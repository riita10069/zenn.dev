---
title: "timm (PyTorch Image Models)"
---

# timm (PyTorch Image Models)

## ひとことで言うと
timm(ティム。PyTorch Image Models の略)は、画像認識用の有名なニューラルネットワーク(ResNet、Swin、ConvNeXt、ViT、EfficientNet など千種類以上)を、たった1行の `create_model("モデル名")` で呼び出せるようにまとめた Python ライブラリです。事前学習済みの重み(ImageNet などで訓練済みのパラメータ)も自動でダウンロードしてくれます。今日の画像系の研究・開発では、自前でバックボーンを実装する代わりに、この timm から調達するのが事実上の標準になっています。

## 直感的な理解
料理に例えると、timm は「下ごしらえ済みの食材が並んだ巨大な冷蔵庫」のようなものです。ResNet-50 が欲しければ棚から1つ取り出すだけ。しかもその食材(モデル)は、すでに最高の状態に仕込まれて(事前学習されて)おり、正しい使い方(前処理の設定)のラベルまで貼ってあります。自分で畑から育てる(ゼロから実装・学習する)必要はありません。研究者が時間を使うべきなのは「どの食材をどう組み合わせるか(新しいアイデアの検証)」であって、食材の下ごしらえそのものではない、という発想です。

## 基礎: 前提となる概念
「バックボーン」とは、画像を入力するとその特徴を数値の塊(特徴マップ)に変換する土台のネットワークです。物体検出・領域分割・俯瞰図生成などの応用は、すべてこの出力の上に乗ります。

「事前学習(pretraining)」とは、本番のタスクの前に、大規模な汎用データ(代表例は ImageNet。約128万枚、1000カテゴリ)でモデルを訓練しておくことです。こうして得た重みには「エッジ、テクスチャ、物体の部品、物体カテゴリ」といった汎用的な特徴抽出能力が詰まっており、これを別タスクの初期値として再利用する手法を転移学習と呼びます([ImageNet 事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0559_imagenet-pretraining)、[凍結バックボーン vs 微調整](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0557_frozen-vs-finetune-backbone))。

新しいモデルを作るとき、バックボーンを毎回ゼロから実装するのは非常に大変です。ResNet-50 だけでも数百行、Swin Transformer や ConvNeXt はもっと複雑です。さらに困るのは、論文の著者が公開した重みが、ファイル形式・パラメータのキー名・入力正規化のやり方がバラバラで、自分のコードに読み込ませるのに苦労する点です。

具体例で困りごとを示します。あるチームが ResNet-50 を、別のチームが ViT を実装したとします。両者のコードは、モデルの作り方(コンストラクタの引数)が違い、事前学習重みのダウンロード先・読み込み方が違い、入力画像の前処理(平均・標準偏差で正規化する値)が違い、中間特徴の取り出し方も違います。これでは「バックボーンを差し替えて比べる」という、研究・開発で最も頻繁にやる作業が地獄になります。具体的には、ある実装は画素値を [0,1] に正規化し、別の実装は ImageNet 統計(平均 0.485/0.456/0.406、標準偏差 0.229/0.224/0.225)で標準化し、さらに別の実装は [-1,1] にスケールする、といった食い違いが頻発します。前処理が事前学習時とずれると、モデルは見たことのない画素分布を受け取り、精度が静かに落ちます。

timm は、Ross Wightman 氏(現 Hugging Face)が長年かけて整備した、この問題を解決するライブラリです。何百ものモデルを「同じ API(呼び出し方)」「同じ重み管理」「同じ前処理規約」に統一しました。

## 仕組みを詳しく
中心となるのは `create_model` という関数です。

```python
import timm

# ResNet-50を、ImageNet事前学習重み付きで作る
model = timm.create_model("resnet50", pretrained=True)

# 1000クラス分類ヘッドではなく、特徴ベクトルだけ欲しい
model = timm.create_model("resnet50", pretrained=True, num_classes=0)

# 各stageのマルチスケール特徴を全部返してほしい
model = timm.create_model("resnet50", pretrained=True, features_only=True)
```

引数の意味を順に説明します。第1引数の文字列がモデル名です。`timm.list_models()` で一覧が見られ、`timm.list_models("*convnext*")` のようにワイルドカードで絞り込めます。`timm.list_models(pretrained=True)` で「重みが用意されているモデル」だけに絞れます。`pretrained=True` にすると、事前学習済み重みを自動でダウンロード・ロードします(`False` ならランダム初期化)。`num_classes` で出力クラス数を変えられ、`0` にすると分類ヘッドが外れて特徴ベクトルが出ます(`global_pool=""` を併用すれば pooling 前の空間特徴も取れます)。`features_only=True` にすると、最終出力だけでなく各 stage の中間特徴(マルチスケール特徴)をリストで返します(詳細は [features_only マルチスケール特徴抽出](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1106_features-only-multiscale))。

入力画像の前処理も、モデルごとに正しい設定を timm が知っています。

```python
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config)
# data_config の例: {"input_size": (3, 224, 224),
#                    "mean": (0.485, 0.456, 0.406),
#                    "std":  (0.229, 0.224, 0.225),
#                    "interpolation": "bicubic", "crop_pct": 0.95, ...}
```

これで「そのモデルが事前学習時に使ったのと同じ正規化・リサイズ・補間方法」を自動適用できます。たとえば一部の ViT は bicubic 補間と特定の crop 比を前提にしており、これを bilinear で代用すると数値が微妙にずれて精度が下がることがあります。timm はこの細部までモデルごとに記録しています。

`features_only=True` のとき、各特徴マップのチャンネル数や縮小率を、forward を回さずに `feature_info` から取得できます。

```python
m = timm.create_model("resnet50", features_only=True)
print(m.feature_info.channels())   # 例: [256, 512, 1024, 2048]
print(m.feature_info.reduction())  # 例: [4, 8, 16, 32] (元画像に対する縮小率)
```

これにより、後段のネットワーク(特徴融合層や検出ヘッド)を「何チャンネル受け取るか」自動で配線できます。バックボーンを差し替えてもチャンネル数を手で書き換える必要がなく、実験の回転が速くなります。

timm のもう一つの強みは、訓練レシピそのものを提供している点です。AutoAugment / RandAugment / Mixup / CutMix といったデータ拡張、EMA(指数移動平均。学習中の重みを平滑化して汎化を上げる)、LAMP や AdamW などのオプティマイザ、cosine など各種学習率スケジュールが整備されており、「同じ条件でモデルを公平に比べる」基盤になっています。付属の `train.py` / `validate.py` は、多くの論文の数値を再現するために実際に使われています。

## 手法の系譜と主要論文
timm 自体は論文ではなくソフトウェアライブラリ(Ross Wightman, GitHub: huggingface/pytorch-image-models、初公開 2019 年頃)ですが、学術論文でも `Wightman, R. PyTorch Image Models. 2019.` の形で広く引用されます。

timm が研究上の価値を最も鮮やかに示した例が ResNet strikes back(Wightman, Touvron, Jégou, 2021, arXiv:2110.00476)です。この論文の主張は「古い ResNet-50 も、最新の訓練レシピ(強いデータ拡張・改良された最適化・長いスケジュール・適切な正則化)で鍛え直せば、ImageNet top-1 精度が当初報告の 76% 前後から 80% 超まで伸びる」というものでした。動機は、新しいアーキテクチャの論文が「古いモデルの弱い数値」と比較して優位を主張しがちな状況への警鐘です。効果として、アーキテクチャの優劣を語る前に「訓練手順を揃えて比較すべき」という規範を定着させました。トレードオフは、強い訓練レシピは学習エポックが増え計算コストが上がる点です。この研究の実験基盤がまさに timm であり、「バックボーンを公平に比較するには統一基盤が要る」という timm の存在意義そのものを裏付けています。

背景として ResNet の原典(He et al., Deep Residual Learning, CVPR 2016, arXiv:1512.03385)があり、残差接続(層の入力を出力にショートカットで足し込む)という発想が深いネットワークの学習を可能にしました。その後、ConvNeXt(Liu et al., CVPR 2022, arXiv:2201.03545)が「ViT の設計上の工夫を CNN に逆輸入すれば CNN でも Swin に匹敵する」と示し、CNN と Transformer の優劣論争を訓練レシピ込みで再検証しました。timm はこうした歴史的モデルから最新の ViT 系・自己教師ありモデルまで、同じ土俵で扱えるようにした点に意義があります。

## 論文の実験結果(定量データ)
ResNet strikes back の数値を具体的に見ます。指標は ImageNet-1K の top-1 精度(モデルが最も自信を持って出した1つのラベルが正解と一致した割合)です。同じ ResNet-50 という構造でありながら、訓練レシピの違いだけで結果が大きく変わりました。論文は複数の訓練手順(A1/A2/A3 と名付けられた、エポック数やデータ拡張の強さが異なるレシピ。A1 が約 600 エポックで最も手厚く、A3 が約 100 エポックで軽量)を提示し、最も手厚い A1 レシピでは ResNet-50 が約 80.4% に達したと報告しています。これは、当時新しいとされたいくつかの Transformer 系モデルに匹敵する水準でした。原典 ResNet-50 の 76% 前後と比べると約 4 ポイントの差で、これは ImageNet では非常に大きな差です。

この結果の意味は重大です。差が「構造の違い」ではなく「訓練の違い」から来ていたことを示したからです。アブレーション的に見れば、データ拡張(Mixup/CutMix/RandAugment)を弱める、エポックを減らす、正則化(重み減衰や stochastic depth)を外すといった操作のそれぞれが精度を着実に下げ、訓練レシピの各要素が積み上がって最終精度を作っていることが分かります。つまり「新アーキテクチャ A がベースライン B より良い」と主張するには、A と B を同じレシピで学習しなければ意味がない、という方法論的教訓を数値で示したのです。

timm が提供する千以上のモデルの ImageNet 精度・パラメータ数・FLOPs・推論速度は公開されており(results フォルダの CSV やリーダーボード)、研究者はこれを参照して「精度と計算コストのトレードオフ曲線」上で自分のタスクに合うモデルを選べます。これは新しいバックボーン研究のベースライン整備として、コミュニティ全体の生産性を底上げしています。

## メリット・トレードオフ・限界
メリットは、千以上のモデルを1行・統一 API で呼べること、事前学習重みの自動ダウンロードとモデルごとの正しい前処理設定を提供すること、`features_only` と `feature_info` でマルチスケール特徴抽出が標準化されていること、再現可能な訓練レシピを同梱していること、そしてメンテナンスが活発で新しいモデルが速やかに追加されることです。

トレードオフと限界としては、外部依存が増える(バージョン差でモデル名・重み URL・返り値の形が変わることがあり、再現性のためにバージョン固定が望ましい)こと、事前学習重みのダウンロードにネット接続が要る(オフライン環境やセキュアな閉域環境では事前にキャッシュするか、ランダム初期化に切り替える必要がある)こと、抽象化が厚いぶん内部で何をしているか追いにくい場面があること、そして timm がまだ対応していない最新研究モデルは結局自前実装が必要になることが挙げられます。

研究上の注意点として、timm のモデル間で前処理規約(入力サイズ・正規化・補間)が異なるため、複数モデルを混在させるパイプラインでは前処理の取り違えが性能劣化の隠れた原因になりやすい点があります。`resolve_model_data_config` を必ずモデルごとに呼ぶのが安全策です。また、同じモデル名に複数の事前学習タグ(例: `.a1_in1k`、`.in21k_ft_in1k` のように学習データ・レシピを表すサフィックス)が存在することがあり、どのタグを使うかで精度が変わるため、論文を再現するときはタグまで明示するのが望ましいです。

## 発展トピック・研究の最前線
timm は Hugging Face に統合され、Hugging Face Hub から重みを直接ロードできるようになりました。これにより、コミュニティが学習した独自重みの共有・再利用が容易になっています。最近は ViT 系の自己教師あり事前学習重み(MAE、DINOv2 など)も timm 経由で扱えるものが増え、教師ありと自己教師ありのバックボーンを同じ API で比較できる環境が整いつつあります。DINOv2(Oquab et al., Meta AI, 2023, arXiv:2304.07193)のような、ラベルなしで強力な汎用視覚特徴を学んだモデルを `create_model` で呼び、凍結特徴として下流タスクに使う、という使い方が広がっています。新しいバックボーンを提案する研究では、timm への取り込みが事実上の「普及の通過儀礼」になっており、ライブラリ自体が研究のインフラとして機能しています。

## さらに学ぶための関連トピック
- [features_only マルチスケール特徴抽出](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1106_features-only-multiscale)
- [ImageNet 事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0559_imagenet-pretraining)
- [凍結バックボーン vs 微調整](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0557_frozen-vs-finetune-backbone)
- [Hiera](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0389_hiera_note)
- [FPN (Feature Pyramid Network)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1067_feature-pyramid-network)
