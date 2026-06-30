---
title: "build_* ファクトリ関数とレジストリパターン"
---

# build_* ファクトリ関数とレジストリパターン

## ひとことで言うと
深層学習モデルは「画像から特徴を取り出す部分(backbone)」「複数の入力をまとめる部分(fusion)」「最終的な予測を出す部分(head/planner)」のように、役割の違う部品(モジュール)を組み合わせて作られます。ファクトリ関数とは、こうした部品を「名前の文字列ひとつ」で作り分ける小さな関数のことです。たとえば `build_backbone("swin_v2_tiny")` のように頼むと、対応する部品を組み立てて返してくれます。間違った名前を渡すと「その名前は登録されていない。使える名前はこれだよ」と候補を並べて教えてくれます。実験設定を「コードではなくデータ(文字列)」として扱えるようにする、現代の機械学習フレームワークの土台となる設計です。

## 直感的な理解
レストランの厨房を作るとします。注文票に「ハンバーグ」と書けば、厨房の誰かがレシピ棚からハンバーグの作り方を引き出して作ってくれます。注文する側は「ハンバーグの作り方」を知る必要がありません。名前を言うだけです。新しいメニューを足したいときは、レシピ棚に1枚レシピを追加するだけで、注文の仕組みそのものは変えなくていい。

ファクトリ関数とレジストリ(登録簿)はまさにこの「注文票とレシピ棚」です。モデルを使う側のコード(注文する側)は部品の具体的な作り方を知らず、名前だけ渡す。部品の作り方の知識はレジストリ(レシピ棚)に集約される。新しい部品を足すときはレジストリに1行追加するだけで、モデル本体は1行も変えなくていい。

なぜこれが要るのか。素朴に書くと、モデル本体のコードの中に次のような巨大な if-elif の塊(条件によって処理を切り替える「分岐」)が散らばります。

```python
if backbone_name == "swin_v2_tiny":
    backbone = create_model("swinv2_tiny_window8_256", ...)
elif backbone_name == "res_net_50":
    backbone = create_model("resnet50", ...)
elif backbone_name == "conv_next_v2_tiny":
    backbone = create_model("convnextv2_tiny", ...)
elif ...  # 部品が増えるほど際限なく伸びる
```

新しい部品を1つ足すたびに、モデル本体のあちこちの if-elif を書き換える必要があり、書き換え漏れやバグの温床になります。研究では「backbone を10種類試したい」「fusion を5通り試したい」といった組み合わせ爆発が日常茶飯事なので、この素朴なやり方はすぐ破綻します。10種類の backbone と5種類の fusion を試すと、素朴な分岐では条件の組み合わせが膨れ上がりますが、ファクトリなら 10 + 5 = 15 個の登録で済みます(掛け算ではなく足し算で済む)。

## 基礎: 前提となる概念

理解に必要な4つの用語を噛み砕きます。

辞書(dictionary / dict)。「キー(鍵となる名前)→ 値」の対応表です。`{"赤": "#FF0000", "青": "#0000FF"}` のように、名前を引くと中身が返ってきます。レジストリの正体はこの辞書です。辞書の検索は平均的に一定時間(O(1))で済むので、部品が何百種類になっても名前から実体を引く速さは変わりません。

ラムダ(lambda)と高階関数。`lambda` とは「名前のない小さな関数」です。Python では関数も「値」として変数に入れたり辞書に格納したりできます。これを「関数は第一級オブジェクト(first-class object)」と呼びます。レジストリは「キー → 部品を作る関数」という辞書なので、辞書から関数を取り出して呼び出す、という二段構えになります。「関数を返す/受け取る関数」を高階関数(higher-order function)と呼び、ファクトリの根幹をなします。

キーワード引数の束(`**kwargs`)。Python で `**kwargs` と書くと、呼び出し側が渡した名前付き引数(例 `bev_h=450`)を、中身を見ずにまとめて受け取り、そのまま別の関数へ受け渡せます。これは「途中の関数が引数の意味を知らなくても、終点まで運べる」配管(パイプ)のような仕組みです。ファクトリが多様な部品に対応できるのは、この `**kwargs` のおかげで「どんな引数でも素通しできる」からです。

デコレータ(decorator)。`@register` のように関数やクラスの定義の前に付ける記法で、「定義した瞬間にそれをレジストリへ自動登録する」ために使われます。後述の MMDetection や Detectron2 はこのデコレータ式登録を広めました。

## 仕組みを詳しく

典型的なファクトリ関数群は、すべて同じ形をしています。

- `build_backbone(name, pretrained=True, **kwargs)` — 画像特徴抽出器を作る
- `build_view_fusion(fusion_mode, num_views, embed_dim=256, **kwargs)` — 複数カメラ特徴の融合モジュールを作る
- `build_planner(planner_mode, **kwargs)` — 軌道プランナーを作る
- `build_temporal_memory(memory_mode, **kwargs)` — 時系列メモリを作る

どれも内部に「レジストリ(登録簿)」という辞書を持ちます。backbone のレジストリは概念的に次の形です。

```python
BACKBONE_REGISTRY = {
    "swin_v2_tiny":      lambda pretrained=True, **kw: create_model("swinv2_tiny_window8_256", pretrained=pretrained, **kw),
    "conv_next_v2_tiny": lambda pretrained=True, **kw: create_model("convnextv2_tiny",         pretrained=pretrained, **kw),
    "res_net_50":        lambda pretrained=True, **kw: create_model("resnet50",                pretrained=pretrained, **kw),
}
```

build 関数の処理は、例外なく3ステップに分解できます。

ステップ1: 渡された名前がレジストリのキーに存在するか確認する。

ステップ2: 無ければ `ValueError`(値が不正というエラー)を投げる。このとき必ず「使える名前の一覧」を一緒に出す。

```python
def build_backbone(name, pretrained=True, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. "
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )
    return BACKBONE_REGISTRY[name](pretrained=pretrained, **kwargs)
```

例えば `build_backbone("swim_v2_tiny")`(swin の綴り間違い)と呼ぶと、

```
ValueError: Unknown backbone 'swim_v2_tiny'.
Available: ['swin_v2_tiny', 'conv_next_v2_tiny', 'res_net_50']
```

と表示され、何が正しいかすぐ分かります。この「候補列挙付きエラー」は地味ですが極めて重要です。エラーメッセージが `KeyError: 'swim_v2_tiny'` だけだと、何が正しいのか分からず、ソースを開いてレジストリの中身を探す羽目になります。候補を出すだけで、デバッグ時間が数分から数秒に縮みます。さらに親切な実装では、`difflib.get_close_matches` で綴りの近い候補を「もしかして swin_v2_tiny?」と提案します。

ステップ3: キーがあれば対応する値(関数やクラス)を呼び出し、追加引数 `**kwargs` をそのまま転送して結果を返す。

`**kwargs` の流れを図示します。

```
build_view_fusion("bev", num_views=8, bev_h=450, bev_w=300)
        |
        |  fusion_mode="bev" を FUSION_REGISTRY で照合
        v
   FUSION_REGISTRY["bev"]  ->  BEVViewFusion クラス
        |
        |  num_views=8, embed_dim=256, bev_h=450, bev_w=300 を素通し転送
        v
   BEVViewFusion(num_views=8, embed_dim=256, bev_h=450, bev_w=300) を生成して返す
```

ここで `bev_h=450, bev_w=300` は `build_view_fusion` 自身は意味を理解せず、生成する `BEVViewFusion` クラスへそのまま渡しています。融合方式が concat のときは `bev_h` など渡しませんし、bev のときだけ渡す。このように「方式ごとに必要な引数が違う」状況を、`**kwargs` の素通しで吸収しています。

注意点として、この設計は「kwargs を中身を見ずに丸ごと転送する」ため、選んだ部品が受け取れない引数を渡すと部品側でエラーになります。レジストリは事前にフィルタしないので、呼び出し側が「選んだ部品の受け取れる引数を渡す」責任を持ちます。これは柔軟性と引き換えの代償です。

### レジストリを動的選択肢の源にする

このパターンの実利が最も効くのは、コマンドライン引数や設定ファイルの「選択肢」をレジストリのキーから自動取得する設計です。

```python
parser.add_argument("--backbone", choices=list(BACKBONE_REGISTRY.keys()))
```

こう書いておけば、レジストリに新しい部品を1行登録するだけで、学習スクリプトを1行も書き換えずに `--backbone <新しい名前>` で選べるようになります。さらに不正な値はコマンドラインの段階で弾かれ、エラーメッセージに候補が自動で並びます。「追加は1行、本体は無改変」が達成される。これは設計原則の「開放閉鎖原則(Open-Closed Principle): 拡張に対して開いていて、変更に対して閉じている」を体現しています。

## 手法の系譜と主要論文

ファクトリ関数とレジストリは特定の論文の発明ではなく、ソフトウェア工学の古典的な設計パターンが、機械学習の実装慣習として再発見・標準化されてきたものです。系譜を追います。

Gang of Four(1994)。Gamma, Helm, Johnson, Vlissides の "Design Patterns: Elements of Reusable Object-Oriented Software"(Addison-Wesley)が Factory Method パターンと Abstract Factory パターンを定式化しました。「どのクラスをインスタンス化するかの決定を、クライアントコードから切り離して別の場所に委ねる」手法です。狙いは「使う側のコードを具体的なクラス名から切り離す」こと、効果は「新しい型を追加してもクライアントを変えずに済む」こと、トレードオフは「間接層が1枚増え、コードを追うのにレジストリを経由する手間が生じる」ことです。

MMDetection(Chen et al., arXiv:1906.07155, 2019, OpenMMLab)。物体検出フレームワークとして `@DETECTORS.register_module()` のようなデコレータ式レジストリを広めました。設定ファイル(Python dict)に書いた `type='FasterRCNN'` という文字列だけでモデル全体を再帰的に構築できる仕組みで、「実験設定をコードではなくデータ(設定ファイル)として扱う」研究文化を決定づけました。1つの config ファイルが1つの実験を完全に再現できるため、論文の追試・比較が劇的にやりやすくなりました。

Detectron2 / fvcore Registry(Wu et al., Facebook AI Research, 2019)。`Registry("BACKBONE")` のように汎用のレジストリクラスを提供し、`@BACKBONE_REGISTRY.register()` でクラスを登録、`build_backbone(cfg)` で取り出す、という今日の標準形を確立しました。本稿で説明した `build_*` + `*_REGISTRY` の組はこの直系の子孫です。

Hugging Face Transformers(Wolf et al., EMNLP 2020, arXiv:1910.03771)。`AutoModel.from_pretrained("bert-base-uncased")` のように、モデル名の文字列から適切なクラスを解決して生成する巨大なファクトリ群(Auto クラス)を整備しました。`AutoModel`, `AutoTokenizer`, `AutoConfig` はいずれも名前 → クラスの解決表を内部に持ち、「名前で部品を呼ぶ」発想がエコシステム全体に行き渡っていることが分かります。

Hydra(Yadan, Facebook, 2019)。設定をツリー状に合成し、コマンドラインから `model.backbone=resnet50` のように上書きできるフレームワークで、レジストリ + 設定駆動の発展形です。OmegaConf による型・補間機能を持ち、ハイパーパラメータ探索(sweep)とも統合されます。

## 論文の実験結果(定量データ)

設計パターンそのものは精度を上げる手法ではないため、ベンチマーク数値で「何点改善」とは測れません。しかし効果は別の形で定量化できます。

MMDetection が報告する最大の貢献は「再現性とモジュール再利用」です。論文によると、Faster R-CNN, Mask R-CNN, RetinaNet, Cascade R-CNN など数十のモデルが、共通のレジストリ基盤の上で、それぞれ数十行の config ファイルだけで定義・再現されています。COCO データセット(物体検出の標準ベンチマーク、約12万枚の訓練画像・80カテゴリ)上で、各モデルの mAP(mean Average Precision、検出精度の代表指標。0-100で高いほど良い)が原論文とほぼ一致する形で再現され、これが「設定駆動 + レジストリ」の正しさの実証となりました。例えば ResNet-50 backbone の Faster R-CNN で COCO box AP 約37、Mask R-CNN で約38といった値が、config の backbone 文字列を差し替えるだけで ResNeXt-101 などへ切り替えられ、AP が数ポイント単位で変動することを系統的に比較できます。

「どの要素を抜くとどうなるか」というアブレーション的な観点では、レジストリを使わない実装との比較で「新規モデル追加に要するコード変更行数」が桁違いに減ることが知られています。素朴な if-elif 実装では新 backbone 追加が「本体の複数箇所 + テスト + 設定」で数十〜数百行に波及するのに対し、レジストリ実装では「登録1行 + 部品クラス」で閉じます。バグ混入率はコード変更量にほぼ比例するため(欠陥密度は1KLOCあたり数件〜数十件という経験則がある)、変更行数の削減はそのまま品質に効きます。

## メリット・トレードオフ・限界

メリット。第一に、部品追加がレジストリへの1行追加で完結し、モデル本体や学習スクリプトを変更しなくてよい(開放閉鎖原則)。第二に、不正な名前に対し候補一覧つきの分かりやすいエラーが出るので、綴り間違いをすぐ直せる。第三に、実験設定を文字列(コマンドライン引数や設定ファイル)として扱え、コード変更なしに構成を切り替えられる。設定ファイル1枚が1実験を完全に再現するので、研究の追試性が劇的に上がる。

トレードオフと限界。第一に、`**kwargs` を中身を見ずに転送する設計のため、間違った引数を渡してもファクトリ自身は検出できず、生成先でエラーになる。これは「設定ミスがモデル生成の瞬間まで気づけない」という遅延を生みます。第二に、「名前 → 実体」の対応がレジストリ辞書に隠れるため、IDE のコード補完や定義ジャンプが効きにくく、初見では実体を追いづらい(間接層の代償)。第三に、登録のタイミング問題。デコレータ式レジストリでは「そのモジュールが import されないと登録されない」ため、import 漏れで「Unknown ...」が出る罠があります。第四に、型安全性の喪失。文字列キーは静的型チェックの対象外なので、タイプミスはコンパイル時には捕まらず実行時まで残ります。

研究上の未解決課題としては、レジストリの肥大化と名前空間の衝突(別ライブラリが同じキーを登録する)、設定ファイルの再帰的構築が深くなりすぎて「config が読めない」問題、そして「設定の自由度が高すぎて意味のない組み合わせも作れてしまう」検証不足の問題が挙げられます。近年は dataclass や Pydantic、Hydra(構造化設定)による型付き設定で、文字列の弱点を補う方向の研究・実装が進んでいます。

## 発展トピック・研究の最前線

構造化設定(structured config)。Hydra + OmegaConf や Pydantic は、設定を「型付きの dataclass」として定義し、レジストリの文字列キーが妥当か、引数の型が正しいかを早期に検証します。これにより「実行直前まで気づけない設定ミス」を、実行前のバリデーションで捕まえられるようになります。

プラグインアーキテクチャ。レジストリを「外部パッケージが後から部品を注入できる拡張点」として使う設計が広がっています。Python の entry_points 機構と組み合わせると、別の pip パッケージをインストールするだけで新しい backbone が選択肢に現れる、といったことが可能になります。

研究の最前線では、NAS(Neural Architecture Search、ネットワーク構造を自動探索する研究)がレジストリを「探索空間の定義」として使います。レジストリに登録された部品の組み合わせを探索アルゴリズムが自動で試すので、ファクトリパターンが「人間が手で組む」から「機械が自動で組む」への橋渡しになっています。さらに AutoML フレームワーク(Optuna, Ray Tune など)は、レジストリのキー集合を探索空間のカテゴリ変数として受け取り、ベイズ最適化や進化的探索で最良の部品の組み合わせを見つけます。

## さらに学ぶための関連トピック
- [隠れ次元の統一（embed_dim=256 という設計判断）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0957_embed-dim-unification)
- [Channel Projection（1x1 Conv によるチャネル次元射影）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0947_channel-projection-layer)
- [マルチスケール特徴のプーリング連結](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0988_multi-scale-feature-pooling)
- [feature_infoによるチャネル自動発見](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0960_feature-info-channel-discovery)
