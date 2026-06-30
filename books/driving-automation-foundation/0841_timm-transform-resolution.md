---
title: "timmによるbackbone駆動の前処理解決"
---

# timmによるbackbone駆動の前処理解決

## ひとことで言うと
画像をモデルに入れる前の「リサイズと色の正規化(前処理)」を、人が数値をベタ書きするのではなく、使う backbone(画像エンコーダ)が事前学習時に使った設定をライブラリから自動で読み出し、それに合わせた前処理を実行時に組み立てる仕組みです。backbone を差し替えても前処理が自動で追従するため、書き間違いによる静かな性能劣化を防げます。

## 直感的な理解
ある料理人(事前学習済みモデル)は、特定の下ごしらえをされた食材(特定の解像度・特定の色味に正規化された画像)に慣れています。本番でいきなり違う切り方・違う味付けの食材を渡すと、その料理人は本来の腕を発揮できません。前処理を事前学習時とそろえるとは、料理人が慣れた下ごしらえを本番でも再現してあげることです。

画像認識モデルは普通「事前学習(pretraining)」されています。大量の画像(ImageNet など)で先に学習させ、画像の見方を覚えさせておくのです。重要なのは「事前学習のときの前処理と、本番で使う前処理を一致させる」こと。解像度や色の正規化がずれると、モデルは学習時と違う見た目の画像を受け取り、覚えた特徴抽出がうまく働かず、性能が落ちます。しかも多くの場合エラーは出ないので、気づきにくい劣化になります。backbone を差し替え可能にしている設計では、差し替えるたびに正しい前処理値も変わるため、手書き管理は特に危険です。

## 基礎: 前提となる概念

事前学習と転移学習(pretraining / transfer learning)。事前学習とは、大規模データセット(典型的には ImageNet、約128万枚・1000クラス)で先にモデルを学習させ、汎用的な画像特徴を獲得させること。転移学習は、その重みを別タスクに流用することです。事前学習の恩恵を受けるには、入力の見た目(解像度・色の分布)を事前学習時と揃える必要があります。

正規化(normalization)。画像の各色チャネル(R, G, B)から決まった平均(mean)を引き、決まった標準偏差(std)で割る処理です。ImageNet で学習したモデルは慣例的に mean=[0.485, 0.456, 0.406]、std=[0.229, 0.224, 0.225] を使います。これは「画素値の分布を、モデルが学習時に見た分布に合わせる」ための前処理です。間違った mean/std を使うと、明るさや色味が学習時とずれます。

リサイズと補間(resize / interpolation)、クロップ。モデルは特定の入力解像度を期待します(たとえば 224x224 や 256x256)。元画像をその大きさに縮める・拡大するのがリサイズで、ピクセルをどう作るか(bicubic、bilinear など)が補間方法です。CenterCrop は中央を正方形に切り出す処理。crop_pct(クロップ比率)は「リサイズ後に中央の何割を残すか」を表し、たとえば 0.9 なら短辺を入力サイズの 1/0.9 倍に縮めてから中央を切り出す、という意味です。

解像度の不一致問題。学習時と評価時で入力解像度がずれると、物体の見かけの大きさ(スケール)がずれ、精度が落ちます。これは見えにくい劣化を生む、よくある落とし穴です。

## 仕組みを詳しく

timm(PyTorch Image Models)というライブラリは、各モデルに `default_cfg` というメタデータ(入力サイズ・mean・std・補間・crop_pct)を持たせています。これを使って前処理を自動生成する典型的なコードは次の通りです。

```python
backbone = timm.create_model(backbone_name, pretrained=False)
data_config = timm.data.resolve_model_data_config(backbone)
transform = timm.data.create_transform(**data_config, is_training=False)
del backbone
```

ステップごとに見ます。

1. `create_model(name, pretrained=False)`。backbone を作ります。`pretrained=False` なので学習済み重みはダウンロードせず、設定(config)を読むためだけの軽い生成です。ネット通信や巨大ファイル取得は起きません。

2. `resolve_model_data_config(backbone)`。そのモデルに紐づくデータ設定を辞書で返します。中身はおおむね次のようです。

```
{
  'input_size': (3, 256, 256),     # チャネル・高さ・幅
  'mean': (0.485, 0.456, 0.406),   # 正規化の平均
  'std':  (0.229, 0.224, 0.225),   # 正規化の標準偏差
  'crop_pct': 0.9,                 # リサイズ後に中央を残す比率
  'interpolation': 'bicubic',      # 補間方法
}
```

これは「このモデルが事前学習時に使った前処理」をモデルのメタデータから引いた値です。

3. `create_transform(**data_config, is_training=False)`。上の設定から、実際に画像を変換する torchvision の Compose(処理の連結)を生成します。`is_training=False`(評価用)なのでランダム拡張は入れず、決定的な並びになります。256x256 入力・crop_pct=0.9 なら「短辺を 256/0.9≒284 に Resize → 256x256 を CenterCrop → ToTensor → ImageNet 正規化」という具合です。

4. `del backbone`。設定はもう取れたので backbone オブジェクトは捨てる(メモリ節約)。

得られた transform を、複数のビュー(複数カメラ画像、ラスタライズした地図タイルなど)に同じく適用すれば、全ビューが同じ形 (3, H, W)・同じ正規化に揃い、モデルの入力契約が一貫します。

幾何との一貫性。リサイズや CenterCrop は画像に幾何変形を加えます。カメラ射影でピクセル対応を使う場合、この同じ幾何変形をカメラ内部行列 K にも適用しないと射影が狂います(→[前処理（Resize/Crop）に合わせたカメラ内部行列のスケーリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0801_camera-intrinsic-scaling))。前処理を1か所(transform)で決めると、画像と K を同じ加工で一貫させやすくなります。

before/after。

```
[before] 手書き
  transform = Compose([Resize(256), CenterCrop(256), ToTensor(),
                       Normalize(mean=[...], std=[...])])
  → backbone を変えるたびに数値を調べて書き直し。間違えても気づきにくい

[after] backbone 駆動
  cfg = resolve_model_data_config(create_model(name))
  transform = create_transform(**cfg, is_training=False)
  → name を変えるだけで解像度・mean/std・補間が正しく追従
```

## 手法の系譜と主要論文

「事前学習時と推論時で前処理(特に正規化と解像度)を一致させる」重要性は、転移学習・事前学習の文献で繰り返し指摘されています。

Deng et al., "ImageNet: A Large-Scale Hierarchical Image Database" (CVPR 2009)。ImageNet を提示した論文。以降、ImageNet の統計値(mean/std)での正規化が事実上の標準になりました。そろえる理由は、モデルが学習時の入力分布(明るさ・色の範囲)に最適化されているため、推論時に分布がずれると性能が落ちるからです。

Touvron et al., "Fixing the train-test resolution discrepancy" (NeurIPS 2019, FixRes, arXiv:1906.06423, Facebook AI)。学習時と評価時で画像解像度がずれると精度が落ちる現象を定量的に分析し、評価解像度を上げて軽く微調整するだけで精度が回復することを示しました。重要なのは、解像度ミスマッチが見えにくい劣化を生むという点で、これがモデルごとに正しい解像度を自動で引く動機になります。

Wightman, "PyTorch Image Models (timm)" (2019-、オープンソースライブラリ)。何百ものモデルそれぞれに `default_cfg`(入力サイズ・mean/std・補間・crop_pct)を持たせ、`resolve_model_data_config` / `create_transform` で前処理を自動生成できるようにしました。動機は、モデルごとに正しい前処理が違うため人手管理だとミスが多発すること。効果は、モデルを変えるだけで正しい前処理が付いてくること。トレードオフは、timm のメタデータに依存するため、timm 外の独自モデルでは config を自分で用意する必要があること。timm はその後、多数の事前学習済みモデルの再現性評価のハブとしても広く使われています(Wightman et al., "ResNet strikes back", 2021 などが代表)。

## 論文の実験結果(定量データ)

FixRes の実験は、解像度整合の効果を端的に示します。ImageNet 分類(指標は top-1 accuracy、テスト画像を正しく1位に当てた割合。高いほど良い)で、学習 224x224・評価 224x224 のモデルを、評価解像度を上げてから軽く再調整すると top-1 が数ポイント改善することが報告されています。たとえば ResNet-50 系で、学習解像度のまま評価するより、解像度を整合・調整したほうが top-1 がおよそ 1〜2 ポイント向上する、という結果です。1〜2 ポイントは ImageNet では大きな差で、論文1本に値する改善幅です。これは「解像度を合わせるだけで、モデルを変えずに精度が上がる」ことを意味します。

正規化のミスマッチについては、間違った mean/std を使うと(たとえば正規化を全くかけない、または別データセットの統計を使う)、事前学習特徴が崩れて転移学習の精度が顕著に落ちることが、転移学習の各種研究で一貫して観察されています。これは「正規化は単なる前処理の作法ではなく、事前学習の性能を引き継ぐための必須条件」であることを示します。

アブレーション的な観点。補間方法(bicubic か bilinear か)や crop_pct の違いも、最後の数十分の一ポイントを左右します。timm の系統的な再現実験では、これらの設定を学習時と揃えるかどうかで報告精度がぶれることが示されており、だからこそ「モデルから設定を引く」自動化が再現性の担保に有効だと結論づけられています。

## メリット・トレードオフ・限界

メリット。backbone が事前学習時に使った解像度・正規化・補間に自動で一致し、事前学習の性能を活かせる。backbone を差し替えても前処理が追従するので、手書き修正が不要でミスが減る。同じ transform を複数ビューや内部行列補正で共有でき、一貫性が保てる。

トレードオフ・限界。第一に、timm が各モデルに持つメタデータに依存します。timm 外の独自モデルでは config を自分で用意する必要があります。第二に、全ビューに同じ前処理を当てる場合、ImageNet 正規化が最適でないビューがありえます。たとえば地図のラスタ画像のような単純で人工的な画像は、自然画像向けの ImageNet 統計より、タスク特化の前処理(コントラスト強調など)が合う可能性があります。第三に、`is_training=False` 固定だと評価用の決定的変換しか得られず、学習時のデータ拡張(ランダムリサイズ・色揺らし等)は別途組み込む必要があります。拡張を足すと、その幾何効果を内部行列補正にも反映する手当てが要ります(→[前処理（Resize/Crop）に合わせたカメラ内部行列のスケーリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0801_camera-intrinsic-scaling))。

研究上の論点として、入力解像度を後から上げる FixRes 流の微調整、ドメイン特化の正規化統計の学習、複数解像度で頑健にする学習(multi-scale training)などが、前処理の最適化として継続的に研究されています。

## 発展トピック・研究の最前線

事前学習が ImageNet 教師ありから、自己教師あり(MAE, He et al. CVPR 2022 など)や大規模画像テキスト対照学習(CLIP, Radford et al. ICML 2021)へ広がる中でも、「事前学習時の前処理を推論時に再現する」原則は変わりません。CLIP は独自の正規化統計を使い、その値を推論時にそろえないと性能が落ちることが知られています。モデルごとに前処理メタデータを携帯させ自動適用する timm の発想は、こうした多様な事前学習モデルを安全に使い回すための実務基盤として定着しています。

## さらに学ぶための関連トピック
- [前処理（Resize/Crop）に合わせたカメラ内部行列のスケーリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0801_camera-intrinsic-scaling)
- [カメラ射影行列 P = K_scaled @ T_ref_to_cam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0802_camera-projection-matrix)
- [履歴・未来ウィンドウによるサンプル列挙](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0217_dataset-sampling-window)
- [マップのラスタ化 (semantic raster)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0826_map-rasterization)
