---
title: "画像分類"
---

# 画像分類

## ひとことで言うと

画像分類(image classification)とは、1枚の画像に対して「これは何か」を1つのラベルで答えるタスクです。たとえば犬の写真を見て「犬」と答える。コンピュータビジョンで最も基本的な課題でありながら、深層学習の歴史を切り開いた象徴的なタスクであり、ここで学ばれた特徴抽出器(バックボーン)が検出・セグメンテーションなど他のあらゆる視覚タスクの土台になります。

## 直感的な理解

人間は猫の写真を一瞬で「猫」と認識しますが、コンピュータにとって画像は単なる数値の格子です。たとえばカラー画像は縦 H ・横 W ・色3チャンネル(赤緑青、RGB)の数値の塊で、tensor shape で書くと [3, H, W]。224x224 のカラー画像なら 3×224×224 = 150,528 個の数字です。この膨大な数字の並びから「猫らしさ」を取り出すのが画像分類の核心です。

難しさは、同じ「猫」でも見え方が無数にあること。明るさ、角度、品種、隠れ具合が変わっても「猫」と答えねばならない(不変性、invariance)。一方で、猫と犬のようなわずかな違いは見分けねばならない(弁別性、discriminability)。この両立をどう実現するかが、モデル設計の歴史そのものです。

## 基礎: 前提となる概念

出力の作り方を押さえます。分類モデルは最後に、各クラスに対するスコア(ロジット、logit)を並べたベクトルを出します。1000クラスなら長さ1000のベクトル。これを確率に変換するのがソフトマックス(softmax)関数です。

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

z_i はクラス i のロジット、exp は指数関数、分母は全クラスのロジットを指数化して足したもの。これで全クラスの確率が正の値になり合計1になります。最も確率が高いクラスを予測ラベルとします。

学習に使う損失は交差エントロピー(cross-entropy)です。

```
L = - Σ_c y_c log p_c
```

y_c は正解なら1、他は0(one-hot ラベル)、p_c はモデルが出した確率。正解クラスの確率 p に対して -log p を最小化する、つまり正解クラスの確率を1に近づける圧力です(関連: [損失関数の概観](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0593_loss-functions-overview))。

ベンチマークの定番は ImageNet(正確には ILSVRC データセット)。約128万枚の学習画像を1000クラスに分類します。評価指標は:

- Top-1 accuracy: 最も確率が高い1つが正解と一致する率。
- Top-5 accuracy: 確率上位5つの中に正解が含まれる率。1000クラスは似たものが多いので、歴史的に Top-5 もよく報告されました。

## 仕組みを詳しく

### 畳み込みニューラルネット(CNN)

CNN の主役は畳み込み層(convolution)です。小さなフィルタ(たとえば 3×3 の重み)を画像全体にスライドさせ、局所的なパターン(エッジ、角、模様)を検出します。同じフィルタを画像のどこにでも使い回す(重み共有)ことで、「左上にあっても右下にあっても同じ模様を検出できる」位置不変性とパラメータ削減を同時に実現します。

層を重ねるほど、検出するパターンが「エッジ → 模様 → 部品 → 物体」と抽象化されていきます。浅い層は単純な特徴、深い層は意味的な特徴を捉える、という階層構造です。途中でプーリング(pooling、領域内の最大値や平均を取って解像度を下げる)を挟み、徐々に空間サイズを縮めながらチャンネル(特徴の種類)を増やします。

### 勾配消失と残差接続

層を深くするほど表現力は上がるはずですが、単純に深くすると学習が進まなくなります。誤差を逆向きに伝える逆伝播(backpropagation)で、勾配(損失を減らす方向の情報)が層を遡るうちに小さくなって消えてしまう「勾配消失」が起きるためです。

ResNet(He et al. 2016)はこれを残差接続(residual connection / skip connection)で解決しました。ある層の出力を H(x) とするとき、層には差分 F(x) = H(x) - x だけを学ばせ、入力 x をショートカットで足し直します。

```
入力 x ──────────────┐(ショートカット)
   │                  ↓
 [畳み込み層] → F(x) → (+) → 出力 H(x)=F(x)+x
```

こうすると勾配がショートカット経由でまっすぐ流れ、100層を超える深さでも学習できるようになりました。「何もしない(恒等写像)が簡単に表現できる」ので、深くしても性能が劣化しにくいのが利点です。

### Vision Transformer(ViT)

長らく CNN が主流でしたが、Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT), ICLR 2021(arXiv:2010.11929)が、自然言語処理の Transformer をほぼそのまま画像に適用しました。

仕組みは、画像を 16×16 のパッチに分割し、各パッチを1つの「単語」とみなして系列にし、自己注意(self-attention)機構で全パッチ間の関係を直接学びます。自己注意とは、各要素が他のすべての要素にどれだけ注目するかを学習する仕組みで、CNN のように局所しか見ないのではなく最初から画像全体の関係を捉えられます。224×224 を16刻みにすると 14×14=196 パッチ、これにクラス分類用のトークンを1つ足して系列にします。

ViT は CNN が持つ「局所性」「位置不変性」といった事前の思い込み(帰納バイアス、inductive bias)が弱いぶん、大量のデータがないと真価を発揮しません。逆に大規模データでは CNN を上回ることを示しました。

## 手法の系譜と主要論文

- AlexNet(Krizhevsky et al. NeurIPS 2012): 深層 CNN を GPU で学習し ImageNet で圧勝。深層学習ブームの起点。
- VGG(Simonyan & Zisserman ICLR 2015): 3×3 の小フィルタを積む単純で深い設計。
- GoogLeNet / Inception(Szegedy et al. CVPR 2015): 複数サイズのフィルタを並列に使う。
- ResNet(He et al. CVPR 2016, arXiv:1512.03385): 残差接続。深さの壁を突破。
- DenseNet(Huang et al. CVPR 2017): すべての前層と接続。
- EfficientNet(Tan & Le ICML 2019): 深さ・幅・解像度をバランス良く拡大する複合スケーリング。
- ViT(Dosovitskiy et al. ICLR 2021, arXiv:2010.11929): Transformer の導入。
- Swin Transformer(Liu et al. ICCV 2021, arXiv:2103.14030): 窓を区切って注意を計算し階層構造を持たせ、検出・セグメンテーションのバックボーンに適した ViT。
- ConvNeXt(Liu et al. CVPR 2022, arXiv:2201.03545): ViT の設計思想を CNN に取り込み、CNN でも Transformer に匹敵する性能を示した。

## 論文の実験結果(定量データ)

歴史を ImageNet の Top-5 誤り率(低いほど良い)でたどると流れがつかめます。報告値はおおよそ次の通りです。

- AlexNet(2012): Top-5 誤り率 およそ 16〜17%。前年までの非深層手法(およそ26%)を大きく更新し、衝撃を与えました。
- VGG / GoogLeNet(2014): およそ 6〜7% 台。
- ResNet-152(2015): およそ 3.5〜4% 台。これは当時報告された人間の Top-5 誤り率(およそ 5%)を下回り、「機械が人を超えた」と話題になりました。

Top-1 accuracy では、ResNet-50 がおよそ 76% 前後、EfficientNet や大規模事前学習を使った最新モデルは 85〜88% 超に達すると報告されています。ViT 論文では、ViT を中規模データ(ImageNet のみ)で学習すると同規模の ResNet にやや劣るが、巨大データ(JFT-300M、約3億枚)で事前学習してから ImageNet に転移すると Top-1 でおよそ 88% 超を達成し、CNN を上回ったと報告されています。これが「ViT はデータを食う」という有名な教訓の根拠です。

数値の読み方として、ImageNet 上位の競争は今やコンマ数パーセントを争う段階で、誤り率1ポイントの差にも大きな工学的努力が要ります。一方で「事前学習データを大きくすると、同じモデルでも数ポイント上がる」ことは繰り返し確認されており、データ規模の効果がモデル設計に匹敵するほど大きいというのが現代の理解です。

## メリット・トレードオフ・限界

意義:
- 事前学習の土台。ImageNet で学習したバックボーンを、検出やセグメンテーションへ転移すると少ないデータで高性能が得られる(転移学習、関連: [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation))。
- 評価の明快さと再現性。1画像1ラベルなので指標が単純。

限界・トレードオフ:
- 1画像1ラベルの前提が現実とずれる。1枚に複数物体があるのが普通で、そこから検出・セグメンテーションへ発展しました。
- ショートカット学習: モデルが本質でなく背景や質感など「ズル」の手がかりに頼ることがある(牛は草原にいる、で判断するなど)。
- 頑健性の弱さ: わずかなノイズや敵対的摂動、分布シフト、見たことのないクラスに弱い(関連: [異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection)、[ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning))。
- 校正のずれ: 深いネットは過信しがちで、出力確率が実際の正解率とずれる(関連: [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration))。
- データとアノテーションのコスト: ImageNet 級のラベル付けは高コスト(関連: [能動学習(Active Learning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0874_active-learning))。

## 発展トピック・研究の最前線

- 自己教師あり事前学習: ラベルなし画像から表現を学ぶ(MAE, DINO, SimCLR 等)。ラベル不要で強力なバックボーンを得る流れ(関連: [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining))。
- 基盤モデルとゼロショット分類: CLIP は画像とテキストを同じ空間に埋め込み、ラベルを「文」で与えるだけで未知クラスを分類できる(関連: [ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning))。
- 効率化・蒸留: モバイルや車載向けに、軽量化・量子化・蒸留で精度を保ちつつ計算を削る。
- 頑健性研究: 分布シフト・敵対的攻撃への耐性、ショートカット学習の抑制。

## さらに学ぶための関連トピック

- [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation)
- [損失関数の概観](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0593_loss-functions-overview)
- [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining)
- [ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning)
- [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration)
- [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods)
