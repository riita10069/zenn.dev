---
title: "Mixup / CutMix"
---

# Mixup / CutMix

## ひとことで言うと

学習に使う画像を2枚ずつ取り出して混ぜ合わせ、ラベル (正解) も同じ割合で混ぜて学習させる、という意外なほど単純で強力なデータ拡張 (data augmentation, 学習データを人工的に増やす・変形する技術) です。Mixup は2枚を半透明に重ねるように画素単位で線形混合し、CutMix は片方の画像から四角い領域を切り取ってもう片方に貼り付けます。これだけでモデルの汎化性能 (未知データへの当たりやすさ) が上がり、過学習や敵対的入力への頑健性も改善します。

## 直感的な理解

普通の画像分類の学習は「この画像は猫」「この画像は犬」と、1枚に1つの正解を対応づけて教えます。しかしこのやり方だと、モデルは学習データを丸暗記しがちで、データとデータの「間」については何も知りません。たとえば猫と犬の中間のような曖昧な入力に対し、根拠なく猫 100% と言い切ってしまう (過信, over-confidence)。

Mixup の発想は「2枚を混ぜた中間画像を作り、ラベルも中間にする」ことです。猫画像と犬画像を 70:30 で重ねたら、ラベルも「猫 0.7, 犬 0.3」にする。こうするとモデルは「データの間の領域」でも滑らかに振る舞うよう学習し、決定境界 (クラスを分ける線) が滑らかになります。

```
Mixup (画素を線形混合):
  画像A(猫) ──┐
              ├─→ 0.7*A + 0.3*B  (半透明に重ねた一枚)   ラベル: 猫0.7 犬0.3
  画像B(犬) ──┘

CutMix (領域を切り貼り):
  ┌───────────┐         ┌───────────┐
  │           │         │      ┌────┐│
  │  猫(A)     │   +     │ 猫(A)│犬B ││   ラベル: 面積比で 猫0.75 犬0.25
  │           │         │      └────┘│
  └───────────┘         └───────────┘
                         (Bの四角い切片を貼る)
```

CutMix は混ぜ方が違います。半透明に重ねるのではなく、片方の四角いパッチを切って貼る。猫画像に犬の一部が貼られた画像になり、ラベルは「貼られた面積の割合」で混ぜます。Mixup より画像が自然 (実在しうる見た目) で、物体の位置を捉える力も育ちます。

## 基礎: 前提となる概念

**経験損失最小化 (ERM, Empirical Risk Minimization)**: 通常の学習は手元のデータ点だけで損失を最小化します。Mixup の原論文の副題が "Beyond ERM" なのは、データ点そのものでなく「データ点の間を線形補間した仮想サンプル」でも学習する点で ERM を超えるからです。これを近傍リスク最小化 (Vicinal Risk Minimization, VRM) と呼びます。

**ソフトラベル (soft label)**: 「猫 1.0」のように1クラスだけ 1 の硬いラベル (one-hot) ではなく、「猫 0.7, 犬 0.3」のように確率的に分散したラベル。Mixup は混合比をそのままソフトラベルにします。

**ベータ分布 (Beta distribution)**: 0 から 1 の値を生む確率分布。Mixup は混合比 $\lambda$ をベータ分布 $\text{Beta}(\alpha, \alpha)$ から毎回サンプルします。パラメータ $\alpha$ が小さい (例 0.2) と $\lambda$ は 0 か 1 に近い値ばかり出て「ほぼ混ぜない」状態が多くなり、$\alpha$ が大きい (例 1.0) と 0.5 付近の「がっつり混ぜる」が増えます。$\alpha$ で混合の強さを調整します。

**正則化 (regularization)**: 過学習を防ぎ汎化を促す仕組みの総称。Mixup / CutMix はデータ拡張であると同時に強力な正則化として働きます。

## 仕組みを詳しく

### Mixup の数式

ミニバッチからランダムに2サンプル $(x_i, y_i)$, $(x_j, y_j)$ を取り (画像と one-hot ラベル)、混合比 $\lambda \sim \text{Beta}(\alpha, \alpha)$ をサンプルして

$$\tilde{x} = \lambda x_i + (1-\lambda) x_j, \qquad \tilde{y} = \lambda y_i + (1-\lambda) y_j$$

画素値もラベルも同じ $\lambda$ で線形に混ぜます。$\tilde{x}$ を入力、$\tilde{y}$ をターゲットにして普通に交差エントロピー損失で学習するだけ。実装は数行で、計算コストはほぼゼロです。実際には、バッチをシャッフルしたものとの間で混ぜると2回フォワードせずに済みます。

数値例: テンソル形状 (B, C, H, W) = (256, 3, 224, 224) のバッチに対し、$\lambda=0.7$ なら $\tilde{x} = 0.7 x + 0.3 \cdot \text{shuffle}(x)$ という要素ごとの加重和。ラベルは (256, 1000) のソフトラベル行列になります。

### CutMix の数式

CutMix は画素を重ねず、矩形領域を入れ替えます。バイナリマスク $M$ (貼り付ける四角の中が 0、外が 1 のマスク) を使い

$$\tilde{x} = M \odot x_i + (1 - M) \odot x_j$$

$\odot$ は要素ごとの掛け算。四角の大きさは $\lambda \sim \text{Beta}(\alpha,\alpha)$ から決め、矩形の面積比が $(1-\lambda)$ になるよう一辺を $\sqrt{1-\lambda}$ に比例させて切ります。位置はランダム。ラベルは実際に貼られた面積比で混ぜます。

$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$

ポイントは、Mixup と違って画像のどの画素も「本物のどちらかの画像の画素」である (ぼやけた半透明合成にならない) こと。これにより物体の局所特徴 (localizable features) が壊れず、分類だけでなく弱教師あり物体位置推定 (どこに物体があるか) にも効きます。

### なぜ効くのか

- **決定境界の平滑化**: データ間を線形補間したサンプルで学ぶことで、クラス間で予測が急変しない滑らかな境界になり、汎化が向上。
- **過信の抑制 (label smoothing 効果)**: ソフトラベルにより「猫 100%」と言い切らなくなり、予測確率の較正 (calibration, 確信度が実際の正解率と合うこと) が改善。
- **頑健性**: 学習分布の周辺を埋めるので、ノイズ・敵対的摂動・破損画像に強くなる。
- CutMix は **情報を捨てない**: Cutout (一部を黒く塗りつぶす拡張) は画素を捨てるが、CutMix はそこに別画像を入れるので情報量を保ちつつ正則化できる。

## 手法の系譜と主要論文

- **Dropout (Srivastava et al., 2014)** や **Cutout (DeVries & Taylor, 2017, arXiv:1708.04552)**: 情報を一部消して頑健性を出す系統。Cutout は画像の四角を黒塗り。
- **Label Smoothing (Szegedy et al., CVPR 2016)**: 硬いラベルを少しなまらせる正則化。Mixup のソフトラベルと思想が通じる。
- **mixup (Zhang, Cisse, Dauphin, Lopez-Paz, ICLR 2018, arXiv:1710.09412)**: 画素とラベルの線形混合。VRM の枠組みを提示した原点。
- **Manifold Mixup (Verma et al., ICML 2019, arXiv:1806.05236)**: 入力画素ではなく中間層の特徴を混ぜる。隠れ表現を滑らかにする。
- **CutMix (Yun et al., ICCV 2019, arXiv:1905.04899)**: 領域の切り貼り。Cutout の情報損失と mixup の不自然さの両方を解消。
- **AugMix (Hendrycks et al., ICLR 2020, arXiv:1912.02781)**: 複数の拡張を確率的に合成し一貫性損失と組む。分布シフト頑健性に特化。
- **PuzzleMix / SaliencyMix / FMix など**: 顕著性 (どこが重要か) を考慮して混ぜる位置を賢く選ぶ後続研究。CutMix がランダムに矩形を貼ると重要物体を隠してラベルと不整合になる問題への対処。
- 系譜の本質: 「情報を消す (Cutout)」→「滑らかに混ぜる (mixup)」→「自然に切り貼り (CutMix)」→「賢く混ぜる (saliency 系)」という洗練の流れ。現代の画像分類レシピ (DeiT, ConvNeXt など) では mixup と CutMix を両方ランダムに切り替えて使うのが標準装備になっています。

## 論文の実験結果 (定量データ)

評価は ImageNet (1000 クラスの大規模分類) と CIFAR が中心。指標は Top-1 誤り率 (正解が1位に来なかった割合、低いほど良い)。

- **mixup (ICLR 2018)**: ImageNet で ResNet-50 の Top-1 誤りを、ベースライン約 23.5% から約 22.1% へ改善 ($\alpha$ 適切時)。ResNet-101 や ResNeXt でも一貫して 1〜1.5 ポイント改善。CIFAR-10 でも誤りを低減。さらに、ラベルノイズ (わざと間違ったラベルを混ぜたデータ) への頑健性が大幅に向上し、敵対的サンプルへの耐性も上がることを示しました。
- **CutMix (ICCV 2019)**: ImageNet で ResNet-50 の Top-1 誤りを、ベースライン約 23.7% から **約 21.4%** へ改善。これは Mixup (約 22.6% 程度) や Cutout (約 22.9%) を上回り、当時の単純データ拡張で最良クラス。さらに、CutMix で学習したモデルは ImageNet の弱教師あり物体位置推定 (画像ラベルだけで物体位置を当てる) でも精度が向上し、転移学習で物体検出 (Pascal VOC) の mAP (mean Average Precision, 検出の総合精度) も改善しました。CIFAR-100 でも PyramidNet で誤りを大きく下げました。
- **頑健性の定量**: CutMix モデルは入力の遮蔽 (occlusion, 一部を隠す) に強く、ImageNet の一部を黒塗りした評価で精度低下が小さい。Mixup は較正誤差 (ECE, Expected Calibration Error) を下げ、予測確信度が信頼できるようになることが報告されています。

数値の意味: ImageNet の 1〜2 ポイントの Top-1 改善は、アーキテクチャを変えずデータの混ぜ方だけで得られる「ただ乗り」の利得であり、計算コストがほぼ増えない点で実務的価値が極めて高い。だからほぼすべての近年の学習レシピに組み込まれています。

## メリット・トレードオフ・限界

メリット:
- 実装が数行、追加計算コストほぼゼロ。
- 汎化・較正・頑健性 (ノイズ/遮蔽/敵対的) を同時に改善。
- アーキテクチャ非依存で、CNN にも Vision Transformer にも効く (DeiT は mixup/CutMix なしでは ImageNet で大きく劣化する)。
- CutMix は物体位置を捉える特徴を育て、検出・位置推定への転移に有利。

トレードオフ・限界:
- **ラベルと内容の不整合リスク**: CutMix でランダムに貼った矩形が、もう片方の重要な物体を完全に隠してしまうと「ラベルには犬 0.3 とあるのに画像に犬が写っていない」状態が起き、ノイズになる。saliency 系手法はこれを補正するために生まれた。
- **混合が不自然な領域**: Mixup の半透明合成は現実にはありえない見た目で、人間の知覚とはずれる。タスクによっては有害。
- **ハイパーパラメータ $\alpha$ の調整が要る**: 大きすぎると混ぜすぎて学習が難しくなり、小さすぎると効果が出ない。データセットごとに最適値が違う。
- **少エポック学習では逆効果のことがある**: 強い正則化のため、学習が長く回せる前提。短時間学習や小データでは収束を遅らせる。
- **密な予測タスク (セグメンテーション・物体検出) への素直な適用は難しい**: 画素ごとにラベルがあるタスクでは「ラベルを面積比で混ぜる」が自然に定義できず、専用の派生 (検出向けの mosaic 拡張など) が必要。
- **回帰や時系列・点群への一般化**: 線形混合がラベルとして意味を持つかはタスク依存。安易な適用は禁物。

## 発展トピック・研究の最前線

- **顕著性ベース混合 (PuzzleMix, SaliencyMix, Co-Mixup)**: 重要領域を保ちつつ混ぜる位置・量を最適化し、CutMix の不整合を解消。
- **Vision Transformer 必須レシピ化**: DeiT (Touvron et al., 2021) 以降、ViT を ImageNet だけで学習するには mixup + CutMix + RandAugment + label smoothing の併用がほぼ必須。
- **TokenMix / TransMix**: Transformer のトークン単位で混ぜ、Attention に応じてラベル比を補正する ViT 専用拡張。
- **検出向け Mosaic / Copy-Paste**: YOLO 系の Mosaic (4枚を1枚に並べる) や Copy-Paste はインスタンスを切り貼りする、CutMix の検出版とも言える拡張。
- **半教師あり・自己教師ありとの結合 (MixMatch, FixMatch)**: ラベルなしデータの擬似ラベルと mixup を組み合わせて少ラベル学習を強化。
- **自動運転への応用**: 稀少シーン (ロングテール) のデータ増強として、Copy-Paste で稀な物体を様々な背景に貼り込み、検出器の稀少カテゴリ性能を底上げする使い方が実用的。較正改善は安全な確信度推定にも寄与します ([ロングテール認識](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0592_long-tail-recognition))。

## さらに学ぶための関連トピック

- [データ拡張](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0166_data-augmentation)
- [ラベルスムージング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0357_label-smoothing)
- [ロングテール認識](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0592_long-tail-recognition)
- [Vision Transformer (ViT)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0411_vision-transformer)
- [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration)
- [Copy-Pasteデータ拡張](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0162_copy-paste-augmentation)
