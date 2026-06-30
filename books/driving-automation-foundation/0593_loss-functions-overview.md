---
title: "損失関数の概観"
---

# 損失関数の概観

## ひとことで言うと

損失関数(loss function)とは、モデルの予測が正解からどれだけ外れているかを1つの数値で表す尺度です。学習とは、この数値をできるだけ小さくするようにモデルの重みを調整する作業に他なりません。「何を小さくするか」を決めるのが損失関数なので、損失の選び方はモデルが何を学ぶかそのものを決める、最も本質的な設計判断のひとつです。

## 直感的な理解

弓矢で的を狙う練習を思い浮かべてください。矢が的の中心からどれだけ外れたかを測り、その「外れ具合」を減らすように構えを直していきます。この「外れ具合の測り方」が損失関数です。

ここで測り方には流儀があります。中心からの距離をそのまま測るのか、距離の二乗(大きく外したときに強く罰する)で測るのか。あるいは「当たったか外れたか」だけを問うのか。測り方を変えると、矯正の方向性が変わります。大外しを特に嫌うなら二乗、たまの大外しは無視して全体を整えたいなら別の測り方、というように。損失関数を学ぶとは、タスクごとに「何を重く罰し、何を許すか」を選ぶ感覚を身につけることです。

## 基礎: 前提となる概念

損失関数が満たすべき性質を押さえます。

- 微分可能性: 深層学習は勾配降下法(損失を下げる方向へ少しずつ重みを動かす)で学習するため、損失は重みについて微分できる(なめらかである)必要があります。勾配(gradient)とは、損失を最も増やす方向を表すベクトルで、その逆向きに進めば損失が減ります。
- 最小値が「良い予測」に対応すること: 損失が最小になる点が、本当に望ましい予測になっていなければなりません。
- タスク整合性: 分類・回帰・対照学習など、タスクの性質に合った形であること。

損失はタスクで大別されます。回帰(連続値を当てる)、分類(カテゴリを当てる)、対照・距離学習(似ているか否かを学ぶ)、生成。以下、代表を見ます。

## 仕組みを詳しく

### 回帰の損失

平均二乗誤差 MSE(Mean Squared Error)。
```
L_MSE = (1/N) Σ_i (y_i - ŷ_i)²
```
y_i は正解、ŷ_i は予測、N はサンプル数、Σ_i は全サンプルの和。誤差を二乗するので、大きく外したサンプルを強く罰します。なめらかで扱いやすい反面、外れ値(極端に変なデータ)に過敏で、1つの大外れに全体が引きずられます。

平均絶対誤差 MAE(Mean Absolute Error)。
```
L_MAE = (1/N) Σ_i |y_i - ŷ_i|
```
誤差の絶対値。外れ値に頑健(二乗しないので大外れの影響が穏やか)ですが、ゼロ点で微分が折れていて勾配が一定なため、最適解付近での収束がやや雑になります。

Huber 損失はこの両者の良いとこ取りです。誤差が小さいうちは二乗(なめらかで収束しやすい)、ある閾値 δ を超えたら絶対値に切り替える(外れ値に頑健)。
```
小さい誤差: 二乗で扱う / 大きい誤差: 線形で扱う(δで切替)
```
物体検出のボックス回帰でよく使われる Smooth L1 損失は Huber の特別な場合です。閾値 δ をいくつにするかが調整点になります。

### 分類の損失

交差エントロピー(cross-entropy)が王道です。2クラスなら、
```
L = - [ y log p + (1-y) log(1-p) ]
```
y は正解(0か1)、p はモデルが出した「クラス1である確率」。正解が1のとき -log p を最小化し、p を1へ押し上げます。多クラスなら正解クラスの確率 p に対し -log p。確率が0に近いほど -log p は急激に大きくなり、強い勾配で矯正します(関連: [画像分類](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0391_image-classification))。

Focal Loss(Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017, arXiv:1708.02002)は、クラス不均衡を扱う交差エントロピーの改良です。物体検出では背景(負例)が圧倒的に多く、簡単な背景の損失が積み上がって、難しい前景の学習がかき消されます。Focal Loss は簡単に当たっているサンプルの損失を抑える係数を掛けます。
```
L_focal = - (1 - p_t)^γ log(p_t)
```
p_t は正解クラスの予測確率、γ(ガンマ)は集中度を決める調整値(典型的に2)。すでに正しく自信を持って当てているサンプル(p_t が1に近い)は (1 - p_t)^γ がほぼ0になり損失が小さくなる。一方、外しているサンプルは係数が1に近く損失が残る。結果、「難しいサンプルに学習資源を集中」させます。

### 対照・距離学習の損失

似たものは近く、違うものは遠くなる埋め込み空間を学びます。Contrastive Loss(Hadsell et al. 2006)は、ペアが同類なら距離を縮め、異類なら一定マージン m まで離します。Triplet Loss(Schroff et al., FaceNet, CVPR 2015)は、基準(anchor)・同類(positive)・異類(negative)の3つ組で、「anchorはpositiveより、negativeへ少なくともマージンm遠く」を課します。
```
L_triplet = max(0,  d(a,p) - d(a,n) + m)
```
d は距離、a/p/n は anchor/positive/negative。これらは顔認証や再同定(関連: [カルマンフィルタによる追跡](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0489_kalman-filter-tracking))、自己教師あり学習(関連: [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining))の核心です。

### 自己教師あり・確率の損失

InfoNCE(対照学習で使う)、KL ダイバージェンス(2つの確率分布の隔たり)、負の対数尤度(NLL)など。NLL は「正解にどれだけ高い確率を割り当てたか」を測り、確率予測の良し悪し全般の物差しになります(関連: [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods))。

## 手法の系譜と主要論文

- 古典統計: 最小二乗(Gauss/Legendre, 19世紀初頭)、最尤推定。交差エントロピーは情報理論(Shannon 1948)とロジスティック回帰に根ざす。
- ロバスト統計: Huber, "Robust Estimation of a Location Parameter", 1964。外れ値に強い損失の理論的基礎。
- 距離学習: Contrastive(Hadsell et al. CVPR 2006)、Triplet / FaceNet(Schroff et al. CVPR 2015, arXiv:1503.03832)。
- 不均衡対策: Focal Loss(Lin et al. ICCV 2017, arXiv:1708.02002)。
- 重なり最適化: IoU Loss / GIoU(Rezatofighi et al. CVPR 2019)— 検出ボックスの重なりを直接最適化(関連: [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation))。
- ラベルの不確かさ: Label Smoothing(Szegedy et al. CVPR 2016 で普及)— 正解確率を1でなく0.9などに緩め、過信を抑える(関連: [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration))。

## 論文の実験結果(定量データ)

Focal Loss 論文(ICCV 2017)は、提案した一段階検出器 RetinaNet を COCO で評価しました。報告によると、通常の交差エントロピーで学習した一段階検出器が背景に押し流されて精度が伸びないのに対し、Focal Loss を使うと box AP がおよそ 39% 前後に達し、当時の二段階検出器(Faster R-CNN 系)の精度に並びつつ一段階の速度を保ったとされています。集中度パラメータ γ のアブレーションでは、γ=0(普通の交差エントロピー)から γ=2 へ上げると AP が数ポイント改善し、γ が大きすぎても効果が頭打ち・悪化することが示され、「難サンプルへの集中度には最適値がある」ことが定量的に確認されました。

距離学習側では、FaceNet(CVPR 2015)が Triplet Loss を用いて顔認証ベンチマーク LFW で当時としては極めて高い精度(報告でおよそ 99% 台)を達成し、Triplet Loss が大規模な顔埋め込み学習に有効であることを示しました。ただし三つ組の選び方(どの negative を選ぶか、いわゆる hard negative mining)が性能を大きく左右し、選び方が悪いと学習が停滞することも報告されています。これは「損失の式だけでなく、損失を計算する対象(サンプリング)も同じくらい重要」という普遍的な教訓です。

数値の読み方として、損失関数の論文は「同じモデル・同じデータで損失だけ差し替えたときの指標差」を示すのが基本です。Focal の数ポイント、Label Smoothing の校正改善などは小さく見えますが、アーキテクチャを変えずに損失の定義だけで得られる改善である点に価値があります。

## メリット・トレードオフ・限界

- 損失と指標の不一致: 学習で最小化する損失と、最終的に評価したい指標(精度、mAP など)はしばしば一致しません。微分可能性のために代理損失(surrogate)を使うので、損失が下がっても目的の指標が伸びないことがあります。
- 外れ値とのトレードオフ: MSE は外れ値に弱く、MAE は収束が雑。Huber が折衷だが閾値調整が要る。
- 不均衡への弱さ: 素の交差エントロピーは多数クラスに引きずられる。Focal や重み付けで補う。
- 過信(校正不良): 交差エントロピーは正解確率を1へ押し続けるため過信を生みやすい。Label Smoothing 等で緩和(関連: [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration))。
- 複数損失の重み付け: 実システムは分類+回帰+補助損失を足し合わせるが、その重み比の決定が難しく、性能を大きく左右します。

## 発展トピック・研究の最前線

- 損失の自動設計・探索: 損失そのものをメタ学習や探索で見つける研究。
- 不確実性を含む損失: 予測の分散も学ばせ、難しいサンプルの重みを自動調整する(aleatoric uncertainty の学習)。
- ランキング・指標直結型損失: mAP や IoU を直接最適化する微分可能な近似。
- 頑健・公平性志向の損失: ノイズラベルや分布シフト、グループ間公平性を考慮した損失設計。

## さらに学ぶための関連トピック

- [画像分類](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0391_image-classification)
- [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation)
- [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining)
- [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration)
- [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods)
