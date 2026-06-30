---
title: "異常検知"
---

# 異常検知

## ひとことで言うと

異常検知(anomaly detection)とは、大多数の「正常」なデータの中から、それと毛色の違う「異常」なデータを見つけ出す技術です。ここで「異常」とは故障・不正・見たことのない入力など、平常時のパターンから外れたものを指します。自動運転では、学習時に想定していなかった物体やシーン(分布外、OOD: out-of-distribution)を検出して安全に対処するための基盤技術になります。

## 直感的な理解

工場のラインに流れる製品をイメージしてください。99.9%は正常品で、ごくたまに傷物が混じります。傷物のサンプルはほとんど手に入らないので、「傷物とは何か」を直接教えるのは難しい。そこで「正常品とはどういうものか」だけを大量に学び、その正常像から外れたものを異常と判定します。

これが異常検知の本質的な難しさであり面白さです。普通の分類は「犬」と「猫」のように両方の例から境界を学びますが、異常検知では異常の例がほとんど(あるいは全く)ない。だから「正常の輪郭」を覆う風船を作り、その外に出たものを異常とみなす、という片側だけのアプローチになります。これを one-class 分類(片側クラス分類)と呼びます。

## 基礎: 前提となる概念

異常検知の問題設定はラベルの有無で分かれます。

- 教師なし(unsupervised): ラベルは一切なし。データの大部分が正常という前提で、少数の外れ値を見つける。
- 半教師あり(semi-supervised / one-class): 正常データのみでモデルを学習し、テスト時に異常を判定する。実用上いちばん多い設定です。
- 教師あり(supervised): 正常・異常両方にラベルがある。異常が極端に少ない不均衡分類になり、これは異常検知というより不均衡分類問題に近い。

評価指標も独特です。異常は希少なので、単純な正解率(accuracy)は役に立ちません(全部「正常」と答えても99.9%正解になってしまう)。よく使うのは次の2つです。

- AUROC(Area Under the ROC Curve): 異常スコアをしきい値でずらしながら、真陽性率(本物の異常を当てる率)と偽陽性率(正常を誤って異常とする率)の関係を描いた曲線の下の面積。0.5 が当てずっぽう、1.0 が完璧です。正常と異常をどれだけ順位付けできているかを表します。
- AUPRC(Area Under the Precision-Recall Curve): 異常が極端に少ないとき、ROC より厳しく実力を測れます。

「異常スコア」とは、各サンプルがどれだけ異常らしいかを表す実数値です。多くの手法はこのスコアを出し、しきい値を超えたら異常と判定します。

## 仕組みを詳しく

代表的な3つの考え方を見ます。

### 1. 再構成ベース(autoencoder)

オートエンコーダ(autoencoder)は、入力をいったん低次元に圧縮(エンコード)してから元に戻す(デコード)ニューラルネットです。たとえば 28x28=784 次元の画像を 32 次元のベクトルに潰し、また 784 次元へ復元します。

正常データだけで学習すると、ネットは「正常な画像を上手く復元する」ことを覚えます。すると、テスト時に異常な画像を入れると、見たことのないパターンなので復元に失敗し、入力と出力の差(再構成誤差、reconstruction error)が大きくなります。この誤差を異常スコアにします。

```
正常入力 → [圧縮] → 32次元 → [復元] → 出力   差は小さい(≒0)
異常入力 → [圧縮] → 32次元 → [復元] → 出力   差は大きい → 異常!
```

数式では再構成誤差は ||x - x̂||² と書きます。x は入力、x̂(エックスハット)はネットが復元した出力、|| · ||² はベクトルの各要素の差を二乗して足した量(二乗ユークリッド距離)で、要するに「どれだけずれたか」の総量です。

### 2. 距離・密度ベース(古典)

- One-Class SVM(Schölkopf et al. 2001): 特徴空間で原点と正常データの間に「できるだけ大きなマージンの超平面」を引き、正常データを片側に囲い込む。外に出たら異常。
- SVDD(Support Vector Data Description, Tax & Duin 2004): 正常データを最小半径の球(超球、hypersphere)で囲み、球の外を異常とする。
- kNN・LOF(Local Outlier Factor, Breunig et al. 2000): あるデータの近傍密度が周りより低ければ異常とみなす。

これらは特徴量が良ければ強いですが、生の高次元データ(画像など)には弱い。良い特徴を人手で設計するのが難しいからです。

### 3. Deep SVDD(深層化)

Ruff et al., "Deep One-Class Classification" (Deep SVDD), ICML 2018 は、SVDD の「正常を球で囲む」発想をニューラルネットと融合させました。ニューラルネット φ(・; W) で入力を特徴空間へ写し、その空間であらかじめ決めた中心点 c に正常データを集めるよう学習します。

目的関数(最小化したい量)はおおまかに次の形です。

```
min over W :  (1/N) Σ_i || φ(x_i; W) - c ||²  +  正則化項
```

意味を言葉にすると、「各正常サンプル x_i をネットに通した出力 φ(x_i) が、中心 c にできるだけ近づくよう、ネットの重み W を調整する」。Σ_i は全 N 件の正常サンプルについての和、c は固定された中心ベクトルです。学習後、テストサンプルの異常スコアは中心からの距離 ||φ(x) - c||² で測ります。中心から遠いほど異常です。

ここで注意点があり、何も制約しないとネットは「全部を中心 c に潰す(定数を出力する)」自明解に落ちてしまいます(hypersphere collapse と呼ぶ)。これを防ぐため、バイアス項を使わない・有界でない活性化(ReLU など)を使う、といった設計上の工夫が必要だと論文は指摘しています。

## 手法の系譜と主要論文

- 古典: One-Class SVM(Schölkopf et al. 2001)、SVDD(Tax & Duin 2004)、LOF(Breunig et al. 2000)、Isolation Forest(Liu et al. 2008、ランダムな分割で孤立しやすいものを異常とみなす木のアンサンブル、関連: [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods))。
- 深層・距離: Deep SVDD(Ruff et al. ICML 2018)。
- 深層・再構成/敵対: AnoGAN(Schlegl et al. 2017、生成器が再現できない部分を異常とする)、Deep Autoencoding Gaussian Mixture(DAGMM, Zong et al. ICLR 2018)。
- OOD 検知(分類器を流用): Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples" (MSP), ICLR 2017(arXiv:1610.02136)— ソフトマックスの最大確率が低いものを OOD とみなす素朴で強いベースライン。ODIN(Liang et al. ICLR 2018, arXiv:1706.02690)、Mahalanobis 距離法(Lee et al. NeurIPS 2018)。
- 自己教師あり×異常: CSI(Tack et al. NeurIPS 2020)など、対照学習で正常表現を作ってから検出する流れ(関連: [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining))。

## 論文の実験結果(定量データ)

Deep SVDD 論文(ICML 2018)は MNIST と CIFAR-10 を使い、「ある1クラスだけを正常、残り全クラスを異常」とする one-vs-rest 設定で評価しました。評価指標は AUROC(%)。報告によると、MNIST では平均でおよそ 94〜95% 前後、CIFAR-10 では平均でおよそ 64〜65% 程度の AUROC を達成し、従来の浅い OC-SVM やオートエンコーダ系のベースラインを上回ったとされています。CIFAR-10 の数値が MNIST より大きく低いのは、自然画像のほうが背景や色のバリエーションが豊かで「正常の輪郭」を1つの球で表すのが難しいためで、自然画像の異常検知の難しさを端的に示しています。

OOD 検知側では、Hendrycks & Gimpel の MSP ベースライン(ICLR 2017)が「学習済み分類器のソフトマックス最大値を見るだけ」という単純さにもかかわらず、in-distribution(学習分布内)と OOD を AUROC で高い水準で見分けられることを示し、以後の研究の出発点になりました。ODIN や Mahalanobis 法はこれをさらに数ポイント改善したと報告されています。数値の比較で大事なのは、「AUROC が 0.5 に近ければ無力、1.0 に近いほど良い」という基準と、データセット間で難易度が桁違いに変わる(MNIST は簡単、自然画像は難しい)点です。

## メリット・トレードオフ・限界

メリットは、異常の例がなくても運用できること。希少で事前収集が困難な異常(新種の故障、未知の障害物)に対応できます。

限界とトレードオフ:

- 「正常」の定義が曖昧: 何を正常とみなすかは恣意的で、データの偏りがそのまま判定の偏りになります。
- しきい値設定: 異常スコアのどこで線を引くかは、見逃し(偽陰性)と誤報(偽陽性)のトレードオフ。安全クリティカルでは見逃しを極端に避けたいので誤報が増えがちです(関連: [SOTIF(意図機能の安全性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0784_safety-of-the-intended-functionality))。
- 自然画像での性能限界: 上記の CIFAR-10 の数値が示す通り、高次元・多様な入力では難易度が跳ね上がります。
- 校正の問題: ニューラルネットは OOD 入力に対しても自信満々の確率を出しがちで、ソフトマックスベースの検知が当てにならないことがあります(関連: [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration))。
- 概念ドリフト: 時間とともに「正常」が変化すると(季節・新製品)、再学習が必要になります。

## 発展トピック・研究の最前線

- OOD 検知と不確実性推定の融合: アンサンブルやベイズ的手法でモデルの不確実性を測り異常判定に使う(関連: [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods))。
- 基盤モデルの埋め込み利用: CLIP など大規模事前学習モデルの特徴空間で距離を測ると、ゼロショット的に異常検知できる(関連: [ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning))。
- 産業画像異常(MVTec AD など): 製造業向けに、画像のどこが異常かを画素単位で局在化(localization)するタスクが盛んです。PatchCore などが高い性能を報告しています。
- 自動運転の安全監視: 走行中にセンサ入力が学習分布から外れたことを検知し、減速・停止・人間への移譲(フォールバック)につなげる運用研究。

## さらに学ぶための関連トピック

- [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods)
- [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining)
- [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration)
- [SOTIF(意図機能の安全性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0784_safety-of-the-intended-functionality)
- [ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning)
