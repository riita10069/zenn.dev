---
title: "ゼロショット学習"
---

# ゼロショット学習

## ひとことで言うと

ゼロショット学習(zero-shot learning、ZSL)とは、学習時に一度も例を見たことのないクラスを、補助的な知識を頼りに認識する手法です。「ゼロショット」は「そのクラスの教師例がゼロ枚でも当てる」という意味。たとえばシマウマの画像を一度も学習に使わなくても、「シマウマは縞模様のある馬に似た動物」という言葉の知識から、初見のシマウマ画像を「シマウマ」と当てる、といったことを目指します。近年は CLIP のような大規模基盤モデルがこれを実用レベルに押し上げました。

## 直感的な理解

子どもに「ペンギンは黒と白で、飛べない鳥で、立って歩く」と言葉で教えれば、本物のペンギンを見たことがなくても初めて動物園で見たときに「あれペンギンだ」と分かります。人間は、視覚の例そのものではなく、言葉や属性という別経路の知識を使って未知を認識できる。

ゼロショット学習はこの能力を機械にやらせます。鍵は「見たことのあるクラス」と「見たことのないクラス」を橋渡しする共通の意味空間です。画像から直接クラス名へ対応づけるのではなく、いったん「属性」や「言葉のベクトル」という中間の意味表現に写してから、その意味表現を介して未知クラスにたどり着きます。橋さえあれば、橋の向こう側に行ったことがなくても渡れる、というわけです。

## 基礎: 前提となる概念

用語を整理します。

- 既知クラス(seen classes): 学習時に画像とラベルがあるクラス。
- 未知クラス(unseen classes): 学習時に画像がなく、テスト時にだけ現れるクラス。
- 補助情報(side information / semantic embedding): 既知と未知を橋渡しする知識。代表は2種類。
  - 属性ベクトル(attribute vector): 「縞模様あり/なし」「4本足/2本足」など、人間が定義した特徴の有無を並べたベクトル。各クラスにこのベクトルが付与される。
  - 単語埋め込み(word embedding): クラス名を言葉のベクトルにしたもの(word2vec など)。意味の近い言葉が近くに配置される空間。

関連する設定の区別:

- ゼロショット(0例)/ ワンショット(1例)/ フューショット(few-shot、数例)。few-shot は少数なら見られるのに対し、zero-shot は完全に0例。
- 通常ZSL(conventional ZSL): テスト時に未知クラスだけが現れる前提(やや非現実的)。
- 一般化ZSL(generalized ZSL, GZSL): テスト時に既知クラスと未知クラスが混在する、より現実的で難しい設定。

評価指標は、未知クラスの精度に加え、GZSL では既知精度と未知精度の調和平均 H を使います。調和平均は、片方だけ高くても引き上げられないので「既知に偏らず未知も当てられているか」をきちんと測れます。

## 仕組みを詳しく

### 属性ベースの古典手法(DAP)

Lampert et al., "Learning to Detect Unseen Object Classes by Between-Class Attribute Transfer", CVPR 2009 が ZSL の出発点です。DAP(Direct Attribute Prediction)の流れ。

```
1. 各クラスに属性ベクトルを人手で定義
   例: シマウマ = [縞=1, 4本足=1, 尻尾=1, 海に住む=0, ...]
2. 既知クラスの画像で「画像 → 各属性の有無」を学習
   (属性予測器。これは既知/未知に依らず使える)
3. テスト: 未知クラス画像から属性を予測 → 属性ベクトルが最も近いクラスへ
```

画像をいったん属性空間に写し、属性空間で最も似たクラスを選ぶ。属性予測器は既知クラスで学びますが、属性そのものはクラスをまたいで共通なので、未知クラスにも転用できる。これが「属性転移(attribute transfer)」です。

### 埋め込みベースと CLIP

属性は人手定義が大変です。そこで、画像とテキストを同じ空間に埋め込んでしまう発想が登場します。その決定版が CLIP(Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", 2021, arXiv:2103.00020)。

CLIP は、インターネットから集めた約4億組の(画像, テキスト)ペアで学習します。画像エンコーダと文章エンコーダの2つを用意し、対照学習(contrastive learning)で「対応する画像と文は埋め込み空間で近く、対応しない組は遠く」なるよう訓練します(関連: [損失関数の概観](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0593_loss-functions-overview)、[自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining))。

```
バッチ内 N 組の (画像, 文):
  画像埋め込み I_1..I_N と 文埋め込み T_1..T_N を計算
  正しいペア (I_i, T_i) の類似度を上げ、
  間違ったペア (I_i, T_j, i≠j) の類似度を下げる
```

学習後のゼロショット分類はこうです。分類したいクラス名を「a photo of a {クラス名}」という文に埋め込み、入力画像の埋め込みと最も類似度の高い文(クラス)を選ぶ。クラスを文で与えられるので、学習時に一度も見ていないクラスでも、その名前さえあれば分類できます。属性を人手定義する必要がなく、自然言語がそのまま橋になる。これが CLIP の革新でした。

tensor のイメージでは、画像埋め込みも文埋め込みも同じ次元(たとえば512次元)のベクトルに正規化され、内積(コサイン類似度)で近さを測ります。

## 手法の系譜と主要論文

- 属性転移: Lampert et al., DAP/IAP, CVPR 2009(および PAMI 2014 の拡張)。Animals with Attributes(AwA)データセットを提供。
- 単語埋め込み利用: DeViSE(Frome et al., NeurIPS 2013)— 画像を word2vec 空間へ写す。ConSE(Norouzi et al., ICLR 2014)。
- 埋め込み学習の整理: ESZSL(Romera-Paredes & Torr, ICML 2015)、SJE(Akata et al., CVPR 2015)。
- 生成ベース: f-CLSWGAN(Xian et al., CVPR 2018)— 未知クラスの特徴を生成して擬似的に学習データを作る。GZSL の偏り対策として有力。
- 大規模対照学習: CLIP(Radford et al. 2021, arXiv:2103.00020)、ALIGN(Jia et al. ICML 2021)。これ以降、ZSL は「専用手法」から「基盤モデルの能力」へと重心が移りました。
- セグメンテーションへの波及: SAM(arXiv:2304.02643)などプロンプト型の汎用知覚(関連: [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation))。

## 論文の実験結果(定量データ)

古典 ZSL の標準ベンチマークは AwA2、CUB(鳥の細粒度)、SUN(シーン)など。Xian et al. の整理(PAMI 2018, "Zero-Shot Learning — A Comprehensive Evaluation")によると、属性ベースの古典手法は通常ZSLでは AwA で未知精度およそ 60% 前後を出すものの、現実的な GZSL 設定にすると未知クラスの精度が大きく落ち、既知クラスへ予測が偏る(既知精度は高いが未知精度が低い)問題が顕著でした。調和平均 H で見ると初期手法は低く、この「既知への偏り」が GZSL の中心課題だと定量的に示されました。生成ベースの f-CLSWGAN はこの H を大きく改善したと報告されています。

CLIP(2021)のインパクトは桁違いでした。報告によると、CLIP はゼロショット(ImageNet の画像を一切ラベル付きで学習せず、クラス名を文で与えるだけ)で ImageNet の Top-1 精度およそ 76% を達成し、これは ImageNet を実際にラベル付きで学習した ResNet-50 と同等水準でした。さらに30種類以上のデータセットでゼロショット評価し、多くで強力な転移性能を示しました。とりわけ重要なのは分布シフトへの頑健さで、ImageNet-Sketch(スケッチ画)や ImageNet-R(イラスト等)のような見た目が大きく変わったデータで、通常の教師あり ResNet が大幅に精度を落とすのに対し、CLIP のゼロショットは崩れにくいと報告されました(関連: [画像分類](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0391_image-classification)、[異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection))。

数値の読み方として、古典 ZSL の「AwA で60%」と CLIP の「ImageNet で76%」は対象もスケールも違うので直接比較はできません。前者は数十クラスの限定設定、後者は1000クラスの大規模設定です。CLIP の意義は精度の絶対値というより、「専用に作り込まなくても、巨大データで学んだ基盤モデルがゼロショットで実用精度を出す」というパラダイム転換にあります。

## メリット・トレードオフ・限界

メリット:
- 未知クラスのラベル収集が要らない。新クラス追加が「名前を足すだけ」で済む(関連: [能動学習(Active Learning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0874_active-learning) の対極的な省ラベル戦略)。
- 自然言語で柔軟に指定できる(CLIP系)。クラス定義を後から自由に変えられる。

トレードオフ・限界:
- 既知への偏り(GZSL): テストに既知が混ざると、未知クラスがほぼ当たらなくなる根強い問題。
- 補助情報の質に依存: 属性定義が雑、あるいはクラス名が曖昧だと精度が出ない。橋が悪ければ渡れない。
- 細粒度(fine-grained)の弱さ: 似た鳥の種など、言葉や粗い属性では区別しきれない対象に弱い。
- 基盤モデルのバイアスと不透明性: CLIP は学習データ(ウェブ)の偏りをそのまま受け継ぎ、安全クリティカルな用途では不確実性の扱いが課題(関連: [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration)、[SOTIF(意図機能の安全性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0784_safety-of-the-intended-functionality))。
- ドメイン差: 自動運転の特殊センサ(LiDAR、夜間カメラ)など、ウェブ画像と分布が大きく違う領域への転移は限定的。

## 発展トピック・研究の最前線

- プロンプト学習(prompt tuning): CLIP のテキストプロンプトを学習で最適化(CoOp など)し、ゼロショット精度をさらに上げる。
- オープンボキャブラリ認識: 固定クラスを捨て、任意の語彙で検出・セグメンテーションする(関連: [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation))。
- マルチモーダル基盤モデル: 画像・テキスト・音声などを統合し、より広い未知への汎化を狙う。
- 安全領域での活用と監視: ゼロショットで未知物体を拾いつつ、不確実性を測って危険時にフォールバックする運用(関連: [異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection)、[SOTIF(意図機能の安全性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0784_safety-of-the-intended-functionality))。

## さらに学ぶための関連トピック

- [画像分類](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0391_image-classification)
- [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining)
- [損失関数の概観](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0593_loss-functions-overview)
- [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation)
- [異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection)
- [SOTIF(意図機能の安全性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0784_safety-of-the-intended-functionality)
