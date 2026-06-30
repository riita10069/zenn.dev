---
title: "回転予測などpretext task"
---

# 回転予測などpretext task

## ひとことで言うと

ラベルのない画像から特徴を学ぶ初期の自己教師あり学習です。「画像を90度ずつ回して何度回したか当てさせる」「画像を9枚のパズルに切ってバラバラに並べ、正しい並びを当てさせる」といった、本来の目的ではないが解くと特徴が身につく「お題 (pretext task, 代理タスク)」を解かせます。お題を解くにはネットワークが物体の形・向き・空間配置を理解する必要があるため、副産物として汎用的な視覚特徴が得られ、それを別タスクに転用します。

## 直感的な理解

強い画像認識を作るには大量のラベル付き画像が必要でした。ImageNet は人手で120万枚以上にラベルを付けた巨大データセットで、これがあったから画像認識が飛躍しました。しかしラベル付けは膨大なコストがかかります。「ラベルなしの画像なら無限にあるのに活かせないか」が自己教師あり学習の動機です。

鍵は「正解を人手で与えなくても、データの加工そのものから自動で正解が作れる」点です。画像を回転させたとき「何度回したか」は、回転させたプログラム自身が知っています。だから人手のラベルなしで正解が自動的に手に入ります。これが「自己」教師の意味です。そして「回転を当てる」という建前のお題を真剣に解くには、ネットワークは「空は上、タイヤは下、顔の目は上のほう」といった物体の正しい姿勢や構造を理解せねばならず、結果として物体認識に役立つ特徴が育ちます。お題の正解(回転角)自体はどうでもよく、解く過程で育つ特徴が目的です。

ここに pretext task の哲学が表れています。「お題そのもの」には誰も興味がありません。誰も「画像が何度回っているか当てるアプリ」を欲しがってはいません。欲しいのは、お題を解くために嫌でも身につけてしまう「物体を理解する力」です。お題は、その力を引き出すための口実(pretext)にすぎないのです。

## 基礎: 前提となる概念

自己教師あり学習 (self-supervised learning, SSL) とは、人間が正解を与えるのではなく、データの加工から自動で正解(教師信号)を作る枠組みです。

プレテキストタスク (pretext task) の pretext は「口実・建前」の意味です。「本当に解きたいタスク(画像分類など、下流タスク downstream task と呼ぶ)ではないが、それを口実に特徴を学ばせるための、自動で正解が作れるお題」を指します。

転移の使い方は2段階です。(1) pretext task で大量のラベルなし画像から backbone(特徴抽出部。画像を特徴ベクトルに変換する CNN の本体)を事前学習 (pretraining)。(2) 本当に解きたいタスク用に、少量のラベル付きデータで微調整 (fine-tuning) するか、backbone を凍結して上に線形分類器だけ載せて性能を測る linear probing(linear evaluation)を行います。linear probing の精度は「特徴がどれだけ線形分離しやすい良い表現か」の標準的な指標です。fine-tuning は backbone も含めて全部学習し直すので性能は高いが「特徴そのものの良さ」は測りにくく、linear probing は特徴の質を純粋に測れる、という使い分けがあります。

ショートカット (shortcut) も重要な概念です。ネットワークが「お題を本質理解なしに解くズルの手がかり」を見つけてしまうと、狙った特徴が育ちません。例えばジグソーで、パッチ境界の色の連続性だけ見て並びを当てると、物体を理解せずに解けてしまいます。pretext task の設計はこのショートカット潰しとの戦いでもあります。「お題が簡単すぎてズルで解けると、特徴が育たない。難しすぎると学習が進まない」という、ちょうど良い難易度の設計が腕の見せどころです。

## 仕組みを詳しく

### 回転予測 (RotNet)

1. 1枚の画像を 0°・90°・180°・270° の4通りに回転させる。回転はプログラムが行うので正解(0/1/2/3)は自動で分かる。
2. CNN に回転後の画像を入力し、「4つの回転角のどれか」を当てる4択分類として学習する。損失は交差エントロピー。
3. 正解率を上げるにはネットワークが物体の正しい向き(空・地面・物体の上下関係)を理解する必要がある。

形状イメージ: 入力 `[B, 3, 224, 224]`(B枚のRGB画像)→ CNN → 出力 `[B, 4]`(4回転角のスコア)。学習後、最後の分類層は捨て、途中の特徴抽出部だけを学習済み backbone として下流タスクに転用します。RotNet が驚かれたのは、これほど単純なお題(4択を当てるだけ)で、当時の凝った手法に匹敵する特徴が得られた点です。

### ジグソーパズル (Jigsaw)

1. 画像から正方領域を取り、3×3の9マスのパッチに切る。
2. 9枚を、あらかじめ決めた並べ替えパターン(例: 9! = 362880 通りの順列の中からハミング距離が大きいものを選んだ 1000 通りの中の1つ)でシャッフルする。どのパターンかはプログラムが知っている。
3. 9枚のシャッフル済みパッチを(各パッチを重み共有の別々の枝に通す siamese 構造で)入力し、特徴を結合して「どの並べ替えパターンか」を当てる分類として学習する。
4. 正解には各パッチが物体のどの部分か(耳・胴体)を理解し、正しい空間配置を推論する必要がある。

ショートカット対策として、パッチ間に隙間(gap)を空けて境界の連続性で当てられないようにし、各パッチの色を独立に正規化して低レベルの色統計(色収差 chromatic aberration など)に頼れなくする工夫が要ります。これらを外すとネットワークは物体を理解せず「境界がつながる並べ方」だけで解いてしまい、特徴が育ちません。

before / after で pretext task の役割を整理:
- before(ラベル付き教師あり): 人手で「これは犬」と教える → 大量のラベルが必要。
- after(pretext task): プログラムが自動で「これは90°回転」と正解を作る → ラベル不要。お題を解く過程で物体理解の特徴が育ち、それを転用する。

他の初期 pretext task: 相対位置予測(中央パッチに対する周囲パッチの位置関係を8方向で当てる、Doersch 2015)、彩色([colorization (彩色) プレテキスト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0034_colorization-pretext)。白黒画像に色を塗らせる)、欠損復元([Context Encoder (inpainting)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0035_context-encoder-inpainting)。穴を開けて埋めさせる)など。これらはすべて「データ自身から正解を自動生成する」という共通の発想に立っています。

## 手法の系譜と主要論文

- Doersch, Gupta, Efros, "Unsupervised Visual Representation Learning by Context Prediction" (ICCV 2015, arXiv:1505.05192)。中央パッチに対する周囲パッチの相対位置(8方向)を当てる、pretext task の先駆けの一つ。深層 SSL の事実上の出発点。
- Noroozi & Favaro, "Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles" (ECCV 2016, arXiv:1603.09246)。3×3パッチの並べ替えパターンを当てる。位置予測より多くの空間配置の推論を要求する設計で、ショートカット対策の重要性も明確化した。
- Zhang et al., "Colorful Image Colorization" (ECCV 2016) と Pathak et al., "Context Encoders" (CVPR 2016, [Context Encoder (inpainting)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0035_context-encoder-inpainting))。彩色系・復元系の代表で、同時期に「生成的に欠けた情報を補う」タイプの pretext task を確立。
- Gidaris, Singh, Komodakis, "Unsupervised Representation Learning by Predicting Image Rotations" (RotNet, ICLR 2018, arXiv:1803.07728)。4通りの回転角を当てる極めて単純なお題で、複雑な手法に匹敵する特徴を達成。「pretext は凝らずとも良い」を示し、SSL のベースラインとして広く使われた。
- Kolesnikov, Zhai, Beyer, "Revisiting Self-Supervised Visual Representation Learning" (CVPR 2019, arXiv:1901.09005)。各 pretext task を公平な条件で再評価し、「アーキテクチャ(特に幅広い ResNet)を変えると手法間の優劣が入れ替わる」「pretext task 本来の精度と下流転移精度は必ずしも一致しない」という重要な知見を示した。SSL の評価作法を変えた。
- その後 [CPC (Contrastive Predictive Coding)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0037_contrastive-predictive-coding) の InfoNCE を起点とする対照学習(SimCLR, MoCo)が転移性能で pretext task を上回り主役を奪い、さらに He et al. "Masked Autoencoders" (MAE, CVPR 2022, arXiv:2111.06377) など欠損復元系が Transformer 時代に再び主流化しました。pretext task は「歴史的な出発点」として位置づけられます。

系譜を一言でまとめると、位置・回転・パズルといった「単一の手がかりを当てる」初期 pretext task → 複数ビューの一致を学ぶ対照学習 → 入力の大半を隠して復元する大規模マスク復元、という3世代の流れです。

## 論文の実験結果(定量データ)

標準的な評価は、pretext で事前学習した特徴を ImageNet 分類(linear probing)や PASCAL VOC の検出・セグメンテーションに転移し、その精度を測るものです。

- RotNet (Gidaris 2018): ImageNet で、AlexNet の中間層(conv4 あたり)の特徴を凍結して線形分類すると top-1 でおよそ 38〜43% 程度(層により変動。最良層で約 43.8%)。当時の他の pretext task(Jigsaw, Context Prediction など)を上回るか同等で、しかも実装が圧倒的に単純。PASCAL VOC 検出への転移でも教師あり事前学習に近い水準を報告。
- Jigsaw (Noroozi 2016): PASCAL VOC 2007 検出で mAP(mean Average Precision、平均適合率。検出の正確さの標準指標で、高いほど良い)およそ 51% 前後を報告し、当時のランダム初期化(約 44%)や他の SSL を上回った。
- Context Prediction (Doersch 2015): VOC 2007 検出で mAP 約 51% を報告し、ランダム初期化を大きく上回り、教師あり事前学習(約 57%)との差を縮めた最初期の成果。

数値の意味の補足: VOC 検出で「ランダム初期化 44% → SSL 51% → 教師あり 57%」という並びは、「pretext task が、ラベルなしだけでランダム初期化と教師あり事前学習のちょうど中間まで来た」ことを示し、当時としては大きな前進でした。

アブレーション・知見:
- RotNet の回転は 4 通り(0/90/180/270)が最良で、8通り(45°刻み)に増やすと補間で画像がぼけて手がかりが乱れ、精度が下がる。直交回転は補間不要で画像が劣化しないことが効いている。2通り(0/180)では情報が足りず劣る。
- Jigsaw は順列の数(クラス数)を増やすと難しくなりすぎ、減らすと易しすぎる。1000通り前後が良いバランスと報告。ショートカット対策(パッチ間の隙間、色正規化)を外すと境界の連続性で当てる「ズル」が起き、転移精度が大きく落ちる。
- Kolesnikov et al. (2019): 同じ pretext task でも ResNet の幅を広げると下流精度が大きく伸びる一方、手法間の順位が入れ替わる。さらに「pretext task の精度が高い=下流転移が良い、とは限らない」ことを定量的に示し、「pretext task の比較は、固定されたネットワーク前提でしか意味をなさない」という、SSL ベンチマーク設計への重要な警鐘を鳴らした。
- 自然画像で向きが固定の領域(常に「地面が下」のシーン)では回転予測の手がかりが弱く効果が限定的、という弱点も指摘される。

## メリット・トレードオフ・限界

メリット:
- 人手ラベルが一切不要でラベルなし画像から特徴を学べる。
- 実装が単純(特に回転予測)で計算も軽く、理解しやすい。SSL を学ぶ最初の教材として最適。
- 自己教師あり学習の歴史的出発点で、後の手法の発想の土台になった。

トレードオフ・限界:
- お題に特化したショートカットを学ぶリスク(パッチ境界の連続性、低レベル色統計、色収差で当てる等)があり、本質特徴に育たないことがある。
- 物体の向きが固定の画像(車載カメラ映像など、常に地面が下で向きが一定)や対称物体(コップ・ボールなど回しても同じに見えるもの)では回転予測が効きにくい。
- 単一の手がかり(回転・順列)に特化しすぎ、それ以外の構造を学びにくい。お題が捉える不変性が限定的。
- 最終的な転移性能で、後発の対照学習やマスク復元系に抜かれた。
- 研究上の課題として、pretext task の「お題の精度」と「下流転移の精度」が一致しないため、どのお題が良い特徴を生むかを事前に予測する理論がなく、経験的な試行錯誤に頼る面が残ります。

## 発展トピック・研究の最前線

- マスク復元系の再興: 入力の一部を隠して復元する MAE / BEiT が、Vision Transformer と大規模データの下で「欠損を補う」タイプの pretext task を現代的に蘇らせ、SSL の主流の一角になりました(復元系は [Context Encoder (inpainting)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0035_context-encoder-inpainting) の直系)。MAE は画像の 75% を隠して残りから復元させる極端な設計で、ImageNet fine-tuning で 87% 超を達成しました。
- 対照学習との融合: 回転やジグソーの「不変性/同変性(変換に対して特徴がどう変わるべきか)」を対照学習の補助損失として組み合わせる研究(例: 回転に対する同変性を保つ表現)があり、pretext task の知見は今も部分的に活用されています。
- ベンチマーク方法論: Kolesnikov et al. 以降、SSL の比較は「アーキテクチャ・データ拡張・評価プロトコルを揃える」ことが必須という認識が確立し、現在の SSL 研究の評価作法の基礎になっています。pretext task 時代の「比較が公平でないと結論が逆転する」という教訓が、今の厳密な評価文化を生みました。

## さらに学ぶための関連トピック

- [colorization (彩色) プレテキスト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0034_colorization-pretext)
- [Context Encoder (inpainting)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0035_context-encoder-inpainting)
- [CPC (Contrastive Predictive Coding)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0037_contrastive-predictive-coding)
- [Deep InfoMax (DIM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0042_deep-infomax)
