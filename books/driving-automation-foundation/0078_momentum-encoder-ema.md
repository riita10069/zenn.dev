---
title: "momentum encoder / EMA target"
---

# momentum encoder / EMA target

## ひとことで言うと
学習する側のモデル(student)とは別に、その「ゆっくり追従するコピー」(teacher)をもう1つ用意し、teacher を学習の目標(target)として使う仕組みです。teacher は勾配で直接学習せず、student の重みを少しずつ平均して取り込む(EMA、指数移動平均)ことで更新します。目標がフラフラ動かず安定するので、多くの自己教師あり手法(MoCo, BYOL, DINO)の共通の土台になっています。

## 直感的な理解
自分で問題を作り、自分で答え合わせをする状況を想像してください。もし「答え(目標)」を出すのも、それを「解く」のも、毎ステップ更新される同じネットワークだと、答えそのものが解くたびに動いてしまう。動く的を撃ち続けるようなもので、学習が落ち着きません。さらに、解く側と答える側が同じだと、両者が手を取り合って「全部同じ答え」に潰れる崩壊(→ [predictor (予測ヘッド非対称性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0083_prediction-head-predictor))も起きやすい。

そこで「答えを出す側」を、解く側とは切り離し、ゆっくりだけ動かす。teacher を student の「なめらかな平均的な姿」にしておけば、目標は急には変わらず、安定した的になります。平均化されたモデルは個々のステップのブレを吸収するので、単体より頑健で性能も出やすい、という経験則(古くは Polyak averaging、最適化で反復の平均を取ると収束が安定する)もここに乗っています。

## 基礎: 前提となる概念
EMA(Exponential Moving Average、指数移動平均): 過去の値を少しずつ混ぜて平滑化する更新方法です。新しい値を毎回ほんの少しだけ取り込み、大半は過去を保ちます。teacher の重みを、毎ステップ student の重みでわずかに更新する(例: 99.9% は元のまま、0.1% だけ student を取り込む)と、teacher は student の「最近の平均的な軌跡」をたどり、瞬間的なブレに動じません。

stop-gradient: teacher 側には勾配(学習の更新信号)を流しません。teacher は EMA でだけ動き、誤差逆伝播では一切学習しません。これにより teacher は「固定された目標」として振る舞います。EMA と stop-gradient はセットで、「自分では学ばず、student の平均だけで動く」目標を作ります。

自己蒸留(self-distillation): 蒸留(distillation, Hinton et al. 2015)はもともと「大きなモデル(teacher)の出力分布を小さなモデル(student)に真似させる」技術です。自己蒸留は、同じか似た構造の teacher の出力を student に真似させること。DINO はこれを SSL に応用し、teacher を student の EMA とすることで「ラベルなしの蒸留」を実現しました。

崩壊(collapse): 全出力が同一に潰れる失敗。EMA teacher は「逃げない安定した目標」を与えることで、特に負例を使わない手法での崩壊回避に寄与します。

## 仕組みを詳しく
2つのネットワークを用意します。

- student(オンライン、パラメータ θ): 普通に勾配で学習する側。
- teacher(モメンタム/ターゲット、パラメータ ξ): student の EMA で更新し、勾配では学習しない(stop-gradient)。

EMA 更新式:

```
ξ ← m · ξ + (1 - m) · θ
```

m はモメンタム係数(0〜1)で、1 に近いほど teacher はゆっくり動きます。MoCo は m=0.999、BYOL は学習中に m を 0.996→1.0 へコサインスケジュールで増やす、DINO も 0.996 付近から 1.0 へ増やす、といった設定が使われます。

数値例: m=0.999 のとき、毎ステップ teacher は student を 0.1% だけ取り込みます。teacher の重みは student のおおよそ直近 1/(1-m)=1000 ステップ分の指数加重平均に近く、瞬間的なブレを吸収します。m を 1 に近づけるほど目標は安定しますが、追従が遅すぎると student が学んだ進歩を teacher が取り込めず学習が停滞します。逆に m が小さすぎると teacher が student とほぼ同じになり、安定化の効果も崩壊回避の効果も失われます。学習序盤は student が急速に変わるので m をやや小さめ(追従を速く)にし、後半は m を 1 に近づけて目標を固める、というスケジュールが理にかなっています。

手法ごとの使われ方:

MoCo(Momentum Contrast)。対照学習(→ [InfoNCE / NT-Xent ロス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0061_info-nce-loss))では大量の負例が要りますが、毎回計算し直すと重い。そこで「キュー(queue、過去の特徴をためる待ち行列)」に過去ミニバッチの特徴をためて負例に使います。ただし student が速く変わると、キュー内の古い特徴と今の特徴で「ものさし」がズレてしまう(特徴の意味が時間とともに変質する、feature drift)。teacher を EMA でゆっくり動かし、キューに入れる特徴をすべて teacher で計算することで、古い特徴も今の特徴も一貫したものさしで比較でき、大バッチなしに大量の負例を使えます。MoCo の EMA の主目的は「特徴の一貫性(consistency of the dictionary)」です。

BYOL。負例なし。teacher の出力を student が predictor 越しに予測します(→ [predictor (予測ヘッド非対称性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0083_prediction-head-predictor))。teacher を EMA で動かすことで「逃げない目標」を与え、崩壊を防ぎつつ正例予測だけで学習します。BYOL の EMA の主目的は「崩壊回避と目標の安定」です。後に SimSiam が「EMA は必須でなく predictor + stop-gradient で足りる」と示したため、BYOL における EMA は「必須要素」から「性能を底上げする安定化装置」へと位置づけが変わりました。

DINO(self-distillation with no labels)。teacher も student も同じ画像の別ビューを見て、各自「どのプロトタイプっぽいか」の確率分布を出します。student は teacher の分布を当てるよう交差エントロピーで学習します。teacher は EMA 更新で、さらに2つの仕掛けを併用します。centering(中心化、teacher 出力から移動平均的な中心ベクトルを引き、特定の1次元へ全部偏るのを抑える)と sharpening(鋭利化、teacher 側の温度を student より低くして分布をはっきりさせる)。この2つは崩壊の2方向を互いに打ち消すよう設計されています。centering だけだと出力が一様分布に潰れ(全クラス均等)、sharpening だけだと1クラスに潰れる。両者を組み合わせることで、どちらの崩壊も避けます。DINO は ViT で使うと、教師なしなのに teacher の自己注意(self-attention)マップが物体の輪郭やセマンティックな領域を自然に捉えるという顕著な創発的性質を示しました。

共通の骨格を before/after で言うと、単一ネットワークが目標を兼ねる(before)と目標が動いて不安定・崩壊しやすい。EMA teacher を分離(after)すると目標が安定し、負例ありでも負例なしでも学習が安定します。

なぜ EMA が崩壊を防ぐのか、直感的な説明。崩壊は「student と target が一緒に同じ点へ走る」ときに起きます。EMA target は student の過去の平均なので、student が急に動いても target はゆっくりしかついてこない。この「遅れ」が、両者が同時に潰れることを物理的に妨げます。target は常に「少し前の student」であり、student は「少し前の自分」に追いつこうとし続けるので、定数解に静止しにくいのです。

## 手法の系譜と主要論文
Mean Teacher(Tarvainen & Valpola, NeurIPS 2017, "Mean teachers are better role models", arXiv:1703.01780)。半教師あり学習で、teacher を student の EMA とし、両者の予測を一致させる consistency 正則化を提案。「平均化したモデルは単体より良い目標になる」という、その後の SSL EMA teacher の直接の源流です。先行する Temporal Ensembling(過去エポックの予測の平均を目標にする)を、重みの EMA に置き換えてオンライン化したものと位置づけられます。

MoCo(He, Fan, Wu, Xie, Girshick, CVPR 2020, "Momentum Contrast for Unsupervised Visual Representation Learning", arXiv:1911.05722, Facebook AI Research)。モメンタムエンコーダ + キューによる負例供給。動機: 大量の負例を、一貫したものさしで、大バッチなしに供給したい。MoCo v2 では SimCLR の投影ヘッドと強拡張を取り込んで改良され、MoCo v3 では ViT への適用と学習安定化を扱いました。

BYOL(Grill et al., NeurIPS 2020, arXiv:2006.07733, DeepMind)。EMA teacher + predictor 非対称性で負例なし学習。後続の SimSiam(Chen & He, 2021)が「EMA は必須でなく、predictor + stop-gradient で足りる」と示し、EMA の役割が「必須」から「性能を底上げする安定化装置」へと位置づけ直されました。

DINO(Caron et al., ICCV 2021, "Emerging Properties in Self-Supervised Vision Transformers", arXiv:2104.14294, Facebook AI Research)。EMA teacher + centering/sharpening による自己蒸留で、ViT の創発的性質(教師なしセグメンテーション的な注意)を引き出しました。後継の DINOv2(Oquab et al., 2023)は同枠組みを大規模キュレーション済みデータで拡張し、fine-tuning 不要の汎用視覚基盤モデルへ発展しました。

## 論文の実験結果(定量データ)
評価指標は線形評価(エンコーダ凍結+線形分類器、Top-1 精度)に加え、DINO は k-NN 評価(特徴を凍結し、最近傍法で分類)も使います。k-NN は線形分類器すら学習しないので「特徴の素の質」をさらに直接測れます(→ 線形評価の説明は [InfoNCE / NT-Xent ロス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0061_info-nce-loss))。

MoCo(ResNet-50、ImageNet 線形評価、報告による)。MoCo v1 で約 60.6%、v2 で約 71.1%。さらに重要なのは転移性能で、PASCAL VOC や COCO の物体検出で、教師あり ImageNet 事前学習を上回る転移を報告。「ラベルなし事前学習が下流で教師ありに勝てる」初期の決定的証拠の一つです。EMA のモメンタム m のアブレーションでは、m=0.999 付近が最良で、m=0.9 では精度が大きく低下、m=0(EMA なし、teacher を毎ステップ student と同一にする)では学習がうまく進まず精度が大幅に落ちる(報告では学習が崩れる)とされました。EMA による特徴一貫性が効いていることを示します。

BYOL(ResNet-50、線形評価、報告による)。Top-1 約 74.3%。バッチサイズ 256〜4096 で精度がほぼ不変という頑健性も示しました。EMA を外す(m=0、target を student と同一にする)と崩壊・劣化することが報告され、BYOL では EMA が崩壊回避に寄与します(ただし後に SimSiam が predictor + stop-gradient だけでも代替可能と示しました)。m を 0.9〜0.999 の範囲で振ると、低すぎても高すぎても性能が落ち、適度な追従速度に最適があると報告されています。

DINO(報告による)。ViT-S/16 で線形評価 約 77%、k-NN 評価 約 74〜75%。同サイズの ResNet ベース SSL を上回り、特に k-NN 精度の高さは特徴がそのまま近傍構造として意味的にまとまっていることを示します。centering を外すと一様分布への崩壊、sharpening(温度)を不適切にするともう一方向(1クラス)の崩壊が起き、両者のバランスが崩壊回避の鍵だとアブレーションで示されました。teacher の EMA モメンタムを 0(EMA なし)にすると学習が崩れ、EMA teacher が安定した蒸留目標として不可欠であることも確認されています。さらに興味深い観察として、DINO では teacher の方が student より一貫して高精度に推移し、teacher を最終モデルとして使うのが最良という「teacher が student を上回る」現象が報告されました。

## メリット・トレードオフ・限界
メリット
- 目標(target)が安定し、学習が安定する(逃げない目標を与える)。
- 崩壊回避に寄与する(特に負例なし手法 BYOL/DINO で重要)。
- MoCo では大バッチなしに一貫した大量の負例を供給できる(特徴の意味的ドリフトを抑える)。
- 平均化されたモデルは単体より汎化しやすい傾向(Mean Teacher 以来の知見)。teacher 自体を最終モデルとして使えることも多い(DINO では teacher が student を上回る)。

トレードオフと限界
- teacher のコピーを別途保持する分、メモリを使う(2モデル分のパラメータ)。
- モメンタム係数 m の調整が必要で、大きすぎると追従が遅く学習が停滞、小さすぎると安定化・崩壊回避の効果が消える。スケジュール(学習中に m を上げる)も設計事項。
- BYOL では EMA・predictor・バッチ正規化のどれが効くか直感的に分かりづらく、SimSiam が「EMA は必須でない」と示したことで、必須要素ではなく性能向上装置と再評価された。
- DINO は centering/sharpening の温度バランスが崩壊回避の鍵で、調整に手間がかかる。
- EMA がなぜ崩壊を防ぐのかの完全な理論的特徴づけは未確立(線形モデルでの解析が中心)。
- 単画像ベースが基本で、時系列・行動・物理は別途設計が必要。

## 発展トピック・研究の最前線
EMA teacher は SSL の枠を超えて広く使われています。動画・予測型の SSL では「未来の本物の特徴」という目標を EMA teacher で安定化させる設計が自然で、LeCun らの JEPA(Joint Embedding Predictive Architecture)系(I-JEPA, V-JEPA)はまさに EMA target encoder + predictor で「一部から残りを特徴空間で予測する」枠組みです(→ predictor の非対称性は [predictor (予測ヘッド非対称性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0083_prediction-head-predictor)、未来予測は [行動条件付き潜在予測 (counterfactual rollout)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0001_action-conditioned-prediction))。data2vec(Baevski et al., 2022)は音声・画像・テキストを EMA teacher の潜在表現の予測という共通枠組みで統一しました。半教師あり学習・自己訓練(self-training)・強化学習(DQN の target network として、また方策の平均化として)・連合学習(モデル平均)・拡散モデル(サンプリング用に重みの EMA を使うのが標準)でも EMA/平均化は基本道具です。研究上の open question としては、最適なモメンタムスケジュールの理論的決定、EMA が崩壊回避に効く正確な条件、predictor 非対称性や正則化(VICReg/Barlow Twins)とのどの組み合わせが最良かといった点が残っています。動画・運転領域では、空間的 SSL より時間方向の予測との相性が良く、EMA teacher はその目標側の安定化機構として参照価値があります。

## さらに学ぶための関連トピック
- [predictor (予測ヘッド非対称性)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0083_prediction-head-predictor)
- [InfoNCE / NT-Xent ロス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0061_info-nce-loss)
- [SwAV (Swapping Assignments between Views)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0095_swav_note)
- [MAE (Masked Autoencoders)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0070_mae_note)
- [行動条件付き潜在予測 (counterfactual rollout)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0001_action-conditioned-prediction)
