---
title: "Predictor (予測ネットワーク)"
---

# Predictor (予測ネットワーク)

## ひとことで言うと
「いま見えている情報(context, 文脈)」から「まだ見えていない/別の情報(target, 標的)」を、画素ではなく特徴ベクトルのレベルで当てにいく小さなネットワークのことです。自己教師あり学習(正解ラベルなしで自分で学ぶ方式)の世界モデル系で、学習が崩壊(collapse)せず、かつ意味のある予測ができるための心臓部になります。左右非対称な構造のうち、片側だけに置かれる「もう一手間」が predictor です。たった1個の小さなネットを足すだけで、負例なしでも表現が崩壊しなくなる、という不思議な効果を持ちます。

## 直感的な理解
クイズで「写真の左半分を見せられて、右半分に何が写っているか言い当てる」状況を考えてください。左半分(context)から右半分(target)を当てるには、世界の構造を理解している必要があります。道路の左に車線があれば右にもある、人の左半身が見えれば右半身も続く、といった知識です。Predictor は「context を手がかりに target を言い当てる係」です。

ここで重要なのは、当てる対象が「画素そのもの」ではなく「特徴(抽象化された意味)」だという点です。右半分の影の落ち方や葉っぱ一枚の質感まで正確に当てる必要はありません。むしろそんな細部は予測不可能で、当てようとすると無駄に苦労します。「右側には歩道があり人が歩いている」という意味さえ当たればよい。だから特徴空間で予測します。この発想は、世界モデルは予測すべき本質だけを抽象的な空間で扱うべきだ、という考え方(LeCun の JEPA=Joint-Embedding Predictive Architecture 構想)に基づいています。ピクセルを当てる生成モデル(画素を1つ残らず復元する)と、特徴を当てる JEPA の最大の違いがここにあります。

## 基礎: 前提となる概念

自己教師あり学習(self-supervised learning, SSL)とは、人間が正解ラベルを付けるのではなく、データから自分で問題を作って学ぶ方式です(SSL の SS は self-supervised の略で、教師=正解ラベルがなくても学べる、の意味)。

崩壊(representational collapse, 表現崩壊)が最大の落とし穴です。画像を2通りに加工し(context と target)、「context を見たら target の特徴に近づける」と学習させると、ネットワークが「どんな入力にも同じ定数ベクトルを返す」とズルをすれば、context と target の特徴は完璧に一致します。損失(loss、予測のズレを表す数値)はゼロになりますが、特徴は何の情報も持たない無意味なものになります。これが崩壊です。崩壊には「全部同じ定数になる(complete collapse)」と「一部の次元しか使わない(dimensional collapse)」の2種類があることも知られています。

崩壊をどう防ぐかが大問題でした。対比学習(contrastive learning、似たペアは近く・違うペアは遠くに押し分ける方式)は「違うものは遠ざける」項(negative, 負例)を入れて崩壊を防ぎますが、大量の負例が必要で計算が重い欠点がありました。

そこで登場したのが「非対称な構造 + Predictor」というアイデアです。target 側の枝には predictor を付けず、context 側の枝にだけ predictor を1個足します。この左右非対称(asymmetry)が、定数ベクトルへの崩壊を回避する鍵になります。なぜ非対称だと崩壊しないのか、その直感は後述します。

## 仕組みを詳しく

典型的な構成は次の3部品です。

1. encoder(エンコーダ、入力を特徴ベクトルに変換する本体ネットワーク)
2. target encoder(教師側エンコーダ。多くの場合 encoder の重みを EMA=指数移動平均でゆっくり追従させたコピー)
3. predictor(予測器。context 側の特徴を受け取り target 側の特徴を予測する小さな MLP や Transformer)

ステップで書くと:

```
context画像 --encoder--> z_context  --predictor--> z_pred  (予測)
target画像  --target_encoder--> z_target            (ターゲット, 勾配は流さない = stop-gradient)

loss = || z_pred - z_target ||²   (予測と本物の特徴のズレ)
```

ポイントは2つです。

- stop-gradient(勾配を止める): target 側には誤差逆伝播の勾配を流しません。両側を同時に学習させると「お互いに定数へ寄せる」崩壊が起きやすいので、target を一旦「動かない目標」として扱います。
- predictor だけが context→target の写像を学ぶ: predictor が「context から見て target がどうなっているか」という関係を吸収するため、encoder は崩壊せず意味ある特徴を持てます。

### なぜ非対称だと崩壊しないのか

直感はこうです。もし encoder が全入力を同じ定数 `c` に潰そうとしても、predictor は「context の `c` から target の `c` を予測する」恒等写像になればよいだけで一見成立しそうに見えます。しかし stop-gradient によって target 側は固定目標として扱われ、かつ EMA で target encoder が生徒よりゆっくり動くため、生徒が定数へ突進しても target は即座には追随しません。さらに predictor が「平均的な target を出す」線形成分を担うことで、encoder 自身が定数化する圧力が逃がされます。

Tian, Chen, Ganguli(ICML 2021)の理論解析は、これを線形モデルで厳密に調べました。彼らは predictor の重み行列の固有値ダイナミクス(学習が進むにつれ各固有値がどう動くか)を解析し、崩壊解(=固有値ゼロ、情報を持たない解)が学習の不動点として不安定になり、情報を持つ解へ収束することを示しました。さらに重要な発見として、predictor が EMA(stop-gradient つき教師)と組み合わさると、予測対象の相関行列の固有空間に predictor が「整列(align)」していく動きが生まれ、これが崩壊を能動的に避ける、と定量的に説明しました。要は「predictor が関係を吸収する役を引き受けるので、encoder は中身のある特徴を持つしかなくなる」という分業です。

### 数値例(テンソル形状で見る)

- z_context: `[B, 768]`(B=バッチサイズ、768=特徴次元)
- predictor: 768 → 512 → 768 のような細くしてから戻すボトルネック MLP(中間で次元を絞るのが定石。bottleneck により恒等写像へ落ちにくくする)
- z_pred: `[B, 768]`
- z_target: `[B, 768]`
- loss はこの2つのコサイン距離や L2 距離

I-JEPA(画像版)では encoder/predictor とも Vision Transformer(画像をパッチに区切って Transformer で処理する方式)で、「画像の一部パッチ(context)から、隠した別パッチ(target)の特徴を、target の位置情報を条件に予測」します。predictor は「どの位置を予測したいか」を表す位置トークン(mask token + position embedding)を受け取り、その位置の特徴を出力します。つまり predictor は単なる MLP ではなく、「予測したい場所を指定できる条件付き予測器」になっています。この「条件を取れる predictor」という発想が、後の行動条件付き予測(取る行動を条件に未来特徴を出す世界モデル)に直結します。

## 手法の系譜と主要論文

- BYOL (Bootstrap Your Own Latent, Grill et al., NeurIPS 2020, DeepMind): 負例なしで表現学習できることを示した代表作。online ネットワーク(context 側)に predictor を1個足し、target ネットワークは online の EMA。提案理由は「負例を集める計算コストを無くしたい」。当初は「なぜ崩壊しないのか」が不明瞭で、後に predictor + stop-gradient + EMA の組み合わせが効いていると解析された。
- SimSiam (Chen & He, CVPR 2021, Meta): BYOL から EMA すら外し、「predictor と stop-gradient さえあれば崩壊しない」ことを実験で示した。提案理由は「BYOL の何が本質か切り分けたい」。predictor が崩壊回避の主役であることを最も鮮明にした論文。
- 理論解析 (Tian, Chen, Ganguli, "Understanding self-supervised learning dynamics without contrastive pairs", ICML 2021, Best Paper Honorable Mention): predictor の重みのダイナミクスと EMA が崩壊解を避ける理由を線形モデルで解析。「predictor が必要」という経験則に理論を与えた。さらに、解析から導かれた DirectPred(predictor を解析的に直接計算する手法)が、学習した predictor と同等の性能を出すことを示し、理論の正しさを裏づけた。
- I-JEPA (Image-based Joint-Embedding Predictive Architecture, Assran et al., CVPR 2023, Meta): 画素を再構成せず「隠した領域の特徴を予測」する設計。提案理由は「世界モデルは抽象的な表現空間で予測すべき(画素は予測不要な細部が多すぎる)」という LeCun の構想の具現化。
- V-JEPA (Video-JEPA, Bardes et al., 2024, Meta): 動画に拡張。時間方向に隠したフレーム/領域の特徴を予測。「動画から世界の動きを表現として学ぶ」を狙う。動画の時空間ブロックをマスクし、その特徴を predictor で当てる。
- 崩壊回避の別路線 (2021-2022): VICReg(Bardes et al.)や Barlow Twins(Zbontar et al.)は predictor の代わりに、特徴の分散を一定以上に保つ正則化や、特徴次元間の相関をゼロに近づける正則化で崩壊を防ぐ。predictor が唯一の解ではないことを示した。

いずれも共通して、predictor という非対称な1枝(または明示的な正則化)が「崩壊回避」と「context→target の関係学習」を同時に担っています。

## 論文の実験結果(定量データ)

### BYOL (Grill et al., 2020)

- ImageNet 線形評価(凍結特徴の上に線形分類器を学習)で 74.3%(ResNet-50)を達成。同時期の対比学習 SimCLR の約 69% を大きく上回った。線形評価は「特徴がどれだけ素直に分類に使えるか」の標準指標。より大きな ResNet-200(×2)では 79.6% に達した。
- アブレーション: predictor を取り除くと精度が大きく崩れ(崩壊に近づく)、predictor が崩壊回避に必須であることを示した。target ネットワークの EMA 係数を 0(=毎ステップ即同期)にすると不安定化することも報告。一方で、対比学習と違いバッチサイズへの依存が小さく(256〜4096 でほぼ安定)、負例不要の利点を実証した。

### SimSiam (Chen & He, 2021)

- ImageNet 線形評価で 71.3%(ResNet-50, 100 エポック)を達成。EMA なしでも崩壊しないことを実証。
- アブレーション(この論文の核心): predictor を外す、あるいは stop-gradient を外すと、損失は急速にゼロへ落ちるが特徴は崩壊し、線形評価精度がチャンスレベル(ほぼランダム、約 0.1%)まで低下することを示した。「損失が下がる=学習成功」ではないことを鮮明にした重要な実験。predictor を固定(ランダム初期化のまま学習しない)にしても崩壊し、predictor を学習することが必須と示した。

### I-JEPA (Assran et al., 2023)

- ViT-H/14 を ImageNet で SSL 事前学習し、線形評価で高精度を達成しつつ、ピクセル再構成型の MAE(Masked Autoencoder)より少ない計算量・少ないエポックで同等以上の表現品質に到達したと報告。特徴空間予測の効率の良さを示した。
- アブレーション: 予測対象のマスク領域を「大きめの連続ブロック(画像の 15〜20% を 4 ブロック程度)」にすると表現が良くなり、細かくバラまくと劣化する。これは「意味のあるまとまり」を予測させることが本質的という示唆。target の特徴を作る encoder の質(EMA で安定した target encoder)も精度に直結することを確認。

### V-JEPA (Bardes et al., 2024)

- 動画でマスク特徴予測を事前学習し、凍結特徴(バックボーンを固定)での動作認識(Kinetics-400, Something-Something-v2)で、ピクセル再構成型(VideoMAE など)を上回る凍結評価精度を報告。特に動きの理解が要る Something-Something-v2 での優位が大きく、「特徴予測は動きの本質を捉える」ことを示した。

## メリット・トレードオフ・限界

メリット
- 負例を集めずに自己教師あり学習ができる(計算が軽い、大バッチ不要)。
- 画素ではなく特徴を予測するので、予測不可能な細部(影・質感)を無視でき、本質的な構造に集中できる。
- 小さなネットワーク1個の追加で崩壊回避という大きな効果を得られる。

トレードオフと限界
- なぜ崩壊しないのかの理論的理解は線形モデル中心で、深層・非線形での完全な保証はない。ハイパーパラメータ(EMA 係数・predictor の幅・学習率)に敏感。
- 特徴空間の予測は人間が直接見て検証しづらい(可視化に別工夫が必要で、生成・デコードは別モジュールが要る)。
- target encoder の更新速度など、設計次第で学習が不安定になりうる。EMA が速すぎると崩壊、遅すぎると学習停滞。
- 未解決課題: 予測目標(target)をどの粒度・どの抽象度で作るのが最適か、運転のような時系列・行動条件付きの予測でどう設計するかは研究途上。

## 発展トピック・研究の最前線
- 行動条件付き predictor: context に加えて「取る行動」を入力し、行動の結果としての未来特徴を予測する設計。これは世界モデル・モデルベース強化学習の核で、運転の未来状態予測に直結する。JEPA を行動条件付きに拡張する研究(I-JEPA から世界モデルへ)が進む。
- 時系列・複数ステップ予測: 1ステップ先だけでなく複数の未来時刻の特徴を予測する場合、誤差累積([長期予測ホライズンと誤差累積](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0795_vista-temporal-horizon))や教師強制 vs ロールアウト([Teacher Forcing vs 自己回帰ロールアウト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0790_teacher-forcing-vs-rollout))の論点が predictor の学習戦略に絡む。複数ステップ予測では、predictor の出力をまた predictor に入れる自己回帰が必要になり、崩壊回避と誤差累積の両方を同時に扱う設計が要る。
- predictor の表現力: MLP から Transformer・条件付き拡散へと predictor 自体を高度化する流れ。位置・行動・時刻を条件に取れる柔軟な予測器が求められている。
- 崩壊回避の代替: predictor の代わりに、分散最大化(VICReg)や白色化(Barlow Twins)で崩壊を防ぐ系統もあり、設計選択肢が広がっている。どの方式が運転のような連続・高次元系列に最適かは未確定。

## さらに学ぶための関連トピック
- [DINO (自己蒸留)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0045_dino-self-distillation)
- [Teacher Forcing vs 自己回帰ロールアウト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0790_teacher-forcing-vs-rollout)
- [長期予測ホライズンと誤差累積](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0795_vista-temporal-horizon)
- [JEPA系 vs 生成系世界モデルの対比](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0066_jepa-vs-generative-world-models)
- [反実仮想ロールアウト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1033_counterfactual-rollout)
