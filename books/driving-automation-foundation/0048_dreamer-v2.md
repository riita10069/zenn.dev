---
title: "DreamerV2"
---

# DreamerV2

## ひとことで言うと

DreamerV2 は、Dreamer V1 の潜在状態を「連続的な数値」から「離散的なカテゴリ変数(32択を32グループ並べたもの)」に変えた後継です。この変更により、Atari の57ゲームで「単一GPUでの学習」かつ「世界モデルの想像だけで方策を学ぶ」という条件で、初めて人間の平均的プレイヤーを上回るスコア(人間正規化中央値)を達成しました。「離散潜在の世界モデルが本当に効く」ことを大規模に実証した記念碑的研究です。

## 直感的な理解

V1 の潜在状態は連続ガウス分布、つまり「1つの山を持つ釣鐘型の分布」でした。これは「だいたいこのあたり」という連続的な状態を表すのは得意ですが、「敵が右に出るか左に出るか」のようにはっきり分かれた複数の可能性(マルチモーダル性)を表すのが苦手です。1つの山しか持てないので、右と左の中間という「ありえない平均」にぼやけてしまいます。世界モデルがこの平均を予測すると、想像の中で起きるはずのない曖昧な未来が生まれ、その上で学んだ方策は実環境で役に立ちません。

ゲーム映像にはこうした「飛び飛びで複数の未来」が頻出します。場面が瞬間的に切り替わる、敵がどちらに出るか分岐する、ブロックが壊れる瞬間に画面が一変する、といった具合です。DreamerV2 は「潜在状態を離散カテゴリにすれば、こうした飛び飛び・複数の未来を自然に表現できる」と考えました。カテゴリ分布は「右に 0.5、左に 0.5」のように複数の選択肢に同時に確率を割り当てられるため、無理な平均化が起きず、サンプルすると右か左かどちらかにはっきり決まります。

## 基礎: 前提となる概念

### 連続分布とカテゴリ分布

連続分布(ガウスなど)は実数全体に滑らかな確率を割り当て、典型的には1つのモード(山)を持ちます。カテゴリ分布(categorical distribution)は「サイコロの目」のように有限個の選択肢それぞれに確率を割り当て、複数の選択肢に高い確率を同時に持てます。マルチモーダルな(複数の答えがありうる)未来の表現にはカテゴリ分布が向きます。

### one-hot 表現

「one-hot(ワンホット)」とは、たとえば「3番目を選んだ」を [0,0,1,0,...,0] のように、選んだ場所だけ1、他は0で表す方式です。32択の one-hot ベクトルは長さ32で、1箇所だけが1です。離散サンプリングの結果はこの one-hot ベクトルになります。

### KL ダイバージェンス

KL ダイバージェンス(Kullback-Leibler divergence)は2つの確率分布の「ずれ」を測る非対称な量で、KL(p‖q) と KL(q‖p) は一般に値が異なります。[RSSM (Recurrent State-Space Model)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0088_rssm_note) では、観測を見ずに予測した分布(prior)と観測を見て補正した分布(posterior)を KL で近づけ、「観測なしでも posterior に近い予測ができる」世界モデルを育てます。この非対称性が後述の KL バランシングで重要になります。

### 微分できない離散変数

ニューラルネットは勾配で学習しますが、「どのカテゴリを選ぶか」という離散的なサンプリングは段差のある操作で、そのままでは微分できません。これを学習可能にする技術(straight-through, Gumbel-Softmax)が離散潜在の鍵になります。

## 仕組みを詳しく

### 離散カテゴリ潜在とは

V2 の確率状態は、32個のカテゴリ変数の集まりです。各カテゴリ変数は32択(one-hot)です。

```
確率状態 s_t   =  32 グループ × 32 カテゴリ
              =  [B, 32, 32]   ← 各グループで32択のうち1つを選ぶ(各行が one-hot)
              → flatten して [B, 1024] で actor/critic/デコーダに渡す
```

32グループあるので状態は「32桁の32進数」のようなもので、32^32 ≈ 10^48 通りという膨大な組み合わせを表せます。連続ガウスと違い、はっきり区別された状態を選び取れるので、マルチモーダルな未来を表現できます。

### 微分できない問題と straight-through gradient

離散的な選択(どのカテゴリを選ぶか)は段差のある関数なので、そのままでは微分できず勾配を逆方向に流せません。そこで straight-through gradient(ストレートスルー勾配、Bengio 2013)という近似を使います。順伝播では普通に離散サンプリングして one-hot を作り、逆伝播ではあたかも連続的な確率(softmax の出力)だったかのように勾配を通します。式で書くと、sample = one_hot(argmax) + (probs − stop_grad(probs)) のように「順方向では one-hot、勾配計算では probs」になるトリックです。「行きは離散、帰りは連続として扱う」ことで、離散変数を含むネットワークも学習できます。近似なので勾配にバイアスが乗りますが、実用上は十分機能します。

### KL バランシング(KL balancing)

V2 は KL 損失を「2つの方向で重みを変える」工夫を導入しました。KL(posterior ‖ prior) を、prior を posterior に近づける方向(prior を賢くする、重み 0.8)と、posterior を prior に近づける方向(posterior が情報を捨てる、重み 0.2)に分け、stop-gradient で勾配の流れを制御します。離散潜在では prior(観測なしの予測)の学習が難しく、prior 側を重く更新することで世界モデルの予測力を優先しつつ、posterior が情報を捨てすぎないようにします。このバランス調整が学習安定化の鍵になりました。

### 学習の全体像と actor 学習

世界モデル([RSSM (Recurrent State-Space Model)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0088_rssm_note) の離散版)を画像再構成・報酬予測・KL で学習し、その想像ロールアウト上で actor-critic を学習する流れは V1 と同じです。ただし離散潜在では V1 のような滑らかな解析的勾配(reparameterization)が直接は使えません。そこで actor の学習には、世界モデルを通す動力学勾配(straight-through で離散を通す)と、報酬の良し悪しに応じて行動確率を上下させる REINFORCE 系の方策勾配を、混合比 ρ で組み合わせます。Atari のような離散行動・不連続な報酬では REINFORCE 寄りが効き、連続制御では動力学勾配寄りが効く、という使い分けが報告されています。critic の学習(λ-return への回帰)は V1 と同様です。

### tensor のイメージ

```
決定論状態 h_t   [B, 600]        ← GRU の隠れ状態(V1 より大きめ)
離散状態   s_t   [B, 32, 32]     ← 32グループ×32カテゴリの one-hot
状態全体        h_t と flatten(s_t) を結合 [B, 600+1024] → actor/critic/デコーダ
```

## 手法の系譜と主要論文

### 離散潜在の源流: VQ-VAE (van den Oord et al., NeurIPS 2017)

連続な潜在をコードブックの最近傍コードに量子化して離散表現を学ぶ VQ-VAE が離散潜在表現の源流の一つです。画像・音声生成で高品質を実現し、離散コードが豊かな表現を持てることを示しました。DreamerV2 のカテゴリ潜在は VQ-VAE とは量子化の仕組みが異なります(VQ-VAE はコードブック最近傍、V2 は softmax からのサンプリング)が、「離散潜在が世界の表現に有効」という思想を共有し、それを世界モデルの潜在ダイナミクスに持ち込んで成功させた例と位置づけられます。

### Gumbel-Softmax / straight-through (Jang et al. ICLR 2017; Maddison et al. ICLR 2017; Bengio et al. 2013)

離散変数を微分可能に扱う技術の系譜。Gumbel-Softmax は離散サンプリングを温度パラメータ付きの連続緩和で近似し、温度を下げると one-hot に近づきます。DreamerV2 はこれらの straight-through 版を採用しました。

### Hafner, Lillicrap, Norouzi, Ba "Mastering Atari with Discrete World Models" (ICLR 2021, arXiv:2010.02193)

[Dreamer (V1)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0047_dreamer-v1) が連続制御で強力だった一方、Atari のような場面転換が急でマルチモーダルな環境では伸び悩みました。V2 は潜在を離散カテゴリにし、straight-through gradient と KL balancing で安定化することでこれを解決しました。「世界モデルの想像だけで学んだエージェントが、同条件のモデルフリー強豪を Atari で上回る」ことを初めて大規模に示し、これが [DreamerV3](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0049_dreamer-v3) の汎用化へと続きます。

## 論文の実験結果(定量データ)

### Atari 57(人間正規化中央値)

評価は Atari の57ゲームのベンチマークで、指標は gamer-normalized median(人間正規化中央値: 各ゲームのスコアを「ランダムプレイ=0、人間プレイヤー=1」に正規化し、57ゲームの中央値を取る)です。この指標が 1.0 を超えると「半数以上のゲームで人間プレイヤーを上回った」ことを意味します。

DreamerV2 は 2億フレーム(200M フレーム ≈ 50M エージェントステップ、フレームスキップ4)・単一GPU(約10日)・世界モデルの想像のみという条件で、人間正規化中央値がおよそ 1.6 前後に達したと報告されています。これは「半数以上のゲームで人間の1.6倍前後のスコア」を意味し、当時のモデルフリーの代表である Rainbow(DQN 系の集大成)、IQN を同条件で上回りました。世界モデルベースの手法が、同じ計算予算でモデルフリーの強豪に勝ったことは大きなインパクトを持ちました。なお、人間正規化「平均」では Rainbow などが上回る場合もあり、論文は「中央値で測ると典型的なゲームで安定して強い」という点を主張しています。

数値の意味を補足すると、中央値を使うのは平均だと一部のゲームの極端な高スコア(例: 桁違いに稼げる特定ゲーム)に引きずられて全体像を見誤るためで、中央値1.6は「典型的なゲームで安定して人間超え」を示す堅実な指標です。

### アブレーション(何を抜くとどうなるか)

論文のアブレーションで示された主要な知見は次の通りです。第一に、離散カテゴリ潜在を連続ガウス潜在に戻すと中央値が大きく低下し、離散化が Atari 性能の主因であることが示されました。これが V2 の中心的主張の裏付けです。第二に、KL balancing を外す(prior と posterior を等しい重みで近づける)と学習が不安定になり性能が落ちます。離散潜在では prior の学習が難しいため、このバランスが不可欠です。第三に、画像再構成損失を弱める/外すと表現が育たず性能が低下し、V2 が純粋な潜在予測ではなく画像再構成を併用するハイブリッドであることを反映しています([潜在空間予測 vs ピクセル予測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0068_latent-vs-pixel-prediction) の議論に対する一つの立場)。第四に、想像での actor 学習における勾配の混合(REINFORCE と動力学勾配)で、Atari では REINFORCE 成分が重要であることが示されました。第五に、モデルサイズや報酬予測の精度を削ると当然性能が落ち、世界モデルの容量が効いていることが分かります。

## メリット・トレードオフ・限界

メリット: 離散カテゴリ潜在で飛び飛び・マルチモーダルな未来を自然に表現でき、世界モデルの予測が「ありえない平均」にぼやけない。Atari で人間超え(中央値)を達成し離散潜在世界モデルの有効性を実証。単一GPU・想像のみで学習でき計算資源の面で現実的。KL balancing と straight-through gradient で離散潜在の学習を安定化。

トレードオフ・限界: 離散変数で V1 のような滑らかな解析的勾配が直接使えず、分散の大きい方策勾配(REINFORCE)を併用する必要がある。straight-through gradient は近似なので勾配にバイアス(偏り)が乗る。各損失の重み・勾配混合比などタスクごとのハイパラ調整がまだ残り、完全な汎用性には未到達で、これが [DreamerV3](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0049_dreamer-v3) の動機になりました。また Atari は比較的決定論性が高い環境であり、確率的で部分観測性が強く長期な現実世界タスクへの一般化は別途検証が必要でした。

## 発展トピック・研究の最前線

DreamerV2 が示した「離散潜在は急峻でマルチモーダルな未来の表現に有利」という知見は、世界モデル設計の指針になりました。続く [DreamerV3](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0049_dreamer-v3) は離散潜在を維持しつつ symlog 変換・twohot 予測・free bits・パーセンタイル正規化などの頑健化を加え、単一設定での汎用性を実現しました。

自動運転や軌跡予測の文脈でも、未来は「直進か右折か左折か」のように離散的分岐を含むため、離散潜在や複数モードを出力するマルチモーダルヘッド(混合密度ネットワーク、アンカー軌跡など)が有力な選択肢になります。また、離散潜在トークンと Transformer 系列モデルを組み合わせて世界モデルを構築する流れ(Micheli らの IRIS, Chen らの TransDreamer, Zhang らの STORM など)が活発で、言語モデル流の「次トークン予測」を世界のダイナミクスに適用しています。離散コードの数や粒度の自動調整、straight-through の勾配バイアス低減、部分観測・確率的環境での頑健性が現在の研究テーマです。

## さらに学ぶための関連トピック
- [Dreamer (V1)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0047_dreamer-v1)
- [DreamerV3](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0049_dreamer-v3)
- [RSSM (Recurrent State-Space Model)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0088_rssm_note)
- [PlaNet](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0082_planet_note)
- [潜在空間予測 vs ピクセル予測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0068_latent-vs-pixel-prediction)
