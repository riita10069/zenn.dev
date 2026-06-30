---
title: "Flow Matching による連続軌道生成"
---

# Flow Matching による連続軌道生成

## ひとことで言うと
自動運転の軌道(trajectory、車がこれから走る道筋)を、Flow Matching(フローマッチング、ノイズから本物のデータへ「流す」生成モデルの学習手法)で作り出すアプローチです。連続的で滑らかな軌道を、複数の可能性(多モード)を保ったまま、数ミリ秒の低遅延で生成することを狙います。拡散モデルの親戚でありながら、生成経路をなるべく真っ直ぐに設計することで、はるかに少ないステップで軌道を生成できるのが核心で、近年の運転モデルや行動生成モデルが好んで採用するようになりました。

## 直感的な理解
川の流れを想像してください。上流(完全にランダムなノイズ)に木の葉を一枚落とすと、流れに沿って下流(本物そっくりの軌道)へ運ばれていきます。この「どの向きへどれだけ流れるか」を表すのが速度の場(velocity field)です。Flow Matching は、この速度の場をニューラルネットに学ばせ、推論時はノイズという木の葉を場に沿って流すことで軌道を生成します。

なぜ「流す」のか。軌道を一気に当てにいく回帰だと、交差点で「直進も右折もありうる」ときに両方の平均(縁石に突っ込む危険な中間)を出してしまいます([生成的デコーダ vs 回帰的デコーダ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0144_generative-vs-regressive-decoder) 参照)。流れとして生成すれば、落とす葉(初期ノイズ)を変えるたびに別の下流(別の軌道候補)へ運ばれ、複数の可能性を自然に表せます。しかも流れの道筋をなるべく真っ直ぐに設計すれば、少ない歩数(少ステップ)で下流に着くので速い。これが滑らかさ・多モード・低遅延を同時に狙える理由です。拡散モデルが「ジグザグに曲がりながら下流へ向かう」のに対し、Flow Matching(特に整流したもの)は「ほぼ直線で下流へ向かう」と考えると違いがつかめます。

## 基礎: 前提となる概念
軌道プランナ(trajectory planner)とは、次の数秒間に車がどう進むかを出力するモジュールです。出力はたとえば 0.1 秒刻みで 6 秒先までの位置 `(x, y)` の列(60 点なら `(60, 2)`)や、加速度・曲率(ハンドルの切れ具合)の列です。

生成モデルの大きな系譜を押さえます。VAE([VAE (変分オートエンコーダ)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0255_vae_note))・GAN([GAN (敵対的生成ネットワーク)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0236_gan_note))・Normalizing Flows([Normalizing Flows](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0245_normalizing-flows))・拡散モデル(diffusion)が代表で、いずれも「データの確率分布を学び、そこからサンプリングする」点が共通です。Flow Matching はこのうち、連続正規化フロー(continuous normalizing flows、CNF、[Continuous Normalizing Flows (CNF) / Neural ODE](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0226_continuous-normalizing-flows))を効率よく学習する手法として登場しました。

ODE(常微分方程式、ordinary differential equation)とは、「ある量が時間とともにどう変化するか」を変化の速さ(微分)で記述した式です。`dx/dt = v(x, t)` は「位置 x が時刻 t で速度 v の向きへ変わる」という意味で、これを t について積分すれば軌跡が得られます。Flow Matching が生成時に解くのはまさにこの ODE です。拡散モデルが解く確率微分方程式(SDE、ランダムな揺らぎを含む)と違い、ODE は決定的(同じノイズからは同じ軌道が出る)なので、ノイズと出力が一対一に対応し、扱いやすいという利点があります。

速度の場(velocity field)とは、各点・各時刻で「データへ近づくにはどちらへどれだけ動くべきか」を表すベクトルです。記号では `v_θ(x, t)`(θ はネットの重み)と書きます。

## 仕組みを詳しく
条件(入力)として、周囲の状況を表す特徴を使います。たとえば BEV(Bird's-Eye-View、鳥瞰図。カメラ画像などを上空から見下ろした座標系に変換した特徴マップ)`(B, C, H, W)` です。B はバッチ、C はチャネル数、H・W は鳥瞰グリッドの縦横です。

生成の流れは次の通りです。

1. BEV 特徴を畳み込みで圧縮し平坦化して「条件ベクトル(context)」にする。これが「今どんな道路状況か」を表す。
2. ノイズ軌道 `z`(標準正規分布から引いたランダムな `(B, T, 2)`、T は軌道の点数)を用意する。これが川上の木の葉。
3. 時刻 t(0 から 1)を sinusoidal embedding(時刻を高次元ベクトルに変換する仕組み、Transformer の位置エンコーディングと同じ)で表し、AdaLN(Adaptive Layer Norm、時刻に応じて正規化のスケールとシフトを変える層)でネットへ注入する。「今どのくらい流れたか」をネットに伝えるためです。
4. ノイズ軌道(クエリ)が BEV 条件(キー・バリュー)を cross-attention(交差注意、軌道の各点が状況マップのどこに注目すべきかを学習)で参照する。
5. 出力は速度の場 `v_θ` `(B, T, 2)`。今のノイズ軌道を本物の滑らかな軌道へ近づけるための「向き」です。

```
[BEV特徴 (B,C,H,W)]──[Conv/Flatten]──┐ (条件 context)
                                      ▼
[ノイズ軌道 z (B,T,2)]──[投影]──[Cross-Attn]──[AdaLN(時刻t)]──> [速度場 v_θ (B,T,2)]
推論: z から v_θ に沿って t:0→1 を積分 → 滑らかな軌道
```

学習(Flow Matching、シミュレーションフリー)はこうです。本物軌道 x とノイズ z を直線で結び、中間点を `x_t = (1-t)·z + t·x` と置きます。この直線に沿った正解の速度は `x - z`(始点から終点へ向かう一定ベクトル)です。ネット `v_θ(x_t, t, 条件)` がこの `x - z` を出すよう二乗誤差で訓練します。

```
損失 L = E[ || v_θ(x_t, t, 条件) - (x - z) ||² ]
   x_t = (1-t)·z + t·x,   t ~ Uniform(0,1)
```

ここが重要です。各記号の意味は、x が本物軌道(教師データ)、z がノイズ、t が 0〜1 の進行度、`v_θ` がネットが予測する速度、`x - z` が教えたい正解速度です。E は期待値(平均)。なぜこの単純な損失で正しい生成モデルが学べるのかは Flow Matching の理論の肝で、「個々のデータ点へ向かう条件付きの流れを平均すると、データ分布全体へ向かう正しい流れ(marginal velocity field)になる」という事実に基づきます。学習者は条件付きの速度 `x - z` だけ教えればよく、複雑な平均化はネットが暗黙に行います。この学習は ODE を実際に解く必要がない(シミュレーションフリー)ので、CNF を従来通り解いていた頃に比べて学習が格段に安定し速くなりました。

推論は、ノイズ z から始め、`dx/dt = v_θ` を t=0 から 1 までオイラー法(微小な歩幅で順に足していく数値積分)で数ステップ積分します。直線的な経路なので少ステップ(おおむね 5〜10 回、整流すれば 1 回)で済み、低遅延につながります。終点 t=1 が出力軌道です。

この設計が運転に向く理由を言葉でまとめます。連続軌道をそのまま生成するので、動きを記号(「右」「左」など)に離散化する大規模言語モデル風アプローチと違い、座標・速度・曲率の連続性(運動学的な滑らかさ)を保てます。速度場を回帰するので、始状態から多モードな最適未来軌道への「最も真っ直ぐな道」を描けます。直線経路ゆえ少ステップで済み、実時間制御に必要な数ミリ秒級の推論が現実的になります。

## 手法の系譜と主要論文
数学的土台は Chen, Rubanova, Bettencourt, Duvenaud の "Neural Ordinary Differential Equations"(NeurIPS 2018、arXiv:1806.07366、トロント大)です。ニューラルネットを ODE の速度場として使う発想を導入し、連続正規化フロー([Continuous Normalizing Flows (CNF) / Neural ODE](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0226_continuous-normalizing-flows))へつながりました。ただし当時は生成のたびに ODE を解いて学習する必要があり、計算が重い・尤度計算に高コストなトレース推定が要るという問題がありました。

転機は Lipman, Chen, Ben-Hamu, Nickel, Le の "Flow Matching for Generative Modeling"(ICLR 2023、arXiv:2210.02747、Meta AI)です。提案は「速度場を直接回帰すれば CNF を ODE 求解なしに学習できる」こと(conditional flow matching)。動機は CNF の学習が重く不安定だったこと。新規性は、ノイズとデータを結ぶ確率の道(probability path)を設計し、その道に沿った速度を教師信号として与えるだけでよいと示した点です。これにより拡散モデルと同等以上の生成品質を、より直線的な経路で達成できるようになりました。

ほぼ同時期に Liu, Gong, Liu の "Flow Straight and Fast: Rectified Flow"(ICLR 2023、arXiv:2209.03003、テキサス大オースティン)が、ノイズとデータを直線で結ぶ経路を整流(rectify、reflow という手続きで自己生成したペアを再度直線で結び直す)し、可能な限り真っ直ぐにして、1〜数ステップ生成を可能にしました。Flow Matching の少ステップ化を推し進めた重要論文です。さらに Albergo & Vanden-Eijnden の "Stochastic Interpolants"(ICLR 2023、arXiv:2209.15571、NYU)は、拡散と Flow Matching を統一的に見る理論枠組み([Stochastic Interpolants](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0252_stochastic-interpolants))を与え、ノイズとデータをつなぐ「補間子」を自由に設計できることを示しました。これらが同時期に出たことで、Flow Matching は理論・実用の両面で急速に成熟しました。

制御・行動生成への応用としては Chi, Feng, Du, Xu, Cousineau, Burchfiel, Song の "Diffusion Policy"(RSS 2023、arXiv:2303.04137)が、ロボットの行動を拡散で生成し多モードを扱える点を示しました。Flow Matching は拡散の親戚で、より少ないステップで生成できる利点があるため、後続の運転・操作モデルが Flow Matching を採用する流れができました。研究コミュニティでは、こうした「連続軌道を Flow Matching で生成する運転デコーダ」が、回帰系・自己回帰トークン系に対する有力な代替として議論されています。連続表現を保ちつつ生成の多様性と速度を両立する点が、運転の軌道デコーダに据える動機になっています。

## 論文の実験結果(定量データ)
Flow Matching の基礎論文(Lipman et al., 2023)は画像生成(ImageNet 等)で評価し、生成品質の指標 FID(Fréchet Inception Distance、生成画像と本物画像の分布の近さ。小さいほど良い)を、拡散モデルと同等以上に達成しつつ、より少ない関数評価回数(NFE、number of function evaluations、ODE を解くステップ数に相当)で到達できることを報告しました。特に最適輸送に基づく条件付き経路(OT path)を使うと、拡散の代表手法より少ない NFE で同等以下の FID を出すと示されています。要点は「同じ品質を少ない計算で」です。

Rectified Flow(Liu et al., 2023)は、reflow による経路の整流を繰り返すことで NFE=1(1 ステップ生成)でも実用的な画質を出せると示し、少ステップ生成の有効性を定量的に裏付けました。整流前は多ステップ必要だったものが、整流後は 1〜数ステップで同等の FID に近づく、という改善が報告されています。

行動生成では Diffusion Policy(Chi et al., 2023)が、複数のロボット操作ベンチマーク(Push-T、RoboMimic、実機操作など計十数タスク)でタスク成功率を従来の回帰系・LSTM-GMM・暗黙的方策(IBC)より大きく改善したと報告しています(平均でおよそ数十パーセントポイント、報告では平均約 46.9% の相対改善という記述があります)。これは生成系がマルチモーダルな行動分布を扱える利点を定量的に示した例で、Flow Matching による軌道生成が運転で期待される理由の根拠になっています。

運転への直接適用(連続軌道の Flow Matching 生成)については、公開ベンチマーク横断での標準化された数値が限られるため、報告による定量値は一次資料で確認するのが確実です。一般に主張される利点は、回帰系に比べて滑らかさ(jerk、加加速度の小ささ)と多モード性が改善し、自己回帰トークン系に比べて推論ステップが少なく低遅延、という定性的・準定量的なものです。アブレーションとしては、推論ステップ数を 1 に減らすと滑らかさ・正確さがやや落ち、5〜10 に増やすと品質が飽和に近づく、という速度と品質のトレードオフが各種報告で共通して観察されています。条件付け(地図・ルート)を外すと多モードが過剰に広がり実行不能な軌道が増える、という観察も生成制御の重要性を裏づけます。

## メリット・トレードオフ・限界
メリットは、滑らかで連続な軌道が出ること(自己回帰のカクつきがない)で、乗り心地と安全に直結します。多モードな未来(直進・右折など)を分布として自然に扱えること。推論が少ステップで済み数ミリ秒級の低遅延を狙えること。離散トークン化を避けるので座標・運動学の連続性を保てること。ODE が決定的なので拡散の SDE より積分が安定し、ノイズと出力の対応が明確で解析しやすいこと。

トレードオフと限界としては、生成モデルなので学習データの質と分布に敏感です。推論ステップ数を減らすと滑らかさ・正確さが落ちます(速度と品質のトレードオフ)。一本の軌道を確定的に出す回帰系よりパイプラインが複雑(条件付け、時刻埋め込み、ODE ソルバ等)です。多モードに出せる反面、最終的に一本を選ぶ後段の選択ロジック(安全制約との整合、最尤モードや最安全モードの選択)が別途必要です。また Flow Matching の生成は確率的なので、安全制約を厳密に保証するには後段の検証や射影が要ります。整流(reflow)は強力ですが、自己生成ペアの品質に依存し、繰り返すと多様性が落ちる(モード崩壊に近い現象)おそれもあります。

未解決の課題は、安全性の保証付き生成、希少シナリオへの頑健性、そして 1 ステップ生成での品質維持(蒸留や整流の改良)です。

## 発展トピック・研究の最前線
Flow Matching は急速に発展しています。Rectified Flow による 1 ステップ化、最適輸送(optimal transport)に基づく経路設計でさらに真っ直ぐな流れを作る研究、拡散と Flow Matching を統一的に見る確率補間([Stochastic Interpolants](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0252_stochastic-interpolants))の理論、そして条件付け(地図・ルート・安全制約)を強める制御可能な生成が活発です。運転分野では、知覚から軌道生成までを一体で学ぶ end-to-end モデルに Flow Matching デコーダを組み込む流れ、言語モデル的アプローチとの比較([生成的デコーダ vs 回帰的デコーダ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0144_generative-vs-regressive-decoder))、そして大規模な運転基盤モデルのデコーダとして Flow Matching を据える研究が最前線にあります。少ステップ化と安全制約の両立が、実車展開へ向けた中心的な研究課題です。

## さらに学ぶための関連トピック
- [Continuous Normalizing Flows (CNF) / Neural ODE](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0226_continuous-normalizing-flows)
- [生成的デコーダ vs 回帰的デコーダ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0144_generative-vs-regressive-decoder)
- [Normalizing Flows](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0245_normalizing-flows)
- [Diffusion Policy](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0170_diffusion-policy)
- [Stochastic Interpolants](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0252_stochastic-interpolants)
- [VAE (変分オートエンコーダ)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0255_vae_note)
