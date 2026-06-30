---
title: "Score-based Generative Models (SDE定式化)"
---

# Score-based Generative Models (SDE定式化)

## ひとことで言うと
これは [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note) と [NCSN (Noise Conditional Score Networks)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0243_ncsn_note) という、それまで別々に発展していた2つの生成モデルが、実は同じものの離散版(時間を1000ステップなどに区切ったもの)だったことを示し、両者を「連続時間の確率微分方程式(SDE)」という1つの数学の枠組みで統一した理論です。「データを少しずつノイズで壊す過程」をステップではなく滑らかに連続する時間の流れとして書き直し、その逆流(ノイズからデータへ戻る流れ)を解けば生成できる、と定式化しました。拡散モデルの「大統一理論」と呼べる仕事で、ICLR 2021 で Outstanding Paper Award を受けました。

## 直感的な理解
コップに垂らしたインクが水中で広がっていく様子を想像してください。最初は一点に固まったインク(=きれいなデータ)が、時間とともにランダムに拡散し、やがて水全体に均一に混ざる(=純粋なノイズ)。この「広がっていく流れ」を逆再生できれば、均一な水からインクの一滴を復元できます。これがまさに生成です。

DDPM も NCSN も、本質的にはこの「広がる→逆再生する」をやっていました。違いは「インクをどう薄めるか(信号を縮めるか縮めないか)」だけ。ところが両者はステップ単位(1000ステップなど)で書かれ、語り口も「変分下限の最適化」(DDPM)と「複数水準のスコアマッチング」(NCSN)で別々だったため、同じものに見えなかった。Song らはステップ幅を限りなく 0 に近づけた連続時間極限を取り、両者を一本の微分方程式 dx = f(x,t)dt + g(t)dw の特別な場合として書き直しました。一度連続で書けば、逆再生の流れも、決定論的な高速版も、厳密な確率計算も、すべて統一的な道具(確率論と数値解析)で扱えるようになります。これは「離散の漸化式」から「連続の微分方程式」へ視点を上げたことで、微積分と数値解析の200年分の道具が一気に使えるようになった、という構図です。

## 基礎: 前提となる概念

### スコアとは何か

スコアとは、データ分布の対数の勾配 ∇_x log p(x) です(詳しくは [Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching))。直感的には「今いる点 x から、データが密集している方向はどっちか、どれくらい急か」を指し示す矢印です。スコアの嬉しい点は、確率の正規化(全部足して1にする割り算、高次元では計算不能)を考えなくてよいことです。log を微分すると正規化定数が定数項として消えるため、スコアだけなら学習できます。

### 微分方程式・SDE・ブラウン運動

常微分方程式(ODE, ordinary differential equation)は dx = f(x,t)dt のように「次の瞬間どっちへどれだけ動くか」を決める式です。出発点を決めれば未来は一意に決まります(決定論的)。確率微分方程式(SDE, stochastic differential equation)は、それにランダムな揺れを足したものです。

```
dx = f(x, t) dt + g(t) dw
```

- f(x,t)dt はドリフト項(drift)。決定論的にどっちへ動くか。
- g(t)dw は拡散項(diffusion)。dw はブラウン運動(Brownian motion、Wiener 過程とも。水中の微粒子のようなランダムな揺れの数学的モデル)の微小増分で、平均 0・分散 dt の正規乱数。g(t) はその揺れの強さ。

つまり SDE は「決まった流れ + ランダムな揺れ」で粒子がどう動くかを記述します。SDE が与えられると、時刻ごとに「粒子の居場所の分布」 p_t(x) が定まり、それは Fokker-Planck 方程式という偏微分方程式に従って時間発展します。この p_t こそ、後でスコアを学ぶ対象になります。

## 仕組みを詳しく

### 前向き過程を SDE で書く

DDPM や NCSN の「少しずつノイズを足す」過程を、時間 t を 0 から 1(または 0 から T)まで連続に流れる量として書き直します。t=0 が元のきれいなデータ p_0 = p_data、t=1 がほぼ純粋なノイズ p_1 ≈ N(0, σ_max²I) です。Song らは f と g の選び方で、有名な2系統が再現されることを示しました。

- VP-SDE(Variance Preserving、分散保存型): dx = -(1/2)β(t)x dt + sqrt(β(t)) dw。信号 x を少しずつ縮めながらノイズを足し、全体の分散を一定(おおむね 1)に保つ。DDPM の連続版で、ドリフトに -x が入っているのが「信号を縮める」効果。
- VE-SDE(Variance Exploding、分散発散型): dx = sqrt(d[σ²(t)]/dt) dw。ドリフトがゼロ(信号は縮めない)で、ノイズだけを足していくので分散が時間とともに発散する。NCSN の連続版。
- sub-VP-SDE: VP を改良し、各時刻の分散が VP 以下に抑えられる変種。尤度(bits/dim)がさらに良くなる。

つまり DDPM も NCSN も、この一般 SDE の f, g を変えただけの兄弟だったのです。VP は「縮めながら混ぜる」、VE は「混ぜるだけ」と覚えると違いが直感的です。

### 逆向き SDE で生成する

ここが核心です。Anderson(1982)という古い確率論の定理により、「前向きの SDE には対応する逆向きの SDE が必ず存在し、それはスコアを使って書ける」ことが知られています。逆向き SDE は次の形です。

```
dx = [ f(x, t) - g(t)² · ∇_x log p_t(x) ] dt + g(t) dw̄
```

ここで dt は負の方向(t=1 から t=0 へ)に進み、dw̄ は逆時間のブラウン運動です。注目すべきは ∇_x log p_t(x)、つまり各時刻のスコアが入っていることです。前向き SDE のドリフト f から「スコアに g² を掛けた項」を引くと、流れが逆転してノイズからデータへ戻る、というのが Anderson の結果の意味です。このスコアをニューラルネット s_θ(x, t) に学習させれば、t=1(ノイズ)から t=0(データ)へ向かって逆向き SDE を数値的に解くだけで、新しいデータを生成できます。スコアの学習は、全時刻にわたるデノイジングスコアマッチングを重み付き積分した連続版の目的関数で行います。

```
L(θ) = E_t { λ(t) E_{x_0, x_t} [ || s_θ(x_t, t) - ∇_{x_t} log p_{0t}(x_t | x_0) ||² ] }
```

t は [0,1] から一様(または重み付き)に引き、p_{0t} は前向き SDE が与える条件付き分布(線形な f を選べばガウスになり、平均と分散が解析的に書ける)です。条件付きスコア ∇ log p_{0t}(x_t|x_0) は -ε/(対応する標準偏差) という閉じた式になるので、教師信号は計算できます。これは [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note) のノイズ予測損失や [NCSN (Noise Conditional Score Networks)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0243_ncsn_note) の複数水準 DSM を連続時間に一般化したものに他なりません。重み λ(t) を「尤度が厳密に下界される値」に選ぶと、後述の厳密尤度計算と整合します。

### 確率フロー ODE(おまけの強力な道具)

Song らはさらに、同じ前向き SDE と「各時刻のデータの周辺分布 p_t が完全に一致する」決定論的な常微分方程式が存在することを示しました。これを確率フロー ODE(probability flow ODE, PF-ODE)と呼びます。

```
dx = [ f(x, t) - (1/2) g(t)² · ∇_x log p_t(x) ] dt
```

逆向き SDE と見比べると、スコアの係数が g² から (1/2)g² に変わり、ランダム項 dw が消えています。これがポイントです。同じスコア s_θ を使い、ODE なので高精度な数値解法(Runge-Kutta、Heun 法など)で少ステップ高速生成ができます。これは [DDIM (Denoising Diffusion Implicit Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0227_ddim_note) の決定論的サンプリングの連続時間版です。さらに ODE は連続正規化フロー(continuous normalizing flow, CNF)の一種なので、瞬間変化率の発散(ヤコビ跡)を時間積分することで対数尤度を厳密に計算でき(instantaneous change-of-variables、Chen et al. 2018)、また t=0 の画像を t=1 のノイズへ可逆に戻せる(=画像を潜在表現にエンコードして補間・編集する)という理論的に重要な性質を持ちます。SDE 経路は揺れがあるので毎回違うサンプル、ODE 経路は初期ノイズが同じなら同じサンプル、という使い分けになります。

### Predictor-Corrector サンプラー

連続化の実用的な果実が Predictor-Corrector(PC)サンプラーです。各時刻で、まず逆向き SDE の数値積分で次の時刻へ進み(predictor)、続いてその時刻のスコアを使った Langevin 動力学で分布を補正する(corrector、[NCSN (Noise Conditional Score Networks)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0243_ncsn_note) のアニーリング Langevin に相当)。predictor は時間方向に進めるが離散化誤差を持ち込み、corrector は同じ時刻に留まって誤差を吸収する、という役割分担です。予測の数値誤差を補正が吸収するので、同じ関数評価回数でも品質が上がります。これは「片方(SDE 積分)と片方(Langevin)を組み合わせる」という、統一定式化があって初めて自然に出てくる設計です。DDPM の祖先サンプラーは predictor のみ、NCSN は corrector のみ、と整理でき、両者の良いとこ取りができます。

### 具体的なイメージ

形状 [3, 256, 256] の画像を考えます。t=1 で純粋なノイズ(全ピクセルがランダム)から始め、逆向き SDE または確率フロー ODE を t=1 から t=0 へ数値的に積分します。各積分ステップでネット s_θ がスコア(どっちが画像らしい方向か)を教え、その矢印に沿って少しずつ画像を整えていきます。t=0 に着くと完成した画像が得られます。SDE 経路はランダム揺れを伴うので毎回違う画像、ODE 経路は決定論的で初期ノイズが同じなら同じ画像が出ます。逆向き SDE をオイラー法で 1000 ステップ刻むのが素朴なやり方ですが、PF-ODE を Heun 法で 20〜50 ステップ刻む、あるいは専用ソルバー(DPM-Solver)で 10〜20 ステップにする、と劇的に速くできます。

## 手法の系譜と主要論文

- Anderson (1982), "Reverse-time diffusion equation models" (Stochastic Processes and their Applications 12): 逆向き SDE がスコアで書けるという数学的基礎を与えた古典。Song らの理論はこの定理に依拠している。
- Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015), "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" (ICML 2015): 拡散による生成という発想の原点。非平衡熱力学を比喩に離散時間の前向き/逆向き過程を提案。
- Song & Ermon (2019, [NCSN (Noise Conditional Score Networks)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0243_ncsn_note)) / Ho et al. (2020, [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note)): 独立に高品質化した2系統。SDE 統一の前夜。
- Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole (2021), "Score-Based Generative Modeling through SDEs" (ICLR 2021, Outstanding Paper Award): 本トピックの主役。DDPM(VP-SDE)と NCSN(VE-SDE)を連続時間 SDE で統一し、逆向き SDE・確率フロー ODE・PC サンプラー・厳密尤度計算・条件付き生成(逆問題)を一つの枠組みで導いた。
- Karras, Aittala, Aila, Laine (2022, EDM), "Elucidating the Design Space of Diffusion-Based Generative Models" (NeurIPS 2022): SDE/ODE の設計空間(ノイズスケジュール σ(t)、前処理係数、サンプラー、訓練時のノイズ分布)を直交する設計選択として整理し、確率フロー ODE を Heun 法など高次の ODE ソルバーで解く実践的レシピを提示。SDE 定式化の実用面を大きく前進させた。
- Lu, Zhou, Bao, Chen, Li, Zhu (2022), "DPM-Solver" (NeurIPS 2022) / "DPM-Solver++" (2022): 確率フロー ODE の線形項を解析的に解き、非線形部分だけを数値近似する半解析的高速ソルバー。10〜20 ステップで高品質生成を可能にし、統一定式化の実用価値を決定づけた。

## 論文の実験結果(定量データ)

評価指標は FID(生成分布と本物分布の特徴空間距離、低いほど良い)、IS(Inception Score、高いほど良い)、bits/dim(尤度ベースモデルの圧縮効率、1次元あたり何ビットで表せるか、低いほど良い)です。

- Song et al.(2021)は CIFAR-10(32×32)で、当時の最高記録を更新しました。確率フロー ODE と高精度ソルバーを用いた生成で FID 2.20、Inception Score 9.89 を達成。これは同時期の最良の GAN(StyleGAN2-ADA の FID 約 2.42、StyleGAN2 系で FID 約 2.9〜3.2)を上回る数値で、「拡散モデルが GAN を画質で凌ぐ」流れを定量的に示しました。
- 尤度: 確率フロー ODE による厳密尤度計算で、CIFAR-10 で 2.99 bits/dim(尤度重み付け版でさらに改善)という当時最良クラスの圧縮効率を報告。生成品質(FID)と尤度(bits/dim)の両方で最先端という、それまで両立が難しかった結果を出しました。GAN は尤度を持たず、従来の尤度モデル(PixelCNN, Glow)は FID が悪い、という二者択一を崩した点が重要です。
- 高解像度: VE-SDE と PC サンプラーで 1024×1024 の CelebA-HQ(顔画像)の生成に成功し、高解像度でのスケーラビリティを実証しました。
- アブレーション(PC サンプラーの効果): predictor 単独(逆向き SDE 積分のみ)や corrector 単独(Langevin のみ)に対し、両者を組み合わせた PC サンプラーが同一の関数評価回数(NFE, Number of Function Evaluations)で一貫して低い FID を出すことが示されました。たとえば CIFAR-10 で同じ予算なら predictor のみより PC の方が FID で明確に改善し、「予測の数値誤差を補正が吸収する」という設計の正当性を裏づけています。
- アブレーション(VP vs VE vs sub-VP): sub-VP-SDE は VP より良い尤度(bits/dim)を与えつつ同程度の FID を保つことが示され、ドリフト・拡散の設計が尤度と品質のバランスを変えることを定量化しました。VE は高解像度で強く、VP は CIFAR-10 で安定、という棲み分けも報告されています。
- 続く EDM(Karras et al., 2022)は、この設計空間をさらに最適化し、CIFAR-10 で FID 約 1.79〜1.97(条件付き/無条件)という新記録を、Heun 法で約 35 NFE という少ステップで達成しました。同時に ImageNet 64×64 でも当時の最良級を出しています。統一定式化が「設計を系統的に探索する地図」として機能したことの証拠です。
- DPM-Solver(Lu et al., 2022)は、同じ学習済みモデルのまま、サンプリングを 10〜20 NFE まで削っても FID をほとんど劣化させないことを示し、PF-ODE の半解析解法が品質を保ったまま桁で高速化できることを定量化しました。

これらの数値が示すのは、連続時間 SDE という共通言語を持ったことで、(1) 品質(FID)と尤度(bits/dim)を同時に最高水準へ押し上げ、(2) PC サンプラーや高精度 ODE ソルバーといった新しい生成手続きを系統的に設計でき、(3) 後続研究(EDM, DPM-Solver, Consistency Models)が同じ地図の上で急速に最適化を進められた、ということです。

## メリット・トレードオフ・限界

メリット
- DDPM と NCSN を1つの理論に統一し、研究の見通しが劇的に良くなった。VP/VE/sub-VP がパラメータ違いの兄弟だと分かった。
- 連続定式化により新サンプラー(PC、高次 ODE ソルバー)を体系的に設計できる。離散の漸化式では見えなかった設計自由度が明示される。
- 確率フロー ODE により決定論的・少ステップ・高速な生成と、厳密な尤度計算・可逆エンコード(編集・補間)が可能。
- 条件付き生成や逆問題(画像補完・超解像・MRI 再構成など)へ自然に一般化できる(逆向き SDE のスコアに条件項 ∇ log p(y|x) を足すだけ。[Classifier Guidance](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0224_classifier-guidance) の理論的根拠)。

トレードオフ・限界
- 理論が高度で、確率微分方程式・数値積分・確率論(伊藤積分、Fokker-Planck)の素養を要する。初心者には参入障壁が高い。
- 逆向き SDE をそのまま素朴に数値積分すると依然として多ステップ(数百〜千)で遅い。実用には ODE 化や DPM-Solver 等の専用ソルバーが必要。
- 数値解法のステップ幅・ノイズスケジュール σ(t)・離散化点の取り方が品質に強く影響し、無策だと離散化誤差で品質が落ちる(とくに t→0 近傍の数値的扱いが繊細)。
- PF-ODE による尤度推定は Hutchinson 跡推定や ODE ソルバーの誤差で偏りが入りうる。
- 未解決の課題として、超少ステップ(1〜4)での品質維持(蒸留・Consistency Models が挑戦中)、離散・多様体・グラフデータへの SDE 定式化の拡張、SDE と ODE のどちらがどの指標で有利かの理論的理解などが研究の最前線に残っています。

## 発展トピック・研究の最前線

- EDM(Karras et al., 2022): ノイズ前処理(c_skip/c_out/c_in/c_noise)・スケジュール・ソルバー・訓練時ノイズ分布を連続定式化の上で直交設計し、CIFAR-10 で FID 約 1.79 という到達点を示した実践的バイブル。多くの後続実装の標準設計になっている。
- DPM-Solver / DPM-Solver++ / DEIS / UniPC: 確率フロー ODE の半解析的高速ソルバー群。線形項を解析的に解き、10〜20 ステップで高品質生成を実現。テキスト画像生成の推論を実用速度にした立役者。
- Consistency Models(Song, Dhariwal, Chen, Sutskever, 2023): 確率フロー ODE の軌道上の任意点を始点(終点)へ直接写す関数を学び、1〜数ステップ生成を達成。SDE/ODE 定式化の最も先鋭な高速化。蒸留版と単独学習版の両方がある。
- Flow Matching / Rectified Flow(Lipman et al.; Liu et al., 2023): スコアではなく速度場を回帰で学ぶ枠組みで、ODE 経路を直線化して少ステップ化を狙う。PF-ODE と相互翻訳可能で、近年の大規模画像・動画生成の主流設計になりつつある。
- 逆問題への応用: 確率フロー ODE/逆向き SDE に観測の尤度項を足すだけで、画像補完・超解像・MRI/CT 再構成などの逆問題を統一的に解ける(diffusion posterior sampling, ΠGDM, DPS 系)。学習済みスコアを汎用事前分布として再利用する流れ。

## さらに学ぶための関連トピック
- [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note)
- [DDIM (Denoising Diffusion Implicit Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0227_ddim_note)
- [NCSN (Noise Conditional Score Networks)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0243_ncsn_note)
- [Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching)
- [Classifier Guidance](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0224_classifier-guidance)
