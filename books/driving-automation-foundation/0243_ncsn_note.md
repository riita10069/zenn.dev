---
title: "NCSN (Noise Conditional Score Networks)"
---

# NCSN (Noise Conditional Score Networks)

## ひとことで言うと
NCSN は「データ分布のスコア(ありそうな方向を指す矢印)」を、1つの強さのノイズではなく、強いノイズから弱いノイズまで何段階ものノイズ水準で同時に学習し、強いノイズ向けの矢印から弱いノイズ向けの矢印へと段階的に乗り換えながらサンプリングする生成モデルです。[DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note) と並ぶ拡散モデルのもう一つの源流であり、[Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching) のアイデアを「実際に高品質な画像を生成できる」レベルまで初めて引き上げた成功例です。

## 直感的な理解
スコアに沿って動けばデータの谷底へ行ける、という [Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching) の発想を、素朴にそのまま実装すると画像生成は失敗します。なぜか。本物の自然画像は、巨大な空間(196608 次元など)の中で「ごく薄い膜」のようにしか存在しないからです。ランダムな出発点はほぼ確実にこの膜から遠く離れた「何もない真空」にあり、そこには訓練データが一切ないので、ネットはその場所のスコア(下り方向)を一度も学んでいません。霧の中で杖を突いても、訓練していない地点では杖がデタラメな方向を指すのです。

NCSN の解決策は「データに大きなノイズを足してぼかす」ことです。膜にノイズを足すと、膜がぼやけて空間全体に広がり、真空だった場所にも「うっすらデータがある」状態になる。すると真空地帯でもスコアが定義され学習できます。ただしノイズを足しすぎると元の鋭い画像から離れてしまう。そこで「最初は強いノイズで大まかに膜の方へ運び、徐々にノイズを弱めて細部を整える」――焼きなまし(annealing)の発想を持ち込むのが NCSN の核心です。金属を高温から徐々に冷やして欠陥のない結晶を作る焼きなましと同じで、最初は揺さぶりを大きくして全体配置を決め、だんだん揺さぶりを小さくして細部を固めます。

## 基礎: 前提となる概念

### スコアと Langevin 動力学

スコア ∇_x log p(x) は「データが密になる方向の矢印」です(詳しくは [Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching))。このスコアだけを使ってサンプルを得る古典的な手法が Langevin 動力学(Langevin dynamics)です。物理で粒子が力(ポテンシャルの勾配)を受けつつ熱的にランダムに揺れる運動を記述する方程式に由来し、機械学習では Welling & Teh(2011)の確率的勾配 Langevin 動力学(SGLD)として広まりました。更新式は次の通りです。

```
x ← x + (η/2) · s_θ(x) + sqrt(η) · z       (z ~ N(0, I)、η は小さな歩幅)
```

第1項がスコアに沿って進む決定論的な動き、第2項がランダムな揺れです。η を十分小さく、ステップ数を十分多く取ると、この反復で得られる x の分布は p(x) に収束する、という理論保証があります(正確には離散化誤差を補正する Metropolis-adjusted Langevin algorithm, MALA で厳密化されます)。第2項のランダム性があるおかげで、同じ出発点からでも毎回違うサンプルが出る(=多様性が生まれる)点が重要です。揺れがなければ単なる勾配上昇で、最寄りのピーク1点に貼りつくだけになります。

### 低次元多様体仮説と、単一スコアの2つの失敗

「自然データは高次元空間の薄い低次元多様体(膜)に集中する」という見方を多様体仮説(manifold hypothesis)と呼びます。Song & Ermon(2019)は、この仮説のもとで単純な Langevin + 単一スコアが失敗する理由を2つに整理しました。これは単なる経験則ではなく、なぜ素朴なスコアベース生成がそれまで誰も成功させられなかったかを説明する診断です。

- 失敗1(スコアが学べない領域、manifold hypothesis の帰結): 膜から離れた真空では訓練データがゼロなので、スコアが未学習でデタラメ。生成はこの真空から始まるため、最初の一歩を踏み出せない。さらに、データがちょうど低次元多様体に乗っていると、その上では log p の勾配がそもそも定義しにくい(法線方向に密度が発散する)という理論的問題もある。
- 失敗2(低密度領域・モード間の質量配分): データの谷間ではサンプルが少なくスコア推定が不正確。さらにデータが複数のかたまり(モード)に分かれていると、有限ステップの Langevin がモード間を正しい比率で行き来できず、生成されるサンプルの種類が偏る(あるモードばかり出る、混合比が崩れる)。Song & Ermon は、2つのモードを持つ単純な混合分布で、真のスコアを与えても Langevin が混合比を正しく再現できない例を構成して示しました。

NCSN はこの両方を「複数ノイズ水準」で同時に解決します。大きなノイズは多様体を空間全体へ広げて真空を埋め(失敗1)、かつモード間の谷を埋めて行き来を容易にします(失敗2)。

## 仕組みを詳しく

### 複数ノイズ水準でスコアを学ぶ(Noise Conditional)

NCSN は、大きい順に並べたノイズ強度の幾何級数列 σ_1 > σ_2 > ... > σ_L を用意します。元論文では L = 10、σ_1 = 1.0 から σ_L = 0.01 まで対数的に等間隔(隣接比 σ_i/σ_{i+1} が一定)に取りました。各 σ_i ごとに、データに σ_i のノイズを足した分布 p_{σ_i} のスコアを学びます。

重要なのは、これらを別々のネットで学ぶのではなく、1つのネット s_θ(x, σ) に「今どの強さのノイズを見ているか(σ、またはそのインデックス)」を条件として与え、全水準を同一ネットで扱う点です。これが Noise Conditional(ノイズ条件付き)の意味です。隣り合う σ のスコアは似ているので、共有することで効率よく学べます。学習則は [Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching) のデノイジングスコアマッチングそのもので、

```
ℓ(θ; σ_i) = (1/2) E_{x~p_data, x̃~N(x, σ_i²I)} [ || s_θ(x̃, σ_i) + (x̃ - x) / σ_i² ||² ]
```

を各水準について重み λ(σ_i) = σ_i² を掛けて平均し、全体損失とします。

```
L(θ) = (1/L) Σ_i λ(σ_i) · ℓ(θ; σ_i)
```

λ(σ_i) = σ_i² と置くのが肝で、こうすると各水準の損失の大きさがおおよそ揃い(教師信号 -(x̃-x)/σ_i² = -ε/σ_i が σ_i⁻¹ のスケールを持つので、二乗すると σ_i⁻²、そこに σ_i² を掛けて打ち消す)、全水準が偏りなく学習されます。これを怠ると、小さい σ の損失だけが巨大になり、大きい σ のスコアが学習されず失敗します。要するに「いろいろな強さのノイズを足し、それぞれについて、どっちへ戻せばきれいになるかを当てる」学習です。NCSNv2 ではネット出力自体を 1/σ でスケールする設計に変え、この重み付けをアーキテクチャ側に組み込んでいます。

### アニーリングされた Langevin 動力学でサンプリング

生成では「強いノイズ向けの矢印から弱いノイズ向けの矢印へ、段階的に乗り換える」アニーリングされた Langevin 動力学(annealed Langevin dynamics)を使います。

```
1. 純粋なノイズ x_0 ~ 一様 or N(0, σ_1²I) から始める。
2. for i = 1 .. L:
3.     step size α_i = ε · σ_i² / σ_L²  に設定(弱い水準ほど歩幅を小さく)
4.     for t = 1 .. T:        # 各水準で T ステップ(例 T=100)
5.         z ~ N(0, I)
6.         x ← x + (α_i / 2) · s_θ(x, σ_i) + sqrt(α_i) · z
7. 出力 x
```

強い σ_1 のスコアは空間全体で定義されているので、どこから始めても「だいたいデータのある方向」へ大きく動けます(失敗1の解決)。徐々に σ を弱めることで、ぼけた大まかな構図から細部へと焦点を移します。モードをまたぐ移動は谷が埋まっている強いノイズ水準のうちに済ませ、弱いノイズ水準では位置をほとんど動かさず細部だけ整えるので、モード間の質量配分も正しくなりやすい(失敗2の解決)。歩幅 α_i ∝ σ_i² は、各水準で「ノイズに対する信号対雑音比」を一定に保つよう理論的に設計されています。

数値イメージ: [3, 32, 32] の CIFAR-10。最初は全ピクセルが砂嵐。σ_1=1.0 の矢印で大まかな構図(空は上、地面は下のような塊)ができ、σ が小さくなるにつれ物体の輪郭、最後にテクスチャの細部が整い、CIFAR-10 らしい画像が出来上がります。L=10、各水準 T=100 で、合計約 1000 回のネット評価が必要です。これが GAN の1回フォワードに比べ桁違いに遅い原因です。

### NCSNv2 の改良(2020)

元の NCSN は 32×32 程度なら動きますが、高解像度(64〜256)では学習が発散したり生成が崩壊したりして不安定でした。Song & Ermon(2020)は5つの設計指針を理論的根拠とともに与えました。(1) σ_1 を訓練データ点間の最大ユークリッド距離程度に取る――そうしないと強い水準でモードをまたげない。(2) σ を幾何級数で選び、隣接水準の比を 1 に近づける(高次元ほど L を増やす)――比が大きすぎると水準間で分布が急変し乗り換えに失敗する。(3) ネットの出力を 1/σ でスケールする s_θ(x,σ) = s̃_θ(x)/σ ――出力の大きさを水準間で揃える。(4) サンプリングのステップ数 T と歩幅 ε を、各水準のノイズに対する信号対雑音比が一定になるよう理論式から決める。(5) パラメータの指数移動平均(EMA)を使い、学習後半の振動を平滑化する。これらにより 256×256 の顔画像生成まで安定して回るようになりました。これらは後に [Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde) の連続化や EDM の前処理設計へ受け継がれます。

### DDPM との関係

NCSN と [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note) は独立に提案されましたが、後に [Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde) によって、両者は「ノイズを足す過程の連続時間極限」の特別な場合だと示されました。NCSN はノイズを足しても元の信号を縮めない分散発散型(VE-SDE, Variance Exploding: dx = sqrt(d[σ²]/dt) dw)、DDPM は信号を縮めつつノイズを足し全体の分散を一定に保つ分散保存型(VP-SDE, Variance Preserving)です。NCSN の「足したノイズを当てる学習」も DDPM の「ε を予測する学習」も、正体はどちらもデノイジングスコアマッチングです。アニーリング Langevin は、SDE の言葉でいえば各時刻の補正(corrector)ステップに相当し、[Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde) の Predictor-Corrector サンプラーへ一般化されます。

## 手法の系譜と主要論文

- Hyvärinen (2005) / Vincent (2011): スコアマッチングと DSM の理論的基礎。NCSN の学習則の直接の祖先([Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching))。
- Welling & Teh (2011), "Bayesian Learning via Stochastic Gradient Langevin Dynamics" (ICML 2011): スコア(勾配)だけでサンプリングする Langevin の機械学習版を確立。NCSN のサンプラーの基礎。
- Song & Ermon (2019), "Generative Modeling by Estimating Gradients of the Data Distribution" (NeurIPS 2019): 本トピックの主役。複数ノイズ水準を1つのネットで同時に学ぶ NCSN と、アニーリング Langevin を提案。単一スコアの2つの失敗を「ノイズで空間を埋める」ことで克服し、スコアベース手法で初めて GAN に迫る画質を実現した。
- Song & Ermon (2020), "Improved Techniques for Training Score-Based Generative Models" (NeurIPS 2020): NCSNv2。σ スケジュール・出力スケーリング・EMA などを体系化し高解像度で安定化。
- Ho, Jain, Abbeel (2020), "Denoising Diffusion Probabilistic Models" (NeurIPS 2020): 並行して発展した [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note)。変分下限の言葉だが NCSN と表裏一体。
- Song et al. (2021), "Score-Based Generative Modeling through SDEs" (ICLR 2021): NCSN(VE-SDE)と DDPM(VP-SDE)を連続時間で統一([Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde))。

## 論文の実験結果(定量データ)

評価指標は FID(生成分布と本物分布の特徴空間距離、低いほど良い)と Inception Score(IS、鮮明さと多様性、高いほど良い)です。

- NCSN(Song & Ermon, 2019)は CIFAR-10(32×32)で Inception Score 8.87、FID 25.32 を達成しました。当時の代表的 GAN(WGAN-GP の IS ≈ 7.86、SNGAN の IS ≈ 8.22、PixelCNN++ などの尤度モデルの IS ≈ 5 前後)と比べ、IS では GAN を上回り、尤度ベース以外で初めて GAN と真正面から競える数値でした。FID 25 は「人間が一見して自然画像と分かるが、近づくとほころびがある」水準です。
- NCSNv2(Song & Ermon, 2020)は CIFAR-10 で FID をおよそ 10.87 まで改善し、さらに 256×256 の CelebA-HQ(顔画像)や LSUN(寝室・教会など)といった高解像度データでも安定生成を示しました。元 NCSN が発散・破綻していた解像度域に到達したことが最大の貢献です。
- アブレーション(NCSNv2 論文の核心): 提案した5指針を1つずつ外す実験が示されています。σ スケジュール則を外す、出力の 1/σ スケーリングを外す、EMA を外す、をそれぞれ試すと、いずれも生成が崩壊またはノイズ混じりになる。とくに「σ_1 をデータ点間最大距離に合わせる」を守らないと、強い水準でモードを横断できず多様性が落ちる。EMA を外すと学習後半でサンプル品質が振動する。これは「複数ノイズ水準の設計そのものが品質を決める」という NCSN の主張を定量的に裏づけるものです。
- 速度の代償: NCSN/アニーリング Langevin は L×T ≈ 1000 回のネット評価を要し、1サンプルに数秒オーダー(GPU)。GAN の1回フォワードに比べ桁違いに遅く、これが後の [DDIM (Denoising Diffusion Implicit Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0227_ddim_note)、確率フロー ODE、蒸留・少ステップ化研究の直接の動機になりました。
- 後継との比較: 同じ VE 系を連続化・最適化した EDM(Karras et al., 2022)は CIFAR-10 で FID 約 1.97 まで到達し、NCSN 系の設計自由度(ノイズ範囲・前処理・ソルバー)を理論的に詰めれば最先端に届くことを示しました。NCSN→NCSNv2→VE-SDE→EDM という改良の系譜は、同一の物理的枠組み(信号を縮めずノイズだけ増やす)を保ったまま設計を磨いた好例です。

これらの数値が示すのは、「ノイズで空間を埋める」というアイデアが、理論止まりだったスコアマッチングを実用的な画像生成器へ変えたこと、そして σ スケジュールという設計自由度が品質を大きく左右することです。

## メリット・トレードオフ・限界

メリット
- スコアベース生成を「実際に高品質画像を作れる」実用水準に初めて引き上げた。多様体仮説に基づく失敗の診断と処方箋を与えた。
- 複数ノイズ水準により、データのない真空領域からでも生成を正しく始められる(失敗1の解決)。
- 単一ネットで全ノイズ水準を扱え、学習は DSM(単純な二乗誤差回帰)で安定。
- Langevin のランダム項により多様なサンプルが出て、モード崩壊(GAN の弱点)が起きにくい。

トレードオフ・限界
- アニーリング Langevin を L×T 回(約1000回)評価するため生成が遅い。
- ノイズ水準のスケジュール(σ の範囲・段数・隣接比)とステップ幅の調整が品質に強く影響し、設計が職人芸になりがち(NCSNv2 で一部理論化されたが完全ではない)。
- 高解像度では学習が不安定で、NCSNv2 の5工夫を要した。出力スケーリングや EMA を欠くと容易に崩壊する。
- Langevin の収束保証は無限小の歩幅・無限ステップを仮定したもので、有限ステップでは離散化バイアスが残る(MALA の Metropolis 補正は高次元で受理率が落ちるため画像では使えない)。
- 未解決の課題として、超高解像度でのスケーラビリティ、低密度領域でのスコア推定誤差の制御、条件付き生成への自然な拡張(後の [Classifier Guidance](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0224_classifier-guidance) や Classifier-Free Guidance が担う)などが残されました。

## 発展トピック・研究の最前線

- 連続時間統一: 離散的な複数 σ を連続のノイズ過程に拡張し、逆向き SDE / 確率フロー ODE を導いた [Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde)。NCSN は VE-SDE として再解釈される。
- Predictor-Corrector サンプラー: SDE 数値積分(predictor)と各時刻での Langevin 補正(corrector)を組み合わせる方式。NCSN のアニーリング Langevin を「補正ステップ」として一般化したもの。
- EDM(Karras et al., 2022): VE 系の前処理・ノイズスケジュール・ソルバー設計を整理し、CIFAR-10 で FID 約 1.97 まで到達。NCSN 系の設計自由度を理論的に最適化した到達点。前処理(c_skip, c_out, c_in, c_noise の係数設計)は NCSNv2 の出力スケーリングの一般化と読める。
- 少ステップ化: 約1000ステップという遅さに対し、[DDIM (Denoising Diffusion Implicit Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0227_ddim_note)、progressive distillation、Consistency Models、Flow Matching などが1〜数十ステップ生成を追求している。
- 条件付き生成: スコアにベイズ分解で条件項を足す [Classifier Guidance](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0224_classifier-guidance) や、その分類器を不要にした Classifier-Free Guidance が、NCSN/拡散の枠組みに「何を作るか」を指定する能力を加えた。

## さらに学ぶための関連トピック
- [Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching)
- [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note)
- [Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde)
- [DDIM (Denoising Diffusion Implicit Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0227_ddim_note)
- [Classifier Guidance](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0224_classifier-guidance)
