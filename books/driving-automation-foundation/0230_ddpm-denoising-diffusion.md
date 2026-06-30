---
title: "DDPM（拡散モデル基礎）"
---

# DDPM（拡散モデル基礎）

## ひとことで言うと
DDPM（Denoising Diffusion Probabilistic Model、拡散確率モデル）は、「きれいなデータに少しずつノイズを足して最後は完全なランダムノイズにする」過程を逆向きにたどることで、ノイズからデータを作り出す生成モデルです。逆過程の各ステップで「混ざっているノイズはどれか」をニューラルネットに当てさせ、その予測を使って少しずつノイズを剥がします。画像生成（Stable Diffusion など）で一躍主流になり、ロボットの行動生成や自動運転の軌道生成（未来の走行経路をノイズから生成するプランナ）の土台にもなっています。

## 直感的な理解
1枚のきれいな写真に、ほんの少しだけ砂嵐のようなノイズを足します。これを何百回も繰り返すと、最後は元が何だったか分からない真っ白な砂嵐になります。この「壊していく過程」は単純で、誰でもできます。難しいのは逆で、砂嵐から写真を復元することです。

拡散モデルのアイデアは「壊す過程を細かく刻めば、1ステップ分の逆操作（ほんの少しノイズを剥がす）は学習で当てられるくらい簡単になる」というものです。一気に砂嵐から写真を作るのは無理でも、「今より少しだけノイズが少ない状態」を作るだけなら、ニューラルネットに学習させられます。それを何百回も繰り返せば、砂嵐からきれいな写真にたどり着けます。生成を「小さな逆操作の積み重ね」に分解したのが拡散モデルの本質です。難しい問題を、解ける小問題の連鎖に分割する、というのが核心の発想です。

自動運転・ロボットとの関係: 「次にどう動くか」は1つに決まりません。交差点で直進・左折・停止など、複数の妥当な未来があります。これをマルチモーダル（多峰性、答えの山が複数あること）と呼びます。1本だけ軌道を出す回帰モデルは、これらの山の平均（どこにも行かない中途半端な軌道）を出しがちですが（mode averaging）、拡散モデルはノイズの引き方を変えるだけで複数の妥当な軌道を確率的にサンプリングできるため、この問題に向いています。

## 基礎: 前提となる概念

生成モデル: 「本物っぽい新しいデータを作り出せるモデル」のこと。大量の犬の画像を学習すれば新しい犬の画像を、過去の運転データを学習すればありえそうな未来の軌道を作れます。数学的にはデータ分布 p_data(x) を近似し、そこからサンプリングできるようにすることです。

拡散モデル以前の生成モデルとそれぞれの弱点:
- GAN（敵対的生成ネットワーク）: 生成器と判別器を競わせる。鮮明な画像を作れるが学習が不安定で、本来多様であるべき出力が一部に偏るモード崩壊（mode collapse）が起きやすい。
- VAE（変分オートエンコーダ）: エンコーダで潜在変数に圧縮しデコーダで復元する。学習は安定するが出力がぼやけがち。
- 正規化フロー（normalizing flow）: 確率を厳密に計算できるが、可逆性とヤコビアン計算の都合からネットワーク構造に強い制約がある。

ガウスノイズ: 正規分布 N(0, I) から引いた乱数。平均0・分散1の「素直なばらつき」を持つランダムな値で、拡散の「砂嵐」の正体です。ガウスを使うのは、ガウスを足し合わせてもガウスのままという再生性があり、複数ステップの合成が解析的に閉じるからです。

マルコフ過程（Markov chain）: 「次の状態は直前の状態だけで決まる」確率過程。前向き・逆向きとも各ステップが直前だけに依存する形で定義されます。

ELBO（変分下限、Evidence Lower BOund）: データの対数尤度（モデルがデータをどれだけよく説明するか）を直接最大化するのは難しいので、それを下から押さえる扱いやすい量を代わりに最大化する標準的な手法。DDPM の理論的な損失はこの ELBO から導かれます。

スコア（score）: 対数密度の勾配 ∇_x log p(x)。各点で「密度が増える方向」を指すベクトル場で、これが分かればランジュバン動力学でサンプリングできます。後述するように DDPM のノイズ予測はスコア推定と本質的に同じです。

## 仕組みを詳しく

拡散モデルは「前向き過程（ノイズを足す）」と「逆向き過程（ノイズを除く）」の2つから成ります。

前向き過程（forward / diffusion process）: きれいなデータ x_0 に、少しずつガウスノイズを足していきます。ステップ数 T（典型的に T=1000）として、t 番目のステップで

```
x_t = sqrt(1 - β_t) * x_{t-1} + sqrt(β_t) * ε      （ε ~ N(0, I)）
```

β_t（各ステップで足すノイズ量を決める小さな数、例えば 0.0001 から 0.02 へ徐々に増やす）はあらかじめ決めた「ノイズスケジュール」です。重要な性質として、t ステップ後の状態は1ステップずつ計算しなくても一発で書けます（ガウスの再生性のおかげ）。

```
x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
ただし ᾱ_t = (1-β_1)(1-β_2)...(1-β_t)
```

ᾱ_t（アルファバー）は「元データがどれだけ残っているか」を表す 0〜1 の係数。t が小さいと ᾱ_t ≈ 1 でほぼ元データ、t=T では ᾱ_t ≈ 0 でほぼ純粋ノイズです。この一発計算のおかげで、学習時に好きな t をランダムに選んで、その時点のノイズ入りデータを即座に作れます（1000 回足す必要がない）。

tensor shape の例: 軌道が [B, 128] のベクトル（B はバッチサイズ、128 = 64ステップ × 2 信号など）なら、x_0 は [B, 128]、足すノイズ ε も [B, 128]、t は [B] の整数ベクトル、x_t も [B, 128]。形は変わりません。画像なら [B, 3, H, W] のまま。

逆向き過程（reverse / denoising process）: 生成時は純粋ノイズ x_T ~ N(0, I) から始め、ノイズを少しずつ取り除いて x_0 に戻します。この「1ステップ分のノイズの除き方」をニューラルネットに学習させます。理論的には逆過程の各ステップ q(x_{t-1} | x_t, x_0) は条件付きでガウスになることが計算でき、その平均と分散を予測する問題に帰着します。

DDPM の巧妙な点は「ノイズそのものを予測する」形に問題を変えたことです。ネットワーク ε_θ(x_t, t)（θ は重み）に「x_t に混ざっているノイズ ε はどれだったか」を当てさせます。理論上は ELBO から各時刻の KL ダイバージェンスの和という複雑な損失が出ますが、Ho らは時刻ごとの重み係数を1に落とした単純化版を提案し、実質ただの二乗誤差で済むようにしました。

```
損失 L_simple = E_{t, x_0, ε} || ε - ε_θ(x_t, t) ||²
```

学習手順:
1. データ x_0 を取る。
2. t を 1〜T から一様ランダムに選ぶ。
3. ノイズ ε を引いて x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε を作る。
4. ネットワークに x_t と t を入れて ε_θ を出させる。
5. ε と ε_θ の二乗誤差を最小化する。

生成（サンプリング）はこの逆をたどります。x_T からスタートし、各 t で ε_θ を使って少しノイズを取り、わずかなランダム性 z（z ~ N(0,I)、t=1 では足さない）を足しながら x_{t-1} を作り、これを T 回繰り返して x_0 に到達します。

```
x_{t-1} = (1/sqrt(α_t)) ( x_t - (β_t/sqrt(1-ᾱ_t)) ε_θ(x_t,t) ) + σ_t z
ノイズ x_T ──ε_θで除去──> x_{T-1} ──> … ──> x_1 ──> きれいな x_0
（この矢印を 1000 回くらい繰り返す）
```

時刻 t の渡し方: ネットワークに「今どのノイズレベルか」を伝えるため、t を正弦波埋め込み（sinusoidal embedding、整数 t を異なる周波数の sin/cos の組で連続ベクトルに変換したもの。Transformer の位置エンコーディングと同じ発想）にして与えるのが定番です。条件付き生成では、これに加えてテキストや BEV 特徴などの文脈をクロスアテンションや AdaLN（条件からスケールとシフトを作り特徴を変調）で注入します。

なぜ「ノイズ予測」が効くのか: x_t と予測ノイズ ε_θ が分かれば、前向き式を逆に解いて「元のきれいなデータ x_0 の推定値」x̂_0 = (x_t - sqrt(1-ᾱ_t) ε_θ) / sqrt(ᾱ_t) が即座に得られます。さらにスコア ∇ log p_t(x) とノイズ予測は ε_θ ≈ -sqrt(1-ᾱ_t) ∇ log p_t(x) という定数倍の関係にあり、DDPM は Song & Ermon のスコアマッチングと本質的に同じことをしています。この同一視が後に SDE/ODE による統一理論につながりました。予測対象としては ε（ノイズ）、x_0（データ）、v（その線形結合、v-prediction）の3通りがあり、ε 予測が画像で安定して良いとされる一方、v 予測は蒸留や少ステップで好まれます。

## 手法の系譜と主要論文

- Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli (2015, ICML)「Deep Unsupervised Learning using Nonequilibrium Thermodynamics」(arXiv:1503.03585): 拡散モデルの原典。非平衡熱力学（物質が拡散して均一になる過程）にヒントを得て「データを徐々に壊し、その逆過程を学べば生成できる」枠組みを最初に提案。理論は美しかったが当時の生成品質は他手法に及ばず、注目は限定的でした。
- Song & Ermon (2019, NeurIPS)「Generative Modeling by Estimating Gradients of the Data Distribution」(arXiv:1907.05600): スコアマッチングという別角度から本質的に同じ枠組みに到達。ノイズレベルごとにスコアを学び、ランジュバン動力学でサンプリングする NCSN を提案。
- Ho, Jain, Abbeel (2020, NeurIPS)「Denoising Diffusion Probabilistic Models」(arXiv:2006.11239): 本トピックの中心。ELBO から導かれる複雑な損失を「予測ノイズの単純な二乗誤差 L_simple」に簡略化し、ノイズスケジュールと U-Net 構造を工夫して、GAN に匹敵する高品質画像生成を初めて安定的に達成しました。動機は「厳密な ELBO より各時刻の重み係数を落とした単純化版のほうが画質も安定性も良かった」こと。
- Nichol & Dhariwal (2021, ICML)「Improved Denoising Diffusion Probabilistic Models」(arXiv:2102.09672): コサイン型ノイズスケジュール（線形より情報の壊し方がなめらか）と分散の学習を導入し、対数尤度とサンプル効率を改善。
- Dhariwal & Nichol (2021, NeurIPS)「Diffusion Models Beat GANs on Image Synthesis」(arXiv:2105.05233): classifier guidance（別途学習した分類器の勾配で生成を条件方向へ押す）により ImageNet で当時最強だった BigGAN を FID で上回り、拡散が主流であることを決定づけました。
- Song ら (2021, ICLR)「Score-Based Generative Modeling through Stochastic Differential Equations」(arXiv:2011.13456): DDPM とスコアマッチングを連続時間の SDE/ODE として統一。確率的サンプラ（reverse SDE）と決定論的サンプラ（probability flow ODE）の両方を導き、[DDIM 高速サンプリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0228_ddim-sampling)・[Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching)・[Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow) へつながる連続時間の見方の基礎を作りました。
- Chi ら (2023, RSS)「Diffusion Policy」(arXiv:2303.04137): 拡散をロボットの行動（連続アクション列）生成に応用。マルチモーダルな行動分布を表現でき、模倣学習で従来手法を上回ることを示しました。自動運転の軌道生成（MotionDiffuser, Diffusion-ES など）もこの流れにあります。

## 論文の実験結果(定量データ)

指標の意味: FID（Fréchet Inception Distance）は生成画像と本物の分布の近さで、小さいほど良い。IS（Inception Score）は鮮明さと多様性の指標で、大きいほど良い。NLL（bits/dim、1次元あたり何ビットで符号化できるか）はデータの説明力で、小さいほど良い。

- Ho ら (2020): CIFAR-10（32×32 の小画像データセット）で、DDPM は FID 3.17 を達成し、当時の有力 GAN を上回る最先端級の品質を示しました。これは「学習が不安定と言われた生成モデルで、二乗誤差だけの安定学習が GAN 並みの鮮明さを出せる」ことを定量的に示した点で画期的でした。アブレーションでは、ELBO そのもの（L_vlb）より単純化損失 L_simple のほうが FID が良いこと、ノイズ ε を予測する形が x_0 を直接予測する形より安定して良いことを報告。LSUN（寝室・教会など 256×256）でも高品質サンプルを生成しました。
- Nichol & Dhariwal (2021): コサインスケジュールと分散学習により、線形スケジュールの DDPM より bits/dim とサンプル効率が改善することを示し、少ステップでの劣化も緩和しました。
- Dhariwal & Nichol (2021): classifier guidance を導入した拡散モデルが ImageNet 128/256/512 で BigGAN の FID を上回り、論文題「Diffusion Models Beat GANs」の通り拡散優位を定量的に確立しました。guidance を強めると FID と多様性（recall）がトレードオフになることも示し、以後の guidance 研究の出発点になりました。
- Diffusion Policy (2023): ロボット操作の複数ベンチマーク（数十タスク）で、行動を拡散生成する方式が、回帰や混合密度ネットワーク（[Mixture Density Network 軌道予測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0189_mixture-density-network-planner)）など従来の方策表現に対し成功率を平均で大きく（タスクによっては数十パーセントポイント、報告による）上回りました。アブレーションで「マルチモーダルな行動を1本に潰す回帰系は、分岐のある状況（押すか引くか等）で性能が落ちる」ことが示され、生成的方策の利点を裏づけました。

共通して示されているのは「拡散は安定学習で高品質・高多様性を達成できる」ことと、その代償としての「生成の遅さ」です。CIFAR-10 の高品質サンプルにも T=1000 ステップを要しており、この遅さの克服が次の研究（[DDIM 高速サンプリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0228_ddim-sampling) 以降）の主題になりました。

## メリット・トレードオフ・限界

メリット
- 学習が安定。単純な二乗誤差を最小化するだけで、GAN のような不安定さやモード崩壊がない。損失曲線が素直に下がる。
- 多様な出力を確率的にサンプリングでき、マルチモーダルな分布（複数の妥当な軌道や画像）を表現できる。
- 損失も生成も実装が素直で、条件付け（テキスト・画像・BEV 特徴での誘導、classifier-free guidance）を後付けしやすい。

トレードオフ・限界
- 生成が遅い。標準 DDPM は1サンプルに数百〜千回のネットワーク呼び出しが必要で、リアルタイム制御が要る自動運転には重い。これが最大の弱点。
- ノイズスケジュール（β_t の決め方）、予測対象（ε/x_0/v）、guidance 強度など調整すべきハイパーパラメータがある。
- 1回ごとの生成にランダム性が入るため、同条件でも出力が毎回変わる。評価の再現には乱数シード固定が必要。
- 未解決の課題として、超少ステップ生成での品質維持、誘導の強さと多様性のトレードオフ、離散・構造化データへの拡張などがあります。

## 発展トピック・研究の最前線
- サンプリング高速化: [DDIM 高速サンプリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0228_ddim-sampling)（決定論的・少ステップ）、DPM-Solver（高次 ODE 解法）、蒸留（progressive distillation, consistency models）。生成の遅さを直接攻める系統。
- 連続時間の統一理論（SDE/ODE）から派生した [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) と [Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow)。これらは「最初からまっすぐな ODE を学ぶ」ことで拡散の遅さを構造的に解消する後継手法です。
- 潜在拡散（Latent Diffusion / Stable Diffusion）: 画素空間でなく VAE で圧縮した潜在空間で拡散し、計算量を大幅に削減。テキスト→画像の実用化を牽引しました。
- classifier-free guidance: 別の分類器を使わず、条件付き・無条件のノイズ予測を内挿/外挿して条件遵守を強める手法。現在の条件付き拡散の標準。
- 行動・軌道生成への応用: Diffusion Policy, MotionDiffuser, 自動運転のプランナ。マルチモーダルな未来を表現できる利点が活き、安全制約をサンプリング時に注入する研究（guided sampling）も進んでいます。

## さらに学ぶための関連トピック
- [DDIM 高速サンプリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0228_ddim-sampling)
- [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching)
- [Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow)
- [Mixture Density Network 軌道予測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0189_mixture-density-network-planner)
- [Winner-Takes-All / min-of-N 損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0602_winner-takes-all-loss)
