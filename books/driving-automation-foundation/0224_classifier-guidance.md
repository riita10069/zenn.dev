---
title: "Classifier Guidance"
---

# Classifier Guidance

## ひとことで言うと
Classifier Guidance(分類器ガイダンス)は、拡散モデルで「ただ画像を作る」のではなく「猫の画像を作る」のように指定した条件に従わせる初期の手法です。やり方は、拡散モデルとは別に「ノイズだらけの画像を見て、それが何のクラスか当てる分類器」を学習しておき、その分類器が示す「もっと猫らしくなる方向」を生成の各ステップに少しずつ足し込みます。これで無条件の拡散モデルを後付けで条件付き生成に変えられ、しかも「条件への忠実さ」をつまみ1つで自在に強められます。「拡散モデルが画質で GAN を超えた」と宣言した論文の中心技術です。

## 直感的な理解
無条件の拡散モデル([DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note) や [NCSN (Noise Conditional Score Networks)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0243_ncsn_note))は「データらしい何か」を作れますが、何を作るかは指定できません。ImageNet で学習したモデルに「犬が欲しい」と頼んでも、出てくるのは1000クラスのどれかランダムです。

Classifier Guidance のイメージはこうです。生成の各ステップで、モデルは2つの矢印を受け取ります。1つ目は無条件モデルが出す「(クラスを問わず)もっと自然な画像になる方向」。2つ目は分類器が出す「もっと犬らしくなる方向」。この2本を合成した方向へ進めば、自然でかつ犬らしい画像へ向かいます。2つ目の矢印を何倍に強調するか(guidance scale)で、「無難だが多様」から「いかにも犬らしいが似たり寄ったり」までを連続的に調整できます。鳥の群れを誘導する2つの力――自然に飛ぶ本能と、餌のある方へ向かう誘い――を合成するようなものです。重要なのは、この「2本目の矢印」を、拡散モデル本体を一切再学習せずに後付けできることです。

## 基礎: 前提となる概念

### 条件付き生成とスコアのベイズ分解

理論的な下地は [Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde) にあります。スコア(ありそうな方向を指す矢印 ∇_x log p)の言葉で条件付き生成を書くと、必要なのは「条件 y のもとでのスコア ∇_x log p(x|y)」です。逆向き SDE のスコアをこれに差し替えれば条件付き生成になる、というのが SDE 定式化が明示した点でした。ベイズの定理 p(x|y) = p(y|x)p(x)/p(y) の対数を取り x で微分すると、

```
∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)
```

- 第1項 ∇_x log p(x) は無条件の拡散モデルがすでに学んでいるスコア。
- 第2項 ∇_x log p(y|x) は「画像 x が与えられたとき、それがクラス y である確率の勾配」、つまり分類器の出力(対数尤度)を x で微分したもの。p(y) は x に依存しないので微分で消える。

だから「無条件拡散モデルのスコア + 分類器の勾配」で条件付きスコアが作れる――これが核心です。新しい確率モデルを学び直す必要はなく、既存の無条件モデルに分類器の勾配を「足す」だけで条件付けできます。生成モデルと識別モデル(分類器)という、機械学習の2大系統を1本のサンプリング式の中で合流させているのが美しい点です。

### ノイズ付き画像を分類できる分類器

ただし普通の分類器ではいけません。生成は純粋なノイズから始まり、序盤(時刻 t が大きい)は画像がほぼ砂嵐です。その砂嵐が「将来どのクラスになりそうか」を判定できないと誘導が始まりません。普通の ImageNet 分類器はノイズ画像を入れるとデタラメな勾配を返します。そこで、いろいろな時刻 t のノイズ入り画像 x_t を入力にクラス y を当てる、時刻条件付き分類器 p_φ(y | x_t, t) を別途学習します。前向き拡散過程で訓練画像に時刻 t 相当のノイズを足し、そのラベルを当てるよう普通の交差エントロピーで訓練するだけです。Dhariwal & Nichol は拡散モデルの U-Net のエンコーダ(ダウンサンプリング部)を流用した分類器を使い、各時刻のノイズ画像で訓練しました。

## 仕組みを詳しく

### 各サンプリングステップで勾配を足す

DDPM は通常、ノイズ予測 ε_θ(x_t, t) を出します。これはスコアと ε_θ = -√(1-ᾱ_t)·s_θ という係数関係にあります(ᾱ_t は時刻 t までに信号が縮む割合を表す累積係数)。条件付きスコアの分解にこの関係を代入すると、修正後のノイズ予測は次になります。

```
ε̂(x_t, t) = ε_θ(x_t, t) - s · √(1-ᾱ_t) · ∇_{x_t} log p_φ(y | x_t, t)
```

- ∇_{x_t} log p_φ(...) は「x_t をどう動かせばもっとクラス y らしくなるか」を指す矢印。分類器を x_t で逆伝播して得る(各ステップで分類器の backward が1回必要)。
- s は guidance scale(ガイダンス強度)という調整つまみ。理論的には s=1 が厳密なベイズ条件付けに対応し、s>1 は「条件項を過剰に効かせる」ことを意味する。

直感的には、各ステップで「データらしさ(無条件スコア)」と「クラス y らしさ(分類器勾配)」の2本の矢印を 1:s の比で合成し、その方向へ進めています。s=0 なら無条件生成、s を上げるほど指定クラスへ強く引き寄せられます。同じ操作は逆向き SDE / 確率フロー ODE のスコア項に分類器勾配を加える形でも書け、どのサンプラーにも組み込めます。

### なぜ s を1より大きくするのか

ベイズ的には s=1 が正しいのに、Dhariwal & Nichol は s>1(典型的には s=1〜10)で品質が大きく上がることを発見しました。理由は、分類器勾配をスカラー s 倍することが、条件付き分布を p(x) · p_φ(y|x)^s に比例する形にする(分類器分布を s 乗してシャープにする)ことに相当し、「より典型的で、より確実にそのクラスに見える」サンプルへ質量を集中させるからです。これは数学的には条件付き分布の温度を下げる(low-temperature sampling)操作で、分布の裾を切り捨ててピーク付近に集める効果を持ちます。後述のトレードオフはここから生まれます。

### guidance scale のトレードオフ

s を上げると、人間が見た確からしさや事後分類精度は上がり、FID(生成画像と本物の分布の近さ、低いほど良い)も適度な範囲では改善します。しかし s を上げすぎると、いかにもそのクラスらしい典型例ばかりになり、生成の多様性(Recall、本物の分布のどれだけを覆えるか)が下がります。逆に s が小さいと多様だが条件への忠実さ(Precision、生成物がどれだけ本物らしいか)が弱い。この「つまみで忠実さと多様性を交換できる」性質が実用上きわめて便利でした。Precision/Recall(Kynkäänniemi et al., 2019 の改良版)は、生成サンプルが本物分布の高密度領域に入る割合(Precision、品質)と、本物サンプルが生成分布に覆われる割合(Recall、多様性・網羅性)で測ります。FID 単独では「品質を上げたのか多様性を犠牲にしたのか」が区別できないため、この2指標を併用するのが誠実な評価です。

### 具体的な数値イメージ

ImageNet 256×256(形状 [3, 256, 256]、1000クラス)で「クラス=ゴールデンレトリバー(y)」を生成するとします。純粋なノイズから始め、各ステップで無条件モデルのノイズ予測に、分類器が出す「もっと犬らしくなる方向」を s 倍して引き(上式)、x_t を更新します。s≈1 では犬は出るが背景や姿勢が多様、s≈10 では教科書的なゴールデンレトリバーが安定して出るが似た構図が増える、といった具合です。各ステップで拡散モデルの forward と分類器の forward+backward が走るため、無条件生成より1ステップあたりの計算が増えます。

## 手法の系譜と主要論文

- Sohl-Dickstein et al. (2015) / Song et al. (2021, SDE): 条件付きスコアをベイズの定理で「無条件スコア + 条件項」に分解できるという理論的下地を提供。SDE 定式化では、逆向き SDE のスコアを条件付きスコアに差し替えるだけで条件生成になることが明示された([Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde))。
- Nichol & Dhariwal (2021), "Improved DDPM" (ICML 2021): 分散の学習やノイズスケジュール改良で DDPM の尤度と品質を底上げし、Dhariwal & Nichol の ADM の前提を整えた直前の仕事。
- Dhariwal & Nichol (2021), "Diffusion Models Beat GANs on Image Synthesis" (NeurIPS 2021): 本トピックの主役。ノイズ入り画像用の時刻条件付き分類器を別途学習し、その勾配で拡散サンプリングを誘導する Classifier Guidance と、guidance scale による品質・多様性の制御を提案。あわせて U-Net の改良(チャンネル数増、複数解像度での self-attention、BigGAN 風のアップ/ダウンサンプリング、適応的グループ正規化 AdaGN)も行い、これらを総称して ADM(Ablated Diffusion Model)と呼ぶ。
- Ho & Salimans (2021/2022), "Classifier-Free Diffusion Guidance" (NeurIPS 2021 Workshop, 後に広く普及): Classifier Guidance の弱点(別の分類器が必要、勾配逆伝播が必要)を解消した後継。条件あり予測 ε_θ(x_t, t, y) と条件なし予測 ε_θ(x_t, t, ∅) を1つのネットで学び(訓練時にランダムに条件 y をドロップして ∅ に置換)、推論時に ε̂ = ε_θ(∅) + w·[ε_θ(y) - ε_θ(∅)] で誘導する。分類器が不要で実装が簡潔、勾配の逆伝播も要らないため、現在のテキスト画像生成(Stable Diffusion, Imagen, DALL·E 2 等)の主流。
- GLIDE(Nichol et al., 2022)/ Imagen(Saharia et al., 2022): Classifier-Free Guidance をテキスト条件に拡張し、guidance scale を上げることでテキスト整合性を高める実務を確立。GLIDE 論文は CLIP guidance(分類器の代わりに CLIP を使う classifier guidance)と CFG を直接比較し、CFG の優位を示した。Classifier Guidance はその直接の先駆けとして歴史的に重要。

## 論文の実験結果(定量データ)

Dhariwal & Nichol(2021)の中心的な数値です。FID は低いほど良く、IS(Inception Score)は高いほど良い。sFID(空間特徴を重視した FID 変種)、Precision/Recall は忠実さ/多様性を測ります。

- ImageNet 128×128(クラス条件付き): ADM + Classifier Guidance が FID 2.97 を達成し、当時最強の GAN である BigGAN-deep(FID 6.02)を大きく下回りました。論文タイトル通り「拡散モデルが GAN を画質で上回る」を初めて明確に示した結果です。
- ImageNet 256×256: ADM-G(guidance あり)が FID 4.59 で、BigGAN-deep の FID 6.95 を上回りました。guidance なしの ADM は FID 10.94 程度で、guidance がこの大幅改善の主因です。
- ImageNet 512×512: ADM-G が FID 7.72 で、同じく BigGAN-deep(FID 8.43)を上回り、高解像度でも優位を示しました。
- guidance scale のアブレーション: s を 0 から増やすと、Inception Score と Precision(忠実さ)は単調に上がる一方、Recall(多様性)は単調に下がることが定量的に示されました。FID は中庸の s で最小になる谷型(上げすぎると典型例集中で再び悪化)で、これが「品質と多様性のトレードオフ点を s で選ぶ」という実務指針の根拠になりました。たとえば 256×256 では s=1.0 付近が FID の最良点付近で、それを超えると IS は伸び続けるが FID は悪化します。
- アーキテクチャ改良のアブレーション: U-Net の改良(深さ・幅・attention 解像度・AdaGN)を1つずつ加えると FID が段階的に改善することが示され、「guidance だけでなくバックボーン強化も GAN 超えに寄与した」ことが分離されています。とくに複数解像度での self-attention と AdaGN(時刻・クラス情報で正規化のスケール/シフトを変える)の寄与が大きいと報告されています。これは「画質の勝利は手法の組み合わせの結果」であることを誠実に示したアブレーションです。
- BigGAN との多様性比較: BigGAN は FID は競争力があるものの Recall が低く(モード崩壊で本物分布の一部しか覆えない)、ADM-G は同等以下の FID でより高い Recall を示しました。つまり「同じ画質なら拡散モデルの方が多様」という、GAN に対する本質的な優位を数値で裏づけています。

これらが示すのは、(1) ベイズ分解に基づく単純な勾配の足し込みで強力な条件付けが実現すること、(2) guidance scale が忠実さと多様性を連続的に交換する制御つまみとして機能すること、(3) この組み合わせ(guidance + 強化した U-Net)が画質と多様性の両面で GAN を上回り、生成モデルの主役を GAN から拡散モデルへ移す転換点になったことです。

## メリット・トレードオフ・限界

メリット
- 無条件で学習済みの拡散モデルを、後付けで条件付き生成に変えられる(本体の再学習不要)。
- guidance scale 1つで「条件への忠実さ」と「多様性」を自在に交換できる。実務上きわめて扱いやすい制御軸。
- 理論がベイズの定理で明快(条件付きスコア = 無条件スコア + 分類器勾配)。
- 分類器に限らず、任意の「望ましさ」を表す微分可能な関数(CLIP 類似度・コスト関数・報酬モデル・物理制約など)の勾配で誘導に応用できる汎用性。これがプランニングや逆問題への橋渡しになる。

トレードオフ・限界
- 拡散モデルとは別に、ノイズ入り画像でも働く時刻条件付き分類器を訓練・保守する必要がある。データとラベルのペアが要る。
- 分類器の精度・頑健性が生成の上限を縛る。分類器が間違えれば誘導も間違う方向へ向かう。ノイズ画像上の勾配は敵対的摂動のように脆く、誘導が不自然なテクスチャを生むことがある。
- guidance scale を上げすぎると多様性(Recall)が落ち、典型例ばかりになる。s が小さすぎると条件への忠実さが弱い。
- 各ステップで分類器の勾配(逆伝播)を計算するため推論コストが増える。
- 後続の Classifier-Free Guidance に比べ実装が煩雑で、現在の主流からは外れている。未解決の論点として、guidance による分布歪み(s>1 は厳密なベイズ事後分布ではなく p·p(y|x)^s への偏り、サンプルが本来の条件分布から系統的にずれる)の理論的扱いや、誘導と尤度・多様性の最適バランスは今も議論が続いています。

## 発展トピック・研究の最前線

- Classifier-Free Guidance(Ho & Salimans, 2021): 分類器を捨て、条件あり/なし予測の差で誘導する後継。実装簡潔・分類器不要で、テキスト画像生成の標準。同じ「分布をシャープにする」効果を分類器なしで得る。
- テキスト条件への拡張(GLIDE, Imagen, Stable Diffusion, DALL·E 2): CLIP やテキストエンコーダ(T5 等)の埋め込みを条件に、CFG の guidance scale でテキスト整合性を制御する実務が確立。guidance scale を上げると prompt 追従が強まるが過飽和・コントラスト過多になる、という CG と同型のトレードオフが現れる。
- 報酬/コスト誘導とプランニング: 分類器の代わりに「衝突しない」「目標に近い」などの微分可能なコスト関数の勾配で誘導する発想は、Diffusion Policy(Chi et al., 2023)や拡散ベースの軌道生成・プランニング(Diffuser, Janner et al., 2022)で実用化され、制約付き生成・制御の一般的な道具になっている。ロボティクスや自動運転の軌道生成では、この「学習済み拡散事前分布 + タスク特化の誘導」という分業が有効。
- 逆問題への誘導: 観測 y(ぼけ画像・低解像度・スパース計測)に対する尤度 ∇ log p(y|x) を guidance として足す diffusion posterior sampling(DPS, ΠGDM 等)。Classifier Guidance のベイズ分解を逆問題へ一般化したもので、超解像・画像補完・MRI/CT 再構成に応用される。
- 誘導の理論再検討: guidance scale を上げると分布が歪む問題に対し、自己誘導(self-guidance)、時刻依存・動的な guidance scale、品質と多様性を分離制御する手法、CFG の理論的正当化と過飽和の緩和(CFG++ 等)が研究されている。

## さらに学ぶための関連トピック
- [DDPM (Denoising Diffusion Probabilistic Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0229_ddpm_note)
- [DDIM (Denoising Diffusion Implicit Models)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0227_ddim_note)
- [Score-based Generative Models (SDE定式化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0250_score-based-sde)
- [NCSN (Noise Conditional Score Networks)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0243_ncsn_note)
- [Score Matching / Denoising Score Matching](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0251_score-matching)
