---
title: "DDIM 高速サンプリング"
---

# DDIM 高速サンプリング

## ひとことで言うと
DDIM（Denoising Diffusion Implicit Models）は、[DDPM（拡散モデル基礎）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0230_ddpm-denoising-diffusion) が生成に数百〜千回もネットワークを呼ぶ遅さを解決する手法です。逆過程の各ステップから「ランダムさ」を抜いて決定論的（同じ入力なら必ず同じ出力）にすることで、生成ステップを 1000 回から 20〜50 回程度へ大幅に減らしても、ほぼ同じ品質の出力が得られます。学習はやり直さず、DDPM で学習済みのノイズ予測ネットワークをそのまま流用できるのが最大の実用的価値です。

## 直感的な理解
DDPM の生成は「砂嵐から写真へ、ほんの少しずつノイズを剥がす操作を1000回繰り返す」作業です。各ステップが小さなランダムジャンプ（毎回ノイズ z を足す）なので、刻みを大きくすると軌跡がガタガタに崩れます。だから細かく刻む（=1000回）必要がありました。

DDIM の発想は「ランダムジャンプをやめて、なめらかな決まった道（決定論的な軌道）に沿って進めば、刻みを荒くしても道が崩れない」というものです。道がなめらかなら、1000 個の細かい点を打たなくても、20〜50 個の点で十分その道を近似できます。各点でネットワークを1回呼ぶので、点の数を減らすことがそのまま生成の高速化になります。しかも重要なのは、この変更が「生成手順だけの工夫」で、DDPM の学習済みネットワークをそのまま使える点です。新しく学習し直す必要がありません。

自動運転では、車載コンピュータが「次の数秒の軌道」を毎秒何回も計算し直します。1回の生成に1000回の推論がかかってはリアルタイム制御に間に合いません。「品質を保ったままステップ数を減らす」ことが切実に必要で、DDIM がその最初の有力な答えでした。

## 基礎: 前提となる概念

DDPM の逆過程の復習: 拡散モデルは前向きに x_0 へノイズを足して x_T（純粋ノイズ）にし、逆向きにネットワーク ε_θ(x_t, t)（混ざっているノイズの予測）を使ってノイズを剥がします。前向きの一発式 x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε と、そこから得られるデータ推定 x̂_0 = (x_t - sqrt(1-ᾱ_t) ε_θ)/sqrt(ᾱ_t) が以下で繰り返し使われます。詳細は [DDPM（拡散モデル基礎）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0230_ddpm-denoising-diffusion)。

マルコフ過程（Markov process）: 「次の状態は直前の状態だけで決まる」確率過程。DDPM の逆過程は各ステップで小さなランダムノイズを足すマルコフ過程として定義されており、この「毎ステップのランダム性」が刻みを荒くできない理由でした。

決定論的 vs 確率的サンプリング: 確率的（stochastic）は毎回ランダム性が入り、同じ初期ノイズでも出力が変わる。決定論的（deterministic）はランダム性がなく、初期ノイズが決まれば出力が一意に決まる。DDIM は後者を可能にします。

常微分方程式（ODE, ordinary differential equation）: ランダム性のない、決まった軌道を描く方程式 dx/dt = f(x, t)。これに対し確率微分方程式（SDE）はランダムな揺らぎ項を含みます。DDIM の決定論的サンプリングは、拡散に対応する probability flow ODE（reverse SDE と同じ周辺分布を持つ決定論的 ODE）を粗く数値積分することと数学的に等価です。

## 仕組みを詳しく

DDIM の鍵となる発見は「学習し直す必要はない。DDPM で学習済みの ε_θ をそのまま使い、生成手順だけを賢く変えればよい」という点です。これが成り立つ理由は次の洞察にあります。

ポイント1: 非マルコフ過程に作り替える。DDPM の損失 L_simple は、実は前向き過程の各時刻の周辺分布 q(x_t | x_0) にしか依存しません。つまり「途中の道筋（同時分布）」を変えても、各時刻の周辺分布さえ同じなら、学習済みネットワークはそのまま正しいノイズ予測を与えます。DDIM はこの自由度を使い、逆過程を「非マルコフ的（non-Markovian、x_{t-1} が x_t だけでなく x_0 推定にも依存する）」に再定義します。要は「各ステップでランダムノイズを足すのをやめ、現在の状態 x_t から最終的なきれいなデータ x_0 を直接推定し、そこへなめらかに向かう」やり方です。同じ周辺分布を保つので、同じ学習済みネットワークが使えます。

ポイント2: ランダム性をパラメータ σ_t で制御する。DDIM は1ステップ更新に、ランダム性の強さを表すパラメータ σ_t（シグマ）を導入します。

- σ_t を大きく（DDPM 相当の値に）取ると DDPM と同じ確率的サンプリングに戻る。
- σ_t = 0 にすると完全に決定論的になる。これが DDIM の代表的な使い方です。

σ_t = 0 のときの1ステップ更新は、概念的に次の3段階です。

1. 現在の状態 x_t と予測ノイズ ε_θ(x_t, t) から、最終的なきれいなデータの推定 x̂_0 を求める。前向き式 x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε を逆に解いて、x̂_0 = (x_t - sqrt(1-ᾱ_t) ε_θ) / sqrt(ᾱ_t)。
2. その x̂_0 を使い、より少ないノイズしか残らない時刻 t'（< t）の状態 x_{t'} を直接構成する。x_{t'} = sqrt(ᾱ_{t'}) x̂_0 + sqrt(1-ᾱ_{t'}) ε_θ（同じ予測ノイズの向きを「指す方向」として再利用）。
3. これを t = T から t = 0 へ、好きな粗さの時刻列（例: 1000, 980, 960, … と 50 個だけ、あるいは等間隔に間引いた部分列）でたどる。

数式で書くと σ_t=0 の1ステップは

```
x_{t'} = sqrt(ᾱ_{t'}) · ( (x_t - sqrt(1-ᾱ_t)·ε_θ(x_t,t)) / sqrt(ᾱ_t) ) + sqrt(1-ᾱ_{t'}) · ε_θ(x_t,t)
```

ここが速さの源です。逆過程がランダムジャンプでなく「なめらかな ODE に沿った移動」になるため、刻みを荒くしても軌道が大きく崩れません。1000 を 50、20 へ減らしても品質低下が緩やかです。

```
DDPM:  ノイズ ──小さなランダムジャンプ×1000──> データ
DDIM:  ノイズ ──なめらかな ODE に沿った移動×20〜50──> データ（同じ入力→同じ出力）
```

ポイント3: 潜在変数としての意味づけ。DDIM は決定論的なので「初期ノイズ x_T」と「生成結果 x_0」が1対1で対応します。これにより x_T が意味のある潜在表現になり、2つのノイズの球面線形補間（slerp）でなめらかに変化する出力列を作る、生成画像を順過程の ODE で x_T に逆写像（DDIM inversion）して編集する、といった操作が可能になります。これは確率的サンプリングでは得られない性質です。

ポイント4: ODE 視点での位置づけ。Song らの SDE 論文（2011.13456）は、DDIM の決定論的更新が拡散に対応する probability flow ODE を1次のオイラー法的に離散化したものと等価であることを示しました。この見方は「刻みを減らす=ODE を粗く解く」と捉えるもので、より高次の数値解法（DPM-Solver）や、最初からまっすぐな ODE を学ぶ [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) / [Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow) へと自然につながります。

## 手法の系譜と主要論文

- Ho, Jain, Abbeel (2020, NeurIPS)「DDPM」(arXiv:2006.11239): DDIM が高速化の対象とした元の手法。逆過程をマルコフ過程として定義し、T=1000 の逐次サンプリングを要しました。DDIM はこの学習済みネットワークを丸ごと流用できます。
- Song, Meng, Ermon (2021, ICLR)「Denoising Diffusion Implicit Models」(arXiv:2010.02502): 本トピックの中心。DDPM の前向き過程を「同じ周辺分布を持つ非マルコフ過程の一族」へ一般化し、その逆過程を σ_t で連続的に制御できることを示しました。動機は「学習はそのままに、サンプリングの数学的定義を緩めて高速化する」こと。σ_t=0 の決定論的版が DDIM です。
- Song ら (2021, ICLR)「Score-Based Generative Modeling through SDEs」(arXiv:2011.13456): DDIM の決定論的サンプリングが probability flow ODE の離散化と等価であることを明確化。拡散を ODE として解くという見方を確立し、[Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) / [Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow) へつながりました。
- Lu, Zhou, Bao ら (2022, NeurIPS)「DPM-Solver」(arXiv:2206.00927): probability flow ODE を、拡散の半線形構造を利用した高次の指数積分子（exponential integrator）で解き、10 ステップ前後でも高品質を達成。DDIM のさらなる高速化系として広く使われます。
- Lu ら (2022)「DPM-Solver++」(arXiv:2211.01095): guidance を強くかけた条件付き生成での安定性を改善し、少ステップでもサンプルが破綻しないようにした改良版。テキスト→画像の標準サンプラの一つになりました。

## 論文の実験結果(定量データ)

指標の意味: FID（小さいほど本物に近い）、NFE（ネットワーク呼び出し回数、小さいほど速い。サンプリング時間にほぼ比例）。

- Song ら (2021, DDIM): CIFAR-10 と CelebA（顔画像）で、同じ学習済み DDPM モデルに対しサンプリング手順だけを変えて評価。DDPM が 1000 ステップで出す品質に対し、DDIM は 50 ステップ（20倍速）でほぼ同等の FID を達成し、20 ステップ（50倍速）でも実用的な品質を保ちました。アブレーションの核心は「ステップ数を 1000 → 100 → 50 → 20 → 10 と減らしたときの FID の劣化曲線」で、確率的な DDPM サンプリングは少ステップで急激に FID が悪化するのに対し、決定論的 DDIM は緩やかにしか悪化しないことを示しました。例えば CIFAR-10 で DDIM は 100 ステップで FID 4 台、50 ステップでも 5 前後（報告による）と、1000 ステップの DDPM に肉薄しました。さらに σ を中間値にして DDPM と DDIM を内挿する実験も行い、σ を上げる（確率性を強める）ほど少ステップでの劣化が大きくなることを定量的に示しています。
- 決定論性の検証: 同じ初期ノイズ x_T から異なるステップ数で生成しても、似た画像が得られる（ステップ数によらず x_T が出力をほぼ決める）ことを示し、x_T が潜在表現として機能する証拠としました。これは確率的サンプリングでは得られない性質です。
- DPM-Solver (2022): probability flow ODE を高次解法で解くことで、DDIM がおよそ 50 ステップ要した品質を 10〜15 ステップで達成。少 NFE 領域で DDIM を明確に上回り、拡散サンプリング高速化の標準的選択肢になりました。DPM-Solver++ では guidance を強めた条件付き生成でも少ステップで破綻しないことを示しています。

これらの数値が示すのは「決定論的にして ODE として滑らかに解けば、刻みを 1/20〜1/50 に減らしても品質をほぼ保てる」こと、そして「その高速化が学習し直しなしで既存モデルに後付けできる」ことです。

## メリット・トレードオフ・限界

メリット
- 学習し直さずに DDPM を高速化できる（既存の学習済みネットワークをそのまま流用）。最大の実用的価値。
- 決定論的なので、同じ初期ノイズから必ず同じ出力が得られ、評価の再現性が高い。デバッグや A/B 比較がしやすい。
- 生成ステップを 1000 から 20〜50 へ減らせ、リアルタイム用途に一歩近づく。
- 潜在補間・DDIM inversion による画像編集など、確率的サンプリングでは難しかった操作ができる。

トレードオフ・限界
- ステップ数を極端に減らす（数ステップ）と品質が落ちる。さらなる高速化には別手法（DPM-Solver、蒸留、最初から ODE を学ぶ flow 系）が必要。
- 決定論的にすると出力の多様性が下がる。マルチモーダルな未来を多数サンプリングしたい場合はランダム性（σ_t > 0）を残す調整が要り、その分速さの利点が薄れる。多様性と少ステップ品質はトレードオフ。
- 「拡散モデルを後から ODE で解く」設計のため、最初からまっすぐな ODE を学習する [Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow) / [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) ほどには超少ステップ（1〜数ステップ）生成に強くない。1次のオイラー離散化なので、刻みを粗くすれば積分誤差が残る。
- 未解決の課題として、少ステップでの離散化誤差の理論的解析や、決定論性と多様性を両立させるサンプラ設計があります。

## 発展トピック・研究の最前線
- DPM-Solver / DPM-Solver++ / UniPC など、ODE を高次解法で解く高速サンプラ群。10 ステップ前後を狙い、guidance 下での安定性を高めています。
- 蒸留による超少ステップ化（progressive distillation, consistency models）。多ステップの教師を1〜数ステップの生徒に圧縮し、DDIM のさらに先（NFE=1〜4）を目指します。
- 「最初からまっすぐな ODE を学ぶ」発想の [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) と [Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow)。DDIM が後付けで得た ODE 的高速化を、学習目標として内包する後継です。DDIM はこの「ステップ削減」という大きな流れの最初の一歩でした。
- 自動運転・ロボットの軌道/行動生成で、リアルタイム制約に合わせてサンプリングステップを削る実務的な選択（決定論的少ステップ生成）の基礎になっています。決定論性は安全評価の再現性という観点でも好まれます。

## さらに学ぶための関連トピック
- [DDPM（拡散モデル基礎）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0230_ddpm-denoising-diffusion)
- [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching)
- [Rectified Flow](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0248_rectified-flow)
- [Mixture Density Network 軌道予測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0189_mixture-density-network-planner)
