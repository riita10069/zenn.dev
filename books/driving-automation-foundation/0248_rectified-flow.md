---
title: "Rectified Flow"
---

# Rectified Flow

## ひとことで言うと
Rectified Flow（整流フロー）は、ノイズとデータを「できるだけまっすぐな直線」で結ぶ輸送をベクトル場として学習する生成手法です。直線にこだわるのは、生成時に少ない積分ステップ（極端には1ステップ）でゴールに着けるようにするためです。[Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) とほぼ同じ家族に属し、「reflow（リフロー）」という反復手順で生成パスをさらにまっすぐ整える点が独自の貢献です。Stable Diffusion 3 が学習目標に採用したことで、研究にとどまらず実用の主流になりました。

## 直感的な理解
生成は「ノイズからデータへ向かう道を、数値積分で一歩ずつたどる」作業です。道が曲がりくねっていると、粗く刻んで進めば真の道から外れて品質が落ちます。だから細かく刻む（=多ステップ）必要がありました。逆に、道が完全な直線なら、刻みが1回でも正確にゴールに着けます。「道をまっすぐにすれば、少ないステップで生成できる」——これが Rectified Flow の出発点です。

身近な例えで言えば、目的地まで車で行くとき、道が真っすぐなら一気にアクセルを踏んで到着できますが、くねくね山道だと何度もハンドルを切り直さなければなりません。生成における「ハンドルを切り直す回数」がネットワーク呼び出し回数（NFE）で、これがそのまま計算コストとレイテンシになります。道そのものをまっすぐに整えてしまえば、切り直しが要らなくなる、というのが整流（rectify）の発想です。

ここで重要なのは、「直線で結ぼうと意図して学習しても、結果の生成パスは曲がってしまう」という落とし穴があることです。Rectified Flow の本当の貢献は、その曲がりを反復手順で系統的にほどく方法（reflow）を見つけたことにあります。

## 基礎: 前提となる概念

ベクトル場と ODE 生成: ノイズ x_0 からデータ x_1 へ点を運ぶには、各点・各時刻での速度を表すベクトル場 v(x, t) を用意し、常微分方程式 dx/dt = v(x, t) を t=0 から t=1 まで積分します。詳しくは [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) を参照。Rectified Flow も flow matching も、この「ベクトル場を学んで ODE で積分する」骨格は同じです。

カップリング（coupling、対応づけ）: ノイズ側の点 x_0 とデータ側の点 x_1 をどうペアにするか、を決める同時分布のこと。素朴には独立にランダムに引いてペアにします（独立カップリング）。誰と誰を結ぶかで引く直線の集合が変わり、それが生成パスの曲がりやすさを左右します。

生成パスの曲がり（straightness）: 学習したベクトル場でノイズから生成したとき、点が実際にたどる軌跡が直線からどれだけ外れているかを「straightness（まっすぐさ）」という量で測ります。定義は S = E∫₀¹ |(x_1 - x_0) - v(x_t, t)|² dt のような形で、各時刻の速度が「終点-始点」一定ベクトルからどれだけずれるかの積算です。S=0 なら完全な直線で、どんなに刻みを粗くしても積分誤差はゼロ。だから straightness を上げる（パスを直線化する）ことが少ステップ化と直結します。

蒸留（distillation）: 学習済みの「教師モデル」が時間をかけて出す結果を、「生徒モデル」が一発で真似るよう学習させる手法。多ステップの教師を1ステップの生徒に圧縮するのに使われます。Rectified Flow の超少ステップ化の最終段で組み合わされます。

## 仕組みを詳しく

ステップ1: 基本は直線パスのベクトル場回帰。出発点は [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching) と同じです。ノイズ x_0 とデータ x_1 を直線で結び、その上の点 x_t における一定速度 x_1 - x_0 をネットワーク v_θ に二乗誤差で学習させます。

```
x_t = (1 - t) * x_0 + t * x_1
損失 = E_{t, x_0, x_1} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Rectified Flow はこれを「Flow Straight（まっすぐな流れ）」を目指す思想として強調した呼び名です。1回目に学習したベクトル場を v¹ と呼びます。

ステップ2: 問題——直線同士が交差する。ノイズとデータを独立にランダム対応づけ（独立カップリング）して直線を引くと、複数の直線が空間内で交差します。砂粒 A が左上から右下へ、砂粒 B が左下から右上へ向かうと、真ん中で進む向きがぶつかります。ベクトル場は1つの点に1つの向きしか持てないので、交差点では両者を平均した速度になり、結果として実際の生成パスは曲がってしまいます。これでは「直線で結んだ」つもりでも少ステップでは済みません。

```
理想（交差なし）        現実（独立対応で交差あり）
 →→→→→               ↘     ↗
 →→→→→                ↘   ↗     ← 交差点で速度が平均化され
 →→→→→                 ↘ ↗        生成パスが曲がる
```

ステップ3: 解決——reflow で整流する。Rectified Flow の独創的貢献が reflow という反復手順です。

1. まず素朴な直線パス（独立カップリング）でベクトル場 v¹ を学習する。
2. 学習した v¹ を使ってノイズ x_0 から ODE 生成し、「ノイズ x_0 → 生成結果 z_1 = ODE(x_0; v¹)」というペアを大量に作る。
3. この新しいペア (x_0, z_1) を改めて直線で結び直し、ベクトル場 v² を学習する。

ここがポイントです。v¹ で生成したペア (x_0, z_1) は「もともと交差していた対応づけを、v¹ がほどいた後の決定論的な対応づけ」になっています。これを再び直線で結ぶと、交差が減って straightness が上がります。reflow を 1 回、2 回と繰り返すほどパスは直線に近づきます。理論的にも論文は2つの性質を証明しています。1つは「reflow は出力の周辺分布を変えない」（z_1 の分布はデータ分布のまま）こと、もう1つは「reflow を1回行うと輸送コスト（凸コストの総和）が減り、繰り返すほど直線に単調に近づく」ことです。

ステップ4: 蒸留で1ステップ化。十分整流された後は蒸留を組み合わせます。教師である k-rectified flow の ODE 解（時間をかけてたどった終点）を、生徒が1ステップで直接予測するよう学習させます。具体的には x̂_1 = x_0 + v_θ(x_0, 0) のように初速だけで終点を当てる形にし、教師の ODE 出力との差を最小化します。パスがすでにほぼ直線なので、1ステップ予測でも誤差が小さく、品質を保ったまま NFE=1（ネットワーク1回呼び出し）の生成が実現します。

```
素朴な flow:     ノイズ ──曲がった道を数十ステップ──> データ
reflow 後:       ノイズ ──ほぼ直線を数ステップ──> データ
reflow + 蒸留:   ノイズ ──直線を1ステップで一気に──> データ
```

## 手法の系譜と主要論文

- Liu, Gong, Liu (2023, ICLR)「Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow」(arXiv:2209.03003): 本トピックの中心。直線輸送のベクトル場回帰と、reflow による反復整流を提案。動機は「生成パスを直線化できれば数値積分の刻みを極限まで減らせ、少ステップ・1ステップ生成が可能になる」こと。flow matching（Lipman ら）とほぼ同時期に独立に直線補間の考えに到達した姉妹研究です。画像生成だけでなくドメイン変換（画像→画像）にも使える一般枠組みとして提示されました。
- Lipman ら (2023, ICLR)「Flow Matching for Generative Modeling」(arXiv:2210.02747): 姉妹的研究。直線パス=OT 的パスを学ぶ点で数学的に近く、両者を合わせて flow matching 系と総称されます。詳細は [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching)。
- Song, Dhariwal, Chen, Sutskever (2023, ICML)「Consistency Models」(arXiv:2303.01469): 別系統の少ステップ手法。ODE 軌道上のどの点からでも同じ終点を予測するよう学習し、1〜数ステップ生成を狙う。Rectified Flow と問題意識（少ステップ化）を共有し、しばしば比較対象になります。
- Liu, Zhang, Ma ら (2024, ICLR)「InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation」(arXiv:2309.06380): Rectified Flow + reflow + 蒸留で Stable Diffusion を1ステップ化した実証。動機はテキスト→画像生成を実用的な速度に乗せること。reflow を経てから蒸留することで、蒸留だけ（reflow なし）より大幅に品質が上がることを示しました。
- Esser ら (2024, ICML)「Scaling Rectified Flow Transformers for High-Resolution Image Synthesis」(arXiv:2403.03206, Stable Diffusion 3): Rectified Flow を学習目標として大規模 Transformer（MM-DiT、テキストと画像で別々の重みを持つマルチモーダル DiT）に採用。学習時刻のサンプリング分布（logit-normal で中間時刻を厚く）が品質に効くことを大規模に検証し、Rectified Flow が実用の主流になる決定打となりました。

## 論文の実験結果(定量データ)

指標の意味: FID（Fréchet Inception Distance）は生成画像と本物の分布の近さで、小さいほど良い。NFE（Number of Function Evaluations）は1サンプル生成に必要なネットワーク呼び出し回数で、小さいほど速い。recall は生成の多様性（本物の分布をどれだけ広くカバーするか）の指標で、大きいほど多様。straightness はパスの直線度で、小さいほど直線（少ステップに強い）。

- Liu ら (2023): CIFAR-10 で、reflow を経た Rectified Flow は素朴な1回学習の flow（1-rectified）よりも straightness が大きく改善し、NFE を1桁まで減らしても FID の劣化が緩やかになると報告。アブレーションの核心は「reflow を外す（1-rectified のまま少ステップ生成する）と、同じ NFE での FID が悪化する」一方で「2-rectified, 3-rectified と reflow 回数を増やすと straightness が単調に下がり、少 NFE での FID が改善する」ことを示した点です。1ステップ（NFE=1）の蒸留でも、reflow なしの直接蒸留より良い FID を達成し、整流が少ステップ品質の鍵だと裏づけました。
- InstaFlow (2024): Stable Diffusion 由来のモデルで、reflow + 蒸留により NFE=1 で MS-COCO の FID がおよそ 23 前後（報告による）に到達し、これは従来の1ステップ蒸留手法を上回る品質でした。要点は「reflow で道をまっすぐにしてから蒸留すると、蒸留単独よりはるかに少ステップ品質が良くなる」という定量的証拠です。多数ステップの教師（Stable Diffusion、数十ステップ）に迫る品質を、1回のネットワーク呼び出しで出せることを示しました。学習コストの面でも、必要な計算量を A100 GPU 日数で具体的に報告し、reflow の追加コストが許容範囲であることを示しています。
- Stable Diffusion 3 (2024): Rectified Flow + logit-normal 時刻サンプリングが、拡散の損失重み（EDM など）や一様時刻サンプリング、他の補間（拡散型・線形型）より一貫して良い FID / CLIP スコアを与え、モデル・データをスケールするほど優位が安定すると大規模に報告。複数の学習目標・時刻分布を同条件で比較する徹底したアブレーションを行い、Rectified Flow 系 + 中間時刻を厚くする時刻分布が総合的に最良と結論づけました。

これらの数値が示すのは「直線化（reflow）と蒸留を組み合わせると、生成ステップを極限まで減らしても品質を保てる」こと、そして「直線化を経るかどうかが少ステップ品質を分ける」ことです。

## メリット・トレードオフ・限界

メリット
- 生成パスがまっすぐになり、少ステップ（極端には1ステップ）で高品質生成ができ、レイテンシを大きく削れる。
- 学習目標自体は flow matching と同じ単純な二乗誤差で安定している。
- reflow は学習済みモデルの上に重ねる手順なので、既存の flow をそのまま強化できる。出力の周辺分布を変えない理論保証があるため、reflow してもデータ分布から外れない。

トレードオフ・限界
- reflow は「学習→生成→再学習」を反復するため学習コスト・時間がかさむ。各段階で生成サンプルを大量に作る必要がある（合成ペアの生成自体が ODE 積分でコストになる）。
- reflow の累積誤差: v¹ の生成は完全でないため、生成ペア z_1 に v¹ の誤差が混入し、それが v² の学習データの質を下げる。reflow を重ねるほど直線になる一方で、この誤差蓄積と品質低下のせめぎ合いがあり、闇雲に回数を増やせば良いわけではない。
- 1ステップ化には蒸留が必要で、生徒の最終品質は教師モデルの品質に上限を縛られる。
- 直線化を追求するほど、出力のばらつき（多様性、recall）が下がる場合があり、マルチモーダル性とのバランス調整が要る。
- 未解決の課題として、条件付き生成での整流の理論、超高解像度・長系列での直線化の効き、reflow を1段に圧縮する効率化などが研究対象です。

## 発展トピック・研究の最前線
- Consistency Models（Song ら, 2023）や Consistency Trajectory Models など、別系統の少ステップ手法との比較・統合。これらも「ODE の任意の点から終点を一発予測する」発想で Rectified Flow と問題意識を共有します。
- reflow を1段に圧縮する効率化や、蒸留と reflow を同時に行う手法、敵対的損失（GAN 的な判別器）を蒸留段に足して1ステップ品質をさらに上げる手法。
- 動画・3D・行動生成への展開。長い系列や高次元連続行動で直線化の効きを検証する研究。
- 自動運転・ロボットの軌道生成での応用。flow matching ベースの軌道デコーダをさらに少ステップ化して車載のリアルタイム制約（数十ミリ秒で軌道を出す）に合わせる方向で、reflow は自然な発展経路になります。マルチモーダルな未来を保ちつつ少ステップ化するため、多様性低下とのトレードオフ調整が実務上の論点です。

## さらに学ぶための関連トピック
- [Conditional Flow Matching (CFM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0214_conditional-flow-matching)
- [DDPM（拡散モデル基礎）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0230_ddpm-denoising-diffusion)
- [DDIM 高速サンプリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0228_ddim-sampling)
- [Mixture Density Network 軌道予測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0189_mixture-density-network-planner)
