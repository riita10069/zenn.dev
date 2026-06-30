---
title: "AdaLN による条件付け（DiTパターン）"
---

# AdaLN による条件付け（DiTパターン）

## ひとことで言うと

AdaLN（Adaptive Layer Normalization、適応的レイヤ正規化）は、「時刻 t」や「クラスラベル」「自車状態」などの条件情報を、ニューラルネットの途中の特徴に注入するための技です。具体的には、正規化した特徴を、条件から作った二つの数 gamma（スケール）と beta（シフト）で「拡大・平行移動」します。拡散 Transformer（DiT, Diffusion Transformer）で標準手法になり、いまや Flow Matching や拡散ベースの生成モデルで条件を効かせる定番です。「条件は足すより、正規化後にチャネルごとのアフィン変調で注入する方が効く」という、生成モデル設計の一つの結論を体現しています。

## 直感的な理解

生成モデルは「何かの条件にしたがって生成」したいことがほとんどです。「この時刻ステップに合った速度を出して」「このクラス（犬・猫）の画像を作って」「この自車状態に合った軌道を出して」というように。Flow Matching の速度場ネットも、今が時刻 t のどこか、シーンや自車がどんな状態か、を知らないと正しい速度を出せません。この外部情報をネット内部にうまく流し込むことを「条件付け（conditioning）」と呼びます。

素朴な条件付けは「条件ベクトルを特徴に足す、あるいは連結（concat）する」やり方です。しかしこれだと条件の効き方が弱かったり、層ごとに均一に効かせにくかったりします。比喩でいえば、足し算は「全部の音に同じ音量のノイズを重ねる」ようなもので、各楽器（チャネル）を別々に強調したり抑えたりはできません。AdaLN は「各チャネルのつまみ（音量と位置）を、条件に応じて個別に回す」イメージです。これにより、条件は単なる味付けではなく、特徴の各次元をどれだけ強調しどれだけずらすか、という強い制御として効きます。

## 基礎: 前提となる概念

**正規化（normalization）と LayerNorm**。ニューラルネットの学習を安定させる定番の部品です。LayerNorm（レイヤ正規化）は、特徴ベクトルを「平均 0・分散 1」にならす操作です。通常はその後に学習パラメータ gamma・beta を掛けて足し戻します（`正規化 × gamma + beta`、これをアフィン変換と呼びます）。重要なのは、この gamma・beta が通常「全サンプル共通の固定パラメータ」だという点です。どんな入力でも同じ伸縮・シフトを掛けます。

**チャネル（channel）**。特徴ベクトルの各成分（次元）のことです。画像なら色や抽象的な特徴の種類、Transformer のトークンなら埋め込みの各次元に対応します。AdaLN は「チャネルごとに別々の gamma・beta」を持つので、特徴の次元ごとに異なる制御ができます。

**アフィン変調（affine modulation）**。`出力 = 入力 × scale + shift` という、掛け算と足し算による線形変換です。FiLM・AdaIN・AdaLN はすべてこのアフィン変調を、scale と shift を条件から動的に作る、という共通骨格を持ちます。違いは正規化の種類（インスタンス正規化かレイヤ正規化か）と適用箇所だけです。

**SiLU（Sigmoid Linear Unit）**。活性化関数の一つで、x · sigmoid(x)。Swish とも呼ばれ、滑らかで深いネットで安定しやすいため、条件処理の小さな MLP によく使われます。

## 仕組みを詳しく

AdaLN の発想は「LayerNorm の gamma・beta を固定パラメータにせず、条件から動的に計算する」ことです。つまりサンプルごと・時刻ごとに違う gamma・beta を条件から作り、正規化済み特徴を変調します。Flow Matching の速度場ネットを例に処理を追います。注入したい条件は「時刻 t」と「自車状態（視覚履歴＋自車運動）」だとします。シーンの空間特徴（BEV）は AdaLN には混ぜず、クロスアテンションで別経路から参照します（空間構造を潰さないため）。

1. **時刻を埋め込む**: スカラーの t `(B,)` を正弦波エンベディング（sin/cos を並べたベクトル化、Transformer の位置エンコーディングと同じ要領）にし、MLP で t_emb `(B, 256)` を作ります。

2. **自車条件を作る**: 視覚履歴を線形層で 256 次元に射影、自車運動も 256 次元に射影して足す → mod_cond `(B, 256)`。

3. **gamma, beta を生成**: `mod_cond + t_emb` を小さな MLP（SiLU → Linear で 2×256=512 次元出力）に通し、前半 256 を gamma、後半 256 を beta に分割します。
   ```python
   gamma, beta = self.adaln_modulation(mod_cond + t_emb).chunk(2, dim=-1)
   ```

4. **正規化 → 変調**: クロスアテンション出力 attended `(B,64,256)` を、アフィンなし LayerNorm（`elementwise_affine=False`、固定 gamma/beta を持たない）で正規化し、条件から作った gamma・beta で変調します。
   ```python
   normed = self.attn_norm(attended)
   modulated = normed * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
   ```
   `(1 + gamma)` としているのは「gamma=0 のとき変調なし（恒等写像）」を初期状態にし、学習開始時の振る舞いを安定させるためです（DiT の AdaLN-Zero に近い発想で、学習初期にブロックを恒等に近づけると深いネットが安定して立ち上がります）。`unsqueeze(1)` は 64 個のトークン全部に同じ gamma/beta を効かせるための形合わせ（ブロードキャスト）です。

数値イメージ。ある条件で gamma の 0 番チャネルが 0.3、beta が −0.1 なら、そのチャネルの正規化値 z は z·1.3 − 0.1 に変調されます。条件が変われば gamma/beta が変わり、同じ特徴でも違う変調を受けます。これが「条件に応じて各チャネルのつまみを回す」の実体です。

DiT の完全版（AdaLN-Zero）では、各 Transformer ブロックがさらに「残差スケール alpha」も条件から作り、アテンションや MLP の出力（残差）を `x + alpha · sublayer(modulate(norm(x)))` のように変調します。alpha を 0 で初期化すると学習初期に各ブロックが恒等写像になり、深いネットでも壊れずに立ち上がります。つまり一ブロックあたり 6 個（shift・scale・gate を attention と MLP の二系統に）の変調パラメータを条件から生成するのが典型です。

なぜ BEV を AdaLN に混ぜないのか。AdaLN は条件を一本のベクトルに集約してから gamma/beta を作るため、「地図のどこに何があるか」という空間的に細かい条件は表現できません。gamma/beta は全トークンに同じ値が掛かるので、空間ごとに違う条件を効かせられないのです。空間構造を保ちたい条件はクロスアテンションへ回し、潰してよい大域的な条件（時刻・自車運動）だけを AdaLN に回す、という役割分担が定石です。

## 手法の系譜と主要論文

AdaLN は突然生まれたのではなく、「条件由来のアフィン変調」という発想の十年がかりの蓄積の到達点です。

- **Dumoulin, Shlens, Kudlur, "A Learned Representation for Artistic Style (Conditional Instance Normalization)", ICLR 2017（arXiv:1610.07629）**。一つのネットで複数の画風を切り替えるため、正規化のスケール・バイアスをスタイルごとに持たせるという、条件依存のアフィン変調の初期形を提案しました。

- **Huang & Belongie, "Arbitrary Style Transfer with Adaptive Instance Normalization (AdaIN)", ICCV 2017（arXiv:1703.06868）**。任意のスタイル画像から、その場でチャネルごとの平均・分散を取り出してコンテンツ特徴に移植する AdaIN を提案しました。スタイルベクトルから動的にスケール・シフトを作る、という AdaLN の直系の祖先です。

- **Perez, Strub, de Vries, Dumoulin, Courville, "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018（arXiv:1709.07871）**。「条件からチャネルごとのスケール gamma とシフト beta を作って特徴を線形変調する」一般原理（Feature-wise Linear Modulation）を定式化しました。視覚質問応答で、条件を足す/concat するより効くと示し、AdaLN/AdaIN を包含する枠組みを与えました。

- **Karras, Laine, Aila, "A Style-Based Generator Architecture for GANs (StyleGAN)", CVPR 2019（arXiv:1812.04948）**。スタイルベクトルから各チャネルのスケール・バイアスを作り特徴を変調する AdaIN を画像生成の中核に据え、潜在空間での強い制御性と高品質を実現しました。

- **Peebles & Xie, "Scalable Diffusion Models with Transformers (DiT)", ICCV 2023（arXiv:2212.09748）**。AdaLN を拡散 Transformer の標準にした論文です。時刻ステップとクラスラベルから gamma・beta（さらに残差スケール alpha）を作って各 Transformer ブロックを変調する「AdaLN-Zero」（学習初期に残差を 0 にして恒等から始める版）を提案し、in-context 条件付けや cross-attention 条件付けより性能とスケーラビリティで優れると示しました。Flow Matching の速度場や軌道生成での AdaLN 利用は、この DiT パターンの直接の流用です。

- **Esser et al., "Stable Diffusion 3 (MM-DiT)", ICML 2024（arXiv:2403.03206）**。大規模テキスト画像生成で、AdaLN による時刻条件付けと、テキスト条件のための双方向アテンションを組み合わせた MM-DiT を提案しました。AdaLN が大規模生成 Transformer の標準部品として今も中核にあることを示す代表例です。

これらに共通する結論は「条件は足すより、正規化後にチャネルごとのアフィン変調で注入する方が効く」です。

## 論文の実験結果（定量データ）

DiT（Peebles & Xie, ICCV 2023）の条件付け方式アブレーションが最も重要な定量証拠です。ImageNet 256×256 のクラス条件付き画像生成で、評価指標は FID（生成分布と本物分布の距離、小さいほど良い）です。論文は同じ DiT バックボーンに対し、条件の入れ方を (a) in-context（条件をトークンとして系列に足す）、(b) クロスアテンション、(c) AdaLN、(d) AdaLN-Zero、と変えて比較しました。報告によれば FID は AdaLN-Zero が最良で、in-context やクロスアテンションより明確に小さい FID を達成しました。具体的には、最大規模の DiT-XL/2 で ImageNet 256×256 の FID 2.27 という当時の最先端を記録し、その性能を支えたのが AdaLN-Zero 条件付けでした。アブレーションのもう一つの示唆は「AdaLN-Zero の zero 初期化（残差を 0 から始める）」が効くことで、これを外して普通の AdaLN にすると収束が遅く最終 FID も悪化した、と報告されています。計算コストの面でも、AdaLN は条件を gamma/beta に圧縮して注入するためクロスアテンションより演算量（GFLOPs）が小さく、品質と効率を両立しました。ただし AdaLN はパラメータ数を増やす（各ブロックに変調 MLP が付く）ため、DiT のパラメータの少なくない割合が変調生成に使われる、という指摘もあります。

FiLM（Perez et al., AAAI 2018）は視覚質問応答ベンチマーク CLEVR で評価し、評価指標は **正答率** です。条件（質問）を FiLM で各層に変調注入する方式が、当時の手法を上回り 97.7% の正答率を達成したと報告しました。アブレーションとして、FiLM 層を抜く（条件を concat だけにする）と正答率が大きく下がり、チャネルごとのアフィン変調が推論能力に効いていることが示されました。これは「足す/concat より変調」という結論の最も古い定量的根拠の一つです。

StyleGAN（Karras et al., CVPR 2019）は顔画像生成 FFHQ などで、AdaIN ベースのスタイル注入により FID を大きく改善し（FFHQ で FID 4 台前半を当時記録）、かつ潜在空間での属性の分離（disentanglement）が進んだと報告しました。アフィン変調による条件注入が品質と制御性の双方に効く、という流れを決定づけた研究です。

## メリット・トレードオフ・限界

メリット。条件（時刻・状態・クラス）を特徴のチャネルごとに強く柔軟に効かせられ、足し算より表現力が高いです。`(1 + gamma)` の初期恒等（AdaLN-Zero）で学習が安定し、モデルを大きくしても性能が素直に伸びます。多数のトークン全体に一括で条件を効かせられ、形合わせ（unsqueeze）だけで済みます。条件を gamma/beta に圧縮して注入するため、クロスアテンション条件付けより演算量が小さいです。

トレードオフと限界。gamma/beta（および alpha）を作る MLP のぶんパラメータが増え、これが大規模モデルでは無視できない割合になります（ただしクロスアテンションより演算量自体は軽い）。条件を一本のベクトルに集約してから変調するため、空間的に細かい条件（地図上の位置依存、テキストのトークンごとの違いなど）はこの経路では表しにくく、そうした条件はクロスアテンションに回す必要があります。実装上は gamma/beta の分割やブロードキャスト（unsqueeze）の形を間違えやすく、無言のバグになりがちです。研究上の課題として、複数の条件（時刻・テキスト・空間特徴）をどう役割分担して注入するのが最適か、AdaLN とクロスアテンションのハイブリッドの設計は依然として経験則に頼る部分が残っています。

## 発展トピック・研究の最前線

最前線では、AdaLN を超える/補う条件付けの研究が続いています。一つは「マルチモーダル条件の統合」で、時刻・テキスト・画像・空間特徴を、AdaLN とクロスアテンションを組み合わせて注入する大規模生成モデル（Stable Diffusion 3 の MM-DiT、動画生成など）の設計です。二つ目は「効率化」で、gamma/beta 生成 MLP を層間で共有したり低ランク化したりしてパラメータを節約する工夫（変調パラメータが全パラメータの大きな割合を占める問題への対処）です。三つ目は「制御性の強化」で、ガイダンス（classifier-free guidance）と AdaLN を組み合わせ、生成時に条件の効き具合を調整する手法です。AdaLN-Zero は DiT 以降、動画拡散・3D 生成・ロボットアクション生成・自動運転軌道生成と幅広く転用され、生成 Transformer の事実上の標準部品になっています。

## さらに学ぶための関連トピック

- [Flow Matching 軌道プランナ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0123_flow-matching-trajectory-planner)
- [Flow Matching（フローマッチング）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0121_flow-matching)
- [条件付き最適輸送（OT）パス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0215_conditional-optimal-transport-path)
