---
title: "GAN ベース軌道予測（敵対的学習で社会的に妥当な多峰軌道を生成する）"
---

# GAN ベース軌道予測（敵対的学習で社会的に妥当な多峰軌道を生成する）

## ひとことで言うと
「軌道を生成する側」と「その軌道が本物の人間の動きか偽物かを見破る側」の2つのネットワークを競わせる（敵対させる）ことで、人間らしく、しかも複数のありうるパターンを持つ未来軌道を作る手法です。GAN（Generative Adversarial Network、敵対的生成ネットワーク）を軌道予測に応用したもので、Social GAN が代表例です。「何が人間らしいか」を人間が数式で書くのではなく、判別器にデータから学ばせるのが核心です。

## 直感的な理解
偽札づくりと鑑定officer の追いかけっこを想像してください。偽造者（生成器）は本物そっくりの札を作ろうとし、鑑定officer（判別器）はそれを見破ろうとする。officer が見破るたびに偽造者は腕を上げ、最後には officer でも区別できないほど精巧な札ができあがる。GAN はこの構図をニューラルネットで再現したものです。

なぜこれが軌道予測に向くのか。歩行者の動きの予測には、軌道予測一般の難しさ（未来が一通りに決まらない多峰性）に加えて、もう一つ特有の難しさがあります。それは「社会的に妥当でなければならない」ことです。人は群衆の中で、無意識のうちに他人とぶつからないよう経路を選び、譲り合い、グループで一緒に動きます。こうした暗黙のルールを社会的相互作用（social interaction）と呼びます。良い予測は「人間が見て自然と感じる動き」を再現しなければならない。物理的に可能でも、人混みの真ん中を突っ切る予測は不自然です。

ところが「人間らしさ」を二乗誤差や明示的なルールで書くのは至難です。条件付き VAE（[CVAE 軌道生成（条件付き変分オートエンコーダによる多峰な未来の生成）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0165_cvae-trajectory-generation)）は確率分布を明示的に学んで多峰性を出しますが、ELBO 最適化ゆえ出力がややぼやけがちでした。そこで「人間らしさを数式で定義する代わりに、判別器に学ばせよう」という発想が出てきます。判別器が「これは群衆の中で自然な動きか？」を採点してくれるなら、生成器はその採点を上げるように学べばよい。これが GAN ベース軌道予測の動機です。

## 基礎: 前提となる概念

### GAN の基本: 生成器と判別器の競争
GAN（Goodfellow et al., NeurIPS 2014）は2つのネットからなります。
- 生成器（Generator, G）: ノイズ z（と条件）から偽物のデータ（ここでは未来軌道）を作る。
- 判別器（Discriminator, D）: 与えられた軌道が「本物（実データ）」か「G が作った偽物」かを当て、本物である確率を出力する。

学習はミニマックスゲームとして定式化されます。D は「本物に高い確率、偽物に低い確率」を付けるよう学習し、G は「D が本物と誤認するような軌道」を作るよう学習します。両者が拮抗していくと、理論的には G が生成する分布が真のデータ分布に一致します。ポイントは、何が「本物らしい」かを人間が数式で書く必要がなく、D がデータから自動で学ぶ点です。社会的妥当性のような言語化しにくい性質を扱うのに向いています。

### RNN / LSTM（時系列を読む道具）
軌道は時間に沿った点列なので、過去の動きを読むのに RNN（Recurrent Neural Network、再帰型ネット）、特に LSTM（Long Short-Term Memory）が使われます。LSTM は「どの過去情報を覚え、どれを忘れるか」をゲートで制御し、長い系列の文脈を保持できる RNN の改良版です。歩行者1人の過去軌道を1つの状態ベクトルに要約するのに使います。

## 仕組みを詳しく

### Social GAN の構成
Social GAN（略して SGAN）は、歩行者群の予測のために以下を組み合わせました。

1. エンコーダ（LSTM）: 各歩行者の過去軌道を時系列で読み、状態ベクトルにする。
2. プーリングモジュール（Pooling Module, PM）: シーン内の全歩行者の状態を集約する。ここが社会性の肝。先行手法の Social LSTM は近傍のグリッド内の人だけを集約したが、SGAN は全員の相対位置を埋め込んで対称的なプーリング（max-pooling）で集約し、近くの人だけでなく遠くの人の情報も取り込めるようにした。これにより「人混み全体の中での妥当な経路」を考慮できる。
3. デコーダ（LSTM）: プーリングした社会的文脈と、ランダムなノイズ z を受け取り、未来軌道を生成する。
4. 判別器（LSTM ベース）: 生成された（または本物の）未来軌道全体を見て、本物／偽物を判定する。軌道全体を評価するので「途中までは自然だが最後が不自然」のような軌道も見破れる。

### 多峰性をどう出すか — ノイズと Variety Loss
生成器の入力にランダムなノイズ z を入れるので、同じ過去でも z を変えれば違う未来が出ます。これで多峰性を表現します。

ただし素朴にやると、生成器が z を無視して同じ軌道ばかり出すモード崩壊（mode collapse）が起きがちです（VAE の posterior collapse に対応する GAN 版の悩み）。SGAN はこれに対し Variety Loss（多様性損失、別名 best-of-many loss）を導入しました。同じ条件から k 本生成し、その中で「本物に最も近い1本」の L2 誤差だけを罰する:

```
L_variety = min_{i=1..k} || Y_pred^(i) − Y_gt ||_2
```

こうすると、生成器は「k 本のどれかが当たればよい」ので、わざと互いに違う候補（右に避ける／左に避ける）を出すよう促され、多様性が保たれます。これは多峰予測ベンチマークの best-of-K 評価とも整合する設計です。

### 数値・形状の具体例
```
入力  : 各歩行者の過去軌道 [8, 2] = 過去8ステップ × xy (ETH/UCY 慣例: 観測8フレーム=3.2秒)
ノイズ z : 8〜16次元の正規乱数
出力  : 1サンプル 未来軌道 [12, 2] = 未来12ステップ × xy (予測12フレーム=4.8秒)
k回サンプリング → [k, 12, 2] のk本の候補
```

## 手法の系譜と主要論文

- GAN 基礎（Goodfellow et al., "Generative Adversarial Nets", NeurIPS 2014）: 敵対的学習の枠組み自体を提案。画像生成で爆発的に発展し、後に時系列・軌道へ波及。
- Social LSTM（Alahi, Goel, Ramanathan, Robicquet, Fei-Fei, Savarese, CVPR 2016, Stanford）: GAN 以前の古典。Social pooling という「近傍グリッド内の人の隠れ状態を共有する」仕組みで相互作用を扱ったが、出力は単峰で多峰性も GAN もない。SGAN はこの pooling を全員に拡張し GAN を足した発展形。
- Social GAN（Gupta, Johnson, Fei-Fei, Savarese, Alahi, "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks", CVPR 2018, Stanford）: 代表的原典。LSTM エンコーダ・デコーダ + 全員プーリング + GAN + Variety Loss。社会的妥当性を判別器に学ばせ、モード崩壊を Variety Loss で抑えて多峰性を保った。
- SoPhie（Sadeghian, Kosaraju, Sadeghian, Hirose, Rezatofighi, Savarese, CVPR 2019）: SGAN に「シーン画像の CNN 特徴」と「物理的注意・社会的注意（attention、重要な相手や場所に注目する仕組み）」を加えた。壁・障害物などの物理環境と社会的相互作用の両方を考慮し精度を向上させたが、構成は複雑化。
- Social-BiGAT（Kosaraju et al., NeurIPS 2019）: Graph Attention Network と BicycleGAN の考え方で、潜在変数と出力の対応（latent-output mapping）を強め、モード崩壊をさらに抑えた。
- 哲学の対比: VAE 系（[CVAE 軌道生成（条件付き変分オートエンコーダによる多峰な未来の生成）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0165_cvae-trajectory-generation)、[Trajectron++（CVAEと動力学制約による多エージェント多峰予測）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0206_trajectron-plus-plus)）が「確率分布を明示的にモデル化する」のに対し、GAN 系は「分布を明示せず、本物らしさを判別器で学ぶ」点が根本的に異なる。VAE は出力がぼやけやすいが学習は比較的安定、GAN は鮮明だが学習が不安定、という一般的傾向がある。

## 論文の実験結果（定量データ）

### 評価データセットと指標
歩行者軌道予測の事実上の標準ベンチマークが ETH と UCY です。これらは大学キャンパスや街路を俯瞰撮影した実映像で、歩行者の座標が時系列でアノテーションされています。慣例として「過去 3.2 秒（8 フレーム）を観測し、未来 4.8 秒（12 フレーム）を予測」します。指標は多峰手法用の best-of-K:
- minADE_K（minimum Average Displacement Error, メートル）: K 本のうち最良の1本の、全タイムステップ平均の位置誤差。小さいほど良い。
- minFDE_K（minimum Final Displacement Error, メートル）: 最良の1本の、最終フレームでの位置誤差。長期の到達点ズレを直接測る。
Social GAN は K=20 サンプルでこれらを報告しました。

### Social GAN の結果
報告によれば、Social GAN（20 サンプル版、SGAN-20VP-20V）は ETH/UCY 5 シーン平均で minADE 約 0.58 m、minFDE 約 1.18 m 前後を達成し、決定的な Social LSTM（単峰、平均 ADE 約 0.72 m / FDE 約 1.54 m 程度）を明確に上回りました。誤差にして 2 割前後の改善で、これは「未来が複数ありうる状況で best-of-K を許すと、多峰生成が決定的予測より当たりやすい」ことを定量的に示します。

### アブレーション（どの要素を抜くとどうなるか）
論文のアブレーションで重要なのは Variety Loss の効果です。Variety Loss を外して L2 1 本だけで学習すると、生成器が z を無視して同じ軌道ばかり出すモード崩壊が起き、20 本サンプルしても multimodal な利得がほとんど得られず minADE が悪化します。逆に Variety Loss と全員プーリングの両方を入れると、社会的に妥当かつ多様な候補が出て誤差が最小になりました。つまり「プーリングで社会性」「Variety Loss で多様性」が独立に効いていることが確認されています。

定性的にも、SGAN は2人が向かい合って歩くシーンで「互いに左右に避ける」複数の妥当な未来を出し分けるなど、社会的相互作用を反映した予測ができることが示されました。

### k と Variety Loss の関係（数値で理解する）
Variety Loss の k（学習時に1条件あたり何本生成して best を取るか）は、多様性と当てやすさを左右するハイパーパラメータです。報告では k を 1 から増やすほど minADE_20 が下がり、ある程度（k=20 程度）で頭打ちになります。直感的には、k=1 だと「1本を正解に合わせる」ので多様性が出ず（実質モード崩壊）、k を増やすと「どれか1本が当たればよい」ので生成器が候補をばらけさせる動機を得ます。ただし k を大きくしすぎると学習が「外れ候補を放置」しやすくなり、外れ候補の質が下がる副作用も指摘されています。これは後述の best-of-K 評価の弱点（外れ候補が罰せられない）と表裏一体です。

## メリット・トレードオフ・限界

メリット
- 「人間らしさ」「社会的妥当性」のような言語化しにくい性質を、判別器がデータから自動で学んでくれる。
- 生成された軌道が VAE よりも鮮明（平均化されにくい）になりやすい。
- Variety Loss などでモード崩壊を抑えれば、多様な候補を best-of-K で当てやすい。

トレードオフ・限界
- GAN 特有の学習不安定性。生成器と判別器のバランスが崩れると学習が発散したり、品質が振動する。学習率やアーキテクチャに敏感で再現が難しい。
- モード崩壊（同じ軌道ばかり出す）が起きやすく、Variety Loss 等の対策が必須。
- 確率分布を明示しないので「各候補がどれくらいありそうか」という尤度が素直には得られず、下流の安全評価に使いにくい場合がある。
- best-of-K（K=20）で良くても、実用上は少数の候補しか扱えないことが多く、K=1 では弱いことがある。
- 未解決の課題: 多峰の「網羅性」と学習安定性の両立、そして open-loop の minADE が良くても closed-loop で安全に動くとは限らない問題（[PDM（ルールベース予測駆動プランナー）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0197_pdm-rule-based-planner)）。

## 発展トピック・研究の最前線
- GAN から拡散モデルへ: 近年、軌道生成の主役は拡散モデル（diffusion）に移りつつある。拡散はモード崩壊が起きにくく、多様性と鮮明さを両立しやすいが推論が遅い。GAN は学習困難さから第一選択ではなくなりつつある。
- 注意機構・グラフへの統合: SoPhie / Social-BiGAT の流れで、相互作用を Transformer や Graph Neural Network で捉える方向が主流化。生成手法（GAN/VAE/diffusion）と相互作用モデルは直交する設計選択になった。
- 評価の見直し: best-of-K は「当たりの1本」しか見ないため、外れ候補が危険でも罰せられない。closed-loop 評価や分布全体の質を測る指標（カバレッジ、確率較正）への関心が高まっている。
- 実用での選好: 学習安定性を重視する実基盤では、GAN より VAE 系・拡散系・ゴール条件付き手法（[ゴール条件付き計画（先に目標点を予測し、そこへ至る軌道を生成する2段構成）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0179_goal-conditioned-planning)）が選ばれやすい。GAN ベース軌道予測は「相互作用と多峰性をどう学習で扱うか」を最初に体系化した重要なマイルストーンとして位置づけられる。

## さらに学ぶための関連トピック
- [CVAE 軌道生成（条件付き変分オートエンコーダによる多峰な未来の生成）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0165_cvae-trajectory-generation)
- [Trajectron++（CVAEと動力学制約による多エージェント多峰予測）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0206_trajectron-plus-plus)
- [ゴール条件付き計画（先に目標点を予測し、そこへ至る軌道を生成する2段構成）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0179_goal-conditioned-planning)
- [PDM（ルールベース予測駆動プランナー）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0197_pdm-rule-based-planner)
- [LaneGCN（車線グラフ畳み込みによる動き予測）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0444_lanegcn-planning)
