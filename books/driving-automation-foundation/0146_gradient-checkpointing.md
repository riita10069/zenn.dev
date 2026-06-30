---
title: "Gradient Checkpointing"
---

# Gradient Checkpointing

## ひとことで言うと
ニューラルネットの学習では、逆伝播 (backward) のために順伝播 (forward) の途中結果 (activation) をすべて GPU メモリに取っておく必要があり、これが大量のメモリを食います。Gradient Checkpointing は「途中結果のほとんどを捨てておき、backward で必要になったらその場でもう一度 forward を計算し直す」ことで、メモリを大きく節約する手法です。メモリを減らす代わりに計算が増える、時間とメモリのトレードオフです。日本語では「勾配チェックポイント」「activation 再計算 (recomputation)」とも呼ばれます。

## 直感的な理解
長い計算の途中経過をすべてメモに残しながら進む人を想像してください。後で検算 (backward) するとき、全ステップのメモがあれば楽ですが、メモ用紙 (メモリ) を大量に消費します。

別のやり方として「要所要所だけメモを残し、間の計算は捨てる。検算でその区間が必要になったら、直前のメモから素早く計算し直す」があります。メモ用紙は激減し、代わりに計算をやり直す手間が少し増えます。これが gradient checkpointing です。

なぜこの交換が割に合うのか。現代の GPU は「計算は速いがメモリは高価で有限」という非対称性を持ちます。たとえば最新の GPU は 1 秒に数百兆回の演算ができますが、メモリは数十 GB しかありません。多くの大規模学習では、計算能力が余っているのにメモリが先に枯渇して学習が止まります。それなら「余っている計算を使ってメモリを買う」のは合理的な取引です。ここが、計算もメモリも両方足りない状況では成り立たない前提で、checkpointing が「計算に余裕がある現代だからこそ効く」道具である理由です。

## 基礎: 前提となる概念

### activation がメモリを食う理由
forward では、各層が入力を受け取って出力 (activation = 活性化、層の出力テンソル) を計算します。backward では連鎖律 (chain rule = 合成関数の微分則) に従って勾配を計算しますが、そのとき「forward 時の各層の入力・出力」が必要になります。

具体例を挙げます。ReLU (負の値を 0 にする活性化関数) の微分には forward 時の入力の符号が要ります。行列積 y = Wx の重み W に関する勾配は ∂L/∂W = (∂L/∂y) x^T で、forward 時の入力 x が要ります。だから素朴な学習は、forward で全層の activation を保持したまま backward を待ちます。これを保持しないと backward の式が計算できないのです。

層が L 個あり各層の activation サイズがほぼ同じなら、必要メモリは層数 L に比例します。深いネットワークや高解像度の入力 (大きな特徴マップを扱う検出・セグメンテーション・俯瞰視点 (BEV) 系のモデルなど) では、この activation がパラメータ本体よりはるかに大きなメモリを占めることがよくあります。

### activation メモリの数値感
batch B、特徴マップの空間サイズ H×W、チャネル C の activation は B×C×H×W 要素を持ちます。たとえば B=8、C=256、H=W=300、fp16 (2 バイト) なら 1 層で 8×256×300×300×2 ≈ 1.1GB。高解像度な特徴マップを多数の層で保持すると、これだけで数十 GB に達することがあります。「重みは数 GB で入るのに、activation で OOM (Out Of Memory) する」というのが典型的な行き詰まりです。このとき出番になるのが gradient checkpointing です。Transformer でも、attention の中間テンソル (系列長の 2 乗に比例) が長い系列で巨大になり、同じ問題が起きます。

## 仕組みを詳しく

### 基本の考え方: 捨てて、後で作り直す
forward 時に全層の activation を保持する代わりに、一部の層 (チェックポイント) の出力だけを残し、それ以外は捨てます。backward でチェックポイント間の勾配が必要になったら、直前のチェックポイントから forward を「もう一度だけ」実行して必要な activation を一時復元します。つまり「保存しておく」のをやめて「必要なときに再計算する」のです。再計算した activation は使い終わったらまた捨てるので、一度にメモリに乗るのは「1 区間分」だけになります。

### ステップ分解 (4 層の例)
層 A → B → C → D があるとします。

標準の backward:
```
1. forward で A,B,C,D の出力を全部メモリに保持
2. backward で D→C→B→A と勾配を計算 (保持した activation を使う)
   メモリ消費 = 4 層分の activation
```

checkpointing (B の出力だけチェックポイントとして残す例):
```
1. forward で B の出力だけ保持。A,C,D の中間 activation は捨てる
2. backward で C,D の勾配が必要 → 保持した B の出力から C,D を再 forward
   して activation を一時復元 → 勾配計算 → また捨てる
3. A の勾配が必要 → 入力から A,B を再 forward
   メモリ消費 = チェックポイント(B) + 一時的に再計算する 1 区間分だけ
```

### メモリと計算のトレードオフ (理論値)
原論文の有名な結果は「層数 L のネットワークで、チェックポイントを √L 個おきに置くと、メモリは O(√L) に減り、追加計算は forward 1 回分だけ」というものです。

直感的な導出: チェックポイントを m 個置くと、各区間は L/m 層。backward 時に保持するのは「チェックポイント m 個」+「再計算中の 1 区間 L/m 層」なので、メモリ = m + L/m。これを m で最小化すると m = √L のとき最小値 2√L が得られます (相加相乗平均より)。L = 100 層なら、メモリは 100 に比例していたのが 2√100 = 20 程度に下がり、約 5 分の 1 です。計算コストは「元の forward + 再計算 1 回」で約 2 回ぶん相当、backward と合わせた学習時間としてはおおむね 1.3〜1.5 倍に増えます。

```
before (標準):       メモリ ∝ L      計算 = forward 1 回 + backward
after (checkpoint):  メモリ ∝ √L     計算 = forward 約 2 回 + backward
                     ↑ 大幅減        ↑ 30% 前後増
```

### selective recomputation (選択的再計算)
すべてを再計算するのではなく「再計算が安く、メモリを多く食う演算」だけを選んで捨てる戦略もあります。たとえば Transformer の attention は activation が巨大 (系列長の 2 乗) ですが、その再計算 (softmax の再評価など) は比較的軽いので、ここだけ checkpoint すると「少ない計算増で大きなメモリ削減」が得られます。逆に、行列積のように計算が重い部分の activation はなるべく保持する。全部捨てる full recomputation より効率的な配置として、大規模 Transformer 学習で実用されています。

### PyTorch での使い方
```python
from torch.utils.checkpoint import checkpoint

# 通常: out = self.block(x)              ← block の中間 activation を全保持
# checkpointing:
out = checkpoint(self.block, x, use_reentrant=False)  # activation を捨て、backward で再計算
```
Transformer 系では各エンコーダ層を checkpoint で包む「層単位チェックポイント」が定番です。`use_reentrant=False` は新しい実装で、再計算の挙動 (RNG 状態の扱いや高階微分との互換) がより正しく、推奨されます。FSDP などフレームワークによっては、層を wrap するのと同じ感覚で checkpointing を有効にする API が用意されています。

### 乱数を使う層への注意
dropout のように乱数を使う層では、forward 時と再計算時で同じ乱数を引かないと結果がズレ、勾配が誤ります。たとえば dropout は forward でどのニューロンを 0 にしたかを backward で再現できないと、保存した出力と再計算した出力が食い違ってしまいます。PyTorch の checkpoint は内部で RNG (乱数生成器) の状態を保存・復元してこれを防ぎますが、`use_reentrant` の設定によって挙動が変わるため注意が要ります。

### 他のメモリ最適化との関係
checkpointing は activation を削るのが専門で、パラメータや optimizer 状態は削りません。そこは [ZeRO](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer) / [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel) の領分です。両者は直交するので併用が前提で、FSDP がパラメータ系を、checkpointing が activation を削ることで、初めて超大規模モデルが現実的なメモリに収まります。[混合精度](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision) とも独立に併用でき、activation を低ビットで持つことでさらに削減できます。「3 つの軸 (モデル状態・activation・数値精度) を別々に攻める」という整理が、現代の大規模学習のメモリ設計の基本です。

## 手法の系譜と主要論文

- Chen, Xu, Zhang, Guestrin, "Training Deep Nets with Sublinear Memory Cost" (2016, arXiv:1604.06174)。本トピックの原典。「activation を全保持せず、戦略的に選んだチェックポイントだけ残し、残りを backward 時に再計算する」ことを提案。動機は「深いネットの学習はメモリが律速で、メモリさえ減れば同じハードで深く・大きくできる」。L 層のメモリを O(L) から O(√L) に下げ、追加コストは forward 1 回分のみと理論・実験で示しました。「計算は安く、メモリは高い」状況での合理的交換と位置づけました。

- Gruslys, Munos, Danihelka ら, "Memory-Efficient Backpropagation Through Time" (NeurIPS 2016, DeepMind)。RNN (時系列を扱うネット) 向けに、どの中間状態を保存し・どこを再計算するかを動的計画法で最適に決める手法 (BPTT = 時間方向の逆伝播の文脈) を提案。与えられたメモリ予算の下で再計算回数を最小化する配置を解き、checkpointing の「保存と再計算の最適配置」を理論的に深めました。

- Shoeybi ら, "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019, NVIDIA, arXiv:1909.08053)。Transformer の大規模学習で activation recomputation を標準採用し、checkpointing が大規模学習の必須技術であることを実用面で確立しました。

- Korthikanti, Casper, Shoeybi ら, "Reducing Activation Recomputation in Large Transformer Models" (2022, NVIDIA, arXiv:2205.05198)。Transformer の activation メモリを詳細に解析し、sequence parallelism と selective activation recomputation を組み合わせることで、full recomputation の計算オーバーヘッドを大幅に削りつつメモリも抑える手法を提案。「全部再計算するのは無駄が多い」ことを定量的に示し、現代の大規模 LLM 学習の標準的なレシピになりました。

系譜: 「Chen 2016 (基本アイデアと √L の理論) → Gruslys 2016 (最適配置の動的計画) → Megatron 2019 (大規模 Transformer での実用化) → Korthikanti 2022 (selective recomputation で計算増を最小化)」と発展し、現在は LLM 学習のデフォルト装備です。

## 論文の実験結果 (定量データ)

Chen ら (2016) の主要な実測です。指標はメモリ使用量と学習時間 (1 イテレーションあたり)。

- 1000 層級の深い ResNet 系ネットワークで、メモリを大幅に削減 (報告では数分の 1 〜 桁レベル) しつつ、追加の計算コストはおよそ forward 1 回分に収まることを示しました。具体的には、メモリが O(L) から O(√L) に落ちることで、同じ GPU に乗らなかった深さのネットが乗るようになったと実証。たとえば一定のメモリ予算で学習できる層数が数倍に伸びました。
- 計算オーバーヘッドは典型的に学習時間の +30% 前後。これが「メモリを √L まで削る代償」であり、論文はこのトレードオフを明示しています。

大規模 Transformer 系の知見 (Megatron、Korthikanti 2022) として重要なアブレーション:
- full activation recomputation を入れると activation メモリがほぼ消える代わりに、計算時間 (forward FLOPs) が約 30〜40% 増加。
- selective recomputation (attention など一部だけ再計算) にすると、Korthikanti 2022 の報告ではメモリ削減の大半を保ちつつ計算オーバーヘッドを数% (full の 30% 超に対し大幅減) まで抑えられました。sequence parallelism と併用することで、activation メモリを層数や系列長に対してさらに有利にスケールさせられることも示されています。つまり「どこを再計算するか」の選び方で、トレードオフの位置を有利に動かせることが定量的に示されました。これは「全部捨てる」のが最善ではないという重要な教訓です。

指標の意味の補足: ここで効くのはピーク activation メモリ (学習中に一瞬でも到達する最大メモリ) で、これが GPU 容量を超えると OOM して学習自体が不可能になります。checkpointing の価値は「速度を多少犠牲にして、不可能を可能にする」点にあり、速度指標だけ見ると損に見えても、そもそも回らなかった設定 (大きい batch・高解像度・深いモデル) が回るようになることが本質です。逆に言えば、メモリに余裕がある小規模学習では入れるだけ遅くなる純損なので、必要なときだけ使う道具です。

## メリット・トレードオフ・限界

メリット
- activation メモリを大幅に削減 (層数 L に対し理論上 O(√L))。重みは入るが activation で OOM するケースに特効
- より大きい batch・高解像度・深いモデルを同じ GPU で学習できる
- モデルの出力・精度は変わらない (同じ計算を再現するだけで、近似ではない)。対象ブロックを包むだけで導入でき、[bf16](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision) や [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel) と併用できる

トレードオフ・限界
- forward を余分に計算するため学習が遅くなる (典型的に 30% 前後、適用範囲が広いほど増える)
- 「どの区間をチェックポイントにするか」で効果が変わり、最適配置にはチューニングが要る (機械的に √L 間隔が最善とは限らない)
- パラメータや optimizer 状態のメモリは減らない。activation 専用の対策
- 乱数を使う層 (dropout など) は再現性のため RNG 状態の保存・復元が必要 (`use_reentrant` の設定に注意)

研究上の未解決課題: 与えられたメモリ予算と計算予算の下で「どこを保存し、どこを再計算するか」を自動で最適化する問題は、計算グラフが複雑な現代モデル (分岐・スキップ接続・attention) では非自明です。コンパイラ的なアプローチ (計算グラフを解析して最適な checkpoint を自動挿入する) が研究・実装されつつありますが、汎用的な最適解は未確立です。

## 発展トピック・研究の最前線
- selective / smart recomputation: 演算ごとの「再計算コスト vs メモリ削減量」を見て、得な演算だけ再計算する自動化。Korthikanti 2022 はこの方向の代表
- コンパイラ統合: 計算グラフを解析し checkpoint を自動挿入する仕組み (深層学習コンパイラの一機能)。手で区間を切る代わりに、min-cut 的な解析で最適な保存点を決める研究がある
- offload との比較・併用: activation を捨てて再計算する代わりに、CPU メモリへ退避する activation offload という別解もあり、計算を増やしたくない場合の選択肢。再計算 (計算を払う) か offload (転送帯域を払う) かの使い分けや併用が研究対象
- 超大規模学習での標準化: FSDP × activation checkpointing × 混合精度 × sequence parallelism を組み合わせ、限られた GPU で最大規模のモデルを学習する構成が事実上の標準になっている

## さらに学ぶための関連トピック
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision)
- [FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel)
- [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0604_ddp-distributed-data-parallel)
