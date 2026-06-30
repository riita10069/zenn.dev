---
title: "最適化・学習テクニック"
---

# 最適化・学習テクニック

オプティマイザ・学習率・正則化・データ拡張など、学習を成功させるためのテクニックを扱います。

この章では次の60個のトピックを順に解説します。


## bf16 Mixed Precision

### ひとことで言うと
ニューラルネットの学習中、計算に使う数値を「32ビットの高精度」ではなく「16ビットの軽い形式 bfloat16」に切り替えて、メモリを減らし計算を速くするテクニックです。bfloat16 は表現できる数の「範囲 (ダイナミックレンジ)」が32ビットと同じくらい広いので、float16 で必要だった面倒な調整 (損失スケーリング / GradScaler) なしで、ほぼそのまま安全に学習できるのが最大の特徴です。新しい世代の GPU や TPU で標準的に使われています。

### 直感的な理解
モデルが大きくなると、学習に2つの困りごとが出ます。一つはメモリ不足。重み・勾配・最適化器の状態・中間結果 (activation) をすべて GPU メモリ (VRAM) に置く必要があり、32ビットだと容量が足りずクラッシュ (OOM = Out Of Memory) します。もう一つは遅さ。GPU には16ビット計算を高速にこなす専用回路 (Tensor Core 等) があり、32ビットのままだとこの専用回路を活かせません。

そこで「全部を16ビットにするのではなく、速度が欲しい行列計算は16ビット、精度が要る一部は32ビットのまま」という使い分け = 混合精度 (Mixed Precision) が生まれました。問題は「どの16ビット形式を使うか」です。float16 は範囲が狭く、学習中に出る極小の勾配がゼロに潰れて学習が止まるため、勾配を一時的に拡大する仕掛け (GradScaler) が必須でした。bfloat16 はこの範囲を float32 並みに広げることで、仕掛けなしで安全に使えるようにした形式です。つまり bfloat16 は「float16 の高速・省メモリは欲しいが、面倒は避けたい」という要求への答えです。

### 基礎: 前提となる概念

#### 浮動小数点数とは
コンピュータは小数を「浮動小数点数 (floating point)」で表します。1個の数値を3つの部分で構成します。

- **符号 (sign): 1ビット**。プラスかマイナスか。
- **指数部 (exponent)**。数の「大きさのスケール」を決める。これが広いほど、ものすごく大きい数や小さい数まで表せる = ダイナミックレンジが広い。指数部は「2の何乗か」を表すので、ビットが1つ増えると表せる範囲が指数的に広がる。
- **仮数部 (mantissa)**。数の「細かさ・有効ケタ数」を決める。これが多いほど、隣り合う表現可能な数の間隔が細かくなる = 精度が高い。

ディープラーニングで標準だったのは float32 (32ビット、FP32) で、1個の数値に4バイト使います。

#### float16 と bfloat16 の違い
16ビット形式は2種類あり、同じ16ビットを指数部と仮数部にどう配分するかが違います。

| 形式 | 全ビット | 指数部 | 仮数部 | ダイナミックレンジ (おおよそ) | 細かさ |
|---|---|---|---|---|---|
| float32 (FP32) | 32 | 8 | 23 | 1e-38 〜 3e38 | 高い |
| float16 (FP16) | 16 | 5 | 10 | 6e-5 〜 65504 | そこそこ |
| bfloat16 (BF16) | 16 | 8 | 7 | 1e-38 〜 3e38 | 低い |

ポイントは指数部です。bfloat16 は指数部8ビットで float32 と同じ。つまり表せる数の範囲が float32 と同等です。実際、float32 の上位16ビットをそのまま切り取った形が bfloat16 で、これは「float32 とは指数部の構造が完全に同じ、下位の細かいビットを捨てただけ」という意味になります。だから float32 ↔ bfloat16 の変換が単純なビット切り捨てで済むという実装上の利点もあります。float16 は指数部5ビットしかなく、65504 より大きい数や 6e-5 より小さい数を表せません。bfloat16 は範囲を取った代わりに仮数部が7ビットと少なく、細かい桁の精度は float16 より劣ります。

#### なぜ範囲が重要か — アンダーフロー
学習では「勾配 (gradient)」= 重みをどう動かせば誤差が減るかを表す値を計算します。学習が進むと勾配はしばしば 1e-7 のような極小値になります。float16 でこれを表そうとすると最小値 6e-5 を下回り、ゼロに丸められます。これを「アンダーフロー (underflow)」と呼びます。勾配がゼロになると重みが更新されず学習が止まります。float16 ではこれを防ぐため、損失を大きく拡大してから逆伝播する GradScaler が必須でした (詳細は [fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。bfloat16 は範囲が広いのでアンダーフローがほぼ起きず、GradScaler 不要で済みます。

ここに「学習における精度と範囲のどちらが効くか」という重要な経験則があります。Kalamkar らの研究が定量的に示したのは、学習の安定性に効くのは「細かさ (仮数部)」より「範囲の広さ (指数部)」だという点です。勾配は大小に何桁も振れるので、それを潰さず表せる範囲のほうが、各値の細かい精度より大事だ、という直感です。bfloat16 はこの直感に沿って指数部を優先した形式です。

### 仕組みを詳しく

#### 自動混合精度 (AMP) の流れ
フレームワークでは「この計算ブロックの中の、安全な演算だけ自動で bfloat16 に切り替える」というコンテキストを使います。PyTorch なら `torch.autocast` です。典型的な学習ループは次の形です。

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model(x)
    loss = loss_fn(output, target)

# bf16 は fp32 並みの range を持つので GradScaler は不要
loss.backward()
optimizer.step()
```

段階で説明します。

1. `autocast` の内側で forward (順伝播) を実行。autocast は行列積 (`matmul`)・畳み込み (`conv`) など「丸めても安全で速度が欲しい演算」を自動で bfloat16 に落とす。一方、総和 (`sum`)・softmax・正規化 (LayerNorm/BatchNorm)・指数や対数のような「精度が要る/値が大きくなりやすい/桁が飛びやすい演算」は自動で float32 に残す。どの演算をどちらにするかはフレームワークの allow/deny リストで決まり、プログラマが一つずつ指定する必要はない。
2. `loss.backward()` で勾配を計算 (逆伝播)。bfloat16 は範囲が広いので勾配がゼロに潰れず、GradScaler の拡大処理が不要。
3. `optimizer.step()` で重みを更新。重みのマスターコピー自体は float32 で保持され、更新の足し算は float32 精度で行われるので、小さな更新が消えない。

#### 重みは float32、計算は bf16 — なぜ両方持つのか
混合精度の要は「重みの float32 マスターコピー」を持つことです。重み更新 `w ← w - lr * grad` で、`lr * grad` が極めて小さいと、16ビットの `w` に足しても丸めで消えてしまいます。これは「大きな数に微小な数を足すと吸収される」という浮動小数点の性質です。具体例: bfloat16 で `1.0` の次に表せる数は約 `1.0078` (仮数部7ビットなので刻みが粗い)。もし更新が `0.001` だと、`1.0 + 0.001 = 1.001` は最も近い表現に丸められて `1.0` のまま、つまり更新が完全に消えます。float32 で重みを保持し更新もそこで行えば (float32 では `1.0` の次の刻みは約 `1.0000001`)、この消失を防げます。forward/backward だけ bf16 にして速度とメモリを稼ぐ、というのが基本構図です。

#### 数値例: メモリと速度
あるテンソルが形状 `[4, 256, 450, 300]` (batch=4、チャネル256、高さ450、幅300) だとします。要素数は `4×256×450×300 = 138,240,000` 個。

- float32: `138.24M × 4バイト ≈ 553MB`
- bfloat16: `138.24M × 2バイト ≈ 276MB`

中間の activation (各層の出力。逆伝播のために保持される) がこの調子で半分になるので、実際の VRAM 削減は数 GB 単位になります。高解像度の特徴マップを多数保持するモデル (例えば俯瞰図 = BEV のような空間表現を作る知覚モデル) では、bf16 によって同じ GPU で扱える解像度やバッチサイズが大きく変わります。速度面でも、新しい GPU の Tensor Core は bf16 行列積を float32 の数倍のスループット (理論ピークで2〜数倍) で処理します。

なお、混合精度はメモリを「半分」にはしません。最適化器の状態 (Adam なら運動量と分散の2つ) は通常 float32 で持ち、forward/backward の activation が主に削れる部分です。よって全体の削減率はモデル構成に依存し、activation が支配的なモデルほど効果が大きくなります。

### 手法の系譜と主要論文
- **2017/2018年 — 混合精度の確立**: Micikevicius ら "Mixed Precision Training" (ICLR 2018、NVIDIA/Baidu、arXiv:1710.03740) が float16 を使う枠組み (float32 マスターコピー + 損失スケーリング + float32 累積) を確立。混合精度の出発点です。詳細は [fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。
- **2019年 — bfloat16 の提案・検証**: Kalamkar, Mudigere, Mellempudi ら "A Study of BFLOAT16 for Deep Learning Training" (Intel/Google、arXiv:1905.12322) が、指数8ビット・仮数7ビットの bfloat16 を体系的に検証。「float16 は損失スケーリングが要るが、bfloat16 は float32 と同じ指数幅なのでスケーリングなしで学習できる」ことを示しました。
- **2019年 — 業界標準化**: Wang & Kanwar "BFloat16: The secret to high performance on Cloud TPUs" (Google) が、Google TPU が bfloat16 を標準採用した背景を解説。これと NVIDIA の Ampere 世代 (A100、2020) が bf16 を Tensor Core でネイティブ対応したことで、bf16 は学習のデファクト標準形式になりました。
- **2022年 — 最適化器状態の量子化**: Dettmers ら "8-bit Optimizers via Block-wise Quantization" (ICLR 2022) が、混合精度では float32 で残しがちな Adam の最適化器状態 (運動量・分散) を8ビットに圧縮し、精度をほぼ保ったままメモリをさらに削れることを示しました。bf16 と組み合わせる省メモリ手法として広く使われています。
- **2022年以降 — さらに低ビットへ**: Micikevicius ら "FP8 Formats for Deep Learning" (arXiv:2209.05433) が8ビット浮動小数点 (E4M3 / E5M2) を提案。Hopper/Ada 世代以降の GPU が FP8 を Tensor Core で対応し、巨大モデルの学習・推論で使われ始めました。bf16 → fp8 が低精度化の次の段階です。

### 論文の実験結果 (定量データ)
- **bfloat16 の精度同等性 (Kalamkar et al., 2019)**: 画像分類 (ResNet-50 on ImageNet)、物体検出、機械翻訳、音声、推薦、GAN など幅広いタスクで、bfloat16 学習が float32 とほぼ同等の最終精度に到達することを示しました。例えば ResNet-50/ImageNet で、bf16 学習の top-1 精度 (1000クラスで最も確信したクラスが正解と一致する割合) は float32 とコンマ数 % 以内の差にとどまる、と報告。意味: 「半分のビット幅でも、最終的な賢さはほとんど落ちない」。しかも float16 と違い損失スケーリングのチューニングが不要です。
- **混合精度の速度・メモリ効果 (Micikevicius et al., ICLR 2018)**: ImageNet 分類・物体検出・音声認識・翻訳・GAN などで、float32 と同等精度を保ちつつ、メモリ使用量を約半減し、Tensor Core 対応 GPU で大幅な高速化を達成。「同等精度・半分メモリ・高速」が要点です。
- **アブレーション的な知見**: float16 で損失スケーリングを「外す」と、勾配のヒストグラムの大部分が float16 の表現可能範囲を下回り (アンダーフロー)、学習が発散・停滞します (= スケーリングが必須)。一方 bfloat16 では同じ実験でスケーリングを外しても学習が安定する — これが bf16 の核心的な利点を示すアブレーションです。逆に bf16 で仮数部が7ビットしかないことの影響を見ると、大きな値の総和や正規化統計など一部の演算で精度劣化が出るため、それらは float32 に残すのが安定 (autocast はこれを自動でやっている)。
- **8-bit 最適化器 (Dettmers et al., 2022)**: GLUE/言語モデル/画像分類など多様なタスクで、Adam の状態を8ビット化しても32ビット Adam とほぼ同等の最終性能を保ち、最適化器が占めるメモリを大きく削減できることを報告。bf16 と直交する省メモリ手段として価値が示されました。
- **FP8 の限界 (Micikevicius et al., 2022)**: FP8 では多くのタスクで bf16 同等精度を出せるが、レンジ/精度がさらに厳しく、層ごとのスケーリングや慎重なキャスト設計が必要。bf16 ほど「何も考えず差し替え」とはいかない、というのが現状の知見です。

数値の意味: ここでの「精度差コンマ数 %」は、半分のメモリと数倍の速度という大きな利得に対して無視できるほど小さい、という点が重要です。だからこそ bf16 は標準になりました。

### メリット・トレードオフ・限界
メリット:
- VRAM をおおむね半減 (activation 部分) でき、より大きい batch やより高い解像度を同じ GPU で回せる。
- Tensor Core を使えるので行列積・畳み込みが高速化する。
- float16 と違いダイナミックレンジが float32 並みに広く、GradScaler が不要 → コードが単純で学習が安定。
- 多くのタスクで float32 と同等の最終精度が得られる。実験ループを短縮できる。
- float32 の上位ビットを切るだけの単純な形式のため、float32 ↔ bf16 変換が安価。

トレードオフ・限界:
- 仮数部が7ビットと少なく有効ケタが粗い。大きな値の総和・正規化統計・指数/対数など数値的に敏感な演算は精度低下が起こりうるため、autocast は内部でそれらを float32 に残す。
- bfloat16 をネイティブ対応する GPU/TPU (NVIDIA Ampere 以降、Google TPU など) が必要。古い GPU (Pascal/Turing 世代の一部、T4 など) では bf16 の恩恵が小さい/非対応で、float16 + GradScaler を使うことになる ([fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。
- メモリ削減には限界がある。さらに削るには [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning) (中間結果を捨てて再計算する)、8-bit 最適化器、[FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) (重み・勾配・最適化器状態を複数 GPU に分割保持する) が必要。
- 推論時に bf16 をそのまま使うと、仮数部の粗さで出力がわずかに変わることがある。精度が重要な検証では float32 と突き合わせるのが安全。
- 未解決領域として、bf16 と FP8 を層ごとに使い分ける最適な混合精度方策、低精度下での学習安定性の理論的理解などが研究の途上にある。

### 発展トピック・研究の最前線
- **FP8 学習・推論**: E4M3 (精度寄り) と E5M2 (範囲寄り) を forward/backward で使い分ける手法。巨大言語モデルの事前学習で実用化が進む。
- **層ごと/テンソルごとのスケーリング**: 低ビットになるほど、テンソルごとに動的なスケール係数を持たせて範囲を合わせる手法が重要に。
- **重みの量子化 + 低精度最適化器状態**: Adam の最適化器状態 (運動量・分散) を8ビットに量子化してメモリを削る手法 (8-bit Adam など)。
- **TF32 などの中間形式**: NVIDIA の TF32 (内部で仮数部を削った行列積) は「コード変更なしで bf16 的な速度」を得る折衷案。
- **数値安定性の理論**: 低精度が収束に与える影響、確率的丸め (stochastic rounding) で低ビットの偏りを減らす手法など。

### さらに学ぶための関連トピック
- [fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## DDP (Distributed Data Parallel)

### ひとことで言うと
1台の GPU では遅すぎる、あるいはデータが多すぎて時間がかかりすぎる学習を、複数の GPU に分散して速くする最も標準的な方法です。すべての GPU に「同じモデルのコピー」を置き、学習データを GPU 間で分割して各 GPU が独立に計算し、計算した勾配 (重みをどちらにどれだけ動かすべきかを表す量) を all-reduce という集合通信で合算・平均することで、全 GPU の重みを常に同一に保ちます。結果として「巨大な batch を複数 GPU で並列に処理する」のと数学的に等価になり、台数にほぼ比例して学習が速くなります。これがデータ並列 (data parallelism) と呼ばれる戦略で、DDP はその PyTorch における標準実装です。

### 直感的な理解
たとえば 100 万枚の画像を分類するモデルを学習するとします。1台の GPU は 1 秒間に処理できる画像枚数に上限があり、データセット全体を 1 周 (これを 1 エポックと呼びます) するのに何時間もかかります。学習は通常何十エポックも回すので、1台では数日かかることも珍しくありません。

ここで「分業」を考えます。GPU が 4 台あれば、データを 4 等分して各台に配り、4 台が同時に働けば理論上 4 倍速くなります。料理にたとえると、1 人で 100 人分の料理を作る代わりに、4 人で 25 人分ずつ分担するイメージです。

ただし単純な分業には落とし穴があります。各 GPU が違うデータで学習すると、求める勾配がバラバラになります。GPU0 は「この重みを増やせ」、GPU1 は「減らせ」と言うかもしれません。放置すると 4 台の重みがどんどんズレて、最終的に 4 つの別物のモデルになってしまいます。これでは分業の意味がありません。

そこで毎ステップ、4 台が出した勾配を「平均」して全員が同じ更新を適用します。こうすれば 4 台はずっと一卵性の双子のように同一の重みを保ち、「4 倍のデータを 1 台で処理した」のと完全に同じ結果になります。この「全員の勾配を集めて平均し、結果を全員に配り戻す」通信が all-reduce です。DDP の本質はこの一点に尽きます。

なぜ「データを分ける」のであって「モデルを分ける」のではないのか、という疑問が出るかもしれません。モデルそのものを切り分ける方式 (テンソル並列・パイプライン並列) もありますが、それらは層の内部や層の境界で頻繁に通信が必要で実装が難しく、通信効率も悪くなりがちです。データ並列は「1 ステップに 1 回まとめて勾配を交換するだけ」で済むので、最も単純で通信効率がよく、モデルが 1 台に収まる限り第一選択になります。

### 基礎: 前提となる概念

#### ニューラルネット学習の 1 ステップ
ニューラルネットの学習は次の繰り返しです。

1. forward (順伝播): データを入力し、層を順に通して出力を出す
2. loss (損失): 出力と正解の差を 1 つの数値で測る
3. backward (逆伝播): その損失を各重みで微分し、勾配を得る。勾配は「損失を下げるには各重みをどちらにどれだけ動かせばよいか」を表すベクトル
4. step: optimizer (最適化器、Adam や SGD など) が勾配を使って重みを少しだけ更新する

この 4 つを何百万回も回して、損失が小さくなる重みを探します。

#### batch (バッチ) という単位
1 ステップで一度に処理するデータのまとまりを batch と呼び、その枚数を batch size と言います。batch size を大きくすると、1 ステップで多くのサンプルから勾配を計算するので、勾配のノイズ (サンプルのばらつきによる推定誤差) が減って安定します。データ並列とは要するに「大きな batch を複数 GPU に切り分けて並列計算する」ことです。各 GPU が処理する分を local batch、全 GPU 合計を global batch (実効 batch) と呼びます。

#### 集合通信 (collective communication) の用語
複数のプロセスが協調して値をやり取りする通信を集合通信と呼びます。本トピックで登場するのは主に次の 3 つです。

- broadcast: 1 つのプロセスの値を全プロセスに配る (初期重みを揃えるのに使う)
- reduce: 全プロセスの値を 1 つのプロセスに集めて演算 (合算など) する
- all-reduce: 全プロセスの値を集めて演算 (合算や平均) し、その結果を全プロセスに配り戻す。「reduce (集約)」を「all (全員に)」行うので all-reduce

#### world size と rank
- world size: 参加するプロセス (通常 GPU 1 台 = 1 プロセス) の総数。4 台なら 4
- rank: 各プロセスに振られた通し番号 (0, 1, 2, 3)。rank 0 が代表 (チェックポイント保存やログ出力を担う) になることが多い
- local rank: 1 ノード内での GPU 番号。複数ノードにまたがる場合、global な rank とノード内の local rank を区別する

#### ノードとインターコネクト
GPU が物理的にどう繋がっているかは性能に直結します。1 台のサーバ (ノード) 内の GPU は NVLink (NVIDIA の GPU 間直結バス、数百 GB/s) などで高速に繋がりますが、ノードをまたぐと InfiniBand や Ethernet (数十〜数百 Gb/s、バイト換算で 1 桁以上遅い) を通ります。「node 内は速い、node 間は遅い」というこの非対称性が、後述する scaling 効率の頭打ちの主因です。

### 仕組みを詳しく

#### 1 ステップの流れ (GPU 4 台、global batch 256 の例)
1. 学習開始時、rank 0 の初期重みを broadcast で全 GPU にコピーし、4 台の重みを完全に一致させる
2. global batch 256 サンプルを 4 分割し、各 GPU に 64 サンプルずつ配る。データの分配は、サンプリングが重複しないよう各 rank に異なるインデックス集合を割り当てる仕組み (PyTorch では `DistributedSampler`) が担う
3. 各 GPU が自分の 64 サンプルで forward → loss → backward を実行し、ローカルな勾配を得る (この時点では 4 台の勾配はバラバラ)
4. all-reduce で 4 台の勾配を合算し、4 で割って平均する。通信後は全 GPU が同一の平均勾配を持つ
5. 各 GPU が同じ平均勾配で `optimizer.step()` を実行 → 更新後も 4 台の重みは完全に一致
6. 次のステップへ

数式で書くと、各 GPU i がローカル勾配 g_i を計算し、all-reduce が g = (1/N) Σ_i g_i を全員に届ける、という処理です。N は world size。これは global batch 256 サンプルで計算した勾配と数学的に等しくなります。重要なのは、loss 関数がサンプル平均 (batch 内で平均する形) のとき、各 GPU のローカル勾配を平均すれば自動的に global batch の勾配になるという点で、ここがズレると (たとえば loss を sum で取ると) 実効学習率が台数倍に化けるバグになります。

#### 計算と通信のオーバーラップ (DDP の核心的最適化)
素朴に実装すると「backward が全部終わってから all-reduce 開始 → 終わってから step」となり、通信時間がまるごと待ち時間になります。DDP の賢い点は、この通信を backward の計算と重ねて隠すことです。

逆伝播は出力側の層から入力側へ進みます。つまり最後の層の勾配が最初に確定し、最初の層の勾配が最後に確定します。DDP はある層の勾配が出来上がった瞬間に、まだ計算中の手前の層を待たず、その層の all-reduce を裏で開始します。

```
時間 →
backward:  [層4勾配][層3勾配][層2勾配][層1勾配]
通信:            [層4 all-reduce][層3 all-reduce][層2..][層1..]
                 ↑ backward と並行して通信が走る
```

これにより、通信時間の大半が計算時間の裏に隠れ、見かけ上ほぼ通信ゼロのように振る舞います。実装上は、各パラメータの勾配が確定したことを autograd の hook (勾配確定時に呼ばれるコールバック) で検知して通信を発火させます。

#### gradient bucketing (勾配バケツ化)
パラメータ 1 個ごとに all-reduce すると、小さい通信が大量に発生して非効率です (通信には 1 回ごとに固定の立ち上げコスト、いわゆるレイテンシがある)。そこで DDP は複数パラメータの勾配を 1 つのバケツ (典型的に 25MB 程度) にまとめ、バケツ単位で all-reduce します。大きなまとまりで送ることで帯域を有効に使え、オーバーラップの単位にもなります。バケツの境界は backward で勾配が確定する順 (おおむね層の逆順) に合わせて区切られ、最初に埋まったバケツから順に通信を発火させることでオーバーラップを最大化します。

#### ring all-reduce のアルゴリズム
all-reduce の効率的な実装として ring all-reduce が広く使われます。N 台の GPU を論理的にリング状に並べ、データを N 分割して、各台が隣へ少しずつ送りながら部分和を積み上げます。reduce-scatter フェーズ (N-1 ステップ) で各台が「自分担当の部分の全合計」を持ち、all-gather フェーズ (N-1 ステップ) でそれを全員に配ります。

総通信量は台数 N にほぼ依存せず一定で、各台が送受信するのは約 2(N-1)/N × データサイズ ≈ 2 × データサイズです。つまり台数が増えても 1 台あたりの通信量がほぼ増えないため、スケールしやすいのが ring all-reduce の決定的な利点です (素朴な「全員が rank 0 に送って rank 0 が配る」方式だと rank 0 に通信が集中して N に比例して悪化します)。NVIDIA の NCCL ライブラリがこれを GPU 間で高速実行し、トポロジ (NVLink/InfiniBand の繋がり方) を見て ring や tree などのアルゴリズムを自動選択します。

#### batch サイズと学習率: 線形スケーリング則
4 GPU で各 64 サンプルなら実効 batch は 256 です。batch を大きくすると 1 ステップの勾配が安定する反面、同じ学習率では「1 エポックあたりの更新回数」が減るので収束が遅くなります。経験則として線形スケーリング則 (batch を k 倍にしたら学習率も k 倍) が知られ、ただし学習初期に学習率を徐々に上げる warmup を併用しないと、大きな学習率で発散します。

これは DDP 導入時に最も見落とされやすい落とし穴で、「台数を増やしたら精度が落ちた」のほとんどはこの調整漏れです。直感的には、batch が大きいほど勾配は正確 (低ノイズ) なので大きな歩幅で進めますが、学習のごく初期は重みがまだランダムに近く、大きな歩幅で進むと壊れるため、最初だけ歩幅を抑えるのが warmup です。さらに極端な大 batch では線形スケーリングが破綻し、LARS / LAMB (層ごとに学習率を適応させる optimizer) などが必要になります。

#### 典型的な実装
```python
import os, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group(backend="nccl")          # GPU 間通信の初期化。NCCL は NVIDIA の集合通信ライブラリ
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().to(local_rank)
model = DDP(model, device_ids=[local_rank])      # モデルを DDP で包む

sampler = DistributedSampler(dataset)            # データを rank ごとに重複なく分割
loader = DataLoader(dataset, sampler=sampler, batch_size=64)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)                     # エポックごとにシャッフルを変える (これを忘れると毎エポック同じ順)
    for batch in loader:
        loss = model(batch)                      # forward
        loss.backward()                          # backward + 裏で all-reduce
        optimizer.step()
        optimizer.zero_grad()
```

起動は `torchrun --nproc_per_node=4 train.py` のように行い、torchrun が 4 プロセスを立ち上げて `RANK` / `LOCAL_RANK` / `WORLD_SIZE` を各プロセスの環境変数に注入します。クラスタ上では、ジョブスケジューラ (Kubernetes 上の分散学習オペレータなど) がこの torchrun 相当の役割を担い、各 Pod に rank を割り当てて DDP プロセス群を構成するのが一般的です。

実務で踏みやすい落とし穴がいくつかあります。`DistributedSampler` を使わず通常の loader を使うと全 GPU が同じデータを見て勾配が冗長になり高速化しません。BatchNorm は各 GPU が自分の local batch だけで統計を取るので、batch が小さいと統計がぶれます (これを嫌う場合は SyncBatchNorm で統計を all-reduce します)。また各 GPU の入力長が不揃いだと、ある GPU だけ余分に backward を回して all-reduce の回数がズレ、ハングする (永久に待ち合う) ことがあります。

### 手法の系譜と主要論文

データ並列の歴史は「同期 SGD をどう効率よく分散するか」の歴史です。

- Dean ら, "Large Scale Distributed Deep Networks" (NeurIPS 2012, Google)。DistBelief。parameter server (重みを集中管理するサーバ) と非同期 SGD を提案。各 worker が勝手なタイミングで勾配を送る方式で、初期の大規模分散学習を支えましたが、勾配の遅延 (stale gradient = 古い重みで計算された勾配が遅れて届くこと) で精度が不安定になる問題がありました。

- Goyal, Dollár, Girshick ら, "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Facebook, 2017, arXiv:1706.02677)。データ並列で巨大 batch (8192) を使う際の学習率調整 (線形スケーリング則 + warmup) を確立し、ImageNet 学習を 256 GPU・1 時間で完了させました。「batch を増やしただけでは精度が出ない、学習率を正しく合わせろ」という、DDP 実用の必読の教訓を与えた論文です。

- Sergeev & Del Balso, "Horovod: fast and easy distributed deep learning in TensorFlow" (2018, Uber, arXiv:1802.05799)。ring all-reduce を MPI ベースで実装し、parameter server より高効率な all-reduce 方式のデータ並列を広めました。少ないコード変更で分散化できる API 設計も含め、PyTorch DDP の通信設計の源流の 1 つです。

- Li, Zhao, Varma ら, "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (VLDB 2020, Facebook, arXiv:2006.15704)。本トピックの中心となる論文で、PyTorch `DistributedDataParallel` の設計と最適化を解説。中核の工夫は (1) gradient bucketing、(2) 計算と通信のオーバーラップ、(3) 勾配が出ない (forward で使われなかった) パラメータのスキップ。near-linear scaling を多くのモデルで達成したことを実測で示しました。

その後、データ並列はメモリの冗長性を解く方向へ進化します ([ZeRO](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、[FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。DDP は「モデルが 1 台に収まる範囲で、最も単純で速い」基本形として今も標準であり続けています。

### 論文の実験結果 (定量データ)

PyTorch DDP 論文 (VLDB 2020) の主要な実測は次の通りです。評価には ResNet-50 (画像分類) と BERT (自然言語) を使い、指標は scaling efficiency (台数に対する高速化の比率) と 1 秒あたりの処理サンプル数です。

- ResNet-50 と BERT を最大 256 GPU まで増やしたとき、ideal (台数に完全比例) に対しておよそ 90% 前後の scaling efficiency を維持。たとえば 256 GPU で理想の約 9 割の速度が出る、という意味で、これは通信を計算に隠せていることを示します。
- bucketing の効果: バケツサイズを適切に (報告では数十 MB) 取ると、パラメータごと通信に比べてスループットが大きく向上。バケツが小さすぎると通信回数が増えて遅く、大きすぎると最初のバケツが埋まるまで通信を始められずオーバーラップの粒度が粗くなって遅くなる、という凸型 (途中に最適点がある山型) の関係が示されました。
- オーバーラップの効果: backward と all-reduce を重ねることで、重ねない場合に比べて 1 イテレーションの時間が短縮されることをプロファイルで実証。論文の例では、通信を計算の裏に隠すことで、通信が見かけ上ほぼ消える挙動が示されました。

Goyal ら (2017) のアブレーション的な知見も重要です。線形スケーリング則だけで warmup を入れないと、大 batch (8192) では学習初期に発散し精度が出ません。warmup を入れると、batch 256 の小規模学習とほぼ同等の top-1 accuracy (ImageNet で約 76%) を、batch 8192・256 GPU で再現できました。さらに batch を 8192 より大きくすると warmup を入れても精度が劣化し始め、純粋な線形スケーリングには限界があることも示されています。「どの要素を抜くと壊れるか、どこまでなら通用するか」が明確に示された例で、DDP を実用に乗せる際の前提知識になっています。

指標の意味の補足: scaling efficiency は「N 台にしたとき本当に N 倍速くなったか」を測る比率で、1.0 (100%) が理想です。これが落ちる主因は通信オーバーヘッドと負荷の不均衡で、効率の数値を見れば「台数を増やす価値がまだあるか (これ以上増やしても頭打ちか)」を判断できます。top-1 accuracy は「予測 1 位が正解だった割合」で、ImageNet 1000 クラス分類の標準指標です。0.x% の差でも順位やモデルの優劣を論じるのが慣例なので、76% を維持できたことは「分散化で精度を損なわなかった」ことの強い証拠になります。

### メリット・トレードオフ・限界

メリット
- 実装が単純。既存の単一 GPU コードに数行足すだけで動く
- GPU 台数にほぼ比例した高速化 (near-linear scaling) が得られやすい
- PyTorch 標準で実績が豊富。ドキュメントとツール (torchrun、分散学習オペレータ) が充実

トレードオフ・限界
- モデル・勾配・optimizer 状態を各 GPU にまるごと複製するため、メモリ削減はゼロ。1 台に入らない巨大モデルには使えない (その場合は [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) / [ZeRO](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))
- 通信帯域 (GPU 間・ノード間ネットワーク) がボトルネックになると、台数を増やしても速度が頭打ちになる。特にノードをまたぐ multi-node では帯域が GPU 内より桁違いに細く、scaling 効率が落ちる
- 実効 batch が増えるため学習率の再調整が必須 (線形スケーリング則 + warmup)。怠ると精度が出ない、あるいは発散する
- 各 GPU の処理時間がバラつく (データの長さが不揃い、一部ノードが遅いなど) と、最も遅い GPU に全員が同期で待たされる (stragglers 問題)

研究上の未解決課題: 同期 all-reduce は「最も遅い 1 台」に律速されるため、超大規模 (数千 GPU) では straggler や通信競合が深刻化します。勾配圧縮 (量子化やスパース化で通信量を減らす)、非同期・準同期手法、通信トポロジ最適化などが研究され続けていますが、精度を保ちつつ通信を減らすのは依然として難問です。また「大 batch で精度を落とさず収束させる」最適化アルゴリズム (LARS/LAMB を超える設計) も、超大規模学習の継続的なテーマです。

### 発展トピック・研究の最前線
- 勾配圧縮: PowerSGD (Vogels ら, NeurIPS 2019) のように勾配を低ランク近似して通信量を削減する手法。精度をほぼ保ったまま通信を数分の 1 に減らせるが、圧縮の計算コストとのバランスが課題。1-bit SGD や top-k sparsification なども同系統
- 大 batch 最適化: LARS (You ら, 2017)、LAMB (You ら, ICLR 2020)。層ごとに勾配のノルムに応じて学習率を調整し、BERT を数千 GPU・極端な大 batch で短時間学習することを可能にした
- ハイブリッド並列: データ並列・テンソル並列・パイプライン並列を組み合わせる 3D parallelism (Megatron-LM など)。DDP 単独では足りない超大規模モデルで、各並列軸を使い分ける
- 通信最適化: NCCL のトポロジ認識、NVLink / InfiniBand などの高速インターコネクト活用、gradient accumulation (複数ステップの勾配をためてから 1 回通信し、通信頻度と実効 batch を同時に調整する) との組み合わせ

### さらに学ぶための関連トピック
- [FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Kubeflow Training Operator (PyTorchJob)

### ひとことで言うと
Kubeflow Training Operator は、機械学習の学習ジョブを Kubernetes 上で「宣言的に」動かすための拡張機能です。具体的には「PyTorchJob を1つ作って」とだけ宣言すれば、分散学習に必要な煩雑な準備 (複数のワーカーをどう起動し、どう通信させ、失敗したらどう再起動するか) を裏側で全部面倒見てくれます。利用者は学習コードを書き、ワーカー数と資源を指定するだけで、複数 GPU・複数サーバにまたがる学習を起動できます。

### 直感的な理解
1枚の GPU では時間がかかりすぎる、あるいはモデルがメモリに乗りきらない学習を、複数の GPU に分けて並列にやりたい、という場面を考えます。これを素朴にやろうとすると、各 GPU で動くプロセス (ワーカー) に「お前は何番目だ」「全部で何人いる」「リーダーのアドレスはここだ」を矛盾なく教え込む必要があります。サーバの IP は起動するまで分からないので、手で設定ファイルを書くのは現実的ではありません。さらに、1つのワーカーが落ちたら全体を整合性を保って立て直し、全員が揃うまで学習を始めない、といった気配りも要ります。

Training Operator は、この「分散学習のお膳立て係」です。「オペレータ (Operator)」という Kubernetes 用語は、「ある種類のリソースの面倒を見続ける常駐プログラム」を指します。人間の運用担当者 (operator) が手作業でやっていた配線・監視・再起動を、プログラムが肩代わりする、というニュアンスです。利用者は宣言を1枚書くだけで、配線は Operator がやってくれます。

### 基礎: 前提となる概念
- PyTorch: ディープラーニング (深い層のニューラルネットの学習) で最も広く使われる Python ライブラリ。モデルの定義、勾配の自動計算 (autograd)、最適化を担います。
- ニューラルネットの学習の基本ループ: データを入力してモデルの予測を出し (forward)、予測と正解のズレ (loss) を測り、loss を小さくする方向 (勾配) を逆向きに計算し (backward)、その方向に少しパラメータを動かす (optimizer step)。これを大量のデータで何百万回も繰り返します。
- 分散学習の主要な型:
  - データ並列 (data parallel): 全ワーカーが同じモデルのコピーを持ち、データを分担して別々のミニバッチで勾配を計算し、勾配を平均して同期する。最も一般的。
  - モデル並列 / テンソル並列 (tensor parallel): 1つの巨大なモデルの各層の行列を複数 GPU に分割して載せる。モデルが1枚に乗らないときに使う。
  - パイプライン並列 (pipeline parallel): モデルの層をステージに分け、流れ作業のように複数 GPU で処理する。
- 分散実行に必要な情報 (PyTorch の `torch.distributed` の場合):
  - RANK: 自分が何番目のワーカーか (0 始まり)。
  - WORLD_SIZE: 全部で何ワーカーいるか。
  - MASTER_ADDR / MASTER_PORT: 代表ワーカー (rank 0) のアドレスとポート。全員がここに集合して接続を確立する。
- all-reduce: 全ワーカーの勾配を集めて平均し、結果を全員に配り戻す集団通信 (collective communication)。データ並列学習の同期の核です。

これらの環境変数を、起動する全 Pod に正しく・矛盾なく注入することが分散学習の最初の関門です。Training Operator はここを自動化します。

### 仕組みを詳しく
#### CRD = Kubernetes に学習専用の「型」を足す
Training Operator は Kubernetes に `PyTorchJob` という CRD (Custom Resource Definition) を追加します。標準の Kubernetes には Pod や Deployment という型しかありませんが、CRD を入れると「PyTorchJob」という機械学習専用の型を `kubectl apply` で扱えるようになります。歴史的には `TFJob` (TensorFlow 用)、`MPIJob` (MPI / Horovod 用)、`XGBoostJob`、`PaddleJob` などフレームワークごとに型が用意されてきました。Operator は「宣言された CRD の状態」と「実際に起きている Pod の状態」を絶えず突き合わせ、差があれば Pod を作る・消す・再起動するという制御ループ (reconciliation loop) を回し続けます。

#### PyTorchJob の構造
PyTorchJob は `pytorchReplicaSpecs` という欄に、役割ごとのワーカー構成を書きます。

```yaml
apiVersion: kubeflow.org/v1
kind: PyTorchJob
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1            ← 代表ワーカー (rank 0)
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: <学習コードを焼いたコンテナイメージ>
              resources:
                limits: { nvidia.com/gpu: "1" }
    Worker:
      replicas: 3            ← その他のワーカー (rank 1,2,3)
      ...
```

ポイントを噛み砕きます。
- `Master` (代表) と `Worker` (その他) に分けて、それぞれ何個 (`replicas`) 起動するかを指定します。GPU 1枚なら `Master: 1` のみ、4 GPU なら `Master: 1 + Worker: 3` のように書きます。
- `nvidia.com/gpu` で各 Pod が要求する GPU 枚数を指定します。
- `restartPolicy` で、ワーカーが落ちたときの再起動方針を決めます。

#### 環境変数の自動配布 (分散学習の本体)
`Master: 1`, `Worker: 3` の構成では、Operator は4つの Pod を起動し、各 Pod に次の環境変数を自動注入します。

```
Pod 0 (Master): RANK=0  WORLD_SIZE=4  MASTER_ADDR=<masterのDNS名>  MASTER_PORT=23456
Pod 1 (Worker): RANK=1  WORLD_SIZE=4  MASTER_ADDR=<同上>           MASTER_PORT=23456
Pod 2 (Worker): RANK=2  WORLD_SIZE=4  MASTER_ADDR=<同上>           MASTER_PORT=23456
Pod 3 (Worker): RANK=3  WORLD_SIZE=4  MASTER_ADDR=<同上>           MASTER_PORT=23456
```

代表 Pod 用に Kubernetes の Service (固定の DNS 名) を作り、その名前を全 Pod の MASTER_ADDR に流し込むことで、起動前には分からない IP を経由せず安定して集合できます。学習コード側は `torch.distributed.init_process_group()` を呼ぶだけで、これらの環境変数を読んでお互いに接続します。各ワーカーは異なるデータの断片で勾配を計算し、`all-reduce` で勾配を平均して同期します。役割分担は明確で、Operator は「環境変数の配布」と「Pod の生成・監視・再起動」を担当し、通信そのものは PyTorch (内部では NCCL という GPU 間集団通信ライブラリ) が行います。

#### ライフサイクル
```
kubectl apply -f pytorchjob.yaml
   │
   ▼
Operator が PyTorchJob を検知し、ReplicaSpec の数だけ Pod を作成
(suspend=true なら外部のキュー管理が許可するまで停止。許可後に Pod 起動)
   │
   ▼
各 Pod に RANK / WORLD_SIZE / MASTER_ADDR を注入 + 代表用 Service を作成
   │
   ▼
学習実行 (失敗した Pod は restartPolicy に従って再起動)
   │
   ▼
全 Pod 完了 → PyTorchJob を Succeeded に → TTL 経過後に自動削除
```

#### キュー管理との連携 (suspend)
PyTorchJob は `runPolicy.suspend: true` を持てます。これが true の間、Operator は Pod を起動しません。外部のクォータ・優先度管理 (キューイングシステム、例えば [Kueue (Job Queueing)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)) が「いま実行してよい」と判断して suspend を解除したら、初めて起動します。これにより「学習ジョブの抽象化 (Training Operator)」と「資源配分の門番 (キュー管理)」をきれいに分離できます。Operator 単体には「全ワーカー揃うまで起動を待つ」gang scheduling の能力は弱いため、この分離が gang scheduling を外部に委ねる素直な手段になります。

### 手法の系譜と主要論文
Training Operator 自体は Kubeflow プロジェクト (2017年に Google が中心となって開始した「Kubernetes 上で ML を回すツール群」) の一部で、論文発表されたシステムではありません。しかし支える分散学習の技術には明確な研究史があります。

- データ並列分散学習の起源: Dean et al. "Large Scale Distributed Deep Networks" (NeurIPS 2012, Google) は DistBelief というシステムで、parameter server (パラメータを集約・配布する中央サーバ) を介して多数のワーカーが勾配をやり取りする分散学習を提案しました。動機は「1台では学習が遅すぎる・大きすぎる」こと、効果は「ワーカー数に応じた高速化」、トレードオフは「通信コストと、非同期更新による収束の不安定さ」でした。Li et al. "Scaling Distributed ML with the Parameter Server" (OSDI 2014) はこの方式を汎用フレームワークとして体系化しました。

- 大規模ミニバッチの安定化: Goyal et al. "Accurate, Large Minibatch SGD" (arXiv:1706.02677, 2017, Facebook AI Research) は、ワーカーを増やすと実効バッチサイズが膨らんで学習が不安定になる問題に対し、学習率の線形スケーリング (バッチを k 倍にしたら学習率も k 倍) とウォームアップ (最初の数エポックは学習率を徐々に上げる) を提案しました。これは「ワーカーを増やせばよいわけではなく、最適化の工夫が要る」ことを示し、台数を増やすときに必ず意識すべき限界を表しています。You et al. の LARS / LAMB (arXiv:1708.03888, arXiv:1904.00962) はさらに大きなバッチ (数万) でも安定させる層ごとの適応学習率を提案し、超大規模バッチ学習を可能にしました。

- all-reduce 通信の効率化: Sergeev & Del Balso "Horovod" (arXiv:1802.05799, 2018, Uber) は、parameter server 方式に代えて ring-allreduce (ワーカーをリング状につなぎ、帯域を均等に使う集団通信) で勾配同期を効率化しました。PyTorch の `DistributedDataParallel` も同系統の all-reduce を使い、勾配計算と通信をオーバーラップさせて待ち時間を隠します。Training Operator はこの通信を PyTorch / NCCL に任せ、自身は Pod とネットワークのお膳立てに徹します。

- メモリ効率化と超大規模化: Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020, arXiv:1910.02054, Microsoft) は、最適化状態 (optimizer state)・勾配・パラメータをワーカー間で分割保持することで、データ並列のメモリ消費を劇的に減らしました。Narayanan et al. の Megatron-LM (SC 2021) はテンソル並列・パイプライン並列・データ並列を組み合わせた 3D 並列を確立しました。これにより数千億〜兆パラメータ級の学習が現実的になりました。Training Operator はこうした大規模分散の「実行基盤」を提供し、ZeRO 等の手法はその上で動きます。

- 後継への展開: Kubeflow コミュニティは v1 系の Training Operator (PyTorchJob 等の個別 CRD) から、より統一的な新世代 API (TrainJob / Trainer) へと発展させています。仕様が移行期にあるため、バージョン追従が運用上の論点になります。

### 論文の実験結果 (定量データ)
- ImageNet を1時間で学習 (Goyal et al. 2017): ResNet-50 を ImageNet (約128万枚の画像分類データセット、1000クラス) で学習する際、256個の GPU を使い実効バッチサイズ 8192 まで拡大しても、線形スケーリング則とウォームアップを使えば、単一ノード学習とほぼ同じ精度 (top-1 accuracy で誤差1%以内) を保ったまま、学習時間を従来の何日かから約1時間に短縮できたと報告されています。指標の top-1 accuracy は「最も確信度の高い予測クラスが正解と一致した割合」で、ImageNet では当時 76% 前後が標準的でした。ここで重要なのは「台数を64倍にしても精度を落とさなかった」点で、ウォームアップを抜くアブレーションでは精度が明確に劣化することも示され、最適化の工夫の必要性が裏付けられました。バッチサイズをさらに大きく (例えば 16384 以上に) すると線形スケーリングだけでは精度が崩れ始め、ここに素朴な台数増加の限界が現れます。

- Horovod のスケーリング効率 (Sergeev & Del Balso 2018): 標準的な分散 TensorFlow (parameter server 方式) では GPU を増やしてもスケーリング効率 (理想的な台数倍速に対する実測の割合) が頭打ちになりますが、ring-allreduce では Inception V3 や ResNet-101 で128 GPU 規模でもおよそ 90% 前後のスケーリング効率を達成したと報告されています。スケーリング効率が高いほど「通信で時間を食わず、増やした台数がそのまま速度に反映される」ことを意味します。

- ZeRO のメモリ削減 (Rajbhandari et al. 2020): ZeRO の各段階 (Stage 1: 最適化状態の分割 → Stage 2: 勾配の分割 → Stage 3: パラメータの分割) を入れるほど、ワーカーあたりのメモリ消費が下がります。報告では Stage 1+2 で従来比およそ8倍のモデルを同じハードウェアに載せられ、全段階を組み合わせて1兆パラメータ級の学習が視野に入ること、かつ400 GPU 規模でほぼ線形のスループット向上を示しました。これは「どの要素を分割するか」というアブレーションそのものが性能曲線になっており、分散の設計選択が直接メモリと到達可能な規模を決めることを示しています。

これらの数値はいずれも「分散にすればただ速くなるのではなく、最適化と通信とメモリの工夫が伴って初めてスケールする」ことを物語っており、Training Operator はその工夫を載せる土台にすぎない、という位置づけを理解するのが研究上重要です。

### メリット・トレードオフ・限界
メリット:
- 分散学習の煩雑な環境変数設定 (RANK, MASTER_ADDR 等) を自動化し、宣言1枚で起動できる。
- フレームワークごとの CRD (PyTorchJob, TFJob 等) で宣言的に書け、再現性と可読性が高い。
- 失敗時の再起動・終了後の自動掃除 (TTL) など運用を肩代わりする。
- suspend を介して外部のキュー管理にきれいに乗る。

トレードオフ・限界:
- Kubernetes と CRD の知識が前提で、ローカルで `python train.py` するより導入の敷居が高い。
- 単一 GPU の用途では分散自動配布の旨味が薄く、コンテナ起動などのオーバーヘッドの方が目立ちうる。
- gang scheduling (全ワーカー揃ってから起動) は Operator 単体では弱く、キュー管理と組み合わせる必要がある。
- ワーカー数を増やしても、最適化 (学習率) と通信 (帯域・トポロジー) を整えないと精度劣化や速度頭打ちが起きる。これは Operator では解決できない、上で動く学習設計の責任。
- 静的な構成 (replicas 固定) が基本で、学習中のワーカー増減 (弾力性) や故障からの高速復旧は Operator だけでは完結せず、TorchElastic 等の機構が必要。
- API が移行期 (v1 系 → 新 Trainer/TrainJob) にあり、バージョン追従の手間がある。

### 発展トピック・研究の最前線
- Elastic training: ワーカー数を学習中に増減させても継続できる仕組み (PyTorch の TorchElastic / `torchrun`)。スポットインスタンスでワーカーが突然落ちても、生き残ったワーカーで rendezvous を再構成して学習を続けられます。
- フォールトトレランス: 大規模・長時間学習では必ず故障が起きるため、頻繁なチェックポイントと高速復旧 (例: in-memory checkpoint、非同期チェックポイント) の研究が活発です。数千 GPU 規模では平均故障間隔が数時間〜数十分にまで縮むため死活問題になります。
- 3D 並列とハイブリッド: データ並列 × テンソル並列 × パイプライン並列を組み合わせた超大規模学習 (Megatron-LM, Narayanan et al. SC 2021) が標準化しつつあり、これを Kubernetes 上で宣言的に組む方向が探られています。
- スケジューラとの統合: トポロジーアウェア配置 (高速ネットワークでつながった GPU に固める) や gang scheduling を、キュー管理システムと密に連携させる流れがあります。

### さらに学ぶための関連トピック
- [Kueue (Job Queueing)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)
- [Flyte (Pipeline Orchestration)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)
- [MLflow (Tracking + Model Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)


## AdamW (Decoupled Weight Decay)

### ひとことで言うと
AdamW(アダム・ダブリュー)は、Adam の「weight decay(重み減衰)」という正則化のかけ方が実は間違っていた、という問題を直した改良版オプティマイザです。weight decay とは、重みが大きくなりすぎないように毎ステップ少しずつ 0 へ縮めて過学習を防ぐ手法です。AdamW は、この「重みを縮める処理」を勾配の計算から切り離して(decouple、デカップル=分離して)、純粋に重みだけに直接かけることで、正則化が設計通りに効くようにしました。Transformer や ViT(Vision Transformer、画像を Transformer で扱うモデル)、大規模言語モデルを学習するときの事実上の標準オプティマイザです。

### 直感的な理解
ニューラルネットを学習させると、重みの値がどんどん大きくなっていくことがあります。重みが大きいモデルは、入力のわずかな違いに敏感に反応し、訓練データの細かいノイズまで覚え込みやすく、未知データで性能が落ちる「過学習(overfitting)」を起こします。これを抑える定番の手段が weight decay で、毎ステップ重みをほんの少しだけ 0 へ引き戻して「大きくなりすぎないよう手綱を締める」操作です。

ここで素朴な疑問が生まれます。「Adam を使うとき、この手綱の締め方はどうすればいいのか?」 長らく多くの実装は、SGD と同じやり方(損失関数に重みの二乗を足す=L2正則化)で済ませていました。ところが Adam はパラメータごとに歩幅を割り算で調整する仕組み(勾配を `√v` で割る)を持つため、この素朴なやり方だと手綱の締め方まで割り算でゆがめられ、重みによって締めたり緩めたりがバラバラになってしまいます。AdamW は「手綱は手綱として、割り算とは別に、まっすぐ一律にかける」ことで、本来意図した正則化を取り戻します。たったこれだけのことで、汎化性能(未知データでの性能)とハイパーパラメータの扱いやすさが目に見えて改善しました。

### 基礎: 前提となる概念
- 正則化(regularization): モデルが複雑になりすぎる(過学習する)のを防ぐ仕掛けの総称。weight decay はその一種。
- weight decay(重み減衰): 毎ステップ重みを `w ← w − η·λ·w` のように少しずつ 0 へ縮める操作。`λ`(ラムダ)は縮める強さを決める係数(典型的には 0.01〜0.1)。
- L2正則化(L2 regularization): 損失関数に重みの二乗和 `(λ/2)·‖w‖²` を足す方法。これを微分すると勾配に `λ·w` という項が増える。`‖w‖²` は全重みを二乗して足したもの。
- なぜ「同じはず」だったのか: 純粋な SGD では、L2正則化(損失に二乗を足す)と weight decay(重みを直接縮める)は数学的に完全に一致します。だから長年区別されずに同義語として使われてきました。AdamW の本質的な貢献は「Adam ではこの2つが一致しない」と明確に示した点にあります。
- 適応的スケーリング(adaptive scaling): Adam が勾配を `√v`(勾配二乗の移動平均の平方根)で割る操作。これが正則化をゆがめる元凶になる。前提として [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) を理解しておくと良いです。

### 仕組みを詳しく
Adam では、weight decay を「勾配に `λ·w` を足す」L2方式で入れると、その `λ·w` まで `√v` で割られてしまいます。何が起きるか。勾配が大きくよく動く重み(`√v` が大きい)は weight decay の効きが弱まり、逆に勾配が小さくあまり動かない重み(`√v` が小さい)には weight decay が強く効きすぎます。つまり「重みを一律に λ の割合で縮める」という本来の意図が崩れ、パラメータごとにバラバラの正則化になります。皮肉なことに、本来は最も縮めて抑えたい(よく動いて大きくなりがちな)重みほど、weight decay が弱まってしまうのです。

AdamW の解決策はシンプルで、weight decay を勾配にではなく、更新の最後で重みに直接かけます。両者を並べます。

通常の Adam(L2正則化を勾配に混ぜる方式):
```
g ← g + λ · w                      ← weight decay を勾配に足してしまう(ここが問題)
m ← β1·m + (1−β1)·g
v ← β2·v + (1−β2)·g²
w ← w − η · m̂ / (√v̂ + ε)           ← λ·w も √v̂ で割られて歪む
```

AdamW(weight decay を分離):
```
m ← β1·m + (1−β1)·g                ← 勾配 g には weight decay を混ぜない
v ← β2·v + (1−β2)·g²
w ← w − η · m̂ / (√v̂ + ε) − η · λ · w   ← 最後に weight decay を別項として直接かける
```

違いは `λ·w` を勾配 `g` に混ぜるか、更新式の最後に独立した項として足すか、だけです。AdamW では `λ·w` が `√v̂` で割られないので、すべての重みが学習率 η に応じて一律 `η·λ` の割合で縮みます。これが本来意図した weight decay の挙動です。

#### 数値例
`η = 0.001`、`λ = 0.01` とします。AdamW では勾配の大小に関係なく、すべての重みが毎ステップ `η·λ·w = 0.001·0.01·w = 0.00001·w` ぶん 0 へ縮みます。重み `w=2.0` なら毎回 `0.00002` 縮む、という一律の正則化です。一方の通常 Adam だと、`√v` が大きい重み(例えば `√v=10`)では縮みが 10 分の 1 になり、`√v` が小さい重み(例えば `√v=0.1`)では 10 倍に効きすぎ、重みによって縮み方が 100 倍も変わってしまいます。

#### λ と η の解きほぐし(decoupling の効能)
L2方式では、`η` を変えると正則化の実効的な強さも一緒に変わってしまい、2つが絡み合っていました。学習率を下げると正則化も弱まる、という連動が起きるのです。これは、学習率スケジュール(学習が進むにつれ η を下げていく手順)を使うと、終盤で正則化が勝手に弱まるという厄介な副作用を生みます。AdamW でも `η·λ` が weight decay 項なので確かに `η` を変えれば正則化も変わりますが、`λ` を独立に調整できるため、ハイパーパラメータ探索の格子(grid、η と λ の組み合わせを総当たりする表)を `η` 軸と `λ` 軸で素直に切れるようになります。Loshchilov & Hutter はこの「探索空間がほぼ直交する(2つの軸が独立に効く)」ことを大きな実用メリットとして強調しました。

#### メモリ・計算
AdamW のメモリと計算量は Adam と同じです(`m` と `v` の2セット、追加メモリは重みの2倍)。違うのは weight decay を足す場所だけなので、コストはほぼゼロで実装できます。多くの深層学習フレームワークが標準で AdamW を提供しています。「Adam に weight_decay 引数を渡す」古い実装は内部で L2方式になっていることがあるため、正しい正則化を効かせたいなら明示的に AdamW を選ぶ必要があります。この「同じ名前の引数なのに中身が違う」落とし穴は、再現性の問題としてもしばしば指摘されてきました。

### 手法の系譜と主要論文
- Adam — Kingma & Ba (ICLR 2015)。AdamW の更新本体(`m`, `v`, バイアス補正)はそのまま Adam を踏襲します。[Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。
- L2正則化と weight decay の歴史的混同 — Hanson & Pratt (1988) ら初期のニューラルネット研究以来、両者は同義として扱われてきました。SGD では実際に一致するため、誰も区別する必要を感じなかったのが背景です。
- Decoupled Weight Decay — Loshchilov & Hutter。初出は arXiv:1711.05101(2017)、改訂を経て ICLR 2019 に採択。Adam の weight decay を分離型に変更し、SGD でも cosine 学習率スケジュール下では L2 と weight decay の挙動が異なることを指摘しました。同論文は SGDW(SGD の分離型 weight decay 版)も提案しています。
- 普及の歴史: AdamW はその後 BERT(Devlin et al., NAACL 2019)、GPT 系、ViT(Dosovitskiy et al., ICLR 2021)、画像系の DeiT(Touvron et al., 2021)・Swin Transformer(Liu et al., 2021)など Transformer/近代アーキテクチャの学習でほぼ標準採用され、「Transformer を学習するならまず AdamW」という慣習を作りました。
- 後続: AdamW は AMSGrad(収束修正)や warmup(RAdam)、メモリ削減(Adafactor, 8-bit Adam)などと組み合わせて使われます。更新本体を軽くする方向では [Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) が AdamW の分離型 weight decay を踏襲しつつ更新式を sign+単一モーメントに置き換えました。

### 論文の実験結果(定量データ)
Loshchilov & Hutter (ICLR 2019) の主な定量的主張は次の通りです。

- CIFAR-10 の画像分類(ResNet 系)で、Adam(L2方式)に比べて AdamW はテスト誤差が改善し、よく調整した SGD+Momentum との差を縮めました。具体的には、Adam が SGD に明確に劣っていたテスト精度のギャップ([Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) でも触れた Wilson et al. 2017 の問題)を、weight decay を分離するだけでほぼ埋められると報告。ここでの「テスト誤差」は未知データでの誤分類率で、低いほど良い指標です。
- ハイパーパラメータの解きほぐし: `η` と `λ` を変えたときのテスト精度の等高線(ヒートマップ)を示し、L2方式では性能の良い領域が斜めに伸びて(η と λ が連動して)探索しにくいのに対し、分離型では良い領域が軸に沿った楕円としてまとまり、`η` と `λ` を独立に探索できることを可視化しました。これは厳密な数値比較というより「探索のしやすさ」を示す重要な結果で、実務での調整コストを下げる効果があります。
- ImageNet32(ImageNet を 32×32 に縮小したデータセット)など複数設定でも分離型が一貫して同等以上の汎化を示し、cosine 学習率スケジュール(学習率を周期的に下げる)や warm restarts(学習率を周期的に再加熱する)との相性が良いと報告しました。

アブレーション的に言えば、変更点は「weight decay を足す場所」のただ1か所です。それ以外(`m`, `v`, バイアス補正、学習率スケジュール)は Adam と完全に同一。にもかかわらず汎化が改善する、という事実が「場所が本質的に重要だった」ことの証拠になっています。逆に言えば、weight decay を 0 にすれば AdamW と Adam は完全に一致するので、効果はすべて weight decay の効き方の違いから来ています。

実務上の重要な注意点として、後続のレシピ研究では「weight decay をかけるべきでないパラメータ」を除外する設定が定番化しました。具体的には LayerNorm / BatchNorm のスケール(γ)・バイアス(β)、各層のバイアス項、および埋め込み層の一部です。これらは「重みを 0 へ縮めること」が意味をなさない、あるいは正規化の働きを壊すパラメータです(例えば LayerNorm のスケールを 0 に縮めると出力がつぶれる)。ViT・BERT の標準レシピでは、これらを decay 対象から外す「パラメータグループ分け(parameter grouping)」が行われます。この設定を怠ると、ViT 系で数ポイント精度が落ちることが経験的に報告されています。

### メリット・トレードオフ・限界
メリット
- weight decay が設計通り一律に効くため正則化が予測しやすく、汎化性能が改善する。
- `λ` と `η` をほぼ独立に調整でき、ハイパーパラメータ探索が楽。学習率スケジュールを使っても正則化の強さが勝手に変わらない。
- Adam と同じ計算量・メモリで、変更は weight decay の場所だけ。導入コストが低い。
- Transformer/ViT 系で実績豊富。デファクトスタンダード。

トレードオフ・限界
- Adam 同様 `m` と `v` の2セットで追加メモリが重みの2倍([Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) が削りにいく点)。
- `λ` と `η` の調整は依然必要で、最適値はモデル規模・データセット・バッチサイズで変わる。
- Adam の収束理論上の課題(AMSGrad 系の指摘)は weight decay の修正とは別問題として残る。
- LayerNorm のスケール・バイアスや埋め込みを decay 対象から除外する設定が必要。怠ると性能低下。

未解決の課題として、「分離型 weight decay がなぜ sharp minima(尖った極小値=過学習しやすい)を避けて汎化を助けるのか」の理論的説明はまだ発展途上です。実務では効くことが確立しているものの、最適 `λ` をモデル規模から事前に予測する一般則はまだありません。

### 発展トピック・研究の最前線
- スケール則と weight decay: 大規模言語モデルでは、weight decay と学習率スケジュールの相互作用が損失曲線に与える影響(損失地形を緩やかな川筋に見立てる「river valley」的な解釈など)が研究されており、weight decay を学習率に応じて調整するレシピが議論されています。weight decay が実質的に重みノルムの定常状態(平衡点)を決めるという見方も提示されています。
- 独立 weight decay(independent weight decay): `η` に依存しない形で `w ← w − λ·w` とする変種もあり、スケジューラとの組み合わせで挙動が変わります。実装によって `η` を掛けるか掛けないかが異なるため、論文の数値を再現する際は注意が必要です。
- 軽量化との両立: Adafactor + 分離 weight decay、8-bit AdamW など、メモリ削減と正しい正則化を両立する実装が巨大モデル学習で使われています。低ランク勾配射影の GaLore なども AdamW の更新を低メモリで近似する方向の研究です。
- スケール不変性の理解: BatchNorm/LayerNorm を含むネットでは、正規化層の前の重みのスケールは出力に影響しない(スケール不変)ため、weight decay は実質的に「実効学習率を調整するノブ」として働くという解釈(Zhang et al., van Laarhoven ら)があり、weight decay の役割を再定義する研究が続いています。

### さらに学ぶための関連トピック
- [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Checkpoint / Resume(チェックポイントと再開)

### ひとことで言うと
学習の途中経過(モデルの重みや、学習を進めるための内部状態)をファイルに保存しておき、学習が止まっても続きから再開できるようにする仕組みです。ゲームのセーブデータと同じ発想で、長時間の学習や、いつ中断されるか分からない安価なマシンで学習するときに必須になります。

### 直感的な理解
機械学習の「学習」とは、モデルの中にある大量の数値(これを「重み(weight、パラメータ)」と呼びます。モデルの予測を決めるツマミのようなもの)を、少しずつ正しい値へ調整していく作業です。大きなモデルでは、この調整を何時間も、ときに何日も繰り返します。

ここで困るのが、途中でプログラムが止まる場合です。原因はいくつもあります。

- マシンが突然落ちる(電源・ハードウェア障害)。大規模クラスタでは、台数が多いほど「どこかが壊れる」確率が上がり、故障は「起きるか」ではなく「いつ起きるか」の問題になります。
- GPU のメモリ不足(VRAM 不足、OOM = Out Of Memory)で異常終了する。
- スポットインスタンス(正規料金の数分の一で借りられる代わりに、クラウド事業者の都合で多くは2分前の予告つきで強制回収される計算資源)を使っていて、回収される。
- 自分でプログラムを止めて設定を変えたい。

何も対策しないと、止まった瞬間に「それまで何時間もかけて調整した重み」がメモリ上から消え、次回はゼロからやり直しです。10時間学習して9時間目で落ちたら、9時間が無駄になります。これを防ぐのが Checkpoint(チェックポイント)です。一定間隔で「今の学習状態」をディスクや S3 などに保存しておけば、落ちても直前の保存地点から続き(Resume)を始められます。

### 基礎: 前提となる概念
理解の前提となる用語を噛み砕きます。

- 重み(model weights / parameters): モデル本体の数値。これが学習で調整される対象です。
- 勾配(gradient): 「重みをどちらにどれだけ動かせば誤差が減るか」を示す方向と大きさ。各ステップで計算され、重み更新に使われます。
- オプティマイザ(optimizer): 勾配を受け取って実際に重みを更新するアルゴリズム。代表例が Adam で、各重みについて「過去の勾配の移動平均(1次モーメント)」と「過去の勾配の2乗の移動平均(2次モーメント)」という履歴を内部に保持します。
- エポック(epoch)とステップ(step): エポックはデータ全体を1周することを1と数える単位。ステップはミニバッチ(数枚〜数十枚をまとめたかたまり)を1回処理することを1と数える単位です。
- 検証データ(validation set): 学習には使わず、成績測定専用に取り分けたデータ。学習データだけ良くて新しいデータで悪くなる「過学習(overfitting)」を見抜くために使います。

ここで大事なのは、「学習状態」は重みだけではない、という点です。次のセクションで、なぜ重み以外も保存しないと正しく再開できないのかを見ます。

### 仕組みを詳しく
「モデルの重みだけ保存すればいい」と思いがちですが、それだけでは続きから正しく再開できません。最低でも次の4つを一緒に保存します。

1. モデルの重み(model state)。モデル本体の数値そのもの。これが無いと話になりません。

2. オプティマイザの状態(optimizer state)。Adam は各重みについて1次・2次モーメントという履歴を持ち、それを使って次の更新量を決めます。この履歴を捨てて重みだけ復元すると、再開直後の数ステップで更新量が不自然になり、loss が一時的に跳ね上がります(学習が一瞬崩れる)。だから必ず一緒に保存します。

3. 学習率スケジューラの状態(scheduler state)。学習率(learning rate、1回の更新で重みをどれくらい大きく動かすかの倍率)は、学習が進むほど少しずつ小さくしていくのが普通で、その進行を管理するのがスケジューラです。今がスケジュールの何合目かを保存しないと、再開時に学習率が初期値に戻ってしまいます。

4. 進行カウンタ(epoch / step)。今が何エポック・何ステップ目か。これが無いと「どこまで進んだか」が分からず、データを読み直したり早すぎる打ち切りをしたりします。

PyTorch での典型的な保存コードはこうです。

```python
torch.save({
    "step": step,
    "model": model.state_dict(),          # 重み
    "optimizer": optimizer.state_dict(),  # Adam の内部履歴
    "scheduler": scheduler.state_dict(),  # 学習率の進行度
    "rng": torch.get_rng_state(),         # 乱数の状態(厳密な再現用)
    "args": vars(args),                   # 学習設定(あとで照合用)
}, "checkpoint_step5000.pt")
```

`state_dict()` は「名前と数値テンソルの対応表(辞書)」を返すメソッドです。中身は例えば `{"backbone.layer1.weight": <shape [64, 3, 7, 7] のテンソル>, ...}` のようになります。再開時は逆をたどります。

```python
ckpt = torch.load("checkpoint_step5000.pt", map_location=device)
model.load_state_dict(ckpt["model"])
optimizer.load_state_dict(ckpt["optimizer"])
scheduler.load_state_dict(ckpt["scheduler"])
start_step = ckpt["step"] + 1   # 次のステップから続ける
```

保存のタイミングは2種類を組み合わせるのが定番です。

- 定期保存: 例えば 1000 ステップごと、または 1 エポックごと。突然死やスポット中断に備える保険。
- ベスト保存: 検証データの成績が過去最高を更新したときだけ別名で保存。最終的に使うのはこの「いちばん成績が良かった重み」です。学習を続けると過学習で悪化することがあり、最後の重みが最良とは限らないからです。

容量にも注意します。モデルの重みは数百MB〜数GBあり、Adam のオプティマイザ状態は1次・2次モーメントで重みの約2倍の容量を食います。つまりチェックポイント全体は「重みの約3倍」になりがちです。全ステップ分残すとディスクが溢れるので、「最新3個だけ残して古いものを消す」世代管理(rotation)をします。

タイムラインを ASCII で描くとこうなります。

```
step:  0 ----1000----2000----3000(中断!)
保存:        ●       ●               × ここで落ちる
再開:                ↑ 2000 から続行(1000ステップ分しか損しない)
```

保存先の選択も重要です。ローカルディスクは速いが、ノードごと消えると失われます。スポット中断を前提にするなら、ノードの外(オブジェクトストレージ、例えば S3。バージョニングを有効にすると上書き事故にも強い)へ書き出すのが安全です。S3 への保存は遅いので、ローカルに書いてから非同期でアップロードする、といった工夫もよく使われます。

### 手法の系譜と主要論文
Checkpoint/Resume は特定の論文が発明した「手法」というより、大規模学習を回すための標準的なエンジニアリング技術です。しかし大規模学習の論文・技術報告では、その重要性と高速化の工夫が繰り返し論じられてきました。系譜を追います。

- Shoeybi ら "Megatron-LM"(2019、NVIDIA、arXiv:1909.08053)や Brown ら "GPT-3"(NeurIPS 2020、OpenAI、arXiv:2005.14165)などの大規模学習報告では、数百〜数千 GPU を何週間も回すため、ハードウェア故障が常態として扱われます。頻繁なチェックポイントと自動再開が運用の前提に組み込まれています。

- Rajbhandari ら "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"(SC 2020、Microsoft、arXiv:1910.02054、DeepSpeed の中核)は、巨大モデルでは重みもオプティマイザ状態もチェックポイント自体が巨大化する問題に取り組み、状態を複数 GPU に分割して保存・復元する仕組みを示しました。動機は「1台のメモリやディスクに状態が収まらないから」、効果は「桁違いに大きいモデルを学習・再開できる」、トレードオフは「保存・復元の手順が複雑になり、GPU 構成を変えると単純には復元できない」ことです。

- Mohan ら "CheckFreq: Frequent, Fine-Grained DNN Checkpointing"(USENIX FAST 2021、Microsoft)は、「どのくらいの頻度で保存すべきか」を自動で決め、保存処理を計算と重ねて(パイプライン化して)止め時間を最小化する手法を提案しました。

- Eisenman ら "Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models"(USENIX NSDI 2022、Meta)は、チェックポイント保存が学習を一時停止させてスループットを下げる問題に対し、差分のみ保存・非同期書き込み・量子化(数値を粗く丸めて容量を減らす)で保存コストを下げました。効果は「保存頻度を上げても学習が遅くならない」、トレードオフは「実装の複雑化と量子化によるわずかな状態劣化」です。

共通する教訓は「保存は安いうちに頻繁にやれ。ただし保存コストと容量が学習速度を殺さないよう工夫しろ」です。

### 論文の実験結果(定量データ)
チェックポイント研究の効果は、主に「保存にかかる停止時間(stall、保存のために学習を止める時間)」と「その削減によるスループット改善」で測られます。これらが重要なのは、保存頻度を上げるほど障害時の損失は減る一方、素朴に実装すると保存のたびに学習が止まって全体が遅くなる、というトレードオフがあるからです。

- CheckFreq(FAST 2021)は、報告によると、保存処理を計算とパイプライン化することで、頻繁にチェックポイントを取りながらも学習のオーバーヘッド(余分な遅延)を概ね 3.5% 以下に抑えたとされます。素朴な「保存中は学習を止める」方式だと、頻度を上げると停止時間が積み上がって大きな遅延になるところを、ほぼ無視できる水準に下げた点が成果です。アブレーション的に見ると、保存を計算と重ねる(オーバーラップさせる)工夫を外すと、停止時間がそのまま遅延として現れ、頻繁な保存が現実的でなくなります。

- Check-N-Run(NSDI 2022)は、巨大な推薦モデル(数 TB 規模の埋め込みテーブルを持つ)で、差分保存と量子化を組み合わせ、チェックポイントの書き込み量を大幅に(報告ではおよそ 6〜17 倍)削減したとされます。書き込み量が減れば、同じネットワーク帯域でより頻繁に・より速く保存でき、障害からの復旧で失う学習時間を小さくできます。アブレーションでは、量子化を外すと容量削減効果が縮み、差分保存を外すと毎回の保存量が膨らむことが示され、両者が相補的に効くことが分かります。

- ZeRO(SC 2020)の文脈では、状態を分割することで、単一 GPU では決して保持・保存できない規模(数千億〜兆パラメータ)のモデルのチェックポイントが現実に取れるようになりました。ここでの「改善」は速度というより「そもそも保存可能になる」という質的な飛躍です。

数値は環境依存ですが、一貫した結論は「保存を計算と重ね、書き込み量を減らせば、頻繁な保存と高いスループットを両立できる」という点です。

### メリット・トレードオフ・限界
メリット:
- 中断しても続きから再開でき、計算時間と費用の無駄を最小化できる
- 安価なスポットインスタンスを中断前提で安心して使える
- ベスト保存により、過学習で悪化する前の最良の重みを確実に確保できる
- 学習を意図的に止めて設定を変え、続きから試す柔軟な運用ができる

トレードオフ・限界:
- 保存にはディスク/ネットワークの I/O 時間がかかり、頻度を上げすぎると学習が遅くなる(CheckFreq / Check-N-Run はこれを緩和する研究)
- 重み + オプティマイザ状態は大きく(Adam なら重みの約3倍)、世代管理しないとディスクを食い尽くす
- 重みだけ保存してオプティマイザ・スケジューラ状態を忘れると、再開時に学習が一時的に乱れる(最頻出の落とし穴)
- GPU 台数やモデルの分割構成を変えると、分散学習のチェックポイントは単純には復元できないことがある
- 乱数の状態(シード、DataLoader の進行位置を含む)まで完全に復元しないと、厳密には「中断しなかった場合」と同一の学習にはならない(多くの場合は実害ないが、再現実験では注意)

研究上の未解決課題としては、(1) 巨大モデルのチェックポイントを GPU 構成に依存せず再分配して復元すること(elastic / resharding checkpoint)、(2) 保存頻度と障害確率から最適な保存間隔を理論的に決めること、(3) 学習ジョブ全体(複数ノード)を一貫した時点で止めて保存する分散スナップショットの整合性保証、などがあります。

### 発展トピック・研究の最前線
近年は、チェックポイントを GPU メモリから直接ストレージへ流す高速 I/O、あるいは一度 CPU メモリへ退避してから非同期で書き出す方式が一般化しています。さらに「フォールトトレラント学習(fault-tolerant training)」として、故障を検知したら自動で生きているノードだけで学習を継続し、後から復旧したノードを編入する弾力的(elastic)な仕組みが、PyTorch の分散学習ランチャ等で実用化されています。チェックポイントはこの弾力性の土台です。

大規模言語モデルの学習では、チェックポイントの書き込みがボトルネックにならないよう、テンソル並列・パイプライン並列・データ並列を組み合わせた状態を効率よく分割保存し、後で別の並列構成へ読み替える「resharding(再分割)」の研究が活発です。また実験管理の観点では、保存したチェックポイントを「どの実験のどのステップの重みか」として追跡し、モデルレジストリ(モデルのバージョン管理台帳)へ登録して再現性を担保する運用が標準になりつつあります。

### さらに学ぶための関連トピック
- [TensorBoard Logging(学習の可視化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)
- [Hyperparameter Sweep(ハイパーパラメータ探索)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)
- [DataLoader Prefetch / num_workers(データの並列先読み)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/13_datasets-dataprocessing)
- [Warm GPU Node (pause Pod)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)


## Cosine Annealing 学習率スケジュール

### ひとことで言うと
Cosine Annealing(コサイン・アニーリング)は、学習の進み具合に合わせて「学習率(1 回の更新で重みをどれだけ動かすかの大きさ)」をコサイン曲線の形でなめらかに下げていくスケジューラです。最初は大きく、終盤にかけてゆっくり小さくしていき、最後はほぼゼロにします。曲線の両端が平らで中央が急、という人間が手で設計しにくい自然な減り方を、たった数個の設定値で実現します。Transformer や ViT(画像用 Transformer)の学習で最も広く使われる定番スケジューラです。

### 直感的な理解
損失関数を「山あり谷ありの地形」、学習を「その地形をボールが転がって谷底を探す過程」だと思ってください。学習率は「1 歩の歩幅」にあたります。

歩幅が大きすぎると、谷底を通り越して反対側の斜面に飛んでしまい、損失が上下に振動したり、最悪は発散します。逆に歩幅が小さすぎると、なかなか谷底に着かず学習が遅く、浅い窪み(局所解)で止まりやすくなります。

理想は「最初は大きな歩幅でざっくり谷の在りかを探し、近づいたら歩幅を縮めて谷底にそっと収める」ことです。これを学習の進行に応じて歩幅を変えることで実現するのが学習率スケジューリングです。

素朴なやり方として「一定回数ごとに学習率を 1/10 にする」階段状の減衰(step decay)があります。これでも効きますが、学習率が急にガクッと落ちる瞬間があり、損失の進み方が不連続にカクつきます。また「いつ落とすか」を手作業で決める必要があり調整が面倒です。Cosine Annealing は、この「滑らかに、かつパラメータ少なく学習率を減らす」要求にぴったり応えます。

### 基礎: 前提となる概念
コサイン関数 cos(θ) は、θ=0 で 1、θ=π/2 で 0、θ=π で −1 を取る波の関数です。重要なのは「両端(0 と π)で傾きがほぼ平ら、中央(π/2)付近で最も急に変化する」という形です。この形を学習率の減衰に流用すると、序盤は高い学習率をしばらく保ち、中盤で一気に下げ、終盤は低い学習率をしばらく保つ、という挙動になります。

学習率スケジューラ全般の役割は、学習が進むほど「探索(広く良い領域を探す)」から「収束(見つけた良い領域の底へ丁寧に収める)」へ重心を移すことです。序盤は高学習率で広く探索し、終盤は低学習率で精密に収束する。コサインはこの移行を滑らかな曲線一本で表現します。

「アニーリング(annealing)」という語は金属の焼きなまし(高温からゆっくり冷ます)に由来します。最初は熱く(高学習率で激しく動き回り)、徐々に冷ます(学習率を下げて落ち着かせる)イメージがそのまま名前になっています。物理の焼きなまし法(simulated annealing)では、高温では悪い状態へ移る確率が高く広く探索でき、温度を下げると良い状態に落ち着く、という理屈があり、学習率の役割と対応します。

なぜ「滑らかさ」が嬉しいのか、もう一段。最適化理論では、学習率を `1/t` や `1/√t` のように緩やかに 0 へ近づけることが確率的勾配降下(SGD)の収束に必要だと知られています。step decay はこの「0 へ近づける」を階段で粗く近似するのに対し、コサインは連続的かつ終端で確実に 0 近くへ落とすので、理論的な収束条件とも整合的に「最後にきちんと収束させる」挙動になります。

### 仕組みを詳しく
総ステップ数(または総エポック数)を T、最大学習率を η_max、最小学習率を η_min とします。現在のステップ t における学習率 η(t) は次式で決まります。

```
η(t) = η_min + 0.5 · (η_max − η_min) · (1 + cos(π · t / T))
```

各記号の意味を確認します。t は現在のステップ番号、T は学習全体のステップ数、η_max は最初に与える最大学習率、η_min は最後に到達させる最小学習率です。`π · t / T` は学習の進捗(0 から始まり、終了時に π になる)をコサインの角度に変換しています。

挙動を具体的な数値で追います。η_max=0.001、η_min=0、T=100000 ステップとすると、

- t=0(開始): cos(0)=1 なので η = 0.5·0.001·(1+1) = 0.001(最大)。
- t=25000(1/4): cos(π/4)≈0.707 なので η ≈ 0.5·0.001·1.707 ≈ 0.000854。
- t=50000(中間): cos(π/2)=0 なので η = 0.5·0.001·1 = 0.0005(ちょうど半分)。
- t=75000(3/4): cos(3π/4)≈−0.707 なので η ≈ 0.5·0.001·0.293 ≈ 0.000146。
- t=100000(終了): cos(π)=−1 なので η = 0(最小)。

曲線の形は、両端(開始直後と終了直前)が平ら(傾きが 0)で、中央付近で最も急に下がる「S 字を寝かせたような」形になります。

```
η
η_max |*..
      |   *.
      |     *.
      |       *.
      |         *.
      |            *
η_min |________________*___  → t
      0       T/2        T
```

この「両端が平ら」という性質が効きます。序盤は高い学習率をしばらく保って広く探索でき、終盤は低い学習率をしばらく保って解を丁寧に磨き込めます。step decay の「ある瞬間だけ急落して、あとは一定」とは対照的に、減衰が常に滑らかに連続している点が特徴です。半分の進捗(t=T/2)でちょうど学習率が中間値になる、という覚えやすい性質も実用的です。

論文 SGDR ではさらに Warm Restart(温め直し)を組み合わせます。コサインで η_min まで下げきったら、いきなり η_max に戻して再びコサインで下げる、を繰り返すのです。下げきった瞬間に「再加熱」することで、いったん落ち着いた解から飛び出して別のより良い谷を探せます。restart の周期を毎回伸ばす(各サイクル長 T_i を T_mult 倍ずつ大きくする、例えば 10, 20, 40 エポック)使い方も提案されています。ただし実務では restart なしの「1 回だけ下げきる」単純なコサインも非常に多く使われます。

多くの場合、コサインは [学習率 Warmup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)(最初の数%で学習率を 0 から η_max まで上げる)と組み合わせ、「warmup → cosine decay」という形で運用されます。これは warmup なしで高学習率から始めると初期に発散しやすいためです。実装上の注意として、warmup 区間を T に含めるか別勘定にするかで曲線が微妙にずれるため、フレームワークごとの定義を確認する必要があります。

### 手法の系譜と主要論文
- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017, フライブルク大, arXiv:1608.03983)。コサイン形の減衰と warm restart の組み合わせを提案。動機は「滑らかな減衰で安定に収束させつつ、周期的な再加熱で局所解から脱出し、複数解のアンサンブル的効果も得る」こと。これがコサインスケジュールの源流です。

- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017, arXiv:1706.03762)。Transformer の原論文は warmup 付きの逆平方根減衰(学習率を warmup 後に 1/√t で下げる)を使いました。コサインそのものではありませんが、「warmup してから滑らかに下げる」という発想を広めました。その後の ViT・大規模言語モデルの多くは、この逆平方根減衰から warmup + cosine decay へ移行していきます。

- Dosovitskiy et al., "An Image Is Worth 16x16 Words (ViT)" (ICLR 2021, Google, arXiv:2010.11929)、Liu et al., "Swin Transformer" (ICCV 2021, Microsoft, arXiv:2103.14030)。これら視覚 Transformer の学習レシピは warmup + cosine annealing をほぼ標準採用しました。畳み込みネットより学習が不安定になりやすい ViT 系で、コサインが収束安定化に有効であることが広く確認され、以後デファクトスタンダードになりました。

- Hu et al., "MiniCPM" (2024, arXiv:2404.06395)。大規模言語モデルの学習で Warmup-Stable-Decay(WSD)スケジュールを体系化し、コサインの「総ステップ T を事前固定しないといけない」弱点に正面から応えました。コサインの後継候補として位置づけられます(後述)。

### 論文の実験結果(定量データ)
評価指標は主に画像分類の test error(誤分類率。小さいほど良い)あるいは accuracy(正解率)、そしてそこへ到達する速さです。

SGDR 論文では CIFAR-10 と CIFAR-100(それぞれ 32×32 のカラー画像を 10 クラス・100 クラスに分類するデータセット)で評価しました。Wide ResNet(WRN-28-10 など)を用いた実験で、コサイン + warm restart は従来の step decay と同等以上の最終精度に、より少ないエポックで到達したと報告されています。具体的には CIFAR-10 で 4% 前後、CIFAR-100 で 19% 前後の test error が報告され、warm restart のチェックポイントを集めるスナップショットアンサンブル(restart 直前のモデルを複数保存して平均/投票する手法)によってさらに 1〜2% 程度誤差を減らせることが示されました。これは「1 回の学習で複数の良いモデルを無料で得られる」点が新しく、追加学習なしで精度を底上げできる意味で重要です。

その後の ViT・Swin 系の学習レシピでも、warmup + cosine は線形減衰や step decay と比較され、ImageNet(約 128 万枚、1000 クラスの大規模画像分類データセット)分類で安定して高い top-1 accuracy を出すことが報告されています。ViT-B/16 や Swin-T の標準レシピはいずれも warmup(数エポック)+ cosine decay を採用し、それぞれ ImageNet で 80% 前後以上の top-1 を達成しています。アブレーション的な観察として、(1) warmup を外すと ViT は学習初期に発散しやすく、コサイン単体では恩恵を活かせない、(2) 終盤に η_min まで十分下げきらないと最終精度が頭打ちになる、(3) step decay と比べてコサインは「いつ落とすか」の手調整が不要なぶん再現性が高い、という点が繰り返し確認されています。

数値の意味を補足すると、ImageNet top-1 で 0.3〜0.5% の差は、この規模のベンチマークでは手法の優劣を分ける有意な差とみなされます。スケジューラだけでこの程度の差が出るため、学習レシピの中でコサインは「安価だが効く」要素として重視されています。大規模言語モデルの学習でも、Chinchilla(Hoffmann et al., 2022)など多くの代表的研究がコサイン減衰を採用しており、「学習トークン総量にコサインの T を合わせる」ことが標準作法になりました。逆に言えば、T を実際の学習長より長く設定して途中で打ち切ると、学習率が十分下がりきらず損失が悪化することが知られ、これがコサインの実務上の落とし穴です。

### メリット・トレードオフ・限界
メリット。学習率が滑らかに減り、step decay のような不連続なカクつきがありません。終盤に学習率が十分小さくなり解を丁寧に磨けるため最終精度が上がりやすい。設定パラメータが少ない(η_max, η_min, T 程度)。主要な深層学習フレームワークに標準実装があり 1 行で使えます。warm restart を使えば局所解脱出やスナップショットアンサンブルの効果も得られます。

トレードオフと限界。総ステップ数 T を事前に決める必要があり、途中で学習を延ばすと曲線がずれます(計画変更に弱い)。T を学習長より長く取って途中で止めると学習率が下がりきらず損失が悪化する、逆に短く取って学習を続けると η_min に張り付いて進まなくなる、という両側の罠があります。warmup なしで使うと序盤の高学習率で発散しやすく、ほぼ常に warmup との併用が必要です。warm restart は精度が一時的に落ちる谷ができ、運用上の見え方が複雑になります。そして「コサインが常に最良」ではなく、タスクによっては線形減衰(特に大規模言語モデルの一部レシピでは線形 decay が好まれる報告もある)や WSD が勝つこともあります。

未解決・議論中の課題として、学習途中で総ステップ数を変えたいときにスケジュールをどう滑らかに引き延ばすか(continual / resumable scheduling)、最終 η_min をゼロにすべきか小さな正の値(例 η_max の 10%)にすべきか、warm restart の周期をどう自動決定するか、といった点が依然として経験則に頼っている部分です。

### 発展トピック・研究の最前線
近年の大規模言語モデルでは、コサインに代わって学習率を一定に保ったあと最後に急減衰させる WSD(Warmup-Stable-Decay)スケジュールが提案され(MiniCPM, Hu et al., 2024)、「総ステップ数を固定しなくても済む」「途中のどこからでも短い decay を足せば良いチェックポイントが得られる」という利点で注目されています。これはコサインの「T を事前固定しないといけない」弱点への直接の応答です。WSD は安定相(stable phase)の損失曲線を見てから decay を開始できるため、データ追加や計画変更に強く、スケーリング則の調査にも便利だと報告されています。また、学習率を後から自由に再スケジュールできる連続学習向けスケジューラ、損失の二次情報(曲率)からスケジュールを自動調整する手法など、スケジューリングをハイパーパラメータ探索から解放する研究が進んでいます。コサインは依然として最も無難で広く使われる基準線(ベースライン)であり、新しいスケジューラはほぼ常にコサインと比較して提案されます。

### さらに学ぶための関連トピック
- [学習率 Warmup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [One-Cycle 学習率ポリシー](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [学習率スケジューラ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)


## Data Augmentation 概論

### ひとことで言うと
Data Augmentation(データ拡張)とは、手元の訓練データに「左右反転」「少し回転」「色を変える」などの加工を加えて、見た目の違う訓練サンプルを人工的に増やす技術です。新しくデータを集めなくても、モデルが「初めて見るデータ」に強くなる(汎化する)ように仕向ける、ほぼすべての画像系深層学習で前提となる基本テクニックです。

### 直感的な理解
深層学習のモデル、特に画像を扱うニューラルネットワークは、学習で調整される内部パラメータが数千万から数億個もあります。これだけ自由度が大きいと、訓練データを「丸暗記」してしまう危険があります。これを過学習(overfitting)と言います。訓練に使った画像では完璧に正解するのに、少しでも違う本番の画像では間違える状態のことです。

具体例で考えます。あるカメラ画像で「前方に歩行者がいるか」を学習させたいとします。もし訓練データに写っている歩行者がたまたま全員「画面の右側、晴れた昼間」だったら、モデルは「右側にいて明るい人影=歩行者」と誤った相関を覚えてしまいます。すると本番で「左側の歩行者」や「夕暮れの薄暗い歩行者」を見逃すかもしれません。本来学ばせたいのは「人の形をしたもの=歩行者」という本質的な特徴で、「右側」「明るい」はたまたまの偶然(交絡)です。

ここで訓練画像を左右反転すれば歩行者は左側に来ます。明るさを下げれば夕暮れに近づきます。こうして「意味は同じだが見た目が違う」サンプルを増やすことで、モデルは位置や明るさに依存しない本質的な特徴を学ばざるをえなくなります。言い換えると、データ拡張は「このような変化が起きても答えは変わらないはずだ」というドメイン知識(不変性、invariance)をデータを通じてモデルに教え込む手段です。

新しいデータを実際に集めるには、現場でデータを取得し、人手でラベル(正解)を付ける必要があり、莫大なコストがかかります。データ拡張は既存データをプログラムで加工するだけなので、ほぼ無料で訓練サンプルの多様性を増やせる点が大きな魅力です。

### 基礎: 前提となる概念

過学習(overfitting)とは、モデルが訓練データの偶然の特徴やノイズまで覚え込み、訓練データでの誤差は下がり続けるのに検証データでの誤差は途中から上がる現象です。データ拡張は、この過学習を抑える「正則化(regularization)」の一種です。正則化とはモデルが訓練データに過度に適合するのを防ぐあらゆる工夫の総称で、重みに罰則を与える weight decay や、ニューロンをランダムに無効化する dropout などもその仲間です。データ拡張は「入力側を多様にすることで正則化する」アプローチに位置づけられます。

画像の tensor(数値の多次元配列)を理解しておくと拡張の中身が見えます。1枚のカラー画像は典型的に形状 [C, H, W] = [3, 256, 256] で表されます。C=3 は RGB の3チャンネル、H=256 は縦のピクセル数、W=256 は横のピクセル数です。各要素は0〜255(または0〜1に正規化した)輝度値です。データ拡張の多くは、この tensor に対する数値変換として実装されます。

不変性(invariance)と同変性(equivariance)の区別も重要です。分類タスクでは「画像を左右反転しても答えは犬のまま」という不変性が望ましく、ラベルは変えずに入力だけ変換します。一方、検出やセグメンテーション、軌跡予測のように出力が空間構造を持つタスクでは「入力を反転したら出力も対応して反転する」という同変性を守る必要があり、入力とラベルを必ずセットで変換しなければなりません。この違いを誤ると、拡張がかえって学習を壊します。

### 仕組みを詳しく

代表的なデータ拡張を tensor [3, H, W] のレベルで説明します。

幾何変換(位置や形をいじる):
- 水平反転(horizontal flip): 横方向のピクセル順を逆にする。`x[:, :, ::-1]` のイメージ。[3,256,256] の形は変わらず中身だけ左右が入れ替わる。最も安全で効果が高い定番。
- 回転(rotation): 例えば ±5〜15度回す。地平線やカメラ姿勢のわずかな傾きを模す。
- ランダムクロップ(random crop): [3,256,256] から [3,224,224] の領域を切り出し、再び目的サイズに拡大。被写体の位置・大きさのばらつきを作る。分類の標準前処理。
- スケーリング・平行移動・アフィン変換: 被写体を少し拡大縮小、上下左右にずらす、せん断する。

色・輝度変換(見た目をいじる):
- カラージッター(color jitter): 明るさ・コントラスト・彩度・色相を ±20% などランダムに変える。昼/夕方/曇りの違いを模す。
- ガウシアンノイズ付加: 各ピクセルに小さな乱数を足す。センサーノイズを模す。
- ガウシアンブラー: 軽くぼかす。ピントの甘さや雨滴を模す。

オクルージョン・消去系(隠す):
- Cutout / Random Erasing(DeVries & Taylor 2017): 画像の矩形領域を一様な値やノイズで塗りつぶす。被写体の一部が隠れても認識できる頑健性を促す。

合成系(複数画像を混ぜる):
- Mixup(Zhang et al. 2018): 2枚の画像を λ:(1−λ) の比率で重ね合わせ、正解ラベルも同じ比率で混ぜる。λ=0.7 なら「猫7割+犬3割の画像」を作り、ラベルも「猫0.7 犬0.3」とする。
- CutMix(Yun et al. 2019): 画像Aの一部の矩形を画像Bの同じ位置の領域で置き換え、ラベルは置き換えた面積比で混ぜる。Mixup より自然な画像になりやすく、領域情報を保つ。

処理の流れ(before/after):
```
元画像 [3,256,256]
  → ランダム水平反転(50%の確率で実行)
  → ランダムクロップ&リサイズ → [3,224,224]
  → カラージッター(明るさ・彩度を ±20%)
  → 正規化(チャンネルごとに平均0・分散1付近へ)
  → モデルへ入力
```
重要なのは、これらが「毎エポック、ランダムなパラメータで」適用される点です。同じ元画像でも1周目と2周目では微妙に違う画像としてモデルに渡るので、実質的なデータ量が増えます。拡張は通常 GPU 学習と並行して CPU のデータローダ側で実行され、I/O やデコードと重ねることで学習速度をほぼ落とさず適用できます。

### 手法の系譜と主要論文

Krizhevsky, Sutskever, Hinton「ImageNet Classification with Deep Convolutional Neural Networks」(2012, NIPS、通称 AlexNet)。深層学習が画像認識で一躍注目された論文です。ここで既に「ランダムクロップ+水平反転」と「PCA color augmentation(主成分分析で求めた色の方向に沿って色をずらす)」を使い、過学習を大きく減らしたと報告しました。提案理由はネットワークが巨大で過学習しやすいから。以降、拡張は画像分類の標準技法になりました。

Shorten & Khoshgoftaar「A Survey on Image Data Augmentation for Deep Learning」(2019, Journal of Big Data)。本トピックの代表サーベイです。幾何変換・色変換・カーネルフィルタ・ランダム消去・特徴空間での拡張・GAN による生成・自動探索などを体系的に整理しました。重要な指摘は「拡張は元データの分布を不当に歪めない範囲で行うべき」という点で、たとえば手書き数字の「6」を180度回すと「9」になりラベルが変わってしまうため使ってはいけない、という安全性の原則を明示しています。

DeVries & Taylor「Improved Regularization of CNNs with Cutout」(2017、arXiv:1708.04552)。画像の一部を矩形で消す単純な拡張で、被写体の一部が隠れても判断できる頑健性を促し、CIFAR で安定した精度向上を示しました。後の CutMix の発想の源流です。

Zhang et al.「mixup: Beyond Empirical Risk Minimization」(2018, ICLR、arXiv:1710.09412)。画像とラベルを線形に混ぜる手法。提案理由は「サンプル間を補間することで決定境界を滑らかにし、誤ラベルや敵対的入力に強くする」。汎化向上とロバスト性向上を示しました。トレードオフは、混ざった不自然な画像が出るため、位置情報が重要なタスクではそのままだと相性が悪いことです。

Cubuk et al.「AutoAugment: Learning Augmentation Strategies from Data」(2019, CVPR、arXiv:1805.09501)。どの拡張をどの強さでどの確率で適用するかを、人手ではなく強化学習で自動探索する手法。提案理由は「最適な拡張はデータセットごとに違い、人手調整が大変だから」。効果は当時の最高精度。トレードオフは探索に膨大な計算が必要なことでした。

Yun et al.「CutMix」(2019, ICCV、arXiv:1905.04899)と Cubuk et al.「RandAugment」(2020, NeurIPS、arXiv:1909.13719)。CutMix は Mixup と Cutout の利点を統合し、RandAugment は AutoAugment の重い探索を捨てて「N個の変換をランダムに選び強さ M で一律に適用」という2パラメータだけに単純化し、同等性能を桁違いに安い計算で達成しました。これにより自動拡張が実用的になりました。

### 論文の実験結果(定量データ)

指標はおもに画像分類の Top-1 精度(最も確信したクラスが正解だった割合、%)、または Test error(誤り率、%)です。ベンチマークは CIFAR-10/100(小画像、10/100クラス)と ImageNet(大規模、1000クラス)が中心です。

- AlexNet(2012)では、データ拡張を加えることで ImageNet の誤り率が約1ポイント下がり、当時としては過学習を大きく抑える効果があったと報告されています。
- Cutout(2017)では、Wide ResNet を用いた CIFAR-10 でテスト誤り率がおよそ 0.3〜0.5 ポイント、CIFAR-100 で 1 ポイント前後改善したと報告されています。単純な実装で安定した利得が出る点が評価されました。
- mixup(2018)では、ImageNet で ResNet-50 の Top-1 精度がベースラインからおよそ 1.2〜1.5 ポイント向上したと報告されています。加えて、ラベルノイズを人工的に注入した条件での頑健性や、敵対的サンプルへの耐性が向上することも示されました。
- AutoAugment(2019)では、ImageNet で当時の最高精度(報告で Top-1 約 83.5%)を達成し、CIFAR-10 でも誤り率を最先端水準へ押し下げました。RandAugment(2020)は探索計算をほぼゼロにしながら AutoAugment と同等以上の精度を再現し、「探索の重さ」というトレードオフを解消しました。
- CutMix(2019)では、ImageNet で ResNet-50 の Top-1 が Mixup や Cutout を上回り、報告で約 78.6%(ベースライン約 76.3% から 2 ポイント超の改善)を達成しました。さらに、領域を混ぜる性質から物体の局在化(どこに物体があるか)が改善し、弱教師あり物体検出の性能も上がったと報告されています。

アブレーションで一貫して観察されたのは、(1) 拡張を強くしすぎると元データ分布から離れすぎて精度が落ちる「強度の最適点」が存在すること、(2) 学習データが少ないほど拡張の利得が大きいこと、(3) Mixup/CutMix のような混合系は分類では強力だが、出力が空間構造を持つタスク(検出・セグメンテーション・軌跡予測)ではラベル混合の意味づけが難しく、そのまま使うと逆効果になりうること、です。これは「適切な拡張はタスク依存」というサーベイの主張を定量的に裏づけます。

### メリット・トレードオフ・限界

メリット
- 追加のデータ収集・ラベル付けコストなしで、訓練サンプルの多様性を増やせる。
- 過学習を抑え、本番(未知データ)での精度を上げる正則化として広く機能する。
- 位置・明るさ・隠れ・天候など、実環境のばらつきへのロバスト性が上がる。
- 多くのフレームワークに定番の実装があり、導入コストが小さい。

トレードオフ・限界
- やりすぎると元データの分布を歪め、かえって性能が落ちる。拡張の種類と強度はチューニング対象。
- ラベルとの整合に注意が必要。空間構造を持つ出力では入力変換に合わせてラベルも変換しなければならない(後述の運転系の例)。
- データに存在しない種類の変動(例: 訓練に夜間が皆無)は拡張だけでは生み出せない。本質的に新しい状況は実データ収集や合成データ生成が必要。
- Mixup/CutMix のような混合系はタスク適合性が限定的で、回帰や構造化予測へは慎重な適用が要る。

軌跡予測・運転系タスクでの固有の注意: 出力が「将来どう動くか」という空間的な値である場合、入力画像を左右反転したら、操舵やコースの横方向の正解も符号反転させなければなりません(同変性)。模倣学習(behavior cloning)で軌跡を教師として学ぶ設定では、画像を反転したら教師軌跡の横座標も反転する、といったラベル整合が拡張導入の前提条件になります。これを怠ると「右に曲がる画像なのに左に曲がる教師」を与えることになり、学習が破綻します。

### 発展トピック・研究の最前線

データ拡張は近年、設計済みの変換だけでなく「学習可能・自動化」の方向へ発展しています。AutoAugment から RandAugment、さらにオンラインで方策を最適化する手法へと続く自動探索の系譜に加え、特徴空間での拡張(中間表現を混ぜる manifold mixup)、敵対的に最も損失を上げる摂動を加える adversarial augmentation、そして近年は拡散モデルなどの生成モデルで訓練画像そのものを合成して水増しする合成データ拡張が活発です。自己教師あり学習(SimCLR など)では、強いデータ拡張による2つのビューを「同じものとして近づける」ことが学習の中心原理になっており、データ拡張は単なる前処理から表現学習の本質的構成要素へと役割を広げています。一方で、生成データに頼りすぎるとモデルが生成器の偏りを学んでしまう「モデル崩壊(model collapse)」のリスクも指摘されており、合成と実データのバランスが新たな研究課題になっています。

### さらに学ぶための関連トピック
- [Imitation / Behavior Cloning Loss](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [System1/System2分離（Reactive/Reasoning）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/14_model-architecture-design)
- [Layer Normalization](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## frozen backboneによる特徴キャッシュ

### ひとことで言うと
画像を特徴ベクトルに変換する重い部品(バックボーン)の重みを固定(凍結)しておくと、各フレームの計算結果が「そのフレームだけで決まる独立な値」になります。すると毎回すべてのフレームを計算し直す必要がなく、新しく来た1フレームだけ計算し、残りは前回の結果を使い回せます(キャッシュできます)。これにより、いちばん重いバックボーン計算をフレーム数ぶんのおよそ N 分の1に減らせる、という効率化の話です。しかもこの最適化は近似ではなく、出力が完全に一致する厳密な節約です。

### 直感的な理解
動画を1秒ごとに処理して、毎回「直近4枚の画像をまとめて見て判断する」システムを想像してください。素朴に作ると、毎秒「直近4枚すべてを重い画像処理にかける」ことになります。しかしよく考えると、4枚のうち3枚は前の秒にもう処理したばかりの、まったく同じ画像です。同じ画像を同じ計算機で処理すれば結果は当然同じはずなのに、毎回ゼロから計算し直すのは大きな無駄です。

ここで「前回の計算結果をメモ(キャッシュ)に取っておいて、変わらないものは使い回す」というのは、誰でも思いつく自然な節約です。料理で言えば、毎食ごとに出汁を取り直すのではなく、前回取った出汁を冷蔵庫に保存しておいて使い回すようなものです。重要なのは「結果が変わらない」という保証で、これが成り立つときだけキャッシュは安全に使えます。本トピックの主張は、バックボーンを凍結すればこの保証が厳密に成り立つ、という点にあります。出汁の比喩で言えば、レシピ(重み)が一切変わらないので、同じ材料(同じ画像)からは必ず同じ味(同じ特徴)が出る、ということです。

### 基礎: 前提となる概念
バックボーン(backbone)とは、画像から特徴を抽出するニューラルネットの本体部分を指します。CNN(畳み込みニューラルネット)や Vision Transformer(ViT)などで、画像 `[3, H, W]`(3チャンネル=RGB、縦H横W)を受け取り、抽象的な特徴マップやベクトルに変換します。モデル全体の中でいちばん計算が重い部分で、推論時間の大半をここが占めることが珍しくありません。たとえば ResNet-50 で1枚あたり約40億回の積和演算(4 GFLOPs)、ViT-Large なら数十 GFLOPs に達し、後段の軽い融合層やデコーダの計算量を桁で上回ります。

凍結(freeze)とは、学習中もバックボーンの重み(ネットワーク内部の数百万〜数億のパラメータ)を一切更新しないことです。逆に学習可能(trainable)とは、誤差逆伝播(backpropagation、損失を小さくする方向にパラメータを少しずつ調整する仕組み)で重みを更新することです。凍結すると、ある画像 X をバックボーンに通した結果は「いつ通しても完全に同じ」になります。重みが変わらず関数として固定だからです。数学的には、凍結バックボーンを f とすると f は時間に依存しない決定的な関数で、f(X) は X だけの関数になります。

フレーム独立(frame-wise independence)とは、各フレームの埋め込みが「そのフレームの画像だけ」から決まり、他のフレームには依存しない性質です。バックボーンの中にフレーム間で情報を混ぜる処理(時間方向の attention や 3D 畳み込みなど)がなければ、この性質が成り立ちます。逆に言えば、もしバックボーンが時間方向に情報を混ぜるなら、あるフレームの特徴は周囲のフレームに依存するので、新フレームが来るたびに過去の特徴も変わってしまい、キャッシュは使えません。

この2つ、すなわち「重みが不変」と「フレーム独立」が揃うと、過去フレームの特徴は時刻が進んでも変わらないので、一度計算したら保存して再利用できます。これが特徴キャッシュ(feature caching)です。

なお、なぜバックボーンを凍結するのかには別の動機もあります。自己教師あり学習では、予測する側とされる側(ターゲット)が同じ重みを共有して一緒に動くと、両方が「何でも同じ値を出す」状態に潰れてしまう崩壊(representation collapse)という失敗が起きます。ターゲット側を凍結したり、勾配を流さない(stop-gradient)、ゆっくり更新する(EMA)などするのは、この崩壊を防ぐ標準的な手法です。つまり「凍結だからキャッシュできる」という効率の利点は、崩壊防止という別目的の副産物として得られる、という関係になります。

### 仕組みを詳しく
具体例で計算量を比べます。直近 N=4 フレーム、カメラ V=7 台、1秒に1回(1 Hz)処理するとします。

凍結なし・キャッシュなしの素朴版では、毎秒の処理量は次のとおりです。

```
毎秒: 4フレーム × 7カメラ = 28 枚をバックボーンに通す
```

凍結+キャッシュ版では、次のように動きます。

```
時刻 t:   e_{t-3}, e_{t-2}, e_{t-1}, e_t をバッファに保持(各 D 次元)
時刻 t+1: 新フレーム e_{t+1} だけ計算(7カメラ分)
          e_{t-2}, e_{t-1}, e_t は前回の結果をそのまま流用(再計算しない)
          e_{t-3} は FIFO(先入れ先出し)で押し出して破棄
毎秒: 1フレーム × 7カメラ = 7 枚だけバックボーンに通す
```

バックボーンの計算量がおよそ N 分の1(この例では 28 → 7、4倍の削減)になります。仮に ResNet-50(1枚 4 GFLOPs)なら、28枚で 112 GFLOPs/秒 が 7枚で 28 GFLOPs/秒 に落ちます。キャッシュに保持するのは小さな埋め込みベクトル(`[B, D]` を N 個)だけなので、メモリ負担はごくわずかです。仮に D=256、B=1、float32 なら1ベクトルは 256×4 = 1024 バイト = 1 KB、4個でも 4 KB に満たず、画像1枚ぶんの生データ(224×224×3×4 ≈ 600 KB)よりはるかに小さいことが分かります。

処理を整理すると次の4段です。第一に、各時刻、新しく到着した1フレームだけを凍結バックボーンとフレーム符号化器に通し、D次元の埋め込みを作る。第二に、その埋め込みを FIFO バッファに push し、最古を pop する。第三に、バッファ内の N 個は「過去に計算済みの値そのまま」で再計算しない。第四に、その後段(連結・時間方向の attention・デコーダ)は軽い部品なので毎回計算してよい。後段を毎回計算してよいのは、後段こそが「時刻が進むと出力が変わるべき」処理だからです。重い不変部分だけをキャッシュし、軽い可変部分は再計算する、という役割分担です。

ここで決定的に重要なのは、この最適化が近似ではなく厳密だという点です。凍結によって結果が完全に一致するため、キャッシュを使っても出力は再計算した場合とビット単位で同じになります(浮動小数点の演算順序が変わらない限り)。精度を一切犠牲にせず速度だけを得られる、稀に都合のよい最適化です。多くの高速化(量子化・蒸留・枝刈り)は精度を多少落として速度を得ますが、特徴キャッシュは精度のロスがゼロです。

前提が崩れるケースも押さえておきます。もしバックボーンを学習で更新する(凍結しない)設計にすると、重みが毎ステップ変わるため「前回の埋め込み」はもう正しくなくなり、キャッシュは無効になります。自己教師あり学習でよく使われる構成では、ターゲット側のバックボーンは凍結コピー(stop-gradient、勾配を流さない複製)にする一方、オンライン側(予測する側)のバックボーンは学習可能にすることがあります。その場合キャッシュが効くのは凍結ターゲット側に限られ、学習可能なオンライン側は毎回計算が必要です。また、データ拡張(augmentation、画像をランダムに変形してデータを水増しする手法)を毎回かける学習時には、同じフレームでも入力画像が毎回違うため、厳密にはキャッシュできません。キャッシュが完全に効くのは「拡張をかけない推論時」が典型です。

### 手法の系譜と主要論文
「過去の計算結果を保存して使い回す」という発想は、複数の分野で独立に確立されてきました。

言語モデルの世界では KVキャッシュ(key-value cache)が標準です。Transformer(Vaswani et al., 2017, NeurIPS, arXiv:1706.03762)に基づく自己回帰生成では、新しいトークンを1つ生成するたびに、過去すべてのトークンの key と value を再計算すると、系列長の二乗に比例する計算量がかかります。そこで過去トークンの key/value を保存しておき、新トークンぶんだけ計算する。これにより生成が大幅に高速化されます。トレードオフはキャッシュのメモリが系列長に比例して増えることで、長文生成ではメモリが律速になります。本トピックの特徴キャッシュは、これの「時系列フレーム版」アナロジーと言えます。KVキャッシュは「過去トークンの key/value は重みが変わらない限り不変」という性質を使い、特徴キャッシュは「過去フレームの埋め込みは凍結重みなら不変」という性質を使う、いずれも同じ「不変なものを使い回す」原理です。

運転ドメインの空間認識では、BEVFormer(Li et al., 2022, ECCV, arXiv:2203.17270)が過去のBEV特徴(Bird's-Eye-View、車を上空から見下ろした座標系の特徴)をキューに保持して再利用します。毎フレーム全履歴を作り直さず、過去のBEV結果を持ち回って時間方向の手がかりに使う設計で、リアルタイム性を確保します。トレードオフは履歴保持のメモリと、過去特徴が(モデルが更新されると)古い重みで作られたものになりうる点です。

自己教師あり学習では、ターゲット側を凍結または EMA(Exponential Moving Average、指数移動平均でゆっくり更新する)で安定化する設計が BYOL(Grill et al., 2020, NeurIPS, arXiv:2006.07733)や I-JEPA(Assran et al., 2023, arXiv:2301.08243)で確立しました。BYOL は対照学習で必須とされていた負例(違う画像のペア)を使わずに、ターゲット側を EMA で更新するだけで崩壊を防げることを示し、大きな驚きを与えました。さらに SimSiam(Chen & He, 2020, arXiv:2011.10566)は、EMA すら使わず単純な stop-gradient(片側に勾配を流さないだけ)で崩壊を防げることを示し、「ターゲット側を止める」ことの本質的な役割を明らかにしました。凍結/EMA/stop-gradient はいずれも崩壊防止のための設計ですが、その結果としてターゲット側の特徴がキャッシュ可能になります。

### 論文の実験結果(定量データ)
KVキャッシュの効果は計算量の理論オーダーで明確です。系列長を L とすると、キャッシュなしの素朴な再計算は全体で L の二乗(各ステップで全履歴を再計算)に比例し、KVキャッシュありなら新トークンぶんの線形コストに落ちます。長い系列ほど差が開き、生成速度が数倍から桁違いに改善します。代償としてキャッシュのメモリが L に比例し、長文では GPU メモリの大半を占めることもあります。vLLM の PagedAttention(Kwon et al., 2023, SOSP, arXiv:2309.06180)は、KVキャッシュを OS の仮想メモリのようにページ単位で管理してメモリ断片化をほぼ解消し、報告ではスループットを既存システム比で約2〜4倍に改善しました。これは「キャッシュをどう効率管理するか」が実務性能を左右する好例です。

BEVFormer は nuScenes(自動運転の標準ベンチマーク。多数の都市走行シーンに3D物体やマップの正解が付与されている)で評価され、過去BEVの時間融合を入れることで物体検出の指標 NDS(nuScenes Detection Score、検出の総合スコア。0〜1で高いほど良く、位置・速度・属性などの誤差を統合した指標)が約 0.51〜0.57 規模に達し、時間融合なしの構成を上回ったと報告されています。とくに速度推定の誤差(mAVE)が時間情報で大きく改善し、これは「過去フレームの特徴を保持して使い回す」ことの実利を示します。アブレーションでは、時間融合(temporal self-attention)を外すと速度誤差が悪化し検出スコアも下がることが確認され、履歴の再利用が単なる効率化でなく精度にも寄与する設計であることが分かります。

BYOL は ImageNet 線形評価(凍結特徴の上に線形分類器だけ学習)で、当時の対照学習(SimCLR など)に匹敵またはそれを上回る精度を、負例なしで達成しました。報告ではトップ1精度で 74.3%(ResNet-50)に届き、ターゲット側を EMA で安定化する設計が崩壊を防ぎつつ高品質な表現を学べることを示しました。SimSiam は EMA なし・負例なし・大バッチなしでも約 71% に達し、stop-gradient だけが崩壊防止の鍵であることを切り分けました。この「ターゲットを止めて安定させる」発想が、本トピックの凍結=キャッシュ可能の前提に直結します。

### メリット・トレードオフ・限界
メリットは、いちばん重いバックボーン計算をおよそ N 分の1に削減でき、1 Hz 推論の負荷とレイテンシが大きく下がること。キャッシュするのは小さな埋め込みベクトルだけなのでメモリ増は軽微で、しかも結果は再計算と完全一致する厳密な最適化であることです。

トレードオフと限界もあります。第一に、バックボーンを凍結している前提が崩れると(学習で更新する設計を採ると、あるいは時間方向に情報を混ぜるバックボーンを使うと)キャッシュは無効になります。第二に、バッファ管理(push/pop、GPU/CPU 間のデバイス配置、バッチ整合、シーンの切れ目でのバッファクリア)の実装ミスが入りやすく、古い特徴を間違った時刻に使うバグは出力が「それっぽく」見えるため検出しづらいです。第三に、これは効率ではなく精度側のトレードオフですが、凍結ゆえバックボーンが下流タスクに適応しないため、表現がタスク最適でない可能性があります。汎用の画像認識(ImageNet など)で学んだ特徴が、運転の細部(遠方の歩行者、夜間、悪天候)に最適とは限りません。

研究上の論点として、凍結による効率と、適応による精度のどちらを取るかは状況依存です。データが少なくバックボーンの過学習が怖い場合は凍結が有利、データが潤沢でタスク特化を狙う場合は学習可能(その場合キャッシュは諦める)が有利、といった判断になります。EMA で「ゆっくりだけ更新する」中間策もありますが、その場合は厳密なキャッシュではなくなり、近似(古い特徴を許容する)や定期的な再計算が必要になります。実務では「学習はバックボーン微調整あり(キャッシュなし)、デプロイは重みを凍結してキャッシュあり」と段階を分ける折衷も取られます。

### 発展トピック・研究の最前線
動画認識やストリーミング処理では、フレーム独立を意図的に設計してキャッシュ可能性を確保する「ストリーミング推論」アーキテクチャが研究されています。時間方向の混ぜ込みを軽量な後段に限定し、重いバックボーンはフレーム独立に保つことで、ライブ映像に対する低遅延推論を実現する方向です。逆に「時間方向に混ぜたいが、混ぜる範囲を限定して部分的にキャッシュする」工夫(直近 k フレームだけ再計算する近似)も検討されています。

また LLM のサービングで磨かれた KVキャッシュ管理の技術(PagedAttention によるメモリ断片化の解消、量子化によるキャッシュ圧縮、プレフィックス共有による複数リクエスト間のキャッシュ再利用)は、時系列特徴キャッシュにも応用が考えられます。多数の系列を同時に処理する状況で、共通する過去フレームのキャッシュをどう共有・管理するかは実務的に重要なテーマです。特にマルチカメラ・マルチエージェントの設定では、同じシーンの重複したフレームをまたいでキャッシュを共有できる余地があり、メモリと計算の両面で効きます。

### さらに学ぶための関連トピック
- [RollingHistoryBuffer (FIFO履歴)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/08_temporal-memory)
- [temporal attention-pool (時間方向集約)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/08_temporal-memory)
- [並列クエリ型JEPAデコーダ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [pooled特徴 vs 空間特徴マップのJEPAターゲット](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)


## FSDP (Fully Sharded Data Parallel)

### ひとことで言うと
[DDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) は全 GPU にモデルを丸ごとコピーするのでメモリを大量に使い、1 台に入らない大きなモデルでは学習できません。FSDP は「モデルの重み・勾配・optimizer の状態を GPU 間で分割 (shard) して保持し、計算で必要になった瞬間だけ全 GPU から集めてくる」ことで、1 台あたりのメモリを大幅に減らす手法です。これにより、本来なら 1 台に収まらない大規模モデルでも、複数 GPU を束ねれば学習できるようになります。FSDP はデータ並列の一種であり続けます (各 GPU は違うデータを処理する)。違うのは「重みを全コピーで持つか、分割で持つか」だけです。

### 直感的な理解
4 人で 1 冊の分厚い百科事典を使って調べ物をする状況を想像してください。DDP のやり方は「全員が同じ百科事典を 1 冊ずつ持つ」です。確実ですが、本棚 (メモリ) を 4 冊ぶん占有します。本がもっと分厚くなれば、各人の本棚に 1 冊すら入らなくなります。

FSDP のやり方は違います。「百科事典を 4 分割し、各人は 1/4 だけ持つ。今 A の項目を調べたい人がいたら、その瞬間だけ全員から A のページを集めて完全な状態を作り、調べ終わったらまた手放す」。普段は各人の本棚は 1/4 しか使わないので、4 倍分厚い本でも扱えます。集める手間 (通信) は増えますが、調べる項目を順番に処理していけば、一度に集めるのは「今調べている数項目分」だけで済みます。

ニューラルネットの学習も同じです。ある層を計算する瞬間だけその層の重みを全部そろえ、終わったら手放す。これを層ごとに繰り返せば、各 GPU が常時保持するメモリは「モデル全体の 1/台数」で済みます。鍵は「常時保持する量」と「一瞬だけ集める量」を分けて考えることで、後者を小さく保てる限り、前者は台数分だけ薄くできるという発想です。

### 基礎: 前提となる概念

#### 学習で消費するメモリの 3 要素
1 台の GPU が学習中に保持する主なものは次の 3 つです。重みの個数を Ψ (プサイ) と書きます。

1. パラメータ (重み): Ψ 個
2. 勾配: Ψ 個 (各重みに 1 つの更新方向)
3. optimizer 状態: 最適化器が追加で持つ補助変数。Adam の場合、各重みについて移動平均 m と分散 v の 2 つ、さらに混合精度学習では float32 のマスターコピーも持つため、パラメータの数倍にふくらむ

具体例: 重みが 1 億個・float32 (1 個 4 バイト) なら、パラメータだけで約 400MB。Adam では加えて勾配 400MB、optimizer 状態 (m, v で) 約 800MB、計 1.6GB。これに forward の中間 activation が乗ります。重みが 10 億・100 億と増えると、optimizer 状態だけで GPU の VRAM を超えます。

#### DDP の冗長性
DDP は全 GPU がこの 3 要素を「まるごと 1 セットずつ複製」します。4 台なら同じ optimizer 状態が 4 セット存在し、3 セットは完全に冗長です。台数を増やしても 1 台あたりのメモリは 1 ミリも減らず、扱えるモデルサイズの上限が変わりません。FSDP はこの冗長コピーを排除します。

#### 関連する集合通信
- all-gather: 各 GPU が持つ「重みの 1/N」を全部集めて、その層の完全な重みを一時的に作る
- reduce-scatter: 各 GPU が計算した勾配を合算しつつ、合算結果を分割して「各 GPU の担当 1/N だけ」を配る。DDP の all-reduce が「全員が全合計を持つ」のに対し、reduce-scatter は「全合計を切り分けて配る」点が違い、勾配も分散保持できる

実は all-reduce = reduce-scatter + all-gather と分解できます。DDP はこの 2 つを一体で行って全員が全勾配を持ち、FSDP は forward/backward で all-gather を使ってパラメータを一時復元し、勾配は reduce-scatter だけで止めて分散保持する、と理解すると両者の関係が見えます。

### 仕組みを詳しく

#### 1 層の forward の流れ (GPU 4 台)
1. 普段、ある層の重みは 4 分割され、各 GPU に 1/4 ずつ置かれている (定常メモリ = モデルの 1/4)
2. この層の forward 直前に all-gather で完全な重みを各 GPU に一時構築
3. 各 GPU が自分の担当データ (DDP と同じくデータも分割されている) に対してこの層の forward を計算
4. 計算が終わったら、一時的に集めた重みを即座に解放し 1/4 に戻す
5. 次の層へ。これを層ごとに繰り返す

backward も同様に、各層で all-gather により重みを集めて勾配を計算し、reduce-scatter で勾配を合算しつつ分散配置します。最後に各 GPU が「自分の担当 1/4 の optimizer 状態」を使って担当分の重みだけを更新します。注意すべきは、forward で all-gather したパラメータを backward でも再び all-gather する必要がある点で (forward 後に解放しているため)、これが FSDP の通信が DDP より多くなる根本理由です。

#### 通信と計算のオーバーラップ
FSDP も DDP と同様、通信を計算で隠します。具体的には、層 i を計算している最中に、次の層 i+1 の all-gather (prefetch = 先読みで集めておくこと) を裏で先に始めておきます。これにより all-gather の待ち時間を計算時間に重ねられます。backward では「逆方向の prefetch」、すなわち次に backward する層のパラメータを先に集めます。この prefetch が効くかどうかが FSDP のスループットを大きく左右し、prefetch を切ると通信が計算に隠れず大幅に遅くなります。

#### wrap の粒度 (sharding の単位)
「どのまとまりで分割し、どのタイミングで all-gather するか」を決めるのが wrap の粒度です。FSDP では分割の最小単位を「FSDP unit」と呼び、unit ごとに all-gather/解放が起こります。

- 細かく wrap (各層を個別に unit にする): all-gather する単位が小さく、一時メモリは少ないが通信回数が増える
- 粗く wrap (複数層をまとめて 1 unit にする): 一時メモリは増えるが通信回数が減りオーバーラップしやすい

Transformer なら「各ブロックを 1 unit として wrap」するのが定番です。粒度はメモリと通信のトレードオフを決める主要なチューニング項目で、`auto_wrap_policy` で「パラメータ数がしきい値を超えたら自動で unit を区切る」といった指定ができます。

#### sharding strategy の段階
何を分割するかには段階があり、これは [ZeRO](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) のステージと対応します。

- FULL_SHARD: パラメータ + 勾配 + optimizer 状態すべてを分割。最もメモリ効率が高い (ZeRO Stage 3 相当)
- SHARD_GRAD_OP: 勾配と optimizer 状態だけ分割し、パラメータは複製 (ZeRO Stage 2 相当)。パラメータを集め直す all-gather が要らない分だけ通信は減るが、パラメータのメモリは削れない
- HYBRID_SHARD: ノード内では FULL_SHARD、ノード間では複製。遅いノード間通信 (all-gather/reduce-scatter) を減らしつつメモリも削る折衷で、multi-node で実用的

#### メモリ削減の数値イメージ
重み 1 億個・Adam・GPU 4 台。DDP では各 GPU が optimizer 状態 800MB を持ちますが、FSDP FULL_SHARD では 800MB ÷ 4 = 200MB。パラメータ・勾配も同様に 1/4。一時的に集めるのは「同時に計算する 1〜数 unit 分」だけなので小さく済みます。台数 N を増やすほど 1 台あたりの定常メモリは 1/N に近づきます。

```
DDP (各GPU):    [param 400MB][grad 400MB][opt 800MB]  = 1.6GB 固定 (台数で減らない)
FSDP (各GPU):   [param 100MB][grad 100MB][opt 200MB]  = 0.4GB + 一時 all-gather 分
                 ↑ N=4 で各要素 1/4
```

ただし「一時 all-gather 分」を忘れてはいけません。一度に集める unit の最大サイズが大きいと、定常メモリが小さくてもピークでそこに乗り、OOM します。粒度のチューニングはこのピークを抑えるためでもあります。

#### 典型的な実装
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=transformer_wrap_policy,   # どの単位で wrap するか
)
# 以降は DDP とほぼ同じ。loss.backward(); optimizer.step()
```
新しい PyTorch では `fully_shard` (FSDP2) という per-parameter sharding の API も提供され、より柔軟な分割が可能になっています。チェックポイントの保存・復元では、分散して持つ重みを 1 つの完全な state_dict に集約する full state dict か、各 rank が自分の shard だけ保存する sharded state dict かを選べ、巨大モデルでは後者が現実的です。

### 手法の系譜と主要論文

- Rajbhandari ら, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020, Microsoft, arXiv:1910.02054)。FSDP の理論的源流。optimizer 状態・勾配・パラメータの冗長保持を段階的に排除する ZeRO ステージを提案しました。詳細は [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。

- Rasley ら, "DeepSpeed" (KDD 2020, Microsoft)。ZeRO を実装したライブラリで、PyTorch ネイティブ FSDP 登場以前から大規模分散学習の事実上の標準でした。FSDP はしばしば DeepSpeed ZeRO Stage 3 と比較されます。

- FairScale FSDP (Meta, 2021)。PyTorch ネイティブ FSDP の前身となる実験的実装で、ZeRO Stage 3 のアイデアを PyTorch の autograd と統合する初期の試みでした。ここでの知見が後に PyTorch コアへ取り込まれます。

- Zhao, Gu, Varma ら, "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (VLDB 2023, Meta, arXiv:2304.11277)。本トピックの中心論文。ZeRO のアイデアを PyTorch コアに統合し、(1) deferred initialization (巨大モデルを物理メモリに乗せずにメタデバイス上で初期化してから分割配置)、(2) 通信と計算のオーバーラップ・prefetch、(3) 混合精度との統合、(4) 柔軟な wrap ポリシーを実装したことを報告。数千億パラメータ規模を限られた GPU メモリで学習可能にしました。

系譜としては「DDP (冗長だが単純) → ZeRO/DeepSpeed (冗長を消すアイデアと実装) → FairScale FSDP (PyTorch への移植実験) → PyTorch FSDP (フレームワークネイティブに、使いやすく) → FSDP2 (per-parameter で更に柔軟に)」という流れです。

### 論文の実験結果 (定量データ)

PyTorch FSDP 論文 (VLDB 2023) の主な実測です。指標は主に TFLOPS/GPU (1 GPU が 1 秒に実行する浮動小数点演算数で、計算資源をどれだけ使い切れているかを表す。理論ピークに近いほど効率がよい) と、学習可能な最大モデルサイズです。

- A100 GPU クラスタで、GPT 系の大規模モデルを学習し、175B (1750 億) パラメータ級まで含む広いサイズ域で高い演算効率を達成。報告では多くの設定で 100+ TFLOPS/GPU 台、条件次第で A100 の理論ピーク (bf16 で約 312 TFLOPS) の 5〜6 割程度の利用率を維持しました。これは「メモリを削りつつも計算機を遊ばせていない」ことを意味します。
- スケーラビリティ: モデルサイズと GPU 数を同時に増やしても、TFLOPS/GPU がほぼ一定に保たれる (near-constant) 領域が示され、weak scaling (規模に応じて資源も増やす) が良好であることを実証。数百 GPU 規模まで効率の崩れが小さいことが報告されています。
- DeepSpeed ZeRO Stage 3 との比較で、同等のメモリ効率を PyTorch ネイティブに、より少ない外部依存で実現できることを示しました。

アブレーション的な知見として重要なのは prefetch と wrap 粒度の効果です。all-gather の prefetch を無効にすると通信が計算に隠れず TFLOPS/GPU が顕著に低下します。また wrap 粒度を細かくしすぎると通信回数増で遅くなり、粗くしすぎると一時メモリがふくらんで OOM します。「通信を隠す工夫を抜くと効率が崩れる」ことが、FSDP がただの分割ではなくシステム最適化の塊であることを示しています。さらに HYBRID_SHARD は、ノード間帯域が細い環境で FULL_SHARD よりスループットが上がる場合があることも報告され、「メモリ最小化が常に最速ではない」という実務上重要な教訓を与えています。

指標の意味の補足: TFLOPS/GPU は速度の絶対量ではなく「GPU の計算能力をどれだけ使い切ったか」の指標です。同じ TFLOPS/GPU を大規模でも保てるなら、台数とモデルを増やしても効率が落ちないということで、これが大規模学習で最も重視される性質です。MFU (Model FLOPs Utilization、理論ピークに対する実効の割合) という形で報告されることも多く、大規模 LLM 学習では 40〜60% 程度が良好とされます。

### メリット・トレードオフ・限界

メリット
- パラメータ・勾配・optimizer 状態を分割するため、1 台あたりのメモリがおおむね 1/台数 に減る
- DDP では 1 台に入らない大規模モデルでも学習できる
- PyTorch ネイティブで、[混合精度](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) や [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning) と自然に組み合わせられる (FSDP がパラメータ系を削り、checkpointing が activation を削る、と役割分担)

トレードオフ・限界
- all-gather / reduce-scatter の通信が DDP の all-reduce より多く (forward と backward の両方でパラメータを集め直す)、ネットワークが遅い環境ではスループットが落ちる。特にノードをまたぐ all-gather は帯域が細く効率が下がる (HYBRID_SHARD で緩和)
- wrap 粒度・prefetch・sharding strategy などチューニング項目が多く、DDP より導入が複雑
- 小〜中規模モデルでは、メモリに余裕があるのに通信オーバーヘッドだけ増え、DDP より遅くなる (大モデルでこそ効く)
- state_dict (重みの保存・復元) の扱いが分散しているため複雑。チェックポイントを full state にまとめる際の集約コストや、再開時のシャーディング整合に注意が要る

研究上の未解決課題: 「メモリは限界まで削れるが通信が増える」というトレードオフの最適点をどう自動で見つけるかは依然難しく、wrap ポリシーや並列構成の自動探索は研究途上です。また FSDP (データ並列の拡張) 単独では超大規模に届かず、テンソル並列・パイプライン並列との組み合わせ (3D parallelism) が必要になり、その構成探索も難問です。

### 発展トピック・研究の最前線
- FSDP2 (per-parameter sharding): パラメータを flat な 1 本のバッファにまとめて分割する旧方式に対し、パラメータ単位で分割し、量子化・部分凍結 (一部の層だけ学習) や LoRA との相性を改善
- ハイブリッド並列: FSDP × テンソル並列 × パイプライン並列。各並列軸を直交させて超大規模 LLM を学習する 3D parallelism。FSDP をテンソル並列の外側に置く構成が一般的
- 通信圧縮・量子化通信: all-gather する重みを低ビット (例: fp8) で送って通信量を削る試み。ZeRO++ の発想と共通
- offload との併用: パラメータや optimizer 状態を CPU/NVMe に逃がし、さらに大きいモデルを少ない GPU で扱う ([ZeRO-Offload/Infinity](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の発想)。FSDP も CPU offload オプションを持つ

### さらに学ぶための関連トピック
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## 学習率 Warmup

### ひとことで言うと
Warmup(ウォームアップ)は、学習のいちばん最初だけ、学習率(1 回の更新で重みを動かす大きさ)をゼロに近い値から目標値まで徐々に上げていく工夫です。いきなり大きな学習率で始めると学習が壊れる(発散する)ことがあるため、最初の数%のステップで「準備運動」をさせて安定させます。大きなバッチや Transformer の学習ではほぼ必須とされる手法です。それ単体では学習率を下げないので、必ず後段の減衰スケジュール(コサインなど)と組み合わせて使います。

### 直感的な理解
冷えたエンジンをいきなり全開で回すと壊れることがあるので、暖機運転してから回転を上げます。warmup はこれと同じです。学習開始直後のモデルは「冷えたエンジン」、つまりまだランダムに近い重みを持つ最も壊れやすい状態にあります。この時期に大きな学習率で更新すると、信用できない方向に大股で進んで損失が跳ね上がり、発散することがあります。

なぜ初期が危ないのか。学習開始直後の勾配(損失を減らす方向の傾き)はノイズだらけで方向が信用できません。さらに、後で詳しく述べますが、Adam のような適応的オプティマイザは「過去の勾配の統計」を使って各パラメータの学習率を内部調整しますが、開始直後はその統計サンプルが少なすぎて推定が暴れます。これらが重なって、初期に高学習率を当てると一気に崩れるのです。warmup は「最初だけ学習率を抑えて、勾配の統計が安定してきてから本来の学習率に上げる」ことでこの危険な区間を安全にやり過ごします。

### 基礎: 前提となる概念
学習率とは、勾配降下法における 1 歩の歩幅です。重み W を勾配 g で更新するとき W ← W − η·g とし、この η が学習率です。大きいほど速く動けますが、危険です。

バッチサイズとは、1 回の更新でまとめて処理する訓練データの枚数です。これを大きくすると 1 ステップで多くのデータから平均勾配を取れるため学習を高速化できますが、後述する線形スケーリング則により学習率も比例して大きくしたくなり、初期の発散リスクが増します。

適応的オプティマイザ(Adam など)とは、各パラメータごとに学習率を自動調整するオプティマイザです。具体的には、過去の勾配の 2 乗の移動平均(分散の推定 v_t)で勾配を割ることで、勾配が大きい方向は控えめに、小さい方向は大胆に更新します。問題は、この分散推定が学習開始直後はサンプル不足で不安定なことです。割る値 √v_t が小さく振れると実効学習率 η/√v_t が一時的に巨大化し、これが初期発散の主因になります。

線形スケーリング則(Goyal et al.)とは、「バッチサイズを k 倍にしたら学習率も k 倍にすると、学習の進み方がだいたい揃う」という経験則です。大バッチ学習の高速化に不可欠ですが、その大きくした学習率を初期にそのまま当てると発散するため、warmup とセットで使われます。

正規化の置き方(Post-LN / Pre-LN)も後で効いてきます。Transformer の各サブ層(自己注意や全結合)に対し、層正規化(LayerNorm)をサブ層の後に置くのが Post-LN、前に置くのが Pre-LN です。この置き方で初期の勾配の安定性が変わり、warmup の要否に直結します。

### 仕組みを詳しく
warmup は「最初の W ステップで学習率を η_start(0 か非常に小さい値)から目標 η_max まで上げる」処理です。最も一般的な線形 warmup の式は次のとおりです。

```
warmup 中 (t ≤ W): η(t) = η_max · (t / W)
warmup 後 (t > W): 本来のスケジュール(例: cosine decay)に従う
```

記号の意味。t は現在のステップ、W は warmup の長さ(ステップ数)、η_max は到達させたい目標学習率です。`t / W` は warmup の進捗(0 から 1 へ)を表し、それを η_max に掛けることで学習率を線形に立ち上げます。

数値例で見ます。η_max=0.001、W=2000 ステップとすると、

- t=0: η = 0.001·(0/2000) = 0(実質ゼロ。ここから始める)。
- t=500: η = 0.001·(500/2000) = 0.00025。
- t=1000: η = 0.001·(1000/2000) = 0.0005。
- t=2000: η = 0.001·(2000/2000) = 0.001(目標に到達)。
- t>2000: ここから cosine decay などで下げていく。

つまり学習率の時間変化は「最初に斜めに立ち上がる三角の入り → ピーク到達 → あとはゆっくり減衰」という山型になります。warmup と [Cosine Annealing 学習率スケジュール](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) を組み合わせた典型的な形は次のようになります。

```
η
η_max |      .-*-._
      |    .'      `-.
      |  .'            `-._
      | /                  `--._
    0 |/__________________________ → t
      0   W                      T
       warmup     cosine decay
```

設計の勘どころは「warmup の長さ W」です。総ステップの 1〜10% 程度がよく使われます(例: 10 万ステップなら 1000〜5000)。大規模言語モデルでは固定ステップ数(例 2000 ステップ)で warmup する流儀も一般的です。バッチが大きい・学習率が高い・モデルが深いほど、長めの warmup が安全側です。立ち上げ方は線形が定番ですが、ゆるやかな曲線(指数・多項式)を使う実装もあります。Transformer の原論文は逆平方根スケジュールの一部として warmup を組み込み、warmup_steps の前は線形に上げ(η ∝ step·warmup_steps^{-1.5})、後は 1/√step で下げる形を使いました。

なぜ効くかをもう一歩深掘りします。Adam 系では「初期の勾配分散推定が不安定 → 実効学習率が暴れる」のが発散の主因の一つです。warmup はこの不安定な区間を低学習率でやり過ごすことで安定させます。この理解を理論として精緻化したのが RAdam(下記)で、warmup の役割をオプティマイザ側に内蔵しました。もう一つの理解として、Transformer のアーキテクチャ側の事情があります。Post-LN(各サブ層の後に正規化を置く構成)の Transformer は初期に勾配が層をまたいで増幅されやすく warmup が必須ですが、Pre-LN(サブ層の前に正規化を置く構成)に変えると勾配が安定し warmup の必要性が下がる、という分析(Xiong et al., 2020)もあります。つまり warmup の要否はアーキテクチャ設計とも結びついています。さらに別の視点として、warmup は「鋭い損失領域に序盤で捕まらず、より平坦で汎化しやすい領域へ初期解を導く暗黙の正則化」として働くという観察(損失地形の鋭さ sharpness を warmup が下げるという報告)もあります。

### 手法の系譜と主要論文
- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (2017, Facebook, arXiv:1706.02677)。核心は線形スケーリング則ですが、それを安定させる鍵として「最初の数エポックの gradual warmup(徐々に上げる warmup)」を導入しました。動機は、大学習率で始めると初期に発散するため。warmup を広く知らしめた代表的論文です。

- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017, arXiv:1706.03762)。Transformer の学習に warmup(最初に上げてから逆平方根で下げる)を採用。Transformer は warmup なしだと学習初期に不安定で発散しやすいため。以後、Transformer 系学習で warmup は事実上の必須になりました。

- Gotmare et al., "A Closer Look at Deep Learning Heuristics" (ICLR 2019, arXiv:1810.13243)。warmup・コサイン・知識蒸留などの経験則を、層ごとの重み変化の解析を通じて実証的に調べました。warmup が初期の大きな重み更新を抑えて学習を安定させることを定量的に示した位置づけです。

- Liu et al., "On the Variance of the Adaptive Learning Rate and Beyond (RAdam)" (ICLR 2020, arXiv:1908.03265)。warmup がなぜ効くかを「Adam の初期の適応学習率(実効学習率)の分散が大きすぎるから」と分析し、その補正をオプティマイザに内蔵した RAdam を提案。手動 warmup なしでも安定すると示しました。warmup の理論的理解を深めた研究です。

- Xiong et al., "On Layer Normalization in the Transformer Architecture" (ICML 2020, arXiv:2002.04745)。Post-LN と Pre-LN の勾配挙動を理論・実験で比較し、Pre-LN なら初期勾配が安定して warmup を省略・短縮できることを示しました。warmup を「アーキテクチャ設計で吸収できる」方向の研究です。

### 論文の実験結果(定量データ)
Goyal et al. の実験は ImageNet(約 128 万枚、1000 クラス)で ResNet-50 を学習するものです。バッチサイズ 8192(通常の数十倍)という大バッチで分散学習を行い、線形スケーリング則 + gradual warmup を用いることで、小バッチで学習した場合とほぼ同じ top-1 accuracy(約 76%)を保ったまま、学習を 256 個の GPU で約 1 時間に短縮したと報告しました。指標の意味を補足すると、ImageNet top-1 で 1% の精度差は大きく、「精度を落とさず 1 時間」という点がこの論文の核心です。アブレーションがはっきりしており、warmup を外して大バッチ・大学習率で始めると初期に学習が崩れ精度が大幅に劣化することが示されています。彼らは constant warmup(最初だけ一定の低学習率)より gradual warmup(徐々に上げる)の方が有効だとも報告しており、立ち上げ方そのものが重要であることを示しました。つまり大バッチ高速学習を成立させているのは線形スケーリングと warmup の組み合わせです。

RAdam 論文では、画像分類と言語モデルの両方で、warmup なしの素の Adam は初期に損失が悪化し最終性能も不安定になる一方、RAdam(内蔵 warmup)や明示的 warmup 付き Adam は初期から安定して学習が進むことを示しました。学習率を変えたときの感度(robustness)が高い、すなわち学習率を多少間違えても壊れにくい点も報告されています。アブレーションとして、warmup 区間を長くするほど初期の発散は確実に防げるが、長すぎると本格学習の開始が遅れて到達精度が頭打ちになる、というトレードオフが観察されています。

Xiong et al. の Transformer 実験では、Pre-LN 構成にすると warmup を大幅に短縮または省略しても機械翻訳タスク(BLEU スコアで評価。翻訳の正確さを測る指標で大きいほど良い)が安定して学習でき、Post-LN は warmup を外すと発散することが定量的に示されました。具体的には、Pre-LN は warmup なし・大学習率でも収束し学習を高速化できる一方、Post-LN は warmup ステップ数を十分取らないと初期勾配が爆発して学習が立ち上がらない、という対比が示されています。これは「warmup は万能のおまじないではなく、アーキテクチャの初期勾配の性質に応じて必要量が決まる」という理解を裏付けます。

### メリット・トレードオフ・限界
メリット。学習初期の発散を防ぎ、特に大バッチ・高学習率・Transformer で安定します。Adam 系の初期の実効学習率の暴れを低学習率区間でやり過ごせます。実装が簡単(数行)で、ほとんどのスケジューラと自由に組み合わせられます。

トレードオフと限界。warmup の長さ W という追加のハイパーパラメータが増えます。短すぎると発散を防げず、長すぎると本格的な学習開始が遅れて時間を損します。小バッチ・低学習率・浅いモデルでは効果が薄く不要なこともあります。warmup 単体では学習率を下げないため、必ず後段の減衰スケジュールと組み合わせる必要があります。さらに、Pre-LN のように初期勾配が安定するアーキテクチャでは warmup の必要性自体が下がるため、「とりあえず warmup を入れる」のではなく、なぜ必要かを理解して長さを決めることが望ましいです。

研究上の未解決課題としては、最適な warmup 長を事前に予測する理論(現状は経験則に強く依存)、warmup の役割をどこまでオプティマイザやアーキテクチャ側に吸収できるか、低精度学習(bf16 など)での初期数値安定性と warmup の関係、が挙げられます。

### 発展トピック・研究の最前線
warmup は単独で語られるより、warmup + cosine、warmup + 線形 decay、あるいは Warmup-Stable-Decay(WSD: warmup したあと学習率を一定に保ち、最後だけ急減衰させる新しい大規模言語モデル向けスケジュール)の「先頭部品」として組み込まれる方向に進んでいます。RAdam のようにオプティマイザへ内蔵する流れ、Pre-LN や正規化手法(DeepNorm など、極端に深い Transformer を warmup を緩和しつつ安定化する手法)の改良でそもそも warmup を不要・短縮する流れ、学習率と並んで weight decay や勾配クリッピングをまとめて初期安定化の枠組みで扱う研究などが活発です。低精度・大規模分散学習が標準になるにつれ、初期数値安定性の確保手段としての warmup の重要性はむしろ増しています。

### さらに学ぶための関連トピック
- [Cosine Annealing 学習率スケジュール](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [One-Cycle 学習率ポリシー](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [bf16 混合精度学習 (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)


## Weight Decay (L2 正則化)

### ひとことで言うと
ニューラルネットの「重み」(モデルの中にある無数の数値パラメータ)が大きくなりすぎないように、学習中にじわじわと小さく引っ張る仕組みです。これによってモデルが訓練データを丸暗記する「過学習」を抑え、見たことのないデータでも良い予測ができるようになります。実装はほぼタダ(最適化手法に引数を1つ渡すだけ)で、ほとんどの深層学習で標準的に使われます。

### 直感的な理解
試験勉強にたとえます。賢い勉強法は「本質的なパターンを少数の原則で理解する」ことです。一方ダメな勉強法は「過去問の答えを一字一句丸暗記する」ことで、問題が少し変わると太刀打ちできません。

ニューラルネットも放っておくと丸暗記に走ります。重みの一部に極端に大きな値(たとえば `w = 350.0`)を割り当てると、入力のごくわずかな違いに過敏に反応して、訓練データ1つ1つの細かいノイズまで覚え込めてしまうからです。Weight Decay は「使う重みは小さく保て、本当に必要なときだけ大きくしてよい」という圧力を常にかけ続け、モデルを「少数の本質的なパターンで説明する、シンプルな解」へ誘導します。シンプルな解ほど、新しいデータにも素直に当てはまる傾向があります。これがオッカムの剃刀(同じ説明力なら単純な方が良い)の数値版です。

自動運転のような連続値予測では、これが安全性に直結します。重みが暴れていると、カメラ画像の数ピクセルの明るさの違いや、訓練データに偶然なかった影の形に過敏反応して、出力(ハンドル角など)が大きくぶれます。重みを小さく保てば出力は入力の小さな変化になめらかに追従し、過敏さが減ります。

### 基礎: 前提となる概念
- 重み (weight): ニューラルネットは「入力に数値を掛けて足す」を大量に繰り返す計算です。掛ける数値が重みです。`出力 = w1*入力1 + w2*入力2 + ...` の `w1, w2` が重み。学習とはこの重みを良い値に調整する作業です。

- 過学習 (overfitting): 訓練に使ったデータには完璧に答えられるのに、新しいデータではまるでダメになる現象。

- 汎化 (generalization): 見たことのないデータに対しても正しく動く能力。これが機械学習の本当のゴールです。訓練誤差ではなくテスト誤差を下げることが目的です。

- 正則化 (regularization): 訓練誤差を下げるのとは別に、「モデルを単純に保つ」制約を課して汎化を助ける手法の総称。Weight Decay はその代表で、ほかに [Dropout](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、データ拡張([Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) など)、[Stochastic Depth (DropPath)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) などがあります。

- 損失 (loss): 予測がどれだけ間違っているかを測る数値。学習はこの損失を小さくするよう重みを更新します。

過学習が起きる仕組みをもう一段。重みが大きいと、その重みが掛かる入力方向にモデルの出力が鋭く反応します。多数の大きな重みが組み合わさると、訓練点のあいだで暴れる「ギザギザの関数」が作れてしまい、ノイズまで再現できます。重みを小さく保つと関数がなめらかになり、ノイズに引っ張られにくくなります。

### 仕組みを詳しく
通常、モデルは「データの損失」を小さくするように重みを更新します。Weight Decay は、この損失に「重みの大きさそのもの」のペナルティ項を足します。

L2 正則化の式(古典的な定義):

```
全体の損失 = データの損失 + (λ/2) * (w1^2 + w2^2 + ... + wn^2)
```

- `λ`(ラムダ)は正則化の強さを決めるハイパーパラメータ。大きいほど重みを強く小さくします。典型値は `1e-2`(0.01)、`5e-2`(0.05)、`1e-4`(0.0001)など。
- 重みの2乗を足しているので、大きい重みほど急激にペナルティが増えます。`w=10` なら 2乗で 100、`w=0.1` なら 0.01。だから大きい重みが優先的に縮められます。

このペナルティ項を重みについて微分すると、勾配に `λ*w` が加わります。更新式に展開すると、更新のたびに重みに `(1 - 学習率*λ)` を掛ける操作になります。つまり毎ステップ、重みがほんの少し(たとえば 0.99999 倍に)縮みます。これが「decay(減衰)」と呼ばれる理由です。

数値例で更新の流れを追います。学習率 `lr=0.001`、`λ=0.01`、ある重み `w=2.0` の場合:

1. データの損失から来る勾配で更新(仮に重みを 0.005 増やす方向だったとする): `w = 2.0 + 0.005 = 2.005`
2. Weight Decay の効果で縮める: `w = w * (1 - lr*λ) = 2.005 * (1 - 0.001*0.01) = 2.005 * 0.99999 ≈ 2.00498`

1ステップではわずかですが、数万ステップ積み重なると、データから強く必要とされない重みは着実にゼロへ近づきます。逆に「予測に本当に必要な重み」は、データの損失がそれを大きく保とうと押し返すので、ゼロにはなりません。結果として「必要なものだけ残る、シンプルなモデル」に整います。釣り合いの式で言えば、データ損失の勾配 `g` と decay の引き戻し `λ*w` が拮抗する点 `g ≈ λ*w` で重みが落ち着きます。

ベイズ的な解釈も押さえると深まります。「重みは 0 付近に集まっているはずだ」という事前分布(ガウス分布)を仮定して最尤推定すると、ちょうど L2 ペナルティ付きの損失が出てきます。つまり Weight Decay は「重みはたぶん小さい」という事前知識を学習に注入していると見なせます(Goodfellow et al. 2016, 7章)。

#### L2 正則化と Weight Decay の微妙な違い(重要)
「L2 正則化」と「Weight Decay」は同じものとして語られますが、最適化手法によっては別物になります。

- 単純な SGD(確率的勾配降下法)では両者は数学的に等価です。損失に L2 項を足しても、更新式で重みを直接縮めても、同じ結果になります。
- ところが Adam のような「勾配を変数ごとにスケーリングする」最適化手法では、損失に L2 項を足す方式だと、スケーリングの影響でペナルティが歪みます。具体的には、勾配が大きくよく動く重みほど L2 の効きが弱まり、本来縮めたい重みが縮みにくくなります。Loshchilov & Hutter(2019)はこれを問題視し、ペナルティを損失に混ぜるのではなく、重み更新の最後で直接 `w = w*(1 - lr*λ)` と縮める「分離した(decoupled)Weight Decay」を提案しました。これが AdamW です。

この区別は現代の学習で実務上きわめて重要で、Adam 系を使うなら AdamW(分離型)を選ぶのが標準的な推奨になっています。

### 手法の系譜と主要論文
- 起源。Hanson & Pratt, "Comparing Biases for Minimal Network Construction with Back-Propagation", NIPS 1988。重みを小さく保つペナルティがネットワークを単純化し汎化を助けることを早期に示しました。

- 理論的定式化。Krogh & Hertz, "A Simple Weight Decay Can Improve Generalization", NIPS 1991。提案は、学習に重みの2乗ペナルティを加えると汎化が改善することの理論分析。動機は、大きい重みが入力ノイズを増幅するため、それを抑えると予測が安定するという考え。効果は、過学習が減りテスト誤差が下がること。トレードオフは、`λ` が強すぎると重みが縮みすぎてデータに必要な表現力まで失い「未学習(underfitting)」になること。

- 正則化全般への位置づけ。Goodfellow, Bengio & Courville, "Deep Learning"(2016)7章。Weight Decay をベイズ事前分布の観点・パラメータ空間の制約の観点から整理し、dropout やデータ拡張など他の正則化と並べて体系化しました。

- 現代化。Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019(AdamW, arXiv:1711.05101)。提案は、Adam に L2 項を足す従来方式は Weight Decay として正しく働かないことを指摘し、Weight Decay を勾配計算から切り離して直接重みを縮める方式(AdamW)。動機は、Adam が勾配を成分ごとにスケーリングするため L2 ペナルティの効きが不均一になること。効果は、画像分類などで Adam より一貫して汎化が改善し、学習率と Weight Decay を独立に調整しやすくなったこと。トレードオフは、`λ` のチューニングは依然必要で最適値がデータ・モデル規模に依存すること。AdamW は今や Transformer / CNN 学習のデファクト最適化手法です。

### 論文の実験結果(定量データ)
- AdamW(ICLR 2019)。CIFAR-10(10 クラス・6 万枚の画像分類)で、従来の Adam(L2 を損失に足す方式)に比べ、AdamW(分離型)はテスト誤差を一貫して低減しました。重要なアブレーションは「学習率と weight decay の探索しやすさ」で、Adam では両ハイパーパラメータが強く絡み合い最適点が斜めに伸びるのに対し、AdamW では2軸がほぼ独立になり、探索が格段に楽になることを示しました。これは「最適な λ を見つけやすくなる」という実務上大きな利点です。ImageNet 系でも AdamW は Adam より良い汎化を示し、以後の Vision Transformer 学習の標準になりました。

- 代表的なモデルでの設定値。Vision Transformer や Swin Transformer、ConvNeXt の元論文では、weight decay を 0.05 前後に設定するのが定番です。一方 CNN を SGD で学習する古典的設定では 1e-4 程度が定番で、最適化手法とモデルによって2桁以上違う値が使われます。これは「λ は普遍的な値がなく、設定全体に依存する」ことの具体例です。

- 何にかけるかのアブレーション。多くの実験で、バイアス項や正規化層(LayerNorm / BatchNorm)のスケール・シフトパラメータには weight decay をかけない方が良いと報告されています。これらは「大きさ」が意味を持つパラメータなので、ゼロへ引っ張ると表現力が落ちるためです。実務では「重み行列にはかけ、バイアスと正規化層パラメータには除外する」パラメータグループ分けが標準です。これを怠ると、ImageNet 級で数十分の1ポイント〜1ポイント規模の精度低下が出ることがあります。

- 古典的な定量例(Krogh & Hertz 1991)。小規模ネットワークでの解析・実験で、weight decay を加えるとテスト誤差が下がり、適切な `λ` のもとで汎化誤差が最小化されることを示しました。彼らの定式化では、weight decay の最適強度はデータのノイズレベルと訓練サンプル数に依存し、データが少なくノイズが多いほど強い decay が有利になるという、いまの大規模学習にも通じる傾向を早期に明らかにしています。

- decay の強さに対する U 字。`λ` を 0 から増やしていくと、テスト誤差はいったん下がり(過学習が抑えられる)、ある点を越えると再び上がる(必要な表現力まで削られて未学習に転じる)という U 字カーブを描きます。最適点はデータ量・モデル規模・学習率に依存し、普遍的な値は存在しません。この U 字の底を探すのがチューニングの実体で、AdamW が「学習率と λ をほぼ独立に動かせる」ようにしたことの実務的価値は、この底を 2 次元グリッドで効率よく探せる点にあります。

指標の意味。テスト誤差(test error)は「学習に使っていないデータでの間違いの割合」で、汎化性能そのものです。weight decay の効果は訓練誤差ではなくこのテスト誤差の改善として現れる点が肝心で、「訓練誤差は同じか少し悪くても、テスト誤差が下がる」のが正則化が効いているサインです。

### メリット・トレードオフ・限界
メリット
- 実装が極めて簡単(最適化手法に引数1つ)で、計算コストはほぼゼロ。
- 過学習を抑え、テスト誤差が安定して下がることが多い。
- 重みを小さく保つことで学習が数値的に安定しやすい(勾配爆発の抑制にも寄与)。

トレードオフ・限界
- `λ` を強くしすぎると未学習になり、訓練誤差すら下がらなくなる。
- 最適な `λ` はデータセット・モデル・学習率と絡み合うため探索コストがかかる(SGD では特に学習率と密結合。AdamW で緩和される)。
- バイアス項や正規化層パラメータには通常かけない方が良く、「どのパラメータにかけるか」の取捨選択が要る。
- Weight Decay 単体では強力な正則化にならず、dropout やデータ拡張と併用するのが普通。
- 研究上の未解決点として、Weight Decay が正規化層と組み合わさったときの「有効学習率」への影響(重みノルムが小さくなると正規化後の勾配が相対的に大きくなる効果)は完全には理解されておらず、近年も解析が続いています。

### 発展トピック・研究の最前線
- AdamW と learning rate の解離。Weight decay は正規化層と相互作用して「有効学習率」を変えることが知られ(Zhang et al., van Laarhoven 2017 など)、見かけの λ と実際の正則化強度がずれる現象が研究されています。
- L1 正則化と疎性。重みの絶対値(2乗でなく)をペナルティにすると、重みがちょうど 0 になりやすく、モデルが疎(sparse)になります。プルーニング(枝刈り)と結びつきます。
- 層ごと・パラメータグループごとの decay。バックボーンと予測ヘッドで別の λ を使う、層が深いほど decay を変えるなど、きめ細かい設計が大規模モデルで使われます。
- スケール不変性の観点。正規化層を持つネットでは重みのスケールが出力に効かないため、weight decay の役割が「過学習抑制」から「有効学習率の調整」へ意味が変わる、という解釈が提案されています。

### さらに学ぶための関連トピック
- [Dropout](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Stochastic Depth (DropPath)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [重みの指数移動平均 (EMA of Weights)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Adadelta

### ひとことで言うと
Adadelta(アダデルタ)は [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の弱点「学習が進むと実効学習率がどんどん0に近づいて止まってしまう問題」を直すために作られたオプティマイザです。過去の勾配を「全部足し続ける」のではなく「最近のものほど重視する移動平均」にすることで学習率が消えないようにし、さらに人間がグローバル学習率(全体に共通の学習率)を最初に決めなくても動くように設計されています。

### 直感的な理解
Adagrad は「過去の頑張りを全部記憶する」タイプの人だと思ってください。働けば働くほど疲れがたまり、やがて動けなくなります(=学習率が0に消える)。Adadelta は「最近の頑張りだけを覚えて、古いものは忘れる」タイプです。最近サボっていればまた元気に動けるし、最近働きすぎていれば自然にペースを落とす。だから永遠に止まることがありません。

もう一つの工夫は「自分のこれまでの歩幅を基準に次の歩幅を決める」ことです。人間が「学習率はこれくらい」と外から指定しなくても、過去にどれだけ動いてきたかを見て自分で歩幅を決めます。これがグローバル学習率を不要にする発想です。歩幅の単位そのものを物理的に整合させる、という少し変わった着眼点から生まれています。

### 基礎: 前提となる概念

前提の言葉を確認します(詳しくは [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照)。

- 勾配(gradient): パラメータをどっちに動かせば誤差が減るかを示す数字。記号 g。
- 学習率(learning rate): 1回の更新でパラメータをどれくらい動かすかの係数。記号 η。
- 移動平均(moving average): 過去の値を平均するとき、古い値を少しずつ忘れて新しい値を重視する平均の取り方。「指数移動平均(EMA、exponential moving average)」がよく使われ、減衰係数 ρ(ロー)を使って `新EMA = ρ×旧EMA + (1-ρ)×新しい値` と更新します。ρ が 1 に近いほど過去を長く覚えます。
- RMS(root mean square、二乗平均平方根): 値を二乗して平均し平方根を取ったもの。`RMS[g] = sqrt(E[g^2] + ε)`。符号に関係なく「典型的な大きさ」を表します。
- 単位(unit / dimension)の整合: 物理で「速度は m/s、加速度は m/s^2」のように量には単位があり、足し算は同じ単位どうしでないと意味をなさない、という考え方。最適化でもパラメータ θ と更新量 Δθ は本来同じ単位であるべき、というのが Adadelta の出発点です。

Adagrad の学習率消失問題を改めて確認します。Adagrad は各パラメータについて「過去の勾配二乗を全部足し続ける」変数 G を持ち、学習率を `1/sqrt(G)` で割ります。足し続けるだけなので G は単調に増え続け、実効学習率が時間とともに0へ向かいます。

```
学習率 0.1、毎ステップ勾配の絶対値 1.0 のパラメータ:
ステップ 1:     G = 1       実効学習率 = 0.1 / sqrt(1)     = 0.1
ステップ 100:   G = 100     実効学習率 = 0.1 / sqrt(100)   = 0.01
ステップ 10000: G = 10000   実効学習率 = 0.1 / sqrt(10000) = 0.001
```

何も悪さをしていなくても、ただ時間が経つだけで学習率が下がり続けます。深いネットワークを長時間学習すると、必要な学習が終わる前に止まってしまうのが致命的でした。

### 仕組みを詳しく

Adadelta の工夫は2段階です。

#### 工夫1: 勾配二乗を「移動平均」にする(RMSProp と同じ考え)

足し算ではなく、減衰係数 ρ(例: 0.9 や 0.95)を使った指数移動平均で勾配二乗を覚えます。

```
E[g^2]_t = ρ × E[g^2]_(t-1) + (1-ρ) × (g_t)^2
```

- E[g^2] は「最近の勾配二乗の平均」。古い勾配は ρ 倍ずつ薄まっていずれ忘れられます。
- ρ = 0.9 なら、おおよそ直近10ステップ分の勾配を覚えているイメージ(`1/(1-ρ)` ステップ分が目安)。

これだけだと RMSProp(Hinton が Coursera 講義で提案)と同じです。足し続けないので、勾配が落ち着けば E[g^2] も落ち着き、学習率が0に消える問題は起きません。固定窓ではなく指数移動平均にしているのは、過去の勾配二乗をすべて保存せずに済ませる(状態を1変数で持てる)ための実装上の工夫でもあります。

#### 工夫2: グローバル学習率をなくす(単位を合わせる)

Zeiler は「更新量の単位が物理的におかしい」点に着目しました。普通の更新 `Δθ = -η·g` では、勾配 g は「loss / θ」の単位を持つため、更新量 Δθ の単位は「η × loss / θ」となり、パラメータ θ の単位と一致しません。一方ニュートン法のような二次法は `Δθ = -H^{-1}g`(H はヘシアン)で、ここでは単位がきれいに θ に揃います。一次法でこの整合を近似的に取り戻すため、Adadelta は「過去の更新量の二乗の移動平均」も覚えて、それを分子に使います。

```
Δθ_t = - ( sqrt(E[Δθ^2]_(t-1) + ε) / sqrt(E[g^2]_t + ε) ) × g_t   # 更新量
θ_t  = θ_(t-1) + Δθ_t                                              # パラメータ更新
E[Δθ^2]_t = ρ × E[Δθ^2]_(t-1) + (1-ρ) × (Δθ_t)^2                  # 更新量の移動平均も更新
```

ポイントは、更新量を決める係数が「過去の更新量の RMS ÷ 最近の勾配の RMS」になっていることです。ここに人間が決める学習率 η が出てきません。過去に大きく動いていたパラメータは引き続き大きく、小さく動いていたパラメータは小さく、という調整が自動で決まります。なお E[Δθ^2] は1ステップ遅れた値(t-1)を使う点に注意してください。これは現ステップの Δθ_t がまだ計算前だからです。分子の RMS[Δθ] と分母の RMS[g] を比で割ることで、`(θ の単位)/(loss/θ の単位)` の比となり、結果として更新量が θ の単位に揃う、というのが「単位を合わせる」の意味です。

数値例でイメージします。

```
あるパラメータで、最近の勾配 RMS  sqrt(E[g^2])  = 2.0
                過去の更新量 RMS sqrt(E[Δθ^2]) = 0.4 だったとする。
更新係数 = 0.4 / 2.0 = 0.2
勾配 g = 1.0 のとき 更新量 Δθ = -0.2
```

勾配が大きい局面では分母が大きくなり更新量が抑えられ、安定します。逆に勾配が小さい局面では更新量が相対的に保たれます。学習初期は E[Δθ^2] が小さく(更新がまだ蓄積されていない)、ε に支配されて慎重なステップから始まる、という自然な warmup のような立ち上がりも持ちます。

tensor shape の話: パラメータが `[1000, 256]` の行列なら、E[g^2] と E[Δθ^2] も同じ `[1000, 256]` で、要素ごとに別々に持ちます。Adagrad より状態が1つ多い分メモリは増えますが、Adam(m と v の2状態)と同程度です。

### 手法の系譜と主要論文

Adadelta は適応的学習率の系譜の中で「Adagrad → 移動平均化 → Adam」へ至る途中に位置します。

1. [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)(Duchi et al., JMLR 2011): パラメータごとの学習率調整を導入。だが累積を足し続けるため学習率消失。
2. RMSProp(Tieleman & Hinton, Coursera 2012): 累積を移動平均に置き換えて消失問題を解消。論文化されず講義スライドのみが出典という珍しい手法。引用の際は「Lecture 6.5, COURSERA: Neural Networks for Machine Learning」が一般的な出典表記です。
3. Adadelta(Zeiler, arXiv 2012): RMSProp とほぼ同時期に独立に登場。移動平均化に加え、更新量の RMS を分子に入れてグローバル学習率を不要にした点が独自。
4. Adam(Kingma & Ba, ICLR 2015): RMSProp の二次モーメントに、一次モーメント(モメンタム)とバイアス補正を加えた決定版。Adadelta が省いたグローバル学習率を、バイアス補正という別の形で扱い、安定性と速度を両立させました。Adam は結局、明示的な学習率 η を残したまま広く普及し、「学習率を消す」という Adadelta の路線とは異なる方向で成功しました。
5. その先に [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、[RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、AdamW などが続きます。

Matthew D. Zeiler は当時 Google におり、後に畳み込みネットの可視化(ZFNet、Zeiler & Fergus, ECCV 2014、ILSVRC 2013 優勝)でも知られる研究者です。

### 論文の実験結果(定量データ)

Zeiler (2012) の論文は MNIST と音声系タスクで評価しています。

- 評価指標の説明: MNIST(28×28 の手書き数字 0〜9、学習6万枚・テスト1万枚)では「テスト誤分類率(test error rate、間違えた割合)」を見ます。数値が小さいほど良い。
- MNIST の結果: ハイパーパラメータをほとんど調整せずに、SGD や Adagrad と同等以上のテスト誤分類率を達成。具体的には、SGD・モメンタム SGD・Adagrad が学習率の手調整を必要としたのに対し、Adadelta は ρ(論文では 0.95 前後)と ε(1e-6 前後)を決めるだけで競争力のある結果を出し、ε と ρ を広い範囲で変えてもテスト誤差が大きく崩れないという頑健性を示しました。論文の主眼は「学習率を手で探さなくても安定して学習できる」点の実証です。
- 音声タスク(大規模分散学習): Google の音声認識システムの大規模ニューラルネット(数千万パラメータ規模)を多数のマシンで並列学習する設定で、Adadelta が安定して動くことを示しました。学習率を環境(マシン台数、データ分割)ごとに調整する手間が省ける点が、大規模実応用での魅力でした。

アブレーション的な観点: Adadelta から工夫2(分子の更新量 RMS)を抜くと、それはほぼ RMSProp になり、グローバル学習率を再び指定する必要が出ます。逆に工夫1(移動平均化)を抜くと Adagrad に戻り、学習率消失が復活します。つまり Adadelta の2つの工夫はそれぞれ別々の問題(消失問題・学習率指定の手間)を解いており、両方そろって初めて「学習率指定なしで止まらない」が実現します。

ただし後年の比較では、Adam(一次モーメント=モメンタムを併用)のほうが多くのタスクで速く安定することが繰り返し報告され、Adadelta は現在の深層学習では主役ではありません。大規模なオプティマイザ公平比較(Schmidt et al. "Descending through a Crowded Valley", ICML 2021、15手法を多数タスクで比較)でも、Adadelta は中位以下に位置づけられ、Adam/AdamW/Nadam が上位グループでした。Adam がモメンタムで「方向の勢い」を加えるのに対し、Adadelta は勢いの項を持たないため、谷の細長い領域(ill-conditioned な loss 地形)での進みが遅くなりやすいのが一因です。

### メリット・トレードオフ・限界

メリット

- 学習率消失問題を解消(移動平均化により実効学習率が0に消えない)。
- グローバル学習率を人間が指定しなくても動く(調整するハイパーパラメータが ρ と ε 程度)。
- 更新量とパラメータの単位が揃うように設計されており、理屈の上で素直。
- ε と ρ に対する頑健性が高い(論文のアブレーションで確認)。

トレードオフ・限界

- 学習率で収束速度を積極的に制御しにくい。明示的な学習率がないため、収束を速めたい/遅くしたいといった介入が難しい(実装によっては係数を掛けられるが、本来の設計思想からは外れる)。
- パラメータと同サイズの状態を2つ持つためメモリ消費が増える(Adam と同程度)。
- 一次モーメント(モメンタム)を持たないため、細長い谷では Adam より進みが遅い。
- 実務では Adam / AdamW のほうが速く安定することが多く、現在は主流ではない。
- 「単位を合わせる」議論は厳密な収束保証を伴うものではなく、ヒューリスティックな動機づけにとどまる。

### 発展トピック・研究の最前線

- 学習率フリーの最適化: 「人間が学習率を決めなくてよい」という Adadelta の問題意識は、近年の D-Adaptation(Defazio & Mishchenko, ICML 2023)や Prodigy(Mishchenko & Defazio, 2023)、Schedule-Free(Defazio et al., NeurIPS 2024)といった「学習率スケジュール不要」手法に受け継がれています。これらは理論的なステップサイズ推定を持ち込み、Adadelta のヒューリスティックを精緻化したと位置づけられます。Adadelta はこの長い系譜の先駆けです。
- ニュートン法的な単位合わせ: 工夫2の「二次法では単位が揃う」という観察は、近年の二次最適化(Shampoo, Sophia など)の再注目とも問題意識を共有しています。曲率情報を安価に近似する流れの源流の一つと見なせます。
- 実務的選択: 今でも一部のフレームワークでデフォルト候補に残っていますが、密な深層学習では AdamW が事実上の標準です。Adadelta は「なぜグローバル学習率を消せるのか」「更新量の単位とは何か」を学ぶ教材として価値があります。

### さらに学ぶための関連トピック

- [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Adagrad

### ひとことで言うと
Adagrad(アダグラッド、Adaptive Gradient の略)は「ニューラルネットワークの各パラメータごとに、別々の学習率を勾配の履歴から自動で決める」最適化手法です。これまで活発に動いてきたパラメータは慎重に小さく、まだほとんど動いていないパラメータは大胆に大きく更新します。この「パラメータごとに学習率を変える」という発想は、後に RMSProp、Adam をはじめとする現代の主流オプティマイザすべてに受け継がれた、適応的最適化(adaptive optimization)の出発点です。

### 直感的な理解
たくさんの人が乗った満員のシーソーを水平にしたい、と想像してください。すでに何度も大きく揺らされた側は、これ以上強く押すと行きすぎてしまうので、そっと触る程度にしたい。一方まだ一度も触っていない側は、思いきって押したい。全員に同じ力で押すと、よく動く側だけが暴れて、動かない側は放置されます。

機械学習でも同じことが起きます。あるパラメータは毎回の学習で大きな勾配(=どっちに動かすべきかの信号)を受け取り、別のパラメータはたまにしか信号を受け取りません。これらに同じ学習率を使うと、よく信号を受け取るパラメータに合わせれば珍しいパラメータが永遠に育たず、珍しいパラメータに合わせればよく動くパラメータが発散します。Adagrad は「過去にどれだけ動いたか」をパラメータごとに記録し、その記録に応じて押す力(学習率)を自動で弱めることで、この不公平を解消します。

### 基礎: 前提となる概念

以下の言葉を順に噛み砕きます。

- パラメータ: ニューラルネットワークが学習で調整する数字のことです。画像分類ネットなら内部に数百万〜数億個の重み(weight)があり、これらすべてがパラメータです。学習とは、この数字を少しずつ良い値に変える作業です。記号では θ(シータ)と書くのが慣例です。
- 勾配(gradient): 「今のパラメータをどっちの方向にどれだけ動かせば誤差(loss、予測と正解のズレ)が減るか」を表す数字です。各パラメータごとに1個ずつ計算されます。記号では g と書きます。「誤差を最も急に増やす向き」が勾配で、その逆向きにパラメータを動かすと誤差が減ります。
- 学習率(learning rate): パラメータを1回の更新でどれくらい大きく動かすかを決める係数です。大きすぎると行きすぎて発散し、小さすぎると学習が遅すぎます。記号では η(イータ)と書くのが慣例です。
- SGD(Stochastic Gradient Descent、確率的勾配降下法): 最も基本的な最適化法。全パラメータに同じ1つの学習率を使い、`θ ← θ - η·g` で更新します。「確率的」とは、データ全体ではなく一部(ミニバッチ)から勾配を計算することを指します。
- 疎(sparse): 「ほとんどの場面で値が0で、たまにしか0以外にならない」という意味です。たとえば自然言語処理で単語を one-hot(該当単語だけ1、他は0のベクトル)で表すと、入力はほとんど0になります。0の入力に掛かる重みには勾配が来ないので、対応するパラメータは更新されません。
- 劣勾配(subgradient): 微分できない点でも使える勾配の一般化です。たとえば絶対値関数 |x| は x=0 で微分できませんが、-1 から 1 の任意の値を「劣勾配」として使えます。Adagrad の理論はこの一般化された勾配で成り立っています。

疎な特徴の典型例を出します。テキスト分類で「the」「is」のような頻出単語に対応するパラメータは、ほぼ毎回の学習で勾配が出ます(=よく更新される)。一方「珍しい専門用語」に対応するパラメータは、その単語がデータに現れたときだけ勾配が出ます(=ほとんど更新されない)。全パラメータで同じ学習率を使うと、この頻度の違いをうまく扱えません。頻出単語に合わせて学習率を小さくすると珍しい単語が学習されず、珍しい単語に合わせて大きくすると頻出単語が暴れます。Adagrad はまさにこの問題を解くために生まれました。

### 仕組みを詳しく

Adagrad の中心アイデアは「各パラメータについて、過去の勾配の二乗を全部足し続け、その合計の平方根で学習率を割る」ことです。各パラメータ i ごとに専用の累積変数 G_i を持ちます。

```
G_i ← G_i + (g_i)^2                          # 勾配二乗を累積し続ける(初期値 0)
θ_i ← θ_i - ( η / sqrt(G_i + ε) ) × g_i      # 累積が大きいほど実効学習率が小さくなる
```

- G_i: パラメータ i 専用の「これまでの勾配二乗の合計」。各パラメータが自分用に1個持ちます。
- ε(イプシロン): ゼロ割りを防ぐごく小さい数(例: 1e-8 や 1e-10)。
- sqrt: 平方根。
- η / sqrt(G_i + ε) の部分を「実効学習率(そのパラメータに実際に効いている学習率)」と呼びます。

言葉に直すと、そのパラメータが過去に大きく動いてきた(勾配が大きかった)なら累積 G_i が大きくなり実効学習率を小さくする。逆に過去にあまり動いていない(勾配が小さかった)なら G_i は小さいままで実効学習率を大きく保つ。

数値例で追います。η = 0.1、ε を無視します。

パラメータ A は毎回勾配の絶対値がずっと 1.0 だったとします。

```
ステップ 1:   G = 1.0   実効学習率 = 0.1 / sqrt(1.0)   = 0.100
ステップ 4:   G = 4.0   実効学習率 = 0.1 / sqrt(4.0)   = 0.050
ステップ 100: G = 100   実効学習率 = 0.1 / sqrt(100)   = 0.010
```

パラメータ B はたまにしか勾配が出ず、100ステップのうち1回だけ勾配 1.0 が出たとします。

```
そのステップ: G = 1.0   実効学習率 = 0.1 / sqrt(1.0)   = 0.100
```

同じ「100ステップ目」でも A の実効学習率は 0.010、B は 0.100 と10倍の差がつきます。よく動くパラメータは慎重に、ほとんど動かないパラメータは大胆に、という調整が自動で実現します。これが「疎な特徴に強い」と言われる根拠です。一般に、勾配が一定の大きさで来続けると実効学習率は `1/sqrt(t)`(t はステップ数)のオーダーで減衰します。

tensor shape の話: パラメータが `[1000, 256]` という行列なら、G も同じ `[1000, 256]` で、要素ごとに別々の累積値を持ちます。つまりメモリはパラメータと同じだけ余分に必要です。1.28 億パラメータのモデルなら G も 1.28 億個分(float32 で約 0.5 GB)です。

なぜこの設計なのか、を理論面から少し深掘りします。Adagrad の原論文は「対角プレコンディショニング(diagonal preconditioning)」という枠組みで導かれました。最適化では本来、loss 地形の曲がり方(曲率、二階微分のヘシアン行列)を考えてステップ幅を決めたい。理想的には全パラメータ間の相関を考えた行列(フルマトリクス Adagrad)で勾配を変換したいのですが、それは行列の逆平方根計算が必要でパラメータ数 d に対して O(d^2) のメモリと O(d^3) の計算がかかり非現実的です。そこで行列の対角成分だけを使う近似が、上の `1/sqrt(G_i)` で各成分を独立にスケールする式になります。この近似により実用的な計算量(パラメータ数に比例)に収まりました。直感的には「過去の勾配二乗の和」が、そのパラメータ方向の loss のばらつき(曲率に相当する量)の推定になっており、ばらつきの大きい方向ほど慎重に進む、という二次最適化の近似になっています。

### 手法の系譜と主要論文

適応的学習率の系譜は Adagrad を起点に枝分かれします。

1. Adagrad(Duchi, Hazan, Singer, JMLR 2011 / 同時期に McMahan & Streeter, COLT 2010 が独立に類似手法を提案)。パラメータごとに過去の劣勾配の二乗和で学習率を個別調整する枠組みを確立。オンライン凸最適化のリグレット(regret、毎ステップ最善を尽くした仮想の相手と比べてどれだけ累積で損したか)の理論的上界を証明した点が大きな貢献です。具体的には T ステップで `O(sqrt(T))` のリグレット境界を、疎なデータでは標準的な勾配降下より定数倍良い形で達成しました。

2. RMSProp(Tieleman & Hinton, Coursera 講義 2012、論文化されず講義スライドが出典)。Adagrad の「足し続ける」累積を、減衰係数つきの移動平均 `E[g^2] ← ρ·E[g^2] + (1-ρ)·g^2` に置き換えた手法。これにより後述の学習率消失問題を回避しました。

3. [Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)(Zeiler, arXiv 2012)。RMSProp とほぼ同じ移動平均化に加え、更新量の移動平均を分子に入れてグローバル学習率を不要にした手法。

4. Adam(Kingma & Ba, ICLR 2015)。RMSProp の二次モーメント(勾配二乗の移動平均)に、一次モーメント(勾配の移動平均=モメンタム)とバイアス補正を組み合わせた、いいとこ取りの手法。現在のデファクトスタンダードです。

5. その先に [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)(Nesterov モメンタム導入)、[RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)(初期分散補正)、AdamW(重み減衰の分離、Loshchilov & Hutter, ICLR 2019)などが続きます。

6. 疎なドメインの実応用では FTRL-Proximal(McMahan et al., KDD 2013)が、Adagrad の per-coordinate 学習率を L1 正則化と組み合わせて広告クリック率予測の標準にしました。Adagrad の血統が「密な深層学習(Adam 系)」と「疎な大規模線形モデル(FTRL 系)」の両方向に分かれた点は系譜として重要です。

つまり Adagrad は「適応的学習率」という発想そのものを世に出した祖先であり、現代のオプティマイザはほぼすべてその子孫です。

### 論文の実験結果(定量データ)

Adagrad の原論文(Duchi et al., 2011)は理論寄りの論文で、リグレット境界の証明が主軸ですが、実験も含まれます。

- 評価指標の説明: テキスト分類では「精度(accuracy、正しく分類できた割合)」や「F1スコア(適合率と再現率の調和平均、不均衡データで使う総合指標)」を見ます。F1 は「正例をどれだけ取りこぼさず、かつ誤検出を抑えたか」を1つの数字にまとめたもので、珍しいクラスがあるデータで重要になります。オンライン学習では上記のリグレットを見ます。
- 結果の傾向: Reuters RCV1(ロイターのニュース記事分類、特徴が極めて疎)などのテキストタスクで、Adagrad は手調整した固定学習率の SGD を上回るか同等の精度を、学習率の手調整なしで達成しました。論文では、ImageNet 系の画像タスクや RCV1 でのオンライン分類において Adagrad が SGD より速く良い解に到達することが示されています。疎な特徴を持つデータほど Adagrad の優位が明確になる、という主張が実験で裏づけられています。

実応用での重要なエピソードとして、Google が大規模な分散学習で Adagrad を採用した例があります。猫の動画から特徴を教師なしで学習した有名な大規模ニューラルネット研究(Le et al., ICML 2012)や、その後の単語分散表現の学習などで、疎なデータと大規模並列環境において Adagrad が安定して動いたことが知られています。さらに前述の FTRL-Proximal は、Google の広告システムで Adagrad ベースの per-coordinate 学習率がモデルサイズを大幅に削減しつつ精度を保ったことが報告され、産業界での Adagrad 系の実用性を示しました。

アブレーション的な観点(どの要素を抜くとどうなるか): Adagrad から「履歴で割る」部分を抜くと素の SGD に戻ります。素の SGD は疎なデータで珍しい特徴を学習しきれず、F1 が伸びにくくなります。逆に Adagrad の最大の弱点は「累積 G を減らさない」ことで、これを移動平均に直したのが RMSProp/Adadelta です。後続研究では「移動平均に直すと長時間学習でも学習が止まらず、最終精度が改善する」ことが繰り返し示されました。これが Adagrad 単体が密な深層学習で主役にならなかった直接の理由です。

### メリット・トレードオフ・限界

メリット

- パラメータごとに学習率を自動調整するため、初期学習率の手調整がある程度楽になる。
- 疎な特徴を持つデータで、まれな特徴も十分に学習できる(理論・実験の両面で確認済み)。
- 凸最適化の設定でリグレットの理論保証がある。理論的に挙動が理解しやすい数少ないオプティマイザの一つ。

トレードオフ・限界

- 学習率消失問題(decaying learning rate problem): 勾配二乗を足し続けるだけで G が単調増加するため、長く学習すると実効学習率が限りなく0に近づき、必要な学習が終わる前にパラメータがほとんど更新されなくなる。深層ネットの長時間学習では致命的。これが最大の弱点です。
- パラメータと同サイズの累積変数 G を持つため、メモリを余分に消費する(これは Adam なども同様)。
- 密(dense)な特徴(画像の畳み込み特徴など、ほぼ毎回勾配が出る特徴)では「疎に強い」という利点が効きにくく、Adam 系のほうが良いことが多い。
- 初期学習率 η に対する感度は完全には消えず、依然として η と ε はある程度のチューニングが要る。

研究上の未解決課題的な観点としては、対角近似(パラメータ間の相関を無視)による性能の頭打ちがあります。フルマトリクス版は理論的には優れますが計算量が非現実的で、Shampoo(Gupta et al., ICML 2018)のように二次情報をブロック単位で近似する研究が、この方向の発展として続いています。

### 発展トピック・研究の最前線

- AdaGrad-Norm / 適応的勾配法の収束理論: 非凸(深層学習は非凸)設定での収束証明は近年も活発な研究テーマで、Ward, Wu & Bottou (ICML 2019, "AdaGrad stepsizes") が AdaGrad-Norm という変種について、学習率の事前調整なしに非凸関数で収束することを示しました。学習率に対する頑健性を理論的に保証した点が重要です。
- 二次情報の近似最適化: Shampoo(Gupta et al., ICML 2018)や K-FAC(Martens & Grosse, ICML 2015)のように、対角近似を超えてパラメータ間の相関(曲率)を取り込む手法が、大規模言語モデルの学習で再注目されています。Shampoo はまさに「フルマトリクス Adagrad の実用的近似」を狙った手法で、Adagrad の理論的理想に立ち返る系譜です。
- 疎な埋め込み学習: 推薦システムや広告 CTR 予測のような超大規模かつ疎なデータでは、今でも Adagrad 系(FTRL-Proximal など)が現役です。密な深層学習で主役を譲った一方、疎なドメインでは依然として強力です。
- ディープラーニングフレームワークでの位置づけ: 主要フレームワークには標準実装があり、埋め込み層のみ Adagrad、残りは Adam という混在運用も実務で見られます。

### さらに学ぶための関連トピック

- [Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Adam

### ひとことで言うと
Adam(アダム)は、深層学習でいちばん広く使われている「オプティマイザ(optimizer)」です。オプティマイザとは、誤差(loss、損失=モデルの予測がどれだけ間違っているかを数値化したもの)を小さくする方向に、ニューラルネットの重み(weight、モデルが学習で調整するパラメータ)を少しずつ動かす計算手順のことです。Adam は「Momentum(モーメンタム、慣性=これまで進んできた方向を覚えて勢いをつける工夫)」と「RMSProp(パラメータごとに歩幅を自動調整する工夫)」という2つの考え方を1つにまとめ、さらに学習序盤のズレを直す小さな補正(バイアス補正)を加えたものです。ほとんど何も設定をいじらなくてもそこそこ学習が進む「とりあえず使える」万能型として、2015年の登場以来デファクトスタンダードであり続けています。

### 直感的な理解
ニューラルネットの学習は、暗い山の中で谷底(loss が最小になる点)を探す登山にたとえられます。手元にあるのは、いま立っている地点の傾き(勾配=gradient、ある重みを少し動かすと loss がどれだけ変わるかを表す量)だけです。最も素朴な方法は SGD(Stochastic Gradient Descent、確率的勾配降下法)で、「傾きの逆向きに一定の歩幅で進む」だけです。ところがこれには2つの困りごとがあります。

1つ目は「方向がぶれる」問題です。谷が細長い形(ある方向には急で、別方向にはなだらか)だと、急な向きに行ったり来たりジグザグしてしまい、本来進みたいなだらかな方向になかなか進めません。これを解決するのが Momentum で、過去の進行方向を「速度」として貯めることで、ジグザグを打ち消して谷へまっすぐ速く進めます。重いボールが斜面を転がるとき、小さな凹凸では止まらず勢いで進み続けるのと同じ発想です。

2つ目は「歩幅が場所によって合わない」問題です。重みごとに傾きの大きさが何桁も違うことがあり(例えばある層の勾配は 100、別の層は 0.01)、同じ歩幅では片方は暴れ、もう片方は止まります。これを解決するのが RMSProp で、各重みの傾きの大きさで割り算をして、歩幅をそろえます。

Adam は「方向の安定」と「歩幅の自動調整」という別々の課題を同時に解きます。だから初心者が初めて深層学習を組むとき、まず Adam(または後述の AdamW)を選べば大きく外すことはまずありません。

### 基礎: 前提となる概念
Adam を理解するには、次の言葉を押さえておくと十分です。

- 勾配(gradient, `g`): 損失を各重みで偏微分した値。「この重みを増やすと loss がどう変わるか」の傾き。学習はこの逆向きに重みを動かす。
- 学習率(learning rate, `η`、イータ): 1ステップで重みを動かす基本の歩幅。大きすぎると loss が発散(どんどん大きくなって学習が壊れること)し、小さすぎると進みが遅い。
- 移動平均(moving average): 過去の値を少しずつ忘れながら平均し続ける手法。`x ← β·x + (1−β)·新しい値` の形を取る。`β`(ベータ)が大きいほど過去を長く覚える(=なめらかになる)。指数移動平均(EMA, Exponential Moving Average)とも呼ぶ。
- モーメント(moment): ここでは統計の言葉。1次モーメント=平均(勾配そのものの平均)、2次モーメント=二乗の平均(勾配の大きさ=ばらつき)を指す。Adam の名 Adaptive Moment Estimation は「モーメントを推定して適応的(状況に合わせて)に動く」の意味。
- ε(イプシロン): ゼロ割りを防ぐための極小の数(典型的には 1e-8)。
- ミニバッチ(mini-batch): 全データではなく、一部のサンプルだけを使って勾配を計算する単位。だから勾配にはノイズ(ばらつき)が乗る。このノイズへの強さがオプティマイザ選びで重要になる。

これらが分かれば、以下の式は読めます。

### 仕組みを詳しく
Adam はパラメータごとに2つの状態(内部メモリ)を持ちます。`m` が1次モーメント(勾配の移動平均=Momentum 相当)、`v` が2次モーメント(勾配二乗の移動平均=RMSProp 相当)です。`β1`(通常 0.9)、`β2`(通常 0.999)が減衰率、`t` はステップ番号(1, 2, 3, …)です。

```
m ← β1 · m + (1 − β1) · g        ← 1次モーメント(平均的な勾配の向き=慣性)
v ← β2 · v + (1 − β2) · g²        ← 2次モーメント(勾配の大きさ=スケール)

m̂ = m / (1 − β1ᵗ)                 ← バイアス補正
v̂ = v / (1 − β2ᵗ)

w ← w − η · m̂ / (√v̂ + ε)          ← 慣性 m̂ を、スケール √v̂ で割って更新
```

#### バイアス補正がなぜ要るか
`m` も `v` も初期値は 0 から始めます。すると学習の最初の数ステップでは過去の蓄積が足りず、移動平均の値が本来あるべき大きさより小さく(0 寄りに)偏ります。β1=0.9 なら、1ステップ目は `m = 0.9·0 + 0.1·g = 0.1·g`、つまり本来の勾配 `g` の1割の大きさしかありません。これを「ゼロ初期化バイアス」と呼びます。

これを直すのが `m̂ = m / (1 − β1ᵗ)` です。1ステップ目なら `1 − 0.9¹ = 0.1` で割るので `m̂ = 0.1·g / 0.1 = g` と正しい大きさに戻ります。ステップが進むと `β1ᵗ`(0.9 の t 乗)は 0 に近づき `(1 − β1ᵗ) ≈ 1` になるので、補正はほぼ効かなくなります。つまり「最初だけ補正、あとは素通し」です。`v` 側は β2=0.999 なので補正がさらに長く効きます(`1 − 0.999¹ = 0.001` で割るので、補正がなければ序盤の `v̂` は本来の1000分の1)。この補正が、学習序盤から安定した歩幅を可能にします。補正がないと、序盤に実効学習率が小さくなりすぎて立ち上がりが遅れたり、逆に分母が小さすぎて暴れたりします。

#### 更新式の意味
最終行 `η · m̂ / (√v̂ + ε)` の中身は、

- 分子 `m̂` = 慣性のかかった平均的な勾配の向き(どっちに進むべきか)
- 分母 `√v̂` = そのパラメータの勾配の典型的な大きさ(どれくらいバラついているか)

割り算によって、勾配が大きい重みは控えめに、小さい重みは大きめに更新され、更新量(ステップ幅)はおおよそ `±η` に揃います。「進む方向は慣性で安定、歩幅はパラメータごとに自動調整」が一度に実現します。この `m̂/√v̂` という比は「シグナル(信号、平均的方向)/ ノイズ(ばらつき)」の比とも読め、勾配の符号が安定しているパラメータほど大きく、ぶれているパラメータほど小さく動くという信頼度の重み付けとも解釈できます。

#### 数値例
`η = 0.001`、β1=0.9、β2=0.999(デフォルト)とします。ある重みの勾配がしばらく安定して `g ≈ 0.5` なら `m ≈ 0.5`、`v ≈ 0.25`、`√v ≈ 0.5`、更新量 `≈ 0.001 · 0.5 / 0.5 = 0.001`。別の重みで `g ≈ 5` でも `v ≈ 25`、`√v ≈ 5` なので更新量はやはり `≈ 0.001`。勾配の絶対的な大きさに関係なく歩幅が揃うのが分かります。逆に、勾配の符号がランダムに揺れて平均が 0 に近いパラメータでは `m̂ ≈ 0` なので更新がほぼ止まります。

#### tensor の形・メモリ
パラメータごとに `m` と `v` の2セットを、それぞれ重みと同じ形(shape)で保持します。重みが `[512, 256]` なら `m` も `v` も `[512, 256]`。したがって追加メモリは重みの2倍です。パラメータ1億個のモデルなら、重み 0.4GB(fp32、1個4バイト)に対し `m`+`v` で 0.8GB が上乗せされる計算です。さらに勾配 `g` 自体も重みと同サイズなので、学習時の総メモリは「重み + 勾配 + m + v」で重みの4倍が目安になります。この「状態2倍」が大規模学習でメモリのボトルネックになり、後述の Lion や 8-bit Adam、Adafactor がここを削りにいきます。

ASCII で更新の流れを表すと:

```
   勾配 g
     │
     ├──► m(慣性で平滑化) ──┐
     │                       ├──► m̂/√v̂ ──► 歩幅は ±η に正規化 ──► w 更新
     └──► v(大きさを記録) ──┘
```

### 手法の系譜と主要論文
Adam は突然生まれたのではなく、適応的勾配法(adaptive gradient methods、パラメータごとに学習率を変える手法群)の系譜の合流点です。

1. Momentum / Nesterov — Polyak (1964)、Nesterov (1983)。過去の進行方向を速度として貯め、ジグザグを抑える。Adam の1次モーメント `m` はこれに相当。Nesterov は「先読み」してから勾配を測る改良。[SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) [Nesterov Accelerated Gradient](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。
2. AdaGrad — Duchi, Hazan & Singer (2011, JMLR)。各パラメータについて過去の勾配二乗を「全部足し合わせた量」で割り、パラメータごとに学習率を変える発想を確立。スパース(まれにしか勾配が立たない)な特徴に強い反面、足し続けるので分母が単調増加し学習率が枯れて止まる弱点があった。詳しくは [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。
3. RMSProp — Tieleman & Hinton (2012, Coursera 講義 Lecture 6e)。AdaGrad の「全部足す」を「最近を重視する移動平均」に変え、学習率が枯れない形にした。Adam の2次モーメント `v` はこれそのもの。
4. Adam — Kingma & Ba (ICLR 2015, arXiv:1412.6980)。RMSProp の `v` と Momentum の `m` を1つの式に統合し、初期バイアスを消す補正を追加。デフォルト設定で多くのタスクに効くロバストさ(設定を変えても壊れにくい性質)を示した。同論文には AdaMax(`v` を L∞ ノルムに置き換えた変種)も併載されている。
5. AMSGrad — Reddi, Kale & Kumar (ICLR 2018, Best Paper)。Adam が単純な凸問題でさえ収束しない反例を示し、`v̂` を単調非減少にする修正を提案。
6. AdamW — Loshchilov & Hutter (ICLR 2019)。weight decay(重み減衰、正則化の一種)の扱いを正した改良版で、現在 Transformer 学習の標準。詳しくは [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。

つまり Adam は「AdaGrad → RMSProp(スケール調整)」と「Momentum(方向安定)」の2系統を束ねた合流点であり、その後さらに AMSGrad(収束)・AdamW(正則化)・RAdam(序盤の分散)などへと枝分かれしていきます。

### 論文の実験結果(定量データ)
Kingma & Ba (2015) の原論文は、いくつかの標準タスクで Adam が他手法より速く収束することを示しました。

- ロジスティック回帰(MNIST、L2正則化あり): Adam は AdaGrad・SGD+Nesterov と比べて訓練 loss の下がり方が速く、初期から安定。MNIST は手書き数字 0–9 の 28×28 画像 6万枚の分類タスクで、機械学習のベンチマークの定番。
- 多層パーセプトロン(MNIST): 学習 cost(=訓練 loss)の減少が SGD・AdaGrad・RMSProp・SGD+Nesterov より速いことを図で示した。
- 畳み込みネット(CIFAR-10、画像分類): 学習初期の cost 減少が他手法を上回ると報告。CIFAR-10 は 32×32 のカラー画像 6万枚を10クラスに分類するタスク。

ここで言う「収束が速い」とは、同じ計算量(エポック数=データを何周したか)でより低い訓練 loss に到達する、という意味です。これは開発サイクルの短縮(同じ時間でより良いモデルが得られる)に直結するため重要視されました。

ただし「速い=最終的に良い」ではない点が後続研究で問題になりました。

- Wilson et al. (NeurIPS 2017, "The Marginal Value of Adaptive Gradient Methods") は、CIFAR-10 の画像分類や文字レベル言語モデルなどで、Adam を含む適応手法は学習は速いが、最終的なテスト精度(汎化性能=未知データでの正解率)はよく調整した SGD+Momentum に劣ることがある、と報告。例えば CIFAR-10 で SGD のテスト誤差のほうが数ポイント低いケースを示した。「テスト精度」は学習に使っていないデータでの正解率で、実用で本当に効くかを測る指標です。訓練 loss が低くても過学習(訓練データだけに過剰に適合)していればテスト精度は上がりません。
- Reddi et al. (ICLR 2018) は、ある人工的な1次元の確率的凸問題で Adam が最適解から離れた点に収束し続ける反例を構成しました。原因は `v` の移動平均が状況次第で増減し、有効学習率(`η/√v̂`)が単調に減らないことにある、と分析。たまに来る大きく正しい方向の勾配が、頻繁に来る小さく逆向きの勾配に `v` の更新を通じて押し負ける、という構造です。修正版 AMSGrad は `v̂ ← max(過去の v̂, 今の v̂)` として有効学習率を単調非増加にし、収束を保証しました。ただし実務での AMSGrad の精度改善は限定的だったとの報告も多く、理論的な穴を塞いだ意義のほうが大きいと評価されています。

アブレーション(ablation、構成要素を1つずつ抜いて寄与を測る分析)の観点では、Adam の3要素を抜くと何が起きるかが理解の助けになります。Momentum(`m`)を抜くと RMSProp に戻り、ジグザグ抑制が弱まる。スケール調整(`v`)を抜くと Momentum 付き SGD に戻り、勾配スケールが層ごとに違う問題が再燃する。バイアス補正を抜くと、学習序盤の数十〜数百ステップで実効学習率が小さくなりすぎ、立ち上がりが遅れる(あるいは `v̂` が過小で暴れる)。3つそれぞれが別の役割を担っていることが分かります。

### メリット・トレードオフ・限界
メリット
- Momentum(方向安定)と RMSProp(スケール調整)の利点を両立。
- デフォルト値(η=0.001, β1=0.9, β2=0.999, ε=1e-8)のままで幅広い問題が動く。ハイパーパラメータにロバスト。
- バイアス補正で学習序盤から安定。たまにしか勾配が立たないスパースなパラメータ(単語埋め込みなど)にも強い。
- 勾配のスケール変化が激しい RNN(リカレントネット)や Transformer の学習で安定して機能する。

トレードオフ・限界
- `m` と `v` の2セットで追加メモリが重みの2倍。大規模モデルで重い。
- 一部の凸問題で収束しない反例があり理論的に万能ではない(AMSGrad で修正提案)。
- よく調整した SGD+Momentum に最終的な汎化性能で負けることがある(特に CNN の画像分類)。
- 標準実装の weight decay の扱いが不正確で正則化が意図通り効かない([AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) で修正)。

未解決の課題として、「なぜ適応手法は速いのに汎化で劣ることがあるのか」は完全には解明されていません。鋭い極小値(sharp minima、少しずれると loss が急増する尖った谷底=過学習しやすい)に落ちやすい説、勾配ノイズの構造(ノイズの方向の偏り)の違い説などがありますが、決着していません。実務では「画像 CNN は SGD+Momentum、Transformer 系は AdamW」という経験的な使い分けが広く採られています。

### 発展トピック・研究の最前線
- メモリ削減: 8-bit Adam(Dettmers et al., 2022)は `m`・`v` を8ビットに量子化(値を粗く丸めて保存)して状態メモリを大幅削減し、精度をほぼ維持。Adafactor(Shazeer & Stern, 2018)は2次モーメントを行ベクトル・列ベクトルに因子分解して `v` のメモリを `O(行+列)` に節約し、巨大言語モデル学習で多用される。[Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) は `v` 自体を捨てて状態を半分にする。
- 収束と汎化の橋渡し: RAdam(Liu et al., ICLR 2020)は学習序盤に `v̂` の分散が大きく不安定になる問題を、序盤は適応をオフにする整流(rectification)で緩和し、warmup(学習率を徐々に上げる手順)の代替を狙った。AdaBelief(Zhuang et al., NeurIPS 2020)は `v` を「勾配と予測 `m` の差の二乗」に置き換え、勾配が予測通りなら大きく進むことで汎化を改善すると主張。
- スケール則(scaling law)との関係: 大規模言語モデルでは β2 を 0.95 に下げる(より直近を重視し急変に追従)、ε をやや大きめにする、といった調整が定番化しており、モデル規模・バッチサイズに応じた最適設定の研究が続いています。学習率を層やパラメータの幅に応じて自動調整する muP(maximal update parametrization, Yang et al.)とも組み合わせられます。

### さらに学ぶための関連トピック
- [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Nesterov Accelerated Gradient](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## AutoAugment

### ひとことで言うと
「どの画像加工を、どんな確率で、どれくらいの強さで組み合わせると、モデルの精度が一番上がるか」を、人間が手で決めるのではなく機械(強化学習)に自動で探させる手法です。ImageNet など主要ベンチマークの精度を当時の最先端まで押し上げ、「データ拡張も、ネットワーク構造やハイパーパラメータと同じく最適化の対象にできる」と示した、自動データ拡張研究の起点となった仕事です。後続の RandAugment や TrivialAugment はすべて、この論文への応答として生まれました。

### 直感的な理解
それまでデータ拡張の設計は職人芸でした。「回転は何度まで許すか」「色をどのくらいいじるか」「明るさ変換と回転をどう組み合わせるか」を、研究者が経験と勘で1つずつ決め、試し、調整していました。これは時間がかかるうえ、CIFAR(小さい画像のデータセット)で良かった設定が ImageNet(大きく多様な画像のデータセット)で最適とは限らず、データセットごとにやり直しでした。

ちょうど同じ頃、ニューラルアーキテクチャ探索(NAS: ネットワークの構造そのものを機械に自動設計させる研究、Zoph & Le, ICLR 2017)が「人間が手で設計してきたものを、機械に探させると人間を超える」ことを示し始めていました。AutoAugment の発想はシンプルで、「NAS と同じ探索の枠組みを、ネットワーク構造ではなくデータ拡張ポリシーに向けたらどうか」というものです。つまり「拡張の設計」という職人芸を、報酬(うまくいった度合いを表す数値)を最大化する自動探索問題として定式化し直したのです。アイデア自体は素直ですが、「拡張も学習対象にできる」という視点の転換が、その後の分野を作りました。

### 基礎: 前提となる概念
- データ拡張(data augmentation): 学習画像に加工を施して人工的に種類を増やし、過学習(学習データだけ覚えて未知データで失敗する状態)を防ぐ技術。
- 強化学習(reinforcement learning, RL): 「行動して、結果として得た報酬を頼りに、次はもっと良い行動を取る」よう試行錯誤で学ぶ枠組み。ここでは「ポリシーを提案する」が行動、「そのポリシーで学習したモデルの精度」が報酬。
- コントローラ(controller): ポリシーを提案する側のモデル。AutoAugment では RNN(リカレントニューラルネットワークの略。過去の出力を次の入力に使い、可変長の系列を順に生成できるニューラルネットワーク)を使います。RNN が「変換・確率・強度」を1つずつ順番に吐き出して1つのポリシーを組み立てます。
- 代理タスク(proxy task): 探索を安く済ませるための縮小設定。小さいモデルを小さいデータの一部で学習・評価します。本番より軽いので何千回も試せる一方、代理での最適が本番で最適とは限らない(proxy gap=代理と本番のズレ)という弱点があります。
- ポリシーとサブポリシー: ポリシーは複数のサブポリシーの集合。各サブポリシーは「2つの操作の連なり」で、操作は (変換, 適用確率, 強度) の3つ組です。

### 仕組みを詳しく
#### ポリシーの構造
- ポリシーは 25 個のサブポリシーからなる(論文設定)。
- 各サブポリシーは 2 つの操作の連なり。
- 各操作は3つ組 (変換, 適用確率, 強度)。
  - 変換は ShearX/Y、TranslateX/Y、Rotate、Color、Posterize、Solarize、Contrast、Sharpness、Brightness、AutoContrast、Equalize、Invert など 16 種類前後。
  - 適用確率は離散化(0.0, 0.1, …, 1.0 の 11 段階)。
  - 強度も離散化(0〜9 の 10 段階)。

学習時は、25 個のサブポリシーから画像 1 枚ごとにランダムに 1 つを選び、その 2 操作を順に適用します。探索空間はこの離散化で約 10^32 通り規模と見積もられ(16 変換 × 11 確率 × 10 強度を2操作分、それを 25 サブポリシー並べる組み合わせ)、人手では到底試せません。この巨大さこそが「機械に探させる必然性」を生んでいます。

#### 探索のしくみ(RL ループ)
1. コントローラ(RNN)がポリシーを1つ提案する。
2. そのポリシーで小さい子モデルを代理データセットで学習する。
3. 子モデルの検証精度(学習に使わなかったデータでの正解率)を「報酬」としてコントローラに返す。
4. コントローラを報酬が高くなる方向へ強化学習(方策勾配の一種である Proximal Policy Optimization 系)で更新する。
5. これを約 1 万 5 千回繰り返し、最良ポリシーを得る。

「ポリシーを出す → 試す → 精度を測る → ポリシーを改善する」というループを延々と回し、人手では試せない数の組み合わせを機械が探索します。約 1 万 5 千回の子モデル学習という回数が、後述の高コストの正体です。

#### 数値イメージ
あるサブポリシーが [(Equalize, 確率 0.8, 強度 -), (Rotate, 確率 0.6, 強度 2)] なら、1枚の画像に対し:
- 80% の確率でヒストグラム均等化(明暗分布を整える)を適用。
- 60% の確率で強度 2 相当の角度で回転。
(Equalize は強度を持たない変換なので強度欄が空になる、といった変換ごとの違いも探索空間に含まれます。)

別の画像では別のサブポリシーが選ばれ、毎回異なる拡張が施されます。出力テンソルの形(例 `[3, 224, 224]`)は不変で、中身だけが変換されます。

#### なぜ確率と強度を別々に持たせたか
現実の最適拡張は変換ごとに「適度な頻度」と「適度な強さ」が違うはずだ、という仮定です。例えば色変換は強くかけても良いが回転は控えめが良い、といった非対称を表現するために、変換ごとに確率・強度を独立に最適化できる柔軟な空間を設けました。この柔軟さが高精度の源である反面、探索空間を巨大にし、後述の高コストを招きました。皮肉なことに、後続の RandAugment はこの柔軟さの大半が不要だったと示すことになります。

### 手法の系譜と主要論文
- NAS with RL(Zoph & Le, ICLR 2017): RNN コントローラ + RL でネットワーク構造を探索。AutoAugment が借用した方法論の母体。
- AutoAugment(Cubuk, Zoph, Mané, Vasudevan, Le, CVPR 2019): 拡張ポリシーを RL で探索。自動データ拡張の起点。
- Fast AutoAugment(Lim et al., NeurIPS 2019): 密度マッチング(拡張後のデータ分布を元の分布に近づける方向で評価)で「子モデルを再学習せずに」良いポリシーを推定し、探索を桁違いに高速化。
- Population Based Augmentation(Ho et al., ICML 2019): 学習途中でポリシーを進化させる「スケジュール」を探索。固定ポリシーより柔軟で、学習が進むにつれて最適な拡張が変わることを利用。
- Faster AutoAugment(Hataya et al., ECCV 2020): 拡張を微分可能化し勾配法で直接最適化。
- Learning Data Augmentation for Object Detection(Zoph et al., ECCV 2020): 検出タスク向けに、バウンディングボックスを意識した変換集合で探索を拡張。
- RandAugment(Cubuk et al., NeurIPS 2020): 探索を捨て (N, M) の2パラメータに単純化し、同等精度を達成([RandAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。
- TrivialAugment(Müller & Hutter, ICCV 2021): さらに単純化しパラメータを実質ゼロに([TrivialAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。

この一連は「探索を発明し(AutoAugment)→ 探索を速め(Fast/PBA/Faster)→ 探索を縮め(RandAugment)→ 探索を捨てる(TrivialAugment)」という、研究の自己批判的な成熟の物語として読めます。

### 論文の実験結果(定量データ)
指標の意味: CIFAR は誤分類率(間違えた割合、低いほど良い)、ImageNet は Top-1 精度(最も確信した予測が正解の割合、高いほど良い)で測ります。CIFAR-10 は 10 クラス 32×32 の小画像 6 万枚、ImageNet は 1000 クラス約 128 万枚で、後者での 0.5 ポイント改善は有意とされます。

AutoAugment(Cubuk et al. 2019)の報告:
- CIFAR-10 で誤分類率を当時の最先端水準まで低減。Wide-ResNet や Shake-Shake、PyramidNet 系で、拡張なし比でおよそ 0.5〜1.5 ポイント規模の改善を報告。PyramidNet + ShakeDrop の組み合わせで誤分類率およそ 1.5% という当時最高水準を達成。
- CIFAR-100、SVHN(街の番地数字)でも一貫して改善。SVHN では誤分類率を約 1.0% 近くまで下げたと報告。
- ImageNet で ResNet-50 の Top-1 精度をベースライン約 76.3% からおよそ 77.6% 程度へ(報告による、約 1.3 ポイント改善)。大規模分類でも明確に効くことを示しました。AmoebaNet-C など大規模モデルではさらに高い精度を報告。
- 転移性(transferability)のアブレーション(要素を変えて影響を見る実験): ImageNet で探索したポリシーを、Stanford Cars(車種の細かい分類)や FGVC Aircraft(航空機の細かい分類)など別の細粒度分類データセットに「探索しなおさず」適用しても精度が向上。すなわち探索コストを一度払えば、似た性質のデータに使い回せると報告。これは実用上きわめて重要な発見で、「探索は高いが一度きりで済む」という主張の根拠になりました。
- コストの実態: 論文設定の探索は数千 GPU 時間規模(代理タスクでの約 1 万 5 千回の子モデル学習)。個人や中小規模では再現が困難で、これが後続の高速化・単純化研究を強く動機づけました。

アブレーション的観点としては、探索を代理タスク(小モデル・データ部分集合)で行う設計が、本番の大規模設定での最適性を犠牲にしている可能性(proxy gap)が指摘され、RandAugment がこの点を正面から突くことになります。実際 RandAugment 論文は、本番設定で直接 (N, M) を選ぶと、AutoAugment の代理探索ポリシーと同等以上になると示しました。

### メリット・トレードオフ・限界
メリット
- 人手のチューニング不要で拡張ポリシーを自動最適化でき、職人芸を体系的最適化に置き換えた。
- ImageNet など主要ベンチマークで明確な精度向上を示した。
- 見つけたポリシーが似たデータセットへ転移しうる(使い回せる)。
- 「データ拡張を最適化対象として扱う」という研究の方向性を切り開いた歴史的意義。

トレードオフと限界
- 探索が数千 GPU 時間規模で極めて高価。再現性のハードルが高く、ほとんどの研究者は公開済みの探索済みポリシーを流用するしかなかった。
- 代理タスクで探すため、本番の大規模設定では最適とは限らない(proxy gap)。
- 確率・強度を離散化しており、真の最適点を取りこぼす可能性。
- 探索済みポリシーは固定であり、学習の進行に応じて変える(スケジュール)柔軟性がない(PBA がこの点を改善)。
- 後続の RandAugment / TrivialAugment に「もっと安く同等」と示され、実用では置き換えられがち。ただし「なぜ単純化で十分なのか」を理解するための比較基準として今も価値があり、自動拡張を語るうえで避けて通れない原点です。

### 発展トピック・研究の最前線
- 探索の高速化: Fast AutoAugment(密度マッチング)、Faster AutoAugment(微分可能拡張で勾配最適化)、DADA(微分可能 AutoAugment)など、proxy gap を抑えつつコストを下げる方向。これらは探索を数千 GPU 時間から数 GPU 時間規模に縮めました。
- 探索の放棄: RandAugment / TrivialAugment は「そもそも探索が必要だったのか」を問い、単純なランダム化で匹敵すると示した。AutoAugment はこの議論の出発点として常に引かれます。
- スケジュール最適化: Population Based Augmentation のように、固定ポリシーではなく学習段階ごとに変わる拡張スケジュールを探す研究。学習初期は弱く、後期は強く、といった動的制御の可能性を開きました。
- 適用領域の拡大: 物体検出向けの拡張探索(Learning Data Augmentation Strategies for Object Detection, Zoph et al. 2020)、音声・テキストへの自動拡張など、画像分類の外への展開。
- AutoML 文脈での位置づけ: AutoAugment は「学習パイプラインの構成要素すべてを自動最適化する」という AutoML(機械学習の自動化)の理念を、データ拡張という具体例で実証した代表例として参照され続けています。

### さらに学ぶための関連トピック
- [RandAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [TrivialAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cutout / Random Erasing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Mixup / CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Fine-tuning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)


## Barlow Twins

### ひとことで言うと
Barlow Twins(バーロウ・ツインズ)は、ラベルのないデータから良い特徴を学ぶ自己教師あり学習の手法です。同じ画像から作った2つのビューの特徴を比べ、その「相互相関行列」を単位行列(対角が1、それ以外が0)に近づけることで、崩壊(全部同じ定数に潰れる失敗)を防ぎつつ、次元間の情報の重複(冗長性)を削ります。負例も EMA も stop-gradient も使わず、損失式が実質1本というシンプルさが特徴です。発想の源は神経科学の「冗長性削減原理」にあります。

### 直感的な理解
良い特徴とは何でしょうか。1つの答えは「無駄がない」ことです。たとえば 100 次元の特徴ベクトルがあっても、その 100 個が全部「画像の明るさ」を少しずつ違う形で言い換えているだけなら、実質1次元ぶんの情報しかなく無駄です。理想は、各次元が互いに別々の(独立した)情報を担い、しかも同じ画像の別ビューでは安定して同じ値を取ることです。

Barlow Twins は、この理想を「相互相関行列を単位行列にせよ」という1つの目標に凝縮しました。対角を1にする=「同じ次元は2ビュー間で完全に一致せよ」(安定性・不変性)、非対角を0にする=「違う次元は無相関であれ」(独立性・冗長性削減)。この2つを同時に満たすと、崩壊もせず無駄もない表現が得られます。崩壊しないのは、全部が定数に潰れると(正規化のあとで)相関を定義できず対角を1にできないため、定数への逃げ道が自動的に塞がれるからです。

### 基礎: 前提となる概念
冗長性削減原理(redundancy reduction principle)とは、神経科学者ホレス・バーロウ(Horace Barlow, 1961)が提唱した古典的仮説です。「脳の感覚処理は、入力に含まれる冗長な(重複した)情報を取り除き、互いに統計的に独立した成分で世界を表現するように進化した」という考えです。情報理論の言葉では「効率的符号化(efficient coding)」とも呼ばれます。Barlow Twins という名前はこの研究者に由来し、その原理を表現学習に持ち込んだのが手法の核心です。

相関(correlation)とは、2つの変量が一緒に動く度合いを -1〜1 で表したものです。1 は完全に同じ動き、0 は無関係、-1 は逆向きの動きを意味します。

相互相関行列(cross-correlation matrix)とは、2つのビューの特徴間で、全次元ペアの相関を並べた `[D, D]` の行列です(D は特徴次元)。(i, j) 成分は「ビュー A の i 次元目とビュー B の j 次元目が、バッチ全体でどれだけ一緒に動くか」を表します。注意したいのは、これが「2つのビュー間」をまたぐ量である点です(同じビュー内の共分散ではない)。

単位行列(identity matrix)I は、対角が1・それ以外が0の行列です。相互相関行列を I に近づけるとは、「対角(同じ次元同士)は完全相関、非対角(違う次元同士)は無相関」を目指すことです。

正規化(normalization)。Barlow Twins は各次元をバッチ方向で平均0・標準偏差1にそろえてから相関を計算します。これにより各成分が素直に相関係数として解釈できます。

### 仕組みを詳しく
処理の流れを数値例で説明します。

1. 1枚の画像にランダムなデータ拡張を2回かけ、ビュー A, B を作る。
2. 同じネットワーク(重み共有のエンコーダ + 射影 MLP)に通し埋め込みを得る。バッチ N 枚なら Z_A, Z_B は形状 `[N, D]`(論文では projector の出力 D=8192 など高次元)。
3. 各次元をバッチ方向で平均0・標準偏差1に正規化する。
4. 2ビュー間の相互相関行列 C(形状 `[D, D]`)を計算する。

```
C_{ij} = Σ_b (z_A[b,i] · z_B[b,j]) / sqrt( Σ_b z_A[b,i]^2 · Σ_b z_B[b,j]^2 )
```

各成分は -1〜1 の値を取ります。

5. C を単位行列 I に近づける損失を最小化する。

```
loss = Σ_i (1 - C_ii)^2   +   λ · Σ_i Σ_{j≠i} C_ij^2
       └── 不変項 ──────┘     └──── 冗長性削減項 ────┘
```

2項の意味を噛み砕きます。
- 第1項(対角を1に):同じ次元 i は2ビュー間で完全相関 `C_ii = 1` であれ。「同じ画像の別ビューなら、同じ特徴次元は同じ値を取れ」という不変性の要求。
- 第2項(非対角を0に):違う次元 i, j は無相関 `C_ij = 0` であれ。各次元が別々の情報を担い重複しないようにする(冗長性削減)。λ は2項の重み(論文では λ≈0.0051 など小さめ)。

数値例。ある次元の対角が `C_11 = 0.7` なら、第1項は `(1-0.7)^2 = 0.09` の罰を出し、その次元を2ビュー間でもっと一致させる方向へ押します。次元1と次元2の相関が `C_12 = 0.4` なら、第2項は `λ·0.16` の罰を出し、両次元を無相関化する方向へ押します。

なぜ崩壊しないか。全特徴が同じ定数に潰れると、正規化(標準偏差1で割る)が定義できず、相関 C_ii を1にできません。各次元が「独立に意味のある変動」を持たないと損失が下がらないので、定数への逃げ道がふさがれます。stop-gradient や EMA という構造的トリックを使わず、損失そのものに崩壊回避を埋め込んでいるのが特徴です。

情報理論的な見方。損失は、2ビューの埋め込み間の情報を残しつつ(対角=不変)、次元間の冗長性を減らす(非対角=脱相関)という、情報ボトルネック(information bottleneck)に近い目標と解釈できます。Tsai et al.(2021, arXiv:2104.13712)は、Barlow Twins の損失が、適切な仮定のもとで「負例なしの対照学習」や相互情報量(mutual information)の最大化と関係づけられることを示し、神経科学的直感に情報理論の裏づけを与えました。

高次元に強いという珍しい性質。多くの対照学習は射影次元を大きくしてもさほど伸びませんが、Barlow Twins は D を大きくするほど性能が上がります。直感的には、非対角を0にする冗長性削減のおかげで、増やした次元が「同じことの繰り返し」にならず新しい情報を担えるためです。

[VICReg](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) との関係。VICReg の「不変+分散+共分散」と狙いがよく似ています。Barlow Twins は2ビュー間の相互相関1本で「対角=不変」「非対角=脱相関」をまとめており、VICReg は分散項と共分散項を各枝で独立に置くことで、片枝の統計だけでも崩壊を防げるよう一般化した、と整理できます。言い換えると、Barlow Twins の相互相関の対角を VICReg の不変項+分散項に、非対角を VICReg の共分散項に対応づけられます。

### 手法の系譜と主要論文
- Barlow(Horace Barlow, 1961)。冗長性削減原理を提唱した神経科学の古典。手法名と思想の源流。
- SimCLR(Chen et al., ICML 2020, arXiv:2002.05709)/ MoCo(He et al., CVPR 2020, arXiv:1911.05722)。負例で崩壊を防ぐ対照学習。
- BYOL(Grill et al., NeurIPS 2020, arXiv:2006.07733)。EMA + predictor + stop-gradient で負例なし崩壊回避。
- SimSiam(Chen & He, CVPR 2021, arXiv:2011.10566)。predictor + stop-gradient が最小要件と特定([SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl))。
- Barlow Twins(Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny, ICML 2021, arXiv:2103.03230)。相互相関行列を単位行列に近づける情報理論的アプローチ。負例・EMA・stop-gradient のいずれも使わず崩壊を防ぐ点と、高次元ほど良くなる性質が新規性。
- Tsai et al.(2021, arXiv:2104.13712)。Barlow Twins を負例なし対照学習・相互情報量最大化と結びつけた理論的考察。
- VICReg(Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906)。Barlow Twins の発想を分散・不変・共分散の3項に分解し、片枝統計で崩壊を防げるよう一般化した後続([VICReg](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。

### 論文の実験結果(定量データ)
評価は ImageNet 線形評価(特徴を凍結し線形分類器1層で top-1 正解率。1000クラス、チャンスレベル約 0.1%)が中心です。

Barlow Twins は ResNet-50・1000エポックの事前学習で ImageNet 線形評価 top-1 約 73.2% を報告。BYOL(約 74.3%)や SwAV(約 75.3%)に迫り、SimCLR(約 69%)を上回ります。負例・EMA・stop-gradient なしでこの水準に達したことが成果です。

半教師あり(ラベル 1%/10%)では、ResNet-50 で 1% 時 top-1 約 55.0%、10% 時 約 69.7% を報告し、ラベルが乏しい状況でも有用な特徴を学べることを示しました。

射影次元 D への依存(特徴的なアブレーション)。Barlow Twins は projector の出力次元を 256 → 1024 → 8192 → 16384 と上げるほど線形評価精度が単調に向上する傾向を示しました(8192 付近で約 73% に到達)。これは冗長性削減項が大きな次元を有効活用させるためで、対照学習が次元飽和しやすいのと対照的です。一方で D を大きくすると `[D,D]` 行列の計算・メモリが二乗で増える代償があります。

バッチサイズ依存性。Barlow Twins は相関をバッチ内統計で推定するため、極端な小バッチでは推定が荒れますが、SimCLR ほど大バッチ必須ではなく、中程度のバッチ(数百〜2048)で安定します。論文はバッチ 128〜2048 で精度がほぼ平坦になることを示し、SimCLR(大バッチで顕著に伸びる)との対比を示しました。

冗長性削減項のアブレーション。非対角を抑える項(λ 項)を外すと、次元間の相関が残って有効次元が減り、精度が低下します。逆に不変項(対角を1にする項)を弱めると2ビューを揃える信号が薄れます。両項のバランス(λ)が重要で、論文は λ≈0.005 付近が良いと報告しています。

データ拡張への頑健性。Barlow Twins は、対照学習に比べて拡張の種類・強さの選択にやや頑健(性能の落ち込みが緩やか)という報告があり、これは負例に依存せず統計量で学ぶ性質の利点とされます。

### メリット・トレードオフ・限界
メリット
- 崩壊回避と冗長性削減を1本の損失式で同時に実現し、思想が明快(情報理論的・神経科学的な根拠を持つ)。
- 負例・EMA・stop-gradient のいずれも不要で、構成が素直。
- 射影次元 D を大きくするほど性能が伸びやすい(高次元に強い)。
- データ拡張の選択に比較的頑健。

トレードオフ・限界
- 相互相関行列が `[D, D]` のため、D を大きくすると計算・メモリが二乗で増える。高次元の利点と計算コストがトレードオフ。
- 2項の重み λ の調整が必要。
- バッチ内統計に基づくため、極端に小さいバッチでは相関推定が不安定になりやすい。
- 相関(線形の関係)を消すだけなので、次元間の非線形な依存(相関は0でも独立でない関係)までは除けない。真の統計的独立を保証するわけではない。
- 相互相関は「2ビュー間」をまたぐ量なので、片枝が固定された別モダリティや異なる次元の枝には素直に適用しにくい(この制約を外したのが VICReg)。

### 発展トピック・研究の最前線
Barlow Twins は、崩壊回避を「構造のトリック」から「損失に書く情報理論的目標」へ移した代表例であり、VICReg をはじめとする正則化ベースの自己教師あり学習の直接の先駆けです。研究コミュニティでは、対照学習・非対照学習・正則化系を「表現の分散を保ち次元間相関を減らす」という一つの原理で統一的に理解する流れが強まっており、Barlow Twins の相互相関の対角・非対角という分解はその原理を最も簡潔に示した形と見なされています。発展としては、相関だけでなく高次の統計的独立を狙う手法、相互相関行列の計算を低ランク近似や部分サンプリングで軽量化する工夫、画像以外(動画・点群・マルチモーダル)への拡張、そして特徴空間で予測する世界モデル系の補助正則化として冗長性削減項を使う試みなどが進んでいます。

### さらに学ぶための関連トピック
- [VICReg](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Stop-gradient (勾配停止)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [V-JEPA 2](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [LeCunの自律機械知能構想](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)


## Batch Normalization

### ひとことで言うと
Batch Normalization (バッチ正規化、略して BatchNorm や BN) とは、ニューラルネットワークの途中を流れる数値 (活性、activation) を、ミニバッチ (一度に処理する数枚〜数十枚のデータのまとまり) 単位で「平均 0・分散 1」に揃え直す処理です。各層に入ってくる値のスケールを毎回そろえることで、学習が速く・安定して進むようになります。畳み込みニューラルネット (CNN、画像処理の定番アーキテクチャ) の学習では事実上の標準部品になっています。

### 直感的な理解
工場の組み立てラインを想像してください。各工程 (層) は、前の工程から流れてくる部品 (数値) を受け取って加工し、次へ渡します。もし前工程が渡す部品のサイズが日によって 1cm だったり 100cm だったりとバラバラだと、各工程はそのたびに設定を調整し直さねばならず、ライン全体が混乱します。逆に、各工程の手前で部品のサイズを毎回一定の規格にそろえてやれば、後ろの工程は安定した入力を前提に作業でき、ライン全体がスムーズに流れます。

ニューラルネットでも同じことが起きます。学習中、ある層の重みが更新されると、その層が出す数値の分布 (平均や広がり) が変わります。すると次の層から見れば「入力の分布がコロコロ変わる」状態になり、層が深いほど上流の小さな変化が下流で増幅されて伝わります。BatchNorm は「各層の入力分布を毎回そろえてしまえば、下流の層は安定したスケールの入力を受け取れる」という発想で、この問題を緩和します。

### 基礎: 前提となる概念
活性 (activation) とは、層の出力として次の層へ渡される数値のことです。画像なら各位置・各チャンネルの値の集まりで、何百万個もの数値が層から層へ流れます。

ミニバッチ (mini-batch) とは、訓練データ全体を一度に処理する代わりに、数枚〜数十枚ずつ小分けにして処理するそのまとまりです。バッチサイズ (batch size) はその枚数で、たとえば 32 や 64 がよく使われます。BatchNorm はこのミニバッチ内の複数サンプルから統計 (平均・分散) を計算するので、バッチサイズが BatchNorm の挙動を左右します。

活性化関数 (activation function) とは、層の出力に非線形性を与える関数で、sigmoid や tanh、ReLU などがあります。sigmoid や tanh は入力が大きすぎると出力が端 (0 や 1) に張り付き、そこでは勾配 (微分) がほぼ 0 になります。これを飽和 (saturation) と呼び、飽和すると勾配が流れず学習が止まります。入力のスケールが暴れると飽和が起きやすくなるため、スケールをそろえる BatchNorm が効きます。

論文の著者はこのスケール変動を Internal Covariate Shift (内部共変量シフト、層への入力分布が学習中に変化し続けること) と名付け、これが学習を遅く・不安定にする原因だと主張しました。具体的には、(1) 値が大きくなりすぎると活性化関数が飽和して学習が止まる、(2) 分布が変わるたびに各層が適応をやり直すので学習が遅い、(3) 安定のため学習率を小さくすると収束に時間がかかる、という連鎖が起きます。

### 仕組みを詳しく
BatchNorm 層は、入力されたミニバッチに対して 1 つの特徴 (チャンネル) ごとに次の処理をします。

ステップ 1: ミニバッチ内の平均と分散を計算
ミニバッチに B 個のサンプルがあるとき、その特徴の値 `x_1, ..., x_B` から平均 `μ` と分散 `σ²` を計算します。

```
μ  = (x_1 + ... + x_B) / B            ← バッチ内平均
σ² = Σ(x_i − μ)² / B                  ← バッチ内分散
```

ステップ 2: 正規化 (平均 0・分散 1 に揃える)

```
x̂_i = (x_i − μ) / √(σ² + ε)
```

`ε` (イプシロン) は 0 割りを防ぐ極小の定数 (例: 1e-5) です。これで `x̂` は平均 0・分散 1 になります。

ステップ 3: スケールとシフトを学習で取り戻す
「平均 0・分散 1」に固定するとネットワークの表現力が下がることがあります。そこで学習可能なパラメータ `γ` (gamma、スケール) と `β` (beta、シフト) を掛けて足し、必要なら正規化を打ち消せるようにします。

```
y_i = γ × x̂_i + β
```

`γ, β` は普通の重みと同じく学習で最適化されます。極端には `γ=√σ², β=μ` を学べば正規化を完全に元に戻すこともできる柔軟さを持ちます。

数値例: ミニバッチ内のある特徴が `[2, 4, 6, 8]` だったとします。`μ=5, σ²=5`。正規化すると約 `[−1.34, −0.45, 0.45, 1.34]` になり、平均 0・分散 1 にそろいます。

CNN での tensor 形状: 入力が `(B, C, H, W)` (B=バッチ, C=チャンネル, H×W=空間サイズ) のとき、BatchNorm はチャンネル `C` ごとに、`B×H×W` 個の値全体で平均・分散を取ります。つまり `γ, β` はチャンネル数 `C` ぶんだけ存在します。たとえば `(32, 64, 56, 56)` の特徴マップなら、64 チャンネルそれぞれについて 32×56×56 = 約 10 万個の値から平均と分散を計算します。この「空間方向も込みで統計を取る」点が、画像 1 枚だけで正規化する InstanceNorm との違いです。

学習時と推論時の違い (重要):
- 学習時: 上記のとおりミニバッチの統計 (`μ, σ²`) を使う。同時に、各 BatchNorm 層は running mean / running variance という移動平均を `running = (1−m)·running + m·batch_stat` の形 (m はモメンタム、例 0.1) で更新し続けます。
- 推論時: 1 枚ずつ処理することもあり、バッチ統計が不安定になります。そこで学習中に蓄積した移動平均の統計 (running mean / running variance) を使って正規化します。だから推論前にはモデルを評価モード (`model.eval()`) に切り替える必要があります。これを忘れると推論結果がおかしくなる、というのは初心者がよく踏むワナです。

なぜ表現力を取り戻す `γ, β` が必要かをもう少し説明します。たとえば sigmoid の手前で常に平均 0・分散 1 にしてしまうと、sigmoid のほぼ線形な中央付近しか使われず、非線形性が活かせません。`γ, β` があれば、ネットワークは「正規化を効かせる」か「元に戻して非線形領域を使う」かを学習で選べます。この自由度が表現力の低下を防ぎます。

逆伝播 (backpropagation) の観点も触れておきます。BatchNorm は単に値を割るだけでなく、`μ` と `σ²` 自体が同じミニバッチの全サンプルに依存するため、勾配計算ではこの依存を通る経路も考慮されます。結果として、あるサンプルの勾配が他のサンプルにも影響する「バッチ内の相互作用」が生じ、これがバッチ統計のノイズ (同じサンプルでも一緒に入るバッチが違えば正規化結果が少し変わる) とともに、わずかな正則化効果の源になっています。

### 手法の系譜と主要論文
Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015 (arXiv:1502.03167) が BatchNorm を提案した原論文です。Internal Covariate Shift を問題として定式化し、その緩和策として上記の正規化層を提案しました。動機は「層への入力分布を安定させれば、もっと大きな学習率を使えて速く学習できる」こと。副次効果として、バッチ統計に含まれるノイズがわずかな正則化として働き、Dropout を減らせることも報告しました。

Santurkar, Tsipras, Ilyas, Madry, "How Does Batch Normalization Help Optimization?", NeurIPS 2018 (arXiv:1805.11604) は、BatchNorm がなぜ効くのかを再検証しました。主張は「効果の本質は Internal Covariate Shift の削減ではなく、損失関数の地形 (loss landscape) を滑らかにすること」だというものです。彼らは、BN 層の直後にわざとノイズを注入して分布をわざと暴れさせても (= ICS を意図的に悪化させても)、BN ありなら学習がうまくいくことを実験で示し、ICS 削減説に疑問を投げかけました。代わりに、BN が損失や勾配の変動 (Lipschitz 定数で測られる滑らかさ) を小さくし、最適化を楽にしていることを定量化しました。「BN は確かに効くが、その理由は原論文の説明とは別かもしれない」という重要な議論を提起した点で、手法理解を更新させた論文です。

Ba, Kiros, Hinton, "Layer Normalization", 2016 (arXiv:1607.06450) は、バッチ方向ではなく 1 サンプル内の特徴方向で正規化する手法を提案しました。バッチサイズに依存しないため、系列長が可変な RNN や、Transformer のような系列モデルで好まれます。

Wu & He, "Group Normalization", ECCV 2018 (arXiv:1803.08494) は、BatchNorm の弱点 (小さいバッチで不安定) を克服する代替手法です。バッチ方向ではなくチャンネルをグループに分けて正規化するため、バッチサイズ 1 でも安定します。動機は、物体検出やセグメンテーションのようにメモリ制約でバッチを小さくせざるを得ないタスクへの対応でした。

この系譜は「正規化の軸をどこに取るか」の探求として整理できます。`(B, C, H, W)` の tensor で、BatchNorm は `B×H×W` 方向 (チャンネルごと)、LayerNorm は `C×H×W` 方向 (サンプルごと)、GroupNorm は `C` をグループに割って各グループ×`H×W`、InstanceNorm は `H×W` 方向 (1 サンプル 1 チャンネル) で正規化します。タスクとアーキテクチャに応じて使い分けられ、Transformer 系が LayerNorm を標準採用する流れも、この「バッチ非依存正規化」の系譜上にあります。

### 論文の実験結果(定量データ)
評価指標は ImageNet 画像分類の top-1 / top-5 accuracy (正解率)、および「同じ精度に到達するまでに必要な学習ステップ数」です。後者は学習の速さを測る指標で、少ないステップで同じ精度に届けば学習が高速化したことを意味します。

Ioffe & Szegedy (2015) の結果は劇的でした。当時の代表的な画像分類ネットワーク (Inception) に BatchNorm を入れると、同じ精度に到達するのに必要な学習ステップ数を約 1/14 に削減できたと報告されています。さらに BatchNorm により学習率を従来の約 5〜30 倍に上げても安定して学習でき、最終精度も改善しました。ImageNet で当時の最高水準を更新し、BatchNorm を入れた複数モデルのアンサンブルで top-5 error を 4.9% 程度まで下げ、人間の性能に匹敵する水準に達したと報告されています。アブレーション的には、BatchNorm を外すと大きな学習率では発散 (学習が破綻) し、小さな学習率に戻すと収束が大幅に遅くなることが示され、BatchNorm が「大きな学習率を可能にする」中心的役割を果たしていることが裏づけられました。

Santurkar ほか (2018) は、VGG ネットワークで BN ありとなしを比較し、BN ありの方が学習中の損失・勾配の変動が一貫して小さいこと (= 地形が滑らか) を測定しました。さらに、BN 層の後に分布を乱すノイズを加えても学習速度がほとんど落ちないことを示し、「速さの源泉は ICS 削減ではない」という主張を支えました。

Wu & He (2018) の Group Normalization では、ImageNet で ResNet-50 を学習する際、バッチサイズを 32 から 2 へ小さくすると BatchNorm のエラーが大きく悪化する (報告ではバッチ 2 で誤差が 10 ポイント超も増加) のに対し、Group Normalization はバッチサイズにほぼ依存せず安定した精度を保ちました (バッチ 32 と 2 の差がおよそ 0.x ポイント程度)。これは「BatchNorm は小バッチに弱い」という弱点と、その克服策の有効性を定量的に示した代表的な結果です。

### メリット・トレードオフ・限界
メリット:
- 学習が速く収束し、大きな学習率を使えるようになる。
- 学習が安定し、重みの初期値に対する敏感さが減る。
- バッチ統計のノイズによるわずかな正則化効果があり、過学習をいくらか抑える。

トレードオフ・限界:
- 小さいバッチサイズでは統計が不正確になり、性能が落ちる (バッチ依存)。
- 学習時と推論時で挙動が変わり、`eval` モードへの切り替え忘れがバグの温床になる。
- 系列長が可変な RNN や Transformer とは相性が悪く、そこでは LayerNorm が好まれる。
- 分散学習でバッチ統計をデバイス間で同期するか否か (SyncBN) など、実装上の考慮が増える。サンプルが各 GPU に分散していると、各 GPU ローカルのバッチで統計を取るか全体で同期するかで結果が変わる。
- 一部の層を凍結する転移学習では、凍結したつもりでも running statistics が更新され下層の挙動が変わる落とし穴がある。
- バッチ内のサンプルが互いに影響し合うため、各サンプルの出力がバッチ構成に依存してしまい、契約上「1 サンプルの出力は他サンプルに依存しない」ことを要求する一部のタスク (一部の生成モデルや対照学習) で問題になることがある。

研究上の未解決の論点として、BatchNorm がなぜ効くのかの完全な理論的説明は依然として議論が続いており (ICS 説 vs 地形平滑化説 vs 他の要因)、正規化の最適な軸や、正規化を一切使わずに同等の学習安定性を得る方法 (normalizer-free な設計) の探求が続いています。

### 発展トピック・研究の最前線
正規化を使わずに深いネットワークを安定して学習する研究も進んでいます。Brock ほかの NFNets (Normalizer-Free Networks, ICML 2021, arXiv:2102.06171) は、BatchNorm を取り除く代わりに適応的勾配クリッピング (Adaptive Gradient Clipping、勾配がパラメータの大きさに対して大きくなりすぎたら抑える) と注意深い重みスケーリング (Scaled Weight Standardization) を組み合わせ、BatchNorm 付きモデルに匹敵する、あるいは ImageNet で上回る精度を達成しました。動機は、BatchNorm のバッチ依存性・学習推論の不一致・分散学習での同期コストといった欠点を避けることです。

また、正規化と残差接続 (residual connection、層をスキップして信号を通す経路) の相互作用や、Transformer における Pre-LN と Post-LN (LayerNorm を残差ブロックの前に置くか後に置くか) の違いが学習安定性に与える影響も活発な研究テーマです。Pre-LN は学習初期の勾配を安定させ warmup なしでも学習しやすい一方、最終精度は Post-LN がわずかに上回る場合がある、といったトレードオフが報告されています。さらに、BatchNorm の代替として正規化を簡略化した RMSNorm (分散だけで割り平均は引かない、計算が軽い) が大規模言語モデルで広く採用されるなど、正規化レイヤの設計は今も進化を続けています。

実務的には、画像系の CNN backbone では今も BatchNorm が標準ですが、Transformer 系では LayerNorm、メモリ制約で小バッチになる検出・セグメンテーションでは GroupNorm、というように、アーキテクチャとバッチサイズに応じた使い分けが定着しています。

### さらに学ぶための関連トピック
- [Transfer Learning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [Early Stopping](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Layer-wise LR Decay (LLRD)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [Gradual Unfreezing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [Discriminative / Layer-wise Learning Rates](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)


## BEVPool / BEVPoolv2 (高速voxel pooling)

### ひとことで言うと
カメラ画像から作った特徴を「真上から見た地図(BEV)」に並べ替える計算は、自動運転の認識システムの中でとても重い処理です。BEVPool と BEVPoolv2 は、その並べ替えに必要な「どの画像ピクセル・どの奥行き候補が、BEV のどのマスに入るか」をあらかじめ計算してメモリに焼き込んでおき、推論(実際に走らせるときの計算)を劇的に速くする実装テクニックです。新しい認識アルゴリズムを提案したのではなく、既存のアルゴリズム(LSS の Splat)を精度を一切変えずに高速に動かす「計算の段取り」の工夫である点が最大の特徴です。車載デプロイ(エッジ GPU、TensorRT)で BEV 検出を現実的な速度にする鍵になりました。

### 直感的な理解
郵便の仕分けを想像してください。何十万通もの手紙を、宛先の地区ごとの棚に振り分ける作業です。素直なやり方は、手紙を1通ずつ見て宛先を読み、対応する棚まで歩いて入れることです。これだと毎回「宛先を読む」「棚まで歩く」を繰り返すので遅い。

しかし、もし手紙の差出人と宛先のパターンが毎回まったく同じだと分かっていたら、「この差出人の手紙は必ずこの棚」という対応表を一度作ってしまえば、本番では中身を読まずに表を見るだけで一気に振り分けられます。BEVPoolv2 がやっているのはまさにこれです。車載カメラは走行中に位置・向きが変わらないので、「画像のどこ・どの奥行きが、BEV のどのマスに行くか」の対応関係は走り出す前に1回計算すれば、あとは毎フレーム使い回せるのです。さらに賢いのは、その対応表を引くときに「ネットワークが出した特徴と奥行き確率」だけを実行時に掛け合わせる形にした点で、重い座標変換とソートをすべて事前計算側に追い出しています。

### 基礎: 前提となる概念
言葉を平易に整理します。

BEV (Bird's-Eye-View) は鳥が真上から見たような俯瞰視点です。前後左右の複数カメラ画像を、最終的に自車を中心とした真上の地図に統合します。この地図上で「ここに車」「ここが車線」と判断するほうが、複数カメラの情報をまとめやすく距離も扱いやすいからです。詳しくは [BEVグリッド](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)。

voxel (ボクセル) は3次元空間を細かい立方体のマスに区切ったときの1マスです。pixel が画像の2次元のマスなら、voxel はその3次元版です。BEV では高さ方向もマス目に区切ることがあります。

pooling (プーリング) は、ここでは「たくさんの値を1つのマスに集めて合計や平均をとる集約処理」を指します。Splat は「同じ BEV マスに落ちた点をすべて足す」sum pooling です。

LSS (Lift-Splat-Shoot、[Lift-Splat-Shoot (LSS)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)) は、各ピクセルについて「奥行きの確率分布」を予測します。たとえば各ピクセルの特徴ベクトルに「1m先20%、2m先30%、3m先50%」のような奥行き確率を掛け、視線方向に伸びる多数の3次元点を作ります。これを Lift(持ち上げる)と呼びます。式で書けば、視線上の深度 d の点の特徴は `α_d · c`(c は文脈特徴 C 次元、α_d は深度確率)です。

CUDA カーネル は、NVIDIA の GPU 上で多数のスレッド(並列の処理単位)が同時に走るように書かれたプログラムです。GPU は「同じ処理を大量のデータに一斉にかける」のが得意な一方、バラバラなメモリアクセス(ランダムアクセス)や、複数スレッドが同じ場所に同時に書き込む atomic 加算は苦手です。高速化はこの特性に合わせて処理を組み直すことが鍵になります。

メモリ帯域 (memory bandwidth) は GPU がメモリを読み書きできる速度です。深層学習の多くの処理は計算ではなくメモリ読み書きが律速(memory-bound)で、無駄な中間配列を作るとここが詰まります。

### 仕組みを詳しく
#### 何が重いのか
LSS で Lift した後、作られた膨大な3次元点を BEV のマス目に振り分けて合計する処理を Splat(叩きつける)と呼びます。これがとても重い。具体的な数で考えます。

- 6台のカメラ、各カメラの特徴マップが 16x44 ピクセル、奥行き候補が 59 段階、特徴次元が 80 だとすると、Lift で生成される点(frustum 点)の数は 6 x 16 x 44 x 59 = 約 249,000 点になります。高解像度設定ではこれが数百万点に跳ね上がります。
- これらの点を BEV グリッド(例 128x128 マス)に振り分け、同じマスに落ちた点を全部足し合わせる必要があります。1マスに数十〜数百点が落ちます。

「どの点がどのマスに入るか」を毎回計算し、しかも複数の点が同じマスに入るので合計する処理は、素直に書くとメモリアクセスがバラバラで、GPU が苦手とするパターンになります。

#### もとの遅い方法 (LSS のソート + 累積和)
LSS の元実装は次の手順を取りました(cumsum trick)。

1. Lift で約24.9万点と、それぞれが落ちる先の BEV マスのインデックスを作る。
2. インデックスでソート(並べ替え)する。同じマスに入る点が隣り合うようにするため。
3. ソート済みの特徴に対して累積和(prefix sum、前から順に足していった合計の列)をとる。
4. マスの境界で差分(隣り合うマス境界の累積和の引き算)をとり、各マスの合計を取り出す。

この方法は正しいのですが、(a) ソートが毎フレーム必要で点数に比例して遅く、(b) 累積和のために全点ぶんの巨大な中間配列をメモリに置く必要があり、メモリと帯域を食います。BEVDet という車載向け検出器を作る過程で、Huang らはこの Splat(view transformation)が推論時間の大半を占めると気づき、高速化に取り組みました。

#### BEVPool の改善
BEVPool(v1)は専用の CUDA カーネルを書いて、各 BEV マスごとにそこへ落ちる点を直接合計します。GPU の多数のスレッドにマス(と区間)を割り当て、それぞれが担当マスに属する点だけを足すので、巨大な累積和の中間配列を作らずに済みます。これだけでも数倍速くなりました。ただし v1 はまだ Lift 後の `[N, D, C, H, W]` 相当の大きな特徴テンソルを実体として作る必要が残っていました。

#### BEVPoolv2 の核心: 計算を「事前計算」と「実行時」に分け、Lift を遅延させる
BEVPoolv2 の最大の発見は2つあります。

第一に、「カメラの取り付け位置・向きと BEV グリッドの定義が固定なら、どの画像ピクセル・どの奥行き候補がどの BEV マスに対応するかは毎回同じ」という点です。車載カメラは走行中に動かないので、この対応関係(マッピング)は走り出す前に1回だけ計算すれば使い回せます。

第二に、Lift の外積(`α_d · c`)を事前に実体化しない点です。v1 までは Lift 後の巨大テンソルを作っていましたが、v2 では「特徴 c」と「深度確率 α」を別々に持っておき、BEV マスへ集約する瞬間に初めて掛け算します(乗算を集約カーネルの中に融合する)。これで `D` 倍に膨らんだ中間テンソルをメモリに置かずに済みます。

そこで処理を2段階に分けます。

事前計算(プリプロセス、1回だけ):
- すべての (カメラ, ピクセル位置, 奥行き候補) の組について、対応する BEV マスのインデックスを計算する(座標変換)。
- そのインデックスでソートしておき、`ranks_bev`(どのマスに行くか)、`ranks_feat` / `ranks_depth`(特徴・深度のどの要素を引くか)、`interval`(各 BEV マスに対応する区間の開始位置と長さ)といった補助テーブルを作ってメモリに保存する。
- 重い計算(座標変換とソート)をここで全部済ませる。

実行時(毎フレーム、リアルタイム):
- ネットワークが出力するのは特徴ベクトル(値、`[N,C,H,W]`)と奥行き確率(重み、`[N,D,H,W]`)だけ。
- 事前計算したテーブルを引いて、各 BEV マスに対応する区間の (特徴 × 奥行き確率) を直接合計する。
- ソートも座標変換も実行時には不要。テーブルを引いて掛けて足すだけ。

before / after を tensor の形で見ると次のようになります。

```
入力(実行時):
  feat  (文脈特徴) : [N, C, H, W]   ネットワーク出力
  depth (奥行き確率): [N, D, H, W]  ネットワーク出力
  事前計算テーブル(固定):
    ranks_bev   : [M]        各有効点が落ちる BEV マスの通し番号
    ranks_feat  : [M]        その点が引く feat の要素インデックス
    ranks_depth : [M]        その点が引く depth の要素インデックス
    interval    : [K, 2]     各 BEV マスの (区間開始位置, 個数)
    （M = ソート後の有効な (点, マス) 対応の総数）

出力:
  bev_feat : [C, Y, X]       BEV グリッド上の集約済み特徴(例 80 x 128 x 128)
```

実行時の CUDA カーネルは、各 BEV マス(出力の1要素、あるいはマス×チャンネル)に1スレッドを割り当て、interval テーブルが示す区間の点だけを「feat[ranks_feat] × depth[ranks_depth]」で足し合わせます。スレッドごとに出力先が決まっているので atomic 加算が不要で、中間の巨大配列も作らないため、メモリ使用量が小さく、メモリ帯域の無駄が減ります。GPU が得意な「整然とした coalesced(連続)アクセス」に処理を作り変えているのがポイントです。

### 手法の系譜と主要論文
この系譜は「アルゴリズムは LSS のまま、実装だけを段階的に速くしていく」流れとして読めます。

出発点が LSS (Philion & Fidler, ECCV 2020, arXiv:2008.05711、[Lift-Splat-Shoot (LSS)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)) です。ピクセルごとの奥行き確率分布で特徴を3次元に持ち上げ、BEV に集約する枠組みを作りましたが、Splat の実装はソート + 累積和で、メモリと速度の両面で重いものでした。

BEVDet (Huang, Huang, Zhu, Du; 2021, arXiv:2112.11790、[BEVDet](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)) と BEVDet4D (Huang & Huang, 2022, arXiv:2203.17054) は LSS を多カメラ3次元検出に実用化したパイプラインです。BEVPool / BEVPoolv2 はもともと、この BEVDet を実機に載せて動かせるようにするための工学的改良として生まれました。BEVDet の論文・コードリリースの中で view transformation の高速化が段階的に進みました。

そして BEVPoolv2 (Huang & Huang, 2022, arXiv:2211.17111「BEVPoolv2: A Cutting-edge Implementation of BEVDet Toward Deployment」) が、Splat を「事前計算テーブル + Lift 遅延 + 専用 CUDA カーネル」に分解する決定版を提示しました。論文というより技術レポート / 実装最適化レポートに近く、新しい認識アルゴリズムではなく既存アルゴリズムを高速に動かす技法である点が際立っています。同じ「投影インデックスの事前計算」という発想は、幾何投影ベースの BEV 手法 [M2BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform) や軽量 BEV の [FastBEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) でも採用されており、リアルタイム BEV 実装の共通テクニックになっています。

### 論文の実験結果(定量データ)
BEVPoolv2 の評価は、認識精度ではなく速度とメモリで測ります(精度は数学的にほぼ不変なので測る意味が薄い)。

評価する量は主に2つです。レイテンシ (latency) は1回の処理にかかる時間で、短いほど良い(ミリ秒、ms)。リアルタイム性が要る車載では決定的に重要です。メモリ消費 は処理に必要な GPU メモリで、小さいほど車載の限られたメモリに収まりやすい。

技術レポートの報告では、BEVDet の view transformation(視点変換、すなわち Splat)部分が、もとの実装に比べて桁違いに速くなったとされています。

- view transformation 単体の高速化は、もとの LSS 実装に対しておおむね 数十倍規模(報告では従来比で大幅、ある設定で 1ms 未満まで短縮)と報告され、view transformation がもはやパイプラインのボトルネックでなくなったと述べられています。
- 中間テンソルを実体化しない(Lift を遅延する)ことで、メモリ消費も大幅に削減され、より高い BEV 解像度や多フレーム入力が同じメモリで扱えるようになりました。

重要なのは、これが「計算の段取り」を変えただけで、出力される BEV 特徴は数学的にほぼ同じということです。つまり mAP / NDS のような検出精度を一切犠牲にせず、速度とメモリだけを得ています。これは精度と速度がトレードオフになりがちな通常の研究とは対照的な、純粋な実装利得です。浮動小数点の加算順序が変わるため厳密にビット一致はしませんが、検出指標に有意な差は出ません。

view transformation がパイプライン全体のレイテンシに占める割合は大きかったため、ここをほぼゼロコストにすることでモデル全体の推論速度が実用域に入り、TensorRT(NVIDIA の推論高速化ライブラリ)へのカスタムプラグイン組み込みも容易になったと報告されています。これにより BEVDet 系がエッジ GPU(車載 SoC)で動かせるようになりました。

### メリット・トレードオフ・限界
メリット

- 認識精度を変えずに view transformation を桁違いに高速化できる(段取りの最適化だから)。
- Lift を遅延し中間配列を作らないのでメモリ消費が小さく、車載の限られた GPU メモリに優しい。高解像度・多フレーム化の余地が生まれる。
- TensorRT などの推論エンジンにプラグインとして組み込みやすく、実機デプロイの実用性が高い。

トレードオフ・限界

- カメラの取り付け位置・向き(校正値)と BEV グリッドの定義が固定であることが前提。これが変わると事前計算テーブルを作り直す必要がある。校正が走行中に変動する設計や、カメラごとに異なる校正を動的に扱う設計とは相性が悪い。
- 専用 CUDA カーネルの実装・保守が必要で、フレームワークやハード(GPU 世代、TensorRT バージョン)が変わると移植コストがかかる。
- LSS 系(奥行き確率で持ち上げる方式)の Splat に特化した最適化であり、まったく別の BEV 変換方式(MLP ベース、純粋な幾何投影、クロスアテンション型)には直接は適用できない。
- 奥行き予測そのものの精度は改善しない(あくまで集約の高速化)。精度は元の LSS / BEVDet の性能に依存する。
- sum pooling 前提。max pooling や学習可能集約には別実装が要る。

研究・実装上の残る課題は、可変校正(校正が動的に変わる、複数車種を1モデルで扱う)への対応、別系統の BEV 変換への一般化、そして「事前計算テーブルのメモリ」と「実行時の速度」のバランス(高解像度ではテーブル自体が大きくなり別のボトルネックになりうる)です。

### 発展トピック・研究の最前線
BEVPoolv2 が示した「固定校正なら投影を事前計算でき、Lift は遅延できる」という洞察は、その後のリアルタイム BEV 実装の標準テクニックになりました。

幾何投影系への波及。均一深度仮定でボクセルに投影する [M2BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform) や、軽量化を狙う [FastBEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、[Simple-BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform) も、投影インデックスの事前計算という同じ発想を共有しています。

デプロイ最適化全般。TensorRT カスタムプラグイン化、量子化(INT8 など計算精度を下げてさらに速くする)、ONNX エクスポート、車載 SoC への移植など、BEV 検出器をエッジに載せるための一連の最適化が研究・実装の主戦場になっています。BEVPoolv2 はその「最初の関門(view transform を速くする)」を解いた位置づけです。

差別化された集約。単純な合計ではなく、最大値プーリングや重み付き集約、あるいは Splat 自体を学習可能にする試みもあり、速度を保ちつつ表現力を上げる方向が探られています。

クエリベースとの棲み分け。前向き投影(LSS/BEVPool 系)はインデックス事前計算で高速化できる一方、後ろ向きのクロスアテンション型 BEV([M2BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform) の対比で言う query-based)は別種の最適化(deformable attention の効率化、サンプリング点の事前計算)が必要で、両者でデプロイ最適化のレシピが分かれています。

### さらに学ぶための関連トピック
- [Lift-Splat-Shoot (LSS)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [FastBEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Simple-BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [M2BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [BEVグリッド](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [BEVDet](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)


## Cosine Annealing with Warm Restarts (SGDR)

### ひとことで言うと
学習率(ニューラルネットを学習させるとき「パラメータを1回どれくらいの幅で更新するか」を決める数値)を、なめらかな下り坂(コサインカーブ)で少しずつ下げていき、底まで下がったら一気に元の高さに戻す、という上げ下げを何度も繰り返す学習率スケジュール(学習率を時間とともに変化させる仕組み)です。底に張り付いて動けなくなった状態から、もう一度勢いを与えて別のもっと良い場所を探させるのが狙いです。Loshchilov と Hutter が 2017 年に提案した SGDR(Stochastic Gradient Descent with Warm Restarts)が出発点で、現在の深層学習でもっとも広く使われる「単一サイクルのコサイン減衰」の親にあたります。

### 直感的な理解
ニューラルネットの学習は、広大な地形のどこかにある「一番低い谷」を探して降りていく作業に例えられます。あなたは目隠しをしてボールを転がしながら、できるだけ深い谷の底を探しています。

このとき、ボールを動かす力(=学習率)が大きすぎると、谷を飛び越えてしまって落ち着けません。小さすぎると、たまたま落ちた浅い谷(局所解。周りより低いがもっと深い谷が他にある凹み)から二度と出られなくなります。

昔ながらのやり方は「最初は大きな力で大胆に動かし、だんだん力を弱めて細かく調整する」というものでした。これはこれで合理的です。しかし、もし途中で浅い谷に落ちてしまっていたら、力を弱めた後ではもう抜け出せません。

SGDR の発想はシンプルです。「いったん力を弱めきって谷の底に着いたら、もう一度力を強く戻して、別の谷を探させればいい」。何度か繰り返すうちに、もっと深い谷に行き着く可能性が高まります。しかも、ボール(=これまで学習したパラメータ)はリセットせず、位置はそのままで「押す力だけ」を戻します。だから「温かい(warm)再起動」と呼ばれます。すべてをランダムからやり直す「冷たい(cold)再起動」とは違い、これまでの成果を捨てません。ここが warm restart という名前の核心です。

### 基礎: 前提となる概念
理解のために、いくつかの用語を噛み砕いておきます。

- 損失(loss): モデルの予測がどれだけ間違っているかを表す1つの数値。小さいほど良い。学習とはこの損失をできるだけ小さくすることです。
- パラメータ(重み): モデルが持つ調整可能な数値。大きなネットでは数百万から数十億個あります。学習とはこれらの値を少しずつ動かす作業です。
- 勾配(gradient): 「今いる場所で、各パラメータをどちらにどれだけ動かせば損失が下がるか」を示す矢印(ベクトル)。微分で計算します。坂の傾きにあたります。
- 確率的勾配降下法(SGD): 勾配の向きと逆(=損失が下がる向き)へパラメータを少しずつ動かしていく基本アルゴリズム。「確率的」とは、データ全体ではなく一部(ミニバッチ)で勾配を概算するという意味です。ミニバッチごとに勾配がばらつくため、更新には自然なノイズが乗ります。このノイズが探索の役割を果たすことが、後の議論で重要になります。
- 学習率(learning rate, lr): 勾配の向きに1回でどれくらいの幅で動かすかを決める係数。更新式は大まかに「新しい重み = 古い重み − 学習率 × 勾配」です。学習率はモデルの精度を左右する最重要のハイパーパラメータ(人間が事前に決める設定値)です。
- 局所解(local minimum): 周囲より低いが、地形全体で見ればもっと低い谷が他にある凹み。SGD はここに留まりやすい。
- 鞍点(saddle point): ある方向には下りでも別方向には上りになっている平らな地点。勾配がほぼゼロなので進みにくく、高次元の深層学習では局所解より鞍点のほうが圧倒的に多いとされます(Dauphin et al. 2014 の高次元解析)。
- 平坦な極小 vs シャープな極小(flat / sharp minima): 谷の底の「広さ」のこと。広くなだらかな谷(flat minimum)に落ちたモデルは、未知データでも性能が安定しやすい(汎化が良い)とされ、急峻で狭い谷(sharp minimum)は学習データに過剰適合しやすいと議論されてきました(Hochreiter & Schmidhuber 1997、Keskar et al. 2017)。

学習率を時間とともに変える仕組みをスケジューラと呼びます。コサイン減衰(cosine annealing)は、コサインカーブの形にそって学習率をなめらかに下げる方式です。「annealing(焼きなまし)」は冶金の用語で、金属を高温から徐々に冷ますと欠陥の少ない安定な結晶になることに由来します。学習率を高いところから徐々に下げる様子が、この冷却過程に似ているため使われます。SGDR はそこに「再起動」を足したものです。

### 仕組みを詳しく
1回の下り坂を1サイクル(cycle)と呼びます。あるサイクル内での学習率は次の式で決まります。

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * T_cur / T_i))
```

各記号の意味を言葉で説明します。

- `lr_max`: そのサイクル開始時(坂の一番上)の学習率。
- `lr_min`: そのサイクル終了時(坂の一番下)の学習率。多くは 0 か、0 に近い小さな値。
- `T_i`: そのサイクルの長さ。何ステップ(または何エポック)で下りきるか。
- `T_cur`: そのサイクルが始まってから今が何ステップ目か(0 から `T_i` まで進む)。
- `cos`: コサイン関数。`cos(0)=+1`、`cos(pi)=−1`。

`cos(pi * x)` は `x=0` で +1、`x=1` で −1 になります。式に入れると、`T_cur=0`(サイクル開始)のとき lr は `lr_max`、`T_cur=T_i`(サイクル終了)のとき lr は `lr_min` となり、その間をコサインカーブでなめらかに下がります。

数値例を出します。`lr_max = 0.1`、`lr_min = 0`、`T_i = 10`(10エポックで1サイクル)とすると、

- エポック0: 0.1(満タン)
- エポック2.5: 約 0.085
- エポック5(真ん中): 0.05
- エポック7.5: 約 0.015
- エポック10(底): 0.0

底に着いた瞬間、次サイクルのために lr を `lr_max = 0.1` に戻します。これが再起動です。コサインカーブを選ぶ理由は、序盤と終盤で傾きがゆるやか(中盤で最も急)になり、終盤に学習率がゆっくり 0 に近づくため、最後の細かい調整がきれいにできるからです。直線だと終盤まで一定の速さで落ち、終わり際の詰めが甘くなります。終盤に lr がゆっくり 0 へ近づく区間が、SGD のノイズを徐々に絞り込み、谷の底へ静かに沈めていく効果を持つと解釈されています。

学習率の時間変化を ASCII 図で描くと、ノコギリの歯がなめらかになった形です。

```
lr
0.1 |\        /\           /\
    | \      /  \         /  \
    |  \    /    \       /    \
    |   \  /      \     /      \
0.0 |    \/        \___/        \___
    +-------------------------------> epoch
     サイクル1   サイクル2     サイクル3
```

論文ではさらに、サイクルをだんだん長くする工夫を入れています。`T_mult`(サイクル長の倍率)というパラメータを使い、新しいサイクルが始まるたびに長さを `T_mult` 倍します。たとえば `T_0 = 10`、`T_mult = 2` なら、サイクル長は 10 → 20 → 40 → 80 … と伸びます。最初は頻繁に再起動して地形を広く探索し、後半はじっくり長く下げて精度を詰める、という設計思想です。`T_mult = 1`(等間隔再起動)も使え、その場合はノコギリ歯が等間隔で並びます。

再起動時の挙動も研究の対象です。素朴な warm restart は lr を `lr_max` に一気に戻しますが、戻した瞬間の損失の跳ね上がりが大きすぎることがあります。これを緩和するため、再起動後に短い warmup(数百ステップで lr を 0 から `lr_max` へ線形に上げる)を挟む実装もよく使われます。

副次的だが重要な効果があります。各サイクルの底に達したときのパラメータは「それぞれ違う良い谷の底」にいることが多いのです。1回の学習で複数の優秀なモデルが手に入る、ということです。これを利用したのが後述の Snapshot Ensembles と SWA です。

### 手法の系譜と主要論文
学習率を周期的に動かすという発想は、2015〜2018 年に集中的に発展しました。

1. Smith, "Cyclical Learning Rates for Training Neural Networks", WACV 2017(arXiv:1506.01186, 初版 2015)。学習率を上限と下限の間で三角波(直線的なギザギザ)で上下させる CLR を提案しました。動機は「単調に下げるだけの既存スケジュールは局所解・鞍点を脱出できない」という問題意識です。学習率を周期的に跳ね上げると、勾配がほぼゼロの鞍点や狭い谷を乗り越えやすい、と論じました。あわせて、学習率を徐々に上げながら損失を観察し、損失が急に増え始める直前を上限とする「LR range test」という実用的な手順も提案しました。CLR は SGDR の直接の先行研究です。

2. Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", ICLR 2017(arXiv:1608.03983)。CLR の三角波をコサインカーブに置き換え、さらに「サイクルごとに長さを `T_mult` 倍する」周期拡大を加えました。コサインを使う理由は、終盤に学習率を 0 へなめらかに近づけられ、各サイクルの底での収束が良くなるからです。これが本トピックの中心となる手法です。歴史的に重要なのは、この論文が広めた「コサイン形状」が、後に再起動を外した「1サイクルのコサイン減衰」として深層学習全体の事実上の標準になった点です。

3. Huang et al., "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017(arXiv:1704.00109)。SGDR のコサイン周期スケジュールを使い、各サイクルの底でモデルの重みを保存(スナップショット)し、それらを推論時にアンサンブル(複数モデルの予測を平均して精度を上げる手法)しました。各底は異なる局所解にいるため、予測を平均すると個々のモデルより頑健になります。「1回分の学習コストで M 個のモデルが手に入る」という主張です。

4. Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization"(SWA, UAI 2018, arXiv:1803.05407)。Snapshot Ensembles が「予測を平均」したのに対し、SWA は周期的(または高め一定)スケジュールで集めた重みそのものを平均します。推論時のコストが1モデル分のままで、より平坦な極小に到達でき汎化が良くなる、と示しました。SGDR の「複数の良い底を巡る」性質を、推論コストを増やさずに活用する流れです。

5. Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"(1cycle, arXiv:1708.07120, 2017-2019)。再起動を使わず、学習の前半で lr を一気に大きく上げ、後半で下げる(同時にモメンタムを逆方向に動かす)1cycle ポリシーを提案。非常に大きな学習率を一時的に許すことで「super-convergence(超収束)」と呼ぶ高速学習が起きると示しました。SGDR と問題意識を共有しつつ、再起動ではなく単一の大きな山で探索する設計です。

6. その後、SGDR の周期は Goyal らの warmup(序盤に学習率を徐々に上げる工夫)や AdamW(Loshchilov & Hutter 自身が 2019 年に提案した、重み減衰を勾配から切り離した最適化手法)と組み合わせて使われるようになりました。近年の大規模学習では「再起動なしの1サイクルのコサイン減衰 + 線形 warmup」が事実上の標準で、再起動つきの SGDR はむしろ特定の場面(長時間学習で複数チェックポイントを得たいとき、アンサンブルしたいとき)で選ばれます。

### 論文の実験結果(定量データ)
SGDR 論文の主な検証は CIFAR-10 と CIFAR-100(それぞれ 10 クラス・100 クラスの 32×32 画像分類データセット。CIFAR-10 は約 6 万枚)、加えて EEG データと downsampled ImageNet で行われました。評価指標は分類誤り率(test error、テスト画像を何パーセント間違えたか。低いほど良い)です。

報告された主な結果は次の通りです。Wide ResNet(WRN)というモデルで、通常の学習(固定学習率で長く学習し、終盤に手動で下げる従来法)と SGDR を比較しています。

- WRN-28-10(28層・幅10倍)で、従来の学習を 200 エポック回した場合とほぼ同じかそれ以上の精度に、SGDR は少ない総エポック数(たとえば B0=50 エポックなど短いサイクル構成)でも到達できると報告されました。CIFAR-10 の test error はおよそ 3.7〜4% 台、CIFAR-100 でおよそ 18〜19% 台で、いずれも従来法と同等以上です。重要なのは「同じ最終精度に、より早く・安定して到達した」点です。
- SGDR のサイクル底でのスナップショットをアンサンブルすると、単一モデルよりさらに誤り率が下がることも示されました。
- Snapshot Ensembles では、CIFAR-100 で単一モデルの test error からアンサンブルにより誤り率が下がることが報告されました(報告によると、単一モデル比でおよそ 1〜2 ポイント前後の改善)。M 個のスナップショットを平均するほど精度が上がる傾向で、追加の学習コストはゼロ(同じ1回の学習から取り出すだけ)です。ResNet-110、Wide-ResNet-32、DenseNet など複数アーキで一貫して効果が確認されました。
- SWA 論文では、CIFAR や ImageNet で、同じ学習予算のベースライン SGD に対して test error をおよそ 0.5〜1.3 ポイント改善できると報告され、しかも推論コストは1モデル分のままです。

アブレーション(どの要素を抜くとどうなるかの検証)の観点では次が読み取れます。再起動を完全になくして単調なコサイン減衰だけにすると、局所解脱出の効果が失われ、特定のタスクで最終精度がわずかに落ちます。逆に再起動だけ多用してサイクルを短くしすぎると、各サイクルが十分に収束する前に再起動がかかり、底での精度が出ません。`T_0`(初期サイクル長)と `T_mult`(拡大率)のバランスが性能を左右する、という構造です。コサインを直線(linear)や指数に替えるアブレーションでは、終盤の 0 への滑らかな近づき方がコサインの優位の源だと示唆されています。

これらの数値が示す意味は明確です。test error の数パーセントの差は、画像分類では実用上きわめて大きく(誤り率 4% と 5% では誤分類数が 25% 違う)、しかも SGDR は「追加コストなしで複数モデルが得られる」という別次元の利点を持ちます。

### メリット・トレードオフ・限界
メリット

- 局所解・鞍点から抜け出しやすく、最終精度が上がりやすい。
- 1回の学習で複数の良いモデル(各サイクルの底)が得られ、追加学習なしでアンサンブル(Snapshot)や重み平均(SWA)に使える。
- なめらかなコサインカーブなので、段差(後述の Step decay)による不安定さがない。
- 再起動を外した単一サイクル版は、ハイパーパラメータが「総ステップ数」だけと少なく、現代の事実上の標準として極めて使いやすい。

トレードオフと限界

- 再起動のたびに損失が一時的に跳ね上がるため、学習曲線がギザギザになります。途中の任意のタイミングで止めると性能が悪く、必ずサイクルの底で評価・保存する必要があります。これは運用上の制約です。
- `T_0`、`T_mult`、`lr_max`、`lr_min` など調整すべきハイパーパラメータが増え、最適値はタスク依存です。
- 再起動直後の高い学習率が、すでによく学習できたモデルを一時的に壊すリスクがあります。特に fine-tuning(学習済みモデルを少しだけ追加学習する場面)では、大きな再起動が事前学習の知識を破壊しやすく、SGDR は不向きとされます。
- 「なぜ周期的再起動が効くのか」の理論的説明は完全には確立していません。flat minima(平坦な極小)へ誘導される効果と説明されることが多いですが、近年の大規模学習では再起動なしの単一コサインで十分という報告も多く、再起動の必要性自体が問い直されています。これは未解決の研究課題です。
- コサインは「総ステップ数を事前に決め打ち」する必要があり、学習を途中で延長したいときに後付けが難しい。最近はこの弱点を突いて、決め打ち不要のスケジュール(後述)が研究されています。

### 発展トピック・研究の最前線
- 損失地形(loss landscape)の研究: なぜコサインや再起動が効くのかを、損失地形のフラットさ・シャープさの観点から説明しようとする研究が進んでいます(Hochreiter & Schmidhuber 以来の flat minima 仮説、Keskar らのバッチサイズと汎化の研究、Li et al. 2018 の loss landscape 可視化など)。Mode connectivity(複数の良い解が低損失の経路でつながっているという発見、Garipov et al. 2018)は、Snapshot/SWA がなぜ効くのかの理論的背景になっています。
- Stochastic Weight Averaging(SWA): 上述。SWA-Gaussian(SWAG)はベイズ的に不確実性も推定する拡張です。
- 1cycle policy と super-convergence: 上述。fast.ai 系のコミュニティで広く普及しました。
- warmup との統合: 近年は「線形 warmup → 単一コサイン減衰」が Transformer 系学習の事実上の標準です。LLM の事前学習でもコサインが基本ですが、学習途中で停止・再開・延長しにくいという欠点が顕在化し、Warmup-Stable-Decay(WSD)など「最後だけ急減衰し、それまでは一定」にして任意長で止められるスケジュールが提案されています(MiniCPM 2024 など)。
- スケジューラ自体の不要化: schedule-free optimizer(Defazio et al. 2024)のように、明示的なスケジュールを使わず最適化アルゴリズム側で同等の効果を出そうとする研究も登場しており、「コサイン一強」の前提が再検討されています。

### さらに学ぶための関連トピック
- [Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Linear LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## CutMix

### ひとことで言うと
CutMix は、ある画像の一部分(四角い領域)を切り取って別の画像をそこに貼り付け、1枚の合成画像を作り、ラベルも貼り付けた面積の割合で混ぜて学習するデータ拡張です。Mixup が全面を半透明で重ねるのに対し、CutMix は「切り貼り」なので画像が二重写しにならず、各ピクセルはどちらか一方の画像のものになります。これにより、画像の局所的な特徴をしっかり学ばせ、物体が部分的に隠れた状況にも強いモデルを作れます。

### 直感的な理解
猫を見分ける訓練をするとき、いつも猫の全身がきれいに写った写真ばかり見せていると、モデルは「画像全体の雰囲気」で判断する癖がつきます。すると、猫の半分が物陰に隠れた写真や、画面の隅にしか写っていない写真ではうまく判断できません。

CutMix は、猫の写真の一角に犬の四角いパッチを貼り付け、「この画像は70%が猫、30%が犬」と教えます。すると、モデルは「画像のこの部分は猫の特徴、この部分は犬の特徴」と局所ごとに見分ける力を鍛えられます。物体の一部だけからでも認識できるようになるため、現実によくある「物が部分的に隠れる」状況(オクルージョン)に強くなります。先行手法には「一部を黒く塗りつぶす Cutout」と「全面を半透明で重ねる Mixup」がありましたが、CutMix はその両方の良いところを取った設計です。

### 基礎: 前提となる概念
- データ拡張 (data augmentation): 訓練データを加工して水増しし、過学習を防ぐ技術です。
- ラベル (label): そのデータの正解で、分類なら `猫 = [1,0]` のようなベクトルです。
- オクルージョン (occlusion): 物体が他の物に隠れて一部しか見えない状態です。現実の画像では頻繁に起きるため、これに強いことが実用上重要です。
- 弱教師あり物体位置特定 (weakly-supervised object localization): 「画像にどのクラスが写っているか」だけの弱いラベルから、「そのクラスが画像のどこにあるか」を推定するタスクです。CAM(Class Activation Map、どこを見てクラスを判断したかの可視化)などで評価し、CutMix で鍛えたモデルはこの能力が高いと報告されています。
- ベータ分布 (Beta distribution): 0から1の混合比を生成する確率分布で、CutMix ではボックスの大きさを決めるのに使います。`α=1` なら一様分布になり、ボックスサイズがまんべんなくばらつきます。

先行手法を整理すると、2つの系統がありました。

1. Cutout(DeVries & Taylor 2017): 画像の一部を四角く黒く塗りつぶす拡張です。一部が隠れても正しく分類できるよう、画像全体の手がかりを使わせる狙いでした。欠点は、塗りつぶした領域が完全に無駄なピクセル(情報ゼロ)になり、訓練効率が落ちることです。
2. Mixup(Zhang et al. 2018、[Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)): 2枚を半透明で重ねる手法です。全画素を有効に使いますが、二重写しの不自然な画像になり、「物体のどこに何があるか」という局所構造が曖昧になります。

CutMix はこの2つの「いいとこ取り」を狙います。Cutout のように一部を別物に置き換えますが、黒塗りではなく別画像を貼ることで塗りつぶし領域も有効なピクセルとして使えます。Mixup のようにラベルを混ぜますが、全面重ねではなく切り貼りなので各ピクセルは「猫の画素」か「犬の画素」のどちらか一方であり、二重写しになりません。

### 仕組みを詳しく
2つのサンプル `(画像_A, ラベル_A)` と `(画像_B, ラベル_B)` を用意します。

1. 混合比 `λ` をベータ分布 `Beta(α, α)` から引く(`α` は典型的に 1.0、これは `λ` が一様にばらつく設定)。
2. 画像_A 上に、面積割合が `1 - λ` になる四角い領域(ボックス)をランダムな位置に決める。
3. その領域に画像_B の同じ位置を切り貼りする。
4. ラベルは「貼り付けた面積の割合」で混ぜる。

```
合成画像   = 画像_A の大部分 + 画像_B の切り貼り領域
合成ラベル = λ * ラベル_A + (1 - λ) * ラベル_B
```

ここで `λ` は「画像_A が占める面積の割合」になるよう、ボックスの大きさを決めます。具体的には、引いた `λ` から面積が `(1-λ)` になるよう、ボックスの幅・高さをそれぞれ `√(1-λ)` 倍で決め、中心位置を一様ランダムに選びます。最後に実際に貼られた面積から `λ` を計算し直してラベルに使います(ボックスが画像端からはみ出して切れた場合の面積ずれを補正するため)。

数値例を見ます。画像サイズ `224 x 224`(= 50176 ピクセル)、`λ = 0.7` を引いたとします。

- 画像_B を貼る領域の面積は全体の `1 - 0.7 = 0.3`、つまり約 15053 ピクセル。
- 正方形なら一辺は `224 * √0.3 ≈ 224 * 0.548 ≈ 123` ピクセル。
- この `123 x 123` のボックスをランダム位置に置き、そこだけ画像_B(たとえば犬)に差し替える。残りは画像_A(猫)のまま。
- ラベルは `0.7 * [1,0]猫 + 0.3 * [0,1]犬 = [0.7, 0.3]`。

できあがる画像は「猫の写真の一角に犬の四角いパッチが貼られた」もので、Mixup のような半透明の二重写しにはなりません。各ピクセルは猫か犬のどちらか一方です。

テンソル形状の観点では、バッチ `[B, 3, H, W]` をシャッフルしたコピー `x[index]` を用意し、ボックス座標 `(x1, y1, x2, y2)` の領域だけを差し替えます。

```
x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
λ = 1 - (x2-x1)*(y2-y1) / (H*W)        # 実面積から再計算
```

Mixup と同様、学習時のみ適用し推論時は使いません。実務では Mixup と CutMix を確率的に切り替えて両方使うレシピが一般的で、Vision Transformer の標準的な学習設定(DeiT など)に組み込まれています。

なぜ局所特徴の学習につながるのかを補足します。全面を重ねる Mixup では、画像のどの場所も両クラスの情報が混ざるため、モデルは「画像のここに猫の特徴がある」という空間的な手がかりを学びにくくなります。一方 CutMix では、画像の特定の領域だけが別クラスに置き換わるので、モデルは「この領域は猫、この領域は犬」と空間的に対応づけて学ぶ必要が生じます。この空間対応の学習が、物体位置特定能力やオクルージョン耐性につながります。さらに、ラベルを面積比に合わせることは「モデルの注目領域(CAM)が両クラスの物体にバランスよく分散すること」を暗黙に促し、1つの物体だけに過度に注目する偏りを抑える効果があります。

### 手法の系譜と主要論文
- DeVries & Taylor, "Improved Regularization of CNNs with Cutout", arXiv:1708.04552 (2017): 画像の一部を黒くマスクする Cutout を提案しました。部分的に隠れても認識できるよう全体の手がかりを使わせる狙いで、CIFAR で汎化を改善しました。欠点はマスク領域が情報ゼロで非効率な点です。
- Zhang et al., "mixup", ICLR 2018([Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)): 全面の線形補間でデータを混ぜる手法です。CutMix の直接の比較対象で、「補間ではなく置換」という対比で理解すると分かりやすくなります。
- Yun, Han, Oh, Chun, Choe, Yoo, "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features", ICCV 2019: 本手法の原典です。画像の領域を別画像で置換し、ラベルを面積比で混ぜる CutMix を提案しました。Cutout の「無駄ピクセル問題」と Mixup の「二重写しで局所構造が曖昧になる問題」を同時に解消し、分類精度と局所特徴の学習を両立しました。
- Kim, Choo, Song, "Puzzle Mix: Exploiting Saliency and Local Statistics for Optimal Mixup", ICML 2020: 物体の重要領域(saliency)と局所統計を考慮して、貼り付け方を最適化する派生です。
- Uddin, Monira, Shin, Chung, Bae, "SaliencyMix", ICLR 2021: 貼り付け位置を物体の重要領域に合わせ、「無関係な背景を貼ってラベルがずれる」問題を緩和しました。
- Touvron et al., "DeiT", ICML 2021: Vision Transformer の標準学習レシピに Mixup と CutMix を併用で組み込み、これらが大規模 Transformer の汎化に不可欠であることを示しました。

### 論文の実験結果(定量データ)
ICCV 2019 の主要結果を、指標の意味とともに紹介します。基本指標は ImageNet の Top-1 誤り(最も確信したクラスが間違っている割合)で、小さいほど良い値です。

- ImageNet 分類(ResNet-50): ベースラインの Top-1 誤りがおよそ 23.7% だったのに対し、CutMix を適用するとおよそ 21.4% まで低下したと報告されています(およそ2.3ポイントの改善)。これは同条件の Mixup(約 22.6%)や Cutout(約 22.9%)を上回る数値で、3手法の中で CutMix が最も誤りを下げました。
- 転移学習(検出・キャプション): CutMix で事前学習した ResNet-50 を物体検出(Pascal VOC)や画像キャプション(MS-COCO)に転用すると、ベースラインより良い結果が得られたと報告されています。局所特徴を鍛えた効果が他タスクにも波及することを示しています。
- 弱教師あり物体位置特定: CutMix モデルは、クラスラベルだけから物体の位置を推定する精度(CAM ベース)が Cutout や Mixup より高いと報告されています。これは「空間的にどこに何があるか」を学んだ証拠です。
- 頑健性: 入力に破損(ノイズ、ぼかし)を加えた設定や FGSM 敵対的攻撃に対して、CutMix モデルは精度の低下がベースラインより小さいと報告されています。CIFAR-100 でも PyramidNet などで一貫した改善が示されました。

アブレーションでは、ラベルを面積比で混ぜることの重要性が示されています。ボックスを貼っても元のラベルのままにすると精度向上が大きく落ちることから、「貼った面積に応じてラベルも混ぜる」設計が核心だと確認されています。また、貼り付ける別画像を黒塗り(Cutout相当)に置き換えると性能が下がることから、「無駄ピクセルを別画像で埋める」ことが効いていると分かります。中央固定のボックスよりランダム位置の方が良いこと、`α=1`(一様な面積分布)が安定して良いことも示されています。

### メリット・トレードオフ・限界
メリット:
- 全ピクセルが有効で、Cutout の無駄ピクセル問題がありません。
- 二重写しにならず、局所的な物体特徴をしっかり学べます(Mixup の弱点を補う)。
- オクルージョンや入力破損、敵対的攻撃への頑健性が向上します。
- 物体の位置特定能力が副次的に付き、転移学習にも有利です。
- 実装が軽量で計算コストがほぼ増えません。

トレードオフ・限界:
- 分類向けの設計で、回帰・検出・セグメンテーションにはそのまま使いにくいです。
- 貼り付け領域に肝心の物体が含まれないと、ラベルと中身がずれて学習ノイズになります(たとえば犬の背景の空だけを貼ると「30%犬」というラベルが不正確になる)。この弱点を緩和するのが saliency 系の後続手法(PuzzleMix, SaliencyMix)です。
- `α` とボックスのランダム性の調整が必要です。
- 切り貼りの境界が物理的にありえない不自然な画像になるため、現実画像の整合性が重要なタスク(運転シーンの生画像など)には不向きな場合があります。突然パッチ内に出現した車に対する正しい操舵は定義できません。

研究上の未解決課題として、貼り付け領域とラベルの不一致(ラベルノイズ)をどう原理的に抑えるか、そして分類以外のタスク(密な予測や系列予測)へどう正しく一般化するかが、引き続き研究対象になっています。saliency を毎ステップ計算すると計算コストが増えるため、「貼り付け位置の質と計算コストのトレードオフ」も実用上の論点です。

### 発展トピック・研究の最前線
CutMix は「領域置換による混合」という研究系統を確立しました。物体の重要領域を考慮する SaliencyMix・PuzzleMix・SnapMix、複数画像を組み合わせる Mosaic 拡張(物体検出で広く使われる)、特徴空間で領域を混ぜる手法など、多数の派生が生まれています。Transformer 向けには、パッチ単位で混ぜる TokenMix や、Attention マップを使って意味のある領域を貼る手法も登場しています。自己教師あり学習や半教師あり学習でも、データの多様性を増やす要素として取り込まれています。理論面では、CutMix のような領域置換がモデルに与える帰納バイアスや、ラベルノイズと汎化の関係を解析する研究が進んでおり、「どの混合則が最も効率よく汎化を高めるか」という問いの探索が続いています。

### さらに学ぶための関連トピック
- [Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Dropout](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Stochastic Depth (DropPath)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Weight Decay (L2 正則化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Cutout / Random Erasing

### ひとことで言うと
学習中に、入力画像のどこか一部を「四角く塗りつぶして消す」だけの、とても単純なデータ拡張(学習データを人工的に水増ししてモデルを賢くする工夫)です。わざと一部を隠すことで、モデルが「画像のごく一部だけを見て答えを決める」近道学習を防ぎ、物体が部分的に隠れている(遮蔽されている)場面にも崩れにくくします。実装は数行で済み、計算コストはほぼゼロ、それでいて画像分類・物体検出・人物照合(同じ人物を別カメラ間で見つける課題)など幅広いタスクで安定して精度を上げることが知られています。「情報をわざと捨てることで、かえって強くなる」という、正則化(過学習を抑える工夫)の代表例です。

### 直感的な理解
画像認識モデルは、放っておくと「楽をして当てる」習性を持ちます。たとえば「馬」の画像にいつも左下にカメラマンのクレジット文字が入っていると、モデルは馬そのものではなくその文字を見て「馬」と判定するようになる、という有名な失敗例があります。これは「ショートカット学習(shortcut learning: 本質的でない手がかりに頼ってしまう現象)」と呼ばれます。学習データの中でたまたま正解と相関している「ニセの手がかり」にモデルが飛びついてしまうのです。

人間は、犬の顔が物陰に半分隠れていても「犬だ」と分かります。鼻先だけ、しっぽだけ、耳だけ見えていても全体を補完して認識できる。これは私たちが物体の多くの部位を冗長に(重複して、保険のように)手がかりとして持っているからです。一方、ショートカット学習をしたモデルは、頼りにしていた1か所が隠れただけで途端に分からなくなります。冗長性が無いので脆いのです。

Cutout / Random Erasing の発想は単純で、「だったら学習中にわざと、いろんな場所をランダムに隠してしまえ」というものです。隠れた残りの情報からでも正解を出さなければ損失(モデルの予測と正解とのズレを表す数値、これを小さくするのが学習)が下がらないので、モデルは自然と「1か所に頼らず、画像全体の複数の手がかりを使う」ようになります。結果として、見たことのないデータへの汎化(generalization: 未知データでも正しく予測できる能力)と、現実の遮蔽への頑健性(robustness: 入力が乱れても性能が落ちにくい性質)の両方が高まります。重要なのは、これがテスト時には一切適用されないことです。隠す操作はあくまで学習中のトレーニングだけで、推論時はフル解像度のきれいな画像をそのまま使います。

### 基礎: 前提となる概念
理解に必要な用語を順に噛み砕きます。

- 画像のテンソル表現: コンピュータは画像を数値の並び(テンソル: 数値が多次元に並んだ配列)として扱います。カラー画像は普通 `[C, H, W]` の形で、C はチャンネル数(RGB なら 3)、H は高さ(縦のピクセル数)、W は幅(横のピクセル数)です。「ピクセル(画素)」は画像を構成する1つ1つの点で、各点が色の強さを表す数値(たとえば 0〜255)を持ちます。

- 過学習(オーバーフィッティング): モデルが学習データを丸暗記してしまい、学習データでは高精度なのに未知データでは精度が落ちる状態。パラメータ数(モデルが持つ調整可能な数値の個数)に対してデータが少ないほど、また学習を回しすぎるほど起きやすい。

- 正則化(regularization): 過学習を抑えるための仕掛けの総称。モデルの自由度をわざと制限したり、学習を少し難しくしたりして、丸暗記ではなく一般的なパターンを学ばせます。データ拡張はその一種で、「データ側を増やす」タイプの正則化です。

- ドロップアウト(Dropout, Srivastava et al., JMLR 2014): 学習中にニューロン(ネットワークの計算単位)をランダムに一定割合だけ無効化(出力を 0 に)する正則化。「特定のニューロンに依存しすぎない」ようにします。Cutout はこの考えを「中間層」ではなく「入力画像の空間領域」に持ち込んだものと見ることができます。実際、論文でも Cutout は「入力に対する空間的な Dropout」として位置づけられます。

- 遮蔽(occlusion): ある物体の一部が、別の物体に物理的に隠されること。現実世界では普遍的に起こります(歩行者が車の陰に半分隠れる、など)。

なぜ「1ピクセルずつランダムに消す」のではなく「まとまった矩形を消す」のか。画像は隣り合うピクセルが強く相関しています(隣の点はだいたい似た色)。1ピクセルだけ消しても、周囲から容易に推測できてしまい、消す効果が薄い。まとまった領域をごっそり消すことで初めて「その領域の情報が本当に失われた」状態を作れ、現実の遮蔽の状況にも近づきます。これが Cutout が空間的にまとまったマスク(覆い)を使う本質的な理由です。実は同じ理由で、中間層の特徴マップ(畳み込み層が出す「どこに何があるか」を表す数値の格子)に普通の Dropout を素朴にかけても効きが弱く、連続した領域をまとめて落とす DropBlock(Ghiasi et al., NeurIPS 2018)が提案された経緯があります。空間的に相関した信号には、点ではなく塊で介入する必要があるのです。

### 仕組みを詳しく
基本は驚くほど単純で、学習のたびに各画像へ次を行います。

1. 画像内からランダムに位置を1つ選ぶ。
2. その位置を中心(または左上)とする矩形を決める。
3. その矩形内のピクセルを一定の値で塗りつぶす。

#### Cutout(DeVries & Taylor 2017)
- 形は正方形に固定。サイズもあらかじめ決めた1つの値(例: CIFAR-10 なら 16×16 ピクセル、CIFAR-100 なら 8×8 程度)で固定。
- 塗りつぶす値はゼロ(あるいは正規化後の 0、すなわちデータ平均)。
- 矩形の中心は画像全体から一様ランダムに選び、矩形が画像の外にはみ出してもよい(はみ出た分は無視)。これにより、画像中央が丸ごと消えるパターンだけでなく、端の一部だけが欠けるパターンも自然に生成されます。中心を画像内に取りつつはみ出しを許すこの設計は、「常に固定面積を消す」より多様なマスクを生む工夫であり、結果的に「画像のどの隅・どの中心領域が消えても答えられる」訓練になります。

テンソルで言うと、入力 `[3, 32, 32]`(RGB、32×32)で座標 (10, 12) を中心に 16×16 を 0 にすると、出力の形は `[3, 32, 32]` のままで中身の一部だけが 0 になります。形が変わらないので、後段のネットワークは何も変更せずそのまま使えます。

before(元画像) → after(中央付近を消す):
```
□□□□□□□□        □□□□□□□□
□□□□□□□□        □□■■■□□□
□□□□□□□□   →    □□■■■□□□
□□□□□□□□        □□□□□□□□
```
(■ が塗りつぶされた領域)

#### Random Erasing(Zhong et al., AAAI 2020)
Cutout をより一般化したもので、ほぼ同時期に独立に提案されました(両者とも 2017 年に arXiv 公開)。違いは主に4点。

- 適用確率 p(論文では 0.5 程度)で「消すかどうか」を確率的に決める。消さない画像も混ぜることで、加工していない原画像の分布も学習に残す。
- 消す領域の面積を画像全体の面積の一定割合(論文の標準設定で下限 s_l = 0.02、上限 s_h = 0.4、すなわち 2%〜40%)からランダムに選ぶ。
- 領域の縦横比(アスペクト比)もランダム(r ∈ [0.3, 3.3] 程度)。だから正方形だけでなく縦長・横長の長方形にもなり、現実の細長い遮蔽(電柱、看板)に近づく。
- 塗りつぶす値は「ランダムなノイズ(各ピクセルに乱数)」「ゼロ」「データ平均(ImageNet 平均など)」から選べ、論文ではランダムノイズが最も良かったと報告。

具体的なサンプリング手順は次の通りです。面積 S_e = Area × Uniform(s_l, s_h)(Area は画像の総ピクセル数)、アスペクト比 r_e = Uniform(r_1, r_2) を引き、矩形の高さ h_e = sqrt(S_e × r_e)、幅 w_e = sqrt(S_e / r_e) とし、左上座標 (x, y) を画像内にランダム配置します。矩形が画像内に収まらなければ引き直します。ここで sqrt は平方根で、面積 S_e とアスペクト比 r_e から高さ・幅を逆算する式です(h_e × w_e = S_e、h_e / w_e = r_e を解いた形)。

Cutout は Random Erasing の特殊ケース(確率1、正方形固定サイズ、ゼロ埋め)とみなせます。Random Erasing のほうが「消える場所・大きさ・形・色」のバリエーションが豊かで、より多様な遮蔽を模擬できます。物体検出向けには object-aware の変種(画像全体だけでなく各バウンディングボックス内にも独立に消去をかける)も提案されており、「物体ごとに部分遮蔽を起こす」ことで検出器の遮蔽耐性を直接鍛えます。

#### 関連バリエーション
- Hide-and-Seek(Singh & Lee, ICCV 2017): 画像をグリッドに分割し、各セルを確率的に隠す。弱教師あり物体局在(weakly-supervised localization: 画像レベルのラベルだけから物体の位置を当てる課題)で、モデルが物体の最も目立つ部分だけでなく全体に注意を広げる効果を示しました。Cutout より前の系譜の一つで、「目立つ部分を隠して、次に目立つ部分も使わせる」というアイデアの原型です。
- GridMask(Chen et al. 2020): 規則的な格子状のマスクで消す。「ランダムに大穴を1つ」だと物体を丸ごと消すか全く消さないかの両極端になりがちな問題を、格子で「ほどよく分散して消す」ことで緩和します。格子の周期と比率を調整して、消去領域が物体全体をくまなく虫食い状に覆うようにします。
- DropBlock(Ghiasi et al., NeurIPS 2018): 入力画像ではなく中間層の特徴マップに対して、連続した正方形ブロックを落とす。Cutout の「塊で消す」思想を深層内部に持ち込んだもので、ResNet-50 の ImageNet 精度を 76.5% から 78.1% 程度に引き上げたと報告されています。

#### なぜ効くのか(より深く)
情報理論的には、入力の一部を欠落させることでモデルは「冗長な特徴表現」を学ばざるを得なくなります。1つの決定的手がかりに賭けると、それが消えたときに損失が跳ね上がるので、勾配(損失を減らすためにパラメータをどちらへ動かすべきかを示すベクトル)は「複数の手がかりに分散して依存する」方向にネットワークを押します。これは Dropout が特徴の共適応(co-adaptation: ニューロン同士が特定の組み合わせに過度に依存すること)を壊すのと同じ原理を、入力の空間次元で行っているものです。別の見方として、これは「学習データに人工的な多様性を注入し、実効的なデータ量を増やす」操作でもあり、データが少ないほど効果が大きくなる傾向があります。

### 手法の系譜と主要論文
時系列でたどると、データ拡張による「情報削除(information dropping)」系の流れが見えます。

1. Dropout(Srivastava et al., JMLR 2014): 中間層のニューロンをランダムに落とす。空間構造は考えない、すべての出発点。
2. Hide-and-Seek(Singh & Lee, ICCV 2017): 入力画像のグリッドセルを確率的に隠し、物体局在を改善。空間的削除の先駆け。
3. Cutout(DeVries & Taylor, arXiv 2017): 入力に正方形マスク1つ。最も単純で広く使われる定番。
4. Random Erasing(Zhong et al., AAAI 2020): 面積・形・確率・塗り値を全てランダム化。分類だけでなく検出・人物再同定にも有効と実証。
5. DropBlock(Ghiasi et al., NeurIPS 2018): 削除を中間層へ移し、塊で落とすことの重要性を実証。
6. GridMask(Chen et al. 2020): 格子状の構造化マスクで「消しすぎ/消さなすぎ」を緩和。
7. 別系統として、2枚の画像を混ぜる Mixup(Zhang et al., ICLR 2018)や、片方の矩形を別画像で置き換える CutMix(Yun et al., ICCV 2019)があります。CutMix は「Cutout で消す代わりに別画像を貼り、ラベルも面積比で混ぜる」点で Cutout の発展形とも言え、消した領域を無駄なく学習信号として使う改良です([Mixup / CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning) 参照)。
8. さらに先には Masked Image Modeling、特に MAE(He et al., CVPR 2022)があり、画像の 75% という大部分のパッチを隠して復元させる事前学習へと「隠す」発想が昇華しています。

### 論文の実験結果(定量データ)
評価指標の意味から押さえます。分類の主指標は「誤分類率(error rate)」または「Top-1 精度」です。誤分類率はテスト画像のうち間違えた割合(低いほど良い)、Top-1 精度は最も確信した予測が正解だった割合(高いほど良い)です。CIFAR-10 は 10 クラス・32×32 ピクセルの小画像 6 万枚、CIFAR-100 は 100 クラスの同様のデータセットで、いずれも画像が小さく枚数も限られるため過学習しやすく、拡張の効果が出やすいベンチマークです。1ポイント(1%)の差は、小画像分類では誤分類数の 2〜3 割を動かすこともある大きな差です。

Cutout(DeVries & Taylor 2017)の報告:
- CIFAR-10 で WideResNet-28-10 を用い、Cutout なしの誤分類率およそ 3.9% が、Cutout 適用でおよそ 3.1% 前後まで低下(報告による。約 0.8 ポイント改善は小画像分類では大きい)。
- CIFAR-100 でも誤分類率がおよそ 18.8% → 18.4% 前後へ低下し、当時の最先端を更新する組み合わせを報告。
- SVHN(街の番地数字、約 60 万枚)でも誤分類率の低下を確認(おおむね 1.3% 台へ)。
- マスクサイズが鍵で、データセットごとに最適サイズが異なる(CIFAR-10 は 16、CIFAR-100 は 8 など)。大きすぎると物体を消しすぎて逆効果という、小さすぎても効かず大きすぎても害になる「U 字」の傾向が観察され、サイズが唯一かつ重要なハイパーパラメータであることが示されました。

Random Erasing(Zhong et al. 2020)の報告:
- CIFAR-10 で ResNet 系に適用し、ベースラインから誤分類率を一貫して低減(報告で約 0.5〜1 ポイント規模)。WideResNet との併用でさらに改善。
- 物体検出(PASCAL VOC、指標は mAP = mean Average Precision: 各クラスの検出精度を平均した値、高いほど良い)で Fast R-CNN 系に適用し mAP が数ポイント改善。検出は「位置を当てる」タスクなので、遮蔽耐性が直接効きます。
- 人物再同定(person re-ID、同じ人を別カメラ間で照合。指標は Rank-1 精度=最も似ているとされた候補が本人だった割合、と mAP)で大幅改善を報告。たとえば Market-1501 データセットで Rank-1 と mAP がともに数ポイント向上。人物は他人や物に隠れやすいため、遮蔽拡張の効果が顕著に出る代表例です。
- アブレーション(ある要素を抜いて影響を見る実験): 塗りつぶし値はランダムノイズが最良、次いで平均、ゼロの順。適用確率 p は 0.5 付近が安定。面積上限 s_h を大きくしすぎる(例 0.6 超)と劣化し、消しすぎの害が出ました。

GridMask(Chen et al. 2020)の報告:
- ImageNet 分類で ResNet-50 の Top-1 精度をベースライン約 76.5% から約 77.9% 程度へ(報告による)、Cutout を上回る改善。
- COCO 物体検出でも Cutout を上回る mAP 改善を報告し、「構造化された(規則的に分散させた)削除」が単一の大穴より有効なことを示しました。

総じて、これら情報削除系は「数行の追加で、誤分類率を 0.5〜1.5 ポイント、検出 mAP を 1〜数ポイント改善する」コスト対効果の極めて高い拡張です。

### メリット・トレードオフ・限界
メリット
- 実装が数行、計算コストはほぼゼロ(ピクセルを書き換えるだけで、追加の forward 計算もない)。前処理段階に挟むだけで他の拡張と自由に併用できる。
- 遮蔽への頑健性が直接向上し、ショートカット学習を抑えて汎化を改善する。
- 分類・検出・セグメンテーション・人物照合と幅広いタスクで効く汎用性。

トレードオフと限界
- 情報を「捨てる」操作なので、学習データが極端に少ない/対象物体が小さい場合は、消した結果が無情報な画像になり逆効果になりうる。
- マスクサイズ・面積範囲・適用確率といったハイパーパラメータの調整が必要(特に Random Erasing)。最適値はデータセット依存で、Cutout の論文自身がサイズの U 字依存を示しています。
- 物体検出のように位置が重要なタスクでは、重要物体を丸ごと消すと、その画像が誤った教師信号になりうる(消したのにラベルは「そこに物がある」のまま、という矛盾)。GridMask や object-aware Random Erasing はこの問題への部分的な回答です。
- 「四角く均一に消す」だけなので、現実の複雑な遮蔽(半透明、影、不規則な輪郭、モーションブラー)を完全には再現できない。
- 回転・反転と異なり画素値そのものを破壊するため、強くかけすぎると低レベル統計(色分布など)が学習データとテストデータで乖離し、分布シフトを招く危険がある。
- 方向や空間構造が意味を持つドメイン(運転シーンで上が空・下が路面、左右に車線の意味がある等)では、消す位置の偏りが意図せぬバイアスを生む可能性があり、検証が要る。

未解決の課題としては、「どこを消すと最も学習に効くか」を入力依存で適応的に決める方向(注意マップに基づく削除など)や、削除拡張と混合拡張(CutMix 等)の最適な組み合わせ方、大規模事前学習が普及した時代に削除拡張の限界利益がどう変わるか、が今も研究対象です。

### 発展トピック・研究の最前線
- 適応的・敵対的マスキング: 一様ランダムではなく、モデルが最も依存している領域を狙って消すことで、より強い正則化を狙う研究(Adversarial Erasing 系)。弱教師付き局在の文脈で、最も識別的な領域を順に消して別の領域も学習させる手法が発展しました。
- 混合系との統合: CutMix は Cutout の「消す」を「別画像で埋める」に変え、捨てていた領域を学習信号として活用します。Mixup/CutMix と Random Erasing の併用は、近代的な画像分類の学習レシピで標準的に組み合わされます。
- 自己教師あり学習との接続: Masked Image Modeling(MAE 等、画像のパッチを大量に隠して復元させる事前学習)は、Cutout の「隠す」発想を表現学習の中心原理にまで押し上げたものと見ることができ、系譜上のつながりがあります。MAE は画像の 75% を隠してもなお意味のある特徴が学べることを示し、「隠す」ことの可能性を大きく拡張しました。
- 3D/時系列への拡張: ビデオや点群、BEV(鳥瞰図、上空から見下ろした視点で周囲を表す表現)での時空間的な削除拡張は、遮蔽が支配的な運転・ロボティクスで関心が高い領域です。時間方向にフレームを落とす、特定の方位のセンサ情報を隠すなど、ドメイン固有の削除設計が研究されています。

### さらに学ぶための関連トピック
- [Mixup / CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [RandAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AutoAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [TrivialAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Dropout](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## DeepSpeed

### ひとことで言うと
DeepSpeed（ディープスピード）は、Microsoft が公開した「大きなニューラルネットワークを、限られた台数の GPU でも学習できるようにする道具箱（ライブラリ）」です。中心となるアイデアは ZeRO（ゼロ）と呼ばれるメモリ節約の仕組みで、複数の GPU が「同じものを重複して抱えている無駄」を取り除き、互いに分担して持ち合うことで 1 枚あたりのメモリ使用量を劇的に減らします。これにより、本来なら数十枚必要だった巨大モデルを、ずっと少ない枚数で学習できるようになります。さらに、足りない分を CPU メモリや SSD に逃がす機能（offload）や、計算そのものを速くする最適化カーネルも束ねています。一言でいえば「巨大モデルを現実的な台数で回すための、メモリと通信と計算の総合最適化スイート」です。

### 直感的な理解
4 人で同じ分厚い辞書を 1 冊ずつ全員が抱えて持ち歩く状況を想像してください。重い辞書を 4 冊ぶん全員が持っているのは明らかに無駄です。もし「あなたは A〜F の章、私は G〜M の章」と分担して 1 人 1 冊の 1/4 だけ持ち、必要なページが来たら持っている人に聞けば、1 人あたりの荷物は 4 分の 1 で済みます。

GPU を使ったモデル学習でも、まったく同じ無駄が起きています。素朴な「データ並列」という方式では、GPU 全員がモデルの重み・勾配・オプティマイザの内部状態という巨大な荷物を丸ごとコピーして抱えます。GPU を 8 枚に増やしても、8 枚それぞれが同じ巨大な荷物を抱えたままなので、1 枚に乗らない巨大モデルは何枚 GPU を足しても乗りません。ここが分散学習で多くの人がつまずく直感の落とし穴です。「GPU を増やせばメモリも増えるはず」と思いがちですが、素朴なデータ並列ではメモリは増えず、計算スループット（さばける枚数）だけが増えます。ZeRO は「全員が同じものを持つのをやめ、分担して持ち合おう」という発想で、この壁を壊し、GPU を増やすと本当に 1 枚あたりのメモリが減るようにします。

### 基礎: 前提となる概念
深く理解するために、まず「学習で何がメモリを食うのか」を分解します。

ニューラルネットワークの学習は次を繰り返します。(1) データを入力して予測を出す（順伝播 forward）、(2) 予測と正解のズレ（損失 loss）を計算する、(3) 各パラメータを少し変えると損失がどう変わるかという「勾配 gradient」を計算する（逆伝播 backward）、(4) 勾配を使ってパラメータを少し更新する（オプティマイザ step）。

このとき GPU メモリを占有するのは大きく次の 4 種類です。

- パラメータ（重み）本体: モデルが学習する数値そのもの。
- 勾配: 各パラメータに対する損失の傾き。パラメータと同じ個数だけある。
- オプティマイザ状態: Adam（アダム、最もよく使われる更新アルゴリズム）は各パラメータについて「過去の勾配の移動平均（1 次モーメント、momentum）」と「勾配の 2 乗の移動平均（2 次モーメント、variance）」の 2 つを保持する。
- 活性化（activation）: 順伝播の途中結果。逆伝播のときに使うので保持される。バッチサイズ・系列長・層数に比例して膨らむ。

混合精度学習（計算は 16bit の半精度 float16/bfloat16 で速く行い、更新の正確さのために 32bit のパラメータのコピーを別途持つ手法）で Adam を使う典型構成では、1 パラメータあたりのメモリは次のように積み上がります。これは ZeRO 論文（Rajbhandari ら 2020）が「モデル状態（model states）」と呼んで定式化した内訳そのものです。

```
パラメータ本体 (fp16)        : 2 バイト
勾配 (fp16)                  : 2 バイト
fp32 パラメータのコピー       : 4 バイト  ┐
Adam 1次モーメント (fp32)     : 4 バイト  ├ オプティマイザ状態 = 12 バイト
Adam 2次モーメント (fp32)     : 4 バイト  ┘
----------------------------------------
合計                         : 16 バイト/パラメータ（活性化は別）
```

パラメータ 75 億個（7.5B）のモデルなら、活性化を除いても 16 × 7.5e9 ≈ 120GB。45GB の GPU 1 枚には到底乗りません。注目すべきは、最大の犯人が「オプティマイザ状態（12 バイト）」であって、パラメータ本体（2 バイト）ではないことです。素朴な直感だと「重みが重い」と思いがちですが、実は Adam の内部状態とその fp32 コピーが全体の 4 分の 3 を占めます。ここが ZeRO の攻めどころになります。

ここで「データ並列（Data Parallelism, DP）」を確認します。同じモデルを全 GPU にコピーし、各 GPU が違うミニバッチを処理し、計算した勾配を全 GPU で平均（all-reduce、[NCCL (集合通信)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照）して、全員が同じ更新を適用する方式です。実装が簡単でスケールしやすい一方、上記の 16 バイト/パラメータを全 GPU が重複して持つため、メモリ効率が極端に悪いという欠点があります。対照的に「モデル並列（テンソル並列 [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) / パイプライン並列 [Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)）」はモデル自体を割って各 GPU に分散するためメモリは節約できますが、通信が重く実装が難しい。ZeRO の動機はまさに「データ並列の単純さ・計算効率を保ったまま、モデル並列のメモリ効率を手に入れる」ことでした。

### 仕組みを詳しく

#### ZeRO の 3 段階
ZeRO（Zero Redundancy Optimizer、冗長性ゼロのオプティマイザ）は、データ並列の「重複保持」を段階的に取り除きます。GPU が N 枚あるとします。各段階で「何を 1/N に分割（shard、シャード化）するか」が異なります。

- ZeRO Stage 1 (P_os): オプティマイザ状態（12 バイト/パラメータ）を N 分割。各 GPU は 1/N だけ持つ。最大の犯人を分割するので効果が大きい。
- ZeRO Stage 2 (P_os+g): 上に加えて勾配（2 バイト）も N 分割。
- ZeRO Stage 3 (P_os+g+p): さらにパラメータ本体（2 バイト）も N 分割。各 GPU はモデルの一部しか常時保持しない。これは PyTorch の FSDP（後述）と同じ発想。

メモリの目安を式で書きます。混合精度 Adam の 1 パラメータあたり 16 バイトのうち、分割される量が段階で変わります（記号 N は GPU 枚数）。

```
方式            1パラメータあたりメモリ
素朴なDP        16 バイト（全部重複）
Stage 1         4 + 12/N バイト（fp16重み2+勾配2 は重複、状態12を分割）
Stage 2         2 + (2+12)/N バイト（fp16重み2 だけ重複）
Stage 3         16/N バイト（全部分割）
```

具体例: 120GB のモデルを GPU 8 枚で動かす場合、素朴な DP では各 GPU 120GB（不可能）。Stage 1 では 4 + 12/8 = 5.5 バイト/param ≒ 41GB。Stage 2 では 2 + 14/8 = 3.75 バイト/param ≒ 28GB。Stage 3 では理論上 16/8 = 2 バイト/param ≒ 15GB まで下がり、45GB GPU に余裕で乗ります。N（枚数）を増やすほど 1 枚あたりが軽くなるのが ZeRO の本質です。N=64 の Stage 3 なら 120/64 ≒ 1.9GB まで縮みます。

ただしタダではありません。Stage 3 では、ある層を計算する瞬間にはその層の全パラメータが手元に必要です。分担して持っているので、計算直前に他の GPU から「この層のパラメータを集める（all-gather）」通信が走り、計算が終わると手放します（free）。逆伝播でも各層のパラメータを再び all-gather し、勾配を集約する reduce-scatter が走ります。つまりメモリを「追加の通信」と引き換えに節約しています。通信量の理論を押さえると、Stage 1/2 のデータ並列はステップあたり約 2×(モデルサイズ) の通信（all-reduce 相当）ですが、Stage 3 は順伝播の all-gather + 逆伝播の all-gather + reduce-scatter で約 3×(モデルサイズ) と通信量が 1.5 倍に増えます。ここが本質的なトレードオフであり、GPU 間ネットワークが細いと Stage 3 の速度が頭打ちになる根本原因です。

#### 活性化メモリと checkpointing の併用
ZeRO が削るのは上記「モデル状態」の部分で、活性化（activation）は別問題です。系列長の長い Transformer ではむしろ活性化がメモリ支配的になることもあります。DeepSpeed は activation checkpointing（順伝播の途中結果を一部だけ残し、逆伝播時に必要になったら再計算する手法。メモリと計算を交換する）を ZeRO と組み合わせられます。ZeRO 論文では活性化分割（partitioned activation checkpointing）も提案され、活性化もテンソル並列の軸で分割して重複を除けることを示しました。

#### ZeRO-Offload と ZeRO-Infinity
ZeRO だけでも N を増やせばメモリは減りますが、GPU が少ないとそもそも限界があります。そこで DeepSpeed は「GPU メモリに乗り切らない分を、より大きいが遅い記憶階層に逃がす（offload）」機能を持ちます。

- ZeRO-Offload（Ren ら 2021）: オプティマイザ状態と fp32 パラメータコピーを CPU の主記憶（RAM、GPU メモリより桁違いに大きいが遅い）に置き、Adam の更新計算自体も CPU で行う。GPU は順伝播・逆伝播に専念できる。賢い点は「計算グラフのどの部分を CPU/GPU どちらに割り当てるか」を最小通信量の観点で定式化し、forward/backward は GPU、parameter update は CPU、という分担が通信最小であることを理論的に導いたこと。さらに CPU 側に最適化した高速 Adam 実装（CPUAdam、SIMD/マルチスレッド利用）を用意し、CPU 更新が律速にならないよう工夫した。GPU 1 枚でも数十億パラメータ級を学習可能にした。
- ZeRO-Infinity（Rajbhandari ら 2021）: さらに NVMe（高速 SSD）まで使い、GPU メモリ → CPU RAM → NVMe という記憶階層を総動員する。理論上は GPU 1 枚でも兆パラメータ級を視野に入れる。鍵は「帯域中心設計（bandwidth-centric partitioning）」と積極的な prefetch（次に要るデータを先読みして転送を計算の裏に隠す）、転送の重複（overlap）で、NVMe の遅さを実効的に隠す。

記憶階層の速度差を直感的に押さえておきます。GPU メモリ（HBM）は毎秒 1〜3 TB、CPU RAM は数十〜百数十 GB/s、NVMe は数 GB/s 程度です。下に逃がすほど容量は増えるが転送が遅くなる、というのが offload のトレードオフの根です。NVMe は HBM より 2〜3 桁遅いため、prefetch で隠しきれないと GPU が待たされ TFLOPS が落ちます。

#### ZeRO++ — 通信を削る方向の進化
通信が Stage 3 のボトルネックになる問題に対し、ZeRO++（Wang ら 2023, arXiv:2306.10209）は 3 つの通信圧縮を提案しました。(1) qwZ: all-gather するパラメータを int8 に量子化して通信量を半減、(2) hpZ: パラメータをノード内に二次的に複製して逆伝播の all-gather をノード内（高速 NVLink）に閉じ込め、ノード間通信を消す、(3) qgZ: 勾配の reduce-scatter を量子化通信に置き換える。これにより低帯域環境や小バッチ時の通信量を最大 4 分の 1 に削減し、スループットを大きく改善したと報告しています。これは「ZeRO は通信が重い」という弱点への直接的な回答です。

#### 道具箱としての DeepSpeed
DeepSpeed は ZeRO だけのライブラリではありません。

- 3D 並列の統合: データ並列 ＋ パイプライン並列（[Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)）＋ テンソル並列（[Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)）を組み合わせて、巨大モデルを多次元に分割する。Megatron-DeepSpeed として大規模言語モデルの学習に使われた。各軸の使い分けは「テンソル並列は通信が重いのでノード内 NVLink に閉じる、パイプライン並列はノード間、データ並列＝ZeRO で外側」というのが定石。
- 最適化カーネル: 複数の演算を 1 つにまとめた fused Adam、効率的な Attention 実装、transformer kernel などで計算自体を速くする。
- 大規模推論（DeepSpeed-Inference / DeepSpeed-MII）: 学習済み巨大モデルを少ない GPU で動かす最適化（テンソル並列推論、KV キャッシュ管理）。
- 導入のしやすさ: 既存の PyTorch コードに対し `model_engine, optimizer, _, _ = deepspeed.initialize(model=model, ...)` でモデルとオプティマイザを包み、Stage や offload の設定を JSON（ds_config）で渡すだけで導入できる。コードの大改修が要らないのが普及した一因。

#### FSDP との関係
PyTorch 本体には FSDP（Fully Sharded Data Parallel）という、ZeRO Stage 3 と発想がほぼ同じ機能が標準で入っています（パラメータ・勾配・オプティマイザ状態を分割保持し、計算直前に all-gather）。そのため DeepSpeed は「FSDP の代替または補完」として語られます。選択基準は、CPU/NVMe offload や 3D 並列の作り込み、推論最適化まで欲しいなら DeepSpeed、PyTorch 標準で完結させたく追加依存を避けたいなら FSDP、というのが一般的な目安です。Hugging Face Accelerate などの上位ライブラリはどちらも切り替えて使えるようになっています。

### 手法の系譜と主要論文
- ZeRO（Rajbhandari ら, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020, arXiv:1910.02054, Microsoft）: データ並列の冗長保持を分割で取り除く中核アイデア。動機は「データ並列はスケールしやすいがメモリ効率が悪く、モデル並列は通信が重く実装が難しい。両者の良いとこ取りをしたい」。新規性は、メモリ削減を実現しつつデータ並列の単純さと計算効率を保った点。
- DeepSpeed システム論文（Rasley ら, "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters", KDD 2020）: ZeRO を含む学習システム全体を実用ライブラリとして提示。
- ZeRO-Offload（Ren ら, USENIX ATC 2021, arXiv:2101.06840）: 計算と状態を CPU に賢く配分（CPU には更新、GPU には順逆伝播）し、転送と計算を重ねて隠す。動機は「潤沢な GPU を持たない研究者に大規模学習を開放する（democratizing）」。
- ZeRO-Infinity（Rajbhandari ら, SC 2021, arXiv:2104.07857）: NVMe まで階層化し、帯域中心分割・prefetch・重複転送で速度低下を緩和。
- ZeRO++（Wang ら, 2023, arXiv:2306.10209）: 量子化通信とノード内複製で Stage 3 の通信量を削減。低帯域・小バッチ環境を救う。
- PyTorch FSDP（Zhao ら, VLDB 2023, arXiv:2304.11277, Meta）: Stage 3 と同等の分割発想を PyTorch コアに統合し、標準機能化した。
- Megatron-LM（Shoeybi ら, arXiv:1909.08053; Narayanan ら, SC 2021, arXiv:2104.04473）: テンソル並列・パイプライン並列の系譜で、DeepSpeed と組み合わせた Megatron-DeepSpeed が超大規模 LLM 学習（Megatron-Turing NLG 530B など）の標準構成になった。

系譜を一言でまとめると、「データ並列のメモリ無駄を削る（ZeRO 2020）→ 足りない分を CPU/NVMe に逃がす（Offload/Infinity 2021）→ 増えた通信を量子化で削る（ZeRO++ 2023）」という、メモリ・通信のトレードオフを順に潰してきた流れです。並行して PyTorch がその核を標準機能（FSDP）に取り込み、Megatron 系のモデル並列と組み合わさって超大規模 LLM 学習のデファクト構成が固まりました。

### 論文の実験結果（定量データ）
ここで指標の意味を補足します。大規模学習では「学習できる最大モデルサイズ」「GPU あたりのスループット（TFLOPS、1 秒あたり浮動小数点演算数。GPU の理論ピークに対しどれだけ引き出せたか）」「スケーリング効率（GPU を N 倍にしたとき速度が何倍になるか）」が主要指標です。

- ZeRO 論文（SC 2020）の報告: ZeRO により、当時のデータ並列が扱えた数十億パラメータ級から、100B（1000 億）超のモデルを学習可能にした。400 GPU（V100）で 100B パラメータのモデルを学習し、GPU あたり約 38 TFLOPS（V100 のピーク約 125 TFLOPS に対しおよそ 3 割）、ほぼ線形（super-linear に近い）なスケーリングを示したと報告。Stage を上げるほど扱えるモデルが大きくなることをメモリ内訳の理論計算とともに示した。アブレーション的な観点では、Stage 1 だけでもオプティマイザ状態が最大の犯人なので約 4 倍のメモリ削減が効く一方、Stage 3 まで進めるとパラメータも分割され通信が 1.5 倍に増えるという段階的トレードオフが定量化されている。
- ZeRO-Offload 論文（ATC 2021）の報告: 単一の V100 GPU（32GB）で 13B パラメータのモデルを学習可能にした。これは offload なしの同条件（約 1.4B が上限）に対し約 10 倍。複数 GPU でも 128 GPU で 70B 級まで拡張。「CPU に状態を置くと遅くなる」懸念に対し、計算配分と通信のオーバーラップ、および CPUAdam により単一 GPU で 40 TFLOPS 前後を維持したと報告。これは offload のナイーブ実装（GPU が CPU の更新完了を待つ）に対する大きな改善で、PyTorch 単体に比べ単一 GPU で最大約 10 倍のスループットを報告。
- ZeRO-Infinity 論文（SC 2021）の報告: 単一 DGX-2 ノード（16 GPU）で 1 兆（1T）パラメータ級、512 GPU で 32T パラメータ級の学習が原理的に可能と示した。NVMe の遅さは prefetch と帯域重ね合わせで隠し、512 GPU で 25 PetaFLOPS 級（ピークの約 4 割の実効効率）の総スループットを報告。GPU 数に対しほぼ超線形のスーパースケーリングも観測。
- ZeRO++ 論文（2023）の報告: 低帯域クラスタや小グローバルバッチの設定で、通信量を最大 4 分の 1 に削減し、ZeRO-3 比で最大約 2.4 倍のスループット向上を報告。通信律速の現実環境で効くことを示した。
- PyTorch FSDP 論文（VLDB 2023）の報告: 175B パラメータの GPT 規模モデルを多数 GPU でほぼ線形にスケールさせ、A100 クラスタで GPU あたり高い利用率（実装やモデルにより 100〜180 TFLOPS 級、A100 ピーク 312 TFLOPS に対し 3〜6 割）を達成。DeepSpeed ZeRO-3 と同等の機能・性能を PyTorch ネイティブで実現できることを示した。

数値の意味を初心者向けに補うと、TFLOPS が大きいほど「買った GPU の馬力を無駄なく引き出せている」ことを表します（ピークの 3〜5 割引き出せれば大規模学習としては良好な部類）。スケーリング効率が線形（8 枚で 8 倍）に近いほど、台数を増やした投資が素直に速度に変わるということです。Offload はメモリは稼げても TFLOPS が下がりやすく、これが「メモリと速度のトレードオフ」の正体です。

### メリット・トレードオフ・限界
メリット
- ZeRO によりデータ並列の冗長保持を取り除き、1 GPU あたりメモリを大幅削減。GPU を増やすほど 1 枚あたりが軽くなる（素朴な DP にはない性質）。
- ZeRO-Offload / Infinity により、GPU が少なく（極端には 1 枚でも）大規模モデルを学習可能。研究室レベルの計算資源を底上げする。
- 3D 並列・効率カーネル・推論最適化まで束ねた道具箱で、JSON 設定中心の低い導入コスト。
- 巨大モデル学習の事実上の標準ツールの 1 つで、知見・ドキュメントが豊富。

トレードオフ・限界
- Stage を上げるほど all-gather / reduce-scatter の通信が増え（Stage 3 は約 1.5 倍）、GPU 間ネットワークが遅いと速度が頭打ち。通信が同期点を作るためストラグラー（遅い 1 台）に弱い。
- Offload は転送の遅さで TFLOPS が下がる。メモリと速度のトレードオフが激しく、NVMe offload は特に遅い。
- 設定項目が多く、最適な Stage・offload 量・bucket サイズ・通信オーバーラップのチューニングに経験が要る。設定ミスで OOM や速度激減が起きやすい。
- 小規模で 1 枚に乗るモデルには不要で、導入が複雑さを増やすだけになりうる。
- PyTorch 標準 FSDP と機能が重複し、エコシステムの選択判断が必要。

研究上の未解決課題としては、通信と計算のオーバーラップをいかに自動で最適化するか、異種環境（GPU と CPU/NVMe の帯域差が大きい構成）での自動配分、そして MoE（Mixture of Experts）など疎なモデルでのシャーディング戦略、低精度（FP8）学習との統合などが活発に研究されています。

### 発展トピック・研究の最前線
- 3D 並列の自動探索（どの軸をどれだけ分割すれば最速かを自動決定する Alpa などの研究）。
- 活性化メモリ削減（activation checkpointing / recomputation、selective recompute と ZeRO の併用）。
- 量子化を絡めた学習（8bit Adam、FP8 学習）でオプティマイザ状態のバイト数自体を削る方向。ZeRO の 12 バイト/param を半分以下にできる。
- MoE 学習向けの専用シャーディング（DeepSpeed-MoE）。expert parallelism と ZeRO の組み合わせ。
- 推論側の最適化（KV キャッシュのシャーディング、テンソル並列推論、continuous batching）。
- 通信圧縮の進化（ZeRO++ の量子化通信を低帯域クラウド学習へ展開）。

### さらに学ぶための関連トピック
- [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [NCCL (集合通信)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Spot Instance 学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)
- [Model Registry (Stage 遷移)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)


## Dropout

### ひとことで言うと
Dropout は、ニューラルネットワークの学習中に、内部の計算ユニット(ニューロン)をランダムに一時的に「お休み」させる正則化手法です。毎回違うメンバーで計算させることで、特定のニューロンだけに頼りきった脆い構造になるのを防ぎます。結果として、訓練データを丸暗記する過学習が抑えられ、見たことのない新しいデータでも性能が落ちにくくなります。追加の重みパラメータが要らず実装が数行で済むため、深層学習が普及した初期から現在まで使われ続けている基本ツールです。

### 直感的な理解
あるチームでいつも同じ2人がペアで仕事をしていると、片方が休むともう片方も仕事ができなくなる、という事態が起きます。一人ひとりが「相方がいる前提」でしか動けないからです。これは効率がよく見えても、人が抜けたとたんに崩壊する脆いチームです。

ニューラルネットの中でも同じことが起きます。ニューロンAが過剰に反応すると必ずニューロンBがそれを打ち消す、というような相互依存(共適応)ができあがると、訓練データには都合よく合いますが、少しでも入力が変わると予測が崩れます。これが過学習の正体の一つです。

Dropout のアイデアは「メンバーをランダムに休ませて、誰も特定の仲間に頼れなくする」ことです。今日はAが休み、明日はBが休む、という状況では、各ニューロンは「自分一人でもそこそこ役立つ、汎用的な特徴」を学ばざるを得なくなります。これによって相互依存が壊れ、頑健で汎化する表現が育ちます。別の見方をすると、Dropout は「少しずつ違う無数のネットワークを同時に訓練し、推論時にそれらを平均する」アンサンブル(複数モデルの合議)を、1個のネットワークの中で安く実現する仕掛けでもあります。

### 基礎: 前提となる概念
このトピックを理解するための言葉を、知らない人向けに噛み砕きます。

- ニューロン (neuron) / ユニット (unit): ネットワーク内の1個の計算単位です。入力に重みを掛けて足し合わせ、活性化関数(非線形変換)を通して次の層へ値を渡します。隠れ層(入力と出力の間にある中間の層)には数百から数千個のニューロンが並びます。
- 過学習 (overfitting): モデルが訓練データには完璧に当てはまるのに、新しいデータでは性能が落ちる現象です。訓練データのノイズや偶然の偏りまで覚え込んでしまうのが原因です。逆に学習が足りず訓練データにも当てはまらない状態を未学習 (underfitting) と呼びます。
- 正則化 (regularization): 過学習を抑えるために、モデルが複雑になりすぎないよう制約や工夫を加える総称です。Dropout はその代表格です。重み減衰(weight decay)、データ拡張、早期終了(early stopping)なども正則化の一種です。
- 共適応 (co-adaptation): 複数のニューロンが「お互いの存在を前提に」役割を固定してしまう状態です。これは訓練データには合っても、新しいデータには脆くなります。Dropout が直接狙うのはこの共適応の解消です。
- 期待値 (expected value): ランダムな量を「平均するとどれくらいか」を表す値です。Dropout では学習時にランダムにニューロンを落とすため、出力の期待値を推論時とそろえる工夫が必要になります。
- アンサンブル (ensemble): 複数の異なるモデルの予測を平均・多数決して、単体より良い精度を得る手法です。一般に汎化を改善しますが、複数モデルを保持・実行するコストがかかります。Dropout はこのコストを払わずアンサンブルに近い効果を得ようとします。

### 仕組みを詳しく
各層の出力に対し、ドロップ率 `p`(たとえば 0.5)を決めます。学習時には、各ニューロンの出力を確率 `p` でゼロにします(=お休み)。生き残ったニューロンの出力は `1/(1-p)` 倍にスケールします(理由は後述)。

数値例を見ます。ある層の出力が4個のニューロンで `[2.0, 4.0, 1.0, 3.0]`、`p = 0.5`(半分休ませる)とします。

1. ランダムに2個を選んでゼロにする。仮に2番目と3番目が休みなら → `[2.0, 0.0, 0.0, 3.0]`
2. 生き残った値を `1/(1-0.5) = 2` 倍にする → `[4.0, 0.0, 0.0, 6.0]`

次のステップでは別のニューロンが休みます。`[0.0, 8.0, 0.0, 6.0]` のように毎回マスクが変わります。これにより、毎ステップ実質的に違うネットワークを訓練していることになります。

なぜ `1/(1-p)` 倍するのかは重要なポイントです。推論時には全ニューロンを使いたい(ランダムに休ませると予測が毎回ブレるため)。しかし、学習時に半分休ませていたなら、推論時は2倍のニューロンが活動するので、次の層に渡る信号の合計が約2倍になってしまいます。これを防ぐため、学習時に生き残った出力をあらかじめ `1/(1-p)` 倍に増やしておきます。こうすると推論時はスケール調整なしで全ニューロンをそのまま使え、学習時と推論時で出力の期待値が一致します。この方式を inverted dropout(逆ドロップアウト)と呼び、現代のフレームワークの標準実装です。原論文では逆に「推論時に重みを `(1-p)` 倍する」方式が提示されましたが、推論を素直に保てる inverted dropout が後に主流になりました。

まとめると、こうなります。

- 学習時 (training): ランダムにゼロ化 + 生存分を `1/(1-p)` 倍にスケールアップ。
- 推論時 (inference): 何もしない(全ニューロンを普通に使う)。

主要な深層学習フレームワークでは、Dropout 層を層の間に挟むだけで使え、学習モードと評価モードの切り替えで上記の挙動が自動的に切り替わります。評価モードへの切り替えを忘れると、推論時にもランダムにニューロンが落ち、予測が不安定になり性能が大きく劣化します。これは実装上の典型的な落とし穴です。

ドロップ率 `p` の選び方の目安として、全結合層(ニューロンが密につながった層)では 0.5 前後、入力層に近い側では 0.1〜0.2 程度、畳み込み層(CNN、画像を扱う層)では 0.1〜0.2 程度が経験的に使われます。畳み込み層で効きにくい理由は後述します。

数学的な見方を補足します。Dropout 率 `p` でニューロンをマスクするとき、各ニューロンの活性 `h` に対し、確率 `1-p` で 1、確率 `p` で 0 をとるベルヌーイ確率変数 `m` を掛けます。学習時の出力は `m · h / (1-p)` で、`m` の期待値が `1-p` なので、出力の期待値は `h` に一致します。これが「期待値を保つ」という性質の正体です。一方、期待値は保たれても分散は学習時に増えます。この分散の増加こそがノイズ注入による正則化として働き、同時にバッチ正規化との相性問題の根にもなります。

ネットワーク全体としては、`N` 個のニューロンがあれば理論上 `2^N` 通りのサブネットワークが存在します。Dropout は毎ステップそのうち1つをランダムに選んで訓練し、推論時には重みを共有した1個のネットでそれらを近似的に平均する、と解釈できます。原論文ではこれを「重み共有によるアンサンブルの近似的な幾何平均」として説明しました。線形回帰の特殊な場合には、Dropout が入力特徴の大きさに応じた L2 正則化(重み減衰)と等価になることも示されており、Dropout が暗黙の正則化として働く理論的根拠の一つになっています。

### 手法の系譜と主要論文
- Hinton, Srivastava, Krizhevsky, Sutskever, Salakhutdinov, "Improving neural networks by preventing co-adaptation of feature detectors", arXiv:1207.0580 (2012): Dropout の最初の提示です。特徴検出器の共適応が過学習の主因という仮説のもと、学習時にニューロンを確率的に落とすアイデアと初期実験を示し、当時の音声認識・画像認識タスクで誤りを削減しました。
- Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)", NeurIPS 2012: 最後の全結合層に Dropout を採用し、ImageNet で当時の最先端を大きく更新しました。Dropout を外すと過学習が顕著に悪化したと報告され、これが深層学習の標準ツールになる大きなきっかけになりました。
- Wan, Zeiler, Zhang, LeCun, Fergus, "Regularization of Neural Networks using DropConnect", ICML 2013: ニューロンの出力ではなく、個々の重み(結合)をランダムにゼロにする一般化版を提案しました。MNIST で当時の最先端を更新しました。
- Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", JMLR 2014: Dropout を体系化した決定版の論文です。共適応の防止と暗黙のアンサンブル効果という二つの解釈を整理し、MNIST、CIFAR-10/100、ImageNet、SVHN、音声認識(TIMIT)、文書分類(Reuters)など当時の主要ベンチマークで一貫した改善を実証しました。
- Tompson, Goroshin, Jain, LeCun, Bregler, "Efficient Object Localization Using Convolutional Networks", CVPR 2015: 畳み込み特徴マップで点ごとに落とすと隣接ピクセルの相関のため効きにくい問題を指摘し、チャンネル(特徴マップ)単位で丸ごと落とす SpatialDropout を提案しました。
- Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016: 推論時にも Dropout を有効にして複数回サンプリングする Monte Carlo Dropout を提案し、予測の不確実性を推定できることを理論的に示しました。Dropout をベイズ近似として再解釈した重要な研究です。
- Kingma, Salimans, Welling, "Variational Dropout and the Local Reparameterization Trick", NeurIPS 2015 / Gal, Hron, Kendall, "Concrete Dropout", NeurIPS 2017: ドロップ率自体を勾配で学習可能にする方向の研究です。手で `p` を調整する必要を減らそうとしました。
- Ghiasi, Lin, Le, "DropBlock: A regularization method for convolutional networks", NeurIPS 2018: CNN 向けに「点ではなく連続した領域(ブロック)」を落とすことで畳み込みでの効きを改善しました。ImageNet 分類と COCO 物体検出で改善を報告しています。

Transformer 系では、Attention の重み、残差経路、全結合部分に Dropout を入れるのが標準になりました。層ごと丸ごと落とす Stochastic Depth([Stochastic Depth (DropPath)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))は、ニューロンではなく層を落とす Dropout の親戚と位置づけられます。

### 論文の実験結果(定量データ)
JMLR 2014 の主要な定量結果を、指標の意味とともに紹介します。ここでの基本指標は分類誤り率(error rate)で、「全テストデータのうち何パーセントを間違えたか」を表します。数値が小さいほど良く、たとえば 1.60% から 1.35% への低下は、間違いがおよそ16%減ったことを意味します。

- MNIST(手書き数字認識、10クラス): 標準的な全結合ネットワークでドロップなしのテスト誤りがおよそ 1.60% だったのに対し、Dropout を入れると 1.35% 程度まで低下したと報告されています。さらに DropConnect や畳み込みと組み合わせると 1% を切る水準まで改善しました。
- CIFAR-10(自然画像、10クラス): Dropout なしで約 14.98% の誤りだったところ、Dropout 導入でおよそ 12.6% 程度へ改善したと報告されています。
- SVHN(街頭の数字、より難しい): Dropout により誤りが約 2.8% から 2.5% 前後へ下がったと報告されています。
- ImageNet(大規模画像、1000クラス): AlexNet は最後の全結合層に Dropout を使うことが性能の鍵の一つで、Dropout を外すと過学習が顕著に悪化したと報告されています。
- 音声認識(TIMIT、音素認識): Dropout により音素誤り率が数ポイント改善したと報告されています。

アブレーション(どの要素を抜くとどうなるか)の観点では、論文は次の点を示しています。第一に、ドロップ率 `p` を 0.5 付近にしたとき全結合層で最も効果が高く、`p` を上げすぎると未学習に、下げすぎると過学習抑制が弱くなる、というU字型の関係があります。隠れ層の保持確率 `1-p` を横軸にとると、おおむね 0.4〜0.8 の広い範囲で誤りが低く、その外で悪化する谷型のカーブが報告されています。第二に、Dropout は重みの最大ノルム制約(max-norm regularization、各ニューロンへの重みベクトルの大きさに上限を設ける制約)と併用するとさらに効果が高まると報告されています。第三に、Dropout を入れると収束に必要な学習回数が約2〜3倍に増える、という代償も明記されています。これは各ステップで一部のニューロンしか更新されないためです。第四に、ガウスノイズを掛ける乗算的ノイズ版(Gaussian Dropout)もベルヌーイ版と同等以上に効くことが示され、Dropout の本質が「特定のマスク形」ではなく「乗算ノイズの注入」であることを示唆しました。

### メリット・トレードオフ・限界
メリット:
- 実装が簡単で、追加の重みパラメータが不要です。
- 過学習を効果的に抑え、特に全結合層やデータが少ない設定で有効です。
- 暗黙のアンサンブル効果で予測が頑健になります。
- MC Dropout を使えば、追加コストほぼなしで予測の不確実性を推定できます。

トレードオフ・限界:
- 学習の収束が遅くなり、必要な学習回数が増える傾向があります(JMLR 論文で2〜3倍)。
- `p` が大きすぎると未学習になり、必要な情報まで落としてしまいます。
- 畳み込み層では情報が空間的に冗長(隣のピクセルが似た情報を持つ)なため、点単位の Dropout は効きにくく、SpatialDropout / DropBlock / DropPath の方が適します。
- バッチ正規化 (Batch Normalization) と併用すると、学習時と推論時で分散の推定がずれる相性問題が報告されており(Li, Chen, Hu, Yang, "Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift", CVPR 2019)、置く位置や順序に注意が要ります。一般には Dropout を BN の後ろに、かつ全結合部分に限定して置くのが安全とされます。
- 評価モードへの切り替えを忘れると性能が大きく劣化する実装上の落とし穴があります。

未解決・研究上の課題としては、最適なドロップ率を自動で決める方法(Concrete Dropout など学習可能化の試み)、バッチ正規化との両立、そして大規模 Transformer での最適な配置(どの経路にどの率で入れるか)が依然として経験則に頼っている点が挙げられます。近年の超大規模モデルでは、十分なデータ量があれば Dropout を弱める・外す方が良い場合もあり、「データ量と正則化強度のバランス」という観点でも研究が続いています。

### 発展トピック・研究の最前線
学習可能な Dropout として、Variational Dropout や Concrete Dropout は、ドロップ率自体を勾配で最適化できるようにしました。不確実性推定の文脈では MC Dropout がベイズ深層学習の入り口として広く使われ、安全性が重要な分野で「モデルがどれくらい自信を持っているか」を測る手段として研究が続いています。構造化 Dropout(SpatialDropout, DropBlock)は畳み込みの空間冗長性に対処する系統で、これを層単位に拡張したのが Stochastic Depth です。大規模言語モデルや Vision Transformer では、Dropout 単独よりも DropPath とデータ拡張([Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) / [CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))を組み合わせたレシピが主流になっており、「どの正則化をどう混ぜるか」というレシピ設計そのものが研究テーマになっています。シーケンスモデルでは、時刻ごとに同じマスクを使い続ける Variational RNN Dropout のような、アーキテクチャに合わせたマスク設計も提案されています。

### さらに学ぶための関連トピック
- [Stochastic Depth (DropPath)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Weight Decay (L2 正則化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [重みの指数移動平均 (EMA of Weights)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)


## Early Stopping

### ひとことで言うと
Early Stopping (早期終了) とは、学習中に検証データでの性能を見張り、それが悪化し始めたら学習を途中で打ち切る手法です。学習を続けすぎると、モデルが訓練データを丸暗記してしまい未知のデータに弱くなる「過学習 (overfitting)」が起きます。その手前で止めることで、過学習を防ぎつつ無駄な計算も節約します。追加パラメータがほぼ不要で、どんなモデルにも後付けできる、最も手軽で効果の高い正則化の 1 つです。

### 直感的な理解
試験勉強にたとえます。過去問を何周も解くうちに、問題の解き方を理解する段階を越えて「この問題の答えは選択肢の 3 番」と丸暗記してしまうと、本番で少し違う問題が出たときに対応できません。これがモデルにおける過学習です。ちょうど良いのは「理解はしたが暗記には至っていない」段階で勉強をやめることです。

ニューラルネットを長く学習させると典型的に次のことが起きます。訓練データに対する誤差 (training loss) は学習が進むほど下がり続けます。一方、学習に使っていない検証データに対する誤差 (validation loss) は、最初は一緒に下がりますが、ある時点を境に上昇に転じます。

```
誤差
 │＼
 │ ＼＿＿＿＿＿＿＿＿  training loss (下がり続ける)
 │      ＼
 │       ＼＿＿,-‐''  validation loss (途中から上昇 = 過学習開始)
 │          ↑
 │       この谷の底で止めたい
 └────────────────────→ エポック
```

検証誤差が最も低くなった点を過ぎてさらに学習を続けると、モデルは訓練データへの当てはめだけが上手くなり、本番では性能が下がります。だったら検証誤差が底を打った付近で止めればよい——これが Early Stopping の動機です。同時に「もう良くならないのに回し続ける」無駄な計算時間も省けます。

### 基礎: 前提となる概念
まず 3 つのデータの役割を整理します。

訓練データ (training data) はモデルが重みを学習するために使う正解付きデータです。検証データ (validation data) は学習には使わず、学習途中の性能チェックだけに使う別の正解付きデータです。テストデータ (test data) は最終的な性能を一度だけ測るために取っておくデータで、Early Stopping の判断にも使いません。検証データを停止判断に使うと、検証データに対してもわずかに最適化が進む (停止点という 1 個のハイパーパラメータを検証データに合わせて選んでいる) ため、本当に未知の性能はテストデータで測る、という分業が大切です。

過学習 (overfitting) は、訓練データの細部やノイズまで丸暗記してしまい、未知データへの性能 (汎化性能、generalization) が落ちる現象です。逆に学習が足りず訓練データすら当てられない状態を未学習 (underfitting) と呼びます。Early Stopping は、この両者の間にある「ちょうど良い学習量」を狙う技術です。

正則化 (regularization) とは、モデルが複雑になりすぎて過学習するのを抑える工夫の総称です。重みを小さく保つ罰則 (L2 正則化)、ニューロンをランダムに無効化する Dropout などがあり、Early Stopping もこの仲間です。Early Stopping の特徴は「モデルの容量 (表現できる複雑さ) そのものは変えず、学習の進行時間を制限することで実効的な複雑さを抑える」点にあります。学習途中の重みは初期値 (小さい値) から少しずつ大きくなっていくので、早く止めれば重みが大きくなりきらず、結果的に「単純なモデル」に近い状態で止まるのです。

### 仕組みを詳しく
実装の核心は patience (我慢回数) というカウンタです。

ステップ 1: 設定値を決める
- `patience = 5`: 検証性能がベスト記録を更新できないエポックが何回続いたら止めるか。
- `best_val = ∞` (最良の検証誤差。最初は無限大)。
- `counter = 0` (連続して改善しなかった回数)。
- `min_delta`: 「改善した」とみなす最小の差 (微小な揺らぎを改善と数えないため)。

ステップ 2: 各エポックの終わりにチェック

```python
val_loss = evaluate(model, val_data)      # 検証誤差を測る
if val_loss < best_val - min_delta:        # 意味のある改善なら
    best_val = val_loss
    counter = 0
    save_checkpoint(model)                 # この時点の重みを保存 (ベスト)
else:                                       # 改善しなかったら
    counter += 1
    if counter >= patience:                # patience 回連続で改善なし
        stop_training()                    # 学習打ち切り
```

ステップ 3: 終了時
保存しておいた「ベストだったときの重み」を最終モデルとして採用します。止めた瞬間の重みではなく、谷の底の重みを使うのがポイントです。

数値例で追います。`patience=3`、`min_delta=0` で、検証誤差が `0.50 → 0.42 → 0.40 → 0.41 → 0.43 → 0.44` と推移したとします。

```
epoch1: 0.50  (更新, best=0.50, counter=0)
epoch2: 0.42  (更新, best=0.42, counter=0)
epoch3: 0.40  (更新, best=0.40, counter=0)  ← ここがベスト
epoch4: 0.41  (改善せず, counter=1)
epoch5: 0.43  (改善せず, counter=2)
epoch6: 0.44  (改善せず, counter=3 = patience → 停止)
→ 採用するのは epoch3 の重み
```

`patience` を大きくすると「もう少し様子を見る」、小さくすると「すぐ諦める」挙動になります。検証誤差は確率的勾配降下法 (stochastic gradient descent、ランダムなミニバッチで学習するため毎回少し違う) の影響で揺らぐので、`patience=1` だと一時的な揺らぎで早く止まりすぎることがあります。実務では patience を 5〜20 程度に取り、揺らぎを吸収します。`min_delta` も同じ目的の道具で、たとえば 0.001 に設定すれば「0.001 未満の改善は改善とみなさない」ことで微小な揺らぎによるカウンタのリセットを防げます。

なぜ「学習時間の制限」が「重みの大きさの制約」と等価になるのか、もう少し踏み込みます。学習開始時、重みは 0 付近の小さな値です。勾配降下を繰り返すと重みは少しずつ最適点に向かって伸びていきます。途中で止めれば、重みは最適点まで到達せず、原点 (小さい値) と最適点の間で止まります。これは「重みを原点付近に引き戻す罰則 (L2 正則化)」をかけたときに得られる解とよく似た位置になります。具体的には、損失を最小点まわりで二次近似した線形モデルでは、t 回反復で止めた Early Stopping の解と、罰則係数 λ の L2 正則化の解が、おおよそ「学習率 × 反復回数 ≈ 1/λ」という対応で一致することが示せます。だから Early Stopping は L2 正則化の近似と見なせるのです。

### 手法の系譜と主要論文
Early Stopping は古くから経験的に使われてきましたが、その分析と理論化が段階的に進みました。

Morgan & Bourlard, "Generalization and Parameter Estimation in Feedforward Nets: Some Experiments", NIPS 1990 は、検証集合を使って過学習の開始を検知し学習を止めるという考えを早い段階で実験的に示した研究の 1 つです。ニューラルネットの汎化を「パラメータ数を増やしすぎず、学習も進めすぎない」観点から論じ、検証集合による停止が有効であることを報告しました。

Prechelt, "Early Stopping — But When?", in Neural Networks: Tricks of the Trade, 1998 は、Early Stopping を体系的に分析した古典です。彼は複数の停止基準 (検証誤差の絶対値、これまでのベストからの相対的な悪化度を表す generalization loss、訓練誤差の改善の停滞度との比 (progress quotient) など) を定義し、それぞれの挙動を多数のタスクで比較しました。動機は、当時「いつ止めるべきか」が完全に経験則任せだったため、定量的で再現可能な基準を与えること。結論として、「より緩い (遅めに止める) 基準は精度をわずかに改善するが学習時間が大幅に伸び、より厳しい基準は時間を節約するが精度を少し犠牲にする」という、精度と計算コストの明確なトレードオフを実験で示しました。

Goodfellow, Bengio, Courville, "Deep Learning", MIT Press 2016 (第 7 章 Regularization) は、深層学習の標準教科書として Early Stopping を正則化手法の 1 つに位置づけました。重要な理論的指摘は「Early Stopping は、二次近似のもとで線形モデルにおいて L2 正則化と近似的に等価である」という点です。学習を途中で止めることが、重みが大きくなりすぎないよう制約するのと数学的に似た効果を持つ、という洞察を与えました。さらに、Early Stopping の利点として「正則化の強さ (どこで止めるか) を 1 回の学習の中で自動的に探索できる」ことを挙げています。L2 正則化なら罰則係数を変えて何度も学習し直す必要がありますが、Early Stopping は 1 回の学習で検証誤差の谷を見つけるだけで済みます。逆に欠点として、停止判断のために学習を最後まで監視する必要があり、L2 のように学習中の重みに直接作用するわけではないことも指摘しています。

### 論文の実験結果(定量データ)
評価指標は文脈により異なりますが、回帰なら mean squared error (平均二乗誤差)、分類なら error rate や accuracy が使われます。Early Stopping の評価では「停止基準ごとに、最終的な汎化性能 (テスト誤差) と、それに要した学習時間 (エポック数) のペア」を比較するのが定石です。

Prechelt (1998) の実験では、多数のベンチマークタスクにわたって複数の停止基準を比較し、次のことが定量化されました。緩い基準 (generalization loss がある閾値を超えるまで粘る、など) を使うと、最も厳しい基準に比べてテスト誤差をわずかに (典型的に数パーセント程度) 改善できる一方で、学習時間は数倍に伸びることがあると報告されています。逆に最速で止める基準は、時間を大幅に節約しつつ精度低下を小さく抑えられる場合が多いことも示されました。結論として「万能の最適基準は存在せず、精度と計算予算のどちらを重視するかで選ぶべき」という実務的指針が得られました。実用的な落としどころとして、「ベスト更新から一定回数改善がなければ止める」という単純な patience 方式が、複雑な基準と遜色なく機能することも示唆されています。

深層学習の教科書的なアブレーションとしては、同一モデル・同一データで Early Stopping を入れた場合と、最大エポックまで学習し続けた場合を比較すると、後者は train loss が極小になる一方でテスト誤差が悪化する (過学習する) のが典型で、Early Stopping ありの方がテスト性能が良いことが繰り返し確認されています。これは Early Stopping が正則化として機能している直接的な証拠です。

### メリット・トレードオフ・限界
メリット:
- 過学習を防ぎ、未知データへの汎化性能を保てる。
- 「もう良くならない」学習を打ち切るので計算時間とコストを節約できる。
- 追加パラメータがほぼ不要で、どんなモデルにも後付けできる。
- 1 回の学習の中で正則化の強さ (止める時点) を自動探索できる。

トレードオフ・限界:
- 検証データを別に確保する必要があり、その分を訓練に回せない (データが少ない設定では痛い)。
- 検証誤差は揺らぐので、停止タイミング (patience や min_delta) の設定が難しい。
- 早く止めすぎると未学習、遅すぎると過学習になる。
- 学習率スケジュールと干渉することがある。cosine decay などで学習率が後半に向けて下がりきる設計だと、途中で止めると学習率が下がりきっておらず、本来到達できたはずの良い解に届かない場合がある。このため、スケジュールを最後まで回す前提では Early Stopping を使わず最終モデルを採用する、という選択もある。
- 二重降下 (double descent、過学習を越えてさらに学習すると再び汎化性能が改善する現象) が起こるモデルでは、検証誤差の最初の谷で止めるのが最適とは限らない。

### 発展トピック・研究の最前線
近年、Early Stopping の前提を揺るがす現象として deep double descent (Nakkiran ほか, ICLR 2020, arXiv:1912.02292) が注目されています。これは、モデルやデータの規模、あるいは学習エポック数を増やしていくと、テスト誤差が「下がる → 上がる (過学習) → 再び下がる」という二度目の降下を見せる現象で、十分に大きいモデルでは「過学習の谷を越えて学習し続けた方が良い」場合があることを示しました。特に「エポック方向の double descent (epoch-wise double descent)」では、検証誤差が一度上昇してから再び下がるため、古典的な Early Stopping の「最初の谷で止める」直感が、過剰パラメータ化された現代の巨大モデルでは必ずしも最適でないことを意味します。とはいえ、この再降下が起きる条件は限定的で、多くの実務設定では最初の谷で止める Early Stopping が依然として有効です。

実務面では、Early Stopping は学習率スケジューラ (cosine decay など) やチェックポイント保存と組み合わせて使うのが標準です。検証指標として loss ではなくタスク本来の評価指標 (accuracy や IoU など) を監視する方が、最終目的に直結した停止判断ができることもあります。また、複数指標を組み合わせたり、検証誤差の移動平均で揺らぎを抑えたり、複数チェックポイントの重みを平均する (model soup / SWA、確率的重み平均) ことで停止点の選択に頑健性を持たせる工夫が現場で使われます。

### さらに学ぶための関連トピック
- [Batch Normalization](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Transfer Learning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [Gradual Unfreezing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [Layer-wise LR Decay (LLRD)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [Discriminative / Layer-wise Learning Rates](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)


## Exponential LR Decay

### ひとことで言うと
学習率(ニューラルネットのパラメータを1回どれくらい動かすかを決める数値)を、毎ステップ(または毎エポック)わずかな一定割合ずつ掛け算で減らしていくやり方です。「毎エポック0.95倍する」のように、なめらかな下り坂を描きながら、決して0には到達しないまま限りなく小さくなっていきます。設定する値が実質1つ(減衰率)だけで済む、最も単純な連続減衰スケジューラ(学習率を時間とともに変化させる仕組み)です。

### 直感的な理解
温かいコーヒーが冷めていく様子を思い浮かべてください。最初は熱いので急速に温度が下がりますが、室温に近づくにつれて冷め方はゆっくりになります。これは「今の温度と室温の差に比例して冷める」という指数的な減衰(ニュートンの冷却法則)で、自然界のあちこちに現れる形です。放射性物質の崩壊、コンデンサの放電なども同じ数学です。

指数減衰の学習率もこれと同じ形をしています。学習率が大きいうち(序盤)は1回あたりの減り幅も大きく、小さくなってきた終盤は減り幅も小さくなります。これは「序盤は大胆に、終盤は慎重に」という学習率減衰の理想にちょうど合っています。

階段状に下げる Step decay には2つの不満がありました。1つは下げるタイミングを人間が決めねばならないこと、もう1つは節目で学習率が不連続にジャンプして学習が一瞬不安定になることです。指数減衰は「毎回ほんの少しだけ掛け算で減らす」ことで、この階段の段差を無限に細かくした極限を作ります。結果としてなめらかなカーブの下り坂になり、不連続なジャンプがありません。

### 基礎: 前提となる概念
理解に必要な概念を噛み砕きます。

- 学習率(learning rate): 勾配の向きに1回でどれくらいパラメータを動かすかの係数。大きいと速いが粗く、小さいと細かいが遅い。
- 減衰(decay): 学習が進むにつれて学習率を小さくしていくこと。
- 等差的に減らす(引き算): 毎回一定量を引く。たとえば毎回 0.01 ずつ減らす。終盤にマイナスや 0 になりやすく扱いにくい。
- 等比的に減らす(掛け算): 毎回一定倍率を掛ける。たとえば毎回 0.95 倍。必ず正の値を保ち、序盤大きく・終盤小さくと減る。指数減衰はこちらです。
- ステップ単位とエポック単位: 学習率を更新する頻度。1ステップ = パラメータ1回更新、1エポック = データ1周。どちらの単位で減衰させるかで挙動が大きく変わります(後述)。
- 半減期(half-life)の考え方: 毎回 gamma 倍するとき、学習率が半分になるまでの回数は `log(0.5)/log(gamma)` 回です。指数減衰の「速さ」を半減期で把握すると、gamma の感覚がつかみやすくなります。

掛け算で減らすという性質には数学的な意味があります。引き算(等差)で減らすと、序盤も終盤も同じ幅で減るため、終盤に学習率がマイナスになったり急に 0 付近になったりして扱いにくいのです。掛け算(等比)なら必ず正の値を保ったまま、なめらかに小さくなり続けます。指数減衰は数学的には「dlr/dt = −k·lr」という微分方程式の解で、その解が `lr = lr_init · e^(−k t)` という指数関数になります。

### 仕組みを詳しく
指数減衰の式はとてもシンプルです。

```
lr(t) = lr_init * gamma ^ t
```

各記号の意味です。

- `lr_init`: 学習開始時の学習率。
- `gamma`(ガンマ): 1ステップ(または1エポック)ごとに掛ける倍率。1未満の正の数。0.95、0.99 など。
- `t`: 何回 step したか(エポック数またはイテレーション数)。

数値例。`lr_init = 0.1`、`gamma = 0.9`(毎エポック0.9倍)とすると、

- エポック0: 0.1
- エポック1: 0.09
- エポック2: 0.081
- エポック5: 約 0.059
- エポック10: 約 0.035
- エポック20: 約 0.012
- エポック50: 約 0.0005

掛け算なので、最初の数エポックでは大きく減り(0.1→0.09→0.081)、後半は同じ 0.9 倍でも減り幅が小さくなる(0.012→0.011…)のが分かります。この場合の半減期は `log(0.5)/log(0.9) ≈ 6.6` エポックです。学習率の時間変化を ASCII 図にすると、最初急で後ろがゆるやかな、なめらかな下り坂です。

```
lr
0.1 |*
    | *
    |  *
    |   **
0.05|     ***
    |        *****
    |             *********
0.0 |                      ************
    +--------------------------------------> epoch
```

gamma の選び方には注意が必要です。1ステップごとに掛けるのか、1エポックごとに掛けるのかで、必要な gamma が大きく変わります。たとえば1エポックが 1000 ステップで「1エポックで 0.9 倍にしたい」なら、1ステップあたりの gamma は 0.9 の 1000 乗根、つまり約 0.99989 という1に非常に近い値になります。ここを取り違えると、学習率があっという間に 0 近くまで落ちて学習が止まる、という事故が起きやすく、初心者がつまずく典型的な落とし穴です。逆に gamma を 1 に近くしすぎると、ほとんど減らず固定学習率と変わらなくなります。

実用上は、`scheduler.step()` 相当の呼び出しが「エポックごとか、ステップごとか」をフレームワークの実装に合わせて意識する必要があります。`ExponentialLR` 系の実装は呼ぶたびに gamma 倍するだけなので、呼ぶ頻度の設計が結果を直接左右します。

なお、TensorFlow/Keras の `ExponentialDecay` には `staircase=True` というオプションがあり、これを使うと連続の指数カーブではなく階段状になります。これは指数減衰と Step decay が地続きの関係にあることを端的に示しています。階段の段差を細かくしていった極限が連続な指数減衰、という関係です。

### 手法の系譜と主要論文
指数減衰は特定の論文の発明ではなく、最適化の古典として深層学習に持ち込まれた手法です。位置づけは次の通りです。

1. Robbins & Monro(1951)の確率近似理論。SGD の理論的源流で、収束のために学習率 `a_t` が「Σ a_t = ∞ かつ Σ a_t^2 < ∞」を満たすべきと示しました。指数減衰は後者を満たす一方で前者を満たさない(等比級数は収束する)ため、厳密な理論的収束保証の観点では `1/t` 系の減衰のほうが好まれるという背景知識があります。実務では十分小さくなる前に学習を打ち切るので問題になりにくいです。

2. Szegedy et al., GoogLeNet / Inception 系(CVPR 2015 など)。Google 系の画像分類レシピで「数エポックごとに一定割合で学習率を下げる」指数減衰(例: 8エポックごとに ×0.94 など)が広く使われました。RMSProp との組み合わせが定番で、後の EfficientNet までこの系譜が続きます。

3. Goodfellow, Bengio & Courville, "Deep Learning"(MIT Press, 2016)。深層学習の標準教科書。学習率減衰の基本形のひとつとして指数減衰を紹介しています。連続でなめらかな減衰の最も単純な実現方法という位置づけです。同時に「適切な gamma を見つけるのが難しく、間違えると学習率が早く落ちすぎる、または減らなさすぎる」という実務上の難しさも指摘しています。

4. Smith, "Cyclical Learning Rates for Training Neural Networks", WACV 2017(arXiv:1506.01186)。周期的に学習率を上下させる自分の手法(CLR)の優位性を示すため、当時広く使われていた既存手法に指数減衰を含めて整理し、「これらは単調に減るだけで局所解を脱出できない」と問題提起しました。指数減衰は安定だが探索能力に乏しい、という位置づけです。

5. Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019(arXiv:1905.11946)。実際の最先端モデルでも指数減衰が使われた例です。EfficientNet の学習では「2.4 エポックごとに学習率を 0.97 倍する」という指数減衰(RMSProp 最適化、momentum 0.9 と組み合わせ)を採用し、ImageNet で当時の最高精度を達成しました。GoogLeNet/Inception 系の伝統を引き継いだレシピで、指数減衰が大規模学習でも通用することを示しました。

その後、Transformer 系の学習では warmup + コサイン、または warmup + linear が主流になり、指数減衰は CNN 系の一部レシピ(EfficientNet、MobileNet など Google 系)を除いて実務での出番が減りました。強化学習では今も探索率(epsilon)やエントロピー係数の指数減衰として頻繁に使われます。

### 論文の実験結果(定量データ)
EfficientNet 論文は指数減衰を含むレシピで ImageNet(約 120 万枚・1000 クラス)を学習し、評価指標 top-1 精度(第1候補が正解の割合。高いほど良い)で当時の最高水準を達成しました。

報告された代表的な数値です。

- EfficientNet-B0(最小モデル)で top-1 精度およそ 77.1%、パラメータ数約 530 万。当時の ResNet-50(約 2600 万パラメータ、top-1 約 76%)と同等以上の精度を、約 5 分の 1 のパラメータで達成しました。
- EfficientNet-B7(最大モデル)で top-1 精度およそ 84.3%、top-5 およそ 97.0%。当時の最高精度モデル(GPipe など)と同等の精度を、約 8.4 分の 1 のパラメータ・約 6.1 倍の推論高速化で実現しました。
- 学習レシピは RMSProp(decay 0.9、momentum 0.9)、学習率は 0.256 開始で「2.4 エポックごとに ×0.97」の指数減衰、加えて weight decay とドロップアウト。指数減衰部分だけを取り出した効果のアブレーションは論文の主眼ではありませんが、Google 系のこの指数減衰レシピが Inception 以来安定して機能してきた実績があります。

指標の意味として、top-1 精度の差は実用上きわめて大きく、76% と 77% の差でも誤分類が無視できない量変わります。EfficientNet の貢献の主役は複合スケーリング(深さ・幅・解像度を同時に調整する手法)であり、指数減衰はそれを支える安定した学習スケジュールという役割です。つまり指数減衰自体が精度を押し上げるというより、なめらかで予測可能な減衰として土台を担っています。

なめらかな方式の横並び比較(独立研究)では、画像分類で指数・linear・コサインの差は概しておよそ 1 ポイント以内で、コサインがわずかに有利という報告が多いものの、Google 系の指数減衰レシピは依然として競争力を保っています。

### メリット・トレードオフ・限界
メリット

- 実質1つのパラメータ(gamma)だけで設定でき、極めて単純。
- なめらかな連続減衰なので、Step decay のような不連続ジャンプによる不安定さがない。
- 掛け算減衰なので学習率が必ず正の値を保ち、序盤大きく・終盤小さくという理想的な減り方をする。
- 半減期で速さを直感的に把握でき、強化学習の探索率減衰など幅広い場面で再利用できる。

トレードオフと限界

- gamma の選び方がシビアで、ステップ単位かエポック単位かを取り違えると学習率が早く落ちすぎて学習が止まる事故が起きやすい。
- 0 に漸近するだけで決して 0 に到達しないため、「ちょうど何エポックで 0 にしたい」という設計はやりにくい(その用途は linear 減衰や polynomial 減衰のほうが向く)。
- 単調に減るだけなので、局所解・鞍点から抜け出す能力はない(コサイン warm restart のような再起動はできない)。
- 検証指標を見て適応はしないので、データに合わせた自動調整ができない(plateau 監視型と対照的)。
- warmup と組み合わせないと、開始直後の高い学習率で不安定になることがある。
- 確率近似理論の観点では等比減衰は収束条件 Σ a_t = ∞ を満たさず、厳密には大域収束を保証しにくい(実務では問題になりにくいが理論的には弱点)。
- 未解決の論点として、指数減衰・linear 減衰・コサイン減衰のどれがどのタスクで最適かを事前に予測する理論はなく、いまだ経験則とハイパーパラメータ探索に頼っています。

### 発展トピック・研究の最前線
- warmup との組み合わせ: 開始直後に学習率を線形に上げてから指数減衰に入る構成が、不安定さを抑える定番です。
- 最適化手法との相互作用: 指数減衰は SGD よりも RMSProp や Adam と組み合わせて使われることが多く、最適化手法ごとに相性の良い減衰が異なるという観察があります。Adam 系は内部で勾配を正規化するため、外側のスケジュールの効き方が SGD とは異なります。
- なめらかな減衰の比較研究: 指数・linear・polynomial・コサインを横並びで比較する研究では、多くの画像分類タスクでコサインがわずかに有利という報告が多い一方、Google 系の指数減衰レシピも依然として競争力があり、決定的な勝者はいません。
- 学習率と weight decay の結合: 学習率を変えると実効的な weight decay も変わるため、両者を切り離す AdamW(Loshchilov & Hutter 2019)などの研究が、指数減衰の解釈にも影響を与えています。
- 強化学習・探索スケジュールでの利用: epsilon-greedy の epsilon やエントロピー係数の指数減衰は今も標準で、「探索から活用へ」の移行を滑らかに表現する道具として現役です。

### さらに学ぶための関連トピック
- [Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Linear LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## FastBEV

### ひとことで言うと

FastBEV は、複数のカメラ画像から「真上から見た周辺地図(BEV: Bird's-Eye-View、鳥の目線の地図)」を作る処理を、自動運転車に載せる非力な計算機(車載チップ)でもリアルタイムに動かせるほど軽くした手法です。速さの肝は2つあります。第一に、各ピクセルの奥行き(カメラからの距離)をニューラルネットで予測する重い処理をやめ、単純な幾何投影で済ませること。第二に、「カメラ画像のどのピクセルが BEV のどのマスに対応するか」をあらかじめ計算してルックアップテーブル(対応表)に焼き込み、推論時はその表を引くだけにすることです。これにより、データセンター級 GPU を前提にした重い手法を、車載 GPU 上で数十ミリ秒/フレームで動かせるようになりました。

### 直感的な理解

自動運転車は前後左右に6台前後のカメラを付けています。それぞれが別々の方向の「前から見た画像」を撮りますが、運転判断に使うには、これらを1枚の「上から見た周辺地図」にまとめたい。なぜなら、上から見た地図(BEV)では車や障害物の位置関係が距離によらず素直に表現でき、経路計画や衝突回避がやりやすいからです。

問題は速さです。車は時速60kmなら1秒に約17m進みます。カメラが30fps(1秒に30枚)で映像を送ってくるなら、1枚あたり約33ミリ秒以内に「画像→BEV地図→判断」を終えないと、処理が追いつかず古い地図で運転することになり危険です。研究のベンチマークで最高精度を出す手法の多くは、奥行き予測ネットワークや巨大な中間テンソル、複雑な集約処理を持ち、これらは強力なデータセンター GPU では動いても、車に積めるサイズ・電力・コストの車載チップでは間に合いません。

FastBEV は「ベンチマーク最強」ではなく「実車のチップで、十分な精度を、間に合う速度で」を狙ったデプロイ志向の設計です。料理に例えると、毎回包丁で野菜を計りながら切る(=毎フレーム奥行きを予測する)のをやめ、あらかじめ「この食材はここに置く」というレシピ表を作っておき、本番では表を見て配置するだけにした、というイメージです。レシピ表は最初に1回作る手間がかかるだけで、本番では一切の計算をせずに済みます。FastBEV の LUT(ルックアップテーブル)はまさにこのレシピ表に当たります。

### 基礎: 前提となる概念

理解に必要な言葉を一つずつ噛み砕きます。

- BEV (Bird's-Eye-View): 真上から見下ろした視点の地図表現。車の周りを格子状のマス(グリッド)に区切り、各マスに「そこに何があるか」の特徴を持たせます。複数カメラを統合する共通の土俵になります。詳しくは [BEVグリッド](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)。
- 推論 (inference): 学習済みのモデルを実際に走らせて答えを出すこと。学習は1回だけデータセンターでやればよいですが、推論は走行中の車の上で1秒間に何十回も繰り返す必要があります。だから推論の速さが死活問題になります。
- リアルタイム: ここでは「カメラ映像が来るたびに遅れずに処理が間に合う」状態。30fps なら1フレーム33ミリ秒以内が目安です。
- 車載チップ (edge device): 例えば NVIDIA Jetson シリーズや車載 SoC(System on Chip)。データセンターの A100/H100 のような GPU に比べてメモリも演算性能も一桁以上小さく、消費電力にも厳しい制約があります。たとえば Jetson AGX Orin の演算性能はおおむね数十〜275 TOPS の領域で、データセンター GPU の数分の一から十分の一程度です。
- カメラ校正 (calibration): カメラの内部パラメータ(焦点距離やレンズ歪み)と外部パラメータ(車に対する取り付け位置・向き)の情報。これがあれば「実世界の3次元の点が画像のどのピクセルに写るか」を幾何だけで計算できます。
- 奥行き予測: 画像の各ピクセルが「カメラからどれだけ遠いか」を当てる処理。これが分かれば2次元画像を3次元空間に正しく持ち上げられますが、当てるのは難しく、専用ネットワークが必要で重い。
- TensorRT: NVIDIA が提供する推論最適化ライブラリ。学習済みモデルを GPU 向けに変換・融合・量子化し、推論を高速化します。ただし「サポートされた演算子しか速くならない」「動的な分岐やソート、scatter は苦手」という制約があり、モデルがこのライブラリに「載りやすい」形かどうかが実機速度を大きく左右します。

BEV を作る代表手法に Lift-Splat-Shoot([Lift-Splat-Shoot (LSS)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform), 2020)があります。これは各ピクセルについて「奥行きの確率分布」を予測し(例: このピクセルは10m先の確率0.6、20m先の確率0.3…)、その確率で重み付けして特徴を3次元空間にばらまき(lift)、BEV グリッドに集約します(splat)。精度は高いのですが、奥行きネットワーク・巨大な中間テンソル・splat 集約のどれもが重く、しかも splat は各点を BEV マスへ散らす scatter 操作なので、TensorRT などの推論エンジンに綺麗に載せにくいという実務上の壁がありました。FastBEV はこの壁を乗り越えるために生まれました。

### 仕組みを詳しく

FastBEV は論文では大きく4つの要素から成ります。中心は Fast-Ray Transformation(高速な視点変換)で、これに Multi-View/Multi-Scale データ拡張、時間方向の集約、そしてエッジ向けの最適化が組み合わさります。

#### 要素1: Fast-Ray Transformation(奥行き予測なしの均一投影 + LUT)

画像から BEV への変換を、奥行き予測なしの純粋な幾何投影で行います。発想の出発点は M2BEV([M2BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform))の「奥行きを厳密に当てず、視線上で均一に仮定する」考え方です。

仕組みはこうです。まず BEV のボクセル格子(車の周りの3次元のマス目。voxel = volume + pixel、3次元版のピクセル)を定義します。各ボクセルの中心は実世界の3次元座標 (x, y, z) を持ちます。カメラ校正があれば「この3次元点はカメラ k の画像の (u, v) ピクセルに写る」が幾何だけで計算できます。LSS のように奥行き分布を予測する代わりに、ある視線(レイ)に沿って並ぶ複数のボクセルは、すべて同じ画像ピクセルから特徴を受け取る、と割り切ります(均一深度仮定: uniform depth assumption)。つまり「カメラから出た1本の光線上では、特徴は奥行きが分からないので一様にコピーする」という近似です。

FastBEV の核心は、この「ボクセル→(カメラ, ピクセル)」の対応を実行時に計算せず、学習・デプロイ前に1回だけ計算してルックアップテーブル(LUT)に焼き込むことです。

```
事前計算(1回だけ。カメラ校正が固定なら学習前に確定):
  各 BEVボクセル → どのカメラの・どのピクセルから特徴を取るか の対応表を作る
  例) voxel[12][34][2]  ←  camera 1 の feature pixel (87, 203)

推論時(毎フレーム):
  表を引いて、画像特徴をボクセルへ gather(集めてコピー)するだけ
  座標変換も射影行列の掛け算も奥行き予測も実行時には一切不要
```

実行時の処理が「表を引いてメモリをコピーする」だけなので、計算量はメモリ転送に近く非常に軽い。FastBEV はさらに、密(dense、全ボクセル全画素)ではなく疎(sparse、対応がある所だけ)な投影を使い、無駄な計算を省きます。BEV グリッドのうち実際にカメラに写るのは一部なので、この sparse 化が効きます。論文では、密な投影では全ボクセルにアクセスするのに対し、レイ上で実際に有効なボクセルだけを処理することで投影部分の計算を大幅に削減できると述べています。

tensor の形でイメージすると次の通りです。

```
画像特徴:     [N, C, H, W]        N=カメラ台数(例6), C=特徴次元, H/W=特徴マップ縦横
LUT:          各 voxel → (cam_idx, h, w) の整数インデックス表(事前計算・固定)
gather:       LUT で indexing → BEV voxel 特徴 [C, X, Y, Z]
高さ集約:     Z をまとめて → BEV 特徴 [C', X, Y]
```

before / after を一言でまとめると、LSS が「奥行き予測 → 巨大テンソル lift → splat 集約(ソート/scatter)」だったのを、FastBEV は「事前計算した整数表を引く → gather するだけ」に置き換えた、という違いです。後者は分岐もソートもなく、TensorRT で素直に最適化できます。具体的な数値感で言えば、LSS 系では奥行き分布の確率(例えば 60 ビン)と特徴チャンネル数を掛け合わせた巨大な中間テンソルが生じますが、FastBEV はこれを一切作りません。

#### 要素2: Multi-View / Multi-Scale データ拡張

投影が固定の LUT である分、モデルが投影の作為に過学習しないよう、学習時に頑健性を補う工夫を入れます。画像側(image-space)と BEV 側(BEV-space)の両方で、ランダムな回転・反転・スケール変更・クロップなどを加えます。BEV 側の拡張は当時の高速 BEV 手法ではまだ珍しく、FastBEV の精度向上に寄与しました。BEV 空間でデータ拡張をすると、固定 LUT が前提とするカメラ-BEV 対応が崩れるため、拡張に合わせて LUT を再計算するか、拡張を考慮した投影を行う必要があります。Multi-Scale(複数解像度)の特徴を使うことで、近距離の細かい物体と遠距離の大きな構造の両方を捉えます。

#### 要素3: 時間方向の集約(Temporal fusion)

1フレームだけでは動いている物体や一時的に隠れた物体に弱い。そこで過去フレームの BEV 特徴を、自車の動き(エゴモーション)で現在の座標系に合わせてから連結・集約します。発想は BEVDet4D(Huang & Huang, 2022)と同系統で、速度推定の精度や遮蔽への頑健性が上がります。FastBEV では複数フレームの BEV を効率よく重ねることで、計算を増やしすぎずに時間情報を取り込みます。論文では3〜4フレーム程度を扱い、過去フレームの特徴は計算済みのものをキャッシュして再利用するため、追加コストは BEV の連結とエゴモーション補正だけで済みます。

#### 要素4: エッジ最適化

LUT gather と sparse 投影に加え、演算子を TensorRT に載りやすい形に保ち、必要なら CUDA/C++ で投影部分を実装するなど、推論エンジンへのデプロイを意識した実装上の工夫が随所にあります。たとえば LUT gather を専用カーネルとして書き、ビュー変換とその前後をひとつの最適化可能なグラフに収めることで、CPU-GPU 間の往復や中間テンソルの確保を減らします。これが「論文の中だけで速い」ではなく「実機で速い」を実現する地味で重要な部分です。

### 手法の系譜と主要論文

幾何ベース BEV 変換の系譜の中で FastBEV の位置を押さえます。

- Roddick, Kendall, Cipolla, "Orthographic Feature Transform" (OFT, BMVC 2019, arXiv:1811.08188)。ボクセルから画像へ投影し対応領域を積分する pull 型幾何変換の祖。奥行きを当てない発想の源流。[OFT (Orthographic Feature Transform)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)。
- Philion & Fidler, "Lift, Splat, Shoot" (ECCV 2020, arXiv:2008.05711)。奥行き確率を明示予測する push 型。精度は高いが重く、FastBEV が高速化で置き換えようとした元手法。[Lift-Splat-Shoot (LSS)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)。
- Huang, Huang, Zhu, Du, Wang, "BEVDet" (2021, arXiv:2112.11790)。LSS を多カメラ3次元検出に持ち込み、画像エンコーダ・ビュー変換・BEV エンコーダ・検出ヘッドという定番パイプラインを確立。
- Xie, Geng, Hu, Li, Sun, Hua, Zhou, "M2BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified BEV Representation" (2022, arXiv:2204.05088)。「ピクセルの奥行きはボクセル軸に沿って均一」と仮定し、奥行き予測を捨てた。FastBEV の直接の出発点。FastBEV はこれを LUT 化・sparse 化してさらに速くした。[M2BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)。
- Huang & Huang, "BEVDet4D" (2022, arXiv:2203.17054)。過去 BEV を現在に合わせて重ねる時間集約。FastBEV の temporal fusion と同系統。
- Li, Huang, Chen, Lu, Lu, Wang, Li, Zhou, Yu, Wang, Wei, "Fast-BEV: A Fast and Strong Bird's-Eye View Perception Baseline" (2023, arXiv:2301.12511、ショート版 arXiv:2301.07870 は NeurIPS 2022 の自動運転向け機械学習ワークショップ系)。M2BEV の均一投影を LUT + sparse 化し、Multi-View/Multi-Scale 拡張と temporal fusion を足して、車載チップ向けに最適化した軽量フレームワークを提案。

系譜を一言で言うと、OFT(2019, 幾何の pull)→ LSS(2020, 奥行き予測の push)→ BEVDet(2021, 多カメラ検出への定番化)→ M2BEV(2022, 奥行きを捨てる割り切り)→ FastBEV(2023, それを LUT で実装し車載へ)という流れです。FastBEV は新しい原理を発明したというより、既存の割り切りを「実機で動く形」に徹底的に工学化した点に価値があります。研究の世界では「新しい原理」だけでなく「既存の良いアイデアを本当に使える形にする」工学的貢献も重要であり、FastBEV はその好例です。

### 論文の実験結果(定量データ)

評価指標と数値の意味から説明します。FastBEV は主に nuScenes(自動運転の標準ベンチマーク。6カメラ・LiDAR・レーダーを積んだ車で集めた大規模データ。約1000シーン)で3次元物体検出を評価します。

- mAP (mean Average Precision): 検出の正確さ。値が大きいほど良い。0〜1(または0〜100%)。nuScenes では中心距離ベースのマッチングで複数しきい値の AP を平均します。
- NDS (nuScenes Detection Score): nuScenes 独自の総合指標で、mAP に加えて位置・大きさ・向き・速度・属性の誤差(mATE, mASE, mAOE, mAVE, mAAE)をまとめたもの。0〜1。これも大きいほど良い。検出の「当たり外れ」だけでなく「どれだけ正確か」も測るため、実用的な評価指標とされます。
- レイテンシ (latency): 1フレームの処理にかかる時間。小さいほど速い。FPS(1秒あたりの処理枚数)は 1000/レイテンシ(ms)で求まります。

報告によると、FastBEV はモデルサイズを M0〜M5 のように複数用意しています。最小の M0 は軽量バックボーンと低解像度入力で、最大の M5 は重いバックボーンと高解像度入力です。軽量設定では NVIDIA 車載グレード GPU(Jetson AGX Orin 等)上でリアルタイム域(報告ではおおむね 50fps を超える設定も可能で、最大モデルでも数十 fps 級)で動作しつつ、nuScenes 検証セットで実用的な検出精度(報告では小型モデルで mAP 0.3 前後・NDS 0.4 前後、大型モデルではさらに高く mAP 0.4 前後・NDS 0.5 前後の領域。設定やモデル規模・解像度で変動)を達成したとされています。重要なのは「データセンター GPU が前提の重い手法と比べ、必要な計算量・メモリが大幅に小さい」点で、同等精度帯で見たときの推論速度が数倍規模で速いことが売りです。論文は同程度の精度を持つ既存手法に対し、推論を大きく加速したと報告しています。

アブレーション(どの要素を抜くと何が起きるか)で報告されている傾向は次の通りです。これらは FastBEV の設計判断の根拠になります。

- LUT を使わずに毎フレーム投影を計算すると、精度はほぼ同じだが推論が大幅に遅くなる。つまり LUT は精度を犠牲にせず速度だけを稼ぐ「ただ働き」の最適化で、これが FastBEV の核心だと示されます。投影計算は本来カメラ台数×ボクセル数の行列演算を要しますが、LUT 化でこれをゼロにできます。
- BEV 側のデータ拡張を外すと精度が低下する。固定投影への過学習を抑える効果があることの裏付けです。
- temporal fusion(時間集約)を外すと、特に速度推定の精度(mAVE のような速度誤差指標)と遮蔽下の検出が悪化する。動きの情報が効いていることを示します。複数フレームを使うほど NDS が改善する傾向が報告されています。
- 入力解像度やバックボーン(特徴抽出 CNN)の規模を上げると精度は上がるが、その分レイテンシも増える。M0〜M5 はこの速度-精度トレードオフ上の点を並べたものです。

なお、これらの数値は論文・実装・ハード・量子化の有無によって幅があるため、おおよその傾向として捉えるのが適切です。とくに Jetson 上の実測値は TensorRT のバージョンや FP16/INT8 の選択で大きく変わります。

### メリット・トレードオフ・限界

メリット:
- 奥行き予測ネットワークと巨大な中間テンソルが不要で、計算・メモリが非常に軽い。
- LUT で gather するだけなので分岐もソートもなく、TensorRT 等の推論エンジンに載せやすく、車載チップでのリアルタイム推論が現実的。
- データ拡張と時間集約で、軽量でも実用的な精度を確保できる。M0〜M5 で速度-精度トレードオフを選べる。
- 検出・セグメンテーションなど複数の下流タスクに同じ BEV 表現を流用できる(統一表現の利点)。

トレードオフ・限界:
- 均一深度の近似なので、奥行きの細かい弁別が決定的に重要な場面(同一視線上の前後の物体の分離など)では精度の上限がある。LiDAR 教師で奥行きを学ぶ手法には及ばないことが多い。
- カメラ校正が固定である前提。LUT は校正に依存するため、校正がずれたり可変焦点・可変取り付けだと表の再計算が必要で、オンライン校正とは相性が悪い。サスペンションの沈み込みや経年で外部パラメータが変化する実車では、定期的な再校正と LUT 更新が要る。
- ベンチマーク最高精度を狙う Transformer ベースの BEV(BEVFormer など)や丁寧な奥行き手法には純粋な精度では及ばないことがある。FastBEV はあくまで速度と精度のバランス重視。
- 視線方向の特徴のにじみ(均一深度ゆえ同じ特徴が前後ボクセルに重複)は後段の BEV エンコーダと複数カメラの重なりに依存して解消する設計で、M2BEV と同じ弱点を引き継ぐ。

研究上の未解決課題としては、(1)校正フリー化やオンライン校正への対応、(2)均一深度近似と本物の奥行き予測の中間で速度を保ちつつ精度を上げる方法、(3)LUT を可変解像度・可変カメラ構成に動的対応させる仕組み、(4)INT8 量子化と精度維持の両立、などが挙げられます。

### 発展トピック・研究の最前線

FastBEV 以降、車載リアルタイム BEV の効率化は複数方向に進んでいます。一つは集約処理自体の高速化で、LSS 系の splat を専用 CUDA カーネルで一気に速くする BEVPoolv2 のようなアプローチ([BEVPool / BEVPoolv2 (高速voxel pooling)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))が、push 型の弱点だった集約速度を改善しました。FastBEV の LUT pull 型と、BEVPoolv2 の高速 push 型は、どちらも「車載で間に合わせる」という同じ目標に異なる経路で迫っています。pull 型は「埋めたい BEV マスから画像を引く」ので出力サイズが制御しやすく、push 型は「画像点をどこへ飛ばすか」が起点で奥行き表現が豊かという相補関係があります。

もう一つは、奥行きを完全には捨てずに軽量に扱う中間路線です。LiDAR で奥行きを教師として与えて軽い奥行きヘッドを学習させる BEVDepth のような手法は、FastBEV の「奥行きを捨てる」割り切りと LSS の「奥行きを丁寧に当てる」の間を埋めます。さらに、量子化(重みや活性を低ビットで表現して速くする)、知識蒸留(大きいモデルの知識を小さいモデルへ移す)、構造の枝刈りといった一般的なモデル圧縮技術と組み合わせることで、車載デプロイの限界がさらに押し広げられています。研究の最前線では「精度を保ったまま、どこまで安く速くできるか」のフロンティアを各手法が押し続けており、FastBEV はその速度側の代表的な里程標であり続けています。

### さらに学ぶための関連トピック

- [M2BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [Simple-BEV](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [Lift-Splat-Shoot (LSS)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [BEVPool / BEVPoolv2 (高速voxel pooling)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [OFT (Orthographic Feature Transform)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)
- [BEVグリッド](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/04_bev-view-transform)


## FlashAttention

### ひとことで言うと
FlashAttention とは、Transformer の中核である attention (注意機構)の計算を、GPU のメモリ事情を考慮して賢く組み替えることで、メモリ使用量を大幅に減らし同時に高速化する計算手法(GPU カーネル)です。出力する答えは普通の attention とまったく同じ(厳密、exact)ですが、途中で現れる巨大な中間行列を GPU の遅いメモリに書き出さずに済ませるのが鍵です。「IO-aware(入出力=メモリの読み書きを意識した)」設計と呼ばれます。

### 直感的な理解

膨大な答案を採点する場面を想像してください。全員ぶんの答案を一度に机に広げる(=巨大な行列を全部メモリに置く)と机からあふれます。代わりに「一束ずつ取り出して採点し、途中集計だけ手元に残して束を片付ける」方式にすれば、机が小さくても全員ぶん採点でき、しかも答案の出し入れ(=遅いメモリへの往復)が減ります。FlashAttention はまさにこの「一束ずつ処理して途中集計を持ち回る」やり方を attention に適用したものです。結果(最終的な採点)は全部広げた場合と完全に一致します。

なぜ重要かというと、attention は系列が長くなるほど中間データが爆発的に増え、これが Transformer を重くする最大の原因だからです。そして実は、その重さの正体は計算量よりも「遅いメモリへの読み書き回数」にあります。FlashAttention はそこを直撃します。FLOP(浮動小数点演算回数)を1つも減らさずに、メモリトラフィックだけを減らして実時間を縮める、という点が他の高速化と決定的に異なる発想です。

### 基礎: 前提となる概念

#### attention とは何か
Transformer の attention は「系列の中の各要素が、他のどの要素にどれだけ注目すべきか」を計算する仕組みです。文章中の単語どうし、画像中のパッチ(小領域)どうしの関係を測ります。各要素から Query(質問)Q、Key(鍵)K、Value(値)V という3つのベクトルを作り、次を計算します。

```
S = (Q × Kの転置) / √d   … 全要素ペアの類似度スコア (行列)。√d でスケール
P = softmax(S)            … 各行を確率(注目度)に変換
O = P × V                 … 注目度で重み付けした値の合計
```
ここで d は各ベクトルの次元数で、√d で割るのは内積が大きくなりすぎて softmax が尖りすぎるのを防ぐためです。softmax とは、数値の並びを「合計が1の確率分布」に変換する関数で、大きい値ほど大きな重みを与えます。

#### N^2 の壁
S の大きさが問題です。系列長(要素数)を N とすると、S は N×N 行列になります。全ペアの組み合わせだからです。

数値例: 系列長 N=4096、ヘッド数 16(attention は複数の視点=ヘッドで並列に行う)の場合、S は 4096×4096 ≈ 1670万要素を16ヘッドぶん持ちます。fp16(2バイト)でも1ヘッド約33MB、全ヘッドで0.5GB超。同サイズの P も別に必要で、バッチ方向にも掛かります。N を倍の8192にすると N^2 なので4倍に膨れます。つまり attention のメモリと計算は系列長の2乗で増えます。これが長系列・高解像度で Transformer が重くなる根本原因です。

#### 本当のボトルネックはメモリの往復
GPU には大容量だが遅い HBM(High Bandwidth Memory、VRAM、数十GB)と、計算ユニット隣の爆速だが極小の SRAM(数十〜数百KB)があります。両者の帯域差は1桁以上(HBM が毎秒1〜3TB、SRAM は実効で毎秒10〜20TB 級)あります。普通の attention 実装は巨大な S と P を HBM に書き出し、また読み戻します。この HBM 往復が、実は行列積の計算そのものよりずっと時間を食っています。GPU の演算ユニットは速すぎて、データの到着を待っている状態(メモリ律速、memory-bound)なのです。FlashAttention はこの無駄な往復を消すことを狙います。これが「計算量(FLOP)を減らす」のではなく「IO を減らす」という独自の着眼点です。アルゴリズムを評価するとき、FLOP ではなく HBM アクセスバイト数で良し悪しを測る、という見方を持ち込んだのが思想的な核心です。

### 仕組みを詳しく

#### 巨大行列を作らず、ブロックごとに計算する (tiling)
FlashAttention の核心は「N×N の S と P を一度も全体としては作らない」ことです。Q, K, V を小さなブロック(タイル)に分け、SRAM に載るぶんだけを順番に処理します。

```
1. Q を行ブロックに、K, V を列ブロックに分割
2. 各 Q ブロックについて、K, V ブロックを1つずつ SRAM に読み込み
3. その小ブロックぶんだけ S, softmax 寄与, V との積を SRAM 上で計算
4. 結果を「走りながら」足し込む (オンラインで合算)
5. 巨大な S, P は HBM に一度も書かない。出力 O だけを HBM に書く
```
これにより HBM への往復回数が劇的に減り、メモリ使用量も O(N^2) から O(N) (系列長に線形)に下がります。理論解析では、SRAM サイズを M とすると HBM アクセスが O(N^2 d^2 / M) となり、素朴な実装の O(N^2 + N d) より大幅に小さい(d^2/M が 1 より十分小さい現実設定で効く)ことが示されています。

#### online softmax (分割しても正しい softmax)
難所は softmax です。softmax は「行全体の合計で割る」操作なので、本来は行を全部見ないと正規化できません。ブロックに分けると全体が見えません。

FlashAttention は online softmax (Milakov & Gimelshein, 2018 が確立した手法)を使います。ブロックを処理するたびに「これまで見た最大値 m」と「これまでの指数和 l」を保持し、新しいブロックが来たら過去の部分結果をスケールし直して足し込みます。

```
新ブロックの最大値が過去より大きければ、
  過去の累積和 l と累積出力 O を exp(m_old - m_new) 倍して縮め、
  新ブロックぶんを加える
```
最大値を引いてから exp を取るのは数値安定化(オーバーフロー防止)のためです。この補正係数 exp(m_old − m_new) を掛け直す操作(rescale)により、行全体を一度に持たなくても最終的に厳密に同じ softmax 結果が得られます。近似ではない点が決定的に重要です。理論的には Rabe & Staats (2021) が「self-attention は O(n^2) メモリを必要としない」ことを先に示しており、FlashAttention はこれを IO 最適なカーネルとして実装で具現化したものです。

#### 逆伝播 (backward) の工夫
学習では勾配計算が必要ですが、ここで普通なら順伝播で作った P を保存して使い回します。しかし FlashAttention は P を保存しません。代わりに、保存しておいた小さな統計量(各行の最大値 m と正規化定数 l をまとめた logsumexp)から逆伝播時に P を必要なブロックだけ再計算します。これを recomputation(再計算)と呼びます。計算を少し増やす(FLOP は増える)代わりにメモリを大幅に節約し、しかも増えた再計算ぶんが SRAM 上で完結するため、HBM 律速の世界では実時間も速くなる、という一石二鳥です。典型的な計算とメモリのトレードオフが、IO 律速ゆえに「両得」に転じる好例です。

#### before / after まとめ
```
                       普通の attention        FlashAttention
N×N 行列 S, P の保存    HBM に書く (N^2 メモリ)   作らない (SRAM 上で消費)
メモリ使用量           O(N^2)                   O(N) (線形)
HBM 往復               多い (律速の原因)         少ない (O(N^2 d^2/M))
FLOP                   同じ                     同じ (むしろ再計算で微増)
出力の正しさ           -                        まったく同じ (厳密)
速度                   遅い                     数倍速い
```

#### 使い方
専用カーネルなので自分で実装する必要はありません。
```python
# PyTorch 2.x: 条件が合えば内部で FlashAttention 系カーネルを自動選択
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```
PyTorch の SDPA はハードと入力条件(対応 GPU、対応 dtype、head_dim 制約など)が合えば内部で FlashAttention カーネルを自動選択し、合わなければ memory-efficient 版や数式どおりの実装にフォールバックします。`flash-attn` ライブラリを直接呼ぶ方法もあります。

### 手法の系譜と主要論文

- Milakov & Gimelshein, "Online normalizer calculation for softmax" (NVIDIA, 2018): ブロックを逐次処理しながら厳密な softmax を計算する online softmax を確立。FlashAttention の数学的土台。
- Rabe & Staats, "Self-attention Does Not Need O(n^2) Memory" (Google, 2021): attention を線形メモリで計算できることを理論と実装(JAX)で示した先駆け。ただし実速度の最適化よりメモリ削減の理論証明が主眼でした。
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022, スタンフォード): tiling + online softmax + recomputation を IO 最適な単一 GPU カーネルにまとめ上げ、近似なしで実速度を数倍にした決定版。従来研究が FLOP 削減の近似に注力していたのに対し、真のボトルネックは HBM の IO だと見抜いた点が新規性。
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023): GPU 内のスレッドブロック/warp への作業分割を見直し、非行列積演算(rescale など)を減らして行列積の比率を上げ、系列長方向にも並列化(causal でも負荷を均す)。GPU 利用率を大きく改善。
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (NeurIPS 2024, NVIDIA H100 向け): Hopper 世代の非同期実行(warp-specialization で行列積とソフトマックスを別 warp で重ねる、TMA による非同期コピー)と FP8 低精度を活用。
- 対照的に Linformer (Wang et al., 2020) や Performer (Choromanski et al., 2020)、Reformer (Kitaev et al., 2020) は attention を低ランク/カーネル/ハッシュで線形化しますが、これらは答えが近似で精度劣化リスクがあります。FlashAttention の価値は「厳密な答えを保ったまま IO だけ減らす」点で、近似系とは思想が異なります。

### 論文の実験結果 (定量データ)

FlashAttention (NeurIPS 2022) の主要な結果。

- BERT-large (系列長512) の学習を、当時の最速記録(MLPerf 1.1 の NVIDIA 実装)比でおよそ15%短縮(end-to-end の壁時計時間)。
- GPT-2 の学習で、HuggingFace 実装比で最大約3倍、Megatron-LM 実装比で約1.8倍の高速化を報告(系列長512〜1K)。
- 系列長を伸ばせることを活かし、Long-Range Arena(長系列ベンチマーク)で標準 Transformer より速く、近似 attention 系と比べて速度・精度ともに優位を示した。
- メモリが O(N) に線形化するため、同じ GPU で扱える系列長が大幅に伸び、Path-X(系列長16K)・Path-256(系列長64K)など従来 Transformer がメモリ不足で解けなかった長系列タスクで初めて学習が回り、chance(でたらめ)を上回る精度を達成。これは速度向上を超えた質的変化です。
- attention 単体のマイクロベンチで、標準実装比で最大7.6倍(系列長が長いほど比が伸びる)。

数値の意味: 学習3倍速とは、同じ精度のモデルを3分の1の時間・コストで得られることを意味し、大規模学習で極めて大きな差です。さらに「メモリ線形化により長系列が初めて扱える」というのは単なる速度向上を超えて、それまで不可能だった規模の問題を可能にする質的な変化です。

アブレーション的観察: 論文はブロックサイズ(タイルの大きさ)を変えた感度分析を行い、SRAM 容量に合わせた適切なブロックサイズで HBM アクセスが最小化されることを示しています。ブロックが小さすぎると並列度・再利用が落ち、大きすぎると SRAM に載らずあふれます。FlashAttention-2 では、この作業分割をさらに最適化し、A100 で理論ピーク FLOP の50〜70%程度という、attention としては極めて高い GPU 利用率を達成(FlashAttention-1 のおよそ2倍速、標準実装比で最大9倍)。FlashAttention-3 は H100 で FP16 でおよそ1.5〜2.0倍(FA2 比)、FP8 ではほぼ1.2 PFLOP/s 近くに達し、低精度ながら誤差は標準的なベースライン量子化より小さいと報告されています。

### メリット・トレードオフ・限界

メリット
- attention のメモリを O(N^2) から O(N) に削減し、長系列・高解像度を同じ VRAM で扱える。
- HBM 往復を減らし学習・推論を数倍高速化。
- 答えが厳密に普通の attention と同じ(近似ではないので精度劣化なし)。
- PyTorch の SDPA 経由ならほぼ書き換えなしで使える。

トレードオフ・限界
- GPU アーキテクチャ専用カーネルで、対応 GPU(おおむね Ampere 以降、最新版は Hopper 最適化)と精度(fp16/bf16 が主、fp32 は対象外なことが多い)、head_dim の制約(古い版では 128 まで等)がある。
- attention 部分だけを速くする手法で、モデル全体のボトルネックが他(MLP や通信)にあると end-to-end の効果は限定的。
- 任意のマスクや任意の bias(相対位置バイアスなど)を伴う特殊な attention 変種に、カーネルが未対応の場合があり、その際は遅い経路にフォールバックする。
- ライブラリ/CUDA/フレームワークのバージョン組み合わせ依存が強く、環境構築でつまずきやすい(ビルドに時間がかかる)。
- 厳密同一とはいえ低精度演算なので、極端な数値で微小差が出ることはある。
- 研究上の課題として、各世代の新 GPU 機能(非同期、低精度 FP8/FP4)への追随、可変長・ブロックスパース・ツリー構造マスクへの対応拡張が継続テーマ。

### 発展トピック・研究の最前線
- PagedAttention / vLLM: 推論時の KV キャッシュをページ単位で管理し、長系列バッチ推論のメモリ断片化を抑えてスループットを上げる流れ(FlashAttention と相補的)。
- 低精度 attention(FP8/FP4)と量子化推論の組み合わせ。
- 高解像度ビジョンや BEV(鳥瞰図)表現のように系列長が数万規模になる空間的 attention での適用。
- スパース attention / sliding window / Flash Decoding(推論時の系列方向並列)との融合(長文脈 LLM 向け)。
- ハードウェアとアルゴリズムの協調設計(IO-aware の思想を畳み込みや正規化など他の演算へ展開する研究)。

### さらに学ぶための関連トピック
- [torch.compile](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Transformer 時間方向アテンション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/05_multicamera-fusion)
- [鳥瞰図特徴融合（BEVFormer風 空間クロスアテンション）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/05_multicamera-fusion)


## fp16 AMP + GradScaler

### ひとことで言うと
学習を float16 (16ビットの軽い数値形式) で高速化したいとき、float16 は「とても小さい数」をゼロに丸めて消してしまう弱点があります。GradScaler は「勾配を一時的に大きく拡大してゼロに潰れないように守り、重みを更新する直前に元に戻す」道具です。混合精度学習の元祖となる手法で、bfloat16 が使えない古い GPU では今も現役の標準テクニックです。

### 直感的な理解
混合精度の狙いは、計算を16ビットにしてメモリと速度を稼ぐことです ([bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照)。ところが float16 には致命的な弱点があります。表せる数の範囲が狭く、学習中に出る極小の勾配 (例: 1e-7) を「ゼロ」に丸めてしまうのです。勾配がゼロになると重みが動かず学習が止まります。

解決のアイデアは拍子抜けするほど単純です。「小さくてゼロに潰れるなら、潰れる前に大きくしておけばいい」。損失 (loss) に大きな定数を掛けてから逆伝播すると、微分の連鎖律によって計算される勾配もすべて同じ倍率で拡大され、float16 で表せる範囲に収まります。重みを更新する直前に同じ倍率で割り戻せば、元の正しい勾配に戻ります。この拡大・縮小を自動でやるのが GradScaler です。秤に小さすぎて読めない重りを乗せるとき、いったん大きな分銅と束ねて測り、後で分銅ぶんを引く、というイメージです。

### 基礎: 前提となる概念

#### float16 の弱点をおさらい
16ビット形式には float16 (FP16) と bfloat16 (BF16) があり、16ビットを指数部と仮数部にどう配分するかが違います。

- **float16**: 指数部5ビット、仮数部10ビット。表せる最小の正規化数は約 6e-5。これより小さい数はゼロ付近に丸められる。仮数部は10ビットあるので「細かさ」自体は16ビット形式の中では良い。
- **bfloat16**: 指数部8ビット、仮数部7ビット。float32 並みに広い範囲を表せる代わり細かさは粗い。

float16 の問題は指数部が5ビットしかないこと = 表せる範囲 (ダイナミックレンジ) が狭いことです。値の大きさが激しく振れる勾配を扱うとき、この狭さが致命傷になります。

#### アンダーフローとオーバーフロー
- **アンダーフロー (underflow)**: 数が小さすぎて表現できずゼロになる現象。float16 では勾配でこれが起きやすい。学習が静かに止まる原因になる (エラーは出ず、ただ重みが動かなくなる)。
- **オーバーフロー (overflow)**: 数が大きすぎて表現できず無限大 (Inf) になる現象。損失を拡大しすぎると今度はこちらが起きる。Inf が一度でも混じると、その後の計算が NaN (非数) に汚染され学習が壊れる。

GradScaler はこの2つの崖の間 (ゼロにも Inf にもならない範囲) に勾配を収めるよう倍率を自動調整します。

#### 損失スケーリング (loss scaling) の原理
微分の連鎖律より、損失を S 倍してから逆伝播すると、すべてのパラメータの勾配が一律 S 倍になります (`∂(S·L)/∂w = S·∂L/∂w`)。だから損失に S を掛けるだけで全勾配を一律に持ち上げられます。更新直前に勾配を 1/S すれば正しいスケールに戻ります。鍵は「S 倍は全勾配に対して一律」なので、勾配同士の相対的な大小関係 (= 最適化に必要な情報) は一切壊さず、ただ表現可能な範囲へ平行移動させるだけ、という点です。

### 仕組みを詳しく

#### なぜ勾配は小さく、活性化や重みは小さくないのか
float16 の範囲問題が「勾配」で特に深刻なのには理由があります。Micikevicius らが示したのは、典型的なネットワークでは活性化や重みの値は float16 の範囲によく収まる一方、勾配は分布がずっと小さい側 (1e-7 付近) に寄りがちで、その大部分が float16 の最小値 (約 6e-5) を下回ってしまう、という事実です。つまり「重みや活性化はそのまま float16 で大丈夫だが、勾配だけ範囲外に落ちる」。だから損失スケーリングは勾配にだけ効けばよく、forward は素の float16 のままでよい、という設計が成り立ちます。

#### PyTorch での典型コード
float16 を使う場合の典型的な書き方です。

```python
scaler = torch.amp.GradScaler()            # スケーラを作る (初期倍率 S は典型的に 65536)
for batch in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(batch)
        loss = loss_fn(output, target)

    scaler.scale(loss).backward()   # 1) loss を S 倍してから逆伝播
    scaler.step(optimizer)          # 2) 勾配を 1/S に戻し、Inf/NaN がなければ更新
    scaler.update()                 # 3) 倍率 S を自動調整
```

ステップを分解します。

1. `scaler.scale(loss)`: loss に現在の倍率 S を掛ける。`backward()` するとすべての勾配が S 倍されて計算され、小さい勾配もゼロに潰れずに float16 で表現される。
2. `scaler.step(optimizer)`: 重みを更新する前に勾配を内部で 1/S して元のスケールに戻す。このとき勾配に Inf や NaN が含まれていないかチェックする。含まれていたら (倍率が大きすぎてオーバーフローした証拠)、そのステップの更新をスキップする (壊れた勾配で重みを動かさない安全装置)。
3. `scaler.update()`: 直近で Inf/NaN が出なかったなら倍率 S を少しずつ大きくし (例: 一定回数連続で成功したら2倍)、出たなら即座に S を小さくする (例: 半分)。これで「ゼロに潰れない程度に大きく、しかしオーバーフローしない程度に」自動調整される。

#### 数値例
倍率 S = 1024 とします。ある勾配が本来 5e-8 (float16 ではゼロに潰れる) だと、`5e-8 × 1024 = 5.12e-5`。これは float16 の最小値 6e-5 にまだ少し足りませんが、S = 65536 なら `5e-8 × 65536 ≈ 3.3e-3` となり余裕で表せます。更新直前に `÷ 65536` で `5e-8` に戻すので、最終的な重みの更新は正しい値になります。逆に S が大きすぎて、どこかの勾配が 65504 (float16 の最大) を超えると Inf になり、そのステップはスキップして S を下げます。

#### 動的損失スケーリング (dynamic loss scaling)
固定倍率だと「大きすぎてオーバーフロー」か「小さすぎてアンダーフロー」のどちらかに当たりがちです。しかも勾配の大きさは学習の進行とともに変わる (序盤と終盤で何桁も違うことがある) ので、一つの固定値で全期間をまかなうのは困難です。GradScaler は上記ステップ3の仕組みで倍率を実行時に自動で上げ下げします。これにより人間が倍率をタスクごと・期間ごとに手調整する必要がなくなりました。学習序盤は倍率を探る過程で数ステップがスキップされることがありますが、すぐ安定します。

#### float32 マスターコピーとの併用
GradScaler だけでは不十分で、混合精度のもう一つの柱「重みの float32 マスターコピー」と併用します。forward/backward は float16、重みの保持と更新は float32 で行うことで、微小な更新が大きな重みに足したとき丸めで消えるのを防ぎます。さらに行列積などは float16 入力でも内部の累積 (たくさんの積を足し合わせる工程) を float32 で行うことで、長い和での誤差蓄積を抑えます。bf16 でも float32 マスターコピーと float32 累積は同じく使いますが、bf16 は範囲が広いので GradScaler の部分だけが不要になる、という関係です ([bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。

### 手法の系譜と主要論文
- **2017/2018年 — 元祖**: Micikevicius, Narang, Alben ら "Mixed Precision Training" (ICLR 2018、NVIDIA/Baidu、arXiv:1710.03740)。混合精度学習の枠組みを確立した基礎論文。3つの技術を提案: (1) 重みの float32 マスターコピー、(2) 損失スケーリング、(3) 行列積などの float32 累積。GradScaler はこの (2) の自動化実装です。
- **2018年 — 整数演算による別系統**: Das ら "Mixed Precision Training of CNNs using Integer Operations" (ICLR 2018) は INT16 を使う混合精度を示し、低精度学習の方向性が浮動小数点だけではないことを示しました。後の INT8 推論・量子化学習につながる流れです。
- **実装の普及**: NVIDIA Apex ライブラリ (`amp` モジュール) がこの手法を広め、後に PyTorch 本体に `torch.cuda.amp` / `torch.amp` として取り込まれ、動的損失スケーリングが標準機能になりました。TensorFlow も `mixed_float16` ポリシーで同等機能を提供しています。
- **2019年 — bfloat16 による代替**: Kalamkar ら (arXiv:1905.12322) の bfloat16 は、指数部を float32 並みに広げることで損失スケーリングそのものを不要にしました。つまり fp16+GradScaler の「次の世代」が bf16 で、ハードが対応していれば bf16 のほうが単純で安定する、という流れになりました。

### 論文の実験結果 (定量データ)
- **損失スケーリングの必要性 (Micikevicius et al., ICLR 2018)**: 物体検出 (SSD) などの実験で、損失スケーリングを「使わない」と勾配のヒストグラムの大部分が float16 の表現可能範囲を下回り (アンダーフロー)、学習が発散・劣化することを示しました。スケーリングを入れると float32 と同等の精度を回復。これは「スケーリングを抜くと壊れる」という明確なアブレーションで、この手法の必要性を定量的に裏づけています。論文中の勾配ヒストグラムは、スケーリング前は値の塊が float16 の最小値より左 (= ゼロに潰れる領域) に集まっていることを視覚的に示しており、なぜ単純な平行移動 (S 倍) で救えるかが直感的に分かります。
- **精度同等性 (Micikevicius et al., ICLR 2018)**: ImageNet 分類 (AlexNet, VGG, GoogLeNet, ResNet, Inception)、物体検出、音声認識、機械翻訳、言語モデル、GAN など多様なタスクで、float16 混合精度が float32 と同等の最終精度を達成しつつ、メモリを約半減・Tensor Core 対応 GPU で高速化したことを報告しています。意味: 「半分のビット幅で同じ賢さ・速い・省メモリ」。
- **bf16 との比較 (Kalamkar et al., 2019)**: 同じタスク群で、bfloat16 は損失スケーリングなしで float32 同等精度を達成。つまり fp16 で必須だった調整が bf16 では消える。これが「新しい GPU なら bf16 を選ぶ」根拠の定量的裏づけです。
- **動的スケーリングのオーバーヘッド (一般的観察)**: 学習序盤に倍率探索で数ステップがスキップされますが、全体の学習時間に対する影響は小さいです。一方で固定倍率を手で当てると、タスクが変わるたびに再調整が要りコストになる — これが動的スケーリングが標準になった理由です。

数値の意味: 「精度が float32 と同等」かつ「メモリ半減・高速」であれば、追加コストはスケーリングの実装と序盤のわずかなスキップだけ。古い GPU ではこの取引が依然として有利です。

### メリット・トレードオフ・限界
メリット:
- bfloat16 非対応の古い GPU (Pascal/Turing 世代の一部、T4 など) でも、float16 の高速化・メモリ削減の恩恵を受けられる。
- 動的損失スケーリングにより倍率の手動調整がほぼ不要。
- float16 は仮数部10ビットと、bfloat16 (7ビット) より細かさは上。範囲さえ確保できれば数値表現は細かい。一部の用途では、この細かさの差が効くこともある。

トレードオフ・限界:
- GradScaler という追加の仕組みが必要で、コードが bf16 より複雑になる (scale → step → update の三段)。勾配クリッピングなど他の処理と組み合わせるときは `scaler.unscale_()` を挟むなど順序に注意が要る。
- 倍率が大きすぎる初期段階では Inf/NaN が出てステップがスキップされ、学習序盤がわずかにもったいない。
- それでも float16 のダイナミックレンジの狭さに起因する数値不安定 (特に勾配の極端な分布、勾配爆発を起こしやすいモデル、極端に大きい活性化を出す層) は完全には消えない。学習率や正規化との相互作用で発散することがある。
- 新しい GPU が手に入るなら bfloat16 のほうが単純で安定するため、新規に fp16+GradScaler を選ぶ理由は薄い。位置づけは「ハードが古いとき/メモリ表現の細かさが効くときの代替手段」です。
- 未解決の課題というより、実務的には「いつ fp16 を選び、いつ bf16/fp8 に移るか」の判断と、量子化推論との一貫性をどう保つかが論点です。

### 発展トピック・研究の最前線
- **bf16 / fp8 への移行**: 新世代ハードでは bf16 が既定、さらに FP8 (E4M3/E5M2) が登場し、損失スケーリングは「テンソルごとの動的スケール」という形で姿を変えて残っています (低ビットほどスケーリングが再び重要になる)。
- **per-tensor / per-channel スケーリング**: 単一の損失スケールではなく、テンソルやチャネルごとにスケールを持たせて範囲を最適化する手法。量子化学習 (QAT) とも接続します。
- **確率的丸め (stochastic rounding)**: 低精度で更新が消える問題に対し、丸め方向を確率的に決めて期待値として正しい更新を残す手法。
- **数値安定性の解析**: 低精度下での勾配分布・損失地形の理論的理解、混合精度が収束に与える影響の解析。
- **最適化器状態の低精度化**: 8-bit Adam など、勾配だけでなく最適化器の内部状態も低精度化してメモリを削る方向。

### さらに学ぶための関連トピック
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Gradient Accumulation

### ひとことで言うと
Gradient Accumulation（勾配の貯め込み）とは、小さなバッチを何回かに分けて計算し、その勾配（重みをどちら向きにどれだけ動かすかの情報）を足し合わせてから、まとめて1回だけ重みを更新する手法です。これによって、GPU のメモリ（VRAM）には収まらない「大きなバッチ」で学習したのと同じ効果を、メモリを増やさずに得られます。GPU を増やさずに実効バッチサイズを拡大する、ほぼ無料の手段です。

### 直感的な理解
大きな鍋でカレーを一度に作りたいけれど、手元の鍋が小さい状況を想像してください。小さい鍋で4人分ずつ4回作り、最後に1つの大鍋に全部移して味を整えれば、結果として16人分の大鍋で作ったのとほぼ同じになります。Gradient Accumulation はこれと同じで、小さいバッチを何回かに分けて計算し、勾配だけを足し合わせてから一度に「味を整える」（重みを更新する）のです。

なぜ大きなバッチが欲しいかというと、勾配のノイズが減るからです。少数サンプルから計算した「重みを動かす向き」は偶然のブレを含みますが、たくさんのサンプルで平均すると「全データを使ったときの本当の向き」に近づき、更新が安定します。けれどバッチを大きくするほど VRAM を食い、高解像度の入力や大きなモデルではすぐにメモリの壁（Out Of Memory, OOM）にぶつかります。GPU を買い増せば解決しますが高価です。Gradient Accumulation は手元の1枚の GPU のまま、メモリを増やさずに実効バッチを大きくします。

### 基礎: 前提となる概念

#### バッチサイズとは
学習では画像などのサンプルを一度に何枚かまとめて GPU に入れ、その平均的な誤差から重みを更新します。この「一度に入れる枚数」がバッチサイズです。バッチサイズ8なら8枚を同時に処理して1回更新します。バッチを大きくすると、(1) 勾配のノイズが減って学習が安定、(2) GPU の並列計算を効率よく使える、(3) BatchNorm などバッチ内統計を使う仕組みが安定する、という利点があります。一方で VRAM 消費が増えます。VRAM を主に食うのは、逆伝播のために順伝播時の中間出力（活性値）を保持する分で、これはバッチサイズにほぼ比例して増えます。

#### 勾配(gradient)と最適化ステップ
- 順伝播(forward): 入力をモデルに通して予測を出す。
- 損失(loss): 予測と正解のずれを数値化したもの。
- 逆伝播(backward): loss を各重みで微分し、「その重みをどちらにどれだけ動かせば loss が減るか」を表す勾配を計算する。
- 更新(optimizer step): 勾配に従って重みを実際に動かす。

PyTorch では `loss.backward()` を呼ぶと、各重みの `.grad` 属性に勾配が「加算」されます（上書きではない）。この加算性が Gradient Accumulation の鍵です。`optimizer.zero_grad()` を呼ぶまで勾配は貯まり続けます。

#### micro-batch と effective batch
- micro-batch（マイクロバッチ）: 実際に GPU に1回で載せる小さなバッチ（例 4枚）。
- effective batch（実効バッチ）: 1回の重み更新が反映するサンプル総数（micro-batch × 貯め込み回数）。

Gradient Accumulation は「micro-batch は小さく保ちつつ、effective batch を大きく見せる」技術です。

### 仕組みを詳しく

#### 普通の学習（accumulation なし）
```python
for batch in loader:              # batch は例えば4枚
    optimizer.zero_grad()         # 勾配をゼロにリセット
    loss = compute_loss(model, batch)
    loss.backward()               # 逆伝播 → 各重みの .grad を計算
    optimizer.step()              # 勾配に従って重みを更新
```
ここで `optimizer.step()` が「実際に重みを動かす」操作で、バッチ4枚ごとに1回更新されます。

#### Gradient Accumulation あり
実効バッチ16枚で学習したいが、メモリ的には4枚しか載らない状況を考えます。4枚ずつ4回計算して勾配を貯め、4回目にまとめて更新します。
```python
accum_steps = 4                   # 4回ぶん貯める → 実効バッチ 4*4 = 16
optimizer.zero_grad()
for i, batch in enumerate(loader):     # batch は4枚
    loss = compute_loss(model, batch)
    loss = loss / accum_steps          # ← 重要: 平均にするため割る
    loss.backward()                    # 勾配が「足し算」で貯まる
    if (i + 1) % accum_steps == 0:
        optimizer.step()               # 4回に1回だけ更新
        optimizer.zero_grad()          # 貯めた勾配をリセット
```

#### なぜこれで「大きいバッチ」と同じになるのか
loss が各サンプルの平均で、勾配が線形演算であることから、「4枚ずつ4回の勾配の和」は「16枚を一度に計算した勾配の和」と数学的に等しくなります。式で書くと、N サンプルのバッチの平均損失の勾配は

```
g = (1/N) Σ_{k=1..N} ∇ loss_k
```

であり（Σ は総和、∇ は微分=勾配、loss_k は k 番目サンプルの損失）、これを M 個の micro-batch（各サイズ N/M）に分け、各 micro-batch の損失を M で割って backward すると、`.grad` に貯まる総和は上式の g に一致します。だから各 loss を `accum_steps` で割っておけば、16枚を1回処理したときとほぼ同じ更新になります。割り忘れると勾配が M 倍になり、実質的に学習率が M 倍になって発散しやすいので、これがよくあるバグです。

#### 数値例
- micro-batch: 4 枚 / accum_steps: 4 / effective batch: 16
- VRAM 使用量: 4枚ぶんのまま（ほぼ増えない。貯めるのは勾配で、勾配は重みと同じサイズなので一定）
- 1更新あたりの計算時間: 4枚×4回ぶんなので約4倍

メモリは増えませんが、その代わり1更新にかかる時間が伸びます。これがトレードオフの核心です。「メモリと時間のトレードオフ」と覚えるとよいです。総サンプル処理数は同じなので1エポックの計算量は変わりませんが、勾配を実際に使う更新回数が1/4に減る点が普通の学習との違いです。

#### BatchNorm との注意
完全に同一とは言えない部分があります。BatchNorm（バッチ正規化）は「そのバッチ内の平均・分散」で各特徴を正規化するため、4枚での統計と16枚での統計は異なります。したがって BatchNorm を含むモデルでは、accumulation の結果は本物の16枚バッチと厳密には一致しません。一方 LayerNorm（サンプル単位で正規化）を使う Transformer 系ではこの問題は起きません。ConvNeXt 系も LayerNorm を採用しており、この差は小さくなります。BatchNorm モデルで厳密さが要るときは、SyncBatchNorm（分散時にバッチ統計を集約）や GroupNorm への置き換えで補う必要があります。

#### 分散学習との関係
複数 GPU の DistributedDataParallel（DDP）では、各 GPU が異なる micro-batch を処理し、backward 時に勾配が自動で全 GPU 平均されます。accumulation 中の途中ステップでは毎回通信すると無駄なので、`model.no_sync()` コンテキストで通信を抑え、最後の step でだけ同期するのが定石です。これを忘れると accum_steps 回ぶん余分に勾配を all-reduce 通信し、通信律速の環境では学習が大きく遅くなります。accumulation と DDP を併用すれば、実効バッチ = micro-batch × GPU 台数 × accum_steps とさらに大きくできます。

### 手法の系譜と主要論文

Gradient Accumulation 自体は特定の論文の「発明」というより、大規模分散学習の現場で自然に使われてきた標準技法です。その価値を支える研究の流れは次の通りです。

- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"（Facebook AI Research, 2017, arXiv:1706.02677）。提案: バッチサイズを256から8192まで大きくしても精度を保つレシピ。学習率をバッチサイズに比例させる Linear Scaling Rule と、最初に学習率をゆるやかに上げる warmup（[LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)）を組み合わせた。動機: バッチを大きくすると1更新が大胆になり初期に発散しやすいため。効果: 大バッチでも精度劣化なく学習時間を大幅短縮。トレードオフ: 学習率の調整が必須で、無調整に大バッチへ移すと精度が落ちる。この研究が「大バッチは有効」という前提を確立し、その大バッチを安価なハードで再現する手段として accumulation の価値が高まった。
- You et al., LARS（"Large Batch Training of Convolutional Networks", 2017, arXiv:1708.03888）/ LAMB（"Large Batch Optimization for Deep Learning: Training BERT in 76 minutes", 2019, arXiv:1904.00962）。提案: バッチを数万まで増やすため、層ごとに学習率を適応させる最適化手法。動機: 単純な線形スケーリングだけでは超大バッチで破綻するため。効果: BERT の事前学習を約76分まで短縮。トレードオフ: 実装と調整が複雑。
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"（ICLR 2017, arXiv:1609.04836）。報告: 大バッチは sharp minima（鋭い谷）に落ちやすく、汎化が悪化しうる。つまり accumulation で無限にバッチを増やせばよいわけではなく、適切な範囲がある。これに対し前述の warmup や適切な学習率調整で汎化ギャップを縮められることも後続研究で示されている。
- McCandlish et al., "An Empirical Model of Large-Batch Training"（OpenAI, 2018, arXiv:1812.06162）。提案: 勾配ノイズスケール（gradient noise scale）という指標で「どこまでバッチを大きくすると効率が落ちるか（critical batch size, 臨界バッチサイズ）」を予測できる。臨界バッチサイズより小さい領域ではバッチを2倍にすると必要更新回数がほぼ半分になり時間効率が良いが、それを超えると更新回数があまり減らず計算が無駄になる。実務でバッチ・accum_steps を決める理論的指針になる。

### 論文の実験結果(定量データ)

評価で測られる指標と、その意味を示します。

- 検証精度（Top-1 accuracy など。テスト画像で最も確率の高い予測が正解と一致した割合）: バッチを大きくしても精度が保たれるかが論点。Goyal et al. は ImageNet で ResNet-50 を、バッチ256のベースラインと同等の Top-1 精度（約76%台）を、バッチ8192でも維持できることを示した。学習率を線形にスケールし warmup を入れたのが鍵で、これらを抜くと大バッチで精度が数ポイント落ちる（アブレーション）。さらにバッチ16384以上に増やすと、レシピを守っても精度が目に見えて低下し始め、臨界バッチサイズの存在を裏づけた。
- 壁時計時間（wall-clock time）: 大バッチ + 多 GPU で ResNet-50 ImageNet 学習を約1時間（256 GPU 使用）に短縮。LAMB は BERT 事前学習を、通常の数日規模から約76分（1024 TPU 相当の大規模並列）に短縮。これらは「同じ精度に到達するまでの時間」を大幅に縮めたという定量成果。
- generalization gap（汎化ギャップ。訓練精度とテスト精度の差）: Keskar et al. は、大バッチ学習が小バッチ学習に比べてテスト精度で数ポイント劣化しうること、そしてその解が損失地形の「鋭い谷」に位置することをヘッシアン（損失の2階微分）の固有値の大きさで定量化した。鋭い谷は重みの微小変化で損失が急増するため、訓練分布とテスト分布のわずかなズレに弱い。

アブレーション的な要点: (1) 大バッチ化に伴い学習率を線形スケールしないと精度が落ちる、(2) warmup を抜くと初期に発散する、(3) critical batch size を超えてバッチを増やしても、必要な総更新回数があまり減らず計算効率の改善が頭打ちになる。Gradient Accumulation はこれらの「大バッチの利点と注意点」をそのまま継承します。なお accumulation は「実効バッチを大きくする」だけで臨界バッチサイズ自体を動かさないため、効率改善には自ずと上限がある点を理解しておくのが重要です。

### メリット・トレードオフ・限界

メリット
- メモリを増やさずに実効バッチを拡大できる（GPU 増設不要）。
- 勾配のノイズが減り、学習が安定しやすい。BatchNorm 以外では本物の大バッチとほぼ等価。
- 実装が数行で済み、既存ループにほぼそのまま足せる。
- 分散学習（DDP）と併用でき、実効バッチをさらに伸ばせる。

トレードオフ・限界
- 学習が遅くなる。accum_steps 回ぶんの forward/backward をしてから1回更新するので、更新あたりの計算時間が増える（壁時計時間が伸びる）。
- loss を accum_steps で割り忘れると、実質学習率が accum_steps 倍になり発散（頻出バグ）。
- BatchNorm を使うモデルでは本物の大バッチと厳密に一致しない。
- 大バッチ化に伴い学習率の再調整（warmup, linear scaling）が必要になることが多い。
- バッチを大きくしすぎると汎化がむしろ落ちる場合がある（sharp minima 問題、critical batch size を超えると効率も頭打ち）。

未解決・研究課題: 「最適な effective batch をデータ・モデルから事前に決める一般理論」「大バッチでも sharp minima を避ける最適化」は今も研究が続く領域です。勾配ノイズスケールはその一つの答えですが、学習の進行中に臨界バッチサイズ自体が変化する（後半ほど大きくなる）ことが観察されており、万能ではありません。

### 発展トピック・研究の最前線
- Gradient checkpointing（活性値の再計算）との併用: accumulation はバッチ次元のメモリ節約、checkpointing は深さ方向のメモリ節約（順伝播の中間出力を保存せず逆伝播時に再計算する）で、組み合わせるとさらに大きなモデル・バッチを1枚に載せられる。
- ZeRO / FSDP: オプティマイザ状態や勾配を GPU 間で分割保持し、巨大モデルを学習する手法。accumulation と組み合わせて超大規模学習で使われる。
- 適応的 accumulation: 学習の進行に応じて accum_steps を変える、勾配ノイズスケールを測って自動調整する研究。学習後半に臨界バッチサイズが上がる性質に合わせてバッチを段階的に増やす batch ramp-up も使われる。
- 大バッチ向け最適化（LARS/LAMB の後継、Adafactor, Sophia など）: 大きな実効バッチでも安定・高速に収束する optimizer の探索が続いている。

### さらに学ぶための関連トピック
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [WebDataset (Shard Packing)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/13_datasets-dataprocessing)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)


## KLダイバージェンス損失

### ひとことで言うと
「2つの確率分布がどれだけ違うか」を1つの数字で測る道具です。機械学習では、モデルの出した分布を「お手本の分布」や「理想の分布」に近づけたいときに、この値を小さくするよう学習します。値は常に0以上で、2つの分布が完全に一致したときだけ0になります。情報理論を起源としながら、現代の深層学習(生成モデル・蒸留・強化学習)の至る所で使われる基礎部品です。

### 直感的な理解
天気予報を例にします。明日の天気を予報官Aが「晴れ60%・曇り30%・雨10%」と予報し、別の予報官Bが「晴れ20%・曇り30%・雨50%」と予報したとします。この2人の予報は「どれくらい食い違っているか」を数字にしたい。単純に各項目を引き算するだけでは、確率分布の特殊な性質(全部足すと1、マイナスにならない)をうまく扱えません。

KLダイバージェンスは、この食い違いを情報理論的に正しく測る指標です。直感的には「真の分布 P で物事が起きるのに、間違った分布 Q を信じていたとき、平均してどれだけ余分に驚く(=情報を失う)か」を表します。Q が P と一致していれば驚きはゼロ、ズレているほど大きな驚きになります。

別の言い方をすると、KL は「符号化のムダ」です。本当は P に従ってデータが来るのに、Q に最適化された符号(短いビット列)を割り当ててしまうと、平均してビット数が余分にかかります。その余分なビット数こそ KL です。

なぜこれが機械学習で重要かというと、「分布Aを分布Bに近づける」という操作が至る所で必要だからです。
- VAE(変分オートエンコーダ、画像などを圧縮して再生成するモデル)で、内部表現を「きれいな正規分布」に近づけたいとき。
- 知識蒸留(大きな教師モデルの予測分布を小さな生徒モデルに真似させる)とき。
- 強化学習で「新しい方策が古い方策から離れすぎないよう」制約するとき。
- 大規模言語モデルの人間フィードバック学習(RLHF)で、元のモデルから暴走しないよう縛るとき。

これらすべてで近さの指標として KL が使われます。

### 基礎: 前提となる概念

確率分布 (probability distribution) とは「どの結果がどれくらい起こりやすいか」を表す割合の集まりです。例えば信号機の色を当てるモデルが「赤70%・黄10%・緑20%」と出力したら、これが1つの離散確率分布です。性質として、各値は0以上で、全部足すと1になります。

対数 (log) も前提です。KL の式には log が出てきます。対数は「掛け算を足し算に変える」道具で、情報理論では「ある事象が起きたときの驚きの量(情報量)」を −log(確率) で測ります。起こりにくい(確率が小さい)ことほど log が大きな負になり、驚きが大きいと解釈します。log の底が2ならビット、自然対数(底 e)なら nat という単位になります。

エントロピー (entropy) は分布の「不確かさ」の平均で、H(P) = −Σ P(i) log P(i) と書きます。クロスエントロピー (cross-entropy) は「真の分布 P を、別の分布 Q だと思って符号化したときの平均符号長」で、H(P,Q) = −Σ P(i) log Q(i) です。実は KL はこの2つの差です。

```
KL(P || Q) = H(P, Q) - H(P) = クロスエントロピー − エントロピー
```

つまり KL は「Q を信じたことで余分にかかったコスト」を表します。分類で使うクロスエントロピー損失は、真の分布 P(エントロピー H(P) は固定)に対して KL を最小化することと等価なので、KL と密接につながっています([クロスエントロピー損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception))。言い換えると、毎日使っている分類の損失は実は KL 最小化の特殊ケースです。

### 仕組みを詳しく

2つの離散確率分布 P(真の分布、お手本)と Q(モデルの分布、近似)があるとき、KLダイバージェンスは次で定義されます。

```
KL(P || Q) = Σ_i  P(i) * log( P(i) / Q(i) )
```

Σ(シグマ)は「全部の結果 i について足す」、log は対数です。直感的には「P が確率を置いている場所で、Q がどれくらいズレた確率を置いているか」を P で重み付けして合計しています。重みが P であることが後で効いてきます(非対称性の理由)。

数値例で計算します。3択問題で、log は自然対数とします。

```
真の分布 P = [0.5, 0.3, 0.2]
モデル Q1  = [0.5, 0.3, 0.2]   (Pと完全一致)
モデル Q2  = [0.2, 0.3, 0.5]   (AとCを取り違えている)
```

Q1 では P(i)/Q(i) がすべて1で log(1)=0 なので KL = 0。「完全一致のとき KL はゼロ」という大事な性質です。

Q2 を計算します。

```
A項: 0.5 * log(0.5/0.2) = 0.5 * log(2.5)  = 0.5 *  0.916 =  0.458
B項: 0.3 * log(0.3/0.3) = 0.3 * log(1)    = 0
C項: 0.2 * log(0.2/0.5) = 0.2 * log(0.4)  = 0.2 * (-0.916) = -0.183
KL  = 0.458 + 0 - 0.183 = 0.275
```

KL = 0.275 > 0 で、ズレが正の値で表現されます。KL は常に0以上で、一致時だけ0(ギブスの不等式)。個々の項はマイナスになり得ますが(C項)、全体の合計は必ず非負になります。

非対称性(重要)。KL(P||Q) と KL(Q||P) は一般に異なります。だから KL は「距離(distance)」ではなく「ダイバージェンス(divergence、隔たり)」と呼ばれます。距離なら A→B と B→A は同じはずです。この非対称性は実用上の意味を持ちます。

```
KL(P || Q) を最小化(forward KL):  P が確率を置く所を Q は必ずカバーしないと罰される
                                  → mode-covering(取りこぼしを嫌い、全部を薄く覆う)
KL(Q || P) を最小化(reverse KL):  Q が確率を置く所だけ P と合えばよい
                                  → mode-seeking(一つの山に集中、他の山は無視しうる)
```

直感的に言うと、forward KL は「P で起こり得ることを Q が見落とすと激しく罰される(P(i) が大で Q(i) が小だと log が爆発)」ため、安全側に全部を覆おうとします。reverse KL は「Q が確率を置いた場所だけ気にする」ため、複数の山がある P に対して1つの山に潜り込みがちです。VAE や変分推論では reverse KL を、最尤推定や蒸留では forward KL 的な向きを使うことが多く、向きの選択が学習挙動を左右します。

#### VAE での具体的な使われ方

VAE では、エンコーダ(入力を圧縮する部分)が出力する潜在分布 Q(z|x) を、標準正規分布 N(0,1)(平均0・分散1の釣鐘型)に近づけます。両方が正規分布のときは KL に閉じた式(解析解)があり、平均 μ と分散 σ² から直接計算できます。

```python
# Q = N(mu, sigma^2), P = N(0, 1) のときの KL
# mu, logvar はどちらも形状 [B, latent_dim]
kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)  # 形状 [B]
kl_loss = kl.mean()
```

`logvar` は分散の対数 log σ² です。対数で持つのは、分散が必ず正になることを保証し、数値的に安定させるためです。この式は μ が0から離れるほど、σ² が1から離れるほど KL が大きくなります。VAE 全体の損失は

```
total_loss = reconstruction_loss + beta * kl_loss
```

で、`reconstruction_loss` は「圧縮→復元した画像が元と似ているか」、`kl_loss` は「潜在空間がきれいな正規分布になっているか」を見ます。`beta` でバランスを取ります(beta-VAE)。KL 項があることで潜在空間に「穴」ができず連続になり、ランダムにサンプリングして新規データを生成できるようになります。

### 手法の系譜と主要論文

- 原典: Kullback & Leibler, "On Information and Sufficiency" (Annals of Mathematical Statistics, 1951)。統計的推定や情報量の文脈で「ある分布を別の分布で近似したときに失う情報量」を測る量として導入されました。情報理論の親である Shannon (1948) のエントロピーの直後に位置づけられる古典です。

- VAE: Kingma & Welling, "Auto-Encoding Variational Bayes" (ICLR 2014, arXiv:1312.6114)。生成モデルを勾配降下で学習可能にするため、変分下限(ELBO = Evidence Lower BOund)という目的関数を導きました。その中に潜在分布を事前分布(標準正規)に近づける KL 項が現れます。再パラメータ化トリック(z = μ + σ·ε, ε~N(0,1))により KL 項を含む目的関数を微分可能にしたのが核心です。狙いは潜在空間を連続でなめらかにし、そこからサンプリングして新規データを生成することでした。

- beta-VAE: Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (ICLR 2017)。KL 項に重み β を掛けて強めると、潜在変数の各次元が独立で解釈しやすい表現(disentanglement、もつれ解き)になることを示しました。β が情報のボトルネックを締める強さを直接制御します。

- TRPO/PPO: Schulman et al., "Trust Region Policy Optimization" (ICML 2015) と "Proximal Policy Optimization" (arXiv:1707.06347, 2017)。TRPO は新方策と旧方策の KL を一定以下に制約する信頼領域(trust region)で方策更新を安定化しました。PPO はこれをクリッピングや KL ペナルティで近似実装し、実装が簡単で安定なため強化学習の事実上の標準になりました。

- 蒸留: Hinton et al. (2015) は温度付きソフトマックス出力間の KL を蒸留損失に使いました([知識蒸留損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。GAN の系譜では、Jensen-Shannon ダイバージェンス(対称化した KL)の最小化として元祖 GAN が解釈され、その不安定さを Wasserstein 距離で置き換えた WGAN へつながります。

### 論文の実験結果(定量データ)

- VAE の評価指標は対数尤度(log-likelihood、データをどれだけうまく説明できるか、高いほど良い)や、画像なら下限である負の ELBO(nats や bits/dim 単位)です。Kingma & Welling は MNIST(手書き数字)・Frey Face で、変分下限を勾配降下で直接最適化でき、従来のサンプリングベース推論(Monte Carlo EM など)より高速かつ安定に良い尤度に到達することを示しました。

- posterior collapse(事後分布の崩壊)という有名な失敗が定量的に観測されます。KL 項が強すぎたりデコーダが強力すぎると、KL 項が0付近に張り付き(潜在変数が無視される)、生成は再構成損失だけで進むようになります。指標としては「KL 項の値がほぼ0」「潜在次元の活性度(active units、実際に使われている次元数)が激減」で検出されます。対策として KL annealing(β を0から徐々に上げる)や free bits(各次元の KL に下限を設ける)が提案され、活性次元数を回復させる効果が報告されています。

- beta-VAE のアブレーション。β を1から上げていくと、disentanglement の指標(各潜在次元が単一の生成要因に対応する度合い)は改善する一方、再構成誤差は悪化します。論文では合成データ(dSprites、位置・大きさ・向きを既知に作った図形データ)で β を上げるほど位置・大きさ・向きといった要因が別々の次元に分離する一方、画像の細部がぼやけるトレードオフが定量的に示されています。β はおおよそ 4〜250 の範囲で探索され、タスクごとに最良点が異なります。

- TRPO/PPO の効果。連続制御ベンチマーク(MuJoCo)や Atari で、KL 制約を入れない素朴な方策勾配に比べ、学習が崩壊しにくく最終報酬が高いことが報告されています。KL 制約は「1回の更新で方策が壊れるほど大きく動くこと」を防ぐ安全装置として働きます。PPO は KL を直接制約する代わりに確率比をクリップする近似で、TRPO に匹敵する性能を大幅に少ない計算で達成しました。

- 指標の意味の補足。KL が大きいとは「2つの分布が情報的に大きく食い違う」こと。蒸留では KL が小さいほど生徒が教師をよく真似ており、RL では更新前後の KL が大きいほど方策が急変したことを意味します。RLHF では参照モデルからの KL が大きいほど「元の言語能力から逸脱した」サインで、報酬ハッキング(報酬モデルの穴を突く奇妙な出力)の予兆になります。

### メリット・トレードオフ・限界

メリット
- 確率分布同士の「ずれ」を情報理論的に正しく測れる(符号化長という明確な意味を持つ)。
- 正規分布同士など多くのケースで微分可能な閉じた式があり、勾配降下で直接最適化できる。
- 蒸留・VAE・変分推論・RL の信頼領域制約など用途が極めて広く、理論との接続も豊富。

トレードオフ・限界
- 非対称(KL(P||Q) ≠ KL(Q||P))なので、向きの選択を間違えると意図しない学習挙動(mode-covering vs mode-seeking)になる。
- Q(i) が0に近い所で P(i) が正だと log(P/Q) が発散しうる(数値不安定)。実装では log_softmax 経由で計算する、台(support、確率が正である範囲)を共有させる等の配慮が要る。
- 重みのチューニングが必要で、強すぎると posterior collapse などの副作用が出る。
- 分布の台が重ならないと無限大や勾配消失になり、距離としての滑らかな尺度にならない。この弱点を埋めるため、対称な Jensen-Shannon ダイバージェンスや、台が離れていても有限で滑らかな Wasserstein 距離(最適輸送)が代替として使われる。GAN の学習安定化(WGAN, Arjovsky et al. 2017)はこの問題意識から生まれました。

### 発展トピック・研究の最前線

- 一般化ダイバージェンス。KL は α-ダイバージェンスや f-ダイバージェンスという広い族の特殊ケースです。Rényi ダイバージェンスや χ²(カイ二乗)ダイバージェンスを使った変分推論で、mode-covering と mode-seeking のバランスを連続的に調整する研究があります。
- 最適輸送(Wasserstein 距離)。KL の発散問題を回避し、台が重ならない分布間でも意味のある勾配を与えます。生成モデルや分布マッチング、ドメイン適応で KL の代替・補完として重要です。
- レート歪み視点。β-VAE の β は情報ボトルネック(レート歪み理論)の Lagrange 乗数として解釈でき、潜在変数が運ぶ情報量(レート)と再構成品質(歪み)のトレードオフを制御します。この視点から VAE を設計する研究が進んでいます。
- RLHF/方策制約。大規模言語モデルの人間フィードバック強化学習でも、参照方策からの KL ペナルティで暴走を防ぐのが標準で、KL 係数の選び方が出力品質と多様性を左右します。DPO(Direct Preference Optimization)はこの KL 制約付き最適化を、明示的な報酬モデルなしに閉じた形で解く新しい流れです。

### さらに学ぶための関連トピック
- [知識蒸留損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [コサイン類似度損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [クロスエントロピー損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [変分オートエンコーダ(VAE)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/03_diffusion-flow-generative)


## 知識蒸留損失

### ひとことで言うと
賢いけれど重い「教師モデル」の答え方を、軽くて速い「生徒モデル」に真似させる学習方法です。教師が出す「自信の度合い(確率)」まで含めて真似することで、生徒は小さいのに賢くなれます。中心になる道具は「温度付きソフトマックス」と「分布間の距離を測る損失(KL)」です。

### 直感的な理解
試験勉強にたとえます。正解だけが書かれた答案(「問3は7」)から学ぶより、よくできた先輩が「これは7だね。でも形が崩れていて1にも見えるし、ちょっと9っぽくもある」と注釈をつけてくれた方が、はるかに多くを学べます。「7は1や9と似ているが3や8とは似ていない」という、答えそのものには現れない関係性の知識が手に入るからです。

ニューラルネットワークでも同じです。大きなモデル(教師、teacher)は精度が高いですが計算が重く、車載コンピュータのような限られた資源では動かしにくい。自動運転では毎秒何十回も推論する必要があるため、小さくて速いモデル(生徒、student)が望ましい。しかし小さいモデルをゼロから普通に学習すると精度が教師に届きません。

そこで発想を変えます。「正解ラベルだけで学ぶ」のではなく「教師モデルの出力分布そのものを真似させる」のです。これが知識蒸留(knowledge distillation)です。蒸留(distillation)は、教師の中の知識を純度を保ったまま小さい器(生徒)に移すイメージから来ています。

### 基礎: 前提となる概念

ロジット (logit) は、ニューラルネットが各クラスに対して出す生のスコア(まだ確率になっていない実数)です。これをソフトマックス (softmax) という関数で確率に変えます。ソフトマックスは各ロジットを exp(指数関数)で正の値にし、合計が1になるよう正規化します。

ハードラベル (hard label) は正解だけを示す白黒の情報です。手書き数字「7」なら

```
0:0  1:0  2:0  3:0  4:0  5:0  6:0  7:1  8:0  9:0
```

ソフトラベル (soft label) は教師が出す確率の分布です。よく学習された教師は

```
0:0.00  1:0.05  2:0.00  3:0.00  4:0.00  5:0.00  6:0.00  7:0.90  8:0.00  9:0.05
```

のように「7だと思うが1や9に少し似ている」という情報まで持ちます。この「クラス同士の関係」を Hinton らは dark knowledge(暗黒知識、表に出ない知識)と呼びました。ハードラベルにはこの情報がありません。1ビット(7か否か)しか伝えないハードラベルに比べ、ソフトラベルは全クラスにまたがる豊かな分布を伝えるため、1サンプルあたりの情報量が桁違いに多いのです。

温度 (temperature) T は、ソフトマックスに入れる「やわらかさ」のつまみです。ロジットを T で割ってからソフトマックスを取ります。

```
softmax_T(z_i) = exp(z_i / T) / Σ_j exp(z_j / T)
```

T=1 が通常。T を大きくすると分布がなだらかになり、小さい確率(上の「1:0.05」など)が拡大されて見えやすくなります。これが蒸留で重要なクラス間関係をはっきりさせる効果です。逆に T→0 でほぼハードラベル(一番大きいクラスだけ1)に近づきます。数値で見ると、ロジット [6, 4, 2] を T=1 で softmax すると約 [0.84, 0.11, 0.02] ですが、T=4 にすると約 [0.49, 0.30, 0.21] と、2位・3位の情報がはっきり現れます。

### 仕組みを詳しく

蒸留損失は、教師と生徒の「温度を上げたソフトマックス出力」の差を測ります。差の指標には KLダイバージェンス([KLダイバージェンス損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))を使います。これにハードラベルでの通常の分類損失を混ぜるのが標準形です。

```python
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.7):
    # 形状: student_logits, teacher_logits ともに [B, num_classes]
    # 1) ソフトターゲット: 教師の確率を生徒が真似る(温度Tでなめらかに)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)        # [B, C]
    soft_student = F.log_softmax(student_logits / T, dim=-1)    # [B, C]
    kd = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (T * T)

    # 2) ハードターゲット: 本物の正解ラベルでも学ぶ
    ce = F.cross_entropy(student_logits, labels)

    # 3) 2つを重み alpha で混ぜる
    return alpha * kd + (1 - alpha) * ce
```

ステップを言葉で追います。

1. 教師のロジットを温度 T で割ってソフトマックス。やわらかい確率分布 `soft_teacher`(形状 `[B, C]`、C はクラス数)。
2. 生徒のロジットも T で割って log-softmax。KL を計算するには生徒側を対数で渡すのが PyTorch `kl_div` の仕様です。
3. KL に `T*T` を掛ける。温度で割ると勾配が 1/T² 倍に縮むので、それを打ち消して元のスケールに戻すため(Hinton 論文の指摘)。これを忘れると、ハードターゲット側との重みバランスが温度に依存して崩れます。
4. 別に本物の正解ラベルでの通常の分類損失(クロスエントロピー)も計算。教師が間違っているときの保険になります。
5. `alpha` で2つを混ぜる。alpha が大きいほど教師の真似を重視します。

数値例。教師がソフトに「7:0.90, 9:0.05, 1:0.05」と出し、生徒が当初「7:0.99, 他ほぼ0」と固すぎる出力をしていたとします。蒸留損失はこの差を縮めるよう生徒を更新し、生徒も「7と9・1は似ている」という構造を学びます。結果として、生徒は教師の「自信の分布」を引き継ぎ、未知データへの汎化が向上します。

#### 出力蒸留と特徴蒸留

ここまでは最終出力(ロジット)を真似る出力蒸留(logit/response distillation)でした。もう一つの系統が特徴蒸留(feature distillation)で、途中の層の特徴(中間表現)も真似させます。出力だけでは情報が足りない深い生徒の学習に有効です。教師と生徒の特徴次元が違うときは、変換層(projection)で次元を合わせてから MSE やコサイン類似度([コサイン類似度損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl))で揃えます。出力蒸留は「最終的な答え方」を、特徴蒸留は「途中の考え方」を移すと考えると整理できます。

#### 回帰の蒸留

連続値の予測(例: 軌跡や制御指令)では、ソフトマックスの代わりに「教師の連続出力を生徒が MSE などで真似る」形になります。自動運転の方策蒸留(policy distillation)ではこちらが使われます。教師が出す行動分布(平均と分散)ごと真似れば、教師の自信度(どの程度ばらつかせるか)も移せます。分散まで真似ることが、教師の不確実性の表現を生徒へ伝える鍵になります。

### 手法の系譜と主要論文

- 前身: Buciluă, Caruana & Niculescu-Mizil, "Model Compression" (KDD 2006)。大きなアンサンブルの予測を1つのニューラルネットに圧縮するアイデアを最初に示しました。これを後にソフトターゲットと温度として一般化したのが Hinton らです。

- Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network" (NIPS 2014 Deep Learning Workshop, arXiv:1503.02531)。蒸留の原典。大きなモデルやアンサンブル(複数モデルの平均)の知識を、温度付きソフトターゲットで小さな1つのモデルに移せることを示しました。狙いは推論を軽くしつつ精度を保つこと。`T*T` 補正もこの論文で導入されました。

- FitNets: Romero et al., "FitNets: Hints for Thin Deep Nets" (ICLR 2015)。最終出力だけでなく中間層の特徴も真似させる hint learning を提案。深く細い生徒を学習可能にしました。教師と生徒の特徴次元を合わせる変換層が必要なのがコストです。

- Attention Transfer: Zagoruyko & Komodakis (ICLR 2017)。中間特徴そのものより、特徴の注目マップ(空間のどこに反応が集中しているか)を真似る方が効くことを示しました。注目マップは特徴を空間方向に集約した低次元の量で、転移しやすいのが利点です。

- DistilBERT: Sanh et al. (2019, arXiv:1910.01108)。大規模言語モデル BERT を蒸留で約40%小さく、約60%速くしつつ性能の約97%を保ちました。大規模モデルの実用化に蒸留が効くことを示した代表例。

- 自己蒸留・オンライン蒸留。生徒が同時に教師にもなる Deep Mutual Learning (Zhang et al., CVPR 2018) や、生成過程で出力分布を揃える DINO (Caron et al., ICCV 2021) など、固定教師を使わない流れも生まれました。Born-Again Networks (Furlanello et al., 2018) は生徒を教師と同じ構造にして蒸留すると元の教師を上回ることを示しました。

- 自動運転文脈では、VLM(視覚言語モデル、画像と言語を理解する大規模モデル)を教師にし、その豊かな状況理解を軽量な運転方策へ蒸留する研究が増えています。VLM は推論が重く車載で直接動かせないため、その判断を小さい方策へ移して賢さと速さを両立させる狙いです。

### 論文の実験結果(定量データ)

- Hinton et al. の MNIST 実験。大きな教師ネット(隠れ層 1200 ユニット×2、ドロップアウトあり)が出すソフトターゲットで小さな生徒(800×2、正則化なし)を蒸留すると、ハードラベルだけで学習した同じ生徒のテスト誤りが 146 個から 74 個へと半減しました。特に劇的なのが汎化の実験で、学習データから数字「3」を全部除いて蒸留しても、生徒はテスト時に「3」を高い精度で認識できました(誤りは数個程度)。教師のソフトターゲットに「3らしさ」の情報が滲んでいるためで、dark knowledge の存在を示す象徴的な結果です。

- 音声認識(Google の社内システム)では、10個のモデルのアンサンブルの知識を1つのモデルに蒸留し、アンサンブルが達成したフレーム精度の改善のかなりの部分を、単一モデルのコストで再現できたと報告されました。

- DistilBERT は GLUE ベンチマーク(自然言語理解の9タスク総合)で、教師 BERT-base のスコアの約97%(平均スコアで数ポイント以内)を、パラメータ約40%減(約 1.1 億 → 約 6,600 万)・推論約60%高速で達成。指標の意味は「GLUE スコアが高いほど言語理解が正確」で、ほぼ無劣化で大幅に軽くできたことを意味します。

- 温度 T の感度。T が小さすぎると分布がハードラベルに近づき dark knowledge が消え、大きすぎると分布が一様に近づき情報が薄まります。論文では中間の値(タスクにより 2〜10 程度)で最良になる凸型の傾向が報告され、`T*T` 補正を入れないと最適 T が大きくずれることが示されています。

- alpha の役割。教師が誤りやすいタスクでは alpha を下げてハードラベル側を重くした方が良く、教師が十分強いタスクでは alpha を上げてソフトターゲットを重視した方が良い、という傾向が一般に観測されます。実務では alpha 0.5〜0.9 程度がよく使われます。

### メリット・トレードオフ・限界

メリット
- 小さく速いモデルで、大きいモデルに近い精度を出せる(車載・エッジ推論に好適)。
- ソフトラベルにはクラス間関係という追加情報があり、ハードラベルだけより効率的に学べる(少ないデータでも効く)。
- 既に学習済みの強い教師があれば生徒の学習を大きく加速できる。
- ラベルのないデータでも教師の出力を疑似ラベルとして使え、半教師あり的に活用できる。

トレードオフ・限界
- 良い教師が前提。教師が弱いと生徒もそれ以上にならず、教師の誤り(バイアスや誤分類)もそのまま継承する。
- 温度 T と混合重み alpha という追加ハイパーパラメータの調整が必要。
- 教師の推論コストが学習時にかかる(事前にソフトラベルをキャッシュすれば軽減できるが、データ拡張と併用しにくくなる)。
- 教師と生徒のアーキテクチャ差(capacity gap)が大きすぎると蒸留がうまくいかないことがある。教師が強すぎると生徒には真似きれず、かえって性能が落ちる現象が報告されている(対策に助教師=teaching assistant を挟む TAKD、Mirzadeh et al. 2020 など)。
- 未解決の課題として「なぜ蒸留が効くのか」の理論的説明がまだ完全でない。ラベル平滑化との関係、暗黙の正則化効果、最適化の容易化など複数の説が併存しています。

### 発展トピック・研究の最前線

- データフリー蒸留。元の学習データが手に入らない状況で、教師から逆生成した合成データで蒸留する研究(DeepInversion, Yin et al. 2020 等)。プライバシーやデータ保持の制約がある現場で重要です。
- 自己蒸留と Born-Again Networks。生徒を教師と同じ構造にして蒸留すると、教師より良くなることがある(Furlanello et al., 2018)。圧縮目的でなく性能向上の道具としての蒸留です。
- 大規模言語モデルの蒸留。近年は LLM の連鎖推論(chain-of-thought、考える過程を文章で出させる手法)そのものを蒸留する、あるいは大モデルの出力で小モデルを指示チューニングする手法が盛んです。出力分布だけでなく推論過程を移すのが新しい流れです。
- マルチモーダル/クロスモーダル蒸留。VLM やカメラ・LiDAR など異なるモダリティの教師から、単一モダリティの軽量生徒へ知識を移す研究。自動運転で特に活発で、高価な LiDAR 教師の知識をカメラのみの生徒へ移す試みなどがあります。

### さらに学ぶための関連トピック
- [KLダイバージェンス損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [コサイン類似度損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [クロスエントロピー損失](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)
- [模倣学習 (Behavior Cloning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)


## LAMB

### ひとことで言うと
LAMB は、巨大なバッチサイズ(一度に大量のデータをまとめて学習する設定)でも壊れずに学習できるよう、Adam に「層ごとの学習率調整」を組み込んだオプティマイザです。これにより、大規模言語モデル BERT の事前学習を、従来の約 3 日から 76 分まで短縮しました。LAMB = Layer-wise Adaptive Moments optimizer for Batch training の略です。一言でいえば「Adam の賢さ(要素ごとの学習率調整)はそのままに、LARS の層ごと正規化を重ねて、超大バッチでも発散しないようにした」手法です。

### 直感的な理解
大規模言語モデルの事前学習は、とにかく時間とお金がかかります。BERT は当時、数台規模の TPU で 3 日ほど回す必要がありました。これを速くする一番素直な方法は「バッチサイズを大きくして、多数の演算装置で並列に処理する」ことです。1 回の更新でより多くの文を一度に食わせれば、同じデータ量を少ない更新回数で学習し切れます。更新回数(イテレーション)はそのまま「全演算装置が待ち合わせる同期回数」でもあるので、これを減らすことが分散学習の高速化に直結します。

ところがバッチを大きくすると学習率も上げる必要があり([LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の線形スケーリング則を参照)、上げると学習が発散します。画像分類ではこの問題を LARS が解決していました。しかし LARS のベースはモメンタム付き SGD です。BERT のような Transformer 言語モデルでは、SGD ベースより Adam ベースの方が学習が圧倒的に安定することが経験的に知られていました。Transformer は層ごとに勾配のスケールが大きく異なり(埋め込み層・Attention の射影・FeedForward・LayerNorm のパラメータがそれぞれ違う桁の勾配を出す)、Adam の「要素ごとに勾配の大きさで割る」適応がほぼ必須だったのです。

そこで「Adam に LARS の層ごと調整を組み合わせれば、Transformer の超大バッチ学習が安定するのでは」という発想で生まれたのが LAMB です。Adam が各パラメータ内部の事情に合わせ、その上から LARS 風の層ごと正規化が層全体のバランスを取る、という二段構えです。役割分担がきれいに分かれているのがこの手法の美点です。

### 基礎: 前提となる概念

前提を 1 つずつ噛み砕きます。

- **Adam**: モメンタム(過去の勾配の指数移動平均で「進むべき方向」をならす。詳しくは [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) / [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))と、勾配二乗の移動平均(各パラメータがどれだけ激しく動くかを測る。[Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の系譜)を組み合わせた定番オプティマイザです。「要素ごとに学習率を自動調整する」のが特徴で、勾配が大きく暴れる方向は小さく、おとなしい方向は大きく進めます。
- **モーメント(moment)**: 勾配の 1 次モーメント `m`(平均、=方向)と 2 次モーメント `v`(二乗平均、=ばらつき)のことです。Adam も LAMB もこの 2 つをパラメータごとに保持します。指数移動平均なので、減衰係数 β1(典型 0.9)、β2(典型 0.999)で過去をどれだけ引きずるかを決めます。
- **bias correction(バイアス補正)**: 移動平均は学習初期に 0 から始まるため値が小さく出すぎます。これを `1 − β^t`(t はステップ番号)で割って補正したものが `m_hat = m/(1−β1^t)`, `v_hat = v/(1−β2^t)` です。t が大きくなると分母は 1 に近づき、補正の効果は薄れます。
- **大バッチ学習**: バッチサイズを数千〜数万にして多数の GPU/TPU で並列に学習し、学習時間を短縮する手法です([LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照)。
- **BERT**: 2018 年に Google が発表した Transformer ベースの双方向言語モデル(Devlin et al., NAACL 2019)です。大量のテキストでマスク言語モデル(文中の単語を隠して当てる課題)と次文予測として事前学習します。この事前学習が重く、当時 3 日ほどかかっていました。
- **信頼比(trust ratio)**: LARS から受け継いだ概念で「層の重みノルム ÷ その層の更新量ノルム」です。更新を重みのサイズに見合った大きさへ調整する係数です。これがゼロを正規化の中心思想として引き継いでいます。

### 仕組みを詳しく

LAMB は「各層について、Adam が計算した更新量を、その層の重みの大きさに合わせて再スケールする」ことをします。LARS との決定的な違いは、**再スケールする対象が『素の勾配』ではなく『Adam が適応的に計算した更新量』である**点です。これにより、要素ごとの適応(Adam)と層ごとの正規化(LARS)が両立します。

各層(正確にはパラメータグループ)l について、ステップを分解します。

1. Adam とまったく同じく、モメンタム `m` と勾配二乗移動平均 `v` を更新し、バイアス補正した `m_hat`, `v_hat` を求める。
   ```
   m ← β1·m + (1−β1)·g
   v ← β2·v + (1−β2)·g²
   m_hat = m/(1−β1^t),  v_hat = v/(1−β2^t)
   ```
2. Adam の素の更新方向を作る。重み減衰は Adam ではなく更新方向側に足す([AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses) と同じ decoupled weight decay の流儀):
   ```
   r_l = m_hat / ( sqrt(v_hat) + ε ) + λ·W_l
   ```
3. その層の重みノルム `||W_l||` と、更新方向のノルム `||r_l||` を計算する。
4. 信頼比を計算する:
   ```
   trust_ratio = φ(||W_l||) / ||r_l||
   ```
   φ は重みノルムに対する関数で、単純には φ(x) = x(そのまま)を使います。実装では `φ(x) = min(max(x, lower), upper)` のようにクリップして、ノルムが極端なときの暴走を防ぐこともあります。`||W_l||` が 0(重みが全部ゼロ)のときは trust_ratio = 1 にフォールバックするのが定石です。
5. 学習率にこの信頼比を掛けて更新する:
   ```
   W_l ← W_l − γ × trust_ratio × r_l
   ```
   γ はグローバル学習率です。

骨格を簡略化して並べると:

```
Adam の更新方向 r_l = m_hat / (sqrt(v_hat) + ε) + λ·W_l
信頼比 = ||W_l|| / ||r_l||
W_l ← W_l − γ × 信頼比 × r_l
```

つまり「要素ごとの適応(Adam の `1/sqrt(v_hat)`)」と「層ごとの正規化(LARS 由来の信頼比)」を二段重ねにしています。Adam がパラメータ内部のスケール差を吸収し、信頼比が層と層の間のスケール差を吸収する、という役割分担です。

直感を数値で見ます。グローバル学習率 γ = 1.0 とします。

- 層A: ||W|| = 10、Adam 更新方向のノルム ||r|| = 2 → 信頼比 = 5 → 実効的に大きく動かす。
- 層B: ||W|| = 0.5、||r|| = 2 → 信頼比 = 0.25 → 控えめに動かす。

各層は「自分の重みの大きさに対して一定の割合」で動くようになり、重みが小さい層が相対的に大きく動かされて壊れることを防ぎます。重要な観察は、LAMB の 1 ステップでの相対的な重み変化 `||ΔW_l|| / ||W_l||` が γ にほぼ等しくなる点です。つまり「すべての層が同じ割合 γ だけ進む」よう正規化されており、これが超大バッチでも発散しない理由です。Adam 単体ではこの相対変化が層ごとにバラバラになり、一番大きく動く層が学習を壊します。

**tensor shape の話**: 信頼比は「層ごとに 1 個のスカラー」です。状態(`m` と `v`)は Adam と同じくパラメータと同サイズの配列を 2 つ持ちます。層ごとのノルム計算が毎ステップ加わりますが、要素数に線形のコストで小さいです。メモリは実質 Adam / AdamW と同等と考えてよいです。分散学習では、各層のノルムを all-reduce で集約する必要があるため、通信パターンが Adam よりわずかに複雑になります。

### 手法の系譜と主要論文

LAMB は 2 つの系譜の合流点にあります。

- 適応的オプティマイザの系譜: [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) → RMSProp/[Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) → Adam(Kingma & Ba, ICLR 2015)→ [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)/[Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)/[RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) などの改良。
- 大バッチ対応の系譜: Goyal et al.(線形スケーリング則+warmup, 2017)→ [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)(You et al., 2017, SGD ベース、CNN 向け)→ LAMB(Adam ベース、Transformer も含む)。

**Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh, "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (ICLR 2020, arXiv:1904.00962)** が LAMB の原論文です。

- 提案: LARS の層ごと正規化を Adam に組み込んだ LAMB。要素ごとの適応(Adam)と層ごとの信頼比スケーリングを組み合わせました。
- 動機: BERT のような Transformer を超大バッチで安定して学習し、TPU/GPU を大量に使って学習時間を劇的に短縮したかったから。SGD ベースの LARS では Transformer に十分でなかったため、Adam ベースに作り替えました。
- 新規性: 信頼比に「Adam の更新方向のノルム」を使う定式化と、その収束に関する理論解析(滑らかな非凸目的かつ有界分散という仮定の下で、確率的勾配法として O(1/√(Tb)) オーダで停留点へ収束するという保証。T はステップ数、b はバッチサイズ)を与えた点です。理論が「バッチを増やせば必要ステップが減る」ことを裏づけている点が、大バッチ学習の正当化として重要でした。

その後、LAMB は Hugging Face / NVIDIA(APEX)/ TensorFlow など主要フレームワークに実装され、大規模事前学習の標準的な選択肢の 1 つになりました。

### 論文の実験結果(定量データ)

主な数値を、指標の意味とともに説明します。

最も有名な結果は **BERT 事前学習の高速化** です。BERT の事前学習は 2 段階(系列長 128 のフェーズ 1 と、系列長 512 のフェーズ 2)で行います。フェーズ 2 だけ系列が長く Attention のコストが系列長の二乗で効くため重くなります。

- バッチサイズ 32K(フェーズ 1)/ 32K(フェーズ 2)という当時として桁外れの大バッチで、LAMB は精度を落とさずに学習を完了しました。総ステップ数はベースラインの約 100 万ステップから 8599 ステップ程度まで激減します。
- 学習時間は、ベースライン(Adam, バッチ 512 程度)の約 3 日に対し、LAMB + 1024 TPUv3 チップで **76 分**。論文タイトルの「76 minutes」はこれを指します。
- 精度の指標は下流タスク **SQuAD v1.1 の F1 スコア**(質問応答で、予測した答えの範囲が正解とどれだけ重なるかを測る指標。100 点満点に近いほどよい)です。LAMB で大バッチ学習した BERT は、ベースラインと同等以上の F1(およそ 90.4〜91.5、ベースラインは約 90.4)を達成しました。つまり「速くしても賢さは落ちていない」ことが示されています。
- さらに大きなバッチ 64K/32K(フェーズ 1 を 64K、フェーズ 2 を 32K)でも学習が成立し、ほぼ線形のスケーリング(76.7% のスケーリング効率)で時間短縮できることを報告しています。

アブレーション・対比の要点:

- **Adam をそのまま大バッチにすると壊れる**: バッチを 16K 以上に増やすと、素の Adam では精度が大きく劣化(あるいは発散)します。LAMB はこの領域で精度を維持し、層ごと正規化の効果を裏づけました。
- **信頼比を外すと劣化**: 信頼比のスケーリングを取り除くと(=素の AdamW に戻すと)、超大バッチでの安定性が失われます。LAMB の効きどころが「層ごと正規化」であることを示すアブレーションです。
- **LARS では Transformer に不十分**: 同じ大バッチで SGD ベースの LARS を BERT に使うと精度が出ません。Adam ベースに移植したことが本質だったことを示します。
- 画像分類(ResNet-50 on ImageNet)でも検証され、バッチ 16K〜32K で LARS と同等以上(top-1 約 76.7%、わずか数エポックの追加)の精度を出すことを示しています。Transformer 専用ではなく汎用に使えることの傍証です。

注意すべきは、これらの利点が出るのは **本当に超大バッチ・大規模分散** の領域だという点です。小〜中バッチの普通の学習では、LAMB は素の Adam / AdamW とほぼ差が出ません。

### メリット・トレードオフ・限界

メリット

- 巨大バッチ(数万)でも発散させずに学習でき、多数の演算装置で学習時間を劇的に短縮できる。
- Adam の要素ごと適応と層ごとの正規化を両立させているため、Transformer 言語モデルの大バッチ事前学習に強い。
- 層ごとに持つのはスカラーの信頼比だけで、状態のメモリ追加は Adam / AdamW とほぼ同じ。
- 画像・言語の両方で効果が確認されており、汎用性が高い。

トレードオフ・限界

- 利点が出るのは本当に大規模・大バッチの領域に限られる。小〜中バッチでは素の Adam / AdamW と差が出にくく、むしろ調整の手間だけが増えることがある。
- 信頼比・層ごとノルム・φ のクリップ範囲など実装がやや複雑で、学習率・重み減衰の調整に手間がかかる。バイアスや LayerNorm のパラメータには信頼比を適用しない(あるいは φ をかけない)のが定石で、この扱いを誤ると劣化する。
- 大バッチ学習自体に、線形スケーリング則・warmup([RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の暖機との関連も)・学習率スケジュール([LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning))など別の工夫を併せて必要とすることが多い。
- 超大バッチ領域には依然として「汎化ギャップ」(大バッチで学習したモデルは小バッチよりわずかに汎化が劣る傾向)という未解決問題があり、LAMB はこれを緩和するが消し去るわけではない。どのバッチサイズが品質と速度の最適点かは、モデル・データごとに探索が必要です。

### 発展トピック・研究の最前線

- **より大きなモデルへの適用**: LAMB は BERT 以降、GPT 系や T5 など大規模事前学習でも使われ、データ並列のスケールアップを支えました。一方、超大規模では Adam(AdamW)+ 慎重な warmup と学習率スケジュールでも十分安定するケースが多く、LAMB を使うかどうかは実装・インフラとの相性で判断されるようになっています。実際、近年の代表的な LLM では AdamW を使い続ける例も多く、LAMB の「大バッチ必須性」は計算資源とのトレードオフで決まります。
- **2 次・行列構造を使う系統との比較**: 同じ「大規模学習を速く」という目的で、対角ヘシアンを使う [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) や、行列前処理の [Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) が登場しました。LAMB が「層ごとスカラー正規化」なのに対し、これらは曲率や行列構造まで使う点が異なります。どれが最良かはモデル規模・タスク・計算予算に依存し、決着していません。
- **理論の深化**: 信頼比正規化がなぜ大バッチを安定させるのか、層のスケール不変性(正規化層との相互作用)との関係など、最適化理論側からの理解が進んでいます。とくに「重みを定数倍しても出力が変わらない層では、絶対的な更新量より相対的な更新量が本質的」という視点が、信頼比の正当化につながっています。
- **NLP 以外への展開**: 大バッチ対照学習(自己教師あり表現学習)など、バッチサイズが品質に直結するタスクで LARS / LAMB が使われ続けています。

### さらに学ぶための関連トピック

- [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## LARS

### ひとことで言うと
LARS は「一度にたくさんのデータをまとめて学習する(大バッチ学習)とき、ネットワークの層ごとに別々の学習率を自動で決めることで学習を安定させる」オプティマイザです。これにより、画像分類モデル(ResNet など)を多数の GPU で巨大なバッチサイズで高速に学習できるようになりました。LARS = Layer-wise Adaptive Rate Scaling(層ごとの適応的学習率スケーリング)の略です。要するに「全層を同じ学習率で動かすのをやめて、各層を自分の重みの大きさに見合った速さで動かす」という発想を、たった1つの正規化ルールに落とし込んだ手法です。

### 直感的な理解
モデルの学習を「大勢で一斉に山を下る」作業にたとえてみます。バッチサイズを大きくするというのは、参加人数(=1回の更新に使うデータ枚数)を増やすことに相当します。人数が多いほど一度に進む距離(=処理できるデータ量)は増えるので、同じ道のりを少ない回数で歩き切れます。これが大バッチ学習で学習が速く終わる理屈です。分散学習では、更新回数がそのまま全 GPU の同期回数なので、これを減らすことが速度の本質になります。

ところが問題があります。人数が増えたぶん歩幅(学習率)も大きくしないと、結局はゆっくりにしか進めません。かといって歩幅を大きくすると、足場の悪い箇所(損失地形の急な方向)でつまずいて転びます。「転ぶ」とは学習が発散すること、つまり損失が無限大に向かって暴走し、モデルが使い物にならなくなることです。

さらに厄介なのは、ニューラルネットの「足場の固さ」が層によってまるで違う点です。ある層は重みが大きくて頑丈、別の層は重みが小さくて壊れやすい。全員に同じ歩幅を強いると、壊れやすい層だけが先に転んで、その崩れが連鎖して全体が発散します。LARS は「層ごとに歩幅を変えればいい。重みが大きい層は大きく、小さい層は小さく動かせば、どの層も自分にとって安全な割合だけ進める」という単純で強力なアイデアです。

### 基礎: 前提となる概念

理解に必要な言葉を1つずつ噛み砕きます。

- **バッチサイズ(batch size)**: 1回のパラメータ更新で、まとめて使うデータの枚数です。バッチサイズ 32 なら画像 32 枚分の勾配を平均して 1 回更新します。
- **勾配(gradient)**: 損失(モデルの間違いの大きさ)を各パラメータで微分した値です。「このパラメータをどちらに、どれだけ動かせば損失が減るか」を表すベクトルだと思ってください。
- **学習率(learning rate)**: 勾配の方向にどれだけの幅で踏み込むかを決める係数です。一歩の大きさそのものです。
- **大バッチ学習(large batch training)**: バッチサイズを非常に大きく(数千〜数万)する学習です。たくさんの GPU で並列計算すれば、同じ枚数のデータを少ない更新回数で処理でき、学習を短時間で終わらせられます。
- **層(layer)**: ニューラルネットの構成単位です。畳み込み層・全結合層などが何十層も積み重なってネットワークを作ります。
- **ノルム(norm)**: ベクトルや行列の「大きさ」を1つの数で表したものです。`||W||` と書けば、層の重み全体をひとまとめにしたベクトルの長さ(各成分を二乗して足して平方根を取った値、ユークリッドノルム)を指します。
- **モメンタム(momentum)**: 過去の更新方向を一定割合だけ引き継いで、勾配のノイズをならし加速する仕組みです([SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。LARS のベースはこのモメンタム付き SGD です。

ここで大事な経験則が1つあります。**線形スケーリング則(linear scaling rule)** です。Goyal ら(Facebook, 2017, arXiv:1706.02677)は「バッチサイズを k 倍にしたら学習率も k 倍にすると、小バッチと同じ精度に到達できる」ことを ImageNet で示しました。例えばバッチ 256・学習率 0.1 が基準なら、バッチ 8192(32倍)では学習率 3.2 にします。直感的には、バッチが k 倍になると 1 ステップで使うデータが k 倍なので、1 ステップで k 倍の距離を進んでよい、という理屈です。ただしこの大きな学習率は学習初期に発散しやすいので、最初の数エポックだけ学習率を 0 から徐々に上げる **warmup(ウォームアップ)** が必須でした。Goyal らはこの工夫でバッチ 8192 まで持ちこたえましたが、それ以上(16K, 32K)に増やすと warmup だけでは安定しなくなります。LARS はこの壁を越えるために生まれました。

### 仕組みを詳しく

LARS の核心は「各層の更新量を、その層の重みの大きさに合わせる」ことです。

You らが最初に観測した事実から始めます。彼らは ResNet-50 や AlexNet の各層について「重みノルム `||W_l||`」と「勾配ノルム `||g_l||`」の比を測りました。すると、この比が層ごとに桁違いに違うことが分かりました。論文の報告では、AlexNet の最初の畳み込み層 conv1 では `||W||/||g||` がおよそ 5.8、別の全結合層では 1345 といったように、200 倍以上の開きがありました。全層で同じ学習率を使うと次のことが起きます。

- 比が小さい層(重みに対して勾配が大きい層)では、1 ステップの更新が重みに対して大きすぎて暴れる。
- 比が大きい層(重みに対して勾配が小さい層)では、更新が重みに対して小さすぎて学習が進まない。

つまり、全体の学習率を上げると、最初に壊れる層に引きずられて発散します。LARS は層ごとにこの不均衡を打ち消します。

各層 l について、次のように更新量を決めます。

1. その層の重み全体のノルム `||W_l||` を計算する。
2. その層に来た勾配全体のノルム `||g_l||` を計算する。
3. 「ローカル学習率(local learning rate)」を次で決める:
   ```
   local_lr_l = η × ||W_l|| / ( ||g_l|| + λ·||W_l|| )
   ```
   ここで η(イータ)は「信頼係数(trust coefficient)」と呼ぶ小さな定数(例 0.001)、λ は重み減衰(weight decay)の係数です。
4. グローバル学習率 γ にこのローカル学習率を掛けて、その層だけの実効学習率として更新する(モメンタム付き SGD がベース)。

式の骨格を簡略化して書くと:

```
実効学習率_l = γ × η × ||W_l|| / ||g_l||
更新方向 v_l ← μ·v_l + 実効学習率_l × (g_l + λ·W_l)
W_l ← W_l − v_l
```

μ はモメンタム係数(過去の更新方向をどれだけ引き継ぐか、典型値 0.9)です。

数値例で見ます。グローバル学習率 γ = 1.0、η = 0.001 とします。

- 層A: ||W|| = 10、||g|| = 1 → 実効学習率 = 1.0 × 0.001 × 10/1 = 0.01
- 層B: ||W|| = 0.1、||g|| = 1 → 実効学習率 = 1.0 × 0.001 × 0.1/1 = 0.0001

重みの大きい層 A には大きめ、小さい層 B には小さめの学習率が自動で割り当てられます。別の見方をすると、LARS は「1 ステップで重みが変わる相対的な割合 `||ΔW||/||W||` を、全層でほぼ一定に保つ」装置です。どの層も自分のサイズの一定割合(上の例では η = 0.1%)だけ動く、というふうに揃えるわけです。これが「重みの大きさで正規化する」ことの本質です。

この設計が理論的に正当化されるのは、BatchNorm を含む現代の CNN が **スケール不変性** を持つからです。BatchNorm 付きの層では、重み W を定数 α 倍しても前向き出力は変わらず、勾配が逆に 1/α 倍になります。つまり重みの絶対的なサイズには意味がなく、本質的なのは「重みに対して相対的にどれだけ動くか」です。LARS の `||W||/||g||` 正規化は、この相対量を直接制御するため、スケール不変な層にとって自然な学習率になります。

**tensor shape の話**: ローカル学習率は「層ごとに 1 個のスカラー」です。Adam のようにパラメータ要素ごとに状態を持つわけではなく、層単位なので追加メモリはごくわずか(層の数ぶん)です。ベースはモメンタム付き SGD なので、状態はモメンタム 1 つ分(パラメータと同サイズの配列が1つ)だけです。毎ステップ層ごとに 2 つのノルム(重みと勾配)を計算しますが、これは要素数に線形のコストで、行列演算に比べれば無視できます。分散学習では各層のノルムを all-reduce で集約します。

実装上の細かい注意がいくつかあります。バイアスや BatchNorm のパラメータ(スカラーやベクトル)には LARS を適用せず、通常の SGD で更新するのが定石です。これらは重みノルムが小さく信頼比が不安定になりやすいためです。また `||g_l||` がゼロに近いとローカル学習率が発散するので、分母に小さな値を足すか上限を設けます。

### 手法の系譜と主要論文

大バッチ学習の流れの中で LARS を位置づけます。

- **Krizhevsky, "One weird trick for parallelizing CNNs" (2014, arXiv:1404.5997)**: 大バッチ・データ並列の初期の検討で、学習率を √k 倍にするスケーリングを提案。後の線形則の前段にあたります。
- **Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Facebook, 2017, arXiv:1706.02677)**: 線形スケーリング則と warmup を確立し、バッチ 8192 で ResNet-50 を 256 GPU・1 時間で学習しました。LARS の直前の到達点で、「ここまでは素朴な SGD でいけるが、これ以上は壊れる」という境界を示しました。
- **You, Gitman, Ginsburg, "Large Batch Training of Convolutional Networks" (2017, arXiv:1708.03888)**: LARS の原論文。層ごとの信頼比による正規化を提案し、バッチ 16K〜32K でも ResNet-50 / AlexNet を学習できることを示しました。動機は「層ごとの `||W||/||g||` の不均衡を直接打ち消せば、warmup だけでは越えられなかった壁を越えられる」という観察です。
- **You et al., "ImageNet Training in Minutes" (2018, arXiv:1709.05011)**: LARS を使い、バッチ 32K で ResNet-50 を多数 CPU/GPU 上で 20 分、AlexNet を 11 分で ImageNet 学習する記録を出しました。LARS が「大バッチを実用的なツールに変えた」ことを示すマイルストーンです。
- **You et al., "LAMB" (ICLR 2020)**: LARS のアイデアを Adam ベースに移植し、Transformer 言語モデル(BERT)の大バッチ学習を可能にしました。詳しくは [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照。LARS が CNN 向け、LAMB が Adam ベースで Transformer も含む、という住み分けで理解すると整理しやすいです。

### 論文の実験結果(定量データ)

LARS 原論文と続編の主な数値を、指標の意味とともに説明します。

評価指標は **ImageNet 検証セットの top-1 / top-5 精度** です。top-1 は「モデルが最も確信したクラスが正解と一致した割合」、top-5 は「確信度上位 5 個に正解が含まれた割合」です。ResNet-50 のベースライン(小バッチ・通常の SGD)は top-1 でおよそ 75〜76% です。大バッチ学習の評価では「バッチを巨大にしても、この基準からどれだけ精度が落ちないか」が問われます。0.5〜1% の差でも、これだけ成熟したベンチマークでは大きな意味を持ちます(1% は ImageNet では 1 年分の研究進歩に相当することもあります)。

- **ResNet-50 / バッチ 8192**: 素朴な線形スケーリング+warmup でも維持できる範囲。LARS でも同等の top-1(およそ 75.3%)を達成。
- **ResNet-50 / バッチ 16384〜32768**: warmup だけの素朴な手法は精度が大きく崩れる(数%の劣化や発散)領域。LARS を使うと、バッチ 32K でも top-1 の劣化を 1% 以内(約 74.9〜75.3%)に抑えられたと報告されています。
- **AlexNet / バッチ 8192**: 素朴な大バッチでは top-1 がおよそ 53〜57% まで落ちるのに対し、LARS は小バッチ基準(約 57〜58%)に近い精度を回復しました。LARS の効果が最も劇的に出た例の 1 つです。BatchNorm を入れた AlexNet-BN ではさらに改善が顕著でした。
- **学習時間(続編 2018)**: LARS でバッチ 32K にすると、ResNet-50 の ImageNet 学習が約 20 分、AlexNet が約 11 分。1 GPU で数日かかっていた学習が、多数演算装置で分単位に短縮されました。

アブレーション的な観察として重要なのは、「**LARS を外すと(=全層同一学習率に戻すと)バッチ 16K 以上で精度が急落する**」という対比です。逆に、warmup を併用しないと LARS でも初期発散が起きやすいことも報告されており、LARS と warmup は補完関係にあります。つまり LARS 単独で万能なのではなく「線形スケーリング+warmup+層ごと正規化」の組み合わせで初めて超大バッチが安定する、という構図です。

### メリット・トレードオフ・限界

メリット

- 大バッチ(数千〜数万)でも層ごとの学習率調整で安定して学習でき、超大バッチ領域(16K 以上)で素朴な手法を明確に上回る。
- 多数 GPU での分散学習による大幅な学習時間短縮(ImageNet を分単位)を可能にした。
- 層単位のスカラーしか追加で持たないため、メモリ追加はごくわずか。SGD ベースなので状態量も Adam より軽い。

トレードオフ・限界

- 信頼係数 η など新しいハイパーパラメータが増える。値が大きすぎると発散、小さすぎると学習停滞する。
- 毎ステップ層ごとのノルム計算が必要(コストは小さいが追加される)。
- 主に CNN の大バッチ画像分類で効果が検証された手法で、小バッチや非画像タスクでは利点が薄い。Transformer 系では Adam 系の安定性が必要なため、後継の LAMB が向く。
- バイアスや正規化層パラメータの扱い、ゼロ勾配時の数値処理など、実装の細部に発散の落とし穴がある。
- 大バッチ学習そのものに「汎化ギャップ」(大バッチで学習したモデルは小バッチより汎化がやや悪くなりがち)という未解決の課題が残る。LARS はこれを和らげるが完全には消さない。研究上は「大バッチでなぜ汎化が落ちるのか(損失地形の鋭い極小に落ちやすいという Keskar らの仮説)」「層ごと正規化が汎化に与える影響」が今も議論されています。

### 発展トピック・研究の最前線

LARS の「層ごとに重みノルムで正規化する」という発想は、その後さまざまな方向へ広がりました。

- **LAMB**: Adam の適応的更新量を信頼比で再スケールし、Transformer の大バッチ事前学習を可能にしました([LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。
- **正規化と最適化の理論**: LARS が成立する背景には、重みノルムと勾配の関係が層の正規化(BatchNorm / LayerNorm)とスケール不変性に深く関わるという理論があります。「重みを定数倍しても出力が変わらない層では、更新の絶対量より相対量が本質的」という視点が、信頼比の正当化につながっています。スケール不変な重みは球面上を動くと見なせ、有効学習率の解析(weight norm が暗黙に有効学習率を決める)が進んでいます。
- **直交化・行列構造を使う2次手法**: 別系統として、勾配を行列・テンソルとして扱い前処理する [Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) や、その近似である Muon が登場しました。LARS が「層ごとスカラー正規化」なら、これらは「層内の行列構造まで使う正規化」と位置づけられます。
- **自己教師あり学習との相性**: 大バッチが品質に効く対照学習(SimCLR など)で LARS が標準的に使われ、大バッチ・大規模事前学習の基盤技術になりました。
- **適応的バッチサイズ・学習率スケジュール**: 学習の進行に応じてバッチや学習率を動的に変える手法とも組み合わせられます([LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning))。

### さらに学ぶための関連トピック

- [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Layer Normalization

### ひとことで言うと
Layer Normalization(レイヤー正規化、略して LayerNorm)は、ニューラルネットワークの途中を流れる数値を「ちょうど良い大きさ」に整える処理です。1つのデータ(1枚の画像、1つの単語ベクトルなど)の中の特徴の値をならして、平均がほぼ0・ばらつきがほぼ1になるように作り直します。これにより数値が暴れず、深い層を重ねても学習が安定して速くなります。Transformer や ViT(Vision Transformer、画像をパッチに割って Transformer で処理するモデル)では必ずと言っていいほど使われる、いわば「定番部品」です。

### 直感的な理解
料理に例えると、各層は「材料に何かを加えて混ぜる」工程です。ところが工程を経るたびに、量が勝手に10倍になったり1/10になったりするとしたら困ります。次の工程は前の工程の出力をそのまま受け取るので、量が膨らみ続ければ最後には鍋からあふれ(発散)、縮み続ければ味がしなくなる(消失)。LayerNorm は各工程の出口で「全体をいったん一定の量・濃さにそろえ直す」役割を果たします。

なぜ「データ1件の中で」そろえるのか。これが LayerNorm の最大の特徴で、後で説明する Batch Normalization(バッチ正規化)との決定的な違いです。バッチ(一度にまとめて処理する数十個のデータの束)の中の他のデータを一切参照せず、目の前の1件だけで正規化を完結させます。だからバッチに何個データがあっても、極端には1個でも、まったく同じ計算ができます。自動運転のオンライン推論(1フレームずつ来る入力をその場で処理する)や、文の長さがバラバラな自然言語処理に、この性質がぴたりと合いました。

### 基礎: 前提となる概念
理解に必要な用語を先に噛み砕きます。

- 浮動小数点数: コンピュータが小数を表す形式。深層学習の数値はほぼこれです。
- 層(レイヤー): 「入力に重み(かけ算の係数)をかけて足し、非線形関数を通す」という処理の一単位。これを何枚も重ねたものがニューラルネットワークです。
- 活性化(activation): ある層の出力、つまり次の層への入力になる数値の束のこと。
- 平均 μ(ミュー): 数値の合計を個数で割ったもの。中心の位置を表す。
- 分散 σ²(シグマ二乗)/ 標準偏差 σ: 数値が平均からどれくらい散らばっているかの指標。分散の平方根が標準偏差。
- 勾配(こうばい、gradient): 損失(誤差)をパラメータで微分した値。「このパラメータをどちら向きにどれだけ動かせば誤差が減るか」を示す矢印。学習はこの矢印に沿ってパラメータを少しずつ動かす作業です。
- 内部共変量シフト(Internal Covariate Shift): 学習が進んで各層の重みが変わると、その層が次の層に渡す活性化の分布も変わってしまう現象。次の層は「動く的」を相手に学ぶことになり、学習が遅くなる、という主張です。Ioffe と Szegedy(2015)が命名しました。

なぜ正規化が要るのか、具体例で。ある層に平均0・ばらつき1のきれいな入力が来たとします。その層の重みがやや大きいと、出力は平均5・ばらつき100の巨大な値になるかもしれません。次の層はこれをさらに変形し、層を10枚20枚と重ねると数値が指数的に膨らんで無限大に飛んだり、逆に0に潰れたりします。さらに困るのは、活性化が極端に大きい/小さい領域に入ると、sigmoid や tanh のような非線形関数の傾き(=勾配)がほぼ0になり、学習信号が伝わらなくなることです(勾配消失)。正規化はこの暴走を毎層で抑え込む安全弁です。

### 仕組みを詳しく
LayerNorm の中身を、テンソル形状(tensor shape、配列の各次元のサイズ)と数値で追います。

入力を `x` とします。Transformer なら形状はたとえば `[B, L, D]`。
- `B` = バッチサイズ(同時処理するデータ数)。例: 8
- `L` = トークン数(単語やパッチの個数)。例: 196(ViT で画像を 14x14 のパッチに割った場合)
- `D` = 特徴次元(各トークンを表すベクトルの長さ)。例: 256

LayerNorm は最後の次元 `D` の中だけで平均と分散を取ります。つまり1つのトークンが持つ256個の数値を見て、その256個の平均 `μ` とばらつき `σ²` を求めます。各トークン(全部で `B×L = 8×196 = 1568` 個)について独立に次を行います。

1. 平均: `μ = (256個の値の合計) / 256`
2. 分散: `σ² = ((各値 - μ) の二乗の平均)`
3. 正規化: `x_hat = (x - μ) / sqrt(σ² + ε)`
   - `ε`(イプシロン)は 1e-5 程度の極小値。分散が0のとき0で割らないための保険。
4. スケールとシフト: `y = γ ⊙ x_hat + β`
   - `γ`(ガンマ)と `β`(ベータ)は学習で動く、それぞれ256個のパラメータ。「平均0・ばらつき1が常に最適とは限らない」ので、ネットワークが必要なら元のスケールに戻したり、別のスケールに移したりできる自由度を残しています。`⊙` は要素ごとのかけ算。

数値例。あるトークンの256次元のうち先頭3つが `[10.0, -2.0, 4.0]` で、256個全体の平均が `μ=3.0`、`sqrt(σ²+ε)=4.0` だったとします。正規化後は `(10-3)/4 = 1.75`、`(-2-3)/4 = -1.25`、`(4-3)/4 = 0.25`。巨大な値も小さい値も、おおむね -2〜+2 の扱いやすい範囲に収まります。`γ` と `β` の初期値は普通 `γ=1, β=0` で、学習開始直後は「正規化しただけ」の挙動から始まり、必要に応じてネットワークが分布をずらしていきます。

ここで重要な性質が「スケール不変性(scale invariance)」です。入力 `x` を定数 `a` 倍した `a·x` を LayerNorm に通すと、平均も `a` 倍・標準偏差も `a` 倍になるので、`(a·x − a·μ)/(a·σ) = (x − μ)/σ` となり、出力は元の `x` を入れたときと完全に一致します。つまり LayerNorm の前段にある重み行列を何倍にしても出力は変わらない。この性質が「層ごとの活性化の絶対スケールに学習が左右されなくなる」効果を生み、後で述べる「効いているのは勾配側」という分析につながります。逆に言えば、LayerNorm の前の重みは大きさ方向の自由度を失い、方向だけが意味を持つようになります。

BatchNorm との違いを図で。同じテンソル `[B, L, D]` でも、どの方向の値をまとめて平均するかが両者で逆です。

```
テンソル [B=バッチ, L=トークン, D=特徴]

BatchNorm: 同じ特徴チャンネルを B方向(と必要なら L方向)にわたって平均
           → バッチ内の他データに依存。推論時は学習中の移動平均を使う
           例: D の 17番目の特徴を、B×L 個すべて集めて平均・分散

LayerNorm: 1トークンの中の D方向(256個)だけで平均
           → そのデータ1件で完結。バッチに非依存。学習と推論で挙動が同じ
```

なぜこの設計なのか。BatchNorm はバッチの統計に依存するため、(a) バッチサイズが小さいと平均・分散の推定が不正確になる、(b) 学習時はバッチ統計、推論時は学習中に貯めた移動平均、と挙動が変わる、(c) 系列長が可変だと統計が安定しない、という弱点を持ちます。LayerNorm はこれらを「バッチを参照しない」ことで一掃しました。代償として、画像のように「同じチャンネルの空間統計」を使えると有利なタスクでは BatchNorm に精度で劣る場合があります。

CNN の画像 `[B, C, H, W]`(C=チャンネル、H=高さ、W=幅)では、正規化の方向の取り方で派生が分かれます。Group Normalization(Wu & He, 2018)は C をいくつかのグループに分けてグループ内 `×H×W` で正規化し、バッチ非依存と空間統計の両取りを狙います。Instance Norm は1チャンネル `×H×W` ごと、ConvNeXt 等は「1ピクセルの C 方向」だけで取る LayerNorm を採用します。Vision Transformer ではパッチをトークン化して `[B, L, D]` にしてから、上で説明した素の LayerNorm を使うのが標準です。

Post-LN と Pre-LN。Transformer の各サブブロックは「正規化 → 演算 → 残差接続(入力を出力に足し戻す)」で構成されますが、正規化をどこに置くかで2流派あります。
```
Post-LN:  x → SubLayer → (+x) → LayerNorm     (残差を足した後に正規化)
Pre-LN:   x → LayerNorm → SubLayer → (+x)     (サブレイヤの前に正規化)
```
この置き場所が学習の安定性を大きく左右します(後述)。

### 手法の系譜と主要論文
- Ioffe & Szegedy「Batch Normalization」(2015, ICML)。正規化を学習に持ち込んだ元祖。内部共変量シフトを抑えるとして提案され、ImageNet 分類で学習を劇的に高速化しました。ただしバッチ依存という宿題を残しました。
- Ba, Kiros & Hinton「Layer Normalization」(2016, arXiv:1607.06450)。発想を90度回し、特徴方向で正規化。動機は、リカレントネットワーク(RNN、時系列を1ステップずつ処理)では時刻ごとに統計が変わり BatchNorm が扱いにくかったこと。RNN の学習を大きく安定・高速化しました。
- Vaswani ら「Attention Is All You Need」(2017, NeurIPS)。Transformer を提案。各サブレイヤの後に LayerNorm を置く Post-LN を採用し、LayerNorm を一気に主流へ押し上げました。
- Xu ら「Understanding and Improving Layer Normalization」(2019, NeurIPS)。LayerNorm の効きは平均・分散の「順伝播での再中心化・再スケール」よりも、それらの計算が生む「勾配の正規化」効果が大きいと分析。具体的には、正規化の中で平均・分散を計算する操作それ自体が逆伝播時に勾配へ作用し、大きすぎる勾配を抑え小さすぎる勾配を持ち上げる「勾配の再スケール・再中心化」を起こします。著者らは平均・分散の勾配を切り離す(止める)実験で、順伝播の正規化だけ残し勾配側の効果を消すと性能が落ちることを示し、「効いているのは勾配の正規化のほうだ」と主張しました。あわせて `γ, β`(学習可能なスケール・シフト)が過学習を招く場合があるとして、それを取り除いた AdaNorm を提案しています。
- Zhang & Sennrich「Root Mean Square Layer Normalization (RMSNorm)」(2019, NeurIPS)。平均を引く処理(再中心化)を省き、二乗平均平方根(RMS)で割るだけに簡略化。`y = x / RMS(x) ⊙ γ`。計算量を減らしつつ精度を保ち、近年の大規模言語モデルで広く採用されています。
- Xiong ら「On Layer Normalization in the Transformer Architecture」(2020, ICML)。Post-LN と Pre-LN を勾配の観点から理論解析。Post-LN は学習初期に出力層付近の勾配が大きくなり不安定で、学習率を徐々に上げる「ウォームアップ」が事実上必須。Pre-LN は勾配が層をまたいで安定し、ウォームアップなしでも学習できると示しました。

### 論文の実験結果(定量データ)
- 元論文(Ba et al., 2016)。機械翻訳・質問応答・言語モデリング・画像-文章マッチングなど RNN 系タスクで、BatchNorm より学習が速く収束する例を多数提示。特に系列長が可変なタスクで、BatchNorm がそのまま適用しづらい中、LayerNorm は素直に効きました。一方で畳み込み主体の画像分類では、当時の BatchNorm に精度・速度で及ばないと正直に報告しています(指標は分類精度。差の意味は「どちらが汎化的に正解率が高いか」)。
- Xiong et al.(2020)。WMT(機械翻訳ベンチマーク。指標は BLEU、訳文の n-gram 一致度で高いほど良い)と IWSLT で、Pre-LN はウォームアップを完全に省いても Post-LN(ウォームアップあり)に匹敵する BLEU を達成。Post-LN はウォームアップを外すと学習が発散して BLEU が大きく崩れることを示し、「Post-LN の不安定さは LN の置き場所が原因」という主張を定量的に裏づけました。学習の手間(ハイパーパラメータ調整の労力)が減る点が実務上の改善です。
- RMSNorm(Zhang & Sennrich, 2019)。複数の機械翻訳・言語モデリングタスクで、LayerNorm から平均計算を省いても BLEU やパープレキシティ(言語モデルの予測の当たりやすさ。低いほど良い)がほぼ同等で、正規化レイヤの実行時間をおよそ7〜64%削減できたと報告。アブレーションとして「再中心化(平均引き)を外しても性能が保たれるが、再スケール(RMS で割る)を外すと崩れる」ことを示し、効いているのはスケール調整のほうだと裏づけました。

これらの数値が示すのは、(1) 正規化の効きどころは順伝播の見かけより勾配側にある、(2) Transformer では「どこに置くか」が「置くこと自体」と同じくらい重要、という研究上の知見です。

### メリット・トレードオフ・限界
メリット
- バッチサイズに非依存。バッチ1でもオンライン推論でも同一の計算。
- 系列長(トークン数)が可変でも問題なく動く。Transformer/ViT と好相性。
- 学習と推論で挙動が変わらない(BatchNorm の移動平均切り替えが不要)。
- 数値スケールをそろえて学習を安定させ、深いネットワークを組みやすくする。

トレードオフ・限界
- 畳み込み主体タスクでは BatchNorm/GroupNorm のほうが精度・速度で有利な場面が残る。
- トークンごとに平均・分散を計算するためわずかに計算コストが増える(通常は無視できるが、メモリ帯域を食う「memory-bound」演算で、極端に多数のトークンを処理すると無視できなくなる)。
- Post-LN 構成では学習初期が不安定で、ウォームアップ等の工夫が要る(Pre-LN で緩和、ただし Pre-LN は最終性能が Post-LN にわずかに劣る報告もある)。
- 「内部共変量シフトを抑えるから効く」という当初の説明は、その後の研究(Santurkar et al., 2018 など)で「むしろ損失地形を滑らかにすることが本質では」と再解釈されており、なぜ効くのかは完全には決着していません。

### 発展トピック・研究の最前線
- RMSNorm の普及。近年の大規模言語モデルでは LayerNorm より RMSNorm が標準になりつつあります。簡略化で速度を稼ぎつつ精度を保てるためです。
- 正規化を「重み側」で行う流派。Weight Normalization(重みを方向と大きさに分解)や、正規化レイヤを完全に取り去って初期化と残差スケーリングで安定化を狙う研究(NFNet、DeepNet 等)もあり、「正規化は本当に必須か」という問いが続いています。
- Pre-LN と Post-LN のハイブリッドや、残差経路にスケール係数を学習させる手法(ReZero、LayerScale など)で、両者の良いとこ取りを狙う動きが活発です。LayerScale(Touvron ら, CaiT, 2021)は残差ブランチの出力に層ごとの学習可能な対角係数(初期値 1e-4 程度の極小値)を掛け、学習初期は各ブロックをほぼ恒等写像に近づけることで深い ViT の学習を安定させ、数十〜百層規模の ViT を学習可能にしました。DeepNet(Wang ら, 2022)は Post-LN を残しつつ残差を定数倍する DeepNorm を導入し、1000層級の Transformer を安定学習できると報告しています。
- 「なぜ正規化が効くのか」の理論研究。損失地形の平滑化(Santurkar ら, 2018, NeurIPS は BatchNorm の効果を内部共変量シフトではなく損失曲面の平滑化=リプシッツ性の改善で説明)、勾配のスケール不変性、暗黙の学習率調整(正規化のスケール不変性が実効学習率を動的に変える効果)といった複数の説明が提案され、研究テーマとして未解決のまま残っています。
- 「正規化なし」の流派。NFNet(Brock ら, 2021, ICML)は BatchNorm を完全に取り去り、Scaled Weight Standardization と Adaptive Gradient Clipping で安定化して ImageNet で当時の最高精度級に到達。正規化レイヤが「必須」ではなく「便利な道具の一つ」であることを示し、なぜ正規化が効くのかという問いを別角度から照らしました。

### さらに学ぶための関連トピック
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Loss Scaling / GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Sharpness-Aware Minimization (SAM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [LR Range Test (LR Finder)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Stochastic Weight Averaging (SWA)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)


## Linear LR Decay

### ひとことで言うと
学習率(モデルのパラメータを1回の更新でどれくらい動かすかを決める数値)を、まっすぐな直線で一定のペースで減らしていき、学習の終点でちょうど 0(または指定の最小値)に到達させるスケジューラ(学習率の時間変化のさせ方)です。仕組みはとても単純ですが、序盤に学習率を少しずつ上げる「warmup(ウォームアップ)」と組み合わせた「warmup + 線形減衰」が、BERT をはじめとする Transformer(自己注意機構を使う現代の言語・画像モデルの基本構造)の学習で事実上の標準になっています。

### 直感的な理解
車で目的地に着いて駐車する場面を思い浮かべてください。最初は速度を出し(大きな学習率)、目的地が近づくにつれてアクセルを緩め、最後はぴたっと 0 km/h で止まりたい。線形減衰は「ゴールまでの残り距離に比例して一定の割合で減速し、ゴール地点でちょうど停止する」という素直な減速の仕方です。

ただし Transformer には独特の事情があります。エンジンが冷えている(パラメータがランダムな初期状態の)ときに、いきなりフルスロットルを当てると暴走(損失が発散、つまり下がるどころか無限大へ飛ぶ)します。だから発進直後だけ、ゆっくりアクセルを踏み込んで暖機運転をする。これが warmup です。warmup で速度を目標値まで上げきってから、線形減衰でゴールまで一定ペースで減速していく。この「上って下る」三角形が warmup + linear decay です。

なぜ 0 にきっちり着地させたいのか。学習の終盤に学習率が残っていると、パラメータは谷底の周りで揺れ続け、最終モデルがたまたま揺れの端にいることになります。終点で 0 に着地させれば、谷底でぴたっと止まった重みが得られます。

### 基礎: 前提となる概念
- 学習率: 学習は「予測誤差を測り、誤差を減らす方向にパラメータを少し動かす」の繰り返しで、この「少し」の大きさが学習率です。大きすぎると発散、小さすぎると遅い。

- ステップ (step / iteration): データを1バッチ見てパラメータを1回更新する単位。Transformer の学習ではエポックより「総ステップ数」で語ることが多いです。数万〜数百万ステップ回します。

- 自己注意 (self-attention): 入力の各要素が、ほかのどの要素にどれだけ注目するかを学ぶ仕組みで、Transformer の心臓部です。学習初期はこの注意の重みがランダムなため、出力が極端な分布に偏りやすく、大きな学習率を当てると一気に壊れます。これが Transformer の「初期不安定性」です。

- 発散 (divergence): 損失が下がるどころか急激に増えて NaN(計算不能)になる現象。warmup はこの発散を防ぐための安全装置です。

- スケジューラの位置づけ。[Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) は理論上 0 に到達せず、[Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) は不連続な段差があります。線形減衰はこれらの中間で、「決めた終点でちょうど 0 に着地」かつ「途中はなめらかで一定ペース」という最も単純な下り坂です。

### 仕組みを詳しく
warmup を含めた全体は2本の直線でできています。

warmup 区間(0 から `warmup_steps` まで、学習率を上げる):
```
lr(t) = lr_peak * (t / warmup_steps)
```

減衰区間(`warmup_steps` から `total_steps` まで、学習率を下げる):
```
lr(t) = lr_peak * (total_steps - t) / (total_steps - warmup_steps)
```

- `lr_peak`: 頂点(warmup が終わった瞬間)の学習率
- `warmup_steps`: 何ステップかけて 0 から lr_peak まで上げるか
- `total_steps`: 学習全体の総ステップ数(ここで学習率が 0 になる)
- `t`: 現在のステップ数

数値例。`lr_peak = 5e-4`、`warmup_steps = 1000`、`total_steps = 10000` とすると、

- ステップ 0: 0
- ステップ 500(warmup 半分): 2.5e-4
- ステップ 1000(頂点): 5e-4
- ステップ 2000: 約 4.44e-4
- ステップ 5000(全体の半分あたり): 約 2.78e-4
- ステップ 10000(終点): 0

学習率の時間変化をASCII図にすると、左肩上がりの短い上り坂(warmup)のあと、長い直線の下り坂が続く三角形です。

```
lr
5e-4|     /\
    |    /  \__
    |   /      \__
    |  /          \__
    | /              \__
  0 |/                  \__
    +------------------------> step
     ↑warmup  ↑頂点      ↑total_steps で0
```

実装は複数あります。warmup なしの単純な線形減衰なら、開始倍率 1.0 から終了倍率 0.0 まで `total_iters` ステップかけて線形補間するスケジューラ(PyTorch の `LinearLR` 相当)。warmup と組み合わせるなら、warmup_steps と total_steps を渡すと2本の直線を1つにまとめてくれる関数(HuggingFace の `get_linear_schedule_with_warmup` 相当)が広く使われます。BERT 系の学習では後者がデファクトです。

線形減衰は [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)(多項式減衰)の power=1 の特殊ケースである、という関係も整理に役立ちます。多項式減衰の乗数を 1 にすると、まっすぐな直線になります。power を 2 にすれば終盤がゆっくり減る曲線、power を 0.5 にすれば序盤がゆっくり減る曲線になります。

なぜ線形なのか、コサインとの違い。コサイン減衰(cosine annealing)は終盤の減り方がゆるやかで、「最後にじっくり谷底を詰める」効果が強いと言われます。線形は終盤も一定ペースで 0 へ向かうため、その効果はやや弱い代わりに、挙動が直感的で、warmup の三角形と相性がよく、実装も最も単純です。タスクによってどちらが上かは変わり、決着はついていません。

### 手法の系譜と主要論文
- warmup の源流。Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, "Attention Is All You Need", NeurIPS 2017(arXiv:1706.03762)。オリジナルの Transformer 論文で、warmup の重要性を最初に強く示しました。提案した「Noam スケジュール」は、warmup 後に学習率をステップ数の平方根の逆数で下げるもので、純粋な線形ではありませんが、「warmup + 単調減衰」という枠組みの源流です。動機は自己注意の学習初期の発散を防ぐこと。新規性は、固定スケジュールに warmup を明示的に組み込んだ点。トレードオフは、平方根減衰が終点を 0 にきっちり決めにくいことで、これを後に BERT が分かりやすい線形へ置き換えました。

- 線形減衰の定着。Devlin, Chang, Lee & Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019(arXiv:1810.04805)。1万ステップの warmup のあと学習率を線形に 0 まで減衰させるスケジュールで BERT を事前学習しました。動機は Transformer の初期不安定性を warmup で抑え、終盤は線形に下げてきれいに収束させること。効果は、多数の言語理解ベンチマークで当時の最高精度を更新し、このスケジュールを業界標準にしたこと。トレードオフは、warmup_steps と total_steps を事前に決め打ちする必要があり、学習量を変えると再調整がいることです。

- warmup の理論的説明。Liu, Jiang, He, Chen, Liu, Gao & Han, "On the Variance of the Adaptive Learning Rate and Beyond"(RAdam, ICLR 2020, arXiv:1908.03265)。なぜ warmup が必要なのかを理論的に分析し、Adam 系最適化の初期は適応学習率の分散が大きく、それが発散の原因だと示しました。新規性は、warmup を内部に組み込んだ最適化手法 RAdam の提案。トレードオフは、線形 warmup ほど単純ではなく、実務では結局 warmup + linear/cosine が広く残っていることです。

- 画像系への波及。ConvNeXt(Liu et al., CVPR 2022)や多くの Vision Transformer 系は、AdamW + warmup + cosine/linear 系のスケジュールで学習されており、線形減衰系の知識は言語に限らず画像モデルの学習にも直結します。

### 論文の実験結果(定量データ)
- BERT(NAACL 2019)。GLUE(General Language Understanding Evaluation、9 個の言語理解タスクをまとめたベンチマーク)で、BERT-large は平均スコア 80.5 を記録し、当時の最良手法を約 7 ポイント上回りました。SQuAD v1.1(質問応答データセット、指標は F1 = 適合率と再現率の調和平均)では F1 93.2 を達成し人間の性能(91.2)を上回りました。これらは warmup + linear decay スケジュールのもとで得られた数値で、スケジュールが Transformer の安定学習に不可欠だったことを示します。

- warmup を抜くアブレーション。RAdam 論文(ICLR 2020)では、warmup なしの Adam が学習初期に大きく発散し、最終精度が大きく劣化することを示しました。warmup ありに比べ、機械翻訳(IWSLT データセット、指標は BLEU = 機械翻訳の n-gram 一致度を測るスコアで高いほど良い)で数 BLEU の差が出る設定が報告されています。逆に言えば「warmup の有無」だけで Transformer の成否が分かれることがある、という強いメッセージです。RAdam は warmup なしでもこの発散を内部で抑え、warmup あり Adam に近い精度を出せることを示しました。

- warmup 長さの感度。BERT 系の実務報告では、warmup を総ステップの 1〜10% 程度に取るのが定番で、短すぎる(0.5% 未満)と初期発散、長すぎる(20% 超)と学習が遅れて最終精度がやや落ちる、という傾向が知られています。この「warmup_steps を何にするか」は依然チューニング対象です。

- 線形 vs コサインの実証比較。多くのベンチマークで、適切にチューニングした線形減衰とコサイン減衰の最終精度差はごく小さい(多くの場合 1 ポイント未満)と報告されています。コサインは終盤の減りが緩やかで「最後の詰め」がわずかに有利な設定がある一方、学習を途中で止めた場合は線形のほうがその時点の学習率が低く、中断時の性能が安定するという実務的利点も指摘されています。総ステップ数を最後まで使い切る前提ならコサイン、途中で止めうるならスケジュールの形を選び直せる柔軟さが重要、という整理になります。

- ファインチューニングでの線形減衰。BERT を下流タスクに適応させる(fine-tuning)際も、warmup 約 10% + 線形減衰が定番です。GLUE の小規模タスク(数千〜数万例)では学習が数エポックで終わるため、減衰の終点を 0 に合わせて「最後はほとんど更新しない」状態にすることが、小データでの過学習を抑える効果も持ちます。

指標の意味の補足。BLEU は 0〜100 で、数ポイントの差でも翻訳品質の体感差が出ます。GLUE 平均や SQuAD F1 も、終盤の 1 ポイントを詰めるのが難しい領域なので、スケジュールの違いで生まれる数ポイントの差は実用上大きな意味を持ちます。

### メリット・トレードオフ・限界
メリット
- 仕組みが直線2本だけで極めて単純、かつ終点でちょうど 0 に着地する。
- warmup と組み合わせることで Transformer の初期不安定性を抑え、発散を防げる。
- BERT 以降の膨大な実績があり、Transformer 学習の安全な既定値として信頼できる。

トレードオフ・限界
- warmup_steps と total_steps を事前に決め打ちする必要があり、学習量を変えると再設定がいる(適応はしない。[ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) と対照的)。
- 単調に減るだけで、局所解・鞍点からの脱出機能はない([Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) のような再起動はできない)。
- 終盤に学習率が直線的に 0 へ近づくため、コサイン減衰のように「終盤をゆっくり下げて精度を詰める」効果はやや弱い(タスクによりコサインのほうが最終精度が上という報告もある)。
- warmup の長さが短すぎると発散、長すぎると学習が遅れる、というシビアさがある。
- 研究上の未解決点として、warmup と減衰のスケジュールを最適化問題として原理的に導く理論はまだ十分でなく、多くは経験則です。

### 発展トピック・研究の最前線
- one-cycle / super-convergence。Smith & Topin(2018)は、学習率を一度大きく上げてから下げる「1サイクル」スケジュールで、通常より大幅に少ないステップ数で高精度に到達できる「super-convergence」を報告しました。線形 warmup の発展形と見なせます。
- WSD スケジュール(warmup-stable-decay)。大規模言語モデルの学習で、warmup の後に学習率を長く一定に保ち、最後の 10〜20% だけ急減衰させる方式が近年注目されています(MiniCPM, Hu et al. 2024 などで採用)。長い定常区間のあいだは total_steps を確定させずに学習を続けられ、いつでも「最後の減衰区間」を後付けして打ち切れるため、学習量を事前に決め打ちしなくてよいのが利点です。線形/コサイン減衰の「終点を最初に決める」という制約を外した発展形と言えます。
- 学習量に依存しないスケジュール。total_steps を事前に決めずに済むスケジュール(無限学習向けの一定学習率 + 後付け減衰など)が、学習を途中で止めたり延長したりしたい大規模学習の文脈で研究されています。
- スケーリング則との結びつき。モデル規模・データ量と最適学習率・warmup 長の関係(Chinchilla 系の研究)が、スケジュール設計を経験則から理論へ近づけつつあります。

### さらに学ぶための関連トピック
- [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Lion (EvoLved Sign Momentum)

### ひとことで言うと
Lion(ライオン、EvoLved Sign Momentum の略)は、重みを動かす「向き」を符号関数(sign、プラスかマイナスかだけを取り出す関数)だけで決める、とてもシンプルでメモリ効率の良いオプティマイザです。Adam/AdamW がパラメータごとに2つの状態(`m` と `v`)を持つのに対し、Lion は1つ(`m` だけ)で済むので、オプティマイザのメモリを約半分に減らせます。それでいて大規模モデルでは AdamW を上回る性能を出すことが報告されています。さらにこの更新式は、人間が設計したのではなく、機械(プログラム探索)が自動で発見したものだという点でも画期的でした。

### 直感的な理解
Adam は、各重みについて「どっちに進むか(向き)」と「どれくらい進むか(歩幅)」の両方を精密に計算します。歩幅の計算には2次モーメント `v`(勾配二乗の移動平均)が要り、これが重み1個ぶんの追加メモリを食います。

ここで素朴な問いが立ちます。「向きさえ正しければ、歩幅は全部一律でもいいのではないか?」 Lion の答えは「その通り」です。Lion は更新の向きだけを符号(+1 か −1 か 0)で決め、大きさは全パラメータ一律に学習率 `η` ぶんだけ動かします。歩幅を計算する `v` がいらなくなるので、状態は `m` ひとつで済み、メモリが半分になります。

たとえるなら、Adam は「各方向ごとに地面の硬さを測って踏み込む力を調整する慎重な歩き方」、Lion は「向きだけ見て一定の歩幅でリズミカルに歩く」やり方です。直感的には大ざっぱに思えますが、ミニバッチの勾配にはどのみちノイズが乗っているので、勾配の大きさまで信じて細かく歩幅を決めることに大きな意味はない、という見方もできます。意外なことに、巨大なモデルを大きなバッチで学習する場面では、後者(Lion)のほうが速く・安く・良い結果になることがあると報告されています。

### 基礎: 前提となる概念
- 符号関数 sign(x): 中身が正なら +1、負なら −1、ちょうど 0 なら 0 を返す関数。大きさ情報を捨てて向きだけ残す。例えば sign(3.2) = +1、sign(−0.001) = −1。
- 1次モーメント `m`: 勾配の移動平均。過去の進行方向を覚えておく「慣性」。[Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の `m` と同じ役割だが、Lion では更新の仕方が違う。
- 2次モーメント `v`: 勾配二乗の移動平均。Adam が歩幅を決めるのに使う。Lion はこれを持たない(ここがメモリ半減の正体)。
- weight decay(分離型): [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) と同じく、重みを直接 `λ·w` ぶん縮める正則化。Lion も分離型を採用。
- プログラム探索(symbolic program search): オプティマイザの更新式を「プログラム(命令の組み合わせ)の空間」とみなし、進化的アルゴリズム(良いものを選んで少しずつ変異させ世代を重ねる探索)で良い式を自動的に探す手法。Lion はこれで発見された。
- signSGD: 勾配の符号だけで重みを更新する素朴な手法。Lion の先祖にあたる。

### 仕組みを詳しく
Lion の更新式は次の通りです。`m` は唯一の状態、`sign()` は符号関数、`β1`(通常 0.9)、`β2`(通常 0.99)、`λ` は weight decay 係数です。

```
c ← β1 · m + (1 − β1) · g           ← 更新方向の候補(現在の勾配を多めに混ぜる)
w ← w − η · ( sign(c) + λ · w )      ← sign で向きだけ取り、一律の歩幅 η で更新 + weight decay
m ← β2 · m + (1 − β2) · g            ← モーメントは別の係数 β2 で更新(次のステップ用に保存)
```

ポイントを噛み砕きます。

- `sign(c)` により更新の各要素は +1 か −1 か 0 になります。すべての重みが、向きはそれぞれですが大きさは一律 `η` で動きます。Adam のような `√v` でのパラメータごとの歩幅調整がありません。
- 状態は `m` ひとつだけ。`v` がないのでメモリは Adam/AdamW の約半分です。
- 更新方向 `c` を作るときの混合比 `β1` と、状態 `m` を更新するときの `β2` が別の値である点が非自明です(Adam は1次モーメントに 1 つの β しか使いません)。これにより「いま進む方向 `c` には直近の勾配を強めに反映しつつ、状態 `m` にはもう少し長期の履歴を残す」挙動になります。これも探索で見つかった構造で、人間の直感では出にくいものでした。実質的に Nesterov 加速(先読みしてから勾配を測る)に似た効果を持つと解釈されています。
- weight decay は AdamW と同じく分離型(`λ·w` を直接かける)で、勾配には混ぜません。

#### 数値例で効果を見る
`η = 0.0001`(Lion は AdamW より小さい学習率を使うのが定石)とします。あるステップで `c` の各要素が `[2.3, −0.01, 0.0, 5.0]` なら `sign(c) = [+1, −1, 0, +1]`。更新量は `[−0.0001, +0.0001, 0, −0.0001]`(weight decay 前)。値の大小(2.3 でも 5.0 でも)に関係なく、向きだけ見て一律 ±0.0001 動く、というのが Lion の挙動です。一方 Adam なら 5.0 の勾配と 0.01 の勾配で実効的な扱いが変わりますが、Lion は両者を「正の方向」としてのみ区別します。

この「一律の歩幅」は更新の各要素のノルムが必ず 1 になるため更新が暴れにくく安定する反面、実効的な更新量のスケールが Adam と違います。経験則として、Lion の学習率は AdamW の 3〜10 分の 1、weight decay は逆に 3〜10 倍にすると良いとされます。理由は、Adam の更新量が `m̂/√v̂` で平均的に 1 未満に収まりやすいのに対し、Lion の更新量は要素ごとに必ず大きさ 1(`η` ぶん)になり、実効的な更新がやや大きく出やすいためです。学習率を下げてバランスを取り、その分だけ正則化を相対的に強める、というのが定石になります。

#### メモリ比較(具体)
```
              追加状態        例: 70億パラメータ・fp32 のオプティマイザ状態
AdamW   :   m + v(2倍)        約 56GB
Lion    :   m のみ(1倍)        約 28GB
```
オプティマイザ状態のメモリが半分になるため、同じ GPU でより大きなバッチやモデルを載せられます。さらに `m` を bfloat16(16 ビット浮動小数点)で持てる実装では一段と削れます。大規模学習でこの差は決定的に効きます。

#### なぜ sign で安定するのか
sign 更新は、勾配の絶対値が大きい外れ値(まれに来る巨大な勾配)に引きずられません。Adam では巨大勾配が `m` を大きく動かしますが、Lion は符号しか見ないので 1 ステップで暴走しにくい。これは勾配クリッピング(勾配が大きすぎたら頭打ちにする処理)と似た安定化効果を内蔵していると見ることもできます。

### 手法の系譜と主要論文
- 手設計の系譜: SGD → Momentum → AdaGrad → RMSProp → Adam → AdamW と、人間が数式を考えて改良してきた歴史があります([Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。Lion はこの「人手設計」というパラダイム自体を問い直しました。
- sign ベースの先行研究: signSGD(Bernstein et al., ICML 2018)は勾配の符号だけで更新する手法で、分散学習(複数マシンで並列に学習)の通信量削減(符号は 1 ビットで送れる)で注目されました。Momentum 付きの signum も同論文系で提案されています。Lion はこの系譜に Momentum と分離 weight decay、そして二重 β を加えたものと位置づけられます。
- Lion — Chen et al. (NeurIPS 2023, "Symbolic Discovery of Optimization Algorithms", arXiv:2302.06675, Google)。オプティマイザの更新プログラムを進化的探索で自動発見し、何千もの候補プログラムを ViT などの小規模代理タスクで評価して選別、見つかった最良の式を簡約化して Lion と命名しました。「機械がアルゴリズムを発見する」AutoML 的なアプローチの代表例です。論文では、探索が複雑な式を見つけたあと、それを人間が解釈・簡約できる形に削っていく過程(program simplification)も重視されています。

### 論文の実験結果(定量データ)
Chen et al. (2023) は幅広いタスクで Lion と AdamW を比較しました。主な報告は次の通りです。

- ViT の ImageNet 分類: ViT(Vision Transformer)を ImageNet(約 128 万枚の画像を 1000 クラスに分類する標準ベンチマーク)で学習し、Lion は AdamW と同等以上の top-1 精度(正解クラスが予測の最上位に来る割合)を達成。特に大規模 ViT(ViT-G/14 など)を JFT(Google 内部の約 30 億枚規模の巨大画像データセット)で事前学習し ImageNet に転移(fine-tune)した設定で、Lion が AdamW を上回る精度を報告しました。論文では大規模 ViT で ImageNet の top-1 精度が約 1 ポイント程度上振れする結果が示されています(例えば ViT-G で AdamW を約 +1% 上回るなど)。
- 言語モデル(自己回帰 LM、マスク言語モデル): 同等以上の精度を、より少ない学習計算量(ステップ数や FLOPs)で達成したと報告。事前学習の perplexity(パープレキシティ、言語モデルが次の語をどれだけ当てられるかの指標、低いほど良い)で AdamW に匹敵または改善し、一部設定で AdamW より少ない計算量で同じ品質に到達したとしています。
- 拡散モデル(画像生成): FID(Fréchet Inception Distance、生成画像と本物画像の分布の近さ、低いほど良い)で AdamW を上回るケースを報告しました。
- 計算・メモリ効率: 状態が `m` のみのためメモリが約半分。さらに更新が sign 中心で軽い(平方根や除算がない)ため、大規模学習で実時間あたりのスループットが改善するとしました。論文は「規模が大きいほど、バッチが大きいほど、データが多いほど Lion の優位が出やすい」と総括しています。

アブレーション・限界に関する報告と後続の知見:
- 二重 β の重要性: `β1` と `β2` を同じ値にすると(=Adam 的な単一モーメントに退化させると)性能が落ちることが示され、更新方向 `c` と状態 `m` を別係数で扱う構造が効いていることが確認されています。
- 学習率と weight decay のスケールが AdamW と大きく違い、AdamW の設定をそのまま流用すると不安定になります。再調整が必須で、学習率は AdamW の 3〜10 分の 1 が目安です。
- sign による一律更新はバッチサイズが小さいと不安定になりやすく、大バッチ前提の傾向があります。小バッチでは勾配ノイズが相対的に大きく、符号がころころ変わって更新が荒れるためです。論文も大バッチ(数千〜)で優位が顕著と報告しています。
- 小規模モデルや一部タスクでは AdamW に対する優位が出ず、効果は規模・タスク依存です。コミュニティの追試では「Lion が常に勝つわけではない」「データセットやモデルによっては同等か微減」という報告も多くあります。
- 探索で見つかった式のため、なぜ効くかの理論的理解は後追いで、完全には解明されていません。後続研究では「sign 更新が暗黙の正則化(implicit regularization)として働き、平坦な極小値(flat minima、少しずれても loss があまり増えない=汎化しやすい谷)に導く」という解釈が提案されています。

### メリット・トレードオフ・限界
メリット
- オプティマイザ状態が `m` 1 つだけで追加メモリが AdamW の約半分。大規模モデル・大バッチで効く。
- 更新式が sign と単一モーメントだけで非常にシンプル。除算・平方根がなく計算も軽い。
- ViT・拡散モデル・言語モデルで AdamW と同等以上の精度を報告。
- weight decay は分離型で AdamW の正しい正則化の利点を引き継ぐ。
- sign 更新が外れ値勾配に強く、暗黙の安定化効果を持つ。

トレードオフ・限界
- 学習率・weight decay のスケールが AdamW と大きく異なり、再調整しないと不安定・発散しやすい。
- 小バッチでは sign 更新が不安定になりやすく、大バッチ前提。
- 効果が規模・タスク依存で、小規模では AdamW に勝てないこともある。
- 探索由来の式のため理論的理解が後追いで、挙動の予測が難しい面がある。

未解決の課題として、「どの条件(モデル規模・バッチ・データ量)で Lion が AdamW を上回るか」を事前に判定する一般則は確立していません。実務では「まず AdamW で動かし、大規模化でメモリが厳しくなったら Lion を試し、学習率を 1/3〜1/10 にして weight decay を 3〜10 倍にする」という使い分けが現実的です。

### 発展トピック・研究の最前線
- AutoML によるアルゴリズム発見: Lion は「学習則そのものを機械が発見する」流れの象徴で、損失関数・データ拡張・正則化・アーキテクチャの自動発見研究(AutoML-Zero など)と地続きです。今後さらに高性能な更新則が自動発見される可能性があります。
- sign 系オプティマイザの理論: sign 更新の暗黙正則化、平坦な極小値との関係、分散学習での通信効率(符号は 1 ビットで送れるので勾配通信を 32 分の 1 に圧縮できる)など、理論・システム両面で研究が進んでいます。sign 更新は L∞ ノルムに関する最急降下とみなせる、という最適化理論からの解釈もあります。
- メモリ効率オプティマイザの競争: Lion(状態半減)、Adafactor(`v` を行・列に因子分解)、8-bit Adam(量子化)、GaLore(勾配を低ランク部分空間に射影して状態を圧縮)、Adam-mini(`v` を層ごとに 1 つにまとめる)など、巨大モデル学習のメモリ削減は活発な研究領域です。Lion はこの競争に「2次モーメントを完全に捨てる」という最も大胆な選択肢を提示しました。

### さらに学ぶための関連トピック
- [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## LR Range Test (LR Finder)

### ひとことで言うと
LR Range Test(学習率レンジテスト、LR Finder とも呼ぶ)は、ニューラルネットワークの「学習率」をどれくらいに設定すればよいかを、短い実験で素早く見つける方法です。学習率をとても小さい値からだんだん大きくしていきながら、数百ステップだけ学習を回し、損失(誤差)がどう変化するかを観察します。損失が一番よく下がっている学習率の範囲をグラフから読み取り、本番の学習率やスケジューラの設定値として使います。

### 直感的な理解
学習率の調整は、深層学習で最初にぶつかる、しかし最も性能を左右する作業の一つです。例えるなら、目隠しをして谷を下りるとき「一歩の大きさ」を決める問題に似ています。歩幅が小さすぎると谷底まで永遠にたどり着けません(学習が遅すぎる)。歩幅が大きすぎると谷を飛び越えて反対側の斜面に着地し、行ったり来たりして発散します(損失が NaN に飛ぶ)。ちょうど良い歩幅は、速く・安定して谷を下れる絶妙な範囲にあります。

問題は、この「ちょうど良い歩幅」がモデル・データ・バッチサイズ・最適化アルゴリズムの組み合わせごとに毎回変わることです。あるモデルで最適だった学習率が、層構成を変えたり別のデータセットに移ったりすると、もう最適ではありません。だから毎回探し直す必要があります。

従来はこれを「グリッドサーチ」、つまり `1e-2, 1e-3, 1e-4, 1e-5` をそれぞれ最後まで学習して結果を比べる方法で探していました。しかしこれはフル学習を何度も繰り返すので非常に重い作業です。LR Range Test の発想は「最後まで学習しなくても、学習率を上げながら数百ステップ走らせるだけで、損失の挙動から良い範囲が見える」というものです。1回の短い実験で済むので、グリッドサーチより桁違いに速い、というのが動機です。

### 基礎: 前提となる概念

学習率(learning rate、略して lr)は、1回の更新でパラメータをどれだけ大きく動かすかを決める係数です。勾配(損失を下げる方向)に学習率を掛けた量だけパラメータを動かします。深層学習で最も重要なハイパーパラメータ(人が事前に決める設定値)の一つです。

損失(loss)は、モデルの予測が正解からどれだけずれているかを表す1つの数値で、学習はこの値をできるだけ小さくする作業です。学習率が適切なら損失はなめらかに下がり、大きすぎると損失が暴れて発散します。

スケジューラ(learning rate scheduler)は、学習の進行に合わせて学習率を変化させる仕組みです。最初は固定でも学習はできますが、学習が進むほど小さくしていく(減衰)と最終的な精度が上がることが知られています。LR Range Test は、このスケジューラの「ピーク学習率」や「上限・下限」を決めるための事前調査として最も価値を発揮します。

ウォームアップ(warmup)も関連概念です。学習開始直後はパラメータが不安定なので、最初の数百〜数千ステップは学習率を小さい値から目標値まで徐々に上げる工夫で、大バッチ学習(Goyal et al., 2017)では事実上の標準になっています。LR Range Test の「学習率を徐々に上げる」操作はウォームアップと形が似ていますが、目的が異なります(片や調査、片や安定化)。

### 仕組みを詳しく

手順は次の通りです。

1. 学習率をとても小さい値(たとえば `1e-7`)から始める。
2. 1ステップ(または数ステップ)学習するたびに、学習率を少しずつ大きくしていく。指数的に増やすのが一般的で、たとえば毎ステップ一定倍率(1.05〜1.1 倍など)を掛けて、最終的に大きな値(たとえば `1` や `10`)まで上げる。これを数百ステップ続ける。線形に増やす変種もある。
3. 各ステップでの「学習率」と「損失」を記録する。損失はノイズが乗るので、指数移動平均で平滑化してから記録することが多い。
4. 横軸を学習率(対数スケール)、縦軸を損失にしてグラフを描く。

このグラフは典型的に次の形になります。

```
損失
 |  ＼                              ／  ← ここで発散(学習率が大きすぎ)
 |    ＼                          ／
 |      ＼____________＿＿＿＿＿／
 |          急に下がる   平ら
 |________________________________________→ 学習率(対数)
   小さすぎ  良い範囲      大きすぎ
```

読み取り方:
- 学習率が小さすぎる左側では、損失はほとんど下がらない(歩幅が小さすぎて動かない)。
- ある点から損失が急に下がり始める。ここが「学習が効き始める」境界。
- さらに上げると損失の下がりが鈍り、やがて急上昇する。急上昇する直前が「これ以上上げると壊れる」上限。
- 良い学習率の選び方には流儀があり、(a)「損失が最も急に下がっている点」(損失曲線の傾きが最も負になる学習率)を選ぶ、(b)「損失が最小になる学習率の1桁ほど手前(およそ1/10)」を選ぶ、という二つがよく使われます。後者は、最小点はすでに発散の手前で危ういため、安全マージンを取る考え方です。

数値例: テストの結果、学習率 `3e-3` あたりで損失の下がりが最も急で、`1e-1` を超えると損失が跳ね上がって発散したとします。すると本番の最大学習率は安全をみて `1e-3 〜 3e-3` 程度に設定する、というように決められます。`3e-3` は「3 × 10⁻³ = 0.003」を意味し、`1e-1` は 0.1 です。学習率は桁(オーダー)で効くので、対数スケールで考えるのが基本です。

このテストの結果は、固定学習率を決めるだけでなく、スケジューラの設計に直結します。代表例が同じ Smith の One-Cycle(ワンサイクル)です。LR Range Test で見つけた上限をピーク学習率にして、学習中に学習率を「低→高(ピーク)→さらに低」と1回大きく動かすスケジュールを組みます。前半で大きな学習率を使うと、損失地形の鋭い谷を避けて平らな領域を探索しやすくなり、最後に学習率を絞り込んで谷底に収束させる、という理屈です。これにより通常より少ないエポックで高精度に達する「Super-Convergence(超収束)」が報告されています。

注意点として、LR Range Test の最中の重みは捨てます。学習率を発散域まで意図的に上げて壊すので、本番学習はテスト後に最初からやり直します。テストはあくまで「どの学習率が安全で効果的か」を測る計測専用の実験です。

### 手法の系譜と主要論文

Leslie N. Smith「Cyclical Learning Rates for Training Neural Networks」(2017, WACV、米海軍研究所、arXiv:1506.01186)。LR Range Test を初めて明文化した論文です。本来の主題は「学習率を下限と上限の間で三角波状に周期的に上下させる Cyclical Learning Rate(CLR)」で、その上下の範囲を決める前準備として LR Range Test を導入しました。提案理由は、最適な学習率範囲をフル学習なしで見つけるためです。周期的に学習率を上げることで鞍点(勾配がほぼゼロで学習が止まりやすい平坦な場所)を脱出しやすくなり、収束が速くなると示しました。

Smith, Topin「Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates」(2019、arXiv:1708.07120)。LR Range Test で見つけた上限を使う One-Cycle スケジュール(学習率を一度大きく上げてから下げ、最後に非常に小さくする。同時にモメンタムを逆位相で動かす)を提案。効果は通常の数分の一のエポックで同等以上の精度に達する超収束。トレードオフは、大きな学習率を使うため設定を誤ると発散しやすく、LR Range Test での上限見極めが特に重要になることです。

Loshchilov, Hutter「SGDR: Stochastic Gradient Descent with Warm Restarts」(2017, ICLR、arXiv:1608.03983)。学習率をコサインカーブで滑らかに下げ、周期の終わりで一気に上限へ戻す(ウォームリスタート)手法を提案。LR Range Test とは別手法ですが、「学習率を動的に動かして汎化と収束を改善する」という同じ流れに属し、適切な最大学習率を事前に知るうえで LR Range Test 的な調査が役立ちます。

Goyal et al.「Accurate, Large Minibatch SGD」(2017、arXiv:1706.02677)。バッチサイズに比例して学習率を上げる「線形スケーリング則」と、開始直後のウォームアップを組み合わせ、ImageNet を非常に大きなバッチで短時間学習できることを示しました。LR Range Test と直接の手法ではありませんが、最適学習率がバッチサイズに依存することを定量的に裏づけ、「設定が変われば学習率を測り直す」という LR Range Test の前提を補強します。

### 論文の実験結果(定量データ)

Smith の一連の研究で報告された定量的な効果を、指標の意味とともに整理します。

- CLR(2017)では、CIFAR-10 を ResNet などで学習する際、固定学習率では約7万回の反復(iteration)を要した精度に、三角波の CLR は約2.5万回で到達したと報告されています。指標は「ある目標テスト精度に到達するまでの反復回数」で、これが少ないほど学習が速いことを意味します。固定学習率と同等以上の最終精度を、明らかに少ない反復で達成しました。
- Super-Convergence(2019)では、CIFAR-10 を One-Cycle で学習すると、通常のスケジュールが数百エポックを要するところを、ごく少数のエポック(報告では従来の数分の一)で同等の精度に到達できたと示されました。特に訓練データが少ない設定ほど超収束の効果が大きいことも観察されています。
- これらの実験に共通する重要な含意は、「LR Range Test で測った上限ぎりぎりの大きな学習率を使うこと自体が一種の正則化として働く」という点です。大きな学習率は鋭い谷への落ち込みを妨げ、結果として汎化が改善する方向に作用します。

アブレーション的な観察として、One-Cycle のピーク学習率を LR Range Test の発散点より十分大きく取ると発散し、小さく取りすぎると超収束が起きない、という「上限の見極めが成否を分ける」感度の高さが報告されています。これが LR Range Test を事前に回す実用的な動機になります。なお、グラフの読み取りに主観が入ること自体は弱点でもあり、後年は損失曲線の傾きが最小になる点を自動検出する実装(各種ライブラリの LR Finder)が普及して、再現性を高めています。

### メリット・トレードオフ・限界

メリット
- フル学習を何度も繰り返さずに、1回の短い実験(数百ステップ)で良い学習率範囲が分かる。
- グリッドサーチより桁違いに速く、計算資源を節約できる。
- One-Cycle やコサイン減衰など強力なスケジューラのピーク学習率を、勘ではなく根拠を持って設定できる。
- モデル・データ・バッチサイズが変わるたびに再実行すれば、その都度適切な範囲を得られる。

トレードオフ・限界
- グラフの読み取りに多少の経験・主観が要る(どこを最急降下点や安全上限とみなすか)。自動検出ライブラリで緩和できるが万能ではない。
- 結果はバッチサイズ・データ・最適化アルゴリズム(SGD か Adam か)に強く依存し、設定が変われば再実行が必要。
- あくまで初期範囲の目安であり、最終的な微調整は別途必要になることがある。
- テスト用に学習率を発散域まで上げるため、その間の重みは捨てる。テスト後に本番学習を最初からやり直すコストが別途かかる。
- 研究上の論点として、Adam 系の適応的最適化では各パラメータが内部で実効学習率を調整するため、LR Range Test の解釈が SGD ほど単純でないことが指摘されています。また Transformer の事前学習のような長大な学習では、初期の短いテストが全体最適を保証しない場合があります。

### 発展トピック・研究の最前線

LR Range Test を出発点とする学習率設計は、近年さらに自動化が進んでいます。スケジューラの選択自体(One-Cycle / コサイン減衰 / 線形減衰 / 定数+ウォームアップ)をベンチマークで比較する研究が積み重なり、大規模 Transformer ではウォームアップ付きコサイン減衰が事実上の標準になりました。一方で、近年は学習率スケジュールの形をできるだけ事前に決めずに済ませる方向の研究も活発で、学習終盤に学習率を急減衰させる WSD(Warmup-Stable-Decay)スケジュールや、スケジュール自体を不要にする schedule-free optimizer の提案が出ています。これらはいずれも「ピーク学習率をどう決めるか」という LR Range Test が扱った問題意識の延長にあり、最適学習率の探索は今なお最適化研究の中心テーマであり続けています。

### さらに学ぶための関連トピック
- [Stochastic Weight Averaging (SWA)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Sharpness-Aware Minimization (SAM)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Layer Normalization](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Mixup

### ひとことで言うと
Mixup は、2枚の訓練画像を「半透明で重ね合わせる」ように混ぜて新しい画像を作り、正解ラベルも同じ割合で混ぜて学習する、データ拡張(訓練データの水増し)の手法です。モデルが「データとデータの間(中間)」でも素直に振る舞うようになり、過学習やノイズへの脆さが減ります。実装は数行で済むほど軽量なのに、画像分類の標準レシピに組み込まれるほど効果が安定している点が特徴です。

### 直感的な理解
ふつうの学習では、モデルに「この画像は100%猫」「この画像は100%犬」とだけ教えます。すると、モデルは訓練データのまさにその点だけで自信満々に答えるようになりますが、その「間」では予測が不安定になりがちです。たとえば猫と犬の中間的な見た目に対して、極端で過信した予測を返したり、わずかなノイズで答えが大きく変わったりします。これは過学習の一形態で、入力にほんの少しノイズを加えただけで誤分類する脆さ(敵対的サンプルへの弱さ)の原因にもなります。

Mixup のアイデアは「データの間も学習で埋めてしまえ」です。2枚の画像を線形に混ぜた中間画像を作り、ラベルも同じ割合で混ぜます。「70%猫 + 30%犬の画像なら、答えも70%猫・30%犬」と教えるのです。こうすると、モデルは「入力がなめらかに変われば出力もなめらかに変わる」という素直な(線形に近い)振る舞いを学びます。データ点と点の間を直線で補間したような領域まで学習が及ぶため、汎化が上がり、ノイズや破損データへの頑健性も改善します。

なぜ「ラベルも混ぜる」ことが重要かを直感で言うと、画像だけを混ぜて「これは100%猫」と教えると、犬が30%写っているのに猫と断言させることになり、矛盾した過酷な要求になります。ラベルも30%犬に混ぜることで、画像の中身とラベルが整合し、モデルは無理なく中間を学べます。

### 基礎: 前提となる概念
- データ拡張 (data augmentation): 手持ちの訓練データに加工を施して、見かけ上のデータ量を増やすことです。画像の左右反転、回転、明るさ変更などが典型です。データが多いほどモデルは過学習しにくくなります。
- ラベル (label): そのデータの正解です。画像分類なら「これは猫」「これは犬」。
- ワンホット (one-hot): 正解クラスだけ 1、他は 0 にしたベクトルです。3クラスで2番目が正解なら `[0, 1, 0]`。
- ソフトラベル (soft label): ワンホットのように0か1ではなく、`[0.7, 0.3]` のように中間的な確率を持つラベルです。Mixup が作るのはこのソフトラベルです。
- 経験的リスク最小化 (Empirical Risk Minimization, ERM): 手元の訓練データ上での損失を最小化する、標準的な学習の定式化です。Mixup はこれを拡張した近傍リスク最小化(VRM、後述)に基づきます。
- 決定境界 (decision boundary): モデルがクラスを分ける境目のことです。境界が鋭く複雑だと過学習しやすく、なめらかだと汎化しやすい傾向があります。
- ベータ分布 (Beta distribution): 0から1の間の値を生成する確率分布です。Mixup の混合比をこの分布から引きます。パラメータ `α` が小さいほど0や1の端に偏り、大きいほど0.5付近に集まります。

### 仕組みを詳しく
2つの訓練サンプル `(画像_i, ラベル_i)` と `(画像_j, ラベル_j)` を取り、混合比 `λ`(ラムダ、0〜1)で線形補間します。

```
混合画像  = λ * 画像_i + (1 - λ) * 画像_j
混合ラベル = λ * ラベル_i + (1 - λ) * ラベル_j
```

`λ` は毎回ベータ分布 `Beta(α, α)` からランダムに引きます。`α`(アルファ)はハイパーパラメータで、典型値は 0.2〜0.4 です。`α` が小さいと `λ` は 0 か 1 に近い値(ほぼ片方の画像)になりやすく、大きいと 0.5 付近(半々の混合)になりやすい性質があります。つまり `α` は「どれくらい強く混ぜるか」のつまみです。`α → 0` の極限では Mixup は通常の学習(ERM)に戻ります。

数値例を見ます。

- 画像_i: 猫の画像。あるピクセルの RGB が `[200, 180, 150]`、ラベル `[1, 0]`(猫)。
- 画像_j: 犬の画像。同じ位置のピクセルが `[60, 90, 40]`、ラベル `[0, 1]`(犬)。
- `λ = 0.7` を引いたとします。

```
混合ピクセル = 0.7 * [200,180,150] + 0.3 * [60,90,40]
             = [140,126,105] + [18,27,12]
             = [158,153,117]
混合ラベル   = 0.7 * [1,0] + 0.3 * [0,1] = [0.7, 0.3]
```

見た目は猫が濃く犬がうっすら透けた画像になり、「70%猫・30%犬」というソフトラベルを学びます。

テンソル形状の観点では、画像バッチの形状が `[B, 3, H, W]`(B枚、3チャンネル、高さH、幅W)のとき、実装上はバッチをシャッフルしたコピーを作り、元バッチと混ぜます。

```
index = ランダムにシャッフルした並び
mixed = λ * x + (1 - λ) * x[index]     # 形状は [B, 3, H, W] のまま
```

ラベルも `[B, クラス数]` のまま同様に混ぜます。損失計算では、2つのラベルそれぞれの交差エントロピー損失を `λ` と `1-λ` で重み付けして足す実装が一般的で、これはソフトラベルに対する交差エントロピーと数値的に等価になります。バッチ全体で同じ `λ` を1つ引く実装が標準ですが、サンプルごとに別々の `λ` を引く実装もあり、後者の方が多様性が増します。

なぜこの設計が効くのかは、近傍リスク最小化(Vicinal Risk Minimization, VRM)という枠組みで説明されます。通常の ERM は訓練データの各点だけで損失を測りますが、VRM は各データ点の「近傍」も含めて損失を測ります。Mixup は、2つのデータ点を結ぶ線分上を仮想的な訓練データとみなすことで、この近傍を線形補間で定義しています。結果として、モデルにはデータ点の間で線形にふるまうという帰納バイアス(事前の好み)が入り、決定境界がなめらかになります。後の解析(Carratino らの JMLR 2022 など)では、Mixup が入力へのデータ依存的なノイズ注入と、ヤコビアン(出力の入力に対する感度)・ヘシアンへの罰則の組み合わせとして近似的に展開できることが示され、なぜ汎化とロバスト性が同時に上がるのかが理論的に整理されつつあります。

重要なのは、Mixup は学習時にだけ適用し、推論時には適用しないことです。あくまで「データの間も学ばせるための訓練時の工夫」です。

### 手法の系譜と主要論文
- Zhang, Cisse, Dauphin, Lopez-Paz, "mixup: Beyond Empirical Risk Minimization", ICLR 2018: 入力とラベルを線形補間する Mixup を提案した原典です。VRM の考え方に基づき、「サンプル間で線形にふるまう」帰納バイアスが汎化を改善するという仮説を実証しました。
- Tokozume, Ushiku, Harada, "Between-class Learning for Image Classification", CVPR 2018: ほぼ同時期に、クラス間を混ぜて学習する近い発想を音声・画像で提案しており、混合系拡張の源流の一つです。
- Verma, Lamb, Beckham, Najafi, Mitliagkas, Lopez-Paz, Bengio, "Manifold Mixup", ICML 2019: 入力画像ではなく中間層の特徴を混ぜる手法です。生のピクセルより意味的に整理された特徴空間で補間するため、より滑らかでクラスが分離した表現を学べると報告しました。
- Thulasidasan, Chennupati, Bilmes, Bhattacharya, Michalak, "On Mixup Training: Improved Calibration and Predictive Uncertainty", NeurIPS 2019: Mixup が予測信頼度のキャリブレーション(モデルの自信と実際の正答率の一致)を大きく改善することを示しました。
- Hendrycks, Mu, Cubuk, Zoph, Gilmer, Lakshminarayanan, "AugMix", ICLR 2020: 複数の拡張を確率的に混ぜて、分布シフト(訓練時と違うノイズや天候の画像)への頑健性を高めました。
- Carratino, Cissé, Jenatton, Vert, "On Mixup Regularization", JMLR 2022: Mixup を正則化として展開し、理論的な性質を整理した代表的な解析研究です。
- 後続の CutMix(Yun et al. 2019、[CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))は「重ね合わせ」ではなく「切り貼り」でデータを混ぜ、画像の局所構造が壊れない利点を持ちます。Mixup と CutMix は、現代の Vision Transformer/CNN の学習レシピ(DeiT など)で確率的に切り替えて併用するのが定番になっています。

### 論文の実験結果(定量データ)
ICLR 2018 の主要結果を、指標の意味とともに紹介します。基本指標は分類誤り率(Top-1 error、最も確信したクラスが間違っている割合)で、小さいほど良い値です。

- ImageNet(1000クラス、ResNet-50): ベースラインの Top-1 誤りがおよそ 23.5% だったのに対し、Mixup を適用するとおよそ 22.1% まで低下したと報告されています(およそ1.4ポイントの改善)。これは大規模データで簡単には縮まらない誤りを、追加データなしで下げた点で注目されました。ResNet-101 や ResNeXt-101 などより大きいモデルでも一貫して改善が見られました。
- CIFAR-10(10クラス): PreAct ResNet-18 で、ベースラインの誤りがおよそ 5.6% だったのに対し、Mixup でおよそ 4.2% へ改善したと報告されています。
- ラベルノイズへの頑健性: 訓練ラベルの一部をわざと間違ったものに置き換える設定で、Mixup はノイズ率が高いほどベースラインとの差が広がり、誤って覚え込む度合い(memorization)を大きく抑えると報告されています。たとえばラベルの大部分を破損させても、ベースラインが破損ラベルを暗記して汎化を失うのに対し、Mixup はテスト精度の低下が緩やかでした。
- 敵対的サンプルへの耐性: 入力に微小な悪意あるノイズを加える FGSM 攻撃に対して、Mixup を使ったモデルは正答率の低下が小さく、より頑健だと報告されています。
- キャリブレーション(NeurIPS 2019): 予測確率と実際の正答率のずれを測る ECE(Expected Calibration Error)が、Mixup でベースラインより大きく下がると報告されています。通常の学習では過信(自信95%なのに正答70%など)が起きやすいのに対し、Mixup はこの過信を緩和します。
- 学習の安定化: GAN(生成モデル)の学習に Mixup を適用すると学習が安定したという結果も示されています。

アブレーションとして、`α` の影響が分析されています。`α` を 0.1〜0.4 程度にすると安定して効果が出る一方、`α` を大きくしすぎて(たとえば 1.0 以上)半々の混合が増えると、画像が不自然になりすぎて未学習(under-fit)に傾き、精度がかえって落ちることがある、というトレードオフが示されています。また、ラベルを混ぜずに画像だけ混ぜる変種では効果が大きく落ちることから、「ラベルも同じ割合で混ぜる」ことが本手法の核心だと確認されています。同一クラス同士だけを混ぜる変種より、異なるクラス間で混ぜる方が効果が高いことも示されています。

### メリット・トレードオフ・限界
メリット:
- 追加データなしでデータの「間」を学べ、過学習を抑え汎化が向上します。
- ラベルノイズや敵対的ノイズ、破損データへの頑健性が上がります。
- 実装が軽量(バッチをシャッフルして線形に混ぜるだけ)で計算コストがほぼ増えません。
- 予測信頼度のキャリブレーションが改善し、過信しにくくなります。

トレードオフ・限界:
- 主に分類向けの設計で、回帰・物体検出・セグメンテーションにはそのまま使いにくいです。連続値の出力や精密な位置が重要なタスクでは、線形補間したラベルの意味が曖昧になります。
- `α` の調整が必要で、強すぎると不自然な画像で未学習になり性能が落ちます。
- 重ね合わせた画像は現実には存在しない見た目になるため、物理的整合性が重要なタスク(運転シーンの生画像など)には不向きな場合があります。たとえば右折シーンと直進シーンを重ねた中間画像は二重写しで、対応する「軌跡の線形補間」も安全な経路とは限りません。
- 局所的な構造を全面で壊すため、局所特徴を重視するタスクでは [CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の方が適することがあります。
- まれに「manifold intrusion」と呼ばれる問題、すなわち混合画像がたまたま第三のクラスの見た目になり、混合ラベルと矛盾する事態が起きます(AdaMixup などがこれに対処します)。

研究上の未解決課題として、なぜ Mixup がこれほど広く効くのかの理論的説明は完全には確立しておらず、線形補間が最適な近傍の定義なのか(より良い混合則があるのか)、そして分類以外のタスクへの正しい一般化方法が、引き続き研究対象になっています。

### 発展トピック・研究の最前線
Mixup は「サンプル混合(sample mixing)」という大きな研究系統の起点です。Manifold Mixup(特徴空間での混合)、CutMix(領域置換)、PuzzleMix や SaliencyMix(物体の重要領域を考慮して混ぜる)、AutoMix(混合自体を学習する)など、多数の派生が生まれました。自己教師あり学習や半教師あり学習でも、ラベルなしデータの一貫性正則化と組み合わせて使われています(MixMatch など)。理論面では、Mixup が暗黙の正則化やマージン最大化として働くという解析が進んでおり、「データ拡張がなぜ汎化を改善するのか」を理解する手がかりとして注目されています。自然言語処理や音声、表形式データなど画像以外のモダリティへの拡張も進み、汎用的なノイズ注入・正則化の枠組みとして位置づけが広がっています。

### さらに学ぶための関連トピック
- [CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Dropout](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Stochastic Depth (DropPath)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Weight Decay (L2 正則化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Muon オプティマイザ

### ひとことで言うと
Muon は、ニューラルネットの中にある「行列の形をした重み」を更新するための比較的新しいオプティマイザ(最適化アルゴリズム)です。普通の方法では勾配(損失を減らす方向の傾き)をそのまま使って重みを動かしますが、Muon は更新量の行列を「直交化(あとで説明します。すべての方向の伸び縮みを 1 にそろえる操作)」してから重みに適用します。これによって、同じ計算量でも学習が速く進むことが報告され、大規模言語モデル(LLM)の学習速度記録を更新する文脈で一躍注目されました。名前は "MomentUm Orthogonalized by Newton-schulz"(ニュートン・シュルツ反復でモメンタムを直交化する)の頭文字です。

### 直感的な理解
ニューラルネットの計算の大部分は「行列とベクトルの掛け算」です。入力ベクトル x に重み行列 W を掛けて y = Wx を作り、次の層へ渡す。これを層ごとに繰り返します。だからモデルの中で本当に重い、本質的なパラメータは「行列の形をした重み」、つまり全結合(linear)層やアテンションの射影行列です。

ところが Adam のような既存のオプティマイザは、重みを「ただの数の集まり(ベクトル)」として扱い、各要素を独立に更新します。行列ならではの構造、すなわち「その行列がベクトルをどの方向にどれだけ伸ばすか縮めるか」という情報を完全に無視しています。

ここで問題が起きます。勾配の行列をそのまま更新に使うと、更新の「効き目」が少数の特定方向(後で説明する大きな特異値を持つ方向)に極端に偏りがちなのです。たとえると、ハンドルを切ったとき一部の車輪だけが強く動き、残りはほとんど動かない状態です。これでは限られたステップで学習を均等に進められません。実測でも、学習途中の勾配行列の特異値分布は「上位数個が支配的で、残りはほぼゼロ」という長い裾を引いた分布になることが多く、Adam はこの偏りをほとんど補正しません。

Muon の発想はシンプルです。「更新行列の特異値を全部 1 にそろえてしまえ(=直交化しろ)」。特異値が全部 1 の行列は直交行列に近く、入力ベクトルをどの方向にも均等に動かします。つまり「どの方向にも公平に、まんべんなく更新を効かせる」ようにする。これは後述する理論で「スペクトルノルム(行列が引き起こす最大の伸び)を一定に保ったまま、損失を最も速く減らす最急降下」に対応すると整理されました。

### 基礎: 前提となる概念
Muon を理解するには、行列に関するいくつかの言葉を噛み砕いておく必要があります。

特異値(singular value)とは何か。どんな行列 W も「回転 → 各軸方向への伸縮 → 別の回転」という 3 段階の操作に分解できます。これを特異値分解(SVD: Singular Value Decomposition)と呼び、W = U Σ Vᵀ と書きます。ここで U と V は回転を表す直交行列(掛けても長さを変えない行列)、Σ は対角行列で、その対角に並ぶ非負の数が特異値 σ₁ ≥ σ₂ ≥ … です。特異値は「その方向に入力をどれだけ伸ばすか」を表します。σ₁ が一番大きい方向に勾配が偏っていると、更新がその方向ばかり強く効いてしまう、というのが先ほどの問題でした。

直交行列(orthogonal matrix)とは、掛けてもベクトルの長さや角度を保つ行列です。回転や鏡映がその例です。特異値が全部 1 の行列はちょうど直交行列(行が縦より多い/少ない非正方なら半直交行列)で、すべての方向を平等に扱います。

スペクトルノルム(spectral norm)とは、行列が引き起こしうる最大の伸び、すなわち最大特異値 σ₁ のことです。記号では ‖W‖₂ と書きます。「この更新行列を掛けたら、最悪のケースでベクトルが何倍に伸びるか」を表します。学習を暴走させないには、この最大の伸びを制御することが本質的に重要だ、というのが Muon の理論的支柱になります。対照的に、Adam が間接的に制御しているのはフロベニウスノルム(全要素の 2 乗和の平方根、=全特異値の 2 乗和の平方根)で、これは「全体のエネルギー」は見ても「最悪方向の伸び」は見ません。

モメンタム(momentum)とは、過去の勾配の指数移動平均を取って更新方向をならす工夫です。一歩ごとにジグザグする勾配を平滑化し、坂を転がるボールのように一定方向の勢いを蓄えます。Muon はこのモメンタム付き更新行列を直交化の入力に使います。

極分解(polar decomposition)も鍵概念です。任意の行列 M は M = Q P と分解でき、Q は半直交行列(M に最も近い直交行列)、P は対称半正定値行列です。Muon が作りたいのはこの Q、すなわち「M に最も近い、特異値が全部 1 の行列」です。SVD で言えば M = UΣVᵀ のうち Σ を単位行列に置き換えた UVᵀ がちょうど Q に一致します。

### 仕組みを詳しく
Muon の 1 ステップは次の手順です。対象は 2 次元の重み行列(linear 層やアテンション射影)に限り、埋め込み層・出力層・バイアス・正規化のスケールなどは対象外で、それらは Adam(正確には AdamW)で更新します。

1. 勾配 G にモメンタムをかけ、ならした更新方向 M(行列)を作る。ここは SGD with momentum と同じです。
2. M を直交化する。理想は SVD で M = UΣVᵀ と分解し、Σ を単位行列に置き換えた `U Vᵀ`(これが M に最も近い半直交行列。専門的には極分解の直交成分)を作ることです。
3. ただし SVD は重いので、Muon は Newton-Schulz 反復という多項式の繰り返し計算で `U Vᵀ` を近似します。具体的には正規化した X に対して `X ← a·X + b·(X Xᵀ)X + c·(X Xᵀ)²X` という 5 次の多項式を、係数 (a, b, c) を選んで 5 回ほど反復します。行列積だけで済むので GPU と非常に相性がよく、bf16(半精度)でも安定に動きます。
4. 直交化した更新行列を学習率倍して重みから引きます。

数式の骨格はこうです。

```
M ← μ·M + G            # モメンタム(μ は 0.9〜0.95 程度)
O ← NewtonSchulz5(M)   # M を半直交行列に近づける(特異値を ≈ 1 に)
W ← W − η · O
```

ここで μ はモメンタム係数、η は学習率、NewtonSchulz5 は 5 回反復の直交化です。

Newton-Schulz 反復がこの手法の心臓部です。SVD は計算量 O(n³) で重いうえ GPU で並列化しにくく、半精度では数値が崩れやすい。これに対し Newton-Schulz は数回の行列積(これも O(n³) だが定数が小さく、GPU の Tensor Core などの行列積ユニットに最適化されている)で済み、半精度でも壊れません。さらに特異値を厳密に 1 にする必要はなく、「だいたい 0.7〜1.3 の範囲に押し込む」程度で十分効果が出る、という割り切りが実用性を生んでいます。係数 (a, b, c) は、わざと収束を「だいたい」で止めて反復回数を減らすよう調整されます(過収束させると無駄に反復が増える)。Jordan の公開実装では (a, b, c) ≈ (3.4445, −4.7750, 2.0315) といった値が使われ、これは特異値が 0 に近い小さな成分も 1 桁台のステップで 1 付近へ素早く引き上げられるよう最適化されています。

数値例で直交化の効果を見ます。仮に更新行列 M の特異値が (5.0, 0.3, 0.05) だったとします。これをそのまま使うと、第 1 方向は第 3 方向の 100 倍も強く更新され、ほとんど一方向しか動きません。Newton-Schulz で直交化すると特異値が概ね (1.0, 1.0, 1.0) になり、3 方向が均等に更新されます。「眠っていた弱い方向」が初めて学習に参加できるわけです。tensor shape の感覚をつかむと、隠れ次元 d=4096 の linear 層なら M は 4096×4096 の行列で、Newton-Schulz 1 反復はこの行列の行列積を数回行う程度。5 反復しても 1 ステップの追加コストは前方計算+逆伝播全体のおよそ数%に収まることが多く、これが「総計算では得になる」根拠です。

なぜ埋め込み層や出力層を Adam に任せるのか。それらは「行列としての回転変換」ではなく「個々の語彙ベクトルの大きさ・向き」が意味を持ち、直交化の前提(行列がベクトルを変換する作用素である)と性質が合わないためです。埋め込み行列は語彙数×次元という極端に縦長の行列で、行ごとに頻度が大きく違う(高頻度語と低頻度語で更新頻度が桁違い)ため、全方向を均等化する操作と相性が悪い。実際、埋め込みや出力を Muon で更新すると性能が落ちることが経験的に知られています。したがって Muon は「Muon(隠れ層の行列本体) + AdamW(埋め込み・出力・正規化・バイアス)」のハイブリッドで使うのが標準です。

### 手法の系譜と主要論文
Muon は突然出てきたのではなく、「更新を前処理(preconditioning)して幾何を整える」という流れの中にあります。

- Gupta, Koren & Singer, "Shampoo: Preconditioned Stochastic Tensor Optimization" (ICML 2018, Google, arXiv:1802.09568)。各層の勾配統計から左右の前処理行列を作り、勾配を `L^{-1/4} G R^{-1/4}` のように整形する手法です。これは更新行列の特異値を平滑化する効果を持ちます。後の理論研究で、Shampoo の前処理は本質的に更新行列を直交化することに対応すると整理され、Muon の「祖先」と位置づけられました。難点は逆 4 乗根行列の計算が重いことです。

- Vyas et al., "SOAP: Improving and Stabilizing Shampoo using Adam" (2024, arXiv:2409.11321)。Shampoo を Adam の枠組みで再解釈し、前処理した固有基底の中で Adam を走らせることで安定性と効率を高めました。Shampoo 系と Muon の橋渡しになる位置づけです。

- Keller Jordan et al., "Muon: MomentUm Orthogonalized by Newton-Schulz" (2024、技術ノート/実装として公開され広まる)。提案の核心は「モメンタム付き更新行列を、SVD ではなく安価な Newton-Schulz 反復で直交化する」こと。動機は NanoGPT の高速学習コンペ(speedrun)で、限られた計算で最速の収束を競う中から生まれました。Shampoo の重い逆ルート分解を、行列積だけの反復に置き換えた点が新規性です。

- Bernstein & Newhouse, "Old Optimizer, New Norm: An Anthology" (arXiv:2409.20325) および "Modular Duality in Deep Learning" (arXiv:2410.21265)(ともに 2024)。Muon がなぜ効くのかを理論で説明しました。彼らは「各層のパラメータ更新を、その層の作用に合ったノルム(行列ならスペクトルノルム)の下での最急降下とみなすべきだ」という枠組み(modular norm / modular duality)を提示し、スペクトルノルム制約下の最急降下がちょうど更新行列の直交化に帰着することを示しました。これにより Muon が経験則ではなく原理に基づくことが明らかになりました。

- Liu et al., "Muon is Scalable for LLM Training"(2025, Moonshot AI ら。Moonlight モデル、arXiv:2502.16982)。Muon を数十億〜兆パラメータ級まで拡張し、(1) 重み減衰(weight decay)の追加、(2) パラメータごとの更新 RMS をそろえるスケール調整(行列の形に応じて学習率を `√(out_dim/in_dim)` 等で補正)、(3) 分散環境向けの効率的実装、を加えることで大規模でも安定に動くことを示しました。AdamW 比でおよそ 2 倍のサンプル/計算効率を報告しています。

### 論文の実験結果(定量データ)
Muon の評価でよく使われる指標は、言語モデルの validation loss(検証損失)あるいは perplexity(困惑度。損失の指数を取った値で、モデルが次の単語をどれだけ「迷う」かを表す。小さいほど良い)と、そこに到達するまでに要した計算量(FLOPs)や実時間(wall-clock time)です。

NanoGPT speedrun(GPT-2 規模、約 124M パラメータ、FineWeb / OpenWebText 系データ)の文脈では、Muon は AdamW と同じ目標 validation loss(GPT-2 small 相当の約 3.28)に到達するのに必要な学習ステップ数・実時間を明確に削減したと報告されています。speedrun は記録更新の積み重ねで進み、当初 8 GPU で数時間かかっていた学習が一連の改良で数分台まで短縮されましたが、Muon の導入はその到達時間短縮の主要因の一つに数えられました。直交化反復のぶん 1 ステップあたりの計算はわずかに増えますが、必要ステップ数の減少がそれを大きく上回るため、総計算では得になります。報告されている代表的な数字として、Muon は AdamW に比べて目標損失到達までのサンプル数をおよそ 1.3〜1.4 倍効率化したケースが speedrun 文脈で言及されています。

Moonlight 論文(Liu et al., 2025)では、より大規模な検証が行われました。報告によれば、同じ計算予算(FLOPs)で比べたとき Muon は AdamW より低い損失に到達し、逆に同じ損失に到達するための計算量がおよそ半分(約 2 倍のサンプル効率)で済んだとされています。具体的には 3B 活性 / 16B 総パラメータ規模の Mixture-of-Experts モデルを 5.7T トークンで学習し、同等の計算予算で学習した AdamW ベースの公開モデル群と同等以上の下流タスク性能(MMLU などのベンチマーク)を、より少ない学習トークンで達成したと報告しています。「2 倍の効率」という数字は、同じ GPU 予算で実質 2 倍のデータを学習したのと同じ進み方ができる、という意味で、大規模学習のコストに直接効くため重要です。

アブレーション(どの要素を抜くとどうなるかの検証)からの示唆もはっきりしています。(1) 直交化(Newton-Schulz)を外して単なるモメンタム SGD にすると、Muon の優位は消えます。直交化こそが効果の源泉です。(2) 埋め込み層・出力層まで Muon で更新すると性能が劣化します。これらを AdamW に戻すハイブリッドが必須です。(3) 大規模では weight decay を入れないと、更新が直交化されて方向が均等になる一方で重みのノルム(特にスペクトルノルム)が単調に膨らみ続け、学習が不安定になります。Moonlight はこの weight decay 追加を安定化の鍵として挙げており、weight decay なしでは学習後半で損失曲線が崩れることを示しています。(4) 行列の形に応じた学習率スケーリングを外すと、縦長/横長の層で更新の大きさがそろわず、大規模での収束が悪化します。

### メリット・トレードオフ・限界
メリット。更新を直交化することで全方向に均等な学習が進み、AdamW より少ない計算・実時間で同じ損失に到達した報告が複数あります。Newton-Schulz 反復は行列積だけで GPU と相性がよく、bf16 でも安定。状態としてモメンタム 1 つを持てばよく、2 つの状態(1 次・2 次モーメント)を持つ Adam よりオプティマイザのメモリを軽くできる場合があります(隠れ層を Muon、その他を AdamW にするハイブリッドのため、総メモリは構成依存)。

トレードオフと限界。行列パラメータ専用で、埋め込み・出力・正規化・バイアスは別途 AdamW が必要(ハイブリッド前提)。直交化反復のぶん 1 ステップの計算が増え、小さな行列では割に合わないこともあります(行列積コストは隠れ次元の 3 乗で効くため、極端に幅広い層では反復回数とのバランスを見る必要があります)。検証は主に Transformer 言語モデルで行われており、視覚・マルチモーダル・自動運転といったタスクでの有効性はまだ広く確立されていません。比較的新しい手法で、学習率や反復回数のチューニング指針、超大規模での分散安定化のベストプラクティスがまだ成熟途上です。畳み込み層(4 階テンソルの重み)は素直に 2 次元行列に reshape して扱うが、その reshape の仕方で性能が変わるという報告もあり、設計が確立していません。

未解決の研究課題としては、(1) 直交化が有効なタスク・アーキテクチャの境界を明らかにすること、(2) スペクトルノルム以外のノルム(層の種類に応じた最適なノルム)での最急降下を体系化すること、(3) 畳み込み層など 2 次元行列でないパラメータへの一般化、(4) Muon と Adam の最適なハイブリッド比率や学習率スケールの理論的決定、が挙げられます。

### 発展トピック・研究の最前線
Bernstein & Newhouse の「層ごとに適切なノルムを選び、そのノルム下の最急降下を行う」という modular な視点は、オプティマイザ設計を「各パラメータの幾何に合わせて更新を整形する」一般原理へと押し広げつつあります。Muon はその最初の実用例の一つにすぎず、今後は層タイプごとに異なる前処理を自動で選ぶ枠組みへ発展する可能性があります。また、Muon と Shampoo・SOAP・二次情報を使う手法群との関係を統一的に理解しようとする研究、Muon を分散学習(データ並列・テンソル並列・ZeRO 系の最適化状態分割)で効率よく動かす実装研究が活発です。直交化が暗黙に持つ「特異値正則化」が汎化に与える影響、視覚 Transformer や拡散モデルなど言語以外への適用検証も今後の焦点です。

### さらに学ぶための関連トピック
- [Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Nadam

### ひとことで言うと
Nadam(ナダム、Nesterov-accelerated Adaptive Moment Estimation)は、現代で最もよく使われるオプティマイザ Adam に「Nesterov モメンタム」という一歩先読みの工夫を組み込んだ改良版です。Adam とほぼ同じ使い勝手・同じメモリ消費のまま、収束(学習が最適な状態に落ち着くこと)がわずかに速く・安定することを狙います。

### 直感的な理解
坂道をボールが転がる様子を思い浮かべてください。普通のモメンタム(慣性)は「今いる場所で坂の傾きを見て、過去の勢いと合わせて進む」やり方です。Nesterov モメンタムは「まず勢いで少し先に進んでみて、その着地点で坂の傾きを見る」やり方です。先回りして傾きを見るので、谷底を行きすぎそうなときに一歩早く気づいてブレーキをかけられます。カーブの手前で先を見て減速するドライバーのイメージです。

モメンタム単体では Nesterov のほうが普通のモメンタムより収束が良いことが古くから知られていました。Adam も内部でモメンタムを使っているので、その部分を Nesterov 版に差し替えれば Adam も少し良くなるはず、という素直な発想で生まれたのが Nadam です。

### 基礎: 前提となる概念

前提を順に噛み砕きます。

- モメンタム(momentum、慣性): パラメータ更新に「これまで動いてきた方向の勢い」を加える工夫。毎回の勾配だけでなく過去の更新方向も少し混ぜて進むことで、ジグザグが減り谷底に速く着きます。式では `v ← μ·v + g`、`θ ← θ - η·v`(μ がモメンタム係数で 0.9 など、v が速度)。坂の細長い谷で勾配が左右に振れても、振れの成分が打ち消し合い、谷に沿った成分だけが積み上がって加速します。
- Nesterov モメンタム(NAG、Nesterov Accelerated Gradient): Nesterov が1983年に提案した加速法。通常のモメンタムが「今の場所 θ で勾配を測ってから勢いを足す」のに対し、Nesterov は「勢いで先に進んだ場所 θ+μ·v で勾配を測る」先読みをします。Sutskever et al. (ICML 2013) が深層学習で NAG の有効性を再評価し、広く使われるようになりました。凸最適化では NAG が `O(1/t^2)` の収束率(通常の勾配降下の `O(1/t)` より速い)を達成することが理論的に示されています。t はステップ数で、t^2 で割れるほうが誤差の減りが速いという意味です。
- Adam(Adaptive Moment Estimation、Kingma & Ba, ICLR 2015): 一次モーメント(勾配の移動平均=モメンタム)と二次モーメント([Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)/RMSProp 系の勾配二乗の移動平均)を両方使い、さらにバイアス補正を加えた手法。多くの深層学習でデフォルトに近い存在です。
- バイアス補正(bias correction): 移動平均 m や v は初期値0から始まるため、学習初期は本来の値より小さく偏ります(0に引っ張られる)。これを `m_hat = m/(1-β1^t)` のように割って補正します(t はステップ数、β1 は減衰係数)。t が大きくなると `1-β1^t → 1` で補正は効かなくなります。

問題意識はこうです。Adam が内部で使うモメンタムは「素朴なモメンタム」です。モメンタム単体では Nesterov が優位なのだから、その優位性を適応的オプティマイザ Adam でも享受したい。これが Nadam の動機です。

### 仕組みを詳しく

まず Adam の更新を3ステップでおさらいします。各パラメータについて、

```
1. m ← β1·m + (1-β1)·g            # 一次モーメント(モメンタム)
2. v ← β2·v + (1-β2)·g^2          # 二次モーメント(勾配二乗の移動平均)
3. m_hat = m/(1-β1^t), v_hat = v/(1-β2^t)   # バイアス補正
   θ ← θ - η · m_hat / (sqrt(v_hat) + ε)
```

β1(例 0.9)、β2(例 0.999)は減衰係数、ε(例 1e-8)はゼロ割り防止です。

Nadam が変えるのはステップ3で「分子にどんなモーメントを使うか」です。Dozat の出発点は「Nesterov モメンタムは、過去の勢いを使うのではなく、今まさにこれから使う勢い(更新後のモーメント)を先取りして勾配ステップに混ぜることと等価に書き換えられる」という観察です。この書き換えを Adam の分子に適用すると、「今のモーメント m_hat」だけでなく「今まさに測った勾配 g を前倒しで足し込んだ」値を使う形になります。式の形を比べると、

```
Adam:   θ ← θ - η · m_hat / (sqrt(v_hat) + ε)
Nadam:  θ ← θ - η · ( β1·m_hat + (1-β1)·g/(1-β1^t) ) / (sqrt(v_hat) + ε)
```

直感的には「過去の勢い(モーメント m_hat)に、今の勾配 g を少し前倒しで混ぜる」ことで Nesterov 流の先読みを近似しています。Dozat の貢献は、この先読みを「わざわざ先の場所 θ+μ·v で勾配を測り直す」のではなく、現在の勾配とモーメントの重み付けだけで実現する書き換えを示した点です。これにより実装は Adam とほぼ同じ計算量・同じ状態で済みます。なお厳密な Nadam は β1 をステップごとにわずかに変化させる係数(モメンタムスケジュール)を導入する版もありますが、実用上は固定 β1 でほぼ同じ挙動になります。

数値例でイメージします。補正済みモーメント m_hat = 0.5(これまで正方向に動いてきた)、今の勾配 g = 0.5、β1 = 0.9、t が十分大きく `1-β1^t ≈ 1` とします。

```
Adam の分子:  0.5
Nadam の分子: 0.9·0.5 + 0.1·0.5 ≈ 0.5
```

定常状態では両者ほぼ同じです。差が出るのは「勾配の向きが変わり始めた局面」です。たとえば谷底に近づいて g が +0.5 から -0.5 に反転し始めると、

```
Adam の分子:  m_hat = 0.45 程度(過去の勢いがまだ正)→ まだ正方向に進む
Nadam の分子: 0.9·0.45 + 0.1·(-0.5) = 0.405 - 0.05 = 0.355 → 一歩早く減速
```

Nadam は今の勾配 g を前倒しで反映する分、Adam より一歩早くブレーキがかかり、行きすぎ(オーバーシュート)を抑えます。これが Nesterov 化の効きどころです。

tensor shape とメモリ: 状態(m と v)は Adam とまったく同じで、パラメータと同サイズを2つ持ちます。Nadam は Adam に対して追加メモリをほぼ必要としません。

### 手法の系譜と主要論文

- Nesterov(1983): 凸最適化の加速勾配法。`O(1/t^2)` 収束を達成する理論的に重要な手法。後の加速法・近接勾配法すべての基礎です。
- モメンタム SGD の再評価(Sutskever et al., ICML 2013): 深層学習で良い初期化と Nesterov モメンタムを組み合わせると、当時主流だったヘシアンフリーな二次法に迫る性能が出ることを示し、NAG を深層学習に普及させた。Nesterov 更新を実装しやすい等価形に書き換える定式化もこの論文が広めました。
- Adam(Kingma & Ba, ICLR 2015): 適応的学習率の決定版。
- Nadam(Dozat, ICLR 2016 Workshop): Adam のモメンタム項を Nesterov 化。
- その後の Adam 系改良: 重み減衰を正しく分離した AdamW(Loshchilov & Hutter, ICLR 2019)、学習初期の分散を補正した [RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、大バッチ向けの [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。Nadam は「モメンタムを Nesterov 化する」という軸の改良で、これらとは直交します(組み合わせ可能)。

### 論文の実験結果(定量データ)

Timothy Dozat "Incorporating Nesterov Momentum into Adam" (ICLR 2016 Workshop) の主張と実験。

- 評価指標の説明: 言語モデルでは「パープレキシティ(perplexity、モデルが次の単語をどれだけ当てられるかの指標。1単語あたりの平均的な選択肢の数のようなもので、低いほど良い)」、画像分類では「検証誤差(validation loss、学習に使っていないデータでの誤差)」、単語埋め込みでは類推タスクの精度を見ます。
- 結果の傾向: Word2Vec 系の単語埋め込み学習や、画像分類(MNIST 系)・言語モデル(Penn Treebank)のいくつかのタスクで、Nadam は Adam より検証誤差・パープレキシティがわずかに小さく、収束も少し速いことを示しました。とくに学習初期の収束速度で差が出やすい傾向が報告されています。
- 正直な限界の記述: 論文自身が「改善は劇的ではなくわずか(modest)」と述べています。タスクによっては Adam と差が出ない、あるいは Adam のほうが良いこともあります。

その後のコミュニティの大規模ベンチマーク(Schmidt et al. "Descending through a Crowded Valley", ICML 2021。15個のオプティマイザを8つのタスク・約5万回の学習実行で公平比較した研究)では、Nadam は Adam・AdamW と並んで一貫して上位グループに入る安定した選択肢であることが確認されました。同研究の重要な全体結論は「どのオプティマイザも、十分にチューニングすれば最終性能の差は小さく、新しいオプティマイザがよくチューニングした Adam を明確に上回る証拠は乏しい」というもので、Nadam の優位もこの文脈で「条件依存の小さな改善」と理解すべきです。

アブレーション的な観点: Nadam から Nesterov 化を外せばただの Adam です。Adam との唯一の違いはステップ3の分子に今の勾配を前倒しで混ぜるかどうかだけなので、Nadam の効果はまさにこの「先読み項」が生み出すものです。先読み項の寄与が大きいのは勾配が変動する初期や、loss 地形のカーブが急な局面です。逆に勾配がほぼ一定方向に流れる局面では Adam と区別がつきません。

### メリット・トレードオフ・限界

メリット

- Adam とほぼ同じ使い勝手・同じメモリで、Nesterov の先読みによる安定化・高速化を狙える。
- 谷底付近で勾配が反転し始めたとき、Adam より一歩早くブレーキがかかりやすい。
- 既存の Adam 学習コードからの置き換えコストが小さい(状態構造が同一)。
- 学習初期の収束速度で僅差ながら優位が出やすい。

トレードオフ・限界

- 改善はわずかで、タスクや十分なチューニング後には Adam と差が出ないことが多い。
- Adam の根本的な弱点(重み減衰の扱い、学習初期の不安定さ)は解決していない。これらはそれぞれ AdamW、[RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) が対象とする別軸の改良。
- ハイパーパラメータ感度は Adam とほぼ同等で、β1・β2・η の調整は依然必要。
- 適応的手法全般の汎化性能への懸念(後述)は Nadam も共有する。

### 発展トピック・研究の最前線

- AdamW との組み合わせ: 実務では NadamW(Nadam に decoupled weight decay を入れたもの)として使われることもあり、主要フレームワークに NAdam の実装があります。重み減衰の分離(AdamW の貢献)と Nesterov 化(Nadam の貢献)は独立なので合成できます。
- オプティマイザの公平比較と汎化: Nadam を含む適応的手法は、Wilson et al. (NeurIPS 2017, "The Marginal Value of Adaptive Gradient Methods") の「適応的手法はよくチューニングした SGD+モメンタムより汎化(テスト精度)で劣ることがある」という指摘の対象でもあります。学習(訓練 loss)は速いがテスト性能で SGD に及ばない場面がある、というトレードオフは Adam 系共通の論点です。
- 大規模学習での選択: 大規模言語モデルの事前学習では AdamW が事実上の標準で、Nadam が選ばれる場面は限定的です。一方、収束速度を少しでも稼ぎたい中小規模タスクや、ハイパーパラメータ探索を最小限にしたいケースでは依然有効な選択肢です。

### さらに学ぶための関連トピック

- [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## NCCL (集合通信)

### ひとことで言うと
複数の GPU で 1 つのモデルを学習するとき、GPU どうしは頻繁に数値をやり取りします。たとえば「全 GPU が計算した勾配を平均する」「分割して持っているパラメータを集める」といった操作です。NCCL（ニックル、NVIDIA Collective Communications Library）は、こうした GPU 間のデータ交換を、GPU 間の高速リンク（NVLink など）やネットワークをフル活用してできるだけ速く行う、NVIDIA 製の通信ライブラリです。PyTorch の分散学習（DDP や FSDP）で、裏側の通信を担う標準の部品になっています。分散学習の「速さの天井」を決めるのが、実はこの集合通信の効率です。

### 直感的な理解
合唱団で全員が別々の小節を練習したあと、最後に「全員の音量を平均して同じ強さに揃える」場面を想像してください。一番素朴なやり方は、指揮者 1 人が全員の音量を聞いて回り、平均を計算し、また全員に伝えに回ることです。これは指揮者にすべての通信が集中し、人数が増えるほど指揮者がボトルネックになります（これがパラメータサーバ方式の弱点です）。

代わりに「隣の人に自分の値を渡し、受け取った人は足し込んでまた隣に渡す」というバケツリレーをすれば、全員が同時に動けて、誰か 1 人に負荷が集中しません。NCCL がやっているのはまさにこの「賢いバケツリレー」で、GPU を何枚に増やしても 1 枚あたりの通信量が爆発しないように工夫されています。

### 基礎: 前提となる概念
まず「なぜ GPU 間で通信が必要なのか」をデータ並列で具体化します。データ並列とは、同じモデルのコピーを GPU 4 枚に置き、それぞれに違うデータを処理させる方式です（[DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の冒頭も参照）。

```
GPU0: データA を処理 → 勾配 g0 を計算
GPU1: データB を処理 → 勾配 g1 を計算
GPU2: データC を処理 → 勾配 g2 を計算
GPU3: データD を処理 → 勾配 g3 を計算
```

各 GPU は自分の見たデータ分の勾配しか持っていません。学習を正しく進めるには、4 枚分を平均した (g0+g1+g2+g3)/4 を全 GPU が共有し、同じ更新を適用する必要があります。そうしないと 4 枚のモデルがバラバラの方向に進み、平均化の意味が失われます。この「全員の値を合計（または平均）して全員に配る」操作を all-reduce（オールリデュース）と呼びます。

ここで「集合通信（collective communication）」という用語を噛み砕きます。1 対 1 の通信（point-to-point、ある GPU から別の GPU へ送る）に対し、集合通信は「グループ全員が協調して 1 つの操作を成し遂げる」通信です。all-reduce、all-gather、broadcast などが代表で、いずれも全員が参加して初めて完了します。並列計算の世界では MPI（Message Passing Interface）が古くからこの集合通信を標準化してきましたが、汎用 MPI は GPU 特有の高速リンク（NVLink など）を活かしきれなかったため、NVIDIA が GPU に特化したのが NCCL です。

なぜ速さが死活問題かというと、勾配はパラメータと同じ個数だけあるので、数十億パラメータのモデルでは毎ステップ数 GB〜十数 GB の数値を交換するからです。これを愚直にやると、計算（GPU の本業）より通信に時間がかかり、GPU をいくら増やしても全体が速くなりません。「集合通信をいかに速く・無駄なく行うか」が分散学習の性能を決めます。

通信コストは伝統的に「レイテンシ項 α（1 回のメッセージ発信にかかる固定時間）」と「帯域項 β×データ量（データを流すのにかかる時間）」の和で表す α-β モデルでモデル化されます。後述のアルゴリズム選択は、この 2 項のどちらを重視するか（小データはレイテンシ、大データは帯域）で変わります。

### 仕組みを詳しく

#### 代表的な集合通信操作（プリミティブ）
NCCL が提供する主な操作と、分散学習での使いどころを対応させます。

- all-reduce: 全 GPU の値を合計（や平均）して全 GPU に配る。データ並列の勾配同期で使う中心操作。
- all-gather: 各 GPU が持つ断片を全部集め、全 GPU が全体を得る。ZeRO Stage 3 / FSDP で分割保持したパラメータを計算直前に集める（[DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)）。
- reduce-scatter: 合計しつつ結果を分割して各 GPU に配る。all-reduce は実は reduce-scatter + all-gather に分解できる。FSDP の勾配集約でも使う。
- broadcast: 1 つの GPU の値を全 GPU に配る。学習開始時に全 GPU のモデルを同じ初期値に揃える。
- reduce / gather / scatter: 1 つの GPU に集約・収集・分配する非対称な操作。
- all-to-all: 各 GPU が他の全 GPU に異なる断片を送る。Mixture of Experts のトークンルーティングやテンソル並列の一部で使う。

#### Ring all-reduce で通信を効率化する
NCCL の効率の鍵は通信アルゴリズムです。代表が「Ring（リング、輪）」方式です。N 枚の GPU を輪状につなぎ、データを N 個のチャンク（塊）に分割して、隣から隣へバケツリレーのように渡しながら少しずつ足し合わせます。手順は 2 フェーズです。

```
フェーズ1: reduce-scatter（N-1 ステップ）
GPU0 → GPU1 → GPU2 → GPU3 → (GPU0 へ戻る輪)
各ステップで「隣に送る」「隣から受け取って足し込む」を繰り返す
終了時、各 GPU が「全 GPU 分が合算された 1 チャンク」を持つ

フェーズ2: all-gather（N-1 ステップ）
合算済みチャンクを同じ輪で配り回し、全 GPU が全チャンクを得る
```

この方式の核心は通信量の分析にあります。1 つの GPU が送受信する総データ量は、データ全体を S バイトとすると約 2 × (N-1)/N × S です。N が大きくなると (N-1)/N は 1 に近づくので、各 GPU の通信量は GPU 枚数 N にほとんど依存せず、約 2S で頭打ちになります。つまり 8 枚でも 256 枚でも 1 枚あたりの通信量がほぼ一定で、これが「帯域最適（bandwidth-optimal）」と呼ばれる理由です。この性質は Patarasuk & Yuan（2009）が理論的に示し、Baidu が 2017 年ごろ深層学習に応用して広めました。

Ring の弱点はレイテンシです。reduce-scatter と all-gather でそれぞれ N-1 ステップ、合計 2(N-1) ステップの逐次的な hop が必要で、N が大きい（多ノード・多 GPU）と固定遅延が積み上がります。そこで NCCL は Tree（木構造）方式も持ちます。木はデータを根から葉へ log(N) 段で配るためレイテンシ項が O(log N) に抑えられ、多ノードの小〜中サイズ通信で Ring より速くなります。NCCL は実際には「double binary tree」（2 本の二分木を組み合わせて全ノードが内部/葉の両役を持ち帯域を使い切る）を採用し、レイテンシと帯域の両立を図っています。NCCL は GPU 台数・配置・データサイズ・ネットワーク構成に応じて Ring / Tree / CollNet などから速いアルゴリズムを自動選択します（NCCL_ALGO で固定も可能）。

#### トポロジを認識してハードウェアを使い分ける
NCCL は起動時にシステムのトポロジ（GPU・リンク・ネットワークの接続関係）を調べ、経路を最適化します。

- 同一サーバ内の GPU どうし: NVLink / NVSwitch（GPU 間を直結する超高速リンク。PCIe より桁違いに速く、世代により数百 GB/s〜1.8TB/s 級。Hopper 世代の NVLink 4 で GPU あたり 900 GB/s 双方向）を使う。
- サーバをまたぐ場合: InfiniBand や高速イーサネット、可能なら GPUDirect RDMA（CPU を経由せず GPU メモリ間で直接転送する技術。CPU バウンスコピーのオーバーヘッドを除ける）を使う。AWS では EFA（Elastic Fabric Adapter）がこの役割を担う。
- in-network reduction: 対応スイッチ（NVIDIA SHARP）があれば、合算処理をネットワークスイッチ上で実行し、各ノードの送受信量をさらに削る。

この自動最適化のおかげで、通信が頻繁なテンソル並列を同一サーバ内 GPU に閉じ込める（[Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)）といった配置が、ハードウェアの強みを活かして速く回ります。トポロジを誤検出すると遅い経路（PCIe 経由など）に落ちることがあり、`NCCL_TOPO_DUMP_FILE` や `NCCL_DEBUG=INFO` でどの経路・アルゴリズムが選ばれたかを確認するのが診断の定番です。

#### PyTorch からの使われ方
利用者が NCCL を直接書くことはほぼありません。PyTorch の分散学習を初期化するとき `torch.distributed.init_process_group(backend="nccl")` を指定すると、DDP の勾配 all-reduce、FSDP の all-gather / reduce-scatter が裏で NCCL を呼びます。GPU 分散学習ではこれがほぼ既定です（CPU では Gloo という別バックエンドを使う）。DDP は通信と逆伝播の計算をオーバーラップさせる工夫を持ちます。逆伝播は出力層側から入力層側へ進むので、勾配ができた層から順に、複数の勾配をまとめた「bucket（バケツ、既定 25MB 程度）」単位で all-reduce を非同期に発火させ、まだ計算中の他層の逆伝播の裏に通信を隠します（gradient bucketing）。これにより理想的には通信時間がほぼ見えなくなり、計算律速に近づきます。

### 手法の系譜と主要論文
- Ring all-reduce の理論（Patarasuk & Yuan, "Bandwidth optimal all-reduce algorithms for clusters of workstations", JPDC 2009）: 各ノードの通信量が台数に依存せず帯域最適になることを示した古典。HPC コミュニティ発で、後に深層学習へ輸入された。
- Baidu の深層学習応用（2017 年ごろの技術ブログ・実装、ring-allreduce）: HPC の Ring all-reduce を深層学習の勾配同期に持ち込み、パラメータサーバ方式より素直にスケールすることを実証。
- Horovod（Sergeev & Del Balso, "Horovod: fast and easy distributed deep learning in TensorFlow", arXiv:1802.05799, 2018, Uber）: Ring all-reduce ベースの分散学習を数行で書けるようにしたフレームワーク。通信バックエンドに NCCL や MPI を使う。動機は「当時の分散学習がパラメータサーバ方式で書きにくくスケールしにくかった」こと。Tensor Fusion（小さなテンソルをまとめて 1 回の通信にする最適化）など実務的工夫も導入。
- NVIDIA NCCL（製品。仕様・実装が一次情報）: トポロジ認識で最適アルゴリズム（Ring / double-binary Tree / CollNet）と経路（NVLink / IB / RDMA / SHARP）を選ぶ集合通信ライブラリ。汎用 MPI では GPU 特有の高速リンクを活かしきれなかった問題を解く。NCCL 2.x で double binary tree を導入し多ノードのレイテンシを大幅改善。
- Megatron-LM（Narayanan ら, SC 2021, arXiv:2104.04473）: テンソル並列・データ並列の通信を NCCL で実行する前提で超大規模 LLM を学習し、NCCL が大規模学習の通信基盤の標準であることを裏づけた。

系譜を一言でまとめると、「HPC で生まれた帯域最適 all-reduce 理論（2009）→ 深層学習への応用と使いやすい API 化（Baidu/Horovod 2017-18）→ GPU ハードを知り尽くした NVIDIA 製ライブラリとして標準化（NCCL）→ 超大規模 LLM 学習の通信基盤（Megatron 系）」という流れです。

### 論文の実験結果（定量データ）
指標の意味を先に説明します。集合通信の性能は「実効帯域（algorithm bandwidth / bus bandwidth、実際に何 GB/s でデータを動かせたか）」と「スケーリング効率（GPU を N 倍にしたとき学習速度が何倍になるか。1.0 に近いほど良い）」で測ります。bus bandwidth はアルゴリズムの理論通信量で割り戻した「ハードの本来の帯域をどれだけ使えたか」を表し、リンク理論値に近いほど効率が良いとされます。

- Horovod 論文（2018）の報告: Inception V3 / ResNet-101 / VGG-16 を多数 GPU で学習する際、従来の TensorFlow 標準（グラフ内分散・パラメータサーバ的）に比べ、Ring all-reduce + NCCL で 128 GPU 規模でスケーリング効率を大きく改善し、モデルによりスループットがおよそ 2 倍、スケーリング効率 80〜90% 台を達成と報告。通信が GPU 数に対してほぼ一定量で済むことが効いている。
- NCCL のベンチマーク（NVIDIA 公表値、nccl-tests）: NVLink/NVSwitch で結ばれた 8 GPU の all-reduce で、リンクの理論帯域に近い実効 bus bandwidth（DGX A100 で 200〜250 GB/s 級）を達成。multi-node でも IB/EFA + GPUDirect RDMA により高い bus bandwidth を維持し、double binary tree で多ノードの中サイズメッセージのレイテンシを Ring より大きく短縮。
- Megatron-LM（SC 2021）の報告: 数千 GPU（A100）で数千億〜兆パラメータ級を学習し、テンソル並列を同一ノード（NVLink）に、パイプライン並列をノード間に割り当てる配置で、GPU あたり高い達成 TFLOPS（A100 ピーク 312 TFLOPS に対し約 50% 超の利用率、約 163 TFLOPS）を実現。NCCL の通信効率がこのスケーラビリティの前提になっている。

数値の意味を初心者向けに補うと、実効帯域がリンクの理論値に近いほど「配線の太さを無駄なく使えている」ことを意味します。スケーリング効率が 0.9 なら「8 枚にして 7.2 倍速くなった（理想の 90%）」という読み方で、通信が下手だとこれが 0.5（半分しか活きない）まで落ちることもあります。多ノードでネットワークが細い（EFA/IB がない）と、まさにこの 0.5 付近まで落ちるのが現実です。

### メリット・トレードオフ・限界
メリット
- 集合通信（all-reduce、all-gather、reduce-scatter、all-to-all など）をハードウェアの高速リンクを活かして高速実行。
- Ring などのアルゴリズムにより、GPU を増やしても 1 枚あたり通信量が爆発せず大規模にスケール。Tree でレイテンシも抑える。
- トポロジ自動認識で最適経路・アルゴリズムを選び、利用者は通信詳細をほぼ意識しなくてよい。
- PyTorch DDP / FSDP の標準バックエンドで `backend="nccl"` だけで使える。エコシステムが成熟。

トレードオフ・限界
- NVIDIA GPU 専用。他社アクセラレータには別ライブラリ（AMD は RCCL、Intel は oneCCL）が必要。
- ノード間が遅いネットワークだと、アルゴリズムが優れていても通信がボトルネックになりスケールが頭打ち。GPUDirect RDMA / EFA / IB の有無が効く。
- 集合通信は同期点（全 GPU がそろうのを待つ点）を作るため、1 台でも遅い・落ちた GPU があると全体が引きずられる（ストラグラー問題）。デッドロックやハングの原因にもなりやすく、`NCCL_DEBUG=INFO` での診断や `NCCL_ASYNC_ERROR_HANDLING` の設定が定番。
- トポロジ誤検出や環境変数の設定ミスで遅い経路に落ちることがあり、運用知識を要する。
- 1 枚学習では出番がなく、分散学習を始めて初めて意味を持つ部品。

研究上の話題としては、通信圧縮（勾配を量子化・スパース化して通信量を減らす）、計算と通信のさらなるオーバーラップ自動化、耐障害性（途中で GPU が落ちても続行する弾力的集合通信）などが続いています。

### 発展トピック・研究の最前線
- 勾配圧縮（gradient compression, PowerSGD, 1-bit Adam など）で all-reduce のバイト数自体を削る方向。
- in-network reduction（スイッチ上で合算する NVIDIA SHARP）で通信を物理層に押し込み、各ノードの送受信量をさらに削減。
- FP8 / 低精度通信、通信と計算の重ね合わせをコンパイラ的に最適化する研究（例: 通信を計算カーネルに融合する手法）。
- 耐障害・弾力的な分散学習（elastic training, torchrun の elastic 機能）における集合通信の再構成・communicator 再生成。
- 超大規模での通信トポロジ最適化（rail-optimized ネットワーク、GPU 配置と通信パターンの協調設計）。

### さらに学ぶための関連トピック
- [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Spot Instance 学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)


## Nesterov Accelerated Gradient

### ひとことで言うと

Nesterov Accelerated Gradient (ネステロフ加速勾配法、略して NAG) は、普通の Momentum を少し賢くした改良版です。普通の Momentum が「今いる場所の坂を見てから慣性で進む」のに対し、NAG は「まず慣性で進むと仮定して、その着地予定地点の坂を先に見てから動く」ことで、行き過ぎ (オーバーシュート) を早めに察知してブレーキをかけられます。凸最適化では理論上最速の収束レートを達成し、深層学習でも古典 Momentum より安定して速いことが知られています。

### 直感的な理解

[SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) で見たように、Momentum はボールに慣性を持たせて学習を速くしますが、慣性が強いと谷底を通り過ぎてしまう弱点があります。谷底に近づいたとき、これまで貯めた速度のせいで「気づいたら反対側の壁まで登っていた」という行き過ぎが起こります。

運転にたとえると、普通の Momentum は「今の路面状況だけ見てアクセルを踏み続ける」運転です。これに対し NAG は「このまま進んだら数秒後にカーブに差し掛かる」と先読みして、カーブの手前で減速できる運転に近い。先読みすることで、無駄なオーバーシュートを減らし、より滑らかに谷底へ着けます。反応が一歩早いぶん、同じ慣性の強さでも安定するのです。逆に言えば、同じ安定性を保ったまま、より強い慣性 (大きな μ) を使えるので、結果として速くできます。

### 基礎: 前提となる概念

NAG を理解するには、まず古典 Momentum の更新式を押さえる必要があります。`w` が重み、`v` が速度、`μ` が momentum 係数 (例 0.9)、`η` が学習率、`g(·)` が指定した場所での勾配です。

```
v ← μ · v − η · g(w)    ← 勾配を「今の場所 w」で計算
w ← w + v
```

ここで重要なのは「次に重みが進む量は、慣性 `μ·v` の分と、今回計算した勾配 `−η·g` の分の合計」だという点です。慣性で `μ·v` だけ進むことは、勾配を計算する前から (ほぼ) 確定しています。NAG のアイデアは「どうせ `μ·v` だけ進むなら、勾配は今の場所ではなく、進んだ先で測るべきだ」というものです。確定している移動分を先に織り込んでから坂を測る、という発想です。

もう一つの前提が「凸関数」と「収束レート」です。凸関数 (convex function) とは、お椀型のきれいな関数で、極小が 1 つしかなく必ず大域最適に到達できるもの。収束レートとは、反復回数 `k` に対して誤差がどれだけ速く減るかを表します。普通の勾配法は `O(1/k)` (k に反比例) で誤差が減りますが、NAG は `O(1/k²)` (k の 2 乗に反比例) で減り、これは 1 次手法 (勾配しか使わない手法) で理論上達成可能な最速です。「1 次手法」とは、勾配 (1 階微分) だけを使い、ヘッセ行列 (2 階微分、地形の曲がり具合) を計算しない手法のこと。ヘッセ行列はパラメータ数の 2 乗のサイズになり巨大すぎて深層学習では使えないので、1 次手法の中で最速であることが実用上重要になります。

### 仕組みを詳しく

普通の Momentum と NAG の違いは「勾配をどこで計算するか」だけです。

普通の Momentum:
```
v ← μ · v − η · g(w)         ← 勾配を「今の場所 w」で計算
w ← w + v
```

NAG:
```
w_lookahead = w + μ · v        ← まず慣性 μ·v の分だけ仮に進んだ「先読み地点」を作る
v ← μ · v − η · g(w_lookahead) ← 勾配を「先読み地点」で計算する (ここが違う)
w ← w + v
```

違いは `g(w)` か `g(w + μ·v)` か、それだけです。先読み地点で測った坂のほうが、これから着地する場所の状況を正確に反映しているので、補正 (ブレーキや方向修正) が一歩早く効きます。

#### 数値例でイメージする

谷底のすぐ手前にいて、慣性 `μ·v` のせいで谷底を通り過ぎようとしている状況を考えます。

- 普通の Momentum: 今いる手前側の坂は「まだ下り」なので、勾配はさらに前へ押す方向。慣性とあわせて、谷底を大きく通り過ぎてしまう。
- NAG: 先読み地点 (= 通り過ぎた向こう側) で坂を測ると、そこは「もう上り坂」なので、勾配は戻れと押し返す方向。この戻す力が今回の更新に先取りで入るため、通り過ぎる量が小さくなる。

ASCII で描くと:

```
谷底 V
      \    /
   w ● \  /        Momentum: w で測る → まだ下り → 押しすぎる
       \/
       w+μv ○      NAG: 先読み地点で測る → もう上り → 早めに戻す
```

このように NAG は反応が一歩早いぶん、振動が小さく収まり、同じ問題でも古典 Momentum より大きな μ を安定して使えます。前述のとおり実効加速は `1/(1−μ)` 倍なので、より大きな μ を安全に使えること自体が「より速い収束」に直結します。

#### 実装上の工夫

上の式は毎回「別の場所 `w + μ·v` で勾配を計算する」ように見えますが、変数を先読み点側に置き換えると、古典 Momentum とほぼ同じ計算量・同じメモリ (速度バッファ 1 個) で実装できる書き換え (Sutskever らの定式化) が知られています。具体的には、追跡する変数を `w` ではなく先読み点 `θ = w + μ·v` に取り直すと、毎ステップ「現在地で勾配を 1 回測る」だけの形に整理でき、追加の forward/backward は不要になります。多くのフレームワークでは Momentum の SGD に nesterov オプションを付けるだけで有効になり、勾配計算の回数は 1 回のまま変わりません。つまり NAG は「ほぼタダで」古典 Momentum を改良できる、コスト対効果の良い選択肢です。

### 手法の系譜と主要論文

Nesterov, Y. (1983) "A method for solving the convex programming problem with convergence rate O(1/k²)": 原典です。提案内容 — 凸関数の最小化で、普通の勾配法の収束速度 `O(1/k)` を `O(1/k²)` に改善する加速法。理由 — 先読み点での勾配を使うことで、理論上達成可能な最速の収束レート (凸最適化の下限、Nemirovski-Yudin による) に到達できる。効果 — 反復回数 `k` に対し誤差が `1/k²` で減る、理論的に最適な 1 次手法。トレードオフ — 理論保証はあくまで凸関数 (とくに滑らかな凸関数) の話で、ニューラルネットのような非凸 (でこぼこで極小が無数にある) 関数では最適性はそのままは成り立たない。

Sutskever, Martens, Dahl & Hinton (2013) "On the importance of initialization and momentum in deep learning" (ICML 2013): 凸最適化用だった NAG を、深層学習という非凸問題でも実用的に使える形に書き換え、古典 Momentum より安定して速いことを実験的に示した。理由 — 先読みにより、慣性で行き過ぎそうなときに勾配が早めに補正をかけるため、大きな μ を使っても発散しにくい。効果 — 深い RNN を含む難しいネットワークの学習で、古典 Momentum より良い結果。トレードオフ — それでも μ と η の調整は必要で、適切な momentum スケジュール (序盤は小さく) が重要と指摘。

Su, Boyd & Candès (2014/2015) "A Differential Equation for Modeling Nesterov's Accelerated Gradient" (arXiv:1503.01243): NAG を連続時間の 2 階微分方程式 (ODE) として解釈し、加速のメカニズムを「時間とともに弱まる摩擦をもつ運動」として説明しました。`ẍ + (3/t)·ẋ + ∇f(x) = 0` という方程式で、`3/t` という時間依存の摩擦項が NAG 特有の加速を生むことを示し、なぜ NAG が速いのかを物理的・幾何学的に理解する道を開いた重要な仕事です。

Dozat (2016) "Incorporating Nesterov Momentum into Adam" (Nadam): NAG の「先読み」を Adam の 1 次モーメントに組み込んだ Nadam を提案。Adam の適応的学習率に Nesterov 流の先読み補正を加えることで、一部のタスクで Adam より速い収束を示した。NAG の核心が現代の適応オプティマイザにも生きていることを示す例です。

### 論文の実験結果 (定量データ)

Nesterov (1983) の貢献は理論的で、滑らかな凸関数において誤差 `f(w_k) − f(w*)` が `O(1/k²)` で減ることを証明しました。これは古典勾配降下の `O(1/k)` に対し、例えば k=100 回反復したとき、誤差がおよそ 1/100 ではなく 1/10000 のオーダーまで減ることを意味し、大きな k で劇的な差になります。`O(1/k²)` は 1 次手法の理論下限に一致するため、「これ以上速い 1 次手法は (最悪ケースでは) 存在しない」最適性が保証されます。

Sutskever ら (2013) は深い feedforward ネットや RNN で、古典 Momentum と NAG を同条件で比較し、NAG のほうが高い μ (0.99 付近) でも安定して学習でき、最終的な学習成功率と loss / perplexity で古典 Momentum を上回ることを報告しました。特に「慣性が強いほど加速するが発散しやすい」領域で、NAG の先読み補正が発散を防ぐことが鍵だと示しています。彼らの実験では、初期化と momentum スケジュールを適切に選んだ NAG が、当時最先端だった Hessian-Free のような計算の重い 2 次手法に匹敵する学習成功を、はるかに低コストで達成しました。

アブレーション的な観点 (どの要素を抜くとどうなるか): NAG から先読み (`w + μ·v` での勾配計算) を外すと古典 Momentum に戻り、同じ μ では振動・オーバーシュートが増えます。逆に NAG では同じ安定性をより大きな μ で保てるため、実効的な加速量 `1/(1−μ)` を大きく取れます。momentum スケジュール (序盤 μ を小さく、徐々に 0.99 へ) を外して固定値にすると、学習初期に発散しやすくなることも報告されています。ただし非凸な深層学習では、古典 Momentum との差が問題によっては小さく、必ず勝つわけではないことも示されています。

指標の意味の補足: perplexity (パープレキシティ) は言語モデルの予測の良さを測る指標で、「次の単語の候補をどれだけ絞れているか」を表し、小さいほど次の単語をうまく当てられていることを意味します (例えば perplexity が 100 なら平均的に 100 択で迷っている感覚)。NAG が同条件で低い perplexity に到達するのは、より速く・より安定に良い極小へ収束できていることの表れです。

### メリット・トレードオフ・限界

メリット:
- 先読みによって行き過ぎを早めに補正でき、古典 Momentum より振動が小さく収束が安定・高速。
- 凸問題では理論上最速 (`O(1/k²)`) の収束保証がある。
- 追加メモリ・計算コストは古典 Momentum とほぼ同じ。フレームワークでは nesterov フラグだけで有効。同じ μ ならほぼタダで安定性が上がる。

トレードオフ・限界:
- 非凸なニューラルネットでは理論上の最適性はそのまま成り立たず、効果は問題依存。
- 依然として μ と η の手調整が必要。
- パラメータごとの学習率自動調整はない (その点は Adam/RMSProp の領分)。
- 古典 Momentum との差は問題によっては小さく、必ず勝つわけではない。ミニバッチ勾配のノイズが大きい設定では、先読みの利点がノイズに埋もれて差が縮むことがある。
- 研究上の論点: なぜ Nesterov 加速が起こるのかの幾何学的・力学的な理解 (連続時間 ODE としての解釈、微分方程式の視点からの解析) は今も研究テーマであり、加速の本質の解明が続いています。

### 発展トピック・研究の最前線

NAG は最適化理論と深層学習の両方で発展の起点になっています。Su, Boyd & Candès (2014) は NAG を連続時間の微分方程式 (2 階 ODE) として解釈し、加速のメカニズムを物理的な「摩擦付きの運動」として理解する道を開きました。これを発展させた variational/Hamiltonian な視点 (Wibisono, Wilson & Jordan 2016) は、加速法を「ある作用汎関数 (Bregman Lagrangian) を最小化する運動」として統一的に説明する枠組みを与え、離散化の仕方が安定性を左右することを明らかにしました。実用面では Nadam が NAG を Adam に取り込み、近年の大規模学習でも適応オプティマイザに先読みの発想が取り入れられています。restart 付き加速法 (O'Donoghue & Candès 2015) は、非凸領域で加速が裏目に出るのを「速度をリセットする」ことで防ぎ、実用的な堅牢性を上げました。古典 Momentum と NAG の比較は「慣性をどう制御すれば速く・安定に収束するか」という最適化の根本問題への入口であり、すべての加速法を理解する基礎です。

### さらに学ぶための関連トピック

- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## One-Cycle 学習率ポリシー

### ひとことで言うと
One-Cycle(ワンサイクル)は、学習の最中に学習率を「いったん大きく上げて、それから下げきる」という 1 回だけの山を描かせるスケジューラです。一時的にかなり高い学習率を通すことで、通常より大幅に速く高精度へ収束する現象(super-convergence、超収束)を狙います。同時にモメンタム(過去の勾配をどれだけ引き継ぐかの度合い)を学習率と逆向きに動かして安定性を保ちます。

### 直感的な理解
ほとんどの学習率スケジューラは「大きいと速いが危険、小さいと安全だが遅い」というトレードオフに対し、安全側に倒して学習率を最初から単調に下げていきます([Cosine Annealing 学習率スケジュール](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) など)。

Leslie Smith はここに逆張りの問いを立てました。「学習の途中で、あえて普段なら発散しそうなくらい高い学習率を通したら何が起きるか」。彼の観察では、適切に設計すれば発散せず、むしろ学習が劇的に速まり、汎化性能(未知データでの精度)まで上がることがありました。これを super-convergence と名付けました。

なぜ高学習率が「効く」のか。直感的にはこうです。損失地形には「鋭く狭い谷(過学習しやすい解)」と「広くなだらかな谷(汎化しやすい解)」があります。高い学習率の区間では歩幅が大きすぎて鋭い谷には留まれず弾き出され、結果として広い谷へモデルが押しやられます。つまり高学習率区間が一種の正則化(過学習を防ぐ働き)になるのです。その後で学習率を十分下げきると、たどり着いた広い谷の底に丁寧に収まります。One-Cycle は「高学習率による探索と正則化」+「低学習率による収束」を、たった 1 つの山で両立させる設計です。

### 基礎: 前提となる概念
モメンタムとは、勾配降下に「慣性」を与える仕組みです。更新方向を毎回ゼロから決めるのではなく、過去の更新方向を一定割合で引き継ぎます。坂を転がるボールのように勢いがつき、ジグザグを減らし収束を速めます。モメンタム係数(0.9 など)が大きいほど過去を強く引き継ぎます。One-Cycle ではこのモメンタムを学習率と逆位相で動かすのが特徴です。

正則化とは、モデルが訓練データに過剰に適合する(過学習する)のを抑え、未知データでの性能を保つための工夫の総称です。weight decay やドロップアウトが代表ですが、Smith は「高学習率そのものが正則化として働く」と主張しました。背景には「最適化と汎化は別物で、収束を速める設定がそのまま汎化を良くするとは限らない」という深層学習の難しさがあり、One-Cycle はその両方を同時に狙う点が野心的です。

LR range test(学習率レンジテスト)とは、One-Cycle の上限学習率 η_max を決めるための事前手続きです。学習率を非常に小さい値(例 1e-7)から指数的に上げながら数百ステップだけ流し、損失が下がり続ける限界の手前(損失が上昇に転じる直前)の値を η_max に採ります。「壊れる一歩手前まで攻める」ための測定です。具体的には、横軸を学習率(対数)、縦軸を損失にプロットし、損失が最も急に下がっている付近、または最小値の少し手前を選びます。

### 仕組みを詳しく
One-Cycle は学習全体を 1 つのサイクルとして、学習率を 3 局面で動かします。η_max を上限、η_min をその 1/10〜1/25 程度とします。

1. 上昇フェーズ(全体の約 30〜45%): 学習率を η_min から η_max へ上げる。
2. 下降フェーズ(残りの大半): 学習率を η_max から η_min へ下げる。
3. 最終アニーリング(終盤の数%): η_min よりさらに桁違いに小さい値(例: η_max/1000)まで一気に下げきり、解を磨き込む。

上昇・下降の形は当初は三角(線形)で提案され、その後コサイン形(`cos` で滑らかに上げ下げ)がよく使われます。同時に重要なのがモメンタムを学習率と逆向きに動かすことです。

- 学習率が高いとき → モメンタムを低く(例 0.95 → 0.85)。理由: 学習率が高くて動きが大きいのに過去まで強く引き継ぐと、勢いがつきすぎて発散しやすい。だから高学習率区間ではモメンタムを下げてブレーキ役にする。実効的な「過去の勾配の蓄積倍率」は 1/(1−momentum) なので、0.95 だと 20 倍、0.85 だと約 6.7 倍と大きく変わり、高学習率時に勢いを抑える意味が定量的にも分かります。
- 学習率が低いとき → モメンタムを高く(例 0.85 → 0.95)に戻す。理由: 動きが小さいので、過去を強く引き継いで収束を加速したい。

数値例。総ステップ 10000、η_max=0.01、上昇 45%、最終 5% とすると、

- t=0: η=0.0004(≈ η_max/25)、モメンタム 0.95。
- t=4500(上昇終わり): η=0.01(ピーク)、モメンタム 0.85。
- t=9500(下降終わり): η≈0.0004、モメンタム 0.95。
- t=10000(最終): η≈0.00001(η_max/1000)で締める。

```
η                       モメンタム
η_max|   .-*-.        高 |*-.        .-*
     |  /     \          |   `-.  .-'
     | /       \         |      `*'
η_min|/         `*..   低 |________________
     0  ↑up  ↓down end       (η と逆位相)
```

設計の心臓部は η_max の決め方で、ここで LR range test を使います。学習率を小さい値から指数的に上げながら短く流し、損失曲線を見て「まだ下がり続けている最大の学習率」を η_max に選びます。高すぎれば発散し、低すぎれば super-convergence の恩恵が得られないため、この測定が成否を分けます。

なぜ warmup(上昇フェーズ)が前半に必要かというと、初期のランダムに近い重みでいきなり η_max を当てると壊れるためで、上昇フェーズが事実上の warmup を兼ねています。One-Cycle は warmup・高学習率正則化・最終磨き込みを 1 本のスケジュールに統合した、と見ることができます。ここが [学習率 Warmup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) + [Cosine Annealing 学習率スケジュール](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の組み合わせとの大きな違いで、後者は「上げてから単調に下げる」だけなのに対し、One-Cycle は「上げて・下げて・さらに下げきる」3 段構成かつモメンタムを連動させる点が独自です。

### 手法の系譜と主要論文
- Smith, "Cyclical Learning Rates for Training Neural Networks" (WACV 2017, 米海軍研究所 USNRL, arXiv:1506.01186)。One-Cycle の前身で、学習率を周期的に上下させる CLR(Cyclical Learning Rates)を提案。動機は「最適学習率を 1 つに固定するより、ある範囲を行き来させた方が局所解(鞍点)に捕まりにくく、手動チューニングも減る」こと。LR range test もこの論文で導入されました。

- Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates" (2018, arXiv:1708.07120)。One-Cycle そのものの提案。CLR の周期を「ただ 1 回」に絞り、高学習率区間を正則化として積極利用して超収束を狙いました。仮説は「高学習率区間が鋭い解を避けて広く汎化する解へ導く」こと。論文は損失地形のヘッセ行列(曲率)の近似解析を通じて、高学習率が大きな曲率の方向を避ける効果を議論しています。

- Smith, "A Disciplined Approach to Neural Network Hyper-Parameters: Part 1" (2018, arXiv:1803.09820)。学習率・バッチサイズ・モメンタム・weight decay を体系的に決める手順をまとめ、One-Cycle と LR range test を実務手順として整理しました。学習率とモメンタムは逆方向に、学習率と weight decay も連動して調整すべき、という実践的指針を与えています。

- Foret et al., "Sharpness-Aware Minimization (SAM)" (ICLR 2021, arXiv:2010.01412)。One-Cycle が暗黙に狙っていた「平坦な解=汎化しやすい解」を、損失地形の鋭さ(sharpness)を明示的に最小化することで直接追求する手法。One-Cycle の「高学習率による暗黙の正則化」という主張を、明示的最適化として後から定式化した位置づけとして対比されます。

- 実装面では fast.ai ライブラリの `fit_one_cycle` として普及し、画像分類の高速学習レシピとして広く使われました。主要フレームワークにも `OneCycleLR` として標準実装があります。これは [Cosine Annealing 学習率スケジュール](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) + [学習率 Warmup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) と並ぶ実用スケジューラの一系統です。

### 論文の実験結果(定量データ)
評価指標は test accuracy(正解率)と、目標精度に到達するまでのエポック数・反復回数(少ないほど速い)です。

Super-Convergence 論文の代表的な実験は CIFAR-10(32×32 カラー画像を 10 クラスに分類)です。ResNet 系のネットワークで、通常の学習が数万〜十数万反復を要するところ、One-Cycle は数分の 1 の反復数で同等以上の test accuracy(おおむね 90% 台前半)に到達したと報告されています。論文中の象徴的な結果として、ある設定で通常レシピより大幅に少ないエポック(報告では従来の十分の一程度の反復で済んだ例も)で同等精度に達した事例が示され、これが「super-convergence」と呼ばれる根拠になりました。データが少ない設定ほど高学習率の正則化効果が相対的に大きく、超収束が顕著に出やすいことも観察されています。

重要なアブレーションが複数あります。(1) 一定学習率と比べ、One-Cycle の高学習率ピークを通したほうが最終精度が同等以上かつ到達が速い。これが核心の主張です。(2) モメンタムを学習率と逆に動かさず固定すると、高学習率区間で不安定になりやすく、逆位相にすることで安定と速度が両立する。(3) 効果は weight decay などの他の正則化設定と強く相互作用し、One-Cycle で高学習率を使うときは weight decay をやや弱める、といった調整が必要になる(高学習率自体が正則化を担うため、二重に効かせると過正則化になる)。(4) η_max を LR range test より高く取りすぎると発散し、低すぎると恩恵が消える。

数値の意味を補足すると、CIFAR-10 で 90% 台前半の精度は標準的な達成ラインで、One-Cycle の貢献は「精度を保ったまま学習時間を大きく削る」点にあります。一方で、この超収束効果はデータセット・モデル・正則化設定への依存が強く、論文の劇的な高速化が他の設定で必ず再現するとは限らないことも繰り返し指摘されています。特に ImageNet 規模や大規模言語モデルでは、CIFAR ほど劇的な超収束は報告されにくいことが知られています。

### メリット・トレードオフ・限界
メリット。高学習率区間の正則化効果で、通常より大幅に速い収束(super-convergence)が得られることがあります。学習率とモメンタムを連動させることで安定性と速度を両立できます。warmup・高学習率・最終アニーリングが 1 本のスケジュールに統合されており、主要フレームワークに標準実装があって導入が容易です。中規模モデルのプロトタイピングやファインチューニングで、短時間で良い精度を出すのに向きます。

トレードオフと限界。η_max が高すぎると発散するため LR range test など慎重な事前設定が必要です。super-convergence の効果はデータ・モデル・正則化に強く依存し、再現しないこともあります。総ステップ数を事前固定する前提で、学習延長や途中変更に弱いです。高学習率区間は数値的に攻めるため、bf16 など低精度では発散により注意が要ります。また理論的裏付けは「広い谷へ導く正則化」という仮説段階で、コサインや warmup ほど普遍的な定番にはなりきっていません。大規模言語モデルの事前学習ではほとんど使われず、適用範囲が中規模の教師あり学習に偏ります。

研究上の未解決課題としては、どんなタスク・モデルで超収束が起きるのかの予測理論、LR range test の自動化と頑健化、超大規模モデルでの再現性検証(現状の代表的成功例は中規模の画像分類に偏る)が挙げられます。

### 発展トピック・研究の最前線
One-Cycle の「高学習率を正則化として使う」「学習率とモメンタムを協調させる」という発想は、その後のスケジューラ設計やハイパーパラメータ自動調整の研究に影響を与えました。近年の大規模言語モデルでは、計画を固定せず途中延長しやすい WSD(Warmup-Stable-Decay)スケジュールが好まれる傾向があり、One-Cycle の「総ステップ固定が前提」という弱点が相対的に目立つ場面もあります。一方で、限られた計算で素早く一通り学習を回すプロトタイピングや、中規模モデルの高速ファインチューニングでは依然有力な選択肢です。損失地形の鋭さ(sharpness)と汎化の関係を直接最適化する SAM のような手法と、高学習率による暗黙の正則化という One-Cycle の主張を結びつけて理解しようとする研究、また大バッチ学習で学習率とバッチサイズをスケールさせる線形スケーリング則と One-Cycle を組み合わせる実務的レシピの研究も進んでいます。

### さらに学ぶための関連トピック
- [Cosine Annealing 学習率スケジュール](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [学習率 Warmup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [学習率スケジューラ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Pipeline Parallelism

### ひとことで言うと
とても大きなニューラルネットワーク（たくさんの「層」を積み重ねた計算の連なり）は、GPU 1 枚のメモリに乗り切らないことがあります。Pipeline Parallelism（パイプライン並列、層方向分割）とは、モデルを層のかたまりで前半・中盤・後半…と分割し、それぞれを別の GPU に置いて、工場のベルトコンベア（流れ作業）のようにデータを順番に流して計算する手法です。1 枚に乗らない巨大モデルを複数 GPU に分けて学習できるようにします。

### 直感的な理解
洗濯を 4 工程（洗う → すすぐ → 乾かす → たたむ）に分け、各工程を別の人が担当する場面を想像してください。1 枚の服しか流さないと、洗う人が働いている間、すすぐ・乾かす・たたむ担当は手持ち無沙汰です。これでは 4 人いても 1 人分の速さしか出ません。

ところが服を次々と投入すれば、1 枚目を乾かしている頃には 2 枚目をすすぎ、3 枚目を洗っている、という具合に全員が同時に働けます。これがパイプライン（流れ作業）の発想です。ニューラルネットワークも「層 1 → 層 2 → 層 3 → 層 4」と順番に計算が流れる構造なので、各層を別の GPU に置き、データを次々と流せば全 GPU を働かせられます。立ち上がりと終わりだけは一部の人が遊びますが、これを工夫で減らすのがこの分野の中心テーマです。

深層学習特有の難しさが一点あります。洗濯は一方通行ですが、ニューラルネットの学習は「順方向に流して答えを出し（順伝播）、答えの誤差を逆方向に流して各層の直し方を計算する（逆伝播）」という往復構造です。しかも逆伝播のためには順伝播の途中結果（中間活性）を取っておかねばならず、これがメモリを食い、スケジューリングを複雑にします。パイプライン並列の研究史は、ほぼこの「順伝播と逆伝播をどう噛み合わせて GPU を遊ばせず、かつ活性メモリを抑えるか」というスケジューリングの工夫の歴史です。

### 基礎: 前提となる概念

- 層（layer）: ニューラルネットワークの計算の一単位。入力テンソルを受け取り、変換して出力テンソルを返す。これを多数積み重ねたものが深いネットワーク。
- 順伝播（forward pass, F）: 入力から出力へ計算を進めること。
- 逆伝播（backward pass, B）: 出力側で計算した誤差（loss）から、各層の重みをどう動かせば誤差が減るか（勾配）を、出力側から入力側へさかのぼって計算すること。逆伝播は順伝播のおよそ 2 倍の計算量を要するのが経験則。
- 中間活性（activation）: 順伝播の途中で各層が出力するテンソル。逆伝播で勾配を計算するため保持が必要。これがメモリを大量に食う。
- VRAM / GPU メモリ: GPU が持つ専用メモリ。上限があり、たとえば 1 枚 40〜80GB クラス。
- ミニバッチ（mini-batch）: 一度に処理するデータのかたまり（例: 32 サンプル）。1 回の重み更新の単位。
- マイクロバッチ（micro-batch）: ミニバッチをさらに細かく割ったもの。パイプラインに次々流す単位。

なぜ 1 枚に乗らないのかを数値で押さえます。学習時には VRAM に (1) モデルの重み、(2) 各重みの勾配（重みと同数）、(3) オプティマイザの状態（Adam なら重み 1 個あたり追加 2 値）、(4) 中間活性、をすべて置く必要があります。パラメータ 100 億個（10B）のモデルを混合精度 + Adam で学習すると、重み・勾配・オプティマイザ状態だけでパラメータ 1 個あたり概ね 16〜18 バイト要し、合計 160〜180GB を超えます。これは単一 GPU にはどう逆立ちしても乗りません。

このときモデルそのものを分割する「モデル並列」が必要になります。パイプライン並列はその一種で、層の方向（深さ方向）で輪切りにします。深さ方向に切るのが自然なのは、計算が層を順に流れる構造だからです。なお「データ並列」（同じモデルのコピーを各 GPU に置き、別々のデータを処理して勾配を平均する手法）は、モデルのコピーが各 GPU に丸ごと必要なので、そもそも 1 枚に乗らないモデルには単独では使えません。

### 仕組みを詳しく

#### 単純に分けただけだと GPU が遊ぶ（パイプラインバブル）

4 層のモデルを 4 枚の GPU に 1 層ずつ置きます。

```
GPU0: 層1   GPU1: 層2   GPU2: 層3   GPU3: 層4
```

ミニバッチを 1 つ流すと、まず GPU0 が層1を計算し結果を GPU1 へ送り、GPU1 が層2を…と進みます。問題は、GPU0 が計算している間、GPU1・2・3 は前の結果を待って何もしていないこと。1 枚ずつしか働かず、4 枚あっても実質 1 枚分の速度しか出ません。この遊び時間を「パイプラインバブル（気泡）」と呼びます。GPU が P 個・ミニバッチを丸ごと流す素朴な方式では、バブルの割合は約 (P−1)/P にもなり、4 枚なら 75% が無駄になります。

#### マイクロバッチで気泡を減らす（GPipe の核心）

GPipe のアイデアは、ミニバッチ（例: 32 サンプル）をさらに小さい「マイクロバッチ」（例: 8 サンプル × 4 個）に分割し、ベルトコンベアに次々と流すことです。

```
時間 →
GPU0: m1  m2  m3  m4
GPU1:     m1  m2  m3  m4
GPU2:         m1  m2  m3  m4
GPU3:             m1  m2  m3  m4   ← この区間で全 GPU が同時稼働
```

m1 が GPU1 に進んだら GPU0 はすぐ次の m2 に取りかかります。マイクロバッチ数を M とすると、バブルの割合はおよそ (P−1)/(M+P−1) まで下がります。具体的に P=4, M=4 ならバブル率は 3/7 ≈ 43%、M=16 なら 3/19 ≈ 16%、M=32 なら 3/35 ≈ 9% と、M を大きくするほど効率が上がります。GPipe はさらに、逆伝播時に中間活性を全部保持せず必要時に再計算する「活性の再計算（activation recomputation / checkpointing）」でメモリを節約します。これは「メモリを節約するために計算をやり直す」という、メモリと計算量のトレードオフの典型です。

#### PipeDream の工夫（1F1B スケジューリング）

GPipe は全マイクロバッチを順伝播してから一斉に逆伝播するため、順伝播の途中結果（活性）を全部抱える時間が長く、活性メモリのピークが M に比例して膨らみます。M を増やすとバブルは減るがメモリは増える、という板挟みになります。PipeDream は「1F1B（one-forward-one-backward）」スケジュールを提案しました。各 GPU が「1 つ順伝播したら、できるだけ早く 1 つ逆伝播する」ように交互に回す方式で、同時に抱える活性の量を最大 P 段ぶん程度に抑え、メモリ効率を上げます。バブル率は GPipe と同じ (P−1)/(M+P−1) のまま、メモリだけを削れる点が美味しいところです。

```
GPipe (all-forward then all-backward):
  F F F F | B B B B    ← 活性を 4 つ分ずっと保持

1F1B (PipeDream):
  F F F F B F B F B B   ← 逆伝播を早めに差し込み、保持する活性を減らす
（F=順伝播, B=逆伝播。実際は GPU ごとにずれて並ぶ）
```

PipeDream はさらに非同期に複数バッチを流す方式も検討しましたが、これだと重みの更新タイミングがずれ、「順伝播に使った重み」と「逆伝播時の重み」が食い違う問題（weight staleness、重みの古さ）が生じます。これは数学的には勾配が古い重みに対するものになる現象で、放置すると収束が乱れます。PipeDream は世代ごとに重みのコピーを保持する「weight stashing」で整合を取りました（順伝播に使った重みを逆伝播まで保管して同じものを使う）。

#### 同期型 1F1B と後継のスケジューリング改良

PipeDream の非同期はスループットは高いが収束保証が弱いため、実用では「同期型 1F1B」（マイクロバッチをまたいで重みを更新せず、ミニバッチ末で一括更新する）が主流になりました。DAPPLE（Fan et al., PPoPP 2021, Alibaba）はこの同期型を明確に定式化し、データ並列との組み合わせ最適化を示しました。以下は主な改良です。

- PipeDream-2BW（ICML 2021）: 重みのコピーを 2 つ（double-buffered weights）に抑え、weight stashing のメモリ負担を削減しつつ整合を保つ。
- Megatron の interleaved 1F1B（SC 2021）: 各 GPU に「連続した層のかたまり」でなく「飛び飛びの複数の小さなかたまり（virtual pipeline stage）」を割り当てることで、バブルを v 分の 1（v はステージ細分化数）に縮める。そのぶん通信回数が v 倍になるトレードオフがある。
- Chimera（Li & Hoefler, SC 2021）: 順方向と逆方向の双方向パイプライン（two pipelines が逆向きに流れる）を噛み合わせ、バブルをほぼ半減させる。代償としてモデルの重みを 2 系統持つためメモリが増える。
- Zero Bubble（Qi et al., ICLR 2024）: 逆伝播を「入力に対する勾配（B）」と「重みに対する勾配（W）」の二つに分解し、W を隙間に後回しで詰めることで、理論上バブルをゼロに近づける。最新の到達点。

#### 他の並列化との組み合わせ（3D 並列）

実際の超大規模学習では、パイプライン並列を単独で使わず、データ並列・テンソル並列と組み合わせた「3D 並列」にするのが定番です。テンソル並列で 1 層の中の行列演算を割り、パイプライン並列で層方向に割り、データ並列でさらに横に増やします。配置の鉄則は「通信頻度の高いテンソル並列を高速な同一サーバ内に、通信頻度の低いパイプライン並列をサーバ間に」割り当てることです。詳しくは [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) と [DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) を参照してください。

### 手法の系譜と主要論文

- GPipe（Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", NeurIPS 2019, Google）。マイクロバッチ分割でバブルを減らし、活性再計算でメモリを節約する、パイプライン並列の基礎を確立。動機は、当時のモデルが GPU メモリの壁にぶつかり、複数アクセラレータにまたがって 1 つの巨大モデルを効率よく学習する汎用手法が必要だったこと。
- PipeDream（Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training", SOSP 2019, Microsoft Research）。1F1B スケジューリングと weight stashing を導入し、メモリ効率とスループットを改善。GPipe の「全部順伝播してから逆伝播」が活性を長く抱える点を狙った改良。
- PipeDream-2BW（Narayanan et al., ICML 2021）。double-buffered weights でメモリを抑えつつ整合を取る改良版。
- DAPPLE（Fan et al., PPoPP 2021, Alibaba）。同期型 1F1B を定式化し、パイプライン分割とデータ並列の最適配置を探索。
- Megatron-LM のクラスタ論文（Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC 2021, NVIDIA/Microsoft/Stanford）。テンソル・パイプライン・データの 3D 並列を数千 GPU で高効率に回す配置と、interleaved 1F1B を提示。パイプライン並列を実用の超大規模学習に組み込む定石を確立。
- Chimera（Li & Hoefler, SC 2021, ETH Zurich）。双方向パイプラインでバブルをほぼ半減する新スケジュール。
- Zero Bubble（Qi et al., ICLR 2024, Sea AI Lab）。B/W 分解でバブルを理論ゼロまで詰める最新手法。

（注: 一部の著者・年は要約を含むため、引用時は原典確認を推奨します。）

### 論文の実験結果（定量データ）

指標の意味から説明します。

- スループット（throughput）: 単位時間あたり処理できるサンプル数や、達成した実効的な演算性能（TFLOP/s）。高いほど良い。
- スケーリング効率（scaling efficiency）: GPU を N 倍にしたとき性能が理想の N 倍にどれだけ近いか。1.0 に近いほど良い。
- バブル率（bubble fraction）: パイプラインの遊び時間の割合。低いほど良い。
- MFU（model FLOPs utilization）: GPU の理論ピーク性能に対する実効性能の割合。50% 超えれば大規模分散として優秀とされる。

報告されている代表的な結果です。

- GPipe（NeurIPS 2019）は、AmoebaNet（画像分類）を 8 アクセラレータに分割し、活性再計算と併用してメモリ内に約 25 倍大きいモデルを収められたと報告。マイクロバッチ数を増やすほどスケーリング効率が理想線に近づくことを実験で示し、4 アクセラレータで概ね 3.5 倍前後の高速化を達成。アブレーションとして、マイクロバッチ数を減らすとバブルが増え効率が落ちることを確認している。
- PipeDream（SOSP 2019）は、画像分類・翻訳など複数タスクで、データ並列のみの構成に比べて目標精度到達までの時間を最大で約 5 倍短縮した場合があると報告。1F1B により活性メモリを GPipe 比で大きく削減できることも示した。
- Megatron のクラスタ論文（SC 2021）は、最大 1 兆パラメータ級の言語モデルを 3072 個の A100 GPU で学習し、約 502 PFLOP/s、ピーク性能比でおよそ 52%（MFU 約 52%）という当時として極めて高い実効効率を達成。アブレーションでは、テンソル並列をサーバ内・パイプライン並列をサーバ間に割り当てる配置が最良で、逆にすると通信が遅いネットワークを多用してしまい効率が落ちることを示した。interleaved スケジュールが標準 1F1B よりバブルを減らし、スループットを 1 割前後改善する例も報告。
- Chimera（SC 2021）は、双方向パイプラインにより GPipe や 1F1B 比でバブルを大きく削り、言語モデル学習でスループットを 1〜2 割改善したと報告。代償として重みを 2 系統保持するメモリ増を明示。
- Zero Bubble（ICLR 2024）は、B/W 分解スケジュールにより、同期型 1F1B 比でスループットを最大で約 15〜30% 改善し、メモリ制約下でも従来手法を上回ったと報告。バブルを理論ゼロまで詰められることを実測で裏付けた。

数値の意味を補足します。「MFU 52%」とは、GPU が理論上出せる計算速度の半分を実際に引き出せた、ということ。大規模分散では通信待ちやバブルで効率がすぐ 20〜30% に落ちるため、50% 超えは配置設計が極めてうまくいった証拠です。「目標精度到達まで最大 5 倍短縮」は、本来 5 日かかる学習が 1 日で終わる差であり、研究の反復速度を直接左右します。

### メリット・トレードオフ・限界

メリット
- GPU 1 枚に乗り切らない巨大モデルを、複数 GPU に層方向で分割して学習できる。
- マイクロバッチと 1F1B / interleaved / Zero Bubble スケジューリングで、バブル（遊び時間）を実用上許容できる水準まで減らせる。
- テンソル並列・データ並列と組み合わせ、数千 GPU 規模まで拡張できる。
- ステージ間で送るのは境界のテンソル（活性とその勾配）だけなので、テンソル並列より通信頻度が低く、サーバ間（やや遅い接続）でも使いやすい。

トレードオフ・限界
- 立ち上がりと終わりに必ずバブルが残り、マイクロバッチが少ないと無駄が大きい。バッチを大きくすればバブルは減るが、有効バッチサイズの増大は収束に影響する別の制約を生む。
- 各ステージの計算量を均等に割らないと、重いステージがボトルネックになる（負荷分散が難しい。層ごとに計算量が違い、特に埋め込み層や最終層は重さが偏る）。
- ステージ間でテンソルを送受信する通信が必要で、ネットワークが遅いと性能が落ちる。
- 非同期方式では重みの古さ（staleness）への対処が複雑で、weight stashing 等が追加メモリを使う。同期方式はその問題はないがバブルが残る。
- 1 枚に乗るモデルには不要で、むしろ通信オーバーヘッドぶん遅くなるだけ。

研究上の未解決課題としては、(1) 異種構造モデル（層ごとに計算量が大きく違う、分岐がある）での自動的な最適分割、(2) バブルをゼロに近づけつつメモリ増を抑えるスケジュール、(3) パイプライン・テンソル・データ・シーケンスなど多軸並列の最適配置の自動探索が挙げられます。

### 発展トピック・研究の最前線

- ゼロバブルに近いスケジューリング: 逆伝播を「入力に対する勾配」と「重みに対する勾配」に分割して隙間を埋める Zero Bubble 系の研究と、その後続。
- 自動並列化（auto-parallelism）: Alpa（OSDI 2022）のように、与えられたモデルとクラスタに対し、データ/テンソル/パイプライン並列の最適な組み合わせを自動で探索・生成する。
- メモリと通信の協調最適化: 活性再計算・オフロード（CPU や NVMe へ退避）とパイプラインの併用。
- 耐障害パイプライン: Spot 中断などでステージが消えたときに、巻き戻しなしで復旧するスケジュール（[Spot Instance 学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci) で触れる Bamboo / Oobleck）。
- 長系列・MoE（Mixture of Experts）モデルでの専用スケジューリング。

### さらに学ぶための関連トピック
- [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [NCCL (集合通信)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Spot Instance 学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)
- [Model Registry (Stage 遷移)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)


## Polynomial LR Decay

### ひとことで言うと
学習率(ニューラルネットのパラメータを1回どれくらい動かすかを決める数値)を、多項式(数学で「ある量の何乗か」を使った式。ここでは「残り進捗割合の何乗」)の形にそって減らし、あらかじめ決めた総ステップ数でちょうど 0(または指定の最小値)に到達させるスケジューラ(学習率を時間とともに変化させる仕組み)です。特に乗数を 0.9 にした「poly」スケジュールはセマンティックセグメンテーション(画像の各ピクセルが何の物体かを塗り分けるタスク)の定番として広く使われ、乗数 1(=線形減衰)は BERT 以来の言語モデル事前学習の定番です。

### 直感的な理解
マラソンでゴールまでの距離が決まっているとします。最初は速いペースで走り、ゴールに近づくにつれてペースを落とし、ゴール地点でちょうど止まるよう逆算してペース配分する、という走り方を想像してください。

指数減衰には「決して 0 に到達しない」という性質がありました。これは「学習を○○ステップで終え、その瞬間にちょうど学習率を 0 にしたい」という場面では不便です。マラソンでいえば、ゴールしてもまだ走り続けてしまう感じです。

学習の総量(エポック数や総イテレーション数)が最初から決まっているなら、その終点でぴたりと 0 になるよう逆算して減らすほうが、終盤に学習率が残りすぎず、きれいに収束します。Polynomial decay はこの「終点を指定できる」減衰です。残りの進捗割合を何乗かして掛けることで、終点で必ず 0(または min_lr)に着地します。乗数(power、べき)を変えると、下り坂の形を直線から強い反りまで自由に調整できます。

### 基礎: 前提となる概念
用語を噛み砕きます。

- セマンティックセグメンテーション: 画像の1ピクセルごとに「これは道路」「これは車」と分類するタスク。物体の輪郭まで塗り分ける、密な(dense)予測です。
- mIoU(mean Intersection over Union): セグメンテーションの代表的な評価指標。予測した領域と正解の領域がどれくらい重なっているかを、クラスごとに「重なり ÷ 和集合」で測り、全クラスで平均した値。0〜1(または 0〜100%)で、高いほど良い。重なりも和集合もピクセル数で数えます。
- 総ステップ数(T_max): 学習を何回のパラメータ更新で終えるか、最初から決めた値。Polynomial decay はこれを使って終点を逆算します。
- 多項式とべき(power): `x` を power 乗すること。power=1 は直線、power=2 は放物線。べきの値で曲線の反り方が変わります。

密予測タスク(セグメンテーションのようにピクセル単位で出力するタスク)は、画像分類より細かい空間情報を扱うため、学習率の減衰の仕方が精度に効きやすいことが知られています。終点で学習率をきれいに 0 へ落とすことが、輪郭の精度向上につながると経験的に観察されてきました。輪郭付近の少数ピクセルの予測は学習終盤の細かな調整で決まりやすく、そこで学習率が大きいまま残っていると境界がぶれるためだと説明されます。

### 仕組みを詳しく
多項式減衰の式は次の通りです。

```
lr(t) = (lr_init - lr_end) * (1 - t / T_max) ^ power + lr_end
```

各記号の意味です。

- `lr_init`: 開始時の学習率。
- `lr_end`: 終了時の学習率(多くは 0、または 0 に近い小さな値)。
- `t`: 現在のステップ数(0 から T_max まで)。
- `T_max`: 学習の総ステップ数(終点)。
- `power`(パワー): 多項式の乗数。下り坂の形を決める。

`(1 - t / T_max)` は「残りの進捗割合」で、開始時(t=0)は 1、終了時(t=T_max)は 0 です。これを power 乗して掛けるので、終了時には必ず `lr_end` に着地します。

power の役割を数値例で見ます。`lr_init = 0.01`、`lr_end = 0`、`T_max = 100` のとき、進捗 50%(t=50、残り割合 0.5)での学習率は、

- power = 1.0(直線): 0.01 × 0.5 = 0.005(ちょうど半分。これは線形減衰と同じ)
- power = 0.9(DeepLab の poly): 0.01 × 0.5^0.9 ≈ 0.01 × 0.536 ≈ 0.00536(直線よりわずかに上、序盤を高めに保つ)
- power = 2.0(凸カーブ): 0.01 × 0.5^2 = 0.0025(直線より下、序盤で速く落ちる)

power が 1 なら直線、1 未満なら序盤に学習率を高めに保って終盤で一気に落とす形、1 より大きいと序盤で速く落として終盤を低く引きずる形になります。学習率の時間変化を ASCII 図にすると、power によって反り方が変わる下り坂です。

```
lr        power<1 は上に膨らむ(序盤高め)
0.01 |**
     |  **__   power=1(直線)
     |     \ \__
     |      \   \__
     |       \     \__   power>1 は下に膨らむ(序盤速く落ちる)
0.0  |________\_______\___
     +-----------------------> step (T_max で必ず0)
```

セグメンテーション系のライブラリには `PolyLR` 相当が最初から備わっていることが多く、`power=0.9` が既定値になっていることもよくあります。

DeepLab で使われた power=0.9 は、直線(power=1)よりほんの少しだけ序盤の学習率を高く保つ設定です。これにより、学習序盤にもう少し大胆に進めつつ、終点ではきちんと 0 に収束させる、というバランスを取っています。0.9 という値が広く踏襲されているのは、DeepLab の成功体験に由来する経験則です。

重要な関係として、power=1 の多項式減衰は線形減衰(linear decay)とまったく同じものになります。つまり線形減衰は多項式減衰の特殊ケースであり、多項式減衰は線形減衰を含むより一般的な枠組みだと理解すると整理しやすいです。この線形版(power=1)は、warmup と組み合わせて BERT 以降の言語モデル事前学習・ファインチューニングのほぼ標準スケジュール(線形 warmup → 線形 decay、いわゆる triangular)になっています。

### 手法の系譜と主要論文
多項式(poly)スケジュールは、セマンティックセグメンテーション分野で定着し、線形版が言語モデルへ広がった手法です。系譜は次の通りです。

1. Liu, Rabinovich & Berg, "ParseNet: Looking Wider to See Better", arXiv:1506.04579(2015)。セグメンテーション文脈で poly 系の学習率ポリシーを早期に用いた研究の一つです。グローバルなコンテキスト(画像全体の情報)を取り込む手法とともに、poly に近いなめらかな減衰を採用しました。

2. Chen, Papandreou, Kokkinos, Murphy & Yuille, "DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs", TPAMI 2018(arXiv:1606.00915、初出 2014-2016)。「poly」learning rate policy(power=0.9 の多項式減衰)を明確に採用し、広めた中心的論文です。動機は、それまでの Step decay より、終点に向けてなめらかに 0 へ収束させるほうがセグメンテーション精度が上がると実験で確認したことです。以後、poly はセグメンテーション分野の事実上の標準になりました。

3. Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation"(DeepLabv3, arXiv:1706.05587, 2017)、および "Encoder-Decoder with Atrous Separable Convolution"(DeepLabv3+, 2018)。poly スケジュールを引き続き採用し、atrous(拡張)畳み込みやエンコーダ・デコーダ構造の改良と組み合わせて、PASCAL VOC や Cityscapes で当時の最高精度を更新しました。

4. Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019(arXiv:1810.04805)。事前学習・ファインチューニングともに「線形 warmup → 線形 decay」(=power=1 の polynomial decay)を採用しました。これにより polynomial decay の特殊ケースである線形減衰が、自然言語処理の事実上の標準スケジュールとして定着しました。多項式減衰の系譜が、画像の密予測から言語モデルへ橋渡しされた象徴的事例です。

5. その後の多くのセグメンテーション論文(PSPNet、HRNet、SegFormer、Mask2Former など Transformer ベースを含む)が poly を踏襲し、密予測タスクでの標準的な学習率ポリシーとして定着しました。

### 論文の実験結果(定量データ)
DeepLab 系の主な検証データセットは PASCAL VOC 2012(20 クラス + 背景のセグメンテーション)と Cityscapes(都市の走行シーン、19 クラスのセグメンテーション)です。評価指標は mIoU(高いほど良い)です。

報告された代表的な数値です。

- DeepLab(初期版)は PASCAL VOC 2012 のテストで mIoU およそ 79.7% を達成しました。これは当時の最高水準で、atrous 畳み込み・CRF 後処理・poly スケジュールの組み合わせによるものです。
- poly スケジュール自体のアブレーションとして、DeepLabv3 論文では「step decay(同じ反復数)」から「poly(power=0.9)」へ変えるだけで mIoU が向上することが報告され、報告によるとおよそ 1〜2 ポイント前後の改善です。これは「学習率ポリシーだけを差し替えた」純粋な効果で、poly の有効性を示す重要な根拠です。
- DeepLabv3 は Cityscapes で mIoU およそ 81.3%、DeepLabv3+ で PASCAL VOC 2012 テスト mIoU およそ 89.0% に到達しました。これらは複数の改良の積み重ねによる数値で、poly はその学習基盤を担っています。
- 言語側では、BERT が GLUE ベンチマーク(8〜9 タスクの自然言語理解の総合指標)で当時の最高スコア(GLUE スコアおよそ 80.5、従来比で約 7 ポイントの大幅改善)を達成し、その標準レシピに線形(power=1)decay が組み込まれていました。

指標の意味として、mIoU の 1〜2 ポイントの差はセグメンテーションでは実用上大きく、物体の輪郭や小さなクラスの認識精度に直結します。注目すべきは、その改善が「モデル構造を変えず、学習率の減らし方を poly にしただけ」で得られた点です。これは学習率スケジュールが、構造変更と同等のインパクトを持ちうることを示しています。

### メリット・トレードオフ・限界
メリット

- 総ステップ数の終点でちょうど 0(または min_lr)に着地するよう設計でき、終盤に学習率が残らずきれいに収束する。
- power 1つで直線から強い反りまで下り坂の形を自由に調整できる(線形減衰を特殊ケースとして含む)。
- セグメンテーションなど密予測タスクで豊富な実績があり、線形版は言語モデルでも標準的で、信頼できる。

トレードオフと限界

- 総ステップ数 T_max を事前に決め打ちする必要があり、学習を途中で延ばしたり縮めたりしにくい(適応型と対照的)。学習を延長したいとき、コサインと同様にこの決め打ちが足かせになる。
- 単調に減るだけで、局所解・鞍点からの脱出はできない(再起動機能はない)。
- power というハイパーパラメータの感覚をつかむまで、初心者には最適値が直感的でない。
- warmup と組み合わせないと、開始直後の高い学習率で不安定になることがある。
- 未解決の論点として、なぜ power=0.9 が多くのセグメンテーションタスクで好成績なのか、また密予測でなぜ poly がコサインや step より有利になりやすいのかの理論的説明は確立しておらず、データ・モデル依存の経験則の域を出ていません。

### 発展トピック・研究の最前線
- 線形減衰との統一的理解: power=1 で線形減衰に一致するため、両者を1つの枠組みで扱えます。近年の大規模事前学習では linear(=poly power=1)warmup + linear decay や、warmup + コサインがよく使われます。
- warmup + poly の組み合わせ: 開始直後の不安定を避けるため、線形 warmup の後に poly 減衰へ入る構成が密予測タスクで使われます。
- Transformer ベースのセグメンテーション: SegFormer や Mask2Former など Transformer 系のセグメンテーションモデルでも poly あるいは linear 減衰が引き続き採用されており、poly の系譜が新世代モデルに受け継がれています。
- 言語モデルでの線形 vs コサイン: BERT の線形 decay に対し、GPT 系の事前学習はコサインを好む傾向があり、どちらが優れるかはモデル・データ・打ち切り戦略に依存します。線形は「途中で止めてもそれなりに収束している」という扱いやすさがあります。
- スケジュール横断比較: poly・コサイン・step・linear を同条件で比較する研究では、密予測では poly/linear、画像分類ではコサインがわずかに有利という傾向が観察されますが、タスク依存で決定的な結論はなく、依然として活発な研究領域です。

### さらに学ぶための関連トピック
- [Linear LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## torch.compile

### ひとことで言うと
`torch.compile` とは、PyTorch 2.0 (2023年公開) で導入された「ニューラルネットワークのモデルをコンパイル(最適化された実行コードに変換)して速くする」機能です。`model = torch.compile(model)` と1行足すだけで、PyTorch が裏で計算の流れを解析し、無駄な GPU メモリの読み書きを削り、複数の演算を1つにまとめた高速なコードを自動生成します。学習(training)も推論(inference)も、既存のコードをほとんど書き換えずに速くできるのが特徴です。ここで「コンパイル」とは、人間が書いた高水準のコードを、機械が速く実行できる低水準のコードへ事前変換しておくことを指します。

### 直感的な理解

料理にたとえます。普通の PyTorch は「レシピを1行読むたびにコンロに火をつけ、1つ作っては冷蔵庫にしまい、また取り出して次を作る」やり方です。手順がそのまま見えてデバッグはしやすいのですが、コンロの点火と冷蔵庫の出し入れに時間を食います。`torch.compile` は「レシピ全体を先に読み、まとめて作れるものは一気に作り、冷蔵庫への出し入れを最小限にする」段取りの良い料理人を雇うようなものです。出来上がる料理(計算結果)は同じですが、作業の無駄が消えるぶん速くなります。

なぜこれが必要かというと、深層学習の実際の遅さの大半は「計算そのもの」ではなく「データの移動」にあるからです。GPU の演算ユニット(行列積を担う Tensor Core など)は極めて速く、たいてい遊んでいて、メモリからデータが届くのを待っています。この待ち時間を減らすことが高速化の鍵で、そのためには複数の演算をまとめて見渡して最適化する必要があります。それを自動でやるのがコンパイルです。眼目は「人間がカーネルを手書きせずに、ふだん書いている素朴な PyTorch コードのまま」その最適化を享受できる点にあります。

### 基礎: 前提となる概念

#### eager mode (即時実行モード)
PyTorch はもともと eager mode で動きます。これは Python のコードを1行ずつ、書いた順に GPU で実行する方式です。たとえば `y = a + b; z = relu(y)` と書くと、まず足し算のカーネル(kernel、GPU 上で動く小さなプログラム)を起動して結果を GPU メモリに書き込み、次に relu のカーネルを起動して同じデータを読み直します。直感的で対話的なデバッグに向きます(`print` を差し込んで途中値を見られる、Python の任意の制御フローが書ける)が、速度面では無駄が出ます。TensorFlow 1.x のような「先にグラフを定義してから実行する」define-and-run 方式に対し、PyTorch のこの define-by-run 方式が研究者に愛された理由でもあります。`torch.compile` の設計目標は、この愛された体験を一切壊さずに速さだけ足すことでした。

#### eager mode の3つの無駄
1. カーネル起動のオーバーヘッド: 演算1つごとに CPU から GPU へ「これを計算せよ」と命令を発行する手間(起動コスト、数マイクロ秒程度)がかかります。小さな演算が大量にあると、計算より起動の積み重ねが支配的になります。とくにバッチサイズが小さい推論や、層が薄く演算が細かいモデルで顕著です。
2. メモリの往復が多い: 上の例では足し算の結果をいったん GPU メモリ(後述の HBM)に書き、relu でまた読み出します。「足してすぐ relu」できれば書き戻しは不要です。
3. Python インタプリタのオーバーヘッド: 毎回 Python が1行ずつ解釈するため、演算の合間に Python 自身の処理時間が挟まります。GPU が速い世代になるほど、この CPU 側の足踏み(「CPU-bound になる」と言う)が相対的に目立ちます。

#### GPU の2階層メモリ
GPU には大容量だが相対的に遅い HBM (High Bandwidth Memory、いわゆる VRAM、数十GB) と、計算ユニットのすぐ隣にあって爆速だが極小の SRAM/レジスタ(数十〜数百KB)があります。中間結果を HBM に書いて読み戻す回数(メモリ往復)が多いほど、GPU は待ち時間が増えて遅くなります。これを「メモリ律速(memory-bound)」と呼びます。多くの深層学習演算、特に要素ごとの演算(加算、活性化関数、正規化、ドロップアウト)はメモリ律速です。逆に大きな行列積(GEMM)は計算律速(compute-bound)で、ここはコンパイルで融合しても伸びしろが小さい。この「どこが律速か」の見極めが高速化の核心です。

#### カーネル融合 (kernel fusion)
複数の演算を1つの GPU カーネルにまとめることを融合と呼びます。融合すると、中間結果を HBM に書かずレジスタ上で次の演算に渡せるため、カーネル起動回数とメモリ往復が同時に減ります。`torch.compile` の高速化の主因はこの融合です。たとえば LayerNorm → GeLU → Dropout という3つの要素ごと演算は、それぞれ HBM を読み書きするのではなく、1つのカーネルがデータを SRAM に持ったまま3工程を済ませて1度だけ書き戻せます。

### 仕組みを詳しく

#### 使い方は1行
```python
model = MyModel()
model = torch.compile(model)        # ← これだけ

for batch in loader:
    loss = model(batch)             # 最初の数回はコンパイルが走り遅い
    loss.backward()                 # 2回目以降は最適化済みで高速
    optimizer.step()
```

#### 内部の3段構え
`torch.compile` は大きく3つの部品で動きます。

1. TorchDynamo (グラフキャプチャ): Python のバイトコード(CPython が実行直前に使う中間命令列)をフレーム評価フック(PEP 523 の `_PyEval_EvalFrameEx` 差し替え)で横取りし、PyTorch の演算列を「計算グラフ(どの演算がどの順でどのテンソルに作用するかの設計図)」として安全に取り出します。Python の複雑な if 文やループ、未対応の外部呼び出し、テンソル以外への分岐(`.item()` で値を取り出して条件分岐する等)が出てくると、そこでグラフを区切ります。これを graph break (グラフの分断)と呼びます。区切られた部分は普通の eager で実行し、最適化できる連続部分だけをグラフ化します。これにより「どんな書き方でも壊れずに、できる範囲で最適化する」soundness(健全性)優先の設計を実現しています。これは旧 TorchScript が「対応していない構文はエラーにして書き換えを強いた」のと正反対です。

2. AOTAutograd (順逆両方をグラフ化): 順伝播(forward)だけでなく、勾配計算である逆伝播(backward)のグラフも事前に(ahead-of-time)取り出します。具体的には forward グラフを微分して backward グラフを生成し、両方をまとめて最適化対象にします。これにより学習ループ全体が最適化対象になります。従来の推論専用コンパイラ(ONNX Runtime、TensorRT など)と違い、学習を速くできるのが PyTorch 2 の重要点です。

3. TorchInductor (コード生成バックエンド): 取り出したグラフを実際の高速カーネルに変換します。GPU 向けには OpenAI 発の Triton という言語でカーネルを生成し、連続する要素ごと演算を融合します。CPU 向けには C++/OpenMP コードを生成します。Triton は GPU カーネルを Python ライクに書ける言語で、タイル(小ブロック)単位の並列処理を抽象化してくれます。Inductor はスケジューラがどの演算を融合すべきかを決め、メモリ計画を立て、Triton コードを吐く、というパイプラインを踏みます。

#### 融合の具体例
```
コンパイル前 (eager):
  tmp = a + b        # カーネル起動1、tmp を HBM に書き込み
  out = relu(tmp)    # カーネル起動2、tmp を HBM から読み込み + out を書き込み

コンパイル後 (fusion):
  out = fused_add_relu(a, b)   # カーネル起動1回、tmp はレジスタ上で消費、HBM 往復なし
```
カーネル起動が2回→1回、HBM 書き込みが2回→1回に減ります。要素ごと演算が長く連鎖する箇所(例: 正規化 → 活性化 → ドロップアウト)ほど効果が大きくなります。

#### guard と再コンパイル
TorchDynamo はキャプチャしたグラフに guard (前提条件のチェック)を付けます。たとえば「入力のテンソル形状は [8, 3, 224, 224]」「この Python フラグは True」「この属性は None ではない」といった条件です。次回の実行時に guard が満たされればキャッシュ済みの最適化コードを再利用し、満たされなければ再コンパイルします。guard は「このグラフはこの前提のもとでだけ正しい」という安全装置で、健全性の根拠です。

#### 最初は遅い (コンパイルコストの先払い)
`torch.compile` したモデルを最初に実行するとき、その場でコンパイルが走るため数秒〜数十秒(`max-autotune` では数分)遅くなります。これは入力の形ごとに1回だけで、2回目以降はキャッシュされた高速コードが使われます。したがって「学習を長時間回す」「同じ形の入力で何度も推論する」用途で効果が出ます。1回しか実行しないなら、コンパイル時間のほうが損になります。コンパイル結果はプロセス内キャッシュに加え、永続キャッシュ(`TORCHINDUCTOR_CACHE_DIR`)にも保存でき、起動のたびのコンパイルを避けられます。

#### mode オプション
```python
model = torch.compile(model, mode="max-autotune")
```
`mode` で最適化の強さを選べます。`default` は速くコンパイルしてそこそこ最適化、`reduce-overhead` は CUDA Graphs を併用してカーネル起動オーバーヘッドをさらに削減(同じ形の小さなカーネルを大量に起動する場面で効く)、`max-autotune` は行列積や畳み込みについて複数のカーネル実装(タイルサイズ違いなど)を実際に計測して最速を選びます(コンパイルは遅いが実行は最速)。この autotuning は TVM の探索的最適化を継いだ思想です。

#### 動的な形 (dynamic shapes) への注意
入力の形(tensor shape)が毎回変わると、形ごとに再コンパイルが走り、かえって遅くなります(再コンパイルの嵐)。バッチサイズや系列長が固定なら問題ありませんが、可変長系列などでは `torch.compile(model, dynamic=True)` を指定して、形をシンボリック(記号的、`s0` のような記号で持つ)に扱うグラフを1つだけ作らせる工夫が要ります。動的形対応はグラフの汎用性と最適化度合いのトレードオフになります。シンボリックにすると、コンパイラは「N が 1 なのか 1000 なのか」を仮定できないため、形に依存した特殊化(小バッチ専用の最適化など)を諦めます。

### 手法の系譜と主要論文

グラフコンパイルによる融合という発想は古く、複数の系譜があります。

- TVM (Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", OSDI 2018): 機械学習モデルを多様なハードウェア(CPU/GPU/FPGA/組み込み)向けに自動最適化するコンパイラ。探索(auto-tuning、後に学習ベースの AutoTVM)で各演算の最速実装を見つける考え方は、後の `max-autotune` に通じます。
- XLA (Google, 2017年頃〜): TensorFlow / JAX のバックエンドで、計算グラフを融合・最適化します。静的グラフ前提で高い最適化を得る一方、Python の柔軟さを犠牲にする傾向がありました。JAX の `jit` はこの上に乗っています。
- Triton (Tillet, Kung, Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", MAPL 2019): GPU カーネルをタイル単位の高水準言語で書ける仕組み。低レベルの CUDA を書かずにブロック並列を表現でき、コンパイラがレジスタ割当・共有メモリ管理を担います。TorchInductor の生成バックエンドとして採用され、`torch.compile` の高速性を支えています。
- TorchScript (`torch.jit`, 2019年頃): PyTorch 初期のグラフ化機構。Python の自由な書き方に対応しきれず、ユーザーにコード書き換え(型注釈、サポート構文への限定)を強いたため広く普及しませんでした。この反省が `torch.compile` の「ユーザーに書き換えを強いない」設計動機になりました。
- PyTorch 2 (Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation", ASPLOS 2024): `torch.compile` を解説した公式論文。Python バイトコードを動的に書き換えて計算グラフを安全に取り出し(TorchDynamo)、それを融合カーネルにコンパイルする(TorchInductor)設計を提案。「eager の柔軟さと開発体験を一切損なわず、1行追加で最適化する」ことを最大の目標に置いた点が、先行する静的グラフ系コンパイラとの決定的な差です。

### 論文の実験結果 (定量データ)

PyTorch 2 論文 (ASPLOS 2024) は、HuggingFace(言語モデル中心)・TIMM(画像分類モデル中心)・TorchBench(雑多な実モデル群)という3つのベンチマーク群、計180以上のモデルで評価しています。指標は「eager 実行に対する高速化率(speedup、1.0 が同等、2.0 なら2倍速)」で、幾何平均(geometric mean、比率の平均に適した平均)で集計しています。

- NVIDIA A100 GPU で、float32 推論ではおよそ 1.5〜1.7倍、AMP(自動混合精度、bf16/fp16)では幾何平均でおよそ 1.3〜2.3倍程度の高速化が報告されています。学習でも全ベンチマーク横断で正味の高速化が示されています。
- 報告では検証した180以上のモデルのうち圧倒的多数(論文では約93%)が「正しく動作し、かつ高速化した」とされ、デコレータ1つでの広い適用可能性が示されました。残りはグラフブレイク多発や未対応演算で伸びなかったケースです。
- 高速化率はモデル依存が大きく、要素ごと演算が多くメモリ律速なモデルほど融合の恩恵が大きい一方、すでに大きな行列積が支配的なモデル(融合余地が少ない)では伸びが小さくなります。

数値の意味: speedup 1.3倍とは、同じ学習を100時間かけていたものが約77時間で終わる計算で、長期の学習や大量推論ではコスト削減効果が大きいことを意味します。一方、ベンチマーク全体でばらつきが大きいことは「効くかどうかはモデルとハードに依存し、必ず実測して確認すべき」という実務上の教訓を裏付けます。

アブレーション的な観察として、(1) graph break が多いモデル(複雑な Python 制御フローを含む)ほどグラフが細切れになり融合機会を失って高速化率が下がること、(2) `reduce-overhead`(CUDA Graphs)を有効にするとカーネル起動が多い小規模モデルで追加の改善が出ること、(3) `max-autotune` は行列積支配のモデルで効く一方コンパイル時間が大きく増えること、が示されています。バックエンドを Inductor から他のもの(`aot_eager` などデバッグ用)に切り替える比較で、速度向上の大半が Inductor の融合・コード生成由来であることも確認できます。

### メリット・トレードオフ・限界

メリット
- 1行追加するだけで適用でき、モデルコードの書き換えがほぼ不要(eager の開発体験を保つ)。
- カーネル融合とメモリ往復削減により、学習・推論の双方を高速化(多くのモデルで 1.3〜2倍程度)。
- 逆伝播もグラフ化するため、推論専用コンパイラと違って学習そのものを速くできる。
- 新しい GPU(Ada/Hopper)と低精度(bf16/fp16)で効果が出やすい。Tensor Core を埋めるために CPU 側の足踏みを消す価値が高い世代だから。

トレードオフ・限界
- 初回コンパイルに時間がかかる(数秒〜数分)。短時間しか動かさない用途では損。
- graph break が多いコードでは最適化の取りこぼしが出て期待ほど速くならない。`torch._dynamo.explain(fn)(input)` で break 箇所と理由を調べ、原因の Python パターンを書き換える必要がある。
- 入力の形が頻繁に変わると再コンパイルが走り、かえって遅くなる(dynamic shapes 対策が必要)。`torch._dynamo.config.cache_size_limit` を超えると再コンパイルを諦めて eager に落ちることもある。
- 最適化後のコードはエラーや数値の追跡が難しくなる。コンパイルされた関数内のスタックトレースは読みにくい。
- すでに FlashAttention のような専用最適化カーネルを使う部分では、compile による追加の伸びは限定的になることがある(SDPA は Inductor が温存して呼ぶだけ)。
- カスタムの C++/CUDA 拡張や副作用(グローバル状態の書き換え)を含むコードは graph break の原因になりやすい。
- 研究上の未解決課題として、動的形と最大最適化の両立、コンパイル時間の短縮、分散学習(FSDP など)との安定した併用、コンパイル結果の可搬な配布が継続的に改善されています。

### 発展トピック・研究の最前線
- CUDA Graphs との併用(`mode="reduce-overhead"`)によるカーネル起動オーバーヘッドの一掃。多数の小カーネルを1度の起動でまとめて再生する。
- 分散並列(DDP/FSDP)下での compile の安定運用と、通信(all-reduce/all-gather)と計算のオーバーラップ最適化。
- 大規模言語モデル推論における compile と KV キャッシュ・量子化(int8/fp8)の連携。
- AOTInductor による「コンパイル済みモデルを Python ランタイムなしで配布する」推論デプロイ(共有ライブラリ化)。
- カスタムカーネルを Triton で書いて compile グラフに統合する流れ(手書き最適化とコンパイラ最適化の融合)。
- region compile / `torch.compiler.set_stance` など、モデルの一部だけをコンパイルしたりコンパイル方針を実行時に切り替える細粒度制御。

### さらに学ぶための関連トピック
- [FlashAttention](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Gradient Accumulation](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Transformer 時間方向アテンション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/05_multicamera-fusion)


## RAdam (Rectified Adam)

### ひとことで言うと
RAdam(ラダム、Rectified Adam)は Adam の「学習を始めたばかりのとき更新が不安定で暴れやすい」という弱点を直したオプティマイザです。Adam を安定させるために経験的によく使う warmup(学習率を最初は小さく徐々に上げる手動の準備運動)を、数式で自動的に肩代わりします。これにより warmup を手で設定しなくても学習初期から安定します。「Rectified(整流された)」という名前は、初期の不安定さを補正する整流項に由来します。

### 直感的な理解
新しい仕事を始めた人を想像してください。最初の数日はまだ状況がよく分からないので、慎重に少しずつ動くべきです。いきなり全力で動くと、間違った判断で大きく踏み外してしまう。仕事に慣れてくれば全力を出してよい。

Adam は各パラメータの学習率を「勾配二乗の移動平均 v」で自動調整しますが、学習の最初は観測した勾配がまだ数個しかなく、この v の推定が非常に不安定です。少ないサンプルから「ばらつき」を見積もると値が大きくブレるのと同じで、たまたま小さい v が出ると `1/sqrt(v)` が巨大になり、ある特定のパラメータだけ実効学習率が跳ね上がって暴れます。RAdam は「まだサンプルが足りない初期は v を信用せず慎重に動き、十分たまったら全力を出す」という調整を自動で行います。

### 基礎: 前提となる概念

前提を噛み砕きます。

- Adam: モメンタム([Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照)と勾配二乗の移動平均([Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照)を組み合わせた定番オプティマイザ。各パラメータの学習率を勾配二乗の移動平均 v の平方根で割って自動調整します。`θ ← θ - η · m_hat / (sqrt(v_hat) + ε)`。
- 分散(variance): ばらつきの大きさ。同じ統計量を何度も推定したとき、結果がどれだけ揺れるかを表します。少数のサンプルから統計量を推定すると分散が大きくなります(推定が不安定)。例えばサイコロを3回振った平均と300回振った平均では、後者のほうが真の値に近く安定します。
- warmup: 学習率を最初の数百〜数千ステップだけわざと小さくし(0付近から線形に立ち上げるのが典型)、その後ゆっくり目標値まで上げる手法。Transformer の学習(Vaswani et al., NeurIPS 2017)などで「warmup を入れないと発散する」ことが経験的に知られていました。ただし「何ステップかけてどこまで上げるか」を手で決める必要があり、適切な値はタスクごとに違いました。
- 自由度(degrees of freedom): 推定に使える実質的なサンプル数の指標。多いほど推定が安定します。RAdam は移動平均 v の「実質サンプル数」を自由度として定量化します。

問題の核心: Adam は学習率を `1/sqrt(v)` で割って決めますが、学習開始直後は v を計算するための勾配サンプルが数個しかないため v の推定が極端に不安定です。Liu らはこの「適応学習率の分散が学習初期に発散的に大きくなる(理論上は無限大に発散しうる)」ことを理論的に示しました。warmup は「最初は学習率を絞れば、v が安定するまで暴れない」という対症療法でしたが、なぜ効くのかは経験則のままでした。RAdam はその理由を分散の観点で説明し、効果を数式で自動再現します。

### 仕組みを詳しく

RAdam の核心は「学習初期は適応学習率(v で割る部分)を信用しすぎず、サンプルが十分たまるまで適応を弱める」ことです。各更新で次を行います。

```
1. Adam と同じく m(モメンタム)と v(勾配二乗の移動平均)を計算。
2. ρ_∞ = 2/(1-β2) - 1                       # 適応学習率の自由度の最大値(定数)
   ρ_t = ρ_∞ - 2t·β2^t/(1-β2^t)             # 現ステップの「有効な自由度」
3. もし ρ_t > 4(=分散が信用できる):
     r_t = sqrt( (ρ_t-4)(ρ_t-2)ρ_∞ / ((ρ_∞-4)(ρ_∞-2)ρ_t) )   # 整流項
     θ ← θ - η · r_t · m_hat / (sqrt(v_hat) + ε)               # 補正つき Adam
   そうでなければ(初期、分散が信用できない):
     θ ← θ - η · m_hat                                          # モメンタム SGD として動く
```

順に説明します。

- ρ_∞ は β2 から決まる定数で、適応学習率がどれだけの「自由度(=実質的なサンプル数)」を持ちうるかの最大値です。β2=0.999 なら ρ_∞ ≈ 1999、β2=0.99 なら ρ_∞ ≈ 199。
- ρ_t は現ステップ t での有効な自由度。学習が進むほど ρ_∞ に近づきます。これが小さい(およそ4以下)うちは v の推定サンプルが足りず信用できない、という判定になります。β2=0.999 では ρ_t がしきい値 4 を超えるまでにおよそ最初の4〜5ステップを要し、その後 r_t が 1 に近づくまで数千ステップかけて立ち上がります。
- r_t は「適応学習率の分散を一定に保つための補正係数(rectification term、整流項)」です。ρ_t がまだ小さい初期では r_t も小さく、適応の効きを抑えます。十分たまると `r_t → 1` に近づき、ふつうの Adam に一致します。この r_t は、Student の t 分布の分散補正を Adam の二次モーメント推定に当てはめた形になっており、「サンプルが少ないときの推定分散を一定に保つ」という統計的に明確な根拠を持ちます。
- ρ_t がしきい値(4)を下回るごく初期は、v による割り算を完全にやめ、モメンタム SGD のように分散の小さい安全な更新をします。これが warmup の代わりになります。

直感的な効果を数値でイメージします。

```
学習開始〜数ステップ: ρ_t < 4 → 適応オフ → モメンタム SGD のように更新(暴れない)
それ以降:            ρ_t > 4 → 整流項 r_t で分散を均しながら Adam として動く
            t→大:    r_t → 1 → 通常の Adam に一致
```

結果として、学習率の立ち上がりが「自動的な warmup カーブ」のような形になります。論文では RAdam の実効学習率の立ち上がりが手で設定した warmup と似た上に凸の立ち上がり形になることが示されています。

tensor shape とメモリ: 状態は Adam と同じ m と v(パラメータと同サイズを2つ)だけです。ρ_t と r_t はステップごとのスカラー(全パラメータ共通の1個の数)なので追加メモリはほぼゼロ。Adam からの置き換えコストが小さいのが実用上の利点です。

### 手法の系譜と主要論文

- Adam(Kingma & Ba, ICLR 2015): 適応的学習率の決定版。だが学習初期が不安定で、Transformer などでは warmup が事実上必須だった。
- warmup の経験的確立(Vaswani et al., NeurIPS 2017 など): Transformer 学習で warmup なしだと発散することが知られたが、理論的説明はなかった。学習率を `step^{-0.5}` と `step·warmup^{-1.5}` の小さい方にする独特のスケジュールが提案された。
- RAdam(Liu et al., ICLR 2020): warmup がなぜ効くのかを「初期の適応学習率の分散過大」として説明し、整流項で自動補正。
- Lookahead(Zhang et al., NeurIPS 2019)との組み合わせ: RAdam を内部オプティマイザにし Lookahead で外側から平滑化した Ranger オプティマイザが、画像認識のコンペなどで人気を集めました。Lookahead は「速い内側オプティマイザの軌跡を、遅い外側の重みが追いかけて平均化する」二重ループの仕組みです。
- Ma & Yarats(AAAI 2021, "On the adequacy of untuned warmup for adaptive optimization"): RAdam の整流項が、実は単純な線形 warmup でほぼ近似でき、凝った整流項がなくても十分という反論的研究。
- 関連する別軸の改良: 重み減衰を分離した AdamW、Nesterov 化した [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、大バッチ向けの [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。RAdam は「学習初期の安定化」という独立した軸の改良で、これらと組み合わせ可能です。

### 論文の実験結果(定量データ)

Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond" (ICLR 2020) の主な実験。

- 画像分類(CIFAR-10、ImageNet): warmup なしでも安定して学習でき、warmup ありの Adam と同等の精度を達成。とくに「初期学習率を多少間違えても壊れにくい(頑健)」ことを示しました。論文の代表的な実験では、学習率を `0.1` から `0.0001` の桁違いに変えても、Adam が発散・性能劣化するレンジで RAdam は最終精度がほぼ一定に保たれる、という頑健性が示されています。これが実務上の大きな利点です。
- ニューラル機械翻訳(IWSLT'14 De-En など): BLEU(翻訳の質を測る指標、機械翻訳が正解訳とどれだけ語の並びが一致するかを測る。0〜100 で高いほど良い)で、warmup あり Adam と同等の結果を warmup の手調整なしで達成。Transformer 系で warmup を省ける点が特に評価されました。
- 言語モデル: パープレキシティ(低いほど良い)でも warmup あり Adam に匹敵。

理論的貢献の核心: 論文は「学習初期に適応学習率の分散が無限大に発散しうる」ことを解析し、warmup がこの分散を抑える役割を果たしていたと説明しました。この「warmup の数学的正体は分散低減」という洞察自体が論文の最大の貢献です。経験則だった warmup に初めて定量的な根拠を与えました。

正直な限界(論文・後続研究): よく調整された warmup ありの Adam を最終精度で大きく上回るわけではないことが多く、RAdam の利点は主に「warmup スケジュールの手調整が要らなくなる」ことです。また後続研究(Ma & Yarats, AAAI 2021)では、RAdam の整流項が実は単純な線形 warmup でほぼ近似でき、「凝った整流項より、適当な長さの線形 warmup で十分」という指摘もあります。RAdam が決定版というわけではありません。

アブレーション的な観点: RAdam から整流項 r_t を外し ρ_t によるオン/オフ切り替えもなくせば、ただの Adam に戻り、初期不安定さが復活します。整流項のしきい値付近で挙動が切り替わるため、ごく初期の数ステップは実質モメンタム SGD として動く点も RAdam の本質的な振る舞いです。β2 を小さくする(例 0.99)と ρ_∞ も小さくなり、適応がオンになるまでの初期段階が短くなる、という形でハイパーパラメータが立ち上がり形状に直結します。

### メリット・トレードオフ・限界

メリット

- 手動 warmup を不要にできる(初期の不安定さを数式で自動的に抑える)。
- 初期学習率の選び方に頑健で、多少間違えても壊れにくい。学習率探索の手間が減る。
- Adam からの置き換えコストが小さい(追加メモリほぼゼロ、ステップ計算もほぼ同じ)。
- 整流項の根拠が統計的に明快(t 分布の分散補正)で、理論的に理解しやすい。

トレードオフ・限界

- よく調整された warmup ありの Adam と最終精度が大きく変わらないことが多い。
- 整流項は単純な線形 warmup でほぼ近似できるという指摘があり、必ずしも凝った仕組みが必要とは限らない。
- 単体より Lookahead 等との組み合わせ(Ranger)のほうが良いという報告もあり、決定版ではない。
- 整流項のしきい値付近で挙動が切り替わるため、ごく初期の数ステップは適応の恩恵が得られない。
- 重み減衰の扱い(AdamW が解決した問題)はそのままで、別途 decoupled weight decay を組み合わせる必要がある。

### 発展トピック・研究の最前線

- warmup の理論: RAdam が口火を切った「warmup の数学的説明」は今も研究が続き、勾配ノイズや loss 地形の鋭さ(sharpness、ヘシアンの最大固有値で測る曲がりの急さ)との関係から warmup を説明する研究が進んでいます。warmup が学習初期に sharpness を抑える役割を持つという解析もあります。
- Transformer 学習の安定化: 大規模言語モデルの事前学習では、RAdam そのものより AdamW + 線形/コサイン warmup + 勾配クリッピングの組み合わせが標準ですが、その理論的根拠の一部は RAdam の分散解析にあります。
- Ranger 系: RAdam + Lookahead + 勾配中心化(gradient centralization)などを束ねたオプティマイザが、画像系のコンペティションで実用的に使われ続けています。
- 学習率スケジュールフリーへの流れ: warmup を不要にする RAdam の問題意識は、近年の Schedule-Free 最適化(Defazio et al., NeurIPS 2024)など「スケジュール調整を丸ごと省く」研究へと続いています。RAdam は「スケジュールを理論で自動化する」初期の代表例です。

### さらに学ぶための関連トピック

- [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## RandAugment

### ひとことで言うと
あらかじめ用意した「画像の加工メニュー」(回転、色変え、コントラスト調整など十数種類)から、毎回ランダムに数個を選んで適用する自動データ拡張(学習データを水増しして学習を賢くする工夫)です。設定すべきパラメータがたった2つ——何個適用するか N と、どれくらい強く加工するか M——しかないのに、巨大な探索を行う先行手法(AutoAugment)と同等以上の精度が出る、という実用性の高さが最大の特徴です。複雑な最適化を「2つのつまみ」に置き換えた、データ拡張研究の重要な転換点です。

### 直感的な理解
データ拡張で精度を上げたいとき、「どの加工を、どんな強さで、どんな確率で、どう組み合わせるか」という選択肢は天文学的に多くあります。先行研究の AutoAugment は、この巨大な選択肢を強化学習(試行錯誤しながら良い選択を学ぶ枠組み)で総当たりに近い形で探索し、最良の組み合わせ(これをポリシー=拡張の設計図と呼びます)を見つけました。しかしそれには数千 GPU 時間という法外な計算が必要でした。GPU 時間とは「GPU を 1 台 1 時間動かす」コストの単位で、数千あれば高性能 GPU を何日も占有することを意味します。

RandAugment の問いはこうです。「そもそも、そんなに精密に最適化する必要があるのか? 多様な加工が適度にかかりさえすれば十分なのでは?」。例えるなら、料理に最適なスパイスの配合を何万回も実験して厳密に決めるのではなく、「数種類のスパイスから毎回ランダムに2つ振りかけ、全体の効き具合だけを1つのつまみで調整する」やり方です。配合の細部は気にせず、多様性と総量だけ管理する。これで十分おいしくなる、というのが RandAugment の発見でした。探索という重い工程を丸ごと捨て、人が触るつまみを2つに減らしたのです。

### 基礎: 前提となる概念
- データ拡張(data augmentation): 学習画像に回転・色変換などの加工を施して人工的に種類を増やし、過学習(学習データだけ覚えて未知データで失敗する状態)を防ぐ技術。モデルに「同じ物体でも見え方は様々だ」と教える効果があります。
- ポリシー(policy): 「どの変換を、どの確率で、どの強度で適用するか」の設計図。AutoAugment が探索したのはこのポリシーでした。
- 探索空間(search space): 試しうるポリシーの全集合。広いほど良い解がある可能性は上がるが、探すコストも跳ね上がる。
- 代理タスク(proxy task): 探索を安く済ませるための「縮小版」設定。小さいモデルを小さいデータの一部で学習させて評価します。本番より軽いので何度も試せる一方、代理で最適だった設定が本番の大規模設定でも最適とは限らない、という問題があります。これを「proxy gap(代理と本番のズレ)」と呼びます。
- ハイパーパラメータ: 学習を始める前に人が決める設定値。N や M がこれにあたります。
- グリッドサーチ(grid search): 候補となる設定値を格子状に並べ、全組み合わせを実際に試して最良を選ぶ素朴な探索。候補が少なければ現実的に実行できます。

RandAugment が解いたのは、AutoAugment が抱えた2つの実務的な痛点です。(1) 探索が高価すぎる。(2) 代理タスクで見つけたポリシーが本番に移りにくい(proxy gap)。この2つを同時に解消したのが新規性です。

### 仕組みを詳しく
RandAugment はまず、共通の「変換の集まり」(典型的には 14 種類前後)を用意します。

- 幾何変換: Rotate(回転)、ShearX/Y(せん断: 画像を斜めに歪ませる)、TranslateX/Y(平行移動)
- 色・明るさ系: Color(彩度・色味)、Contrast(コントラスト=明暗の差)、Brightness(明るさ)、Sharpness(鮮鋭化)
- 画素値の操作: Posterize(色階調を粗くする)、Solarize(一定以上の明るさを反転)、AutoContrast(自動コントラスト)、Equalize(ヒストグラム均等化=明暗分布を平らに整える)、Invert(明暗反転)、Identity(何もしない)

学習時、各画像に対して:
1. この集合から N 個の変換を一様ランダム(どれも等確率)に選ぶ。
2. 選んだ N 個を順番に適用する。
3. 各変換の強度は全変換で共通の値 M で決める。M は整数スケール(典型的に 0〜30、あるいは 0〜10)で、各変換は「M を実際の物理量に変換する対応表」を持つ。例えば Rotate なら M が大きいほど回転角度が大きく、Solarize なら M が大きいほど反転しきい値が変わる。

つまり調整するのは N(個数、例 2)と M(強度、例 9)の2つだけ。AutoAugment が「各変換ごとに確率と強度を別々に持つ」巨大な探索空間(報告で約 10^32 通り規模)を持っていたのに対し、RandAugment は探索空間を (N, M) の2次元グリッドにまで縮めました。この程度の小ささなら、代理タスクではなく本番のデータ・モデルで直接グリッドサーチしても安く済みます。これが proxy gap を根本から回避する設計の核心です。「探索を縮めた」のではなく「探索を本番でやれるほど縮めた」のがポイントです。

#### 数値イメージ
入力 `[3, 224, 224]`(RGB、224×224)で N=2、M=9 のとき:
- 1回目: ランダムに Rotate が選ばれ、M=9 相当の角度(例: 約 20 度)で回転。
- 2回目: ランダムに Contrast が選ばれ、M=9 相当の強さでコントラストを上げる。

出力の形は `[3, 224, 224]` のまま(回転で生じる外周の余白は反射パディングや定数埋めで補う)。次の画像では選ばれる2つが TranslateX と Posterize になるかもしれず、毎回違う組み合わせが現れます。確率という概念を持たず「選ばれたら必ず適用」とすることで、AutoAugment の確率パラメータを丸ごと省いています。N=2 なら 14 種類から 2 つ選ぶ組み合わせは多数あり、データセット全体では十分に多彩な加工が現れます。

#### なぜ「ランダムでいい」のか
論文の主張は、(a) 全変換を等確率で選び、(b) 強度を1つの M で共通化する、という乱暴な単純化でも、過学習を抑えるのに十分な多様性が生まれる、というものです。重要なのは「適度な多様性と適度な強度」であり、変換ごとの細かい確率・強度の最適化が生む追加利益は、その探索コストに見合うほど大きくなかった、というわけです。直感的には、ポリシー空間の中に「良い解」は1点ではなく広い領域として存在しており、ランダムにその領域内をうろつくだけで十分良い、という解釈ができます。

### 手法の系譜と主要論文
- AutoAugment(Cubuk, Zoph, Mané, Vasudevan, Le, CVPR 2019): 強化学習でポリシーを探索。データ拡張を最適化問題として定式化した起点。探索が数千 GPU 時間と高価で proxy gap も抱えた([AutoAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。
- Fast AutoAugment(Lim et al., NeurIPS 2019): 密度マッチング(拡張前後でデータ分布が近くなる方向を探す)を使い、子モデルを毎回再学習せずに探索を高速化。コストを大幅削減(数千 GPU 時間を数 GPU 時間規模へ)。
- Faster AutoAugment(Hataya et al., ECCV 2020): 拡張処理を微分可能(勾配で最適化できる形)にし、勾配法で直接ポリシーを最適化してさらに高速化。
- RandAugment(Cubuk, Zoph, Shlens, Le, NeurIPS 2020): 探索そのものを捨て、(N, M) の2パラメータに単純化。AutoAugment と同じ Google Brain チームによる「自己批判」的な発展で、自分たちの先行研究が過剰だったと示した点が珍しい。
- TrivialAugment(Müller & Hutter, ICCV 2021): N すら 1 に固定し、強度もランダムにしてパラメータを実質ゼロに。RandAugment をさらに単純化しても匹敵する性能を示した([TrivialAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。

この系譜は「探索を極める」→「探索を速める」→「探索を縮める」→「探索を捨てる」という、複雑さを削ぎ落としていく流れとして読めます。RandAugment はその「縮める」段階の決定打です。なお RandAugment は、Vision Transformer の代表的な学習レシピ DeiT(Touvron et al., ICML 2021)でも標準採用され、Transformer 系でも有効であることが実証されています。

### 論文の実験結果(定量データ)
指標の意味: ImageNet 分類の主指標は Top-1 精度(最も確信した予測が正解だった画像の割合、高いほど良い)。ImageNet は 1000 クラス・約 128 万枚の大規模データセットで、ここでの 0.5 ポイントの改善は十分有意とされます(多くの新手法がこの程度の差を競っています)。物体検出は COCO データセットの mAP(各クラスの平均検出精度、高いほど良い)で測ります。

RandAugment(Cubuk et al. 2020)の報告:
- ImageNet で ResNet-50 の Top-1 精度がベースライン約 76.3% から約 77.6% へ(報告による、約 1.3 ポイント改善)。AutoAugment が同規模で得た改善と同等以上を、探索コストほぼゼロで達成。
- EfficientNet-B7 で約 84.4% から約 85.0% への押し上げを報告し、大規模・高精度モデルでも有効。
- 重要なアブレーション(要素を変えて影響を見る実験): モデルやデータが大きいほど最適な M が大きくなる、という明確な傾向を発見。すなわち「大きいモデルにはより強い拡張が要る」。これは AutoAugment の固定ポリシーでは捉えにくかった知見で、M を1つ上げ下げするだけでモデル規模に追従できる RandAugment の運用上の利点を示します。小さいモデルに強すぎる拡張をかけると逆に精度が落ちる(キャパシティが足りず加工後の難しい画像を学びきれない)ことも観察されました。
- N の感度は比較的低く、N=1〜3 程度で多くの設定が良好。M のほうが効きが大きく、実務では M のチューニングが中心になります。
- COCO 物体検出でも有効性を確認し、分類以外にも転用できることを示した。
- 探索コスト比較: AutoAugment が数千 GPU 時間規模の事前探索を要したのに対し、RandAugment は本番設定での小規模グリッド(N×M の数十通り)を試すだけ。論文は事実上、探索コストを「ゼロに近い」と表現しています。

これにより、「高価な探索をしなくても、よく設計された変換集合に対して多様性と強度を管理するだけで最先端に届く」ことが実証されました。後続の TrivialAugment による公平比較でも、RandAugment は AutoAugment と統計的に区別できない精度を出すことが追認されています。

### メリット・トレードオフ・限界
メリット
- 調整パラメータが N と M の2つだけで非常に扱いやすい。
- 探索コストがほぼゼロ。AutoAugment の数千 GPU 時間が不要。
- 本番のデータ・モデルで直接チューニングするため proxy gap が起きにくい。
- モデル/データを大きくしたとき M を上げるだけでスケールでき、運用が単純。

トレードオフと限界
- 完全自動ではなく、N・M のグリッドサーチは依然必要(ただし極めて軽量)。本番設定でこのサーチを回すコストは、巨大データ・巨大モデルでは無視できなくなる場合がある。
- 全変換を等確率に扱うため、タスクに有害な変換が混ざるリスクがある。例えば文字認識(数字の 6 と 9 など向きが意味を持つ)で大きな回転や上下反転を含めると意味が壊れる。どの 14 種類を変換集合に入れるかは人が決める必要がある。
- 強度 M を上げすぎると画像が破壊され、かえって精度が落ちる(過剰拡張)。最適 M はモデル規模に依存し、固定の万能値はない。
- 全画像に一律の M を使うので、画像ごと・クラスごとに最適強度が違う場合の細やかな制御はできない(これが画像ごと適応の後続研究の動機の一つ)。
- 方向や空間構造が意味を持つドメイン(自動運転で左側通行/右側通行、上が空・下が地面、など)では、回転・反転・大きな平行移動が意味を壊すため、変換集合と強度範囲を慎重に絞る必要があります。

### 発展トピック・研究の最前線
- TrivialAugment([TrivialAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)): RandAugment の (N, M) のうち N を 1 に固定し M もランダム化。「探索より単純化」の到達点として、チューニング不要でほぼ匹敵すると報告。RandAugment の「2つのつまみ」すら不要かもしれない、という問いを突きつけました。
- UniformAugment / Adversarial AutoAugment: 強度分布の与え方をさらに緩めて連続一様分布から引いたり、逆に敵対的に「モデルが苦手な強い拡張」を選んだりする派生。
- 画像ごと適応の拡張: クラスや難易度に応じて強度を変える方向(class-wise や sample-wise の自動拡張)。一律 M の限界に対する応答で、難しいサンプルには弱く、易しいサンプルには強くかける、といった制御を狙います。
- 検出・セグメンテーション専用の拡張探索: 分類向けに設計された変換集合は位置タスクで最適とは限らず、タスク特化の変換集合(スケールジッタやコピーペースト等)と評価が研究されています。
- 近代的な学習レシピへの統合: RandAugment は Mixup・CutMix・Random Erasing・label smoothing(正解ラベルを少しぼかして過信を防ぐ)などと組み合わせて、強い画像分類レシピ(ResNet の再学習レシピや DeiT など)の標準部品として広く使われています。

### さらに学ぶための関連トピック
- [AutoAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [TrivialAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cutout / Random Erasing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Mixup / CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Fine-tuning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)


## ReduceLROnPlateau

### ひとことで言うと
あらかじめ「何エポックで学習率を下げる」と決め打ちするのではなく、学習の様子を実際に観測しながら、検証指標(validation metric、学習に使っていないデータでの成績。損失や精度など)が一定期間まったく良くならなくなったら、そのタイミングで学習率(モデルのパラメータを1回の更新でどれくらい動かすかを決める数値)を下げる、という「成績連動型」の学習率スケジューラ(学習率の時間変化のさせ方)です。下げるべき瞬間を人間が試行錯誤で探す手間が消える点が最大の利点です。

### 直感的な理解
山下りにたとえます。あなたは霧の中で山を下っていて、できるだけ低い谷底に着きたい。最初は大股(大きな学習率)で速く下る方が効率的ですが、谷底に近づくと大股のままでは底を通り過ぎて反対側の斜面に登ってしまいます。だんだん歩幅を小さくしたい。

問題は「いつ歩幅を小さくすべきか」です。固定スケジュール型のスケジューラ(たとえば「30歩ごとに歩幅を半分にする」)は、地形を見ずに歩数だけで決めます。でも実際には、傾斜が急なうちは大股のままがよく、平らになってきたら初めて小股にすべきです。ReduceLROnPlateau は「もう何歩進んでも標高(検証損失)が下がらなくなった」という事実を観測してから歩幅を縮めます。つまり地形に合わせて適応するのです。

なぜこれが要るのか。深層学習では、最適な「下げどき」がデータセット・モデル構造・バッチサイズ・乱数の初期値によって毎回変わります。事前には知りようがありません。固定スケジュールだと、早すぎれば伸びしろを捨て、遅すぎれば無駄に時間を使います。観測してから下げれば、この当たり外れを自動で吸収できます。

### 基礎: 前提となる概念
理解に必要な言葉を順に噛み砕きます。

- 学習率 (learning rate): ニューラルネットの学習は「予測の誤差を測り、その誤差を減らす方向にパラメータを少し動かす」を繰り返します。この「少し」の大きさを決めるのが学習率です。大きいと速く進むが乱暴、小さいと丁寧だが遅い。学習率は深層学習で最も影響の大きいハイパーパラメータ(人間が事前に決める設定値)だと広く言われます。

- エポック (epoch): 訓練データ全体を1周見ることを1エポックと数えます。学習は数十〜数百エポック繰り返します。

- 検証指標とプラトー (plateau): 学習に使わない別のデータ(検証セット)で測った成績を検証指標と呼びます。学習が進むとこの指標は改善していきますが、やがて頭打ちになります。この「高原のように平らで変化しない状態」をプラトー(plateau)と呼びます。プラトーは、いまの学習率では届かない、より深い谷底の手前で振動している状態に対応することが多く、学習率を下げると再び改善が始まることがよくあります。

- スケジューラの2分類: 学習率の下げ方には大きく2系統あります。(1) 事前に時間で決め打ちする「固定スケジュール型」([Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、[Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、[Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、[Linear LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)、コサイン減衰など)。(2) 学習中の観測に応じて動的に決める「適応型」。ReduceLROnPlateau は後者の代表です。

### 仕組みを詳しく
各エポック(または評価のたび)に、監視対象の検証指標を1つスケジューラに渡します。内部ロジックは次の通りです。

1. これまでの「ベスト(best、最良値)」を記録しておく。
2. 新しい指標がベストを `threshold`(しきい値。改善とみなす最小幅)以上更新したら、ベストを更新し「我慢カウンタ(patience counter)」を 0 にリセット。
3. 更新できなかったら我慢カウンタを 1 増やす。
4. 我慢カウンタが `patience`(我慢する回数)に達したら、学習率を `factor`(倍率、例 0.1)倍して下げ、カウンタをリセット。
5. 下げた直後は `cooldown`(クールダウン、しばらく様子見する期間)エポックだけ判定を休む。下げた効果が現れるまで時間がかかるので、すぐ再判定して連続で下げてしまうのを防ぐため。
6. 学習率が `min_lr`(下限)より下がらないようにする。

数値例。`mode='min'`(損失を監視し、小さいほど良い)、`factor=0.1`、`patience=5`、`threshold=1e-4` とします。検証損失が次のように推移したとします。

```
epoch:  1    2    3     4      5      6      7      8      9     10
loss : 0.50 0.42 0.40 0.399  0.398  0.398  0.398  0.398  0.398 0.398
```

エポック3まで順調に下がりベストは 0.40。エポック4以降は 0.399, 0.398 とごくわずかしか改善せず、threshold=1e-4 を下回る改善は「改善なし」とみなされます(0.40→0.399 は 0.001 の改善で threshold を超えるので一度はベスト更新、その後は停滞)。改善なしが patience=5 回続いた時点(おおよそエポック9〜10あたり)で、学習率が 0.1 倍に下げられます。下げた後、低い学習率で再び損失が下がり始めれば、また改善が続くのでしばらく下げません。

学習率と検証損失の連動をASCII図にすると、損失が平らになるたびに学習率が一段下がり、その直後にまた損失が下がりだす、という階段状の追いかけっこが見えます。

```
loss |＼
     |  ＼___          ← ここで頭打ち(plateau)
     |       ＼＿_      ← lr 下げたら また下がる
     |           ＼__
     +----------------------> epoch
lr   |‾‾‾‾‾‾‾|
     |       |_______
     |               |________
     +----------------------> epoch
            ↑下げ     ↑下げ
```

`threshold_mode` という設定もあり、改善幅を「絶対値(rel/abs)」のどちらで測るかを選べます。`rel`(相対)なら「ベストの 0.0001 倍以上の改善でないと改善とみなさない」、`abs`(絶対)なら「ベストより 0.0001 以上小さくなければ改善とみなさない」という違いです。損失のスケールが大きいタスクでは rel、小さいタスクでは abs が扱いやすいことが多いです。

実装上の最重要ポイント: このスケジューラだけは、更新を進めるメソッド(典型的には `step()`)に必ず監視する指標の値を渡します(例 `scheduler.step(val_loss)`)。固定スケジュール型は内部のステップ数だけで動くので引数なしで呼べますが、ReduceLROnPlateau は「外から渡された成績」を見て判断するため指標が必須です。`mode='max'` にすれば精度や mAP のような「大きいほど良い」指標も監視できます。

数式で1点補足します。我慢カウンタを `c`、ステップ `t` での学習率を `η_t` とすると、おおまかに

```
c が patience に達した瞬間:  η ← max(η * factor, min_lr),  c ← 0
それ以外:                    η は据え置き
```

固定スケジュール型が `η_t = η_0 * f(t)` のように時間 `t` の関数であるのに対し、ReduceLROnPlateau は「過去の検証指標列」の関数になっている、という構造の違いがポイントです。

### 手法の系譜と主要論文
ReduceLROnPlateau は特定の1論文で初めて提案された手法というより、研究者の経験則を実装に落とし込んだ実務テクニックです。系譜は次のように描けます。

- 手動 step decay の時代(2012頃)。Krizhevsky, Sutskever & Hinton, "ImageNet Classification with Deep Convolutional Neural Networks"(AlexNet, NIPS 2012)では、研究者が学習曲線を目で見て「検証誤差が頭打ちになったら学習率を 10 分の 1 にする」という運用をしていました。ReduceLROnPlateau はこの「人間が学習曲線を見て手で下げる」操作を、そっくり自動化したものと理解できます。動機は、手動運用が再現性に乏しく面倒だったこと。新規性は、改善停滞の検出条件(patience, threshold)を明文化して機械に任せた点にあります。

- 実務指針としての定式化(2012)。Bengio, "Practical Recommendations for Gradient-Based Training of Deep Architectures"(arXiv:1206.5533)は、学習率が最重要ハイパーパラメータであり、検証性能を見ながら調整すべきだと論じました。ReduceLROnPlateau はこの指針を機械的に実装したものといえます。

- 早期停止 (early stopping) との血縁。ReduceLROnPlateau の「我慢カウンタ」ロジックは、早期停止(検証指標が patience エポック改善しなければ学習を打ち切る)とまったく同じ構造を持ちます。違いは、停滞を検知したときに「学習を止める」のが早期停止、「学習率を下げてもう一度粘る」のが ReduceLROnPlateau だという点だけです。実務では両者を組み合わせ、「数回下げてもなお改善しなければ最終的に停止する」という二段構えがよく使われます。この意味で ReduceLROnPlateau は、早期停止という古典的正則化手法の「いきなり止める前に学習率で粘る」拡張版と位置づけられます。

- フレームワークによる普及。Keras の `ReduceLROnPlateau` コールバックと、PyTorch の同名スケジューラ(`torch.optim.lr_scheduler.ReduceLROnPlateau`)が広く使われ、デファクトの「適応型スケジューラ」として定着しました。

- 大規模学習での後退(2018以降)。BERT 以降の大規模 Transformer 学習([Linear LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照)では、検証評価が高コストで学習も長いため、事前に決めた warmup + linear/cosine スケジュールが好まれる傾向が強まりました。ReduceLROnPlateau が活きるのは、評価が安いタスクや、学習回数が少なく1回1回を最大限活かしたい中小規模の実験段階です。

### 論文の実験結果(定量データ)
ReduceLROnPlateau 単体を主題にした論文は少ないため、近縁の「学習率を段階的に下げると何が起きるか」を示す結果を引きます。

- AlexNet(NIPS 2012)の手動 step decay。ILSVRC-2012(ImageNet、1000 クラス・約 128 万枚の画像分類データセット)で、検証誤差が頭打ちになるたびに学習率を 10 分の 1 にする運用を 3 回行い、最終的に top-5 誤差(正解が予測上位5位以内に入らなかった割合)を 16.4% まで下げました。これは当時の 2 位(約 26%)を大きく引き離す結果で、深層学習ブームの起点になりました。ここで重要なのは「学習率を下げるたびに検証誤差がもう一段下がった」という観測で、ReduceLROnPlateau が自動化しようとした現象そのものです。

- ResNet(He et al., CVPR 2016)の段階減衰。CIFAR-10(10 クラス・6 万枚の小画像分類データセット)で、誤差が頭打ちになる 32k・48k イテレーションで学習率を 10 分の 1 にすることで、152 層級のネットワークを安定して学習させ、当時の最良精度を達成しました。減衰のたびにテスト誤差曲線がはっきり段差を作って下がる様子が論文の図に示されており、「プラトーで下げると効く」という経験則の代表例です。

- 音声認識・系列モデルでの活用。NewBob と呼ばれる古典的スケジュール(検証フレーム精度が一定幅未満しか改善しなくなったら学習率を半分にし、改善が止まったら停止する)は、HMM-DNN 音声認識システムの標準的な学習法として長く使われてきました。これは ReduceLROnPlateau の `factor=0.5` 相当の運用で、「改善が鈍ったら半減」という経験則が音声分野でも独立に確立されていたことを示します。半減(0.5 倍)は画像の 0.1 倍より穏やかで、検証指標がノイジーなタスクで「下げすぎ」を避ける狙いがあります。

- アブレーション的な見方。固定スケジュールと適応型を比べた多くの実験報告では、適切にチューニングされた固定 cosine/step は、ReduceLROnPlateau とほぼ同等かやや上回ることが多いとされます。一方、最適な減衰タイミングが未知の探索段階では、ReduceLROnPlateau の方が「外す」リスクが小さいという報告があります。`factor` の選び方にも明確なトレードオフがあり、0.1 倍は一度に大きく下げるので段差がはっきり出る代わりに下げすぎのリスクがあり、0.5 倍は緩やかに何段も下げるので滑らかだが収束まで時間がかかります。つまり指標は「最終精度」より「チューニング工数あたりの精度」で見るべきで、ReduceLROnPlateau の価値は後者にあります。

これらの数値が示す本質は、「学習率を下げる」という操作が単なる微調整ではなく、検証誤差を不連続にもう一段下げる強力な手段だということです。AlexNet で 3 回の手動減衰が top-5 誤差を段階的に押し下げ、ResNet でテスト誤差曲線にはっきりした段差を作ったように、減衰のたびに「もう一段深い谷」へ落ちる。だからこそ「いつ下げるか」が重要で、ReduceLROnPlateau はそのタイミング決定を、検証指標の停滞という客観的な合図に基づいて自動化します。

### メリット・トレードオフ・限界
メリット
- 下げるタイミングを人間が事前にチューニングしなくてよい。データに合わせて自動で適応する。
- 学習の実際の成績に連動するので、無駄に早く下げたり遅く下げたりしにくい。
- 既存の学習ループに後付けしやすく、最初に試す適応型として実用的。

トレードオフ・限界
- 検証評価を定期的に回す必要があり、評価が重いタスク(大規模言語モデルなど)ではコストがかさむ。
- patience, threshold, factor, cooldown, min_lr と設定項目はむしろ多く、これらが悪いと反応が遅すぎたり早すぎたりする。
- 本質的に「過去を振り返って下げる」ため反応が遅れる(プラトーに入ってから patience 回ぶん待つ)。warmup を内蔵しないので、Transformer の初期不安定性には別途対処が要る。
- 検証指標のノイズ(評価ごとのばらつき)に振り回されやすく、threshold の設定がシビア。
- 学習率の軌跡が事前に決まらないため、再現性や実験比較がしにくい(同じ設定でも乱数で下げどきが変わる)。研究上は「適応スケジュールの分散をどう抑え、固定スケジュールと公平に比較するか」が未解決の論点として残ります。

### 発展トピック・研究の最前線
- 適応型と固定型のハイブリッド。warmup(序盤に学習率を上げる)+ コサイン減衰を基本にしつつ、終盤だけ ReduceLROnPlateau を併用する、といった折衷が実務で使われます。
- ハイパーパラメータ最適化との統合。population-based training(Jaderberg et al., 2017)や Hyperband のような手法は、学習率の下げどきも含めて自動探索するため、ReduceLROnPlateau の「手で patience を決める」部分を上位から置き換える方向にあります。
- 学習率そのものを学習する研究(hypergradient descent、Baydin et al., 2018)もあり、「成績を見て下げる」を勾配ベースで連続的に行う発展形と位置づけられます。
- メトリック設計の問題。何を監視指標にするか(検証損失か、タスク固有の評価指標か)で挙動が変わります。損失は下がっているのにタスク指標が頭打ち、という乖離が起きるタスクでは、監視対象の選択自体が研究課題になります。

### さらに学ぶための関連トピック
- [Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Linear LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## RMSProp

### ひとことで言うと
RMSProp(アールエムエス・プロップ)は、「パラメータごとに、それぞれ違う学習率を自動で調整する」オプティマイザです。最近よく大きく動く(勾配が大きい)パラメータは慎重に小さく、あまり動かない(勾配が小さい)パラメータは大きめに動かす——という調整を、勾配の二乗の移動平均を使って行います。RMS は Root Mean Square(二乗平均平方根、値を二乗して平均して平方根を取った量=典型的な大きさ)の略。AdaGrad の「学習率が枯れて止まる」弱点を直し、Adam の直接の前身となった、適応的オプティマイザの歴史で要となる手法です。

### 直感的な理解
ニューラルネットの重みは、層や位置によって勾配(傾き)の大きさが何桁も違うことがよくあります。ある重みの勾配は毎回 100 くらい、別の重みは 0.01 くらい、という具合です。すべてに同じ学習率 η を掛けると、勾配が大きい重みは動きすぎて発散しかけ、勾配が小さい重みはほとんど動かず学習が進まない、という板挟みになります。η を大きくすれば前者が暴れ、小さくすれば後者が止まる。1 つの学習率では両立できません。

RMSProp の発想は「各重みの最近の傾きの大きさで割り算してしまえばいい」というものです。傾きが大きい重みは大きい数で割られるので歩幅が抑えられ、傾きが小さい重みは小さい数で割られるので歩幅が押し出される。結果として、すべての重みがだいたい同じ歩幅で進むようになります。坂の急なところは小股で、なだらかなところは大股で、自動的に歩幅を変える歩き方だと思えば直感的です。

### 基礎: 前提となる概念
- 学習率(η): 1 ステップで動かす基本の歩幅。全パラメータ共通だと問題が起きる、というのが本題の出発点。
- 移動平均: 過去を少しずつ忘れながら平均し続ける手法。`s ← ρ·s + (1−ρ)·新しい値` の形。`ρ`(ロー)が大きいほど過去を長く覚える。指数移動平均(EMA)の一種。
- 二乗の移動平均: 勾配 `g` を二乗してから移動平均を取った量 `s`。`√s` がその重みの「最近の典型的な勾配の大きさ(RMS)」を表す。
- 適応的(adaptive)学習率: パラメータごとに実効的な学習率を変えること。AdaGrad と RMSProp が確立した概念。
- 非定常(non-stationary): 学習が進むにつれて、最適な進行方向や勾配のスケールが変わっていくこと。深層学習や RNN(リカレントニューラルネット、系列を順に処理するネット)では普通に起きる。RMSProp はこれに強い。
- 前提条件付け(preconditioning): 更新する前に勾配を「整える」行列を掛ける操作。RMSProp の `1/√s` は、その整える行列を対角だけで近似したものと見なせる(後述)。

### 仕組みを詳しく
RMSProp は AdaGrad の「過去の勾配二乗を全部足す」をやめ、「最近の勾配二乗だけを重視する移動平均」に変えました。`s` が勾配二乗の移動平均、`ρ` は減衰率(通常 0.9)、`ε` はゼロ割りを防ぐ小さな数(例 1e-8)です。

```
s ← ρ · s + (1 − ρ) · g²        ← 勾配の二乗を、最近を重視する移動平均で更新
w ← w − η · g / (√s + ε)         ← 勾配を √s で割って正規化してから更新
```

`(1 − ρ)·g²` により古い情報は `ρ` 倍ずつ薄れます。ρ=0.9 なら、おおむね直近 10 ステップぶんの勾配二乗を見ているイメージです(重みが `ρ^k` で減るので、有効な記憶長はだいたい `1/(1−ρ)` ステップ)。AdaGrad のように無限に足し続けないので、`s` が一定範囲に収まり、学習率が完全に枯れて止まることがありません。

#### なぜ √s で割るのか
`g / √s` の `√s` は、そのパラメータの「最近の勾配の典型的な大きさ(RMS)」です。これで割ると、勾配が大きいパラメータも小さいパラメータも更新量がだいたい同じスケール(おおよそ ±η)に揃います。「動きすぎる重みは抑え、動かなすぎる重みは押し出す」という自動調整が、パラメータ 1 個ごとに効くわけです。式としては「勾配の符号 × 一定の歩幅」に近い挙動になり、勾配が安定して同じ向きを指していれば `g/√s ≈ ±1` に近づきます。

#### 数値例
学習率 `η = 0.001`、ρ=0.9 とします。
- 勾配がずっと `g = 100` の重み: `s` は `100² = 10000` に近づき `√s ≈ 100`。更新量は `η·100/100 = 0.001`。
- 勾配がずっと `g = 0.01` の重み: `s ≈ 0.0001`、`√s ≈ 0.01`。更新量は `η·0.01/0.01 = 0.001`。

どちらも更新量がほぼ `0.001` に揃いました。勾配の絶対的な大きさに関係なく、安定した歩幅で進めるのが RMSProp の効果です。逆に、勾配が急に大きくなった直後は `s` がまだ追いついておらず一時的に大きく動く、という過渡的な挙動も持ちます。

#### tensor の形・メモリ
勾配二乗の移動平均 `s` を、各パラメータと同じ形(shape)で 1 つずつ保持します。重みが `[512, 256]` なら `s` も `[512, 256]`。追加メモリはパラメータ数と同量(1 セット)で、Momentum を含む Adam(`m`+`v` の 2 セット)の半分です。

#### AdaGrad との違いを図で
```
AdaGrad : s = g₁² + g₂² + g₃² + …(全部足す)→ 分母が単調増加 → 学習率が枯れて停止
RMSProp : s = ρ·s + (1−ρ)·g²(移動平均)   → 分母が一定範囲に収まる → 枯れない
```
この一行の違いが、長く回す深層学習で AdaGrad が止まり RMSProp が止まらない理由です。AdaGrad の累積和は必ず増えていくので、ステップ数 t に対して実効学習率がおおよそ `1/√t` で減衰し続けます。短期間の凸最適化では理論保証がある一方、深層学習のように長く回すと早すぎる停止を招くのが弱点でした。

### 手法の系譜と主要論文
- AdaGrad — Duchi, Hazan & Singer (JMLR 2011, "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization")。パラメータごとに学習率を変える適応的勾配法の概念を確立。過去の勾配二乗の累積和で割る。スパースな特徴(自然言語の単語など、まれにしか勾配が立たない特徴)で強いが、累積和が単調増加するため学習率が枯れる弱点があった。理論的にはオンライン凸最適化のリグレット解析で裏付けられた手法。
- RMSProp — Tieleman & Hinton (2012)。正式な論文ではなく、Geoffrey Hinton の Coursera 講義「Neural Networks for Machine Learning」の Lecture 6e スライドで初めて紹介された珍しい出自を持ちます。AdaGrad の累積和を移動平均に置き換えることで、学習率の枯渇を解消しました。引用の慣習として "Tieleman & Hinton, 2012, Lecture 6.5, Coursera" と書かれます。
- AdaDelta — Zeiler (2012, arXiv:1212.5701)。RMSProp とほぼ同時期に独立に提案された近縁手法。勾配二乗の移動平均に加え、過去の更新量の移動平均も保持し、学習率 η を更新量の RMS から自動的に決めることで、η を手で設定しなくて済むようにしました。RMSProp と発想を共有しつつ、単位(次元)の整合性を意識した設計が特徴です。
- Adam — Kingma & Ba (ICLR 2015)。RMSProp の 2 次モーメント(勾配二乗の移動平均)に、Momentum の 1 次モーメント(勾配の移動平均=慣性)を組み合わせ、さらに初期バイアス補正を加えたもの。つまり RMSProp は Adam の直接の前身であり、Adam の `v` 項は RMSProp そのものです。詳しくは [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)。なお Adam 論文には Momentum 付き RMSProp も比較対象として登場します。

この系譜は「パラメータごとの歩幅調整」という一本の問題意識でつながっています。AdaGrad がアイデアを出し、RMSProp が実用化し、Adam が Momentum を足して完成形に近づけた、という流れです。

### 論文の実験結果(定量データ)
RMSProp は講義スライド由来で、標準ベンチマークでの厳密な比較表を伴う論文ではありません。Hinton の講義での主張は「AdaGrad は学習が進むと有効学習率が小さくなりすぎて止まるが、二乗の移動平均にすればこれを避けられ、ミニバッチ学習や RNN で安定して良く動く」という経験的なものでした。

その後の研究で、RMSProp の有効性は他手法との比較を通じて定量的に示されています。

- Kingma & Ba (2015, Adam 論文)は、MNIST(手書き数字 0–9 の 28×28 画像 6 万枚の分類)や CIFAR-10(32×32 カラー画像 6 万枚の 10 クラス分類)での学習 cost(=訓練 loss)の減少を、SGD・AdaGrad・RMSProp・SGD+Nesterov・Adam で比較しました。RMSProp は AdaGrad より明確に速く・安定して学習が進み、SGD 系より良い場面が多いが、Momentum を含む Adam にはやや劣る、という位置づけが示されました。ここで言う「cost の減少が速い」とは、同じエポック数(データの周回数)でより低い訓練 loss に到達することで、開発の効率に直結します。
- RNN の学習: 勾配のスケールが時間方向に大きく変動する非定常性があり、RMSProp が安定して機能することが広く報告されました。系列が長いと勾配が指数的に大きく/小さくなりやすく(勾配爆発・消失)、各時刻でスケールがばらつくため、パラメータごとの正規化が効きます。RMSProp が長く実務(特に DeepMind 系の強化学習や RNN)で使われてきた最大の理由はこの安定性にあります。
- 強化学習: 報酬がまばらで勾配のスケールが安定しない設定で、RMSProp は経験的に頑健とされ、初期の深層強化学習(DQN の一部実装など)で標準的に使われました。

アブレーション的に見ると、RMSProp は「Adam から Momentum(`m`)とバイアス補正を抜いたもの」とほぼ等価です。逆に言えば、RMSProp に Momentum とバイアス補正を足すと Adam になります。したがって「RMSProp 単体 vs Adam」の比較は「Momentum を足すと何が良くなるか」のアブレーションそのもので、一般に Momentum を足すと方向のジグザグが抑えられ収束がさらに速く・安定になる、というのが共通理解です。一方でバイアス補正がない RMSProp は学習序盤に `s` が小さく更新が暴れやすいので、序盤は学習率を控えめにする運用が好まれます。

### メリット・トレードオフ・限界
メリット
- パラメータごとに学習率を自動調整するので、勾配の大きさが層やパラメータで何桁も違っても安定して学習できる。
- AdaGrad と違い移動平均なので学習率が枯れて止まらない。RNN や非定常問題、強化学習に強い。
- 状態が `s` 1 つだけで、Adam の半分のメモリ。
- 学習率 η の調整がやや楽(勾配スケールの違いを吸収するため)。

トレードオフ・限界
- Momentum(慣性)を含まないので、方向のジグザグ抑制は別途必要(Adam は Momentum を足して補う)。
- `ρ` と `η` の 2 つを調整する必要がある。
- バイアス補正がないため学習序盤が不安定になりやすい。
- 正式な収束理論が薄く、設定次第で不安定になることもある。
- 多くの場面で Momentum まで含めた Adam/AdamW のほうが扱いやすいため、現在は単体での出番は減っている。

未解決・注意点として、RMSProp も Adam と同様、有効学習率(`η/√s`)が単調でないため、特定の問題で収束が保証されないことがあります(Adam の収束問題と同根で、Reddi et al. 2018 の AMSGrad が指摘した構造)。また ε の値が大きいと適応性が薄れて SGD に近づき、小さすぎると割り算が暴れる、という感度があります。実装によって ε を平方根の中に入れる(`√(s+ε)`)か外に出す(`√s + ε`)かで挙動が微妙に変わる点も、再現性の落とし穴として知られています。

### 発展トピック・研究の最前線
- 適応的オプティマイザの理論: RMSProp/Adam がなぜ・どんな条件で収束するかは Reddi et al. (ICLR 2018, AMSGrad) 以降も活発に研究されています。非凸最適化での収束保証や、勾配の有界性などの仮定をどこまで緩められるかが論点です。
- 二乗統計の活用先: RMSProp の「勾配二乗で正規化する」発想は、Adam・AdamW・Adafactor・[Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) など現代オプティマイザの大半に何らかの形で受け継がれています(Lion は逆に `v` を捨てる選択をした点が対照的で、二乗統計が本当に必要かという問いを投げかけました)。
- 正規化と前提条件付け(preconditioning): `1/√s` で割る操作は、対角近似のプリコンディショナー(更新を整える行列を対角成分だけで近似したもの)とみなせます。これを対角でなくブロックや行列全体で行うのが K-FAC(Martens & Grosse, 2015)や Shampoo(Gupta et al., 2018)、近年の SOAP・Muon といった準 2 次オプティマイザの研究で、RMSProp の対角プリコンディショニングをより豊かな構造へ一般化する方向として発展しています。
- メモリ効率化: Adafactor は RMSProp 系の `s` を行・列のベクトルに因子分解してメモリを `O(行+列)` に削減し、巨大言語モデルで実用化されました。RMSProp の発想は、規模が大きくなるほど「2 次統計をどう安く近似するか」という問題へとつながっています。

### さらに学ぶための関連トピック
- [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## SGD with Momentum

### ひとことで言うと

ニューラルネットの学習は「坂道をボールが転がって谷底 (誤差が最小の場所) を探す」作業ですが、SGD with Momentum は、そのボールに「勢い (慣性)」を持たせる工夫です。これにより、ジグザグに無駄に揺れるのを抑えながら、谷底へ速く着くことができます。追加の計算はほぼゼロ、メモリは重みと同量の「速度」を覚えるだけで、最も基本的かつ実用上いまだに強力なオプティマイザです。多くの最先端の画像認識モデルが、いまだに Adam ではなくこの SGD+Momentum で学習されています。

### 直感的な理解

最も素朴な学習法は「今いる場所の坂を見て、下る方向に一歩進む」を繰り返すものです。しかしこれだと、細長い谷 (片方向は急で、もう片方向はゆるい地形) では、急な壁を行ったり来たりするピンボールのような動きになり、谷の奥へなかなか進めません。Momentum は、ボールに慣性を与えることでこの無駄な往復を抑えます。同じ方向に進み続けるとどんどん加速し、毎回向きが反転する方向は打ち消し合うので、本当に進むべき方向だけ速くなります。物理の「重いボールが谷を転がる」イメージそのままで、実際に原典の論文ではこれを "heavy ball method (重いボール法)" と呼びます。

なぜ深層学習の地形が「細長い谷」になるのか、というのも直感の助けになります。ニューラルネットの損失関数は、ある方向には急激に、別の方向にはゆるやかにしか変化しない、引き伸ばされた地形 (専門的には「ヘッセ行列の条件数が大きい」状態) を持ちます。素朴な勾配降下はこの地形が大の苦手で、Momentum はその弱点をピンポイントで補います。

### 基礎: 前提となる概念

#### そもそも「学習」とは何をしているのか

ニューラルネットの中には「重み (パラメータ)」と呼ばれる数値がたくさんあります。画像を見て「これは車だ」と当てるモデルなら、何百万個もの重みがあります。学習とは、この重みを少しずつ調整して、モデルの予測の誤差 (loss、ロス。予測がどれだけ外れているかを表す数値) を小さくしていく作業です。

誤差を小さくするために使うのが「勾配 (gradient、グラディエント)」です。勾配とは「重みをほんの少し増やしたら誤差がどれだけ変化するか」を表すベクトル (数値の並び) です。勾配は「今いる場所で最も急に誤差が増える方向」を指すので、その逆方向に進めば誤差が減ります。これが「勾配降下法 (gradient descent)」の考え方です。勾配は誤差逆伝播 (バックプロパゲーション、合成関数の微分を出力側から入力側へ連鎖的に計算する手法) で効率よく求めます。

#### 素朴な勾配降下法 (plain SGD) の問題

最も単純な式は次の通りです。`w` は重み、`η` (イータ) は「学習率 (learning rate)」= 一歩の大きさ、`g` は勾配です。

```
w ← w − η · g
```

「勾配の逆方向に、学習率の分だけ一歩進む」を繰り返します。SGD (Stochastic Gradient Descent、確率的勾配降下法) の「確率的」とは、データ全部ではなく一部 (ミニバッチ、例えば 32 枚の画像) だけを使って勾配を近似計算する、という意味です。全データを使うと計算が重いので一部で近似します。この近似のせいで勾配にはノイズ (揺らぎ) が乗ります。

このやり方には 3 つの弱点があります。1 つ目はジグザグ振動: 細長い谷だと、急な軸方向の勾配が大きすぎて毎回オーバーシュート (行き過ぎ) し、壁を跳ね返るように振動して谷の奥へ進めません。2 つ目は平坦な場所で止まる: 勾配がほぼゼロの平坦な領域 (プラトー) に入ると、一歩がほとんど進まず学習が事実上停止します。3 つ目はミニバッチ由来のノイズ: 一歩ごとに方向がランダムに揺れ、収束が遅くなります。この「細長い谷」「プラトー」「ノイズ」は、深層学習の loss 地形では非常によく起こり、Momentum は 3 つ全てを同時に緩和します。

### 仕組みを詳しく

Momentum (モメンタム、運動量) は、過去の進行方向を「速度 (velocity)」として覚えておき、それを今の勾配に足し込みます。`v` が速度、`μ` (ミュー) が momentum 係数 (通常 0.9 などを使う) です。

```
v ← μ · v − η · g     (速度を更新: 過去の速度を μ 割残し、今の勾配を足す)
w ← w + v             (その速度の分だけ重みを動かす)
```

ポイントは `μ · v` の部分です。前のステップの速度を 9 割 (μ=0.9 のとき) 引き継ぐので、同じ方向に進み続けると速度がどんどん積み上がり (加速し)、逆に毎回向きが反転する方向は打ち消し合って小さくなります。

#### 数値例でイメージする

学習率 `η = 0.1`、momentum `μ = 0.9` とします。ある方向の勾配が毎回 `g = 1` で一定 (つまりずっと下り坂) だったとします。

- plain SGD: 毎回 `−0.1` ずつ動く。10 ステップで `−1.0`。
- Momentum: 速度が `−0.1, −0.19, −0.271, −0.34, …` と増え続け、最終的に速度は `−η·g / (1−μ) = −0.1/0.1 = −1.0` に近づく。つまり一歩あたり最大で plain SGD の 10 倍 (`1/(1−μ)` 倍) の速さになります。

逆に、勾配の向きが `+1, −1, +1, −1, …` と毎回反転する振動方向では、速度が積み上がらず打ち消されるので、振動が抑えられます。これが「ジグザグを抑えて、本当に進むべき方向だけ加速する」効果です。`1/(1−μ)` という倍率は重要で、μ=0.9 なら 10 倍、μ=0.99 なら 100 倍の実効加速になります。だから μ を上げるほど速くなる一方、行き過ぎ (オーバーシュート) のリスクも上がります。

#### tensor の形とメモリ

重み `w`、勾配 `g`、速度 `v` はすべて同じ形 (shape) を持ちます。ある全結合層の重みが `[512, 256]` (出力 512、入力 256) なら、`v` も `[512, 256]` です。Momentum は重み 1 個ごとに速度を 1 個ずつ余分に保持するので、追加メモリは重みと同量です (パラメータ数が 1000 万なら速度も 1000 万個)。後述の Adam が「速度 (1 次モーメント) + 分散 (2 次モーメント)」の 2 つを持つのに比べ、Momentum は 1 つで済む点でメモリ効率が良いです。大規模モデルの学習では、このオプティマイザ状態のメモリが GPU メモリの大きな割合を占めるため、状態が 1 本で済むことは実利上も意味があります。

#### なぜこの設計なのか

数学的には、Momentum は loss を「指数移動平均された勾配」で更新する手法とみなせます。`v` を展開すると `v_k = −η Σ μ^{k−i} g_i` となり、直近の勾配ほど重み (μ の累乗) が大きく、過去ほど指数的に減衰します。これにより、ノイズの多いミニバッチ勾配の平均化 (分散低減。ランダムな揺れが平均で打ち消される) と、一貫した方向への加速の両方を、たった 1 本の速度ベクトルで同時に実現しています。1 つの仕掛けで「ノイズ低減」「加速」「振動抑制」の 3 つを賄える設計の巧みさが、この手法が半世紀使われ続ける理由です。

なお実装には 2 流派があります。上記の式 (PyTorch の dampening=0 の形に近い) のほか、`v ← μ·v + g; w ← w − η·v` と書く流派もあり、学習率 η が速度更新の内側にあるか外側にあるかで挙動がわずかに変わります。フレームワークごとの定義差は、学習率を移し替えるときに注意が必要なポイントです。

### 手法の系譜と主要論文

Polyak (1964) "Some methods of speeding up the convergence of iteration methods": Momentum の原典で、「heavy ball method (重いボール法)」と呼ばれます。重いボールは慣性で細長い谷を効率よく転がる、という物理的直感をそのまま最適化に持ち込みました。提案理由は振動を抑えつつ収束を速めること。効果は条件数の悪い (細長い) 二次関数で収束が大幅に速くなること (理論的には条件数 κ に対し、勾配降下の収束が κ に依存するのを √κ オーダーに改善)。トレードオフは μ を大きくしすぎると行き過ぎて発散・振動が増えること。

Rumelhart, Hinton & Williams (1986) "Learning representations by back-propagating errors" (Nature): 誤差逆伝播 (バックプロパゲーション) の有名論文で、ニューラルネット学習に momentum 項を実用的に導入しました。当時から「学習を安定・高速化する標準テクニック」として扱われています。

Sutskever, Martens, Dahl & Hinton (2013) "On the importance of initialization and momentum in deep learning" (ICML 2013): 深層学習における momentum の重要性を実験的に示した代表的論文です。主張 — 適切な初期化 (重みの初期値の決め方) と momentum を組み合わせれば、当時「難しい」とされた深い・再帰的なネットワークも、もっと凝った 2 次最適化手法 (Hessian-Free など、勾配だけでなく曲がり具合まで使う重い手法) に匹敵する性能で学習できる。理由 — momentum は谷の浅い方向の進みを積み上げて加速し、深く急な方向の振動を相殺するため、実質的に 2 次的な効果の一部を安価に得られる。効果 — 深い RNN (再帰型ネット) の学習に成功。トレードオフ — μ のスケジューリング (序盤は小さく、徐々に大きく) が重要で、固定値だと不安定になりやすいと指摘。

He ら (2016) "Deep Residual Learning (ResNet)" (CVPR): ImageNet で SOTA を取った ResNet を、SGD+Momentum (μ=0.9) + weight decay + 段階的学習率減衰 で学習しました。「最先端の画像モデルが Adam でなく SGD+Momentum で学習される」流れを決定づけた代表例です。

Wilson ら (2017) "The Marginal Value of Adaptive Gradient Methods" (NeurIPS): Adam のような適応的手法が、訓練 loss は速く下げても、テスト精度 (汎化) では SGD+Momentum に劣る場合があることを複数タスクで実証しました。なぜ SGD のほうが汎化しやすいのかという論点を学界に強く意識させた論文です。

この系譜は後に [Nesterov Accelerated Gradient](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) (先読みする改良)、RMSProp / [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) (パラメータごとに学習率を自動調整) へと発展します。Adam の「1 次モーメント m」は、Momentum の速度 `v` の考え方をそのまま受け継いだものです。

### 論文の実験結果 (定量データ)

Sutskever ら (2013) は、深い feedforward ネットや RNN (言語モデリング等) で実験し、適切な初期化と高い momentum (μ を 0.9〜0.99 程度まで段階的に上げる) を組み合わせると、momentum なしの SGD では学習が進まなかった深いネットでも収束し、当時の Hessian-Free 最適化 (計算が重い 2 次手法) に匹敵する性能を、はるかに安いコストで達成できることを示しました。指標としては学習が収束したか/しないか (発散・停滞の有無) と最終的な loss / perplexity (言語モデルの予測の良さ、小さいほど良い) が用いられ、momentum スケジュールの有無が成否を分けることが報告されています。

実務的に重要な定量的事実として、画像分類の代表的なベンチマーク (ImageNet 上の ResNet など) では、SGD with Momentum (μ=0.9) + 学習率スケジュール (cosine や step decay) + weight decay の組み合わせが、Adam 系より高い汎化性能 (テスト精度) を出すことが多数報告されてきました。ResNet-50 は ImageNet で top-1 約 76% 程度を SGD+Momentum で達成し、これが長らく標準の学習レシピでした。Wilson ら (2017) は CIFAR-10 の分類や文字レベル言語モデルなどで、Adam/RMSProp が訓練 loss を SGD より速く下げるのにテスト誤差では SGD+Momentum (や Nesterov 版) のほうが良い、という現象を定量的に示しています。

アブレーション的な観点 (どの要素を抜くとどうなるか): momentum を 0 にする (plain SGD) と細長い谷で振動が増え収束が遅くなる、または同じ学習率では発散しやすくなります。逆に μ を 0.99 など過大にすると、谷底をオーバーシュートして loss が振動・発散します。μ=0.9 が経験的な定番値で、これは `1/(1−μ)=10` 倍の実効的加速に対応します。weight decay を外すと汎化 (テスト精度) が落ち、学習率スケジュール (徐々に下げる) を外すと終盤の収束が甘くなり最終精度が下がる、というのも繰り返し観察される事実です。

指標の意味の補足: 「汎化性能 (テスト精度)」は、学習に使っていない未知データでの正解率で、実用上最も重要な指標です。SGD+Momentum が Adam より汎化が良い傾向は、SGD が見つける解 (loss 地形の「平らな谷 = flat minimum」) が、入力やパラメータの小さな摂動に対して loss があまり増えず、過学習しにくく未知データに強いためと説明されることが多いです。

### メリット・トレードオフ・限界

メリット:
- 細長い谷 (条件数の悪い loss 地形) でのジグザグ振動を抑え、収束が速くなる。
- 平坦な領域 (プラトー) でも貯めた速度で進み続けられ、停滞しにくい。
- ミニバッチ勾配のノイズを指数移動平均で平滑化し、安定する。
- 追加計算がほぼゼロ (足し算 1 回)、メモリも重みと同量の速度のみ。実装が単純で枯れている。
- 適切に調整すれば、しばしば Adam より汎化性能が高い (画像分類などで広く報告)。

トレードオフ・限界:
- momentum 係数 μ と学習率 η の 2 つを手で調整する必要があり、相性が悪いと発散・振動する。
- μ を大きくしすぎると慣性が効きすぎて谷底をオーバーシュートする。
- パラメータごとに学習率を自動調整しない (全パラメータ一律の η)。勾配の大きさが場所ごとに大きく違う問題 (Transformer の学習初期、スパースな埋め込み等) では Adam 系のほうが扱いやすい。
- 学習率を手動でスケジューリング (徐々に下げる) しないと最後の収束が甘くなりやすい。
- 研究上の論点: なぜ SGD の解が Adam より汎化するのかは「平坦な極小」「暗黙の正則化 (implicit regularization、最適化手法そのものが解にバイアスをかける現象)」といった観点から研究が続いており、完全には解明されていません。

### 発展トピック・研究の最前線

Momentum は現代の大規模学習でも生き続けています。LARS / LAMB (You et al. 2017, 2019) は Momentum/Adam に layer-wise の学習率スケーリング (層ごとに重みノルムと勾配ノルムの比でスケール) を足し、ImageNet を数千バッチ、BERT を巨大バッチで短時間学習できるようにしました。Lookahead optimizer (Zhang et al. 2019) は「速い内側のオプティマイザ + ゆっくり追従する外側の重み」という二重構造で安定性を高めます。近年は Lion (Chen et al. 2023) のように符号ベースの更新 (勾配の符号だけを使う) と momentum を組み合わせ、Adam の半分のオプティマイザ状態で同等以上の結果を狙う省メモリオプティマイザも登場しました。SAM (Sharpness-Aware Minimization, Foret et al. 2021) は「平坦な極小を明示的に探す」更新を Momentum 系の上に重ね、SGD の汎化の良さの源泉を理論と実装の両面から追っています。いずれも「過去の勾配を貯めて活用する」という Momentum の根本アイデアの上に立っており、本トピックはすべての勾配ベース最適化を理解する出発点です。

### さらに学ぶための関連トピック

- [Nesterov Accelerated Gradient](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AdamW (Decoupled Weight Decay)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Shampoo オプティマイザ

### ひとことで言うと
Shampoo は、ニューラルネットの重みを更新するルール(オプティマイザ)の一つです。重みが「行列(縦横の表)」の形をしていることに注目し、行方向と列方向それぞれの「クセ(統計)」を別々に学習して更新を調整する 2 次最適化手法です。前処理(preconditioning)と呼ばれる工夫で、SGD や Adam より少ないステップで谷底に近づくことを狙います。「行と列を別々に補正する」ことで、フル 2 次最適化の重さを避けつつ Adam の対角近似より賢く地形へ適応します。

### 直感的な理解
学習は損失(間違いの大きさ)が小さくなる方向に重みを少しずつ動かす作業です。普通の勾配降下(SGD)は全パラメータを同じ調子で動かします。ところが損失の地形は方向によってカーブの急さがまるで違います。ある方向は崖のように急で、別の方向は平原のように緩い。同じ歩幅で歩くと、急な方向では行き過ぎてジグザグし、緩い方向ではなかなか進みません。

理想を言えば、各方向のカーブに合わせて歩幅を調整したい。これを実現するのが「前処理(preconditioning)」です。勾配をある行列(前処理行列)で変換してから進むことで、地形の歪みを打ち消し、どの方向もまっすぐ谷底へ向かえるようにします。ニュートン法という古典的 2 次最適化はヘシアンの逆行列を前処理に使いますが、これはパラメータ数の二乗のサイズで、ニューラルネットでは巨大すぎて扱えません。

Adam はこれを「対角だけ」、つまり各パラメータを完全に独立扱いして近似することで回避しました。軽い代わりに、パラメータ間の相関(行列のクセ)は無視します。Shampoo の発想はその中間です。「重みはどうせ行列(あるいは多次元テンソル)なんだから、行と列それぞれの小さな前処理行列だけ持てば、フル行列ほど重くなく、対角より賢い近似ができる」というものです。Adam(対角)とフル前処理(N×N)の間を、構造を使って埋める、という位置づけです。

### 基礎: 前提となる概念

言葉を 1 つずつ噛み砕きます。

- **前処理(preconditioning)**: 勾配を行列 P で変換してから進む操作。`θ ← θ − η·P·g`。P がヘシアンの逆行列に近いほど、地形の歪みが消えてまっすぐ谷へ向かえます。
- **ヘシアン行列(Hessian)**: 損失の 2 回微分。各方向のカーブ具合を表します。前処理の理想形ですが、N×N(N=パラメータ数)で巨大すぎます。
- **対角近似(Adam)**: 前処理行列を対角(各パラメータ独立)だけに簡略化したもの。N 個の数で済む代わりにパラメータ間相関を捨てます。
- **クロネッカー積(Kronecker product)**: 小さい行列 2 つ(m×m と n×n)から大きい行列(mn×mn)を作る演算。逆に「大きい前処理行列を、行用の小行列と列用の小行列のクロネッカー積で近似する」ことで省メモリにできます。これが Shampoo の数学的核です。同じ発想を K-FAC はフィッシャー行列に対して使います。
- **行列の逆 p 乗根**: `L^(−1/4)` のような操作。固有値分解(行列を回転と伸縮に分解)を使い、各固有値を −1/4 乗して組み直します。Shampoo の計算上の難所です。
- **テンソル(tensor)**: 2 次元の行列を一般化した多次元配列。畳み込み層の重みは 4 次元テンソルです。Shampoo は各次元ごとに前処理行列を持つよう自然に一般化されます(名前の「テンソル最適化」はここから)。

### 仕組みを詳しく

例として、ある層の重みが `W` という m×n の行列(m 行 n 列)だとします。その勾配 `G` も同じ m×n の行列です。Shampoo は次の 2 つの小さな統計行列を蓄積します。

- `L`: 行方向の統計。`L += G · Gᵀ`(m×m の行列)。「どの行どうしが似た動きをするか」を表します。
- `R`: 列方向の統計。`R += Gᵀ · G`(n×n の行列)。「どの列どうしが似た動きをするか」を表します。

更新時に、勾配を左右からこれらの逆 4 乗根で挟みます。

```
W ← W − η · L^(−1/4) · G · R^(−1/4)
```

なぜ「−1/4 乗」かというと、左と右で 2 回掛けるので合わせて「−1/2 乗」、つまりおおよそ前処理行列の平方根の逆数に相当し、フルのヘシアン(正確には勾配外積の和=AdaGrad 行列)を `H^(−1/2)` でスケールする形をクロネッカー積で近似した形になるためです(前処理は分散の平方根の逆数を使うのが定石で、Adam の `1/sqrt(v)` も同じ思想です。Shampoo はその行列版とみなせます)。`L` が m×m、`R` が n×n なので保存量は `m² + n²` で済みます。

**規模感の数値例**: m=n=1000 の層なら、フル前処理は (10^6)² = 10^12 要素ですが、Shampoo は 1000² + 1000² = 2×10^6 要素。およそ 50 万分の 1 です。それでいて行・列の相関は捉えられます。Adam なら 10^6 要素(対角のみ)なので、Shampoo はその約 2 倍のメモリでフル前処理に近い表現力を得ます。

**計算上の難所**は `L^(−1/4)` のような逆 4 乗根です。固有値分解が必要で、m×m 行列に対して O(m³) かかります。毎ステップやると重いので、実装上は数十〜数百ステップに 1 回だけ前処理行列の逆ルートを更新するのが普通です(統計 L, R 自体は毎ステップ積み増す)。多次元テンソル(畳み込み層の 4 次元重みなど)の場合は、各次元 k ごとに前処理行列 `P_k` を持ち、対応する向きからテンソル縮約で挟みます。

**数値安定化**も欠かせません。`L`, `R` は学習初期に特異(逆行列が取れない)になりがちなので、小さな対角(`L + ε·I`)を足す **damping** を入れます。この ε の調整が実務では重要です。逆ルート計算自体も、固有値分解の代わりに行列の Newton 反復(後述の Muon の核)で近似する高速化が研究されています。

まとめると、Shampoo は「勾配を行と列の前処理行列で挟んで歪みを補正する」手法で、Adam の対角近似より構造を活かし、フル 2 次最適化よりずっと軽い、という位置にあります。

### 手法の系譜と主要論文

- **Martens & Grosse, "K-FAC" (ICML 2015)**: フィッシャー情報行列をクロネッカー積で近似する先駆。Shampoo はこの「クロネッカー因子化で 2 次情報を省メモリ化する」流れを汲みつつ、フィッシャーでなく勾配の外積(AdaGrad 系)を前処理に使う点が異なります。
- **Gupta, Koren, Singer, "Shampoo: Preconditioned Stochastic Tensor Optimization" (ICML 2018, arXiv:1802.09568, Google)**: クロネッカー因子分解による前処理を提案。動機は「テンソル形状を活かせばフル前処理を近似しつつ計算・メモリを現実的に抑えられる」こと。凸最適化での後悔(regret)上界の理論保証を与え、実験で SGD/AdaGrad より少ない反復で収束することを示しました。
- **Anil et al., "Scalable Second Order Optimization for Deep Learning" (2020, arXiv:2002.09018, Google)**: 元 Shampoo を実用化した「分散 Shampoo」。元の逆ルート計算がボトルネックだったため、その計算を CPU にオフロードして複数ステップで償却し、巨大な層はブロック分割しました。Transformer 翻訳・BERT・ImageNet などの実タスクで Adam より実時間で速い収束を達成。
- **Shi et al., "A Distributed Data-Parallel PyTorch Implementation of the Distributed Shampoo Optimizer" (2023, arXiv:2309.06497, Meta)**: PyTorch 向けの大規模実装と実践的チューニング指針。Meta の広告推薦モデルで Adam より高品質・高速と報告。前処理行列をワーカ間に分散して持つことで大規模でも回せるようにしました。
- **SOAP / Eigen 系の発展 (Vyas et al., 2024, arXiv:2409.11321)**: Shampoo を「前処理行列の固有空間で Adam を回す」と捉え直し、固有分解を疎にして高速化した発展形。Shampoo の系譜が今も活発に改良されていることを示します。

後発の [Muon オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) は「Shampoo の勾配直交化の効果を、固有値分解なしの Newton-Schulz 反復で安く近似する」という見方ができ、Shampoo の系譜の発展形として理解されています。対角ヘシアンを使う [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) とは「2 次情報を使う」点で同系統ですが、Shampoo は対角でなく行列構造を使う点が異なります。

### 論文の実験結果(定量データ)

主な数値を、指標の意味とともに説明します。

- **凸最適化(原論文 2018)**: 後悔上界という理論指標で AdaGrad と同等の保証を、より構造を使った前処理で達成。実験(ロジスティック回帰など)で AdaGrad/SGD より少ない反復で同じ損失に到達することを示しました。
- **機械翻訳 Transformer(分散 Shampoo 2020)**: WMT 英独翻訳の Transformer 学習で、Adam に比べて **目標品質(BLEU、翻訳の n-gram 一致率を測る指標で高いほど良い)に到達するまでのステップ数を約 40% 削減**したと報告。ステップ削減だけでなく、逆ルート計算を CPU オフロードで償却したことで **実時間(wall-clock)でも Adam より約 1.7 倍速い**ことを示した点が重要です。2 次最適化は「ステップは減るが 1 ステップが重くて結局遅い」という落とし穴があり、ここを実時間で逆転して見せたのが意義です。
- **BERT / ImageNet(分散 Shampoo 2020)**: BERT 事前学習や ResNet-50 ImageNet でも、同じ品質に達するステップ数を Adam/SGD より削減。Transformer 言語モデルでも前処理が効くことを示しました。
- **広告推薦(Meta 2023)**: 大規模なクリック率予測モデルで、Adam 比で精度(AUC・正規化エントロピーなど)向上と学習効率改善を報告。実運用規模で 2 次法が回ることを実証した点に意義があります。
- **AlgoPerf ベンチマーク(2023〜)**: MLCommons の最適化器コンペで Shampoo 系が外的チューニング部門で上位の結果を出し、ベンチマーク横断での競争力を示しました。

アブレーション・対比の要点:

- **前処理を対角に落とすと Adam と同等に退化**: 行・列の非対角な相関こそが Shampoo の効きどころです。
- **逆ルート更新の頻度**: 毎ステップ更新は重く、数十〜数百ステップに 1 回でも収束はほとんど劣化しません。償却が実用上の鍵です。
- **damping(ε)の感度**: 小さすぎると数値不安定、大きすぎると前処理が効かず Adam に近づく、という中間最適があります。
- **指数(1/4 乗 vs 1/2 乗)**: 前処理の指数を変えると Adam 寄り・ニュートン寄りに連続的に動き、タスクごとに最適点が異なります。

留意点として、優位の幅はモデル形状(行列が大きく構造を持つほど効きやすい)とタスクに依存し、常に Adam を上回るわけではありません。

### メリット・トレードオフ・限界

メリット

- 行・列の相関を捉える前処理で、対角近似(Adam)より地形に賢く適応でき、同じ品質に少ないステップで到達できる。
- フルの 2 次前処理に比べてメモリ・計算が桁違いに軽い(`m²+n²` 対 `(mn)²`)。
- 実装の工夫(CPU オフロード・償却)次第で、実時間でも Adam より速い収束の報告がある(分散 Shampoo)。
- テンソルへ自然に一般化でき、畳み込み・全結合の両方に適用できる。

トレードオフ・限界

- 前処理行列(L, R)のぶん、Adam よりメモリを食う。巨大層ではブロック分割が必要。
- 逆 4 乗根(固有値分解)が重く、周期的更新・数値安定化(damping)・ブロック分割の調整が必須。
- 実装が複雑で、分散環境では CPU/GPU 間のオフロード調整や前処理行列の分散保持が要る。チューニング項目が Adam より多い。
- 効果はモデル形状・タスクに依存し、小さい層や構造の薄いパラメータ(埋め込み・バイアス)では利点が出にくい。常に Adam を上回るとは限らない。
- ハイパーパラメータ(前処理更新周期・damping・学習率・ブロックサイズ)の相互作用が複雑で、再現に手間がかかるという未解決の実務課題がある。

### 発展トピック・研究の最前線

- **計算の高速化**: 固有値分解を避ける Newton-Schulz 反復(Muon)や、固有空間で Adam を回す SOAP など、「Shampoo の良さを保ったまま 1 ステップを軽くする」研究が活発です。Muon は近年の LLM 事前学習で AdamW を実用的に上回る報告が出ており、Shampoo 系が再注目されています。
- **2 次情報の粒度の最適化**: 対角([Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))/ クロネッカー因子(Shampoo, K-FAC)/ 直交化(Muon)のどれが計算コスト対効果で最良かは未解決のテーマです。
- **大規模 LLM への適用**: 数十億〜パラメータの言語モデル事前学習で、AdamW に対する優位を公平な条件で検証する取り組みが進んでいます。メモリ削減(前処理行列の量子化・低ランク化)が鍵になっています。
- **理論**: クロネッカー近似がヘシアン(あるいは勾配共分散)のどの成分を捨て、それが収束・汎化にどう効くかの解析が続いています。Shampoo がフル AdaGrad 行列の上界・下界で挟めるという解析も与えられています。

### さらに学ぶための関連トピック

- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [Muon オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Sophia オプティマイザ

### ひとことで言うと
Sophia は、ニューラルネットの学習を進めるときの「重みの更新ルール(オプティマイザ)」の一つです。定番の Adam よりも、損失地形のカーブの急さ(曲率)をざっくり見積もって使うことで、大規模言語モデル(LLM)の事前学習を約半分のステップ数で終わらせることを狙った 2 次最適化手法です。「2 次」とは、傾き(勾配)だけでなく傾きの変化の速さ(曲率)も使う、という意味です。Sophia = Second-order Clipped Stochastic Optimization の略です。

### 直感的な理解
ニューラルネットの学習は、目隠しをして山を下る作業にたとえられます。今いる地点の斜面の傾き(勾配)を測り、その逆方向に一歩進む。これを何百万回も繰り返して、損失(モデルの間違いの大きさ)が最小になる谷底を探します。

ここで問題になるのが「一歩の幅をどう決めるか」です。傾きの情報だけを使うと、急なカーブで行き過ぎてジグザグしたり、平らな谷でなかなか進まなかったりします。坂のカーブ具合(曲率)が分かれば、急な方向は小さく、緩い方向は大きく踏み込めるので、ずっと効率よく谷底に着けます。LLM の損失地形は方向ごとの曲率の差(条件数)が極端に大きく、ここが効くと期待されます。

LLM の事前学習(大量のテキストで言語の基礎を学ばせる工程)は、GPU を何千台も何週間も回す、極めて高価な作業です。学習ステップを半分にできれば、費用も時間もおおよそ半分になります。Sophia は「曲率の中でも一番安く取れる成分だけを使い、かつ暴走しないように頭打ち(クリップ)する」ことで、現実的なコストで 2 次最適化の旨味を取りに行く手法です。

### 基礎: 前提となる概念

言葉を 1 つずつ噛み砕きます。

- **勾配(gradient)**: 損失を各パラメータで 1 回微分した値。「損失が増える方向と量」を表すベクトルです。
- **1 次最適化(first-order)**: 勾配だけを使う手法。SGD や Adam がこれです。計算は軽いが地形のカーブが分かりません。
- **2 次最適化(second-order)**: 勾配に加えて曲率も使う手法。理論上は速い(ニュートン法は条件数によらず収束)が計算が重いのが難点でした。
- **ヘシアン行列(Hessian)**: 損失を 2 回微分して並べた表。各方向の坂のカーブ具合を表します。これが曲率の正体です。問題は大きさです。パラメータが N 個ならヘシアンは N×N で、LLM のように N が数十億〜数千億になると、保存も計算も不可能です。例えば N=10 億ならヘシアンは 10^18 要素という途方もない量になります。
- **対角成分**: ヘシアンの「各パラメータ自身のカーブだけ」を取り出したもの。パラメータ間の相互作用を捨てる代わりに、N 個の数で済みます。Sophia はこれだけを使います。
- **条件数(condition number)**: ヘシアンの最大固有値 ÷ 最小固有値。大きいほど地形が「細長い谷」で、勾配法はジグザグして遅くなります。2 次法はこれを補正するので強い、というのが理屈です。
- **クリッピング(clipping)**: 値が大きくなりすぎたら上限で頭打ちにすること。Sophia では更新量の各要素を上限 ρ で抑え、推定誤差による暴走を防ぎます。
- **Hessian-vector product(HVP)**: フルのヘシアンを作らずに「ヘシアン × あるベクトル」だけを計算する操作。勾配をもう一度微分する 1 回ぶんのコストで済むため、対角推定の道具として使えます。

### 仕組みを詳しく

Sophia の更新は、ざっくり次の 3 要素でできています。

1. 勾配の移動平均(momentum)を取る。Adam と同じで、過去の勾配を指数移動平均でならし、ノイズを減らした「進むべき方向」 `m` を作る。
2. ヘシアンの対角成分 `h` を、たまに(例えば 10 ステップに 1 回)安く推定し、これも移動平均でならす。
3. 更新量を `m / max(h, ε)`(方向 ÷ カーブ)で割り、さらに各要素ごとに上限 ρ で頭打ちにする。

数式で書くと、パラメータ θ の更新はおおむね次の形です(η は学習率、ρ はクリップ上限、ε は 0 割り防止の小さな値)。

```
m ← β1·m + (1−β1)·g            (勾配のモーメント)
h ← β2·h + (1−β2)·ĥ            (k ステップに1回だけ ĥ を推定して更新)
θ ← θ − η · clip( m / max(h, ε) , ρ )
```

ポイントは割り算 `m/h` です。カーブが急な方向(`h` が大きい)では更新を小さく、緩い方向(`h` が小さい)では更新を大きくします。地形に合わせて歩幅を自動調整するわけです。Adam の `m/sqrt(v)` と比べると、Adam は「勾配の二乗(=フィッシャー情報の対角)」で割るのに対し、Sophia は「本物のヘシアン対角」で割る点が違います。勾配二乗は曲率の粗い代用にすぎないので、本物を使えば精度が上がる、というのが動機です。

**clip が安定性の鍵** です。対角ヘシアンの推定は誤差を含み、損失が非凸(谷が一つでなくデコボコ)だと `h` が負になったりゼロに近づいたりして `m/h` が暴発します。そこで各要素ごとに更新量の絶対値を ρ で頭打ちにします。これにより「曲率推定がおかしい方向では、せいぜい符号付きの定数歩(普通の勾配降下くらいの安全な一歩)しか進まない」という保険がかかります。clip がかかった要素は実質「更新量 = η·ρ·sign(m)」になり、SignSGD のような振る舞いに退化して安全に止まります。論文では、典型的に更新要素の数十%がクリップされており、これが「悪条件の方向だけ慎重に、良条件の方向は素早く」を実現します。

ヘシアン対角の推定方法は論文で 2 種類示されています。

- **Hutchinson 推定(Sophia-H)**: ランダムなベクトル u(各要素 ±1 のラデマッハ分布)を使い、`u ⊙ (H u)` の期待値が対角ヘシアンになる性質(E[u ⊙ Hu] = diag(H))を利用します。`H u` は HVP 1 回で計算でき、フルのヘシアンを作らずに済みます。非凸では負値も出るため clip が必須です。
- **Gauss-Newton-Bartlett 推定(Sophia-G)**: 言語モデルの損失構造(クロスエントロピー)を利用し、モデルの出力分布からサンプリングした疑似ラベルで勾配を取ることで、必ず非負の対角ヘシアン近似(Gauss-Newton 行列の対角)を安く得ます。Bartlett の恒等式により期待値が曲率に一致します。常に正の曲率が得られるので `max(h, ε)` が安定し、論文の主推奨です。

数値イメージ: ある方向の `m=0.5`、推定 `h=4` なら更新は 0.5/4 = 0.125。別の方向で `h=0.1` なら 0.5/0.1 = 5 ですが、ρ=1 なら clip されて 1 になります。信頼できる急カーブ方向は素直に縮め、怪しい緩カーブ方向は上限で抑える、という挙動です。

**メモリは Adam とほぼ同じ** です。Adam が「勾配の 1 次・2 次モーメント」の 2 つをパラメータごとに持つのに対し、Sophia は「勾配モーメント `m`」と「ヘシアン対角 `h`」の 2 つを持ちます。1 パラメータあたりの追加状態量は同じなので、Adam が動く環境ならそのまま動きます。計算の追加は HVP(あるいは疑似ラベル勾配)が数ステップに 1 回(論文では k=10)だけで、全体への増分は 5% 程度と報告されています。

### 手法の系譜と主要論文

- **Kingma & Ba, "Adam" (ICLR 2015)**: 勾配の 2 次モーメント(自乗平均)で各パラメータの学習率を適応させる 1 次手法。Sophia は「Adam が使う勾配自乗はヘシアンの粗い代用に過ぎず、本物の曲率を使えばもっと速いはず」という問題意識の上に立っています。
- **Martens & Grosse, "K-FAC" (ICML 2015)**: フィッシャー情報行列をブロック(クロネッカー因子)近似して 2 次最適化する手法。Sophia は K-FAC のような行列近似よりさらに軽い「対角のみ」を選び、スケーラビリティを優先しました。
- **Yao et al., "AdaHessian" (AAAI 2021)**: 同じく Hutchinson 法で対角ヘシアンを使う適応最適化。Sophia は AdaHessian に対し、要素ごとのクリッピングと言語モデル特化の Gauss-Newton-Bartlett 推定を加えた点が新しいといえます。
- **Chen et al., "Lion" (2023, arXiv:2302.06675)**: 探索で見つかった符号ベースの軽量オプティマイザ。Sophia の比較対象として登場し、Sophia は Lion・AdamW の双方を上回る効率を主張しました。
- **Liu, Li, Hall, Liang, Ma, "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training" (2023, arXiv:2305.14342, Stanford)**: 上記の対角ヘシアン + 要素ごとクリップを提案。GPT-2 規模で AdamW のおよそ半分のステップ・実時間・計算量で同じ損失に到達したと報告し、非凸の滑らかな目的に対する収束率(条件数に依存しないオーダ)も解析しました。

行列構造を使う [Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) や、その近似である Muon([Muon オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))とは「2 次情報を使う」点で同じ系譜にありますが、Sophia は対角に絞って軽さを優先した点が特徴です。

### 論文の実験結果(定量データ)

評価の中心は **GPT-2 規模のデコーダ型言語モデル**(125M / 355M / 770M パラメータ)を OpenWebText で事前学習する設定です。比較対象は AdamW と Lion。

主な指標と意味:

- **検証損失(validation loss)/ 言語モデルの perplexity**: テキストの次トークンをどれだけ正確に予測できるかの尺度。低いほど良いモデルです。事前学習では「ある目標損失に到達するまでに何ステップ・何 FLOPs かかったか」で速さを比べます。1 ステップごとの計算量がほぼ同じなので、ステップ削減はそのまま計算削減になります。

報告された主な結果:

- 同じ検証損失に到達するまでの **ステップ数・実時間(wall-clock)・総計算量(FLOPs)を、AdamW のおよそ 0.5 倍(半分)** に削減。例えば AdamW が 100K ステップで届く損失に、Sophia は約 50K ステップで届く、というイメージです。355M モデルでは AdamW の 750K ステップ相当の損失に Sophia が約 400K ステップで到達したと報告されています。
- 同じステップ数で比べると、Sophia は AdamW より一貫して低い検証損失を出します。
- モデルサイズが大きいほど(125M → 770M)、Sophia の相対的な優位が広がる傾向が報告されています。スケールに伴って効果が薄れないことは、LLM 用途として重要な性質です。
- Lion(別の符号ベース最適化)との比較でも、同等以上の効率を示したとしています。

アブレーション・対比の要点:

- **clip を外すと不安定化**: 要素ごとクリッピングを取り除くと、非凸領域で `m/h` が暴れて学習が壊れやすくなります。clip が Sophia の安定性の本体であることを示します。
- **ヘシアン推定間隔 k**: 数ステップに 1 回(k=10)でなく毎ステップ推定しても効率はほとんど変わらず、計算だけ増えます。よって間隔を空けるのが効率上の要点です。
- **ρ の感度**: クリップ上限 ρ が大きすぎると Adam に近づき優位が消え、小さすぎると更新が SignSGD に退化して進みが鈍る、という中間最適があります。
- **Sophia-G ≥ Sophia-H**: Gauss-Newton-Bartlett(常に非負)の方が Hutchinson(負値が出る)よりわずかに安定して良いと報告。

留意点として、これらは主に **数億パラメータ規模・デコーダ型 LLM** での結果です。より大規模(数十億〜)での優位や、他アーキテクチャ・他タスクでの再現性は追試で議論があり、「Adam を常に上回る」とは言い切れない、というのが現状の慎重な見方です。コミュニティの一部追試では、学習率を十分にチューニングした AdamW と大差ないという報告もあります。

### メリット・トレードオフ・限界

メリット

- 同じ損失に到達するまでのステップ数・実時間・計算量を Adam のおよそ半分にできる(論文報告、GPT-2 規模)。
- メモリは Adam と同等(状態は 2 つ)。Adam が動く環境ならそのまま動く。
- 要素ごとクリッピングにより、非凸でも発散しにくい。学習率スケジュールへの感度が Adam より低いとも報告される。
- モデルが大きいほど効きが鈍らない(スケール耐性)。

トレードオフ・限界

- ヘシアン対角推定のために数ステップに 1 回、追加計算(HVP または疑似ラベル勾配)が必要。増分は小さいが 0 ではない。HVP は実装(二重逆伝播)がフレームワーク依存で手間がかかる。
- 新しいハイパーパラメータ(ρ、推定間隔 k、推定法、β2)が増え、調整の手間がかかる。
- 大規模・他タスクでの再現性や優位性はまだ議論の途中で、万能ではない。コミュニティの追試では「条件次第で AdamW と大差ない」報告もある。
- 主にデコーダ型言語モデルで検証されており、視覚モデルや構造の異なるネットワークでの効果は別途検証が要る。

### 発展トピック・研究の最前線

- **2 次情報の取り方の選択**: 対角(Sophia)/ クロネッカー因子(K-FAC, Shampoo)/ 行列直交化(Muon)のうち、どの粒度が「計算コスト対効果」で最良かは未解決の研究テーマです。粒度を上げると速くなるが重くなる、というトレードオフの最前線にあります。
- **理論的理解**: clip 付き 2 次更新がなぜ非凸で安定して速いのかの解析、対角近似が捨てているオフ対角項の影響の評価などが進んでいます。論文は条件数に依存しない更新量の上界を示し、悪条件の谷でも一定の進みを保証しました。
- **大規模再現性**: 公開された大規模事前学習(数十億〜)での Adam 系との比較は活発な検証対象で、ベンチマーク条件(学習率調整・データ・系列長・トークン数)を揃えた公平比較の整備が課題です。
- **他手法との融合**: 信頼比([LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))や行列前処理([Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))と組み合わせる試みもあり、最適化器の設計空間はまだ広がっています。

### さらに学ぶための関連トピック

- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/10_losses)
- [Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Muon オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LAMB](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)


## Step / MultiStep LR Decay

### ひとことで言うと
学習率(ニューラルネットのパラメータを1回どれくらい動かすかを決める数値)を、決められたタイミングが来るたびに「階段状」にガクッと下げるやり方です。たとえば「30エポックごとに学習率を10分の1にする」のように、ある区間は同じ学習率を保ち、節目で一気に小さくします。最も古くからある、最もシンプルな学習率スケジューラ(学習率を時間とともに変化させる仕組み)で、深層学習の黎明期から画像分類の定番レシピとして使われ、いまも「確実なベースライン」として現役です。

### 直感的な理解
山を下って一番低い谷を探す状況を想像してください。最初は遠くにいるので、大股(大きな学習率)でずんずん進みます。谷の近くまで来たら、大股のままだと谷を飛び越えて反対側に登ってしまうので、歩幅を小さくして慎重に降ります。さらに底のすぐ近くでは、もっと細かく歩幅を縮めて微調整します。

Step decay はこの「歩幅を段階的に縮める」を最も素朴に実現したものです。「ある程度進んで頭打ちになったら、歩幅を一気に10分の1にする」。それだけです。なめらかに少しずつ縮めるのではなく、しばらく同じ歩幅で進んで、節目でドンと縮める。階段のような形になります。

なぜ「しばらく同じ歩幅を保つ」のかというと、ある学習率である程度学習が進むと損失の改善が頭打ちになり、そこで学習率を下げると改善が再び進む、という現象が経験的に知られているからです。学習率を下げる前後で損失曲線が「階段状にカクッと下がる」のがよく観察され、これが Step decay の有効性を直感的に裏づけています。なぜ下げると再び進むのかは「探索と収束」の言葉で説明されます。高い学習率では SGD のノイズで谷の底を行ったり来たりしているだけ(これ以上深く沈めない)で、学習率を下げるとそのノイズが小さくなり、より底へ沈み込めるためだと解釈されています。

### 基礎: 前提となる概念
いくつかの用語を噛み砕きます。

- エポック(epoch): 学習データ全体を1回すべて使い切ること。10エポックなら、全データを10周することです。
- イテレーション(ステップ): パラメータを1回更新すること。1エポックは「データ数 ÷ バッチサイズ」回のイテレーションからなります。
- 検証誤差(validation error): 学習に使っていない別のデータでの間違い率。学習データだけ良くて未知データで悪い「過学習」を見抜くために使います。
- 頭打ち(plateau): 損失や誤差がそれ以上下がらず、平らになった状態。学習率を下げる合図になります。
- ハイパーパラメータ: 学習前に人間が決める設定値。学習率、バッチサイズ、エポック数など。学習で自動調整される「パラメータ(重み)」とは区別します。
- 有効学習率(effective learning rate): SGD のノイズの大きさは大まかに「学習率 ÷ バッチサイズ」に比例すると議論されます(Smith & Le 2018)。学習率を下げることは、このノイズを減らして探索から収束へ切り替えることに相当します。Step decay の各段差は、このノイズ水準を一段ずつ落とす操作だと理解できます。

学習率減衰(learning rate decay)とは、学習が進むにつれて学習率を小さくしていく一般的な考え方です。理由は、学習初期はパラメータが正解から遠いので大きく動かすほうが速く近づけるのに対し、終盤は正解のすぐ近くなので、大きいままだと正解を飛び越えて振動し、細かく合わせられないからです。Step decay はこの減衰を「階段」で行う最古の方式です。

### 仕組みを詳しく
Step decay(等間隔版)は、一定エポックごとに学習率を一定倍率で下げます。

```
lr(epoch) = lr_init * gamma ^ floor(epoch / step_size)
```

各記号の意味です。

- `lr_init`: 学習開始時の学習率。
- `gamma`(ガンマ): 1回下げるときの倍率。0.1 なら10分の1、0.5 なら半分。1未満の正の数。
- `step_size`: 何エポックごとに下げるか。
- `floor(...)`: 小数点以下を切り捨てる操作(整数にする)。

数値例。`lr_init = 0.1`、`gamma = 0.1`、`step_size = 30` なら、

- エポック0〜29: 0.1
- エポック30〜59: 0.01
- エポック60〜89: 0.001
- エポック90〜: 0.0001

MultiStep decay は、下げるタイミングを等間隔ではなく「自分で指定したエポックのリスト」で決める版です。たとえば `milestones = [60, 120, 160]`、`gamma = 0.2` なら、

- エポック0〜59: 初期値
- エポック60で 0.2 倍
- エポック120でさらに 0.2 倍
- エポック160でさらに 0.2 倍

学習率の時間変化を ASCII 図にすると、まさに階段です。

```
lr
0.1  |‾‾‾‾‾‾‾|
     |       |
0.01 |       |‾‾‾‾‾‾‾|
     |       |       |
0.001|       |       |‾‾‾‾‾‾‾
     +-------+-------+--------> epoch
        30      60       90
```

各エポック(またはステップ)の終わりにスケジューラを1回進めると内部カウンタが進み、節目に達した瞬間に学習率が gamma 倍されます。多くのフレームワークに `StepLR` / `MultiStepLR` 相当の実装があります。

重要な性質として、節目以外の区間では学習率が完全に一定です。これはコサインや指数減衰(なめらかに連続的に下げる方式)との大きな違いで、「区間内は同じ条件で安定して学習し、節目でだけ条件を変える」という設計です。区間内は挙動が読みやすく、デバッグもしやすい一方で、節目での不連続なジャンプが弱点になります。

設計上の経験則として、(1) 学習が頭打ちになったあたりに milestone を置く、(2) gamma は 0.1(10分の1)が伝統的だが 0.2〜0.5 のほうがなめらかで安定することもある、(3) 終盤に十分小さい学習率で詰める区間を確保する、が知られています。milestone を早く置きすぎると学習が早く止まり、遅すぎると探索が長引いて時間を無駄にします。

### 手法の系譜と主要論文
Step decay は特定の論文が発明したというより、深層学習の実務の中で標準化していった手法です。系譜は次のようになります。

1. Krizhevsky, Sutskever & Hinton, "ImageNet Classification with Deep Convolutional Neural Networks", NIPS 2012(AlexNet)。深層学習ブームの火付け役。学習率 0.01 から始め、「検証誤差が改善しなくなったら学習率を手動で10分の1に下げる」という Step decay の手動版を使い、学習中に計3回下げました。動機はシンプルさと実効性です。これにより人手依存ながら、当時としては画期的な ImageNet の精度(top-5 誤り率およそ 15〜16%、2位に約 10 ポイント差)を達成しました。手動なので再現性・自動化に難がありました。

2. He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition", CVPR 2016(ResNet, arXiv:1512.03385)。論文の主役は残差接続(skip connection、入力を出力に足し込む配線で、層を非常に深くしても学習できるようにした仕組み)ですが、学習レシピとして「lr=0.1 開始、誤差が頭打ちになるたびに ×0.1」という Step decay を採用しました(CIFAR では 32k/48k 反復で ×0.1 など)。理由は、当時もっとも信頼でき、ハイパーパラメータが少なく再現しやすかったからです。152層という当時極端に深いネットを安定して学習でき、「Step decay = 画像分類の標準レシピ」という認識を広めました。

3. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017(arXiv:1706.02677)。大きなバッチサイズでの分散学習で、Step decay の節目(典型的には 30/60/80 エポックで ×0.1)に加えて「最初の数エポックは学習率を徐々に上げる warmup」を組み合わせました。動機は、学習初期に大きな学習率をいきなり使うとモデルが壊れるためです。これにより 256 規模の GPU でも ResNet-50 を安定学習でき、warmup + step decay が大規模学習の定番になりました。

4. Smith & Le, "Don't Decay the Learning Rate, Increase the Batch Size"(arXiv:1711.00489, ICLR 2018)。Step decay の各段差は「学習率を下げる」代わりに「バッチサイズを上げる」ことでも代替できると示しました。両者はともに SGD のノイズ水準(およそ lr/batch)を下げる操作だからです。これは Step decay の本質が「学習率という数値」ではなく「ノイズ水準の段階的低下」にあることを明らかにした重要な視点です。

その後、コサイン減衰や linear 減衰などなめらかな方式が台頭し、Transformer 系の学習では Step decay は主役の座を譲りました。しかし ResNet 系 CNN の再現実験や、確実なベースライン、物体検出(Faster R-CNN / Mask R-CNN の標準スケジュール、いわゆる 1x/2x スケジュールでの step 減衰)では今も広く使われ続けています。

### 論文の実験結果(定量データ)
ResNet 論文の中心的な検証は ImageNet(約 120 万枚・1000 クラスの大規模画像分類データセット)と CIFAR-10 です。評価指標は top-1 / top-5 誤り率(モデルの第1候補・上位5候補に正解が含まれない割合。低いほど良い)です。

報告された代表的な数値です。

- ImageNet で ResNet-152(152層)は top-5 誤り率およそ 4.49%(single model)、アンサンブルでおよそ 3.57% を達成し、ILSVRC 2015 で優勝しました。学習レシピは lr=0.1 開始の step decay(誤差頭打ちで ×0.1)です。
- 残差接続のアブレーションが重要です。同じ Step decay を使っても、残差接続なしの「plain net(ただ層を重ねただけのネット)」は層を深くするほど誤差が悪化(34層が18層より悪い)しました。残差接続を入れると深くするほど誤差が改善し、この対比が ResNet の核心です。つまり Step decay 自体は単なる学習レシピで、性能向上の主因は残差接続でした。
- Goyal らの大規模学習では、warmup + step decay により、バッチサイズ 256 と 8192 でほぼ同じ最終精度(ResNet-50 で top-1 誤り率およそ 23.7%、top-1 精度でおよそ 76%)を達成しました。warmup を抜くと大バッチで精度が崩れる、というアブレーションが warmup の必要性を示しています。
- なめらかな方式との比較研究(各種ベンチマーク)では、同条件のコサイン減衰のほうが Step decay より最終精度がわずかに高い(画像分類でおよそ 0.3〜0.8 ポイント)という報告が多い一方、その差は小さく、Step decay の再現性とシンプルさが選好される場面も多くあります。

これらの数値が示す意味として、Step decay は「派手な性能向上をもたらす手法」ではなく「確実で再現性のある土台」だという点が読み取れます。性能の主役はモデル構造(残差接続)や warmup であり、Step decay はそれらを安定して支える役割です。

### メリット・トレードオフ・限界
メリット

- 仕組みが極めて単純で、実装も理解も容易。
- 区間内は学習率一定なので挙動が読みやすく、デバッグしやすい。学習曲線のどの段差がどの milestone に対応するか一目でわかる。
- ハイパーパラメータが少ない(step_size または milestones と gamma だけ)。
- 多くの画像分類・物体検出タスクで実績があり、まず試す基準(ベースライン)として確実。

トレードオフと限界

- 下げるタイミングを人間が決める必要があり、データ・モデル・バッチサイズが変わると最適な milestones も変わります。自動では適応しません(検証指標を見て適応する手法と対照的)。
- 節目で学習率が不連続にガクッと変わるため、その直後に損失が一瞬不安定になることがあります。
- なめらかに下げる方式(コサイン等)に比べ、最終精度がわずかに劣る報告が多い。
- 終盤まで「区間ごとに学習率が残る」ため、終わり際の超低学習率での詰めはコサイン減衰ほど滑らかではありません。
- 局所解・鞍点から抜け出す能力はありません。単調に下げるだけです。
- 未解決の論点として「なぜ学習率を下げた瞬間に損失が階段状に下がるのか」の理論的説明は完全には確立しておらず、損失地形のシャープな谷への移行や、有効学習率と探索ノイズの関係から説明する研究が続いています。Smith & Le の「ノイズ水準の低下」説はその有力な解釈の一つです。

### 発展トピック・研究の最前線
- warmup の標準化: 大バッチ学習や Transformer では、学習開始直後に学習率を線形に上げる warmup がほぼ必須になりました。Step decay も warmup と組み合わせて使うのが現代的な作法です。
- なめらかな減衰への移行: コサイン減衰、linear 減衰、polynomial 減衰など、不連続のない方式が主流になりつつあります。Step decay はその比較対象(ベースライン)としての役割が大きくなっています。
- 線形スケーリング則(Goyal et al.): バッチサイズを k 倍にしたら学習率も k 倍にする、という経験則。Step decay の初期学習率の設定に直結します。
- バッチサイズ増加による代替(Smith & Le 2018): 学習率を下げる代わりにバッチサイズを上げると、同じ汎化性能をより少ない更新回数で得られると示され、Step decay の本質理解を深めました。
- 適応的減衰: 検証指標を監視して頭打ちを自動検出し学習率を下げる手法は、Step decay の「人手で節目を決める」部分を自動化したものと位置づけられます([ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) を参照)。
- WSD(Warmup-Stable-Decay): 大規模言語モデルの学習で、長く一定にして最後だけ急減衰させるスケジュールが登場し、ある意味で「2段階の Step decay」に近い設計が再評価されています。

### さらに学ぶための関連トピック
- [Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Linear LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Stochastic Depth (DropPath)

### ひとことで言うと
Stochastic Depth は、非常に深いニューラルネット(層が何十・何百と積み重なったもの)の学習中に、層をまるごとランダムにスキップする正則化手法です。Dropout が「ニューロン1個」を休ませるのに対し、これは「層1枚(ブロック)」を丸ごと休ませます。これによって学習が速く安定し、深いモデルの汎化が改善します。Vision Transformer や ConvNeXt など現代の大規模画像モデルでは DropPath という名前で標準装備されている、欠かせない技術です。

### 直感的な理解
本を100冊読んで知識を身につけるとき、毎回かならず全100冊を順番に読み返すのは大変です。今日はそのうちランダムに70冊だけ、明日は別の70冊だけ読む、というやり方なら、一回あたりの負担が軽くなり、しかも特定の本に過剰に依存しない柔軟な知識が身につきます。試験本番では全100冊の知識を総動員すればよいのです。

深いニューラルネットでも同じ発想が使えます。学習中は層をランダムにスキップして、実質的に浅いネットワークとして学ぶ。すると一回の学習が軽く速くなり、各層が「他の層がいなくても、ある程度は役に立つ」汎用的な特徴を学ぶようになります。推論(本番)では全層を使うので、深いネットワークのフルの表現力が手に入ります。「深く訓練しつつ、実効的には浅く学ぶ」という、いいとこ取りを狙う手法です。

この発想の背景には「深い残差ネットワークは、長さの異なる無数の浅い経路の集まり(暗黙のアンサンブル)として振る舞う」という観察があります。であれば、学習時にあえて経路を間引いてもネットワークは壊れず、むしろ各経路が独立に役立つよう鍛えられる、というのが直感です。

### 基礎: 前提となる概念
- 層 (layer) / ブロック (block): ネットワークを構成する処理のひとかたまりです。残差ネットワーク(ResNet)なら「畳み込み + 正規化 + 活性化 + 足し込み」のブロックが何十個も直列に並びます。Transformer なら「自己注意 + 全結合」のブロックが繰り返されます。
- 残差接続 (residual connection / skip connection): 各ブロックの出力を `出力 = 入力 + ブロックの処理(入力)` という形にする仕組みです。「入力をそのまま足す」経路(ショートカット)があるおかげで、深いネットでも勾配(学習の信号)が奥まで届きやすくなります。残差接続の導入が深層化のブレークスルーになりました。
- 勾配消失 (vanishing gradient): 層が深いと、学習信号が手前の層に伝わる途中で小さくなりすぎて、手前の層がほとんど学習できなくなる問題です。残差接続はこれを緩和しますが、完全には解決しません。
- アンサンブル (ensemble): 複数モデルの予測を平均して汎化を上げる手法です。残差ネットワークは「多数の浅い経路の暗黙のアンサンブル」と解釈でき、Stochastic Depth はこの解釈と整合します。
- 期待値の保存: Dropout と同じく、学習時にランダムにスキップするため、推論時と出力の大きさをそろえるスケール調整が必要になります。

### 仕組みを詳しく
残差接続を持つブロックは通常こう計算します。

```
出力 = 入力 + f(入力)      # f はそのブロックの処理(畳み込みや Attention)
```

Stochastic Depth では、学習時にブロックごとに確率 `p_drop`(ドロップ確率)でこの `f` を丸ごとスキップします。

- スキップする場合: `出力 = 入力`(ブロックは何もせず、入力がそのまま通過)
- スキップしない場合: `出力 = 入力 + f(入力)`

残差接続があるおかげで、スキップしても情報が途切れず入力がそのまま流れるのが肝心な点です。これが「残差接続を持つアーキテクチャでしか使えない」理由です。通常の(残差なしの)直列ネットでは、層をスキップすると信号が途切れて壊れてしまいます。

数値例を見ます。ある特徴ベクトルが `入力 = [1.0, 2.0, 3.0]`、ブロックの処理結果が `f(入力) = [0.1, -0.2, 0.5]` のとき:

- スキップされなければ → `[1.1, 1.8, 3.5]`
- スキップされれば → `[1.0, 2.0, 3.0]`(そのまま通過)

推論時のスケーリングは Dropout と同じ考え方です。推論時は全ブロックを使いますが、学習時はブロックが確率 `1 - p_drop` でしか寄与していません。期待値を合わせるため、推論時はブロック出力 `f` に `(1 - p_drop)` を掛けるか、学習時に生存ブロックを `1/(1-p_drop)` 倍するスケーリングを行います。これで学習時と推論時の出力スケールが一致します。

深さに応じて確率を変える線形減衰ルール(linear decay rule)が、原論文の重要な工夫です。全ブロックを同じ確率でドロップするのではなく、奥の層ほどドロップ確率を高くします。手前の層(エッジや色など低レベルの基本特徴を学ぶ)はめったにスキップせず、奥の層(高レベルで冗長になりやすい)を積極的にスキップします。たとえば最初のブロックは `p_drop = 0`、最後のブロックは `p_drop = 0.5`、その間を線形に増やす、という設定です。第 `l` ブロック(全 `L` ブロック)のドロップ率は `p_drop(l) = (l / L) · p_L` と表され、`p_L` は最終層のドロップ率です。

DropPath という呼び名について補足します。Vision Transformer 系の実装では同じ仕組みが「DropPath」という名前で、`drop_path_rate` という引数で提供されています。「経路(path)を落とす」という意味で、Stochastic Depth と実質同義です。現代の標準実装では、バッチ全体ではなくサンプル単位(バッチ内の各データごと)に独立してドロップします。これにより、同じバッチの中でも一部のサンプルはブロックを通り、一部は通らない、というきめ細かいランダム化が実現されます。サンプル単位 DropPath は、ドロップされなかったサンプルの残差出力を `1/(1-p_drop)` 倍にスケールして期待値を保ちます。

計算量の観点では、スキップされたブロックは順伝播・逆伝播の計算自体を省ける実装もあり、その場合は学習時間の短縮にもつながります。ドロップ率が平均0.25なら、おおまかに言って学習中は平均的に層の75%だけが計算される計算量になります(ただしサンプル単位 DropPath ではバッチ内に1つでも生存サンプルがあればそのブロックは計算されるため、実際の短縮はブロック単位スキップより小さくなります)。

### 手法の系譜と主要論文
- He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition", CVPR 2016: 残差接続を導入した ResNet です。Stochastic Depth が成立するための土台で、152層という当時としては極端に深いネットの学習を可能にしました。
- Huang, Sun, Liu, Sedra, Weinberger, "Deep Networks with Stochastic Depth", ECCV 2016: 本手法の原典です。学習時に残差ブロックを確率的にスキップし、奥ほどドロップ率を上げる線形減衰スケジュールを導入しました。深いネットの学習を速く安定させ、暗黙のアンサンブルで汎化を上げることを狙い、CIFAR で1202層という超深層 ResNet の学習に成功しました。
- Veit, Wilber, Belongie, "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", NeurIPS 2016: 残差ネットが「長さの異なる多数の経路の集まり」として振る舞い、推論時に一部のブロックを取り除いても性能が大きく崩れないことを実証しました。Stochastic Depth がなぜ機能するのかの理論的な裏付けとして重要です。
- Touvron, Cord, Douze, Massa, Sablayrolles, Jégou, "Training data-efficient image transformers (DeiT)", ICML 2021: Vision Transformer を ImageNet だけで効率よく訓練するレシピを確立し、DropPath を主要な正則化の一つとして採用しました。これ以降、Transformer 系で DropPath は標準になりました。
- Liu, Lin, Cao, Hu, Wei, Zhang, Lin, Guo, "Swin Transformer", ICCV 2021: 階層的な Vision Transformer で、モデル規模に応じて `drop_path_rate` を調整するのが定石になりました。
- Liu, Mao, Wu, Feichtenhofer, Darrell, Xie, "A ConvNet for the 2020s (ConvNeXt)", CVPR 2022: 現代的な CNN にも DropPath を取り入れ、Transformer に匹敵する性能を達成しました。

これらの論文では、モデル規模が大きいほど `drop_path_rate` を高く設定する(tiny は 0.1〜0.2、base は 0.3〜0.4、large は 0.4〜0.5 程度)のが経験則として確立しています。理由は、大きいモデルほど表現力が過剰で過学習しやすく、強い正則化が必要になるためです。

### 論文の実験結果(定量データ)
ECCV 2016 の原論文の主要結果を、指標の意味とともに紹介します。基本指標はテスト誤り率(test error)で、小さいほど良い値です。

- CIFAR-10(自然画像、10クラス): 110層 ResNet で、通常学習のテスト誤りがおよそ 6.41% だったのに対し、Stochastic Depth を適用するとおよそ 5.23% まで低下したと報告されています。誤りが約2割減ったことになります。
- CIFAR-100(100クラス、より難しい): 110層 ResNet で約 27.76% から約 24.58% へ改善したと報告されています。
- SVHN(街頭の数字): 152層 ResNet で約 1.80% から 1.75% 前後へ、すでに低い誤りをさらに削ったと報告されています。
- 超深層の実証: 通常の1202層 ResNet は過学習で性能が頭打ちになる(110層より悪化する)のに対し、Stochastic Depth を使った1202層 ResNet は CIFAR-10 でさらに誤りを下げ(約 4.91%)、深さを増やすほど性能が上がるという「深さの恩恵」を引き出せることを示しました。これは Stochastic Depth の最も象徴的な結果です。
- 学習時間: ブロックをスキップする分の計算を省けるため、110層 ResNet で学習時間が約25%短縮したと報告されています。

アブレーションでは、線形減衰スケジュールが一様なドロップ率より優れることが示されています。すべての層を同じ確率で落とすと、手前の低レベル特徴まで頻繁に失われて学習が不安定になりますが、奥ほど強く落とす線形減衰なら、基礎を保ちつつ冗長な高レベル層だけを正則化できるためです。また、勾配の伝わり方を可視化した分析では、Stochastic Depth により学習中の有効な勾配が手前の層まで届きやすくなる(勾配のスケールが深さによらず保たれやすくなる)ことが確認されています。最終層のドロップ率 `p_L` についても、0.5 付近が CIFAR で安定して良く、極端に高くすると未学習に傾くことが示されています。

### メリット・トレードオフ・限界
メリット:
- 深いネットワークの学習を速く・安定させます(実効的に浅く学ぶため勾配が届きやすい)。
- スキップされたブロックの計算を省ける実装では学習時間が短縮します。
- 画像系深層モデル(ResNet, ViT, Swin, ConvNeXt)で Dropout より汎化に効くことが多いです。
- 推論時は全層を使うのでフルの表現力を保てます。

トレードオフ・限界:
- 残差接続を持つアーキテクチャでしか使えません(入力がそのまま通る経路が必要)。
- ドロップ率と深さスケジュールの調整が必要で、強すぎると未学習、弱すぎると効果が薄くなります。
- 推論時のスケーリング調整を正しく実装しないと、学習時と出力の大きさがずれます。
- 浅いネットワークでは効果が薄くなります(そもそもスキップする層が少ない)。
- サンプル単位 DropPath では、バッチ内に1つでも生存サンプルがあればそのブロックは計算されるため、計算削減効果はブロック単位スキップより小さくなります。

研究上の未解決課題としては、最適な `drop_path_rate` をモデル規模・データ量から自動的に決める原理がまだ確立しておらず経験則に頼っている点、そして Stochastic Depth がバッチ正規化や層正規化(Layer Normalization)の統計推定にどう影響するかの理論的理解が不十分な点が挙げられます。

### 発展トピック・研究の最前線
Stochastic Depth は、より一般的な「構造のランダム化による正則化」という考え方の一例です。学習中にネットワークの幅・深さ・解像度をランダムに変える試み(例: スーパーネット学習やニューラルアーキテクチャ探索)につながっています。また、推論時にあえて層をスキップして計算量を削る早期終了(early exit)や動的推論(dynamic inference)とも発想を共有しており、エッジデバイスでの効率的な推論研究に波及しています。大規模 Transformer の学習レシピでは、DropPath とデータ拡張([Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) / [CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))と重み減衰([Weight Decay (L2 正則化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))を組み合わせた総合的な正則化設計が標準で、その最適な組み合わせの探索が続いています。最近では、層ごとの寄与度を学習しながら不要な層を恒久的に枝刈り(pruning)する手法とも結びつき、「学習時のランダムスキップ」が「推論時の構造圧縮」へ橋渡しされる方向の研究も進んでいます。

### さらに学ぶための関連トピック
- [Dropout](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Weight Decay (L2 正則化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [重みの指数移動平均 (EMA of Weights)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Mixup](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)


## Stop-gradient (勾配停止)

### ひとことで言うと
Stop-gradient(勾配停止)とは、ニューラルネットワークの計算経路のある場所に「ここから先は学習で更新しない」という壁を立て、勾配(重みをどう直すべきかを示す修正情報)がそこから逆流しないように遮断する操作です。順方向の計算では値はそのまま通りますが、逆方向では止まります。一見ささいなこの操作が、2つの枝(branch)で同じ画像の別ビューを処理して出力を近づけるタイプの自己教師あり学習において、「全部同じ答えを出せば損失がゼロになる」という崩壊(collapse)を防ぐ、地味だが決定的な仕掛けになります。

### 直感的な理解
2人で「同じ絵を見て、なるべく似た感想を述べよ」というゲームをすると想像してください。2人とも本気で絵を観察すれば良い感想が揃いますが、最も楽に「完全一致」を達成する方法は、絵を一切見ずに2人とも「うー」とだけ言うことです。これが崩壊です。損失(2人の感想のずれ)はゼロになりますが、感想は絵の中身を何も表していません。

ここで片方の人を「先生」役にして「先生は感想を変えてはいけない。生徒だけが先生に合わせる」という非対称ルールにすると、2人で示し合わせて「うー」に逃げることができなくなります。生徒は先生の(その時点での)感想に合わせるしかなく、先生の感想は絵から作られているので、結局は絵をちゃんと見るしかありません。stop-gradient は、まさにこの「先生は動かない的(まと)になる」という役割を、勾配を止めることで作り出します。

なぜ崩壊が起きるのか。損失を下げる勾配が両方の枝に流れると、ネットワークは「中身を見分ける難しい仕事」より「両方を同じ定数に潰す簡単な仕事」を選びます。学習は楽な方へ流れます。損失の地形(loss landscape)には「定数解(trivial solution)」という広くて深い谷があり、何の制約もなければ最適化はその谷に転がり落ちます。stop-gradient はこの逃げ道を物理的に塞ぎ、最適化を「中身を見る」方向へ押し戻します。

### 基礎: 前提となる概念
理解に必要な用語を順に噛み砕きます。

勾配(gradient)とは、「損失をほんの少し減らすには、各重みをどちら向きにどれだけ動かせばよいか」を表す方向と大きさの情報です。山下りに例えると、今いる地点で最も急に下る方向を指す矢印です。

誤差逆伝播(backpropagation)とは、出力側で計算した損失を、合成関数の微分の連鎖律(チェインルール)を使って入力側の各層へと逆向きに伝え、各重みの勾配を求める手続きです。順伝播(forward)で出力と損失を計算し、逆伝播(backward)で勾配を計算し、その勾配で重みを更新する、というのが学習の1ステップです。

自己教師あり学習(self-supervised learning, SSL)とは、人間が正解ラベルを付けず、データ自身から問題と答えを作って学ぶ方式です。たとえば「同じ画像から少しずつ違う2枚(ビュー)を作り、その特徴を互いに近づけよ」という問題は、ラベルなしで作れます。学習した特徴は後段の分類・検出などに転用されます。

崩壊(collapse)とは、ネットワークが入力に関係なく常に同じ出力(たとえば全成分が同じ定数のベクトル)を出すようになる現象です。「2つのビューの出力を近づけよ」という目標だけだと、両方が同じ定数を出せば距離はゼロ、損失は完璧に最小化されますが、表現は無意味になります。崩壊には2種類あります。完全崩壊(complete collapse、全サンプルが同一の定数ベクトルになる)と次元崩壊(dimensional collapse、出力が高次元空間の一部の低次元部分空間に潰れ、共分散行列の固有値の多くがゼロに近づいて多くの次元が情報を持たなくなる)です。stop-gradient が主に防ぐのは前者ですが、次元崩壊の緩和にも寄与します。

stop_gradient は実装上の名前で、PyTorch では `tensor.detach()`、TensorFlow では `tf.stop_gradient(tensor)` に対応します。順方向では恒等写像(値をそのまま返す)、逆方向では勾配をゼロにする、という性質を持つ演算だと理解すれば十分です。数式で書くと、順伝播は `sg(x) = x`、逆伝播は `∂sg(x)/∂x = 0` と定義される、という非対称な「演算」です。

### 仕組みを詳しく
2枝の自己蒸留(self-distillation、ネットワークが自分自身の出力を教師信号にして学ぶ構図)を例にします。

```
画像 → ビューA → encoder f → predictor h → p_A ┐
                                               ├─ D(p_A, sg(z_B)) を最小化
画像 → ビューB → encoder f ─────────────→ z_B ┘ ← ここに stop-gradient (sg)
```

z_B 側に stop-gradient をかけると、逆伝播の際に z_B から encoder f への勾配が流れません。つまり「z_B は固定された目標(ターゲット)とみなし、p_A の方だけを z_B に近づける」という非対称な最適化になります。z_B を動かして p_A に合わせにいくこと(=両方を定数に潰す共謀)が禁止されます。

多くの実装では、両枝を対称に扱いながら、それぞれ相手側にだけ stop-gradient をかけて損失を平均します。

```
loss = D(p_A, sg(z_B)) / 2  +  D(p_B, sg(z_A)) / 2
```

ここで D は負のコサイン類似度(2つのベクトルの向きがどれだけ揃っているか。揃うほど小さい値)です。sg は引数を定数とみなす操作です。コサイン類似度は `cos(p, z) = (p·z) / (||p|| ||z||)` で、向きだけを見るのでベクトルの長さの違いに影響されません。

数値で確かめます。z_B = [3, 4](正規化すると [0.6, 0.8])、p_A = [1, 0]、D = -cos。最初 cos = 0.6 です。勾配は p_A 側にだけ流れ、p_A を [0.6, 0.8] の方向へ回そうとします。z_B 側には勾配が流れないので、z_B はこのステップでは「動かない的」として振る舞います。もし stop-gradient を外すと、勾配は z_B 側にも流れ、「p_A を z_B に近づける」と同時に「z_B を p_A に近づける」の両方が起き、最も簡単な解として両者が同じ短いベクトル(究極的にはゼロ)へ向かう力が生まれます。これが崩壊への滑り台です。

なぜ predictor h が要るのか。多くの非対照(non-contrastive)手法では stop-gradient だけでなく、片枝に小さな MLP の predictor を挟みます。直感的には、predictor が「ターゲットの分布へ写す変換」を担うことで、encoder の出力 z が直接ターゲットに張り付くのを防ぎ、表現に多様性を残します。Tian et al.(DirectPred, ICML 2021)は、線形 predictor + stop-gradient + EMA の力学を理論解析し、predictor の重みの固有空間が表現の共分散行列の固有空間に追従する(eigenspace alignment)ことで崩壊が回避される、と説明しました。彼らは固有値分解から predictor を解析的に設定する DirectPred を提案し、学習可能 predictor とほぼ同等の性能を出して「predictor の役割は固有空間の整列にある」ことを裏づけました。

EMA ターゲットとの関係。BYOL や MoCo、後の自己蒸留系では、ターゲット側に別のエンコーダ(target/teacher network)を用意し、その重みをオンライン側のゆっくりした指数移動平均(EMA, exponential moving average: θ_target ← τ θ_target + (1-τ) θ_online、τ は 0.99〜0.999 など)で更新します。ターゲット側には勾配を流さないので、これも本質的に stop-gradient です。EMA は「自分自身の少し過去のコピー」を目標にすることで、目標が急に動かず、安定した非対称性を作ります。stop-gradient(瞬間的な固定)と EMA(時間平均で緩やかに動くターゲット)は、いずれも「ターゲットに勾配を流さない」点で同じ家族に属します。両者の違いは「ターゲットがどれだけ速く動くか」だけで、EMA は τ→0 で stop-gradient のみ(瞬間コピー)、τ→1 で完全固定の中間にあります。

なぜ非対称性が崩壊を止めるのか、もう少し踏み込みます。対称な目的(両枝に勾配を流す)では、勾配場(gradient field)に「2枝を縮めて同じ点へ寄せる」回転成分が現れ、これが定数解へ向かう引力になります。stop-gradient はこの回転成分の片側を消し、結果として勾配場の「縮める力」と「散らす力」が釣り合う非自明な不動点(固定点)を作ります。DirectPred の解析は、この不動点が「表現の分散を保つ」状態に対応することを固有値の言葉で示しました。

### 手法の系譜と主要論文
- MoCo(Kaiming He et al., CVPR 2020, arXiv:1911.05722)。対照学習の文脈で、キー(ターゲット)側エンコーダを EMA(momentum)で更新し勾配を止める設計を導入。負例はキュー(memory queue)に貯める。stop-gradient 思想を対照学習に組み込んだ早い例です。
- BYOL(Jean-Bastien Grill et al., NeurIPS 2020, arXiv:2006.07733, "Bootstrap Your Own Latent")。ターゲットネットワークを EMA 更新し、そこに勾配を流さない(stop-gradient 相当)構成で、負例を一切使わずに高性能を達成。「負例なしでも崩壊しない」を初めて鮮烈に示し、stop-gradient + EMA + predictor の組み合わせが鍵だと位置づけました。
- SimSiam(Xinlei Chen & Kaiming He, CVPR 2021, arXiv:2011.10566)。EMA すら外し、重み共有の双子ネットに predictor + stop-gradient だけを残しても崩壊しないと示しました。stop-gradient が崩壊回避の主役で predictor が補助、と分離特定。詳細は [SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)。
- DirectPred(Yuandong Tian, Xinlei Chen, Surya Ganguli, ICML 2021, arXiv:2102.06810)。predictor + stop-gradient の学習力学を理論解析し、固有空間整列による崩壊回避を説明。
- DINO(Mathilde Caron et al., ICCV 2021, arXiv:2104.14294)。ViT バックボーンの teacher-student 自己蒸留で、teacher 側に stop-gradient + EMA、出力に centering と sharpening を組み合わせて崩壊を防ぎ、教師なしでセグメンテーションが出現する性質を示しました。
- 特徴空間で予測する世界モデル系(画像・動画版)も、target encoder への stop-gradient + EMA を中核に据えており、この系譜の延長にあります。詳細は [LeCunの自律機械知能構想](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl) / [V-JEPA 2](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)。

### 論文の実験結果(定量データ)
評価には ImageNet 線形評価(linear probing)が標準で使われます。これは学習した特徴を凍結し、その上に線形分類器1層だけを乗せて教師あり学習し、トップ1正解率(top-1 accuracy、最も確信度の高い1クラスが正解と一致する割合)を測る方法です。特徴の良さを「下流で使えるか」で測ります。ImageNet は約128万枚・1000クラスなので、当てずっぽう(チャンスレベル)は約 0.1% です。

SimSiam 論文の決定的なアブレーション(ある要素を抜いて影響を見る実験)。ResNet-50 バックボーンで、stop-gradient を入れた場合の ImageNet 線形評価 top-1 はおよそ 67〜68% に達するのに対し、stop-gradient を外すと損失は数十ステップでほぼ最小(-1 に近いコサイン類似度)へ急落し、出力の各次元の標準偏差はゼロ近くに張り付きます。すなわち完全崩壊です。このとき線形評価の精度はチャンスレベル(1000クラスなら 0.1% 付近)まで落ちます。差は「約 68% vs ほぼ 0%」という極端なもので、stop-gradient が機能しているか否かで表現が「有用」か「完全に無意味」かに二分されることを示します。健全な学習では出力の標準偏差が `1/sqrt(d)`(d は出力次元。L2 正規化された出力が単位球上に一様に散らばるときの理論値)付近で安定するのに対し、崩壊時はゼロへ落ちる、という標準偏差の監視は崩壊検知の実務的な指標として広く使われます。

BYOL は ImageNet 線形評価で ResNet-50 において top-1 約 74.3% を報告し、当時の対照学習(SimCLR の約 69%)を上回りました。BYOL でターゲットの EMA を外して即座にオンライン重みをコピーする(=stop-gradient だけで momentum を τ=0 にする)と性能が大きく劣化し、EMA という緩やかなターゲット更新が安定化に寄与することが示されています。一方で後の研究(SimSiam, DirectPred)は「EMA は必須ではなく、stop-gradient が本質」と示し、両者の役割の切り分けが進みました。整理すると、stop-gradient は崩壊回避に必須、EMA は性能を底上げするが必須ではない、という分業です。

DirectPred は、解析的に設定した predictor でも学習可能 predictor と同等(ImageNet 100クラス・1000クラスの双方で数%以内)の線形評価精度を達成し、predictor が固有空間整列を通じて崩壊回避を担うという理論を実験で裏づけました。さらに、predictor を完全に外す(恒等写像にする)と stop-gradient があっても崩壊が起きることを示し、「stop-gradient と predictor は両方そろって初めて働く」という相補性を明らかにしました。

### メリット・トレードオフ・限界
メリット
- 実装が極めて簡単。該当テンソルに `.detach()` を一つ入れるだけ。
- 計算コストがほぼゼロ。順方向の値も計算グラフ上の出力も変わらず、逆伝播の経路が減るぶんむしろ軽い。
- 負例(対照学習で必要な大量の比較対象)や明示的な正則化なしに、predictor や EMA と組み合わせて崩壊を防げる。

トレードオフ・限界
- なぜ効くのかの直感はつかみにくく、理論的理解(DirectPred など)は後から整備された。今も完全な一般理論があるとは言い難い。
- 単独では足りないことが多く、predictor や EMA ターゲットと組み合わせて初めて安定する構成が大半。
- 入れる場所を誤ると、必要な枝まで更新が止まって学習が進まないか、逆に崩壊するかのどちらかになり、設計が繊細。たとえば両枝とも stop-gradient をかけると勾配が一切流れず学習が止まる。
- 次元崩壊(出力が低次元部分空間に潰れる)は stop-gradient だけでは完全には防げず、[VICReg](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) や [Barlow Twins](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) のような分散・脱相関の正則化が補完的に有効、という見方が研究では一般的です。
- ハイパーパラメータ(EMA の τ、predictor の構造・学習率)への感度があり、設定次第で崩壊の境界が動く。

### 発展トピック・研究の最前線
崩壊回避の「統一的理解」を目指す研究が続いています。対照学習(負例)、非対照(predictor + stop-gradient)、正則化(分散・共分散)を、いずれも「表現の分散を保ち、次元間の相関を減らす」という共通原理で説明しようとする流れです。stop-gradient による非対称性が、暗黙のうちに分散を保つ正則化として働く、という解釈も提案されています。次元崩壊そのものを共分散行列の固有値スペクトルで定量化し、stop-gradient や正則化がそのスペクトルをどう平坦化するかを測る研究(Jing et al., 2022, "Understanding Dimensional Collapse in Contrastive Self-supervised Learning", arXiv:2110.09348)も登場しました。特徴空間で予測する世界モデル系(画像・動画版)では、ピクセル再構成ではなく特徴空間で予測する設計と、target encoder への stop-gradient + EMA を組み合わせることで、効率的かつ崩壊しない表現学習を実現しており、stop-gradient はその中核部品として今も現役です。

### さらに学ぶための関連トピック
- [SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [VICReg](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Barlow Twins](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [V-JEPA 2](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [LeCunの自律機械知能構想](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)


## Tensor Parallelism

### ひとことで言うと
ニューラルネットワークの計算の大半は「行列のかけ算」です。Tensor Parallelism（テンソル並列）とは、この大きな 1 回の行列かけ算を、行や列で分割して複数の GPU に分担させ、1 つの層を複数 GPU で同時に計算する手法です。パイプライン並列が「層と層の間」で分けるのに対し、テンソル並列は「層の中」を分けます。GPU 1 枚に乗らないほど大きな層（巨大な重み行列）を持つモデル、特に大規模言語モデル（LLM）の学習で標準的に使われます。

### 直感的な理解
非常に大きな掛け算の宿題を 2 人で分担する場面を想像してください。1 枚の大きな表（重み行列）の左半分を A さん、右半分を B さんが担当すれば、同じ入力を使ってそれぞれ自分の半分だけ計算でき、最後に答えをつなげれば全体の答えになります。表が大きすぎて 1 人の机（GPU メモリ）に載らなくても、半分ずつなら載るし、2 人が同時に計算するので速くもなります。

ただし分け方によっては、各人が「答えの一部分（足し合わせ前の値）」しか持てない場合があります。その場合は最後に 2 人の値を足し合わせる作業（通信）が必要です。この足し合わせが頻繁に発生するので、2 人の机が隣同士（高速接続）でないと、答えを渡し合う時間ばかりかかって遅くなります。テンソル並列が「同じサーバ内の数枚の GPU に限定して使う」のはこのためです。これがパイプライン並列との決定的な違いで、パイプライン並列は層の境界でしか通信しない（通信回数が少ない）ためサーバをまたいでも使えますが、テンソル並列は層の中で頻繁に通信するため、速い接続が前提になります。

### 基礎: 前提となる概念

- 線形層（全結合層）: 入力ベクトルに重み行列をかけて出力を得る、ニューラルネットの最も基本的な層。計算は本質的に `Y = X·W`。
- 隠れ次元（hidden dimension, d）: Transformer 内部で流れるベクトルの長さ。大きいほど表現力が上がるが、重み行列が一気に巨大化する。
- Transformer: 自己注意（self-attention）と MLP（多層パーセプトロン、2 つの線形層）のブロックを積み重ねた構造。LLM の基盤。
- 自己注意のヘッド（attention head）: 注意機構を複数並列に走らせる単位。各ヘッドは独立に計算できる。
- all-reduce: 全 GPU が持つ値を足し合わせ（または平均し）、結果を全 GPU に配る集団通信操作。NCCL（[NCCL (集合通信)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)）が高速に実行する。データ量に対して概ね 2(N−1)/N 倍の通信が走る（リング all-reduce の場合）。
- all-gather / reduce-scatter: all-reduce を二段に分けた部品。Sequence Parallelism で使う。
- NVLink: 同一サーバ内の GPU 同士を結ぶ高速インターコネクト。サーバをまたぐ通常のネットワーク（Ethernet/InfiniBand）より一桁以上速い（NVLink は数百 GB/s、サーバ間は数十〜数百 Gbps）。

なぜ層の中まで分ける必要があるのかを数値で押さえます。層方向に分けるパイプライン並列（[Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)）には限界があり、1 つの層そのものが巨大で 1 枚に乗らない場合は対処できません。隠れ次元 d を 12,288 とすると、Transformer の MLP の線形層の重みは概ね d×4d、つまり 12,288 × 49,152 ≈ 6 億個のパラメータになり得ます。これを何十層も積むため、1 層の重みだけでも大きく、層内をさらに分割しないと収まりません。「層の境目で切る」だけでは足りず「層の中の行列演算を切る」必要が出てきます。これがテンソル並列の動機です。

### 仕組みを詳しく

#### 行列かけ算を「列分割」する

線形層 `Y = X·W`（X が入力、W が重み、Y が出力）の重み行列 W を列方向に 2 つに割り、別々の GPU に置きます。

```
W = [ W1 | W2 ]   ← 列で 2 分割。W1 を GPU0、W2 を GPU1 へ

GPU0: Y1 = X·W1   （入力 X は両 GPU が同じものを持つ）
GPU1: Y2 = X·W2

出力を連結: Y = [ Y1 | Y2 ]
```

tensor shape で追うと、X が [batch, d]、W が [d, 4d] のとき、W1・W2 はそれぞれ [d, 2d]、出力 Y1・Y2 はそれぞれ [batch, 2d] になります。各 GPU は重みの半分しか持たないので、1 枚に乗らない重みを 2 枚に分けて持てます。計算も 2 枚で同時に走ります。列分割の良い点は、順伝播時に各 GPU が出力の「別々の列」を完全に計算するため、出力を作る段階で通信が要らないことです。

#### MLP ブロックでの巧妙な配置（Megatron の核心）

Megatron-LM は、Transformer の MLP ブロック（線形層 → 活性化（GeLU）→ 線形層）で、1 つ目を列分割、2 つ目を行分割にすると、ブロック内の通信を順伝播・逆伝播それぞれ all-reduce 1 回ずつに抑えられることを示しました。

```
入力 X（全 GPU が同一）
  ↓ 1つ目の線形層: 列分割 → 各 GPU が出力の別々の列を持つ（通信なし）
  ↓ GeLU（要素ごと。列分割のまま独立に計算可）
  ↓ 2つ目の線形層: 行分割 → 各 GPU は「足し合わせ前の部分和」を持つ
  ↓ all-reduce（部分和を全 GPU で合計して配る）← ここで 1 回だけ通信
出力 Y（全 GPU が同一）
```

なぜ「列→行」の順が効くか。1 つ目を列分割すると出力が列ごとに分かれ、その並びは 2 つ目の線形層の「行」とちょうど対応します。だから 2 つ目を行分割にすると、各 GPU が自分の持つ列だけで部分和を作れ、間に通信を挟まずに済みます。最後の all-reduce だけで完全な出力が得られます。GeLU のような要素ごとの非線形関数が間に挟まっても、列分割のまま各 GPU が独立に計算できる（非線形は要素単位なので分割を壊さない）点も、この配置が成立する鍵です。

#### 行分割と all-reduce の必要性

行分割では、各 GPU は出力の「足し合わせ前の部分和」を計算します。正しい出力にするには全 GPU の部分和を合計する必要があり、これが all-reduce です。

```
GPU0 が部分和 a を、GPU1 が部分和 b を計算
   ↓ all-reduce（a+b を計算して両方へ配る）
GPU0: a+b   GPU1: a+b   ← どちらも完全な出力を得る
```

順伝播と逆伝播の両方で、この all-reduce がブロックごとに走ります（Megatron では Transformer 1 層あたり MLP と Attention で順伝播 2 回・逆伝播 2 回の all-reduce）。だからテンソル並列は通信が頻繁で、GPU 間が高速につながっていないと遅くなります。Megatron がテンソル並列を「同一サーバ内の数枚（典型的には 8 枚）」にとどめ、それを超える分割はパイプライン並列やデータ並列に任せるのは、この通信コストのためです。

#### Attention の分割

自己注意は複数のヘッドが並列に動くので、ヘッド単位で GPU に振り分けます。各ヘッドは独立に計算できるため、ヘッドを GPU 間で分けるのは自然で、QKV（query/key/value）を作る線形層を列分割（ヘッドごとに分ける）、出力をまとめる線形層を行分割にして、最後に all-reduce します。MLP と同じ「列→行で通信 1 回」の構造になります。注意点として、テンソル並列度はヘッド数を割り切れる値でなければなりません（例: 32 ヘッドを 8 分割なら各 GPU が 4 ヘッド）。

#### Sequence Parallelism などの拡張

Megatron の後続研究では、層正規化（LayerNorm）や dropout のような行列かけ算以外の部分でも、入力を系列（sequence、トークンの並び）方向に分割してメモリをさらに節約する Sequence Parallelism が提案されました。テンソル並列が all-reduce を使う区間は活性が全 GPU で重複（複製）して保持されますが、その間の非テンソル並列区間（LayerNorm 等）を系列方向に分割すれば、活性の重複を解消できます。実装上は all-reduce を all-gather + reduce-scatter に置き換えることで、通信量を増やさずに活性メモリを削れます。さらに 2D/2.5D/3D テンソル並列（Optimus、Tesseract など Colossal-AI 系）は、行列を縦横両方向に分割して通信量を GPU 数の増加に対してより緩やかにスケールさせる発展形です。

#### ZeRO との関係

テンソル並列とよく比較されるのが ZeRO（Rajbhandari et al., SC 2020）で、これはデータ並列のままオプティマイザ状態・勾配・重みを GPU 間に分散保持してメモリを削る手法です。テンソル並列が「1 層の計算そのものを割る」のに対し、ZeRO は「計算はデータ並列のまま、置き場所だけ割る」点が異なります。実用では両者を組み合わせます。詳しくは [DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) を参照してください。

### 手法の系譜と主要論文

- Megatron-LM（Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", arXiv 2019, NVIDIA）。Transformer の MLP と Attention をどう列分割・行分割すれば通信回数を最小化しつつ層内を複数 GPU で並列計算できるかを示した、テンソル並列の定番手法。動機は、数十億パラメータの言語モデルが 1 枚に乗らず、既存フレームワークを大きく書き換えず（数行の変更で）モデル並列を導入したかったこと。
- ZeRO（Rajbhandari et al., SC 2020, Microsoft）。テンソル並列とは別系統の、データ並列ベースのメモリ削減。比較・併用の対象として重要。
- Megatron のクラスタ論文（Narayanan et al., SC 2021, NVIDIA/Microsoft/Stanford）。テンソル並列（サーバ内）・パイプライン並列（サーバ間）・データ並列（横方向）の 3D 並列を数千 GPU で 50% 超の実効効率で回す配置を確立。
- Sequence Parallelism / 選択的再計算（Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models", MLSys 2023, NVIDIA）。テンソル並列環境での活性メモリを系列方向分割と選択的再計算で削減し、再計算による速度低下を最小化。
- 多次元テンソル並列: Optimus（2D, Xu et al.）、Tesseract（2.5D, Wang et al., ICPP 2021）、Colossal-AI（Bian et al., ICPP 2023）。行列を 2D/2.5D/3D に分割し、テンソル並列の通信量を GPU 数に対してより良くスケールさせる発展形。

（注: 一部の著者・年は要約を含むため、引用時は原典確認を推奨します。）

### 論文の実験結果（定量データ）

指標の意味から説明します。

- スケーリング効率: GPU を N 倍にしたとき性能が理想の N 倍にどれだけ近いか。1.0 に近いほど良い。
- 実効性能（PFLOP/s, TFLOP/s）: 実際に達成した計算速度。理論ピークに近いほど良い。
- MFU（model FLOPs utilization）: 理論ピークに対する実効性能の割合。
- メモリ削減率: 並列化や Sequence Parallelism で活性メモリをどれだけ減らせたか。

報告されている代表的な結果です。

- Megatron-LM（2019）は、83 億パラメータ（8.3B）の GPT 系モデルを 512 個の V100 GPU で学習し、約 15.1 PFLOP/s を達成。単一 GPU のベースライン（約 39 TFLOP/s）に対して 512 GPU で約 76% のスケーリング効率を維持したと報告。アブレーションでは、テンソル並列の分割度を上げるほど 1 層あたりの all-reduce 通信が効き、高速 NVLink がない環境では効率が大きく落ちることを示した。
- Megatron クラスタ論文（SC 2021）は、3072 個の A100 で 1 兆パラメータ級モデルを学習し約 502 PFLOP/s（MFU 約 52%）を達成。アブレーションとして、同一サーバ内 8 枚をテンソル並列、サーバ間をパイプライン並列に割り当てる配置が最良で、テンソル並列をサーバ境界をまたいで広げると（遅い InfiniBand で頻繁な all-reduce を行うことになり）効率が急落することを定量的に示した。これがテンソル並列をサーバ内に限定する根拠データ。
- Sequence Parallelism 論文（MLSys 2023）は、テンソル並列に系列方向分割を加えることで活性メモリを大幅に削減し、従来は全層で活性再計算が必要だったところを選択的再計算だけで済ませ、再計算による計算オーバーヘッドをおよそ 30% 前後から数 % まで圧縮したと報告。メモリと速度の両方を同時に改善した点が新しい。
- 多次元テンソル並列（Tesseract/Optimus 系）は、1D テンソル並列に比べ、GPU 数を増やしたときの通信量の増加を緩やかにでき、大きな並列度でメモリ・通信のバランスを改善できると報告。ただし実装が複雑になり、小規模では 1D に劣る場合もある。

数値の意味を補足します。「512 GPU で効率 76%」とは、理想なら 512 倍速くなるところを実際は約 389 倍速くなった、ということ。残り 24% は通信待ちで失われます。テンソル並列は all-reduce が頻繁なため、この 76% を保つには NVLink が不可欠で、Ethernet 接続だと数十 % まで落ちる、というのが各論文の一貫した教訓です。

### メリット・トレードオフ・限界

メリット
- 1 つの層（巨大な重み行列）が GPU 1 枚に乗らない場合でも、層内を分割して複数 GPU に分散できる。
- 層内の行列演算を複数 GPU で同時に計算するため、うまく組めば計算が速くなる。
- パイプライン並列・データ並列・ZeRO と組み合わせ、数千 GPU 規模の超大規模学習を実現できる。
- フレームワーク（Megatron 等）を使えば、モデルコードの変更を比較的小さく抑えて導入できる。

トレードオフ・限界
- 層ごとに all-reduce 通信が走り通信が頻繁。GPU 間が高速（NVLink など）でないと大きく遅くなる。だいたい同一サーバ内の少数 GPU に限定して使う。
- 分割の仕方をモデル構造（隠れ次元やヘッド数で割り切れる必要がある）に合わせて正しく設計しないと、通信が増えたり結果が壊れたりする。
- all-reduce 区間では活性が全 GPU で重複保持されるため、メモリ削減効果は重みに限られ、活性メモリには効きにくい（Sequence Parallelism がこれを補う）。
- 1 枚に乗るモデル・層には不要で、通信オーバーヘッドぶん損をするだけ。
- デバッグや実装が、単純なデータ並列より難しい。

研究上の未解決課題としては、(1) サーバ境界をまたいでもテンソル並列を効率良く広げる通信圧縮・重なり化（communication-computation overlap）、(2) 不均一なクラスタや低速接続でのテンソル並列度の自動決定、(3) MoE や長系列モデルでの最適な分割軸の選択が挙げられます。

### 発展トピック・研究の最前線

- 通信と計算のオーバーラップ: all-reduce を計算の裏で進めて待ち時間を隠す（fine-grained overlap）。サーバ境界をまたぐテンソル並列を実用化する鍵。
- 自動並列化（Alpa, OSDI 2022 など）: テンソル/パイプライン/データ並列の最適な組み合わせをコンパイラが自動探索・生成する。
- 多次元テンソル並列（2.5D/3D）と Expert Parallelism（MoE 専用の分割）。
- 推論向けテンソル並列: 学習だけでなく LLM 推論でも、1 枚に乗らない巨大モデルを複数 GPU に分けて低レイテンシで動かす実用が広がっている。バッチが小さい推論では all-reduce の固定コストが効くため、量子化や通信削減が研究テーマ。
- コンテキスト並列（context/sequence parallelism の発展）: 超長系列（数十万トークン）の Attention をトークン方向に分割して扱う。

### さらに学ぶための関連トピック
- [Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [NCCL (集合通信)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Spot Instance 学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)
- [Model Registry (Stage 遷移)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/15_infra-mlops-ci)


## TrivialAugment

### ひとことで言うと
データ拡張(学習データを水増しして賢くする工夫)を極限まで単純化した手法です。「画像1枚につき、変換を1つだけランダムに選び、その強さもランダムに決めて、1回だけ適用する」——たったこれだけ。調整するパラメータが実質ゼロ(tuning-free)なのに、複雑で高価な自動拡張(AutoAugment や RandAugment)に匹敵する性能を出すという、意外な結果を示しました。手法そのものの提案であると同時に、「拡張研究の複雑化は本当に価値を生んでいたのか」を問い直す、方法論への批評でもあります。

### 直感的な理解
データ拡張の研究は、AutoAugment 以降「いかに賢く拡張ポリシー(拡張の設計図)を最適化するか」へと進み、強化学習による巨大探索、微分可能化による高速探索、(N, M) への縮小など、年々手の込んだ手法が積み上がっていました。論文の数字は確かに少しずつ良くなっていました。

しかしここには見過ごされがちな落とし穴があります。新手法の論文は「自分の手法」を丁寧にチューニングする一方、比較相手の「単純なベースライン(比較の土台となる素朴なやり方)」のチューニングは甘くなりがちです。すると本来の差より大きく見えてしまう。これは機械学習研究で繰り返し指摘される問題で、「弱いベースラインと比べて勝った」と主張する論文が後から覆る例が多くあります。TrivialAugment の著者(AutoML 研究で著名な Frank Hutter のグループ)は、この公平性の問題に正面から切り込みました。「考えうる最も単純な拡張」をきちんと実装し、最新手法とまったく同じデータ・モデル・学習設定・評価条件で比較したのです。

すると、その単純きわまる方法が、複雑な手法と肩を並べてしまった。料理にたとえれば、何万回も配合実験をして決めたスパイス調合と、「棚から1種類だけ適当につかんで適当な量をかける」やり方とで、できあがりの味がほぼ同じだった、という話です。これは「複雑さに本当に価値があったのか」という、研究のあり方そのものへの一石でした。

### 基礎: 前提となる概念
- データ拡張: 学習画像に加工を施して人工的に多様性を増やし、過学習(学習データだけ覚えて未知データで失敗する状態)を防ぐ技術。
- ベースライン(baseline): 新手法の良し悪しを判断するための比較基準となる、素朴・標準的な方法。ベースラインが弱いと、新手法が実力以上に良く見えてしまいます。
- チューニング不要(tuning-free): 性能を出すために人が調整すべきハイパーパラメータ(事前に決める設定値)がない、という性質。RandAugment は (N, M) の2つを調整する必要がありましたが、TrivialAugment はそれすらありません。
- 公平比較・再現性: すべての手法を同じデータ・同じモデル・同じ学習設定・同じ評価で比べること。これを揃えないと、精度差が手法の差なのか学習設定の差なのか分からなくなります。AutoML コミュニティが特に重視する規範で、TrivialAugment はその立場から生まれました。

### 仕組みを詳しく
TrivialAugment(略して TA)の手順は、文字どおり trivial(自明・ささいな)です。1枚の画像に対して:

1. 変換の集合(RandAugment と同様の 14 種類前後: Rotate、ShearX/Y、TranslateX/Y、Color、Contrast、Brightness、Sharpness、Posterize、Solarize、Equalize、AutoContrast、Invert、Identity)から、変換を1つだけ一様ランダムに選ぶ。
2. その変換が取りうる強度範囲(離散値)から、強度を1つだけ一様ランダムに選ぶ。
3. 選んだ変換を、選んだ強度で、1回だけ適用する。

以上です。RandAugment との違いは明確です。

- RandAugment: 変換を N 個(例 2 個)選び、強度は全変換共通の固定値 M。N と M は人が決める(2つのつまみ)。
- TrivialAugment: 変換は常に 1 個。強度は毎回ランダム(固定値すらない)。決めるつまみは「ない」。

#### 数値イメージ
入力 `[3, 224, 224]`(RGB、224×224)のとき:
- 1枚目: ランダムに Rotate が選ばれ、強度範囲(例 -135〜+135 度に対応する離散値)からランダムに引かれ、たとえば +40 度回転。
- 2枚目: ランダムに Brightness が選ばれ、強度もランダムに引かれ、やや明るく補正。
- 3枚目: ランダムに Identity(何もしない)が選ばれ、無加工のまま。

出力の形は `[3, 224, 224]` のまま。各画像で「どの変換が、どの強さで」効くかが毎回バラバラなので、データセット全体としては弱い加工から強い加工までまんべんなく現れ、豊かな多様性が生まれます。これが「1個しかかけないのに足りる」理由の核心です。

著者は強度範囲の与え方を2種類検討しました。
- TA(RA): RandAugment 由来の控えめな強度範囲。
- TA(Wide): より広い(強い)強度範囲。

どちらが良いかはデータセット依存ですが、いずれも余計なチューニングなしで強い結果を出しました。一般に TA(Wide) のほうが ImageNet など大規模で有利な傾向が報告されています(大きいモデルほど強い拡張に耐えられる、という RandAugment の知見とも整合します)。

#### なぜ「1個・ランダム」で足りるのか
直感的には「複数の強い変換を重ねるほど多様で良さそう」に思えます。しかし変換を重ねすぎると画像が壊れすぎ、かえって学習を妨げます(過剰拡張)。回転の上に色変換、その上にせん断、と重ねると、元の物体がほとんど判別不能になることがあります。TA は「1枚につき1変換」に絞ることで過度な破壊を避けつつ、強度を一様ランダムにすることで弱い加工から強い加工までまんべんなく現れ、ちょうど良い多様性を確保します。「探索で1点の最適を狙う」のではなく「ランダムで強度空間を広くカバーする」ことが、実は十分だったという結論です。さらに Identity を集合に含めることで「無加工の原画像」も一定割合(14 種なら約 1/14)で残り、加工した分布が元データから離れすぎるのを防ぎます。

### 手法の系譜と主要論文
- AutoAugment(Cubuk et al., CVPR 2019): RL で巨大探索。自動拡張の起点で、最も複雑([AutoAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。
- Fast/Faster AutoAugment(2019–2020): 探索を高速化。複雑さはやや残る。
- RandAugment(Cubuk et al., NeurIPS 2020): 探索を捨て (N, M) に縮小([RandAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。
- UniformAugment(LingChen et al. 2020): 探索不要を志向した先行研究の一つで、強度を連続一様分布から引く。TA と問題意識が近く、ほぼ同時期の独立した到達点です。
- TrivialAugment(Müller & Hutter, ICCV 2021): 1変換・ランダム強度・チューニング不要。「単純化の到達点」。

系譜としては AutoAugment(探索)→ RandAugment(探索を縮小)→ TrivialAugment(探索を捨てる)という流れの終着点にあたり、「複雑な最適化より、よく設計された単純なランダム化のほうがコスパが良いことがある」という教訓を残しました。

### 論文の実験結果(定量データ)
指標の意味: CIFAR は誤分類率(間違えた割合、低いほど良い)、ImageNet は Top-1 精度(最も確信した予測が正解の割合、高いほど良い)。著者は全手法を同一の学習設定・評価で揃え、公平比較を徹底した点が結果の信頼性を支えています。論文の主眼は「自分が最強」ではなく「単純な手法でもこれだけ出る」を公平に示すことでした。

TrivialAugment(Müller & Hutter 2021)の報告:
- CIFAR-10 / CIFAR-100 で、AutoAugment・RandAugment・Fast AutoAugment と同等かわずかに上回る精度を、探索もチューニングもなしで達成。差は多くの場合 0.1〜0.3 ポイント程度の僅差で、「複雑な手法の優位はその程度しかない」ことを定量的に可視化しました。
- SVHN(街の番地数字)でも同様に競争力を確認。
- ImageNet で ResNet-50 の Top-1 精度が、RandAugment や AutoAugment と同等水準(報告で約 77.5〜78% 規模)を達成。大規模設定でも単純さが通用することを示しました。
- アブレーション(要素を変えて影響を見る実験): 「1変換」を「2変換」に増やしても明確な改善はなく、むしろ強度次第で悪化しうる。これが「1個で十分」という主張の実証的根拠です。また強度をランダムにせず固定すると、最適な固定値を当てない限り劣化し、ランダム化のほうが堅牢であることが示されました。TA(Wide) と TA(RA) の比較では、データ規模が大きいほど広い強度範囲が有利という傾向が出ました。
- コスト比較: AutoAugment が数千 GPU 時間規模の事前探索を要し、RandAugment が (N, M) の小グリッドサーチを要したのに対し、TA は探索ゼロ・追加チューニングコストゼロ。実装も数行で、外部依存も小さい。

総じて、「公平に比べれば、最も単純な拡張が最先端に匹敵する」という主張を、複数データセット・複数モデルで一貫して裏づけた点が本論文の価値です。精度の新記録ではなく、「研究の進歩の測り方」への問題提起としての価値が大きい論文です。

### メリット・トレードオフ・限界
メリット
- 実装も運用も最も単純。調整パラメータが実質ゼロで、新しいデータセットに即適用できる。
- 探索コスト・チューニングコストがかからない。
- それでいて AutoAugment / RandAugment に匹敵する精度。
- 新規データに対する「強い初期ベースライン」として最適で、より凝った手法を試す前の出発点に向く。論文自身が「まずこれを試せ」と主張する立場です。

トレードオフと限界
- 変換集合(どの 14 種類を使うか)と各変換の強度範囲の定義は依然必要。ドメインに合わせる手間は残り、完全に「何も決めなくてよい」わけではありません。「ハイパーパラメータがゼロ」というより「設計の自由度を最小化した」が正確です。
- 「1画像1変換」が常に最適とは限らず、タスクやデータ規模によってはより複雑・強力な拡張(混合系の併用など)が勝つこともある。
- ランダム性が強いので、ごく小規模データでは結果がばらつきやすく、複数シード(乱数の初期値を変えた複数回の実験)での評価が望ましい。
- 方向や空間構造が意味を持つドメイン(自動運転で左右通行や上下の意味)では、回転・反転・大きな平行移動が意味を壊すため、変換集合と強度範囲の取捨選択が必須。
- Cutout/Random Erasing や Mixup/CutMix のような「削除・混合」系の変換は標準集合に含まれないため、それらの効果を得たければ別途併用が必要です。

### 発展トピック・研究の最前線
- 公平比較という規範: TrivialAugment は「強いベースラインで公平に比べる」ことの重要性を業界に再認識させ、以後の拡張研究はベースラインのチューニングを厳格化する流れが強まりました。これは AutoML 研究全般、さらには機械学習の実験設計一般に通じる教訓です。
- 単純化の限界点: TA でほぼ飽和したとされる画像分類に対し、検出・セグメンテーション・動画・3D など、変換集合の設計自体が非自明な領域では「何を集合に入れ、どの強度範囲にするか」がなお研究課題です。位置や時間の整合性を保つ変換の設計は分類より難しい。
- 混合・削除系との統合: 実務の強い学習レシピでは TA(あるいは RandAugment)に Mixup・CutMix・Random Erasing・label smoothing を重ねるのが定番で、これらの最適な組み合わせ方は依然探索対象です。TA は「幾何・色変換」を担い、混合・削除系は別レイヤとして併用されるのが一般的です。
- 自己教師あり・大規模事前学習との関係: 大規模事前学習が進むと拡張の限界利益は変化しうるため、「どの規模でどの拡張がどれだけ効くか」という規模則(scaling、規模を変えたときの性能の変化則)の視点での再評価が進んでいます。データが豊富な領域では拡張の寄与が相対的に小さくなる傾向も指摘されています。

### さらに学ぶための関連トピック
- [RandAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [AutoAugment](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Cutout / Random Erasing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Mixup / CutMix](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [Fine-tuning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/06_backbone-perception)


## VICReg

### ひとことで言うと
VICReg(ヴィックレグ)は、ラベルのないデータから良い特徴を学ぶ自己教師あり学習の手法で、崩壊(全部同じ答えに潰れる失敗)を、3つの数学的な罰則(正則化項)を損失に足すことで明示的に防ぎます。3つとは分散(Variance)・不変(Invariance)・共分散(Covariance)で、頭文字を取って VICReg です。負例も EMA も stop-gradient も使わず、損失式を見ただけで「なぜ崩壊しないか」が分かる透明さが特徴で、画像と別モダリティを揃える用途や特徴空間で予測する世界モデル系の損失設計にも採用されています。

### 直感的な理解
2つの枝で同じ画像の別ビューを近づける学習は、放っておくと崩壊します。崩壊とは、入力に関係なく全部同じ定数ベクトルを出して損失をゼロにする手抜きです。

これまでの崩壊回避策は、それぞれ「特別な仕掛け」に依存していました。対照学習は大量の負例、BYOL は EMA、SimSiam は stop-gradient と predictor、という具合です([SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl) / [Stop-gradient (勾配停止)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) 参照)。VICReg の発想はもっと直接的で、「崩壊とは要するに特徴の散らばりが消える現象なのだから、散らばりを保てと損失で命令すればよい」というものです。たとえば全員が同じ答えに潰れたら、その答えの分布の散らばり(分散)はゼロになります。ならば「各次元の散らばりを最低でもこれだけ保て」と罰則を課せば、潰れること自体を禁止できます。これは仕組みのトリックではなく、欲しい性質を損失に直書きする素直な設計です。設計者が「崩壊を防ぐ機構」を間接的に祈るのではなく、「崩壊した状態=分散ゼロ」を直接に禁止する、という違いです。

### 基礎: 前提となる概念
埋め込み(embedding)とは、ネットワークが画像を写した特徴ベクトルです。バッチ(同時に処理する画像のまとまり)が N 枚、各埋め込みが D 次元なら、埋め込み行列の形状は `[N, D]` です。

分散(variance)とは、ある数値の集まりがどれだけ散らばっているかの指標です。全部同じ値なら分散ゼロ。標準偏差(standard deviation)は分散の平方根で、同じく散らばりを表します。

共分散(covariance)とは、2つの変量が一緒に動く度合いです。次元 i と次元 j の共分散が大きいと、それらは似た情報を持つ(冗長)ことを意味します。共分散行列(covariance matrix)は全次元ペアの共分散を並べた `[D, D]` の行列で、対角は各次元の分散、非対角が次元間の相関に対応します。

不変性(invariance)とは、同じ画像の別ビューに対して同じ特徴を返してほしい、という要請です。データ拡張で見た目が変わっても本質は同じなので、特徴は変わらないでほしい、という意味です。

ヒンジ関数 `max(0, ・)` は「閾値を下回ったぶんだけ罰する」装置です。`max(0, γ - s)` は s が γ 以上なら 0(罰なし)、γ 未満なら不足分だけ正の値(罰あり)になります。サポートベクターマシンのヒンジ損失と同じ形で、片側だけを罰する非対称な罰則です。

### 仕組みを詳しく
VICReg は2つの枝の出力埋め込み Z, Z'(それぞれ形状 `[N, D]`、論文では N=2048、D=8192 など)に対して、損失を3項の和で定義します。重要なのは、2枝の対応を使うのは不変項だけで、分散項と共分散項はそれぞれの枝の統計だけで計算される点です。

#### 1. Invariance(不変項)
同じ画像の2ビューの埋め込みを近づける、平均二乗誤差です。

```
s(Z, Z') = (1/N) Σ_i ||z_i - z'_i||^2    ← 小さくしたい(2ビューを一致させる)
```

これだけだと両方を定数にすれば差ゼロで崩壊します。残り2項が崩壊を止めます。

#### 2. Variance(分散項)
各次元について、バッチ内の標準偏差が目標値 γ(例 1)以上であることを要求します。

```
v(Z) = (1/D) Σ_j max(0, γ - std(Z の j 列目))    ← 散らばりが γ 未満なら罰
```

ここで `std(x) = sqrt(Var(x) + ε)`(ε は数値安定用の微小値)。各次元の散らばりが γ を下回ると罰則が出るので、特徴が定数に潰れません。これが崩壊回避の主役です。重要なのは、この項が片方の枝の統計だけで計算できることです(2枝の対応関係に依存しない)。`std` の中に ε を入れて平方根の外で罰するのは、分散がゼロのとき勾配が発散するのを避けるための工夫です。

#### 3. Covariance(共分散項)
次元同士が同じ情報を持たない(無相関)よう、共分散行列の非対角成分を小さくします。

```
C(Z) = (1/(N-1)) Σ_i (z_i - z̄)(z_i - z̄)^T    ← [D,D] 共分散行列
c(Z) = (1/D) Σ_{i≠j} [C(Z)]_{i,j}^2           ← 非対角の二乗和を小さく
```

これで D 次元が「同じことの繰り返し」にならず、情報が次元全体に広がります([Barlow Twins](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の冗長性削減と同じ狙い)。

#### 全体の損失
```
loss = λ·s(Z,Z')  +  μ·[v(Z)+v(Z')]  +  ν·[c(Z)+c(Z')]
```

論文の標準設定は λ=25, μ=25, ν=1。2枝は重みを共有してもしなくてもよく、stop-gradient も EMA も負例も不要です。

直感的なまとめ:
- Invariance =「2ビューは似せろ」(近づける引力)
- Variance =「でも全部同じにはなるな」(崩壊を防ぐ反発力)
- Covariance =「各次元はバラバラの情報を持て」(冗長性を消す力)

数値例。ある次元の std が 0.3 で γ=1 なら、分散項は `max(0, 1-0.3)=0.7` の罰を出し、その次元の散らばりを広げる方向に勾配を与えます。逆に std が 1.2 なら罰はゼロで、その次元はもう十分散らばっているとみなされます。共分散項は、次元 i と j がともに動く(相関 0.5 など)と非対角が大きくなり、それを 0 に押し下げます。

なぜ stop-gradient/EMA が要らないのか。分散項が「潰れるな」を、不変項に頼らず片枝の統計だけで保証するため、崩壊への逃げ道がそもそも塞がれているからです。これにより、2枝で構造や次元が違っても、片枝が固定された別モダリティでも、崩壊回避が成立します。これが VICReg の汎用性の源です。Barlow Twins が「2ビュー間の相互相関」という枝をまたいだ量に依存するのに対し、VICReg は分散・共分散を各枝内に閉じて計算できるよう分離したのが、この汎用性をもたらした技術的な核心です。

### 手法の系譜と主要論文
- SimCLR / MoCo(2020)。負例で崩壊を防ぐ対照学習。大バッチやメモリキューが必要。
- BYOL(Grill et al., NeurIPS 2020)。EMA ターゲット + predictor + stop-gradient で負例なしに崩壊回避。
- SimSiam(Chen & He, CVPR 2021)。predictor + stop-gradient が最小要件と特定([SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl))。
- Barlow Twins(Zbontar et al., ICML 2021, arXiv:2103.03230)。2ビュー間の相互相関行列を単位行列に近づける。VICReg と近い「情報冗長性低減」系の先行手法([Barlow Twins](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))。
- VICReg(Adrien Bardes, Jean Ponce, Yann LeCun, ICLR 2022, arXiv:2105.04906)。分散・不変・共分散の3項に分解。Barlow Twins の相互相関を「自枝の分散」と「自枝の共分散」に分離独立させ、片枝の統計だけで崩壊を防げるよう一般化したのが新規性。これにより2枝の対称性・重み共有・次元一致の制約が外れた。
- VICRegL(Bardes, Ponce, LeCun, NeurIPS 2022, arXiv:2210.01571)。VICReg を局所特徴(画像内の場所ごとの特徴)へ拡張。物体検出・セグメンテーションのように位置が重要なタスク向け。
- 対照・非対照の双対性(Garrido et al., ICLR 2023, arXiv:2206.02574)。VICReg のような次元方向の正則化(dimension-contrastive)と、SimCLR のようなサンプル方向の対照(sample-contrastive)が、適切な変換のもとで等価に書けることを示し、両系統を統一的に理解する橋渡しをした。

### 論文の実験結果(定量データ)
評価は ImageNet 線形評価(特徴を凍結し線形分類器1層で top-1 正解率を測る。1000クラス、チャンスレベル約 0.1%)が中心です。

VICReg は ResNet-50・1000エポックの事前学習で ImageNet 線形評価 top-1 約 73.2% を報告。同水準の Barlow Twins(約 73.2%)や BYOL(約 74.3%)、SwAV(約 75.3%)と競合し、対照学習(SimCLR 約 69%)を上回ります。要点は「3項の素直な正則化だけで最先端級に並ぶ」ことです。

半教師あり(ラベルの1%/10%だけ使う設定)でも、VICReg は ResNet-50 で 1% ラベル時 top-1 約 54.8%、10% ラベル時 約 69.5% を報告し、ラベルが乏しい状況での有用性を示しました。1% ラベルとは、1クラスあたり十数枚しか正解を与えない極端に乏しい設定で、ここで 50% を超える精度が出るのは、事前学習した特徴が「ラベルなしでも下流タスクに効く構造」を捉えている証拠です。

転移学習。VICReg 特徴を物体検出(VOC07+12, COCO)へ転移したとき、平均適合率(AP、検出ボックスの正しさを複数の重なり閾値で平均した指標)で教師あり事前学習や他の自己教師あり手法と同等の性能を示しました。

アブレーション(各項の寄与)。分散項(Variance)を外すと即座に崩壊し、線形評価精度がチャンスレベルへ転落します。これは分散項が崩壊回避の主役であることを示します。共分散項(Covariance)を外すと崩壊はしないものの精度が数%低下し(冗長性が残るため有効次元が減る)、共分散項が「次元を有効活用させる」役割を持つと分かります。不変項を外すと、2ビューを揃える信号が消えて表現が無意味化します。3項のバランス(λ, μ, ν)を大きく崩すと性能が落ち、調整が必要だと報告されています。論文はまた、片枝が固定(stop-gradient 的)でも、2枝で異なるアーキテクチャでも崩壊しないことを実験で確認し、対称性に依存しないことを示しました。

VICRegL は、ローカル特徴も学ぶことで、セグメンテーション(ピクセル単位の領域分け)や検出など位置依存タスクで、グローバルのみの VICReg を上回る結果を示しました。グローバル特徴とローカル特徴の重み α を振るアブレーションでは、分類はグローバル寄り、セグメンテーションはローカル寄りが良い、というタスク依存のトレードオフが報告されています。

### メリット・トレードオフ・限界
メリット
- 崩壊回避が明示的。損失式を読めば「なぜ潰れないか(分散項)」「なぜ冗長でないか(共分散項)」が分かる透明さ。
- 負例・EMA・stop-gradient のいずれも不要で、構成が素直。学習が安定しやすい。
- 2枝で構造・次元が異なっても、片枝が固定でも使える。画像と別モダリティ(テキスト・音声・センサ)を揃えるマルチモーダル学習や、片側がターゲット固定の予測学習に応用しやすい。

トレードオフ・限界
- 3項の重み(λ, μ, ν)のバランス調整が必要で、データやモデルが変わると再調整が要ることがある。
- 共分散項は `[D, D]` 行列を扱うため、次元 D を大きくすると計算・メモリが D の二乗で増える。高次元では重い。
- 分散・共分散はバッチ内統計に依存するため、極端に小さいバッチでは推定が不安定になりやすい(分散項の γ 推定が荒れる)。ただし対照学習ほど大バッチには敏感でない。
- 「次元崩壊を完全に防ぐか」は γ や ε の設定に依存し、設定次第では弱い冗長性が残ることがある。
- 共分散項は線形相関しか消せず、次元間の非線形な依存(相関ゼロでも独立でない関係)までは除けない。

### 発展トピック・研究の最前線
VICReg は、崩壊回避を「学習構造のトリック(stop-gradient/EMA)」から「損失の明示的正則化」へ移したことで、自己教師あり学習の統一理解を進めました。対照学習・非対照学習・正則化系のいずれもが「分散を保ち次元間相関を減らす」共通原理で説明できる、という議論において、VICReg はその原理を最も直接的に体現した手法と位置づけられます。Garrido et al.(2023)は、次元方向の正則化(VICReg/Barlow Twins)とサンプル方向の対照(SimCLR)が双対(数式上で互いに書き換え可能)であることを示し、この統一を理論的に裏づけました。LeCun らが推進する世界モデル/予測アーキテクチャ系では、ピクセル再構成ではなく特徴空間で予測する設計と相性がよく、「予測した特徴の分散を保て」という VICReg 的正則化が崩壊回避の道具として組み込まれています。マルチモーダル整列(画像-テキスト、画像-LiDAR など)や、片枝が固定エンコーダの蒸留設定への応用も活発で、2枝対称性に縛られない VICReg の柔軟性が活きる領域として研究が続いています。

### さらに学ぶための関連トピック
- [Barlow Twins](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Stop-gradient (勾配停止)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [V-JEPA 2](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)
- [LeCunの自律機械知能構想](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/01_world-model-jepa-ssl)


## ZeRO (Zero Redundancy Optimizer)

### ひとことで言うと
複数 GPU で学習するとき、各 GPU が「同じもの (重み・勾配・optimizer 状態) を丸ごとコピーで持っている」のは大きな無駄です。ZeRO はこの冗長 (redundancy) をゼロに近づけ、それらを GPU 間で分担して持たせる技術です。これにより、本来なら 1 台に入りきらない超大規模モデル (数十億〜1 兆パラメータ) を、限られた数の GPU で学習できるようになります。Microsoft の DeepSpeed ライブラリの中核機能であり、PyTorch [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の直接の理論的源流でもあります。

### 直感的な理解
[DDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の問題は「全員が全部を冗長に持つ」ことでした。GPU が 4 台なら、まったく同じ optimizer 状態が 4 セット存在し、3 セットはただの重複です。メモリの観点では「4 人で同じノートを 4 冊持っている」状態で、紙の無駄です。

ZeRO の発想は「ノートを 4 分割して各人 1/4 だけ持ち、必要なときだけ貸し借りしよう」です。普段の保持量が 1/4 になるので、4 倍大きなモデルが扱えます。ポイントは「何を分割すると、どれだけメモリが減り、どれだけ通信が増えるか」を段階的に選べることです。最も削減効果が大きいのは optimizer 状態 (Adam では最大の容量を占める) で、ここだけ分割しても通信はほとんど増えずに大きなメモリ削減が得られる、という「お得な順番」が ZeRO の核心の発見です。

もう 1 つ重要なのは、ZeRO がモデル並列 (テンソルを物理的に切る方式) ではなくデータ並列の枠内で動く点です。モデル並列は層の内部を切るのでモデルの実装を書き換える必要があり、層をまたぐたびに通信が要りますが、ZeRO は「データ並列のままメモリの冗長だけ消す」ので、ユーザーから見ればコード変更が最小で済みます。この「使いやすさを保ったままメモリの壁を破る」設計思想が、ZeRO が広く普及した理由です。

### 基礎: 前提となる概念

#### 混合精度 + Adam のメモリ内訳
なぜ optimizer 状態がそんなに大きいのかを正確に押さえます。重みの個数を Ψ とし、混合精度学習 (計算は bf16/fp16、更新は fp32) で Adam を使う標準的な構成を考えます。1 パラメータあたりのバイト数は次の通りです。

- fp16 のパラメータ: 2 バイト
- fp16 の勾配: 2 バイト
- optimizer 状態 (fp32): マスターパラメータ 4 + Adam の m (1 次モーメント = 勾配の移動平均) 4 + Adam の v (2 次モーメント = 勾配の 2 乗の移動平均) 4 = 12 バイト

合計で 2 + 2 + 12 = 16 バイト/パラメータ。つまり 1 パラメータあたり 16Ψ バイトのうち、optimizer 状態だけで 12Ψ バイト (全体の 3/4) を占めます。15 億パラメータ (GPT-2 級) なら 16 × 1.5G = 24GB となり、1 台の GPU に乗せるだけでもギリギリ、複数台で学習しようにも DDP では全台が 24GB 持つので何も解決しません。なぜマスターコピーが fp32 で要るかというと、fp16 のままだと小さな更新量 (学習率 × 勾配) が丸め誤差で消えてしまい学習が進まないためで、更新は fp32 の高精度で蓄積する必要があるからです。

#### DDP との対比
DDP は上記 16Ψ を全 GPU が複製します。台数を N にしても 1 台あたりは 16Ψ のまま。ZeRO は「同じものを N セット持つのをやめ、N 分割して各台 16Ψ/N に近づける」ことを目指します。ここに forward の中間 activation のメモリは含まれていない点に注意してください。activation を削るのは [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning) の役割で、ZeRO はパラメータ・勾配・optimizer 状態という「モデル状態 (model states)」だけを対象にします。両者は直交するので併用が前提です。

### 仕組みを詳しく

#### 3 つのステージ
ZeRO は「何を分割するか」で 3 段階あります。後のステージほどメモリ削減が大きく、その分通信が増えます。GPU 数を N とします。

- ZeRO Stage 1 (optimizer 状態の分割, P_os): 最も大きい optimizer 状態 (12Ψ) だけを N 分割し、各 GPU は 1/N ずつ持つ。パラメータと勾配は従来通り全 GPU が複製。backward 後は通常通り勾配を集約するが、step では各 GPU が「自分が担当するパラメータ分」だけを更新し、更新後のパラメータを all-gather で全員に配り直す。少ない通信増で最大の削減が得られる「お得」なステージ
- ZeRO Stage 2 (+ 勾配の分割, P_os+g): Stage 1 に加えて勾配も N 分割。backward 後に reduce-scatter で「自分の担当分の勾配」だけを各 GPU に集約するので、全勾配を保持する必要がなくなる
- ZeRO Stage 3 (+ パラメータの分割, P_os+g+p): パラメータ本体も N 分割。計算する層の重みを必要なときだけ all-gather で集め、終わったら捨てる。最もメモリ効率が高く、PyTorch [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) の FULL_SHARD はこれに相当する

#### メモリ削減の数値 (論文の試算)
混合精度 + Adam で 1 パラメータあたり 16 バイトを基準に、N = 64 GPU のときおよそ次のように減ります (論文の図表より)。Ψ = 7.5B (75 億パラメータ) のモデルでは、これは GPU あたり 120GB が 1.9GB 程度まで落ちる計算に相当します。

```
ベースライン (DDP):  16Ψ バイト/GPU      (台数で減らない)
Stage 1:             4Ψ + 12Ψ/N ≈ 4Ψ    (optimizer 状態だけ 1/N)
Stage 2:             2Ψ + 14Ψ/N ≈ 2.6Ψ  (勾配も 1/N)
Stage 3:             16Ψ/N ≈ 0.25Ψ      (すべて 1/N、台数に反比例)
```

つまり Stage 3 では「GPU を増やせば増やすほど 1 台あたりの分担が減る」ため、原理的には台数を揃えれば任意のサイズのモデルが乗ります。一方で Stage 1 は、わずかな通信増 (step 後の all-gather だけ) で 16Ψ → 約 4Ψ という 4 倍の大削減が得られるので、コストパフォーマンスが極めて高いステージです。「とりあえず Stage 1 か 2 を試し、足りなければ Stage 3」というのが実務の定石です。

#### 通信量のトレードオフ
重要なのは「Stage 1 と 2 は DDP と同等の総通信量で済む」という解析結果です。DDP の all-reduce は reduce-scatter + all-gather に分解でき (各 Ψ 分)、Stage 1/2 はこれを単に分解して使うだけなので、通信量はほぼ同じ (約 2Ψ) なのにメモリは激減します。一方 Stage 3 はパラメータの all-gather が forward と backward の両方で追加発生するため、総通信量がおよそ 3Ψ、つまり DDP の約 1.5 倍になります。「Stage 3 だけは通信代を払う」という構造で、ここが帯域の細い環境でのボトルネックになります。

#### ZeRO-Offload と ZeRO-Infinity
派生として、状態を GPU メモリ以外に逃がす手法があります。

- ZeRO-Offload (ATC 2021): optimizer 状態と勾配、そして optimizer の更新計算そのものを CPU メモリ・CPU 演算に逃がす。GPU は forward/backward に専念。通信量の解析に基づき「fp16 の forward/backward は GPU、fp32 の重み更新は CPU」と役割を分けることで CPU-GPU 転送を最小化する設計。GPU が少ない環境 (極端には 1 台) でも大モデルを学習できるが、CPU-GPU 転送がボトルネックになる
- ZeRO-Infinity (SC 2021): さらに NVMe (SSD) まで階層的に使い、GPU メモリ → CPU メモリ → NVMe の 3 階層に状態を流す。限られたハードで桁違いに大きいモデルを扱えるが、SSD 帯域が律速でスループットは大きく落ちる。帯域を稼ぐための prefetch やオーバーラップが鍵

### 手法の系譜と主要論文

- Rajbhandari, Rajbhandari, Ruwase, He, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020, Microsoft, arXiv:1910.02054)。本トピックの原典。3 ステージを提案。動機は「DDP はメモリが冗長でモデルサイズが頭打ち、一方モデル並列 (テンソルを物理的に切る方式) は実装が難しく通信効率が悪い。データ並列の使いやすさを保ちつつメモリの冗長だけを消す」こと。理論上 1 兆パラメータ級を可能にすると示しました。

- Rasley, Rajbhandari, Ruwase, He, "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters" (KDD 2020, Microsoft)。ZeRO を実装したライブラリ DeepSpeed。既存の PyTorch コードに数行の JSON 設定で ZeRO を適用できる実用面を確立しました。

- Ren ら, "ZeRO-Offload: Democratizing Billion-Scale Model Training" (USENIX ATC 2021, arXiv:2101.06840)。CPU に optimizer 状態を逃がし、1 台の GPU でも 13B モデルを学習できることを実証。「大規模学習の民主化」を掲げました。

- Rajbhandari ら, "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" (SC 2021, arXiv:2104.07857)。NVMe まで使ってオフロードを拡張し、限られたハードで 30 兆パラメータ規模に手が届くと示した続編です。

- Wang ら, "ZeRO++" (2023, arXiv:2306.10209)。Stage 3 の弱点である通信量増を、重みの量子化 (all-gather する重みを低ビットに圧縮) や階層的な配置で削減し、帯域の細いクラスタでのスループットを改善しました。

系譜としては「DDP の冗長性 → ZeRO (冗長を段階的に消すという発見) → DeepSpeed 実装 → Offload/Infinity (GPU の外へ逃がす) → ZeRO++ (通信圧縮) → PyTorch FSDP (フレームワークネイティブ化)」と発展しました。

### 論文の実験結果 (定量データ)

ZeRO 論文 (SC 2020) の主要な実測です。指標は学習可能な最大モデルサイズ、throughput (1 GPU あたりの TFLOPS や 1 秒あたりの処理量)、scaling efficiency です。

- 400 台の V100 GPU を用いて 100B (1000 億) パラメータのモデルを学習し、1 GPU あたり 30+ TFLOPS、システム全体で 15 PetaFLOPS 級を達成。当時の DDP では数十億パラメータが限界だったのに対し、扱えるサイズの桁を 1〜2 つ押し上げました。V100 の理論ピーク (fp16 で約 125 TFLOPS) に対し 30 TFLOPS は約 1/4 で、超大規模としては良好な利用率です。
- super-linear scaling の観測: Stage が進むほど 1 台あたりメモリが減るため、台数を増やすと「1 台あたりに乗るモデル断片が小さくなり batch を増やせる」効果で、理想を超える高速化 (super-linear、台数を 2 倍にして 2 倍以上速くなる) が一部の領域で観測されました。これは「メモリが減ること自体が速度に効く」ことを示す象徴的な結果です。
- Stage 別の効果: Stage 1 だけで DDP 比 約 4 倍のモデルサイズ、Stage 2 で約 8 倍、Stage 3 で台数 N に比例して上限が伸びることを実測。

ZeRO-Offload (ATC 2021) のアブレーション的知見: optimizer 計算を CPU に逃がしても、CPU 側を効率化 (CPU 用に最適化した Adam 実装、CPU-GPU 転送と GPU 計算のオーバーラップ) すれば、1 台の V100 (32GB) で 13B モデルを学習でき、しかも GPU は forward/backward に集中できるため GPU 利用率を高く保てることを示しました。素朴に全部 CPU に置く naive offload より大幅に速いことも示され、「どこに何を逃がすと全体スループットが最大化するか」を定量的に詰めた研究です。

指標の意味の補足: TFLOPS は GPU の計算能力の使用量で、メモリを削っても TFLOPS が高く保てるなら「節約と速度を両立できている」ことを意味します。super-linear scaling は通常ありえない (通信が増えるので普通は sub-linear、台数を 2 倍にしても 2 倍未満) ため、これが出るのはメモリ削減が batch 拡大を通じて速度に転化した特殊で重要な現象です。

### メリット・トレードオフ・限界

メリット
- optimizer 状態・勾配・パラメータの冗長コピーを排除し、1 台あたりのメモリを劇的に削減
- データ並列の使いやすさ (コード変更が少ない) を保ったまま超大規模モデルを学習可能
- ステージを選べるので、メモリと通信のトレードオフを調整できる (Stage 1 は少ない代償で大削減)
- Offload/Infinity を使えば少ない GPU でも巨大モデルに手が届く

トレードオフ・限界
- ステージが進むほど通信量が増え (特に Stage 3 で約 1.5 倍)、ネットワーク帯域がボトルネックになりやすい
- Offload 系は CPU/NVMe 転送が遅く、スループットが大きく落ちる (メモリと速度の交換)
- 小〜中規模モデルでは通信オーバーヘッドだけ増えて損をする (大モデル専用の道具)
- DeepSpeed 固有の設定・依存が増え、state の保存・復元やデバッグが複雑になる

研究上の未解決課題: ZeRO はデータ並列の枠内でメモリを削るアプローチで、単独では超大規模に届かず、テンソル並列・パイプライン並列との組み合わせが必要です。各並列軸の最適配分、通信トポロジ、offload の階層配置をどう自動最適化するかは依然オープンな問題です。また Stage 3 の通信増を量子化なしでどこまで隠せるか、帯域が世代ごとに伸びる中でステージ選択の最適点がどう動くかも継続的な検討対象です。

### 発展トピック・研究の最前線
- ZeRO++: 通信を量子化して all-gather/reduce-scatter のデータ量を減らし、帯域の細い環境でのスループットを改善する派生。重みを fp8 級に圧縮して集める
- 3D parallelism: ZeRO (データ並列) × テンソル並列 × パイプライン並列を組み合わせる構成 (Megatron-DeepSpeed)。実際の超大規模 LLM 学習で採用される。ZeRO は外側のデータ並列軸を担う
- FSDP との収束: PyTorch FSDP が ZeRO Stage 3 相当をネイティブ提供するようになり、「DeepSpeed ZeRO を使うか、PyTorch FSDP を使うか」が実装上の選択になった ([FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training))
- メモリ削減の他軸との併用: [activation 再計算](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning) が activation を削るのに対し ZeRO はモデル状態を削るので、両者は直交的に併用して相乗効果を出す。[混合精度](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training) とも独立に効く

### さらに学ぶための関連トピック
- [FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/02_trajectory-planning)
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/11_optimization-training)
