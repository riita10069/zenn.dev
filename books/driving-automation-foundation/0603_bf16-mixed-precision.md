---
title: "bf16 Mixed Precision"
---

# bf16 Mixed Precision

## ひとことで言うと
ニューラルネットの学習中、計算に使う数値を「32ビットの高精度」ではなく「16ビットの軽い形式 bfloat16」に切り替えて、メモリを減らし計算を速くするテクニックです。bfloat16 は表現できる数の「範囲 (ダイナミックレンジ)」が32ビットと同じくらい広いので、float16 で必要だった面倒な調整 (損失スケーリング / GradScaler) なしで、ほぼそのまま安全に学習できるのが最大の特徴です。新しい世代の GPU や TPU で標準的に使われています。

## 直感的な理解
モデルが大きくなると、学習に2つの困りごとが出ます。一つはメモリ不足。重み・勾配・最適化器の状態・中間結果 (activation) をすべて GPU メモリ (VRAM) に置く必要があり、32ビットだと容量が足りずクラッシュ (OOM = Out Of Memory) します。もう一つは遅さ。GPU には16ビット計算を高速にこなす専用回路 (Tensor Core 等) があり、32ビットのままだとこの専用回路を活かせません。

そこで「全部を16ビットにするのではなく、速度が欲しい行列計算は16ビット、精度が要る一部は32ビットのまま」という使い分け = 混合精度 (Mixed Precision) が生まれました。問題は「どの16ビット形式を使うか」です。float16 は範囲が狭く、学習中に出る極小の勾配がゼロに潰れて学習が止まるため、勾配を一時的に拡大する仕掛け (GradScaler) が必須でした。bfloat16 はこの範囲を float32 並みに広げることで、仕掛けなしで安全に使えるようにした形式です。つまり bfloat16 は「float16 の高速・省メモリは欲しいが、面倒は避けたい」という要求への答えです。

## 基礎: 前提となる概念

### 浮動小数点数とは
コンピュータは小数を「浮動小数点数 (floating point)」で表します。1個の数値を3つの部分で構成します。

- **符号 (sign): 1ビット**。プラスかマイナスか。
- **指数部 (exponent)**。数の「大きさのスケール」を決める。これが広いほど、ものすごく大きい数や小さい数まで表せる = ダイナミックレンジが広い。指数部は「2の何乗か」を表すので、ビットが1つ増えると表せる範囲が指数的に広がる。
- **仮数部 (mantissa)**。数の「細かさ・有効ケタ数」を決める。これが多いほど、隣り合う表現可能な数の間隔が細かくなる = 精度が高い。

ディープラーニングで標準だったのは float32 (32ビット、FP32) で、1個の数値に4バイト使います。

### float16 と bfloat16 の違い
16ビット形式は2種類あり、同じ16ビットを指数部と仮数部にどう配分するかが違います。

| 形式 | 全ビット | 指数部 | 仮数部 | ダイナミックレンジ (おおよそ) | 細かさ |
|---|---|---|---|---|---|
| float32 (FP32) | 32 | 8 | 23 | 1e-38 〜 3e38 | 高い |
| float16 (FP16) | 16 | 5 | 10 | 6e-5 〜 65504 | そこそこ |
| bfloat16 (BF16) | 16 | 8 | 7 | 1e-38 〜 3e38 | 低い |

ポイントは指数部です。bfloat16 は指数部8ビットで float32 と同じ。つまり表せる数の範囲が float32 と同等です。実際、float32 の上位16ビットをそのまま切り取った形が bfloat16 で、これは「float32 とは指数部の構造が完全に同じ、下位の細かいビットを捨てただけ」という意味になります。だから float32 ↔ bfloat16 の変換が単純なビット切り捨てで済むという実装上の利点もあります。float16 は指数部5ビットしかなく、65504 より大きい数や 6e-5 より小さい数を表せません。bfloat16 は範囲を取った代わりに仮数部が7ビットと少なく、細かい桁の精度は float16 より劣ります。

### なぜ範囲が重要か — アンダーフロー
学習では「勾配 (gradient)」= 重みをどう動かせば誤差が減るかを表す値を計算します。学習が進むと勾配はしばしば 1e-7 のような極小値になります。float16 でこれを表そうとすると最小値 6e-5 を下回り、ゼロに丸められます。これを「アンダーフロー (underflow)」と呼びます。勾配がゼロになると重みが更新されず学習が止まります。float16 ではこれを防ぐため、損失を大きく拡大してから逆伝播する GradScaler が必須でした (詳細は [fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0630_fp16-amp-gradscaler))。bfloat16 は範囲が広いのでアンダーフローがほぼ起きず、GradScaler 不要で済みます。

ここに「学習における精度と範囲のどちらが効くか」という重要な経験則があります。Kalamkar らの研究が定量的に示したのは、学習の安定性に効くのは「細かさ (仮数部)」より「範囲の広さ (指数部)」だという点です。勾配は大小に何桁も振れるので、それを潰さず表せる範囲のほうが、各値の細かい精度より大事だ、という直感です。bfloat16 はこの直感に沿って指数部を優先した形式です。

## 仕組みを詳しく

### 自動混合精度 (AMP) の流れ
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

### 重みは float32、計算は bf16 — なぜ両方持つのか
混合精度の要は「重みの float32 マスターコピー」を持つことです。重み更新 `w ← w - lr * grad` で、`lr * grad` が極めて小さいと、16ビットの `w` に足しても丸めで消えてしまいます。これは「大きな数に微小な数を足すと吸収される」という浮動小数点の性質です。具体例: bfloat16 で `1.0` の次に表せる数は約 `1.0078` (仮数部7ビットなので刻みが粗い)。もし更新が `0.001` だと、`1.0 + 0.001 = 1.001` は最も近い表現に丸められて `1.0` のまま、つまり更新が完全に消えます。float32 で重みを保持し更新もそこで行えば (float32 では `1.0` の次の刻みは約 `1.0000001`)、この消失を防げます。forward/backward だけ bf16 にして速度とメモリを稼ぐ、というのが基本構図です。

### 数値例: メモリと速度
あるテンソルが形状 `[4, 256, 450, 300]` (batch=4、チャネル256、高さ450、幅300) だとします。要素数は `4×256×450×300 = 138,240,000` 個。

- float32: `138.24M × 4バイト ≈ 553MB`
- bfloat16: `138.24M × 2バイト ≈ 276MB`

中間の activation (各層の出力。逆伝播のために保持される) がこの調子で半分になるので、実際の VRAM 削減は数 GB 単位になります。高解像度の特徴マップを多数保持するモデル (例えば俯瞰図 = BEV のような空間表現を作る知覚モデル) では、bf16 によって同じ GPU で扱える解像度やバッチサイズが大きく変わります。速度面でも、新しい GPU の Tensor Core は bf16 行列積を float32 の数倍のスループット (理論ピークで2〜数倍) で処理します。

なお、混合精度はメモリを「半分」にはしません。最適化器の状態 (Adam なら運動量と分散の2つ) は通常 float32 で持ち、forward/backward の activation が主に削れる部分です。よって全体の削減率はモデル構成に依存し、activation が支配的なモデルほど効果が大きくなります。

## 手法の系譜と主要論文
- **2017/2018年 — 混合精度の確立**: Micikevicius ら "Mixed Precision Training" (ICLR 2018、NVIDIA/Baidu、arXiv:1710.03740) が float16 を使う枠組み (float32 マスターコピー + 損失スケーリング + float32 累積) を確立。混合精度の出発点です。詳細は [fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0630_fp16-amp-gradscaler)。
- **2019年 — bfloat16 の提案・検証**: Kalamkar, Mudigere, Mellempudi ら "A Study of BFLOAT16 for Deep Learning Training" (Intel/Google、arXiv:1905.12322) が、指数8ビット・仮数7ビットの bfloat16 を体系的に検証。「float16 は損失スケーリングが要るが、bfloat16 は float32 と同じ指数幅なのでスケーリングなしで学習できる」ことを示しました。
- **2019年 — 業界標準化**: Wang & Kanwar "BFloat16: The secret to high performance on Cloud TPUs" (Google) が、Google TPU が bfloat16 を標準採用した背景を解説。これと NVIDIA の Ampere 世代 (A100、2020) が bf16 を Tensor Core でネイティブ対応したことで、bf16 は学習のデファクト標準形式になりました。
- **2022年 — 最適化器状態の量子化**: Dettmers ら "8-bit Optimizers via Block-wise Quantization" (ICLR 2022) が、混合精度では float32 で残しがちな Adam の最適化器状態 (運動量・分散) を8ビットに圧縮し、精度をほぼ保ったままメモリをさらに削れることを示しました。bf16 と組み合わせる省メモリ手法として広く使われています。
- **2022年以降 — さらに低ビットへ**: Micikevicius ら "FP8 Formats for Deep Learning" (arXiv:2209.05433) が8ビット浮動小数点 (E4M3 / E5M2) を提案。Hopper/Ada 世代以降の GPU が FP8 を Tensor Core で対応し、巨大モデルの学習・推論で使われ始めました。bf16 → fp8 が低精度化の次の段階です。

## 論文の実験結果 (定量データ)
- **bfloat16 の精度同等性 (Kalamkar et al., 2019)**: 画像分類 (ResNet-50 on ImageNet)、物体検出、機械翻訳、音声、推薦、GAN など幅広いタスクで、bfloat16 学習が float32 とほぼ同等の最終精度に到達することを示しました。例えば ResNet-50/ImageNet で、bf16 学習の top-1 精度 (1000クラスで最も確信したクラスが正解と一致する割合) は float32 とコンマ数 % 以内の差にとどまる、と報告。意味: 「半分のビット幅でも、最終的な賢さはほとんど落ちない」。しかも float16 と違い損失スケーリングのチューニングが不要です。
- **混合精度の速度・メモリ効果 (Micikevicius et al., ICLR 2018)**: ImageNet 分類・物体検出・音声認識・翻訳・GAN などで、float32 と同等精度を保ちつつ、メモリ使用量を約半減し、Tensor Core 対応 GPU で大幅な高速化を達成。「同等精度・半分メモリ・高速」が要点です。
- **アブレーション的な知見**: float16 で損失スケーリングを「外す」と、勾配のヒストグラムの大部分が float16 の表現可能範囲を下回り (アンダーフロー)、学習が発散・停滞します (= スケーリングが必須)。一方 bfloat16 では同じ実験でスケーリングを外しても学習が安定する — これが bf16 の核心的な利点を示すアブレーションです。逆に bf16 で仮数部が7ビットしかないことの影響を見ると、大きな値の総和や正規化統計など一部の演算で精度劣化が出るため、それらは float32 に残すのが安定 (autocast はこれを自動でやっている)。
- **8-bit 最適化器 (Dettmers et al., 2022)**: GLUE/言語モデル/画像分類など多様なタスクで、Adam の状態を8ビット化しても32ビット Adam とほぼ同等の最終性能を保ち、最適化器が占めるメモリを大きく削減できることを報告。bf16 と直交する省メモリ手段として価値が示されました。
- **FP8 の限界 (Micikevicius et al., 2022)**: FP8 では多くのタスクで bf16 同等精度を出せるが、レンジ/精度がさらに厳しく、層ごとのスケーリングや慎重なキャスト設計が必要。bf16 ほど「何も考えず差し替え」とはいかない、というのが現状の知見です。

数値の意味: ここでの「精度差コンマ数 %」は、半分のメモリと数倍の速度という大きな利得に対して無視できるほど小さい、という点が重要です。だからこそ bf16 は標準になりました。

## メリット・トレードオフ・限界
メリット:
- VRAM をおおむね半減 (activation 部分) でき、より大きい batch やより高い解像度を同じ GPU で回せる。
- Tensor Core を使えるので行列積・畳み込みが高速化する。
- float16 と違いダイナミックレンジが float32 並みに広く、GradScaler が不要 → コードが単純で学習が安定。
- 多くのタスクで float32 と同等の最終精度が得られる。実験ループを短縮できる。
- float32 の上位ビットを切るだけの単純な形式のため、float32 ↔ bf16 変換が安価。

トレードオフ・限界:
- 仮数部が7ビットと少なく有効ケタが粗い。大きな値の総和・正規化統計・指数/対数など数値的に敏感な演算は精度低下が起こりうるため、autocast は内部でそれらを float32 に残す。
- bfloat16 をネイティブ対応する GPU/TPU (NVIDIA Ampere 以降、Google TPU など) が必要。古い GPU (Pascal/Turing 世代の一部、T4 など) では bf16 の恩恵が小さい/非対応で、float16 + GradScaler を使うことになる ([fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0630_fp16-amp-gradscaler))。
- メモリ削減には限界がある。さらに削るには [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing) (中間結果を捨てて再計算する)、8-bit 最適化器、[FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel) (重み・勾配・最適化器状態を複数 GPU に分割保持する) が必要。
- 推論時に bf16 をそのまま使うと、仮数部の粗さで出力がわずかに変わることがある。精度が重要な検証では float32 と突き合わせるのが安全。
- 未解決領域として、bf16 と FP8 を層ごとに使い分ける最適な混合精度方策、低精度下での学習安定性の理論的理解などが研究の途上にある。

## 発展トピック・研究の最前線
- **FP8 学習・推論**: E4M3 (精度寄り) と E5M2 (範囲寄り) を forward/backward で使い分ける手法。巨大言語モデルの事前学習で実用化が進む。
- **層ごと/テンソルごとのスケーリング**: 低ビットになるほど、テンソルごとに動的なスケール係数を持たせて範囲を合わせる手法が重要に。
- **重みの量子化 + 低精度最適化器状態**: Adam の最適化器状態 (運動量・分散) を8ビットに量子化してメモリを削る手法 (8-bit Adam など)。
- **TF32 などの中間形式**: NVIDIA の TF32 (内部で仮数部を削った行列積) は「コード変更なしで bf16 的な速度」を得る折衷案。
- **数値安定性の理論**: 低精度が収束に与える影響、確率的丸め (stochastic rounding) で低ビットの偏りを減らす手法など。

## さらに学ぶための関連トピック
- [fp16 AMP + GradScaler](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0630_fp16-amp-gradscaler)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing)
- [FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0604_ddp-distributed-data-parallel)
