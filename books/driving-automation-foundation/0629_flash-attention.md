---
title: "FlashAttention"
---

# FlashAttention

## ひとことで言うと
FlashAttention とは、Transformer の中核である attention (注意機構)の計算を、GPU のメモリ事情を考慮して賢く組み替えることで、メモリ使用量を大幅に減らし同時に高速化する計算手法(GPU カーネル)です。出力する答えは普通の attention とまったく同じ(厳密、exact)ですが、途中で現れる巨大な中間行列を GPU の遅いメモリに書き出さずに済ませるのが鍵です。「IO-aware(入出力=メモリの読み書きを意識した)」設計と呼ばれます。

## 直感的な理解

膨大な答案を採点する場面を想像してください。全員ぶんの答案を一度に机に広げる(=巨大な行列を全部メモリに置く)と机からあふれます。代わりに「一束ずつ取り出して採点し、途中集計だけ手元に残して束を片付ける」方式にすれば、机が小さくても全員ぶん採点でき、しかも答案の出し入れ(=遅いメモリへの往復)が減ります。FlashAttention はまさにこの「一束ずつ処理して途中集計を持ち回る」やり方を attention に適用したものです。結果(最終的な採点)は全部広げた場合と完全に一致します。

なぜ重要かというと、attention は系列が長くなるほど中間データが爆発的に増え、これが Transformer を重くする最大の原因だからです。そして実は、その重さの正体は計算量よりも「遅いメモリへの読み書き回数」にあります。FlashAttention はそこを直撃します。FLOP(浮動小数点演算回数)を1つも減らさずに、メモリトラフィックだけを減らして実時間を縮める、という点が他の高速化と決定的に異なる発想です。

## 基礎: 前提となる概念

### attention とは何か
Transformer の attention は「系列の中の各要素が、他のどの要素にどれだけ注目すべきか」を計算する仕組みです。文章中の単語どうし、画像中のパッチ(小領域)どうしの関係を測ります。各要素から Query(質問)Q、Key(鍵)K、Value(値)V という3つのベクトルを作り、次を計算します。

```
S = (Q × Kの転置) / √d   … 全要素ペアの類似度スコア (行列)。√d でスケール
P = softmax(S)            … 各行を確率(注目度)に変換
O = P × V                 … 注目度で重み付けした値の合計
```
ここで d は各ベクトルの次元数で、√d で割るのは内積が大きくなりすぎて softmax が尖りすぎるのを防ぐためです。softmax とは、数値の並びを「合計が1の確率分布」に変換する関数で、大きい値ほど大きな重みを与えます。

### N^2 の壁
S の大きさが問題です。系列長(要素数)を N とすると、S は N×N 行列になります。全ペアの組み合わせだからです。

数値例: 系列長 N=4096、ヘッド数 16(attention は複数の視点=ヘッドで並列に行う)の場合、S は 4096×4096 ≈ 1670万要素を16ヘッドぶん持ちます。fp16(2バイト)でも1ヘッド約33MB、全ヘッドで0.5GB超。同サイズの P も別に必要で、バッチ方向にも掛かります。N を倍の8192にすると N^2 なので4倍に膨れます。つまり attention のメモリと計算は系列長の2乗で増えます。これが長系列・高解像度で Transformer が重くなる根本原因です。

### 本当のボトルネックはメモリの往復
GPU には大容量だが遅い HBM(High Bandwidth Memory、VRAM、数十GB)と、計算ユニット隣の爆速だが極小の SRAM(数十〜数百KB)があります。両者の帯域差は1桁以上(HBM が毎秒1〜3TB、SRAM は実効で毎秒10〜20TB 級)あります。普通の attention 実装は巨大な S と P を HBM に書き出し、また読み戻します。この HBM 往復が、実は行列積の計算そのものよりずっと時間を食っています。GPU の演算ユニットは速すぎて、データの到着を待っている状態(メモリ律速、memory-bound)なのです。FlashAttention はこの無駄な往復を消すことを狙います。これが「計算量(FLOP)を減らす」のではなく「IO を減らす」という独自の着眼点です。アルゴリズムを評価するとき、FLOP ではなく HBM アクセスバイト数で良し悪しを測る、という見方を持ち込んだのが思想的な核心です。

## 仕組みを詳しく

### 巨大行列を作らず、ブロックごとに計算する (tiling)
FlashAttention の核心は「N×N の S と P を一度も全体としては作らない」ことです。Q, K, V を小さなブロック(タイル)に分け、SRAM に載るぶんだけを順番に処理します。

```
1. Q を行ブロックに、K, V を列ブロックに分割
2. 各 Q ブロックについて、K, V ブロックを1つずつ SRAM に読み込み
3. その小ブロックぶんだけ S, softmax 寄与, V との積を SRAM 上で計算
4. 結果を「走りながら」足し込む (オンラインで合算)
5. 巨大な S, P は HBM に一度も書かない。出力 O だけを HBM に書く
```
これにより HBM への往復回数が劇的に減り、メモリ使用量も O(N^2) から O(N) (系列長に線形)に下がります。理論解析では、SRAM サイズを M とすると HBM アクセスが O(N^2 d^2 / M) となり、素朴な実装の O(N^2 + N d) より大幅に小さい(d^2/M が 1 より十分小さい現実設定で効く)ことが示されています。

### online softmax (分割しても正しい softmax)
難所は softmax です。softmax は「行全体の合計で割る」操作なので、本来は行を全部見ないと正規化できません。ブロックに分けると全体が見えません。

FlashAttention は online softmax (Milakov & Gimelshein, 2018 が確立した手法)を使います。ブロックを処理するたびに「これまで見た最大値 m」と「これまでの指数和 l」を保持し、新しいブロックが来たら過去の部分結果をスケールし直して足し込みます。

```
新ブロックの最大値が過去より大きければ、
  過去の累積和 l と累積出力 O を exp(m_old - m_new) 倍して縮め、
  新ブロックぶんを加える
```
最大値を引いてから exp を取るのは数値安定化(オーバーフロー防止)のためです。この補正係数 exp(m_old − m_new) を掛け直す操作(rescale)により、行全体を一度に持たなくても最終的に厳密に同じ softmax 結果が得られます。近似ではない点が決定的に重要です。理論的には Rabe & Staats (2021) が「self-attention は O(n^2) メモリを必要としない」ことを先に示しており、FlashAttention はこれを IO 最適なカーネルとして実装で具現化したものです。

### 逆伝播 (backward) の工夫
学習では勾配計算が必要ですが、ここで普通なら順伝播で作った P を保存して使い回します。しかし FlashAttention は P を保存しません。代わりに、保存しておいた小さな統計量(各行の最大値 m と正規化定数 l をまとめた logsumexp)から逆伝播時に P を必要なブロックだけ再計算します。これを recomputation(再計算)と呼びます。計算を少し増やす(FLOP は増える)代わりにメモリを大幅に節約し、しかも増えた再計算ぶんが SRAM 上で完結するため、HBM 律速の世界では実時間も速くなる、という一石二鳥です。典型的な計算とメモリのトレードオフが、IO 律速ゆえに「両得」に転じる好例です。

### before / after まとめ
```
                       普通の attention        FlashAttention
N×N 行列 S, P の保存    HBM に書く (N^2 メモリ)   作らない (SRAM 上で消費)
メモリ使用量           O(N^2)                   O(N) (線形)
HBM 往復               多い (律速の原因)         少ない (O(N^2 d^2/M))
FLOP                   同じ                     同じ (むしろ再計算で微増)
出力の正しさ           -                        まったく同じ (厳密)
速度                   遅い                     数倍速い
```

### 使い方
専用カーネルなので自分で実装する必要はありません。
```python
# PyTorch 2.x: 条件が合えば内部で FlashAttention 系カーネルを自動選択
out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
```
PyTorch の SDPA はハードと入力条件(対応 GPU、対応 dtype、head_dim 制約など)が合えば内部で FlashAttention カーネルを自動選択し、合わなければ memory-efficient 版や数式どおりの実装にフォールバックします。`flash-attn` ライブラリを直接呼ぶ方法もあります。

## 手法の系譜と主要論文

- Milakov & Gimelshein, "Online normalizer calculation for softmax" (NVIDIA, 2018): ブロックを逐次処理しながら厳密な softmax を計算する online softmax を確立。FlashAttention の数学的土台。
- Rabe & Staats, "Self-attention Does Not Need O(n^2) Memory" (Google, 2021): attention を線形メモリで計算できることを理論と実装(JAX)で示した先駆け。ただし実速度の最適化よりメモリ削減の理論証明が主眼でした。
- Dao, Fu, Ermon, Rudra, Ré, "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (NeurIPS 2022, スタンフォード): tiling + online softmax + recomputation を IO 最適な単一 GPU カーネルにまとめ上げ、近似なしで実速度を数倍にした決定版。従来研究が FLOP 削減の近似に注力していたのに対し、真のボトルネックは HBM の IO だと見抜いた点が新規性。
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (2023): GPU 内のスレッドブロック/warp への作業分割を見直し、非行列積演算(rescale など)を減らして行列積の比率を上げ、系列長方向にも並列化(causal でも負荷を均す)。GPU 利用率を大きく改善。
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (NeurIPS 2024, NVIDIA H100 向け): Hopper 世代の非同期実行(warp-specialization で行列積とソフトマックスを別 warp で重ねる、TMA による非同期コピー)と FP8 低精度を活用。
- 対照的に Linformer (Wang et al., 2020) や Performer (Choromanski et al., 2020)、Reformer (Kitaev et al., 2020) は attention を低ランク/カーネル/ハッシュで線形化しますが、これらは答えが近似で精度劣化リスクがあります。FlashAttention の価値は「厳密な答えを保ったまま IO だけ減らす」点で、近似系とは思想が異なります。

## 論文の実験結果 (定量データ)

FlashAttention (NeurIPS 2022) の主要な結果。

- BERT-large (系列長512) の学習を、当時の最速記録(MLPerf 1.1 の NVIDIA 実装)比でおよそ15%短縮(end-to-end の壁時計時間)。
- GPT-2 の学習で、HuggingFace 実装比で最大約3倍、Megatron-LM 実装比で約1.8倍の高速化を報告(系列長512〜1K)。
- 系列長を伸ばせることを活かし、Long-Range Arena(長系列ベンチマーク)で標準 Transformer より速く、近似 attention 系と比べて速度・精度ともに優位を示した。
- メモリが O(N) に線形化するため、同じ GPU で扱える系列長が大幅に伸び、Path-X(系列長16K)・Path-256(系列長64K)など従来 Transformer がメモリ不足で解けなかった長系列タスクで初めて学習が回り、chance(でたらめ)を上回る精度を達成。これは速度向上を超えた質的変化です。
- attention 単体のマイクロベンチで、標準実装比で最大7.6倍(系列長が長いほど比が伸びる)。

数値の意味: 学習3倍速とは、同じ精度のモデルを3分の1の時間・コストで得られることを意味し、大規模学習で極めて大きな差です。さらに「メモリ線形化により長系列が初めて扱える」というのは単なる速度向上を超えて、それまで不可能だった規模の問題を可能にする質的な変化です。

アブレーション的観察: 論文はブロックサイズ(タイルの大きさ)を変えた感度分析を行い、SRAM 容量に合わせた適切なブロックサイズで HBM アクセスが最小化されることを示しています。ブロックが小さすぎると並列度・再利用が落ち、大きすぎると SRAM に載らずあふれます。FlashAttention-2 では、この作業分割をさらに最適化し、A100 で理論ピーク FLOP の50〜70%程度という、attention としては極めて高い GPU 利用率を達成(FlashAttention-1 のおよそ2倍速、標準実装比で最大9倍)。FlashAttention-3 は H100 で FP16 でおよそ1.5〜2.0倍(FA2 比)、FP8 ではほぼ1.2 PFLOP/s 近くに達し、低精度ながら誤差は標準的なベースライン量子化より小さいと報告されています。

## メリット・トレードオフ・限界

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

## 発展トピック・研究の最前線
- PagedAttention / vLLM: 推論時の KV キャッシュをページ単位で管理し、長系列バッチ推論のメモリ断片化を抑えてスループットを上げる流れ(FlashAttention と相補的)。
- 低精度 attention(FP8/FP4)と量子化推論の組み合わせ。
- 高解像度ビジョンや BEV(鳥瞰図)表現のように系列長が数万規模になる空間的 attention での適用。
- スパース attention / sliding window / Flash Decoding(推論時の系列方向並列)との融合(長文脈 LLM 向け)。
- ハードウェアとアルゴリズムの協調設計(IO-aware の思想を畳み込みや正規化など他の演算へ展開する研究)。

## さらに学ぶための関連トピック
- [torch.compile](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0648_pytorch-compile)
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0133_mixed-precision-training)
- [Transformer 時間方向アテンション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0344_transformer-temporal-attention)
- [鳥瞰図特徴融合（BEVFormer風 空間クロスアテンション）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0299_bev-fusion)
