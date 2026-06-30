---
title: "ZeRO (Zero Redundancy Optimizer)"
---

# ZeRO (Zero Redundancy Optimizer)

## ひとことで言うと
複数 GPU で学習するとき、各 GPU が「同じもの (重み・勾配・optimizer 状態) を丸ごとコピーで持っている」のは大きな無駄です。ZeRO はこの冗長 (redundancy) をゼロに近づけ、それらを GPU 間で分担して持たせる技術です。これにより、本来なら 1 台に入りきらない超大規模モデル (数十億〜1 兆パラメータ) を、限られた数の GPU で学習できるようになります。Microsoft の DeepSpeed ライブラリの中核機能であり、PyTorch [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel) の直接の理論的源流でもあります。

## 直感的な理解
[DDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0604_ddp-distributed-data-parallel) の問題は「全員が全部を冗長に持つ」ことでした。GPU が 4 台なら、まったく同じ optimizer 状態が 4 セット存在し、3 セットはただの重複です。メモリの観点では「4 人で同じノートを 4 冊持っている」状態で、紙の無駄です。

ZeRO の発想は「ノートを 4 分割して各人 1/4 だけ持ち、必要なときだけ貸し借りしよう」です。普段の保持量が 1/4 になるので、4 倍大きなモデルが扱えます。ポイントは「何を分割すると、どれだけメモリが減り、どれだけ通信が増えるか」を段階的に選べることです。最も削減効果が大きいのは optimizer 状態 (Adam では最大の容量を占める) で、ここだけ分割しても通信はほとんど増えずに大きなメモリ削減が得られる、という「お得な順番」が ZeRO の核心の発見です。

もう 1 つ重要なのは、ZeRO がモデル並列 (テンソルを物理的に切る方式) ではなくデータ並列の枠内で動く点です。モデル並列は層の内部を切るのでモデルの実装を書き換える必要があり、層をまたぐたびに通信が要りますが、ZeRO は「データ並列のままメモリの冗長だけ消す」ので、ユーザーから見ればコード変更が最小で済みます。この「使いやすさを保ったままメモリの壁を破る」設計思想が、ZeRO が広く普及した理由です。

## 基礎: 前提となる概念

### 混合精度 + Adam のメモリ内訳
なぜ optimizer 状態がそんなに大きいのかを正確に押さえます。重みの個数を Ψ とし、混合精度学習 (計算は bf16/fp16、更新は fp32) で Adam を使う標準的な構成を考えます。1 パラメータあたりのバイト数は次の通りです。

- fp16 のパラメータ: 2 バイト
- fp16 の勾配: 2 バイト
- optimizer 状態 (fp32): マスターパラメータ 4 + Adam の m (1 次モーメント = 勾配の移動平均) 4 + Adam の v (2 次モーメント = 勾配の 2 乗の移動平均) 4 = 12 バイト

合計で 2 + 2 + 12 = 16 バイト/パラメータ。つまり 1 パラメータあたり 16Ψ バイトのうち、optimizer 状態だけで 12Ψ バイト (全体の 3/4) を占めます。15 億パラメータ (GPT-2 級) なら 16 × 1.5G = 24GB となり、1 台の GPU に乗せるだけでもギリギリ、複数台で学習しようにも DDP では全台が 24GB 持つので何も解決しません。なぜマスターコピーが fp32 で要るかというと、fp16 のままだと小さな更新量 (学習率 × 勾配) が丸め誤差で消えてしまい学習が進まないためで、更新は fp32 の高精度で蓄積する必要があるからです。

### DDP との対比
DDP は上記 16Ψ を全 GPU が複製します。台数を N にしても 1 台あたりは 16Ψ のまま。ZeRO は「同じものを N セット持つのをやめ、N 分割して各台 16Ψ/N に近づける」ことを目指します。ここに forward の中間 activation のメモリは含まれていない点に注意してください。activation を削るのは [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing) の役割で、ZeRO はパラメータ・勾配・optimizer 状態という「モデル状態 (model states)」だけを対象にします。両者は直交するので併用が前提です。

## 仕組みを詳しく

### 3 つのステージ
ZeRO は「何を分割するか」で 3 段階あります。後のステージほどメモリ削減が大きく、その分通信が増えます。GPU 数を N とします。

- ZeRO Stage 1 (optimizer 状態の分割, P_os): 最も大きい optimizer 状態 (12Ψ) だけを N 分割し、各 GPU は 1/N ずつ持つ。パラメータと勾配は従来通り全 GPU が複製。backward 後は通常通り勾配を集約するが、step では各 GPU が「自分が担当するパラメータ分」だけを更新し、更新後のパラメータを all-gather で全員に配り直す。少ない通信増で最大の削減が得られる「お得」なステージ
- ZeRO Stage 2 (+ 勾配の分割, P_os+g): Stage 1 に加えて勾配も N 分割。backward 後に reduce-scatter で「自分の担当分の勾配」だけを各 GPU に集約するので、全勾配を保持する必要がなくなる
- ZeRO Stage 3 (+ パラメータの分割, P_os+g+p): パラメータ本体も N 分割。計算する層の重みを必要なときだけ all-gather で集め、終わったら捨てる。最もメモリ効率が高く、PyTorch [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel) の FULL_SHARD はこれに相当する

### メモリ削減の数値 (論文の試算)
混合精度 + Adam で 1 パラメータあたり 16 バイトを基準に、N = 64 GPU のときおよそ次のように減ります (論文の図表より)。Ψ = 7.5B (75 億パラメータ) のモデルでは、これは GPU あたり 120GB が 1.9GB 程度まで落ちる計算に相当します。

```
ベースライン (DDP):  16Ψ バイト/GPU      (台数で減らない)
Stage 1:             4Ψ + 12Ψ/N ≈ 4Ψ    (optimizer 状態だけ 1/N)
Stage 2:             2Ψ + 14Ψ/N ≈ 2.6Ψ  (勾配も 1/N)
Stage 3:             16Ψ/N ≈ 0.25Ψ      (すべて 1/N、台数に反比例)
```

つまり Stage 3 では「GPU を増やせば増やすほど 1 台あたりの分担が減る」ため、原理的には台数を揃えれば任意のサイズのモデルが乗ります。一方で Stage 1 は、わずかな通信増 (step 後の all-gather だけ) で 16Ψ → 約 4Ψ という 4 倍の大削減が得られるので、コストパフォーマンスが極めて高いステージです。「とりあえず Stage 1 か 2 を試し、足りなければ Stage 3」というのが実務の定石です。

### 通信量のトレードオフ
重要なのは「Stage 1 と 2 は DDP と同等の総通信量で済む」という解析結果です。DDP の all-reduce は reduce-scatter + all-gather に分解でき (各 Ψ 分)、Stage 1/2 はこれを単に分解して使うだけなので、通信量はほぼ同じ (約 2Ψ) なのにメモリは激減します。一方 Stage 3 はパラメータの all-gather が forward と backward の両方で追加発生するため、総通信量がおよそ 3Ψ、つまり DDP の約 1.5 倍になります。「Stage 3 だけは通信代を払う」という構造で、ここが帯域の細い環境でのボトルネックになります。

### ZeRO-Offload と ZeRO-Infinity
派生として、状態を GPU メモリ以外に逃がす手法があります。

- ZeRO-Offload (ATC 2021): optimizer 状態と勾配、そして optimizer の更新計算そのものを CPU メモリ・CPU 演算に逃がす。GPU は forward/backward に専念。通信量の解析に基づき「fp16 の forward/backward は GPU、fp32 の重み更新は CPU」と役割を分けることで CPU-GPU 転送を最小化する設計。GPU が少ない環境 (極端には 1 台) でも大モデルを学習できるが、CPU-GPU 転送がボトルネックになる
- ZeRO-Infinity (SC 2021): さらに NVMe (SSD) まで階層的に使い、GPU メモリ → CPU メモリ → NVMe の 3 階層に状態を流す。限られたハードで桁違いに大きいモデルを扱えるが、SSD 帯域が律速でスループットは大きく落ちる。帯域を稼ぐための prefetch やオーバーラップが鍵

## 手法の系譜と主要論文

- Rajbhandari, Rajbhandari, Ruwase, He, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020, Microsoft, arXiv:1910.02054)。本トピックの原典。3 ステージを提案。動機は「DDP はメモリが冗長でモデルサイズが頭打ち、一方モデル並列 (テンソルを物理的に切る方式) は実装が難しく通信効率が悪い。データ並列の使いやすさを保ちつつメモリの冗長だけを消す」こと。理論上 1 兆パラメータ級を可能にすると示しました。

- Rasley, Rajbhandari, Ruwase, He, "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters" (KDD 2020, Microsoft)。ZeRO を実装したライブラリ DeepSpeed。既存の PyTorch コードに数行の JSON 設定で ZeRO を適用できる実用面を確立しました。

- Ren ら, "ZeRO-Offload: Democratizing Billion-Scale Model Training" (USENIX ATC 2021, arXiv:2101.06840)。CPU に optimizer 状態を逃がし、1 台の GPU でも 13B モデルを学習できることを実証。「大規模学習の民主化」を掲げました。

- Rajbhandari ら, "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" (SC 2021, arXiv:2104.07857)。NVMe まで使ってオフロードを拡張し、限られたハードで 30 兆パラメータ規模に手が届くと示した続編です。

- Wang ら, "ZeRO++" (2023, arXiv:2306.10209)。Stage 3 の弱点である通信量増を、重みの量子化 (all-gather する重みを低ビットに圧縮) や階層的な配置で削減し、帯域の細いクラスタでのスループットを改善しました。

系譜としては「DDP の冗長性 → ZeRO (冗長を段階的に消すという発見) → DeepSpeed 実装 → Offload/Infinity (GPU の外へ逃がす) → ZeRO++ (通信圧縮) → PyTorch FSDP (フレームワークネイティブ化)」と発展しました。

## 論文の実験結果 (定量データ)

ZeRO 論文 (SC 2020) の主要な実測です。指標は学習可能な最大モデルサイズ、throughput (1 GPU あたりの TFLOPS や 1 秒あたりの処理量)、scaling efficiency です。

- 400 台の V100 GPU を用いて 100B (1000 億) パラメータのモデルを学習し、1 GPU あたり 30+ TFLOPS、システム全体で 15 PetaFLOPS 級を達成。当時の DDP では数十億パラメータが限界だったのに対し、扱えるサイズの桁を 1〜2 つ押し上げました。V100 の理論ピーク (fp16 で約 125 TFLOPS) に対し 30 TFLOPS は約 1/4 で、超大規模としては良好な利用率です。
- super-linear scaling の観測: Stage が進むほど 1 台あたりメモリが減るため、台数を増やすと「1 台あたりに乗るモデル断片が小さくなり batch を増やせる」効果で、理想を超える高速化 (super-linear、台数を 2 倍にして 2 倍以上速くなる) が一部の領域で観測されました。これは「メモリが減ること自体が速度に効く」ことを示す象徴的な結果です。
- Stage 別の効果: Stage 1 だけで DDP 比 約 4 倍のモデルサイズ、Stage 2 で約 8 倍、Stage 3 で台数 N に比例して上限が伸びることを実測。

ZeRO-Offload (ATC 2021) のアブレーション的知見: optimizer 計算を CPU に逃がしても、CPU 側を効率化 (CPU 用に最適化した Adam 実装、CPU-GPU 転送と GPU 計算のオーバーラップ) すれば、1 台の V100 (32GB) で 13B モデルを学習でき、しかも GPU は forward/backward に集中できるため GPU 利用率を高く保てることを示しました。素朴に全部 CPU に置く naive offload より大幅に速いことも示され、「どこに何を逃がすと全体スループットが最大化するか」を定量的に詰めた研究です。

指標の意味の補足: TFLOPS は GPU の計算能力の使用量で、メモリを削っても TFLOPS が高く保てるなら「節約と速度を両立できている」ことを意味します。super-linear scaling は通常ありえない (通信が増えるので普通は sub-linear、台数を 2 倍にしても 2 倍未満) ため、これが出るのはメモリ削減が batch 拡大を通じて速度に転化した特殊で重要な現象です。

## メリット・トレードオフ・限界

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

## 発展トピック・研究の最前線
- ZeRO++: 通信を量子化して all-gather/reduce-scatter のデータ量を減らし、帯域の細い環境でのスループットを改善する派生。重みを fp8 級に圧縮して集める
- 3D parallelism: ZeRO (データ並列) × テンソル並列 × パイプライン並列を組み合わせる構成 (Megatron-DeepSpeed)。実際の超大規模 LLM 学習で採用される。ZeRO は外側のデータ並列軸を担う
- FSDP との収束: PyTorch FSDP が ZeRO Stage 3 相当をネイティブ提供するようになり、「DeepSpeed ZeRO を使うか、PyTorch FSDP を使うか」が実装上の選択になった ([FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel))
- メモリ削減の他軸との併用: [activation 再計算](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing) が activation を削るのに対し ZeRO はモデル状態を削るので、両者は直交的に併用して相乗効果を出す。[混合精度](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision) とも独立に効く

## さらに学ぶための関連トピック
- [FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0604_ddp-distributed-data-parallel)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing)
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision)
