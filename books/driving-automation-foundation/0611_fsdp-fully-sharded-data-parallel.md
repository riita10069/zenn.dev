---
title: "FSDP (Fully Sharded Data Parallel)"
---

# FSDP (Fully Sharded Data Parallel)

## ひとことで言うと
[DDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0604_ddp-distributed-data-parallel) は全 GPU にモデルを丸ごとコピーするのでメモリを大量に使い、1 台に入らない大きなモデルでは学習できません。FSDP は「モデルの重み・勾配・optimizer の状態を GPU 間で分割 (shard) して保持し、計算で必要になった瞬間だけ全 GPU から集めてくる」ことで、1 台あたりのメモリを大幅に減らす手法です。これにより、本来なら 1 台に収まらない大規模モデルでも、複数 GPU を束ねれば学習できるようになります。FSDP はデータ並列の一種であり続けます (各 GPU は違うデータを処理する)。違うのは「重みを全コピーで持つか、分割で持つか」だけです。

## 直感的な理解
4 人で 1 冊の分厚い百科事典を使って調べ物をする状況を想像してください。DDP のやり方は「全員が同じ百科事典を 1 冊ずつ持つ」です。確実ですが、本棚 (メモリ) を 4 冊ぶん占有します。本がもっと分厚くなれば、各人の本棚に 1 冊すら入らなくなります。

FSDP のやり方は違います。「百科事典を 4 分割し、各人は 1/4 だけ持つ。今 A の項目を調べたい人がいたら、その瞬間だけ全員から A のページを集めて完全な状態を作り、調べ終わったらまた手放す」。普段は各人の本棚は 1/4 しか使わないので、4 倍分厚い本でも扱えます。集める手間 (通信) は増えますが、調べる項目を順番に処理していけば、一度に集めるのは「今調べている数項目分」だけで済みます。

ニューラルネットの学習も同じです。ある層を計算する瞬間だけその層の重みを全部そろえ、終わったら手放す。これを層ごとに繰り返せば、各 GPU が常時保持するメモリは「モデル全体の 1/台数」で済みます。鍵は「常時保持する量」と「一瞬だけ集める量」を分けて考えることで、後者を小さく保てる限り、前者は台数分だけ薄くできるという発想です。

## 基礎: 前提となる概念

### 学習で消費するメモリの 3 要素
1 台の GPU が学習中に保持する主なものは次の 3 つです。重みの個数を Ψ (プサイ) と書きます。

1. パラメータ (重み): Ψ 個
2. 勾配: Ψ 個 (各重みに 1 つの更新方向)
3. optimizer 状態: 最適化器が追加で持つ補助変数。Adam の場合、各重みについて移動平均 m と分散 v の 2 つ、さらに混合精度学習では float32 のマスターコピーも持つため、パラメータの数倍にふくらむ

具体例: 重みが 1 億個・float32 (1 個 4 バイト) なら、パラメータだけで約 400MB。Adam では加えて勾配 400MB、optimizer 状態 (m, v で) 約 800MB、計 1.6GB。これに forward の中間 activation が乗ります。重みが 10 億・100 億と増えると、optimizer 状態だけで GPU の VRAM を超えます。

### DDP の冗長性
DDP は全 GPU がこの 3 要素を「まるごと 1 セットずつ複製」します。4 台なら同じ optimizer 状態が 4 セット存在し、3 セットは完全に冗長です。台数を増やしても 1 台あたりのメモリは 1 ミリも減らず、扱えるモデルサイズの上限が変わりません。FSDP はこの冗長コピーを排除します。

### 関連する集合通信
- all-gather: 各 GPU が持つ「重みの 1/N」を全部集めて、その層の完全な重みを一時的に作る
- reduce-scatter: 各 GPU が計算した勾配を合算しつつ、合算結果を分割して「各 GPU の担当 1/N だけ」を配る。DDP の all-reduce が「全員が全合計を持つ」のに対し、reduce-scatter は「全合計を切り分けて配る」点が違い、勾配も分散保持できる

実は all-reduce = reduce-scatter + all-gather と分解できます。DDP はこの 2 つを一体で行って全員が全勾配を持ち、FSDP は forward/backward で all-gather を使ってパラメータを一時復元し、勾配は reduce-scatter だけで止めて分散保持する、と理解すると両者の関係が見えます。

## 仕組みを詳しく

### 1 層の forward の流れ (GPU 4 台)
1. 普段、ある層の重みは 4 分割され、各 GPU に 1/4 ずつ置かれている (定常メモリ = モデルの 1/4)
2. この層の forward 直前に all-gather で完全な重みを各 GPU に一時構築
3. 各 GPU が自分の担当データ (DDP と同じくデータも分割されている) に対してこの層の forward を計算
4. 計算が終わったら、一時的に集めた重みを即座に解放し 1/4 に戻す
5. 次の層へ。これを層ごとに繰り返す

backward も同様に、各層で all-gather により重みを集めて勾配を計算し、reduce-scatter で勾配を合算しつつ分散配置します。最後に各 GPU が「自分の担当 1/4 の optimizer 状態」を使って担当分の重みだけを更新します。注意すべきは、forward で all-gather したパラメータを backward でも再び all-gather する必要がある点で (forward 後に解放しているため)、これが FSDP の通信が DDP より多くなる根本理由です。

### 通信と計算のオーバーラップ
FSDP も DDP と同様、通信を計算で隠します。具体的には、層 i を計算している最中に、次の層 i+1 の all-gather (prefetch = 先読みで集めておくこと) を裏で先に始めておきます。これにより all-gather の待ち時間を計算時間に重ねられます。backward では「逆方向の prefetch」、すなわち次に backward する層のパラメータを先に集めます。この prefetch が効くかどうかが FSDP のスループットを大きく左右し、prefetch を切ると通信が計算に隠れず大幅に遅くなります。

### wrap の粒度 (sharding の単位)
「どのまとまりで分割し、どのタイミングで all-gather するか」を決めるのが wrap の粒度です。FSDP では分割の最小単位を「FSDP unit」と呼び、unit ごとに all-gather/解放が起こります。

- 細かく wrap (各層を個別に unit にする): all-gather する単位が小さく、一時メモリは少ないが通信回数が増える
- 粗く wrap (複数層をまとめて 1 unit にする): 一時メモリは増えるが通信回数が減りオーバーラップしやすい

Transformer なら「各ブロックを 1 unit として wrap」するのが定番です。粒度はメモリと通信のトレードオフを決める主要なチューニング項目で、`auto_wrap_policy` で「パラメータ数がしきい値を超えたら自動で unit を区切る」といった指定ができます。

### sharding strategy の段階
何を分割するかには段階があり、これは [ZeRO](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer) のステージと対応します。

- FULL_SHARD: パラメータ + 勾配 + optimizer 状態すべてを分割。最もメモリ効率が高い (ZeRO Stage 3 相当)
- SHARD_GRAD_OP: 勾配と optimizer 状態だけ分割し、パラメータは複製 (ZeRO Stage 2 相当)。パラメータを集め直す all-gather が要らない分だけ通信は減るが、パラメータのメモリは削れない
- HYBRID_SHARD: ノード内では FULL_SHARD、ノード間では複製。遅いノード間通信 (all-gather/reduce-scatter) を減らしつつメモリも削る折衷で、multi-node で実用的

### メモリ削減の数値イメージ
重み 1 億個・Adam・GPU 4 台。DDP では各 GPU が optimizer 状態 800MB を持ちますが、FSDP FULL_SHARD では 800MB ÷ 4 = 200MB。パラメータ・勾配も同様に 1/4。一時的に集めるのは「同時に計算する 1〜数 unit 分」だけなので小さく済みます。台数 N を増やすほど 1 台あたりの定常メモリは 1/N に近づきます。

```
DDP (各GPU):    [param 400MB][grad 400MB][opt 800MB]  = 1.6GB 固定 (台数で減らない)
FSDP (各GPU):   [param 100MB][grad 100MB][opt 200MB]  = 0.4GB + 一時 all-gather 分
                 ↑ N=4 で各要素 1/4
```

ただし「一時 all-gather 分」を忘れてはいけません。一度に集める unit の最大サイズが大きいと、定常メモリが小さくてもピークでそこに乗り、OOM します。粒度のチューニングはこのピークを抑えるためでもあります。

### 典型的な実装
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

## 手法の系譜と主要論文

- Rajbhandari ら, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020, Microsoft, arXiv:1910.02054)。FSDP の理論的源流。optimizer 状態・勾配・パラメータの冗長保持を段階的に排除する ZeRO ステージを提案しました。詳細は [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer)。

- Rasley ら, "DeepSpeed" (KDD 2020, Microsoft)。ZeRO を実装したライブラリで、PyTorch ネイティブ FSDP 登場以前から大規模分散学習の事実上の標準でした。FSDP はしばしば DeepSpeed ZeRO Stage 3 と比較されます。

- FairScale FSDP (Meta, 2021)。PyTorch ネイティブ FSDP の前身となる実験的実装で、ZeRO Stage 3 のアイデアを PyTorch の autograd と統合する初期の試みでした。ここでの知見が後に PyTorch コアへ取り込まれます。

- Zhao, Gu, Varma ら, "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (VLDB 2023, Meta, arXiv:2304.11277)。本トピックの中心論文。ZeRO のアイデアを PyTorch コアに統合し、(1) deferred initialization (巨大モデルを物理メモリに乗せずにメタデバイス上で初期化してから分割配置)、(2) 通信と計算のオーバーラップ・prefetch、(3) 混合精度との統合、(4) 柔軟な wrap ポリシーを実装したことを報告。数千億パラメータ規模を限られた GPU メモリで学習可能にしました。

系譜としては「DDP (冗長だが単純) → ZeRO/DeepSpeed (冗長を消すアイデアと実装) → FairScale FSDP (PyTorch への移植実験) → PyTorch FSDP (フレームワークネイティブに、使いやすく) → FSDP2 (per-parameter で更に柔軟に)」という流れです。

## 論文の実験結果 (定量データ)

PyTorch FSDP 論文 (VLDB 2023) の主な実測です。指標は主に TFLOPS/GPU (1 GPU が 1 秒に実行する浮動小数点演算数で、計算資源をどれだけ使い切れているかを表す。理論ピークに近いほど効率がよい) と、学習可能な最大モデルサイズです。

- A100 GPU クラスタで、GPT 系の大規模モデルを学習し、175B (1750 億) パラメータ級まで含む広いサイズ域で高い演算効率を達成。報告では多くの設定で 100+ TFLOPS/GPU 台、条件次第で A100 の理論ピーク (bf16 で約 312 TFLOPS) の 5〜6 割程度の利用率を維持しました。これは「メモリを削りつつも計算機を遊ばせていない」ことを意味します。
- スケーラビリティ: モデルサイズと GPU 数を同時に増やしても、TFLOPS/GPU がほぼ一定に保たれる (near-constant) 領域が示され、weak scaling (規模に応じて資源も増やす) が良好であることを実証。数百 GPU 規模まで効率の崩れが小さいことが報告されています。
- DeepSpeed ZeRO Stage 3 との比較で、同等のメモリ効率を PyTorch ネイティブに、より少ない外部依存で実現できることを示しました。

アブレーション的な知見として重要なのは prefetch と wrap 粒度の効果です。all-gather の prefetch を無効にすると通信が計算に隠れず TFLOPS/GPU が顕著に低下します。また wrap 粒度を細かくしすぎると通信回数増で遅くなり、粗くしすぎると一時メモリがふくらんで OOM します。「通信を隠す工夫を抜くと効率が崩れる」ことが、FSDP がただの分割ではなくシステム最適化の塊であることを示しています。さらに HYBRID_SHARD は、ノード間帯域が細い環境で FULL_SHARD よりスループットが上がる場合があることも報告され、「メモリ最小化が常に最速ではない」という実務上重要な教訓を与えています。

指標の意味の補足: TFLOPS/GPU は速度の絶対量ではなく「GPU の計算能力をどれだけ使い切ったか」の指標です。同じ TFLOPS/GPU を大規模でも保てるなら、台数とモデルを増やしても効率が落ちないということで、これが大規模学習で最も重視される性質です。MFU (Model FLOPs Utilization、理論ピークに対する実効の割合) という形で報告されることも多く、大規模 LLM 学習では 40〜60% 程度が良好とされます。

## メリット・トレードオフ・限界

メリット
- パラメータ・勾配・optimizer 状態を分割するため、1 台あたりのメモリがおおむね 1/台数 に減る
- DDP では 1 台に入らない大規模モデルでも学習できる
- PyTorch ネイティブで、[混合精度](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision) や [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing) と自然に組み合わせられる (FSDP がパラメータ系を削り、checkpointing が activation を削る、と役割分担)

トレードオフ・限界
- all-gather / reduce-scatter の通信が DDP の all-reduce より多く (forward と backward の両方でパラメータを集め直す)、ネットワークが遅い環境ではスループットが落ちる。特にノードをまたぐ all-gather は帯域が細く効率が下がる (HYBRID_SHARD で緩和)
- wrap 粒度・prefetch・sharding strategy などチューニング項目が多く、DDP より導入が複雑
- 小〜中規模モデルでは、メモリに余裕があるのに通信オーバーヘッドだけ増え、DDP より遅くなる (大モデルでこそ効く)
- state_dict (重みの保存・復元) の扱いが分散しているため複雑。チェックポイントを full state にまとめる際の集約コストや、再開時のシャーディング整合に注意が要る

研究上の未解決課題: 「メモリは限界まで削れるが通信が増える」というトレードオフの最適点をどう自動で見つけるかは依然難しく、wrap ポリシーや並列構成の自動探索は研究途上です。また FSDP (データ並列の拡張) 単独では超大規模に届かず、テンソル並列・パイプライン並列との組み合わせ (3D parallelism) が必要になり、その構成探索も難問です。

## 発展トピック・研究の最前線
- FSDP2 (per-parameter sharding): パラメータを flat な 1 本のバッファにまとめて分割する旧方式に対し、パラメータ単位で分割し、量子化・部分凍結 (一部の層だけ学習) や LoRA との相性を改善
- ハイブリッド並列: FSDP × テンソル並列 × パイプライン並列。各並列軸を直交させて超大規模 LLM を学習する 3D parallelism。FSDP をテンソル並列の外側に置く構成が一般的
- 通信圧縮・量子化通信: all-gather する重みを低ビット (例: fp8) で送って通信量を削る試み。ZeRO++ の発想と共通
- offload との併用: パラメータや optimizer 状態を CPU/NVMe に逃がし、さらに大きいモデルを少ない GPU で扱う ([ZeRO-Offload/Infinity](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer) の発想)。FSDP も CPU offload オプションを持つ

## さらに学ぶための関連トピック
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0604_ddp-distributed-data-parallel)
- [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing)
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision)
