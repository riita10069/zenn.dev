---
title: "DDP (Distributed Data Parallel)"
---

# DDP (Distributed Data Parallel)

## ひとことで言うと
1台の GPU では遅すぎる、あるいはデータが多すぎて時間がかかりすぎる学習を、複数の GPU に分散して速くする最も標準的な方法です。すべての GPU に「同じモデルのコピー」を置き、学習データを GPU 間で分割して各 GPU が独立に計算し、計算した勾配 (重みをどちらにどれだけ動かすべきかを表す量) を all-reduce という集合通信で合算・平均することで、全 GPU の重みを常に同一に保ちます。結果として「巨大な batch を複数 GPU で並列に処理する」のと数学的に等価になり、台数にほぼ比例して学習が速くなります。これがデータ並列 (data parallelism) と呼ばれる戦略で、DDP はその PyTorch における標準実装です。

## 直感的な理解
たとえば 100 万枚の画像を分類するモデルを学習するとします。1台の GPU は 1 秒間に処理できる画像枚数に上限があり、データセット全体を 1 周 (これを 1 エポックと呼びます) するのに何時間もかかります。学習は通常何十エポックも回すので、1台では数日かかることも珍しくありません。

ここで「分業」を考えます。GPU が 4 台あれば、データを 4 等分して各台に配り、4 台が同時に働けば理論上 4 倍速くなります。料理にたとえると、1 人で 100 人分の料理を作る代わりに、4 人で 25 人分ずつ分担するイメージです。

ただし単純な分業には落とし穴があります。各 GPU が違うデータで学習すると、求める勾配がバラバラになります。GPU0 は「この重みを増やせ」、GPU1 は「減らせ」と言うかもしれません。放置すると 4 台の重みがどんどんズレて、最終的に 4 つの別物のモデルになってしまいます。これでは分業の意味がありません。

そこで毎ステップ、4 台が出した勾配を「平均」して全員が同じ更新を適用します。こうすれば 4 台はずっと一卵性の双子のように同一の重みを保ち、「4 倍のデータを 1 台で処理した」のと完全に同じ結果になります。この「全員の勾配を集めて平均し、結果を全員に配り戻す」通信が all-reduce です。DDP の本質はこの一点に尽きます。

なぜ「データを分ける」のであって「モデルを分ける」のではないのか、という疑問が出るかもしれません。モデルそのものを切り分ける方式 (テンソル並列・パイプライン並列) もありますが、それらは層の内部や層の境界で頻繁に通信が必要で実装が難しく、通信効率も悪くなりがちです。データ並列は「1 ステップに 1 回まとめて勾配を交換するだけ」で済むので、最も単純で通信効率がよく、モデルが 1 台に収まる限り第一選択になります。

## 基礎: 前提となる概念

### ニューラルネット学習の 1 ステップ
ニューラルネットの学習は次の繰り返しです。

1. forward (順伝播): データを入力し、層を順に通して出力を出す
2. loss (損失): 出力と正解の差を 1 つの数値で測る
3. backward (逆伝播): その損失を各重みで微分し、勾配を得る。勾配は「損失を下げるには各重みをどちらにどれだけ動かせばよいか」を表すベクトル
4. step: optimizer (最適化器、Adam や SGD など) が勾配を使って重みを少しだけ更新する

この 4 つを何百万回も回して、損失が小さくなる重みを探します。

### batch (バッチ) という単位
1 ステップで一度に処理するデータのまとまりを batch と呼び、その枚数を batch size と言います。batch size を大きくすると、1 ステップで多くのサンプルから勾配を計算するので、勾配のノイズ (サンプルのばらつきによる推定誤差) が減って安定します。データ並列とは要するに「大きな batch を複数 GPU に切り分けて並列計算する」ことです。各 GPU が処理する分を local batch、全 GPU 合計を global batch (実効 batch) と呼びます。

### 集合通信 (collective communication) の用語
複数のプロセスが協調して値をやり取りする通信を集合通信と呼びます。本トピックで登場するのは主に次の 3 つです。

- broadcast: 1 つのプロセスの値を全プロセスに配る (初期重みを揃えるのに使う)
- reduce: 全プロセスの値を 1 つのプロセスに集めて演算 (合算など) する
- all-reduce: 全プロセスの値を集めて演算 (合算や平均) し、その結果を全プロセスに配り戻す。「reduce (集約)」を「all (全員に)」行うので all-reduce

### world size と rank
- world size: 参加するプロセス (通常 GPU 1 台 = 1 プロセス) の総数。4 台なら 4
- rank: 各プロセスに振られた通し番号 (0, 1, 2, 3)。rank 0 が代表 (チェックポイント保存やログ出力を担う) になることが多い
- local rank: 1 ノード内での GPU 番号。複数ノードにまたがる場合、global な rank とノード内の local rank を区別する

### ノードとインターコネクト
GPU が物理的にどう繋がっているかは性能に直結します。1 台のサーバ (ノード) 内の GPU は NVLink (NVIDIA の GPU 間直結バス、数百 GB/s) などで高速に繋がりますが、ノードをまたぐと InfiniBand や Ethernet (数十〜数百 Gb/s、バイト換算で 1 桁以上遅い) を通ります。「node 内は速い、node 間は遅い」というこの非対称性が、後述する scaling 効率の頭打ちの主因です。

## 仕組みを詳しく

### 1 ステップの流れ (GPU 4 台、global batch 256 の例)
1. 学習開始時、rank 0 の初期重みを broadcast で全 GPU にコピーし、4 台の重みを完全に一致させる
2. global batch 256 サンプルを 4 分割し、各 GPU に 64 サンプルずつ配る。データの分配は、サンプリングが重複しないよう各 rank に異なるインデックス集合を割り当てる仕組み (PyTorch では `DistributedSampler`) が担う
3. 各 GPU が自分の 64 サンプルで forward → loss → backward を実行し、ローカルな勾配を得る (この時点では 4 台の勾配はバラバラ)
4. all-reduce で 4 台の勾配を合算し、4 で割って平均する。通信後は全 GPU が同一の平均勾配を持つ
5. 各 GPU が同じ平均勾配で `optimizer.step()` を実行 → 更新後も 4 台の重みは完全に一致
6. 次のステップへ

数式で書くと、各 GPU i がローカル勾配 g_i を計算し、all-reduce が g = (1/N) Σ_i g_i を全員に届ける、という処理です。N は world size。これは global batch 256 サンプルで計算した勾配と数学的に等しくなります。重要なのは、loss 関数がサンプル平均 (batch 内で平均する形) のとき、各 GPU のローカル勾配を平均すれば自動的に global batch の勾配になるという点で、ここがズレると (たとえば loss を sum で取ると) 実効学習率が台数倍に化けるバグになります。

### 計算と通信のオーバーラップ (DDP の核心的最適化)
素朴に実装すると「backward が全部終わってから all-reduce 開始 → 終わってから step」となり、通信時間がまるごと待ち時間になります。DDP の賢い点は、この通信を backward の計算と重ねて隠すことです。

逆伝播は出力側の層から入力側へ進みます。つまり最後の層の勾配が最初に確定し、最初の層の勾配が最後に確定します。DDP はある層の勾配が出来上がった瞬間に、まだ計算中の手前の層を待たず、その層の all-reduce を裏で開始します。

```
時間 →
backward:  [層4勾配][層3勾配][層2勾配][層1勾配]
通信:            [層4 all-reduce][層3 all-reduce][層2..][層1..]
                 ↑ backward と並行して通信が走る
```

これにより、通信時間の大半が計算時間の裏に隠れ、見かけ上ほぼ通信ゼロのように振る舞います。実装上は、各パラメータの勾配が確定したことを autograd の hook (勾配確定時に呼ばれるコールバック) で検知して通信を発火させます。

### gradient bucketing (勾配バケツ化)
パラメータ 1 個ごとに all-reduce すると、小さい通信が大量に発生して非効率です (通信には 1 回ごとに固定の立ち上げコスト、いわゆるレイテンシがある)。そこで DDP は複数パラメータの勾配を 1 つのバケツ (典型的に 25MB 程度) にまとめ、バケツ単位で all-reduce します。大きなまとまりで送ることで帯域を有効に使え、オーバーラップの単位にもなります。バケツの境界は backward で勾配が確定する順 (おおむね層の逆順) に合わせて区切られ、最初に埋まったバケツから順に通信を発火させることでオーバーラップを最大化します。

### ring all-reduce のアルゴリズム
all-reduce の効率的な実装として ring all-reduce が広く使われます。N 台の GPU を論理的にリング状に並べ、データを N 分割して、各台が隣へ少しずつ送りながら部分和を積み上げます。reduce-scatter フェーズ (N-1 ステップ) で各台が「自分担当の部分の全合計」を持ち、all-gather フェーズ (N-1 ステップ) でそれを全員に配ります。

総通信量は台数 N にほぼ依存せず一定で、各台が送受信するのは約 2(N-1)/N × データサイズ ≈ 2 × データサイズです。つまり台数が増えても 1 台あたりの通信量がほぼ増えないため、スケールしやすいのが ring all-reduce の決定的な利点です (素朴な「全員が rank 0 に送って rank 0 が配る」方式だと rank 0 に通信が集中して N に比例して悪化します)。NVIDIA の NCCL ライブラリがこれを GPU 間で高速実行し、トポロジ (NVLink/InfiniBand の繋がり方) を見て ring や tree などのアルゴリズムを自動選択します。

### batch サイズと学習率: 線形スケーリング則
4 GPU で各 64 サンプルなら実効 batch は 256 です。batch を大きくすると 1 ステップの勾配が安定する反面、同じ学習率では「1 エポックあたりの更新回数」が減るので収束が遅くなります。経験則として線形スケーリング則 (batch を k 倍にしたら学習率も k 倍) が知られ、ただし学習初期に学習率を徐々に上げる warmup を併用しないと、大きな学習率で発散します。

これは DDP 導入時に最も見落とされやすい落とし穴で、「台数を増やしたら精度が落ちた」のほとんどはこの調整漏れです。直感的には、batch が大きいほど勾配は正確 (低ノイズ) なので大きな歩幅で進めますが、学習のごく初期は重みがまだランダムに近く、大きな歩幅で進むと壊れるため、最初だけ歩幅を抑えるのが warmup です。さらに極端な大 batch では線形スケーリングが破綻し、LARS / LAMB (層ごとに学習率を適応させる optimizer) などが必要になります。

### 典型的な実装
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

## 手法の系譜と主要論文

データ並列の歴史は「同期 SGD をどう効率よく分散するか」の歴史です。

- Dean ら, "Large Scale Distributed Deep Networks" (NeurIPS 2012, Google)。DistBelief。parameter server (重みを集中管理するサーバ) と非同期 SGD を提案。各 worker が勝手なタイミングで勾配を送る方式で、初期の大規模分散学習を支えましたが、勾配の遅延 (stale gradient = 古い重みで計算された勾配が遅れて届くこと) で精度が不安定になる問題がありました。

- Goyal, Dollár, Girshick ら, "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (Facebook, 2017, arXiv:1706.02677)。データ並列で巨大 batch (8192) を使う際の学習率調整 (線形スケーリング則 + warmup) を確立し、ImageNet 学習を 256 GPU・1 時間で完了させました。「batch を増やしただけでは精度が出ない、学習率を正しく合わせろ」という、DDP 実用の必読の教訓を与えた論文です。

- Sergeev & Del Balso, "Horovod: fast and easy distributed deep learning in TensorFlow" (2018, Uber, arXiv:1802.05799)。ring all-reduce を MPI ベースで実装し、parameter server より高効率な all-reduce 方式のデータ並列を広めました。少ないコード変更で分散化できる API 設計も含め、PyTorch DDP の通信設計の源流の 1 つです。

- Li, Zhao, Varma ら, "PyTorch Distributed: Experiences on Accelerating Data Parallel Training" (VLDB 2020, Facebook, arXiv:2006.15704)。本トピックの中心となる論文で、PyTorch `DistributedDataParallel` の設計と最適化を解説。中核の工夫は (1) gradient bucketing、(2) 計算と通信のオーバーラップ、(3) 勾配が出ない (forward で使われなかった) パラメータのスキップ。near-linear scaling を多くのモデルで達成したことを実測で示しました。

その後、データ並列はメモリの冗長性を解く方向へ進化します ([ZeRO](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer)、[FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel))。DDP は「モデルが 1 台に収まる範囲で、最も単純で速い」基本形として今も標準であり続けています。

## 論文の実験結果 (定量データ)

PyTorch DDP 論文 (VLDB 2020) の主要な実測は次の通りです。評価には ResNet-50 (画像分類) と BERT (自然言語) を使い、指標は scaling efficiency (台数に対する高速化の比率) と 1 秒あたりの処理サンプル数です。

- ResNet-50 と BERT を最大 256 GPU まで増やしたとき、ideal (台数に完全比例) に対しておよそ 90% 前後の scaling efficiency を維持。たとえば 256 GPU で理想の約 9 割の速度が出る、という意味で、これは通信を計算に隠せていることを示します。
- bucketing の効果: バケツサイズを適切に (報告では数十 MB) 取ると、パラメータごと通信に比べてスループットが大きく向上。バケツが小さすぎると通信回数が増えて遅く、大きすぎると最初のバケツが埋まるまで通信を始められずオーバーラップの粒度が粗くなって遅くなる、という凸型 (途中に最適点がある山型) の関係が示されました。
- オーバーラップの効果: backward と all-reduce を重ねることで、重ねない場合に比べて 1 イテレーションの時間が短縮されることをプロファイルで実証。論文の例では、通信を計算の裏に隠すことで、通信が見かけ上ほぼ消える挙動が示されました。

Goyal ら (2017) のアブレーション的な知見も重要です。線形スケーリング則だけで warmup を入れないと、大 batch (8192) では学習初期に発散し精度が出ません。warmup を入れると、batch 256 の小規模学習とほぼ同等の top-1 accuracy (ImageNet で約 76%) を、batch 8192・256 GPU で再現できました。さらに batch を 8192 より大きくすると warmup を入れても精度が劣化し始め、純粋な線形スケーリングには限界があることも示されています。「どの要素を抜くと壊れるか、どこまでなら通用するか」が明確に示された例で、DDP を実用に乗せる際の前提知識になっています。

指標の意味の補足: scaling efficiency は「N 台にしたとき本当に N 倍速くなったか」を測る比率で、1.0 (100%) が理想です。これが落ちる主因は通信オーバーヘッドと負荷の不均衡で、効率の数値を見れば「台数を増やす価値がまだあるか (これ以上増やしても頭打ちか)」を判断できます。top-1 accuracy は「予測 1 位が正解だった割合」で、ImageNet 1000 クラス分類の標準指標です。0.x% の差でも順位やモデルの優劣を論じるのが慣例なので、76% を維持できたことは「分散化で精度を損なわなかった」ことの強い証拠になります。

## メリット・トレードオフ・限界

メリット
- 実装が単純。既存の単一 GPU コードに数行足すだけで動く
- GPU 台数にほぼ比例した高速化 (near-linear scaling) が得られやすい
- PyTorch 標準で実績が豊富。ドキュメントとツール (torchrun、分散学習オペレータ) が充実

トレードオフ・限界
- モデル・勾配・optimizer 状態を各 GPU にまるごと複製するため、メモリ削減はゼロ。1 台に入らない巨大モデルには使えない (その場合は [FSDP](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel) / [ZeRO](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer))
- 通信帯域 (GPU 間・ノード間ネットワーク) がボトルネックになると、台数を増やしても速度が頭打ちになる。特にノードをまたぐ multi-node では帯域が GPU 内より桁違いに細く、scaling 効率が落ちる
- 実効 batch が増えるため学習率の再調整が必須 (線形スケーリング則 + warmup)。怠ると精度が出ない、あるいは発散する
- 各 GPU の処理時間がバラつく (データの長さが不揃い、一部ノードが遅いなど) と、最も遅い GPU に全員が同期で待たされる (stragglers 問題)

研究上の未解決課題: 同期 all-reduce は「最も遅い 1 台」に律速されるため、超大規模 (数千 GPU) では straggler や通信競合が深刻化します。勾配圧縮 (量子化やスパース化で通信量を減らす)、非同期・準同期手法、通信トポロジ最適化などが研究され続けていますが、精度を保ちつつ通信を減らすのは依然として難問です。また「大 batch で精度を落とさず収束させる」最適化アルゴリズム (LARS/LAMB を超える設計) も、超大規模学習の継続的なテーマです。

## 発展トピック・研究の最前線
- 勾配圧縮: PowerSGD (Vogels ら, NeurIPS 2019) のように勾配を低ランク近似して通信量を削減する手法。精度をほぼ保ったまま通信を数分の 1 に減らせるが、圧縮の計算コストとのバランスが課題。1-bit SGD や top-k sparsification なども同系統
- 大 batch 最適化: LARS (You ら, 2017)、LAMB (You ら, ICLR 2020)。層ごとに勾配のノルムに応じて学習率を調整し、BERT を数千 GPU・極端な大 batch で短時間学習することを可能にした
- ハイブリッド並列: データ並列・テンソル並列・パイプライン並列を組み合わせる 3D parallelism (Megatron-LM など)。DDP 単独では足りない超大規模モデルで、各並列軸を使い分ける
- 通信最適化: NCCL のトポロジ認識、NVLink / InfiniBand などの高速インターコネクト活用、gradient accumulation (複数ステップの勾配をためてから 1 回通信し、通信頻度と実効 batch を同時に調整する) との組み合わせ

## さらに学ぶための関連トピック
- [FSDP (Fully Sharded Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0611_fsdp-fully-sharded-data-parallel)
- [ZeRO (Zero Redundancy Optimizer)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0662_zero-optimizer)
- [Gradient Checkpointing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0146_gradient-checkpointing)
- [bf16 Mixed Precision](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0603_bf16-mixed-precision)
