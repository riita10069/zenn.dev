---
title: "NCCL (集合通信)"
---

# NCCL (集合通信)

## ひとことで言うと
複数の GPU で 1 つのモデルを学習するとき、GPU どうしは頻繁に数値をやり取りします。たとえば「全 GPU が計算した勾配を平均する」「分割して持っているパラメータを集める」といった操作です。NCCL（ニックル、NVIDIA Collective Communications Library）は、こうした GPU 間のデータ交換を、GPU 間の高速リンク（NVLink など）やネットワークをフル活用してできるだけ速く行う、NVIDIA 製の通信ライブラリです。PyTorch の分散学習（DDP や FSDP）で、裏側の通信を担う標準の部品になっています。分散学習の「速さの天井」を決めるのが、実はこの集合通信の効率です。

## 直感的な理解
合唱団で全員が別々の小節を練習したあと、最後に「全員の音量を平均して同じ強さに揃える」場面を想像してください。一番素朴なやり方は、指揮者 1 人が全員の音量を聞いて回り、平均を計算し、また全員に伝えに回ることです。これは指揮者にすべての通信が集中し、人数が増えるほど指揮者がボトルネックになります（これがパラメータサーバ方式の弱点です）。

代わりに「隣の人に自分の値を渡し、受け取った人は足し込んでまた隣に渡す」というバケツリレーをすれば、全員が同時に動けて、誰か 1 人に負荷が集中しません。NCCL がやっているのはまさにこの「賢いバケツリレー」で、GPU を何枚に増やしても 1 枚あたりの通信量が爆発しないように工夫されています。

## 基礎: 前提となる概念
まず「なぜ GPU 間で通信が必要なのか」をデータ並列で具体化します。データ並列とは、同じモデルのコピーを GPU 4 枚に置き、それぞれに違うデータを処理させる方式です（[DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0624_deepspeed) の冒頭も参照）。

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

## 仕組みを詳しく

### 代表的な集合通信操作（プリミティブ）
NCCL が提供する主な操作と、分散学習での使いどころを対応させます。

- all-reduce: 全 GPU の値を合計（や平均）して全 GPU に配る。データ並列の勾配同期で使う中心操作。
- all-gather: 各 GPU が持つ断片を全部集め、全 GPU が全体を得る。ZeRO Stage 3 / FSDP で分割保持したパラメータを計算直前に集める（[DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0624_deepspeed)）。
- reduce-scatter: 合計しつつ結果を分割して各 GPU に配る。all-reduce は実は reduce-scatter + all-gather に分解できる。FSDP の勾配集約でも使う。
- broadcast: 1 つの GPU の値を全 GPU に配る。学習開始時に全 GPU のモデルを同じ初期値に揃える。
- reduce / gather / scatter: 1 つの GPU に集約・収集・分配する非対称な操作。
- all-to-all: 各 GPU が他の全 GPU に異なる断片を送る。Mixture of Experts のトークンルーティングやテンソル並列の一部で使う。

### Ring all-reduce で通信を効率化する
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

### トポロジを認識してハードウェアを使い分ける
NCCL は起動時にシステムのトポロジ（GPU・リンク・ネットワークの接続関係）を調べ、経路を最適化します。

- 同一サーバ内の GPU どうし: NVLink / NVSwitch（GPU 間を直結する超高速リンク。PCIe より桁違いに速く、世代により数百 GB/s〜1.8TB/s 級。Hopper 世代の NVLink 4 で GPU あたり 900 GB/s 双方向）を使う。
- サーバをまたぐ場合: InfiniBand や高速イーサネット、可能なら GPUDirect RDMA（CPU を経由せず GPU メモリ間で直接転送する技術。CPU バウンスコピーのオーバーヘッドを除ける）を使う。AWS では EFA（Elastic Fabric Adapter）がこの役割を担う。
- in-network reduction: 対応スイッチ（NVIDIA SHARP）があれば、合算処理をネットワークスイッチ上で実行し、各ノードの送受信量をさらに削る。

この自動最適化のおかげで、通信が頻繁なテンソル並列を同一サーバ内 GPU に閉じ込める（[Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0659_tensor-parallelism)）といった配置が、ハードウェアの強みを活かして速く回ります。トポロジを誤検出すると遅い経路（PCIe 経由など）に落ちることがあり、`NCCL_TOPO_DUMP_FILE` や `NCCL_DEBUG=INFO` でどの経路・アルゴリズムが選ばれたかを確認するのが診断の定番です。

### PyTorch からの使われ方
利用者が NCCL を直接書くことはほぼありません。PyTorch の分散学習を初期化するとき `torch.distributed.init_process_group(backend="nccl")` を指定すると、DDP の勾配 all-reduce、FSDP の all-gather / reduce-scatter が裏で NCCL を呼びます。GPU 分散学習ではこれがほぼ既定です（CPU では Gloo という別バックエンドを使う）。DDP は通信と逆伝播の計算をオーバーラップさせる工夫を持ちます。逆伝播は出力層側から入力層側へ進むので、勾配ができた層から順に、複数の勾配をまとめた「bucket（バケツ、既定 25MB 程度）」単位で all-reduce を非同期に発火させ、まだ計算中の他層の逆伝播の裏に通信を隠します（gradient bucketing）。これにより理想的には通信時間がほぼ見えなくなり、計算律速に近づきます。

## 手法の系譜と主要論文
- Ring all-reduce の理論（Patarasuk & Yuan, "Bandwidth optimal all-reduce algorithms for clusters of workstations", JPDC 2009）: 各ノードの通信量が台数に依存せず帯域最適になることを示した古典。HPC コミュニティ発で、後に深層学習へ輸入された。
- Baidu の深層学習応用（2017 年ごろの技術ブログ・実装、ring-allreduce）: HPC の Ring all-reduce を深層学習の勾配同期に持ち込み、パラメータサーバ方式より素直にスケールすることを実証。
- Horovod（Sergeev & Del Balso, "Horovod: fast and easy distributed deep learning in TensorFlow", arXiv:1802.05799, 2018, Uber）: Ring all-reduce ベースの分散学習を数行で書けるようにしたフレームワーク。通信バックエンドに NCCL や MPI を使う。動機は「当時の分散学習がパラメータサーバ方式で書きにくくスケールしにくかった」こと。Tensor Fusion（小さなテンソルをまとめて 1 回の通信にする最適化）など実務的工夫も導入。
- NVIDIA NCCL（製品。仕様・実装が一次情報）: トポロジ認識で最適アルゴリズム（Ring / double-binary Tree / CollNet）と経路（NVLink / IB / RDMA / SHARP）を選ぶ集合通信ライブラリ。汎用 MPI では GPU 特有の高速リンクを活かしきれなかった問題を解く。NCCL 2.x で double binary tree を導入し多ノードのレイテンシを大幅改善。
- Megatron-LM（Narayanan ら, SC 2021, arXiv:2104.04473）: テンソル並列・データ並列の通信を NCCL で実行する前提で超大規模 LLM を学習し、NCCL が大規模学習の通信基盤の標準であることを裏づけた。

系譜を一言でまとめると、「HPC で生まれた帯域最適 all-reduce 理論（2009）→ 深層学習への応用と使いやすい API 化（Baidu/Horovod 2017-18）→ GPU ハードを知り尽くした NVIDIA 製ライブラリとして標準化（NCCL）→ 超大規模 LLM 学習の通信基盤（Megatron 系）」という流れです。

## 論文の実験結果（定量データ）
指標の意味を先に説明します。集合通信の性能は「実効帯域（algorithm bandwidth / bus bandwidth、実際に何 GB/s でデータを動かせたか）」と「スケーリング効率（GPU を N 倍にしたとき学習速度が何倍になるか。1.0 に近いほど良い）」で測ります。bus bandwidth はアルゴリズムの理論通信量で割り戻した「ハードの本来の帯域をどれだけ使えたか」を表し、リンク理論値に近いほど効率が良いとされます。

- Horovod 論文（2018）の報告: Inception V3 / ResNet-101 / VGG-16 を多数 GPU で学習する際、従来の TensorFlow 標準（グラフ内分散・パラメータサーバ的）に比べ、Ring all-reduce + NCCL で 128 GPU 規模でスケーリング効率を大きく改善し、モデルによりスループットがおよそ 2 倍、スケーリング効率 80〜90% 台を達成と報告。通信が GPU 数に対してほぼ一定量で済むことが効いている。
- NCCL のベンチマーク（NVIDIA 公表値、nccl-tests）: NVLink/NVSwitch で結ばれた 8 GPU の all-reduce で、リンクの理論帯域に近い実効 bus bandwidth（DGX A100 で 200〜250 GB/s 級）を達成。multi-node でも IB/EFA + GPUDirect RDMA により高い bus bandwidth を維持し、double binary tree で多ノードの中サイズメッセージのレイテンシを Ring より大きく短縮。
- Megatron-LM（SC 2021）の報告: 数千 GPU（A100）で数千億〜兆パラメータ級を学習し、テンソル並列を同一ノード（NVLink）に、パイプライン並列をノード間に割り当てる配置で、GPU あたり高い達成 TFLOPS（A100 ピーク 312 TFLOPS に対し約 50% 超の利用率、約 163 TFLOPS）を実現。NCCL の通信効率がこのスケーラビリティの前提になっている。

数値の意味を初心者向けに補うと、実効帯域がリンクの理論値に近いほど「配線の太さを無駄なく使えている」ことを意味します。スケーリング効率が 0.9 なら「8 枚にして 7.2 倍速くなった（理想の 90%）」という読み方で、通信が下手だとこれが 0.5（半分しか活きない）まで落ちることもあります。多ノードでネットワークが細い（EFA/IB がない）と、まさにこの 0.5 付近まで落ちるのが現実です。

## メリット・トレードオフ・限界
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

## 発展トピック・研究の最前線
- 勾配圧縮（gradient compression, PowerSGD, 1-bit Adam など）で all-reduce のバイト数自体を削る方向。
- in-network reduction（スイッチ上で合算する NVIDIA SHARP）で通信を物理層に押し込み、各ノードの送受信量をさらに削減。
- FP8 / 低精度通信、通信と計算の重ね合わせをコンパイラ的に最適化する研究（例: 通信を計算カーネルに融合する手法）。
- 耐障害・弾力的な分散学習（elastic training, torchrun の elastic 機能）における集合通信の再構成・communicator 再生成。
- 超大規模での通信トポロジ最適化（rail-optimized ネットワーク、GPU 配置と通信パターンの協調設計）。

## さらに学ぶための関連トピック
- [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0659_tensor-parallelism)
- [Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0646_pipeline-parallelism)
- [DeepSpeed](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0624_deepspeed)
- [Spot Instance 学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1181_spot-instance-training)
