---
title: "Kubeflow Training Operator (PyTorchJob)"
---

# Kubeflow Training Operator (PyTorchJob)

## ひとことで言うと
Kubeflow Training Operator は、機械学習の学習ジョブを Kubernetes 上で「宣言的に」動かすための拡張機能です。具体的には「PyTorchJob を1つ作って」とだけ宣言すれば、分散学習に必要な煩雑な準備 (複数のワーカーをどう起動し、どう通信させ、失敗したらどう再起動するか) を裏側で全部面倒見てくれます。利用者は学習コードを書き、ワーカー数と資源を指定するだけで、複数 GPU・複数サーバにまたがる学習を起動できます。

## 直感的な理解
1枚の GPU では時間がかかりすぎる、あるいはモデルがメモリに乗りきらない学習を、複数の GPU に分けて並列にやりたい、という場面を考えます。これを素朴にやろうとすると、各 GPU で動くプロセス (ワーカー) に「お前は何番目だ」「全部で何人いる」「リーダーのアドレスはここだ」を矛盾なく教え込む必要があります。サーバの IP は起動するまで分からないので、手で設定ファイルを書くのは現実的ではありません。さらに、1つのワーカーが落ちたら全体を整合性を保って立て直し、全員が揃うまで学習を始めない、といった気配りも要ります。

Training Operator は、この「分散学習のお膳立て係」です。「オペレータ (Operator)」という Kubernetes 用語は、「ある種類のリソースの面倒を見続ける常駐プログラム」を指します。人間の運用担当者 (operator) が手作業でやっていた配線・監視・再起動を、プログラムが肩代わりする、というニュアンスです。利用者は宣言を1枚書くだけで、配線は Operator がやってくれます。

## 基礎: 前提となる概念
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

## 仕組みを詳しく
### CRD = Kubernetes に学習専用の「型」を足す
Training Operator は Kubernetes に `PyTorchJob` という CRD (Custom Resource Definition) を追加します。標準の Kubernetes には Pod や Deployment という型しかありませんが、CRD を入れると「PyTorchJob」という機械学習専用の型を `kubectl apply` で扱えるようになります。歴史的には `TFJob` (TensorFlow 用)、`MPIJob` (MPI / Horovod 用)、`XGBoostJob`、`PaddleJob` などフレームワークごとに型が用意されてきました。Operator は「宣言された CRD の状態」と「実際に起きている Pod の状態」を絶えず突き合わせ、差があれば Pod を作る・消す・再起動するという制御ループ (reconciliation loop) を回し続けます。

### PyTorchJob の構造
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

### 環境変数の自動配布 (分散学習の本体)
`Master: 1`, `Worker: 3` の構成では、Operator は4つの Pod を起動し、各 Pod に次の環境変数を自動注入します。

```
Pod 0 (Master): RANK=0  WORLD_SIZE=4  MASTER_ADDR=<masterのDNS名>  MASTER_PORT=23456
Pod 1 (Worker): RANK=1  WORLD_SIZE=4  MASTER_ADDR=<同上>           MASTER_PORT=23456
Pod 2 (Worker): RANK=2  WORLD_SIZE=4  MASTER_ADDR=<同上>           MASTER_PORT=23456
Pod 3 (Worker): RANK=3  WORLD_SIZE=4  MASTER_ADDR=<同上>           MASTER_PORT=23456
```

代表 Pod 用に Kubernetes の Service (固定の DNS 名) を作り、その名前を全 Pod の MASTER_ADDR に流し込むことで、起動前には分からない IP を経由せず安定して集合できます。学習コード側は `torch.distributed.init_process_group()` を呼ぶだけで、これらの環境変数を読んでお互いに接続します。各ワーカーは異なるデータの断片で勾配を計算し、`all-reduce` で勾配を平均して同期します。役割分担は明確で、Operator は「環境変数の配布」と「Pod の生成・監視・再起動」を担当し、通信そのものは PyTorch (内部では NCCL という GPU 間集団通信ライブラリ) が行います。

### ライフサイクル
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

### キュー管理との連携 (suspend)
PyTorchJob は `runPolicy.suspend: true` を持てます。これが true の間、Operator は Pod を起動しません。外部のクォータ・優先度管理 (キューイングシステム、例えば [Kueue (Job Queueing)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1117_kueue-job-queueing)) が「いま実行してよい」と判断して suspend を解除したら、初めて起動します。これにより「学習ジョブの抽象化 (Training Operator)」と「資源配分の門番 (キュー管理)」をきれいに分離できます。Operator 単体には「全ワーカー揃うまで起動を待つ」gang scheduling の能力は弱いため、この分離が gang scheduling を外部に委ねる素直な手段になります。

## 手法の系譜と主要論文
Training Operator 自体は Kubeflow プロジェクト (2017年に Google が中心となって開始した「Kubernetes 上で ML を回すツール群」) の一部で、論文発表されたシステムではありません。しかし支える分散学習の技術には明確な研究史があります。

- データ並列分散学習の起源: Dean et al. "Large Scale Distributed Deep Networks" (NeurIPS 2012, Google) は DistBelief というシステムで、parameter server (パラメータを集約・配布する中央サーバ) を介して多数のワーカーが勾配をやり取りする分散学習を提案しました。動機は「1台では学習が遅すぎる・大きすぎる」こと、効果は「ワーカー数に応じた高速化」、トレードオフは「通信コストと、非同期更新による収束の不安定さ」でした。Li et al. "Scaling Distributed ML with the Parameter Server" (OSDI 2014) はこの方式を汎用フレームワークとして体系化しました。

- 大規模ミニバッチの安定化: Goyal et al. "Accurate, Large Minibatch SGD" (arXiv:1706.02677, 2017, Facebook AI Research) は、ワーカーを増やすと実効バッチサイズが膨らんで学習が不安定になる問題に対し、学習率の線形スケーリング (バッチを k 倍にしたら学習率も k 倍) とウォームアップ (最初の数エポックは学習率を徐々に上げる) を提案しました。これは「ワーカーを増やせばよいわけではなく、最適化の工夫が要る」ことを示し、台数を増やすときに必ず意識すべき限界を表しています。You et al. の LARS / LAMB (arXiv:1708.03888, arXiv:1904.00962) はさらに大きなバッチ (数万) でも安定させる層ごとの適応学習率を提案し、超大規模バッチ学習を可能にしました。

- all-reduce 通信の効率化: Sergeev & Del Balso "Horovod" (arXiv:1802.05799, 2018, Uber) は、parameter server 方式に代えて ring-allreduce (ワーカーをリング状につなぎ、帯域を均等に使う集団通信) で勾配同期を効率化しました。PyTorch の `DistributedDataParallel` も同系統の all-reduce を使い、勾配計算と通信をオーバーラップさせて待ち時間を隠します。Training Operator はこの通信を PyTorch / NCCL に任せ、自身は Pod とネットワークのお膳立てに徹します。

- メモリ効率化と超大規模化: Rajbhandari et al. "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (SC 2020, arXiv:1910.02054, Microsoft) は、最適化状態 (optimizer state)・勾配・パラメータをワーカー間で分割保持することで、データ並列のメモリ消費を劇的に減らしました。Narayanan et al. の Megatron-LM (SC 2021) はテンソル並列・パイプライン並列・データ並列を組み合わせた 3D 並列を確立しました。これにより数千億〜兆パラメータ級の学習が現実的になりました。Training Operator はこうした大規模分散の「実行基盤」を提供し、ZeRO 等の手法はその上で動きます。

- 後継への展開: Kubeflow コミュニティは v1 系の Training Operator (PyTorchJob 等の個別 CRD) から、より統一的な新世代 API (TrainJob / Trainer) へと発展させています。仕様が移行期にあるため、バージョン追従が運用上の論点になります。

## 論文の実験結果 (定量データ)
- ImageNet を1時間で学習 (Goyal et al. 2017): ResNet-50 を ImageNet (約128万枚の画像分類データセット、1000クラス) で学習する際、256個の GPU を使い実効バッチサイズ 8192 まで拡大しても、線形スケーリング則とウォームアップを使えば、単一ノード学習とほぼ同じ精度 (top-1 accuracy で誤差1%以内) を保ったまま、学習時間を従来の何日かから約1時間に短縮できたと報告されています。指標の top-1 accuracy は「最も確信度の高い予測クラスが正解と一致した割合」で、ImageNet では当時 76% 前後が標準的でした。ここで重要なのは「台数を64倍にしても精度を落とさなかった」点で、ウォームアップを抜くアブレーションでは精度が明確に劣化することも示され、最適化の工夫の必要性が裏付けられました。バッチサイズをさらに大きく (例えば 16384 以上に) すると線形スケーリングだけでは精度が崩れ始め、ここに素朴な台数増加の限界が現れます。

- Horovod のスケーリング効率 (Sergeev & Del Balso 2018): 標準的な分散 TensorFlow (parameter server 方式) では GPU を増やしてもスケーリング効率 (理想的な台数倍速に対する実測の割合) が頭打ちになりますが、ring-allreduce では Inception V3 や ResNet-101 で128 GPU 規模でもおよそ 90% 前後のスケーリング効率を達成したと報告されています。スケーリング効率が高いほど「通信で時間を食わず、増やした台数がそのまま速度に反映される」ことを意味します。

- ZeRO のメモリ削減 (Rajbhandari et al. 2020): ZeRO の各段階 (Stage 1: 最適化状態の分割 → Stage 2: 勾配の分割 → Stage 3: パラメータの分割) を入れるほど、ワーカーあたりのメモリ消費が下がります。報告では Stage 1+2 で従来比およそ8倍のモデルを同じハードウェアに載せられ、全段階を組み合わせて1兆パラメータ級の学習が視野に入ること、かつ400 GPU 規模でほぼ線形のスループット向上を示しました。これは「どの要素を分割するか」というアブレーションそのものが性能曲線になっており、分散の設計選択が直接メモリと到達可能な規模を決めることを示しています。

これらの数値はいずれも「分散にすればただ速くなるのではなく、最適化と通信とメモリの工夫が伴って初めてスケールする」ことを物語っており、Training Operator はその工夫を載せる土台にすぎない、という位置づけを理解するのが研究上重要です。

## メリット・トレードオフ・限界
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

## 発展トピック・研究の最前線
- Elastic training: ワーカー数を学習中に増減させても継続できる仕組み (PyTorch の TorchElastic / `torchrun`)。スポットインスタンスでワーカーが突然落ちても、生き残ったワーカーで rendezvous を再構成して学習を続けられます。
- フォールトトレランス: 大規模・長時間学習では必ず故障が起きるため、頻繁なチェックポイントと高速復旧 (例: in-memory checkpoint、非同期チェックポイント) の研究が活発です。数千 GPU 規模では平均故障間隔が数時間〜数十分にまで縮むため死活問題になります。
- 3D 並列とハイブリッド: データ並列 × テンソル並列 × パイプライン並列を組み合わせた超大規模学習 (Megatron-LM, Narayanan et al. SC 2021) が標準化しつつあり、これを Kubernetes 上で宣言的に組む方向が探られています。
- スケジューラとの統合: トポロジーアウェア配置 (高速ネットワークでつながった GPU に固める) や gang scheduling を、キュー管理システムと密に連携させる流れがあります。

## さらに学ぶための関連トピック
- [Kueue (Job Queueing)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1117_kueue-job-queueing)
- [Flyte (Pipeline Orchestration)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1107_flyte-pipeline-orchestration)
- [MLflow (Tracking + Model Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1125_mlflow-tracking-registry)
