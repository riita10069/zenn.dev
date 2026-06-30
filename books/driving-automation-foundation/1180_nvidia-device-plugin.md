---
title: "NVIDIA Device Plugin"
---

# NVIDIA Device Plugin

## ひとことで言うと
Kubernetes はそのままでは「このマシンに GPU が何枚あるか」を知りません。NVIDIA Device Plugin は、GPU を Kubernetes に「`nvidia.com/gpu` という割り当て可能な資源」として教えるための小さな常駐プログラムです。これがあると、コンテナ(Pod)が「GPU を1枚ください」と宣言でき、Kubernetes が GPU 付きノードへ正しく配置し、1枚の GPU を複数のプログラムが知らずに奪い合う事故も防げます。深層学習をクラスタ上で動かすときの土台となる仕組みです。

## 直感的な理解
たくさんのサーバを束ねて「巨大な1台の計算機」のように使う仕組みを、ここでは Kubernetes と呼びます。Kubernetes は「このプログラムには CPU を2コア、メモリを4GB あげて、ちょうど空いているサーバに置こう」というふうに、資源(リソース)を見て配置先を決める係です。

ところが、Kubernetes が最初から理解している資源は CPU とメモリだけです。GPU は理解していません。GPU を理解しないと何が困るか、身近な例で考えます。会社の共用プリンタを想像してください。誰がどのプリンタを使っているか管理する仕組みが無ければ、2人が同じプリンタに同時に印刷を投げて紙詰まりを起こします。GPU も同じで、「どのサーバの GPU が今空いているか」を Kubernetes が把握していないと、(1) GPU を使いたいプログラムが「GPU が欲しい」と意思表示できず、(2) GPU が1枚も無いサーバに GPU 必須のプログラムが置かれてしまい、(3) 1枚の GPU を複数のプログラムが同時に掴んでメモリ不足で全滅する、という事故が起きます。

NVIDIA Device Plugin は、この「GPU を Kubernetes に見えるようにする通訳」です。これを入れると、GPU が CPU・メモリと同じ「数えられる資源」として扱えるようになり、上の3つの事故がまとめて解決します。

## 基礎: 前提となる概念
このトピックを理解するには、Kubernetes の登場人物を少しだけ知る必要があります。

- ノード(node): Kubernetes が束ねている1台のサーバ。クラウドでは仮想マシン(VM、物理サーバの一部を仮想的に1台のサーバとして使うもの)であることが多いです。
- Pod: Kubernetes が動かす最小単位。1個以上のコンテナ(プログラムとその実行環境をまとめて持ち運べるようにした箱)を入れた箱だと思ってください。
- kubelet: 各ノードに常駐し、そのノードで Pod を実際に起動・監視する Kubernetes のエージェント(現場係)。
- スケジューラ(scheduler): 「この Pod をどのノードに置くか」を決める Kubernetes の頭脳。各ノードの空き資源を見て決めます。
- DaemonSet: 「すべてのノード(または条件に合うノード)にちょうど1個ずつ Pod を配置する」ための仕組み。監視エージェントやドライバ系のように「全ノードに1個欲しい」ものに使います。

ここで鍵になるのが「資源(リソース)」という考え方です。Kubernetes は資源を Capacity(そのノードの総量)と Allocatable(割り当て可能な量)という形でノードごとに把握しています。CPU・メモリは標準で扱えますが、それ以外の特殊なハードウェア(GPU、FPGA、高速ネットワークカードなど)は「拡張資源(extended resource)」という枠で後から追加登録できます。`nvidia.com/gpu` はこの拡張資源の一つです。

そして「Device Plugin(デバイスプラグイン)」とは、Kubernetes が公式に用意した拡張点(プラグインの差し込み口)です。これは Device Plugin API という決まった通信規約として 2018 年前後に Kubernetes に導入されました(Kubernetes Enhancement Proposal、通称 KEP で仕様化)。この差し込み口に対応したプログラムを書けば、好きなハードウェアを拡張資源として登録できます。NVIDIA Device Plugin は、その差し込み口を使って「NVIDIA GPU」を登録する公式実装です。

## 仕組みを詳しく
NVIDIA Device Plugin は通常、DaemonSet として全 GPU ノードに1個ずつ配置されます。各ノードで行う仕事は大きく3段階です。

### 1. GPU を検出する(discovery)
ノードに常駐したプラグインは、NVIDIA ドライバ(GPU をソフトから操作するための土台ソフト)を通じて「このノードに NVIDIA GPU が何枚あり、それぞれの ID は何か」を調べます。内部的には NVML(NVIDIA Management Library、GPU の状態を取得する公式ライブラリ)を使います。

### 2. kubelet に資源を登録する(register / ListAndWatch)
プラグインは kubelet と gRPC(プログラム同士が決まった形式で通信する仕組み)で接続し、「このノードには `nvidia.com/gpu` という資源が N 枚ある」と報告します。さらに ListAndWatch という常時接続を保ち、GPU が故障して使えなくなったらリアルタイムで「使える枚数が減った」と知らせます。報告を受けた kubelet は、そのノードの資源一覧に GPU を載せます。実際に確認するとこう見えます。

```bash
kubectl describe node <gpu-node>
# Capacity / Allocatable に次のような行が現れる
#   nvidia.com/gpu:  1
```

before(プラグイン無し)では `nvidia.com/gpu` の行はそもそも存在せず、Kubernetes は GPU の存在を一切知りません。after(プラグイン有り)では上の行が現れ、スケジューラがこの数を見て配置判断できるようになります。

### 3. Pod に GPU を割り当てる(Allocate)
ある Pod が GPU を要求してそのノードに配置されると、kubelet はプラグインに「この Pod のコンテナに GPU を1枚用意して」と Allocate を呼びます。プラグインは、コンテナから GPU が見えるように、必要な環境変数(例: `NVIDIA_VISIBLE_DEVICES` に割り当てた GPU の ID を入れる)やデバイスファイルへのアクセス権をコンテナに渡します。これにより、コンテナの中から `nvidia-smi` を実行すると割り当てられた GPU だけが見える、という状態になります。

利用する側の Pod は、欲しい枚数を書くだけです。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: training-job
spec:
  containers:
    - name: trainer
      image: <training-image>
      resources:
        limits:
          nvidia.com/gpu: 1   # GPU を1枚ください
```

ここで一つ注意点があります。GPU(`nvidia.com/gpu`)は拡張資源なので、`requests`(最低保証量)と `limits`(上限)を別々に指定することができず、`limits` に書くと自動的に同量が `requests` 扱いになります。CPU のように「最低1コア、上限4コア」といった幅を持たせる指定はできません。GPU は基本的に「整数枚を丸ごと占有する」資源だからです。

スケジューラは `nvidia.com/gpu` が1枚以上空いているノードを探して Pod を置きます。空きが無ければ Pod は Pending(待機)のまま留まります。これで「GPU の取り合い」も「GPU 無しノードへの誤配置」も防げます。

before/after を整理します。

```
before(Device Plugin なし):
- Kubernetes は GPU の存在を知らない
- Pod は GPU を要求する手段が無く、配置は GPU の有無と無関係
- コンテナから GPU が見えるかは運任せ、複数 Pod が同じ GPU を奪い合う

after(Device Plugin あり):
- ノードに nvidia.com/gpu: N という資源が現れる
- Pod が nvidia.com/gpu: 1 と書けば空き GPU のあるノードへ確実に配置
- 1枚の GPU が同時に複数 Pod へ割り当てられない(排他制御)
```

### GPU をもっと細かく分ける: time-slicing と MIG
標準では「1枚=1割り当て」ですが、推論(学習済みモデルで予測するだけの軽い処理)のように1枚を丸ごと使い切らないワークロードでは、もったいない場合があります。これを解決する2つの設定が Device Plugin にあります。

- time-slicing(時間分割): 1枚の GPU を「論理的に複数枚」として広告し、複数 Pod が時間を区切って交代で使う方式。`nvidia.com/gpu: 4` と広告しても物理は1枚なので、メモリは分離されずお互いに干渉しうる点が弱点です。利用率は上がりますが、性能の予測しにくさが代償です。
- MIG(Multi-Instance GPU): A100 や H100 など対応 GPU で、1枚を物理的に分割して独立した小さな GPU(インスタンス)に切る方式。メモリも演算も分離されるため干渉が小さく、time-slicing より安全です。ただし分割パターンが固定で、対応 GPU でしか使えません。

1つの学習ジョブが GPU を占有するような使い方では、time-slicing も MIG も使わず、素直に1枚=1割り当てにするのが最もシンプルで確実です。

## 手法の系譜と主要論文
Device Plugin そのものは Kubernetes の設計文書(KEP)で定義された仕組みで、論文ではありません。しかし「アクセラレータ(GPU 等)を共有クラスタでどう効率よくスケジュールするか」は活発な研究テーマで、Device Plugin はその研究が乗る土台になっています。系譜を追います。

- Xiao ら "Gandiva: Introspective Cluster Scheduling for Deep Learning"(USENIX OSDI 2018、Microsoft Research)は、深層学習ジョブ専用に GPU を時間分割し、実行中のジョブを別 GPU へ移す(マイグレーション)ことで利用率を引き上げるスケジューラを提案しました。深層学習ジョブは「ミニバッチ単位で周期的に動く」性質があるため、その切れ目を狙って安全に切り替えられる、という洞察が新規性です。

- Gu ら "Tiresias: A GPU Cluster Manager for Distributed Deep Learning"(USENIX NSDI 2019、Michigan/Microsoft)は、ジョブの完了時間を短くするスケジューリングに踏み込み、ジョブの長さが事前に分からない前提でも待ち時間を減らす方式を示しました。

- Weng ら "MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters"(USENIX NSDI 2022、Alibaba)は、実運用の巨大 GPU クラスタのトレースを解析し、GPU 共有(1枚を複数ジョブで使う)が利用率改善に有効である一方、メモリ干渉やローカリティ(同じジョブの分散部分を近くに置くこと)が性能を左右することを定量的に示しました。

これらはいずれも「まず GPU を Kubernetes に資源として認識させる(=Device Plugin の役割)」を前提に、その上でどう割り当てるかを工夫する研究です。階層関係は「Device Plugin = 資源を見せる土台」「スケジューラ研究 = その土台の上で効率化」と整理できます。

## 論文の実験結果(定量データ)
スケジューラ研究の効果は「GPU 利用率(GPU がアイドルでなく実際に計算している時間の割合)」や「ジョブ完了時間(JCT, Job Completion Time)」で測られます。これらの指標が重要なのは、GPU は非常に高価で、利用率が低い=お金を捨てているのと同じだからです。

- Gandiva(OSDI 2018)は、報告によると、時間分割とマイグレーションにより GPU 利用率を最大で 26% 程度改善し、複数モデルを同時に試すハイパーパラメータ探索のような場面で全体のスループット(単位時間あたりに処理できるジョブ量)を大きく高めたとされます。アブレーション的に見ると、マイグレーション(ジョブの引っ越し)を外すと「断片化(GPU が虫食い状に埋まって新しいジョブが入らない状態)」が解消できず、利用率改善の多くが失われます。引っ越しが効果の中核だ、ということです。

- Tiresias(NSDI 2019)は、ジョブの完了時間(JCT)の中央値を、当時の標準的なスケジューラ比でおよそ 5 倍程度短縮できたと報告しています。中央値とは「全ジョブを完了時間順に並べたときの真ん中」で、平均より外れ値に強い指標です。これが短いほど、ユーザは結果を早く得られます。

- Weng ら(NSDI 2022)の解析では、共有を使わない素朴な運用では GPU 利用率が低い水準にとどまる一方、適切な共有とパッキング(詰め込み)で利用率を大きく引き上げられることが、実トレースの統計から示されました。同時に、メモリ帯域の干渉により共有したジョブの一部が遅くなる現象も観測され、「共有は万能ではない」という限界も明らかにしました。

数値そのものは環境依存ですが、共通する教訓は「GPU を資源として正しく見せた上で、断片化を防ぎ干渉を抑える割り当てをすると、利用率もジョブ完了時間も大幅に改善する」という点です。

GPU 共有方式そのものの定量比較も研究されています。time-slicing は1枚を時間で輪番に使うため、複数の推論プロセスを同居させると合計スループットは上がる一方、各プロセスのレイテンシ(1リクエストを処理する時間)は他プロセスのカーネル実行に割り込まれてばらつき、95 パーセンタイル(遅い方から 5% の位置)のレイテンシが単独実行比で数倍に伸びる、という報告があります。MIG は物理的にメモリと演算ユニットを区切るため、A100 40GB を 7 分割(各 5GB 相当)した各インスタンスは互いにほぼ干渉せず、レイテンシの予測しやすさ(tail latency の安定)で time-slicing を上回ります。代償は、分割パターンが 1g.5gb / 2g.10gb / 3g.20gb のように離散的で、1枚を 7 等分すると各インスタンスの演算能力は単独 1 枚の約 1/7 に固定される点です。「干渉を許して柔軟に詰めるか(time-slicing)、干渉を断って硬く区切るか(MIG)」というトレードオフが、指標(スループット重視か tail latency 重視か)によって最適解を変えます。

## メリット・トレードオフ・限界
メリット:
- GPU を Kubernetes の第一級資源(`nvidia.com/gpu`)として扱え、Pod が宣言的に GPU を要求できる
- スケジューラが GPU の空き状況を見て配置するため、取り合いや誤配置を防げる
- time-slicing / MIG により GPU 共有も設定でき、推論など軽いワークロードで利用率を上げられる
- NVIDIA 公式実装で広く使われ、情報が豊富

トレードオフ・限界:
- 素の Kubernetes では別途デプロイし、バージョンと GPU ドライバ・CUDA の整合性を維持する運用負担がある(ドライバと CUDA のバージョン不一致は GPU が見えない典型的な障害原因)
- 標準では GPU は整数枚単位で、1枚未満の細かい共有は time-slicing / MIG の追加設定が必要
- time-slicing はメモリ分離が無く干渉しうる。MIG は対応 GPU と固定の分割パターンに縛られる
- 一部のクラウドのマネージド Kubernetes では、GPU ドライバと Device Plugin 相当の機能をノード OS(マシンイメージ)側に内蔵し、利用者が自前で Device Plugin を入れる必要が無い構成もある。この場合、自前で重ねて入れるとかえって競合や複雑化を招くことがある

研究上の未解決課題としては、(1) ジョブの実行時間が事前に分からない中で最適に近い割り当てをすること、(2) メモリ帯域・PCIe・ネットワークといった「GPU 以外の共有資源」の干渉まで考慮した割り当て、(3) MIG の分割パターンを動的に組み替える運用、などが依然として研究対象です。

## 発展トピック・研究の最前線
近年は Device Plugin API の後継として、より柔軟な資源記述を可能にする DRA(Dynamic Resource Allocation、動的資源割り当て)が Kubernetes に導入されつつあります。DRA は「GPU を N 枚」だけでなく「特定のメモリ容量を持つ GPU を、この種類のネットワークに近い位置で」といった構造化された要求を表現できるようにする方向で、複雑な GPU トポロジ(GPU 間の接続関係)を扱う大規模学習で重要になります。

また、推論コストを下げる文脈では、1枚の GPU 上で複数モデルを安全に同居させる研究(メモリ分離・性能隔離)や、需要に応じて GPU の割り当てを自動で伸縮させるオートスケーリングとの統合が進んでいます。学習側では、Gandiva 以降のジョブマイグレーションを実用化し、スポット(中断されうる安価な計算資源)中断と組み合わせて低コストで大規模学習を回す方向が注目されています。

## さらに学ぶための関連トピック
- [Bottlerocket OS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1100_bottlerocket-os)
- [Warm GPU Node (pause Pod)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1157_warm-gpu-node)
- [DCGM Exporter (GPU 監視)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1163_dcgm-gpu-monitoring)
- [Prometheus + Grafana](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1169_prometheus-grafana)
- [Kubecost (コスト可視化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1165_kubecost)
