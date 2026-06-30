---
title: "Kueue (Job Queueing)"
---

# Kueue (Job Queueing)

## ひとことで言うと
Kueue (キュー、と読みます) は、Kubernetes というサーバ群の管理システムの上で動く「バッチジョブ用の順番待ち列 + クォータ (使ってよい資源の上限) 管理」の仕組みです。機械学習の学習ジョブのように「GPU を長時間まるごと占有する」処理は、誰かが好き勝手に投げると資源の取り合いになります。Kueue はジョブをいったん列に受け止め、決められたクォータの範囲内で、優先度のルールに従って実行を許可 (admit) していきます。Kubernetes の標準スケジューラの「上」に乗る門番 (ゲートキーパー) だと考えてください。

## 直感的な理解
人気ラーメン店を想像してください。席 (= GPU) は限られていて、客 (= ジョブ) は次々にやってきます。整理券も列もなく、来た人が空席に勝手に座れる仕組みだと、すぐに混乱します。常連 (締め切りのある本番ジョブ) が来ても、たまたま先に座った客 (実験用のジョブ) が長居していて入れません。

ここに「番号札を配り、席数の上限を守り、VIP には優先入店させる」係を置けば秩序が生まれます。Kueue はこの係です。やってきたジョブを列に並べ、「いま空いている GPU は何枚か」「このジョブはどのくらい偉い (優先度) か」を見て、入店を許可します。GPU は1枚数十万円、クラウドでも1時間あたり数ドル〜十数ドルかかる希少資源なので、この交通整理の有無で計算資源の利用効率と研究の進み方が大きく変わります。

ここで肝心なのは Kueue が「席にどう座らせるか (= どのサーバのどの GPU に配置するか)」までは決めない点です。そこは Kubernetes 標準スケジューラの仕事です。Kueue は「いま入店させてよいか」だけを判定し、許可したジョブを標準スケジューラに引き渡します。役割を意図的に絞っているのが Kueue の設計思想です。

## 基礎: 前提となる概念
まず登場する用語を、知らない前提でひとつずつ噛み砕きます。

- Kubernetes (K8s と略す): たくさんのサーバ (物理マシンや仮想マシン) をまとめて1つの巨大な計算機のように扱うシステムです。「このプログラムを CPU 4コア・メモリ 16GB で動かして」と宣言すると、空いているサーバを探して自動で配置・起動・監視してくれます。
- Pod (ポッド): Kubernetes が動かす最小単位です。1つ以上のコンテナ (アプリと依存物をまとめた箱) が入った「実行される1かたまり」だと思ってください。
- スケジューラ: 「どの Pod をどのサーバ (ノード) に置くか」を決める Kubernetes 内部の部品です。標準スケジューラは「空いているノードに Pod を1つずつ置く」ことに特化しており、Web サーバのように常時起動して大量の Pod を捌く用途に向いています。
- 拡張リソース (extended resource): GPU は CPU やメモリと違い Kubernetes の標準資源ではありません。`nvidia.com/gpu` という名前の「拡張リソース」として、何枚要求するかを Pod に書きます。Kubernetes は GPU の枚数を整数として数えるだけで、「VRAM が何 GB 残っているか」までは原則把握しないことに注意してください。
- バッチジョブ: Web サーバのように常駐するのではなく、「始まって、計算して、終わる」タイプの処理。機械学習の学習は典型的なバッチジョブです。

ここで問題が生まれます。標準スケジューラには「順番待ち」や「クォータ」という概念がほとんどありません。GPU が空いていない Pod は `Pending` (保留) のまま放置され、次に誰を実行すべきかのルールがありません。研究者が10本の実験を投げた直後に締め切り直前の本番学習を投げても、本番が後回しになります。バッチジョブの世界では、HPC (High Performance Computing、スーパーコンピュータ的な大規模計算) の分野で Slurm などが何十年も「列とクォータと優先度」を扱ってきました。Kueue はその発想を Kubernetes に持ち込むものです。

## 仕組みを詳しく
Kueue は Kubernetes に新しい「概念」を CRD (Custom Resource Definition、Kubernetes に独自の型を追加する仕組み) として加えます。主要な登場人物を順に説明します。

### 1. ResourceFlavor (資源の種類)
「どんな種類のハードウェアか」を表す名札です。たとえば「特定の GPU を積んだノード群」「スポットインスタンス (中断され得る安価なノード) のノード群」「オンデマンドのノード群」などをノードラベルで指定し、1つのカテゴリとして束ねます。これにより「GPU の種類ごとにクォータを分ける」「高価なオンデマンドより安いスポットを優先的に使う」といった配分ができます。1つの ClusterQueue は複数の Flavor を優先順位付きで列挙でき、上から順に空きを探します (flavor fungibility)。

### 2. ClusterQueue (クラスタ全体のクォータの貯金箱)
Kueue の心臓部です。「この列を通って実行できるジョブたちは、合計で CPU 何コア・メモリ何 GB・GPU 何枚まで同時に使えるか」という上限 (nominalQuota = 名目クォータ) を Flavor ごと・リソースごとに定めます。たとえば GPU が4枚分のクォータなら、GPU を1枚ずつ要求するジョブは最大4本まで同時に走り、5本目はクォータ不足で admit されず列で待ちます。

ClusterQueue は cohort (コホート、グループ) を組むこともできます。複数の ClusterQueue を同じ cohort に入れると、片方が余らせている資源をもう片方が一時的に借りる (borrowing) ことができます。「チーム A の枠が空いているとき、チーム B が一時的に使い、A のジョブが来たら返す」といった柔軟な共有が可能になります。borrowingLimit で「借りてよい上限」を、lendingLimit で「貸し出してよい上限」を細かく制御できます。

### 3. LocalQueue (各チーム・名前空間が投げる窓口)
ユーザはジョブを直接 ClusterQueue に出すのではなく、自分の namespace (名前空間、リソースを区切る論理的な仕切り) にある LocalQueue に出します。LocalQueue は「この窓口に来たジョブはどの ClusterQueue に流すか」という対応付けを持ちます。これにより「クォータの定義 (ClusterQueue) は管理者が一元管理し、利用者は手元の窓口に投げるだけ」という役割分担ができます。

### 4. WorkloadPriorityClass (優先度)
ジョブに優先度の数値を付けます。たとえば「実験用 = 1000」「本番用 = 10000」のように、大きいほど優先される数値を割り当てます。重要なのは、Kueue のこの優先度は「キューでの順番」と「preemption (押しのけ)」に使われる点で、Kubernetes 標準の PriorityClass (Pod のスケジューリング優先度) とは別物として扱える設計になっていることです。学習ジョブのキュー順序だけを動かしたいのに、Pod レベルの優先度を上げてしまうと他の常駐 Pod まで影響を受けてしまう、という事故を避けられます。

### 5. preemption (押しのけ)
高優先度のジョブが来たとき、すでに走っている低優先度のジョブを一時停止して資源を空ける動作です。設定で「同じ ClusterQueue 内ではより低い優先度を押しのけてよい」「cohort 内で貸した資源は奪い返してよい/奪い返さない」などを細かく決められます。本番ジョブが研究ジョブを自動で押しのけ、本番が終われば研究ジョブが再開する、という流れを人手なしで実現します。

### ジョブ側からの使い方と suspend のからくり
Kueue が巧妙なのは、ジョブを最初から「停止状態 (suspend)」で作る点です。多くのジョブ系 CRD (Kubernetes 標準の Job や、各種の分散学習用 Job) は `suspend: true` というフィールドを持ち、これが true の間は Pod が起動しません。

```
ジョブにラベルを付けて投入:
  kueue.x-k8s.io/queue-name: <投げ先 LocalQueue>
  kueue.x-k8s.io/priority-class: <優先度>
spec.suspend: true            ← 最初は停止状態で作られる
```

Kueue はこのジョブを Workload (Kueue 内部での管理単位) として認識し、クォータと優先度を確認します。「いまこの瞬間に、要求された資源を割り当ててよい」と判断したら、`suspend` を `false` に書き換えます。すると初めて Pod が起動し、GPU ノードへ配置されます。つまり「列で待つ = suspend のまま待機」「実行許可 = suspend 解除」という、標準スケジューラと衝突しないエレガントな実現になっています。

### 全体の流れ (ステップ分解)

```
ユーザ: 学習ジョブを投げる (queue-name, priority, suspend=true)
   │
   ▼
Kueue: Workload を LocalQueue → ClusterQueue の列に登録
   │
   ▼
Kueue: ClusterQueue のクォータに空きがあるか?
   ├─ 空きあり → suspend を解除 → 標準スケジューラが Pod をノードへ配置 → 実行開始
   └─ 空きなし → 列で待機 (suspend のまま)
           │
           ▼  もし高優先度ジョブが来たら
       走っている低優先度ジョブを preempt (中断) して場所を空ける
```

### all-or-nothing admission (まとめて許可)
分散学習では「ワーカー全員が揃って初めて学習が始まる」性質があります。一部だけ起動して残りが起動できないと、起動済みのワーカーが GPU を握ったまま無駄に待ち続ける「デッドロック」になります。Kueue は Workload 単位で「要求する全 Pod 分の資源が確保できるか」を判定し、確保できないなら何も起動しません。これにより半端な起動による GPU の無駄遣いを防ぎます (gang scheduling、すなわち集団同時起動の発想)。Workload は複数の PodSet (役割ごとの Pod 群、例: Master と Worker) を持てるので、全 PodSet 分の資源を一括で確保するか、まったく確保しないかの二択になります。

## 手法の系譜と主要論文
Kueue 自体は学術論文ではなく、Kubernetes コミュニティ (SIG-Scheduling および WG-Batch) が2022年頃から開発するオープンソースですが、背景には長い研究史があります。

- バッチスケジューリングと公平配分の起源: HPC の世界では Slurm が代表格です。Yoo, Jette, Grondona "SLURM: Simple Linux Utility for Resource Management" (JSSPP 2003) は、ジョブを列に並べ、優先度と fair-share (各ユーザに公平に資源を割り当てる方式) で順番を決めるバッチスケジューリングの作法を確立しました。Kueue はこの「HPC 流のバッチ管理」を、もともと常駐サービス向けに作られた Kubernetes に持ち込む試みと位置づけられます。

- gang scheduling の起源: Ousterhout "Scheduling Techniques for Concurrent Systems" (ICDCS 1982) は、相互に通信するプロセス群を同時に実行する必要性 (coscheduling) を論じました。分散学習の「全ワーカーが揃わないと始まらない」性質はまさにこの問題で、all-or-nothing admission の理論的背景になっています。

- Kubernetes 上の先行バッチスケジューラ: Kueue 以前に Volcano (CNCF プロジェクト、もとは Baidu の kube-batch) や Apache YuniKorn がありました。これらは標準スケジューラを置き換える形で、gang scheduling や bin-packing (資源を隙間なく詰め込む最適化) を強力に実装しました。Kueue の設計思想はこれらと対照的で、「標準スケジューラを置き換えず、その手前で admission (実行許可) だけを制御する」点に特徴があります。これにより Kubernetes 本体の進化に追従しやすく標準機能と衝突しにくい代わりに、Volcano ほど高度な配置最適化は持ちません。実務では Kueue (admission/クォータ) と Volcano や標準スケジューラ (配置) を組み合わせる構成も見られます。

- 大規模クラスタスケジューリングの実務知見: Google の Borg を論じた Verma et al. "Large-scale cluster management at Google with Borg" (EuroSys 2015) や、Microsoft の Boutin et al. "Apollo" (OSDI 2014) は、数万台規模で「優先度・クォータ・preemption」を運用する難しさと有効性を示しました。Kueue が採り入れているクォータ借用や preemption は、こうした大規模運用の知見の延長にあります。

- GPU クラスタに特化したスケジューリング研究: Xiao et al. "Gandiva" (OSDI 2018, Microsoft) は、深層学習ジョブが「試行錯誤的で、early feedback が得られればすぐ捨てられる」性質を持つことに着目し、ジョブの途中移動 (migration) や time-slicing で GPU 利用率を高めました。Gu et al. "Tiresias" (NSDI 2019, Michigan/MSR) は、ジョブの所要時間が事前に分からない問題に対し、これまでに消費した GPU 時間に基づく Least-Attained-Service ベースのスケジューリングを提案しました。これらは「クォータと優先度」を超えた、深層学習固有のスケジューリング最適化の系譜で、Kueue の admission 層と組み合わせて使える知見です。

## 論文の実験結果 (定量データ)
Kueue は単一の論文で評価されるシステムではないため、関連研究の定量結果でその意義を裏付けます。

- preemption とクォータ借用による利用率向上 (Borg, EuroSys 2015): Borg の論文では、production (本番) と best-effort (低優先度) のジョブを同じセルに混載し、preemption で本番を守りつつ低優先度ジョブで空きを埋めることで、クラスタの実効利用率を大きく引き上げられることが示されています。報告では、優先度分離・詰め込みにより本番専用と非本番を別々のクラスタで運用するより必要マシン台数を二桁パーセント (おおむね20〜30%程度) 削減できたとされます。Kueue の cohort 借用と preemption は、この「本番を守りながら空き時間を低優先度で埋める」考え方を Kubernetes 上で再現するものです。

- GPU 固有スケジューリングの効果 (Gandiva, OSDI 2018): Gandiva は、深層学習ジョブの introspection (途中の損失曲線を見て見込みのないジョブを早期に止める) と migration を併用することで、ハイパーパラメータ探索のような多数ジョブのワークロードにおいて、素朴な FIFO 配置に対しクラスタ全体のジョブ完了スループットをおよそ最大で数倍に高めた事例を報告しています。要点は「GPU を遊ばせない配置の工夫が、台数を増やさずに研究の回転を速める」ことです。

- 所要時間予測なしでの待ち時間改善 (Tiresias, NSDI 2019): Tiresias は、ジョブ所要時間を予測せずとも Least-Attained-Service と二次元 (GPU 数 × 時間) のスケジューリングにより、平均ジョブ完了時間 (JCT) を既存方式に対し報告でおよそ最大5倍程度短縮できることを示しました。JCT (Job Completion Time) は「投入から完了までの実時間」で、研究者の体感待ち時間そのものを測る指標です。

これらの指標 (実効利用率、ジョブ完了スループット、平均完了時間) は、いずれも「希少資源をどれだけ無駄なく・公平に・締め切りを守って使えたか」を測るものです。GPU が高価であるほど、数パーセントの利用率改善や数倍の待ち時間短縮が大きなコスト差・研究速度差につながります。Kueue 自身は admission とクォータに範囲を絞っていますが、上記研究が示す効果の多くはクォータ分離・preemption・gang scheduling という、Kueue が提供する機能の組み合わせで現場に持ち込めます。

## メリット・トレードオフ・限界
メリット:
- 希少で高価な GPU を「早い者勝ち」ではなく、クォータと優先度で計画的に配分できる。
- ジョブが大量に投げられても、列に並ぶだけで取り合いによる失敗が起きない。
- suspend ベースなので Kubernetes 標準スケジューラと衝突せず、既存の仕組みにそのまま乗る。標準機能の進化に追従しやすい。
- 締め切りのある本番ジョブが、実験ジョブを自動で押しのけられる (preemption)。
- cohort によるクォータ借用で、チーム間の遊休資源を融通できる。

トレードオフ・限界:
- ResourceFlavor / ClusterQueue / LocalQueue / WorkloadPriorityClass と概念が多く、設定の学習コストが高い。
- preemption で中断されたジョブは、チェックポイント保存をしていないと進捗を失う。preemption を活かすには「いつ止められても再開できる」学習設計が前提になる。
- クォータを「GPU 枚数」のような粗い粒度で静的に決めると、実際の GPU メモリ (VRAM) 使用量との乖離で無駄や詰まりが生じる。1枚の GPU を複数の小ジョブで共有する fractional GPU は標準 Kubernetes の整数リソースモデルとは相性が悪く、別の仕組み (MIG, time-slicing) が要る。
- あくまで「実行許可の門番」であり、配置の最適化 (どのノードに固めるか) やジョブ内部の効率 (GPU 使用率や通信効率) は管理しない。そこは標準スケジューラや学習コードの責任。
- 未解決の課題として、ジョブの所要時間を事前に知らないまま最適な順序付けや preemption を決めるのは難しく、所要時間予測を取り入れたスケジューリングは研究途上です (Tiresias はあえて予測しない方向で挑んだ例)。

## 発展トピック・研究の最前線
- トポロジーアウェアスケジューリング: マルチノード分散学習では、GPU 間が高速なネットワーク (NVLink, InfiniBand) でつながったノードに固めて配置すると通信が速くなります。Kueue にも topology-aware なスケジューリング (TopologyAwareScheduling) の取り組みが進んでおり、「どのノード・どのラック・どのブロックに固めるか」をクォータ管理と統合する流れがあります。
- 動的クォータと予算制御: 静的なクォータでなく、コスト予算 (1日いくらまで) や時間帯に応じてクォータを動かす研究・実装が登場しています。fair sharing の重み付けを動的に変える方向も含まれます。
- 所要時間予測スケジューリング: 学習ジョブの所要時間や GPU メモリ使用量を過去の履歴から予測し、SJF (Shortest Job First) 的な並べ替えや backfill (空き時間に短いジョブを差し込む) を行う研究の系譜 (Gandiva, Tiresias) があります。これらの知見をクォータベースの admission と融合させる方向が今後の焦点です。
- GPU 分割と弾力性: MIG (Multi-Instance GPU) や time-slicing による GPU 共有、ワーカー数を動的に伸縮させる弾力的スケジューリングを、クォータ会計とどう整合させるかが新しい研究課題になっています。

## さらに学ぶための関連トピック
- [Kubeflow Training Operator (PyTorchJob)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0605_kubeflow-training-operator)
- [Flyte (Pipeline Orchestration)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1107_flyte-pipeline-orchestration)
- [MLflow (Tracking + Model Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1125_mlflow-tracking-registry)
- [KServe (Model Serving)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0524_kserve-model-serving)
