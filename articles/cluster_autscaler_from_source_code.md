---
title: "Cluster Autoscalerの挙動をざっくり理解する" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["Cluster Autoscaler", "Kubernetes", "k8s"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

## Abstruct

KubernetesのCluster Autoscalerについて仕組みを理解しているだろうか。
なんとなく、「負荷が高まればNodeが増えるんでしょ。」くらいの理解をされているかもしれない。しかし、Production環境でうまく負荷を捌けなかったりすれば、原因を解明する必要が出てくる。そんな時に役に立つ文献になって欲しいと思い、この文章を書くことにした。

実際に、[autoscaler/cluster-autoscaler at 0c3e9d15d183ffa90e9d03ac79c51db22c68a792 · kubernetes/autoscaler · GitHub](https://github.com/kubernetes/autoscaler/tree/0c3e9d15d183ffa90e9d03ac79c51db22c68a792/cluster-autoscaler)のソースコードを読み挙動を考察した。
もし間違いなどがあれば指摘していただきたい。

## Introduction: クラスターオートスケーラーとは何か

KubernetesのNodeを自動で増やしたり減らしたりしてくれるコンポーネントである。
どのNodeを増やすか減らすかという観点では、Node Pool毎に機能するものであり、特定のNodePoolをwatchしていて、それに対してクラスタを増やしたり減らしたりしている。

また、NodeGroupというのは、CA側の概念であり、実態はCloudProviderによって変わる。EKSの場合は、**Amazon EC2 Auto Scaling グループ**である。(出典：[Auto Scaling - Amazon EKS](https://docs.aws.amazon.com/ja_jp/eks/latest/userguide/autoscaling.html) 2022年7月15日時点)

自動で増やしたり減らしたりしてくれると前述したが、そのタイミングについて整理する。

まずは、増やす必要のある時だ。
これは、kube-schedulerがPodにPendingをつけた時だ。
PendingのPodがあれば、CAはNodeのDesiredの値を増やすようにできている。

逆に、減らす必要のある時は、PodのRequest値が小さくて、そのPodを他のNodeへ移行できる時だ。
その時は、そのNodeを終了させることをする。

## Material: クラスターオートスケーラーの挙動について

これから、CAの詳細な挙動をできるだけ、わかりやすく伝えたいと思う。

### メイン関数

まずは、main関数から紹介する。
https://github.com/kubernetes/autoscaler/blob/f990344b6cb681e3c6a3cb344e71aa6b61afdea0/cluster-autoscaler/main.go#L385-L508

main関数では、いくつかのgoroutineを起動している。

1.最もコアなロジックを実行するのが、RunOnce関数だ。
https://github.com/kubernetes/autoscaler/blob/f990344b6cb681e3c6a3cb344e71aa6b61afdea0/cluster-autoscaler/core/static_autoscaler.go#L230-L623

ここでは、Scale Upの必要を考慮し、必要なければScaleDownの必要を考慮します。必要がある方を実行します。というようなロジックになっている。
ScaleUpとDownの詳細は後ほどまた説明する。

2.LeaderElection
3.ClusterStateRegistryの更新
https://github.com/kubernetes/autoscaler/blob/f990344b6cb681e3c6a3cb344e71aa6b61afdea0/cluster-autoscaler/clusterstate/clusterstate.go#L118-L146

Clusterの状態に対してCacheをこのStructに保持するようにしている
4.ClusterSnapshotの更新
https://github.com/kubernetes/autoscaler/blob/f990344b6cb681e3c6a3cb344e71aa6b61afdea0/cluster-autoscaler/simulator/basic_cluster_snapshot.go#L28-L35

Expansion OptionやRescheduling Simulationでは、Scheduler FrameworkのScheduling Cycleの計算を実際に行うため、Clusterの現状のNodeのSnapshotが必要。
NodeのHealthCheck
CloudProvider側でNodeのProvisioningに失敗したかどうかは、Resyncで判断することもある
つまり、CA側で作っている予定のNodeが`--max-node-provision-time`秒後になっても存在しなかったら、CA側の状態としてそのNodeを消す

### スケールアウト

Expansion Optionを生成する

1.どのNodeGroupのノードが何台必要かを計算する
https://github.com/kubernetes/autoscaler/blob/0c3e9d15d183ffa90e9d03ac79c51db22c68a792/cluster-autoscaler/expander/expander.go#L44-L49

Optionそのものは上記のように、「あるNodeGroupを増やした場合、Nodeをいくつ増やすとどのPodがSchedule可能になるか。」を表すものである。
2.どのようにExpansion Optionを計算しているかは下記
https://github.com/kubernetes/autoscaler/blob/0c3e9d15d183ffa90e9d03ac79c51db22c68a792/cluster-autoscaler/core/scale_up.go#L266-L319

実際のClusterSnapshotを用いて、Scheduling FrameworkのlisterとしてPodを振る舞わせることで、「あるPodがあるNodeGroupにSchedule可能であるか」を検証している。
これを実行するInterfaceはPredicateCheckerで、実態のStructは、SchedulerBasedPredicateCheckerというもの
https://github.com/kubernetes/autoscaler/blob/0c3e9d15d183ffa90e9d03ac79c51db22c68a792/cluster-autoscaler/simulator/scheduler_based_predicates_checker.go#L34-L42

また、Nodeを何台増やせば良いかを計算しているのは、Estimaterという別のInterfaceのようで、実態のStructは、BinpackingNodeEstimatorというもの
https://github.com/kubernetes/autoscaler/blob/0c3e9d15d183ffa90e9d03ac79c51db22c68a792/cluster-autoscaler/estimator/binpacking_estimator.go#L38-L43

Binpacking Algolithmは以下のようなもの
 * まずは、Podのスコアリングを行う
   * すべてのPodについてスコアを計算し、PodInfo構造体を返す。
   * スコアは cpu_sum/node_capacity + mem_sum/node_capacity として定義される。
   * より大きな要件を持つ Pod は最初に処理されるべきであり、したがってより高いスコアを持つ。
 * 上記にあるように、より高いスコアを持つPodが優先的に処理されるようにソートを行う
 * スコアの高い順に、PredicateCheckerを利用してスケジューリング可能なNodeにスケジュールする。
   * 見つからなければ、新たにNodeを追加した場合を想定する
 * 最終的に新規に追加したNode数を計算する
 * 詳細は、下記リンクを参照
   * [Bin packing problem - Wikipedia](https://en.wikipedia.org/wiki/Bin_packing_problem)
* Expander Strategyの実行
  * 先ほど計算したOptionの中から適切なOptionを選択する
https://github.com/kubernetes/autoscaler/blob/0c3e9d15d183ffa90e9d03ac79c51db22c68a792/cluster-autoscaler/expander/expander.go#L28-L40

上記のように利用可能なExpanderは6種類
* RandomExpanderName は、ランダムにノードグループを選択
* MostPodsExpanderName は、最も多くのポッドに対応するノードグループを選択
* LeastWasteExpanderName CPUとMemoryの残りが最も少なくなるノードグループを選択
* PriceBasedExpanderName 最も費用対効果が高く、クラスタに望ましいノードサイズと一致するノードグループを選択
* PriorityBasedExpanderName は、グループ名に割り当てられたユーザー設定の優先順位に基づいてノード グループを選択
* GRPCExpanderNameは、gRPCクライアントエクスパンダーを使用して外部のgRPCサーバーを呼び出し、スケールアップのためのノードグループを選択

### スケールイン

* UpdateUnneededNodes
  * UpdateUnneededNodes は、どのノードが不要か、つまりすべてのポッドが別の場所にスケジュールできるかを計算し、それに応じて unneededNodes マップを更新する。また、ポッドを再スケジュールできる場所とノードの使用率も計算する。

https://github.com/kubernetes/autoscaler/blob/0c3e9d15d183ffa90e9d03ac79c51db22c68a792/cluster-autoscaler/core/scaledown/legacy/legacy.go#L339-L514

 * 具体的な手順は以下
   * スケールダウンが可能であるかを確認
     * スケジュールされたexpendable podと優先順位の低いpodの先取りを待つpodが、ノード削除を防ぐことができる。
   * Utilization Check
     * NodeのAllocate可能な値と、全てのPodのリクエスト値の比率を計算する
     * NodeAはCPUが80%使用されているなどの値が得られる
   * Thresholdの値よりも低いNodeを抽出し、CandidateNodeとする。
   * empty nodeを見つける
     * empty nodeとは、全てのPodが以下の条件を満たすようなNodeのこと
       * Daemon Set
       * Mirror Pod
       * ReplicaSet or Job に関連する
       * などなど(多すぎるので割愛)
     * empty nodeはUnneededNodeとなる
     * Not empty nodeでもいくつかの条件を満たすとUnneededNodeとなるが割愛
   * UnneededNodeにあるPodは他のNodeに移行可能かを検証する
     * 移行可能であれば、そのNodeにはUnneededのTaintがつけられる
   * `--max-graceful-termination-sec`秒後に実際にNodeが削除される
     * この間に、kubeletから、Podへの終了処理が行われるので、GracefullShutdownを実行する
 * この計算は、CA が管理するノードに対してのみ行われる。

## Conclusion: まとめ

本記事では、実際のCluster Autoscalerがどのような基準でNodeを増やしたり減らしたりしているのかを説明した。
Cluster Autoscalerの状態は、kube-system namespaceにあるConfigMapである、`cluster-autscaler-status`を見ることでもわかる。疑問が湧いたら、この記事にある基本的な挙動を念頭に前述のConfigMapの中身を見てみてはいかがだろうか。

また、実際のスケールのメカニズムにはSchedulerやHPEの挙動と深く関係している。そのためそれらのコンポーネントについても理解を進める必要がある。

## FAQ

Q. スケールアウトの際に、PodがUnscheduledになってからではタイミングとして遅くないのか。Nodeの空き容量が少なくなってきた時点で早めにスケールする(like HPA)ようなことはできないのか。
A. Cluster Autosaclerの場合は基本的にそういった仕組みはない。あくまでもPodのPendingを起点にトリガーされる。しかし、対処法はいくつか存在する。
有名な方法は、役割のないPodを用意する方法だ。
Nodeの利用率については、PodのCPU利用率ではなく、PodのRequest値によって決まるものである。そのため、ある程度のRequest値を持った空のPodをデプロイしておくことで、実際よりも多くのRequest値があるように見せることができる。
そして、KubernetesのPodには、[Pod Priority and Preemption](https://kubernetes.io/docs/concepts/scheduling-eviction/pod-priority-preemption/)の仕組みがある。
この空のPodのPod Priorityを低くしておくことで、スケールアウトが必要な時にこのPodをPreemptionして別のPodを置くことができる。そして、その時にこの空のPodがUnscheduledになるので、Cluster AutoscalerはNodeのDesired値を上げるような挙動をとる。鋭いスパイクに対応するために、早い段階でスケールアウトさせるような仕組みが出来上がる。

Q. 消されたくないNodeを明示的に指定するには
A. cluster-autoscaler.kubernetes.io/scale-down-disabled をNodeのAnnotationに設定すれば消えないとのこと
