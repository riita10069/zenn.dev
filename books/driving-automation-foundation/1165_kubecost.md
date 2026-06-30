---
title: "Kubecost (コスト可視化)"
---

# Kubecost (コスト可視化)

## ひとことで言うと
Kubernetes クラスタ上で動くそれぞれのワークロード(学習ジョブ、推論サービスなど)が「実際に何ドルかかっているか」を計算して見せるツールです。とくに GPU のように高価な資源について、「どの実験・どのチーム・どの namespace にいくら使ったか」を後から追えるようにします。CNCF のオープンソース標準 OpenCost をベースにしています。

## 直感的な理解
クラウドの請求書を見ると「EC2 にいくら」「ストレージにいくら」という、サービス単位の合計は分かります。しかし機械学習チームが本当に知りたいのは別の粒度です。「あの高解像度の実験を1回回すのにいくらかかったのか」「研究用ジョブと本番推論で GPU 費用はどう配分されているのか」「GPU を常時1台温めている固定費は月いくらか」「誰のジョブが一番コストを食っているのか」——請求書はこれに答えてくれません。

問題の核心は「1台の GPU ノードの上で複数の Pod が動く」点です。クラウドの請求は「インスタンス1台でいくら」までしか分解せず、その上で動く複数 Pod に費用をどう割り振るか(配賦、はいふ)は教えてくれません。割り勘の総額は分かっても、誰が何を頼んだかの内訳が分からない状態です。この内訳を自動で計算するのが Kubecost です。

## 基礎: 前提となる概念

「ワークロード(workload)」とは、クラスタ上で動く処理のまとまりです。Kubernetes では Pod、それを束ねる Deployment / Job / StatefulSet、それらをグループ化する namespace(名前空間、論理的な区切り)などの単位があります。

「resource request と usage」: Kubernetes では各 Pod が「CPU を何コア、メモリを何 GB ください」という request(要求量)を宣言します。一方 usage(実使用量)は実際に使った量で、request と usage は一致しないのが普通です。過大な request は「予約したのに使わない」無駄を生みます。

「配賦(allocation / cost allocation)」とは、共有資源の費用を利用者ごとに割り当てる会計手続きです。「1ノードの費用を、その上の各 Pod に使用量比で割り振る」のがここでの配賦です。

「アイドルコスト(idle cost)」とは、確保したのに使われなかった資源の費用です。warm 維持で常時起動しているノードの空き時間や、過大な request による未使用分が原因で発生します。

「OpenCost」は CNCF が育てるコスト配賦の標準仕様・オープンソース実装で、Kubecost はそれをベースにした製品(コア部分は無償、上位機能は有償エディション)です。

## 仕組みを詳しく

Kubecost のコスト計算は、おおまかに「使った量 × 単価 = 金額」という掛け算を Pod ごとに行うことで成り立っています。ステップで分解します。

1. 使った量(usage)を集める
   Kubecost は前提として動いている監視基盤([Prometheus](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1169_prometheus-grafana))のメトリクスを使い、各 Pod が「CPU を何コア時間」「メモリを何 GB 時間」「GPU を何枚時間」使ったかを時系列で取得します。GPU の量は [DCGM Exporter](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1163_dcgm-gpu-monitoring) のメトリクスや、Kubernetes の `nvidia.com/gpu` リクエスト数から把握します。

2. 単価(price)を取得する
   そのノードがどんなインスタンスタイプ(例: GPU 1枚の高性能機)で、オンデマンドか・スポット(余剰枠の安い従量)か・予約(容量予約や Savings Plan)かを判定し、クラウドの料金表に基づき「1コア時間/1GB 時間/1GPU 時間あたり何ドルか」を割り出します。クラウドの課金明細(AWS なら Cost and Usage Report)と連携すると、より実支払額に近づけられます。

3. 配賦(allocation)する
   1ノードの費用を、その上の各 Pod の「使った量(または要求量 request)」の比率で割り振ります。GPU を占有する学習 Pod が1個乗っていれば、GPU 費用はほぼ全額その Pod に付きます。複数 Pod が CPU/メモリを分け合っていれば、その比率で分割します。

4. ラベルや namespace で集計して見せる
   Pod に付いた Kubernetes ラベル(例: `team=perception`, `experiment=high-res-bev`)や namespace ごとに金額を合計し、Web ダッシュボードで「この namespace は今月 $X」「この実験は $Y」と表示します。

数値例(イメージをつかむための概算):

```
ノード: GPU 1枚のインスタンス  オンデマンド単価 = 仮に $2.0/時間
このノード上の Pod:
  - 学習 Pod A: GPU を1枚占有、6 時間稼働
    → GPU 費用ほぼ全額が A に配賦 ≈ $2.0 × 6 = $12
  - pause Pod(warm 維持用): CPU/メモリほぼゼロ、GPU 不使用
    → 配賦額ほぼ $0
    (ただしジョブが無い空き時間のノード費用は warm 維持の固定費=アイドルコストとして別途見える)
```

「実験 A は約 $12 かかった」と分かれば、ハイパーパラメータ探索(複数設定を試す)のコスト試算や、warm node 維持費の妥当性判断に使えます。

Kubecost はアラートも出せます。「namespace X の今月コストが予算 $Z を超えた」「アイドルコストが全体の N% を超えた」といった通知です。アイドル分を切り分けて見せるので、無駄の発見に直結します。

## 手法の系譜と主要論文

Kubecost / OpenCost は運用ツールで論文の対象ではありませんが、背景にある「共有クラスタで資源費用を公平に配賦し、無駄を減らす」というテーマは、データセンター・スケジューリング研究で扱われてきました。

- Verma ら "Large-scale cluster management at Google with Borg"(EuroSys 2015)。Google の Borg が大規模共有クラスタでジョブに資源を割り当て、利用率を高める仕組みを説明しました。共有クラスタでは「誰がどれだけ使ったか」を測って配分する必要があるという問題意識は、Kubecost のコスト配賦と同根です。Borg は後の Kubernetes の直接の祖先でもあります。

- Reiss ら "Heterogeneity and Dynamicity of Clouds at Scale: Google Trace Analysis"(SoCC 2012)。実クラスタのトレース解析で「資源の要求量(request)と実使用量(usage)に大きな乖離があり、過剰要求が利用率を下げている」ことを示しました。Kubecost の「アイドルコスト可視化」「request と usage の差の指摘」は、まさにこの種の無駄を金額で見える化するものです。

- 公平な配賦の理論的基盤としては、Ghodsi ら "Dominant Resource Fairness: Fair Allocation of Multiple Resource Types"(NSDI 2011)があります。CPU・メモリ・GPU など複数種類の資源を同時に公平配分する DRF という考え方で、共有クラスタのスケジューラ(Mesos, YARN, Kubernetes 系)に影響を与えました。DRF の核心は「各ユーザの dominant resource(そのユーザが最も多く要求している資源種類)のシェアを揃える」という基準です。たとえば A は CPU 寄りのジョブ、B は GPU 寄りのジョブを出しているなら、A の CPU シェアと B の GPU シェアが等しくなるよう配分します。これにより、1種類の資源だけ見て配分すると起きる不公平(GPU を独占したいユーザが CPU を捨て値で受け取ってしまう等)を避けられます。コスト配賦は、この「公平配分」を金額に翻訳した事後会計版と捉えられます。

学術的含意は「共有資源は測って配賦しないと、誰がどれだけ無駄にしているか分からず、最適化もできない」ということ。Kubecost はそれをドル単位で実現します。

## 論文の実験結果(定量データ)

- Reiss ら(SoCC 2012)は、Google の実トレースで、ジョブが要求する CPU/メモリ(request)に対して実使用量(usage)が大きく下回るケースが多数あり、request ベースで割り当てると利用率が低くとどまることを定量的に示しました。これは「request だけで配賦すると、実際には使っていない資源にも費用が付き、過剰要求の無駄が見えなくなる」ことを意味し、Kubecost が request と usage の両面を見せる設計の根拠になっています。指標の意味: request/usage 比は「予約したが使わなかった度合い」を測り、これが大きいほどアイドルコスト=削れる無駄が多いことを示します。

- Borg 論文(EuroSys 2015)では、本番ジョブとベストエフォートジョブを同じマシンに混載(co-location)し、優先度とリソース再利用でマシン利用率を高めていることが報告されています。これは「測って配賦し、空きに低優先度ジョブを詰める」ことでコスト効率を上げる思想で、Kubecost が可視化するアイドル分を「埋める」運用の理論的背景です。

- GPU 特化の文脈では、Weng ら "MLaaS in the Wild"(NSDI 2022、Alibaba PAI)が本番クラスタで GPU 利用率の中央値が低位に偏ること、多くのタスクが GPU を1枚丸ごとは必要としないことを示し、GPU 共有(複数ジョブで1枚を分け合う)の必要性を定量的に裏づけました。GPU を共有すると配賦はさらに難しくなります——「1枚を3ジョブで時分割した」場合、固定費である GPU 1枚分の費用を3者にどう割るのかは自明ではないからです。これは Kubecost / OpenCost が現在も仕様拡張で取り組んでいる課題と直結します。

これらは「共有クラスタでは測定と配賦こそがコスト最適化の出発点である」という主張を、大規模実データで裏づけています。さらに踏み込むと、可視化は「無駄を見つける」までしかできず、見つけた無駄を「埋める/削る」のは別の最適化(右サイジング、GPU 共有、スポット活用)の仕事であり、ここが研究・実務の継続的な主戦場になっています。

## メリット・トレードオフ・限界

メリット:
- クラウド請求書では分からない「Pod / namespace / 実験単位」の細かいコストが分かる
- GPU のような高額資源について、実験ごとの費用対効果を定量化できる
- アイドルコスト(遊休費用)を切り出せるので、無駄の発見・削減につながる
- OpenCost ベースのコア部分はオープンソースで無償。Prometheus があれば追加導入しやすい

トレードオフ・限界:
- 算出されるのは推定値で、配賦ロジックや単価設定の前提に依存し、実請求と完全一致はしない(精度向上には課金明細連携などの設定が要る)
- Prometheus への依存があり、メトリクスが欠ければ計算も欠ける
- スポット価格の変動や予約の前払い費用(Savings Plan、容量予約の予約料)など、複雑な料金体系を正確に反映するには追加設定が要る。とくに「予約は使わなくても課金される」分の配賦は難しい
- 上位機能(長期保存、マルチクラスタ集約、エンタープライズ機能)は有償エディションになる場合がある
- 配賦は「過去の事実の会計」であり、それ自体は無駄を減らさない。可視化を受けて人間/自動化が是正して初めて節約になる(未解決の課題: 可視化から自動最適化への橋渡し)

## 発展トピック・研究の最前線

- FinOps への接続: Kubecost のコスト配賦は、クラウド財務運用(FinOps)の実践——chargeback(部門への費用付け替え)/showback(費用の見える化)、予算管理、コスト異常検知——の入力になります。MLOps では「実験あたりコスト」を学習効率の指標(精度改善あたりのドル)に結びつける動きがあります。
- right-sizing 自動化: request と usage の乖離を検出し、resource request を自動で適正化(VPA, Vertical Pod Autoscaler 等)してアイドルを減らす方向。Reiss らが指摘した過剰要求問題への直接の対処です。
- GPU 共有のコスト配賦: MIG(GPU 分割)や time-slicing で1枚を複数ジョブが共有する場合、「分割単位での配賦」が必要になり、OpenCost コミュニティで仕様拡張が議論されています。
- スポット/予約の最適混在: コスト可視化を起点に、中断耐性のある学習をスポットへ、即応が要るものをオンデマンド/予約へ振り分ける最適化。warm node の固定費([Warm GPU Node (pause Pod)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1157_warm-gpu-node))が見えることで「scale-to-zero にしない判断」のコスト面の裏づけも取れます。

## さらに学ぶための関連トピック
- [Prometheus + Grafana](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1169_prometheus-grafana)
- [DCGM Exporter (GPU 監視)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1163_dcgm-gpu-monitoring)
- [Warm GPU Node (pause Pod)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1157_warm-gpu-node)
- [NVIDIA Device Plugin](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1180_nvidia-device-plugin)
- [Bottlerocket OS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1100_bottlerocket-os)
