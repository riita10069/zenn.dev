---
title: "Prometheus + Grafana"
---

# Prometheus + Grafana

## ひとことで言うと
Prometheus(プロメテウス)は「システムのあちこちから数値データ(メトリクス)を定期的に集めて時系列で保存する」道具、Grafana(グラファナ)は「その数値をグラフやダッシュボードで見やすく表示する」道具です。2つを組み合わせると、「クラスタや GPU が今どうなっているか・過去どうだったか」を一目で把握でき、異常があれば通知できます。クラウドネイティブ監視の事実上の標準スタックです。

## 直感的な理解
たくさんのサーバやプログラムが動くシステムは、計器盤のない飛行機を飛ばすようなものです。高度(メモリ使用量)、速度(リクエスト数)、エンジン温度(GPU 温度)が分からなければ、問題が起きてから墜落して初めて気づくことになります。Prometheus は各部にセンサーを付けて値を読み取り続ける「データロガー」、Grafana はそれを操縦席の計器盤に表示する「ディスプレイ」です。

「問題が起きてから手作業で調べる」のでは遅すぎ、過去との比較もできません。たとえば「学習が遅い」と気づいたとき、過去のメトリクスがあれば「3時間前から GPU 利用率がじわじわ下がっている」「同じ頃ストレージ I/O が跳ねている」と原因を時系列でたどれます。記録していなければ、何も分かりません。これが継続監視の価値です。

## 基礎: 前提となる概念

「メトリクス(metrics)」とは「ある瞬間の数値の計測値」で、時刻とセットで記録されます。例: GPU 利用率97%、メモリ使用量12.4GB。これを時間順に並べたものが「時系列データ(time series)」です。

メトリクスにはいくつかの型があります。最も基本的なのが「カウンタ(counter)」=増え続ける数(累計リクエスト数など)と「ゲージ(gauge)」=上下する瞬間値(メモリ使用量、温度など)です。カウンタは「累計」なので、そのままでは意味が薄く、「1秒あたりどれだけ増えたか(レート)」に変換して使います。

「CNCF(Cloud Native Computing Foundation)」は、Kubernetes をはじめとするクラウドネイティブ技術を中立的にホストする団体です。Prometheus も Grafana 関連プロジェクトも、この周辺で育ち広く採用されています。

監視には大きく2つの仕事があります。
1. 集めて溜める(収集・保存): あちこちのコンポーネントから数値を定期的に取得し、時系列 DB に貯める
2. 見せる・知らせる(可視化・通知): 溜めたデータをグラフにし、異常があればアラートを出す

1を担うのが Prometheus、2の「見せる」を担うのが Grafana、「知らせる」を担うのが Prometheus 付属の Alertmanager です。

## 仕組みを詳しく

### Prometheus 側

Prometheus の中核動作は「スクレイプ(scrape)」です。スクレイプとは「監視対象が公開する HTTP エンドポイント(例: `http://target:9400/metrics`)を、決まった間隔で取りに行く」ことです。多くの監視ツールが「対象がデータを送りつける(push)」方式なのに対し、Prometheus は「Prometheus 側が取りに行く(pull)」方式である点が特徴です。pull 方式だと、(a) どの対象を監視しているかが Prometheus の設定に集約され、(b) 対象が生きているか(取りに行けるか)自体も `up` メトリクスとして監視でき、(c) 監視対象側は単に値を出すだけでよく送信先を知らなくて済む、という利点があります。動的にスケールする Kubernetes では、Service Discovery(対象の自動発見)と組み合わせ、Pod の増減を自動追従します。

取りに行った先には「メトリクス名{ラベル} 値」のテキストが並びます。例:

```
node_memory_MemAvailable_bytes{instance="gpu-node-1"} 1.34e10
DCGM_FI_DEV_GPU_UTIL{gpu="0",Hostname="gpu-node-1"} 97
```

集めた数値は時刻つきで時系列 DB(TSDB, Time Series DataBase)に保存されます。各データ点には「ラベル」が付いており、これが Prometheus の強力さの源です。ラベルとは `{gpu="0", Hostname="gpu-node-1"}` のような key=value の付帯情報で、同じメトリクス名でも「どの GPU か」「どのノードか」を多次元で区別します。

保存データには「PromQL(プロムキューエル、Prometheus Query Language)」という専用言語で集計をかけられます。例:

```promql
# 全 GPU の利用率の平均
avg(DCGM_FI_DEV_GPU_UTIL)

# ノードごとの空きメモリ(GB 換算)
node_memory_MemAvailable_bytes / 1024 / 1024 / 1024

# 直近5分間の HTTP リクエスト増加レート(秒あたり)
rate(http_requests_total[5m])
```

`rate(...[5m])` は「直近5分間で1秒あたりどれだけ増えたか」を計算し、カウンタから「秒あたりの速度」を取り出す定番です。`avg by (Hostname)(...)` のようにラベルでグループ集計もできます。

### Grafana 側

Grafana は「データソース(data source)」として Prometheus を登録し、PromQL クエリを投げて結果をグラフにします。Grafana 自体はデータを保存せず、表示と操作に専念します。

Grafana でできることの例:
- 折れ線グラフ・ゲージ・ヒートマップなど多彩なパネルで可視化
- 複数パネルを並べた「ダッシュボード」を作る(例: 「GPU クラスタ概況」に利用率・温度・VRAM・電力を並べる)
- 変数(variable)で「どのノードを表示するか」をドロップダウンで切り替え
- しきい値超過でパネルを赤くする、アラートを出す

全体の流れを ASCII 図で示します。

```
[監視対象たち]              [収集・保存]            [可視化/通知]
node-exporter (CPU/メモリ) ─┐
DCGM Exporter (GPU)        ─┼─ pull(scrape) ─> Prometheus ──PromQL──> Grafana ──> 人間が見る
kube-state-metrics (Pod等) ─┤                  (時系列DB)
各種アプリ(/metrics)      ─┘                       │
                                                    └─ alert ─> Alertmanager ─> Slack/メール等
```

GPU 監視に限れば、DCGM Exporter が GPU の数値を出し、Prometheus がスクレイプして溜め、Grafana が「GPU ダッシュボード」として見せる分担になります。NVIDIA や Grafana コミュニティが公開している DCGM 向けの定番ダッシュボード JSON をインポートすれば、ゼロから作らずにすぐ使えます。

## 手法の系譜と主要論文

Prometheus と Grafana はプロダクトであり論文の主題ではありませんが、その設計思想は大規模運用の経験と研究に根ざしています。

- Prometheus は SoundCloud で生まれ(2012年頃)、Google 社内の監視システム Borgmon に強く影響を受けたデータモデル(メトリクス名+ラベルの多次元時系列)を採用しています。2016年に CNCF に参加し、Kubernetes に次ぐ2番目のホストプロジェクトになりました。

- Beyer, Jones, Petoff, Murphy 編 "Site Reliability Engineering: How Google Runs Production Systems"(O'Reilly, 2016)。この本の監視の章は「4つのゴールデンシグナル(latency / traffic / errors / saturation = 遅延・流量・エラー・飽和)を測れ」という指針を示し、現代のメトリクス監視のベースになりました。Prometheus + Grafana はこの4シグナルを測る実装手段として広く採用されています。

- 時系列 DB の効率的保存については、Pelkonen ら "Gorilla: A Fast, Scalable, In-Memory Time Series Database"(VLDB 2015、Facebook)が代表的です。タイムスタンプの差分の差分(delta-of-delta)エンコードと、浮動小数の XOR 圧縮という2つのアイデアで、時系列データを高い圧縮率で保存し高速クエリする手法を提案しました。具体的には、メトリクスは多くの場合「ほぼ等間隔のタイムスタンプ」と「ゆっくり変化する値」を持つため、前回値との差分(さらにその差分)だけを記録すれば、ほとんどのデータ点はごく少ないビット数で表せる、という観察を利用します。Prometheus の TSDB(v2 以降のブロックストレージ)もこの系譜の圧縮アイデアを取り入れ、大量メトリクスを現実的なストレージで扱えるようにしています。

- データモデルの「メトリクス名 + ラベル」という多次元設計は、Google の社内監視 Borgmon の経験を一般化したものです。従来の監視ツールが「サーバごとに別々のメトリクスファイル」を持っていたのに対し、Borgmon/Prometheus は同じメトリクス名にラベルを付けて区別する設計にしたことで、`avg by (datacenter)(...)` のような「軸を選んで集計する」操作が自然にできるようになりました。これはディープラーニングのテンソルで言えば、メトリクスを「(時刻, ラベル次元1, ラベル次元2, ...)」の多次元配列として扱い、任意の軸で reduce できるようにしたことに相当します。

これらの含意は「メトリクスを正しいデータモデル(多次元ラベル付き時系列)で集め、効率よく圧縮保存し、人間が読める形に可視化する」という監視の基本設計であり、Prometheus + Grafana はその実践形です。

## 論文の実験結果(定量データ)

- Pelkonen ら(VLDB 2015)は、Gorilla の圧縮で時系列データを1点あたり平均およそ 1.37 バイト程度まで圧縮できたと報告しています(素朴に float64 のタイムスタンプ+値で16バイト持つのに比べ、おおむね10倍以上の圧縮)。クエリレイテンシも、ディスクベースの従来システムに比べ桁違いに低い(ミリ秒オーダー)としています。指標の意味: 圧縮率は「同じストレージでどれだけ多くのメトリクスを保持できるか=監視のスケーラビリティ」を左右し、クエリレイテンシは「ダッシュボードやアラート判定の即応性」を決めます。

- Gorilla 論文ではさらに、メモリ常駐の設計によりクエリレイテンシが従来のディスクベース時系列ストアに比べおおむね2桁(数十倍〜)速くなったと報告されています。これが「ダッシュボードを開いた瞬間に過去24時間のグラフが描ける」体験や「数秒間隔でアラート条件を評価する」運用を可能にしました。圧縮率(1点あたりおよそ1.37バイト)とクエリ速度はトレードオフではなく両立しており、適切なエンコーディングが両方を同時に改善できることを示した点が研究的に重要です。

- SRE 本のゴールデンシグナルは定量的なベンチマークではなく方針ですが、その後の多くの運用組織が「saturation(飽和)= リソースがどれだけ限界に近いか」を継続監視することでインシデントを予兆的に検知できることを実務で示してきました。GPU クラスタでいえば VRAM 飽和や GPU 利用率の頭打ちがこれに当たります。たとえば VRAM 使用量が物理容量の95%に張り付き始めたら、次のバッチで OOM クラッシュする予兆と読めます——これはエラーが起きてから対応する事後対応ではなく、飽和を見て事前に手を打つ予兆検知の典型です。

これらは「適切なデータモデルと圧縮を備えた時系列監視が、大規模システムでも現実的なコストで成立する」ことの裏づけです。

## メリット・トレードオフ・限界

メリット:
- pull 方式とラベル付き多次元データモデルにより、Pod が増減する Kubernetes のような動的環境と相性が良い
- PromQL で柔軟な集計・加工ができ、Grafana で表現力の高いダッシュボードを作れる
- CNCF 標準で導入事例・既製ダッシュボード・ノウハウが豊富。多数の exporter(node-exporter, DCGM Exporter, kube-state-metrics 等)と素直に連携
- `up` メトリクスで「監視対象が生きているか」自体も監視でき、Service Discovery で動的追従

トレードオフ・限界:
- Prometheus は単一サーバの TSDB が基本で、超長期保存や超大規模・マルチクラスタ集約になると Thanos / Cortex / Mimir といった追加コンポーネントが必要
- カーディナリティ(ラベルの組み合わせ数=系列数)が爆発するとメモリ・ストレージを大量消費する。高カーディナリティなラベル(例: ユーザ ID、リクエスト ID)を付けると破綻する。ラベル設計が要
- セットアップとダッシュボード作成・保守に運用コストがかかる
- メトリクスは「数値の集計」に強い一方、個別イベントの詳細追跡は不得手で、ログ(Loki 等)やトレース(Tempo / Jaeger 等)は別ツールが必要(可観測性の3本柱)
- pull 方式は短命なバッチジョブ(終わってしまう前にスクレイプできない)が苦手で、Pushgateway という補助が要る

## 発展トピック・研究の最前線

- 可観測性の3本柱(メトリクス・ログ・トレース)の統合: OpenTelemetry が計装(instrumentation)の共通規格として広がり、Prometheus メトリクスと分散トレースを相関させる動きが進んでいます。Grafana 側も Loki(ログ)・Tempo(トレース)を統合したダッシュボードを提供します。
- 高カーディナリティと長期保存: Mimir / Thanos / VictoriaMetrics など、Prometheus 互換でスケールアウトと長期保存を実現する系の競争が活発です。
- 機械学習ワークロード特化の監視: GPU 利用率・データストール・キュー待ち・分散学習の通信量を相関させ、ボトルネックを自動診断する MLOps 監視。GPU メトリクス源は [DCGM Exporter](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1163_dcgm-gpu-monitoring)、コスト換算は [Kubecost (コスト可視化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1165_kubecost) が担い、Prometheus はその共通の土台になります。
- アラートの賢さ: 単純なしきい値から、季節性を考慮した異常検知や、SLO ベースのエラーバジェット(許容できる失敗量を予算として管理する)監視への発展。

## さらに学ぶための関連トピック
- [DCGM Exporter (GPU 監視)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1163_dcgm-gpu-monitoring)
- [Kubecost (コスト可視化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1165_kubecost)
- [Warm GPU Node (pause Pod)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1157_warm-gpu-node)
- [Bottlerocket OS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1100_bottlerocket-os)
- [NVIDIA Device Plugin](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1180_nvidia-device-plugin)
