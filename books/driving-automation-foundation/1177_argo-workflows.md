---
title: "Argo Workflows"
---

# Argo Workflows

## ひとことで言うと
Argo Workflows は「複数の処理を、順番や依存関係を決めて自動で次々に実行する」仕組みを、Kubernetes（コンテナを大量に管理するソフトウェア）の上で実現するワークフローエンジンです。たとえば「データを集める → 前処理する → モデルを学習する → 評価する」という機械学習の一連の流れを、人間が手作業で 1 つずつ実行する代わりに、定義どおり自動で走らせてくれます。各処理は独立したコンテナとして起動するため、何百もの処理を並列に流したり、失敗したところから再試行したりが得意です。

## 直感的な理解
料理に例えます。カレーを作るには「野菜を切る」「肉を炒める」「煮込む」「盛り付ける」という工程があり、順番（切ってから炒める）や、同時にできること（野菜を切る間に米を炊く）があります。レシピを 1 枚紙に書いておけば、誰でも同じ手順を再現でき、途中で焦がしても「どの工程からやり直すか」が分かります。

機械学習でも事情は同じです。モデルを作る作業は、1 つの大きな処理ではなく、たくさんの小さな処理の集まりです。これを毎回ターミナルで「python preprocess.py を実行 → 終わったら python train.py を実行 …」と手で打つと、途中で失敗したらどこから再開すべきか分からない、誰がどの順でやったか記録が残らない、GPU を大量に使う処理を複数同時に走らせたいのに管理しきれない、といった問題が起きます。Argo Workflows は、この「レシピ（パイプライン）」を一度書けば自動で正しい順序・並列度で実行してくれる存在です。

## 基礎: 前提となる概念
いくつか用語を噛み砕きます。

「パイプライン（pipeline）」とは、データが管（パイプ）を流れるように、複数の処理を順に通っていく一連の流れのことです。機械学習では典型的に次の段階があります。

```
1. 生データ取得   : カメラ画像やセンサーログをオブジェクトストレージから取り出す
2. 前処理         : 学習に使える形に整える（クレンジング、特徴抽出、シャード化）
3. 学習           : モデルを GPU で学習する
4. 評価           : 学習済みモデルの性能を測る
5. 登録           : 良ければ本番用に登録する（モデルレジストリ）
```

「ワークフローエンジン」または「パイプラインオーケストレーター」は、この流れをあらかじめ定義しておき自動実行するソフトウェアです。

「Kubernetes（クバネティス、略して K8s）」は、コンテナ（アプリを必要な部品ごと箱詰めしたもの）を何百個・何千個も自動で配置・起動・再起動・スケールしてくれる管理基盤です。Argo Workflows は「Kubernetes ネイティブ」、つまり Kubernetes の標準的な仕組み（Custom Resource Definition = CRD という拡張機能や Pod という実行単位）にぴったり乗っかって動くように作られています。具体的には Argo は `Workflow` という新しいリソース種別を CRD で K8s に追加し、Workflow Controller がそれを監視して Pod を作ります。これに対し、Apache Airflow のように Kubernetes に依存せず独自スケジューラ（と独自 DB）で動くオーケストレーターもあり、ここが設計思想の分かれ目です。

「宣言的（declarative）」という考え方も重要です。Kubernetes では「あるべき状態」を YAML（設定を書くテキスト形式）で宣言すると、Controller（管理役のプログラム）が現実をその状態に合わせ続けます（reconciliation loop と呼ぶ調整ループ）。「手順を命令する（命令的）」のではなく「結果を宣言する」スタイルで、Argo もこの流儀を踏襲します。失敗しても Controller が宣言された状態へ戻そうとするため、堅牢性が高いのが利点です。

## 仕組みを詳しく
Argo Workflows の中心概念は「DAG（Directed Acyclic Graph、有向非巡回グラフ）」です。名前は難しそうですが、意味は「矢印付きで、ぐるっと一周して戻らない、処理のつながり図」です。

```
        +-- 前処理A --+
データ取得              合流 --> 学習 --> 評価
        +-- 前処理B --+
```

- 有向（矢印に向きがある）: 「データ取得が終わってから前処理が始まる」順序が決まっている
- 非巡回（一周して戻らない）: 「評価 → データ取得」のように無限ループしない（ループは withItems の展開で表現し、循環は作らない）
- 前処理A と前処理B は互いに依存しないので同時に並列実行できる

Argo ではこの図を YAML で記述します。記述スタイルは大きく 2 つで、依存関係を明示する `dag` と、上から順に流す `steps` があります。

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: ml-pipeline-
spec:
  entrypoint: main
  templates:
    - name: main
      dag:
        tasks:
          - name: fetch-data
            template: fetch
          - name: preprocess
            template: prep
            dependencies: [fetch-data]   # fetch-data が終わってから動く
          - name: train
            template: train-gpu
            dependencies: [preprocess]
```

各ステップは Kubernetes の Pod（コンテナを動かす最小単位の箱）として起動します。実際には Argo は各 Pod に main コンテナ（実処理）と sidecar/init の wait コンテナ（入出力アーティファクトの取得・送出やログ収集を担う）を同居させる構成をとります。「学習ステップ」は GPU を要求する学習用コンテナの Pod になり、`resources: {limits: {nvidia.com/gpu: 1}}` のように GPU 要求を自然に書けます。ステップ間のデータ受け渡しは、小さな値なら `outputs.parameters`（文字列）、大きなファイルなら `outputs.artifacts` としてオブジェクトストレージ（Argo では「アーティファクトリポジトリ」と呼び、S3/GCS/MinIO などを設定）経由で行うのが基本です。

実行の流れは次のとおりです。

```
1. ユーザーが Workflow YAML を投入  (argo submit / kubectl create -f workflow.yaml)
2. Argo Workflow Controller がそれを検知（CRD の watch）
3. Controller が DAG を解釈し、依存が解決されたステップから Pod を作る
4. 各 Pod が処理を実行し、結果ファイルをアーティファクトリポジトリにアップロード
5. 次ステップの Pod が起動し、前段の結果をダウンロードして使う
6. 全完了で Succeeded、失敗で Failed。retryStrategy があれば自動再試行
```

GPU を 8 枚使う学習を 3 種類の設定で同時に流したい、といった「並列で大量に流す」用途にも、各ステップが独立 Pod になるので素直に対応できます。さらに Argo は次の制御構造を持ち、複雑なパイプラインを表現できます。

- loops: `withItems` / `withParam` でパラメータ配列を展開し、同じテンプレートを多数並列実行（ハイパーパラメータ sweep やデータシャードの並列処理に有用）。
- conditionals: `when` で分岐（例: 評価スコアが閾値を超えたときだけ登録ステップへ進む）。
- retries: `retryStrategy` で指数バックオフ付き自動再試行。
- suspend / resume: 人間の承認を待つゲート（手動承認後に本番デプロイへ進む）。
- cron: `CronWorkflow` で定期実行（毎晩再学習など）。
- exit handlers: 成功/失敗にかかわらず後始末（通知、クリーンアップ）を実行。
- memoization / artifact cache: 同一入力ならステップをスキップし結果を再利用。

## 手法の系譜と主要論文
Argo Workflows は学術論文というより、CNCF（Cloud Native Computing Foundation、クラウドネイティブ技術を取りまとめる中立組織）の Graduated（卒業）プロジェクトとして実務で広く使われるツールです。出自は Applatix 社が開発し 2017 年頃にオープンソース化、その後 Intuit、BlackRock、Red Hat などが主要メンテナーとなって発展。2020 年に CNCF Incubating、2024 年に Graduated（最高成熟度）へ昇格しました。Graduated は CNCF の中で「本番運用に十分耐え、ガバナンスと採用実績が確立した」ことを意味する最上位ステータスです。

周辺の系譜を押さえます。

- Apache Airflow（Airbnb 発、2014 年頃、後に Apache 財団）: DAG でワークフローを定義する考え方を広めた先駆。Python でパイプラインを書き、独自の Scheduler と Executor、メタデータ DB で動く。Kubernetes ネイティブではなく（KubernetesExecutor はあるが中核は独自）、データエンジニアリングの ETL 定番。Argo が「K8s が一級市民」なのと対照的。
- Kubeflow Pipelines（KFP, Google 発、2018 年頃）: ML パイプラインを Kubernetes 上で動かす仕組みで、実行エンジン（バックエンド）として長らく Argo Workflows を採用してきた。「Kubeflow Pipelines で書いた ML パイプラインが、裏で Argo の Workflow に変換されて動く」関係。Python SDK で書いた DAG をコンパイルして Argo の YAML を生成する。動機は ML の各ステップをコンテナ化し再現可能・スケール可能にすること。
- Argo Events: 姉妹プロジェクト。「ストレージにファイルが置かれたら」「メッセージキューにメッセージが届いたら」というイベントをトリガーにワークフローを起動。新データ到着で自動学習、というイベント駆動 MLOps を可能にする。
- Argo CD: GitOps（Git の状態を正としてデプロイを自動同期）デプロイツール。Workflows とは別物だが Argo ファミリーとして統合が効く。
- Tekton（CD Foundation）: 同じく K8s ネイティブのパイプラインエンジンだが CI/CD（ソフトのビルド・デプロイ自動化）寄り。Argo が ML・データバッチに強いのと棲み分ける。

ML 向けの抽象化を厚くした対抗として、Python で型付きパイプラインを書け、キャッシュ・データ系統（リネージ）・実験管理連携を最初から備えるオーケストレーター（Flyte, Prefect, Dagster, Metaflow など）の系譜があります。Flyte は内部で K8s 上に実行を展開しつつ Python で強い型を付けキャッシュとリネージを標準装備する点が特徴で、Argo の「Kubernetes そのものに近い低レイヤーの自由度」とは対照的に「ML/データサイエンス向けの高レベル抽象」を持ち味とします。実務では「Argo を薄い実行基盤として使い、その上に型・キャッシュ・リネージの層を載せる（KFP や Flyte 的な発想）」か「Argo の YAML を直接書いて低レイヤーの自由度を取る」かの設計判断になります。

## 論文の実験結果（定量データ）
Argo は研究論文ではなく OSS のため、論文の定量評価という形のデータはありません。代わりに実務上の性能特性と、関連する設計上の数値感覚を挙げます。

- スケーラビリティ: Argo Controller は数千〜数万の並行 Workflow / 数十万 Pod 規模を扱えるよう設計されており、大規模利用ではこの Pod 数が Kubernetes API サーバや etcd（K8s の状態を保存する分散 KV ストア）の負荷上限と直結する。ステップごとに Pod を作る性質上、極端に細かいステップを大量生成すると API サーバが詰まるため、ステップ粒度の設計が性能に効く。実運用では Controller の `--workflow-workers` や parallelism（同時実行 Workflow 数の上限）でスループットと API 負荷を調整する。
- ワークフロー保存方式: 大規模化では各 Workflow の状態（ノードのステータス）を CRD オブジェクトとして etcd に丸ごと持つと、etcd の 1 オブジェクト上限（既定 1.5MB 前後）に当たることがある。これに対し node status をオフロード（外部 DB やアーカイブに退避）する設定や、status の圧縮機能が用意されている。これは「DAG のノード数が増えると状態オブジェクトが線形に膨らむ」という定量的制約への対処。
- 並列度: 各ステップが独立 Pod のため、クラスタのノード/GPU 容量が許す限り水平にスケールする。GPU ジョブをキューイングする仕組み（Kueue など）と組み合わせると、容量超過分を待たせて順次流せる。Pod 起動には数秒〜数十秒のオーバーヘッド（イメージ pull、スケジューリング）があり、これが細粒度ステップで効いてくる。

定量比較として ML オーケストレーター選定でよく問われるのは「ローカル実行のしやすさ」「キャッシュによる再実行スキップ率（同一入力ステップを何割スキップできるか）」「型安全性によるバグ削減」で、これらは Argo（YAML 中心・低レイヤー自由度高）と ML 特化型（Python・高抽象）のどちらを選ぶかのトレードオフとして語られます。

## メリット・トレードオフ・限界
メリット
- Kubernetes ネイティブで、Pod・リソース要求・スケジューリング・ノードセレクタ・トレランスなど K8s の仕組みをそのまま活かせる（GPU 要求やノード選択を自然に記述）。
- DAG による並列実行が得意で、独立ステップを同時に大量に流せる。
- CNCF Graduated でコミュニティ・実績が大きく情報が豊富。本番運用の信頼性が高い。
- Argo Events / Argo CD など周辺エコシステムが充実し、イベント駆動や GitOps と統合しやすい。
- 言語非依存: 各ステップはただのコンテナなので、Python でも C++ でも何でも動かせる。

トレードオフ・限界
- Kubernetes の知識が前提で、前提知識ゼロには学習コストが高い（Pod、PVC、RBAC、CRD の理解が要る）。
- パイプラインを YAML で書くのが基本のため、複雑なロジックや型付き入出力は書きにくく、ML 向けの抽象化（型安全な入出力、自動キャッシュ、リネージ）は弱め。自前で組むか上位レイヤー（KFP/Flyte）に頼る必要がある。
- ステップ間のデータ受け渡しを S3 などのアーティファクト経由で自前設計する必要がある（大きなデータの往復コストに注意）。
- ステップごとに Pod を起動するため、極めて短い処理を大量に並べると Pod 起動オーバーヘッドと API サーバ負荷が無視できなくなる。
- ローカルでの開発・デバッグが、Python ベースで高抽象なオーケストレーターほど手軽ではない（クラスタに submit して見る前提になりがち）。

研究・実務上の未解決課題としては、ML 特化の型・キャッシュ・リネージをどこまで上位レイヤー（Kubeflow Pipelines や Flyte など）に委ね、Argo を純粋な実行基盤として薄く使うかの設計判断や、極大規模での API サーバ・etcd 負荷の抑制、長時間ワークフローの耐障害性（Controller 再起動時の状態復元）が挙げられます。

## 発展トピック・研究の最前線
- Workflow のキャッシュ（同一入力なら結果を再利用しステップをスキップ）と memoization の実用化による計算コスト削減。
- Argo Events による完全なイベント駆動 MLOps（データ到着 → 自動再学習 → 自動評価 → 自動デプロイ）の構築パターン。
- Kueue など GPU ジョブキューと組み合わせた、容量制約下での公平スケジューリングと優先度制御（research/production の 2 段階優先度など）。
- ML 特化型オーケストレーター（Flyte, Metaflow, Dagster）との比較・使い分けの設計論。Argo を実行基盤に、上に型・リネージ層を重ねるハイブリッド。
- データリネージ・再現性の標準化（OpenLineage など）との連携。

## さらに学ぶための関連トピック
- [RDS Postgres (Metadata Backend)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1148_rds-postgres-backend)
- [Weights & Biases](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1183_wandb-experiment-tracking)
- [DVC (Data Version Control)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0884_dvc-data-version-control)
- [VPC / Private Subnet + NAT](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1156_vpc-private-subnet-nat)
- [Model Registry (Stage 遷移)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1168_model-registry-staging)
