---
title: "自動運転モデル開発で Flyte・MLflow・Kueue を選んだ理由と設計の全貌"
emoji: "🚗"
type: "tech"
topics: ["MLOps", "Flyte", "MLflow", "自動運転", "Kubernetes"]
published: false
---

## はじめに

普段は AWS の Solutions Architect として自動運転・モビリティ領域のお客様を支援しています。
その傍ら、[Autoware Foundation](https://autoware.org/) の Robotaxi Working Group で Maintainer を務めており、プロジェクトの技術方針の決定や、世界中のコントリビューターからの PR レビューを担当しています。

Robotaxi WG は L2++ および L4 Robotaxi アプリケーションを構築し、2027年5月にクローズドループ自動運転デモの達成を目標としています。その中核が、カメラのみで高速道路から市街地まで走行可能な End-to-End 自動運転モデル [AutoE2E](https://github.com/autowarefoundation/auto_e2e) です。

![AutoE2E Architecture](/images/auto_e2e_architecture.jpg)

7台のサラウンドカメラとマップタイルを入力に、6.4秒先の走行軌跡（加速度と曲率、10Hz）を予測します。
私はこのモデルのアーキテクチャ設計から学習パイプラインの構築、EKS 上の MLOps 基盤の実装までをリードしています。
モデル側の詳細は「[PyTorchで理解する自動運転マルチカメラ統合の全体像](https://zenn.dev/riita10069/articles/auto-e2e-multi-view-fusion-explained)」で解説しました。
今回はその学習を支えるプラットフォーム側の話です。

MLOps 基盤として Flyte、MLflow、Kueue を利用しています。
この記事では、なぜこの3つを選んだのか、自動運転モデル開発のどういった制約がその判断を後押ししたのかを、実装コードを交えて書いていきます。

:::message
本記事のコードはすべて [autowarefoundation/auto_e2e](https://github.com/autowarefoundation/auto_e2e) の `platform/` ディレクトリに公開されています。
:::

---

## 自動運転の ML パイプラインが抱える制約

自動運転モデルの学習パイプラインには、一般的な ML と質的に異なる制約が3つあります。

1つめはデータの規模です。
一般的な画像認識タスクは数 GB から数十 GB のデータセットで成り立ちますが、自動運転では1台のテスト車両が1日に数十 TB の生センサーデータを生成します。
BMW は AWS と協業して[ペタバイト規模のストレージコスト最適化](https://aws.amazon.com/blogs/industries/aws-professional-services-and-bmw-collaborate-to-optimize-petabyte-scale-storage-costs-for-autonomous-driving-data/)に取り組んでいますが、これが業界の標準的なスケールです。
カメラが7台以上、LiDAR、Radar、GPS/IMU がそれぞれ異なるデータレートとフォーマットで動いており、時空間の同期だけで一つのプロジェクトになります。

2つめは**循環する学習ループ**です。
一般的な ML は「データ収集 → 学習 → デプロイ」で完了しますが、自動運転は循環します。
車両が走行してデータを集め、そのデータでモデルを学習し、新しいモデルを車に載せてまた走る。
うまくいかなかったケースは次のデータ収集のターゲットとして戻ってくる。
しかもルーティン走行（まっすぐの高速道路）が大半を占める中から、稀少だが安全上意味のあるコーナーケースを自動抽出し、データセットをリバランスし続ける必要があります。

3つめは**規制**です。
自動運転 AI は EU AI Act（2024/1689）で「高リスク AI」に分類されており、改ざん防止ログの自動生成、学習データの品質文書化、完全な追跡可能性が法的に義務づけられています。
ISO 21448（SOTIF）は ML の性能限界に起因する危険挙動を体系的に特定し、未知の危険シナリオ空間を継続的に縮小することを要求しています。
一般的な ML のように A/B テストして良ければデプロイ、という流れは成り立ちません。
モデル変更のたびに型式認証の再評価が発生しえるため、パイプライン全体が「このモデルがなぜ安全か」の法的証拠を生成する仕組みでなければなりません。

この「ペタバイト規模のデータ」「循環する学習ループ」「法的に要求されるトレーサビリティ」が、ツール選定の判断を規定しています。

---

## Kueue：GPU のアドミッション制御

GPU は自動運転 ML 基盤で最も高価なリソースです。
AutoE2E で使っている g6e.4xlarge（NVIDIA L40S 48GB）は月額約 $1,300 です。
素の Kubernetes にジョブ管理を任せると、先着順で Pod が作られて Pending のまま放置されます。

[Kueue](https://kueue.sigs.k8s.io/) は Kubernetes SIG-Scheduling が開発したジョブレベルのアドミッション制御システムです。
GKE、AWS SageMaker HyperPod、Red Hat OpenShift が採用しています。
動画生成 AI の Runway は Kueue 導入で [GPU 利用率を 20 ポイント以上改善した](https://runwayml.com/news/no-idle-gpus-managing-research-compute-at-runway)と報告しています。

### 素の Kubernetes の問題

素の Kubernetes ではジョブ投入と同時に Pod が作られ、GPU が足りなければ Pending で待ちます。
4 GPU を必要とする分散学習ジョブが 3 GPU しか確保できないと、3つの Pod が GPU を占有したまま4つめを待ち続けます。
その間、他のジョブは 1 GPU も使えない。
これが部分起動によるデッドロックです。

Kueue はジョブ作成時に `spec.suspend: true` としておき、必要なリソースが全て確保可能であることを確認してから `suspend: false` に変更します。
4 GPU のジョブは4 GPU 全部が空いてから起動する。
部分起動は構造的に発生しません。

### AutoE2E での運用

AutoE2E ではジョブの優先度を2段階に分けています。
本番モデルの学習は `production-high`（10000）、実験的な sweep や ablation study は `research-low`（1000）です。
本番学習がキューに入ると、research ジョブは自動的にプリエンプト（中断して再キューイング）されます。

実際の流れはこうです。
Flyte の学習タスクが実行されると、Flyte Propeller が `auto-e2e-training` namespace に PyTorchJob を作ります（`suspend: true`）。
Kueue がこのジョブを検知し、ClusterQueue のクォータと照合して空きがあれば `suspend: false` に変更。
すると Pod が Pending になり、Karpenter が g6e.4xlarge の GPU ノードを起動します。
学習が終わると Pod が消え、Karpenter がノードを自動終了します。
GPU のアイドルコストはゼロです。

```python
@task(
    container_image=TRAINING_IMAGE,
    requests=Resources(cpu="4", mem="16Gi", gpu="1"),
    limits=Resources(gpu="1"),
)
def train_il(shards: List[FlyteDirectory], ...) -> TrainOutput:
    ...
```

ワークフローを書く開発者は `Resources(gpu="1")` と宣言するだけで、Kueue の存在を意識する必要がありません。
アドミッション制御はインフラ層で完結します。

同じ領域のツールとして Volcano がありますが、Volcano はカスタムスケジューラバイナリをクラスタに追加します。
Kueue は CRD を apply するだけで動き、既存のスケジューラに手を入れません。
AutoE2E は現時点でシングルノード学習なので Volcano の強みである厳密な Gang scheduling は不要でした。

---

## Flyte：自動運転パイプラインのオーケストレーター

Flyte は Lyft が社内で開発したワークフローエンジンです。
Airflow では ML の計算要件（大容量データの受け渡し、タスクごとの GPU 割り当て、キャッシュ）に対応しきれなかったことが開発の動機でした。
現在は CNCF プロジェクトとして累計8,000万回以上ダウンロードされています。

自動運転業界では Woven by Toyota（トヨタグループ）がペタバイト規模のセンサーデータ処理に採用し、[年間 $1M 以上のコスト削減と ML イテレーション速度20倍の改善](https://web.union.ai/case-study/how-woven-by-toyota-saved-millions-with-scaled-autonomous-driving-from-union-ai)を達成しています。
End-to-End 自動運転の Wayve は[12以上のオーケストレーションツールを比較検証した末に Flyte を選定](https://web.union.ai/case-study/wayve-accelerates-autonomous-driving-innovation-with-flytes-scalable-orchestration)し、10,000以上の並列タスクを実行しています。

### FlyteFile と FlyteDirectory による大容量データの型付きリレー

自動運転パイプラインでは、ステージ間で数十 GB のデータが行き交います。
データ前処理の出力（WebDataset の tar shards）を学習タスクに渡し、学習タスクの出力（チェックポイント）を評価タスクに渡す。

Airflow でこれを実現するには、XCom が数 KB のメタデータしか渡せないため、自前で S3 パスの管理コードを書くことになります。
パス文字列のタイポでパイプライン全体が失敗する。
しかもそのエラーは高価な GPU 学習が数時間走った後に発覚します。

Flyte では `FlyteFile` と `FlyteDirectory` という型が S3 上のデータへの参照を表現します。

```python
TrainOutput = NamedTuple("TrainOutput", checkpoint=FlyteFile, metadata=FlyteFile)

@task(container_image=TRAINING_IMAGE, requests=Resources(cpu="4", mem="16Gi", gpu="1"))
def train_il(shards: List[FlyteDirectory], ...) -> TrainOutput:
    shard_dir = shards[0].download()  # S3 → ローカルへ自動転送
    ...
    return TrainOutput(checkpoint=FlyteFile(ckpt_path), metadata=FlyteFile(meta_path))

@task(container_image=EVAL_IMAGE, requests=Resources(cpu="2", mem="8Gi", gpu="1"))
def evaluate_il_policy(checkpoint: FlyteFile, ...) -> EvalMetrics:
    ckpt_path = checkpoint.download()  # 前タスクの出力を型安全に取得
    ...
```

`FlyteDirectory` を返せば S3 へのアップロードは自動で行われ、受け取り側で `.download()` を呼べばローカルにダウンロードされます。
型が合わなければ `pyflyte register`（ワークフロー登録）の時点でエラーになります。
GPU 学習を走らせる前に不整合を検出できるので、発見コストが大幅に下がります。

### NamedTuple：複数の型付き出力

学習タスクは「チェックポイント」と「メタデータ」の2つを返す必要があります。
Flyte は Python の `NamedTuple` をネイティブにサポートしており、後続タスクからは名前付きでアクセスできます。

```python
@workflow
def wf_train_il(shards: List[FlyteDirectory], ...):
    out = train_il(shards=shards, ...)
    return evaluate_il_policy(
        checkpoint=out.checkpoint,
        train_metadata=out.metadata,
        shards=shards,
    )
```

`out.checkpoit`（タイポ）と書いた場合、`pyflyte register` の時点でエラーになります。
GPU 時間を消費した後ではなく、パイプライン起動前にエラーを検出できます。

### Enum によるパラメータ選択

backbone、fusion_mode、dataset といった設定を文字列で受け取ると、タイポで学習が無駄になります。
Flyte は Python の Enum をサポートしており、Flyte Console 上ではドロップダウンとして表示されます。

```python
class Backbone(enum.Enum):
    SWIN_V2_TINY = "swin_v2_tiny"
    CONVNEXT_V2_TINY = "conv_next_v2_tiny"
    RESNET_50 = "res_net_50"

class FusionMode(enum.Enum):
    CONCAT = "concat"
    CROSS_ATTN = "cross_attn"
    BEV = "bev"

@workflow
def wf_full_pipeline(
    dataset: Dataset = Dataset.L2D,
    backbone: Backbone = Backbone.SWIN_V2_TINY,
    fusion_mode: FusionMode = FusionMode.CONCAT,
    epochs_il: int = 3,
    lr: float = 1e-4,
    ...
):
```

存在しない値を指定することが不可能になります。
GPU 学習1回に数時間かかる環境では、起動時のバリデーションが防げるコストは大きい。
Airflow にも KFP v2 にも、この Enum の first-class サポートはありません。

### タスクごとに異なるコンテナとリソース

自動運転パイプラインの各ステージは、必要なライブラリもリソースも異なります。
データ取り込みは HuggingFace SDK と ffmpeg が必要で CPU と大メモリを使い、学習は PyTorch + timm で GPU が必須、評価は PyTorch + MLflow クライアントです。
Flyte ではタスクごとにコンテナイメージとリソースをデコレータで宣言します。

```python
@task(container_image=DATA_PREP_IMAGE,
      requests=Resources(cpu="2", mem="24Gi", ephemeral_storage="50Gi"))
def data_ingest(...) -> FlyteDirectory:

@task(container_image=TRAINING_IMAGE,
      requests=Resources(cpu="4", mem="16Gi", gpu="1"))
def train_il(...) -> TrainOutput:
```

1つのパイプライン内で「データ準備は CPU + 大メモリ」「学習は GPU」「評価は別イメージで GPU」が自然に分離されます。
Airflow で同じことをやるには KubernetesPodOperator で各タスクに Pod spec を個別に書かなければなりません。

### 依存関係からの DAG 自動構築

AutoE2E のフルパイプラインでは、複数の異なるデータセット（L2D と NVIDIA Physical AI）を独立に取り込んで前処理し、結果を合流させて学習に渡します。

```python
@workflow
def wf_full_pipeline(dataset: Dataset, episodes: int, ...):
    raw_l2d = data_ingest(dataset=Dataset.L2D, episodes=episodes)
    shards_l2d = data_processing(raw_data=raw_l2d, dataset=Dataset.L2D)

    raw_nv = data_ingest(dataset=Dataset.NVIDIA_PHYSICAL_AI, episodes=episodes)
    shards_nv = data_processing(raw_data=raw_nv, dataset=Dataset.NVIDIA_PHYSICAL_AI)

    all_shards = [shards_l2d, shards_nv]
    il_out = train_il(shards=all_shards, dataset=dataset, ...)
    ...
```

Flyte はこのコードの入出力依存を解析して DAG を構築します。
L2D と NVIDIA のデータ取り込みは互いに依存がないため、自動的に並列実行されます。
「並列にしろ」と明示する必要はありません。

### キャッシュによる再計算回避

「backbone を変えて再学習したいが、データ前処理は同じ」というケースは頻繁に起きます。
前処理に1時間かかるなら、毎回やり直すのは無駄です。

```python
@task(cache=True, cache_version="1.0",
      container_image=DATA_PREP_IMAGE, requests=Resources(cpu="4", mem="16Gi"))
def data_processing(raw_data: FlyteDirectory, ...) -> FlyteDirectory:
    ...
```

`cache=True` を付けるだけで、同じ入力と同じコードなら前回の結果を即座に返します。
キャッシュキーは「タスク名 + 入力のハッシュ + ソースコードのハッシュ」から自動生成されるため、コードを変更すれば自動的に無効化されます。

`cache_serialize=True` を追加すると、同じ入力の GPU タスクが複数同時にトリガーされた場合に1つだけ実行し、残りはキャッシュ結果を待ちます。
5人のチームメンバーが同じデータセットに対して異なるハイパーパラメータで学習を回した場合、データ前処理は1回だけ実行されて結果が共有されます。

### Secret の注入

HuggingFace のアクセストークンなど秘匿情報をワークフローのパラメータとして渡すと、Flyte Console の UI にも実行履歴にも平文で残ります。
Flyte は Kubernetes Secret をタスクに直接注入する仕組みを持っています。

```python
@task(
    container_image=DATA_PREP_IMAGE,
    requests=Resources(cpu="2", mem="24Gi", ephemeral_storage="50Gi"),
    secret_requests=[Secret(group="hf-token", key="HF_TOKEN",
                            mount_requirement=Secret.MountType.ENV_VAR)],
)
def data_ingest(dataset: Dataset, episodes: int) -> FlyteDirectory:
    token = current_context().secrets.get("hf-token", "HF_TOKEN")
    login(token=token)
    ...
```

`secret_requests` で宣言されたシークレットは Kubernetes Secret から環境変数としてタスク Pod に注入されます。
これはワークフローの入力パラメータではないため、Flyte Console の「Launch Workflow」フォームにもトークンの入力欄は現れず、実行履歴にも記録されません。

### 失敗からの部分復旧

自動運転パイプラインは全体で数時間かかることがあります。
データ取り込み30分、前処理1時間、学習3時間、評価30分。
評価タスクでバグが見つかった場合に全部やり直すのは許容できません。

Flyte の recovery モードは、失敗したノードだけを再実行し、成功済みのノードはスキップします。
データ取り込み、前処理、学習の結果をそのまま使い、評価タスクだけやり直せます。

Kubeflow Pipelines にはこの機能がありません。
KFP で途中失敗したパイプラインは最初からやり直す必要があります。
GPU 学習のコストを考えると、この差は月単位で数千ドルの節約になりえます。

### ローカル実行とリモート実行の透過性

Flyte のタスクとワークフローは `pyflyte run` でローカルに実行できます。

```bash
pyflyte run workflows.py wf_train_il \
    --shards '["/tmp/my_shards"]' \
    --dataset L2D \
    --backbone SWIN_V2_TINY
```

ローカル実行時は `FlyteFile` がローカルファイルパスを直接参照し、リモート実行時は S3 を経由します。
同じコードが環境差を吸収するため、手元で小さいデータで動作確認してから `--remote` を付けてクラスタに送る、という開発フローが成り立ちます。

Robotaxi WG では私のようなコアメンバーだけでなく、多くの大学や企業から研究者が貢献してくださっています。
当然、全員が AutoE2E の AWS 環境にアクセスできるわけではありません。
しかし Flyte のローカル実行であれば、GPU を1枚持っているマシンさえあればパイプライン全体を手元で動かせます。
S3 も EKS も Kueue も不要で、`pyflyte run` だけで学習から評価まで通ります。
コントリビューターは自分の環境でモデルの改善を試し、PR として提出し、コアメンバーがリモートのクラスタで本番データを使って再実行する、というワークフローが自然に成立します。

Airflow の DAG はローカルで動かすことができません。
テストするには Airflow サーバーを立てるか、各 Operator のモックを書く必要があります。
OSS プロジェクトにおいて「貢献者がインフラを持っていなくてもパイプラインを動かせる」ことは、Flyte を選んだ理由の中でも実際に最も効いている利点です。

### Kubernetes ネイティブであることの帰結

Flyte は「Kubernetes の上に薄い層を載せる」設計です。
タスクは Pod、ワークフローの状態管理は CRD、データは S3。
Kueue、Karpenter、Training Operator、Pod Identity といった Kubernetes エコシステムのツールがそのまま使えます。

AutoE2E では、Flyte が PyTorchJob を作り、Kueue がアドミッション制御し、Karpenter がノードをプロビジョニングします。
3つのツールは互いの存在を知らず、Kubernetes API という共通インターフェースで連携しています。
もし Flyte が独自の計算エンジンを内蔵していたら（KFP の Argo Workflows のように）、こうした外部ツールとの統合にグルーコードが必要だったでしょう。

### 不変な実行記録

EU AI Act は改ざん防止ログの自動生成を義務づけています。
Flyte は全ての実行を不変レコードとして記録します。
入力パラメータ、各タスクの中間出力（S3 に永続化）、コンテナイメージバージョン、コードバージョン、タイムスタンプが自動で残ります。

「このモデルはどのデータ、どのコード、どのハイパーパラメータ、どのインフラで作られたか」という規制当局の問いに対して、事後調査なしに即答できます。
自動運転開発ではこれは「あると便利」ではなく法的必須要件です。

### Flyte をどう使っているか


AutoE2E の開発では、基本的に Flyte Console（Web UI）からパイプラインを起動します。
左メニューの Workflows から `wf_full_pipeline` を選び、「Launch Workflow」を押すとフォームが出ます。
backbone と fusion_mode もドロップダウンで選び、epochs や lr は数値を入力して Launch。
これだけでデータ取り込みから学習、評価、Model Registry 登録までが一気通貫で走ります。

起動後は DAG ビューでノードが Pending → Running → Succeeded と色が変わっていくのをリアルタイムで確認できます。
GPU ノードの起動に1〜3分かかる（Karpenter がプロビジョニングする時間）ので、`train_il` ノードが Pending のまましばらく待つのは正常です。

あるノードが赤く失敗したら、そのノードをクリックするとエラーメッセージと Kubernetes Pod のログが表示されます。
OOMKilled なら Resources のメモリを上げる、ImagePullBackOff なら ECR のイメージ名を確認する、といった対処が即座にわかります。
修正後は同じ実行ページの「Relaunch」ボタンで同じ入力パラメータのまま再実行できます。
あるいは recovery モードを使えば、失敗ノード以降だけを再実行し、成功済みのノード（数時間かかったデータ前処理や学習）はスキップできます。

コントリビューターが新しい backbone を追加した場合のワークフローは以下のようになります。

1. `Backbone` Enum に新しい値を追加し、`build_backbone` に対応コードを書く
2. ローカルで `pyflyte run workflows.py wf_train_il --backbone NEW_BACKBONE ...` を実行し、小さいデータで動作確認
3. PR を出す。CI で `pyflyte register --dry-run` が走り、型エラーがないことを確認
4. マージ後、CodeBuild が Docker イメージをビルドし、`pyflyte register` でワークフローの新バージョンが Flyte に登録される
5. Flyte Console で最新バージョンのワークフローを Launch。新しい backbone がドロップダウンに出現している

このとき、既存の backbone（swin_v2_tiny など）の過去の実行は全て保存されています。
新しい backbone との比較は、MLflow で同じ条件の Run をフィルタするだけです。
ワークフローのバージョン管理と実験管理が自然に連動するため、「どのコードバージョンでどの backbone を試したか」が曖昧になることがありません。

実際の開発では、パイプライン全体（`wf_full_pipeline`）を毎回通しで起動することはほとんどありません。
日常的には部品ごとに個別のワークフローを起動し、出力 URI を次のワークフローに渡す形で運用しています。

たとえばデータセットの Hz 数や解像度を変えたい場合を考えます。
`data_processing` タスクのコードを修正して `hz=20` や `image_size=512` に変更し、`pyflyte register` でワークフローを再登録します。
Flyte Console から `wf_data_processing` を起動して、新しいパラメータで前処理を走らせます。

```
wf_data_processing:
  raw_data: s3://auto-e2e-platform-artifacts/.../raw/l2d/
  dataset: L2D
  hz: 20           ← 10Hz から 20Hz に変更
  image_size: 512  ← 256 から 512 に変更
  episodes: 10
```

完了すると出力の `FlyteDirectory` URI が Flyte Console の Outputs パネルに表示されます。
`s3://auto-e2e-platform-artifacts/.../shards/l2d-20hz-512px/` のようなパスです。

次に `wf_train_il` を起動し、`shards` 入力にこの URI を貼り付けるだけで、新しいデータセットを使った学習が走ります。

```
wf_train_il:
  shards: ["s3://auto-e2e-platform-artifacts/.../shards/l2d-20hz-512px/"]
  dataset: L2D
  backbone: SWIN_V2_TINY
  fusion_mode: BEV
  epochs: 10
```

データセットの構成を変えたいときはデータ前処理だけ再実行し、モデルのハイパーパラメータを変えたいときは学習だけ再実行する。
前処理の出力は S3 上にアドレス可能な形で残り続けるため、一度作ったデータセットは何度でも再利用できます。
「20Hz / 512px のデータセット」と「10Hz / 256px のデータセット」を両方手元に持っておき、同じ backbone で学習して精度を比較する、ということが URI を差し替えるだけで実現します。

この「パーツを独立に回して URI で繋ぐ」運用は、フルパイプラインを毎回通しで実行するよりも柔軟です。
データ前処理に1時間かかるとして、backbone の ablation を5パターン試したい場合、フルパイプラインなら前処理が5回走ります。
部分実行なら前処理は1回で済み、その出力 URI を5つの `wf_train_il` に渡すだけです。

---

## MLflow：実験管理とモデルガバナンス

### W&B ではなく MLflow を選んだ理由

AutoE2E は Apache 2.0 のオープンソースプロジェクトです。
W&B を使うとコントリビューター全員にアカウントが必要になり、走行データやチェックポイントが W&B サーバーに送信されます。
自動車 OEM や Tier-1 サプライヤーとの協業を考えると、データが第三者クラウドに出ることを許容する企業はほぼありません。

MLflow は Apache 2.0 ライセンスで、Helm 一発で EKS にデプロイできます。
メタデータは RDS PostgreSQL に、アーティファクトは S3 に格納され、全データが自インフラに閉じます。
コントリビューターは URL を開くだけでアクセスでき、アカウント登録も課金も不要です。

### 評価タスクだけがログを書く設計

AutoE2E のパイプラインでは、MLflow にログを書くのは評価タスクだけです。

学習タスクも epoch ごとに loss をリアルタイムで MLflow に書くことは可能です。
しかし学習中に OOM で落ちたり Spot で中断されると、MLflow に不完全な Run が残ります。
そこで学習タスクは全ての情報（パラメータ、epoch ごとの loss、文脈情報）を `metadata.json` に書き出し、評価タスクがチェックポイントと一緒に受け取って、評価完了後に MLflow へ一括ログする設計にしました。

```python
def _run_evaluation(checkpoint, shards, train_metadata, dataset, experiment_name):
    meta = json.load(open(train_metadata.download()))

    with mlflow.start_run(run_name=f"{bb}-{fm}-e{epochs}"):
        mlflow.log_params({
            "data/dataset": meta["data"]["dataset"],
            "model/backbone": meta["model"]["backbone"],
            "train/lr": meta["training"]["lr"],
            "ctx/train_execution_id": meta["context"]["flyte_execution_id"],
        })
        for i, loss in enumerate(meta["training"]["losses_per_epoch"]):
            mlflow.log_metric("train/loss", loss, step=i)
        mlflow.log_metrics({"eval/ade": avg_ade, "eval/fde": avg_fde})
        mlflow.log_artifact("config.yaml")
        mlflow.register_model(model_uri, "auto-e2e-driving-policy")
```

この設計により、MLflow に存在する全ての Run は「学習が完了し、評価も完了した完全な記録」です。
全ての Run が同じスキーマを持つため、backbone や fusion_mode でフィルタして ADE でソートすればベストモデルが見つかります。

### Quality Gate

AutoE2E では ADE（6.4秒間の予測軌跡と正解軌跡の平均偏差）が 2.0m 未満、FDE（最終予測地点の偏差）が 4.0m 未満であれば PASS としています。

```python
passed = avg_ade < 2.0 and avg_fde < 4.0
mlflow.log_metrics({"eval/gate_pass": 1.0 if passed else 0.0})
```

Gate を通過したモデルのみが Model Registry に登録されます。
品質基準を満たさないモデルが下流に流れることを構造的に防ぐ仕組みであり、ISO 21448（SOTIF）が要求する「性能限界の体系的管理」の実装に相当します。

### MLflow をどう使っているか

ここまで設計思想の話をしてきましたが、実際に MLflow のダッシュボードを日常的にどう活用しているかを書きます。

AutoE2E の MLflow には `imitation-learning` と `offline-rl` の2つの Experiment があります。
`imitation-learning` には IL（模倣学習）で学習したモデルの評価結果が、`offline-rl` には IQL で refinement した後の評価結果が入ります。

たとえば「SwinV2-Tiny の concat fusion と cross_attn fusion、どちらが精度が良いか」を調べたいとき、やることは以下だけです。

1. `imitation-learning` Experiment を開く
2. 検索バーに `` params.`model/backbone` = "swin_v2_tiny" `` と入力する
3. `eval/ade` 列をクリックして昇順ソート

これで backbone を固定したまま fusion_mode ごとの ADE が一覧できます。
「concat が 1.82m、cross_attn が 1.64m、BEV が 1.51m」のように一目で差がわかる。

次に、2つの Run にチェックを入れて Compare ボタンを押すと、パラメータの差分がハイライトされ、`train/loss` の epoch 曲線が重ねて表示されます。
「cross_attn は concat より epoch 2 以降で loss が下がっている」「しかし学習時間は 1.3 倍」のようなトレードオフが視覚的に把握できます。

さらに、各 Run の Artifacts タブを開くと `config.yaml` とチェックポイントが保存されています。
`config.yaml` には学習時の全パラメータ（data、model、training、context）がネストされた YAML で記録されているため、任意の Run を完全に再現できます。

```yaml
# config.yaml の中身（抜粋）
data:
  dataset: yaak-ai/L2D
  shard_dir: s3://auto-e2e-platform-artifacts/shards/l2d/v3/
model:
  backbone: swin_v2_tiny
  fusion_mode: cross_attn
  embed_dim: 256
  num_views: 7
training:
  epochs: 10
  batch_size: 4
  lr: 0.0001
  weight_decay: 0.01
  amp: true
  final_loss: 0.0234
context:
  flyte_execution_id: f8a3b2c1d4e5
  docker_image: xxxxxxxxx.dkr.ecr.us-west-2.amazonaws.com/auto-e2e/training:v2.1
```

Model Registry の `auto-e2e-driving-policy` を開くと、Gate を通過した全バージョンが時系列で並んでいます。
各バージョンからソース Run にリンクが張られているので、「v8 は v7 と比べてどのパラメータが変わったか」をワンクリックで確認できます。

Offline RL の効果を検証するときは、同じ backbone/fusion_mode/dataset の IL Run と RL Run の `eval/ade` を比較します。
IL が 1.64m で RL が 1.42m なら、IQL による refinement で 0.22m 改善したと判定できます。
この比較は `ctx/train_execution_id` で IL と RL の親子関係が辿れるため、正しい対応関係を取り違えることがありません。

週次のモデル改善ミーティングでは、MLflow の Compare 画面をそのまま共有しています。
「先週から今週で ADE が 0.3m 改善した」「改善の原因は learning rate を 3e-4 → 1e-4 に下げたこと」のような議論が、パラメータ差分の画面を見ながらできます。

全ての Run が同じスキーマでパラメータを記録しているため、「ある1つの条件だけを変え、残りを全て揃えた比較」がフィルタだけで実現できます。
たとえば「データセットを L2D に固定し、epochs=10, batch_size=4, lr=1e-4 を揃えた状態で、backbone だけを swin_v2_tiny / convnext_v2_tiny / resnet_50 で変えた場合の ADE/FDE」を抽出するのに、追加のコードは不要です。
検索バーに条件を並べてソートするだけです。

同様に fusion_mode を固定して backbone を変える、backbone を固定してデータセットを変える、learning rate だけを振る、といった ablation study の結果が MLflow に自然に蓄積されていきます。
パラメータ数（モデルサイズ）も `model/backbone` と `model/fusion_mode` の組み合わせから一意に決まるため、「精度 vs パラメータ数」「精度 vs 学習時間」のトレードオフも Run を並べるだけで見えます。

この仕組みのおかげで、論文を書く際の比較実験テーブルが格段に作りやすくなりました。
従来は「あのときどのパラメータで回したか」をノートやログから掘り起こす作業が発生していましたが、MLflow では条件を揃えた Run をフィルタして Export すれば、そのまま論文の Table になります。
ablation study を追加で回したいときも、既存の Run と条件を1つだけ変えたワークフローを起動するだけで、比較可能な結果が同じ Experiment に追加されます。
実験管理の基盤が整っていることで、研究としてのアウトプットの速度も上がっています。

### Flyte との双方向追跡

各 MLflow Run には Flyte の実行 ID が `ctx/train_execution_id` として記録されています。
あるモデルの精度が良いとわかったら、この ID で Flyte Console を検索すれば、どのデータでどのコードで作られたかが確認できます。
逆に Flyte の実行ページからチェックポイントを辿り、MLflow で評価結果を確認することもできます。

EU AI Act が要求する「学習データ → モデル → デプロイの完全な追跡可能性」は、この双方向追跡で満たされます。
後から追跡するのではなく、パイプラインが動くたびに追跡情報が自動的に蓄積される設計です。

---

## 実装のつまずきポイント

### Flyte の S3 認証

Flyte の内部ストレージライブラリ（stow/minio-go）は AWS SDK v1 を使っており、Pod Identity や IRSA をサポートしていません。
EKS で推奨される認証方式が使えないため、静的な IAM アクセスキーを configmap にパッチする null_resource を Terraform に追加しています。

`terraform apply` のたびに Helm が configmap を上書きするため、毎回 post-apply スクリプトで再パッチが必要です。
Flyte の将来バージョンで AWS SDK v2 に移行することが期待されますが、現時点ではこのワークアラウンドが必須です。


---

## まとめ

**Kueue** は GPU の無駄遣いを防ぎます。
ジョブレベルのアドミッション制御で部分起動を防止し、優先度によるプリエンプションで本番学習を最優先にします。

**Flyte** はパイプラインの安全な実行を保証します。
FlyteFile / FlyteDirectory で大容量データを型安全にリレーし、キャッシュで再計算を防ぎ、不変な実行記録で規制要件を満たします。

**MLflow** はモデルの品質管理を担います。
OSS セルフホストでデータ主権を確保し、単一ログポイント設計で完全な実験記録を保ち、Quality Gate と Model Registry で品質基準を満たさないモデルの流出を防ぎます。

AutoE2E はオープンソースの自動運転スタックとして MLOps ガバナンスを組み込んだプロジェクトであり、この基盤の上で2027年の公道デモに向けてモデル改善を進めていきます。

---

## 参考リンク

- [AutoE2E リポジトリ](https://github.com/autowarefoundation/auto_e2e)
- [Woven by Toyota × Flyte Case Study](https://web.union.ai/case-study/how-woven-by-toyota-saved-millions-with-scaled-autonomous-driving-from-union-ai)
- [Wayve × Flyte Case Study](https://web.union.ai/case-study/wayve-accelerates-autonomous-driving-innovation-with-flytes-scalable-orchestration)
- [Flyte vs Kubeflow Pipelines（Theorem LP 移行事例）](https://web.union.ai/blog-post/production-grade-ml-pipelines-flyte-vs-kubeflow)
- [No Idle GPUs: Managing Research Compute at Runway（Kueue）](https://runwayml.com/news/no-idle-gpus-managing-research-compute-at-runway)
- [Kueue ドキュメント](https://kueue.sigs.k8s.io/)
- [BMW × AWS ペタバイト規模ストレージ](https://aws.amazon.com/blogs/industries/aws-professional-services-and-bmw-collaborate-to-optimize-petabyte-scale-storage-costs-for-autonomous-driving-data/)
- [Five-Layer MLOps Architecture for Connected Automated Driving（2025）](https://arxiv.org/html/2605.12719v1)
- [Flyte ドキュメント](https://docs.flyte.org/)
- [MLflow ドキュメント](https://mlflow.org/docs/latest/index.html)
- [前回の記事：PyTorchで理解する自動運転マルチカメラ統合の全体像](https://zenn.dev/riita10069/articles/auto-e2e-multi-view-fusion-explained)
