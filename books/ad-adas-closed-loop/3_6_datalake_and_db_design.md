---
title: "3.6 データレイク／データベース設計"
---

# 3.6 データレイク／データベース設計

この節では、自動運転向けデータレイク (data lake、生データを構造化せずに格納し、後から多様なクエリを当てる方式) とデータベース設計を、具体的な DDL（Data Definition Language、テーブル定義の SQL 文）とエンジン比較に踏み込んで扱います。Drive / Scene / Frame の Iceberg テーブル設計、テーブルフォーマット（Iceberg / Delta / Hudi）の比較、時系列 DB の選定、Apache Arrow / Arrow Flight による高速転送、スキーマ進化と後方互換を整理します。

Closed-Loop の観点では、後段のシーン検索・データセット設計・評価が高速に回るスキーマとは何かを考えます。

## テーブルフォーマットの比較

レイク上で ACID（トランザクションの 4 性質：原子性・一貫性・独立性・永続性）・time travel（過去のスナップショットを読み出す機能）・スキーマ進化（カラム追加などをデータ書き換えなしで行う機能）を実現するオープンテーブルフォーマットを比較します。

| 観点 | Apache Iceberg [ST1](references#st1) | Delta Lake [ST2](references#st2) | Apache Hudi [ST3](references#st3) |
|---|---|---|---|
| 強み | hidden partitioning（パーティション式をスキーマに埋め込む方式）、スキーマ進化、time travel | ACID、Spark 親和、MERGE 性能 | upsert（更新と挿入を兼ねる操作）／ incremental pull |
| エンジン | Spark / Trino / Flink / Dremio 等広範 | Spark 中心 | Spark / Flink |
| 書込パターン | append + MERGE | append + MERGE | 高頻度 upsert |
| 自動運転での適性 | Drive / Scene / Frame の大規模分析 | Databricks 環境 | 高頻度更新メタデータ |

自動運転データレイクでは、**Iceberg + Parquet** [ST1](references#st1)[ST7](references#st7) の組み合わせが、エンジン非依存性と hidden partitioning による運用容易性から有力です。以降は Iceberg を前提に DDL を示します。実プロジェクトでの選定は次の判断軸が実務的です。

- **Databricks 中心（Spark SQL only）で MERGE INTO 多用 → Delta**（生態系の親和性・MERGE 実装の単純さで優位）。
- **複数エンジン（Spark + Trino + Flink）で読み書きする／ time travel を監査要件に使う → Iceberg**。Iceberg の time travel は `SELECT ... FOR VERSION AS OF <snapshot_id>` のように過去スナップショットを SQL で読み戻せる機能で、「特定モデルの学習時点でのデータ」を後日再現するのに使えます。
- **CDC（Change Data Capture、変更データを取り出す方式）のような高頻度 upsert が中心 → Hudi**。

Iceberg の制約として、スキーマ進化は column ID ベースの追加・型昇格・リネームに限定（複雑な構造変更は限定的）、time travel は manifest ファイル数が線形に増えるため定期 compaction（細かいファイルをまとめ直す処理）が必須、という運用負担があります。これも理解した上で採用判断します。

**規模別の指針**：

- TB 級・単一分析チーム → Delta + Databricks で運用負荷を最小化。
- 数百 TB〜PB 級・複数チーム / 複数エンジン → Iceberg を一次採用。Trino から学習・分析の両用クエリを当てる。
- ストリーミング更新（CDC、リアルタイム ETL）が主軸 → Hudi。ただし自動運転ログは追記中心のため少数派。

## Drive / Scene / Frame の Iceberg DDL

ログは 3 粒度で管理します。Drive（連続走行）、Scene（シナリオ単位）、Frame（単一時刻スナップショット）です。

3 つの Iceberg テーブルを以下の方針で設計します。

- **`lake.drives`（連続走行）**：主キー `drive_id`、属性として `vehicle_id`・`sw_version`・`start_ts`・`end_ts`・`odd_segment`・`weather`・`region`・`intervention_count`・`sensor_config_id` を持ちます。パーティションは `months(start_ts)` と `region` の 2 軸にし、月次・地域単位での分析を高速化します。
- **`lake.scenes`（シナリオ単位）**：主キー `scene_id`、外部キー `drive_id`、`start_ts`・`end_ts`・`scenario_tags`（`merge` / `pedestrian` / `rain` などの配列）・`risk_score` を持ちます。パーティションは `bucket(64, drive_id)` を採用し、Drive ごとの局所性とパーティション数のバランスを取ります（バケット分割は `drive_id` のハッシュで 64 個に振り分ける方式）。
- **`lake.frames`（単一時刻スナップショット）**：主キー `frame_id`、外部キー `scene_id`・`drive_id`、`ts_utc`・`blob_uri`・`codec`・`sync_quality`・`label_status`・`model_run_id` を持ちます。パーティションは `days(ts_utc)` と `bucket(256, drive_id)` の 2 軸とし、日次の時間範囲スキャンと Drive 単位の局所性を両立させます。

いずれも Iceberg の `format-version=2` を有効にし、行レベル削除と MERGE INTO（既存行を条件で更新・挿入する SQL）に対応させます。

この設計により、頻出クエリが効率化されます。たとえば「都市高速×夜間×雨で、最新 SW バージョン（`v12.3`）の介入タグ付きシーンを含み、同期品質が `good` の Frame だけを抜き出す」というクエリを考えます。実行は次の流れになります。

1. `drives` × `scenes` × `frames` を 3 段ジョインします。
2. `odd_segment` でパーティションプルーニング（不要パーティションをスキャンしない最適化）を効かせます。
3. `scenario_tags` の配列含有判定と `sync_quality` の述語フィルタを通します。
4. 出力は `frame_id` と `blob_uri` のみとし、後段で対応するバイナリだけ読み込みます。

Drive/Scene/Frame の外部キーと、`odd_segment`・`ts_utc`・`drive_id` のパーティション/バケットにより、モデル評価ビュー・インシデント調査ビュー・フリート統計ビューを高速に実現します。主キーの命名規則は早い段階で固めるとよく、`drive_id = <vehicle>_<yyyymmddhhmmss>`、`scene_id = <drive_id>_<scene_seq>`、`frame_id = <drive_id>_<frame_no:08d>` のような連結キーが扱いやすい例です。代表クエリ 5 本（最新評価、ODD 別カバレッジ、インシデント周辺取得など）を benchmark に登録して p95 応答時間を SLA 化しておくと、compaction 不足によるスロークエリも検知しやすくなります。

## 時系列 DB の選定

テレメトリやモデルスコアのような高頻度スカラ時系列は、レイクとは別に時系列 DB で扱うとドリフト検知やトレンド分析が高速になります。代表エンジンを比較します [ST15](references#st15)。

| エンジン | モデル | 強み | 自動運転での用途 |
|---|---|---|---|
| TimescaleDB | PostgreSQL 拡張 | SQL 完全互換、`time_bucket`、JOIN 容易 | メタデータと結合する時系列 |
| InfluxDB | 専用 TSDB | 高書込スループット、保持ポリシー | 純粋なテレメトリ収集 |
| ClickHouse | 列指向 OLAP | 圧倒的な集計速度、PB 級スキャン | フリート横断の大規模集計 |
| Apache Druid | リアルタイム OLAP | 低レイテンシ集計、取込同時クエリ | リアルタイムダッシュボード |

メタデータと JOIN するなら TimescaleDB、フリート横断の巨大集計なら ClickHouse、という使い分けが実務的です。たとえばモデル不確実性のドリフト監視ダッシュボードでは、TimescaleDB の `time_bucket` 関数を使って `telemetry` テーブルを 1 分間隔で集約し、車両 ID ごとに `model_uncertainty` の平均を直近 24 時間ぶん算出する、といったクエリが標準形になります。これは PostgreSQL の SQL でそのまま書け、車両マスタや Drive メタデータとの JOIN も自然に組み込めます。

役割分担としては、テレメトリ → 時系列 DB、Drive/Scene/Frame メタデータ → Iceberg/RDB、シーン類似検索 → Vector DB（高次元ベクトルの近似最近傍検索に特化した DB、3.7 節）、全文検索 → 検索エンジン、となります。第4・7章のシーン検索やカバレッジ分析はこれらをまたぐため、「どの情報がどこにあり、どう JOIN するか」をアーキテクチャとして整理しておきます。

**規模別の指針**：

- 〜数億行 → TimescaleDB 単体で十分。SQL 互換で運用も容易。
- 数十億〜数千億行、フリート横断集計が中心 → ClickHouse へ移行。圧縮率と集計速度が桁違い。
- 真にリアルタイム（数秒以内のダッシュボード更新）→ Druid を別建てで併用。

## Apache Arrow / Arrow Flight による高速転送

学習クラスタへ大量のメタデータ／特徴量を転送する際、行指向の JDBC / ODBC はシリアライズ（オブジェクトをバイト列に変換する処理）がボトルネックになります。Apache Arrow（列指向のインメモリデータ形式の標準仕様）と Arrow Flight（gRPC ベースの高速転送プロトコル）を使うと、ゼロコピー（メモリコピーを省略する転送）に近い効率で並列転送できます。Iceberg / Parquet は Arrow と相性がよく、PyArrow 経由でそのまま `RecordBatch`（Arrow の基本単位）として読み出せます。

## スキーマ進化と後方互換

センサ構成・ラベルスキーマ・モデル出力形式は時間とともに変化します。Iceberg はカラム ID ベースのスキーマ進化を備え、列の追加・リネーム・型昇格を既存データを書き換えずに行えます。

具体的な進化操作は標準 SQL の `ALTER TABLE` で行います。たとえば `lake.frames` に `ego_velocity DOUBLE` 列を追加する場合、追加だけでよく、既存行は NULL 扱いになります。`model_run_id` を `experiment_id` にリネームする場合も、Iceberg がカラム ID で追跡しているため既存ファイルの書き換えは不要です。

運用方針は「追加は許す、削除・型変更は慎重に」です。非互換変更が必要な場合は新旧スキーマを移行期間中併存させ、変換レイヤで吸収します。スキーマには `schema_version` を持たせ、「どのモデルがどのスキーマバージョンのデータで学習されたか」を追跡し、評価結果との比較可能性を担保します。**読み出し時の挙動**：Iceberg は schema projection によって、新スキーマで open しても旧データファイルを **物理的に書き換えずに** 読めます。新規追加列は default value（指定なしなら NULL）で埋められ、リネームはカラム ID 追跡で旧名と新名を同一視します。

なお、Iceberg の schema projection に任せず変換レイヤを自前で持つ場合は、ETL（Extract Transform Load、抽出・変換・投入の一連のデータ処理）ジョブ内に「旧スキーマ → 新スキーマ」のマイグレーション関数を 1 段差し込みます。具体的には、(1) 旧名カラムを新名にリネームし元の値をコピー、(2) 追加カラムをデフォルト値（多くは NULL）で初期化、(3) `schema_version` フィールドを新バージョンに更新、という 3 ステップです。新規プロジェクトでは Iceberg の schema projection に任せ、レガシーパイプラインの互換維持にのみ自前変換レイヤを使う方針が無難です。スキーマ変更は GitHub PR + 設計レビューを必須化し、列削除・型変更は四半期ごとのメンテナンスウィンドウでまとめて実施することで、突発的な互換崩れを避けられます。

## 本節の振り返り

Drive / Scene / Frame の 3 階層は、自動運転データの「物理的な単位」と「分析的な単位」を一致させた設計です。連続走行の単位（Drive）と、シナリオとして意味を持つ単位（Scene）と、機械学習の最小単位（Frame）を別テーブルにし、外部キーで結ぶことで、シーン検索もインシデント調査もフリート統計もすべて同じスキーマの上で表現できます。テーブルフォーマット選定で Iceberg を中心に据える理由は、エンジン非依存性と hidden partitioning にあります。Spark でも Trino でも Flink でも同じテーブルを読める性質は、学習チームと分析チームが別ツールを使う組織で決定的に効きます。実務でよく見る失敗は、最初に Delta を選んで Databricks に縛られた後、Trino から学習データセットを組みたくなってもエンジン互換性に苦しむケース、もう一つは Iceberg を採用しても compaction を回さずに manifest ファイルが累積してクエリが遅くなるケースです。後者は「Iceberg は ACID テーブルであり、定期メンテナンスが必要」という運用認識の欠落が原因です。データ中心 AI の主張において、データレイクのスキーマは「モデルが学ぶ世界の構造」を表すため、データエンジニアと ML 開発者は `schema_version` を pin した再現可能な評価を Closed-Loop の必須要件として扱う必要があります。

## 次節への橋渡し

レイクとテーブルが整っても、「どこに何があり、どこから派生したか」を人とジョブが辿れなければ活用は進みません。次の 3.7 節では、**メタデータ管理とカタログ**を扱い、DataHub/OpenMetadata 等のカタログ比較、OpenLineage/Marquez によるデータリネージ、SKOS タクソノミ、Vector DB（Milvus/Weaviate/pgvector）を具体化します。
