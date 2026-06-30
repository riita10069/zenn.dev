---
title: "Helm (K8s パッケージング)"
---

# Helm (K8s パッケージング)

## ひとことで言うと
Helm は「Kubernetes (K8s) 上のアプリを、パラメータを差し替えるだけでまとめてインストール・更新・削除できるパッケージマネージャ」です。スマホのアプリストアからアプリを入れるように、ジョブキューや実験管理ツールやパイプライン基盤といった複雑なソフトウェアを、数十枚の設定ファイルを手書きせず1コマンドで K8s に導入できます。

## 直感的な理解
K8s に少し複雑なソフトを入れようとすると、必要な設定ファイル (マニフェスト) が何十枚にもなります。たとえば実験管理ツールを1つ入れるだけでも、本体を動かす Deployment、外部アクセス用の Service、DB 接続情報を持つ Secret、設定値の ConfigMap、権限を表す ServiceAccount と RBAC、永続データ用の PersistentVolumeClaim … と関連オブジェクトが大量に要ります。

これらを手で1枚ずつ書いて適用するのは大変なうえ、次の問題が出ます。

1. 使い回しにくい。別環境 (dev と prod) で「DB のホスト名とレプリカ数だけ変えたい」とき、全ファイルをコピーして該当箇所を手で書き換える必要があり、書き換え漏れが起きる。
2. バージョン管理が辛い。「ツールを v3.12 から v3.13 に上げたい」とき、関連する全ファイルを矛盾なく更新するのが難しい。新バージョンで増えたオブジェクトを足し忘れる。
3. アンインストールが漏れる。入れたものを消すとき、関連リソースを全部覚えていないと消し残し (孤児リソース) が出る。

Helm はこれを解決します。あるソフトの全マニフェストを「テンプレート」としてひとまとめ (これをチャート chart と呼ぶ) にし、変えたい値だけを別ファイル (values.yaml) で指定すれば、Helm がテンプレートに値を埋め込んで完成したマニフェストを生成し、まとめて適用してくれます。値を変えるだけで構成を切り替えられるのが本質です。

## 基礎: 前提となる概念
- コンテナ (container): アプリとその実行環境 (ライブラリ、OS の一部) を1つに箱詰めしたもの。どのマシンでも同じように動きます ([Amazon ECR (Container Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1105_ecr-container-registry) 参照)。
- Kubernetes (K8s): たくさんのコンテナを、何台ものマシン (ノード) の上で「どこで動かすか」「壊れたら再起動する」「負荷に応じて増やす」などを自動で面倒見るコンテナ管理システムです。
- マニフェスト (manifest): K8s に「こういうものを動かして」と指示する YAML ファイル。代表的な種類: Deployment (アプリ本体とレプリカ数)、Service (通信の窓口)、ConfigMap (設定)、Secret (秘密情報)、ServiceAccount (Pod の身分)、PersistentVolumeClaim (永続ストレージの要求)、Ingress (外部公開)。
- 宣言的 (declarative): K8s は「最終的にこうあってほしい状態 (desired state)」を受け取り、現状をそこに近づけ続けます。マニフェストはこの desired state の記述です。
- パッケージマネージャ (package manager): 依存関係を解決してソフトを一括導入・更新・削除する道具。Linux の apt/yum、言語の npm/pip と同じ役割を、Helm は K8s アプリに対して果たします。

## 仕組みを詳しく

### チャート・テンプレート・values の関係
Helm の中心概念は3つです。

```
チャート (chart)        … ソフト一式の「テンプレート集 + デフォルト値」
   ├─ templates/        … 値が穴あきになった YAML テンプレート群
   ├─ values.yaml       … 穴に入れるデフォルト値
   ├─ Chart.yaml        … チャートの名前・バージョン・依存チャート
   └─ charts/           … 依存する子チャート (subchart)

自分の values.yaml        … デフォルトを上書きしたい値だけ書く

[Helm が合成]  →  完成したマニフェスト  →  K8s に適用
```

具体例。チャートのテンプレートにこう書いてあるとします (穴あき部分が `{{ }}`)。

```yaml
# templates/deployment.yaml (チャート側)
spec:
  replicas: {{ .Values.replicaCount }}
  template:
    spec:
      containers:
        - image: "app:{{ .Values.image.tag }}"
          resources:
            limits:
              nvidia.com/gpu: {{ .Values.gpu.count | default 0 }}
```

自分はデフォルトと変えたい値だけ書きます。

```yaml
# 自分の values.yaml
replicaCount: 2
image:
  tag: "3.13"
gpu:
  count: 1
```

Helm はこれらを合成し、最終マニフェスト (`replicas: 2`, `image: "app:3.13"`, GPU 1基) を生成して適用します。テンプレートは Go の text/template をベースにしており、条件分岐 (`{{ if }}`)、ループ (`{{ range }}`)、デフォルト値 (`| default`)、パイプ関数などが書けます。上の例の `| default 0` は「値が未指定なら 0 を使う」という意味で、GPU を要求しない構成にも同じチャートを使い回せるようにしています。

### 基本コマンド
```bash
# チャートの配布元 (リポジトリ) を登録
helm repo add example https://example.github.io/charts/

# values.yaml を渡してインストール (release という名前を付ける)
helm install my-app example/app --version 1.2.3 -f my-values.yaml -n my-namespace

# 設定を変えたいときは upgrade (差分だけ適用)
helm upgrade my-app example/app -f my-values.yaml -n my-namespace

# 前のリビジョンに戻す
helm rollback my-app 1 -n my-namespace

# 入れたものをまとめて削除
helm uninstall my-app -n my-namespace

# 適用せず、生成されるマニフェストだけ確認 (デバッグの定番)
helm template my-app example/app -f my-values.yaml
```

release (リリース) とは「チャートを特定の値でインストールした1つの実体」です。同じチャートを別名・別値で2回入れれば、独立した2つの release になります。Helm は release 単位で「何を入れたか」をクラスタ内 (Secret として、リビジョン番号付きで) に記録するので、uninstall で関連リソースをまとめて消せ、rollback で過去のリビジョンに戻せます。この「release という単位での状態管理」が、生 `kubectl apply` との決定的な違いです。

### Helm v2 から v3 への進化 (歴史的に重要)
Helm v2 (2016 頃〜) は Tiller (ティラー) というクラスタ内のサーバコンポーネントを介して動きました。クライアント (helm) が Tiller に依頼し、Tiller がクラスタにマニフェストを適用する構造です。問題は、この Tiller がクラスタ全体への強い権限を持ち、しかも初期設定では認証が緩かったため、Tiller を踏み台にクラスタを乗っ取れる、というセキュリティ上の弱点があった点です。

Helm v3 (2019 年リリース) は Tiller を完全に廃止し、クライアントが直接 K8s API を叩く方式に変えました。これにより権限管理を K8s 標準の RBAC (Role-Based Access Control、ロールに基づくアクセス制御。[EKS Pod Identity / IRSA](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1138_pod-identity-irsa) と同じ認可基盤) に委ねられ、「Helm 自体が過大な権限を持つ」問題が解消されました。release の記録も Tiller のメモリから K8s の Secret へ移り、複数人が同じ release を扱えるようになりました。現在新規に使うなら v3 一択です。

### バージョン固定の意味
`--version` でチャートバージョンを固定すると「いつインストールしても同じものが入る」再現性が得られます。固定しないと、リポジトリ側でチャートが更新されたタイミングで中身が変わり、「昨日は動いたのに今日は壊れた」が起きます。どのチャートバージョンを今動かしているかを Git に記録しておくことで、後から構成を一意に追えます。これはコンテナイメージにダイジェスト (内容ハッシュ) やコミットハッシュのタグを付けて再現性を担保するのと同じ発想です ([Amazon ECR (Container Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1105_ecr-container-registry))。

### Terraform との役割分担
[Terraform (IaC)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1153_terraform-iac) が「クラウドの土台 (K8s クラスタ、ネットワーク、データベース、権限など)」を作るのに対し、Helm は「その K8s クラスタの上で動くアプリ (ジョブキュー、パイプライン、実験管理、推論サーバなど)」を入れます。土台と上物で道具を分けるのが一般的です。理由は、アプリ層は更新頻度が高く、K8s ネイティブな表現 (マニフェスト) で扱うほうが素直で、ロールバックや差分適用が効くからです。なお Terraform から Helm チャートを適用する (helm provider) こともでき、両者の境界はチームの方針で決まります。土台と上物の両方を1つの巨大 state や1つの巨大チャートに押し込むと、変更の影響範囲が読みにくくなるため避けるのが定石です。

## 手法の系譜と主要論文
Helm は CNCF (Cloud Native Computing Foundation) のプロジェクトで学術論文の手法ではありませんが、背景の考え方を整理します。

- パッケージマネージャの系譜: apt/yum、npm/pip のように「依存を解決してソフトを一括管理する」道具は長い歴史があります。Helm は「K8s アプリ版のパッケージマネージャ」としてこれを K8s に持ち込んだものです。共通の効果は「導入・更新・削除の手間を激減させ、依存の取り違えを防ぐ」点。共通のトレードオフは「提供者が用意した抽象 (values の項目) の範囲でしか細かく制御できない」点。
- テンプレートと設定の分離: 「ロジック (テンプレート) とデータ (値) を分ける」のはソフトウェア工学の古典的な良い習慣で、Web の MVC やテンプレートエンジン (Jinja2 等) と同じ発想です。これで「骨格は共通、環境差は値だけ」という管理ができます。
- Cloud Native の文脈: K8s は Google の Borg (Verma et al., "Large-scale cluster management at Google with Borg", EuroSys 2015) の経験を元にしたオープンソース版です。Borg 論文は「宣言的にジョブを記述しスケジューラに任せる」モデルを大規模に示しました。Helm はその宣言 (マニフェスト) を「再利用可能なパッケージ」として配布・管理しやすくする層を足したもの、と位置づけられます。
- 競合・後続: Kustomize (テンプレートではなく「ベース構成にパッチを重ねる」方式、kubectl に標準同梱で `kubectl apply -k`)、Jsonnet ベースの Tanka、CUE ベースのツールなど、「テンプレート文字列置換」の弱点 (型がない、空白でバグる、デバッグしにくい) を補う後続が登場しています。Helm と Kustomize は対立ではなく併用 (Helm でレンダリングして Kustomize でパッチ) もよく行われます。GitOps ツール (Argo CD、Flux) は Helm チャートを取り込んで宣言的に同期する形が一般的です。

## 論文の実験結果 (定量データ)
Helm はベンチマーク論文を持つ手法ではないため、実務上の効果を定量的な見方で示します。

- マニフェスト削減: 実験管理ツール・ジョブキュー・推論基盤のような構成要素では、生成されるマニフェストが数十オブジェクト (数百〜千行の YAML) に及ぶことが珍しくありません。公式チャート + values 数十行で導入できるため、人が書く YAML の行数を概ね1桁〜2桁分の1に圧縮できます。数値の意味: 書く量が減るほど、書き換え漏れによる構成不整合 (たとえば Service のポートと Deployment のポートがずれる) の確率が下がります。
- 再現性: `--version` 固定 + values.yaml の Git 管理により、別環境で同一構成を再現したときの差分を「values の差分だけ」に限定できます。これは「環境間の差異を最小化する」ことを意味し、dev で動いて prod で動かない問題の主因 (環境差) を減らします。
- ロールバックの即時性: release はリビジョン番号で履歴管理されるため、`helm rollback` でアプリを直前の正常状態へ秒〜分単位で戻せます。手作業で「どのファイルをどの値に戻すか」を思い出す必要がありません。
- アブレーション的な見方 (どの機能を抜くと何が壊れるか): Helm から「release 記録」を外す (生 kubectl apply にする) と、アンインストール時に消し残しが発生し、孤児リソースがクラスタに溜まります。「values 分離」を外す (テンプレートに値を直書き) と、環境差が表現できず使い回せません。「バージョン固定」を外すと、入れるたびに中身が変わり再現性が崩れます。「namespace 分離」を外すと、複数 release のリソースが混ざって管理不能になります。各機能がどの問題を解いているかが分かります。

## メリット・トレードオフ・限界
メリット

- 複雑なソフトの大量マニフェストを1コマンドで導入・更新・削除でき、消し残しが出にくい。
- values.yaml で値だけ差し替えれば、環境ごとの差分 (DB ホスト、レプリカ数、バージョン、GPU 要求量等) を最小の記述で表現できる。
- チャートバージョンを固定でき再現性が高い。コミュニティ提供の公式チャートをそのまま使え、車輪の再発明を避けられる。
- upgrade で差分だけ適用でき、release 単位でリビジョンを遡ってロールバックできる。

トレードオフ・限界

- チャートが用意した values の範囲でしか細かく設定できない。想定外のカスタマイズはテンプレート自体をフォークして手を入れる必要があり、提供元の更新追随コストが上がる。
- Go テンプレートの記法 (文字列ベースのため型がない) は読みにくく、空白やインデントの扱い (`{{- -}}` のトリム) でバグが出やすい。`helm template` で生成結果を必ず確認する習慣が要る。
- 今クラスタに適用されているのが正確にどの values かは release 履歴を見ないと分からない。values.yaml を Git 管理する規律 (= GitOps) が要る。
- アプリ層に向く道具であり、クラウドの土台リソースは [Terraform (IaC)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1153_terraform-iac) 等に任せるのが素直。両方を Helm に押し込むと管理が複雑化する。
- 複雑なチャートは内部のテンプレートロジックがブラックボックス化しやすく、依存子チャートの値の上書き経路 (どの階層の values が効くか) が直感に反することがある。

## 発展トピック・研究の最前線
- GitOps との統合: Argo CD / Flux が Helm チャートをリポジトリの宣言から自動同期し、ドリフト (クラスタ実態と Git の差) を検出・修復する運用が主流化。「Git が真実の源」という原則を Helm 運用に適用します。
- テンプレートの脱・文字列化: Kustomize のパッチ方式、CUE/Jsonnet/Pkl による型付き構成、Helm の値スキーマ検証 (values.schema.json で values の型を JSON Schema で縛る) など、文字列テンプレートの弱点を補う流れ。
- OCI レジストリでのチャート配布: チャートをコンテナイメージと同じ OCI レジストリ ([Amazon ECR (Container Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1105_ecr-container-registry) と同種の保管庫) に push する標準が整備され、イメージとチャートを同じ供給網・同じ認証で扱えるように。
- サプライチェーンの安全性: チャートの署名・検証 (provenance ファイルや cosign による署名)、values に紛れ込む機密の検出、依存子チャートの脆弱性スキャンなど、配布物の信頼性がセキュリティ研究・実践の対象に。

## さらに学ぶための関連トピック
- [Terraform (IaC)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1153_terraform-iac)
- [Amazon ECR (Container Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1105_ecr-container-registry)
- [EKS Pod Identity / IRSA](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1138_pod-identity-irsa)
- [ODCR (On-Demand Capacity Reservation)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1133_odcr-capacity-reservation)
