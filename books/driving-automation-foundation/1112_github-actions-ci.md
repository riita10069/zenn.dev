---
title: "GitHub ActionsによるCI"
---

# GitHub ActionsによるCI

## ひとことで言うと
コードをリポジトリに push したり Pull Request を出すたびに、機械が自動で「文法・スタイルのチェック(lint)」と「テスト」を回してくれる仕組みです。人間がうっかり壊れたコードを混ぜ込んでも、機械が数分で気づいて赤信号を出すので、壊れたコードが本流に入る前に弾けます。GitHub Actions は、その自動実行をクラウド上で代行してくれる GitHub の機能で、リポジトリの中に設定ファイルを1つ置くだけで動き始めます。

## 直感的な理解
複数人で1つの文書を共同編集する場面を想像してください。誰かが章を書き換えるたびに、全員が「他の章と矛盾していないか」を手で読み直すのは非現実的です。確認を忘れると矛盾が紛れ込み、後で「いつ矛盾したのか分からない」状態になります。

ソフトウェアでも同じです。複数人が同じコードベースを編集し、誰かの変更が既存機能を壊しても、すぐには気づきません。時間が経つほど「どの変更が壊したのか」の特定は難しくなります。ここで CI(Continuous Integration、継続的インテグレーション)という考え方が出てきます。CI は「コードを共有の場所に頻繁に統合し、そのたびに自動で検証する」やり方です。検証を機械にやらせると、

- 変更を入れた数分後には「壊れたか」が分かる。
- 壊れていれば、その変更の作者がまだ文脈を覚えているうちに直せる(時間が経つと自分のコードすら忘れる)。
- レビュアー(他人のコードを確認する人)が「テストが緑だから少なくとも基本動作はしている」と安心して、設計やロジックの中身に集中できる。

CI を遅らせて変更を溜め込むほど、統合時の衝突解決が爆発的に難しくなります。これを Fowler は "integration hell"(統合地獄)と呼びました。小さく頻繁に統合し、毎回自動検証するのが安全だ、というのが CI の核心です。

## 基礎: 前提となる概念

- **リポジトリ(repository)**: コードの保管庫。変更履歴をすべて記録する。
- **ブランチ / main**: 並行作業のための枝分かれ。main は本流の最新コードを指す慣習名。
- **push / Pull Request(PR)**: push はローカルの変更をリモートに送る操作。PR は「この変更を本流に取り込んでよいか」をレビューにかける単位。
- **lint(リント)**: コードを実行せずに、スタイル違反・未使用変数・明らかなバグの種を静的に検出するチェック。ruff(Python)、ESLint(JS)などのツールが担う。「実行する前に文章校正する」イメージ。
- **型チェック(type check)**: 「この変数は整数」「これは文字列」といった型の食い違いを実行前に検出する。Python では mypy、pyright など。
- **ユニットテスト**: 部品ごとの小さな自動テスト(詳細は [pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration))。
- **終了コード(exit code)**: コマンドが成功なら 0、失敗なら 0 以外を返す約束事。CI はこの数値だけ見て成功/失敗を判定する。
- **CI ランナー / ジョブ / ステップ / ワークフロー**:
  - **ワークフロー(workflow)**: 自動処理の一連の流れ全体。YAML ファイルで定義する。
  - **ジョブ(job)**: ワークフロー内の仕事のまとまり。ジョブごとに別のマシンで並行に動く。
  - **ステップ(step)**: ジョブ内の手順。上から順に実行される。
  - **ランナー(runner)**: ステップを実行する仮想マシン。GitHub がクラウドで提供する(hosted runner)か、自前マシンを登録する(self-hosted runner)。
- **YAML**: インデント(字下げ)で階層を表すテキスト設定形式。GitHub Actions の設定はこれで書く。

GitHub Actions の最大の利点は、自前でサーバを用意せず、リポジトリ内の `.github/workflows/` フォルダに YAML を1つ置くだけで、GitHub が用意したクラウド VM 上で自動実行してくれる点です。公開リポジトリなら実行時間が無料(または広い無料枠)という点も実務上大きい。

## 仕組みを詳しく

設定はリポジトリ内 `.github/workflows/*.yml` に書きます。典型的な「lint とユニットテストを回す」構成は次のようになります。

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: make setup
      - name: Lint
        run: make lint
      - name: Type-check
        run: make typecheck
      - name: Run unit tests
        run: make test
```

各部分を分解します。

- `name: CI` … ワークフローの表示名。GitHub の画面に出る。
- `on:` … トリガー(起動のきっかけ)。`push:` は誰かが push したとき、`pull_request:` は PR が作成/更新されたときに動く。ブランチを絞りたければ `on: { push: { branches: [main] } }` のように書ける。絞らなければ全ブランチの push と全 PR で動く。`pull_request` トリガはフォークからの外部貢献にも反応するため、公開プロジェクトでは特に重要。
- `jobs:` … 仕事のまとまり。ここでは `lint-and-test` という1ジョブ。複数ジョブは既定で並行に走る。
- `runs-on: ubuntu-latest` … どの VM で動かすか。`ubuntu-latest` は GitHub 提供の最新 Ubuntu を「毎回まっさらな状態で」借りる。前回の残骸が無いので再現性のある検証ができる(逆に毎回セットアップが必要、という代償もある。後述)。
- `steps:` … 手順を上から順に実行。

ステップの中身:

1. `actions/checkout@v4` … リポジトリのコードを VM にダウンロードする。これをしないと VM は空っぽ。`@v4` は公式アクションのメジャーバージョン指定で、安定性のためにピン留めする慣習。
2. `actions/setup-python@v5` … 指定バージョンの Python を導入。
3. `make setup` … 依存ライブラリ(PyTorch など)を入れる。`make` は Makefile に書いた手順の別名を呼ぶ道具で、「セットアップ用にまとめたコマンド列」を実行する。コマンドを Makefile に集約しておくと、ローカルと CI で同じ手順を共有でき、「手元では通るのに CI で落ちる」というズレを防げる。
4. `make lint` … lint ツールで静的チェック(詳細は [ruffによるPython lint](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1151_ruff-lint))。
5. `make typecheck` … 型チェック。
6. `make test` … ユニットテストを実行。

これらのうち1つでも非ゼロの終了コードを返すと、そのステップが失敗し、既定ではジョブ全体が「失敗(赤)」になり、以降のステップは実行されません。すべて成功(終了コード 0)なら「緑」。PR 画面にこの結果がステータスとして表示され、レビュー前に機械的に問題を弾けます。ブランチ保護ルール(branch protection)で「CI が緑でなければマージ不可」と設定すれば、赤いコードが main に入るのを制度的に止められます。

### 実行の流れ(ASCII図)

```
開発者が push / PR を出す
      │
      ▼
GitHub が .github/workflows/ の YAML を読む
      │
      ▼
Ubuntu 仮想マシンを起動(毎回まっさら)
      │
      ▼
checkout → Python導入 → setup → lint → typecheck → test
      │                                  │
      │(どれか失敗 / 終了コード≠0)      │(全部成功)
      ▼                                  ▼
   赤バツ表示                          緑チェック表示
   PR にステータス反映                 (保護ルール下で)マージ可能
```

### 速く保つ工夫

CI が遅いと開発のテンポが落ち、後述の研究が示すとおり開発者は CI を「邪魔」と感じ始めます。よく使われる高速化策:
- **重いテストを除外**: GPU や学習済み重みを要するテストはマーカーで分離し、CI では軽いものだけ回す([pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration))。設定の `addopts` で既定除外にしておけば取りこぼしも減る([pytest addoptsでデフォルト除外](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1141_pytest-addopts-default-exclude))。
- **軽い依存に差し替える**: GPU 不要なら CPU 専用ビルドを入れる([CPU-only PyTorchでのCI実行](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1103_cpu-only-pytorch-ci))。数 GB のダウンロードを数百 MB に縮められる。
- **キャッシュ**: `actions/cache` で依存物のダウンロード結果を再利用し、毎回まっさらな VM のセットアップ時間を削る。
- **並列・マトリクス**: `strategy.matrix` で複数の Python/OS バージョンを並列に検証する。
- **早期失敗(fail-fast)**: lint を最初に置き、安いチェックで先に弾くことで、重いテストに到達する前に多くの誤りを捕まえる。

## 手法の系譜と主要論文

CI は学術手法というより実践知ですが、概念の起源と実証研究があります。

- **概念の起源**: 継続的インテグレーションは Kent Beck の Extreme Programming("Extreme Programming Explained", 1999)で「1日に何度も統合する」プラクティスとして提唱され、Martin Fowler の解説記事 "Continuous Integration"(初版 2000 年代初頭、2006 年に大幅改訂)で広く定着しました。Fowler は CI の本質を「メンバー全員が頻繁に(少なくとも1日1回)統合し、各統合を自動ビルド(テスト込み)で検証すること」と定義し、integration hell を避ける動機を明確にしました。

- **基盤の進化**: 専用 CI サーバ(CruiseControl, 2001 / Hudson→Jenkins, 2005 以降)から、クラウド型 SaaS(Travis CI, 2011 / CircleCI)へ、そしてリポジトリに統合された GitHub Actions(2019 年一般提供開始)へと進化しました。GitHub Actions の新規性は「CI 設定をコードと同じリポジトリに置き、同じ PR でレビューできる」点(Configuration as Code)と、再利用可能な公式/コミュニティ製アクションの市場(Marketplace)です。設定がコードと一緒に履歴管理されるため、「いつ・なぜ CI 構成を変えたか」が追える。

- **ML 文脈**: Sculley らの "Hidden Technical Debt in Machine Learning Systems"(NeurIPS 2015)は、ML システムの壊れやすさはコードだけでなくデータ・設定・パイプラインの結合にも潜むと指摘し、CI/CD と自動テストの重要性を訴えました(MLOps の起点的論文の一つ)。Amershi らの "Software Engineering for Machine Learning: A Case Study"(ICSE-SEIP 2019)は、Microsoft の多数の ML チーム調査から、ML 開発では「速いフィードバックループ」が生産性を強く左右すると報告しています。

## 論文の実験結果(定量データ)

- **Hilton et al. (ASE 2016)**: GitHub 上の 34,544 プロジェクトを分析。CI を採用したプロジェクトは、採用していないプロジェクトに比べ Pull Request の取り込みが速く、より多くの PR を処理していました。さらに 1,529 名の開発者調査で、CI の主要な利得として「壊れた変更を早期に捕捉できる安心感」が挙がる一方、主要なコストとして「ビルド時間」と「CI 設定の保守」が挙げられました。指標の意味として、ここでの「PR 取り込み速度」は、レビュアーがテスト結果を信頼して判断を早められることの代理指標です。テスト時間がコストの筆頭に来る点が、前述の高速化策の動機になります。

- **Vasilescu et al. (FSE 2015)** "Quality and Productivity Outcomes Relating to Continuous Integration in GitHub": GitHub の多数のプロジェクトを統計的に分析し、CI を使うプロジェクトは外部貢献者からの PR をより多く受け入れつつ、バグ報告の増加を伴わなかったと報告しました。つまり「受け入れ量を増やしても品質を落とさない」両立を CI が支えたという定量的示唆です。CI が「門番」として機能することで、見ず知らずの貢献者の変更も安心して受け入れられる、という効果です。

これらが示す核心は、CI 単体が品質を作るのではなく「速く・信頼できるフィードバック」が開発者の行動(小さく頻繁に統合する、レビューを早く回す、外部 PR を受け入れる)を変え、それが結果として品質と速度を両立させる、という因果です。だからこそ「速く保つ」と「信頼できる(flaky でない)」の2つが CI の生命線になります。

## メリット・トレードオフ・限界

メリット:
- 壊れた変更を本流に入る前に自動で弾ける(ブランチ保護と組み合わせると制度的に強制できる)。
- レビュアーの負担が減る(機械が基本チェックを肩代わりし、人間は設計に集中)。
- 自前サーバ不要、設定ファイル1つで完結。公開リポジトリなら実行が無料/広い無料枠。
- 設定がコードと同じ場所にあり、PR でレビュー・履歴管理できる。

トレードオフ・限界:
- **待ち時間**: CI が遅いと開発のテンポが落ちる。速度維持の工夫が継続的に必要(キャッシュ、依存軽量化、テスト分離)。
- **保守コスト**: Python やアクションのバージョン更新、依存の変化に追従する手間がある。`@v4` のようなバージョン固定は安定するが、放置すると古くなりセキュリティ更新も止まる。
- **網羅性の限界**: CI は「書かれたテスト」を回すだけ。テストが無い部分は守られない。緑でもバグはありうる(CI の緑は「既知の壊し方をしていない」証明にすぎない)。
- **flaky test**: 環境起因でたまに落ちるテストがあると、開発者が CI を信用しなくなり「赤でもとりあえずマージ」する文化を生む(最悪のアンチパターン)。CI の価値は信頼性に乗っているため、flaky は単なる不便ではなく文化を壊す。
- **環境差の盲点**: GitHub の標準 VM には GPU が無いなど、本番と環境が違う。CI が緑でも本番特有の問題(GPU メモリ、CUDA カーネル挙動、混合精度の数値差)は検出できない([CPU-only PyTorchでのCI実行](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1103_cpu-only-pytorch-ci))。

未解決の課題として、ML プロジェクトでは「データやモデルの変化をどう CI で検証するか」が難しい。コードのテストは確立しているが、データ分布のドリフトやモデル精度の回帰(regression)を CI ゲートに組み込む方法は、評価が高コストかつ非決定的なため、研究・実務とも発展途上です。精度がしきい値を下回ったら赤にする、といったゲートは導入が進みつつあるものの、しきい値の決め方や評価データの管理が新たな課題になっています。

## 発展トピック・研究の最前線
- **CD(継続的デリバリ/デプロイ)への接続**: CI で緑になった成果物を自動でステージング/本番へ流す。CI/CD パイプラインとして一体運用するのが現代の標準。
- **マージキュー(merge queue)**: main へ入る直前に、複数 PR を順に積み上げた状態で全テストを再実行し、「個別には緑だが組み合わせると壊れる」事故(semantic merge conflict)を防ぐゲート。
- **セルフホストランナー**: GPU が必要なテストや内部リソースへのアクセスが要る場合、自前マシンをランナー登録して回す構成。CPU-only CI とセルフホスト GPU CI を二段で組み合わせるのが ML プロジェクトの定番。
- **ML 向け CI/CD(MLOps)**: モデル評価指標を PR に自動コメントする(CML など)、データバージョンと連動させる、学習を伴う重い検証を別パイプラインに切り出す等、ML 特化の CI 拡張が活発に研究・開発されている。

## さらに学ぶための関連トピック
- [CPU-only PyTorchでのCI実行](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1103_cpu-only-pytorch-ci)
- [pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration)
- [pytest addoptsでデフォルト除外](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1141_pytest-addopts-default-exclude)
- [ruffによるPython lint](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1151_ruff-lint)
- [lint失敗でCIを落とす運用](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1101_ci-fail-on-lint)
