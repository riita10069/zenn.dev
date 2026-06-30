---
title: "プラットフォーム中立な requirements.txt と torch wheel チャネル選択"
---

# プラットフォーム中立な requirements.txt と torch wheel チャネル選択

## ひとことで言うと

ライブラリのバージョンを書き留めた `requirements.txt` から「特定の CUDA バージョン専用」という縛りを外し、インストールするときに「CPU 版か、どの CUDA 版か」を選べるようにする話です。これにより、GPU が無いパソコン・CI(自動テスト)環境・どの世代の GPU マシンでも、同じ 1 つのバージョン固定ファイルを共有できます。バージョン番号(再現性の核)は固定したまま、ハードウェア向けの「味付け」だけをインストール時の引数で切り替える、という設計です。

## 直感的な理解

機械学習のプロジェクトは、たくさんのライブラリに依存します。問題は、同じコードでも「入れたライブラリのバージョンが違う」と挙動が変わることです。数値計算は版が変わると微妙に結果がずれるので、「昨日は通ったのに今日は落ちる」「私の環境では再現しない」という事故が起きます。これを防ぐため、使うバージョンをファイルに書いて固定します。これを「ピン留め(pin)」と呼びます。

ところが PyTorch のような GPU を使うライブラリには、もう一段の難しさがあります。PyTorch は CPU 専用版と、GPU(NVIDIA の CUDA)を使う版が別々に配布されており、さらに CUDA のバージョン(11.8、12.1 など)ごとに専用の部品があります。ここで「CUDA 11.8 専用」とファイルに書き込んでしまうと、GPU が無いパソコンや別の CUDA を使う環境ではインストールが失敗します。

この失敗を避けようとして、現場ではしばしば「じゃあピンファイルを使うのをやめて、なりゆきで最新版を入れよう」という回避策が取られます。しかしこれは本末転倒で、せっかくバージョンを固定したのに、実際には別のバージョンが入る。ピン留めの意味が消えます。

正しい解決は、ファイルからハードウェア依存(CUDA の縛り)だけを切り離し、それをインストール時の引数に追い出すことです。バージョン番号はファイルに固定したまま、「CPU か、どの CUDA か」だけを実行時に選ぶ。これで 1 つのファイルがすべての環境で使え、かつ再現性も保たれます。

## 基礎: 前提となる概念

用語を平易にします。

- requirements.txt: Python で「どのライブラリのどのバージョンを使うか」を 1 行ずつ書いたテキストファイル。`pip install -r requirements.txt` で一括インストールできます。`torch==2.4.1` は「torch をバージョン 2.4.1 にピン留めする」という意味です。
- torch(PyTorch): ニューラルネットワークを構築・学習させるための中心的ライブラリ。
- CUDA(クーダ): NVIDIA 製 GPU を汎用計算に使うための仕組み。GPU を使うと学習が桁違いに速くなります。CUDA にはバージョンがあり、torch はそのバージョンごとに別ビルドを配布します。
- wheel(ホイール): Python ライブラリの配布形式(`.whl` ファイル)。pip がこれをダウンロードして展開します。GPU 向けの torch は、PyPI(標準のパッケージ置き場)ではなく PyTorch 専用の wheel 置き場(チャネル)から取得します。
- PyPI(パイピーアイ): Python の標準パッケージリポジトリ。`pip install` が既定で見に行く場所です。
- index / extra-index-url: pip が wheel を探しに行く URL。`--extra-index-url` は「標準の PyPI に加えて、ここも探していい」という追加の置き場指定です(PEP 503 で定義される Simple Repository API の形式)。
- ローカルバージョン識別子: `2.4.1+cu118` の `+cu118` の部分。PEP 440 で定義される「ローカルバージョン」で、「同じ 2.4.1 だが CUDA 11.8 向けにビルドしたもの」という印です。この `+` 以降が付いていると、その正確な文字列に一致する wheel しか解決できません。

## 仕組みを詳しく

中立化は大きく 2 か所の変更で成り立ちます。

### 1. requirements.txt から CUDA の縛りを外す

before(縛りあり):

```
torch==2.4.1+cu118
--extra-index-url https://download.pytorch.org/whl/cu118
```

after(中立):

```
torch==2.4.1
```

`+cu118` のローカルバージョンタグと、専用 index の指定を削除します。タグの無い `torch==2.4.1` は、ふつうの PyPI にある(あるいは PyTorch チャネルの該当版にある)wheel として解決できます。これだけで `pip install -r requirements.txt` が、追加の URL 指定なしに通るようになります(既定では CPU 版が解決される)。

中立化後の requirements.txt は、ハードウェアに依存しないシンプルなピン一覧になります。例:

```
mypy==2.1.0
numpy==2.4.6
pytest==9.0.3
ruff==0.15.16
timm==1.0.27
torch==2.4.1
```

ここで `torch==2.4.1` という「バージョンだけの固定」が、再現性の核として残ります。`+cu118` を消しても 2.4.1 という番号は固定されたままなので、「どの環境でも torch 2.4.1 系を使う」という保証は失われません。

### 2. インストール時にハードウェアの種類を選べるようにする

ハードウェア依存は、ビルドツール(Make など)側の変数に追い出します。一般化した例:

```make
TORCH_CHANNEL ?= cpu
setup:
	pip install -r requirements.txt \
	  --extra-index-url https://download.pytorch.org/whl/$(TORCH_CHANNEL)
```

- `?=` は「もし指定されていなければこの値を使う」という Make の記法で、既定は `cpu` です。
- `--extra-index-url` で PyTorch のチャネル URL を追加します。`$(TORCH_CHANNEL)` が URL に差し込まれるので、選んだチャネルから torch の wheel を取りに行きます。

これで同じ requirements.txt のまま、次のように使い分けられます。

- `make setup` → `https://download.pytorch.org/whl/cpu` を見て CPU 版 torch 2.4.1
- `make setup TORCH_CHANNEL=cu118` → `.../whl/cu118` を見て CUDA 11.8 版 torch 2.4.1
- `make setup TORCH_CHANNEL=cu121` → CUDA 12.1 版 torch 2.4.1

バージョン(2.4.1)は固定したまま、CPU/GPU の味付けだけを切り替えるイメージです。同名の torch 2.4.1 が、指定したチャネルでは CUDA 対応ビルドとして解決されます。

### 3. CI が勝手に最新へ流れるのを止める

ピンファイルを使わずに「なりゆき任せ」でインストールしていた頃は、CI が実行のたびに最新の torch(たとえば 2.12 のような)を引っ張ってくる、という挙動になりがちでした。中立化したピンファイルを CI でもそのまま使うようにし、CI 用に上書きしていた環境変数(余計な extra-index 指定など)を削除すると、CI はピン通りの torch 2.4.1(CPU 版)を使うようになります。これで「書いたバージョンと実際に入るバージョンが違う」矛盾が消えます。

before / after をまとめます。

| | before | after |
|---|---|---|
| requirements.txt の torch | `torch==2.4.1+cu118`(CUDA 11.8 専用) | `torch==2.4.1`(中立) |
| CPU 環境での `pip install -r` | 失敗しがち | そのまま成功 |
| CI が入れる torch | なりゆきで最新(例 2.12) | ピン通りの 2.4.1+cpu |
| CUDA 11.8 を使いたい人 | 既定で入った | `TORCH_CHANNEL=cu118` と明示 |

ここで重要なトレードオフが 1 つあります。これまで既定で CUDA 11.8 版が入っていた人にとっては、中立化後は `TORCH_CHANNEL=cu118` を明示しないと CPU 版が入る、という挙動の変化(behavior change)が起きます。暗黙の既定が変わるので、移行時にはドキュメントで周知する必要があります。

## 手法の系譜と主要論文

これは依存関係管理(dependency management)と再現性(reproducibility)に関するソフトウェア工学の話で、自動運転のアルゴリズム論文が扱うテーマではありません。背景の系譜を示します。

- PEP 440(Version Identification, 2013): Python のバージョン文字列の規格。`2.4.1+cu118` の `+` 以降を「ローカルバージョン識別子」として定義します。CUDA タグがここに乗る仕組みの根拠です。
- PEP 503(Simple Repository API, 2015): pip が wheel を探すリポジトリの形式を定義。`--extra-index-url` で複数のリポジトリを束ねる挙動の土台です。
- PyTorch のチャネル設計(download.pytorch.org/whl/{cpu,cu118,cu121,...}): CPU と各 CUDA 版を別 URL のチャネルに分けて配布する方式。本トピックの「チャネル選択」はこの設計を前提にしています。
- 機械学習の再現性をめぐる議論: Pineau et al. "Improving Reproducibility in Machine Learning Research"(JMLR 2021, arXiv:2003.12206)や、NeurIPS の Reproducibility Checklist(2019 年導入)は、「環境とバージョンを固定しないと同じコードでも結果が再現できない」ことを繰り返し指摘しています。バージョンのピン留めはこの問題への直接の対策です。

これらは「自動運転の手法」ではなく「実験と開発を安定させる基盤」の文献です。

## 論文の実験結果(定量データ)

依存管理そのものの定量実験はありませんが、再現性研究には具体的な観察データがあります。

- ML の再現性調査では、論文の多くが「コード公開あり」でも、依存バージョンの未固定や環境差のせいで報告された数値を再現できない事例が多数あると報告されています(Pineau ら 2021 ほか)。これは「コードがあれば再現できる」わけではなく、「環境とバージョンの固定」が再現の必要条件であることを示すデータです。
- 数値計算ライブラリの版差による結果のズレは実測されており、たとえば BLAS の実装差や、浮動小数点の縮約順序の違い、torch のマイナーバージョン間でのデフォルト挙動変更(初期化やアルゴリズム選択)によって、同じ学習スクリプトでも最終精度が数値レベルで変わることがあります。差は小さくても、ベンチマークの順位を左右する規模になり得るため、バージョン固定が重視されます。

本トピックの直接の効果は件数では測りにくいものの、「CI が入れる torch がなりゆきの最新から固定の 2.4.1 に変わる」ことで、CI が突然落ちる(最新版の破壊的変更を踏む)種類の失敗がゼロになります。これは再現性の改善を、CI の安定という形で具体化したものです。

## メリット・トレードオフ・限界

メリット:
- 1 つの requirements.txt が CPU・CI・任意の CUDA 世代すべてで使える。環境ごとに別ファイルを保守しなくてよい。
- CI がピン通りのバージョンをテストするようになり、「書いたバージョンと違うものが入る」矛盾が消える。CI が最新版の破壊的変更で突然落ちることがなくなる。
- ハードウェアの違いを「インストール時の引数」に追い出せるので、ファイル本体は読みやすいまま保てる。

トレードオフと限界:
- これまで既定で CUDA 版が入っていた人は、`TORCH_CHANNEL=cu118` のように明示する手間が増える(暗黙の挙動が変わる behavior change)。移行時の周知が必要。
- `--extra-index-url` を使うため、同名 wheel が複数のチャネルに存在すると、意図しない方を拾うリスクがある(dependency confusion と呼ばれる一般的な pip の注意点)。これは厳密には `--index-url` で単一に絞るか、ハッシュ固定(`--require-hashes`)で緩和します。
- バージョンを固定する以上、更新は手動。固定したまま放置すると、セキュリティ修正や新機能を取り逃す。ピンの定期的な見直し(更新の保守コスト)が宿命的に発生します。
- requirements.txt は「直接依存」を固定しますが、その依存がさらに依存するもの(推移的依存)までは完全には固定しません。完全な固定には lock ファイル(後述)が要ります。

## 発展トピック・研究の最前線

- ロックファイルによる完全固定: pip-tools(`pip-compile`)、Poetry、PDM、uv などは、推移的依存まで含めた完全なバージョン・ハッシュの lock ファイルを生成します。requirements.txt のピンより一段強い再現性を与え、近年は uv の高速さもあって急速に普及しています。
- 環境のコンテナ化: Docker イメージにライブラリと CUDA ランタイムを丸ごと封じ込めると、ホストの CUDA バージョンに依存せず「イメージ = 再現可能な環境」になります。本トピックのチャネル選択を、イメージのビルド段階で 1 回だけ行う発想です。CUDA を含む公式 PyTorch イメージはこの典型です。
- プラットフォーム別 wheel の自動解決: 近年の PyTorch は、環境を検出して適切なチャネルを案内する仕組みや、CUDA を同梱した wheel の配布を進めており、「ユーザがチャネルを手で選ぶ」必要を減らす方向の改善が続いています。
- 再現性の標準化: 学会レベルでの Reproducibility Checklist や、実験のメタデータ(乱数シード、ライブラリ版、ハードウェア)を自動記録する実験管理ツール(MLflow、Weights & Biases など)との連携が、依存固定を実験全体の再現性に結びつける流れになっています。

## さらに学ぶための関連トピック

- [Makefile によるテストスイートの選択的実行 (SUITE パラメータ方式)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1123_makefile-test-suite-targets)
- [依存プレゼンスチェックによる遅延セットアップ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1121_make-deps-presence-guard)
- [Makefile テストターゲット(test / test-feature / test-local)による実行の標準化](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1198_makefile-test-targets)
- [pytest の rootdir 解決とマーカー設定の一元化](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1202_pytest-rootdir-marker-config)
- [README をクイックスタート型フロントページに再構成する](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1149_readme-quick-start-restructure)
