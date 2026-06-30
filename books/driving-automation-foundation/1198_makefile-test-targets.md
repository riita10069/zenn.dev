---
title: "Makefile テストターゲット(test / test-feature / test-local)による実行の標準化"
---

# Makefile テストターゲット(test / test-feature / test-local)による実行の標準化

## ひとことで言うと

プロジェクトのテストを「コマンド一発」で走らせるためのショートカット集の話です。`make test` は CI(自動テスト)と同じ範囲のテスト、`make test-feature` は描画・地図のような追加依存が必要な特定機能だけのテスト、`make test-local` は開発者のパソコンで動かせる全範囲のテスト、というように、目的別の実行口を Makefile に用意します。狙いは「長いコマンドを覚えなくていい」ことと、それ以上に「手元の実行と CI の実行を一致させて、片方だけ落ちる事故を防ぐ」ことです。

## 直感的な理解

ソフトウェア開発では、コードを書いたら「壊していないか」をテストで確認します。テストは pytest のような専用ツールで動かしますが、そのコマンドは長くて覚えにくく、しかも引数を少し間違えると違う範囲を走らせてしまいます。たとえば「テストの置き場所」「詳細表示の有無」「どのフォルダを起点にするか」を毎回手で指定するのは負担です。

ここで、もっと深刻な問題があります。開発者ごとに微妙に違うコマンドを使うと、「Aさんの手元では通ったテストが、Bさんの手元では落ちる」「手元では通ったのに CI では落ちる」という食い違いが生まれます。テストは「全員が同じ範囲を同じ条件で走らせる」ことで初めて信頼できる安全網になります。コマンドがバラバラだと、その安全網に穴が空きます。

そこで Makefile を使います。Makefile は「コマンドのあだ名(エイリアス)を定義しておくファイル」だと思ってください。`make test` と打つだけで、裏で長いテストコマンドが実行されます。重要なのは、CI もまったく同じ `make test` を呼ぶことです。こうすると「手元で `make test` が通れば CI でも通る」という安心が成立します。テストの種類ごとにターゲット(test / test-feature / test-local)を分けるのは、「普段は軽く速く、必要なときは追加依存込みで全部」という使い分けを、コマンド名だけで表現するためです。

## 基礎: 前提となる概念

用語を噛み砕きます。

- Make / Makefile: もともと C 言語のビルド自動化ツールとして 1976 年に生まれた仕組み。`make <ターゲット名>` と打つと、Makefile に書かれた一連のコマンドが実行されます。今ではビルドに限らず「定型コマンドの入り口」として広く使われます。ターゲットは「ラベル: 実行するコマンド」の形で書きます。
- ターゲット(target): `make test` の `test` の部分。実行したい作業の名前です。
- pytest: Python のテスト実行ツール。`python -m pytest <フォルダ> -v` のように使い、`-v`(verbose)は結果を 1 件ずつ詳しく表示する指定です。
- CI(Continuous Integration、継続的インテグレーション): コードを送るたびに自動でビルドとテストを動かし、合否を判定する仕組み(GitHub Actions など)。手元と CI でテスト範囲が一致していることが信頼の前提です。
- 依存(dependency): あるテストを動かすのに必要な追加ライブラリ。たとえば描画テストには matplotlib(グラフ・画像を描く)や地図ライブラリが要ります。これらが入っていないとそのテストは失敗します。
- stale(ステイル、賞味期限切れ): 定義が古くなって実態とずれた状態。テストの置き場所を変えたのに Makefile を直し忘れると、`make test` が存在しないフォルダを指して失敗する、といった形で起きます。

## 仕組みを詳しく

目的別の 3 ターゲットの典型的な中身を、一般化した形で示します。`feature_x` は「追加依存が必要な特定機能(描画・地図・可視化など)」を表す仮の名前だと思ってください。

```make
test:        # CI と同じ範囲のテスト(軽い・依存が少ない)
	python -m pytest tests -v

test-feature:  # 特定機能だけ。matplotlib など追加ライブラリが必要
	cd src && python -m pytest feature_x/rendering -v

test-local:  # 開発機で動かせる全範囲(標準 + 特定機能)
	cd src && python -m pytest tests feature_x/rendering -v
```

それぞれの違いを分解します。

- `make test`: 標準のテストフォルダだけを対象にします。これは CI とまったく同じ選び方です。CI が動かすものと同じ範囲を手元でも動かせるので、「自分の環境では通ったのに CI で落ちた」という事故を減らせます。これが 3 つの中で最も重要なターゲットで、「手元と CI の一致点」を担います。
- `make test-feature`: いったん `cd src` でソースのルートに移動してから、特定機能(たとえば描画・地図のような追加依存を伴う機能)のテストだけを動かします。なぜ `cd` するかというと、そのモジュールが `feature_x.something` のような形で内部ファイルを絶対 import するため、ソースのルートを起点(import の基準パス)にしないと「そんなモジュールは無い」と import エラー(`ModuleNotFoundError`)になるからです(これは [pytest の rootdir 解決とマーカー設定の一元化](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1202_pytest-rootdir-marker-config) で扱う rootdir と import 解決の話に関係します)。このテストには描画系・可視化系の追加ライブラリが要ります。
- `make test-local`: 標準テストと特定機能テストの両方を一度に動かします。開発者が「自分のパソコンで動かせるものは全部チェックしたい」というときのターゲットです。CI では重すぎる/外部依存が多すぎるものを、ローカルだけで網羅的に回す用途です。

なぜ 3 つに分けるのか。テストには「速くて依存が少ない=普段から回したいもの」と「遅い・追加依存が要る=必要なときだけ回したいもの」が混在します。全部を 1 つの `make test` に詰めると、普段の確認が遅くなり、依存が足りない環境では失敗します。逆に全部を分離しすぎると使い分けが面倒になります。「CI と一致する軽い既定(test)」「特定機能だけ(test-feature)」「ローカル全部(test-local)」という 3 段は、この粒度のバランスを取った典型的な構成です。

### 実行結果の例(deselected の意味)

実際にこの種のターゲットを回すと、次のような件数が出ます(値は構成例)。

- `make test` — 137 件成功、7 件 deselected。CI と同じ選び方。
- `make test-feature` — 11 件成功。
- `make test-local` — 148 件成功、7 件 deselected。

「deselected」とは、pytest が「今回は対象にしない」と判断して実行しなかったテストのことです。たとえば重い統合テスト(integration マーカー付き)は普段は除外する、という設定(addopts のマーカー除外。[pytest の rootdir 解決とマーカー設定の一元化](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1202_pytest-rootdir-marker-config) 参照)があると、それらが deselected として外れます。`make test` と `make test-local` で deselected が同じ 7 件なのは、両者が同じ除外ポリシーを使っている証拠です。逆に件数がズレていたら、片方だけ違う除外条件で走っているサインで、設定が二重所有になっていないか疑うべきです。

### CI との結線

CI 設定が `make setup` → `make lint` → `make typecheck` → `make test` のように同じ make ターゲットを呼ぶ構成にすると、手元と CI が完全に同じコマンドを通ります。CI のスクリプトと開発者の手順が「同じ Makefile を見る」ことで、ここでも single source of truth(信頼できる唯一の出どころ)が成立します。

## 手法の系譜と主要論文

これはソフトウェア開発の運用(インフラ)の話で、機械学習や自動運転の研究論文が直接対象とするテーマではありません。背景にある確立された文献を系譜で示します。

- GNU Make(Stallman らによる実装、1988 年〜): Make 自体は 1976 年 Stuart Feldman による発明で、ビルド自動化の起点。GNU Make Manual がデファクトの仕様です。「ターゲットと、それを作る手順を宣言的に書く」という発想が、本トピックの土台です。
- Fowler "Continuous Integration"(martinfowler.com, 2006): CI の概念を広めた古典的記事。「コードを頻繁に統合し、毎回自動でビルドとテストを走らせることで統合の苦痛を小さく保つ」。統合を後回しにすると衝突が積み重なって手戻りが爆発する、という問題への処方箋です。
- Duvall et al. "Continuous Integration: Improving Software Quality and Reducing Risk"(Addison-Wesley, 2007): CI を体系化した書籍。「ローカルのビルドが CI のビルドと一致すること(self-testing build)」を重視しており、`make test` が CI と同じ範囲を回す設計の直接の根拠です。
- Humble & Farley "Continuous Delivery"(Addison-Wesley, 2010): CI を本番リリースまで延長する考え方。「ビルド・テストのスクリプト化と、どの環境でも同じコマンドで再現できること」を強調しており、Makefile による標準化の延長線上にあります。

これらは「自動運転の手法」ではなく「開発をどう回すか」の文献である点に注意してください。

## 論文の実験結果(定量データ)

研究論文の定量実験は存在しないテーマですが、CI/自動テスト運用の効果を測った産業界の調査があります。Forsgren, Humble, Kim "Accelerate"(2018、DORA の研究を書籍化)では、自動テストと CI を高度に運用する組織(elite performer)が、低成績の組織に比べてデプロイ頻度で 200 倍以上、変更失敗率で 1/7 以下、といった大差を示すと報告しています。直接 Makefile ターゲットを測ったものではありませんが、「テストを摩擦なく・一貫した方法で走らせること」がチーム成果に効く、という背景データです。

ローカルな効果としては、`make test` を CI と一致させることで「CI でしか再現しない失敗」が構造的に減ります。たとえばテストフォルダの指定ミスや、deselected の食い違いに起因する失敗は、全員が同じ make ターゲットを使えば原理的に起きません。件数では測りにくいものの、デバッグ往復(手元で再現できず CI ログだけ見て推測する作業)の削減として効きます。

## メリット・トレードオフ・限界

メリット:
- 長いコマンドを覚えなくても `make test` だけでテストできる。引数の打ち間違いによる誤った範囲の実行が減る。
- CI と同じ範囲を `make test` に固定したので、手元と CI の食い違いが減る。
- 追加依存が要る特定機能テストを `test-feature` として分離でき、普段の `make test` を軽く速く保てる。

トレードオフと限界:
- Makefile という追加ファイルを保守する必要がある。テストの置き場所を変えたら Makefile も直さないと、また stale(古くなった)状態になり、存在しないフォルダを指して失敗する。
- 特定機能テスト(test-feature)には追加ライブラリが要り、入れ忘れると失敗する。これは「実行前に依存の有無を確認して自動セットアップする仕組み(presence guard)」で改善できます([依存プレゼンスチェックによる遅延セットアップ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1121_make-deps-presence-guard) 参照)。
- Make は Windows ネイティブでは標準で入っておらず、`cd` を含むレシピのシェル方言差などクロスプラットフォームの落とし穴がある。プロジェクトの対象環境を前提に書く必要があります。
- ターゲットを増やしすぎると、どれを使えばいいか分からなくなる。3 段程度に保ち、命名で意図を伝えるのが現実的です。

## 発展トピック・研究の最前線

- タスクランナーの進化: Make の後継として just、Task(Taskfile)、あるいは言語固有のタスク定義(npm scripts、tox、nox、hatch)が広がっています。いずれも「定型コマンドの入り口を一元化する」という本トピックの目的を、よりクロスプラットフォーム・宣言的に実現する方向です。
- スイート選択のパラメータ化: ターゲットを 3 つ並べる代わりに、`make test SUITE=all|map|integration` のように 1 つのターゲットに引数を渡してスイートを切り替える設計があります。ターゲットの増殖を抑えつつ柔軟性を持たせる発展形です([Makefile によるテストスイートの選択的実行 (SUITE パラメータ方式)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1123_makefile-test-suite-targets) 参照)。
- テスト影響度分析(Test Impact Analysis): 変更したコードに関係するテストだけを選んで走らせ、CI 時間を削る技術。大規模リポジトリで実用化が進んでおり、「全部走らせる」と「変更分だけ走らせる」のトレードオフを自動で取る研究・実装が活発です。
- 依存の自己解決: テスト実行前に必要なライブラリの有無を検査し、足りなければ自動でインストールする仕組み。環境差による失敗を吸収する方向で、本トピックのトレードオフを埋める発展です。

## さらに学ぶための関連トピック

- [Makefile によるテストスイートの選択的実行 (SUITE パラメータ方式)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1123_makefile-test-suite-targets)
- [依存プレゼンスチェックによる遅延セットアップ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1121_make-deps-presence-guard)
- [プラットフォーム中立な requirements.txt と torch wheel チャネル選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1137_platform-neutral-requirements-txt)
- [pytest の rootdir 解決とマーカー設定の一元化](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1202_pytest-rootdir-marker-config)
- [巨大テストファイルのコンポーネント別モジュール分割](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1135_per-package-test-module-split)
- [README をクイックスタート型フロントページに再構成する](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1149_readme-quick-start-restructure)
