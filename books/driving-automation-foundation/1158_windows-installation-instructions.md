---
title: "クロスプラットフォーム環境構築 (Windows での pip / CUDA torch 導入)"
---

# クロスプラットフォーム環境構築 (Windows での pip / CUDA torch 導入)

## ひとことで言うと
機械学習プロジェクトのセットアップ手順は、しばしば Linux/macOS を暗黙の前提にして書かれ、`make` などのツールに依存します。ところが `make` は Windows に標準で入っておらず、Windows ユーザーは手順どおりに進められません。そこで、`make` を使わず Python 標準の `pip install -r requirements.txt` だけで環境を作れる手順を併記し、さらに GPU 計算用の CUDA 版 PyTorch を入れるための `--extra-index-url` 指定を明示しておくのが定石です。これは研究手法ではなく、再現性(同じ手順で誰でも同じ環境を作れること)を OS 横断で確保するための運用知識です。

## 直感的な理解
レシピ本に「オーブンで200度20分」とだけ書いてあると、オーブンを持っている家庭は作れますが、コンロしかない家庭は詰まります。良いレシピは「オーブンがない場合はフライパンで弱火10分」という代替手順も書きます。ソフトウェアのセットアップ手順も同じで、特定のOS・特定のツール(オーブン=`make`)を前提にすると、それを持たない環境(Windows)の人が再現できません。Python には全プラットフォーム共通の道具(pip=どの家庭にもあるコンロ)があるので、それを使った代替手順を併記すれば誰でも再現できます。

もう一段の難しさが「GPU 版 PyTorch」です。GPU で計算する版の PyTorch は、Python の標準配布サイト(PyPI)ではなく、PyTorch が運営する専用の配布サーバーに置かれています。pip は普段 PyPI しか見ないので、「標準の場所に加えてこの特別な場所も見て」と教えないと GPU 版を見つけられず、勝手に CPU 版を入れてしまいます。これを教えるのが `--extra-index-url` という指定です。スーパー(PyPI)に売っていない特殊な食材を、専門店(PyTorch のサーバー)から取り寄せる、というイメージです。なぜこの区別が研究者にとって死活問題かというと、CPU 版を間違って入れてしまうと「動くけれど学習が桁違いに遅い」状態になり、しかもエラーが出ないので気づきにくいからです。GPU が認識されているかは学習を始める前に必ず確認すべき項目になります。

## 基礎: 前提となる概念
make と Makefile。`make` はビルドやセットアップの手順を `Makefile` という台本に書いておき、`make setup` のように1コマンドで実行する道具です。Unix 系(Linux・macOS)では標準的に使えますが、Windows には標準では入っていません。Windows で `make setup` と打つと「そんなコマンドは知らない」と止まります。Git Bash や WSL(Windows Subsystem for Linux、Windows 上で Linux 環境を動かす機能)を入れれば使えますが、それ自体が追加のセットアップで初心者の障壁になります。

pip と仮想環境。pip は Python のパッケージ(ライブラリ)を導入する標準ツールで、Python と一緒にどのOSにも入ります。`pip install パッケージ名` で1つ入れ、`pip install -r requirements.txt` でファイルに列挙された複数を一括で入れます。`-r` は "requirements file から読む" の意味です。プロジェクトごとにパッケージを隔離するため、仮想環境(`python -m venv .venv`)を作ってその中に入れるのが定石です。仮想環境を使わないとシステム全体の Python に直接入り、別プロジェクトと依存が衝突します。

requirements.txt。プロジェクトが必要とするパッケージとそのバージョン条件を1行ずつ列挙したテキストファイルです。例えば `torch==2.3.0`、`numpy>=1.24` のように書きます(バージョン指定の文法は PEP 440 で定義されています)。これがあれば、誰の環境でも同じパッケージ群を再現できます。ファイル先頭に `--extra-index-url ...` を書いておくと、その指定をファイル内に埋め込めます(コマンドに毎回付けなくてよい)。

PyTorch と CPU版 / CUDA版。PyTorch(torch)は深層学習の計算を担う中心ライブラリです。CPU だけで計算する CPU 版と、NVIDIA GPU を使って並列高速計算する CUDA 版があります。CUDA は NVIDIA GPU 上で汎用計算を行うための基盤で、`cu118`・`cu121`・`cu124` などの数字は対応する CUDA のバージョン(11.8、12.1、12.4)を表します。GPU を持つ人は CUDA 版を入れると桁違いに速くなります。重要なのは、PyTorch の CUDA 版 wheel には CUDA ランタイムが同梱されているため、ホストに別途 CUDA Toolkit を入れなくても動く点です(ただし NVIDIA のドライバは必要)。

index URL と extra-index URL。pip がパッケージを探しに行く先(パッケージインデックス)です。既定では PyPI を見ます。`--index-url` は探す場所を「丸ごと差し替える」、`--extra-index-url` は「既定の PyPI に加えて追加で見る」指定です。これらのインデックスは PEP 503(Simple Repository API)という共通仕様に従っており、PyTorch の CUDA wheel もこの仕様のサーバーで配布されています。

wheel。`.whl` という拡張子の、ビルド済みパッケージ配布形式(PEP 427)です。ソースからコンパイルする必要がなく、ダウンロードして展開するだけで入るため高速・確実です。PyTorch の CUDA 版は、CUDA バージョンごと・OSごと・Python バージョンごとに別々の wheel が用意されています。wheel のファイル名には対応プラットフォームを示すタグ(例 `cp311-cp311-win_amd64`)が含まれ、pip は自分の環境に合うものを自動選択します。

## 仕組みを詳しく
OS 依存の `make` 手順に対し、Python 共通の pip 手順を併記する典型は次のようになります。

```bash
# どのOSでも共通(ソース取得)
git clone <repository-url>
cd <project-dir>

# 仮想環境を作って有効化(Windows は .venv\Scripts\activate)
python -m venv .venv
source .venv/bin/activate

# CPU 版 torch を含めて一括導入(PyPI から)
pip install -r requirements.txt

# GPU(CUDA)版 torch を入れたい場合は追加インデックスを指定
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
# cu118 の部分を cu121 / cu124 などに変えれば別の CUDA バージョンに合わせられる
```

各ステップを分解します。`git clone` でソース一式を手元へコピーし、`cd` でそのフォルダに入ります。`python -m venv` でプロジェクト専用の隔離環境を作り、有効化します。`pip install -r requirements.txt` は requirements.txt に列挙されたパッケージをまとめて入れ、何も追加しなければ PyPI から CPU 版 torch が入ります。CUDA 版が必要なときだけ `--extra-index-url https://download.pytorch.org/whl/cu118` を付けます。これで pip は「PyPI に加えて PyTorch の CUDA 11.8 向け配布チャネルも探す」ようになり、GPU で動く torch が選ばれます。

導入後の確認も手順に含めるのが望ましいです。

```python
import torch
print(torch.__version__)          # 例: 2.3.0+cu118 なら CUDA 版、2.3.0 だけなら CPU 版
print(torch.cuda.is_available())  # True なら GPU が使える
```

ここで重要な落とし穴があります。`--extra-index-url` は複数のインデックスを「両方」探すため、同じパッケージ名が両方に存在するとき、pip がどちらを選ぶかは状況依存になり得ます(いわゆる依存解決の曖昧さ・dependency confusion の温床)。PyTorch 公式は近年、曖昧さを避けるために `--extra-index-url` ではなく `--index-url`(探す場所を PyTorch チャネルに丸ごと固定)を推奨する場面が増えています。

```bash
# 曖昧さを避けたい場合は index を固定する書き方
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

before/after で運用上の効果をまとめます。before では「`make setup` 前提」のため Windows ユーザーは最初の1コマンドで詰まり、CUDA 版の入れ方も不明でした。after では、Windows を含むどのOSでも `pip install -r requirements.txt` で進められ、GPU が必要なら追加インデックス指定で CUDA 版を入れられる、という再現可能な手順が明示されます。コードの挙動は何も変わらず、変わるのは「誰が・どの環境で・どう環境構築するか」だけです。

なぜ CUDA wheel が PyPI に置かれないのか。PyPI には1ファイルあたり・プロジェクトあたりのサイズ制限があり、CUDA バージョン × OS × Python バージョンの組み合わせで巨大な wheel 群(1つ数百MB〜2GB級)を全部 PyPI に置くのは現実的でないこと、また「どの CUDA 版を入れるか」をユーザーが明示的に選ぶ必要があることから、PyTorch は自前の配布チャネル(`download.pytorch.org/whl/<cuda-tag>`)を運用しています。そのため pip にそのチャネルを教えるステップが不可欠になります。近年は CPU 版とは別に、PyPI 上にも一部の CUDA 版が `nvidia-*` ランタイムパッケージ群への依存付きで置かれるようになり状況は改善しつつありますが、CUDA バージョンを明示したい場合は依然として専用チャネル指定が確実です。

## 手法の系譜と主要論文
これは学術的な研究手法ではなく、ソフトウェア配布・パッケージングのエンジニアリング知識なので、引用すべき自然科学の論文はありません。代わりに依拠する一次情報と標準化の系譜を示します。

Python のパッケージインデックス仕様は PEP 503(Simple Repository API)で標準化され、PyPI も PyTorch の CUDA チャネルもこの共通インターフェースに従います。バージョン指定の文法(`==`, `>=`, `~=` など)は PEP 440(Version Specifiers)で定義されます。配布形式である wheel は PEP 427(Wheel Binary Package Format)で定められ、ソースビルド不要の高速・再現的な導入を可能にしました。`--index-url` と `--extra-index-url` の挙動差、および後者がもたらす依存解決の曖昧さは、Alex Birsan が2021年に公開した "Dependency Confusion" の調査(内部用パッケージ名と同名の悪意あるパッケージを公開インデックスに置くことで、複数のインデックスを見る設定の組織に意図せず取り込ませる攻撃)で広く知られるようになり、pip 公式ドキュメントと Python Packaging User Guide で繰り返し注意喚起されています。

クロスプラットフォーム再現性の流れとしては、(1) OS固有スクリプト(Makefile, シェルスクリプト)依存 → (2) Python 共通の pip + requirements.txt → (3) ロックファイルによる完全固定(pip-tools の `requirements.lock`, Poetry の `poetry.lock`, PDM, そして近年の uv)→ (4) コンテナ化(Docker)で OS ごと固定、という成熟の系譜があります。Windows 向け手順の併記は、この (1)→(2) の橋渡しに当たり、最も低コストに参入障壁を下げる一手です。

## 論文の実験結果(定量データ)
研究手法ではないため定量的な実験結果はありませんが、運用上の「効果」を測る代理指標で整理できます。

再現性の指標として、「クリーンな環境で手順どおり実行したときの成功率(green build rate)」があります。OS固有ツール前提の手順では、対象外OS(ここでは Windows)での成功率が実質0%(最初のコマンドで失敗)になります。pip 手順を併記するとこれが大きく改善します。さらにバージョンを固定しない `requirements.txt`(`torch` とだけ書く)では、実行のたびに最新版が入り、ある日突然依存が壊れる(いわゆる再現性の崩壊)。バージョンを固定(`torch==2.3.0`)し、最終的にロックファイル(依存の依存まで含めた全パッケージのバージョンとハッシュを固定したファイル)まで導入すると、時間が経っても同じ環境を再現できる確率がほぼ100%に近づきます。ハッシュ固定(`--require-hashes`)まで行えば、配布元が差し替えられても検知できます。

依存解決の曖昧さに関しては、`--extra-index-url` を使うと、悪意あるパッケージや同名パッケージが追加インデックスに存在した場合に意図しないものが入るリスク(dependency confusion)が報告されています。Birsan の調査では、この手法で複数の大手テック企業の内部システムにコードを実行させられたことが実証されました。これは指標というより脆弱性事例の系統で、`--index-url` で探索先を限定するか、`--require-hashes` を併用することで回避できる、というのが教訓です。

## メリット・トレードオフ・限界
メリット。Windows を含む全OSのユーザーが標準ツール pip だけでセットアップでき、参入障壁が下がります。CPU 版・CUDA 版の入れ分けが手順として明示され、GPU の有無に応じた環境を再現できます。変更が手順書(ドキュメント)だけで完結するため、コード挙動への影響がゼロで安全です。

トレードオフと限界。セットアップ経路が「make 系」と「pip 系」の2系統になると、片方を更新したらもう片方も合わせる手間(手順の二重管理)が生じます。`requirements.txt` と `Makefile` が指すバージョンがずれると、OS によって入る依存が食い違い、再現性がかえって損なわれます。`--extra-index-url` には前述の依存解決の曖昧さがあり、よりロバストには `--index-url` での固定やロックファイルが望まれます。GPU 環境特有の限界として、ホストの NVIDIA ドライバが古いと新しい CUDA tag の torch が動かないことがあり、「ドライバが対応する最大 CUDA バージョン」を確認してから tag を選ぶ必要があります。研究上というより運用上の未解決課題は、「人手で書いた手順書」が実態とずれること(ドキュメントのドリフト)で、これを根本解決するにはコンテナ化(Docker)や CI でのクリーンインストール検証を入れて、手順そのものを機械的に検証可能にするのが現代的な答えです。

## 発展トピック・研究の最前線
環境構築の最前線は依存固定の高速化と完全再現にあります。uv(Rust 製の超高速 pip 互換ツール)はロックファイルと解決を桁違いに速くし(報告では pip-tools 比で10〜100倍の解決速度)、近年急速に普及しています。Poetry / PDM はプロジェクト単位の依存と仮想環境を統合管理します。さらに OS レベルの差異まで吸収するには Docker / devcontainer でベースイメージごと固定し、CI(継続的インテグレーション)上で「クリーンな環境で手順が通るか」を自動検証します。GPU を含む環境では、ホストの NVIDIA ドライバ・CUDA ランタイム・torch の CUDA tag の三者の整合が依然として悩みどころで、conda の cudatoolkit、Docker の `nvidia/cuda` ベースイメージ、torch の同梱 CUDA ランタイムのどれを正にするかが設計判断になります。最終的に目指すのは「手順書を読まなくても、1つのコマンドかコンテナで誰でも同じGPU環境が立ち上がる」状態です。研究の現場では、実験の再現性危機(同じ論文の結果が環境差で再現しない)への対策として、環境を論文成果物の一部として固定・公開する文化(ロックファイルや Dockerfile の同梱)が定着しつつあります。

## さらに学ぶための関連トピック
- [データパイプラインのドキュメント整理とデッドコード除去](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1119_l2d-nvidia-dataset-docs-cleanup)
- [bf16 混合精度学習 (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0552_bf16-amp-training)
- [デュアルモードデータセット (ローカル直読み と 事前抽出シャードの統一)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0806_dual-mode-dataset)
