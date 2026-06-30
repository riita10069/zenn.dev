---
title: "pip の extra-index-url で PyTorch CUDA ビルドを取得する"
---

# pip の extra-index-url で PyTorch CUDA ビルドを取得する

## ひとことで言うと

`torch==2.4.1+cu118` のような「GPU(CUDA)対応版の PyTorch」は、pip が普段見に行く標準の配布サイト(PyPI)には置かれていません。そこで「この場所も探しに行ってね」と追加の配布サイトを `--extra-index-url` で教えてあげると、pip がその GPU 版を見つけてインストールできるようになります。これにより、リポジトリを取得した人が `pip install -r requirements.txt` と打つだけで、CPU 用と GPU 用が混在する依存関係を一括で解決できます。

## 直感的な理解

本屋で本を探す状況を想像してください。普段は近所の大きな書店(PyPI)に行けばたいていの本が見つかります。ところが、ある専門書(GPU 版 PyTorch)はその書店には置いておらず、出版社が直営する別の店(PyTorch 公式の配布サイト)にしかありません。あなたが「その本ください」と近所の書店で言っても「うちには無い」と断られます。

解決策は、店員に「近所の書店だけでなく、この直営店も探してみて」と伝えておくことです。`--extra-index-url` はまさにこの一言です。普段の書店(PyPI)はそのまま使いつつ、追加の探索先(PyTorch の配布サイト)を 1 つ足す。これで、普通の本(numpy や pytest)は近所の書店から、専門書(GPU 版 torch)は直営店から、と自動で振り分けて取ってきてくれます。

なぜこんな仕組みが必要かというと、GPU 版の PyTorch は CPU 版とは中身(コンパイルされたバイナリ)が根本的に違い、しかも対応する CUDA のバージョンごとに別々のファイルが必要だからです。これらを全部 PyPI に載せるとサイズが膨大になるため、PyTorch は自前の配布サイトを運営しています。

## 基礎: 前提となる概念

用語を順にほぐします。

- **pip**: Python のライブラリ(部品集)をインターネットからダウンロードして入れるツール。`pip install numpy` のように使います。
- **PyTorch (torch)**: 機械学習・ディープラーニングの計算を行う中心的なライブラリ。深層学習モデルの多くがこの上で動きます。
- **CUDA**: NVIDIA 製 GPU 上で計算を高速に走らせるための仕組み(プラットフォームと API の総称)。CPU だけで動かす PyTorch と、GPU を使う PyTorch では、内部のバイナリが違います。GPU 版は CUDA のバージョン(11.8、12.1 など)ごとに別ビルドが用意されます。
- **PyPI (Python Package Index)**: pip が標準で見に行く、世界共通のライブラリ置き場。`pip install torch` と打つと、ここから取ってきます。
- **wheel(.whl)**: ライブラリを「インストール済みに近い形」に固めたパッケージ形式。`torch-2.4.1+cu118-cp312-cp312-linux_x86_64.whl` のようなファイル名で、「torch のバージョン 2.4.1、CUDA 11.8 向け、Python 3.12 向け、Linux 64bit 向け」と中身が決まっています。pip はこの whl を探して落としてきます。
- **index(インデックス)**: pip が whl を探しに行く配布リポジトリ。PyPI はその代表で、PEP 503 が定める「Simple Repository API」という共通の形式に従ったページ群です。`--index-url` で指定できます。

問題の核心は、`torch==2.4.1+cu118` という指定の `+cu118` の部分にあります。これは「CUDA 11.8 向けにビルドされた torch」を意味する **ローカルバージョン識別子(local version identifier)** で、バージョン番号 `2.4.1` の後ろに「どの環境向けか」を示すタグが付いた形です。このような `+cuXXX` 付きビルドは PyPI には置かれておらず、PyTorch が自前で運営する `https://download.pytorch.org/whl/...` にあります。pip は標準では PyPI しか見に行かないため、PyPI だけを探すと「そんなバージョンは無い」と言って失敗します。

典型的な失敗メッセージはこうなります。

```
ERROR: Could not find a version that satisfies the requirement torch==2.4.1+cu118
       (探索先: PyPI のみ → +cu118 版が見つからない)
```

「GPU 版 torch を要求しているのに、pip にその在処を教えていなかった」状態です。

## 仕組みを詳しく

### --extra-index-url が何をするか

`--extra-index-url <URL>` は pip に「標準の PyPI に加えて、この URL も whl の探索先に追加して」と指示するオプションです。似たオプションに `--index-url`(主たる index を置き換える)がありますが、`--extra-index-url` は **追加** なので PyPI も引き続き使えます。だから numpy や pytest など普通のライブラリは PyPI から、torch だけは PyTorch の配布サイトから、という使い分けが自動で起きます。

`requirements.txt` のファイル先頭にこの 1 行を書くのが定石です。

```
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.4.1+cu118
numpy==2.4.6
pytest==9.0.3
```

URL 末尾の `cu118` が「CUDA 11.8 向けの whl が置いてある棚」を指します。CUDA 12.1 なら `cu121`、CUDA 12.4 なら `cu124`、CPU 専用なら `cpu` というように、棚ごとに URL が分かれています。マシンに入っている CUDA(正確には NVIDIA ドライバが対応する CUDA バージョン)に合わせて棚を選びます。

### before / after で見る pip の動き

before(`--extra-index-url` を追加する前):

```
$ pip install -r requirements.txt
探索先: PyPI のみ
ERROR: Could not find a version that satisfies the requirement torch==2.4.1+cu118
```

after(追加後):

```
$ pip install -r requirements.txt
探索先: PyPI + https://download.pytorch.org/whl/cu118
→ torch-2.4.1+cu118-cp312-cp312-linux_x86_64.whl を発見、ダウンロード、インストール成功
```

### ステップ分解

1. pip が `requirements.txt` を読み、先頭の `--extra-index-url` を探索先リストに加える。
2. 各ライブラリについて、登録された全 index(PyPI と extra-index-url の両方)から条件に合う whl を探す。
3. `numpy` や `pytest` などは PyPI で見つかる。
4. `torch==2.4.1+cu118` は extra-index-url(PyTorch の配布サイト)で見つかる。
5. 依存の依存(torch が内部で必要とする別ライブラリ)も芋づる式に解決して、まとめてインストール。利用者の追加手作業はゼロ。

### 複数 index がある場合の解決ルール

pip は複数の index を「フラットに 1 つの集合」として扱います。つまり「PyPI を先に、PyTorch を後に」という優先順位はなく、全 index からバージョン制約を満たす候補を集め、その中から最も適切なもの(通常は最新バージョン)を選びます。これは便利な反面、後述の「依存混乱(dependency confusion)」というリスクの源にもなります。

## 手法の系譜と主要論文

これは研究手法ではなく、Python パッケージング(配布)の実務知識です。一次情報は学術論文ではなく公式仕様・ドキュメントなので、それらを系譜として挙げます。

- **PEP 440 — Version Identification and Dependency Specification**: `2.4.1+cu118` の `+cu118` 部分を「local version identifier(ローカルバージョン識別子)」と定義しています。「同じ公開バージョンだが、ビルド環境(ここでは CUDA バージョン)が違うものを区別するためのタグ」という位置づけで、PyTorch はこれを使って CPU/CUDA 別の同一バージョンを並立させています。仕様上、ローカル識別子付きバージョンは公開 index(PyPI)へのアップロードが制限されるため、別の配布サイトが必要になる、という背景もここに根拠があります。
- **PEP 503 — Simple Repository API**: pip が index を読むときの HTML 形式を定めた規格。`--extra-index-url` で指定する URL は、この Simple Repository API に従ったページであることが前提です。PyTorch の配布サイトもこの形式に従っているため、pip がそのまま読めます。
- **pip 公式ドキュメント(`--index-url` / `--extra-index-url`)**: `--extra-index-url` を「primary index に加えて追加で参照する index」と定義。複数 index がある場合に同名パッケージはバージョン優先で解決される旨、および依存混乱のリスクを併記しています。本手法の直接の根拠です。
- **PyTorch 公式インストールガイド("Get Started")**: GPU 版をインストールする際に `--index-url https://download.pytorch.org/whl/cu118` のような指定を案内しています。PyTorch 側が CUDA 別の whl リポジトリを運営しているという事実の一次情報です。

歴史的には、pip は当初 PyPI 一択でしたが、企業の社内パッケージや GPU ビルドのような「PyPI に載らない/載せたくない」配布物のニーズから、複数 index を扱う `--extra-index-url` が整備されました。さらに後年、依存混乱攻撃の発覚を受けて、より安全な `--index-url` への一本化(PyTorch も近年は extra ではなく index による単一インデックス運用を推奨)が議論されるようになっています。

## 論文の実験結果(定量データ)

定量的な「実験」がある分野ではありませんが、関連する実務上の事実を数値とともに整理します。

- **whl のサイズと CUDA 別ビルドの必要性**: GPU 版 PyTorch の whl は CPU 版より大幅に大きく、CUDA ランタイムや cuDNN を同梱する版ではおよそ数百 MB から 2 GB 超に達します(版により差があり、報告による)。CUDA 11.8、12.1、12.4 といったバージョンごとに別ビルドが必要なため、これらをすべて PyPI に載せると配布サイズが膨大になります。これが PyTorch が独立した配布サイトを持つ実務的理由で、`--extra-index-url` が要るそもそもの背景です。
- **依存混乱攻撃の実例(Birsan, 2021)**: セキュリティ研究者 Alex Birsan は 2021 年、社内向けの内部パッケージ名と同じ名前を公開 index に置くことで、複数 index を混ぜている環境の pip がうっかり公開側の悪意あるパッケージを取得してしまう「dependency confusion」攻撃を実証し、Apple・Microsoft・PayPal を含む 35 社以上のシステムに侵入できたと報告しました(バグバウンティで合計 13 万ドル超を獲得)。これは「複数 index を混ぜると、同名パッケージが両方にあった場合に意図しない方を掴みうる」という `--extra-index-url` の構造的リスクを示す代表事例で、信頼できる URL に限定すべき理由の定量的裏づけです。

## メリット・トレードオフ・限界

メリット

- 取得した人が `pip install -r requirements.txt` の 1 コマンドで、GPU 版 torch まで含めて環境を作れる。新規参加者がつまずかない。
- PyPI の通常ライブラリと PyTorch の特殊ビルドを、設定 1 行で両立できる。`--index-url` のように index 全体を置き換えるわけではないので、副作用が小さい。
- CUDA バージョンを変えるときは URL 末尾の `cuXXX` を差し替えるだけで対応でき、他の行をいじらずに使い回せる。

トレードオフ・限界

- **環境依存**: `cu118` は「CUDA 11.8 を前提とするビルド」を意味する。マシンの NVIDIA ドライバが対応する CUDA とずれていると、インストールはできても GPU で動かない、あるいは遅い、という不一致が起こりうる。ドライバ・CUDA・torch の三者のバージョン整合が前提。
- **依存混乱(dependency confusion)のリスク**: 複数 index を混ぜると、同名パッケージが両方に存在した場合に pip がどちらを選ぶか予期しにくい。悪意ある同名パッケージを掴む供給網攻撃の温床にもなりうるため、信頼できる URL(PyTorch 公式など)に限定すること。近年は `--extra-index-url` より、配布元を 1 つに絞れる `--index-url` の方が安全とされ、PyTorch 公式も単一インデックス運用を案内するようになっている。
- **完全な再現性には不十分なことがある**: OS・Python・CUDA・GPU の組み合わせ次第で取得される whl が変わる。厳密な再現が要るなら、ハッシュ固定(`--require-hashes`)やコンテナ化(Docker など)まで踏み込む必要がある。`--extra-index-url` は「正しい whl の在処を教える」役割であって、「常に同一の whl を保証する」役割ではない。

## 発展トピック・研究の最前線

- **ロックファイルと厳密再現**: pip の `--require-hashes` や、`pip-tools`(`pip-compile` でロックファイルを生成)、Poetry・PDM・uv といった新しいツールは、取得する whl をハッシュ単位で固定し、複数 index が絡んでも再現性を担保します。GPU ビルドの取得と再現性を両立させる現実解として普及が進んでいます。
- **uv による高速・安全な解決**: 近年登場した uv(Rust 製の Python パッケージ管理ツール)は、index の優先順位を明示的に指定できる仕組み(`index-strategy` や index の所属固定)を備え、依存混乱を構造的に防ぎつつ GPU ビルドの取得を扱えます。`--extra-index-url` のフラットな探索が抱えていた曖昧さへの一つの答えです。
- **コンテナによる完全固定**: 学習・推論の本番環境では、`requirements.txt` と `--extra-index-url` だけでは OS や CUDA ランタイムまでは固定できないため、Docker イメージに環境を丸ごと焼き込み、レジストリに push して実行する運用が標準です。Python レイヤ(requirements)とシステムレイヤ(コンテナ)で再現性の責務を分担します。

## さらに学ぶための関連トピック

- [requirements.txt による再現可能な環境構築](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1150_requirements-txt-reproducible-install)
- [推論速度ベンチマーク（FPS/レイテンシ計測）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0685_inference-speed-benchmark)
- [GPU ウォームアップと cuda.synchronize による正確な計測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0681_gpu-warmup-cuda-synchronize-timing)
