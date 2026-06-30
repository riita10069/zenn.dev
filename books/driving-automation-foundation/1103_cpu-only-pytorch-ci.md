---
title: "CPU-only PyTorchでのCI実行"
---

# CPU-only PyTorchでのCI実行

## ひとことで言うと
自動テストを回すマシン(CI ランナー)には高価な GPU を積まず、CPU だけで動く軽量版の PyTorch を入れて、テストを速く・安く回す工夫です。CI で検証したいのは多くの場合「コードのロジックが壊れていないか」であって「GPU で本当に速く学習できるか」ではないので、GPU 用の重い部品を切り落とせる、という割り切りに基づきます。GPU が必須なテストはマーカーで分離して CI から除外し、GPU 検証は別の GPU マシンに任せます。

## 直感的な理解
料理教室で「レシピの手順が正しいか」を確認するのに、毎回プロの業務用大型オーブンを持ち込む必要はありません。手順の正しさ(材料を入れる順番、分量の計算)は家庭用コンロでも確認できます。本番の大量調理のときだけ業務用オーブンを使えばよい。

PyTorch も同じです。PyTorch は機械学習で広く使われるライブラリで、学習(大量データを見せてモデルを賢くする処理)では膨大な行列計算が必要なため、それを並列処理する GPU(Graphics Processing Unit、もとは画像処理向けだった並列計算チップ)を使います。ところが、

- PyTorch を普通にインストールすると、NVIDIA GPU を使うための追加部品(CUDA ランタイム、cuDNN など)が一緒に入る。これらは合計で数 GB ある。
- CI の仮想マシンは毎回まっさらから起動するので、毎回この数 GB をダウンロード・インストールすることになり、テスト本体より「準備」に時間がかかる。
- そもそも CI で標準提供される VM(GitHub Actions の `ubuntu-latest` など)には GPU が付いていない。GPU が無いのに GPU 用の重い部品を入れても、ダウンロード時間を払うだけで一切使われない。

「コードのロジックの正しさは CPU でも検証できる」と発想を変えれば、CI には CPU 専用の軽い PyTorch を入れれば十分、という判断になります。これが CPU-only PyTorch での CI 実行です。

## 基礎: 前提となる概念

- **GPU と CUDA**: GPU は並列計算に特化したチップ。NVIDIA の GPU を汎用計算に使うための基盤ソフトウェアが CUDA。cuDNN は深層学習の畳み込みなどを高速化する最適化ライブラリ。PyTorch の GPU 版はこれらを同梱(または依存)する。
- **PyTorch のビルド種別**: PyTorch には「どの計算バックエンド向けか」で複数のビルドがある。代表的には CPU 専用版、CUDA 各バージョン版、ROCm(AMD GPU)版など。インストール時にどれを入れるか選べる。同じ `import torch` でも中身が違う。
- **wheel(ホイール)と index URL**: Python のパッケージは `.whl`(wheel)という事前ビルド済みの配布形式で配られる。`pip install` はパッケージインデックス(配布元)から wheel を取得する。PyTorch は CPU 版・CUDA 版を別々のインデックス URL で配布しており、`--index-url` で配布元を切り替えると入るビルドが変わる。
- **CI ランナー**: テストを実行する仮想マシン。GitHub などの標準提供のものは CPU のみで GPU 非搭載。
- **smoke test(スモークテスト)**: 「電源を入れて煙が出ないか」程度の最低限の動作確認。小さな入力でモデルの forward が例外なく通り、出力 shape が期待どおりかだけを確かめる軽いテストを指すことが多い。CPU 化しやすい代表例。
- **マーカーによるテスト分離**: GPU や学習済み重みが必須のテストにタグを付け、CI では除外する仕組み([pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration))。

## 仕組みを詳しく

### CPU 専用ビルドを入れる

PyTorch は配布元(index URL)を切り替えることで、入るビルドを変えられます。

```bash
# CPU 専用版を入れる(GPU 用の重い部品を含まない)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 比較: CUDA 版(GPU 用部品を含む、数 GB)
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

`--index-url` は「どの配布元からダウンロードするか」を切り替えるオプションです。`/whl/cpu` という配布元には GPU 用部品を含まない CPU 専用 wheel だけが置かれています。これを使うと PyTorch の導入サイズが大きく縮みます。

数値感(おおよそ、報告・配布物のサイズによる):
- CUDA 版 PyTorch + 依存(CUDA ランタイム、cuDNN 等)= 数 GB 規模。ダウンロード+展開で数分かかることがある。
- CPU 専用版 = 数百 MB 規模。導入が大幅に短い。

この差がそのまま CI の毎回のセットアップ時間に効きます。CI VM は毎回まっさらなので、この準備時間は実行のたびに発生する固定費です。たとえばテスト本体が30秒で終わるのに、セットアップが3分かかっていては、フィードバックループの大半が「待ち」になります。

### GPU 必須テストを除外する

CPU 版を入れただけでは不十分です。`torch.cuda.is_available()` が False の環境で GPU を前提とするコードを走らせると、`RuntimeError: No CUDA GPUs are available`(あるいは `Found no NVIDIA driver`)等で失敗します。そこで GPU が必須のテストにマーカーを付け、CI から除外します。

```python
import pytest

@pytest.mark.integration   # あるいは @pytest.mark.gpu
def test_full_forward_with_pretrained_weights():
    # 学習済み重みのダウンロードと GPU を要する重いテスト
    ...
```

CI 側では GPU 不要なテストだけを選んで実行します。

```bash
pytest -m "not integration"
```

`-m "not integration"` は「integration という印が付いていないテストだけ走らせる」という意味です(詳細は [pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration))。これにより、CPU で十分検証できる軽いユニットテストだけが CI で回ります。さらに設定ファイルの `addopts` にこの除外を既定として書いておけば、`pytest` を素で打つだけで自動的に GPU テストが外れます([pytest addoptsでデフォルト除外](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1141_pytest-addopts-default-exclude))。CPU-only ビルドとマーカー除外は、片方だけでは穴が残るため、両方を組み合わせて初めて「CI が安定して緑になる」状態が完成します。

### device に依存しないコードの書き方

CPU でも GPU でも同じテストを通すには、コード側を device 非依存に書くのが定石です。

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)
```

こう書いておけば、GPU 不要な軽いテストは CPU 上で問題なく通り、GPU マシンでは GPU 上で通ります。テストやモデルが `.cuda()` のように特定 device をハードコードしていると CPU CI で落ちるため、device 抽象を徹底することが CPU-only CI を成立させる前提になります。device をハードコードしないという規律は、後でモデルを別のアクセラレータ(Apple の MPS、TPU 互換層など)へ移すときにも効いてきます。

### before / after で整理

```
before(素朴な構成):
  GPU 版 PyTorch を入れる
    → 数 GB ダウンロード(毎回, 遅い)
    → GPU テストが GPU 不在で失敗
    → CI が無意味に赤くなる / セットアップで時間を浪費

after(CPU-only 構成):
  CPU 版 PyTorch を入れる
    → 数百 MB(毎回, 速い)
    → GPU 必須テストはマーカーで除外
    → GPU 不要なロジックテスト(shape, 分岐, API 誤用)だけ実行
    → 安定して緑, CI が軽い
```

GPU 特有の検証(実際の学習、VRAM 使用量、混合精度の数値挙動、CUDA カーネルの正しさ、`torch.compile` の GPU 経路)は、CI とは別に GPU 搭載マシン上で回します。CPU CI はあくまで「GPU なしで分かる範囲の正しさ」を守る関所、という役割分担です。

## 手法の系譜と主要論文

これは論文の手法ではなく、MLOps(機械学習の運用)の実践的エンジニアリングです。背景となる学術的問題意識を辿ります。

- **環境依存という技術的負債**: Sculley らの "Hidden Technical Debt in Machine Learning Systems"(NeurIPS 2015)は、ML システムでは「設定・データ・環境/インフラの複雑さ」がコード以上に技術的負債(technical debt、後で利息付きで返すことになる手抜き)になりやすいと指摘しました。論文中の "CACE: Changing Anything Changes Everything"(何かを変えると全部が変わる)や、glue code(つなぎコード)・configuration debt の議論が有名です。GPU 依存をテストから切り離し、検証したいロジックだけを軽い環境で回すのは、この環境依存の負債を局所化する具体策の一つです。

- **速いフィードバックループ**: Amershi らの "Software Engineering for Machine Learning: A Case Study"(ICSE-SEIP 2019)は、Microsoft の多数の ML チームへの調査から、ML 開発の生産性は「実験・検証のフィードバックループの速さ」に強く依存すると報告しました。CI を軽く保つ CPU-only 構成は、まさにこのフィードバックを速くする判断にあたります。

- **PyTorch の配布設計**: PyTorch チームは早くから CPU 専用 wheel を独立した index URL で配布しており、これは「GPU を持たない開発者・CI 環境でも `import torch` できるようにする」という明確な意図に基づきます。配布元を分けるこの設計が、CPU-only CI を1行で実現可能にしている前提技術です。同様の発想は他の深層学習フレームワーク(CPU 版と GPU 版を別パッケージにする等)にも共通します。

なお、これらの論文は「CPU-only ビルドを使う」という具体手法そのものを論じてはいないため、本トピックは学術手法ではなく、上記の問題意識に沿った実装上のグッドプラクティスとして理解するのが正確です。

## 論文の実験結果(定量データ)

CPU-only CI の効果を直接測る論文はありませんが、関連する定量的事実を挙げます。

- PyTorch の CPU 版と CUDA 版の配布サイズ差は、おおよそ「数百 MB 対 数 GB」のオーダー(報告・配布物のサイズによる)。CI VM が毎回まっさらで起動する以上、この差は実行のたびに繰り返し発生する固定的なダウンロード時間に直結します。キャッシュ(`actions/cache`)を併用しない限り、毎回数 GB を引くか数百 MB で済むかは、CI 1回あたり数分の差になりえます。月に何百回も回る CI では、この差が累積コストとして無視できなくなります。

- **Hilton ら(ASE 2016)** の調査では、CI 利用者がコストとして最も強く挙げたのが「ビルド/テスト時間」でした。セットアップ時間(依存導入)はビルド時間の無視できない一部であり、依存を軽くすることは開発者満足度に直接効く、と解釈できます。

- **応答時間の閾値**: HCI の古典的知見として、フィードバックが約10秒を超えると開発者は別作業に切り替えやすくなることが知られます。セットアップを含む CI 全体を軽く保つことは、PR ごとの「赤/緑が出るまで」を短縮し、開発者が文脈を保ったまま待てる範囲に近づけます。

指標の意味として、ここで効いているのは精度系の指標ではなく「CI 1回あたりの所要時間」と「CI の安定性(GPU 不在による偽の失敗を出さないこと)」です。前者は開発速度に、後者は CI への信頼(赤を本気で受け止めるか)に直結します。GPU 不在で偽の赤が出る CI は、たとえ速くても「どうせ環境のせい」と無視され、CI 本来の価値を失います。

## メリット・トレードオフ・限界

メリット:
- 導入が軽く速い(数 GB → 数百 MB のオーダー)。CI の固定費を毎回削れる。
- GPU のない標準 CI マシンで安定して動く(GPU 不在による偽の失敗が出ない)。
- CI の待ち時間とクラウドコストを抑えられる(GPU ランナーは時間単価が桁違いに高い)。
- ロジックのバグ(shape ミス、分岐の誤り、API 誤用、None 伝播)は CPU でも十分検出できる。

トレードオフ・限界:
- **GPU 特有の問題は検出できない**: VRAM 不足(OOM)、CUDA カーネルの挙動差、混合精度(fp16/bf16)の数値誤差、`torch.compile` の GPU 経路、マルチ GPU の通信など、GPU でしか出ない不具合は CPU CI をすり抜ける。
- **数値の微差**: CPU と GPU は浮動小数点演算(IEEE 754)の実行順序や実装(融合積和、並列リダクションの順序)が異なり、結果が完全一致しないことがある。CPU で書いたテストの許容誤差(tolerance)が GPU では破れる、あるいはその逆が起こりうる。`assert torch.allclose(a, b, atol=...)` の許容値設定が悩ましくなる。
- **別フロー必須**: GPU を要するテスト・実学習は別の GPU マシンや別 CI フロー(セルフホストランナー、夜間バッチ)で回す必要がある。CPU CI だけでは ML システムの検証は完結しない。
- **部分的な保証**: 「CI が緑=本番 GPU でも完全に動く」とは言い切れない。CPU CI は GPU なしで分かる範囲の保証にとどまる。

未解決の課題として、「GPU 必須ロジックをどこまで CPU で代理検証できるか」の線引きがあります。小さな入力で CPU 上でも forward が通ることを確認する smoke test は CPU 化しやすい一方、メモリ使用量や速度に関わる検証は本質的に GPU でしか測れません。また数値再現性の問題は根が深く、「CPU で緑だったテストが GPU で許容誤差を超える」事例は実務で頻出します。CPU CI と GPU 検証の役割分担、許容誤差の設計、決定的アルゴリズムの採否は、プロジェクトごとに最適点が異なります。

## 発展トピック・研究の最前線
- **セルフホスト GPU ランナー**: 自前の GPU マシンを CI に登録し、GPU 必須テストだけを別ジョブで回す構成。CPU-only の高速 CI と組み合わせる二段運用が定番([GitHub ActionsによるCI](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1112_github-actions-ci))。
- **二段 CI**: PR は CPU-only で軽く、main マージや夜間は GPU マシンで重いテスト・実学習まで回すゲート設計。
- **依存キャッシュと再現性**: `actions/cache` や lockfile(`uv.lock`, `poetry.lock`)で依存導入をさらに高速化・固定化する。再現可能なビルドは「手元では通るのに CI で落ちる」を減らす。
- **クロスデバイス数値整合性**: CPU と GPU、さらに異なる GPU 世代間で数値結果をどこまで一致させるか。`torch.use_deterministic_algorithms(True)` による決定的実行や、許容誤差ベースの比較(`torch.allclose`)の設計が研究・実務の論点。
- **他アクセラレータへの展開**: Apple Silicon の MPS バックエンドや各種 NPU 向けの軽量ビルドが整備されつつあり、device 非依存に書いておく価値が増している。

## さらに学ぶための関連トピック
- [GitHub ActionsによるCI](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1112_github-actions-ci)
- [pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration)
- [pytest addoptsでデフォルト除外](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1141_pytest-addopts-default-exclude)
- [ruffによるPython lint](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1151_ruff-lint)
- [lint失敗でCIを落とす運用](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1101_ci-fail-on-lint)
