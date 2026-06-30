---
title: "pytestのfixtureと共有"
---

# pytestのfixtureと共有

## ひとことで言うと
テスト(プログラムが期待どおり動くかを自動で確かめる小さなチェック)を書くとき、どのテストでも共通して必要になる「下準備」を1か所にまとめて書いておき、各テストから名前で呼び出して使い回す仕組みです。pytest(Python で最も広く使われているテストフレームワーク)では、この下準備のかたまりを fixture(フィクスチャ)と呼びます。fixture を使うと、同じ準備コードを何度もコピペせずに済み、しかも「重い準備を1回だけ実行して全テストで共有する」といった高速化や「テストが終わるたびに自動で後始末する」といった整理もできます。テストを速く・読みやすく・壊れにくくするための、最も基本的な道具です。

## 直感的な理解
料理にたとえます。テスト1個1個が「1皿の料理を作る作業」だとします。どの料理を作るにも、毎回「コンロに火をつける」「まな板と包丁を出す」「だしを取る」という準備が必要です。これを料理のたびに最初からやり直していたら時間がかかります。fixture は、この「いつも必要な準備」をキッチンの段取りとして1か所に書いておき、各料理は「だしください」と頼むだけで受け取れるようにする仕組みです。さらに「火を消す」「まな板を洗う」という後片付けまで段取りに含められます。

機械学習(ML)のコードをテストする状況を想像してください。ニューラルネットワークのある部品が、正しい形のデータを出すか確かめたいとします。その前に毎回、モデル本体を組み立て、計算装置(CPU か GPU か)を決め、ニセの入力データ(ランダムな画像テンソル)を用意する必要があります。この準備が重ければ重いほど、また使い回せる範囲が広いほど、fixture の効果は大きくなります。ML のテストでは「事前学習済みの重み(学習で得た数百MBの数値の塊)を読み込む」という準備が極端に重く、ここを賢く共有できるかどうかでテスト全体の所要時間が桁単位で変わります。

## 基礎: 前提となる概念
最初にいくつかの用語を噛み砕きます。

- テンソル(tensor): 多次元の数値配列です。1つの数(スカラ)が0次元、数の列(ベクトル)が1次元、表(行列)が2次元、それ以上を一般化したものがテンソルです。画像は「縦×横×色チャンネル」の3次元、それをまとめたバッチは4次元のテンソルになります。
- tensor shape(テンソルの形): 各次元のサイズの並びです。たとえば `[8, 3, 256, 256]` は「8枚の画像、各画像は3チャンネル(RGB)、縦256、横256」を意味します。テストで「形が合っているか」は、部品どうしが正しくつながる最低条件です。
- assert(アサーション): 「ここでこの条件が成り立っているはず」とプログラムに宣言し、成り立たなければテストを失敗させる文です。テストの本体は、突き詰めればこの assert の集まりです。
- デコレータ(decorator): 関数の上に `@なんとか` と書いて、その関数に追加の意味づけをする Python の仕組みです。fixture は `@pytest.fixture` というデコレータで宣言します。
- 依存性注入(dependency injection): ある部品が必要とする別の部品を、その部品自身が作るのではなく「外から渡してもらう」設計です。fixture はこの注入をテストに対して自動でやってくれます。

テストの古典的な構造は「準備(setup)→ 実行(exercise)→ 検証(verify)→ 後始末(teardown)」の4段階です。この4段階モデルは Kent Beck らの xUnit 系フレームワークが広めたもので、fixture が担うのは主に最初の準備と最後の後始末です。本体のテスト関数は実行と検証に集中できます。

## 仕組みを詳しく
pytest の fixture は、`@pytest.fixture` を付けた関数として定義します。そのあとテスト関数の「引数名」に fixture 名を書くと、pytest が自動でその fixture を実行し、返り値を引数に渡してくれます。

```python
import pytest, torch

# (1) fixture の定義。名前は関数名 "device"
@pytest.fixture(scope="session")
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (2) テスト関数。引数名 "device" を書くと (1) の返り値が渡ってくる
def test_something(device):
    assert device.type in ("cuda", "cpu")
```

ここで起きていること:
- pytest は `test_something` を実行する前に、引数名 `device` を見て「同名の fixture があるな」と気づく。
- fixture `device()` を呼び、返り値を `test_something` の引数 `device` に注入(inject、外から渡し込むこと)する。
- これが依存性注入です。テスト側は「装置をどう作るか」を知らなくてよく、ただ「device がほしい」と名前で頼むだけ。準備の実装が変わってもテスト本体は無傷でいられます。

### scope: 準備をどれくらいの範囲で使い回すか
fixture の最大の武器が `scope` です。「この準備を何回作り直すか」を決めます。

| scope | いつ作り直すか | 用途 |
|---|---|---|
| `function`(既定) | テスト関数ごとに毎回 | 状態を汚したくないとき |
| `class` | テストクラスごとに1回 | クラス内で共有 |
| `module` | ファイルごとに1回 | ファイル内で共有 |
| `package` | パッケージ(ディレクトリ)ごとに1回 | ディレクトリ内で共有 |
| `session` | テスト全体で1回だけ | 重い準備の使い回し |

重い準備を `scope="session"` にすると、構築回数が「テストの数 N 回」から「1 回」に減ります。

```python
@pytest.fixture(scope="session", params=["bev"])
def model(request, device):
    """全テストで共有する重いモデル。1回だけ構築する。"""
    return build_model(fusion_mode=request.param, device=device)
```

数値例で効果を見ます。仮にモデル構築が1回50ms、テストが90個あるとします。`function` スコープなら 90×50ms = 4.5 秒がモデル構築だけに費やされます。`session` スコープなら 50ms 一度きりです。事前学習済み backbone(画像特徴抽出器)の読み込みのように、準備が数百ms〜数秒かかる ML のテストでは、この差はさらに広がります。実プロジェクトの報告では、重いコンポーネントの session 共有と後述のモック化を組み合わせることで、テストスイート全体が約3分から1分未満へ短縮された例があります。

### conftest.py: fixture をテスト全体で共有する場所
fixture をテストファイルの中に書くと、そのファイルからしか使えません。複数のテストファイルで共有したいときは `conftest.py` という特別な名前のファイルに書きます。pytest はこのファイルを自動で発見して読み込み、同じディレクトリ(とその配下)の全テストに fixture を配ります。`import` 文すら不要です。プロジェクト共通の `device` や `model` といった fixture は、ここに置くのが定石です。`conftest.py` はディレクトリ階層ごとに置け、近い階層のものが優先されるため、共通設定をルートに、特化設定を下層に置くといった階層化もできます。

### params と request: 1つの fixture を複数設定で回す
`@pytest.fixture(params=[...])` と書くと、リストの各要素ごとにそのテストが繰り返し実行されます。fixture の中では `request.param` で現在の値を取れます。上の例の `params=["bev"]` は、将来 `["concat", "cross_attn", "bev"]` のように増やせば、同じテスト本体を3つの設定で自動的に回せます。`request` は「いま誰がこの fixture を呼んでいるか」というメタ情報を持つ特別なオブジェクトで、`request.param` のほかに `request.getfixturevalue("名前")` で別 fixture を動的に取得することもできます。テストの組み合わせを宣言的に表現できる強力な仕組みです。

### 共有 fixture の落とし穴: 状態の汚れ
`scope="session"` で1個のオブジェクトを共有すると、便利な反面、危険もあります。あるテストがモデルに勾配(gradient、誤差をもとに重みをどう直すかの数値)を溜めたり、学習モードと評価モードを切り替えたりすると、次のテストがその「汚れた」状態を受け継ぎます。結果、実行順序によってたまたま通ったり落ちたりする不安定なテスト(flaky test)になります。

定石の対処は、後始末を自動で全テストに適用する `autouse` fixture です。

```python
@pytest.fixture(autouse=True)
def _reset_model_state(request):
    yield                                   # ここまでがテスト前、yield 後がテスト後
    if "model" in request.fixturenames:
        model = request.getfixturevalue("model")
        model.zero_grad(set_to_none=True)   # 溜まった勾配をリセット
        model.train()                       # 学習モードに戻す
```

`yield` を境に、前半が「テスト前の準備」、後半が「テスト後の後始末(teardown)」です。`yield` を使う fixture は、`return` で値を返すだけの fixture と違い、後始末コードを自然に書ける点が利点です。共有 fixture を使うときは、この「後始末で状態を戻す」がほぼ必須になります。`autouse=True` は引数に書かなくても全テストに自動適用される指定で、こうした横断的な後始末に向いています。

### factory fixture: 引数を変えて何度も作りたいとき
「設定を少しずつ変えたオブジェクトを、テストごとに作りたい」こともあります。そのときは fixture が値そのものではなく「関数」を返す factory(工場)fixture が便利です。

```python
@pytest.fixture
def build_mock_model():
    """モデル構築関数そのものを返す factory。"""
    return _build_model_with_mock_backbone   # 関数を返す

def test_views(build_mock_model, device):
    model = build_mock_model(num_views=3, device=device)  # テスト側で引数を決める
```

これで、テストは好きな引数でオブジェクトを組み立てられます。共有 fixture が「1個を使い回す」のに対し、factory fixture は「作り方を渡して、必要なものをその場で作らせる」点が違います。状態を共有したくないが構築は柔軟にしたい、という中間的な需要に応えます。

## 手法の系譜と主要論文
fixture は特定の機械学習論文の手法ではなく、ソフトウェアテスト工学の確立した概念です。系譜をたどると次のようになります。

源流は1990年代後半に Kent Beck が Smalltalk 向けに作った SUnit、それを Erich Gamma と Kent Beck が Java へ移植した JUnit に始まる「xUnit」と呼ばれるテストフレームワーク群です。Kent Beck "Test-Driven Development: By Example"(Addison-Wesley, 2002)は、テストを「setup / exercise / verify / teardown」の4段階で構造化する考え方を広めました。fixture は最初の setup と最後の teardown を担う部品にあたります。

次に Gerard Meszaros "xUnit Test Patterns: Refactoring Test Code"(Addison-Wesley, 2007)が、テスト準備のパターンを体系化しました。テストごとに準備をやり直す方式を Fresh Fixture、使い回す方式を Shared Fixture と呼び分け、Shared Fixture は速い反面テスト間に隠れた依存(状態の漏れ)を生むと整理しています。pytest の `scope` はこの2つを連続的に選べるようにした実装であり、`session` + `autouse` リセットは「Shared Fixture の速さを取りつつ、後始末で状態の漏れを打ち消す」という古典的パターンの実践です。

pytest 自体は2000年代後半に Holger Krekel が中心となって開発を始め、`@pytest.fixture` による引数注入・スコープ・パラメータ化という今日の設計が固まりました。標準ライブラリの `unittest` がクラス継承で setUp/tearDown を書かせる「継承ベース」だったのに対し、pytest は「引数名で必要なものを宣言する」関数ベースのモデルを採用し、合成しやすさと再利用性で広く支持されました。

機械学習の文脈では、Eric Breck・Shanqing Cai・Eric Nielsen・Michael Salib・D. Sculley らによる "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction"(Google, IEEE Big Data 2017 / NeurIPS 2017 ML Systems Workshop)が、ML システムを本番投入する前のテスト成熟度を採点する28項目のルーブリックを示しました。その中で、重い学習済みコンポーネントや外部データ取得を切り離して決定的に検証することが推奨されており、共有 fixture とモック化はその実装手段に位置づけられます。

## 論文の実験結果(定量データ)
fixture そのものは定量評価される研究対象ではありませんが、関連する定量的知見を挙げます。

- 高速化の倍率は「準備コストがテスト全体に占める割合」で決まります。アムダールの法則の素直な応用で、準備が全体の割合 p を占めるとき、準備を N 回から1回に減らした場合の理論上の所要時間は概ね `(1-p) + p/N` 倍になります。事前学習済みモデルを読み込むテストでは p が0.7〜0.9に達することが珍しくなく、p=0.8・N が大きい場合、所要時間はおよそ 1/5 以下になります(残り20%は各テスト固有の処理として残る)。
- 実プロジェクトの報告例では、重いバックボーンのロードと forward がテスト時間の支配項で、session スコープ共有とモック化の併用によりスイート全体が約3分から1分未満に短縮されたとされます。これは「準備を1回に減らす(共有)」「準備自体を軽くする(モック化)」の二段構えの効果です。
- 一方で、Shared Fixture は flaky test(実行順序や環境で結果が変わる不安定なテスト)の発生率を上げます。Google の大規模テスト分析(Micco ら "Flaky Tests at Google" 2016-2017 の公開報告)では、全テストのうち相当割合がいずれかの時点で flaky な振る舞いを示し、再実行コストとシグナル劣化の主因になっていると報告されました。不安定性の原因として「テスト順序依存」「共有状態」が上位に挙がっており、後始末を欠いた Shared Fixture はこのリスクを直接抱えます。
- "The ML Test Score" の調査では、実際の本番 ML システムを採点すると多くが低スコアにとどまり、特に「インフラのテスト(再現性・コンポーネント分離)」の項目が手薄であると報告されました。fixture による準備の構造化と外部依存の分離は、まさにこの手薄な項目を埋める実装手段です。

## メリット・トレードオフ・限界
メリット:
- 準備コードの重複を排除でき、修正が1か所で済む。直し忘れによる事故が減る。
- `scope` で重い準備を使い回せ、テストが大幅に高速化する。
- テスト本体が「確かめたいこと」だけになり読みやすくなる。
- 依存性注入により、準備の実装が変わってもテスト本体は影響を受けにくい。
- `yield` による後始末で、リソースの解放漏れ(ファイル・GPU メモリ・一時ディレクトリ)を防げる。

トレードオフと限界:
- 共有スコープ(session/module)は状態が次のテストに漏れやすく、不安定なテストの温床になる。後始末(teardown)の設計が必須。
- どの fixture がどこで実行されるかが暗黙的(引数名による自動注入)なので、初学者には準備の出所が追いにくい。`pytest --fixtures` で一覧、`pytest --setup-show` で実行順の可視化ができる。
- fixture が別の fixture を呼ぶ連鎖が深くなると、依存関係の全体像が読み解きづらくなる。
- 未解決の課題として、テストの並列実行(`pytest-xdist` など)と session 共有は相性が悪い。プロセスをまたいでオブジェクトを共有できないため、並列度を上げると各ワーカーが個別に重い準備を作り直し、共有の利点が薄れる。並列化と共有のどちらを優先するかは、テストスイートごとに判断が要る。GPU を使うテストでは、複数ワーカーが同時に同じ GPU メモリを取り合う問題もあり、並列化を素直に適用できないことが多い。

## 発展トピック・研究の最前線
- パラメータ化と fixture の組み合わせ(`pytest.mark.parametrize` と `params`)で、設定の直積を宣言的に網羅する手法。設定が増えると組み合わせ爆発するため、すべての組を試さず「2因子の全組み合わせだけ」を網羅するペアワイズ法など、組合せテスト理論(combinatorial testing)と接続する。
- プロパティベーステスト(Hypothesis ライブラリ)との併用。fixture で環境を整え、入力は乱数生成器に任せて「どんな入力でも成り立つべき性質」を検証する。具体例を手書きする代わりに「性質」を書く発想で、エッジケースの発見力が高い。
- ML 特有のテストとして、訓練の決定性(同じシードで同じ結果になるか)、データスキーマの検証、勾配が末端の層まで流れるかの gradient-flow テスト、メトリクスの回帰検出などがあり、これらはいずれも fixture で環境を固定して初めて再現可能になる。"The ML Test Score" 以降、こうした ML テストの体系化(データテスト・モデルテスト・インフラテストの三層)が研究・実務の両面で進んでいる。
- テスト時間が長くなりがちな ML では、変更に関係するテストだけを選んで走らせる Test Impact Analysis や、過去に失敗しやすかったテストを先に走らせる優先度付け実行など、テスト選択・順序付けの研究も活発である。

## さらに学ぶための関連トピック
- [モックbackboneによる軽量テスト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1127_mock-backbone-stub)
- [unittest.mock.patchによる依存差し替え](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1155_unittest-mock-patch)
- [SwinV2の4段階特徴マップ形状](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1016_swinv2-feature-map-shapes)
- [pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration)
- [pytest addoptsでデフォルト除外](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1141_pytest-addopts-default-exclude)
