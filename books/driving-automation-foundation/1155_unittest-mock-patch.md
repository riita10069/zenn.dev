---
title: "unittest.mock.patchによる依存差し替え"
---

# unittest.mock.patchによる依存差し替え

## ひとことで言うと
プログラムの一部(たとえば重い部品を作るクラス)を、テストのあいだだけ別物にこっそり差し替える Python 標準の道具が `unittest.mock.patch` です。これを使うと、テスト対象のコードを1行も書き換えずに、内部で呼ばれている部品を偽物に置き換えられます。代表的な使い方は、モデルが実行時に呼ぶ本物の重い backbone を軽い偽物に差し替え、事前学習済み重みの読み込みそのものを回避することです。本番コードに手を入れずにテストだけを高速化・安定化できる点が肝です。

## 直感的な理解
舞台の本番中に、観客に気づかれないよう小道具を差し替える「すり替え」を想像してください。脚本(本番コード)はそのまま、役者(モデル)も台本どおり動くのに、手に取った道具だけが軽い模型に変わっている。幕が下りれば(テストが終われば)道具は元に戻ります。

機械学習のテストで偽物の backbone([モックbackboneによる軽量テスト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1127_mock-backbone-stub))を用意しても、肝心の「本物の代わりに偽物を使わせる」ところで詰まります。なぜなら、モデルのクラスは自分の初期化処理の中で本物の backbone を直接呼んで部品を作るからです。テスト側から「そこだけ偽物にして」と頼む手段がないと、本番コードを書き換えるしかなくなります。`patch` はこの「本番中のすり替え」を、本番コードに一切触れずに実現します。

## 基礎: 前提となる概念
- import と名前空間: Python では `from foo import Bar` と書くと、`Bar` という名前が「いま実行中のモジュールの名前空間」に取り込まれます。重要なのは、これは「`Bar` の値(クラスそのもの)へのローカルな参照」を作る点です。後で `foo` 側の `Bar` を差し替えても、すでに取り込んだローカル参照は変わりません。これが patch の最重要ポイントの伏線です。
- 名前空間(namespace): 名前と値の対応表。各モジュールは自分の名前空間を持ち、`module.Name` でその中の名前を指せます。
- モンキーパッチ(monkey patching): 実行時にオブジェクトや関数を動的に書き換える手法の総称。`patch` はこれをテスト用に安全に(自動で元に戻す形で)行う仕組みです。
- コンテキストマネージャ(context manager): `with ... :` 構文で使うオブジェクト。ブロックに入るときと出るときに決まった処理を走らせます。`patch` を `with` で使うと、ブロックを抜けた瞬間に差し替えが自動解除されます。
- テストダブル(test double): 本物の代役の総称(Dummy / Stub / Mock / Fake)。`patch` はこの代役を本番コードに注入するための仕組みです。
- 決定的(deterministic): 何度実行しても同じ結果になる性質。外部のネットワークやファイルに依存しないテストは決定的になり、信頼できます。

## 仕組みを詳しく
偽物の backbone を用意しても、モデルは自分の初期化の中で本物を直接作ります。

```python
# モデルの内部(イメージ)
from .backbone import Backbone
class DrivingModel:
    def __init__(self, ...):
        self.image_backbone = Backbone(backbone="swin_v2_tiny", is_pretrained=True)
        #                     ↑ ここで本物が呼ばれ、重みファイルの読み込みが走る
```

この `Backbone(...)` の呼び出しはモデル内部に埋まっており、外から差し替える素直な口がありません。選択肢は3つです。

1. モデルのコードを書き換える(本番コードをテストのために汚す。避けたい)。
2. backbone を外から渡せるよう設計を変える(依存性注入。良い手だが既存設計の改修が要る)。
3. 実行時に `Backbone` という名前が指す中身を、テストのあいだだけすり替える(`patch`)。

`patch` は3を実現します。本番コードもモデル設計も変えずに、テスト実行中だけ `Backbone` が偽物を指すようにできます。さらに重要なのは、本物の `Backbone(...)` を一度も呼ばないので、事前学習済み重み(数百MBになることもある数値ファイル)のダウンロードや読み込み自体が起きないことです。テストがネットワークやファイルに依存せず、速く・決定的になります。

```python
def _build_model_with_mock_backbone(num_views, fusion_mode="bev", device=None, ...):
    from unittest.mock import patch

    with patch('mypkg.driving_model.Backbone', MockBackbone):
        model = build_model(num_views=num_views, fusion_mode=fusion_mode, ...)
    return model.to(device)
```

ステップで分解します。

1. `with patch('mypkg.driving_model.Backbone', MockBackbone):` で「`mypkg.driving_model` モジュールの中の `Backbone` という名前を、`MockBackbone` に差し替える」と宣言します。
2. patch のターゲット文字列が `mypkg.driving_model.Backbone`(使用先)である点が肝心です。`Backbone` クラスは別ファイル(`mypkg.backbone`)で定義されていても、`driving_model.py` の冒頭で `from .backbone import Backbone` と取り込み、`driving_model` の名前空間の `Backbone` として使っています。モデルが実際に呼ぶのはこの「driving_model から見た Backbone」なので、そこを差し替えないと効きません。定義元の `mypkg.backbone.Backbone` を差し替えても、driving_model はすでに自分の名前空間に取り込んだ参照を使うので無効です。これが鉄則「定義された場所ではなく、使われる場所を patch する」の意味です。
3. `with` ブロックの中でモデルを作ると、内部で `Backbone(...)` が呼ばれる瞬間、その名前は `MockBackbone` を指しているので偽物が作られます。本物の重み読み込みは一切走りません。
4. `with` ブロックを抜けると patch は自動解除され、`Backbone` は本物に戻ります。だから本物 backbone を使う統合テストなど他のテストに影響しません。この自動解除こそ `patch` をコンテキストマネージャで使う利点です。

「使われる場所を patch する」という鉄則の例外的注意として、もし `driving_model.py` が `from . import backbone` と取り込み、コード中で `backbone.Backbone(...)` と「モジュール経由で」参照しているなら、patch ターゲットは `mypkg.backbone.Backbone`(定義元)になります。つまり厳密には「コード中でその名前を解決する経路をすべて押さえる」が正確な原則で、import の書き方で patch 先が変わります。

### なぜ偽物が引数を受け取れる必要があるか
差し替え先(MockBackbone)は、本物と同じ引数で呼べないといけません。モデルは `Backbone(backbone="swin_v2_tiny", is_pretrained=True)` のように呼ぶので、偽物の初期化も同じ引数を(無視してでも)受け取れる形にします([モックbackboneによる軽量テスト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1127_mock-backbone-stub))。これをインターフェース(呼び出しの形)を合わせると言います。形が違うと、差し替えても呼び出しの瞬間に `TypeError` などの例外になります。同様に、後段が `model.image_backbone.backbone_channels` のような属性を参照するなら、その属性も偽物に持たせておく必要があります。

### patch の3つの書き方
`unittest.mock.patch` には主に3つの使い方があります。

- コンテキストマネージャ(`with patch(...) as m:`): 差し替える範囲を `with` ブロックに限定できる。範囲が明示的で読みやすく、ブロックを抜ければ自動で戻る。
- デコレータ(`@patch('...')` をテスト関数の上に付ける): 関数全体の実行中だけ差し替える。差し替えた偽物が関数の引数として渡ってくる(複数 patch を重ねると、デコレータの「下に書いた順」から引数が並ぶ独特のルールがある。`patch` は内側=関数に近い側から順に適用されるため)。
- 手動(`patcher = patch(...); patcher.start()` … `patcher.stop()`): 開始と停止を自分で書く。`stop()` を忘れると差し替えが残り、後続テストを壊すので注意が要る。`unittest.TestCase` なら `addCleanup(patcher.stop)`、pytest なら `monkeypatch` fixture を使うと、停止の自動予約ができて安全。

### MagicMock と autospec
差し替え先に独自クラスではなく `MagicMock`(何でも受け取る万能の偽物。属性アクセスもメソッド呼び出しも自動で受け流す)を使う書き方もあります。手軽ですが、本物に存在しないメソッドを呼んでも黙って通ってしまうため、本番のタイプミスを見逃します。これを防ぐのが `autospec=True`(または `create_autospec`)で、本物のシグネチャ(引数の形)と属性を写し取った偽物を作り、誤った呼び出しを例外にします。重い backbone のように「形だけ合った本物相当の偽物」が欲しい場合は、`MagicMock` より専用の Stub クラス(`MockBackbone`)や `autospec` が安全です。なお `autospec` はシグネチャは写しますが、戻り値の tensor shape までは保証しないため、形を返す責務はやはり Stub 側で持つ必要があります。

## 手法の系譜と主要論文
`unittest.mock` は機械学習の手法ではなく、Python 標準ライブラリの機能です。Michael Foord が開発したサードパーティの `mock` ライブラリが Python 3.3(2012年)で標準ライブラリ入りしました。背景にあるのはテスト工学の「テストダブル」と「依存の分離」という考え方です。

Gerard Meszaros "xUnit Test Patterns"(2007)が、本番コードを変えずに依存を差し替えるパターンを体系化しました。`patch` が対応するのは、実行時に依存を置き換える Monkey Patching や、テスト専用の差し替え点を作る各種パターン(Test Double の注入)に近いものです。Martin Fowler "Mocks Aren't Stubs"(2007、エッセイ)は、Stub(決まった応答を返す)と Mock(呼ばれ方も検証する)を区別し、「状態検証(出力を見る)か振る舞い検証(呼ばれ方を見る)か」という設計判断を広めました。`patch` はどちらの代役も注入できる土台です。`patch` が返す `MagicMock` は `assert_called_once_with(...)` などで「振る舞い検証」もできますが、backbone の差し替えでは多くの場合「形を返す Stub」として使われます。

機械学習の文脈では、Breck et al. "The ML Test Score"(IEEE Big Data 2017 / NeurIPS 2017 ML Systems Workshop)が、外部依存(学習済みモデル・データ取得・API)を分離して決定的に検証することを推奨しています。`patch` による依存差し替えは、その推奨を実現する具体手段です。

## 論文の実験結果(定量データ)
`patch` 自体の論文評価はありませんが、依存差し替えがもたらす効果は測定可能です。

- 速度: 本物 backbone の構築は、事前学習済み重みの読み込みを含めて1回あたり数十ms〜数百ms(重みがディスクキャッシュにない初回や、リモートからのダウンロードを含むとさらに長い)。`patch` で偽物に差し替えると、この読み込みがゼロになります。テスト準備の共有([pytestのfixtureと共有](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1142_pytest-fixtures))と軽量 mock([モックbackboneによる軽量テスト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1127_mock-backbone-stub))を併用した報告では、スイート全体が約3分から1分未満に短縮されました。
- 決定性・信頼性: 外部依存を切ると、ネットワーク不通や重みファイルのバージョン差・ハッシュ不一致による失敗が消えます。"The ML Test Score" のルーブリックでは、外部依存の除去とテストの決定性が加点項目で、実調査では多くの本番システムがこの項目で低スコアだったと報告されています。flaky test(実行ごとに結果が揺れるテスト)の主因の1つが外部依存・ネットワークであることは、Google の大規模テスト分析(Micco ら 2016-2017)でも報告されており、相当割合のテストがいずれかの時点で flaky になると示されました。
- アブレーション的観点: patch のターゲットを「定義元(`mypkg.backbone.Backbone`)」に間違えると、import 経路によっては差し替えが効かず本物が読み込まれ、テストは(遅いまま)通ってしまいます。つまり「効いていないのに気づきにくい」失敗が起きます。これを検出するには、テスト内で `MockBackbone` の `__init__` が呼ばれたことを spy で確認するか、重みロード処理側にカウンタを仕込んで「ロードが0回であること」を assert する防御が有効です。テスト実行時間が想定より明らかに長い場合も、patch が効いていない兆候として疑うべきです。

## メリット・トレードオフ・限界
メリット:
- 本番コードもモデル設計も変えずに、内部の依存を差し替えられる。
- 本物を呼ばないので重みのダウンロード・読み込みが起きず、ネットワーク・ファイル非依存で速く決定的なテストになる。
- `with` ブロックを使えば差し替え範囲が明示的で、抜ければ自動で元に戻る。
- ネットワーク API・乱数・時刻・GPU など、テストごとに変わる外部要因も同じ仕組みで固定できる(汎用性が高い)。

トレードオフと限界:
- patch のターゲット文字列を間違えやすい(定義元と使用先の取り違え、import の書き方による経路の違い)。間違えると「patch したのに効かない」という分かりにくい不具合になる。
- 差し替え先が本物の呼び出し形(引数・属性・戻り値の形)と合っていないと、実行時に例外になる。インターフェースを手で揃える保守コストがかかる。
- patch は内部実装(import の仕方やモジュール構成)に依存する。対象コードのリファクタで import 経路や使用先のパスが変わると patch 先がずれて壊れる。これは「テストが実装の内部構造に結合する」という根本的な弱点で、過度な patch は脆い(brittle な)テストを生む。
- 設計上の示唆として、patch を多用しないと差し替えられない構造そのものが「依存性注入の不足」の兆候であることがある。本来は backbone を外から渡せる設計(コンストラクタ注入)にすれば、patch なしでテストできる。patch は便利だが、設計改善のサインを覆い隠す側面もある。テスタビリティ(テストのしやすさ)は、最終的には設計で確保するのが理想で、patch はその移行期の橋渡しと捉えるのが健全。

## 発展トピック・研究の最前線
- 依存性注入(DI)フレームワークとの対比: patch がテスト時の動的すり替えなのに対し、DI は設計段階で差し替え点を明示する。テスト容易性(testability)を「設計でどこまで作り込むか」は継続的な議論で、近年は「構成可能性(composability)を高めて patch を減らす」方向が支持されている。
- コントラクトテスト: patch で注入した偽物が本物と同じ振る舞い契約(同じ引数・同じ戻り値の形)を満たすかを自動照合し、偽物の仕様ずれを検出する手法。本物の出力形をゴールデンファイルに保存し、CI で突き合わせる実装が実務的。
- ML パイプラインのテスト体系化: "The ML Test Score" 以降、データ・モデル・サービングの各層で外部依存をどう切り、どこを本物で検証するかの設計指針が整理されつつある。patch とモック化はその基盤技術として位置づけられている。大規模事前学習モデルを部品に組み込むシステムでは、本物を毎回ロードできないため、patch による分離の重要性がいっそう増している。

## さらに学ぶための関連トピック
- [モックbackboneによる軽量テスト](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1127_mock-backbone-stub)
- [pytestのfixtureと共有](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1142_pytest-fixtures)
- [SwinV2の4段階特徴マップ形状](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1016_swinv2-feature-map-shapes)
- [pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration)
