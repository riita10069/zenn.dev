---
title: "Python dataclassによる型付き構造"
---

# Python dataclassによる型付き構造

## ひとことで言うと

Python の `@dataclass` は、「データをまとめて持つためのクラス」を、短く・型付きで・安全に定義できる仕組みです。属性の名前と型を書くだけで、初期化処理(`__init__`)や比較・表示用のコードを自動で作ってくれます。機械学習のパイプラインで、設定値・サンプル・モデル出力といった構造化データをきれいに受け渡しするために多用されます。

## 直感的な理解

プログラムでは「関連する複数の値を1つの塊として扱いたい」場面が頻繁にあります。例えば自動運転で1フレームのデータを表すなら、画像のパス・タイムスタンプ・速度・操舵角、をひとまとめにしたい。

素朴にはタプルや辞書(dictionary)で持てます。

```python
sample = ("img_0001.jpg", 1000, 12.3, 0.01)        # タプル
sample = {"image": "img_0001.jpg", "ts": 1000,
          "speed": 12.3, "steering": 0.01}          # 辞書
```

タプルは `sample[2]` の 2 が何だったか分からなくなり、辞書は `sample["sped"]` のようなタイプミスを実行時まで気づけません。どちらも「この塊にはどんな項目があり、各々の型は何か」という設計が、コードのどこにも明記されません。

dataclass はこの設計図を、読みやすい形で1か所に書きます。

```python
from dataclasses import dataclass

@dataclass
class FrameSample:
    image: str
    ts: int
    speed: float
    steering: float
```

これで `s = FrameSample("img_0001.jpg", 1000, 12.3, 0.01)` と作れ、`s.speed` と名前でアクセスでき、エディタが補完してくれ、型チェッカが `s.sped` の誤りを書いた瞬間に指摘します。「データの形」がコードとして明文化され、人にも道具にも分かるのが本質です。

`@dataclass` の `@` はデコレータ(decorator)という記法で、「このクラスに追加機能を後付けする印」です。`@dataclass` を付けると、Python が裏で `__init__` など定型コードを生成して、そのクラスに足してくれます。

## 基礎: 前提となる概念

クラス(class)とインスタンス(instance): クラスは「設計図」、インスタンスはその設計図から作った「実物」です。`FrameSample` がクラス、`s = FrameSample(...)` で作った `s` がインスタンスです。

属性(attribute): インスタンスが持つ値で、`s.speed` のようにアクセスします。

型ヒント(type hint, PEP 484): `image: str` の `: str` の部分です。「この属性は文字列のはず」という注釈で、Python の実行時には強制されません(`str` の所に数値を入れてもエラーにはならない)。代わりに、mypy や Pyright といった静的型チェッカ(コードを実行せずに型の矛盾を探す道具)や、エディタの補完がこの情報を使います。「実行前に間違いを見つける」ための情報です。

`__init__` / `__repr__` / `__eq__`: クラスが持つ特殊メソッド。`__init__` は生成時の初期化、`__repr__` は表示用文字列、`__eq__` は等しさの比較を定義します。普通は自分で書きますが、dataclass はこれらを属性定義から自動生成します。手書きの定型コードを大幅に減らせます。

## 仕組みを詳しく

`@dataclass` が自動生成するものを具体的に見ます。前述の `FrameSample` に対し、おおよそ次が自動で作られます。

`__init__`:
```python
def __init__(self, image, ts, speed, steering):
    self.image = image
    self.ts = ts
    self.speed = speed
    self.steering = steering
```
属性の順番どおりに引数を取る初期化です。手で書けば単調で間違えやすい部分が消えます。

`__repr__`: `print(s)` したとき `FrameSample(image='img_0001.jpg', ts=1000, speed=12.3, steering=0.01)` と全属性が見える表示を生成。デバッグが楽になります。

`__eq__`: `s1 == s2` を「全属性が等しいか」で判定。デフォルトのクラスでは「同じオブジェクトか(同一性)」しか比較しませんが、dataclass は「中身が同じか(値の等価性)」を比べます。

デフォルト値とフィールド設定:
```python
from dataclasses import dataclass, field

@dataclass
class TrainConfig:
    backbone: str = "resnet50"          # デフォルト値
    lr: float = 1e-4
    layers: list = field(default_factory=list)   # 可変オブジェクトは factory で
```
注意点として、リストや辞書のような「可変(mutable)なオブジェクト」をデフォルト値に直接書くと、全インスタンスで同じリストが共有される有名なバグになります。dataclass はこれを検知してエラーにし、`field(default_factory=list)`(インスタンスごとに新しいリストを作る関数を渡す)を要求します。default_factory はインスタンス生成のたびに呼ばれ、別々のリストが作られます。

凍結(frozen)による不変性:
```python
@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
```
`frozen=True` を付けると、生成後に属性を書き換えられなくなります(`obj.fx = 0` がエラー)。これを不変(immutable)と呼びます。利点は2つ。第一に、誤って書き換える事故を防げる。第二に、`__hash__` が自動生成され、辞書のキーや集合(set)の要素にできます(可変オブジェクトはハッシュできないため)。設定値やカメラパラメータのように「作った後は変えない」データに向きます。

`__post_init__`(後処理と検証):
```python
@dataclass
class BBox:
    x1: float; y1: float; x2: float; y2: float
    def __post_init__(self):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("invalid box: x2,y2 must exceed x1,y1")
```
`__init__` の直後に自動で呼ばれるメソッドで、値の妥当性チェックや派生値の計算に使います。これにより「不正な状態のオブジェクトをそもそも作れない」設計ができます。

スロット(slots)によるメモリ削減: Python 3.10以降は `@dataclass(slots=True)` が使えます。通常 Python のインスタンスは属性を辞書(`__dict__`)で持つため、属性が少なくてもオーバーヘッドがあります。slots は属性を固定枠に格納し、辞書を作らないため、メモリが減り(オブジェクトあたり数十〜百数十バイト規模)、属性アクセスも速くなります。何百万件のサンプルをメモリに持つ機械学習では効きます。代償として、定義外の属性を後から足せなくなります。

型ヒントとの連携: dataclass の属性型は前述のとおり実行時には強制されませんが、mypy などで `cfg.lr = "fast"`(float に文字列)を書くと静的チェックで弾けます。チームでの大規模コードで、データの形の取り違えを早期に潰せます。

## 手法の系譜と主要論文

dataclass は学術論文というより、Python の標準化提案(PEP, Python Enhancement Proposal)を軸に発展しました。系譜を追います。

- 名前付きタプル(`collections.namedtuple`, Python 2.6, 2008頃): タプルに名前を付けてアクセスできる先駆け。ただし不変で、デフォルト値やメソッドの扱いが不便でした。
- PEP 484(Type Hints, 2014, Guido van Rossum ら): 型ヒントの構文を言語に導入。dataclass が型注釈から構造を読み取る土台になりました。
- attrs ライブラリ(Hynek Schlawack, 2015頃〜): サードパーティ製で、属性定義から `__init__` などを自動生成する考えを普及させました。dataclass の直接の先行実装で、設計思想の多くを共有します。今も attrs の方が機能豊富な場面があります。
- PEP 557(Data Classes, 2017採択, Eric V. Smith, Python 3.7で標準搭載): attrs に着想を得て、標準ライブラリに `@dataclass` を導入。外部依存なしで型付き構造を書けるようになりました。
- `typing.NamedTuple`(Python 3.6, PEP 526併用): 型付きの不変構造。dataclass(frozen)と用途が重なりますが、タプルとしての性質を保ちます。
- Pydantic(Samuel Colvin, 2017〜): 実行時に型を検証(文字列を自動で int に変換、不正値で例外)するライブラリ。dataclass が「設計図の明文化」中心なのに対し、Pydantic は「外部入力の検証」に強く、API や設定ファイルの読み込みで多用されます。Pydantic v2 はコア部分を Rust で書き直し高速化しました。

機械学習基盤では、Hydra や OmegaConf といった設定管理ライブラリが dataclass を設定スキーマとして利用し、コマンドライン引数や YAML を型付きオブジェクトへ流し込む使い方が定着しています。

## 論文の実験結果(定量データ)

dataclass は性能を競う研究対象ではないため、論文ベンチマークではなく、定量的に意味のある計測値を挙げます。

メモリ削減(slots): 一般的な計測では、属性を数個持つインスタンスに `slots=True` を付けると、1インスタンスあたりおよそ40〜60%のメモリ削減が報告されています。例えば通常の dataclass インスタンスが約152バイトのところ、slots版は約56バイト前後といった桁感です(属性数とPythonバージョンで変動)。100万インスタンスなら数十MB〜100MB規模の差になり、メモリ制約のあるデータローダで効きます。

属性アクセス速度: slots は属性を辞書探索ではなく固定オフセットで引くため、属性の読み書きが概ね2〜3割速くなる計測が一般的です。ただしホットループ(最内側で何百万回も回る部分)でなければ体感差は小さいです。

Pydantic v2 の検証速度: Pydantic は v1(純Python)から v2(Rust コア)で、典型的な検証処理が公式ベンチマークでおよそ5〜50倍高速化したと報告されています。これは「外部からのJSONを型検証して取り込む」処理が支配的な API サーバなどで意味を持ちます。dataclass 自体は検証をしないため、この比較は「検証を伴う場合の選択肢」としての参考値です。

生成オーバーヘッド: `@dataclass` の自動生成はクラス定義時に1回だけ走るコード生成で、インスタンス生成のたびの追加コストはほぼありません。手書き `__init__` と同等の速度です。

数値の意味: これらは「設計の良さ」と「実行コスト」を両立できることを示します。可読性のために構造化しても、slots を使えばメモリ・速度面の代償をほぼ消せる、という点が実務的な要点です。

## メリット・トレードオフ・限界

メリット。第一に、定型コード(`__init__`, `__repr__`, `__eq__`)が消え、属性定義に集中できる。第二に、型ヒントにより静的チェックと補完が効き、データの形の取り違えを早期に発見できる。第三に、frozen で不変性、slots でメモリ効率を選べる。第四に、標準ライブラリなので外部依存ゼロ。第五に、デバッグ表示(`__repr__`)が自動でつき、調査が楽。

トレードオフと限界。

実行時の型強制はない: `: int` と書いても、実行時に文字列を入れて止められはしません。実行時検証が必要なら Pydantic などを併用します。dataclass は「ドキュメントとしての型」であり「ガードレールとしての型」ではない、と理解する必要があります。

可変デフォルトの落とし穴: 前述のとおりリスト・辞書を直接デフォルトにできず、`field(default_factory=...)` が必要。初学者がはまりやすい点です。

継承の複雑さ: dataclass を継承し、親にデフォルト値あり・子にデフォルト値なしの属性を足すと、引数順の制約(デフォルトなし引数はデフォルトあり引数より前)に抵触してエラーになることがあります。設計時に順序を意識する必要があります。

過剰適用: ロジック(メソッド)が主役のクラスにまで dataclass を使うと、かえって不自然になります。dataclass は「データを束ねる」のが本分で、振る舞い中心のクラスは通常のクラスが向きます。

ネストと検証の限界: 入れ子になった dataclass の深い検証や、複雑なバリデーションは dataclass 単体では書きにくく、Pydantic や attrs の出番になります。

## 発展トピック・研究の最前線

設定管理との統合: Hydra(Facebook/Meta)+ OmegaConf は dataclass を「構造化設定(structured config)」のスキーマに使い、YAML やコマンドライン上書きを型付きで受け取ります。機械学習実験の再現性(どの設定で回したか)を担保する標準的な構成です。dataclass がそのまま実験設定の単一の真実(single source of truth)になります。

型システムの進化: Python の型ヒントは年々強化され、`Literal`(取りうる値を列挙)、`Annotated`(メタデータ付き型)、ジェネリクス(`list[FrameSample]` のような型パラメータ)を dataclass 属性に使えます。これにより「backbone は resnet50 か swin のどちらかのみ」といった制約を型レベルで表現でき、静的チェックで弾けます。

不変データと関数型志向: frozen dataclass は「作ったら変えない」不変データの推奨手段で、並行処理(複数スレッド・プロセスで安全に共有)やキャッシュのキーに向きます。状態の書き換えを減らすことでバグを構造的に減らす、という設計思想と相性が良いです。

Pydantic の台頭と棲み分け: 外部入力(API・設定ファイル・LLMの出力)を型検証して取り込む用途では Pydantic が事実上の標準になりつつあります。一方、内部のデータ受け渡しで検証が不要な場面では軽量な dataclass が好まれます。「境界(外部との接点)では Pydantic、内部では dataclass」という使い分けが定着しつつあります。

## さらに学ぶための関連トピック

- [Parquet列指向ストレージ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0908_parquet-columnar-storage)
- [ピンホールカメラモデル](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0900_pinhole-camera-model)
- [センサ融合(カメラ+LiDAR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0341_sensor-fusion-camera-lidar)
