---
title: "nn.Module.__getattr__ による mypy 型崩れ"
---

# nn.Module.__getattr__ による mypy 型崩れ

## ひとことで言うと
PyTorch でニューラルネットを書くときに継承する基底クラス `nn.Module` には、属性アクセスを横取りする特殊な仕掛けがあります。その副作用として、静的型チェッカ(mypy)はモジュールの属性を見たときに「これは `Tensor` か `Module` のどちらか」としか推論できなくなります。実際にはそれ以外の型(損失関数・設定オブジェクト・外部ライブラリのメタ情報など)を属性に持たせていると、mypy が誤って型エラーを出します。これを直す定石は、クラス本体に型注釈を書いて mypy に正しい型を教える(自分が型を知っている場合)か、いったん `Any` 型に束ね直して局所的に型チェックを止める(外部ライブラリ由来で型が書きにくい場合)、の2通りです。

## 直感的な理解
普段書く Python なら `self.foo = something` と書けば、`self.foo` の型はそのまま `something` の型です。型チェッカはそれを追えます。ところが `nn.Module` は、属性の「登録」と「取り出し」を普通の代入とは別の経路で行っているため、型チェッカからすると `self.foo` が何の型なのか分かりにくくなります。

例えるなら、普通のロッカーは「自分で鍵をかけて自分で開ける」ので中身が分かりますが、`nn.Module` のロッカーは「受付係(`__getattr__`)が代わりに出し入れする」仕組みで、その受付係が「私が渡せるのは Tensor か Module だけです」と宣言しているようなものです。実際にはロッカーには関数や設定辞書が入っているのに、受付係の宣言だけを見た型チェッカは「Tensor か Module しか出てこない」と思い込み、関数として呼ぼうとすると「Tensor は呼べない」と怒ります。これがこの落とし穴の正体です。

なぜこれが実務で問題になるかというと、多くのプロジェクトは CI(継続的インテグレーション、変更のたびに自動でチェックを走らせる仕組み)で `mypy` を実行し、型エラーが1つでもあるとビルドを失敗扱いにするからです。コードは正しく動くのに型チェックだけが赤くなり、マージできない、という状況が起きます。

## 基礎: 前提となる概念
登場人物を順に噛み砕きます。

静的型チェック (static type checking) とは、プログラムを実行せずにソースコードを読むだけで「型の食い違い」を見つける検査です。「型」とはその値が何者か(整数・文字列・関数など)を表すラベルで、例えば文字列と整数を足そうとすると実行時にエラーになりますが、静的型チェックはそれを実行前に指摘してくれます。Python は本来「動的型付け言語」で型を宣言しなくても動きますが、PEP 484(2014年、Guido van Rossum らが策定)で型ヒント(type hints)の文法が標準化され、後付けで型を書けるようになりました。

mypy とは、その型ヒントを読んで静的型チェックを行う最も普及したツールです。Jukka Lehtosalo らが開発しました。`def f(x: int) -> str:` のような注釈を読み、矛盾があれば警告します。

`nn.Module` とは、PyTorch でニューラルネットを定義するときに継承する基底クラスです。`self.conv = nn.Conv2d(...)` のように子モジュールやパラメータ(学習で更新される重み)を属性として登録すると、`nn.Module` がそれらを内部の辞書(`_parameters`, `_buffers`, `_modules`)に格納します。そして `model.conv` のように取り出すとき、Python の通常の属性探索で見つからなかった場合に呼ばれるフォールバック関数 `__getattr__` が、これらの内部辞書から該当物を返します。

`__getattr__` とは、Python の特殊メソッドの一つで、「通常の属性アクセスで属性が見つからなかったときだけ」呼ばれる関数です。`obj.attr` を評価するとき、Python はまずインスタンスの `__dict__`、次にクラス階層を探し、それでも見つからなければ `__getattr__('attr')` を呼びます。PyTorch はこれを使って、内部辞書に登録した parameter / buffer / submodule を透過的に取り出せるようにしています。

型スタブ (type stub) とは、ライブラリの「型情報だけを書いた別ファイル」(拡張子 `.pyi`)です。実装本体には型注釈が無くても、スタブを用意すれば型チェッカはそこから型を読み取れます。PyTorch が配布する `nn.Module` のスタブでは、`__getattr__` の返り値が次のように宣言されています。

```python
def __getattr__(self, name: str) -> Tensor | Module: ...
```

この宣言が問題の発生源です。mypy はモジュールの属性アクセスでこのスタブを採用するため、ユーザが登録したどんな属性も `Tensor | Module`(Tensor または Module のどちらか、という「ユニオン型」)だと推論してしまいます。

## 仕組みを詳しく
なぜスタブがこう書かれているかを理解すると、対処法も腑に落ちます。`__getattr__` は「内部辞書から何でも取り出しうる」汎用フォールバックなので、型として正確に書こうとすると本来は「何でもありうる」しかありません。とはいえ実際に取り出されるのはほぼ Tensor(parameter / buffer)か Module(submodule)なので、スタブ作者は `Tensor | Module` という現実的な妥協を選びました。これは「だいたい合っているが、それ以外の属性には嘘になる」近似です。

ここで決定的に重要なのが、mypy の属性解決の順序です。mypy は `self.attr` を見たとき、まずクラス本体に書かれた注釈やクラス変数を探し、それが見つからなければ最後の手段として `__getattr__` の返り値型を採用します。つまり、クラス側で明示的に型を書いておけば `__getattr__` のスタブより優先されるのです。これが対処法1の根拠です。

ケース1: クラスレベル注釈で型を確定させる

典型的に問題になるのは、損失関数オブジェクトのような「呼び出せる(callable)が Tensor でも Module でもなく見える」属性です。

```python
class TrajectoryImitationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()        # 損失関数(中身は nn.Module)
        self.temporal_weights = torch.tensor(...)  # 重みテンソル

    def forward(self, pred, target):
        return self.loss_fn(pred, target)        # ← mypy が「Tensor not callable」と怒る
```

`self.loss_fn` の正体は `nn.SmoothL1Loss`(これも `nn.Module` のサブクラス)なので本当は呼び出せます。しかし mypy は `__getattr__` の宣言を見て「`Tensor` かもしれない」と推論し、`self.loss_fn(...)` を「Tensor は呼べない」とエラーにします。

修正は、クラス本体(`__init__` の外、クラス直下)に型注釈を書くことです。PEP 526(2016年)で導入された「変数注釈(代入を伴わない型宣言)」の文法を使います。

```python
class TrajectoryImitationLoss(nn.Module):
    loss_fn: nn.Module
    temporal_weights: torch.Tensor

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SmoothL1Loss()
        self.temporal_weights = torch.tensor(...)
```

`loss_fn: nn.Module` という宣言があると、mypy はこれを `__getattr__` より優先採用し、「これは Module(=callable)だ」と理解して `self.loss_fn(...)` のエラーが消えます。決定的に重要なのは、これは型ヒントを足しただけで実行時の挙動は一切変わらない点です(注釈は実行時には無視される)。リスクの低い純粋な型修正です。

ケース2: `Any` 経由でバインドする

外部ライブラリがモジュールに生やす属性は、こちらで正確な型を書くのが難しいことがあります。例えば画像認識バックボーンを提供する `timm` ライブラリは、`feature_info`(各ステージのチャンネル数などのメタ情報リスト)を提供しますが、これも `__getattr__` 経由のため mypy には `Tensor | Module` に見えます。

```python
self._feature_channels = [
    stage["num_chs"] for stage in self._backbone.feature_info  # ← union-attr エラー
]
```

mypy は `self._backbone.feature_info` を `Tensor | Module` と推論し、「Tensor や Module は for で反復(iterate)できない」「`Tensor` に `feature_info` という属性は無い」(union-attr エラー)と文句を言います。

修正は、いったん `Any` 型の変数に束ね直すことです。

```python
from typing import Any

backbone: Any = self._backbone
self._feature_channels = [
    stage["num_chs"] for stage in backbone.feature_info
]
```

`Any` は mypy にとって「型チェックを放棄する型」です。`Any` 型の値に対してはどんな属性アクセスも反復も呼び出しも許され、エラーになりません。`backbone: Any` とすると、それ以降 `backbone.feature_info` の型チェックは行われず、エラーが消えます。

2つの使い分けの原則:

- 自分が正確な型を知っている属性(loss_fn は Module、temporal_weights は Tensor)→ クラスレベル注釈で正確に教える。型安全を保ちつつ将来の誤用も防げる。
- 外部ライブラリ由来で正確な型を書きにくい属性(timm の feature_info)→ `Any` で局所的にチェックを外す。妥協だが現実的。ただし範囲を最小限(その式の周辺だけ)に留め、`Any` がコード全体に伝播しないようにする。

数値例で型推論の前後を整理すると次のようになります。

```
属性             実際の型           注釈なし(mypy推論)   注釈あり/Anyバインド後
loss_fn          nn.SmoothL1Loss    Tensor | Module       nn.Module   → callable OK
temporal_weights torch.Tensor       Tensor | Module       torch.Tensor → OK
feature_info     list[dict]         Tensor | Module       Any         → iterate OK
```

## 手法の系譜と主要論文
これは特定の学術手法ではなく、漸進的型付けという理論と、PyTorch という具体的ライブラリの実装事情が交差して生まれる、よく知られた工学上の落とし穴です。背景の系譜を辿ります。

漸進的型付け (gradual typing) は、Jeremy Siek と Walid Taha が "Gradual Typing for Functional Languages"(Scheme Workshop, 2006)で定式化した枠組みです。動機は、動的型付け言語の柔軟さと静的型付け言語の安全さを両立させること。アイデアは「型注釈を書いた部分は静的にチェックし、書かない部分(あるいは `Any`/`dynamic` 型を付けた部分)は実行時チェックに委ねる」という、型あり・なしを同一プログラム内で混ぜられる体系です。`Any` はこの理論でいう「動的型」に相当し、「ここは静的チェックの対象外にする」という明示的なエスケープハッチ(逃げ道)として設計されています。ケース2の `Any` バインドは、まさにこの理論が想定した正当な使い方です。

Python への導入は PEP 484(Type Hints, 2014, Guido van Rossum・Jukka Lehtosalo・Łukasz Langa)が起点で、関数引数・返り値の型注釈と `typing` モジュール(`Any`, `Optional`, `Union` など)を標準化しました。続く PEP 526(Variable Annotations, 2016)が「`x: int` のような代入を伴わない変数注釈」を導入し、クラス本体に属性の型だけを宣言できるようになりました。ケース1のクラスレベル注釈はこの PEP 526 の文法に依存しています。さらに PEP 604(2019)で `Tensor | Module` という `|` 記法のユニオン型が書けるようになり、まさに問題のスタブの書き方になっています。

PyTorch 側の事情は、python/typeshed や pytorch/pytorch の issue トラッカで繰り返し議論されてきました。「`nn.Module.__getattr__` が `Tensor | Module` を返すため、ユーザ定義属性の型が崩れる」という問題は、PyTorch に静的型を導入していく過程で広く認識され、PyTorch 公式のコード例でもクラスレベル注釈で型を明示する書き方が増えています。スタブの返り値をより広い型(`Any` に近いもの)にする案も議論されましたが、それでは「本来 Tensor として扱える属性」の型情報まで失われ、別の不便が生じるため、近似 `Tensor | Module` が残っている、という経緯があります。

## 論文の実験結果(定量データ)
この話題は実験で性能を測る種類の手法ではないため、論文のベンチマーク数値は存在しません。代わりに、型システム研究で測られる定性・定量的な観点を挙げます。

漸進的型付けの研究では「健全性 (soundness)」と「漸進性の保証 (gradual guarantee)」がしばしば形式的に証明される対象です。Siek らの後続研究 "Refined Criteria for Gradual Typing"(Siek, Vitousek, Cimini, Boyland, SNAPL 2015)は、型注釈を足しても/外しても、型エラーを除けばプログラムの意味が変わらないこと(gradual guarantee)を満たすべき基準として定式化しました。ケース1の「注釈を足しても実行時挙動が変わらない」という性質は、この保証の実務的な現れです。

`Any` のコストを測る研究もあります。漸進的型付けの実装(特に実行時に型を強制する soundな方式)では、`Any` と具体型の境界を値が通過するたびに実行時チェックが入り、これが性能を大きく劣化させうることが Takikawa らの "Is Sound Gradual Typing Dead?"(POPL 2016)で定量的に示されました。報告では、最悪のケースで実行時間が数倍から数十倍に悪化する構成があったとされます。ただし mypy は「optional typing」(注釈は静的チェックのみに使い、実行時には一切強制しない)方式なので、この実行時コストは発生しません。mypy における `Any` のコストは性能ではなく「バグ検出力の低下」という形で現れます。

## メリット・トレードオフ・限界
メリット:
- CI の mypy を通せるようになり、ビルドが緑に戻る。
- クラスレベル注釈は型を正確化するので、将来その属性を誤った型として使うコードも静的に弾ける。
- いずれの対処も型ヒントの追加・変数バインドの追加だけで、実行時の挙動を一切変えない(リグレッションのリスクが極めて低い)。

トレードオフ・限界:
- `Any` への束ね直しは、その変数に対する型チェックを完全に放棄するので、本物のバグ(存在しない属性へのアクセス、引数型の取り違え)も見逃しうる。範囲を最小限に留めるのが鉄則で、関数全体や戻り値を `Any` にすると `Any` が伝播して型安全が広範囲に崩れる。
- `nn.Module` を継承するクラスでは、Tensor / Module 以外の属性を増やすたびにクラスレベル注釈が必要になりがちで、ボイラープレート(定型的な記述)が増える。
- 根本原因はライブラリのスタブ設計にあるため、PyTorch の型スタブが将来改善されれば、これらの回避策は不要になり陳腐化しうる。回避策をコメントなしで残すと、後任者が「なぜこの注釈/`Any` があるのか」を見失う。

未解決の課題として、`__getattr__` を多用する動的なライブラリ全般(ORM、設定フレームワーク、プラグイン機構など)で同種の型崩れが起きます。Python の型システムには `__getattr__` の返り値を「キーごとに型を変える」ように記述する一般的な手段がなく、`Any` か個別注釈かの二択に追い込まれがちです。これは漸進的型付けの表現力の限界の一例です。

## 発展トピック・研究の最前線
さらに学ぶ方向をいくつか挙げます。

`Protocol`(PEP 544, 構造的部分型)を使うと、「`feature_info` を持つ何か」という形だけのインターフェースを定義し、`Any` より安全に外部オブジェクトを型付けできる場合があります。`Any` で全部諦める前に、必要な属性だけを持つ Protocol を切る選択肢があります。

`TYPE_CHECKING` ガードや `cast()` も実務でよく併用されます。`typing.cast(SomeType, value)` は「mypy にはこの型だと思わせるが実行時は何もしない」ので、`Any` バインドより意図が明確になることがあります。

型チェッカの多様化も最近の流れです。mypy 以外に Microsoft の Pyright(VS Code の Pylance の中核)、Meta の Pyre、Astral の Ty(Rust 実装、高速)などがあり、`__getattr__` の扱いや厳格さが少しずつ異なります。同じコードでもチェッカによってエラーの出方が変わるため、プロジェクトでどのチェッカを CI に採用するかは設計判断になります。

PyTorch 側でも `torch.compile` / TorchDynamo の普及に伴い静的解析の重要性が増しており、型情報をより正確に保つ動きがあります。`nn.Module` の属性型問題は、動的言語に静的型を後付けする際に避けて通れない一般的課題として、今も改善が続いています。

## さらに学ぶための関連トピック
- [pytest マーカによるテスト階層化](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1201_pytest-marker-tiering)
- [モックバックボーンによるテストfixture](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1200_mock-backbone-test-fixture)
- [ラスタ地図エンコーダ (Raster Map Encoder) — 地図画像を鳥瞰特徴へ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1005_raster-map-encoder)
- [地図融合 (Map-BEV Fusion) — ナビ地図を鳥瞰特徴へ重ねる](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0977_map-bev-fusion)
