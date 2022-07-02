---
title: "TLA+チートシート" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["TLA", "PlusCal", "形式手法", "モデル検査"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---

日本語のTLA+に関する文献がほとんど見つからないので、日本語でまとめていきたいと思います。
誤った情報や追記したい情報などは[こちら](https://github.com/riita10069/zenn.dev/blob/main/articles/bc689cae1c7bc0.md)にIssueまたは、PRを送っていただけると助かります。



# TLA+の勉強法・学習法

### ドキュメントや講義など

* [Leslie Lamport's TLA homepage](http://lamport.azurewebsites.net/tla/tla.html): ホームページです。
* [Hyperbook](http://lamport.azurewebsites.net/tla/hyperbook.html): チュートリアルなどを扱ったHyperbookです。こちらのページからダウンロードが可能です。
* [TLA video course](http://lamport.azurewebsites.net/video/videos.html): TLA+の作者であるLamprtさんの動画教材です。
* [Specifying Systems](http://lamport.azurewebsites.net/tla/book.html): Leslie Lamportによるもので、出版社を通じての注文や個人使用のためのダウンロードにリンクしています。
* [Learn TLA](https://learntla.com/) もっとも詳細にTLA+及びPlusCalの文法や使い方について説明している場所です。現状唯一の情報源とも言えます。
	* [TLA+ Toolbox](https://github.com/tlaplus/tlaplus/) リリースタグよりIDEをインストールできます。ARM向けのものは開発されていないのでRosettaが必要になります。
* [Plactical TLA+](https://www.amazon.co.jp/Practical-TLA-Planning-Driven-Development/dp/1484238281): 入門者向けのTLA+の書籍です。例も豊富で[サンプルコード](https://github.com/Apress/practical-tla-plus)もついているので非常におすすめです。これは、O'Reilly online learningで無料で読めます。
* [Dr. TLA+](https://github.com/tlaplus/DrTLAPlus): アルゴリズムとTLA+の実装についての講義です。
* [TLA+ in Practice and Theory](https://pron.github.io/tlaplus): TLA+で使用されている数学と原理・哲学について4部構成での動画講義です。

### 論文など

* [Leslie Lamport papers](http://lamport.azurewebsites.net/tla/papers.html): TLA+の作者であるLamportさんが書いた論文がまとまっています。
* [AWS and TLA+](http://lamport.azurewebsites.net/tla/amazon.html): AWSでは積極的にTLA+を活用しているようです。AWSが発行したTLA+導入に関する非常に有名な論文です。
* [How to integrate formal proofs into software development](https://www.amazon.science/blog/how-to-integrate-formal-proofs-into-software-development): Automated Reasoningにおける形式手法の導入に関する論文です。これもAWSが発行しています。
* [Examples on GitHub](https://github.com/tlaplus/Examples): tlaplusのOrgで管理されているExampleのレポジトリです。あまり活発には動いていないのですが、書き方などを学ぶことができます。

# TLA+の使い方

## そもそもTLA+とは何のためのツールか

TLA+とは、並行処理や分散システムにおける仕様を記述するための言語として開発された。
システムの設計段階において、形式的な検証を行うことができるツールである。
TLA+自体は、記述されたシステムを有限オートマトンとみなして抽象化する。
そこで、簡単な状態遷移図を思い浮かべてほしい。
ある状態から次の状態への推移を一つのステップとしてTLA+は計算する。
そして、連続的な状態の変化によってできる状態の列に対して、不変条件やアサーションで論理的なチェックができる。

TLA+を用いて仕様を記述する際の難点は、システムを正確に有限オートマトンで表現し、さらに機能要件について正確に論理式や時相論理を用いて記述する必要がある。
これらをより直感的な記述で表現できるようにするのが、PlusCalである。PlusCalは手続的なシステムの記述を受け入れ有限オートマトンに変換する。

## 値と演算子

TLA+には、`文字列`、`整数`、`ブール値`、`モデルバリュー`の4種類の値がある。
また、以下の演算子が定義されている。

| 演算子 | 意味 | 例 |
| --- | --- | --- |
| `a = b` |イコール | `1 = 1`  |
| `a /= b` | ノットイコール | `1 /= 2` |
| `a /\ b` | AND | `TRUE /\  FALSE` |
| `a \/ b` | OR |  `TRUE  \/ FALSE` |
| `~a` | ノット | `~TRUE` |

## 型

TLA+には、`集合`、`レコード`、`構造体`、`関数`の4種類の型がある。

### 集合

```
set = {"a", "b", "c"}
```
のように定義する。
集合は順序を持たない要素の集まりなので、同じ要素を2つ持つことはない。

| 演算子 | 意味 | EXTENDS |
| --- | --- | --- |
| `x \in set`  |集合setの要素x|   |
| `x \notin` | 集合setの要素でないx |  |
| `set1 \subsetea set2` | set2の部分集合set1 |  |
| `set1 \union set2` | 和集合 |  |
| `set1 \intersect set2` | 積集合 |  |
| `set1 \ set2` | 差集合 |  |
| `SUBSET {"a", "b"}` | 冪集合 | `{{}, {"a"}, {"b"}, {"a", "b"}}` |
| `Cardinarity(set1)` | 要素数 | FiniteSets |

以下のように集合の変換を行うこともできる

| 演算子 | 意味 | 例 |
| --- | --- | --- |
| `{set2 \in set1: 条件式}` |フィルタリング | `{x \in 1..2: x < 3}` >> `{1, 2}` |
| `{式: set1 \in set2}` | マッピング |  `{x * 3: x \in 1..3}`>> `{3, 6, 9}` |

### レコード

レコードは順序を持つ要素のコレクションであり、1始まりのインデックスを使う。(0始まりではないので注意)
```
seq = <<1, "b", {3, 4}>>
```
のように定義できる。

`EXTENDS Sequences`を使うことで以下の演算子を使うことができる。

| 演算子 | 意味 | 例 |
| --- | --- | --- |
| `Head(seq)` | 先頭| `Head(<<1, 2, 3>>)`>>1  |
| `Tail(seq)` | 先頭以外 | `Tail(<<1, 2, 3>>)`>> `<<2, 3>>` |
| `Append(seq, x)` | 末尾に追加 | `Append(<<1, 2>>, 3)`>> `<<1, 2, 3>>`|
| `seq1 \o seq2` | 結合 | `<<1,  2>> \o <<3, 4>>`>> `<<1, 2, 3, 4>>` |
| `Len(seq)` | 長さ | `Len(<<1, 2, 3>>)`>>3 |
| `SubSeq(seq, index1, index2)` | スライス | `SubSeq(seq, 1, Len(seq) - 1)` >> `末尾の要素以外` |


### 構造体

構造体は文字列に対して、値をマッピングすることができる。
値へのアクセスは、`struct.key` でアクセスできる。
一般のプログラミング言語におけるmapに近い機能に感じる。(構造体は実体ではないため)

```
author = [name |-> "riita", age |-> 22, gender |-> "male"]
author.name
>> "riita"
```
### 関数

`[x \in 1..3 |-> x*2]` のように書くことができる。
これは、`<<2, 4, 6>>`である。


## アルゴリズム

### if

```
if 条件1 then
  body
elsif 条件2 then
  body
else
  body
end if;
```

### while

whileループ内を条件式がTRUEである限り実行し続けます。

```
while 条件式 do
    body
end while;
```

### macros

macroでは、代入、アサーション、if文などを表現できるが、while文は表現できないことに注意。

```
macro name(arg1, arg2) begin
    body
end macro;

begin
  name(x, y);
end algorithm;
```

また、関数とは違いマクロの外側にある値への参照及び代入が可能である。

### either

eitherを使用すると、TLCは全ての分岐を検査する。
このように書くことで、複数の選択肢から一つの分岐が選択されるような処理を表現できる。
この機能は普通のプログラミング言語にない機能なので、TLA+を利用する強い動機になる。

```
either
    分岐1
or
    分岐2
or
    分岐3
end either;
```

### with

withには2つの書き方があり、以下に列挙する。

```
with var = value do
    body
end with;
```

こちらの方法は、一時変数の作成に便利である。
局所的に使用される変数は、可能な限り小さなスコープで用いるべきである。

```
with var \in set do
    body
end with;
```

こちらの方法は非決定論的に検査される。
TLCは、varという変数がsetの要素であるという条件を満たす全ての状態を検査するためだ。


## 不変条件

不変条件とは、各ステップの最後に検査される条件である。


### assert 

不変条件の定義には、アサーションを使う方法がある。
TRUEの場合には、何もせず、FALSEになるとエラーを送出する。
なお、`EXTENDS TLC`が必要なことに注意。
```
begin
  assert x;
end
```

### Invariants

Toolboxを使用している場合、モデルを作成する際に[Invariants]セクションに不変条件を選択することが可能。

## 論理演算子

不変条件は論理式によって定義される。

### \A

 \Aは全称条件を定義する。
```
\A n \in 1..10: n < 5
>> FALSE
```

### \E

 \Eは存在条件を定義する。
```
\A n \in 1..10: n < 5
>> TRUE
```

### =>
`P=>Q`は PならばQ を定義する。

### <=>
`P<=>Q`は PならばQ かつQならばP を定義する。

## 式

論理式により強力な表現を与えることができる構文を紹介する。


### IF THEN ELSE

```
Max(x, y) == IF x > y THEN x ELSE y
```

### CASE

```
CASE n = 1 -> TRUE
  [] n = 2 -> TRUE
  [] OTHER -> FALSE
```

### CHOOSE

`CHOOSE x \in S : 条件式` で条件式がTRUEとなるようなxを選択する。

この仕組みは非常に強力で最大値を求める演算子の定義は以下になる。
```
Max(set) == CHOOSE x \in set: \A y \in set: x >= y
```

## 並行処理

### ラベル

ラベルは原子性の粒度を決定します。TLCはラベルを一つのステップとして実行するような処理を実現することができます。
正しいモデリングのためにいくつかのルールがあることを押さえておく必要があります。

- 各プロセスの初めと、各while文の前にはラベルが必要。
- マクロとwithの中にはラベルが配置できない。
- goto文の後にはラベルが必要。
- ラベル内で同じ変数への複数回のアクセスを行ってはいけない。
- eitherやifなどの分岐構造内にラベルを含んでいるならば、分岐後にもラベルが必要。

```
Label:
    x := 1
```

### プロセス

単一のプロセスを生成する場合

```
fair process p = "p1"
begin 
    body
end process;
```

複数のプロセスを生成する場合

```
fair process p \in 1..5
begin 
    body
end process;
```

プロセスの名前にアクセスする場合は、`self`でアクセス可能

また、`fair`は弱い公平性、`fair+`で強い公平性を持たせることが可能

### Await

`await 条件式`で、条件式がTRUEになるまで、実行を中断することができる。
DeadLockの原因になるので注意。

```
await x > 0;
```

### possibly

サーバーはいつクラッシュするかわからない。通信パケットをロスするかわからない。
失敗時の処理を記述するパターン

```
either
    Success:
        skip;
or
    Fail:
        body
end either;
```

### プロシージャ

```
procedure f(arg1) begin
    body
end procedure;
```

## 時相特性

### []

状態列において、全てが TRUE であること。

### <>

状態列において、いつか TRUE の状態が訪れる

### []<>と<>[]

`[]<>P`は、FALSEになることがあってもいつかまたTRUEになる(無限回trueになる)
`<>[]P`は、あるTRUE状態以降にFALSEに一度もならないようなTRUEが存在する

有限状態においては意味は同じだが、無限状態においては意味が異なる。

### ~>

`P~>Q`で、PとなればQという結果になる。

Pという状態列において、 TRUEになる状態があれば、Qがその状態またはそれ以降の状態においてTRUEになる。
