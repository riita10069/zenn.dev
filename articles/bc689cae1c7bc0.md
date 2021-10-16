---
title: "TLA+チートシート" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["TLA", "PlusCal", "形式手法", "モデル検査"] # タグ。["markdown", "rust", "aws"]のように指定する
published: false # 公開設定（falseにすると下書き）
---

日本語のTLA+に関する文献があまりにも見つからないので、頑張って情報を日本語でまとめていきたいと思います。
誤った情報や追記したい情報などは[こちら](https://github.com/riita10069/tlaplus-cheatsheet/blob/main/articles/bc689cae1c7bc0.md)にIssueまたは、PRを送っていただけると助かります。

# 主要な情報源

### ドキュメントや講義など
* [Leslie Lamport's TLA homepage](http://lamport.azurewebsites.net/tla/tla.html): ホームページです。
* [Hyperbook](http://lamport.azurewebsites.net/tla/hyperbook.html): チュートリアルなどを扱ったHyperbookです。こちらのページからダウンロードが可能です。
* [TLA video course](http://lamport.azurewebsites.net/video/videos.html): TLA+の作者であるLamprtさんの動画教材です。
* [Specifying Systems](http://lamport.azurewebsites.net/tla/book.html): Leslie Lamportによるもので、出版社を通じての注文や個人使用のためのダウンロードにリンクしています。
* [Learn TLA](https://learntla.com/introduction/) もっとも詳細にTLA+及びPlusCalの文法や使い方について説明している場所です。現状唯一の情報源とも言えます。
	* [TLA+ Toolbox](https://github.com/tlaplus/tlaplus/) リリースタグよりIDEをインストールできます。ARM向けのものは開発されていないのでRosettaが必要になります。
* [Plactical TLA+](https://www.amazon.co.jp/Practical-TLA-Planning-Driven-Development/dp/1484238281): 入門者向けのTLA+の書籍です。例も豊富で[サンプルコード](https://github.com/Apress/practical-tla-plus)もついているので非常におすすめです。これは、O'Reilly online learningで無料で読めます。
* [Dr. TLA+](https://github.com/tlaplus/DrTLAPlus): アルゴリズムとTLA+の実装についての講義です。
* [TLA+ in Practice and Theory](https://pron.github.io/tlaplus): TLA+で使用されている数学と原理・哲学について4部構成での動画講義です。

### 論文など
* [Leslie Lamport papers](http://lamport.azurewebsites.net/tla/papers.html): TLA+の作者であるLamportさんが書いた論文がまとまっています。
* [AWS and TLA+](http://lamport.azurewebsites.net/tla/amazon.html): AWSでは積極的にTLA+を活用しているようです。AWSが発行したTLA+導入に関する非常に有名な論文です。
* [How to integrate formal proofs into software development](https://www.amazon.science/blog/how-to-integrate-formal-proofs-into-software-development): Automated Reasoningにおける形式手法の導入に関する論文です。これもAWSが発行しています。
* [Using lightweight formal methods to validate a key-value storage node in Amazon S3](https://www.amazon.science/publications/using-lightweight-formal-methods-to-validate-a-key-value-storage-node-in-amazon-s3): S3のkey-value storageにおいてFormal Validationを導入したという論文です。これもAWSが発行しています。
* [Examples on GitHub](https://github.com/tlaplus/Examples): tlaplusのOrgで管理されているExampleのレポジトリです。あまり活発には動いていないのですが、書き方などを学ぶことができます。

# 文法について
## 値と演算子

TLA+には、`文字列`、`整数`、`ブール値`、`モデルバリュー`の4種類の値がある。
また、以下の演算子が定義されている。

| 演算子 | 意味 | 例 |
| --- | --- | --- |
| a = b |イコール | 1 = 1  |
| a /= b | ノットイコール | 1 /= 2  |
| a /\ b | AND | TRUE /\  FALSE |
| a \/ b | OR |  TRUE  \/ FALSE |
| ~a | ノット | ~TRUE |

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
| x \in set  |集合setの要素x|   |
| x \notin | 集合setの要素でないx |  |
| set1 \subsetea set2 | set2の部分集合set1 |  |
| set1 \union set2 | 和集合 |  |
| set1 \intersect set2 | 積集合 |  |
| set1 \ set2 | 差集合 |  |
| Cardinarity(set1) | 要素数 | FiniteSets |

以下のように集合の変換を行うこともできる

| 演算子 | 意味 | 例 |
| --- | --- | --- |
| {set2 \in set1: 条件式} |フィルタリング | `{x \in 1..2: x < 3}` >> {1, 2} |
| {式: set1 \in set2} | マッピング |  `{x * 3: x \in 1..3}`>> {3, 6, 9} |

### レコード

レコードは順序を持つ要素のコレクションであり、1始まりのインデックスを使う。(0始まりではないので注意)
```
seq = <<1, "b", {3, 4}>>
```
のように定義できる。

`EXTENDS Sequences`を使うことで以下の演算子を使うことができる。

| 演算子 | 意味 | 例 |
| --- | --- | --- |
| Head(seq)  | 先頭| `Head(<<1, 2, 3>>)`>>1  |
| Tail(seq) | 先頭以外 | `Tail(<<1, 2, 3>>)`>> `<<2, 3>>` |
| Append(seq, x) | 末尾に追加 | `Append(<<1, 2>>, 3)`>> `<<1, 2, 3>>`|
| seq1 \o seq2 | 結合 | `<<1,  2>> \o <<3, 4>>`>> `<<1, 2, 3, 4>>` |
| Len(seq) | 長さ | `Len(<<1, 2, 3>>)`>>3 |
| SubSeq(seq, index1, index2) | スライス | `SubSeq(seq, 1, Len(seq) - 1)` >> `末尾の要素以外` |
