---
title: "CUE言語(cuelang)に入門しよう" # 記事のタイトル
emoji: "😸" # アイキャッチとして使われる絵文字（1文字だけ）
type: "tech" # tech: 技術記事 / idea: アイデア記事
topics: ["CUE", "CUE言語", "cuelang", "設定記述用言語", "Configuration"] # タグ。["markdown", "rust", "aws"]のように指定する
published: true # 公開設定（falseにすると下書き）
---



## Abstruct

本記事においては、[`The CUE Configuration Language`](https://cue.googlesource.com/cue/+/572710333821c0c97f670164dc9fc315cf00815a/README.md)を用いて設定ファイルを記述することはどのようなメリットがあるのか。また、どのような使い方が想定されるのかについて議論する。
私は、実際の開発現場において、各マイクロサービスチームがKubernetesのManifestを生成するテンプレートとして活用している。
そのため、KubernetesのManifestをcuelangを用いて記述している。(この件に関する記事は現在執筆中だ。)
その経験から、CUE言語がどのような言語であるかについてここで述べることにする。

**Keyword: CUE, cuelang, configuration, 設定記述言語, JSON, YAML, Kubernetes, manifest**

## Introduction

あなたは普段どんな設定記述言語を利用しているだろうか。すでにJSONやYAMLといった設定ファイルを利用しているだろう。APIのスキーマやKubernetesなどのあらゆるツールのコンフィグレーションに採用されている。
実際のところJSON形式はヒトにとって親しみやすい形式ではないように感じる。具体的な問題点をあげるとすれば、

- 全てに閉じかっこが必要であること
- 末尾のカンマを忘れると機能しないこと
- ネストが深くなるとインデントが大きくなること

などが挙げられる。
読みづらいのはもちろんだが、書くことも難しいだろう。

YAML形式はJSONの上記の問題点を解決することができるため、Kubernetesをはじめヒトが大規模絵な設定ファイルを記述する必要がある場合はほとんどの場合でYAMLが使われる。
実際に、かっこおよびカンマが不要でスッキリと書くことができる上で、配列や辞書などの表現力はJSONと変わらない。一見便利そうに見えるYAMLだが、実際に書いてみると問題に気づく。
それは、かっこがないためインデントに気を使う必要があることだ。スペースの数が一つでもずれたらそのYAMLは機能しない。どこが問題か見つけなければならない。

さらにこれらの設定記述形式では、型定義ができないため不必要な要素があることや必要な要素がないことを指摘できない。また、特定の要素にあまりに大きすぎる値が入っていたりした場合に指摘することもできない。JSONとYAMLは本質的にただのシリアライズでしかないのだ。そのために別の仕組みを使ってバリデーションなどの処理をしなければならない。[conftest](https://github.com/open-policy-agent/conftest)のような確立された仕組みは既にあるが、前述したファイルの生成などを行うユースケースを考えれば、Helmのようなファイル生成の仕組みとは別にconftestを導入する必要があり、問題の検出がどうしても遅くなってしまう。

そこで、CUE(cuelang)という制約ベースの設定記述言語を紹介しよう。これによって複雑な設定が管理しやすくすることができる。具体的な内容は次項以降で語ることにしよう。

## How is the cuelang
### cuelang as better json or yaml

前項で挙げたようなJSONの煩わしさを解決したのがcuelangである。
具体的な改善点としては、以下が挙げられる。

- Cスタイルのコメント
- 特殊文字なしでフィールド名から引用符を省略可能
- フィールドの最後のコンマが不要(ただしリストでは必要)
- 外側のかっこの省略が可能
- 要素が一つしかない場合はインデント不要

実際にcuelangを書いてみたものが以下である。


```cue
apiVersion: "v1"
kind:       "Service"
metadata: {
	name:      "nats-js"
	namespace: "nats-ns"
	labels: {
		"app.kubernetes.io/name":     "nats"
		"app.kubernetes.io/instance": "nats-js"
		"app.kubernetes.io/version":  "2.6.2"
	}
}
spec: {
	selector: {
		"app.kubernetes.io/name":     "nats"
		"app.kubernetes.io/instance": "nats-js"
	}
	clusterIP: "None"
	ports: [{
		name:        "client"
		port:        4222
		appProtocol: "tcp"
	}, {
		name:        "cluster"
		port:        6222
		appProtocol: "tcp"
	}, {
		name:        "monitor"
		port:        8222
		appProtocol: "tcp"
	}, {
		name:        "metrics"
		port:        7777
		appProtocol: "tcp"
	}]
}

```

cueからは、コマンドによって簡単にjsonやyamlを生成できる。
実際に生成されたjsonとyamlと比較してみることにしよう。

`cue export hoge.cue`

```json
{
    "apiVersion": "v1",
    "kind": "Service",
    "metadata": {
        "name": "nats-js",
        "namespace": "nats-ns",
        "labels": {
            "app.kubernetes.io/name": "nats",
            "app.kubernetes.io/instance": "nats-js",
            "app.kubernetes.io/version": "2.6.2"
        }
    },
    "spec": {
        "selector": {
            "app.kubernetes.io/name": "nats",
            "app.kubernetes.io/instance": "nats-js"
        },
        "clusterIP": "None",
        "ports": [
            {
                "name": "client",
                "port": 4222,
                "appProtocol": "tcp"
            },
            {
                "name": "cluster",
                "port": 6222,
                "appProtocol": "tcp"
            },
            {
                "name": "monitor",
                "port": 8222,
                "appProtocol": "tcp"
            },
            {
                "name": "metrics",
                "port": 7777,
                "appProtocol": "tcp"
            }
        ]
    }
}

```


次は、yamlに変換してみよう。
`cue export service.cue --out yaml `

```
apiVersion: v1
kind: Service
metadata:
  name: nats-js
  namespace: nats-ns
  labels:
    app.kubernetes.io/name: nats
    app.kubernetes.io/instance: nats-js
    app.kubernetes.io/version: 2.6.2
spec:
  selector:
    app.kubernetes.io/name: nats
    app.kubernetes.io/instance: nats-js
  clusterIP: None
  ports:
    - name: client
      port: 4222
      appProtocol: tcp
    - name: cluster
      port: 6222
      appProtocol: tcp
    - name: monitor
      port: 8222
      appProtocol: tcp
    - name: metrics
      port: 7777
      appProtocol: tcp
```

こうやってみると、yamlもスッキリしているように見えるが、それは空白がより重要になるような思想で作られているからだ。空白とタブの数を揃えるように努力することが良い解決策ではないだろう。その点で、多くの悩みを解決したのがcueであると言える。

### Type and Values

CUE言語の非常に画期的な仕組みを紹介する。
データ構造の中で、構造を定義することができ、制約を設定することができる。
そして、その制約を実現する仕組みは以下の2点に集約されている。

- 型と値を同様に扱う
- 全ての値はlattice(束)に順序づけられる

「型と値は別のもので同一視できるものではない」と感じるかもしれない。
CUE言語は、この異なるふたつの概念を同様に扱うことを実現している。
それは、それらを**包含関係を順序としてlatticeとみなすことだ**。

厳密に包含関係をどのように順序に対応させるかは、半順序関係≤を用いて以下のようになる。

$$
a ∧ b = a \Leftrightarrow a ≤ b
$$


また、latticeとは何かは以下の3つの条件を満たすものと定義できる。

- 半順序集合である
- 二元の結びの存在
    - 半順序集合$P$において、$∀x,y ∈ P$に対して、最小上界$sup${$x, y$}が存在する
    - 平易に言い換えれば、任意の二要素に対して、一意の最小上界を持つ
- 二元の交わりの存在
    - 半順序集合$P$において、$∀x,y ∈ P$に対して、最大下界$inf${$x, y$}が存在する
    - 平易に言い換えれば、任意の二要素に対して、一意の最大下界を持つ。


包含関係を順序として導入したものが、本当にlatticeであるのかを疑問に思うものが多いであろう。納得できるように簡単に説明したいと思う。

半順序集合であることについて考える。
型と値の集合に対して、反射律、推移律、反対称律が成り立つことは定義から明らかである。
一方で、異なる一元集合(例えば、{int}, {string})の間には包含関係はない。そのため、完全律は成り立たない。よって、全順序集合ではない。

次に、最小上界および最大下界の存在について考える。
まずは、上限定理により数値型においては、最小上界および最大下界が存在することが言える。

CUEが扱う集合は、string型やlist型など数値型だけに留まらないが、これらの集合に順序関係を導入したところで、最大値および最小値が存在する。
そのため、最小上界および最大下界も存在することがわかり、CUE言語の扱う集合全体がlatticeとみなせることがわかる。


:::details 上限定理
Rの空でない部分集合が、上に有界ならば、上限が存在する
同様に、Rの空でない部分集合が、下に有界ならば、下限が存在する
:::

![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/309507/32c04458-f8f3-0d7e-caf7-ffcd063968ec.png)
([公式ドキュメント](https://cuelang.org/docs/concepts/logic/)から引用)

また、これまでの議論によって、二元の結びおよび交わりが常に存在することがわかっている。
そのため、上図のように、常に一つの根と一つの葉が存在することがわかる。

この性質によって、**2つの構成は順序に関係なく、常に明確にマージすることができる**。
これこそがまさに、CUE言語が体現したかった哲学である。

kustomizeのような従来の構成管理技術は、**継承**およびオーバーライドを行うことで構成に柔軟さを持たせてきた。
この**継承**の考え方を使って汎用性を高めることは、大きく保守性を損なうことにつながることは多くのプログラマーが痛感していることでもある。(実際に、Go, Rust, Kotlinなどの現代のプログラミング言語では、継承を排除している。)
kustomizeの場合、マージの際に手続き的な処理を記述する必要がある。どちらをどちらにマージするかが逆になれば、全く逆の構成が出来上がってしまう。
一方、cuelangの哲学では、手続き的な処理は不要で、順序に関係なく、宣言的にマージをすることができる。

そして、この型と値を同一視する手法のもう一つのメリットは、型レベルでしか形式的な検査ができなかったものをより厳密に検査できることだ。

```
// ServiceType
#ServiceType: *"ClusterIP" | "ExternalName" | "NodePort" | "LoadBalancer"
```

上記は、KubernetesのServiceのTypeを受け入れるstructだ。
本来であれば、たった4種類の文字列しか許容しないものを、型付けで検証できるのは、`string`であるかというだけである。
cuelangであれば上記のように、4種類以外のものを受け入れることはない。

実際にこの制約をServiceマニフェストに適用する方法は以下のように記述することである。

```cuelang
#Service: {
	name: string
	expose: [Name=_]: #Port & {name: Name}
	selector: [string]: string
	type: #ServiceType
}

#ServiceType: "ClusterIP" | "ExternalName" | "NodePort" | "LoadBalancer"

#Port: {
	name: string
	port: < 65536
	protocol: #Protocol
	targetPort: < 65536 | *port
}

#Protocol: "tcp" | "udp"

// expression of combination
service: #Service & {
  name: "cue-svc"
  ...
}

```

CUE言語においては、構造体側で値の制限を記述し、実際のオブジェクトは`&`を用いてその制限を受け入れるように記述する。
この式において、serviceは、`#Service`と全く同じプロパティを持つことを意味している。
`#Service`の制限を受け入れていないプロパティや、不完全なプロパティがあれば、CUE言語は検出することができる。


### Generate configuration

cuelangは、Default Valueの設定をサポートしている。これは、フィールドに特定の値が提供されていない場合、デフォルトを指定できるようにする機能である。
つまり、意図的に変更したい箇所以外は、デフォルトの設定を使用することができるのだ。

```
#ServiceType: *"ClusterIP" | "ExternalName" | "NodePort" | "LoadBalancer"
```

ここで、*は、特に指定がない場合、ポートのデフォルトが8080であることを示す。

似たようなものは似たような設定を共有する傾向があるので、デフォルトと同じ値を指定する必要はないだろう。結果的に、記述すべきConfigの内容を非常に小さくすること(もしくは0にすること)ができる。

### Golang Integration

CUEはGo言語で記述されており、いくつかの画期的なインテグレーションを提供している。

まず、一つ目が、GolangからCUE言語の高機能を利用することができる点だ。

```golang
const config = `
msg:   "Hello \(place)!"
place: string | *"world" // "world" is the default.
`

var r cue.Runtime

instance, _ := r.Compile("test", config)

str, _ := instance.Lookup("msg").String()
```

上記のように、cuelangで定義されたものをパースし、Lookupをすることができる。
さらに、

```
merged := cue.Merge(articleIns, analyzerIns)
err = merged.Value().Validate()
if err := nil {
  os.Exit(1)
}
```

このように、　マージやValidationについてもGo側から実行することができる。
つまり、Goでロジックを書く際にもCUE言語の画期的な機能を利用することができると考えられる。
例えば、mDBやAPIのスキーマもcuelangで管理することでリクエストに対して厳密なバリデーションを実装できるかもしれない。

もう一つの強力な機能が、GoパッケージからCUE定義をダウンロードする機能である。

```
cue get go k8s.io/api/core/v1
```

以下のようなコマンドを使うことで、Goのstructの型定義をcuelangの形式で自動生成してくれる。
自動で生成されたファイルと自分で定義したファイルをマージすることで、型情報以上のより厳しい制約を儲けることができる。
ここでも、マージされる順番に依らないことの恩恵を受けていることに注意してほしい。
(もしも、上書きができるとしたら、Goから読み込んだ設定ファイルの恩恵を受けることができない)

## Discussion & Conclusion

ここでは、CUE言語の必要性と、初歩的な機能のいくつかを紹介してきた。
より詳細な機能などは、[cuetorials](https://cuetorials.com/)を参照していただきたい。このサイトが最もわかりやすくcuelangの使い方を解説しているように感じた。

これまでの議論をまとめると、

- ヒトが読み書きをしやすい(よりシンプルで、文法チェックが容易)
- 2つの構成は順序に関係なく、常に明確にマージすることができる
- 強い制約をかけることができる
- プログラミング言語向けのパーサーがある

これらの特徴を兼ねそろえたcuelangは、次世代の構成管理ツールなのだ。
それでも、リリースから数年経つものの実戦投入された事例はあまり見当たらない。Google検索をしても実践投入に向けて、十分な情報が手に入る状態ではない。

何から始めていけばいいのだろうか。
cuelangを使用している事例はあまり見つからないのもあり、全ての構成ファイルをcuelangに移行することは現実的ではない。
より小さいスコープで密かにcuelangを使い始めることもできると私は考える。

それは、既存の設定を検証することから始めることだ。
CUEは、YAMLやJSONを簡単にインポートできる。簡単な定義を書けば、インポートしたYAMLやJSONに意味的な誤りがあるかどうかを確かめてくれる。
そして、CUEはProtobufファイルやGoパッケージ（Kubernetesなど）からスキーマをインポートできるので、そもそも書く必要のない定義も多く存在するのだ。
だから、cuelangを用いたポリシーの作成は本当に簡単に始められる。

次に、参照を使用したり、デフォルトを設定したり、コンフィグを生成したりすることで、これまで必要だった多くのconfigurationを削減することができる。

少しづつcuelangの効果が表れてくるはずだ。

## References

- https://cuetorials.com/introduction/
- https://speakerdeck.com/riita10069/learn-cue
- https://cue.googlesource.com/cue/+/572710333821c0c97f670164dc9fc315cf00815a/README.md
- https://github.com/helm/helm
- https://github.com/GoogleContainerTools/kpt
- https://cuelang.org/docs/concepts/logic/
- https://qiita.com/tkosht/items/a2e100fec6e9fd7ddfaa
- https://qiita.com/riita10069/items/1c9077657fcd62843aaf

