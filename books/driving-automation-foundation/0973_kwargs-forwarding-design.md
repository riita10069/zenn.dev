---
title: "kwargs転送によるモジュール疎結合"
---

# kwargs転送によるモジュール疎結合

## ひとことで言うと
上位のクラスが、下位のモジュールに渡したい設定をひとまとめの辞書(Pythonの `**kwargs`)として受け取り、中身を読まずにそのまま下へ渡す設計パターンです。これにより上位クラスは「下位がどんな引数を必要とするか」を知らずに済み、部品の入れ替えや追加が本体コードを触らずにできるようになります。中身を確認せず宛先に荷物を取り次ぐ宅配便のように、設定を透過的に転送します。

## 直感的な理解
大きな機械学習モデルは、たいてい複数の交換可能な部品から組み立てられます。画像をどう融合するか、地図情報をどう取り込むか、過去の履歴をどう要約するか([Temporal Memory プラグイン](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1020_temporal-memory-plugin))、軌道をどう予測するか — それぞれに「この部品を使う」「あの部品に差し替える」という選択肢があります。

ここで現実的な悩みが生まれます。部品ごとに必要な設定(コンストラクタの引数)がバラバラなのです。ある軌道予測器は「拡散のサンプリングステップ数」を要求し、別の予測器は「アテンションのヘッド数」を要求し、ある融合方式は「BEVグリッドの解像度」を要求します。

もし組み立て役のクラスが、これら全部の引数を自分の引数リストに並べてしまったら、どうなるでしょう。引数が数十個に膨れ上がり、部品を差し替えるたびに組み立て役のコードも書き換える羽目になります。組み立て役が「すべての部品の内部事情」を抱え込み、片方を変えるともう片方も壊れる「密結合」の状態に陥ります。

この悪循環を断ち切る発想が「設定は最終的に使う部品だけが知ればよい。組み立て役は中身を見ずに取り次ぐだけにする」です。これが kwargs 転送による疎結合です。

## 基礎: 前提となる概念
用語を順に噛み砕きます。

- 結合(coupling): モジュール同士の依存の強さ。Aを変えるとBも直さないと壊れる、という関係が強いほど「密結合(tight coupling)」。依存が薄く、片方を変えても他方に波及しないほど「疎結合(loose coupling)」。疎結合なほど変更に強く、再利用しやすいソフトウェアになります。
- 凝集(cohesion): 1つのモジュール内の要素がどれだけ1つの目的に向かってまとまっているか。高凝集が良い設計の指標です。「疎結合・高凝集」は構造化設計の根本原則です。
- `*args` と `**kwargs`: Python の可変長引数の記法。`*args` は位置引数をタプルでまとめて受け取り、`**kwargs`(keyword arguments の略)はキーワード引数を辞書でまとめて受け取ります。`f(**{"a": 1, "b": 2})` は `f(a=1, b=2)` と等価で、辞書を展開して引数として渡せます。
- コンストラクタ(constructor): オブジェクト生成時に呼ばれる初期化処理。Pythonでは `__init__`。ここでどんな引数を要求するかが、その部品の「組み立てに必要な情報」を表します。
- レジストリ(registry): 文字列のキーから実装を引けるよう、名前と部品の対応表を持つ仕組み。`build_xxx(mode, **kwargs)` のように「名前で選び、辞書で設定する」使い方とセットで使われます([Temporal Memory プラグイン](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1020_temporal-memory-plugin) 参照)。

## 仕組みを詳しく
中核は Python の `**kwargs` を使った「中身を決め打ちしない辞書の透過転送」です。組み立て役のクラスは、部品ごとの設定辞書を受け取り、構築時にそのまま下位へ展開します。

```python
class Model(nn.Module):
    def __init__(self, ...,
                 view_fusion_kwargs=None,
                 map_fusion_kwargs=None,
                 temporal_memory_kwargs=None,
                 planner_kwargs=None):
        ...
        # 視点融合へ転送(BEV方式なら bev_h / bev_w / pc_range などが入る)
        self.fusion = build_fusion(..., **(view_fusion_kwargs or {}))

        # 地図融合へ転送(クロスアテンション方式なら num_heads / dropout が入る)
        self.map_fusion = build_map_fusion(
            map_fusion_mode, embed_dim=embed_dim, **(map_fusion_kwargs or {}))

        # 時間メモリへ転送
        self.memory = build_temporal_memory(
            temporal_memory_mode, visual_dim=..., egomotion_dim=...,
            **(temporal_memory_kwargs or {}))

        # 軌道予測器へ転送
        self.planner = build_planner(
            planner_mode, embed_dim=embed_dim, **(planner_kwargs or {}))
```

ポイントを分解します。

### 1. 既定は `None`、使う直前に `or {}` で空辞書に
`**(map_fusion_kwargs or {})` というイディオムは、「呼び出し側が辞書を渡せばそれを展開し、渡さなければ空辞書を展開する(=何も追加しない)」という意味です。`None` をそのまま `**` 展開すると `TypeError` になるため、`or {}` で空辞書へ変換します。

なぜデフォルト値を直接 `{}` にしないのか。Python では `def f(x={})` のように可変オブジェクトをデフォルト引数にすると、その辞書が全呼び出しで共有され、ある呼び出しでの書き換えが次の呼び出しに漏れるという有名なバグ(mutable default argument)を招きます。だから「デフォルトは `None`、使う直前に `or {}`」が定石です。

### 2. 中身を知らずに転送する
組み立て役は `map_fusion_kwargs` の中に `num_heads` が入っているか `dropout` が入っているかを一切気にしません。ただ `build_map_fusion` に丸ごと渡すだけです。何が入っているべきかを知っているのは、最終的に受け取るモジュールだけ。これが疎結合の本質です。

### 3. forward(実行時)でも同じ転送
構築時だけでなく、`forward(..., **kwargs)` で受けた追加引数を `self.planner(features, ctx, **kwargs)` のように下位へそのまま渡すこともあります。構築時の設定と実行時の条件入力の両方を透過させられます。

### 効果を具体例で見る
軌道予測器を「GRU方式」から「フローマッチング方式」に変え、後者が `num_flow_steps=10` という固有引数を要するとします。

kwargs 転送がない場合:
```
組み立て役の __init__ に num_flow_steps を追加 → シグネチャ変更 → 呼び出し側も全部修正
```
kwargs 転送がある場合:
```python
Model(planner_mode="flow_matching", planner_kwargs={"num_flow_steps": 10})
```
組み立て役のコードは1行も変えません。新しい部品の固有設定は、対応する `*_kwargs` に入れるだけ。これが「拡張に開き、修正に閉じる」状態です。

## 手法の系譜と主要論文
これは深層学習の手法ではなく、ソフトウェア設計の古典的原則です。系譜をたどります。

- 疎結合・高凝集(Constantine & Yourdon, "Structured Design", 1979)。構造化設計の中で「モジュール間の結合は薄く、モジュール内の凝集は高く」が良い設計だと体系化しました。結合度には数段階(内容結合・共通結合・制御結合・スタンプ結合・データ結合)があり、データだけを渡すデータ結合が最も疎で望ましいとされます。kwargs 転送は「上位が下位の引数の意味を解釈しない」ことで制御結合を避け、データの取り次ぎに留める実践です。

- 開放閉鎖の原則(Bertrand Meyer, "Object-Oriented Software Construction", 1988)。「ソフトウェア実体は拡張に対して開いていて、修正に対して閉じているべき」という指針です。レジストリ + kwargs 転送の組み合わせは、新しい部品を追加(拡張)できる一方で組み立て役本体を修正しないので、この原則をよく体現します。

- SOLID 原則(Robert C. Martin が2000年前後に整理、"Agile Software Development", 2002 で普及)。開放閉鎖(O)に加え、依存性逆転の原則(D, Dependency Inversion)も関係します。「上位モジュールは下位モジュールの具体実装に依存せず、抽象に依存すべき」という考えで、組み立て役が具体的な部品クラスではなく `build_xxx` という抽象的な構築関数とインターフェースに依存する点が一致します。

- 依存性の注入(Dependency Injection, Martin Fowler の2004年の解説 "Inversion of Control Containers and the Dependency Injection pattern" が有名)。モジュールが必要とするものを内部で生成せず外から渡す考え方です。`planner_mode` と `planner_kwargs` を外から渡して部品を決めるのは DI の一種で、差し替え容易性とテスト容易性(モックの注入)に効きます。

- デザインパターン(Gamma, Helm, Johnson, Vlissides, "Design Patterns", 1994、いわゆる GoF 本)。レジストリと構築関数の組は Factory Method / Abstract Factory パターンに相当し、生成の責務を本体から切り離す古典的解法です。kwargs 転送はその生成時パラメータを透過させる補助手段と位置づけられます。

共通する教訓は一貫しています。「上位が下位の詳細を知るほど変更に弱くなる。詳細は最終的に使う側だけが知ればよい」。

## 論文の実験結果(定量データ)
ソフトウェア設計原則は機械学習の精度指標のような数値で測りにくいですが、結合度・凝集度を定量化する研究はあります。

- 結合・凝集メトリクス(Chidamber & Kemerer, "A Metrics Suite for Object Oriented Design", IEEE TSE 1994)。CBO(Coupling Between Objects、あるクラスが依存する他クラス数)や LCOM(Lack of Cohesion of Methods)といった指標を定義しました。後続の実証研究(Basili et al. 1996 の学生プロジェクト群、NASA/商用コードベースの解析など)では、CBO が高い(密結合な)クラスほど欠陥確率(fault-proneness)が有意に高い、という相関が複数報告されています。報告によっては、CBO の上位グループは下位グループに比べて欠陥混入率が数倍に達する例もあり、疎結合化が保守性に効くという主張の定量的裏づけになっています。

- 経験的には、kwargs 転送のような疎結合化は「変更の波及範囲(change impact)」を縮める効果として表れます。1部品の差し替えで修正が必要なファイル数が、密結合設計では複数ファイルに及ぶのに対し、疎結合設計では呼び出し側1か所で済む、という before/after の差で語られます。具体的な精度ベンチマークは存在しないものの、保守コストの削減が定性的・定量的に観測されます。

指標の意味を補足します。欠陥密度は「1000行あたりのバグ件数」のような単位で、小さいほど品質が高い。CBO/LCOM はクラス設計の健全性の代理指標で、これらが小さい(疎結合・高凝集)ほど後の保守で壊れにくい、という関係が経験則として確立しています。

## メリット・トレードオフ・限界
メリット
- 組み立て役が下位部品の引数を知らずに済み、部品の差し替え・追加が本体無改変でできる。
- コンストラクタの引数が辞書数個に集約され、引数の爆発を防げる。
- 新しい部品を足しても、設定は対応する `*_kwargs` に入れるだけで本体に手を入れない(開放閉鎖の原則)。
- レジストリと組み合わさり、「名前で選んで辞書で設定」という一貫した使い勝手になる。
- 依存性注入の性質を持つため、テスト時にモック部品を差し込みやすい。

トレードオフ・限界
- 型チェックやIDEの補完が効きにくい。辞書の中身が間違っていても、実際に下位モジュールへ渡るまでエラーが出ず、発見が遅れる(fail-late)。
- どの kwargs にどのキーが必要かが本体のシグネチャからは読み取れず、下位モジュールの実装やドキュメントを見に行く必要がある(自己文書化されない)。
- 余分なキーを渡したときの挙動が下位次第(黙って無視か例外か)で一貫しないことがある。typo が静かに無視されると、設定したつもりが効いていない、という発見しにくいバグになる。
- 設定が呼び出し側に散らばるため、全体でどんな設定が使われているかを一覧しづらい。
- 過度に使うと「何でも辞書で渡す」設計になり、かえって追跡困難なコードになる(疎結合のやりすぎ問題)。

## 発展トピック・研究の最前線
- 型付き設定(typed config): `dataclass` や Pydantic、構造化設定ライブラリ(Hydra の structured config, OmegaConf)で、辞書の代わりに型付きオブジェクトを渡す方向。kwargs 転送の「型が緩い」弱点を補い、IDE 補完と検証を取り戻す。
- 設定管理フレームワーク(Hydra, Gin-config): 機械学習の実験で部品とハイパーパラメータを宣言的に組み立てる仕組み。レジストリ + kwargs 転送を、外部設定ファイルから駆動する形に発展させたもの。スイープ(多数のハイパラ組合せの自動探索)とも親和性が高い。
- プロトコル/構造的部分型(Python の `typing.Protocol`): 共通インターフェースを明示的な型として宣言し、疎結合を保ちつつ静的検査も効かせる近年の手法。
- 設計原則の自動検証: 結合度・凝集度を CI で計測し、密結合化を機械的に検知する静的解析ツールの発展。アーキテクチャ適合性検査([アーキ図のコード同期（Mermaid/可視化）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0722_mermaid-arch-as-code))とも接続する。

## さらに学ぶための関連トピック
- [Temporal Memory プラグイン](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1020_temporal-memory-plugin)
- [Frozen Backbone 設計](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0962_frozen-backbone-design)
- [アーキ図のコード同期（Mermaid/可視化）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0722_mermaid-arch-as-code)
- [End-to-End運転パラダイム](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1066_end-to-end-driving-paradigm)
