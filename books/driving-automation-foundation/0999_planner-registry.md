---
title: "プランナーレジストリ (コンポーネント登録機構)"
---

# プランナーレジストリ (コンポーネント登録機構)

## ひとことで言うと
軌道を計画する部品(プランナー)を、`"bezier"` や `"flow_matching"` といった「名前(文字列)」で選んで作れるようにする仕組みです。中身は「名前 → クラス」の辞書(レジストリ)と、名前と設定を受け取って対応するオブジェクトを組み立てて返す関数(ファクトリ)の組です。新しいプランナーを作ったら辞書に1行登録するだけで、それを使う側のコード(モデル定義・学習スクリプト・評価コード)を一切書き換えずに名前指定で差し替えられます。機械学習フレームワークでバックボーンやヘッドを設定ファイルの文字列だけで切り替える、あの仕組みの最小版です。

## 直感的な理解
レストランの注文を思い浮かべてください。客は「ハンバーグ」と名前を言うだけで、厨房が対応する料理を作って出してくれます。客がレシピや調理器具を知る必要はありません。新メニューが増えても、客は新しい名前を言うだけ。厨房側がメニュー表に1行足すだけで対応できます。

レジストリはこの「メニュー表(名前→料理)」と「注文を受けて作る係(ファクトリ)」です。プランナーを使う側(モデルや学習コード)は「bezier ください」と名前を渡すだけで、対応するプランナーが出てきます。新しいプランナーを開発したら、メニュー表に1行追加するだけ。使う側のコードは無傷です。これがないと、プランナーを選ぶ場所に長い if/elif の分岐が並び、しかもその分岐がモデル本体・学習・評価・テストと複数箇所に重複し、新規追加のたびに全部を漏れなく直す羽目になります。

## 基礎: 前提となる概念
言葉を噛み砕きます。

- プランナー (planner): モデルの末端で「これからどう動くか(軌道)」を出す部品。同じ目的でも複数の方式があります(例: ベジェ曲線で滑らかに出す [ベジェ曲線トラジェクトリプランナー](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0116_bezier-trajectory-planner)、生成モデルで多様に出す [Flow Matching トラジェクトリプランナ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0122_flow-matching-planner))。
- レジストリ (registry, 登録簿): 「名前 → クラス」を対応づけた辞書。名前を渡すと対応するクラス(設計図)を取り出せます。クラスはまだ実体(インスタンス)ではない点に注意。
- ファクトリ (factory, 生成関数): 名前と設定(引数)を受け取り、対応するクラスから実体を組み立てて返す関数。よく `build_planner` のような名前で書かれます。
- インスタンス化 (instantiation): クラス(設計図)から実際に動くオブジェクト(実体)を作ること。`BezierPlanner(...)` のように呼ぶと実体ができます。
- 結合度 (coupling): コードの部品どうしが互いをどれだけ知っているか。あちこちが具象クラス名を直接知っていると結合度が高く、1つ変えると芋づる式に直す必要が出ます。レジストリは結合度を下げる道具です。

なぜこの仕組みが要るのか。レジストリなしで素朴に書くと、こうなります。

```python
# レジストリが無いと…(悪い例)
if planner_mode == "bezier":
    planner = BezierPlanner(embed_dim=256, num_controls=5)
elif planner_mode == "flow_matching":
    planner = FlowMatchingPlanner(...)
elif planner_mode == "gru":
    planner = GRUPlanner(...)
else:
    raise ValueError(...)
```

この分岐が、モデル本体・学習コード・ベンチマーク・テストと複数箇所に重複します。新しいプランナーを足すたびに全箇所を直さねばならず、直し忘れるとそこだけ古い構成のまま残り、原因の分かりにくいバグになります。レジストリはこの「名前→クラス」の知識を1か所に集約し、追加を局所化します。

## 仕組みを詳しく
典型的な実装は、モジュールの初期化ファイルに辞書とファクトリを置く形です。

```python
from .base import BasePlanner
from .flow_matching_planner import FlowMatchingPlanner
from .bezier_planner import BezierPlanner

PLANNER_REGISTRY = {
    "flow_matching": FlowMatchingPlanner,
    "bezier": BezierPlanner,
}

def build_planner(planner_mode, **kwargs):
    if planner_mode not in PLANNER_REGISTRY:
        raise ValueError(
            f"Unknown planner_mode {planner_mode!r}. "
            f"Available: {sorted(PLANNER_REGISTRY)}."
        )
    return PLANNER_REGISTRY[planner_mode](**kwargs)
```

`PLANNER_REGISTRY` は「名前 → クラス」の辞書。`build_planner("bezier", embed_dim=256, num_controls=5)` と呼ぶと、辞書から `BezierPlanner` クラスを引き、`**kwargs`(残りのキーワード引数)をそのままコンストラクタに渡して実体を返します。

噛み砕いた流れ。

1. 名前で辞書を引く。`PLANNER_REGISTRY["bezier"]` → `BezierPlanner` クラス本体(まだ実体ではない、設計図)。
2. 知らない名前なら親切なエラー。`"unknown"` を渡すと `Unknown planner_mode 'unknown'. Available: ['bezier', 'flow_matching'].` のように、選べる候補をソートして列挙して教えます。タイプミスにすぐ気づけます。
3. `**kwargs` を素通しでコンストラクタへ。`build_planner` 自身は引数の中身を解釈せず、選ばれたプランナーに丸投げします。

この「kwargs 素通し」には設計上の注意点があります。レジストリは引数を事前にふるい分けしないため、選んだプランナーが受け付けない引数を渡すと、そのプランナーのコンストラクタで初めて `TypeError` になります。事前バリデーションが効かないのは弱点ですが、レジストリを薄く保ち、各プランナーが自分の必要な引数だけを宣言する(共通の `BasePlanner` 契約で揃える)ことで、実用上は問題になりにくくしています。各プランナーが共通インターフェース([プランナーの出力契約 (Planner Output Interface)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0921_base-planner-contract))を実装していれば、使う側はどのプランナーを選んでも同じ呼び出し方ができます。

新しいプランナーを足すときの手順は2ステップだけです。

1. `MyPlanner` クラスを作る(共通の `BasePlanner` を継承し、決められたメソッドを実装)。
2. `PLANNER_REGISTRY["my_planner"] = MyPlanner` を1行足す。

これだけで `build_planner("my_planner", ...)` で作れるようになり、モデルや学習スクリプトの選択コードは一切触りません。これがソフトウェア設計の原則「Open-Closed Principle(拡張に対して開き、修正に対して閉じている)」の具体例です。

レジストリには発展形があり、デコレータで自動登録する流儀もよく使われます。`@register_planner("bezier")` をクラス定義の上に付けると、定義と同時に辞書へ登録される、という書き方です。これは登録漏れを防ぎ、登録と定義を一箇所にまとめられる利点があります(OpenMMLab の `@MODELS.register_module()` がこの形)。

歴史的には、自己回帰デコーダ(GRU = ゲート付き回帰ユニットで1点ずつ軌道を出す方式)を選択肢に含む構成から、リアクティブな(逐次状態を持たず一括で出す)設計へ移行する過程で、登録簿の中身が入れ替わることがあります。レジストリの利点は、こうした「選択肢の世代交代」が辞書の編集だけで完結し、使う側のコードに波及しない点にあります。

## 手法の系譜と主要論文
レジストリは特定の機械学習論文の「手法」ではなく、ソフトウェア工学の設計パターンを ML フレームワークに適用した実装慣習です。系譜をたどります。

- Gamma, Helm, Johnson, Vlissides (1994), "Design Patterns: Elements of Reusable Object-Oriented Software"(いわゆる Gang of Four)が Factory Method / Abstract Factory パターンを定式化しました。主張は「生成するクラスをコードに直接書かず、生成を専用の関数/オブジェクトに委ねよ」。動機は具象クラスへの依存を減らして差し替えを容易にすること。効果は拡張に強い設計。トレードオフは間接層が増え、初見のコードが追いにくくなること。レジストリ+ファクトリはこのパターンの実用形です。
- ML フレームワークでの普及。OpenMMLab の mmcv/mmdetection は `Registry` クラスと `@register_module()` デコレータで、バックボーン・ネック・ヘッド・データセット・最適化器を文字列キーで管理し、設定ファイル(辞書)だけでモデル全体を組み立てます。Wu et al. (2019) の Detectron2 も同様のレジストリ機構を持ちます。Hugging Face transformers の `AutoModel`/`AutoTokenizer` は、モデル名(文字列)から適切なクラスを解決する大規模なレジストリの一種です。
- 動機は一貫して同じ。多数のバックボーン/ヘッド/プランナーを、設定ファイルやコマンドライン引数の文字列だけで切り替え、実験(ablation, ある要素を抜く比較)を量産しやすくするため。研究では「同じ学習コードで部品だけ差し替えて性能を比較する」ことが極めて頻繁なので、レジストリは事実上の標準装備になっています。

## 論文の実験結果(定量データ)
レジストリ自体に精度の数値はありませんが、それが支える「設定駆動の実験量産」がもたらす実利は、フレームワーク論文やベンチマークで間接的に表れます。

- mmdetection / Detectron2 は、レジストリと設定ファイルによって数十種類の検出器(Faster R-CNN, RetinaNet, Mask R-CNN, DETR 等)を共通コードで再現し、COCO ベンチマーク(物体検出の標準データセット、指標は mAP = mean Average Precision、検出の正確さを 0〜100 で測り高いほど良い)上で各手法の mAP を横並びで報告できるようにしました。レジストリがなければ、各手法ごとに別コードを書く必要があり、公平な比較(同じ前処理・同じ評価)が困難でした。レジストリの価値は「比較の公平性と再現性」という、数値そのものではなく数値の信頼性に効きます。
- アブレーション量産の効果。レジストリで部品を文字列差し替えできると、「backbone を ResNet-50 から Swin-Tiny に変えたら mAP が何ポイント動くか」といった比較を、学習ループを書き換えずに設定1行で回せます。研究のスループット(単位時間あたりに回せる実験数)が上がることが、最終的な性能改善の速度に効きます。これは定量化しにくいが、大規模ベンチマークの進展速度に表れる実利です。

要するにレジストリの「実験結果」は、特定の精度向上ではなく「同一条件での比較を安価かつ再現可能にする」という方法論上の貢献であり、現代の ML 研究のアブレーション文化を支える基盤になっています。

## メリット・トレードオフ・限界
メリット。

- 新しいプランナーの追加が1行(登録)で済み、選ぶ側のコードは無改変。重複した if/elif 分岐が消え、保守性が上がる。
- 名前(文字列)で選べるので、設定ファイル・コマンドライン引数・ハイパーパラメータスイープから簡単に切り替えられる。実験量産に向く。
- 未知の名前に対し候補一覧つきの親切なエラーを出せる。タイプミスや古い設定にすぐ気づける。
- 共通インターフェース(BasePlanner 契約)と組めば、どのプランナーでも同じ呼び出し方が保証され、使う側がプランナーの中身を知らずに済む(関心の分離)。

トレードオフ・限界。

- `**kwargs` を素通しするため、選んだプランナーに合わない引数を渡すとコンストラクタで初めてエラーになる。レジストリ側では弾けず、事前バリデーションが効かない。
- 名前という間接層が入るぶん、コードを初めて読む人は「`"bezier"` が実際どのクラスか」を登録簿まで追う必要があり、静的解析やジャンプ機能が効きにくい。
- import 時にすべてのプランナークラスを読み込む素朴な実装だと、依存が重いプランナーがあると import が遅くなる。遅延 import(使うときに初めて読む)やデコレータ自動登録で緩和できるが、複雑さは増す。
- 文字列キーは型安全でない。タイプミスは実行時まで分からず、IDE の補完も効きにくい。Enum や型付きの設定(dataclass + Literal)で補強する設計もある。

## 発展トピック・研究の最前線
- デコレータ自動登録: `@register("name")` をクラス定義に付けて、定義と同時に登録する流儀。登録漏れを防ぎ、定義と登録を一箇所にまとめられる。OpenMMLab が代表。
- 階層的・型付き設定システム: Hydra や OmegaConf、Detectron2 の LazyConfig のように、レジストリと設定ファイルを統合し、ネストした構成(backbone の中の norm 層まで)を文字列+辞書で組み立てる仕組み。大規模な実験管理の主流。
- プラグイン機構との接続: Python の entry points を使い、別パッケージで定義したプランナーを「インストールするだけ」でレジストリに自動参加させる方式。フレームワークを本体改修なしに拡張できる。
- 型安全なレジストリ: 文字列キーの弱点を補うため、Literal 型や Enum で選択肢を型レベルに固定し、IDE 補完と静的チェックを効かせる研究的な設計。表現力と安全性のトレードオフが論点。
- 同種レジストリの多用: バックボーン用、特徴融合用、損失用、データセット用と、同じパターンを複数持つのが大規模 ML コードベースの常套。レジストリ間の依存と初期化順序の管理が、コードベースが育つにつれ新たな設計課題になります。

## さらに学ぶための関連トピック
- [ベジェ曲線トラジェクトリプランナー](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0116_bezier-trajectory-planner)
- [Flow Matching トラジェクトリプランナ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0122_flow-matching-planner)
- [バックボーンのレジストリ構成](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0920_backbone-registry)
- [プランナーの出力契約 (Planner Output Interface)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0921_base-planner-contract)
- [反応的ポリシーと世界モデルの分離(階層的 E2E 運転のリファクタリング)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1006_reactive-e2e-refactor)
