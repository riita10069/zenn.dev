---
title: "Component Registry / Plugin 設計"
---

# Component Registry / Plugin 設計

## ひとことで言うと
Component Registry（コンポーネント登録簿）とは、差し替え可能な部品（画像特徴抽出器、特徴融合器、軌跡プランナなど）を辞書（dict）に名前付きで登録しておき、文字列キー（例 "swin_v2_tiny"）を指定するだけで対応する部品を組み立てられるようにする設計です。新しい部品を辞書に1行足すだけで、学習スクリプト本体を一切いじらずに選択肢を増やせます。研究開発で部品を頻繁に差し替える深層学習フレームワークの拡張性を支える基盤パターンです。

## 直感的な理解
研究開発では「特徴抽出器を Swin から ConvNeXt に替えてみたい」「プランナを GRU から Flow Matching に替えてみたい」といった部品の差し替えが頻繁に起きます。どの組み合わせが最も性能が出るかは事前に分からないので、片っ端から試すことになります。

素朴に書くと、選択のたびに巨大な if/else 分岐を学習コードに書くことになります。

```python
if backbone == "swin":     model = Swin(...)
elif backbone == "convnext": model = ConvNeXt(...)
elif backbone == "resnet":   model = ResNet(...)
# 新しい backbone を足すたびにここを編集...
```

これには2つの問題があります。1つは、部品を追加するたびに学習コード本体を編集する必要があり、変更が広範囲に散らばること。もう1つは、選択肢の一覧（どんな部品が使えるか）がコードのあちこちに散在し、把握しづらいことです。

Registry パターンはこれを解決します。「名前 → 作り方」の対応表を1か所（辞書）に集約し、利用側は「名前を渡すと部品が返ってくる」関数を呼ぶだけにします。新しい部品はその辞書に追加するだけ。利用側コードは無改変で済みます。レストランのメニュー表に例えると、客（利用側）は「カルボナーラ」と注文するだけでよく、厨房（registry）がその名前に対応する作り方を知っている。新メニューを増やしてもメニュー表に1行足すだけで、客の注文の仕方は変わりません。

## 基礎: 前提となる概念
「辞書（dict）」は「キー → 値」の対応表です。Registry では、キーが部品の名前（文字列）、値が「その部品を作る方法」になります。値には2通りあり、クラスそのものを入れる場合（`{"flow_matching": FlowMatchingPlanner}`）と、呼ぶと部品を返す関数を入れる場合（`{"swin": lambda **kw: create_swin(**kw)}`）があります。前者は `value(**kwargs)` で生成、後者も `value(**kwargs)` で呼び出すので、利用側は区別せずに扱えます。

「ファクトリ（factory）」は、オブジェクトを生成する役割を持つ関数やオブジェクトのことです。Registry に付属する `build_xxx(name, **kwargs)` のような関数はファクトリで、「名前を受け取り対応部品を作って返す」のが仕事です。

「`**kwargs`（キーワード引数の可変長受け取り）」は、任意個の名前付き引数をまとめて受け取る Python の仕組みです。Registry のファクトリは `**kwargs` で受けた引数を部品のコンストラクタにそのまま渡すことで、部品ごとに異なる引数を1つの窓口で扱えます。各部品は自分が理解する引数だけ受け取り、余分は無視する約束にすることが多いです。

設計原則として「開放閉鎖の原則（Open/Closed Principle, OCP）」が背景にあります。Bertrand Meyer が "Object-Oriented Software Construction" で提唱した「拡張に対して開いており（新しい部品を足せる）、修正に対して閉じている（既存コードを変えなくてよい）」べきという指針で、Registry はこの典型的な実装です。

## 仕組みを詳しく
典型的な Registry の実装を見ます。backbone（画像特徴抽出器）の例です。

```python
BACKBONE_REGISTRY = {
    "swin_v2_tiny":      lambda pretrained=True, **kw: create_model("swinv2_tiny_window8_256", pretrained=pretrained, features_only=True, **kw),
    "conv_next_v2_tiny": lambda pretrained=True, **kw: create_model("convnextv2_tiny", pretrained=pretrained, features_only=True, **kw),
    "res_net_50":        lambda pretrained=True, **kw: create_model("resnet50", pretrained=pretrained, features_only=True, **kw),
}

def build_backbone(name, pretrained=True, **kwargs):
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}")
    return BACKBONE_REGISTRY[name](pretrained=pretrained, **kwargs)
```

ポイントは2つです。第一に、辞書の値は「呼ぶと部品を作る関数（ここでは lambda）」で、キーを指定するとその関数が呼ばれて部品が組み上がります。第二に、未知のキーには登録済みの一覧を添えた分かりやすいエラーを出します（`Available: [...]`）。利用者がタイポしてもすぐ気づけるのが実務上とても重要です。

プランナの例では値がクラスそのものになることがあります。

```python
PLANNER_REGISTRY = {"flow_matching": FlowMatchingPlanner, "bezier": BezierPlanner}

def build_planner(name, **kwargs):
    if name not in PLANNER_REGISTRY:
        raise ValueError(f"Unknown planner {name!r}. Available: {sorted(PLANNER_REGISTRY)}.")
    return PLANNER_REGISTRY[name](**kwargs)
```

真価が出るのは利用側です。コマンドライン引数の選択肢（choices）を registry のキーから動的に取得すると、registry にエントリを足すだけで新しい選択肢が自動で現れます。

```python
parser.add_argument("--backbone", default="swin_v2_tiny",
                    choices=sorted(BACKBONE_REGISTRY))
```

before/after で効果を示します。

```
before: 新 backbone 追加 → registry に登録 + 学習コードの if/else 編集 + choices を手で更新
after : 新 backbone 追加 → registry の辞書に1行足すだけ（利用側コードは無改変）
```

大規模フレームワークでは、辞書を直接書く代わりにデコレータで登録する形がよく使われます。

```python
@BACKBONE_REGISTRY.register()
class MyNewBackbone(nn.Module):
    ...
```

`@register` を付けるだけで、クラス定義と同時に registry に名前が登録されます。これにより部品の定義と登録が1か所に集約され、新部品を別ファイルに置いて import するだけで使えるようになります（プラグイン的拡張）。設定ファイル（YAML/JSON）に `type: MyNewBackbone` と書けばその部品が選ばれる、という config 駆動のモデル構成がこの仕組みで実現します。

## 手法の系譜と主要論文
これは深層学習の理論論文というより、研究を回すためのソフトウェア工学パターンですが、その普及には主要フレームワークの貢献が大きいです。

起点となる設計原則は GoF の Factory パターン（Gamma et al., 1994）と Meyer の開放閉鎖の原則です。Registry はこれらを「文字列キーで引けるファクトリ表」として具体化したものです。

深層学習での大規模採用例として、Facebook AI Research の Detectron2（Wu et al., 2019）が `Registry` クラスと `@register` デコレータで backbone・proposal generator・ROI head などを文字列で組み替える設計を全面採用しました。同時期に OpenMMLab の MMDetection（Chen et al., 2019, arXiv:1906.07155）と基盤ライブラリ MMCV が `Registry` を中核に据え、backbone・neck・head・loss・optimizer まであらゆる構成要素を config ファイルの文字列だけで再構成できるようにしました。提案動機は明快で、「研究者が部品を頻繁に差し替えるから、設定ファイルの文字列だけでモデルを再構成したい」というものです。効果は実験の組み合わせ爆発を config だけで管理できること。トレードオフは、文字列で繋ぐため静的型チェックが効きにくく、キーのタイポやコンストラクタ引数の不整合が実行時まで分からないことです。

もう1つの代表例が Ross Wightman の timm（PyTorch Image Models）です。`create_model(name, pretrained=True, ...)` というファクトリは、モデル名文字列で1000種類以上のアーキテクチャ（ResNet, ViT, Swin, ConvNeXt など）を生成・事前学習重みのロードまで行う巨大な registry 機構です。多くのプロジェクトは自前の薄い registry を timm の上に重ね、二段のファクトリを構成します。同様に Hugging Face Transformers の `AutoModel.from_pretrained(name)` も、モデル名やアーキテクチャ名から適切なクラスを引く registry 的機構です。

## 論文の実験結果（定量データ）
設計パターンそのものにベンチマーク数値は付きませんが、Registry を採用したフレームワークが研究生産性に与えた影響は間接的に測れます。MMDetection は単一の config システムと registry によって、数十種類の検出器（Faster R-CNN, Mask R-CNN, RetinaNet, DETR ほか）を統一インターフェースで提供し、COCO 物体検出ベンチマーク（一般物体検出の標準データセット、80クラス、評価指標は mAP = mean Average Precision、全クラス・全閾値で平均した検出精度で大きいほど良い）における多数の手法の再現実装を1つのコードベースに集約しました。これにより、異なる手法を同一条件（同じデータ前処理・同じ学習スケジュール）で比較できるようになり、論文間の数値比較の信頼性が向上しました。registry が無ければ各手法ごとに別実装となり、わずかな実装差が mAP に数ポイントの差を生んで公平な比較が困難になります。つまり registry の効果は「公平な比較を可能にすること」という形で、ベンチマーク数値の信頼性に寄与しています。

アブレーション的に言えば、registry の構成要素のうち「一覧表示付きエラー（`Available: [...]`）」を外すと、タイポ時のデバッグ時間が増えます。「choices の動的生成」を外すと、選択肢が二重管理になり、registry に部品を足したのに利用側で選べないという不整合が起きます。「`**kwargs` 素通し」を外して引数を逐一明示すると、部品ごとに窓口が分かれて統一性が失われます。これらはいずれも開発体験（測定しづらいが実務上重要なコスト）に直結します。

## メリット・トレードオフ・限界
メリットは、新しい部品を辞書に登録するだけで選択肢が増え利用側が無改変で済むこと（開発が速い）、選択肢の一覧が registry 1か所に集約され把握しやすいこと、コマンドライン引数や config の選択肢を registry から自動生成でき二重管理を防げること、そして実験（backbone × planner × fusion の組み合わせ）を文字列指定だけで切り替えられることです。

トレードオフと限界もあります。第一に、文字列キーで繋ぐため静的型チェックが効きにくく、タイポやコンストラクタ引数の不整合が実行時エラーになりやすいです（一覧表示エラーで緩和できますが根本解決ではありません）。第二に、ファクトリが `**kwargs` を素通しするだけで事前フィルタしない場合、各部品が受け取れない引数を渡すとエラーになり、呼び出し側が正しい引数を知っている必要があります。第三に、部品間の前提（例えば融合器が backbone の出力チャンネル数を期待する）を registry は保証しないので、組み合わせの整合は別途担保が必要です。第四に、過度に使うと「どの実装が実際に動くか」がコードを追っても分かりにくくなり、間接参照が増えて読みにくくなります（IDE のジャンプ機能が文字列キー越しには効かない）。

研究上というより工学上の課題として、型安全性と動的拡張性をどう両立するかがあります。Python の型ヒントや Pydantic ベースの config（型付き設定）、dataclass による引数定義などで registry の弱点（実行時まで分からないエラー）を補強する取り組みが進んでいます。

## 発展トピック・研究の最前線
近年は registry と型付き設定（Hydra, OmegaConf, Pydantic）を組み合わせ、config ファイルの文字列をエディタ補完・静的検証できるようにする方向が主流です。Hydra（Facebook）は config の合成・上書き・スイープ（複数設定の自動実験）を registry 的な構造化と組み合わせ、ハイパーパラメータ探索を config だけで回せるようにしました。Hydra の `instantiate` 機構は config 中の `_target_: package.module.ClassName` という文字列を解決してそのクラスを生成するもので、これは registry を「import パスそのものをキーにする」形に一般化したものと見なせます。短い別名（"swin"）で引く明示 registry と、完全修飾名で引く Hydra 方式は、可読性（短い名前は覚えやすい）と曖昧さの無さ（完全修飾名は衝突しない）のトレードオフ関係にあります。また、大規模モデルでは部品をプラグインとして別パッケージ化し、import するだけで registry に自動登録される「エントリポイント方式」（Python の entry_points 機構）も使われ、コア本体を変えずに第三者が部品を追加できるエコシステム拡張が可能になっています。本パターンは [薄いラッパー（トップレベルモデルの責務分離）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1008_reactive-e2e-wrapper) の薄いラッパー設計と組み合わせると効果が最大化し、「束ねるトップ（薄いラッパー）」と「何を束ねるかを文字列で選ぶ（registry）」が、研究コードの拡張性を支える両輪になります。

## さらに学ぶための関連トピック
- [薄いラッパー（トップレベルモデルの責務分離）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1008_reactive-e2e-wrapper)
- [System1/System2分離（Reactive/Reasoning）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1017_system1-system2-architecture)
- [Imitation / Behavior Cloning Loss](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0562_imitation-loss)
- [World / Action Model（低頻度推論ブランチ）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1053_world-action-model-placeholder)
