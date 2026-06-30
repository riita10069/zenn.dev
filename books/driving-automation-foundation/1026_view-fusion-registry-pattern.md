---
title: "View Fusion のレジストリパターン"
---

# View Fusion のレジストリパターン

## ひとことで言うと

「複数カメラの特徴をどう1つにまとめるか（view fusion）」には、単純なチャネル連結、カメラ間アテンション、BEV変換など複数のやり方があります。それらを `"concat"`、`"cross_attn"`、`"bev"` といった文字列キーで辞書に登録しておき、設定の文字列を渡すだけで対応するモジュールに差し替えられるようにする設計（レジストリパターン）です。新しい融合手法を辞書に1行足すだけで、それを呼び出す側のコードを一切変えずに使えるようにするのが目的です。

## 直感的な理解

レストランの厨房を想像してください。「カレーをください」と言われたら、店員はメニュー表（注文名 → 調理担当）を引いて、対応するシェフに作らせます。客（呼び出す側）はシェフの名前も調理手順も知る必要がなく、料理名を言うだけでよい。新しいメニューが増えても、メニュー表に1行足してシェフを1人雇えばよく、客側の注文の仕組みは何も変える必要がありません。

機械学習の研究開発でも同じ要求が頻繁に発生します。「融合のやり方をいくつも試して比較したい」「設定ファイルの文字列1つで手法を切り替えたい」。素朴に if-elif の分岐で書くと、手法を増やすたびに本体のコードをいじることになり、変更箇所が散らばってバグの温床になります。メニュー表に相当する辞書を1か所に置き、「文字列で注文する」仕組みにすれば、本体は無改変のまま手法だけを差し替えられます。これがレジストリパターンです。

## 基礎: 前提となる概念

用語を噛み砕きます。

- view fusion（ビューフュージョン）。複数カメラ（view）の特徴を統合して1つの表現にする処理です。具体的な手法は [ConcatViewFusion（チャネル連結融合）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0951_concat-view-fusion) や [CrossAttentionViewFusion（カメラ間アテンション融合）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0952_cross-camera-attention-fusion) を参照してください。

- レジストリ（registry）。「名前 → 実体」の対応表（辞書）です。電話帳のように、名前を引くと中身が取り出せます。プログラム言語によっては「クラスそのもの」を値として辞書に入れられます（Python ではクラスは第一級オブジェクトなので可能です）。

- ファクトリ（factory）。「指定された名前の部品を作って返す係」の関数やメソッドです。客は「何が欲しいか」だけを伝え、どう作るかはファクトリに任せます。デザインパターンの古典 Gamma et al.（1994）の Factory Method / Abstract Factory に由来します。

- 設計パターン（design pattern）。よく出てくる設計上の課題に対する、再利用可能な定型の解き方です。レジストリ + ファクトリは深層学習コードベースで最も普及した組み合わせの1つです。

- 密結合・疎結合（tight / loose coupling）。あるモジュールが別の具体的なモジュールを直接知っていて依存している状態が密結合、文字列やインターフェース越しにしか知らない状態が疎結合です。疎結合だと一部を差し替えても他に波及しません。

研究では「複数手法を比較したい」場面が頻繁にあります。素朴に書くと呼び出す側に次の分岐が生まれます。

```python
if fusion_mode == "concat":
    self.fusion = ConcatViewFusion(...)
elif fusion_mode == "cross_attn":
    self.fusion = CrossAttentionViewFusion(...)
elif fusion_mode == "bev":
    self.fusion = BEVViewFusion(...)
else:
    raise ValueError(...)
```

問題は3つあります。手法を増やすたびにこの if-elif を直接いじる必要がある（変更が散らばる）。モデル本体が個々の融合クラスを全部 import して知っていなければならない（密結合）。設定ファイルや実験スクリプトから「文字列1つで切り替える」ことがやりにくい。レジストリパターンはこれを解決します。

## 仕組みを詳しく

中心は「辞書」と「ファクトリ関数」の2つです。

```python
FUSION_REGISTRY = {
    "concat":     ConcatViewFusion,
    "cross_attn": CrossAttentionViewFusion,
    "bev":        BEVViewFusion,
}

def build_view_fusion(fusion_mode, num_views, embed_dim=256, **kwargs):
    if fusion_mode not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion_mode '{fusion_mode}'. "
            f"Available: {list(FUSION_REGISTRY.keys())}"
        )
    return FUSION_REGISTRY[fusion_mode](
        num_views=num_views, embed_dim=embed_dim, **kwargs
    )
```

ステップで追います。

1. `FUSION_REGISTRY` という辞書が「文字列キー → クラス本体」を保持します。`BEVViewFusion()` と呼ぶ前の、クラスそのものを値として格納している点が肝です。
2. `build_view_fusion`（ファクトリ関数）は、渡された `fusion_mode` を辞書で引いて対応するクラスを取り出し、その場で `(...)` を付けてインスタンス化します。
3. 知らないキーが来たら、登録済みキー一覧を添えてエラーを出します（`Available: [...]`）。何が使えるかが分かるので親切です。
4. `**kwargs`（キーワード引数の受け流し）により、融合手法ごとの固有パラメータ（BEV なら `bev_h, bev_w, pc_range` など）を、ファクトリが中身を知らなくてもそのまま該当クラスへ転送できます。

呼び出す側はこうなります。

```python
self.view_fusion = build_view_fusion(fusion_mode, num_views, embed_dim, **kwargs)
...
fused = self.view_fusion(fused_per_view, B, V, camera_params=camera_params)
```

ポイントは、呼び出す側が `ConcatViewFusion` や `BEVViewFusion` という具体名を一切書いていないことです。文字列 `fusion_mode` を渡すだけ。これが「下流を無改変に保つ」の意味です。

成立の前提はインターフェースの統一です。差し替えが効くのは、全ての融合クラスが同じ呼び出し形（`__init__(num_views, embed_dim, **kwargs)` と `forward(fused_per_view, B, V, camera_params=None)`）を守っているからです。出力の空間サイズは手法で異なり、concat や cross_attn は入力と同じ格子、bev は `(bev_h, bev_w)` の格子を返しますが、`[B, C, H, W]` という型の約束は共通です。この「同じ形で呼べて同じ型を返す」という約束（インターフェース）が崩れると差し替えは破綻します。

学習スクリプトとの連携も重要です。コマンドライン引数 `--fusion-mode` の選択肢を registry のキーから動的に取得する設計にしておけば、registry に手法を足すだけで `--fusion-mode 新手法` がスクリプト無改変で選べます。これが「設定駆動（config-driven）」の実体です。

なお、登録方法には2つの流儀があります。上のように辞書リテラルに直接書く方法と、デコレータで自動登録する方法（後述の Detectron2 の `@FUSION_REGISTRY.register()`）です。後者はクラス定義のそばに登録を書けるので、ファイルを分散させてもインポート時に自動で辞書へ集まる利点があります。

## 手法の系譜と主要論文

レジストリパターンそのものは学術論文の手法ではなく、大規模な深層学習コードベースで広く使われるソフトウェア設計の慣行です。源流は Gamma et al. "Design Patterns: Elements of Reusable Object-Oriented Software"（1994）の Factory Method / Abstract Factory にあり、それを「名前で引ける辞書」と組み合わせたものが深層学習で定着しました。代表的な実装を挙げます。

- Wu et al. "Detectron2"（Facebook AI Research, 2019）。`Registry` クラスと `@BACKBONE_REGISTRY.register()` デコレータで backbone やヘッドを名前登録し、設定ファイルの文字列でモデル構成を組み立てます。研究で多数のバリアントを比較するために、コンポーネントを文字列で差し替え可能にする発想を広く普及させました。

- Chen et al. "MMDetection / MMCV"（OpenMMLab, arXiv:1906.07155, 2019）。`Registry` と `build_from_cfg` により、設定 dict の `type` フィールド（文字列）からモジュールを構築します。`build_view_fusion(fusion_mode, ...)` のようなファクトリはこの「文字列キーでビルドする」流儀の小さな版です。OpenMMLab は検出・セグメンテーション・3D 検出など多数のライブラリでこの統一規約を徹底しました。

- Ross Wightman "timm (PyTorch Image Models)"（2019-）。`create_model("swin_v2_tiny", ...)` のように、モデル名（文字列）から実体を生成する仕組みを提供します。backbone を文字列で差し替えられる設計はここから広く採用されています。

これらに共通する理由は「設定駆動で実験を回しやすくする」「コンポーネントを疎結合にして拡張を局所化する」ことです。効果は再現性と拡張性の向上です。トレードオフは、文字列キーのタイプミスが実行時まで発覚しないこと、IDE の補完や静的型解析が効きにくいことです（登録済みキー一覧をエラーに出す工夫でこれを緩和できます）。

## 論文の実験結果(定量データ)

このパターンは精度指標を持つ「手法」ではないため、定量データは「これを使った研究がどれだけ効率化したか」という間接的な形で現れます。たとえば MMDetection は、レジストリと設定ファイルによって COCO 物体検出ベンチマーク上で数十種類の検出器（Faster R-CNN、RetinaNet、Cascade R-CNN、DETR 系など）を共通フレームで再現し、同一条件での比較表を提供しました。これにより「どの構成要素を差し替えると AP がどう変わるか」のアブレーションが、本体コードを書き換えずに設定の差し替えだけで実施できるようになりました。研究コミュニティではこの再現性の高さが評価され、多数の論文が MMDetection の設定を公開して比較の土台に使っています。

定量的な恩恵を端的に言えば、新手法の追加に必要な変更行数が「本体の if-elif を編集する数十行」から「辞書に1行 + 新クラス1ファイル」へ減ることです。比較実験では、同じファクトリで N 個の手法を切り替えてベンチマークを回せるため、実験スクリプトの重複が消え、設定ファイルの差分だけで実験条件をバージョン管理できます。これは数値精度ではなく開発効率と再現性の指標で測られる利点です。

## メリット・トレードオフ・限界

メリット。拡張が局所的です。新手法は辞書に1行足すだけで、呼び出し側を改変しなくてよい。疎結合になり、モデル本体が個々の融合クラスを import・分岐する必要がありません。設定駆動なので、実験スクリプトや CLI から文字列1つで手法を切り替えられ、比較実験が回しやすい。未知キーのとき利用可能な選択肢を提示でき、エラーが親切になります。

トレードオフと限界。文字列キーの誤りは実行時エラーになり、コンパイル時や型チェックでは見つかりません。IDE 補完も効きにくい。全クラスが同じ呼び出しインターフェースを守る前提に依存するため、約束が崩れると差し替えが壊れます。抽象が1段増えるぶん、コードを初めて読む人には「どの実体が実際に動くのか」が追いにくくなります。さらに、比較対象が将来1つに絞られた場合、レジストリの恩恵は「将来の拡張余地」に縮小し、やや過剰な抽象になりうるという面もあります。実際、手法を比較していた開発初期にレジストリは最も価値を発揮し、標準手法が1つに決まった段階では登録が1件に絞られることがあります。それでも辞書に1行足すだけで別手法を戻せる拡張性は維持されます。

研究上の論点としては、文字列キーの脆さを型で守るために、列挙型（enum）や Literal 型、Pydantic などのスキーマ検証と組み合わせる方向、あるいは設定ファイル全体を型付きデータクラスにして静的検証する方向（Hydra の structured config など）が議論されています。

## 発展トピック・研究の最前線

設定駆動の実験管理は、レジストリ単体から「設定フレームワーク」へ発展しています。Meta の Hydra は YAML 設定を合成・上書きでき、コマンドラインからの一括スイープ（複数設定の自動展開実行）を可能にします。これをレジストリと組み合わせると、`fusion_mode=concat,cross_attn,bev` のように指定して3手法のベンチマークを一括起動できます。OmegaConf による構造化設定や、Hydra の structured config を使えば、文字列キーの誤りを実行前に検出できるようになり、レジストリの最大の弱点である「タイプミスの実行時発覚」を補えます。大規模な基盤モデル開発では、モジュール単位のレジストリに加えて「アーキテクチャ全体を設定で組み立てる」composability が重視されており、レジストリパターンはその最小単位として今も中核を担っています。

## さらに学ぶための関連トピック

- [ConcatViewFusion（チャネル連結融合）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0951_concat-view-fusion)
- [CrossAttentionViewFusion（カメラ間アテンション融合）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0952_cross-camera-attention-fusion)
- [マルチスケール特徴のプーリング連結](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0987_multi-scale-feature-fusion)
- [backbone処理のための B*V reshape](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0918_backbone-batch-view-reshape)
- [学習可能なカメラ位置埋め込み](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0975_learnable-view-position-embedding)
