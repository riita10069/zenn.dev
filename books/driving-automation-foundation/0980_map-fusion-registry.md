---
title: "マップフュージョンレジストリ"
---

# マップフュージョンレジストリ

## ひとことで言うと

地図とカメラの特徴を混ぜ合わせる「融合方式」を、名前(文字列キー)で選べるようにした仕組みです。`"residual"` や `"cross_attn"` という文字列を渡すだけで、対応する融合モジュールが組み立てられます。新しい方式を作ったら辞書に1行足すだけで、呼び出し側のコードを書き換えずに選択肢を増やせます。研究で「どの方式が一番良いか」を何度も入れ替えて比較する作業を、ソフトウェア設計の側から支える土台です。

## 直感的な理解

地図 BEV 特徴とカメラ BEV 特徴を融合する方式は1つではありません。たとえばゼロ初期化ゲートで足す残差融合([マップBEV残差フュージョン（per-channel alpha）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0309_map-bev-residual-fusion))と、注意機構で混ぜるクロスアテンション融合([マップクロスアテンションフュージョン](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0310_map-cross-attention-fusion))があります。研究では「どちらが良いか」を実験で比べたくなります。

ここで素朴に、学習スクリプトやモデル本体の中に `if mode == "residual": ... elif mode == "cross_attn": ...` という条件分岐を直接書いてしまうと何が起きるでしょうか。方式を1つ増やすたびに、この分岐があちこちに散らばって増殖します。学習ループ、評価ループ、推論コードと、同じ分岐を何度も書くことになり、片方を直し忘れてバグになります。これは「コードの変更箇所が一箇所にまとまっていない」状態で、保守の悪夢です。

そこで「文字列キー → クラス」の対応表(辞書)を一箇所に持ち、`build_...(key)` というファクトリ関数で組み立てる設計にします。これがレジストリ(registry、登録簿)パターンです。呼び出し側は具体的なクラス名を一切知らず、文字列だけ知っていればよいので、融合方式を増やしても呼び出し側のコードは1行も変わりません。

## 基礎: 前提となる概念

- レジストリ(registry): 「名前 → 作るべきクラス」の対応表(多くは言語標準の辞書/マップ)。
- ファクトリ関数(factory): 名前を受け取って、対応するオブジェクトを生成して返す関数。生成の詳細を一箇所に閉じ込めます。
- 疎結合(loose coupling): 呼び出し側が「どの具体クラスか」を直接知らなくてよい状態。名前だけ知っていればよいので、一方を変えても他方に影響しにくい。逆に密結合(tight coupling)は、具体クラスを直接 new していて、変更が連鎖する状態。
- インターフェース: 各融合モジュールが満たすべき共通の約束事(同じ引数で構築でき、同じ形の入出力を持つ)。これが揃っているからこそ、文字列で差し替えても呼び出し側が壊れません。
- 設定ファイル駆動(config-driven): 実験条件をコード本体ではなく設定ファイルの文字列で指定し、コードを変えずに条件だけ切り替える流儀。レジストリと相性が良い。

## 仕組みを詳しく

典型的な実装は2階層になります。融合方式のレジストリと、その上位にある地図エンコーダのレジストリです。

### 融合方式のレジストリ

```python
from .residual_fusion import ResidualMapFusion
from .cross_attention_fusion import MapCrossAttentionFusion

MAP_FUSION_REGISTRY = {
    "residual": ResidualMapFusion,
    "cross_attn": MapCrossAttentionFusion,
}

def build_map_bev_fusion(fusion_mode, embed_dim=256, **kwargs):
    if fusion_mode not in MAP_FUSION_REGISTRY:
        raise ValueError(
            f"Unknown map_fusion_mode '{fusion_mode}'. "
            f"Available: {list(MAP_FUSION_REGISTRY.keys())}"
        )
    return MAP_FUSION_REGISTRY[fusion_mode](embed_dim=embed_dim, **kwargs)
```

ポイントを順に。

1. `MAP_FUSION_REGISTRY` が「文字列 → クラス」の辞書。値はインスタンスではなくクラスそのもの(まだ生成していない)。
2. `build_map_bev_fusion("residual", embed_dim=256)` を呼ぶと、辞書から `ResidualMapFusion` を引き、`ResidualMapFusion(embed_dim=256)` として実体化して返します。`"cross_attn"` なら `MapCrossAttentionFusion(...)`。
3. `**kwargs`(可変キーワード引数)で方式固有の引数を素通しできます。クロスアテンションは `num_heads` や `dropout` を持つので `build_map_bev_fusion("cross_attn", embed_dim=256, num_heads=8)` のように渡せます。残差融合はそれらを受け取らないので、各方式のコンストラクタが必要な引数だけ拾います。
4. 未知のキーには、利用可能なキー一覧を添えて `ValueError` を投げます。タイプミス(`"resiudal"` など)を早期に分かりやすく弾けます。文字列キー方式の最大の弱点(タイプミスが実行時まで分からない)を緩和する実用的な工夫です。

### 地図エンコーダのレジストリ

融合の一階層上に、地図そのものをどう符号化するかのレジストリも置けます。

```python
MAP_ENCODER_REGISTRY = {
    "rasterized": RasterizedMapEncoder,
}

def build_map_encoder(map_type, **kwargs):
    if map_type not in MAP_ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown map_type '{map_type}'. "
            f"Available: {list(MAP_ENCODER_REGISTRY.keys())}"
        )
    return MAP_ENCODER_REGISTRY[map_type](**kwargs)
```

ラスタ地図画像を符号化する [ラスタライズドマップエンコーダ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0425_rasterized-map-encoder) を `"rasterized"` キーで登録します。ここが設計上の肝で、`build_map_encoder` のインターフェースは map_type と `**kwargs` を受けるだけなので、Lanelet2 や OpenDRIVE のようなベクトル HD マップ(車線を折れ線で表す高精細地図)用エンコーダを実装したら、辞書に `"vectorized": VectorizedMapEncoder` を1行足すだけで選べるようになります。将来の拡張点を、インターフェースを変えずに先回りで用意しておくのが良い設計です。

### 登録の2つの流儀: 明示辞書 vs デコレータ

上の例は「辞書にクラスを直接書き並べる」明示的な方式です。もう一つ、大規模フレームワークで定番なのがデコレータによる自己登録です。

```python
class Registry:
    def __init__(self): self._table = {}
    def register(self, name):
        def deco(cls):
            if name in self._table:
                raise KeyError(f"'{name}' は既に登録済み")  # 重複検知
            self._table[name] = cls
            return cls
        return deco
    def build(self, name, **kwargs):
        return self._table[name](**kwargs)

MAP_FUSION = Registry()

@MAP_FUSION.register("residual")
class ResidualMapFusion(nn.Module): ...
```

`@MAP_FUSION.register("residual")` と書くだけで、クラス定義と同時に辞書へ登録されます。利点は、クラスのすぐ横に登録名が書かれるので「このクラスがどの名前で呼ばれるか」が一目で分かること。欠点は、そのクラスを定義したモジュールが import されないと登録が走らないことです(登録は import の副作用なので、import し忘れると「未登録」エラーになる)。明示辞書方式はこの落とし穴がない代わり、登録漏れを人間が管理する必要があります。どちらを選ぶかは、方式の数と保守体制次第です。

### 全体の組み立ての流れ(概念図)

```
map_type="rasterized" ──> build_map_encoder ──> RasterizedMapEncoder
                                                  (地図画像 → 地図BEV特徴)
fusion_mode="residual"/"cross_attn"
        └─> build_map_bev_fusion ──> ResidualMapFusion or MapCrossAttentionFusion
                                       (image_bev と map_bev を融合)
```

上位のモデル本体は、この2つの build 関数を呼んでモジュールを組み立てるだけで、個々のクラス名や実装を直接参照しません。これが疎結合の実体です。

## 手法の系譜と主要論文

レジストリ/ファクトリは論文の手法というよりソフトウェア工学のデザインパターンですが、機械学習の大規模フレームワークがその有効性を実証してきました。系譜を追います。

- Gamma et al., "Design Patterns: Elements of Reusable Object-Oriented Software"(1994、通称 GoF 本): Factory Method / Abstract Factory パターンを体系化した古典。動機は「具体的なクラスを直接 new せず、生成を一箇所に閉じ込めて差し替え可能にする」こと。効果は拡張時の変更局所化。トレードオフは間接層が増えてコードの追跡がやや手間になること。レジストリは、この Factory に「名前で引ける辞書」を組み合わせた形です。

- Chen et al., "MMDetection: Open MMLab Detection Toolbox and Benchmark"(2019, arXiv:1906.07155): 物体検出フレームワークで、`@registry.register_module` のデコレータと文字列名による `build_from_cfg` を全面採用したことで知られます。動機は、設定ファイルの文字列だけでバックボーン・ネック・ヘッドを差し替えて大量の実験を回せるようにすること。効果は、新手法の追加が「クラスを書いて登録するだけ」で済む拡張性。トレードオフは、文字列のタイプミスが実行時まで分からない・IDE の補完が効きにくいこと。

- Wu et al., "Detectron2"(2019): 同様の `Registry` クラスを持ち、`@BACKBONE_REGISTRY.register()` のデコレータで登録、`build_backbone(cfg)` で生成する設計。MMDetection と同じ思想で、研究のスピードと保守性を両立しています。

- Wolf et al., "Transformers (Hugging Face)"(EMNLP 2020, arXiv:1910.03771): `AutoModel.from_pretrained("model-name")` のように、モデル名の文字列から適切なクラスを自動解決する `Auto*` ファクトリ群を提供。これも巨大なレジストリ(名前 → クラスの対応表)に支えられており、無数のモデルを統一インターフェースで扱えるようにしています。

これらはいずれも「研究のスピード(手法をすぐ差し替えて比較したい)」と「コードの保守性」を両立させるために registry を採用しており、マップ融合レジストリも同じ目的で導入されます。バックボーンやビュー融合など他の差し替え可能な部品も、同じ registry 思想で統一すると一貫性が保てます。

## 論文の実験結果(定量データ)

レジストリ自体は精度を上げる技術ではないため、定量データは「フレームワークが支えた実験規模・再現性」として現れます。

MMDetection(arXiv:1906.07155)は、論文公開時点で数十種類の検出手法と数百の事前学習済みモデルを統一 API で提供し、COCO 上のベンチマーク結果を一括で再現可能にしました。各手法の AP を同一コードベース・同一データ前処理で比較できることが価値で、registry による config 駆動がこの規模を可能にしています。「手法 A と手法 B を、前処理や学習設定を揃えた公平な条件で比べられる」ことは、研究で最も重要な再現性そのものです。

Hugging Face Transformers(arXiv:1910.03771)は、`Auto*` ファクトリによって数千規模のモデルを同一インターフェースで読み込めるようにし、論文の手法をコード数行で再現・比較できる環境を作りました。これは「レジストリ + 設定駆動」が研究コミュニティ全体の実験効率をどれだけ押し上げるかを示す最大規模の実例です。

なお registry の設計選択にもトレードオフがあり、(a) 文字列キーはコンパイル時の型チェックを通らないためタイプミスが実行時エラーになる、(b) IDE の自動補完が効きにくい、という弱点が共通します。これを緩和するため、未知キーに利用可能キー一覧を添えて例外を投げる・列挙型(Enum)でキーを縛る・登録時に重複チェックする、といった工夫が実務で使われます。

## メリット・トレードオフ・限界

メリット
- 融合方式・地図エンコーダを文字列1つで切り替えられ、どの方式が良いかの比較実験が容易。
- 新方式の追加が「クラスを書いて辞書に1行足す」で済み、呼び出し側(学習・評価・推論)を改変しなくてよい。
- 未知キーに利用可能キー一覧つきの例外を出すので、タイプミスを早期に発見できる。
- ベクトル地図エンコーダなど将来拡張を見越した構造で、後からの差し込みが容易。設定ファイル駆動の実験管理とも相性が良い。

トレードオフ・限界
- キーは文字列なので、タイプミスが実行時まで分からない(静的型チェックで弾けない)。IDE の補完も効きにくい。
- `**kwargs` を素通しする設計上、ある方式にしか無い引数を別の方式に渡すと、どこで弾かれるかが分かりにくいことがある。
- 間接層(build 関数)が1枚増えるため、コードを初めて読む人が「実際にどのクラスが動くか」を追うのに一手間かかる。
- レジストリへの登録が import 副作用に依存する実装だと、その方式を定義したモジュールを import し忘れて「登録されていない」エラーになることがある(遅延 import と相性が悪い)。

研究上の論点として、文字列キーの脆さを型安全にどう近づけるか(Enum・dataclass・Pydantic などによる設定検証)、そして registry が肥大化したときの名前衝突や責務分割をどう管理するか、があります。

## 発展トピック・研究の最前線

設定駆動の極端な形が Hydra(Yadan, 2019)や OmegaConf で、コマンドラインから `model.fusion=cross_attn` のように指定して構成を組み替え、ハイパーパラメータ探索(sweep)を自動化します。registry はこの構成管理の受け皿として機能します。MLOps の文脈では、registry のキーと設定ファイルが「どの構成で学習したか」のメタデータとして実験管理ツール(MLflow など)に記録され、再現性とモデル系譜の追跡を支えます。

研究の最前線では、手法を文字列で選ぶだけでなく、Neural Architecture Search(NAS)のように「どの融合方式を使うか」自体を探索・学習対象にする方向もあります。registry は、その探索空間(選択肢の集合)を宣言的に定義する自然な手段になります。たとえば `fusion in {residual, cross_attn}` という選択を、学習可能な重みで連続緩和して微分可能にする DARTS(Liu et al., ICLR 2019, arXiv:1806.09055)系の探索では、registry に並んだ候補がそのまま探索の選択肢集合になります。

型安全性を高める工夫も実務で進んでいます。文字列キーを Enum や Literal 型で縛り、設定を Pydantic / dataclass で検証すれば、タイプミスを実行前に静的に弾けます。これは「文字列で柔軟に差し替えたい」という registry の利点と、「事前に誤りを検出したい」という型安全の要請を両立させる折衷で、実験を大量に回す研究現場ほど効いてきます。設定ミスで数時間の学習を無駄にする事故を未然に防ぐ、地味だが重要な投資です。

## さらに学ぶための関連トピック

- [マップBEV残差フュージョン（per-channel alpha）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0309_map-bev-residual-fusion)
- [マップクロスアテンションフュージョン](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0310_map-cross-attention-fusion)
- [ラスタライズドマップエンコーダ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0425_rasterized-map-encoder)
- [osmnx による道路ネットワーク取得とマップマッチング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0423_osmnx-map-matching)
