---
title: "モックbackboneによる軽量テスト"
---

# モックbackboneによる軽量テスト

## ひとことで言うと
画像から特徴を取り出す重たい部品(backbone、バックボーン)を、テストのときだけ「正しい形のデータを返すだけの、ほぼ計算しない軽い偽物」に差し替える工夫です。これによりテストが何十倍も速くなります。backbone はテストで確かめたい対象そのものではない(事前学習済みで中身は固定されている)ので、本物そっくりに動く必要はなく、「出てくるデータの形(tensor shape、テンソルの各次元のサイズ)と勾配の流れだけ合っていればよい」という発想です。

## 直感的な理解
劇の稽古を想像してください。主役どうしの掛け合いを練習したいのに、舞台装置の本物の大道具を毎回組み立てていたら時間が足りません。そこで、寸法だけ本物と同じ「段ボールの代用品」を置きます。役者は段ボールでも立ち位置と動線を確認でき、本番の大道具を待つ必要がありません。

機械学習モデルのテストでも同じことが起きます。確かめたいのは「複数の部品が正しくつながり、データが正しい形で流れるか」です。ところが、その流れの入口にある画像特徴抽出器(backbone)が一番重く、テスト時間の大半を食います。しかも backbone は学習済みで中身を変えないので、テストの主役ではありません。そこで backbone を「寸法だけ合った段ボール」、つまり正しい形のテンソルを返すだけの軽い偽物に置き換えます。これが mock backbone です。重要なのは、寸法(形)だけでなく「力の伝わり方(勾配)」も残す点で、ここが単なるダミーとの違いになります。

## 基礎: 前提となる概念
用語を噛み砕きます。

- backbone(バックボーン): 画像を受け取り、エッジ・テクスチャ・物体らしさといった特徴を数値の塊(特徴マップ)に変換する中核部品です。ResNet、Swin Transformer、ConvNeXt などが代表例で、ImageNet などの大規模データで事前学習した重みを使い回すのが一般的です。
- forward(前向き計算): 入力を受け取り、層を順に通して出力を計算する処理です。学習・推論のどちらでも毎回走ります。
- backward(逆伝播): 出力の誤差をもとに、各層の重みをどう直すかの勾配を逆向きに計算する処理。学習のときだけ走ります。
- 特徴マップ(feature map): 画像のように「縦×横×チャンネル」を持つ数値配列です。チャンネルは「特徴の種類の数」で、深い層ほどチャンネルが多く、解像度は低くなります。
- 凍結(frozen): その部品の重みを学習で更新しない設定です。事前学習済み backbone はしばしば凍結して使います。中身が変わらないので、テストでの再現性が高い反面、重いロードコストだけが残ります。
- テストダブル(test double): 本物の代役の総称です。Dummy(使われない穴埋め)、Stub(決まった値を返す)、Mock(呼ばれ方も検証する)、Fake(簡易な本物相当)に分類されます。「mock backbone」は名前は mock ですが、分類上は Stub に近い(正しい形を返すだけで呼ばれ方は検証しない)ことが多いです。

カメラ画像から運転行動を出すような典型的な走行系モデルの構成は次の流れです。
1. backbone: カメラ画像から特徴マップを取り出す(重い、学習済み、凍結)。
2. View Fusion(視点融合): 複数カメラの特徴を1つの俯瞰表現(BEV、Bird's-Eye-View、上から見た地図のような表現)にまとめる。
3. Planner / Head: まとめた特徴から、車がこの先どう動くか(軌跡)を予測する。

テストで確かめたいのは主に 2 と 3 の結線と勾配伝播です。1 は脇役なのに最も重い、というのが mock 化の動機です。

## 仕組みを詳しく
本物の backbone(たとえば Swin Transformer V2 の tiny サイズ)は、256×256 の画像から4段階(stage 0〜3)の特徴マップを出します。mock はこの4段階を、ほぼ計算しない最小の畳み込み(Conv2d)とサイズ合わせのプーリングだけで再現します。

```python
import torch.nn as nn

class MockBackboneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(64),   # 空間サイズを必ず 64x64 にそろえる
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(96, 192, 3, 1, 1), nn.AdaptiveAvgPool2d(32),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 384, 3, 1, 1), nn.AdaptiveAvgPool2d(16),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(384, 768, 3, 1, 1), nn.AdaptiveAvgPool2d(8),
        )

    def forward(self, x):
        s0 = self.stage0(x)   # [B*V, 96, 64, 64]
        s1 = self.stage1(s0)  # [B*V, 192, 32, 32]
        s2 = self.stage2(s1)  # [B*V, 384, 16, 16]
        s3 = self.stage3(s2)  # [B*V, 768,  8,  8]
        return [s0, s1, s2, s3]
```

形の読み方: `[B*V, C, H, W]` は順に「バッチ数 × 視点数」「チャンネル数」「縦」「横」です。B はバッチサイズ(同時に処理するサンプル数)、V は視点数(カメラ台数)。マルチカメラ入力では B 枚 × V 視点をまとめて `B*V` 枚の画像として一括処理することが多く、後段で `[B, V, ...]` に戻します。Swin V2-tiny も入力256×256でこの4段階の形を出すので、形だけぴったり合わせています(詳しくは [SwinV2の4段階特徴マップ形状](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1016_swinv2-feature-map-shapes))。

設計の要点:

- `nn.AdaptiveAvgPool2d(64)` は「入力がどんな解像度でも、出力を必ず64×64にする」プーリングです。これにより、入力画像のサイズが多少違っても後段が期待する形を必ず返せます。テストが入力解像度に縛られなくなります。
- Conv は各段1層だけ。本物の Swin Transformer は各段で複数の Transformer ブロック(window attention + MLP)を通しますが、mock は1層の畳み込みだけなので桁違いに速い。実測例では、本物の forward 約数十ms が mock では1ms 未満になります。
- それでも Conv を通す理由は、勾配(gradient)を末端まで流すためです。完全な定数を返すと、その出力は入力に依存しないので入力側への勾配がゼロになり、「勾配が最後まで伝わるか」を確かめる gradient-flow テストが意味をなさなくなります。Conv は学習可能パラメータを持ち、入力に依存した出力を返すので、勾配が `入力 → Conv → 後段 → loss` と一本につながります。「軽い計算」を残すのが、ただのダミー値と「形・勾配を保つ stub」の決定的な違いです。

本物の backbone クラスと差し替えられるよう、本物と同じ引数を受け取れる薄いラッパも用意するのが定石です。

```python
class MockBackbone(nn.Module):
    """本物 Backbone の drop-in replacement(そのまま置き換え可能な代替)。"""
    def __init__(self, backbone="swin_v2_tiny", is_pretrained=True, **kwargs):
        super().__init__()
        self.backbone = MockBackboneModel()
        self.backbone_channels = 1440   # 本物が公開する属性も合わせておく
    def forward(self, image):
        return self.backbone(image)
```

`is_pretrained=True` など本物と同じ引数を受け取れる(が無視する)ので、本物の代わりにそのまま差し込めます。これを drop-in replacement(そのまま置き換え可能な代替)と呼びます。実際の差し替えは依存差し替えの仕組み(`unittest.mock.patch` など)で行います([unittest.mock.patchによる依存差し替え](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1155_unittest-mock-patch))。`backbone_channels` のような「本物が外部に公開している属性」も模倣しておかないと、後段が `model.image_backbone.backbone_channels` を参照したときに `AttributeError` で落ちます。mock は「呼ばれる API の表面」をすべて揃える必要がある点に注意します。

### mock の限界を明記する
mock は本物の一部しか模倣しません。たとえば上記の mock は4段階の特徴しか返しません。Swin と ConvNeXt は4段階の階層特徴を出しますが、ResNet50 を特徴抽出ライブラリ(features_only モードなど)経由で使うと、最初の stem を含めて5段階の特徴を出すことがあるため、この4段階 mock では合いません。よって5段階を前提とする backbone を試すテストは、本物を使うか、段階数を合わせた別の mock を用意する必要があります。「この mock は何を模倣し、何を模倣しないか」をコードのコメントに正直に書くことが、後で他人が踏む地雷を減らします。

## 手法の系譜と主要論文
mock backbone は機械学習の新手法ではなく、ソフトウェアテストの「テストダブル」という確立した概念の応用です。

Gerard Meszaros "xUnit Test Patterns"(2007)が、代役を Dummy / Stub / Mock / Fake に分類しました。Stub は「決まった応答を返すだけ」、Mock は「正しく呼ばれたかも検証する」点が違います。mock backbone は名前に反して多くの場合 Stub に該当します(出力形を返すだけで、何回どう呼ばれたかは検証しないため)。Martin Fowler "Mocks Aren't Stubs"(2007、エッセイ)は、この区別と「状態検証(出力が正しいか) vs 振る舞い検証(正しく呼ばれたか)」という対立軸を広めた代表的な文献で、なぜ安易に「全部モック」と呼ぶのが誤解を招くかを論じています。

形を合わせる根拠となる backbone 側の系譜は、Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"(ICCV 2021, arXiv:2103.14030)と Liu et al. "Swin Transformer V2: Scaling Up Capacity and Resolution"(CVPR 2022, arXiv:2111.09883)です。これらが「入力の1/4から1/32まで、4段階の階層特徴を出す」構造を定め、mock が再現すべき形の仕様になっています。さらに Liu et al. "A ConvNet for the 2020s"(ConvNeXt, CVPR 2022, arXiv:2201.03545)が、同じ4段階・同じ解像度比を純 CNN で実現したため、Swin と ConvNeXt が同一の mock で兼用できる根拠を与えています(詳しくは [SwinV2の4段階特徴マップ形状](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1016_swinv2-feature-map-shapes))。

機械学習システムの文脈では、Breck et al. "The ML Test Score"(IEEE Big Data 2017 / NeurIPS 2017 ML Systems Workshop)が、重い学習済みコンポーネントを軽量代役に置き換えてパイプライン全体を高速かつ決定的に検証することを推奨しています。mock backbone はまさにこの推奨の実装形です。

## 論文の実験結果(定量データ)
mock backbone 自体の論文評価はありませんが、効果は実測と関連研究で示せます。

- forward 時間: 本物の階層型 Transformer backbone は、256×256 入力で1枚あたりおよそ数十ms(GPU・モデルサイズによる)。1層 Conv の mock はおよそ1ms 未満。倍率にして数十倍の高速化です。マルチカメラ(V=6〜7)で `B*V` をまとめて処理する場合、本物では枚数に比例してさらに時間がかかるため、差はいっそう開きます。
- 重みロードの除去: 事前学習済み backbone は数千万〜数億パラメータを持ち、ファイルにして数百MBになります。Swin-T で約2,800万パラメータ、ConvNeXt-T で約2,900万パラメータ規模です。初回ダウンロードやディスクからの deserialize は数百ms〜秒オーダーで、mock 化するとこれがすべて消えます。
- スイート全体: backbone の forward と重みロードがテスト時間の70〜90%を占めるケースでは、mock 化とテスト準備の共有([pytestのfixtureと共有](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1142_pytest-fixtures))を併用して、スイート全体が約3分から1分未満になった報告があります。
- 決定性: 事前学習済み重みのダウンロード・読み込みが消えるため、ネットワーク不通やキャッシュ差・重みバージョン差による失敗が減り、テストの再現性が上がります。"The ML Test Score" のルーブリックでも、外部依存の除去とコンポーネント分離は加点項目で、実調査では多くの本番システムがこの項目で低評価だったと報告されています。
- アブレーション的な観点(どの要素を抜くと何が壊れるか): mock の Conv をやめて定数返し(あるいは `torch.zeros` 返し)にすると、forward はさらに速くなりますが、出力が入力に依存しないため逆伝播時の入力側勾配がゼロになり、gradient-flow テストが無意味になります。逆に `AdaptiveAvgPool` による形そろえをやめると、入力解像度が変わったとたんに後段が形不一致で落ちます。つまり「Conv(勾配のため)」と「AdaptiveAvgPool(形のため)」はどちらも省けない最小要素です。

## メリット・トレードオフ・限界
メリット:
- forward が数十ms から1ms未満に縮み、テスト全体が劇的に速くなる。
- 事前学習済み重みのダウンロード・読み込みが不要になり、ネットワークやファイル依存を断てる。テストが決定的になる。
- 形と勾配を保つので、後段(視点融合・Planner)の結線や勾配伝播のテストは本物と同じように行える。
- CPU だけでも高速に走るため、GPU のない CI 環境でも回せる。

トレードオフと限界:
- mock は本物の一部しか模倣しない。段階数やチャンネル数が違う backbone とは形が合わず、別の mock が要る。
- 「本物の backbone を含めた全体」が壊れていないかは mock では分からない。別途、本物を使う統合テスト(integration test)が必要([pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration))。
- mock が出すのは無意味な数値なので、出力の意味的な正しさ(物体を本当に捉えているか、精度が出るか)は検証できない。あくまで形と結線の検証用。
- 未解決の課題として、本物と mock の「形の仕様」がずれていく問題(仕様の二重管理)がある。本物の backbone をアップグレードして段階数や次元が変わったとき、mock を直し忘れると、テストは通るのに本番が壊れる。これを防ぐには、本物 backbone の出力形をスナップショットとして自動検証し、mock の前提とつき合わせる仕組み(コントラクトテスト)が望ましい。

## 発展トピック・研究の最前線
- コントラクトテスト(contract testing): mock と本物が「同じインターフェース・同じ出力形」を満たすことを自動で照合する手法。mock の仕様ずれを検出できる。本物 backbone の出力形を一度だけ計測してゴールデンファイルに保存し、mock の出力形をそれと突き合わせるのが実務的な実装。
- 軽量サロゲートモデルの自動生成: 本物のネットワークから入出力形だけを抽出し、ランダム初期化の小さなネットを自動で作るツール。手書き mock の保守コストを下げる方向。`torch.fx` などのグラフトレース機構を使って、本物の層構成から「形だけ同じ」軽量版を派生させる研究も進む。
- ML パイプラインのテスト体系化: "The ML Test Score" 以降、データ・モデル・インフラの各層をどうテストするかの研究が続き、重いコンポーネントの代役化はその基盤技術として位置づけられている。特に大規模事前学習モデル(基盤モデル)を部品として組み込むシステムでは、本物を毎回ロードできないため、形・勾配を保つ軽量代役の重要性が増している。

## さらに学ぶための関連トピック
- [unittest.mock.patchによる依存差し替え](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1155_unittest-mock-patch)
- [SwinV2の4段階特徴マップ形状](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1016_swinv2-feature-map-shapes)
- [pytestのfixtureと共有](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1142_pytest-fixtures)
- [pytestマーカーによる統合テスト分離](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1145_pytest-markers-integration)
