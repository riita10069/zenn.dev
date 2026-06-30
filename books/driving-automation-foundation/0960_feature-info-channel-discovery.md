---
title: "feature_infoによるチャネル自動発見"
---

# feature_infoによるチャネル自動発見

## ひとことで言うと
特徴抽出器（backbone、画像から特徴を取り出す本体）を取り替え可能にすると、下流の部品（例えばチャネル数を合わせる 1x1 の畳み込み）を作るために「各ステージで何チャネル出るか」を事前に知る必要が出てきます。これを人手で調べて数字をコードに書くのは間違いの温床です。そこで、特徴抽出ライブラリが提供する「各ステージの説明書き（メタ情報）」を読んでチャネル数を取得し、もしそれが無い抽出器なら「お試しで1回だけデータを流して、出てきた形を数える」フォールバックで自動発見する、という設計パターンです。

## 直感的な理解
配管工事を思い浮かべてください。上流のタンク（抽出器）から何本ものパイプ（各ステージの特徴）が出ていて、それぞれ太さ（チャネル数）が違います。下流で受け口（1x1 Conv）を作るには、繋ぐパイプの太さを先に知らないと部品の口径が合いません。

素朴なやり方は、タンクのカタログを見て「このタンクは192mm, 384mm, 768mm…」と手で書き写すことです。しかしタンクを別メーカー製に交換するたびにカタログを引き直し、書き間違えれば現場で繋がらず工事が止まります。賢いやり方は2つ。第一に「タンク自身に刻印された口径表示を読む」こと。第二に、刻印が無いタンクなら「水を一瞬流してみて、出口の太さを実測する」ことです。チャネル自動発見はこの2段構えをそのままコードにしたものです。

## 基礎: 前提となる概念
- ステージ（stage）: 抽出器は画像を段階的に縮小しながら抽象度を上げます。各段階がステージで、典型的には4段。浅いステージは解像度が高くチャネル数が少なく（細かい模様を見る）、深いステージは解像度が低くチャネル数が多い（大きな意味を見る）。
- マルチスケール特徴（multi-scale features）: 複数ステージの特徴を同時に使うこと。小さい物も大きい物も捉えるために重要で、検出・分割モデルで標準的に使われます。
- チャネル数（num_chs, number of channels）: あるステージの特徴マップが持つ特徴の本数。下流の層を作るときの必須パラメータです。
- メタ情報（metadata）: データそのものではなく「データについての説明」。ここでは「各ステージのチャネル数・縮小率」などの仕様情報を指します。
- timm（PyTorch Image Models）: Ross Wightman が 2019 年から開発する、数百種類の画像モデルを共通インターフェースで提供する事実上の標準ライブラリ（現在は Hugging Face 配下）。
- 契約 / 規約（contract）: 「このインターフェースに従えば、こういう情報がこういう形で必ず取れる」という約束事。ライブラリ利用側はこの約束に乗ることで、個別実装を知らずに済みます。

なぜ自動発見が要るのか。下流の例として、複数ステージを連結したあとにチャネル数を統一する 1x1 Conv を考えます。

```python
nn.Conv2d(backbone_channels, embed_dim, kernel_size=1)  # backbone_channels を先に確定する必要
```

ここで `backbone_channels`（全ステージのチャネル数の合計など）を間違えると、モデルを走らせた瞬間に形状不一致でエラーになります。抽出器を複数種類サポートするなら、種類ごとに正しい合計を保つのは現実的でなく、自動で問い合わせる仕組みが要ります。手書きの数字は、抽出器を1つ差し替えるたびに「更新漏れ」のリスクを抱え、しかもそのエラーは実行時まで発覚しません。

## 仕組みを詳しく
timm で `features_only=True` を指定してモデルを作ると、そのモデルは「マルチスケール特徴抽出器」になり、各ステージの特徴マップをリストで返すようになります。さらに各ステージのメタ情報を持つ `feature_info` という属性が付き、その要素の `num_chs` キーが「そのステージのチャネル数」です。

自動発見は2段構えにします。

優先パス（feature_info を読む、forward 不要）:

```python
info = getattr(self.backbone, "feature_info", None)
if info is not None:
    try:
        channels = [stage["num_chs"] for stage in info]
    except (TypeError, KeyError, IndexError):
        channels = None
    if channels:
        return channels
```

`getattr(obj, "feature_info", None)` は「その属性があれば取得、無ければ None を返す」安全な読み方です。各ステージから `num_chs` を取り出してリストにします。例えば 4 段の階層型 Transformer 系抽出器なら `[192, 384, 768, 768]` のような結果が返り、その合計（`sum`）が `backbone_channels` になります。`try/except` で囲うのは、規約に部分的にしか従わない実装（要素が辞書でない、キー名が違う等）でも例外で止めずにフォールバックへ落とすためです。

フォールバックパス（feature_info が無いとき、ダミー forward で推定）:

```python
dummy = torch.zeros(1, 3, 256, 256, device=device, dtype=dtype)
was_training = self.backbone.training
self.backbone.eval()
with torch.no_grad():
    out = self.backbone(dummy)
self.backbone.train(was_training)
...
return [f.shape[1] for f in out]
```

`torch.zeros(1, 3, 256, 256)` は「すべて 0 で埋めた、バッチ1・3チャネル(RGB)・256x256 の偽画像」です。これを抽出器に1回だけ流し、返ってきた各特徴マップの `shape[1]`（channels-first を仮定したチャネル軸）を読みます。`torch.no_grad()` は「勾配を計算しない」指定で、形を知りたいだけなので学習用の計算を省いて軽くします。実行前後で元の学習モード（`training` フラグ）を保存・復元し、副作用で学習状態を壊さないよう配慮します。BatchNorm（バッチ正規化、ミニバッチの統計で特徴を正規化する層）などは train/eval で挙動が変わり、しかも train モードのまま forward すると統計を更新してしまうため、この復元は地味に重要です。

フォールバックには安全チェックも入れます。返り値がリストでない／空、あるいは4次元テンソルでない場合は、原因が分かるエラーを投げます。

```python
if not isinstance(out, (list, tuple)) or len(out) == 0:
    raise ValueError("backbone did not return a non-empty list of feature maps")
for f in out:
    if f.dim() != 4:
        raise ValueError(f"returned a feature map with shape {tuple(f.shape)}; expected 4D")
```

なぜ feature_info を優先するか。ダミー forward は1回とはいえ計算コストがかかり、device（CPU/GPU）と dtype（数値精度）を揃える手間もあります。timm 標準の抽出器は feature_info を必ず持つので、その場合は読むだけで済みます。フォールバックは「timm のメタ情報を持たない自作抽出器でも動かす保険」です。

全体の流れ:

```
__init__
  └ build_backbone(name)            # timm features_only モデルを作る
  └ _infer_feature_channels()
        feature_info あり? ─yes─> num_chs を読む ─> [192,384,768,768]
                          ─no──> ダミー forward ─> 各 shape[1] を読む
  └ backbone_channels = sum(...)    # 例: 192+384+768+768 = 2112
```

こうして得たチャネルリストは、下流のレイアウト自動判定（[channels-first/last 自動判定](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0944_channel-layout-autodetect)）でも「期待チャネル数」の基準として再利用されます。つまりこの自動発見は、レイアウト判定とセットで「抽出器非依存の配線」を支える土台です。チャネル数を1か所で確定し、それを全下流が参照する設計は「単一の真実の源（single source of truth）」というソフトウェア設計の基本原則の具体例でもあります。

## 手法の系譜と主要論文
これは単一論文の「アルゴリズム」ではなく、ライブラリの設計規約を活用したエンジニアリングのパターンです。背景には「マルチスケール特徴を当たり前に使う」という潮流があります。

- ResNet（He et al., CVPR 2016, arXiv:1512.03385）。深い CNN を残差接続で学習可能にし、4ステージ構成・段階的ダウンサンプルという「以後の標準的なステージ構造」を確立しました。各ステージのチャネル数（例 ResNet-50 の 256/512/1024/2048）を問い合わせる需要の出発点です。
- FPN（Feature Pyramid Networks, Lin et al., CVPR 2017, arXiv:1612.03144）。抽出器の複数ステージから特徴を取り出し、上の層から下へ情報を流し込んで「どの解像度でも意味の濃い特徴」を作る構造を提案しました。これにより「抽出器から複数段の特徴を取り出して横展開する」設計が一般化し、各段のチャネル数を問い合わせる需要が決定的に高まりました。
- Mask R-CNN（He et al., ICCV 2017, arXiv:1703.06870）。FPN を物体検出・インスタンス分割に組み込み、マルチスケール特徴を前提とする設計を決定的に普及させました。
- timm の `features_only` / `feature_info` 規約（Wightman, 2019-）。上記の潮流を受けて、「どの抽出器でも同じ作法で段ごとの特徴とそのチャネル数・縮小率を取れる」共通インターフェースを提供しました。`feature_info.channels()` や `feature_info.reduction()` で各段のチャネル数・ダウンサンプル率を問い合わせられます。これにより検出・分割・自動運転など多様なフレームワークが、特定抽出器に縛られない配線を実現できるようになりました。

系譜としては「マルチスケールを使いたい（FPN/Mask R-CNN）」→「だから段ごとの仕様を機械的に取れる規約が要る（timm features_only）」→「その規約に乗れば配線を自動化できる（本パターン）」という流れです。

## 論文の実験結果(定量データ)
このパターン自体に固有のベンチマークはありませんが、土台となった研究の効果を押さえると、なぜマルチスケール前提が標準化したかが分かります。

- FPN（CVPR 2017）: COCO 物体検出で、FPN を使わない単一スケールのベースラインに対し、Faster R-CNN の検出精度（AP, Average Precision、検出枠の正確さの平均、高いほど良い）をおよそ +2〜+8 ポイント改善したと報告されています。特に小さい物体（AP_small）の改善が大きく、これは浅いステージの高解像度特徴を活用できたためです。AP は 0〜100 のスケールで、数ポイントの差でも検出タスクでは大きな意味を持ちます。
- Mask R-CNN（ICCV 2017）: COCO のインスタンス分割で当時の最先端を更新し、FPN 付き抽出器を用いることで mask AP（約 35〜37）・box AP ともに明確に向上しました。これにより「抽出器は複数段の特徴を出すもの」という前提が業界標準になりました。
- timm エコシステム: timm は数百種類の抽出器を同一 API で提供し、ImageNet ベンチマークの再現実装の事実上の参照になっています。features_only インターフェースの存在により、研究者は抽出器を1行変えるだけで複数モデルを公平に比較でき、これがアブレーション実験（要素を入れ替えて効果を測る実験）の高速化に大きく寄与しています。

アブレーション的な視点では、「チャネル数を手書きする」設計と「自動発見する」設計を比べると、抽出器を差し替えたときの破損率に明確な差が出ます。手書きは合計値の更新漏れで形状エラーを起こしますが、自動発見は feature_info があれば常に正しい値を返し、追加コードを要しません。

## メリット・トレードオフ・限界
- メリット: 抽出器ごとにチャネル数を人手で調べて書く必要がなく、差し替えてもコードが壊れない。
- メリット: timm 標準なら forward 不要（軽い）、非標準でもダミー forward で動く二段構えで堅牢。
- メリット: 不正な抽出器（リストを返さない、4D でない等）に対し、原因の分かるエラーを早期に出せる。
- トレードオフ: フォールバックのダミー forward は channels-first `(B, C, H, W)` を仮定するため、channels-last しか返さない非 timm 抽出器ではチャネル数を取り違える可能性がある（この縮退は [channels-first/last 自動判定](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0944_channel-layout-autodetect) と組み合わせても完全には防げない）。
- 限界: ダミー forward の入力は固定解像度（例 256x256）なので、極端に小さい入力でしか動かない抽出器や、入力サイズに依存して出力チャネルが変わる特殊な構造では失敗しうる。
- 限界: feature_info の `num_chs` は規約に基づく宣言値であり、実装が規約に反して実際の出力と食い違う抽出器では誤った値を信じてしまう。宣言と実体の乖離は、規約ベース設計に共通する原理的な弱点です。

## 発展トピック・研究の最前線
自動発見の発想は、チャネル数だけでなく「ステージ数・各段の縮小率（reduction）・正規化の有無・出力レイアウト」まで含めた完全な抽出器契約の自動取得へ拡張できます。近年は CNN・Vision Transformer・階層型 Transformer・state-space モデル（Mamba 系）など出力仕様の異なる抽出器を1つのフレームワークで横断比較する需要が高まっており、それぞれの feature_info を統一的に扱えることが実験基盤の前提になっています。さらに、事前学習済みの大規模基盤モデルを凍結して特徴抽出器として使う構成では、内部を改造せずメタ情報だけを頼りに配線する必要があるため、この種の規約ベース自動配線の重要性が増しています。研究の最前線では、特徴抽出器の選択自体を探索対象に含めるニューラルアーキテクチャ探索（NAS）や、複数抽出器の特徴を融合するマルチ backbone 構成も登場しており、いずれも「各抽出器の仕様を機械的に問い合わせられること」が前提になっています。

## さらに学ぶための関連トピック
- [channels-first/last 自動判定](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0944_channel-layout-autodetect)
- [Channel Projection（1x1 Conv によるチャネル次元射影）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0947_channel-projection-layer)
- [マルチスケール特徴のプーリング連結](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0988_multi-scale-feature-pooling)
- [build_* ファクトリ関数とレジストリパターン](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0937_build-factory-functions)
