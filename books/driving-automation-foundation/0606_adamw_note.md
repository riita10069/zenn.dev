---
title: "AdamW (Decoupled Weight Decay)"
---

# AdamW (Decoupled Weight Decay)

## ひとことで言うと
AdamW(アダム・ダブリュー)は、Adam の「weight decay(重み減衰)」という正則化のかけ方が実は間違っていた、という問題を直した改良版オプティマイザです。weight decay とは、重みが大きくなりすぎないように毎ステップ少しずつ 0 へ縮めて過学習を防ぐ手法です。AdamW は、この「重みを縮める処理」を勾配の計算から切り離して(decouple、デカップル=分離して)、純粋に重みだけに直接かけることで、正則化が設計通りに効くようにしました。Transformer や ViT(Vision Transformer、画像を Transformer で扱うモデル)、大規模言語モデルを学習するときの事実上の標準オプティマイザです。

## 直感的な理解
ニューラルネットを学習させると、重みの値がどんどん大きくなっていくことがあります。重みが大きいモデルは、入力のわずかな違いに敏感に反応し、訓練データの細かいノイズまで覚え込みやすく、未知データで性能が落ちる「過学習(overfitting)」を起こします。これを抑える定番の手段が weight decay で、毎ステップ重みをほんの少しだけ 0 へ引き戻して「大きくなりすぎないよう手綱を締める」操作です。

ここで素朴な疑問が生まれます。「Adam を使うとき、この手綱の締め方はどうすればいいのか?」 長らく多くの実装は、SGD と同じやり方(損失関数に重みの二乗を足す=L2正則化)で済ませていました。ところが Adam はパラメータごとに歩幅を割り算で調整する仕組み(勾配を `√v` で割る)を持つため、この素朴なやり方だと手綱の締め方まで割り算でゆがめられ、重みによって締めたり緩めたりがバラバラになってしまいます。AdamW は「手綱は手綱として、割り算とは別に、まっすぐ一律にかける」ことで、本来意図した正則化を取り戻します。たったこれだけのことで、汎化性能(未知データでの性能)とハイパーパラメータの扱いやすさが目に見えて改善しました。

## 基礎: 前提となる概念
- 正則化(regularization): モデルが複雑になりすぎる(過学習する)のを防ぐ仕掛けの総称。weight decay はその一種。
- weight decay(重み減衰): 毎ステップ重みを `w ← w − η·λ·w` のように少しずつ 0 へ縮める操作。`λ`(ラムダ)は縮める強さを決める係数(典型的には 0.01〜0.1)。
- L2正則化(L2 regularization): 損失関数に重みの二乗和 `(λ/2)·‖w‖²` を足す方法。これを微分すると勾配に `λ·w` という項が増える。`‖w‖²` は全重みを二乗して足したもの。
- なぜ「同じはず」だったのか: 純粋な SGD では、L2正則化(損失に二乗を足す)と weight decay(重みを直接縮める)は数学的に完全に一致します。だから長年区別されずに同義語として使われてきました。AdamW の本質的な貢献は「Adam ではこの2つが一致しない」と明確に示した点にあります。
- 適応的スケーリング(adaptive scaling): Adam が勾配を `√v`(勾配二乗の移動平均の平方根)で割る操作。これが正則化をゆがめる元凶になる。前提として [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0616_adam_note) [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0652_rmsprop) を理解しておくと良いです。

## 仕組みを詳しく
Adam では、weight decay を「勾配に `λ·w` を足す」L2方式で入れると、その `λ·w` まで `√v` で割られてしまいます。何が起きるか。勾配が大きくよく動く重み(`√v` が大きい)は weight decay の効きが弱まり、逆に勾配が小さくあまり動かない重み(`√v` が小さい)には weight decay が強く効きすぎます。つまり「重みを一律に λ の割合で縮める」という本来の意図が崩れ、パラメータごとにバラバラの正則化になります。皮肉なことに、本来は最も縮めて抑えたい(よく動いて大きくなりがちな)重みほど、weight decay が弱まってしまうのです。

AdamW の解決策はシンプルで、weight decay を勾配にではなく、更新の最後で重みに直接かけます。両者を並べます。

通常の Adam(L2正則化を勾配に混ぜる方式):
```
g ← g + λ · w                      ← weight decay を勾配に足してしまう(ここが問題)
m ← β1·m + (1−β1)·g
v ← β2·v + (1−β2)·g²
w ← w − η · m̂ / (√v̂ + ε)           ← λ·w も √v̂ で割られて歪む
```

AdamW(weight decay を分離):
```
m ← β1·m + (1−β1)·g                ← 勾配 g には weight decay を混ぜない
v ← β2·v + (1−β2)·g²
w ← w − η · m̂ / (√v̂ + ε) − η · λ · w   ← 最後に weight decay を別項として直接かける
```

違いは `λ·w` を勾配 `g` に混ぜるか、更新式の最後に独立した項として足すか、だけです。AdamW では `λ·w` が `√v̂` で割られないので、すべての重みが学習率 η に応じて一律 `η·λ` の割合で縮みます。これが本来意図した weight decay の挙動です。

### 数値例
`η = 0.001`、`λ = 0.01` とします。AdamW では勾配の大小に関係なく、すべての重みが毎ステップ `η·λ·w = 0.001·0.01·w = 0.00001·w` ぶん 0 へ縮みます。重み `w=2.0` なら毎回 `0.00002` 縮む、という一律の正則化です。一方の通常 Adam だと、`√v` が大きい重み(例えば `√v=10`)では縮みが 10 分の 1 になり、`√v` が小さい重み(例えば `√v=0.1`)では 10 倍に効きすぎ、重みによって縮み方が 100 倍も変わってしまいます。

### λ と η の解きほぐし(decoupling の効能)
L2方式では、`η` を変えると正則化の実効的な強さも一緒に変わってしまい、2つが絡み合っていました。学習率を下げると正則化も弱まる、という連動が起きるのです。これは、学習率スケジュール(学習が進むにつれ η を下げていく手順)を使うと、終盤で正則化が勝手に弱まるという厄介な副作用を生みます。AdamW でも `η·λ` が weight decay 項なので確かに `η` を変えれば正則化も変わりますが、`λ` を独立に調整できるため、ハイパーパラメータ探索の格子(grid、η と λ の組み合わせを総当たりする表)を `η` 軸と `λ` 軸で素直に切れるようになります。Loshchilov & Hutter はこの「探索空間がほぼ直交する(2つの軸が独立に効く)」ことを大きな実用メリットとして強調しました。

### メモリ・計算
AdamW のメモリと計算量は Adam と同じです(`m` と `v` の2セット、追加メモリは重みの2倍)。違うのは weight decay を足す場所だけなので、コストはほぼゼロで実装できます。多くの深層学習フレームワークが標準で AdamW を提供しています。「Adam に weight_decay 引数を渡す」古い実装は内部で L2方式になっていることがあるため、正しい正則化を効かせたいなら明示的に AdamW を選ぶ必要があります。この「同じ名前の引数なのに中身が違う」落とし穴は、再現性の問題としてもしばしば指摘されてきました。

## 手法の系譜と主要論文
- Adam — Kingma & Ba (ICLR 2015)。AdamW の更新本体(`m`, `v`, バイアス補正)はそのまま Adam を踏襲します。[Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0616_adam_note)。
- L2正則化と weight decay の歴史的混同 — Hanson & Pratt (1988) ら初期のニューラルネット研究以来、両者は同義として扱われてきました。SGD では実際に一致するため、誰も区別する必要を感じなかったのが背景です。
- Decoupled Weight Decay — Loshchilov & Hutter。初出は arXiv:1711.05101(2017)、改訂を経て ICLR 2019 に採択。Adam の weight decay を分離型に変更し、SGD でも cosine 学習率スケジュール下では L2 と weight decay の挙動が異なることを指摘しました。同論文は SGDW(SGD の分離型 weight decay 版)も提案しています。
- 普及の歴史: AdamW はその後 BERT(Devlin et al., NAACL 2019)、GPT 系、ViT(Dosovitskiy et al., ICLR 2021)、画像系の DeiT(Touvron et al., 2021)・Swin Transformer(Liu et al., 2021)など Transformer/近代アーキテクチャの学習でほぼ標準採用され、「Transformer を学習するならまず AdamW」という慣習を作りました。
- 後続: AdamW は AMSGrad(収束修正)や warmup(RAdam)、メモリ削減(Adafactor, 8-bit Adam)などと組み合わせて使われます。更新本体を軽くする方向では [Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0638_lion-optimizer) が AdamW の分離型 weight decay を踏襲しつつ更新式を sign+単一モーメントに置き換えました。

## 論文の実験結果(定量データ)
Loshchilov & Hutter (ICLR 2019) の主な定量的主張は次の通りです。

- CIFAR-10 の画像分類(ResNet 系)で、Adam(L2方式)に比べて AdamW はテスト誤差が改善し、よく調整した SGD+Momentum との差を縮めました。具体的には、Adam が SGD に明確に劣っていたテスト精度のギャップ([Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0616_adam_note) でも触れた Wilson et al. 2017 の問題)を、weight decay を分離するだけでほぼ埋められると報告。ここでの「テスト誤差」は未知データでの誤分類率で、低いほど良い指標です。
- ハイパーパラメータの解きほぐし: `η` と `λ` を変えたときのテスト精度の等高線(ヒートマップ)を示し、L2方式では性能の良い領域が斜めに伸びて(η と λ が連動して)探索しにくいのに対し、分離型では良い領域が軸に沿った楕円としてまとまり、`η` と `λ` を独立に探索できることを可視化しました。これは厳密な数値比較というより「探索のしやすさ」を示す重要な結果で、実務での調整コストを下げる効果があります。
- ImageNet32(ImageNet を 32×32 に縮小したデータセット)など複数設定でも分離型が一貫して同等以上の汎化を示し、cosine 学習率スケジュール(学習率を周期的に下げる)や warm restarts(学習率を周期的に再加熱する)との相性が良いと報告しました。

アブレーション的に言えば、変更点は「weight decay を足す場所」のただ1か所です。それ以外(`m`, `v`, バイアス補正、学習率スケジュール)は Adam と完全に同一。にもかかわらず汎化が改善する、という事実が「場所が本質的に重要だった」ことの証拠になっています。逆に言えば、weight decay を 0 にすれば AdamW と Adam は完全に一致するので、効果はすべて weight decay の効き方の違いから来ています。

実務上の重要な注意点として、後続のレシピ研究では「weight decay をかけるべきでないパラメータ」を除外する設定が定番化しました。具体的には LayerNorm / BatchNorm のスケール(γ)・バイアス(β)、各層のバイアス項、および埋め込み層の一部です。これらは「重みを 0 へ縮めること」が意味をなさない、あるいは正規化の働きを壊すパラメータです(例えば LayerNorm のスケールを 0 に縮めると出力がつぶれる)。ViT・BERT の標準レシピでは、これらを decay 対象から外す「パラメータグループ分け(parameter grouping)」が行われます。この設定を怠ると、ViT 系で数ポイント精度が落ちることが経験的に報告されています。

## メリット・トレードオフ・限界
メリット
- weight decay が設計通り一律に効くため正則化が予測しやすく、汎化性能が改善する。
- `λ` と `η` をほぼ独立に調整でき、ハイパーパラメータ探索が楽。学習率スケジュールを使っても正則化の強さが勝手に変わらない。
- Adam と同じ計算量・メモリで、変更は weight decay の場所だけ。導入コストが低い。
- Transformer/ViT 系で実績豊富。デファクトスタンダード。

トレードオフ・限界
- Adam 同様 `m` と `v` の2セットで追加メモリが重みの2倍([Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0638_lion-optimizer) が削りにいく点)。
- `λ` と `η` の調整は依然必要で、最適値はモデル規模・データセット・バッチサイズで変わる。
- Adam の収束理論上の課題(AMSGrad 系の指摘)は weight decay の修正とは別問題として残る。
- LayerNorm のスケール・バイアスや埋め込みを decay 対象から除外する設定が必要。怠ると性能低下。

未解決の課題として、「分離型 weight decay がなぜ sharp minima(尖った極小値=過学習しやすい)を避けて汎化を助けるのか」の理論的説明はまだ発展途上です。実務では効くことが確立しているものの、最適 `λ` をモデル規模から事前に予測する一般則はまだありません。

## 発展トピック・研究の最前線
- スケール則と weight decay: 大規模言語モデルでは、weight decay と学習率スケジュールの相互作用が損失曲線に与える影響(損失地形を緩やかな川筋に見立てる「river valley」的な解釈など)が研究されており、weight decay を学習率に応じて調整するレシピが議論されています。weight decay が実質的に重みノルムの定常状態(平衡点)を決めるという見方も提示されています。
- 独立 weight decay(independent weight decay): `η` に依存しない形で `w ← w − λ·w` とする変種もあり、スケジューラとの組み合わせで挙動が変わります。実装によって `η` を掛けるか掛けないかが異なるため、論文の数値を再現する際は注意が必要です。
- 軽量化との両立: Adafactor + 分離 weight decay、8-bit AdamW など、メモリ削減と正しい正則化を両立する実装が巨大モデル学習で使われています。低ランク勾配射影の GaLore なども AdamW の更新を低メモリで近似する方向の研究です。
- スケール不変性の理解: BatchNorm/LayerNorm を含むネットでは、正規化層の前の重みのスケールは出力に影響しない(スケール不変)ため、weight decay は実質的に「実効学習率を調整するノブ」として働くという解釈(Zhang et al., van Laarhoven ら)があり、weight decay の役割を再定義する研究が続いています。

## さらに学ぶための関連トピック
- [Adam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0616_adam_note)
- [Lion (EvoLved Sign Momentum)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0638_lion-optimizer)
- [RMSProp](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0652_rmsprop)
- [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0653_sgd-momentum)
