---
title: "Linear LR Decay"
---

# Linear LR Decay

## ひとことで言うと
学習率(モデルのパラメータを1回の更新でどれくらい動かすかを決める数値)を、まっすぐな直線で一定のペースで減らしていき、学習の終点でちょうど 0(または指定の最小値)に到達させるスケジューラ(学習率の時間変化のさせ方)です。仕組みはとても単純ですが、序盤に学習率を少しずつ上げる「warmup(ウォームアップ)」と組み合わせた「warmup + 線形減衰」が、BERT をはじめとする Transformer(自己注意機構を使う現代の言語・画像モデルの基本構造)の学習で事実上の標準になっています。

## 直感的な理解
車で目的地に着いて駐車する場面を思い浮かべてください。最初は速度を出し(大きな学習率)、目的地が近づくにつれてアクセルを緩め、最後はぴたっと 0 km/h で止まりたい。線形減衰は「ゴールまでの残り距離に比例して一定の割合で減速し、ゴール地点でちょうど停止する」という素直な減速の仕方です。

ただし Transformer には独特の事情があります。エンジンが冷えている(パラメータがランダムな初期状態の)ときに、いきなりフルスロットルを当てると暴走(損失が発散、つまり下がるどころか無限大へ飛ぶ)します。だから発進直後だけ、ゆっくりアクセルを踏み込んで暖機運転をする。これが warmup です。warmup で速度を目標値まで上げきってから、線形減衰でゴールまで一定ペースで減速していく。この「上って下る」三角形が warmup + linear decay です。

なぜ 0 にきっちり着地させたいのか。学習の終盤に学習率が残っていると、パラメータは谷底の周りで揺れ続け、最終モデルがたまたま揺れの端にいることになります。終点で 0 に着地させれば、谷底でぴたっと止まった重みが得られます。

## 基礎: 前提となる概念
- 学習率: 学習は「予測誤差を測り、誤差を減らす方向にパラメータを少し動かす」の繰り返しで、この「少し」の大きさが学習率です。大きすぎると発散、小さすぎると遅い。

- ステップ (step / iteration): データを1バッチ見てパラメータを1回更新する単位。Transformer の学習ではエポックより「総ステップ数」で語ることが多いです。数万〜数百万ステップ回します。

- 自己注意 (self-attention): 入力の各要素が、ほかのどの要素にどれだけ注目するかを学ぶ仕組みで、Transformer の心臓部です。学習初期はこの注意の重みがランダムなため、出力が極端な分布に偏りやすく、大きな学習率を当てると一気に壊れます。これが Transformer の「初期不安定性」です。

- 発散 (divergence): 損失が下がるどころか急激に増えて NaN(計算不能)になる現象。warmup はこの発散を防ぐための安全装置です。

- スケジューラの位置づけ。[Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0627_exponential-lr-decay) は理論上 0 に到達せず、[Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0656_step-decay-lr) は不連続な段差があります。線形減衰はこれらの中間で、「決めた終点でちょうど 0 に着地」かつ「途中はなめらかで一定ペース」という最も単純な下り坂です。

## 仕組みを詳しく
warmup を含めた全体は2本の直線でできています。

warmup 区間(0 から `warmup_steps` まで、学習率を上げる):
```
lr(t) = lr_peak * (t / warmup_steps)
```

減衰区間(`warmup_steps` から `total_steps` まで、学習率を下げる):
```
lr(t) = lr_peak * (total_steps - t) / (total_steps - warmup_steps)
```

- `lr_peak`: 頂点(warmup が終わった瞬間)の学習率
- `warmup_steps`: 何ステップかけて 0 から lr_peak まで上げるか
- `total_steps`: 学習全体の総ステップ数(ここで学習率が 0 になる)
- `t`: 現在のステップ数

数値例。`lr_peak = 5e-4`、`warmup_steps = 1000`、`total_steps = 10000` とすると、

- ステップ 0: 0
- ステップ 500(warmup 半分): 2.5e-4
- ステップ 1000(頂点): 5e-4
- ステップ 2000: 約 4.44e-4
- ステップ 5000(全体の半分あたり): 約 2.78e-4
- ステップ 10000(終点): 0

学習率の時間変化をASCII図にすると、左肩上がりの短い上り坂(warmup)のあと、長い直線の下り坂が続く三角形です。

```
lr
5e-4|     /\
    |    /  \__
    |   /      \__
    |  /          \__
    | /              \__
  0 |/                  \__
    +------------------------> step
     ↑warmup  ↑頂点      ↑total_steps で0
```

実装は複数あります。warmup なしの単純な線形減衰なら、開始倍率 1.0 から終了倍率 0.0 まで `total_iters` ステップかけて線形補間するスケジューラ(PyTorch の `LinearLR` 相当)。warmup と組み合わせるなら、warmup_steps と total_steps を渡すと2本の直線を1つにまとめてくれる関数(HuggingFace の `get_linear_schedule_with_warmup` 相当)が広く使われます。BERT 系の学習では後者がデファクトです。

線形減衰は [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0647_polynomial-lr-decay)(多項式減衰)の power=1 の特殊ケースである、という関係も整理に役立ちます。多項式減衰の乗数を 1 にすると、まっすぐな直線になります。power を 2 にすれば終盤がゆっくり減る曲線、power を 0.5 にすれば序盤がゆっくり減る曲線になります。

なぜ線形なのか、コサインとの違い。コサイン減衰(cosine annealing)は終盤の減り方がゆるやかで、「最後にじっくり谷底を詰める」効果が強いと言われます。線形は終盤も一定ペースで 0 へ向かうため、その効果はやや弱い代わりに、挙動が直感的で、warmup の三角形と相性がよく、実装も最も単純です。タスクによってどちらが上かは変わり、決着はついていません。

## 手法の系譜と主要論文
- warmup の源流。Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, "Attention Is All You Need", NeurIPS 2017(arXiv:1706.03762)。オリジナルの Transformer 論文で、warmup の重要性を最初に強く示しました。提案した「Noam スケジュール」は、warmup 後に学習率をステップ数の平方根の逆数で下げるもので、純粋な線形ではありませんが、「warmup + 単調減衰」という枠組みの源流です。動機は自己注意の学習初期の発散を防ぐこと。新規性は、固定スケジュールに warmup を明示的に組み込んだ点。トレードオフは、平方根減衰が終点を 0 にきっちり決めにくいことで、これを後に BERT が分かりやすい線形へ置き換えました。

- 線形減衰の定着。Devlin, Chang, Lee & Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", NAACL 2019(arXiv:1810.04805)。1万ステップの warmup のあと学習率を線形に 0 まで減衰させるスケジュールで BERT を事前学習しました。動機は Transformer の初期不安定性を warmup で抑え、終盤は線形に下げてきれいに収束させること。効果は、多数の言語理解ベンチマークで当時の最高精度を更新し、このスケジュールを業界標準にしたこと。トレードオフは、warmup_steps と total_steps を事前に決め打ちする必要があり、学習量を変えると再調整がいることです。

- warmup の理論的説明。Liu, Jiang, He, Chen, Liu, Gao & Han, "On the Variance of the Adaptive Learning Rate and Beyond"(RAdam, ICLR 2020, arXiv:1908.03265)。なぜ warmup が必要なのかを理論的に分析し、Adam 系最適化の初期は適応学習率の分散が大きく、それが発散の原因だと示しました。新規性は、warmup を内部に組み込んだ最適化手法 RAdam の提案。トレードオフは、線形 warmup ほど単純ではなく、実務では結局 warmup + linear/cosine が広く残っていることです。

- 画像系への波及。ConvNeXt(Liu et al., CVPR 2022)や多くの Vision Transformer 系は、AdamW + warmup + cosine/linear 系のスケジュールで学習されており、線形減衰系の知識は言語に限らず画像モデルの学習にも直結します。

## 論文の実験結果(定量データ)
- BERT(NAACL 2019)。GLUE(General Language Understanding Evaluation、9 個の言語理解タスクをまとめたベンチマーク)で、BERT-large は平均スコア 80.5 を記録し、当時の最良手法を約 7 ポイント上回りました。SQuAD v1.1(質問応答データセット、指標は F1 = 適合率と再現率の調和平均)では F1 93.2 を達成し人間の性能(91.2)を上回りました。これらは warmup + linear decay スケジュールのもとで得られた数値で、スケジュールが Transformer の安定学習に不可欠だったことを示します。

- warmup を抜くアブレーション。RAdam 論文(ICLR 2020)では、warmup なしの Adam が学習初期に大きく発散し、最終精度が大きく劣化することを示しました。warmup ありに比べ、機械翻訳(IWSLT データセット、指標は BLEU = 機械翻訳の n-gram 一致度を測るスコアで高いほど良い)で数 BLEU の差が出る設定が報告されています。逆に言えば「warmup の有無」だけで Transformer の成否が分かれることがある、という強いメッセージです。RAdam は warmup なしでもこの発散を内部で抑え、warmup あり Adam に近い精度を出せることを示しました。

- warmup 長さの感度。BERT 系の実務報告では、warmup を総ステップの 1〜10% 程度に取るのが定番で、短すぎる(0.5% 未満)と初期発散、長すぎる(20% 超)と学習が遅れて最終精度がやや落ちる、という傾向が知られています。この「warmup_steps を何にするか」は依然チューニング対象です。

- 線形 vs コサインの実証比較。多くのベンチマークで、適切にチューニングした線形減衰とコサイン減衰の最終精度差はごく小さい(多くの場合 1 ポイント未満)と報告されています。コサインは終盤の減りが緩やかで「最後の詰め」がわずかに有利な設定がある一方、学習を途中で止めた場合は線形のほうがその時点の学習率が低く、中断時の性能が安定するという実務的利点も指摘されています。総ステップ数を最後まで使い切る前提ならコサイン、途中で止めうるならスケジュールの形を選び直せる柔軟さが重要、という整理になります。

- ファインチューニングでの線形減衰。BERT を下流タスクに適応させる(fine-tuning)際も、warmup 約 10% + 線形減衰が定番です。GLUE の小規模タスク(数千〜数万例)では学習が数エポックで終わるため、減衰の終点を 0 に合わせて「最後はほとんど更新しない」状態にすることが、小データでの過学習を抑える効果も持ちます。

指標の意味の補足。BLEU は 0〜100 で、数ポイントの差でも翻訳品質の体感差が出ます。GLUE 平均や SQuAD F1 も、終盤の 1 ポイントを詰めるのが難しい領域なので、スケジュールの違いで生まれる数ポイントの差は実用上大きな意味を持ちます。

## メリット・トレードオフ・限界
メリット
- 仕組みが直線2本だけで極めて単純、かつ終点でちょうど 0 に着地する。
- warmup と組み合わせることで Transformer の初期不安定性を抑え、発散を防げる。
- BERT 以降の膨大な実績があり、Transformer 学習の安全な既定値として信頼できる。

トレードオフ・限界
- warmup_steps と total_steps を事前に決め打ちする必要があり、学習量を変えると再設定がいる(適応はしない。[ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0651_reduce-lr-on-plateau) と対照的)。
- 単調に減るだけで、局所解・鞍点からの脱出機能はない([Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0621_cosine-warm-restarts) のような再起動はできない)。
- 終盤に学習率が直線的に 0 へ近づくため、コサイン減衰のように「終盤をゆっくり下げて精度を詰める」効果はやや弱い(タスクによりコサインのほうが最終精度が上という報告もある)。
- warmup の長さが短すぎると発散、長すぎると学習が遅れる、というシビアさがある。
- 研究上の未解決点として、warmup と減衰のスケジュールを最適化問題として原理的に導く理論はまだ十分でなく、多くは経験則です。

## 発展トピック・研究の最前線
- one-cycle / super-convergence。Smith & Topin(2018)は、学習率を一度大きく上げてから下げる「1サイクル」スケジュールで、通常より大幅に少ないステップ数で高精度に到達できる「super-convergence」を報告しました。線形 warmup の発展形と見なせます。
- WSD スケジュール(warmup-stable-decay)。大規模言語モデルの学習で、warmup の後に学習率を長く一定に保ち、最後の 10〜20% だけ急減衰させる方式が近年注目されています(MiniCPM, Hu et al. 2024 などで採用)。長い定常区間のあいだは total_steps を確定させずに学習を続けられ、いつでも「最後の減衰区間」を後付けして打ち切れるため、学習量を事前に決め打ちしなくてよいのが利点です。線形/コサイン減衰の「終点を最初に決める」という制約を外した発展形と言えます。
- 学習量に依存しないスケジュール。total_steps を事前に決めずに済むスケジュール(無限学習向けの一定学習率 + 後付け減衰など)が、学習を途中で止めたり延長したりしたい大規模学習の文脈で研究されています。
- スケーリング則との結びつき。モデル規模・データ量と最適学習率・warmup 長の関係(Chinchilla 系の研究)が、スケジュール設計を経験則から理論へ近づけつつあります。

## さらに学ぶための関連トピック
- [Polynomial LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0647_polynomial-lr-decay)
- [Cosine Annealing with Warm Restarts (SGDR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0621_cosine-warm-restarts)
- [Exponential LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0627_exponential-lr-decay)
- [Step / MultiStep LR Decay](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0656_step-decay-lr)
- [ReduceLROnPlateau](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0651_reduce-lr-on-plateau)
