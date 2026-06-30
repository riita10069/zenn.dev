---
title: "LR Scheduler (warmup/cosine)"
---

# LR Scheduler (warmup/cosine)

## ひとことで言うと
LR Scheduler（学習率スケジューラ）とは、学習中に「学習率」を時間とともに自動で上げ下げする仕組みです。学習率（learning rate, LR）は「重みを1回でどれだけ大きく動かすか」を決めるつまみで、これを最初はゆるやかに上げ（warmup）、その後なめらかに下げていく（cosine decay）のが、現代の深層学習の定番レシピです。固定の学習率より安定して速く、高い精度に収束できます。

## 直感的な理解
車の運転を思い浮かべてください。発進直後にいきなりアクセル全開にすると車体が暴れます。だからまずそっと踏み込んで加速し（warmup）、目的地が近づいたら速度を落として（decay）、ぴたりと停車します。学習率の調整もこれと同じ発想です。

学習率は深層学習で最も重要なハイパーパラメータ（人が事前に決める設定値）の1つです。大きすぎると1回の更新が大胆すぎて最適解を通り越して跳ね回り、最悪は発散します（loss が爆発して NaN になる）。小さすぎると更新がちょびちょびで収束に膨大な時間がかかります。

ここで決定的なのは「学習の最初と最後で、ちょうど良い学習率が違う」ことです。初期は重みがランダムで出力がめちゃくちゃなので、大きな学習率では更新が暴れて発散しやすい。だから最初はそっと小さく始めて徐々に上げたい（warmup）。終盤は解の近くまで来ているので、大きな学習率では細かい調整ができず最適解の周りで振動する。だんだん小さくして解にそっと着地させたい（decay）。固定の1つの学習率では、初期に安全な値は終盤に大きすぎ、終盤に良い値は全体として遅すぎ、どこを選んでも妥協になります。スケジューラはこの妥協を解消します。

## 基礎: 前提となる概念

### 学習率とハイパーパラメータ
学習率は最適化の更新式 `w ← w - lr × g`（w は重み、g は勾配）に現れる係数です。lr が大きいほど1歩が大きい。ハイパーパラメータとは、学習で自動調整される重みとは別に、人が学習前に決める設定値（学習率、バッチサイズ、エポック数など）を指します。

### ステップとエポック
- ステップ（iteration）: 1回の重み更新。1つのバッチを処理して1ステップ。
- エポック: 学習データ全体を1周すること。1エポック = データ数 / バッチサイズ ステップ。

スケジューラはステップ単位またはエポック単位で学習率を更新します。warmup + cosine ではステップ単位が一般的です。

### なぜ Transformer は warmup がほぼ必須か
Transformer は学習初期に Attention の出力スケールが不安定で、いきなり大きな学習率を使うと層の正規化や残差接続を通じて値が暴走し発散しやすいことが経験的に知られています。原論文も独自の warmup を採用しており、warmup なしだと学習が立ち上がらないことがあります。これがスケジューラを事実上必須にしました。

## 仕組みを詳しく

### warmup（線形ウォームアップ）
学習の最初の数百〜数千ステップで、学習率を0付近から目標値まで一直線に上げます。

数値例: 目標学習率 1e-4、warmup 1000ステップなら、
- ステップ0: lr = 0
- ステップ500: lr = 0.5e-4
- ステップ1000: lr = 1e-4（目標到達）
- 以降: decay フェーズへ

### cosine decay（コサイン減衰）
warmup 後は、コサインカーブ（なめらかな下り坂）に沿って学習率を最小値まで下げます。最初はゆっくり、中盤で速く、終盤でまたゆっくり減るので急激な変化がなく安定します。

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

記号の意味: t は warmup 終了後の経過ステップ、T は decay 全体のステップ数、lr_max は warmup で到達した最大学習率、lr_min は最終的に下げきる最小学習率（0 のことも多い）。cos は warmup 終了直後に1、中間で0、最後に -1 になるので、lr は lr_max → 中間で(lr_max+lr_min)/2 → lr_min と動きます。

数値例（lr_max=1e-4, lr_min=0, T=10000）:
- t=0: lr = 1e-4（cos(0)=1 → 最大）
- t=5000（半分）: lr = 0.5e-4（cos(pi/2)=0）
- t=10000（最後）: lr ≈ 0（cos(pi)=-1 → 最小）

### 組み合わせた全体像（ASCII図）
```
lr
1e-4 |        ___
     |       /   ‾‾‾‾‾‾---___
     |      /                ‾‾--__
     |     /(warmup)  (cosine decay) ‾‾--__
   0 |____/____________________________________ ステップ
     0  1000                            10000
```

### PyTorch での書き方（概念）
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=1000, num_training_steps=10000)
for step, batch in enumerate(loader):
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()      # ← 毎ステップ学習率を更新
    optimizer.zero_grad()
```
`scheduler.step()` を毎ステップ呼ぶと、内部で現在のステップに応じた学習率を計算し optimizer に反映します。

### 他のスケジュール
- step decay: 一定エポックごとに学習率を1/10にする昔ながらの方式。シンプルだが段差ができ、その瞬間に収束が不連続に変わる。
- exponential decay: 毎ステップ一定倍率で減らす。
- ReduceLROnPlateau: 検証 loss が改善しなくなったら学習率を下げる適応的方式。総ステップ数を事前に決めなくてよいのが利点。
- OneCycle（Smith 2018）: 学習率を一度大きく上げてから下げる。高速学習（super-convergence）に使う。
- inverse square root（逆平方根減衰）: Transformer 原論文の方式。warmup 後 1/√step で減衰。
- WSD（Warmup-Stable-Decay）: warmup 後に一定値で長く保ち、最後だけ急減衰する近年の方式。総ステップを後から延長しやすく、大規模 LLM 学習で採用が増えている。

現在の Transformer/画像認識で最も広く使われるのは warmup + cosine decay です。

## 手法の系譜と主要論文

- Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts"（ICLR 2017, arXiv:1608.03983）。コサインスケジュールの代表的提案。学習率をコサインで下げ、ときどき最大値に「再起動（warm restart）」する。動機: 周期的に上げ直すことで浅い谷から抜け、より良い谷を探せると考えた。トレードオフ: restart 周期というハイパーパラメータが増える。近年は restart なしの単純な1回コサイン減衰（warmup と組む）が主流。
- Goyal et al., "Accurate, Large Minibatch SGD"（FAIR, 2017, arXiv:1706.02677）。warmup の重要性を明確化。大バッチ学習で最初の数エポックは学習率を線形に上げる gradual warmup を使う。動機: 大バッチでは初期の大きな学習率が発散を招くため。
- Vaswani et al., "Attention Is All You Need"（NeurIPS 2017）。独自の warmup + 逆平方根減衰スケジュール。Transformer の初期不安定性に対処し、warmup を事実上の必須要素にした。
- Smith, "Cyclical Learning Rates"（2017）/ "Super-Convergence"（2018）。学習率を周期的・大胆に動かすと、かえって速く高精度に収束する場合がある。効果: 学習時間の大幅短縮。トレードオフ: 安定性が問題依存で必ず効くわけではない。
- 近年の LLM 学習: cosine decay が長く標準だったが、総ステップ数を事前固定する必要がある点が弱点とされ、WSD 系（一定+末尾急減衰）への移行が進む。学習を継続・延長しやすいのが理由。

## 論文の実験結果(定量データ)

指標と意味を示します。

- 検証精度・収束速度: SGDR の論文（Loshchilov & Hutter, ICLR 2017）では CIFAR-10/CIFAR-100 で、コサインスケジュール（warm restart 付き）が固定 LR や step decay より高い最終精度と速い収束を示した。具体的には、同じ総エポックで step decay より test error（テスト誤分類率、小さいほど良い）が低下し、CIFAR-10 で当時の強いベースラインに対し誤差を改善。さらに restart の各サイクル終端で得たモデルを集めてアンサンブル（複数モデルの予測を平均）すると、追加学習なしでさらに誤差が下がることも報告された。
- 大バッチでの精度維持: Goyal et al.（FAIR 2017）は warmup を入れることで、バッチ8192の ResNet-50 ImageNet 学習でもバッチ256と同等の Top-1 精度（約76%台）を達成。warmup を抜くと初期に発散、または精度が数ポイント低下する（アブレーション）。gradual warmup（数エポックかけて線形に上げる）と constant warmup（最初だけ小さい固定値）の比較では、前者のほうが安定した。
- Transformer の安定性: Vaswani らの原論文（NeurIPS 2017）の設定で warmup を外すと学習が発散・停滞することが経験的に広く確認されている。warmup ステップ数（原論文では4000）を変えると最終 BLEU（翻訳品質、大きいほど良い）や perplexity（言語モデルの予測の当てにくさ、小さいほど良い）が変わり、短すぎると不安定、長すぎると収束が遅れる。warmup の効果は Adam の二次モーメント（勾配の分散推定）が初期に不安定で実効学習率が暴れることを抑える点にあると後続研究で説明されている。

アブレーション的要点: (1) warmup を抜くと初期発散リスクが上がる、(2) cosine の総ステップ T を実際の学習長より短く設定すると、学習途中で lr がほぼ0になり停滞する、(3) 逆に T を長すぎに設定すると lr が下がりきらず終盤の精度が伸びない。T の設定が精度を左右する最重要ポイントです。

## メリット・トレードオフ・限界

メリット
- 初期の発散を防ぎ（warmup）、終盤の振動を抑える（decay）ことで、固定 LR より安定して高精度に収束しやすい。
- Transformer や大バッチ学習では事実上必須で、これなしだと学習が立ち上がらないことがある。
- 事前学習済みモデルの微調整（fine-tuning）で、初期に大きな lr を使って良い特徴を壊すのを防げる。
- 実装が容易（PyTorch/Hugging Face に既製スケジューラがある）。

トレードオフ・限界
- ハイパーパラメータが増える（warmup ステップ数、最大/最小学習率、総ステップ数 T）。T を間違えると停滞または下がりきらない。
- 総ステップ T を事前に決める必要がある cosine は、学習を途中で延長/短縮しにくい（WSD はこの弱点を緩和）。
- スケジュールが問題依存で、ある設定で最適でも別データでは再調整が要る。
- restart 付き SGDR は周期設定が難しく、効果が安定しない場合がある。

未解決・研究課題: 「データ・モデル規模からスケジュール形状を自動決定する」「学習を延長してもスケジュールを破綻させない」「最適化器（AdamW など）と相性の良いスケジュール形状」は今も探索が続く領域です。

## 発展トピック・研究の最前線
- 学習率と他ハイパーパラメータの相互作用: warmup・学習率・重み減衰・バッチサイズは独立でなく、Linear Scaling Rule のように連動して調整するのが定石。muP（maximal update parametrization）のように、モデル幅を変えても最適学習率が動かないパラメータ化の研究も進む。
- スケジュールフリー最適化: Defazio らの schedule-free optimizer のように、明示的なスケジュールなしで cosine 相当の挙動を得る試み。
- WSD と継続学習: 一定 LR で長く保つ区間を設けることで、いつでもチェックポイントから学習を続けられる設計が大規模事前学習で重視されている。
- warmup の理論的理解: なぜ warmup が安定化に効くのかを Adam の二次モーメント推定の初期分散などから説明する研究が進んでいる。

## さらに学ぶための関連トピック
- [Gradient Accumulation](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0631_gradient-accumulation)
- [EMA (指数移動平均重み)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0173_ema-model-weights)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer)
- [Transformer 時間方向アテンション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0344_transformer-temporal-attention)
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0133_mixed-precision-training)
