---
title: "EMA (指数移動平均重み)"
---

# EMA (指数移動平均重み)

## ひとことで言うと
EMA（Exponential Moving Average, 指数移動平均）重みとは、学習中に揺れ動くモデルの重みの「なめらかな平均」を別途こっそり計算しておき、推論（予測）のときはその平均された重みを使う手法です。学習中の重みは更新のたびに細かく振動しますが、その平均をとると振動が消え、より安定して精度の高いモデルが得られます。追加の計算コストはほぼゼロで、物体検出・生成モデル・半教師あり学習などで広く効果が確認されています。

## 直感的な理解
体重計に毎日乗ると、食事や水分で値が日々±1kg ほど揺れます。1日の値だけ見ても「本当の体重」は分かりませんが、1週間の移動平均をとれば揺れが消え、本当の傾向が見えます。EMA はこれと同じで、毎ステップ揺れる重みの移動平均をとって「本当に良い重み」を取り出します。

学習中、モデルの重みは1ステップごとに更新され、その値は小刻みに振動します。特に終盤、解の近くまで来ると、重みは最適解の周りを行ったり来たり振動するだけでなかなか止まりません。これは「ミニバッチごとに勾配にノイズが乗る」ためで避けられません。ステップ10000、10001、10002 …の重みはどれも「最適解の周りのちょっとずつ違う点」で、どの1点を最終モデルにしてもそれは振動の中の偶然の1点にすぎません。運が悪ければ少し外れた点かもしれない。ならばその平均をとれば真ん中（より良い点）に近づくのでは、という発想が EMA です。

この効果は次の場面で特に顕著です。物体検出（精度が安定して上がる）、生成モデル（GAN・拡散モデルは学習が不安定なため EMA 重みのほうが生成品質が明確に良い。多くの拡散モデルが EMA 重みを配布)、半教師あり学習（EMA 重みを教師として使うと安定。後述の Mean Teacher）。

## 基礎: 前提となる概念

### 重みとパラメータ
モデルの重み（パラメータ）は、入力から出力を計算するための無数の数値です。学習とは、損失が小さくなるようこれらの数値を少しずつ更新していくことです。更新には勾配（loss を各重みで微分したもの）と学習率を使います。

### 勾配ノイズと振動
ミニバッチは全データの一部なので、そこから計算した勾配は「全データの真の勾配」に対してノイズを含みます。このノイズが、解の近くで重みを止まらせず振動させます。振動の幅は学習率に比例し、学習率を下げると振動も小さくなります（だから学習率スケジューラで終盤に下げる、[LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0150_learning-rate-scheduler)）。EMA はもう一つの振動対策で、振動そのものを平均化して消します。

### Polyak averaging（平均化の理論的背景）
確率的最適化では、最後の1点ではなく軌跡の平均を使うとノイズが打ち消され、最適点に統計的に近づくことが古くから知られています（Polyak averaging）。EMA はこの考えの実用版です。

## 仕組みを詳しく

### 単純平均ではなく「指数」移動平均
全ステップの単純平均でもよさそうですが、それだと「学習初期のひどい重み」まで同じ重みで混ざり平均が悪くなります。そこで「最近の重みほど重く、昔の重みほど軽く」効かせる指数移動平均を使います。更新式は単純です。

```
ema_weight = decay * ema_weight + (1 - decay) * current_weight
```

記号の意味:
- `current_weight`: 今ステップの学習中モデルの重み
- `ema_weight`: 平均された影のモデルの重み
- `decay`: 0.999 や 0.9999 のような1に近い数。どれだけ過去を重視するか

decay=0.999 なら毎ステップ「過去の平均99.9% + 今の重み0.1%」を混ぜます。1回の変化は微小で、ema_weight はゆっくりなめらかに動きます。

### 数値例
current_weight がノイズで 0.50 → 0.54 → 0.48 → 0.52 と振動しているとします。decay=0.99 の EMA は、
- ema ≈ 0.50（初期）
- 0.99×0.50 + 0.01×0.54 = 0.5004
- 0.99×0.5004 + 0.01×0.48 = 0.5000
- 0.99×0.5000 + 0.01×0.52 = 0.5002

入力が ±0.04 振れても ema はほぼ 0.500 に張り付き、振動を吸収しているのが分かります。

### decay の意味（実効的な平均ウィンドウ）
decay が1に近いほど過去をたくさん覚え、なめらかになりますが、新しい変化への追従が遅れます。おおよそ「直近 1/(1-decay) ステップの平均」に相当します。
- decay = 0.99 → 約100ステップの平均
- decay = 0.999 → 約1000ステップの平均
- decay = 0.9999 → 約1万ステップの平均

学習総ステップが短いのに decay を大きくしすぎると、EMA がいつまでも初期の悪い重みを引きずり、本体モデルより精度が低くなります。総ステップ数に合わせて decay を選ぶ必要があります。学習初期はバイアス補正（ウォームアップ）を入れる実装も多く、最初の数百ステップは decay を小さめにして本体に素早く追従させます。

### PyTorch での書き方（概念）
```python
ema_model = copy.deepcopy(model)            # 影のモデルを複製
for p in ema_model.parameters():
    p.requires_grad_(False)                 # 影は学習しない
for batch in loader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()                        # 本体だけを更新
    optimizer.zero_grad()
    with torch.no_grad():                   # EMA を手動更新
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(decay).add_(p, alpha=1 - decay)
# 推論・保存には ema_model を使う
```
ポイント: 学習するのは本体 `model` だけで、勾配は EMA に流しません。EMA は本体を毎ステップ「観察して平均する」受動的なコピーです。推論や最終チェックポイント保存に EMA を使います。PyTorch には `torch.optim.swa_utils.AveragedModel`（EMA モードあり）、timm には `ModelEmaV3` という既製実装があります。

### メモリと計算
EMA は本体と同じサイズの重みをもう1組メモリに持つので、パラメータのメモリが約2倍になります。ただし EMA 側は勾配やオプティマイザ状態（Adam なら1次・2次モーメント）を持たないので、学習全体のメモリ増は限定的です。計算は毎ステップの重み平均だけでごく軽量。

### BatchNorm の running statistics に注意
BatchNorm はバッチ統計（running mean/var）を持ち、これは勾配で更新される「重み」ではなく前向き計算で蓄積されるバッファです。EMA で重みだけ平均してこのバッファを放置すると、推論時に統計がずれます。対策は (1) バッファも EMA に含めてコピーする、(2) 推論前に学習データを一度流して EMA 重みでバッファを再計算する、のいずれか。SWA の論文はこの再計算を明示的に推奨しています。

## 手法の系譜と主要論文

- Polyak & Juditsky, "Acceleration of stochastic approximation by averaging"（SIAM J. Control Optim., 1992）。理論的源流。確率的最適化（SGD）で最後の1点でなく軌跡の平均を使うと収束が加速・安定する（Polyak averaging）。平均がノイズを打ち消し最適点に統計的に近づくため、理論的に最適な収束レートを達成。EMA はこの単純平均を「最近重視」に拡張した実用版。
- Tarvainen & Valpola, "Mean teachers are better role models"（NeurIPS 2017）。半教師あり学習での代表例。学習する「生徒モデル」と、その重みの EMA である「教師モデル」を用意し、ラベルなしデータでは教師の予測に生徒を合わせる。動機: 教師（EMA）の予測は生徒の振動を平均化していて安定・高品質なので良い学習目標になる。トレードオフ: 教師モデルぶんのメモリと decay 調整。
- Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization"（SWA, UAI 2018）。学習後半の複数チェックポイントの重みを単純平均する Stochastic Weight Averaging。動機: 平均された重みは「広く平らな谷（wide optima）」に位置し汎化が良くなる。EMA と思想は同じで平均の取り方が違うだけ（SWA は周期的サンプルの単純平均、EMA は毎ステップの指数平均）。
- Ho et al., "DDPM"（NeurIPS 2020）以降の拡散モデル: EMA 重みでサンプリングするのがほぼ標準。EMA なしだと生成画像にノイズや破綻が出やすい。
- 半教師・自己教師あり学習への展開: BYOL（Grill et al., 2020）, MoCo（He et al., 2020）の momentum encoder も EMA で更新される target ネットワークで、Mean Teacher の系譜上にある。EMA target が崩壊（collapse）を防ぐ鍵とされる。

## 論文の実験結果(定量データ)

指標と意味を示します。

- 半教師あり分類精度: Mean Teacher（Tarvainen & Valpola, NeurIPS 2017）は SVHN や CIFAR-10 で、当時の手法（Π-model, Temporal Ensembling）を上回る精度を達成。たとえば SVHN のラベル250枚という極端に少ない設定で誤分類率を大きく下げ（報告では数%台まで低下）、EMA 教師がノイズの少ない学習目標を与えることの効果を示した。EMA をやめて生徒の生重みをそのまま教師にすると、教師の予測が振動して学習目標が不安定になり精度が劣化する（アブレーション）。decay を 0.99 から 0.999 に上げると、ラベルが少ない設定ほど効果が大きいことも報告された。
- 汎化（テスト精度）: SWA（Izmailov et al., UAI 2018）は CIFAR-10/100 や ImageNet で、追加コストほぼなしに通常学習よりテスト精度を一貫して改善（報告では ImageNet の ResNet で約0.6〜0.9ポイント、CIFAR でそれ以上の改善）。損失地形の解析で、平均された重みがより平らで広い谷（wide optima）に位置し、訓練損失とテスト損失の曲面のズレに対して頑健であることを示した。鋭い谷では訓練・テストの最小点が水平にずれると損失が急増するが、平らな谷ではこのズレに強い。
- 生成品質（FID）: 拡散モデル（Ho et al., DDPM, NeurIPS 2020 ほか）では EMA 重みでサンプリングすると FID（Fréchet Inception Distance、生成画像と本物画像の特徴分布の近さを測る指標。小さいほど本物に近く良い）が明確に改善することが広く報告されている。EMA なしの生重みはサンプリングで破綻しやすく、多くの公開拡散モデルが EMA 重みを既定として配布している。

アブレーション的要点: (1) decay が総ステップに対して大きすぎると EMA が初期の悪い重みを引きずり本体より精度が下がる、(2) BatchNorm バッファを EMA で扱わないと推論精度が落ちる、(3) EMA 適用開始を学習初期からにすると初期重みの影響が残るため、warmup 後に開始する実装が多い。

## メリット・トレードオフ・限界

メリット
- 重みの振動を平均化し、最終モデルの精度・安定性が上がることが多い。
- 追加の計算コストがほぼゼロ（毎ステップの軽い重み平均だけ）。
- 学習プロセス自体を変えないので既存ループに後付けしやすい。
- 生成モデルや検出モデルでは効果が大きく、半教師あり・自己教師あり学習では target ネットワークとして中心的に使われる。

トレードオフ・限界
- 重みをもう1組持つためパラメータのメモリが約2倍。
- decay の調整が必要。総ステップに対して大きすぎると初期の悪い重みを引きずる。
- BatchNorm の running statistics は重みでないので別途扱う必要がある（EMA に含めるか推論前に再計算）。
- 必ず効くとは限らない。タスクや設定によっては本体重みと差が出ないこともある。
- どのタイミングで EMA 適用を始めるか（warmup 後など）の判断が要る。

未解決・研究課題: 「最適な decay をデータ・学習長から自動決定する」「重み平均が効くタスクと効かないタスクの境界の理論的理解」「自己教師あり学習で EMA target がなぜ崩壊を防ぐのか」は今も研究が続く領域です。

## 発展トピック・研究の最前線
- Momentum encoder / target network: BYOL・MoCo・DINO など自己教師あり学習で、EMA で更新する target ネットワークが学習の安定化と表現崩壊の防止に不可欠とされる。
- Model soups（Wortsman et al., 2022）: 異なるハイパーパラメータで微調整した複数モデルの重みを平均すると単体より高精度になる。重み平均の有効性を別角度から示し、EMA/SWA と同じ「平均は汎化に効く」系譜にある。
- EMA と学習率スケジュールの相互作用: 学習率を下げる decay フェーズと EMA はどちらも振動抑制で役割が重なるため、両者をどう組み合わせると最良かは実務的な調整対象。
- LLM への適用: 大規模言語モデルでも重み平均（checkpoint averaging）が最終性能をわずかに底上げすることが報告され、EMA 的手法が再評価されている。

## さらに学ぶための関連トピック
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0150_learning-rate-scheduler)
- [Gradient Accumulation](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0631_gradient-accumulation)
- [模倣学習 (Behavior Cloning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0560_imitation-learning)
- [CARLA(閉ループシミュレーション)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0715_closed-loop-simulation-carla)
- [自己教師あり事前学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0090_self-supervised-pretraining)
