---
title: "学習ループ基盤の全体像"
---

# 学習ループ基盤の全体像

## ひとことで言うと
モデルを実際に「賢くする(重みを更新する)」ための繰り返し処理を学習ループといいます。データを少しずつ取り出し、予測させ、正解とのズレ(損失)を測り、そのズレが減る方向に重み(数百万個のパラメータ)を少し動かす、を何万回も回します。このノートは、その流れを成立させるために欠かせない部品 ── DataLoader、Optimizer、Scheduler、forward→loss→backward→step、勾配クリップ、ログ、checkpoint、AMP ── を一つずつ初心者向けに説明し、各部品の理論的根拠まで掘り下げます。これらは個別の研究成果の集積であり、「なぜその部品が要るのか」を理解すると、学習が止まったり発散したりしたときの原因切り分けができるようになります。

## 直感的な理解
ダーツの練習を思い浮かべてください。1投目を投げて(forward)、的の中心からどれだけ外れたかを見て(loss)、外れた方向と量から「次は少し右上に、これくらい力を弱めよう」と修正方針を立て(backward)、実際に構えを直して投げ直す(step)。これを延々と繰り返すと、だんだん中心に集まるようになります。学習ループはまさにこれを、何百万個の「構えのパラメータ」について同時に、自動で行う仕組みです。

ただし練習には段取りが要ります。矢を効率よく手元に供給する係(DataLoader)、修正のさじ加減を決める係(Optimizer)、練習序盤と終盤でさじ加減を変える係(Scheduler)、突風で大きく狂った1投に引きずられない安全装置(勾配クリップ)、上達の記録(ログ)、途中で中断しても再開できる栞(checkpoint)、そして同じ精度なら早く終わらせる工夫(AMP)。どれが欠けても練習は「遅い」「壊れる」「やり直し」になります。学習ループ基盤とは、この段取り一式のことです。最小形は10行ほどで書けますが、実用に耐えるものにするには、これらの周辺部品が不可欠です。

## 基礎: 前提となる概念

- 重み(weight)/パラメータ(parameter): モデルが学習で調整する数値。深層モデルでは数百万〜数十億個あります。
- forward pass(順伝播): 現在の重みで入力から予測を計算する流れ。
- 損失(loss): 予測と正解のズレを表す1つの数値。これを小さくするのが学習の目的です。
- 勾配(gradient): 各重みを「どちらにどれだけ動かせば損失が減るか」を表す量。損失を各重みで微分して得ます。
- backward pass(逆伝播): 自動微分(autograd)で全重みの勾配を一度に計算する流れ。連鎖律(微分の合成則)を機械的に適用します。計算量と必要メモリは forward と同程度かそれ以上です。
- エポック(epoch): 学習データ全体を1周すること。通常は数十エポック回します。
- バッチ(batch): 一度にまとめて処理するサンプルの束。勾配を1サンプルごとでなく束の平均で計算すると、推定が安定し、GPUを効率よく使えます。
- 過学習(overfitting): 訓練データだけに過度に適合し、未知データで性能が落ちる現象。正則化で抑えます。
- 学習率(learning rate, lr): 1回の更新でどれだけ重みを動かすかの歩幅。大きすぎると発散、小さすぎると遅い。
- ハイパーパラメータ: 学習前に人が決める設定値(学習率、バッチサイズ、weight decay など)。学習で自動調整される重みとは区別します。

## 仕組みを詳しく

学習ループの最小形は「forward → 損失 → backward → step」を1バッチごとに回すことです。これに周辺部品を足して実用的な基盤にします。順に見ます。

### 1. DataLoader(データの供給係)

学習データは膨大で一度にメモリに載りません。DataLoader は `Dataset` から少しずつバッチ単位で取り出し、必要なら並列ワーカーで先読みします。

```python
DataLoader(dataset, batch_size=8, shuffle=True,
           num_workers=2, pin_memory=True, drop_last=True)
```

- `batch_size`: 一度に処理するサンプル数。大きいほど勾配推定が安定しGPU効率が上がるが、VRAM を食う。
- `shuffle=True`: 毎エポックで順番を入れ替える。順序の偏り(時間的に近いサンプルが似ていること)による学習の歪みを防ぐ。
- `num_workers`: データ準備を並列化するサブプロセス数。GPUを入力待ちで遊ばせないため。
- `pin_memory`: ページ固定メモリを使い CPU→GPU 転送(`.to(device, non_blocking=True)`)を速くする。
- `drop_last`: 端数バッチを捨ててバッチサイズを一定に保つ。BatchNorm など統計量に敏感な層で効く。

入力が動画デコードのように重い場合は、データ供給がGPUの足を引っ張りがちです。事前抽出済みのシャード(WebDataset 等)をストリーミングで読む経路を用意し、本番ではそちらに切り替えるのが定石です(詳細は [大規模マルチカメラ運転データセットのパースと前処理](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0830_nvidia-physicalai-dataset-parsing))。

### 2. Optimizer(重み更新係)

勾配に基づき重みを動かす担当です。現代の深層学習では AdamW が事実上の標準です(詳細は [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer))。

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
```

- `lr`(学習率): 歩幅。
- `weight_decay`: 重みが大きくなりすぎないよう抑える正則化(過学習対策)。AdamW では適応的歩幅から切り離されて純粋に効きます。

### 3. Scheduler(学習率を時間で変える係)

学習率は最初から最後まで一定が最適とは限りません。Scheduler は時間とともに学習率を変えます。代表的なのは warmup(最初の数百〜数千ステップで0から目標値まで徐々に上げる)と、その後の cosine 減衰(余弦関数の形でゆるやかにゼロへ近づける)の組み合わせです。

```
lr
 |        ___________
 |       /           \___
 |      /                \___
 |     /                     \____
 |____/__________________________\___ step
   warmup      cosine 減衰
```

warmup が要る理由は、学習初期は重みがランダムで勾配が荒れており、いきなり大きな学習率を当てると発散しやすいからです。最初は小さく入って助走をつけ、安定したら本来の学習率に上げ、後半は細かい調整のため小さく絞ります。固定学習率でも学習は進みますが、収束の速さと最終精度で warmup+cosine に劣ることが多く、基盤としては Scheduler を組み込めるようにしておくのが望ましい設計です。cosine が好まれるのは、終盤で滑らかに学習率が0へ近づくため、損失地形の谷底で振動せず落ち着くからです。

### 4. 中核ループ: forward → loss → backward → step

ここが学習の心臓です。1バッチごとに次を回します。

```python
optimizer.zero_grad(set_to_none=True)          # 前回の勾配をクリア
prediction = model(inputs)                     # forward(予測)
loss = loss_fn(prediction, target)             # 予測と正解のズレ
loss.backward()                                # backward(勾配を計算)
clip_grad_norm_(model.parameters(), 1.0)       # 勾配クリップ
optimizer.step()                               # 重みを更新
```

- `zero_grad`: 勾配は加算されていくので、毎回ゼロに戻す。`set_to_none=True` はメモリと速度の両面でわずかに有利(ゼロ埋めでなくテンソル自体を解放)。
- `forward`: 現在の重みで予測を出す。
- `loss`: 予測と正解の差。模倣学習(専門家の運転を真似る学習)では、軌跡の予測と正解軌跡の距離(smooth-L1 や MSE)を使うのが一般的です。smooth-L1 は外れ値に対して MSE より頑健で、L1 と L2 の良いとこ取りをした損失です。
- `backward`: 自動微分で全重みの勾配を計算する。ここで計算グラフを辿るため、forward で保持した中間結果のメモリが必要になる。
- `step`: 勾配に従って重みを実際に動かす。

ここで `zero_grad → backward → step` の順序を間違えると、前ステップの勾配が混ざる(zero_grad を忘れる)、勾配がないのに更新する(backward より前に step)など、静かに学習が壊れます。順序は学習ループの基本文法です。

### 5. 勾配クリップ(暴走防止)

ときどき勾配が異常に大きくなり、重みが吹き飛んで学習が壊れます(発散、損失が nan になる)。勾配クリップは、勾配全体の大きさ(ノルム、ベクトルの長さ)が閾値を超えたら、向きを保ったまま縮めます。

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

`max_norm=1.0` は「全勾配を1本のベクトルとみなし、その長さ(全パラメータの勾配二乗和の平方根)が1を超えたら、長さがちょうど1になるよう一律にスケールする」という意味です。向きは変えず長さだけ抑えるので、1回の暴れたバッチで学習が崩壊するのを防げます。閾値は0.5〜5程度がよく使われ、タスク依存です。なお勾配クリップは `backward` の後・`step` の前に置く必要があります(計算済みの勾配を縮めてから更新する)。AMP を使う場合は、後述する scaler の `unscale_` の後にクリップします。

### 6. ログ(学習の見える化)

損失が下がっているか、GPUが使えているかを記録します。標準出力に加え、実験管理ツール(MLflow、TensorBoard、Weights & Biases など)に損失・ハイパーパラメータ・コードバージョン・GPUメトリクス・学習率の推移を記録するのが標準です。

```python
logger.log_metric("train_loss", loss.item(), step=global_step)
```

ログがないと「学習がうまくいっているか」を判断できず、複数実験の比較もできません。`loss.item()` で Python の数値に落とすのは、テンソルのまま蓄積すると計算グラフが解放されずメモリリークするのを防ぐためです。実験管理は研究の再現性の土台で、「どの設定でこの損失になったか」を後から辿れることが、研究を前に進める条件になります。

### 7. checkpoint(途中保存と再開)

長い学習は途中で落ちることがあります(OOM、ハード障害、ジョブの時間切れ)。checkpoint はモデルの重みと optimizer の状態をファイルに保存し、再開や評価に使います。

```python
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "global_step": global_step}, ckpt_path)
```

optimizer の状態まで保存するのは重要です。Adam/AdamW は内部に「過去の勾配の移動平均(モーメント `m`, `v`)」を持っており、これを捨てて重みだけ復元すると、再開時にモーメントがリセットされて学習が一時的に乱れます。scheduler の状態(現在のステップ)も保存しないと、再開時に学習率が warmup の最初に戻ってしまいます。「重み・optimizer・scheduler・step」をひとまとめで保存・復元するのが完全な再開の条件です。

### 8. AMP(混合精度、メモリと速度の節約)

通常は32bit浮動小数点(fp32)で計算しますが、一部を16bitにすると速くなりメモリも減ります。これがAMP(Automatic Mixed Precision)です。

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
    prediction = model(inputs)
    loss = loss_fn(prediction, target)
loss.backward()   # bf16はfp32と同じ指数レンジなのでGradScaler不要
```

16bit形式には2種類あります。fp16(half)は精度(仮数部10bit)が高いが表せる数値範囲(指数部5bit)が狭く、勾配が小さすぎてゼロに潰れる(アンダーフロー)ため、GradScaler という「損失を一時的に定数倍して勾配を大きくし、更新前に元へ戻す」仕掛けが要ります。一方 bf16(bfloat16)は仮数部が7bitと少ない代わりに指数部が8bitで fp32 と同じ、つまり範囲は fp32 並み。このため勾配の潰れが起きにくく GradScaler が不要で、コードが単純になります。新しめのGPU(L40S, A100, H100 など)は bf16 をハードウェア(Tensor Core)で高速処理できます。autocast は「行列積などは16bit、和の集約や softmax など数値的に敏感な演算は fp32」と演算ごとに精度を自動で使い分けるのが肝で、闇雲に全部16bit にするわけではありません。

## 手法の系譜と主要論文

学習ループの各部品は、それぞれ独立した重要研究に支えられています。

- Optimizer: Kingma & Ba, "Adam"(ICLR 2015, arXiv:1412.6980)が適応的学習率の標準を作り、Loshchilov & Hutter, "Decoupled Weight Decay Regularization (AdamW)"(ICLR 2019, arXiv:1711.05101)が weight decay の扱いを正して現在の標準にしました。詳細は [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer)。

- 勾配クリップ: Pascanu et al., "On the difficulty of training Recurrent Neural Networks"(ICML 2013, arXiv:1211.5063)が、勾配爆発の問題と、勾配ノルムクリッピングで学習が安定することを示しました。本ループの `clip_grad_norm_` はこの成果です。

- 混合精度: Micikevicius et al., "Mixed Precision Training"(ICLR 2018, arXiv:1710.03740)が、16bitで学習しても精度をほぼ保ちつつ高速化・省メモリ化できることを示し、AMPの基礎を作りました。fp32 のマスター重みを保持しつつ計算を fp16 で行い、loss scaling でアンダーフローを防ぐ、という処方箋を確立しました。bf16(Google の TPU 由来)はその後、指数レンジの広さから大規模モデル学習で標準化しました。

- Scheduler: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts"(ICLR 2017, arXiv:1608.03983)が cosine 減衰(と周期的な再スタート)の有効性を示しました。warmup については Goyal et al., "Accurate, Large Minibatch SGD"(2017, arXiv:1706.02677)が、大バッチ学習で初期の学習率 warmup が発散を防ぐことを実証し、現在の warmup+cosine の組み合わせの基礎になっています。

- 分散・大規模化: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"(SC 2020, arXiv:1910.02054)は、optimizer 状態・勾配・パラメータを複数GPUに分割保持してメモリを劇的に削減し、巨大モデルの学習を可能にしました。学習ループの「どこにどれだけメモリを使うか」を理解する上で重要な研究です。

## 論文の実験結果(定量データ)

- 混合精度の効果(Micikevicius et al., 2018): ImageNet 分類(ResNet-50, AlexNet, Inception)、機械翻訳、音声認識など複数タスクで、fp32 と同等の最終精度(トップ1精度で差はほぼゼロ)を保ちつつ、メモリ使用量をおよそ半減し、対応ハードウェアで2〜3倍程度の高速化を達成したと報告されています。「精度を犠牲にせず速く・軽く」が定量的に裏づけられたことが、AMP が標準化した理由です。

- 大バッチ学習と warmup(Goyal et al., 2017): ImageNet の ResNet-50 を、バッチサイズ8192まで拡大しても warmup(最初の数エポックで学習率を線形に上げる)を入れることで、小バッチ学習とほぼ同じ最終精度(トップ1精度で誤差約0.1ポイント以内)を保ちつつ、256GPUで約1時間という当時として劇的な短縮を達成しました。アブレーションとして、warmup を外すと大バッチでは学習初期に発散しやすく精度が落ちることが示されています。これが「warmup は大バッチ・大学習率で特に重要」とされる根拠です。

- 勾配クリップの効果(Pascanu et al., 2013): 勾配クリッピングなしでは、再帰型ネットの学習が勾配爆発でしばしば発散するのに対し、ノルムクリッピングを入れると安定して収束することが示されました。閾値そのものより「閾値が存在すること」が安定性に効くという観察です。深いネットワークや系列モデル、Transformer の学習全般で広く採用されています。

- メモリ削減の効果(Rajbhandari et al., 2020): ZeRO は optimizer 状態の分割だけでも、データ並列に比べて1GPUあたりのメモリを大きく削減し、同じハードでより大きなモデルを学習できることを示しました。Adam の状態が重みの約2倍のメモリを食う(後述)ため、これを分割する効果が大きいのです。

- checkpoint と optimizer 状態: optimizer 状態を捨てて重みだけから再開すると、Adam系では再開直後にモーメントがゼロから再推定されるため、損失が一時的に跳ね上がることが経験的に知られています。状態込みで保存・復元することで、この再開時の乱れを回避できます。

## メリット・トレードオフ・限界

- メリット: forward だけでは進まない学習を、実際に重みを更新する形で端から端までつなげられる。
- メリット: 勾配クリップと AMP で、学習の安定性とGPUメモリ効率を両立できる。
- メリット: checkpoint とログで、中断再開・実験比較・再現性が確保される。
- メリット: backbone やモデル構成をレジストリ(部品を名前で登録・呼び出す仕組み)から選べる設計にすれば、拡張時に学習コードを触らずに済む。
- トレードオフ: Scheduler を省いて固定学習率にすると、実装は単純だが収束の速さ・最終精度で劣りやすい。
- トレードオフ: 補助損失(世界モデル的な未来予測損失など)を主損失に接続しないと、表現学習の恩恵を取りこぼす。
- トレードオフ: 最小実装には分散学習(複数GPU)、EMA(重みの指数移動平均)、early stopping(改善が止まったら停止)などの本格機能が欠けがち。
- 限界: bf16 はメモリ・速度に有利だが、bf16非対応の古いGPUでは使えず fp16+GradScaler に頼る必要がある。
- 未解決の課題: ハイパーパラメータ(学習率・weight decay・バッチサイズ・スケジュール)の相互作用は依然複雑で、自動チューニング(sweep、ベイズ最適化、Population Based Training など)が研究・実務の課題として残ります。

## 発展トピック・研究の最前線

- 分散学習: データ並列(DDP、各GPUが同じモデルを持ち勾配を集約)、モデル並列(モデルを分割)、ZeRO/FSDP(状態を分割してメモリを節約)など、複数GPU・複数ノードへ学習をスケールさせる技術。大規模化の必須要素です。
- 学習安定化技術: gradient accumulation(小バッチを複数回ためて step を遅らせ、疑似的に大バッチにする。VRAM が足りないときに実効バッチを増やす常套手段)、EMA、確率的重み平均(SWA)など。
- 効率的微調整: LoRA など、巨大な事前学習モデルを少ないパラメータ更新で適応させる手法。学習ループの step が触る対象を低ランク行列に絞る形で組み込みます。
- 学習レシピの再現性: ハイパーパラメータ・データ・コード・乱数シードを一括管理し、結果を完全再現する MLOps 基盤。実験管理ツールとデータバージョニングが核になります。
- コンパイルと最適化: `torch.compile` などでループ内の演算をグラフ化・融合し、同じコードのまま学習スループットを上げる流れが進んでいます。

## さらに学ぶための関連トピック

- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer)
- [大規模マルチカメラ運転データセットのパースと前処理](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0830_nvidia-physicalai-dataset-parsing)
- [推論速度ベンチマークとGPUウォームアップ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1152_speed-benchmark-gpu-warmup)
- [履歴ウィンドウと有効サンプル数](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0859_history-window-valid-samples)
