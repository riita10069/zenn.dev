---
title: "Gradient Accumulation"
---

# Gradient Accumulation

## ひとことで言うと
Gradient Accumulation（勾配の貯め込み）とは、小さなバッチを何回かに分けて計算し、その勾配（重みをどちら向きにどれだけ動かすかの情報）を足し合わせてから、まとめて1回だけ重みを更新する手法です。これによって、GPU のメモリ（VRAM）には収まらない「大きなバッチ」で学習したのと同じ効果を、メモリを増やさずに得られます。GPU を増やさずに実効バッチサイズを拡大する、ほぼ無料の手段です。

## 直感的な理解
大きな鍋でカレーを一度に作りたいけれど、手元の鍋が小さい状況を想像してください。小さい鍋で4人分ずつ4回作り、最後に1つの大鍋に全部移して味を整えれば、結果として16人分の大鍋で作ったのとほぼ同じになります。Gradient Accumulation はこれと同じで、小さいバッチを何回かに分けて計算し、勾配だけを足し合わせてから一度に「味を整える」（重みを更新する）のです。

なぜ大きなバッチが欲しいかというと、勾配のノイズが減るからです。少数サンプルから計算した「重みを動かす向き」は偶然のブレを含みますが、たくさんのサンプルで平均すると「全データを使ったときの本当の向き」に近づき、更新が安定します。けれどバッチを大きくするほど VRAM を食い、高解像度の入力や大きなモデルではすぐにメモリの壁（Out Of Memory, OOM）にぶつかります。GPU を買い増せば解決しますが高価です。Gradient Accumulation は手元の1枚の GPU のまま、メモリを増やさずに実効バッチを大きくします。

## 基礎: 前提となる概念

### バッチサイズとは
学習では画像などのサンプルを一度に何枚かまとめて GPU に入れ、その平均的な誤差から重みを更新します。この「一度に入れる枚数」がバッチサイズです。バッチサイズ8なら8枚を同時に処理して1回更新します。バッチを大きくすると、(1) 勾配のノイズが減って学習が安定、(2) GPU の並列計算を効率よく使える、(3) BatchNorm などバッチ内統計を使う仕組みが安定する、という利点があります。一方で VRAM 消費が増えます。VRAM を主に食うのは、逆伝播のために順伝播時の中間出力（活性値）を保持する分で、これはバッチサイズにほぼ比例して増えます。

### 勾配(gradient)と最適化ステップ
- 順伝播(forward): 入力をモデルに通して予測を出す。
- 損失(loss): 予測と正解のずれを数値化したもの。
- 逆伝播(backward): loss を各重みで微分し、「その重みをどちらにどれだけ動かせば loss が減るか」を表す勾配を計算する。
- 更新(optimizer step): 勾配に従って重みを実際に動かす。

PyTorch では `loss.backward()` を呼ぶと、各重みの `.grad` 属性に勾配が「加算」されます（上書きではない）。この加算性が Gradient Accumulation の鍵です。`optimizer.zero_grad()` を呼ぶまで勾配は貯まり続けます。

### micro-batch と effective batch
- micro-batch（マイクロバッチ）: 実際に GPU に1回で載せる小さなバッチ（例 4枚）。
- effective batch（実効バッチ）: 1回の重み更新が反映するサンプル総数（micro-batch × 貯め込み回数）。

Gradient Accumulation は「micro-batch は小さく保ちつつ、effective batch を大きく見せる」技術です。

## 仕組みを詳しく

### 普通の学習（accumulation なし）
```python
for batch in loader:              # batch は例えば4枚
    optimizer.zero_grad()         # 勾配をゼロにリセット
    loss = compute_loss(model, batch)
    loss.backward()               # 逆伝播 → 各重みの .grad を計算
    optimizer.step()              # 勾配に従って重みを更新
```
ここで `optimizer.step()` が「実際に重みを動かす」操作で、バッチ4枚ごとに1回更新されます。

### Gradient Accumulation あり
実効バッチ16枚で学習したいが、メモリ的には4枚しか載らない状況を考えます。4枚ずつ4回計算して勾配を貯め、4回目にまとめて更新します。
```python
accum_steps = 4                   # 4回ぶん貯める → 実効バッチ 4*4 = 16
optimizer.zero_grad()
for i, batch in enumerate(loader):     # batch は4枚
    loss = compute_loss(model, batch)
    loss = loss / accum_steps          # ← 重要: 平均にするため割る
    loss.backward()                    # 勾配が「足し算」で貯まる
    if (i + 1) % accum_steps == 0:
        optimizer.step()               # 4回に1回だけ更新
        optimizer.zero_grad()          # 貯めた勾配をリセット
```

### なぜこれで「大きいバッチ」と同じになるのか
loss が各サンプルの平均で、勾配が線形演算であることから、「4枚ずつ4回の勾配の和」は「16枚を一度に計算した勾配の和」と数学的に等しくなります。式で書くと、N サンプルのバッチの平均損失の勾配は

```
g = (1/N) Σ_{k=1..N} ∇ loss_k
```

であり（Σ は総和、∇ は微分=勾配、loss_k は k 番目サンプルの損失）、これを M 個の micro-batch（各サイズ N/M）に分け、各 micro-batch の損失を M で割って backward すると、`.grad` に貯まる総和は上式の g に一致します。だから各 loss を `accum_steps` で割っておけば、16枚を1回処理したときとほぼ同じ更新になります。割り忘れると勾配が M 倍になり、実質的に学習率が M 倍になって発散しやすいので、これがよくあるバグです。

### 数値例
- micro-batch: 4 枚 / accum_steps: 4 / effective batch: 16
- VRAM 使用量: 4枚ぶんのまま（ほぼ増えない。貯めるのは勾配で、勾配は重みと同じサイズなので一定）
- 1更新あたりの計算時間: 4枚×4回ぶんなので約4倍

メモリは増えませんが、その代わり1更新にかかる時間が伸びます。これがトレードオフの核心です。「メモリと時間のトレードオフ」と覚えるとよいです。総サンプル処理数は同じなので1エポックの計算量は変わりませんが、勾配を実際に使う更新回数が1/4に減る点が普通の学習との違いです。

### BatchNorm との注意
完全に同一とは言えない部分があります。BatchNorm（バッチ正規化）は「そのバッチ内の平均・分散」で各特徴を正規化するため、4枚での統計と16枚での統計は異なります。したがって BatchNorm を含むモデルでは、accumulation の結果は本物の16枚バッチと厳密には一致しません。一方 LayerNorm（サンプル単位で正規化）を使う Transformer 系ではこの問題は起きません。ConvNeXt 系も LayerNorm を採用しており、この差は小さくなります。BatchNorm モデルで厳密さが要るときは、SyncBatchNorm（分散時にバッチ統計を集約）や GroupNorm への置き換えで補う必要があります。

### 分散学習との関係
複数 GPU の DistributedDataParallel（DDP）では、各 GPU が異なる micro-batch を処理し、backward 時に勾配が自動で全 GPU 平均されます。accumulation 中の途中ステップでは毎回通信すると無駄なので、`model.no_sync()` コンテキストで通信を抑え、最後の step でだけ同期するのが定石です。これを忘れると accum_steps 回ぶん余分に勾配を all-reduce 通信し、通信律速の環境では学習が大きく遅くなります。accumulation と DDP を併用すれば、実効バッチ = micro-batch × GPU 台数 × accum_steps とさらに大きくできます。

## 手法の系譜と主要論文

Gradient Accumulation 自体は特定の論文の「発明」というより、大規模分散学習の現場で自然に使われてきた標準技法です。その価値を支える研究の流れは次の通りです。

- Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"（Facebook AI Research, 2017, arXiv:1706.02677）。提案: バッチサイズを256から8192まで大きくしても精度を保つレシピ。学習率をバッチサイズに比例させる Linear Scaling Rule と、最初に学習率をゆるやかに上げる warmup（[LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0150_learning-rate-scheduler)）を組み合わせた。動機: バッチを大きくすると1更新が大胆になり初期に発散しやすいため。効果: 大バッチでも精度劣化なく学習時間を大幅短縮。トレードオフ: 学習率の調整が必須で、無調整に大バッチへ移すと精度が落ちる。この研究が「大バッチは有効」という前提を確立し、その大バッチを安価なハードで再現する手段として accumulation の価値が高まった。
- You et al., LARS（"Large Batch Training of Convolutional Networks", 2017, arXiv:1708.03888）/ LAMB（"Large Batch Optimization for Deep Learning: Training BERT in 76 minutes", 2019, arXiv:1904.00962）。提案: バッチを数万まで増やすため、層ごとに学習率を適応させる最適化手法。動機: 単純な線形スケーリングだけでは超大バッチで破綻するため。効果: BERT の事前学習を約76分まで短縮。トレードオフ: 実装と調整が複雑。
- Keskar et al., "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima"（ICLR 2017, arXiv:1609.04836）。報告: 大バッチは sharp minima（鋭い谷）に落ちやすく、汎化が悪化しうる。つまり accumulation で無限にバッチを増やせばよいわけではなく、適切な範囲がある。これに対し前述の warmup や適切な学習率調整で汎化ギャップを縮められることも後続研究で示されている。
- McCandlish et al., "An Empirical Model of Large-Batch Training"（OpenAI, 2018, arXiv:1812.06162）。提案: 勾配ノイズスケール（gradient noise scale）という指標で「どこまでバッチを大きくすると効率が落ちるか（critical batch size, 臨界バッチサイズ）」を予測できる。臨界バッチサイズより小さい領域ではバッチを2倍にすると必要更新回数がほぼ半分になり時間効率が良いが、それを超えると更新回数があまり減らず計算が無駄になる。実務でバッチ・accum_steps を決める理論的指針になる。

## 論文の実験結果(定量データ)

評価で測られる指標と、その意味を示します。

- 検証精度（Top-1 accuracy など。テスト画像で最も確率の高い予測が正解と一致した割合）: バッチを大きくしても精度が保たれるかが論点。Goyal et al. は ImageNet で ResNet-50 を、バッチ256のベースラインと同等の Top-1 精度（約76%台）を、バッチ8192でも維持できることを示した。学習率を線形にスケールし warmup を入れたのが鍵で、これらを抜くと大バッチで精度が数ポイント落ちる（アブレーション）。さらにバッチ16384以上に増やすと、レシピを守っても精度が目に見えて低下し始め、臨界バッチサイズの存在を裏づけた。
- 壁時計時間（wall-clock time）: 大バッチ + 多 GPU で ResNet-50 ImageNet 学習を約1時間（256 GPU 使用）に短縮。LAMB は BERT 事前学習を、通常の数日規模から約76分（1024 TPU 相当の大規模並列）に短縮。これらは「同じ精度に到達するまでの時間」を大幅に縮めたという定量成果。
- generalization gap（汎化ギャップ。訓練精度とテスト精度の差）: Keskar et al. は、大バッチ学習が小バッチ学習に比べてテスト精度で数ポイント劣化しうること、そしてその解が損失地形の「鋭い谷」に位置することをヘッシアン（損失の2階微分）の固有値の大きさで定量化した。鋭い谷は重みの微小変化で損失が急増するため、訓練分布とテスト分布のわずかなズレに弱い。

アブレーション的な要点: (1) 大バッチ化に伴い学習率を線形スケールしないと精度が落ちる、(2) warmup を抜くと初期に発散する、(3) critical batch size を超えてバッチを増やしても、必要な総更新回数があまり減らず計算効率の改善が頭打ちになる。Gradient Accumulation はこれらの「大バッチの利点と注意点」をそのまま継承します。なお accumulation は「実効バッチを大きくする」だけで臨界バッチサイズ自体を動かさないため、効率改善には自ずと上限がある点を理解しておくのが重要です。

## メリット・トレードオフ・限界

メリット
- メモリを増やさずに実効バッチを拡大できる（GPU 増設不要）。
- 勾配のノイズが減り、学習が安定しやすい。BatchNorm 以外では本物の大バッチとほぼ等価。
- 実装が数行で済み、既存ループにほぼそのまま足せる。
- 分散学習（DDP）と併用でき、実効バッチをさらに伸ばせる。

トレードオフ・限界
- 学習が遅くなる。accum_steps 回ぶんの forward/backward をしてから1回更新するので、更新あたりの計算時間が増える（壁時計時間が伸びる）。
- loss を accum_steps で割り忘れると、実質学習率が accum_steps 倍になり発散（頻出バグ）。
- BatchNorm を使うモデルでは本物の大バッチと厳密に一致しない。
- 大バッチ化に伴い学習率の再調整（warmup, linear scaling）が必要になることが多い。
- バッチを大きくしすぎると汎化がむしろ落ちる場合がある（sharp minima 問題、critical batch size を超えると効率も頭打ち）。

未解決・研究課題: 「最適な effective batch をデータ・モデルから事前に決める一般理論」「大バッチでも sharp minima を避ける最適化」は今も研究が続く領域です。勾配ノイズスケールはその一つの答えですが、学習の進行中に臨界バッチサイズ自体が変化する（後半ほど大きくなる）ことが観察されており、万能ではありません。

## 発展トピック・研究の最前線
- Gradient checkpointing（活性値の再計算）との併用: accumulation はバッチ次元のメモリ節約、checkpointing は深さ方向のメモリ節約（順伝播の中間出力を保存せず逆伝播時に再計算する）で、組み合わせるとさらに大きなモデル・バッチを1枚に載せられる。
- ZeRO / FSDP: オプティマイザ状態や勾配を GPU 間で分割保持し、巨大モデルを学習する手法。accumulation と組み合わせて超大規模学習で使われる。
- 適応的 accumulation: 学習の進行に応じて accum_steps を変える、勾配ノイズスケールを測って自動調整する研究。学習後半に臨界バッチサイズが上がる性質に合わせてバッチを段階的に増やす batch ramp-up も使われる。
- 大バッチ向け最適化（LARS/LAMB の後継、Adafactor, Sophia など）: 大きな実効バッチでも安定・高速に収束する optimizer の探索が続いている。

## さらに学ぶための関連トピック
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0150_learning-rate-scheduler)
- [Mixed Precision Training (AMP)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0133_mixed-precision-training)
- [WebDataset (Shard Packing)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0851_webdataset-shards)
- [DDP (Distributed Data Parallel)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0604_ddp-distributed-data-parallel)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer)
