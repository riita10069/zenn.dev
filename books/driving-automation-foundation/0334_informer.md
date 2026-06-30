---
title: "Informer"
---

# Informer

## ひとことで言うと
長い時系列(例: 過去数百〜数千ステップを見て、未来の数百ステップを予測する)に Transformer を使うと、自己注意の計算量が爆発します。Informer は「重要な Query だけ選んで注意する(ProbSparse 注意)」ことで計算量を O(L^2) から O(L log L) に下げ、長期時系列予測を現実的にした手法です。2021年の AAAI で最優秀論文(Best Paper)に選ばれ、長期時系列 Transformer 研究の事実上の出発点になりました。

## 直感的な理解
教室で100人の生徒が、それぞれ「他の99人のうち誰の意見を聞くべきか」を考える場面を想像してください。素朴にやると、100人×100人=1万通りの「誰が誰を見るか」をすべて計算することになります。生徒が1000人なら100万通り、5000人なら2500万通りと、人数の2乗で爆発します。これが自己注意の O(L^2) 問題です。

ところが実際に観察すると、ほとんどの生徒は「特に誰の意見も重視していない(全員をうっすら見ているだけ)」状態で、ごく一部の生徒だけが「特定の数人を強く参照している」ことが分かります。Informer のアイデアはシンプルで、「強く参照している少数の生徒だけ本気で計算し、残りは平均的な意見で済ませよう」というものです。これで計算する組み合わせが大幅に減り、1000人でも現実的な時間で処理できます。

## 基礎: 前提となる概念

まず **計算量のオーダー記法 O(·)** を押さえます。O(L^2) とは「系列長 L を2倍にすると計算量が4倍になる」という増え方、O(L log L) とは「L を2倍にしても計算量は2倍ちょっとにしか増えない」という、はるかに緩やかな増え方を表します。長い系列ではこの差が桁違いの実行時間・メモリ差になります。L=1000 なら L^2=100万に対し L log L ≒ 7000 で、約140倍の差です。

[Transformer 時間方向アテンション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0344_transformer-temporal-attention) の自己注意は、系列長 L に対して L×L のスコア行列(各時刻が各時刻にどれだけ注目するかの表)を作るため、計算量とメモリが O(L^2) で増えます。これは時系列予測で深刻な壁になります。

時系列予測には2種類あります。短期予測(数ステップ先)と **長期予測(LSTF: Long Sequence Time-series Forecasting、数百〜数千ステップ先)** です。長期予測では、精度を上げるために入力(過去の窓、look-back window)も出力(予測する長さ、horizon)も長くしたい。しかし、

- L=1000 だと L^2=100万。L=5000 だと 2500万。スコア行列だけでメモリが破綻する。
- さらに、出力を1ステップずつ生成する **自己回帰(autoregressive、前の予測を入力に戻して次を出す)** だと、長い予測ほど推論が遅く、誤差も累積する(序盤の小さな誤差が後の予測の入力になり、雪だるま式に膨らむ)。これを exposure bias とも呼びます。

Zhou らの問題意識は3つでした。(1) 自己注意の O(L^2) を下げたい、(2) 層を深くするとメモリが系列長ぶん積み上がるのを抑えたい、(3) 自己回帰の遅さと誤差累積をなくしたい。Informer はこの3つに同時に答えます。

評価に使う指標も先に説明します。**MSE(Mean Squared Error、予測値と正解の差の二乗平均)** と **MAE(Mean Absolute Error、差の絶対値平均)** で、どちらも小さいほど良い。MSE は大きな外れ予測を二乗で強く罰し、MAE は誤差の平均的な大きさをそのまま表します。時系列は事前に標準化(平均0・分散1にスケール)してから評価することが多く、MSE が 0.0X 違うだけでも相対的には大きな差になります。

## 仕組みを詳しく

### 1. ProbSparse Self-Attention(確率的にまばらな注意)

著者らは、訓練済みモデルの注意重みを観察し、「多くの Query は注意がほぼ一様(どこも特に見ていない、出力への貢献が小さい)で、ごく一部の Query だけが特定の Key に強く集中している」ことを発見しました。注意の重み分布が **ロングテール(少数が支配的で、大多数は貢献が小さい)** だったのです。

ならば、貢献の大きい少数の Query だけ本気で計算すればよい。手順は、

1. 各 Query が「どれだけ尖った注意を持つか」を測ります。具体的には、その Query の注意分布が「一様分布(全 Key に均等)」からどれだけズレているかを **KL ダイバージェンス(2つの分布の隔たりを測る量)** に基づく指標(sparsity measurement M)で評価します。ズレが大きい Query ほど「特定の Key に集中している重要な Query」です。M は実装上「最大の内積 − 内積の平均」の形に近似されます。
2. この M を全 Key との内積で正確に計算すると結局 O(L^2) かかるので、ランダムに選んだ少数(約 L log L 個)の Key との内積だけで近似的に見積もります。論文はこの近似に上界・下界の理論保証を与えています。
3. 上位 u = c·log L 個(c は調整係数、サンプリングファクタ)の「アクティブな(尖った)Query」だけを選びます。
4. 選ばれた Query だけ通常の注意を計算し、選ばれなかった Query は入力の平均(Value の平均、ないし系列の累積平均)で埋めます。

これで Query 軸が L から log L に減り、注意全体の計算量・メモリが **O(L log L)** になります。L=1000 なら log L はおよそ7程度なので、O(L^2) の100万に対して 1000×7=7000 と桁違いに軽くなります。

### 2. Self-Attention Distilling(自己注意の蒸留)

Encoder を層ごとに重ねると、各層が長さ L の系列を保持してメモリが積み上がります。Informer は層と層の間に「Conv1d(1次元の畳み込み、カーネル幅3)+ ELU 活性 + 最大プーリング(stride 2)」を挟み、系列長を半分に間引きます。重要な特徴(支配的な Query が作った値)だけ残して長さを圧縮するので、層を重ねても総メモリが O((2-ε)L log L) ≒ O(L log L) に収まります。蒸留(distilling)という言葉は「濃縮して要点だけ残す」イメージです。

### 3. Generative Decoder(一括生成デコーダ)

自己回帰(1ステップずつ)をやめ、予測したい全長を一度の forward(順伝播1回)で出力します。デコーダ入力は「直近の既知系列(start token)+ 予測したい長さぶんのプレースホルダ(0埋め)」を並べ、1回の注意で全予測を生成します。これにより、長い予測でも推論が高速(1回の forward で完結)で、ステップごとの誤差累積が起きません。

### 形状での整理

```
入力: 過去系列 [B, L_enc, d]  (例 L_enc=720, B=バッチ)
Encoder: ProbSparse注意 + distilling で系列長を段階的に半減
         [B, 720, d] → [B, 360, d] → [B, 180, d] 程度
Decoder入力: [既知の直近 L_token, 予測ぶん L_pred の0埋め]
出力: [B, L_pred, d]  を一括生成 (例 L_pred=336 や 720)
```

入力埋め込みは3種類を足します。値そのものの埋め込み、位置エンコーディング([Positional Encoding](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0338_positional-encoding))、そしてタイムスタンプの temporal embedding(月・日・時・曜日など)です。最後のものは時系列に固有で、1日周期や1週間周期といったカレンダー由来の周期性を明示的にモデルへ与えます。

## 手法の系譜と主要論文

- **Zhou, Zhang, Peng, Zhang, Li, Xiong, Zhang, "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021 Best Paper, arXiv:2012.07436)**。提案は上記3点(ProbSparse 注意・蒸留・一括生成)。動機は LSTF の O(L^2)・メモリ積み上がり・自己回帰の3つの壁を同時に壊すこと。
- **(前史) Li et al., "LogSparse Transformer / LogTrans" (NeurIPS 2019, arXiv:1907.00235)** と **Kitaev et al., "Reformer" (ICLR 2020)**。Informer 以前にも、注意のスパース化(対数間隔の参照)やハッシュによる効率化で時系列・長系列に挑む試みがあり、Informer のベースラインになっています。
- **Wu, Xu, Wang, Long, "Autoformer" (NeurIPS 2021, arXiv:2106.13008)**。Informer の翌年、別アプローチ(系列を周期成分とトレンドに分解する series decomposition + 自己相関に基づく Auto-Correlation 機構)で長期予測を改善し、多くのベンチで Informer を上回りました([Autoformer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0321_autoformer))。
- **Zhou et al., "FEDformer" (ICML 2022, arXiv:2201.12740)**。周波数領域(フーリエ変換した空間)でランダムに選んだ少数の周波数成分だけ注意することで O(L) を達成し、Informer/Autoformer の効率化系譜を進めました。
- **Liu et al., "Pyraformer" (ICLR 2022)**。ピラミッド型の注意で多解像度の依存を O(L) で捉える別系統の効率化です。
- **Nie, Nguyen, Sinthong, Kalagnanam, "PatchTST" (ICLR 2023, arXiv:2211.14730)**。時系列をパッチに区切る素直な Transformer(+チャネル独立)が、複雑な注意改造をした Informer/Autoformer を上回ると示し、「凝った注意改造が必ずしも最善ではない」という流れを作りました([PatchTST](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0336_patchtst))。
- **Zeng et al., "Are Transformers Effective for Time Series Forecasting?" (DLinear, AAAI 2023, arXiv:2205.13504)**。トレンドと季節に分解してから1層の線形写像をかけるだけの DLinear が Informer 系の多くを上回ると報告し、Transformer 系全体に強い問い直しを投げかけました。

## 論文の実験結果(定量データ)

Informer は **ETT(Electricity Transformer Temperature、変圧器の油温。ETTh1/h2 が1時間粒度、ETTm1 が15分粒度)・ECL(電力消費)・Weather(気象)** など複数のベンチマークで評価されました。指標は前述の MSE・MAE です。

- 予測長を 24, 48, 168, 336, 720 ステップ(ETTm では 24〜960)と伸ばしていったとき、Informer は LSTM・従来 Transformer・LogTrans・Reformer・DeepAR・ARIMA などのベースラインに対して、ほとんどの設定で MSE/MAE を低く保ちました。論文の集計では、評価した多数の設定の大半で勝ち、特に予測長が長い(336, 720)ほど差が開きます。たとえば ETTh1 の長期設定でベースラインが大きく劣化する一方、Informer は劣化が緩やかで、長期設定では従来手法比でおおむね数十パーセント規模の MSE 削減が報告されています。
- **推論速度・計算量**: 一括生成デコーダにより、自己回帰の従来 Transformer に比べて長期予測の推論が大幅に高速化したことが示されています。予測長に対して推論時間がほぼ一定(1回の forward)である点が、ステップ数に比例して遅くなる自己回帰との決定的な差です。メモリも、入力長を伸ばしたとき full attention がすぐ OOM(メモリ不足)になるのに対し、ProbSparse + distilling は緩やかにしか増えず、より長い入力を扱えたと報告されています。
- **アブレーション**(どの要素を抜くと劣化するか):
  - ProbSparse 注意を通常の full attention に戻すと、メモリと計算が O(L^2) に戻り、長い入力で学習が困難(OOM)になります。逆に ProbSparse は精度をほぼ保ったまま計算量だけ削減できたと報告されています。
  - Distilling を外すと、層を重ねた際のメモリが増え、長い入力で扱える層数や系列長が制限されることが示されました。
  - Generative decoder を自己回帰(動的デコード)に戻すと、推論が遅くなるうえ予測長が伸びるほど誤差累積で精度が落ちることが確認されました。
  - これら3要素がそれぞれ独立に効いており、組み合わせて初めて「長く・速く・正確に」が成立する設計になっています。

なお後続研究の視点では、PatchTST や DLinear が同じ ETT などのベンチで Informer を上回る MSE を報告しています(例えば ETTm1 の長期予測で DLinear や PatchTST が Informer の MSE を半分以下にする設定もある)。Informer は「長期時系列 Transformer の道を開いたが、最終解ではなかった」という位置づけになっています。指標(MSE/MAE)の差が 0.0X 程度に見えても、標準化された時系列では相対的に大きな品質差を意味することが多く、ベンチ上の順位は実務的にも参照されます。

## メリット・トレードオフ・限界

- メリット: 自己注意を O(L log L) に削減し、長い時系列を Transformer で扱えるようにした。長期予測 Transformer の事実上の出発点。
- メリット: 蒸留で層を重ねてもメモリが膨らまない。
- メリット: 一括生成で長期予測が高速、誤差累積もなし。
- トレードオフ: ProbSparse は「注意がロングテールである」という経験的前提に依存する近似で、注意が一様に分散するデータでは近似誤差が出うる。Query 選別のためのランダムサンプリングにも乱数依存の揺れがある。
- トレードオフ: Query 選別・蒸留・専用デコーダと、構成要素が多く実装とハイパーパラメータ調整(サンプリングファクタ c、トークン長など)が複雑。
- トレードオフ: 後発の Autoformer/FEDformer/PatchTST に多くのベンチで抜かれ、さらに DLinear から「単純な線形モデルでも勝てる」という反証を突きつけられた。
- 限界・未解決の論点: 「効率的注意は本当に必要か」という根本的な問いがあります。DLinear の結果は、長期予測の主要な信号がトレンドと季節性という単純な構造に大きく依存し、凝った注意がその差を生んでいないケースがあることを示唆します。一方で、複雑な非線形依存や多変量間の相互作用が強いデータでは Transformer 系が有利という主張もあり、データの性質に応じた使い分けが現実解です。また、効率化のための近似(ProbSparse)が精度をどこまで犠牲にするかは、その後より厳密で軽い手法(FEDformer のフーリエ選択など)に置き換わる流れの中で相対化されました。

## 発展トピック・研究の最前線

効率的注意の系譜(Informer → Autoformer → FEDformer → Pyraformer)と、それへの反動(DLinear の単純モデル、PatchTST のパッチ化回帰)が交互に現れているのが時系列予測研究の構図です。さらに最近は、変数(チャネル)を時間軸ではなくトークン軸に置く [iTransformer — 変数をトークンにする転置型Transformer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0488_itransformer)、複数の時系列をまとめて事前学習する **時系列基盤モデル(TimesFM, Moirai, Chronos など)** が登場し、「データセットごとに学習する」から「事前学習済みモデルをゼロショット/少数ショットで使う」へと関心が移りつつあります。Informer の貢献は、こうした大きな流れの起点として「長系列を Transformer で扱う計算量の壁をどう破るか」という問いを定式化した点にあり、その問いは形を変えて今も中心的なテーマであり続けています。

## さらに学ぶための関連トピック
- [Transformer 時間方向アテンション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0344_transformer-temporal-attention)
- [Autoformer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0321_autoformer)
- [PatchTST](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0336_patchtst)
- [Positional Encoding](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0338_positional-encoding)
