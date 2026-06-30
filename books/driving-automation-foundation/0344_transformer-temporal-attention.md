---
title: "Transformer 時間方向アテンション"
---

# Transformer 時間方向アテンション

## ひとことで言うと
時系列(時間順に並んだデータ列)の各時刻が、他のどの時刻を見るべきかを「内積で一気に計算」する仕組みです。RNN のように1ステップずつ順番に処理するのではなく、全時刻を同時に・並列に処理できるため、遠く離れた時刻どうしの関係(長距離依存)も速く正確に捉えられます。2017年の Transformer によって標準化され、現在の時系列モデル・言語モデル・画像モデルの土台になりました。

## 直感的な理解
ある日の株価を予測したいとします。直感的には、昨日や一昨日だけでなく「1週間前」「1か月前の同じ曜日」「決算発表のあった日」など、過去のさまざまな時点が今日の値に効いてきます。しかも、どの時点がどれだけ効くかは状況によって変わります。

RNN はこれをバケツリレーで解こうとします。情報を1時刻ずつ隣に渡していき、遠い過去の情報は何十回もリレーされて今日に届きます。途中で情報が薄れたり混ざったりするうえ、リレーは順番にしかできないので時間もかかります。

自己注意(self-attention)はバケツリレーをやめ、今日から過去の全時点へ直接「線」を引きます。各時点との関連の強さをいっせいに計算し、強い時点ほど大きく参照する。1か月前の同じ曜日も、決算日も、距離に関係なく「1ホップ(1回の参照)」で届きます。情報が経由する経路の長さ(path length)が、RNN では系列長 L に比例するのに対し、自己注意では定数(1)になる。これが「長距離依存に強い」「並列に処理できる」という Transformer の2大特徴の正体です。

## 基礎: 前提となる概念

まず **RNN(Recurrent Neural Network、再帰型ニューラルネット)** の限界を理解します。RNN、特に LSTM や GRU は「前の時刻の隠れ状態(数値ベクトル)を次の時刻に渡す」形で系列を処理します。時刻 t の計算には時刻 t-1 の結果が必要です。ここに2つの根本問題があります。

1. **並列化できない**。時刻を順番に処理するしかないので、系列長 L のデータには L 回の逐次ステップが必要です。GPU は大量の計算を同時にこなすのが得意なのに、その能力を活かせず学習が遅くなります。

2. **長距離依存が弱い**。時刻1の情報を時刻100に届けるには、隠れ状態を99回バケツリレーする必要があります。その途中で **勾配(学習の信号)** が小さくなりすぎたり大きくなりすぎたりして(**勾配消失・勾配爆発**)、遠い過去の情報が伝わりにくくなります。勾配とは「パラメータをどちらにどれだけ動かせば誤差が減るか」を示す量で、これが消えると学習が進みません。LSTM のゲート機構はこれを緩和しますが限界があります。

[Bahdanau Attention (additive)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0322_bahdanau-attention) の注意機構は「出力時に入力全体を加重和で見る」ことを可能にしましたが、それでも土台は RNN のままで、逐次性は残っていました。Vaswani らの問いは大胆でした。「注意だけでネットワークを組めば、RNN を完全に捨てられるのではないか」。答えが Transformer です。

もう1つの前提が **埋め込み(embedding)** です。これは離散的な記号(単語など)や1時刻ぶんの観測を、固定次元の連続ベクトルに変換したものです。Transformer はこのベクトル列を入力として受け取ります。なお、自己注意がデータについて持つ前提(帰納バイアス、inductive bias)は CNN や RNN より弱い、という点も重要です。CNN は「近い要素ほど関係が強い」、RNN は「順序が重要」という構造をあらかじめ組み込んでいますが、自己注意は全ペアを平等に見るところから始め、関係をすべてデータから学びます。柔軟だが、そのぶん多くのデータを要求する、という性質につながります。

## 仕組みを詳しく

中心は **Self-Attention(自己注意)** です。Bahdanau は「出力(デコーダ)が入力(エンコーダ)を見る」クロスアテンションでしたが、自己注意は「系列が自分自身を見る」点が違います。時系列の各時刻が、同じ系列内の全時刻を参照します。

### Query / Key / Value

入力を時系列の特徴行列 X とします。形状(tensor shape)は `[L, d]`(L = 系列長、例 100、d = 1時刻あたりの特徴次元、例 256)。これを3つに線形変換します。

```
Q = X · W_Q   形状 [L, d_k]   Query (問い合わせ)
K = X · W_K   形状 [L, d_k]   Key   (見出し)
V = X · W_V   形状 [L, d_v]   Value (中身)
```

直感としては、Query は「私はこういう情報を探している」、Key は各時刻の「私はこういう情報を持っているという見出し」、Value は「実際に渡す中身」です。図書館で例えるなら、Query が検索キーワード、Key が本の背表紙のタイトル、Value が本の中身に当たります。Q/K/V を別々の重みに分けることで、「探す軸」と「中身」を独立に学習できます。

### Scaled Dot-Product Attention

注意スコアは Query と Key の内積で一気に計算します。

```
Attention(Q, K, V) = softmax( Q · K^T / sqrt(d_k) ) · V
```

ステップ分解します。

1. `Q · K^T`: 形状 `[L, d_k] × [d_k, L] = [L, L]`。これは「各時刻が各時刻にどれだけ関連するか」の全ペアのスコア表(L×L 行列)です。L=100 なら 100×100 のスコアが一度に出ます。内積は2つのベクトルの「向きの近さ」を測るので、似た情報を持つ時刻どうしほど大きな値になります。
2. `/ sqrt(d_k)`: スケーリング。d_k が大きいと内積の値(d_k 個の積の和)の分散が d_k に比例して大きくなり、softmax が極端に尖って(ほぼ1か0になって)勾配が消えるため、sqrt(d_k)(例 sqrt(64)=8)で割って分散を1程度に戻し安定させます。これが「Scaled(スケール済み)」の由来で、内積型を素朴に使うと加算型に劣るという当時の観測への回答でした。
3. `softmax(...)`: 各行を合計1の重みに正規化します。時刻 i の行が「時刻 i が各時刻に払う注意の配分」です。
4. `· V`: その重みで Value を加重和します。結果は `[L, d_v]` で、各時刻について「他時刻から集めた要約」になります。

ここで重要なのは、すべての時刻のスコアが1個の行列積で同時に出ることです。RNN の逐次ループが消え、GPU で並列計算できます。

### Multi-Head Attention(マルチヘッド注意)

1組の Q/K/V だけだと、捉えられる関係が1種類に偏ります。そこで d 次元を例えば8分割し、8組の小さな注意(**ヘッド**)を並列に走らせ、結果を連結します。あるヘッドは「直前の時刻との関係」、別のヘッドは「周期的な関係」を学ぶ、といった分業ができます。

```
head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)   i=1..8
MultiHead = Concat(head_1, ..., head_8) · W_O
```

各ヘッドは d/8 = 32 次元など小さく、合計するともとの d に戻ります。分割しても総計算量はほぼ変わらず、表現の多様性だけが増えるのが巧妙な点です。元論文は d=512, h=8, d_k=d_v=64 を標準としました。

### 因果マスク(時系列・自己回帰での必須要素)

時系列予測では「未来を見てはいけない」場面が多くあります。時刻 t を予測するのに時刻 t+1 以降の答えを参照したらカンニングです。そこで `[L, L]` のスコア行列の「未来側(上三角)」を -∞ にしてから softmax します。-∞ を softmax に通すと重みが0になるため、各時刻は過去と現在しか見られなくなります。これを **causal mask(因果マスク)** と呼びます。GPT 系の生成モデルや時系列の自己回帰予測で必須です。逆に、入力全体を一度に見てよい場面(分類や、未来を予測しないエンコーディング)ではマスクなしの双方向注意を使います。

### Encoder ブロックの全体

実際の Transformer 層は、自己注意の後に **残差接続(residual connection、入力をそのまま足し戻す)** と **LayerNorm(層正規化、各ベクトルの値の分布を整える)** を置き、続いて位置ごとの小さな全結合層(**Feed-Forward Network、FFN**。元論文は中間次元 2048 の2層 MLP)+ 残差 + LayerNorm を重ねます。これを N 層(元論文は6層)スタックします。残差接続は深い層でも勾配が流れやすくする工夫で、これがないと深い Transformer は学習が安定しません。自己注意は単独では順序を区別できないため、入力に [Positional Encoding](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0338_positional-encoding) を加えるのが前提です。なお元論文は LayerNorm をブロックの後ろに置く Post-LN でしたが、後の研究で前に置く Pre-LN のほうが深いモデルで学習が安定すると分かり、現在の大規模モデルの多くは Pre-LN を採用します。

## 手法の系譜と主要論文

- **Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin, "Attention Is All You Need" (NeurIPS 2017, arXiv:1706.03762)**。自己注意ベースの Encoder-Decoder を提案。動機は RNN の逐次性を排して並列学習し、長距離依存を1ホップで結ぶこと。
- **Devlin, Chang, Lee, Toutanova, "BERT" (NAACL 2019, arXiv:1810.04805)** と **Radford et al., "GPT" (2018-)**。Transformer の Encoder だけ(BERT、双方向、穴埋めで事前学習)/ Decoder だけ(GPT、因果マスク、次トークン予測)を大量データで事前学習する流れを作り、自己注意が言語理解・生成の標準アーキテクチャになりました。
- **Dosovitskiy et al., "An Image is Worth 16x16 Words" (ViT, ICLR 2021, arXiv:2010.11929)**。画像を16×16のパッチに切り、各パッチを1トークンとして Transformer に入れました。時系列ではありませんが、「データをパッチ化して Transformer」という発想は、のちに時系列へ持ち込まれます([PatchTST](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0336_patchtst))。
- **時系列向けの改良系譜**: Zhou et al. (Informer, AAAI 2021)、Wu et al. (Autoformer, NeurIPS 2021)、Zhou et al. (FEDformer, ICML 2022)、Nie et al. (PatchTST, ICLR 2023) は、いずれも Vaswani の自己注意を長期時系列予測向けに改良したものです([Informer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0334_informer) [Autoformer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0321_autoformer) [PatchTST](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0336_patchtst))。一方で Zeng et al. (DLinear, AAAI 2023) は「単純な線形モデルが多くの Transformer を上回る」と報告し、時系列における Transformer の優位性そのものを問い直しました。

## 論文の実験結果(定量データ)

Vaswani et al. (2017) は機械翻訳(WMT 2014)で評価しました。指標は前述の **BLEU**(出力訳と正解訳の n-gram 一致度、高いほど良い)です。

- 英独翻訳(EN-DE)で **28.4 BLEU** を達成し、当時の最良モデル(アンサンブルを含む)を 2 BLEU 以上上回って SOTA(state-of-the-art、その時点の最高性能)を更新しました。
- 英仏翻訳(EN-FR)で **41.8 BLEU** を達成し、単一モデルとして当時の最高水準に並びました。
- 特筆すべきは **学習コスト** です。大きい Transformer モデル(big)でも 8 枚の P100 GPU でおよそ 3.5 日の学習で上記の性能に到達し、従来の最良モデルが要した計算量(報告では FLOPs にして1〜2桁大きい)を大幅に下回りました。これは「並列化できる」という設計上の利点が、実際の学習時間短縮として現れた証拠です。
- **アブレーション**(構成要素を抜くとどうなるかの実験)も豊富です。ヘッド数を1にすると(シングルヘッド)BLEU がおよそ 0.9 低下し、逆にヘッドを増やしすぎても劣化しました(h=16, 32 で頭打ち〜微減)。適度なマルチヘッドが効くことが示されています。また、注意の鍵次元 d_k を小さくしすぎると性能が落ち、内積ベースのスコアの質が次元に依存することも確認されました。位置エンコーディングを正弦波から学習埋め込みに替えてもほぼ同等という結果も報告され([Positional Encoding](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0338_positional-encoding) 参照)、設計選択の感度が定量的に整理されています。

ViT(Dosovitskiy et al. 2021)は、JFT-300M(約3億枚)のような大規模データで事前学習すると、画像分類で ResNet 系を上回り(ImageNet で 88% 超のトップ1精度)、しかも事前学習の計算効率が良いことを示しました。一方、ImageNet-1k 程度の中規模データだけで一から学習すると同規模 CNN に負けることも報告され、「Transformer はデータが多いほど強い(帰納バイアスが弱いぶん大量データを要求する)」という性質が定量的に確認されました。この性質は時系列でも観察され、データが少ない設定で単純モデルが勝つ一因になっています。

## メリット・トレードオフ・限界

- メリット: 全時刻を並列計算でき、GPU を活かして学習が速い。
- メリット: 任意の2時刻を1回の注意で直接結べる(最大経路長が定数)ので、長距離依存に強い。
- メリット: マルチヘッドで複数種類の依存関係を同時に学べる。
- メリット: 同じ仕組みが言語・画像・音声・時系列・点群と分野を越えて使える汎用性。
- トレードオフ: 計算・メモリが系列長 L に対して O(L^2 · d)(L×L 行列を作るため)。長系列でメモリが足りなくなります。これを軽くする工夫が [Informer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0334_informer) や [PatchTST](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0336_patchtst) などです。
- トレードオフ: 自己注意単体は順序を区別できないため [Positional Encoding](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0338_positional-encoding) が必須。
- トレードオフ: データが少ないと、帰納バイアス(構造的な前提。CNN なら「近い画素ほど関係が強い」など)が弱いぶん過学習しやすく、RNN/CNN より多くのデータを要求しがち。
- 限界・未解決の論点: 時系列予測において Transformer が本当に優れているかは論争があります。DLinear(2023)は単純な1層線形モデル(系列をトレンドと季節に分解してから線形写像)が、当時の多くの Transformer を ETT などのベンチで上回ると報告し、「複雑な注意機構が時系列で常に得とは限らない」という流れを生みました。これに対し PatchTST はパッチ化とチャネル独立化で Transformer の優位を取り戻したと主張しており、決着はついていません。

## 発展トピック・研究の最前線

最大の研究方向は **O(L^2) の緩和** です。スパース注意(一部のペアだけ計算。Longformer, BigBird)、線形注意(注意を核関数で近似して O(L) にする。Linformer, Performer)、FlashAttention(Dao et al. 2022、softmax をブロック単位で計算しメモリアクセスを最適化、結果は厳密だが長系列を高速・省メモリで計算する実装)などが活発です。また注意を一切使わない **状態空間モデル(SSM、S4 や Mamba など)** が長系列で Transformer に匹敵あるいは上回る場面が出てきており、「attention is all you need」の前提そのものが問い直されています。時系列分野では、複数変量(多次元の時系列)の扱い方(チャネル独立 vs チャネル混合)、チャネル間依存をどうモデル化するか([iTransformer — 変数をトークンにする転置型Transformer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0488_itransformer) は変数を1トークンに割り当てる)、時系列基盤モデル(Chronos, Moirai, TimesFM など大規模事前学習)の構築などが最前線のテーマです。自動運転では、複数時刻・複数センサのトークンを時間方向と空間方向の両方で注意する時空間 Transformer が、軌跡予測や占有予測で標準的に使われています。

## さらに学ぶための関連トピック
- [Bahdanau Attention (additive)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0322_bahdanau-attention)
- [Positional Encoding](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0338_positional-encoding)
- [Informer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0334_informer)
- [Autoformer](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0321_autoformer)
- [PatchTST](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0336_patchtst)
