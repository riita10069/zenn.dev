---
title: "Activation Recomputation (selective gradient checkpointing)"
---

# Activation Recomputation (selective gradient checkpointing)

## ひとことで言うと
ニューラルネットの学習中、メモリを大量に食う「途中の計算結果(activation、活性値)」を全部保存しておく代わりに、必要になったときにもう一度計算し直すことで GPU メモリを節約する技術です。selective(選択的)というのは「全部を再計算するのではなく、メモリ削減効果が大きく再計算コストが小さい部分だけを選んで再計算する」という賢いやり方を指します。これにより、計算時間の増加をわずかに抑えつつ、大きなメモリ削減を得られます。

## 直感的な理解
試験勉強で、教科書の全ページにびっしり付箋を貼って全部を手元に広げておくか(=メモリを食う)、章の見出しだけ覚えておいて必要になったらそのページを開き直すか(=その場で計算し直す)、を選ぶ場面を想像してください。全部広げておけば速いがスペースを食う。見出しだけにすれば省スペースだが、開き直す手間がかかる。activation recomputation はまさにこのトレードオフを、GPU メモリと計算時間の間で行います。

なぜこれが要るのか。深層学習の学習では、モデルを大きくしたい・バッチを大きくしたい・入力解像度を上げたい、という欲求が常にあります。しかし GPU の VRAM(ビデオメモリ)は有限です。そして学習中のメモリの大きな割合を占めるのが activation です。これを全部抱えると、すぐにメモリ不足(OOM, Out Of Memory)で学習が落ちます。activation を「捨てて必要時に作り直す」ことで、同じ GPU でより大きなモデルや解像度を扱えるようになります。

## 基礎: 前提となる概念
まず「なぜ学習でメモリを食うのか」を押さえます。ニューラルネットの学習は2段階です。

1. forward(順伝播): 入力を層に通して出力を計算する。
2. backward(逆伝播): 出力の誤差(loss)から各層の勾配(gradient、パラメータをどう動かすべきかの方向と大きさ)を計算する。

ここで activation(活性値)とは「forward の途中で各層が出した中間出力」のことです。問題は backward です。各層の勾配を計算するには、連鎖律(chain rule、合成関数の微分のルール)により、その層が forward で出した activation が必要になります。だから普通の学習では forward で計算した activation を全部メモリに取っておき、backward で使い回します。これが activation がメモリを圧迫する理由です。

メモリ量を見積もってみます。Transformer(自己注意機構を使うネット)で、層数 L、バッチサイズ B、系列長 S、隠れ次元 H とすると、保存する activation の総量はおおむね

```
activation メモリ ∝ L × B × S × H
```

に比例します(注意機構の内部はさらに S×S の項も持つ)。層が深く(L 大)、系列が長く(S 大)、バッチが大きい(B 大)ほど、activation は線形〜二次的に増えます。たとえば視覚モデルで高解像度の鳥瞰図(BEV)特徴を扱うと、特徴マップの解像度が H, W で効いてくるため、解像度を上げただけで activation が数倍に膨らみ、VRAM をあっという間に使い切ります。

そこで発想を変えます。「activation を保存しておく代わりに、backward で必要になった瞬間に、保存しておいた入力からその区間の forward をもう一度走らせて activation を作り直す」。これが activation recomputation(活性値の再計算)で、別名 gradient checkpointing(勾配チェックポインティング)とも呼ばれます。メモリは減りますが、forward を一部やり直すので計算時間は増えます。つまり典型的な「メモリと計算時間のトレードオフ」です。

## 仕組みを詳しく
### 基本形(full recomputation)

最も単純なやり方は「forward 中は activation をほとんど捨てる。backward で必要になったら、保存しておいた区間の入口からその区間を再 forward して activation を再生成し、勾配を計算したら再び捨てる」です。

数値感覚で示します。1層あたりの activation メモリを M、層数を L とすると、

```
通常:        activation メモリ = L × M(全層ぶん抱える)
区間チェックポイント: メモリ ≈ √L × M 程度まで削減(区間に区切り中間点だけ保存する古典手法)
                    ただし forward の総計算量は約2倍に増える
```

直感的には「全部覚えるのをやめて、区切りごとの入口だけ覚えておき、必要なら入口から作り直す」ことでメモリを大きく減らせます。最適な区間分割を取ると、メモリは層数 L に対して O(√L) まで下がる、というのが古典的な結果です。代償は、最悪で forward を丸ごともう一度走らせるぶんの追加計算です。

### selective(選択的)recomputation

ここが本トピックの肝です。全層を一律に再計算すると、メモリは大きく減りますが計算時間が約2倍近くになり遅くなります。しかし層の中身を観察すると、「保存するとメモリを大量に食うが、再計算自体は安い」演算と、「メモリはそこそこだが再計算が高い」演算が混在しています。

Transformer の場合、self-attention の内部の softmax 出力・dropout マスク・attention の重み行列(サイズ B×heads×S×S)などは、メモリを大量に食うのに再計算は軽い(主要な行列積に比べて演算量が小さい)ことが知られています。逆に、大きな行列積(QKV 射影や FFN の線形層)の出力は再計算コストが高い。

selective recomputation は「メモリ削減量 / 再計算コスト の比が良い部分だけを再計算対象に選び、再計算が高い部分はあえて保存しておく」やり方です。

```
1層の内訳: [ QKV射影(大きな行列積) ][ attention行列+softmax+dropout ][ FFN(大きな行列積) ]
保存する  :  行列積の出力(再計算が高い → 保存して再計算を避ける)
再計算する:  attention行列・softmax・dropout(メモリ食いだが安い → 捨てて backward で作り直す)
```

これにより「メモリは大きく減るのに、追加の計算時間はわずか(全再計算のような2倍にはならず、数 % 程度)」という良いバランスが取れます。

PyTorch では `torch.utils.checkpoint.checkpoint` を使うと、指定した区間を「forward では activation を捨て、backward で再 forward する」ようにできます。selective にするには「どのモジュール・どの演算を checkpoint で包むか」を選びます。なお dropout のように乱数を使う層を再計算するときは、forward と backward で同じ乱数列を再現しないと結果がずれるため、実装は乱数状態(RNG state)を保存して再現します(PyTorch の checkpoint はこれを内部で扱います)。

## 手法の系譜と主要論文
Chen, Xu, Zhang, Guestrin の「Training Deep Nets with Sublinear Memory Cost」(arXiv:1604.06174, 2016)が gradient checkpointing の基礎を確立しました。系列を区間に区切り、区間の入口だけ保存して backward 時に区間内を再 forward することで、メモリを層数 L に対して O(√L) まで下げられることを示しました。動機は「メモリ制約で大きなモデル・長い系列が学習できない問題」の解決。効果は大幅なメモリ削減。トレードオフは forward の追加計算(報告ではおおむね 30% 程度の計算時間増)です。

Shoeybi らの「Megatron-LM」(arXiv:1909.08053, 2019)系の論文群は、数十億〜数千億パラメータの巨大 Transformer を複数 GPU で学習するため、テンソル並列(1つの層の行列を複数 GPU に分割)と recomputation を組み合わせる文脈を整備しました。巨大化すると activation メモリが支配的になるため、recomputation はほぼ必須の道具になりました。

Korthikanti らの「Reducing Activation Recomputation in Large Transformer Models」(arXiv:2205.05198, 2022)が本トピックの中心的な論文です。full recomputation の計算オーバーヘッドが巨大モデルでは無視できない点に着目し、(1) selective activation recomputation — attention 内のメモリ食いだが再計算が安い部分だけを再計算する設計、(2) sequence parallelism — 系列方向にも並列化して activation を複数 GPU に分散させる手法、を提案しました。この2つを組み合わせることで、full 再計算が生む計算オーバーヘッドを大きく削りつつ、activation メモリも大幅に減らせることを示しました。

その後、FlashAttention(Dao et al., 2022)が登場し、attention を「巨大な S×S の中間行列を materialize(実体化)せずに、ブロック単位でオンチップ計算する」ことで、attention の activation メモリそのものを根本的に削減しました。これにより「attention 部分を recomputation で節約する」必要性は減りましたが、FFN や深い層全体に対する gradient checkpointing は依然として重要な道具であり、両者は併用されます。

## 論文の実験結果(定量データ)
Chen らの 2016 年論文では、メモリを O(√L) に落とせることを理論と実験で示し、その代償として forward の追加計算がおよそ 30% 程度であると報告しました。ここで重要な指標は「学習可能な最大モデルサイズ/系列長」で、同じ GPU メモリでより大きなモデルが回せることが直接の価値です。

Korthikanti らの 2022 年論文では、ベースラインを「full activation recomputation(全層再計算)」とし、selective recomputation + sequence parallelism を適用した場合の比較を行っています。報告の要点は次の通りです。full recomputation は活性メモリをほぼゼロ近くまで削れる代わりに、計算オーバーヘッドが数十 % 規模(典型的に 30〜40% 程度)生じる。これに対し selective recomputation は、再計算する演算を attention 内の安価な部分に限定することで、計算オーバーヘッドを数 % 程度(報告では full 再計算の何分の一)に抑えつつ、activation メモリを大きく削減できた、という結果です。さらに sequence parallelism を併用すると、活性メモリをテンソル並列度に応じて分散でき、超大規模モデルの学習を現実的なメモリ内に収められることを示しました。アブレーションとして「selective を外して full にすると速度が落ちる」「sequence parallelism を外すと活性メモリが GPU あたりで増える」ことを定量的に確認しています。

実務的な数値感覚としては、視覚モデルや BEV のように高解像度特徴で activation が膨らむケースで、エンコーダや fusion ブロックを gradient checkpointing で包むと、ピーク VRAM を数十 % 削減でき、その代わりに 1 イテレーションの時間が十数 % 程度伸びる、というトレードオフが典型です。これによりバッチサイズや入力解像度を1段上げられるなら、最終的な精度に効くことが多く、割の良い選択になります。

## メリット・トレードオフ・限界
メリット
- GPU メモリを大きく削減でき、同じ GPU でより大きなバッチ・高解像度・深いモデルを学習できる。
- selective 版なら計算時間の増加を数 % 程度に抑えつつメモリを減らせる(full 再計算より割が良い)。
- 標準的なフレームワーク機能(PyTorch の `torch.utils.checkpoint` など)で実装でき、モデル構造を大きく変えなくてよい。
- データ並列・テンソル並列・ZeRO などの他のメモリ削減手法と直交して併用できる。

トレードオフ・限界
- forward を一部やり直すので計算時間が増える(full 再計算で 30〜50% 程度、selective でも多少は増える)。メモリと速度の交換。
- どの層・どの演算を再計算対象に選ぶかの判断が要る。選び方を誤ると、メモリも大して削れず速度だけ落ちる。
- 乱数を使う層(dropout など)を再計算するときは、forward/backward で同じ乱数を再現する必要があり実装に注意が要る。
- メモリ不足が桁違いに大きい場合は、これだけでは足りず CPU/NVMe offload や並列化など他手法との併用が必要。
- FlashAttention のように、そもそも中間を materialize しない手法が使える場面では、attention に対する recomputation の効果は薄れる。

研究上の課題として、「どの演算を再計算すべきか」を計算グラフから自動で最適選択する(メモリ予算を制約に計算時間を最小化する離散最適化)アプローチが進んでおり、手動チューニングを置き換えつつあります。

## 発展トピック・研究の最前線
- 自動チェックポイント選択: 計算グラフを解析し、メモリ予算の制約下で再計算箇所を最適化する手法(Checkmate などの整数計画ベースのアプローチ)。
- メモリ削減手法の組み合わせ: ZeRO(optimizer 状態・勾配・パラメータの分割)、CPU/NVMe offload、混合精度(bf16/fp16)、テンソル/パイプライン/シーケンス並列との併用設計。
- FlashAttention 系のカーネル融合(IO-aware な実装)で attention の中間を作らずに済ませる流れと、recomputation の役割分担。
- 学習以外への応用: 長系列推論や RLHF の rollout で同様のメモリ・計算トレードオフを取る場面。
- ハードウェア進化(より大きい HBM、GPU 間高速相互接続)とアルゴリズム的メモリ削減のせめぎ合い。

## さらに学ぶための関連トピック
- [CPU/NVMe Offload](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0164_cpu-offload)
- [優先度プリエンプション (research/production の2段階制御)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1140_priority-preemption-queue)
- [オブジェクトストレージによるデータレイク (datasets/checkpoints/artifacts)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0835_s3-data-lake)
- [Ingest Adapter (データソース取込みの共通インタフェース)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0816_ingest-adapter-protocol)
