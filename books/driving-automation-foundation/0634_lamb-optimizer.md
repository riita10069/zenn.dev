---
title: "LAMB"
---

# LAMB

## ひとことで言うと
LAMB は、巨大なバッチサイズ(一度に大量のデータをまとめて学習する設定)でも壊れずに学習できるよう、Adam に「層ごとの学習率調整」を組み込んだオプティマイザです。これにより、大規模言語モデル BERT の事前学習を、従来の約 3 日から 76 分まで短縮しました。LAMB = Layer-wise Adaptive Moments optimizer for Batch training の略です。一言でいえば「Adam の賢さ(要素ごとの学習率調整)はそのままに、LARS の層ごと正規化を重ねて、超大バッチでも発散しないようにした」手法です。

## 直感的な理解
大規模言語モデルの事前学習は、とにかく時間とお金がかかります。BERT は当時、数台規模の TPU で 3 日ほど回す必要がありました。これを速くする一番素直な方法は「バッチサイズを大きくして、多数の演算装置で並列に処理する」ことです。1 回の更新でより多くの文を一度に食わせれば、同じデータ量を少ない更新回数で学習し切れます。更新回数(イテレーション)はそのまま「全演算装置が待ち合わせる同期回数」でもあるので、これを減らすことが分散学習の高速化に直結します。

ところがバッチを大きくすると学習率も上げる必要があり([LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0635_lars-optimizer) の線形スケーリング則を参照)、上げると学習が発散します。画像分類ではこの問題を LARS が解決していました。しかし LARS のベースはモメンタム付き SGD です。BERT のような Transformer 言語モデルでは、SGD ベースより Adam ベースの方が学習が圧倒的に安定することが経験的に知られていました。Transformer は層ごとに勾配のスケールが大きく異なり(埋め込み層・Attention の射影・FeedForward・LayerNorm のパラメータがそれぞれ違う桁の勾配を出す)、Adam の「要素ごとに勾配の大きさで割る」適応がほぼ必須だったのです。

そこで「Adam に LARS の層ごと調整を組み合わせれば、Transformer の超大バッチ学習が安定するのでは」という発想で生まれたのが LAMB です。Adam が各パラメータ内部の事情に合わせ、その上から LARS 風の層ごと正規化が層全体のバランスを取る、という二段構えです。役割分担がきれいに分かれているのがこの手法の美点です。

## 基礎: 前提となる概念

前提を 1 つずつ噛み砕きます。

- **Adam**: モメンタム(過去の勾配の指数移動平均で「進むべき方向」をならす。詳しくは [SGD with Momentum](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0653_sgd-momentum) / [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0642_nadam_note))と、勾配二乗の移動平均(各パラメータがどれだけ激しく動くかを測る。[Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0615_adagrad) の系譜)を組み合わせた定番オプティマイザです。「要素ごとに学習率を自動調整する」のが特徴で、勾配が大きく暴れる方向は小さく、おとなしい方向は大きく進めます。
- **モーメント(moment)**: 勾配の 1 次モーメント `m`(平均、=方向)と 2 次モーメント `v`(二乗平均、=ばらつき)のことです。Adam も LAMB もこの 2 つをパラメータごとに保持します。指数移動平均なので、減衰係数 β1(典型 0.9)、β2(典型 0.999)で過去をどれだけ引きずるかを決めます。
- **bias correction(バイアス補正)**: 移動平均は学習初期に 0 から始まるため値が小さく出すぎます。これを `1 − β^t`(t はステップ番号)で割って補正したものが `m_hat = m/(1−β1^t)`, `v_hat = v/(1−β2^t)` です。t が大きくなると分母は 1 に近づき、補正の効果は薄れます。
- **大バッチ学習**: バッチサイズを数千〜数万にして多数の GPU/TPU で並列に学習し、学習時間を短縮する手法です([LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0635_lars-optimizer) 参照)。
- **BERT**: 2018 年に Google が発表した Transformer ベースの双方向言語モデル(Devlin et al., NAACL 2019)です。大量のテキストでマスク言語モデル(文中の単語を隠して当てる課題)と次文予測として事前学習します。この事前学習が重く、当時 3 日ほどかかっていました。
- **信頼比(trust ratio)**: LARS から受け継いだ概念で「層の重みノルム ÷ その層の更新量ノルム」です。更新を重みのサイズに見合った大きさへ調整する係数です。これがゼロを正規化の中心思想として引き継いでいます。

## 仕組みを詳しく

LAMB は「各層について、Adam が計算した更新量を、その層の重みの大きさに合わせて再スケールする」ことをします。LARS との決定的な違いは、**再スケールする対象が『素の勾配』ではなく『Adam が適応的に計算した更新量』である**点です。これにより、要素ごとの適応(Adam)と層ごとの正規化(LARS)が両立します。

各層(正確にはパラメータグループ)l について、ステップを分解します。

1. Adam とまったく同じく、モメンタム `m` と勾配二乗移動平均 `v` を更新し、バイアス補正した `m_hat`, `v_hat` を求める。
   ```
   m ← β1·m + (1−β1)·g
   v ← β2·v + (1−β2)·g²
   m_hat = m/(1−β1^t),  v_hat = v/(1−β2^t)
   ```
2. Adam の素の更新方向を作る。重み減衰は Adam ではなく更新方向側に足す([AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer) と同じ decoupled weight decay の流儀):
   ```
   r_l = m_hat / ( sqrt(v_hat) + ε ) + λ·W_l
   ```
3. その層の重みノルム `||W_l||` と、更新方向のノルム `||r_l||` を計算する。
4. 信頼比を計算する:
   ```
   trust_ratio = φ(||W_l||) / ||r_l||
   ```
   φ は重みノルムに対する関数で、単純には φ(x) = x(そのまま)を使います。実装では `φ(x) = min(max(x, lower), upper)` のようにクリップして、ノルムが極端なときの暴走を防ぐこともあります。`||W_l||` が 0(重みが全部ゼロ)のときは trust_ratio = 1 にフォールバックするのが定石です。
5. 学習率にこの信頼比を掛けて更新する:
   ```
   W_l ← W_l − γ × trust_ratio × r_l
   ```
   γ はグローバル学習率です。

骨格を簡略化して並べると:

```
Adam の更新方向 r_l = m_hat / (sqrt(v_hat) + ε) + λ·W_l
信頼比 = ||W_l|| / ||r_l||
W_l ← W_l − γ × 信頼比 × r_l
```

つまり「要素ごとの適応(Adam の `1/sqrt(v_hat)`)」と「層ごとの正規化(LARS 由来の信頼比)」を二段重ねにしています。Adam がパラメータ内部のスケール差を吸収し、信頼比が層と層の間のスケール差を吸収する、という役割分担です。

直感を数値で見ます。グローバル学習率 γ = 1.0 とします。

- 層A: ||W|| = 10、Adam 更新方向のノルム ||r|| = 2 → 信頼比 = 5 → 実効的に大きく動かす。
- 層B: ||W|| = 0.5、||r|| = 2 → 信頼比 = 0.25 → 控えめに動かす。

各層は「自分の重みの大きさに対して一定の割合」で動くようになり、重みが小さい層が相対的に大きく動かされて壊れることを防ぎます。重要な観察は、LAMB の 1 ステップでの相対的な重み変化 `||ΔW_l|| / ||W_l||` が γ にほぼ等しくなる点です。つまり「すべての層が同じ割合 γ だけ進む」よう正規化されており、これが超大バッチでも発散しない理由です。Adam 単体ではこの相対変化が層ごとにバラバラになり、一番大きく動く層が学習を壊します。

**tensor shape の話**: 信頼比は「層ごとに 1 個のスカラー」です。状態(`m` と `v`)は Adam と同じくパラメータと同サイズの配列を 2 つ持ちます。層ごとのノルム計算が毎ステップ加わりますが、要素数に線形のコストで小さいです。メモリは実質 Adam / AdamW と同等と考えてよいです。分散学習では、各層のノルムを all-reduce で集約する必要があるため、通信パターンが Adam よりわずかに複雑になります。

## 手法の系譜と主要論文

LAMB は 2 つの系譜の合流点にあります。

- 適応的オプティマイザの系譜: [Adagrad](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0615_adagrad) → RMSProp/[Adadelta](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0614_adadelta) → Adam(Kingma & Ba, ICLR 2015)→ [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer)/[Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0642_nadam_note)/[RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0649_radam_note) などの改良。
- 大バッチ対応の系譜: Goyal et al.(線形スケーリング則+warmup, 2017)→ [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0635_lars-optimizer)(You et al., 2017, SGD ベース、CNN 向け)→ LAMB(Adam ベース、Transformer も含む)。

**Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli, Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh, "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes" (ICLR 2020, arXiv:1904.00962)** が LAMB の原論文です。

- 提案: LARS の層ごと正規化を Adam に組み込んだ LAMB。要素ごとの適応(Adam)と層ごとの信頼比スケーリングを組み合わせました。
- 動機: BERT のような Transformer を超大バッチで安定して学習し、TPU/GPU を大量に使って学習時間を劇的に短縮したかったから。SGD ベースの LARS では Transformer に十分でなかったため、Adam ベースに作り替えました。
- 新規性: 信頼比に「Adam の更新方向のノルム」を使う定式化と、その収束に関する理論解析(滑らかな非凸目的かつ有界分散という仮定の下で、確率的勾配法として O(1/√(Tb)) オーダで停留点へ収束するという保証。T はステップ数、b はバッチサイズ)を与えた点です。理論が「バッチを増やせば必要ステップが減る」ことを裏づけている点が、大バッチ学習の正当化として重要でした。

その後、LAMB は Hugging Face / NVIDIA(APEX)/ TensorFlow など主要フレームワークに実装され、大規模事前学習の標準的な選択肢の 1 つになりました。

## 論文の実験結果(定量データ)

主な数値を、指標の意味とともに説明します。

最も有名な結果は **BERT 事前学習の高速化** です。BERT の事前学習は 2 段階(系列長 128 のフェーズ 1 と、系列長 512 のフェーズ 2)で行います。フェーズ 2 だけ系列が長く Attention のコストが系列長の二乗で効くため重くなります。

- バッチサイズ 32K(フェーズ 1)/ 32K(フェーズ 2)という当時として桁外れの大バッチで、LAMB は精度を落とさずに学習を完了しました。総ステップ数はベースラインの約 100 万ステップから 8599 ステップ程度まで激減します。
- 学習時間は、ベースライン(Adam, バッチ 512 程度)の約 3 日に対し、LAMB + 1024 TPUv3 チップで **76 分**。論文タイトルの「76 minutes」はこれを指します。
- 精度の指標は下流タスク **SQuAD v1.1 の F1 スコア**(質問応答で、予測した答えの範囲が正解とどれだけ重なるかを測る指標。100 点満点に近いほどよい)です。LAMB で大バッチ学習した BERT は、ベースラインと同等以上の F1(およそ 90.4〜91.5、ベースラインは約 90.4)を達成しました。つまり「速くしても賢さは落ちていない」ことが示されています。
- さらに大きなバッチ 64K/32K(フェーズ 1 を 64K、フェーズ 2 を 32K)でも学習が成立し、ほぼ線形のスケーリング(76.7% のスケーリング効率)で時間短縮できることを報告しています。

アブレーション・対比の要点:

- **Adam をそのまま大バッチにすると壊れる**: バッチを 16K 以上に増やすと、素の Adam では精度が大きく劣化(あるいは発散)します。LAMB はこの領域で精度を維持し、層ごと正規化の効果を裏づけました。
- **信頼比を外すと劣化**: 信頼比のスケーリングを取り除くと(=素の AdamW に戻すと)、超大バッチでの安定性が失われます。LAMB の効きどころが「層ごと正規化」であることを示すアブレーションです。
- **LARS では Transformer に不十分**: 同じ大バッチで SGD ベースの LARS を BERT に使うと精度が出ません。Adam ベースに移植したことが本質だったことを示します。
- 画像分類(ResNet-50 on ImageNet)でも検証され、バッチ 16K〜32K で LARS と同等以上(top-1 約 76.7%、わずか数エポックの追加)の精度を出すことを示しています。Transformer 専用ではなく汎用に使えることの傍証です。

注意すべきは、これらの利点が出るのは **本当に超大バッチ・大規模分散** の領域だという点です。小〜中バッチの普通の学習では、LAMB は素の Adam / AdamW とほぼ差が出ません。

## メリット・トレードオフ・限界

メリット

- 巨大バッチ(数万)でも発散させずに学習でき、多数の演算装置で学習時間を劇的に短縮できる。
- Adam の要素ごと適応と層ごとの正規化を両立させているため、Transformer 言語モデルの大バッチ事前学習に強い。
- 層ごとに持つのはスカラーの信頼比だけで、状態のメモリ追加は Adam / AdamW とほぼ同じ。
- 画像・言語の両方で効果が確認されており、汎用性が高い。

トレードオフ・限界

- 利点が出るのは本当に大規模・大バッチの領域に限られる。小〜中バッチでは素の Adam / AdamW と差が出にくく、むしろ調整の手間だけが増えることがある。
- 信頼比・層ごとノルム・φ のクリップ範囲など実装がやや複雑で、学習率・重み減衰の調整に手間がかかる。バイアスや LayerNorm のパラメータには信頼比を適用しない(あるいは φ をかけない)のが定石で、この扱いを誤ると劣化する。
- 大バッチ学習自体に、線形スケーリング則・warmup([RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0649_radam_note) の暖機との関連も)・学習率スケジュール([LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0150_learning-rate-scheduler))など別の工夫を併せて必要とすることが多い。
- 超大バッチ領域には依然として「汎化ギャップ」(大バッチで学習したモデルは小バッチよりわずかに汎化が劣る傾向)という未解決問題があり、LAMB はこれを緩和するが消し去るわけではない。どのバッチサイズが品質と速度の最適点かは、モデル・データごとに探索が必要です。

## 発展トピック・研究の最前線

- **より大きなモデルへの適用**: LAMB は BERT 以降、GPT 系や T5 など大規模事前学習でも使われ、データ並列のスケールアップを支えました。一方、超大規模では Adam(AdamW)+ 慎重な warmup と学習率スケジュールでも十分安定するケースが多く、LAMB を使うかどうかは実装・インフラとの相性で判断されるようになっています。実際、近年の代表的な LLM では AdamW を使い続ける例も多く、LAMB の「大バッチ必須性」は計算資源とのトレードオフで決まります。
- **2 次・行列構造を使う系統との比較**: 同じ「大規模学習を速く」という目的で、対角ヘシアンを使う [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0655_sophia-optimizer) や、行列前処理の [Shampoo オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0654_shampoo-optimizer) が登場しました。LAMB が「層ごとスカラー正規化」なのに対し、これらは曲率や行列構造まで使う点が異なります。どれが最良かはモデル規模・タスク・計算予算に依存し、決着していません。
- **理論の深化**: 信頼比正規化がなぜ大バッチを安定させるのか、層のスケール不変性(正規化層との相互作用)との関係など、最適化理論側からの理解が進んでいます。とくに「重みを定数倍しても出力が変わらない層では、絶対的な更新量より相対的な更新量が本質的」という視点が、信頼比の正当化につながっています。
- **NLP 以外への展開**: 大バッチ対照学習(自己教師あり表現学習)など、バッチサイズが品質に直結するタスクで LARS / LAMB が使われ続けています。

## さらに学ぶための関連トピック

- [LARS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0635_lars-optimizer)
- [AdamWオプティマイザの選択](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0572_adamw-optimizer)
- [Sophia オプティマイザ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0655_sophia-optimizer)
- [LR Scheduler (warmup/cosine)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0150_learning-rate-scheduler)
- [RAdam (Rectified Adam)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0649_radam_note)
- [Nadam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0642_nadam_note)
