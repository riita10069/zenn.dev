---
title: "Fine-tuning"
---

# Fine-tuning

## ひとことで言うと
Fine-tuning (微調整) とは、他の大規模データで事前学習しておいたモデルを、いまやりたいタスク (下流タスク) のデータでもう一度学習し直すことです。ゼロから学習するのではなく「すでに賢いモデル」を出発点にするので、手元のデータが少なくても高い精度を狙えます。バックボーン (特徴抽出部) の重みまで含めて更新する点が、重みを固定する凍結方式 ([Frozen Backbone](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0355_frozen-backbone)) との決定的な違いで、転移学習の最も基本的で強力なやり方です。

## 直感的な理解
新しい言語を学ぶとき、すでに別の言語を話せる人は文法や音の感覚を流用してすばやく上達します。Fine-tuning はこれに似ています。事前学習済みモデルはすでに「画像を見る目」や「言葉を理解する力」を持っており、そこから本題のタスクへ少しだけ調整 (チューニング) して移行します。

深層ニューラルネットワークは性能が高い反面、ゼロから学習させるには膨大なデータと計算が必要です。画像認識で十分な精度を出すには何十万〜何百万枚ものラベル付き画像と長い学習時間が要ります。多くの現場ではそんな資源はありません。

一方で ImageNet のような巨大データで学習済みのモデルは、すでに「エッジ・質感・物体の部品」といった、多くの画像タスクで共通して役立つ汎用特徴を持っています。だったらそのモデルを初期値 (出発点) として借り、自分のタスクのデータで少し追加学習すれば、ゼロから始めるより圧倒的に有利なはずです。これが Fine-tuning の発想です。たとえば1万枚しか医療画像がなくても、ImageNet 学習済みモデルを微調整すればゼロ学習より高精度になりやすい。言語側でも、BERT のような事前学習済み言語モデルを下流タスクで微調整するのが標準レシピになっています。

## 基礎: 前提となる概念
- 事前学習済みモデル (pretrained model): 大規模データで先に学習を済ませた重み一式。研究機関・企業が公開したものをダウンロードして使うことが多い。
- 重み: 学習で決まる内部の数値。Fine-tuning ではこれを「初期値」として使い、さらに更新する。
- 損失地形 (loss landscape): 重みを座標、損失 (誤差) を高さとした地形のイメージ。学習は谷 (損失が小さい場所) へ下る作業。事前学習済みの重みはすでに良い谷の近くにある。
- 学習率 (learning rate): 1回の更新で重みをどれだけ大きく動かすかの度合い (谷を下る歩幅)。Fine-tuning では、せっかくの良い谷を飛び出さないよう、ゼロ学習時より小さめにするのが定石。
- ウォームアップ (warmup): 学習開始直後だけ学習率を小さく抑え、徐々に上げる手法。初期の不安定な勾配で重みを壊さないため。
- ヘッド (head): タスク専用の出力部分。下流タスクに合わせて付け替える。
- 破滅的忘却 (catastrophic forgetting): 新しいことを学ぶ過程で、事前学習で得た知識を急速に失う現象。Fine-tuning の主要なリスク。

## 仕組みを詳しく
Fine-tuning の基本手順は次のとおりです。

1. 事前学習済みモデルの重みを読み込み、初期値とする。
2. タスク専用の出力部分 (ヘッド) を今のタスクに合わせて付け替える。たとえば1000クラス分類用の最終層を、自分の10クラス用に置き換える。新しいヘッドの重みはランダム初期化。
3. 自分のタスクのデータでモデル全体 (または一部) を再学習する。学習率は小さめにし、しばしばウォームアップを併用する。

### 凍結 (Frozen) との違い
転移学習には2つの代表的な流儀があります。
- Frozen Backbone ([Frozen Backbone](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0355_frozen-backbone)): バックボーンの重みを固定し、ヘッドだけ学習する。速くて省メモリ、少データで安定。
- Fine-tuning: バックボーンの重みも含めて更新する。下流タスクに合わせて特徴そのものを調整できるので、データが十分あれば精度の上限が高い。

両者の中間として、最初はバックボーンを凍結してヘッドだけ学習し、安定したら徐々に解凍して全体を微調整する段階的解凍 ([Gradual Unfreezing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0388_gradual-unfreezing)) という実務テクニックがあります。また、層ごとに学習率を変える識別的学習率 ([Discriminative / Layer-wise Learning Rates](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0380_discriminative-learning-rates)) を併用して下層を守りながら上層を強く適応させるのも定番です。

### 数値イメージ (before / after)
入力画像 `[B, 3, 224, 224]` を分類する場合:
- before: ImageNet で1000クラスを当てるよう学習された重み。最終層は `[特徴次元, 1000]`。
- 付け替え: 最終層を `[特徴次元, 10]` に交換し、ランダム初期化。
- after (学習後): バックボーンの重みも自分のデータに合わせて少しずつ変化し (小さい学習率なので大きくは動かない)、最終層は自分の10クラスを当てられるよう学習される。

凍結との対比では、Fine-tuning ではバックボーンの重みが before と after で「変わる」点が決定的な違いです。なぜ学習率を小さくするのか――事前学習済みの重みはすでに損失地形の良い谷の近くにあるので、大きく動かすとその谷から飛び出して汎用知識を壊す (破滅的忘却) からです。小さな歩幅でその谷の中をターゲット側へ移動させるイメージです。

### よくある落とし穴: ランダムヘッドが下層を壊す
微調整の最初の数ステップで起きやすい問題があります。新しく付けたヘッドはランダム初期化なので、最初は大きく的外れな勾配を出します。全層を同じ学習率で動かすと、この大きな勾配が逆伝播でバックボーンの下層まで届き、せっかくの汎用特徴を初期段階で壊してしまう (feature distortion)。対策として、(1) 数エポックはヘッドだけ学習してから全体を解凍する、(2) ウォームアップで初期学習率を抑える、(3) 層ごとに学習率を変える、などが使われます。これは Kumar et al. (2022) が分布外性能の劣化の主因として指摘した現象でもあります。

## 手法の系譜と主要論文
- Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014。「下層は汎用、上層は特化」という層ごとの性質を定量化し、凍結だけだと深い層で性能が落ちるが fine-tuning で回復することを示した。Fine-tuning が標準レシピになる実験的裏付け。
- Howard & Ruder, "Universal Language Model Fine-tuning for Text Classification (ULMFiT)", ACL 2018。NLP に転移学習を持ち込み、discriminative fine-tuning・slanted triangular learning rates・gradual unfreezing を組み合わせて安定した微調整の作法を確立した。
- Devlin et al., "BERT", NAACL 2019。大規模事前学習 + タスクごとの微調整という現在の標準を決定づけた。多数の NLP タスクで「BERT を fine-tune するだけ」で当時の SOTA を更新。
- Kornblith et al., "Do Better ImageNet Models Transfer Better?", CVPR 2019。画像側で fine-tuning と固定特徴の差をデータセット横断で比較し、データが少ないほど事前学習の恩恵が大きいことを定量化。
- Mosbach et al., "On the Stability of Fine-tuning BERT", ICLR 2021。小データでの fine-tuning が不安定 (シードを変えると精度が大きくぶれる) 問題を分析し、学習率・ウォームアップ・エポック数の設計が安定性に効くことを示した。
- Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution", ICLR 2022。全層 fine-tuning が分布外でかえって弱くなる機序を分析し、二段構えの LP-FT を提案。
- Wortsman et al., "WiSE-FT", CVPR 2022。微調整後の重みと事前学習時の重みを線形補間 (重み平均) するだけで分布外ロバスト性を保つ手法。

これらの先に、全体微調整の計算コストを避けるパラメータ効率的微調整 (LoRA, Hu et al. 2021; アダプタ, Houlsby et al. 2019; プロンプトチューニング) があり、いずれも「事前学習モデルを下流に適応させる」という Fine-tuning の精神を引き継いだ派生です。

## 論文の実験結果(定量データ)
Yosinski et al. (2014) は ImageNet を半分ずつのタスク A/B に分け、A の特徴を B へ転移して top-1 accuracy (最も確率の高い予測が正解だった割合) を測りました。結果、転移した層を凍結したまま使うと層が深いほど精度が低下する一方、転移後に全層を fine-tune すると劣化が回復し、B 単独学習を上回るケースが多いと報告。Fine-tuning の優位を直接示しました。

ULMFiT (2018) は IMDb 感情分類、AG News、TREC など複数のテキスト分類で、誤り率 (error rate, 低いほど良い) を当時の最良手法より相対で18〜24%程度削減したと報告。さらに、ラベル100件程度のごく少数データでも、ゼロ学習で1万〜10万件使った場合に匹敵する精度に達することを示し、転移微調整のデータ効率を印象づけました。指標の意味としては、error rate が18%下がるとは「間違いが約2割減る」こと。少数ラベルで多数ラベル相当に届くという結果は、ラベル収集が高価な現場ほど価値が大きい。

BERT (2019) は GLUE ベンチマーク (複数の言語理解タスクの集合。平均スコアが高いほど良い) で当時の最良を約7ポイント更新し、SQuAD v1.1 質問応答で F1 を当時の人間性能に迫る水準まで押し上げました。「巨大モデルを事前学習し、タスクごとに数エポック微調整する」だけで広範なタスクを制覇できることを実証しました。

Kornblith et al. (2019) は16の下流データセットで、ImageNet 精度の高いモデルほど転移先でも強いという相関を確認しつつ、転移先のデータが豊富になるほど事前学習の利得が縮むこと、細粒度タスクでは fine-tuning と固定特徴の差が小さいことを定量化しました。「いつ fine-tuning の価値が大きいか」を条件ごとに切り分けた点が実務的に重要です。

Mosbach et al. (2021) は、小データ微調整での精度のばらつき (シード間の分散) が、適切な学習率と十分なエポック・ウォームアップで大きく縮むことをアブレーションで示しました。つまり「fine-tuning が不安定」という現象の多くは最適化設定の問題であり、設計次第で安定化できるという実務的知見です。

Kumar et al. (2022) は、分布内では fine-tuning > linear probing だが分布外では逆転しうることを複数ベンチで示し、先に凍結で線形ヘッドを収束させてから全体を fine-tune する LP-FT が ID/OOD の双方で最良に近いと報告しました。

## メリット・トレードオフ・限界
メリット
- 少ないデータでも高精度を狙える (良い初期値から始められる)。
- ゼロ学習より速く収束する。
- バックボーンも下流に合わせて調整できるため、凍結より精度の上限が高い (特に分布内・データが十分なとき)。

トレードオフ・限界
- 全体を学習するため、凍結より計算・メモリコストが高い。
- データが少なすぎると過学習し、汎用特徴を壊しかねない (学習率を小さくする・凍結を併用する等の工夫が要る)。
- 転移元と転移先が大きく異なると効果が薄い (negative transfer)。
- 分布外で弱くなりうる: 全層 fine-tuning は OOD でかえって linear probing に劣ることがある (Kumar et al. 2022)。
- 破滅的忘却のリスクがあり、学習率・凍結範囲・エポック数の設計が成否を分ける。
- 未解決の課題: 小データでの不安定性、最適な凍結範囲の自動決定、ドメインギャップが大きい場合の適応など。これらに対し PEFT やドメイン適応の研究が進行中。

## 発展トピック・研究の最前線
- パラメータ効率的微調整 (PEFT): LoRA・アダプタ・プロンプトチューニングで、少数パラメータだけ学習し計算とストレージを節約。巨大モデル時代の微調整の主役。
- 指示チューニング / RLHF: 大規模言語モデルを人間の意図に合わせる微調整の発展形。教師ありの指示微調整 (SFT) の後に人間/AI のフィードバックで整える。
- ロバスト微調整: WiSE-FT (Wortsman et al. 2022) のように、微調整後と事前学習時の重みを補間して分布外性能を保つ手法。重み平均 (model soup) の系譜にもつながる。
- 安定化テクニック: 再初期化・mixout・層ごと学習率減衰 ([Layer-wise LR Decay (LLRD)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0395_layer-wise-lr-decay)) などで小データ微調整の分散を抑える研究。

## さらに学ぶための関連トピック
- [Frozen Backbone](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0355_frozen-backbone)
- [Transfer Learning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0364_transfer-learning)
- [Discriminative / Layer-wise Learning Rates](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0380_discriminative-learning-rates)
- [Gradual Unfreezing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0388_gradual-unfreezing)
- [Layer-wise LR Decay (LLRD)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0395_layer-wise-lr-decay)
