---
title: "linear probing / kNN評価"
---

# linear probing / kNN評価

## ひとことで言うと
linear probing(リニアプロービング)と kNN 評価は、自己教師あり学習で得た特徴が「どれくらい良いか」を測る、研究コミュニティの標準的な物差しです。学習済みのネットワークを凍結(更新せず固定)したまま、その出力特徴の上に「できるだけ単純なもの」だけ(線形層1枚、または近傍探索)を載せて精度を見ます。狙いは、追加で訓練した部分の頑張りではなく、特徴そのものの実力を公平に浮き彫りにすることです。この2つは異なる側面を測るので、論文では両方を併記することが標準になっています。

## 直感的な理解
ラベルなしで特徴を学んでも、学習が終わった時点で「いい特徴が学べたか」は一目では分かりません。対照損失や復元損失という訓練中の数字が下がっても、それが「猫と犬を見分ける」ような下流タスクに役立つ特徴とは限らないからです([次元崩壊](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse) のように、損失は下がっているのに表現が潰れている例さえあります)。

ここで素朴に「じゃあ、その特徴の上にもう一個ネットワークを足してラベルで学習し、精度を見ればいい」と考えると落とし穴があります。足したネットワークが強力だと、元の特徴が平凡でも後付けで帳尻を合わせてしまい、「元の特徴が良かったのか、足した部分が頑張ったのか」が区別できなくなります。テストの問題が難しくても、優秀な家庭教師(強力な追加層)がいれば点が取れてしまうのと同じです。

そこで「家庭教師をできるだけ無能にする」のが linear probing と kNN です。元の特徴は一切いじらず固定し、その上に乗せるのは線形変換だけ、あるいは何も学習しない近傍探索だけ。こうすれば、点が取れたのは「元の特徴がすでに賢かったから」だと言えます。逆に言えば、特徴を作り直せないほど弱い probe を使うことで、特徴の質に診断の焦点を絞っているのです。

## 基礎: 前提となる概念
- 凍結(freeze): ネットワークの重みを学習中に一切更新せず、事前学習時の値のまま使うこと。バックボーン(特徴を取り出す土台ネットワーク、ResNet や ViT)を凍結することで、評価の前後で特徴が変わらないことを保証します。実装上はパラメータの requires_grad を False にし、推論モード(BatchNorm の統計を固定)にします。

- 線形分離可能性(linear separability): 特徴空間で、あるクラスとそれ以外を1枚の超平面(高次元の平面)で切り分けられること。良い特徴ほど、クラスごとに固まってきれいに並んでいるため、線形(直線・平面)で切り分けられます。「線形分類器で高精度が出る = 特徴が既に十分整理されている」という関係です。

- fine-tuning(微調整): 事前学習済みの重みを出発点に、下流タスクのラベルで全層を学習し直すこと。linear probing が「事前学習だけでどれだけ良い特徴ができたか」を測るのに対し、fine-tuning は「事前学習を出発点にどこまで実タスク性能を伸ばせるか」を測ります。両者は別の問いに答えるので、SSL 研究では両方報告するのが普通です。

- top-1 / top-5 精度: top-1 は「最も確信したクラスが正解と一致した割合」、top-5 は「確信度上位5個に正解が含まれる割合」。ImageNet(1000 クラス)ではランダムだと top-1 は 0.1% です。

- メモリバンク(memory bank / feature bank): kNN 評価のために、訓練画像すべての特徴ベクトルとそのラベルをあらかじめ計算して貯めておく配列。テスト時にこの中から近傍を探します。

## 仕組みを詳しく
### linear probing(線形評価)
ImageNet 分類(1000クラス)を例に手順を分解します。

1. SSL で事前学習したバックボーンを用意し、重みを凍結する。
2. ラベル付きの訓練画像をこのバックボーンに通し、特徴ベクトルを取り出す。ResNet-50 なら最後の global average pooling 後の `[2048]` 次元、ViT なら CLS トークンの `[768]` 次元(ViT-B の場合)、といった1枚あたり1本のベクトル。
3. その特徴の上に、線形層1枚だけ(全結合層、`[2048] → [1000]` の行列 W とバイアス)を載せる。これが線形分類器。バックボーンは固定なので、勾配で学習されるのはこの薄い1層だけ。学習率・エポック数・weight decay などはこの1層用にチューニングする。
4. テストデータでの top-1 精度を報告する。これが linear probing 精度。

なぜ線形(1層)に限るのか。多層の複雑な分類器を載せると、その分類器が特徴を実質的に作り直してしまい、元の特徴の質が見えなくなるからです。線形は「特徴を足し引きして閾値を引く」だけなので、特徴が既に線形分離可能なほど整理されているかを純粋に測れます。

ASCII で構造を表すと:

```
[画像] → [凍結バックボーン] → 特徴 h [2048] → [線形層 W (学習)] → ロジット [1000] → クラス
              ↑ 更新しない               ↑ ここだけ学習
```

### kNN 評価(k近傍評価)
kNN(k-Nearest Neighbors)は、学習パラメータをゼロにした究極の評価です。

1. バックボーンを凍結し、訓練画像すべての特徴ベクトルを計算してメモリバンクに貯める(各ベクトルにクラスラベルを紐づける)。
2. テスト画像1枚の特徴を計算し、メモリバンクの中から距離が近い上位 k 個(DINO では k=20 が標準)を探す。距離は通常コサイン類似度(ベクトルの向きの近さ)。
3. その k 個のラベルを、類似度を温度 τ で重み付けして投票し、テスト画像のクラスを予測する(単純多数決ではなく類似度重み付き投票が一般的)。
4. 精度を報告する。これが kNN 精度。

kNN は分類器の訓練すら不要なので、追加学習の影響をほぼ完全に排除できます。「似た画像が特徴空間で本当に近くに集まっているか」を直接見られます。線形分類器のハイパーパラメータ(学習率・エポック数)に結果が左右されないので、再現性の面でも好まれます。

### 両者の使い分け
- どちらもバックボーンを凍結する点が共通で、「特徴を作り直さない」公平性を担保する。
- linear probing は最適化を1層分含むぶん、特徴を線形に「読み出せる」情報を最大限引き出す。kNN は最適化ゼロで、特徴空間の素の距離構造を反映する。
- kNN と linear probing の差が小さいほど「特徴空間の素の距離が、すでにクラスをきれいに分けている」証拠になる。DINO はこの差が小さいことを ViT 特徴の質の高さの根拠に使いました。
- 注意: linear probing 精度が高い = fine-tuning 後も最強、とは限らない。マスク再構成系(MAE など)は linear probing は控えめでも fine-tuning すると非常に強い、という逆転が起きます。1指標で優劣を断じないことが重要です。

## 手法の系譜と主要論文
- He, Fan, Wu, Xie, Girshick, "Momentum Contrast (MoCo)"(CVPR 2020, FAIR, arXiv:1911.05722)。ImageNet 上の linear probing を SSL の標準評価として前面に据え、凍結特徴 + 線形分類器の精度で手法を比較しました。動機は、特徴の転移性を公平に測る共通土俵が必要だったこと。
- Chen, Kornblith, Norouzi, Hinton, "SimCLR"(ICML 2020, Google, arXiv:2002.05709)。ImageNet linear evaluation を主要指標に採用し、「[projection head](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1082_projection-head) の前(`h`)で評価すべき、後(`z`)で評価すると精度が落ちる」というプロトコルの細部まで議論しました。以後の SSL 論文がほぼこの土俵で比較するようになり、結果の比較可能性が高まりました。半教師あり評価(ラベル 1% / 10% だけで fine-tune)も導入しました。
- Wu, Xiong, Yu, Lin, "Instance Discrimination / Non-parametric softmax"(CVPR 2018, arXiv:1805.01978)。メモリバンクと kNN 評価の原型を提示し、後の DINO などが採用する weighted kNN の土台を作りました。
- Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin, "DINO"(ICCV 2021, Meta, arXiv:2104.14294)。kNN 評価を前面に出し、ViT を SSL で学習した凍結特徴で k=20 の近傍投票だけで高精度が出ることを示しました。「学習済み特徴が分類器なしでもクラスを分けられるほど整っている」ことの強い証拠として使われました。
- He, Chen, Xie, Li, Dollár, Girshick, "Masked Autoencoders (MAE)"(CVPR 2022, FAIR, arXiv:2111.06377)。linear probing と fine-tuning が乖離する代表例で、評価プロトコルの選択がいかに結論を左右するかを浮き彫りにしました。

## 論文の実験結果(定量データ)
指標は ImageNet-1K(約128万枚、1000クラス)の top-1 精度が中心です。

- SimCLR(ResNet-50, 大バッチ・長エポック)は linear probing でおよそ 69〜76% を報告(モデル幅・エポックで変動)。標準幅 ResNet-50 で約 69.3%、幅4倍(ResNet-50 4x)では 76% 超。
- MoCo v2(ResNet-50)は linear probing で約 71.1%。[projection head](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1082_projection-head) と強い拡張の導入で MoCo v1 の約 60.6% から 10 ポイント以上改善しました。
- DINO(ViT-S/16)は linear probing で約 77.0%、kNN(k=20)でも約 74.5% を報告。kNN と linear probing の差が 2.5 ポイント程度と小さいことが「特徴が非常によく整っている」証拠とされました。同じ DINO でも ResNet-50 では kNN(約 67%)と linear(約 75%)の差が大きく、ViT × DINO の特徴の質の高さが際立ちました。
- アブレーションの古典例は SimCLR の projection head 実験です。[projection head](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1082_projection-head) を入れて手前の `h` を使うと、入れない場合(`h` で直接損失)より linear probing 精度が報告でおよそ 10 ポイント以上改善しました。逆に projection 後の `z` で評価すると `h` より大きく(数ポイント〜十数ポイント)落ちます。これは「どの層の特徴を評価に使うか」が結果を左右することを示します。
- MAE の例: linear probing では ViT-B で約 68%(同条件の対照学習系より低い)なのに、fine-tuning では約 83.6%、ViT-L では約 85.9% に達し、当時の対照学習系を上回りました。「linear probing が弱い ≠ 表現が悪い」を端的に示すアブレーションです。MAE のように特徴が「線形には取り出しにくいが fine-tune で開花する」表現は、probe の弱さゆえに過小評価されます。

数値の意味として、SSL では linear probing top-1 の 1〜2 ポイント差が論文の主張を支える重要な差です。1000 クラスの難問で 0.1% から 70% 台まで来ている時点で、1ポイントの差は判定が覆る画像が1万枚規模に及びます。同時に、評価プロトコルを変えると順位が入れ替わりうるため、複数指標の併記が標準になっています。

## メリット・トレードオフ・限界
メリット
- バックボーンを凍結するので、特徴そのものの質を公平に測れる。追加層の頑張りに汚染されない。
- kNN は学習不要・ハイパーパラメータほぼゼロで、再現性が高く、特徴空間の素の構造を直接反映する。
- 手法間の比較が標準化され、論文どうしの数字を突き合わせやすい。

トレードオフ・限界
- 主に分類タスク向けで、物体検出・セグメンテーション・軌跡予測のような構造的タスクでの実力は直接測れない。
- linear probing 精度が高い = fine-tuning 後も最強、とは限らない(MAE が反例)。1指標での断定は誤りを招く。
- linear probing は probe の最適化設定(学習率・エポック・正則化)に依然として敏感で、これを甘く設定すると不当に低く出ることがある。論文によって probe レシピが違うと数字が直接比較できない。
- kNN は訓練特徴を全部メモリに保持するため、データが巨大だとメモリと計算が重い(近似最近傍探索 ANN で緩和はできるが、近似誤差が入る)。
- 評価に使う層(projection の前か後か、ViT の CLS トークンかパッチ平均か)で結果が変わるため、プロトコルの明記が不可欠。

## 発展トピック・研究の最前線
分類精度に依存しない評価への関心が高まっています。ラベルを使わず特徴行列の特異値分布から有効ランクを計算する RankMe(Garrido et al., ICML 2023, arXiv:2210.02885)は、有効ランクが下流性能と強く相関することを利用してモデル選択に使え、「ラベルなしで良いチェックポイントを選ぶ」手段として注目されました([次元崩壊](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse) の裏返しの指標です)。

また、検出やセグメンテーション、深度推定などへの転移を測る「密な予測タスクでの評価」、複数データセットへの転移を測る VTAB(Visual Task Adaptation Benchmark, Zhai et al. 2019)、few-shot 性能、ロバスト性(ImageNet-C / ImageNet-A など分布シフト下の精度)、公平性まで含めた多面的な評価スイートが整備されつつあります。DINO 系では、注意マップが教師なしで物体をセグメントする創発的性質を、ラベルなしセグメンテーション精度(Jaccard 指標)で測る評価も使われます。自動運転のような構造的タスクでは、凍結特徴の上に軽量なタスクヘッドを載せて性能を見る「frozen feature + task probe」という linear probing の発想の拡張(linear segmentation probe、frozen-backbone detection など)が、表現の質を実タスクに近い形で点検する手段として有効です。「単一スカラーで表現の良さを測れるのか」という問い自体が、今も活発に議論されている未解決の論点です。

## さらに学ぶための関連トピック
- [MoCo v3 (ViT対照学習)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0076_mocov3-vit)
- [次元崩壊 (dimensional collapse)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse)
- [projection head (投影ヘッド)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1082_projection-head)
- [data2vec](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0041_data2vec)
- [ConvNeXt V2 / FCMAE](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0039_convnextv2-fcmae)
