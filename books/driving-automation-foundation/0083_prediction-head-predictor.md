---
title: "predictor (予測ヘッド非対称性)"
---

# predictor (予測ヘッド非対称性)

## ひとことで言うと
自己教師あり学習で「片側の枝にだけ小さな MLP(多層パーセプトロン、数層の全結合ネットワーク)を追加して、もう片側を予測させる」という非対称な構造のことです。一見ささいな部品ですが、これがあるおかげで「全部の出力が同じ値に潰れる崩壊」を、負例を一切使わずに防げます。BYOL と SimSiam の中核設計であり、「負例は崩壊回避に必須」という当時の常識を覆した発見の鍵です。

## 直感的な理解
対照学習(→ [InfoNCE / NT-Xent ロス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0061_info-nce-loss))は「正例を近づけ、負例から離す」ことで崩壊を防いでいました。負例から離す圧力が「潰れない」ための支えだったわけです。では負例をなくしたらどうなるか。普通に考えれば、正例ペアを近づけるだけ(離す相手なし)なら、ネットワークは全画像を同じ定数ベクトルに潰してしまい、損失ゼロという自明でずるい解に落ちるはずです。

ところが BYOL は「負例ゼロでも崩壊しない」と示しました。からくりが predictor です。2つの枝を完全に対称(同じ構造、同じ目標)にすると、互いに歩み寄って一瞬で潰れます。そこで片方の枝にだけ predictor という小さな変換を挟み、「予測する側」と「予測される側(目標)」という非対称な関係を作る。さらに目標側には勾配を流さない(stop-gradient)。すると「目標は固定し、予測側だけが目標に合わせに行く」という片方向の運動になり、両者が手を取り合って潰れる経路がふさがれます。

たとえると、自分(オンライン枝)が「相手(ターゲット枝)が次に何を言うか」を当てるゲームです。相手は今この瞬間は固定された答えを持っていて(stop-gradient)、自分はその答えを predictor を通して予測する。もし両者が「いつも『あ』とだけ言う」に合意すれば確かに当たりますが、predictor が間に挟まり、しかも目標が少しずつ動く(EMA)ことで、その安易な合意に滑り落ちにくくなっている、というのが直感です。

## 基礎: 前提となる概念
崩壊(collapse): 全出力が同一の定数ベクトルに潰れる失敗です。representational collapse とも呼ばれ、どんな入力にも同じ特徴を返すので何も区別できません。SSL 設計の最大の敵です。崩壊が起きると損失は最小(正例が完全一致)ですが、特徴の標準偏差がゼロに近づくので、これを監視すれば崩壊を検知できます。

MLP / 投影ヘッド / 予測ヘッド: MLP は全結合層を数層重ねたネットワークです。SSL では、エンコーダ(ResNet や ViT)の後ろに「投影ヘッド(projection head)」という MLP を置き、損失をその出力で計算するのが標準(→ [InfoNCE / NT-Xent ロス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0061_info-nce-loss) の SimCLR が確立)。BYOL/SimSiam はさらにその先の片側にだけ「予測ヘッド(predictor)」という別の MLP を追加します。投影ヘッドは両枝にあり、予測ヘッドは片枝だけにある、という二重構造です。

stop-gradient: 「この値を目標として固定し、ここからは勾配(学習の更新信号)を逆伝播させない」操作です。PyTorch の detach に相当します。SSL では「目標側を動かさない」ために使われ、これがないと両枝が同時に動いて即崩壊します。

EMA(指数移動平均): 目標側のネットワークを、学習側の重みでほんの少しずつ更新する方法(→ [momentum encoder / EMA target](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0078_momentum-encoder-ema))。BYOL は目標枝を EMA で動かしますが、SimSiam は EMA を使わず、ターゲット枝はオンライン枝とパラメータを共有しつつ stop-gradient だけで切り離します。

## 仕組みを詳しく
構造を順に見ます。1枚の画像から2つのビュー(拡張違いの2枚)を作るところは対照学習と同じです。

オンライン枝(学習する側、student): エンコーダ f → 投影ヘッド g → さらに predictor q。出力 p1 = q(g(f(view1)))。
ターゲット枝(目標を与える側、teacher): エンコーダ f' → 投影ヘッド g'。出力 z2 = g'(f'(view2))。predictor は付けません。ここが非対称です。

損失: オンライン枝の予測 p1 で、ターゲット枝の出力 z2 を当てさせます。両者を L2 正規化し、負のコサイン類似度(向きの近さの符号反転、近いほど小さい)を最小化します。

```
L = - cos(p1, stopgrad(z2))   と、ビューを入れ替えた  - cos(p2, stopgrad(z1))  の平均
```

重要な2点。

1. predictor q は片側にだけある。ターゲット枝には q がありません。だから両枝は対称ではなく、「オンライン枝がターゲット枝を予測する」という一方向の関係になります。直感的には、predictor が「ターゲットの平均的な振る舞い」を表現するよう学習し、エンコーダ側まで潰す必要がなくなる、と説明されます。理論的には Tian et al.(DirectPred, ICML 2021)が、線形 predictor の学習ダイナミクスを解析し、predictor の重み行列が「特徴の共分散行列の固有空間」に徐々に整列していくことで崩壊解が不安定化することを示しました。彼らは predictor を勾配で学習せず、特徴の共分散行列の固有値分解から直接設定する DirectPred でも崩壊回避できると実証し、predictor が果たす役割を理論的に裏づけました。

2. stop-gradient。stopgrad(z2) は「z2 を目標値として固定し、勾配を流さない」操作です。これを外すと両枝が互いに歩み寄って一瞬で同じ定数に潰れます。stop-gradient は片方向性を作る装置です。SimSiam の論文は、この学習を期待値最大化(EM, Expectation-Maximization)に似た交互最適化として解釈しました。stop-gradient された目標は「現在の最良の擬似ラベル」に相当し、オンライン枝はそれに合わせ、次のステップで擬似ラベルが更新される、という交互更新だと見なせるという説明です。

SimSiam の切り分け実験(アブレーション)が分かりやすいです。

- predictor あり + stop-gradient あり: 崩壊しない、性能良好。
- predictor を恒等写像(何もしない)に置き換える: 崩壊する。
- stop-gradient を外す: 崩壊する(損失は急速にゼロに落ちるが、特徴の標準偏差がゼロに潰れる)。

つまり predictor の非対称性と stop-gradient はセットで効きます。BYOL はこれに加えてターゲット枝を EMA で更新しますが、SimSiam は EMA なし(ターゲット枝はオンライン枝とパラメータ共有、ただし stop-gradient で勾配は止める)でも成立することを示しました。

形状の例: 投影ヘッド出力 z が [B, 2048]、predictor が 2048 → 512 → 2048 の MLP(中間でボトルネック)で p も [B, 2048]。損失はこの p と z のコサイン類似度。predictor の中間次元(ここでは 512)というボトルネックや、内部のバッチ正規化(batch normalization、ミニバッチ単位で出力を標準化する層)の有無が性能と崩壊回避に効く、繊細な部品です。なぜボトルネックが効くかは完全には解明されていませんが、predictor が過度に表現力を持つとオンライン枝が「目標を丸暗記」して崩壊解に近づきやすくなる、という説明がされます。

BYOL の崩壊回避をめぐっては初期に混乱がありました。当初「投影ヘッド内のバッチ正規化がバッチ統計を通じて暗黙に負例の役割を果たしている(他サンプルと比較している)」という仮説が提示されましたが、Richemond et al.(2020)が「バッチ正規化を group normalization 等の非バッチ統計に置き換えても BYOL は動く」と示し、バッチ正規化が崩壊回避の本質ではないことが確認されました。本質は predictor 非対称性と stop-gradient(と EMA)にある、という理解に落ち着きました。

## 手法の系譜と主要論文
BYOL(Grill, Strub, Altché, Tallec et al., NeurIPS 2020, "Bootstrap Your Own Latent", arXiv:2006.07733, DeepMind)。提案: オンライン枝に predictor を付け、ターゲット枝はオンライン枝の EMA で更新し、負例なしで正例予測のみ学習。動機: 負例集めのコスト(大バッチ・メモリバンク)を不要にしたい。新規性: 負例ゼロで対照学習に並ぶ・超える精度を達成し、SSL の前提を覆した。当初は「なぜ崩壊しないか」が経験的にしか分からず、EMA・predictor・バッチ正規化のどれが本質か議論を呼びました。

SimSiam(Chen & He, CVPR 2021, "Exploring Simple Siamese Representation Learning", arXiv:2011.10566, Facebook AI Research, Xinlei Chen と Kaiming He)。提案: BYOL から EMA を取り除いても、predictor + stop-gradient だけで崩壊回避できると実証。新規性: 崩壊回避の本質的要因を切り分け、stop-gradient が鍵だと特定。仕組みを大幅に単純化し、EM 交互最適化という解釈を与えました。

DirectPred(Tian, Chen, Ganguli, ICML 2021, arXiv:2102.06810, Facebook AI Research)。線形ネットワークでの学習ダイナミクスを解析し、predictor が崩壊を防ぐメカニズムを理論的に説明。predictor を学習せず特徴の共分散から直接構成しても機能することを示し、経験則だった predictor 設計に理論的根拠を与えました。固有値の成長スピードの差が崩壊・非崩壊を分けるという描像です。

関連系統: MoCo(→ [momentum encoder / EMA target](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0078_momentum-encoder-ema))や SwAV(→ [SwAV (Swapping Assignments between Views)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0095_swav_note))は崩壊回避の別アプローチ(負例/クラスタ等分配)で、predictor 非対称性はそれらと並ぶ「負例なし派」の解です。別系統として VICReg(Bardes, Ponce, LeCun, 2022)や Barlow Twins(Zbontar et al., ICML 2021)は、predictor の代わりに「分散を一定以上に保つ項(variance)」「特徴次元間の冗長性・相関を減らす項(covariance/redundancy reduction)」といった正則化で崩壊を防ぎます。崩壊回避には「非対称性(predictor + stop-gradient)」「負例(InfoNCE)」「正則化(分散・冗長性)」「等分配(Sinkhorn)」という大きく4つの流儀がある、と整理できます。

## 論文の実験結果(定量データ)
評価指標は線形評価(エンコーダ凍結、線形分類器だけ ImageNet ラベルで学習、Top-1 精度)です(→ [InfoNCE / NT-Xent ロス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0061_info-nce-loss) 参照)。

BYOL(ResNet-50、ImageNet 線形評価、報告による)。Top-1 約 74.3%。負例を使う SimCLR(約 69.3%)を約5ポイント上回り、負例ありの MoCo v2(約 71.1%)も上回りました。負例ゼロでこの数字は当時衝撃的でした。さらに大きな ResNet(ResNet-200 2x など)では約 79% に達し、教師あり学習に肉薄。

BYOL のバッチサイズ頑健性: SimCLR は小バッチで負例不足から大きく劣化しますが、BYOL はバッチ 256 から 4096 まで精度がほぼ変わらず頑健でした(報告では極端な小バッチでようやく緩やかに低下)。負例数に依存しない設計の利点が定量的に確認されました。データ拡張への依存も SimCLR より緩やかで、color distortion を弱めたときの劣化が小さいと報告されています。

SimSiam(ResNet-50、ImageNet 線形評価、報告による)。Top-1 約 71.3%(100 エポック程度の比較的短い学習でも他手法と競合)。EMA を外しても崩壊せず動くことを示した意義が大きい。SimSiam の決定的アブレーション:

- stop-gradient あり: 学習が安定し Top-1 約 67〜71%(設定による)。
- stop-gradient なし: 損失は急落するが特徴の標準偏差(L2 正規化後の各次元の std)がゼロへ、k-NN 精度がランダム同然(約 0.1%、1000 クラスなので偶然当たる確率)に崩壊。

このコントラストが「stop-gradient が崩壊回避の必須要素」であることを明快に示しました。predictor を恒等写像にしても同様に崩壊します。predictor 内のバッチ正規化を外すと性能が大きく落ち、predictor の学習率を本体より高めに保つ(固定学習率にする)工夫が有効、といった繊細なハイパーパラメータ感度も報告されています。バッチサイズについては SimSiam は 256 程度の標準的なバッチで動き、SimCLR のような巨大バッチを必要としません。

## メリット・トレードオフ・限界
メリット
- 負例が一切不要。大バッチ・メモリバンクを避けられ、バッチサイズへの依存も小さい(BYOL)。
- 仕組みが比較的単純(片側 MLP + stop-gradient)。SimSiam では EMA すら不要にでき、実装がさらに軽い。
- 線形評価で対照学習を上回り、転移性能も高い。
- データ拡張への依存が対照学習よりやや緩やか。

トレードオフと限界
- なぜ崩壊しないかが直感的でなく、設計を間違えると即崩壊する(predictor を外す/stop-gradient を外すと崩壊)。設計の余地が狭い。
- predictor の中間次元・バッチ正規化・学習率など、ハイパーパラメータに敏感。バッチ正規化の役割をめぐって初期に混乱があった。
- 損失が「正例の一致」だけなので、uniformity(特徴を散らす)を陽に課さない。表現の均一性は副作用的に得られるが、対照学習ほど明示的でない。
- 理論的理解は線形ネットワークまでで、深い非線形ネットワークでの崩壊回避の完全な保証はない。
- 単画像ベースが基本で、時系列・行動・物理の情報は別途組み込みが必要。

未解決の研究課題として、非線形ネットワークでの崩壊回避の完全な理論的理解、predictor 設計の原理的な決め方、マルチモーダルや動画への自然な拡張が挙げられます。

## 発展トピック・研究の最前線
predictor 非対称性の発想は、画像 SSL を超えて広がっています。動画 SSL では「現在フレームの特徴から未来フレームの特徴を predictor で予測し、目標側は stop-gradient で固定」という構成が自然で、これは現在から未来を予測する世界モデル的な学習(→ [行動条件付き潜在予測 (counterfactual rollout)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0001_action-conditioned-prediction))と地続きです。実際 LeCun らの提唱する JEPA(Joint Embedding Predictive Architecture)は、まさに「特徴空間で predictor が一部から残りを予測し、目標側は EMA・stop-gradient で安定化」という BYOL/SimSiam の系譜を一般化したもので、I-JEPA(画像、Assran et al. 2023)・V-JEPA(動画、2024)として展開されています。これらは「画素の再構成(→ [MAE (Masked Autoencoders)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0070_mae_note))」ではなく「特徴空間での予測」を行う点が特徴で、不要な低レベル詳細の再現を避けられると主張されています。崩壊回避の流儀(非対称性・負例・正則化・等分配)を組み合わせる研究も進み、どの状況でどれが最適かはまだ open question です。目標側の作り方という観点では EMA(→ [momentum encoder / EMA target](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0078_momentum-encoder-ema))との組み合わせが重要な設計軸になります。

## さらに学ぶための関連トピック
- [momentum encoder / EMA target](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0078_momentum-encoder-ema)
- [InfoNCE / NT-Xent ロス](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0061_info-nce-loss)
- [SwAV (Swapping Assignments between Views)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0095_swav_note)
- [行動条件付き潜在予測 (counterfactual rollout)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0001_action-conditioned-prediction)
- [MAE (Masked Autoencoders)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0070_mae_note)
