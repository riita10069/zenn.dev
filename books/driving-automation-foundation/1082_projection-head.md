---
title: "projection head (投影ヘッド)"
---

# projection head (投影ヘッド)

## ひとことで言うと
projection head(投影ヘッド、projector とも呼ぶ)は、対照学習で「損失を計算する直前」に置く小さな多層パーセプトロン(MLP、全結合層を数枚重ねた小ネットワーク)です。バックボーンが出した特徴を、いったん別の空間に写してから損失を計算します。こうすると、学習の都合で歪みやすい部分を投影ヘッド側に押し込め、バックボーン本体の特徴は汎用的なまま守れる、というのが SimCLR の発見です。学習が終わったら投影ヘッドは捨て、その手前の特徴を使います。実装上はほんの数行の MLP ですが、これを入れるだけで下流精度が10ポイント以上跳ね上がる、対照学習の必須部品です。

## 直感的な理解
対照学習では「同じ画像を色違い・切り抜き違いに加工した2枚を近づける」よう学びます。これは裏を返すと「色や位置の違いは無視せよ」とネットワークに強く命令していることになります。

ここに矛盾が生まれます。色や位置は、確かに「同一画像の判定」には邪魔ですが、下流タスクではしばしば重要な情報です。信号機が赤か青か(色)、歩行者が右にいるか左にいるか(位置)は、捨ててはいけない情報です。もしバックボーンの出す特徴を対照損失で直接引っぱると、特徴が「色も位置も捨てる」方向に最適化されすぎて、本来役立つ情報まで失われます。

projection head は、この矛盾を吸収する緩衝材です。「色や位置を無視せよ」という命令を投影ヘッドが一身に引き受け、バックボーン本体には色や位置の情報を残させます。下流で使うのはバックボーン本体の特徴なので、汎用性が守られる、という仕組みです。学習が終われば投影ヘッドは役目を終え、捨てられます。家庭で言えば、客に出す前に汚れ仕事をする下処理場のようなもので、本体(食材=表現)はきれいなまま、捨てる空間(z)で味を整える、というイメージです。

## 基礎: 前提となる概念
- バックボーン(backbone): 画像から特徴を取り出す土台ネットワーク(ResNet、ViT など)。最終的に下流タスクで使いたいのはこの出力。

- 表現 / representation: バックボーンが出す特徴ベクトル `h`。SimCLR では ResNet-50 の global average pooling 後の `[2048]` 次元など。

- データ拡張不変性(augmentation invariance): 対照学習の目的そのもの。「同じ画像の異なる加工版を区別しない(同じ特徴を出す)」性質。これを強く要求しすぎると下流に有用な情報まで消える、というのが projection head 導入の動機。

- MLP / ReLU / BatchNorm: MLP は全結合層を重ねた小ネットワーク。ReLU は `max(0, x)`、つまり負の値を 0 にする非線形関数で、これがあるから MLP は単なる行列1個より複雑な変換ができる。BatchNorm はバッチ内で各成分を平均0・分散1に正規化する層で、学習を安定させ崩壊を防ぐ補助になる。projection head が非線形であることが、後述の「情報の役割分担」を可能にする鍵。

- prediction head(予測ヘッド): BYOL や [MoCo v3](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0076_mocov3-vit) で、生徒側だけにもう1段足す MLP。projection head とは別物で、生徒の出力を教師の出力に「予測しに行く」非対称性を作り、負例なしでの崩壊回避に効く。

## 仕組みを詳しく
### 構造
SimCLR の構成を例に動作を追います。

1. 入力画像の2ビューをそれぞれバックボーン(ResNet-50)に通し、表現 `h` を得る。`h` は `[2048]` 次元。
2. その `h` を projection head に通す。SimCLR の projection head は「全結合 → ReLU → 全結合」の2層 MLP で、出力 `z` は `[128]` 次元に縮む。式で書くと `z = W2 · ReLU(W1 · h)`。
3. 対照損失(InfoNCE)は `z` どうしで計算する。`h` ではなく `z` で損失を取るのがポイント。損失計算の前に `z` は L2 正規化される。
4. 学習後、下流タスクや [linear probing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0759_linear-probing-evaluation) で使うのは `z` ではなく `h`(projection head の手前)。projection head は捨てる。

ASCII で表すと:

```
[2ビュー] → [バックボーン] → h [2048]  ← 下流で使うのはこっち(汎用情報を保持)
                                |
                          [projection head]  (色・位置を捨てる汚れ仕事を引き受ける)
                                |
                               z [128]   ← 対照損失はこっちで計算
```

### なぜこれで効くか
直感はこうです。損失が要求する「色や位置を無視する不変性」は、`h → z` の変換(projection head)が引き受けます。projection head は非線形なので、「`h` には下流に有用な情報を保ったまま、損失計算用の `z` ではその情報を意図的に落とす」という器用な役割分担ができます。線形変換だけだと情報を「捨てる」操作と「保つ」操作を分離しにくいのですが(線形写像は核空間が単純に決まる)、非線形だからこそ層の出力(`z`)で情報を潰しつつ入力(`h`)には残す、という非対称が成立します。結果として `h` には色・位置などが残り、汎用性が保たれます。

別の見方として、Jing らの [次元崩壊](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse) 解析が与えた説明があります。対照損失で `z` を学習すると、`z` の空間で次元崩壊が起きやすいのですが、projection head が緩衝材になることで、その崩壊が `h` にまで遡及するのを食い止めます。つまり projection head は「崩壊を自分の出力側に局所化し、バックボーンの表現を崩壊から守る」役割を持つ、という解釈です。これは「なぜ手前の `h` が良いか」の理論的な裏づけになっています。

### 設計のバリエーション
- 層数: SimCLR は2層 MLP。SimCLR v2 や BYOL では3層に深くするとさらに良いと報告。深いほど損失の歪みを多く吸収でき、`h` の汚染が減る。
- 幅・出力次元: 隠れ次元はバックボーン出力と同程度(2048)、出力 `z` は 128〜256 が一般的。BYOL は隠れ 4096・出力 256 と広めに取る。
- 正規化: 各層の後に BatchNorm を入れる構成が一般的。BYOL では BatchNorm の有無が崩壊回避に効くという議論があり(BN がバッチ内の暗黙の負例として働くという説と、それを否定する追試の応酬があった)、後に Group Norm でも崩壊しないことが示されて決着しました。
- prediction head との関係: BYOL や MoCo v3 では、生徒側だけにもう1段の MLP(prediction head)を足す。projection head は両側(生徒・教師)に、prediction head は生徒側だけ、という非対称が、負例なしでの崩壊回避の鍵の一つ。
- 崩壊との関係: projection head の設計は次元崩壊と深く関わる。Jing らの分析では、projection head が線形だと表現が崩壊しやすく、非線形にすると崩壊が緩和される。projection head は「データ拡張に固有のノイズ次元」を吸収し、バックボーンの次元が潰れるのを守っている、という解釈が示されている。

## 手法の系譜と主要論文
- Chen, Kornblith, Norouzi, Hinton, "A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)"(ICML 2020, Google Brain, arXiv:2002.05709)。projection head を提案し、「損失は projection 後の `z` で取り、下流では projection 前の `h` を使う」べきことを実験で示しました。動機は、データ拡張不変性を要求する対照損失が表現から有用情報を消すのを避けること。効果は linear probing 精度の大幅向上で、以後ほぼ全ての対照学習が projection head を採用しました。
- Chen, Kornblith, Swersky, Norouzi, Hinton, "Big Self-Supervised Models are Strong Semi-Supervised Learners (SimCLR v2)"(NeurIPS 2020, arXiv:2006.10029)。projection head を3層に深くし、下流では「全部捨てる」のでなく「途中の層(1層目の出力)から取る」と良い場合があることを示しました。深いヘッドほど損失の歪みを吸収できるためです。
- Grill et al., "Bootstrap Your Own Latent (BYOL)"(NeurIPS 2020, DeepMind, arXiv:2006.07733)と Chen, Xie, He, "MoCo v3"(ICCV 2021, FAIR, arXiv:2104.02057)は、projection head に加えて生徒側だけの prediction head を導入。負例なしでも崩壊を防ぐ非対称構造として projection head の役割を拡張しました。
- Jing, Vincent, LeCun, Tian, "Understanding Dimensional Collapse"(ICLR 2022, Meta AI, arXiv:2110.09348)は、projection head が崩壊を吸収する役割を理論・実験で分析し、極端には projection head なしでも一部次元だけで対照学習する DirectCLR が成立することを示しました。
- Gupta, Ajanthan, van den Hengel, Gould ほかの一連の解析("Understanding the Role of the Projector in SSL" 系, 2022)は、projection head が「拡張に固有で下流に無関係な次元」を選択的に吸収していることを、表現の情報量の観点から定量化しました。

系譜としては「projection head の発明(SimCLR)→ 深化と中間層利用(SimCLR v2)→ prediction head による非対称化と負例なし化(BYOL/MoCo v3)→ なぜ手前が良いかの理論解明と最小化(Jing 2022 / DirectCLR、Gupta ら)」と読めます。

## 論文の実験結果(定量データ)
指標は ImageNet-1K の [linear probing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0759_linear-probing-evaluation) top-1 精度。

- SimCLR の中心的アブレーション: projection head を入れて手前の `h` を使うと、入れない(`h` で直接損失を取る、すなわち恒等写像の projector)場合より linear probing 精度が報告でおよそ 10 ポイント以上改善しました。具体的には、projection head なしで `h` を直接対照損失にかけると線形評価が大きく落ち、2層 MLP の projection head を挟んで `h` を使うと大幅に回復します。
- 評価する層の比較: 同じ学習済みモデルでも、projection 後の `z`(128次元)で linear probing すると、projection 前の `h`(2048次元)より精度が明確に低くなりました(SimCLR の報告で十数ポイント規模の差)。これが「最良の表現はヘッドの手前にある」という有名な(やや直感に反する)発見です。`z` は損失のために情報を削ぎ落とした後なので当然とも言えます。
- 層数のアブレーション: SimCLR では projection head を「なし(線形相当)→ 1層 → 2層 → 3層」と深くすると linear probing 精度が段階的に上がる傾向が示され、SimCLR v2 はこれを根拠に3層を採用しました。SimCLR v2 では3層ヘッドの「2層目の出力」から評価すると最良という細かい知見も得ています。
- 崩壊との関係: Jing らは projection head を線形にすると固有値スペクトルが断崖状になり([次元崩壊](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse))、非線形にすると断崖が緩むことを実測しました。DirectCLR(projection head なしで先頭の一部次元のみ対照学習)は projection head 付き SimCLR とほぼ同等の linear probing 精度を達成し、「projection head の本質的役割は崩壊回避」という主張を裏づけました。

数値の意味として、SSL では linear probing top-1 の 1〜2 ポイント差が手法の優劣を分けるため、projection head による 10 ポイント超の改善は決定的でした。これが「ほぼ全ての対照学習が projection head を標準採用する」理由です。

## メリット・トレードオフ・限界
メリット
- 対照損失が要求する不変性を吸収し、バックボーン本体の表現の汎用性を守れる。
- linear probing 精度を大きく改善する(SimCLR で 10 ポイント超の実証)。
- 学習後は捨てられるので、推論時のコストはゼロ。
- 非対称な prediction head と組み合わせると、負例なしでの崩壊回避にも寄与する。

トレードオフ・限界
- 層数・幅・正規化(BatchNorm の有無)・出力次元というハイパーパラメータが増え、調整の手間がかかる。
- 「最良の表現がヘッドの手前にある」挙動は直感に反し、評価時にどの層を使うか([プロトコルの設計](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0759_linear-probing-evaluation))に注意が要る。間違った層で評価すると不当に低い数字が出る。
- なぜ手前が良いかの完全な理論説明は当初なく、後続の解析(ノイズ次元の吸収・崩壊回避)で部分的に解明された段階。深い非線形ネットでの厳密な保証はまだ限定的。
- 対照学習以外(純粋な教師あり学習など)では効果が薄く、用途が SSL 寄りに限定的。教師あり対照学習(SupCon)などでは効くが、通常の交差エントロピー学習では恩恵が小さい。

## 発展トピック・研究の最前線
projection head の「損失を取る空間と下流で使う空間を分ける」という発想は、対照学習を超えて広がっています。マスク再構成系([FCMAE](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0039_convnextv2-fcmae)、MAE など)では再構成のためのデコーダがこの役割に近く、軽量なデコーダが「ピクセル復元という下流に無関係なタスク」を引き受けてエンコーダ表現を守ります。蒸留系([data2vec](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0041_data2vec)、DINO)では予測ヘッドが教師のターゲットに合わせる緩衝材になります。理論面では、projection head が「データ拡張に固有で下流に無関係なノイズ次元」を選択的に吸収しているという解釈が進み、なぜ手前の `h` が良いかが情報量の観点から定量的に説明されつつあります。

実装上の発展として、下流タスクに合わせて projection head の出力次元や深さを調整する、複数の補助損失それぞれに別々のヘッドを割り当てて本体表現を守る(マルチヘッド設計)、といった工夫も一般化しています。大規模化した DINOv2 では projector の設計(iBOT ヘッドと DINO ヘッドの分離、重み共有の有無)が性能に効くことが報告されています。「損失で本体を直接歪めず、専用の射影を一段かませて本体表現を保護する」という設計原則は、SSL に限らず補助損失やマルチタスク学習を扱う多くの場面で応用が効く一般的な知恵です。

## さらに学ぶための関連トピック
- [MoCo v3 (ViT対照学習)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0076_mocov3-vit)
- [次元崩壊 (dimensional collapse)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse)
- [linear probing / kNN評価](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0759_linear-probing-evaluation)
- [data2vec](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0041_data2vec)
- [ConvNeXt V2 / FCMAE](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0039_convnextv2-fcmae)
