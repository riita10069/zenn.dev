---
title: "MoCo v3 (ViT対照学習)"
---

# MoCo v3 (ViT対照学習)

## ひとことで言うと
MoCo v3 は、Vision Transformer(ViT、画像を小さなパッチに切り分けて Transformer で処理する画像認識モデル)を「対照学習(contrastive learning)」というラベルなしの自己教師あり学習で訓練するための研究です。当時、ViT を対照学習させると訓練が不安定で、精度が学習の途中で突然落ちたり、設定をわずかに変えただけで結果が大きくブレたりしました。MoCo v3 はその不安定さの原因を実験で徹底的に調べ上げ、犯人がモデルの一番最初の層(パッチをベクトルに変える層)にあることを突き止め、その層を凍結する(学習させずランダム初期値のまま固定する)という極めて単純な対策で訓練を安定させました。新しいアーキテクチャを提案したのではなく、現象を顕微鏡で観察して病巣を特定し、最小限の処方を出した「実証研究(empirical study)」が論文の本質です。

## 直感的な理解
ラベルなしで画像の特徴を学ぶには、「同じ写真を2通りに加工した2枚は仲間、違う写真は他人」と教えてやる、というのが対照学習の発想です。1枚の猫の写真を、色を変えた版と切り抜いた版にして、その2枚の特徴ベクトルを近づけるよう学習します。こうしてラベルが一切なくても「猫らしさ」のような特徴が浮かび上がってきます。

ところがこの学習を ViT でやると、しばしば壊れます。訓練曲線(横軸が学習ステップ、縦軸が精度のグラフ)を見ていると、順調に上がっていた精度がある瞬間にガクッと落ちて、そこからまた持ち直す「ディップ(dip、くぼみ)」が繰り返し現れます。最終精度もシードや学習率しだいで大きく変わり、論文に「この設定でこの精度が出ます」と断言できるほど安定しません。研究の道具として致命的です。

ここで重要なのは、この不安定さが「目に見えにくい」ことです。最終精度だけ見ていると、たまたまうまくいったランは高い数字を出すので、「ViT の対照学習はうまくいく」と誤解してしまいます。しかし学習率を少し上げたり別のシードで回したりすると突然壊れる。再現性がないのです。MoCo v3 は「なぜ ViT の対照学習はこんなに不安定なのか」という素朴で重要な問いに、観察と実験だけで答えを出しました。

## 基礎: 前提となる概念
順に用語を噛み砕きます。

- 自己教師あり学習(Self-Supervised Learning, SSL): 人手のラベル(「これは猫」など)を一切使わず、データそのものから「擬似的な問題(pretext task)」を作って解かせることで特徴を学ぶ方法です。ラベル付けは高コストなので、大量のラベルなし画像から学べるのは大きな利点です。

- 対照学習(Contrastive Learning): SSL の一種で、「正例ペア(同じ画像の2つの加工版)は特徴空間で近づけ、負例(別の画像)は遠ざける」ように学びます。特徴空間とは、各画像を1本のベクトル(数百次元の数の並び)として置く抽象的な空間です。近い・遠いはベクトル間の距離やコサイン類似度(向きの近さ)で測ります。

- InfoNCE 損失: 対照学習で標準的に使う損失関数です。クエリ q と、その正例 k+、たくさんの負例 k- があるとき、`L = -log( exp(q·k+/τ) / Σ_k exp(q·k/τ) )` という形をしています。q·k はベクトルの内積(似ているほど大きい)、τ(タウ、temperature)は鋭さを調整する温度パラメータです。τ が小さいほど「正例だけを強く近づけ、惜しい負例を強く押しのける」鋭い損失になります。意味は「正例を当てるソフトマックス分類」で、全候補(正例1個+負例多数)の中から正例を選び当てる多クラス分類だと思えば分かりやすいです。

- MoCo(Momentum Contrast): He らが提案した対照学習の枠組みです。鍵は2つのネットワークを使うこと。ひとつは普通に勾配で学習する base encoder(生徒、query encoder)、もうひとつは momentum encoder(教師、key encoder)です。教師の重みは生徒の重みの指数移動平均(EMA、Exponential Moving Average)でゆっくり追従します。式で書くと `θ_teacher ← m · θ_teacher + (1-m) · θ_student`(m は 0.99〜0.999 など 1 に近い値)。教師がゆっくり動くことで、負例として使う特徴が時間的に安定し、学習が落ち着きます。教師には勾配を流しません(stop-gradient、勾配を止める操作)。

- ViT(Vision Transformer): 画像を 16×16 ピクセルのパッチに分割し、各パッチを線形変換でひとつのベクトルに埋め込み(これが patch projection、後述)、そのベクトル列に学習可能な CLS トークン(分類用の特別なトークン)と位置埋め込みを加え、自然言語処理由来の Transformer に通すモデルです。CNN のような畳み込みの帰納バイアス(局所性・並進不変性という事前の思い込み)を持たないぶん、大量データでは強いが、小データや不安定な学習には脆い、という性質があります。Dosovitskiy らが 2020 年に提案しました(ICLR 2021, arXiv:2010.11929)。

## 仕組みを詳しく
MoCo v3 の学習ループ、そして安定化の発見と処方を順に見ます。

### 学習の流れ(MoCo v3 版)
1. 1枚の画像 x に、ランダムなデータ拡張(random resized crop で切り抜き、color jitter で色をずらす、Gaussian blur でぼかす、grayscale 化、solarization など)を独立に2回かけ、2つのビュー x1, x2 を作ります。
2. 生徒 f_q は ViT 本体 → [projection head(投影ヘッド)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1082_projection-head)(3層 MLP)→ prediction head(もう1段の 2層 MLP)という構成です。教師 f_k は ViT 本体 → projection head までで、prediction head は持ちません(非対称、asymmetric)。
3. x1 を生徒に通して q1、x2 を教師に通して k2 を得ます。逆向き(x2→生徒、x1→教師)も計算し、対称化(symmetrized loss)します。各ベクトルはたとえば `[256]` 次元に L2 正規化されています。
4. InfoNCE 損失を計算します。MoCo v1/v2 にあった「負例キュー(過去の特徴をためておくメモリバンク)」を MoCo v3 は廃止し、同じバッチ内の他の画像をそのまま負例に使います(SimCLR 流)。そのためバッチサイズは数千(典型的に 4096)と大きく取ります。
5. 損失を生徒のパラメータについて逆伝播して更新し、教師は生徒の EMA で追従させます。

before/after で構造を比べると:

```
MoCo v1/v2:  query encoder ── InfoNCE ── 負例キュー(過去の key を貯める)
             key encoder(EMA) ── enqueue
MoCo v3:     query encoder + pred head ── InfoNCE ── 同一バッチ内の key を負例に
             key encoder(EMA, pred head なし)            ↑ キュー廃止・大バッチ化
```

### 不安定さの原因の発見
論文の核心はここです。著者らは学習中、各層のパラメータの勾配(その層をどれだけ動かすべきかの大きさ)を層ごとにモニタリングしました。すると、精度のディップが起きるちょうど直前に、ViT の第1層 patch projection の勾配が突発的に巨大化するスパイク(尖ったピーク)を起こしていることを発見しました。スパイクは数十〜数百イテレーション後に出力層側へ伝播し、その後に精度がガクッと落ちる、という時間順序まで観察されています。つまり、最初の層が暴れることが学習崩壊の引き金だったのです。

patch projection は、16×16×3 = 768 個の数(パッチのピクセル値)を受け取り、それを `[768] → [embed_dim]`(ViT-B なら 768 次元)に写す、たった1枚の線形層(実装上は stride=16 の畳み込み1つ)です。モデル全体から見ればパラメータ数のごく一部ですが、すべての入力が最初に通る関門なので、ここが暴れると後段すべてが揺さぶられます。著者らは「最初の勾配スパイクが、後段の不安定の原因なのか結果なのか」を切り分けるために、勾配の時間波形を細かく観察し、スパイクが先行することを示しました。

### 対策: patch projection の凍結
処方は驚くほど単純です。patch projection 層をランダムに初期化したまま、学習を通して一切更新しないで固定します(freeze、凍結。実装上は requires_grad を False にする)。直感的な根拠は、ランダムな線形射影でもパッチの情報はほぼ保たれる(Johnson–Lindenstrauss の補題が示すように、ランダム射影は高次元での距離をおおむね保存する)ので、この層を無理に学習させなくてもモデルは十分な表現を学べる、というものです。

凍結なしでは learning rate を上げると勾配スパイクが頻発しディップが多発、最終精度もブレますが、凍結すると勾配スパイクが消え、ディップがなくなり、learning rate やバッチサイズを変えても精度が安定して再現します。コードでいえば「最初の層の requires_grad を False にする」程度の1行レベルの変更で挙動が劇的に変わる、というのが衝撃的なポイントでした。重要なのは、これは MoCo に固有の現象ではなく、SimCLR や BYOL のフレームワークで ViT を学習させても同じスパイクが観察された点です。つまり「ViT の対照学習一般の問題」であり、フレームワークによらない普遍的な発見でした。

### 安定運用のためのその他のレシピ
- 大きなバッチサイズ(典型 4096)と、それに合わせた learning rate のウォームアップ(最初は学習率をほぼゼロから始め、数〜10エポックかけて目標値まで線形に上げる)。大バッチでいきなり高い学習率を使うと序盤に壊れやすいためです。
- 最適化手法は AdamW を採用(以前の対照学習でよく使われた LARS よりも ViT では安定したと報告)。LayerNorm の配置や weight decay も検証されています。
- これらは ViT に限らず、「Transformer を対照学習する際の安定化レシピ集」として後続研究に参照されました。

## 手法の系譜と主要論文
- He, Fan, Wu, Xie, Girshick, "Momentum Contrast for Unsupervised Visual Representation Learning"(MoCo v1, CVPR 2020, FAIR, arXiv:1911.05722)。momentum encoder と負例キューを提案し、対照学習を ResNet で実用レベルに引き上げました。動機は、対照学習には大量の負例が要るが、巨大バッチは GPU メモリで非現実的だったこと。キューと EMA でそれを解決しました。
- Chen, Fan, Girshick, He, "Improved Baselines with Momentum Contrastive Learning"(MoCo v2, 2020, arXiv:2003.04297)。SimCLR の知見(MLP の projection head、強いデータ拡張、cosine 学習率スケジュール)を MoCo に取り込んだ小改良版です。
- Chen, Kornblith, Norouzi, Hinton, "A Simple Framework for Contrastive Learning(SimCLR)"(ICML 2020, Google Brain, arXiv:2002.05709)。大バッチ・強い拡張・[projection head](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1082_projection-head) を確立しました。MoCo v3 は SimCLR のバッチ内負例方式と projection head を継承します。
- Grill et al., "Bootstrap Your Own Latent (BYOL)"(NeurIPS 2020, DeepMind, arXiv:2006.07733)。負例を一切使わず、予測ヘッド + EMA 教師 + stop-gradient で崩壊を避ける手法です。MoCo v3 の非対称構造(生徒だけ prediction head を持つ)はここに連なります。
- Chen, Xie, He, "An Empirical Study of Training Self-Supervised Vision Transformers"(MoCo v3, ICCV 2021, FAIR, arXiv:2104.02057)。本記事の主役。MoCo を ViT 向けに整理し、不安定さの原因を patch projection の勾配スパイクと特定し、凍結という処方を出しました。

この系譜は「負例キュー(v1)→ MLP ヘッド導入(v2/SimCLR)→ 負例なし化と非対称化(BYOL)→ ViT 対応と安定化(v3)」と読めます。同じ ICCV 2021 の DINO(Caron et al., arXiv:2104.14294)も ViT × SSL の重要研究で、こちらは自己蒸留と centering/sharpening で崩壊を避け、kNN 評価を前面に出しました。MoCo v3 と DINO は ViT を SSL で実用化した双璧として並び称されます。MoCo v3 が「最適化の安定性」に注目したのに対し、DINO は「崩壊回避の仕組み」と「出てくる特徴の質(教師なしセグメンテーションが創発する)」に注目しており、対照的なアプローチです。

## 論文の実験結果(定量データ)
評価は ImageNet-1K(1000 クラス、約 128 万枚)での [linear probing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0759_linear-probing-evaluation)(バックボーンを凍結し線形分類器だけ訓練、top-1 精度を測る)が中心です。top-1 精度とは「最も確信したクラスが正解と一致した割合」で、100% が満点、ランダムなら 0.1% です。

- ViT-B/16(Base、パッチ16)を MoCo v3 で自己教師あり学習し linear probing すると、報告でおよそ 76.7% の top-1 精度に達しました。これは同条件の ResNet-50 ベース対照学習(MoCo v3 の ResNet-50 はおよそ 74.6%、MoCo v2 系は 71 前後)を上回ります。同じ計算予算では ViT が ResNet を上回ることを示した点が重要です。
- より大きい ViT-L(Large)では 77〜78% 程度まで伸び、さらに ViT-BN-L/7(BatchNorm 入り・パッチ7の高解像度設定)では約 81% を報告しました。モデルを大きくしパッチを細かくするほど SSL の伸びしろが大きい、という ViT らしいスケーリング傾向を示しています。
- アブレーション(ある要素を抜くと何が起きるかの実験)が論文の見どころです。patch projection を「学習させた場合」と「凍結した場合(random patch projection)」を比べると、凍結なしでは learning rate を上げたとき(例 lr を 1.0e-4 から 1.5e-4 へ)に精度が大きく不安定化(ディップ多発、最終精度が数ポイント単位で乱高下)するのに対し、凍結すると同じ learning rate でも安定して高精度を保ちました。凍結ありの方が最終精度も平均的に約 1 ポイント高く、かつ分散が大幅に小さくなったと報告されています。
- バッチサイズの実験では、1024〜4096 では概ね安定して精度が向上する一方、6144 のように極端に大きくすると、ウォームアップが十分でないと序盤に壊れて精度が低下する(おかしな谷ができる)ことが示されました。「大きければ良い」という単純な話ではない、という教訓です。

数値の意味を補足すると、SSL 研究では linear probing top-1 の「1〜2 ポイント差」が手法の優劣を分ける重要な差として扱われます。1000 クラスの難問で 0.1% のランダムから 76% まで来ている時点で、1ポイントは約1万3千枚の判定を覆す差です。ただし MoCo v3 の最大の貢献は数ポイントの上積みそのものよりも、「ViT で再現性のある数字を安定して報告できるようにした」点にあります。不安定なまま「最良ランの 78%」を報告するのと、安定して「76.7±小さい分散」を報告するのとでは、科学的な価値が全く違います。

## メリット・トレードオフ・限界
メリット
- ViT 対照学習の不安定さの原因を勾配観察で特定し、凍結という1行レベルの対策で安定化できる。原因究明と処方が明快で、MoCo/SimCLR/BYOL のいずれにも効く普遍的な発見。
- 大バッチと momentum encoder により、ResNet ベースを上回る高品質な SSL 表現を ViT で得られる。
- ラベル不要なので、ラベル付けが高コストな大規模画像・映像データに向く。

トレードオフ・限界
- 大バッチ(数千)前提で、GPU メモリと計算資源を多く要する。小規模環境では再現が難しい(数百 GPU 級の実験が背景にある)。
- 強いデータ拡張に依存し、拡張の設計を誤ると性能が落ちる。拡張は ImageNet 分類向けに調整されており、医用画像・自動運転映像など他ドメインで最適とは限らない。
- 凍結はあくまで経験則であり、「なぜランダム固定で十分か」「なぜ第1層だけが暴れるのか」の完全な理論証明はない。Johnson–Lindenstrauss は直感を支えるが厳密な保証ではない。
- 対照学習は負例の扱いしだいで [次元崩壊](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse) のリスクを抱え、projection/prediction head などの工夫が前提になる。
- 評価が主に分類で、検出・セグメンテーション・予測のような構造的タスクでの優位は別途検証が要る。

## 発展トピック・研究の最前線
MoCo v3 の後、SSL の主流は対照学習(画像どうしを比べる)から、マスク再構成(画像の一部を隠して復元させる)系へと比重を移しました。He らの MAE(Masked Autoencoders, CVPR 2022, arXiv:2111.06377)は ViT のパッチの 75% をマスクして残りから復元させる手法で、対照学習のような大バッチや強い拡張、negative pair を必要とせず、fine-tuning 後の性能(ViT-L で約 85.9% top-1)で対照学習を上回りました。ただし MAE は linear probing 精度では対照学習に劣る傾向(ViT-B で約 68%)があり、「linear probing が強い対照学習 vs fine-tuning が強い MAE」という [評価プロトコル](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0759_linear-probing-evaluation)依存の対比が生まれました。

両者を融合する流れもあります。iBOT(Zhou et al., ICLR 2022)は DINO 流の自己蒸留にマスク予測を組み合わせ、data2vec(Baevski et al., ICML 2022)は教師の中間表現を予測対象にする統一フレームワークを提案しました。畳み込み版では ConvNeXt V2 の FCMAE がマスク再構成を CNN に持ち込みました。MoCo v3 が示した「Transformer の SSL は最初の層と最適化の安定性が鍵」という教訓は、これら後続のマスク系手法の学習安定化や、後に登場した大規模基盤モデル(DINOv2 など)の安定化レシピにも引き継がれています。

## さらに学ぶための関連トピック
- [projection head (投影ヘッド)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1082_projection-head)
- [次元崩壊 (dimensional collapse)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0043_dimensional-collapse)
- [linear probing / kNN評価](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0759_linear-probing-evaluation)
- [ConvNeXt V2 / FCMAE](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0039_convnextv2-fcmae)
- [data2vec](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0041_data2vec)
