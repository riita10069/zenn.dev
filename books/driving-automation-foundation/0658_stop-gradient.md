---
title: "Stop-gradient (勾配停止)"
---

# Stop-gradient (勾配停止)

## ひとことで言うと
Stop-gradient(勾配停止)とは、ニューラルネットワークの計算経路のある場所に「ここから先は学習で更新しない」という壁を立て、勾配(重みをどう直すべきかを示す修正情報)がそこから逆流しないように遮断する操作です。順方向の計算では値はそのまま通りますが、逆方向では止まります。一見ささいなこの操作が、2つの枝(branch)で同じ画像の別ビューを処理して出力を近づけるタイプの自己教師あり学習において、「全部同じ答えを出せば損失がゼロになる」という崩壊(collapse)を防ぐ、地味だが決定的な仕掛けになります。

## 直感的な理解
2人で「同じ絵を見て、なるべく似た感想を述べよ」というゲームをすると想像してください。2人とも本気で絵を観察すれば良い感想が揃いますが、最も楽に「完全一致」を達成する方法は、絵を一切見ずに2人とも「うー」とだけ言うことです。これが崩壊です。損失(2人の感想のずれ)はゼロになりますが、感想は絵の中身を何も表していません。

ここで片方の人を「先生」役にして「先生は感想を変えてはいけない。生徒だけが先生に合わせる」という非対称ルールにすると、2人で示し合わせて「うー」に逃げることができなくなります。生徒は先生の(その時点での)感想に合わせるしかなく、先生の感想は絵から作られているので、結局は絵をちゃんと見るしかありません。stop-gradient は、まさにこの「先生は動かない的(まと)になる」という役割を、勾配を止めることで作り出します。

なぜ崩壊が起きるのか。損失を下げる勾配が両方の枝に流れると、ネットワークは「中身を見分ける難しい仕事」より「両方を同じ定数に潰す簡単な仕事」を選びます。学習は楽な方へ流れます。損失の地形(loss landscape)には「定数解(trivial solution)」という広くて深い谷があり、何の制約もなければ最適化はその谷に転がり落ちます。stop-gradient はこの逃げ道を物理的に塞ぎ、最適化を「中身を見る」方向へ押し戻します。

## 基礎: 前提となる概念
理解に必要な用語を順に噛み砕きます。

勾配(gradient)とは、「損失をほんの少し減らすには、各重みをどちら向きにどれだけ動かせばよいか」を表す方向と大きさの情報です。山下りに例えると、今いる地点で最も急に下る方向を指す矢印です。

誤差逆伝播(backpropagation)とは、出力側で計算した損失を、合成関数の微分の連鎖律(チェインルール)を使って入力側の各層へと逆向きに伝え、各重みの勾配を求める手続きです。順伝播(forward)で出力と損失を計算し、逆伝播(backward)で勾配を計算し、その勾配で重みを更新する、というのが学習の1ステップです。

自己教師あり学習(self-supervised learning, SSL)とは、人間が正解ラベルを付けず、データ自身から問題と答えを作って学ぶ方式です。たとえば「同じ画像から少しずつ違う2枚(ビュー)を作り、その特徴を互いに近づけよ」という問題は、ラベルなしで作れます。学習した特徴は後段の分類・検出などに転用されます。

崩壊(collapse)とは、ネットワークが入力に関係なく常に同じ出力(たとえば全成分が同じ定数のベクトル)を出すようになる現象です。「2つのビューの出力を近づけよ」という目標だけだと、両方が同じ定数を出せば距離はゼロ、損失は完璧に最小化されますが、表現は無意味になります。崩壊には2種類あります。完全崩壊(complete collapse、全サンプルが同一の定数ベクトルになる)と次元崩壊(dimensional collapse、出力が高次元空間の一部の低次元部分空間に潰れ、共分散行列の固有値の多くがゼロに近づいて多くの次元が情報を持たなくなる)です。stop-gradient が主に防ぐのは前者ですが、次元崩壊の緩和にも寄与します。

stop_gradient は実装上の名前で、PyTorch では `tensor.detach()`、TensorFlow では `tf.stop_gradient(tensor)` に対応します。順方向では恒等写像(値をそのまま返す)、逆方向では勾配をゼロにする、という性質を持つ演算だと理解すれば十分です。数式で書くと、順伝播は `sg(x) = x`、逆伝播は `∂sg(x)/∂x = 0` と定義される、という非対称な「演算」です。

## 仕組みを詳しく
2枝の自己蒸留(self-distillation、ネットワークが自分自身の出力を教師信号にして学ぶ構図)を例にします。

```
画像 → ビューA → encoder f → predictor h → p_A ┐
                                               ├─ D(p_A, sg(z_B)) を最小化
画像 → ビューB → encoder f ─────────────→ z_B ┘ ← ここに stop-gradient (sg)
```

z_B 側に stop-gradient をかけると、逆伝播の際に z_B から encoder f への勾配が流れません。つまり「z_B は固定された目標(ターゲット)とみなし、p_A の方だけを z_B に近づける」という非対称な最適化になります。z_B を動かして p_A に合わせにいくこと(=両方を定数に潰す共謀)が禁止されます。

多くの実装では、両枝を対称に扱いながら、それぞれ相手側にだけ stop-gradient をかけて損失を平均します。

```
loss = D(p_A, sg(z_B)) / 2  +  D(p_B, sg(z_A)) / 2
```

ここで D は負のコサイン類似度(2つのベクトルの向きがどれだけ揃っているか。揃うほど小さい値)です。sg は引数を定数とみなす操作です。コサイン類似度は `cos(p, z) = (p·z) / (||p|| ||z||)` で、向きだけを見るのでベクトルの長さの違いに影響されません。

数値で確かめます。z_B = [3, 4](正規化すると [0.6, 0.8])、p_A = [1, 0]、D = -cos。最初 cos = 0.6 です。勾配は p_A 側にだけ流れ、p_A を [0.6, 0.8] の方向へ回そうとします。z_B 側には勾配が流れないので、z_B はこのステップでは「動かない的」として振る舞います。もし stop-gradient を外すと、勾配は z_B 側にも流れ、「p_A を z_B に近づける」と同時に「z_B を p_A に近づける」の両方が起き、最も簡単な解として両者が同じ短いベクトル(究極的にはゼロ)へ向かう力が生まれます。これが崩壊への滑り台です。

なぜ predictor h が要るのか。多くの非対照(non-contrastive)手法では stop-gradient だけでなく、片枝に小さな MLP の predictor を挟みます。直感的には、predictor が「ターゲットの分布へ写す変換」を担うことで、encoder の出力 z が直接ターゲットに張り付くのを防ぎ、表現に多様性を残します。Tian et al.(DirectPred, ICML 2021)は、線形 predictor + stop-gradient + EMA の力学を理論解析し、predictor の重みの固有空間が表現の共分散行列の固有空間に追従する(eigenspace alignment)ことで崩壊が回避される、と説明しました。彼らは固有値分解から predictor を解析的に設定する DirectPred を提案し、学習可能 predictor とほぼ同等の性能を出して「predictor の役割は固有空間の整列にある」ことを裏づけました。

EMA ターゲットとの関係。BYOL や MoCo、後の自己蒸留系では、ターゲット側に別のエンコーダ(target/teacher network)を用意し、その重みをオンライン側のゆっくりした指数移動平均(EMA, exponential moving average: θ_target ← τ θ_target + (1-τ) θ_online、τ は 0.99〜0.999 など)で更新します。ターゲット側には勾配を流さないので、これも本質的に stop-gradient です。EMA は「自分自身の少し過去のコピー」を目標にすることで、目標が急に動かず、安定した非対称性を作ります。stop-gradient(瞬間的な固定)と EMA(時間平均で緩やかに動くターゲット)は、いずれも「ターゲットに勾配を流さない」点で同じ家族に属します。両者の違いは「ターゲットがどれだけ速く動くか」だけで、EMA は τ→0 で stop-gradient のみ(瞬間コピー)、τ→1 で完全固定の中間にあります。

なぜ非対称性が崩壊を止めるのか、もう少し踏み込みます。対称な目的(両枝に勾配を流す)では、勾配場(gradient field)に「2枝を縮めて同じ点へ寄せる」回転成分が現れ、これが定数解へ向かう引力になります。stop-gradient はこの回転成分の片側を消し、結果として勾配場の「縮める力」と「散らす力」が釣り合う非自明な不動点(固定点)を作ります。DirectPred の解析は、この不動点が「表現の分散を保つ」状態に対応することを固有値の言葉で示しました。

## 手法の系譜と主要論文
- MoCo(Kaiming He et al., CVPR 2020, arXiv:1911.05722)。対照学習の文脈で、キー(ターゲット)側エンコーダを EMA(momentum)で更新し勾配を止める設計を導入。負例はキュー(memory queue)に貯める。stop-gradient 思想を対照学習に組み込んだ早い例です。
- BYOL(Jean-Bastien Grill et al., NeurIPS 2020, arXiv:2006.07733, "Bootstrap Your Own Latent")。ターゲットネットワークを EMA 更新し、そこに勾配を流さない(stop-gradient 相当)構成で、負例を一切使わずに高性能を達成。「負例なしでも崩壊しない」を初めて鮮烈に示し、stop-gradient + EMA + predictor の組み合わせが鍵だと位置づけました。
- SimSiam(Xinlei Chen & Kaiming He, CVPR 2021, arXiv:2011.10566)。EMA すら外し、重み共有の双子ネットに predictor + stop-gradient だけを残しても崩壊しないと示しました。stop-gradient が崩壊回避の主役で predictor が補助、と分離特定。詳細は [SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0093_simsiam)。
- DirectPred(Yuandong Tian, Xinlei Chen, Surya Ganguli, ICML 2021, arXiv:2102.06810)。predictor + stop-gradient の学習力学を理論解析し、固有空間整列による崩壊回避を説明。
- DINO(Mathilde Caron et al., ICCV 2021, arXiv:2104.14294)。ViT バックボーンの teacher-student 自己蒸留で、teacher 側に stop-gradient + EMA、出力に centering と sharpening を組み合わせて崩壊を防ぎ、教師なしでセグメンテーションが出現する性質を示しました。
- 特徴空間で予測する世界モデル系(画像・動画版)も、target encoder への stop-gradient + EMA を中核に据えており、この系譜の延長にあります。詳細は [LeCunの自律機械知能構想](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0069_lecun-jepa-vision) / [V-JEPA 2](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0101_v-jepa-2)。

## 論文の実験結果(定量データ)
評価には ImageNet 線形評価(linear probing)が標準で使われます。これは学習した特徴を凍結し、その上に線形分類器1層だけを乗せて教師あり学習し、トップ1正解率(top-1 accuracy、最も確信度の高い1クラスが正解と一致する割合)を測る方法です。特徴の良さを「下流で使えるか」で測ります。ImageNet は約128万枚・1000クラスなので、当てずっぽう(チャンスレベル)は約 0.1% です。

SimSiam 論文の決定的なアブレーション(ある要素を抜いて影響を見る実験)。ResNet-50 バックボーンで、stop-gradient を入れた場合の ImageNet 線形評価 top-1 はおよそ 67〜68% に達するのに対し、stop-gradient を外すと損失は数十ステップでほぼ最小(-1 に近いコサイン類似度)へ急落し、出力の各次元の標準偏差はゼロ近くに張り付きます。すなわち完全崩壊です。このとき線形評価の精度はチャンスレベル(1000クラスなら 0.1% 付近)まで落ちます。差は「約 68% vs ほぼ 0%」という極端なもので、stop-gradient が機能しているか否かで表現が「有用」か「完全に無意味」かに二分されることを示します。健全な学習では出力の標準偏差が `1/sqrt(d)`(d は出力次元。L2 正規化された出力が単位球上に一様に散らばるときの理論値)付近で安定するのに対し、崩壊時はゼロへ落ちる、という標準偏差の監視は崩壊検知の実務的な指標として広く使われます。

BYOL は ImageNet 線形評価で ResNet-50 において top-1 約 74.3% を報告し、当時の対照学習(SimCLR の約 69%)を上回りました。BYOL でターゲットの EMA を外して即座にオンライン重みをコピーする(=stop-gradient だけで momentum を τ=0 にする)と性能が大きく劣化し、EMA という緩やかなターゲット更新が安定化に寄与することが示されています。一方で後の研究(SimSiam, DirectPred)は「EMA は必須ではなく、stop-gradient が本質」と示し、両者の役割の切り分けが進みました。整理すると、stop-gradient は崩壊回避に必須、EMA は性能を底上げするが必須ではない、という分業です。

DirectPred は、解析的に設定した predictor でも学習可能 predictor と同等(ImageNet 100クラス・1000クラスの双方で数%以内)の線形評価精度を達成し、predictor が固有空間整列を通じて崩壊回避を担うという理論を実験で裏づけました。さらに、predictor を完全に外す(恒等写像にする)と stop-gradient があっても崩壊が起きることを示し、「stop-gradient と predictor は両方そろって初めて働く」という相補性を明らかにしました。

## メリット・トレードオフ・限界
メリット
- 実装が極めて簡単。該当テンソルに `.detach()` を一つ入れるだけ。
- 計算コストがほぼゼロ。順方向の値も計算グラフ上の出力も変わらず、逆伝播の経路が減るぶんむしろ軽い。
- 負例(対照学習で必要な大量の比較対象)や明示的な正則化なしに、predictor や EMA と組み合わせて崩壊を防げる。

トレードオフ・限界
- なぜ効くのかの直感はつかみにくく、理論的理解(DirectPred など)は後から整備された。今も完全な一般理論があるとは言い難い。
- 単独では足りないことが多く、predictor や EMA ターゲットと組み合わせて初めて安定する構成が大半。
- 入れる場所を誤ると、必要な枝まで更新が止まって学習が進まないか、逆に崩壊するかのどちらかになり、設計が繊細。たとえば両枝とも stop-gradient をかけると勾配が一切流れず学習が止まる。
- 次元崩壊(出力が低次元部分空間に潰れる)は stop-gradient だけでは完全には防げず、[VICReg](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0661_vicreg_note) や [Barlow Twins](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0618_barlow-twins) のような分散・脱相関の正則化が補完的に有効、という見方が研究では一般的です。
- ハイパーパラメータ(EMA の τ、predictor の構造・学習率)への感度があり、設定次第で崩壊の境界が動く。

## 発展トピック・研究の最前線
崩壊回避の「統一的理解」を目指す研究が続いています。対照学習(負例)、非対照(predictor + stop-gradient)、正則化(分散・共分散)を、いずれも「表現の分散を保ち、次元間の相関を減らす」という共通原理で説明しようとする流れです。stop-gradient による非対称性が、暗黙のうちに分散を保つ正則化として働く、という解釈も提案されています。次元崩壊そのものを共分散行列の固有値スペクトルで定量化し、stop-gradient や正則化がそのスペクトルをどう平坦化するかを測る研究(Jing et al., 2022, "Understanding Dimensional Collapse in Contrastive Self-supervised Learning", arXiv:2110.09348)も登場しました。特徴空間で予測する世界モデル系(画像・動画版)では、ピクセル再構成ではなく特徴空間で予測する設計と、target encoder への stop-gradient + EMA を組み合わせることで、効率的かつ崩壊しない表現学習を実現しており、stop-gradient はその中核部品として今も現役です。

## さらに学ぶための関連トピック
- [SimSiam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0093_simsiam)
- [VICReg](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0661_vicreg_note)
- [Barlow Twins](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0618_barlow-twins)
- [V-JEPA 2](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0101_v-jepa-2)
- [LeCunの自律機械知能構想](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0069_lecun-jepa-vision)
