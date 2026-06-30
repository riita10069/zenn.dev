---
title: "AdaLN による時刻・文脈の条件付け（Adaptive LayerNorm）"
---

# AdaLN による時刻・文脈の条件付け（Adaptive LayerNorm）

## ひとことで言うと

ニューラルネットの途中の信号に対して、条件情報(いまの時刻、車の運動状態、直前の映像など)から計算した2つの量「拡大率 scale」と「上げ下げ量 shift」を掛けたり足したりして、ネットの振る舞いを状況に合わせて変調(modulate、調節)する仕組みです。Flow Matching や Diffusion の速度場・ノイズ予測ネットワークを時刻と文脈に応じて切り替えるために広く使われます。AdaLN は Adaptive Layer Normalization(適応的レイヤー正規化)の略です。

## 直感的な理解

同じ運転手でも、晴れた直線では緩やかに、雨の交差点では慎重にハンドルを切ります。判断の「基本動作」は同じでも、状況(条件)に応じて出力の大きさやオフセットを調整しているわけです。

ニューラルネットでも同じことをしたい。速度場ネットワークは「ノイズ混じりの軌跡」を入力に「進むべき速度」を出しますが、同じ入力でも時刻 t が違えば(ノイズ寄りか完成寄りか)、文脈 c が違えば(高速直進か交差点か)、出すべき速度は変わるべきです。問題は「どうやって t や c をネットに効かせるか」です。

素朴な方法は「入力に t や c をつなげて1本のベクトルにする(concatenation、連結)」ことですが、これだと条件情報が薄まりやすく、深い層まで影響が届きにくい。AdaLN はこれを解決します。アイデアは「正規化層の scale と shift を、固定値ではなく条件から動的に計算する」こと。条件がネットの内部信号を直接スケーリング・オフセットするので、強く確実に効きます。条件が変われば正規化のかかり方そのものが変わるため、1つのネットワークで多様な条件に応答できます。

## 基礎: 前提となる概念

第一に「正規化(normalization)」。ニューラルネットの中間信号は層を重ねるほど分布(平均や分散)がずれて学習が不安定になります。LayerNorm(レイヤー正規化)は各サンプルの特徴ベクトルを平均0・分散1に揃え、その後に学習可能な scale(γ)と shift(β)を掛け足して表現力を戻します。式は `y = γ·(x−μ)/σ + β`。μ,σ はそのベクトルの平均・標準偏差です。BatchNorm がバッチ全体で統計を取るのに対し、LayerNorm は1サンプル内で取るので、バッチサイズや系列長に依存せず安定します。

第二に「条件付け(conditioning)」。生成モデルに「何を生成してほしいか」(クラス、テキスト、時刻、状態など)を伝える操作です。入れ方には連結、クロスアテンション、そして本ノートの変調(modulation)があります。

第三に「アフィン変換」。`scale·x + shift` の形の、拡大と平行移動だけの変換です。AdaLN の変調はこのアフィン変換を、特徴チャンネルごとに条件依存で行います。チャンネルごとに別々の scale/shift を持つので、「ある特徴は強調し、別の特徴は抑える」といった選択的な変調ができます。

第四に「クロスアテンション(cross-attention)」。あるトークン列(クエリ)が別のトークン列(キー/バリュー)を参照して関連情報を引いてくる仕組み。空間的に分布した条件(地図など)を扱うのに向きます。AdaLN とは役割分担します。

第五に「残差接続(residual connection)」。`y = x + f(x)` の形で、層の出力に入力をそのまま足し戻す配線です。深いネットの勾配を安定させる土台で、後述の AdaLN-Zero はこの残差をうまく活かす初期化になっています。

## 仕組みを詳しく

Flow Matching の速度場ネットワークが AdaLN を使う典型的な流れを、形状つきで追います。設定例は埋め込み次元 256、出力が 64 時刻 × 2 信号(加速度と曲率)。

入力:
- `u_t`: `[B, 128]`(= 64×2 をフラットにしたノイズ混じり軌跡)
- `t`: `[B]`(時刻)
- `bev_seq`: `[B, HW, 256]`(地図 BEV を平らに並べたトークン列)
- `mod_cond`: `[B, 256]`(視覚履歴と自車運動を足した低次元の条件ベクトル)

ステップ1: 軌跡をトークン列にして地図にクロスアテンション。
```
u_t_seq  = u_t.reshape(B, 64, 2)            # [B,64,2] 64個の行動トークン
queries  = action_proj(u_t_seq)             # [B,64,256]
attended = cross_attn(queries, bev_seq, bev_seq)  # [B,64,256]
```
各時刻の行動トークンが「地図のどこを見るべきか」を引いてきます。この attended がこれから AdaLN で変調される本体です。

ステップ2: 時刻と文脈から scale/shift を作る(AdaLN の核心)。
```
t_emb        = time_mlp(sinusoidal(t))             # [B,256] 時刻埋め込み
gamma, beta  = adaln_mod(mod_cond + t_emb).chunk(2)# それぞれ [B,256]
# adaln_mod = SiLU → Linear(256 → 512)、出力を半分に割る
```
時刻埋め込み([Timestep Embedding（時刻埋め込み）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0137_timestep-embedding))と文脈を足し合わせ、その合成から scale 用 gamma と shift 用 beta を生成します。条件が変われば gamma/beta が変わり、ネットの振る舞いが変わります。

ステップ3: 正規化してから変調。
```
normed    = norm(attended)                          # アフィン無し LayerNorm
modulated = normed * (1 + gamma[:,None,:]) + beta[:,None,:]
```
ここが AdaLN の本体です。norm は `elementwise_affine=False`、つまり固定の scale/shift を持たない LayerNorm。代わりに条件から作った gamma/beta を使います。
- `1 + gamma`: gamma=0 なら恒等変換(素通し)。学習初期に安全な状態から始められる。
- `beta`: 信号全体を上下にオフセット。
- `[:,None,:]` は `[B,256]` を `[B,1,256]` に広げ、64 トークン全部に同じ変調を適用するブロードキャスト。

ステップ4: 変調済み信号から速度を出力。
```
velocity = velocity_head(modulated).reshape(B, 128) # [B,128]
```

要するに AdaLN は「地図を見て作った行動トークン」を「時刻と運動・視覚文脈」に応じて拡大・平行移動し、出すべき速度を文脈依存に切り替えています。

実際の Transformer ブロックでは、AdaLN はさらに細かく、各サブ層(自己アテンションと MLP)の前後に複数の変調を入れます。DiT 流の AdaLN-Zero では1ブロックあたり6つの変調量(アテンション前の scale/shift、アテンション出力の gate、MLP 前の scale/shift、MLP 出力の gate)を条件から生成します。gate は残差に足し戻す前に出力へ掛ける係数で、0初期化すると学習開始時にブロックが恒等写像(入力をそのまま通す)になります。

設計上の重要な分担: 地図 BEV は scale/shift には混ぜません。空間的な細かさが大事なのでクロスアテンション(ステップ1)で扱い、AdaLN には時刻と低次元の大域文脈だけを渡します。「空間情報はアテンション、大域文脈は AdaLN」という役割分担です。これは AdaLN の本質的限界(後述)に対応した設計判断です。AdaLN は条件を1本のベクトルに要約してチャンネル単位のアフィンに変換するため、空間的にどこを見るかという情報を失うからです。

## 手法の系譜と主要論文

変調による条件付けは複数の系譜が合流して現在の AdaLN に至ります。

源流のひとつは Conditional Batch Normalization と FiLM です。Perez, Strub, de Vries, Dumoulin, Courville(FiLM: Visual Reasoning with a General Conditioning Layer, AAAI 2018, arXiv:1709.07871)が一般化した形を提案しました。条件(質問文など)から各特徴チャンネルごとの scale γ と shift β を計算し `γ·feature + β` で変調します。動機は「条件情報を連結より強く、少ないパラメータで効かせたい」。チャンネル単位のアフィン変換に限られる(空間的にきめ細かい変調はしない)のがトレードオフです。

画像生成では Huang & Belongie(AdaIN, ICCV 2017, arXiv:1703.06868)がスタイル転送のため Instance Normalization の scale/shift をスタイル画像の統計で置き換え、Karras ら(StyleGAN2, CVPR 2020, arXiv:1912.04958)系がスタイルベクトルから正規化の scale/shift を作り生成の様式を制御しました。条件→正規化パラメータという発想の代表例です。

決定的なのが Peebles & Xie(DiT: Scalable Diffusion Models with Transformers, ICCV 2023, arXiv:2212.09748)です。Diffusion のバックボーンを U-Net から Transformer にし、時刻とクラスラベルを AdaLN で各ブロックに注入しました。彼らは gate を0初期化する AdaLN-Zero を提案し、複数の条件付け方式(in-context、cross-attention、AdaLN、AdaLN-Zero)を比較して AdaLN-Zero が計算効率と品質のバランスで最良と報告しました。`1+gamma` の初期素通し設計や gate の0初期化はこの AdaLN-Zero に対応します。

Chen ら(PixArt-α, ICLR 2024, arXiv:2310.00426)は、全ブロックで AdaLN の変調生成層を共有する AdaLN-single を提案し、DiT の AdaLN が持つ大量のパラメータ(各ブロックが独立に変調を出す部分)を削減しました。大規模化時のメモリ・パラメータ効率を改善する変種です。

Esser ら(Stable Diffusion 3, MM-DiT, ICML 2024, arXiv:2403.03206)は、画像とテキストの2つのストリームをそれぞれ AdaLN で時刻変調しつつ、相互は結合アテンションで交わらせる MM-DiT を提案し、AdaLN とアテンションの役割分担をマルチモーダルへ拡張しました。

Lipman ら(Flow Matching, ICLR 2023, arXiv:2210.02747)は速度場 v(x,t) の学習枠組みを与え、条件付き速度場 v(x,t,c) の c の入れ方は実装者に委ねました。Flow Matching の速度場に DiT 流 AdaLN を採用するのは、この自由度を埋める自然な選択です。自動運転の Flow/Diffusion 軌跡生成でも、自車状態や視覚特徴を AdaLN で速度場に注入する構成が一般化しています。

## 論文の実験結果(定量データ)

DiT(Peebles & Xie, 2023)の条件付け方式比較が最も直接的です。ImageNet 256×256 のクラス条件付き生成で、彼らは同程度の計算量(Gflops)のもとで4方式を比較しました。報告では、AdaLN-Zero が cross-attention や in-context 方式より低い FID(生成分布と本物の距離、小さいほど良い)を達成しました。加えて、最大モデル DiT-XL/2 はクラス条件付き ImageNet 256×256 で FID 約2.27(classifier-free guidance 併用)という当時の最良級を記録しました。AdaLN-Zero は cross-attention 方式と比べて追加パラメータ・計算が少ないにもかかわらず品質が高く、「効率と品質を両立する条件付け」という主張を実証しました。

アブレーション(どの要素を抜くとどうなるか)の観点では、DiT は AdaLN の gate を0初期化(AdaLN-Zero)すると、0初期化しない通常の AdaLN より学習が安定し最終 FID も良いと報告しています(論文の図では AdaLN-Zero が他方式より明確に低い FID へ収束)。理由は、学習開始時に各ブロックが恒等変換(素通し)から始まるため、深いネットでも勾配が安定し、残差接続が壊れにくいからです。`1+gamma` と gate の0初期化はこの安定化を狙ったものです。

PixArt-α(Chen et al., 2024)は AdaLN-single により、品質をほぼ保ったままパラメータと計算を削減できることを示し、限られた学習計算量(報告では一般的な大規模拡散モデルの数%程度の GPU 時間)で競争力ある text-to-image 生成を達成しました。これは「変調生成層の共有」が大規模化のボトルネックを緩めることの実証です。

FiLM(Perez et al., 2018)は視覚質問応答(CLEVR データセット)で、条件付けに FiLM を使うことで連結ベースの手法より大幅に精度を上げ、当時ほぼ完璧(98%前後と報告)な正解率を達成しました。条件を変調として効かせることの有効性を初期に示した例です。

自動運転の軌跡生成では公開ベンチマーク(nuScenes など)で、AdaLN ベースの条件付けが自車状態や地図文脈をうまく反映し、衝突率や変位誤差(L2 displacement error)を改善する報告がありますが、条件付け方式単独のアブレーションは画像生成ほど整理されていません。

## メリット・トレードオフ・限界

メリット:
- 条件情報をネット内部信号に直接スケーリングして効かせるので、連結より強く確実に届く。
- `1+gamma`・gate の0初期化(AdaLN-Zero)で学習が安定しやすい。
- 空間的条件(地図)を別経路(アテンション)へ逃がし、大域文脈だけを AdaLN で扱う役割分担ができる。
- 追加パラメータは Linear 数枚程度で、DiT の比較では cross-attention より効率的。AdaLN-single でさらに削減可能。

トレードオフ・限界:
- 変調はチャンネル単位のアフィン(掛け算+足し算)なので、条件を「1本のベクトルに要約」できることが前提。空間的に広がった条件(地図全体)は AdaLN だけでは扱えず別経路が必要。
- gamma/beta を出す層が条件を(おおむね線形に)要約するため、極端に複雑な条件依存はこの層だけでは表しきれないことがある。
- 時刻と文脈を「足してから」gamma/beta を作る実装では、両者の相互作用の表現が限定的。より精密にやるなら別々に注入する設計もある。
- DiT 流の素朴な AdaLN は1ブロックあたり多数の変調量を出すためパラメータが嵩む(AdaLN-single はこの欠点への対処)。

未解決の論点として、複数の異種条件(時刻・状態・地図・言語指示など)を AdaLN・アテンション・連結のどれに割り当てるのが最適かは、まだ経験則に頼る部分が大きいです。

## 発展トピック・研究の最前線

- AdaLN-single / 共有変調: 全ブロックで変調生成層を共有してパラメータを削る変種(PixArt 系)。大規模化時のメモリ節約に有効。
- マルチ条件・マルチモーダルの統合: テキスト・画像・状態など複数モダリティを、AdaLN とクロスアテンションをどう組み合わせるかの設計研究(SD3 の MM-DiT、二重ストリーム結合アテンションなど)。
- 推論時の効率化: 推論ループ(Euler 積分、[Euler 法による ODE 推論（Flow Matching のサンプリング）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0120_euler-ode-inference))で時刻に依存しない条件をループ外で1回だけ計算し、各ステップで変わる時刻埋め込みだけ足し直して gamma/beta を作り直す最適化。条件キャッシュにより1ステップあたりのコストを下げる。
- 自動運転特化: 自車のキネマティクス制約や安全マージンを条件として変調に組み込み、生成軌跡の物理的妥当性を高める方向の研究。

## さらに学ぶための関連トピック

- [Timestep Embedding（時刻埋め込み）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0137_timestep-embedding)
- [プランナ レジストリパターン（BasePlanner + PLANNER_REGISTRY）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1000_planner-registry-pattern)
- [Classifier-Free Guidance（CFG／分類器なし誘導）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0160_classifier-free-guidance)
- [Euler 法による ODE 推論（Flow Matching のサンプリング）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0120_euler-ode-inference)
