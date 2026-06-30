---
title: "TransFuser (CARLA向けセンサ融合E2E)"
---

# TransFuser (CARLA向けセンサ融合E2E)

## ひとことで言うと
TransFuser は、車のカメラ映像と LiDAR(レーザーで周囲の距離を測るセンサ)を Transformer の attention で深く混ぜ合わせ(融合し)、走行経路を直接出力する End-to-End の自動運転モデルです。CARLA(自動運転を試せる無料の運転シミュレータ)上で、当時としては最高水準の運転性能を出し、その後の多くのシミュレータ向け E2E 研究の出発点になりました。融合を「最後に1回」ではなく「処理の途中で何度も」行う点が技術的な核心です。

## 直感的な理解
カメラと LiDAR を使いたい理由は弱点が補完的だからです。カメラは信号の赤青・車線・標識など見た目の意味に強い一方、距離を正確に測れず暗所・逆光に弱い。LiDAR は距離と 3D 形状に強い一方、色や文字は読めません。

問題は混ぜ方です。素朴なのは、両者を別々のネットワークで処理して最後に特徴を連結(横に並べてつなぐ)する late fusion です。実装は簡単ですが、両者が深く絡む場面で弱い。たとえば交差点で「カメラに映る赤信号(意味)」と「LiDAR が捉えた前車の正確な位置(距離)」を結びつけ、さらに視界全体に散らばる複数の物体を同時に考慮する必要があるとき、最後にちょっと混ぜるだけでは関係を捉えきれません。具体的には、信号が赤で前方に車がいるなら止まるべきですが、信号の認識(カメラ)と前車の位置(LiDAR)が別々に処理されて最後に足されるだけだと、「赤信号 かつ 前車近い」という連動した状況判断が立ちにくいのです。

TransFuser の著者らは、こうした広域の文脈(グローバルコンテキスト)を捉えるには、すべての要素同士の関連を計算する attention が向いていると考え、処理の途中段階で繰り返しカメラと LiDAR を attention で混ぜる設計を提案しました。CNN は近くの画素しか一度に見られない(受容野が局所的)のに対し、attention は画像の端と端、あるいはカメラのある場所と LiDAR の別の場所を1ステップで関連づけられる、という違いが効きます。

## 基礎: 前提となる概念
- センサ融合のタイミング: early fusion(入力段で混ぜる)、middle/intermediate fusion(特徴段で混ぜる)、late fusion(各モダリティで結論を出してから混ぜる)の3種類。late は楽だが相互作用を取りこぼし、early は座標系合わせが難しい。TransFuser は中間段で複数回混ぜる立場です。
- attention / self-attention: 要素集合の中で「各要素が他のどれにどれだけ注目するか」を学習し、関連の強い情報を集める仕組み。離れた位置の要素同士でも直接関連づけられるのが CNN との違いで、グローバルな文脈把握に向きます。
- BEV(鳥瞰図): 真上から見下ろした 2D 表現。LiDAR 点群を上から見て、各マスに入った点の密度や高さを画素値にした絵に変換し、画像と同じく 2D で扱えるようにします。
- waypoint(ウェイポイント): 自車がこれから通過する点 (x, y) の列。これを順に辿れば軌道になります。例えば 1 秒ごとに 4 点なら `[4, 2]`。
- PID コントローラ: 目標との誤差から操作量を計算する古典的な制御器(比例 P・積分 I・微分 D の3項)。waypoint をハンドル・アクセル・ブレーキに変換する低レベル制御に使います。学習で出すのは「どこを通るか」だけで、実際のペダル操作は古典制御に任せる分業です。
- 模倣学習(imitation learning): 専門家(人間やルールベースの自動運転 expert)の運転を真似るように学習する方式。教師あり学習で「この状況ならこの操作」を覚えます。CARLA では全情報にアクセスできる「特権 expert(privileged expert)」を教師に使うことが多い。
- CARLA とリーダーボード: オープンソースの運転シミュレータ。CARLA Leaderboard はオンラインで運転性能を競うランキングで、当時の E2E 研究の主戦場でした。closed-loop(自分の操作が次の状況を生む)で評価できるのが利点です。

## 仕組みを詳しく
TransFuser は2本のバックボーン(特徴抽出ネットワーク)を並走させます。

- 画像枝: 前方カメラ画像 `(B, 3, H, W)` を CNN(ResNet など)に通す。
- LiDAR 枝: LiDAR 点群を BEV の 2D 画像に変換したもの `(B, C, H', W')` を別の CNN に通す。

ふつうのマルチセンサ融合は最後に1回混ぜますが、TransFuser は両枝の中間特徴マップ(解像度の異なる複数段、典型的には4段)それぞれに Transformer 融合モジュール(GPT 風の self-attention ブロック)を挿入します。各段の処理を分解すると:

1. 画像特徴マップ `(B, C, h1, w1)` と LiDAR 特徴マップ `(B, C, h2, w2)` を、それぞれトークン列(小さなパッチを1要素とする並び)に平坦化する。例えば 8×8 の特徴マップなら 64 トークン。
2. 2つのトークン列を1本につなぎ(位置エンコーディングを足して)、self-attention にかける。これで「画像のあるパッチ」と「LiDAR のあるパッチ」が直接関連度を計算でき、意味と距離が結びつく。
3. attention の出力を元の特徴マップ形に戻し、各枝に足し戻す(residual)。
4. 浅い段(細かい情報)から深い段(広い文脈)まで複数解像度でこれを繰り返す。浅い段で混ぜると細部、深い段で混ぜると広域文脈が共有されます。

最終的に2枝の特徴を集約し、GRU(再帰型ネットワーク)ベースの自己回帰デコーダで waypoint 列(`[T, 2]`、通過点 (x,y) の系列)を1点ずつ出力します。waypoint を PID に渡してハンドル・アクセル・ブレーキへ変換し、CARLA で走らせます。学習は専門家走行データの模倣(回帰損失)です。

TPAMI 2022 版(拡張版)では、補助タスク(BEV 上のセマンティックセグメンテーション=各画素が道路か車かを塗り分ける、深度推定、HD マップ予測、信号・停止標識の状態予測など)を加えて表現を強化し、より大規模に評価しました。補助タスクは「中間特徴に正しい意味を持たせる」ためのガイドで、解釈性(中間 BEV を可視化して何を見ているか確認できる)と性能を両立させます。補助タスクを足すと、メインの waypoint 予測だけでは学びにくい「シーンの構造的理解」が強制され、結果として運転も良くなる、というのが狙いです。

## 手法の系譜と主要論文
- 初出: Prakash, Chitta, Geiger, "Multi-Modal Fusion Transformer for End-to-End Autonomous Driving", CVPR 2021。提案はマルチ解像度の Transformer attention による camera-LiDAR 融合。動機は、交差点などグローバル文脈で従来の geometric fusion(幾何的に投影して重ねる)や late fusion が安全違反(信号無視・衝突)を起こすことを実験で示し、attention なら離れた物体間の関係を捉えられるという主張です。
- 拡張: Chitta, Prakash, Jaeger, Yu, Renz, Geiger, "TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving", IEEE TPAMI 2022。補助タスクで表現を強化し大規模評価、解釈性と性能を両立。同グループはこの路線を後に幾何ベースの後継(計画特化の手法群)へ発展させていきます。
- 並走系譜: LAV(Chen & Krähenbühl, "Learning from All Vehicles", CVPR 2022)は自車だけでなく周辺車両の視点も学習に使い、データ効率と汎化を改善。InterFuser([InterFuser (安全制約付き解釈可能E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1075_interfuser), Shao, Wang, Chen et al., CoRL 2022)は TransFuser 系の融合に「安全制約付きの解釈可能な中間表現(物体密度マップ・安全マインドマップ)」を足し、危険時に明示的に介入できる安全性を加えた後継です。TCP(Wu et al., NeurIPS 2022)は軌道枝(waypoint)と制御枝(直接ハンドル・アクセル出力)を組み合わせ、両者の長所を融合しました。
- これらはすべて「Transformer でセンサ融合 → CARLA で模倣学習」という TransFuser が築いた路線の上にあり、その後の ReasonNet や Bench2Drive 時代の手法群へ続きます。

## 論文の実験結果(定量データ)
評価は主に CARLA(Town/route ベンチ、Longest6、CARLA Leaderboard)で行われます。指標の意味から。

- Driving Score(運転スコア, 0-100): 走破率(Route Completion)と違反ペナルティ(Infraction Penalty)を掛け合わせた総合点。例えば 80% 走破しても違反でペナルティ係数 0.5 なら 40 点。高いほど良い。
- Route Completion(%): ルートをどれだけ走り切れたか。高いほど良い。
- Infraction(違反): 信号無視・衝突(対車両/対歩行者/対静物)・車線逸脱などの違反回数。少ないほど良い。違反ごとにペナルティ係数が掛かる。
- これらは closed-loop(実際にシミュレータ内を走らせ、自分の行動が次の状況を生む)で測られる点が重要で、open-loop の当てっこより実運転に近い厳しい評価です。誤りが累積する compounding error まで反映されます。

CVPR 2021 版は、当時の geometric fusion や late fusion ベースラインに対し、CARLA の Driving Score と違反率で大きく上回りました。とくに「衝突せず交差点を抜ける」能力で差が出ており、報告では late fusion 系より Driving Score を明確に引き上げ、衝突・信号違反を減らしています。TPAMI 2022 版は補助タスクの追加でさらに性能と解釈性を改善し、Longest6 など長距離ベンチで強い結果を示しました。報告における核心的な結論は「attention 融合が late fusion に比べ衝突・信号違反を明確に減らす」ことです。

アブレーションの方向: (1) 融合を late fusion に落とすと、特に交差点で衝突・違反が増え Driving Score が下がる。これが論文の最重要主張で、Transformer 融合の存在意義そのものを示します。(2) 融合を行う段数を減らすとグローバル文脈の共有が弱まり性能低下。(3) TPAMI 版で補助タスク(セグメンテーション・深度)を外すと表現が劣化し性能・解釈性ともに落ちる。「複数段の attention 融合」と「補助タスク」が効くことが裏付けられます。

## メリット・トレードオフ・限界
メリット
- カメラの意味情報と LiDAR の距離情報を深い段階で結合でき、交差点など難場面に強い。
- attention により視界全体のグローバルな関係を扱える。late fusion の弱点を実証的に克服。
- 中間に BEV/補助タスクを持つので、何を見て判断したかをある程度可視化でき解釈性が高い。

トレードオフ・限界
- 全トークン間 attention はトークン数の2乗で計算が増え、推論が重い。車載リアルタイム化に最適化が要る。複数解像度で何度も融合するため、なおさら。
- LiDAR 前提はセンサコストとキャリブレーション・同期の負担を増やす。
- CARLA(シミュレータ)での強さが、そのまま実世界の安全を保証しない(sim-to-real ギャップ)。シミュレータの見た目・物理・交通エージェントの挙動は実環境と乖離があります。
- 模倣学習ゆえ、専門家データに無い分布外状況(distribution shift)での挙動保証が弱く、自分の小さな誤りが次の入力をさらに学習分布から外し誤りが累積する compounding error の問題を抱えます。
- 初期構成は前方視野中心で全周囲ではなく、視界外・側方からの割り込みに弱い面があります。

## 発展トピック・研究の最前線
- 安全制約の明示化: InterFuser のように中間に解釈可能な安全表現を置き、危険時に介入する流れ。E2E のブラックボックス性への対策です。
- 分布シフトへの頑健化: 模倣学習の compounding error を緩和する DAgger 系のデータ拡張(失敗状況を expert に補正させて再学習)や、強化学習・特権蒸留(privileged distillation、全情報にアクセスできる教師から、センサのみの生徒へ知識を移す)との併用。
- closed-loop 評価の標準化: Bench2Drive など、より体系的で多様な閉ループベンチで手法を公平比較する動き。route ごとの難度を揃えて統計的に比べます。
- カメラのみへの回帰: LiDAR コストを避けカメラだけで同等性能を狙う研究(TCP、カメラ専用 E2E)も並行して進み、融合の必要性は用途・コスト次第という議論があります。
- 実データ E2E への展開: nuScenes/nuPlan など実車データでの E2E([FusionAD (マルチセンサ融合E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1069_fusionad) のような UniAD/VAD 系)と、CARLA 系融合手法の知見をどう橋渡しするかが課題です。シミュレータで磨いた融合設計を実車で活かす流れ。

## さらに学ぶための関連トピック
- [InterFuser (安全制約付き解釈可能E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1075_interfuser)
- [DriveTransformer (統一Transformer E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1063_drive-transformer)
- [FusionAD (マルチセンサ融合E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1069_fusionad)
- [SparseDrive (スパース表現E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1084_sparse-drive)
- [GenAD (生成的軌道計画E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1070_genad_note)
