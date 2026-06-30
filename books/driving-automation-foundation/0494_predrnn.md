---
title: "PredRNN"
---

# PredRNN

## ひとことで言うと
PredRNN は「動画の次のフレームを予測する」ための再帰型ネットです。従来の ConvLSTM は記憶（メモリ）を時間方向にしか流しませんでしたが、PredRNN は記憶を「時間方向」と「層の上下方向（積み重ねた層の間）」の両方に流す Spatiotemporal LSTM（ST-LSTM、時空間 LSTM）という新しい部品を使い、動画予測の精度を大きく高めました。再帰的な時空間予測の代表格で、後続の動画予測研究の基準点になっています。

## 直感的な理解
転がるボールの動画を予測する場面を考えます。予測モデルを多層に積むと、下の層は「ボールの輪郭やテクスチャの細かい変化」を、上の層は「ボール全体がどっちに動くか」という大まかな動きを捉えます。この2つはどちらも予測に必要で、本来は連携すべきです。細部だけ見ると全体の流れを見失い、大局だけ見ると質感が崩れます。

ところが ConvLSTM を多層にすると、各層は自分の記憶（セル状態 C）を時間方向（次の時刻）にしか引き継がず、層と層の間では記憶 C が共有されません。つまり「細部の記憶」と「大局の記憶」が別々の層に閉じこもり、混ざらないのです。下層が捉えた細かな動きの手がかりが、上層の大局判断にうまく届きません。結果として予測がちぐはぐになりがちでした。PredRNN は「層をまたいで記憶を流す」新しい道を作り、細部と大局の時空間情報を一体化することでこれを解決します。

## 基礎: 前提となる概念

動画予測（video prediction）は、過去の数フレームから未来のフレームを生成するタスクです。「次に何が起きるか」を映像レベルで予想でき、自動運転・気象・ロボットの行動計画に役立ちます。教師あり学習の正解は「実際に起きた未来フレーム」そのものなので、ラベル付けが不要（self-supervised に近い）という利点があります。

ConvLSTM（[ConvLSTM](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0483_convlstm)）は LSTM の全結合を畳み込みに置き換え、空間構造を保ったまま時系列を扱う部品です。PredRNN はこれを土台に改良したものなので、ConvLSTM の理解が前提になります。

セル状態（cell state, C）は LSTM/ConvLSTM が持つ長期記憶で、入力ゲートと忘却ゲートで制御しながら時間方向に引き継がれます。忘却ゲートが「過去をどれだけ残すか」、入力ゲートが「今の情報をどれだけ書き込むか」を決めます。

層（layer）を縦に積むとは、ある層の出力を次の層の入力にして、より抽象的な特徴を段階的に作ることです。下層は局所・低次、上層は大域・高次の特徴を担う、というのが多層ネットの一般的な役割分担です。

勾配消失（vanishing gradient）とは、誤差を逆向きに伝えて重みを更新するとき、経路が長いほど勾配（更新の手がかり）が指数的に小さくなり、遠くの層やステップが学習されなくなる現象です。再帰型ネットの長年の課題で、PredRNN の改良版が直接取り組んだ問題でもあります。

## 仕組みを詳しく

### 2種類のメモリを持つ
PredRNN の中核は ST-LSTM という新しいセルで、2系統の記憶を持ちます。

1. 時間メモリ C（temporal memory）: 従来の LSTM と同じく、同じ層の中で時間方向（前の時刻 → 次の時刻）に水平に流れる記憶。各層が自分の時間的連続性を保つ。
2. 時空間メモリ M（spatiotemporal memory）: PredRNN が新しく導入した記憶。層をまたいで垂直に流れ、さらに最上層から次の時刻の最下層へとジグザグに伝わる。層間で情報を橋渡しする役割。

### ジグザグに流れる記憶（zigzag flow）
時空間メモリ M の流れ方が PredRNN の最大の特徴です。ある時刻で、最下層 → 2層目 → ... → 最上層 へと M を上へ流し、最上層まで来たら、次の時刻の最下層へ渡します。これを繰り返すと、M は「右上がりのジグザグ」を描きながら全層・全時刻を一筆書きで貫きます。

ASCII図（縦が層、横が時刻。M の流れを矢印で表す）:

```
層3(上)        ↗──→ ╲              ↗──→ ╲
層2           ↗      ╲ M          ↗      ╲
層1(下)      ↗        ╲──→(次時刻へ)╲
            時刻 t              時刻 t+1
```

これにより、低層の細かい時空間特徴と高層の抽象的な時空間特徴が、同じ記憶ストリーム M の中で出会い混ざり合います。一方で従来どおりの時間メモリ C も各層で水平に流れるので、「層をまたぐ統合（M）」と「層内の時間連続性（C）」の両方を同時に確保できます。ConvLSTM の弱点だった「層に閉じこもる記憶」が、M の縦の流れで解消されるわけです。

### ST-LSTM セルの中身
各 ST-LSTM セルは、入力ゲート・忘却ゲートを C 用と M 用にそれぞれ持ち、最後に両方の記憶を `1x1` 畳み込み（チャンネル方向の混合）で統合して隠れ状態 H を出します。ゲートの計算はすべて畳み込み（ConvLSTM と同じく空間構造を保つ）で行います。M を更新するゲートは、同じ時刻の1つ下の層から来た M を入力に使う点が、C の更新（前の時刻の同じ層から来た C を使う）と異なります。つまり M は「縦（層間・同時刻）」、C は「横（時間・同層）」と、流れる向きが直交しています。

### 形状（tensor shape）で追う
入力動画 `[B, T, C, H, W]`（B=バッチ, T=時間, C=チャンネル, H=高さ, W=幅）を時刻ごとに処理し、各層・各時刻で C、M、H をすべて `[B, C_hidden, H, W]` の空間構造を保ったまま更新します。エンコーダ部で過去フレームを読み、デコーダ部（予測部）で未来フレームを順に生成します。学習時は scheduled sampling（最初は正解フレームを入力に使い、徐々に自前の予測フレームに切り替える）で、推論時の自己回帰のずれに備えます。

## 手法の系譜と主要論文

- Wang, Long, Wang, Gao & Yu (2017), "PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs"（NeurIPS 2017）。ST-LSTM とジグザグに流れる時空間メモリ M を提案しました。動機は ConvLSTM が層ごとに記憶を分断する問題を解き、低層〜高層の時空間情報を一体化することです。

- Wang et al. (2018), "PredRNN++: Towards A Resolution of the Deep-in-Time Dilemma in Spatiotemporal Predictive Learning"（ICML 2018, arXiv:1804.06300）。PredRNN の勾配が深い時空間経路（M のジグザグ経路は層数×時刻ぶん長い）で消えやすい問題に対し、Causal LSTM（記憶の更新順序を直列化＝C を先に更新してから M を更新し、表現力を上げる）と Gradient Highway Unit（勾配の近道を作り、遠い時刻まで勾配を届ける）を追加して改良しました。これは「メモリ経路が長くなると学習が難しくなる」というトレードオフへの直接の回答です。

- Wang et al. (2022), "PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning"（IEEE TPAMI, arXiv:2103.09504、ジャーナル拡張版）。memory decoupling loss（C と M が同じ情報を重複して覚えないよう、両者の差を大きくする損失）や reverse scheduled sampling（エンコーダ側でも正解と予測の混ぜ方を逆向きにスケジュールし、学習と推論のギャップをさらに減らす手法）を加え、精度と安定性をさらに高めました。

## 論文の実験結果(定量データ)

PredRNN（2017）は複数の標準ベンチマークで評価されました。

- Moving MNIST（数字が枠内を跳ね回る合成動画。10フレームから次の10フレームを予測）: 評価指標は MSE（Mean Squared Error、ピクセル誤差の2乗平均。低いほど良い）。PredRNN は当時の ConvLSTM 系手法より MSE を明確に下げ、特に予測が進むほど（先のフレームほど）誤差の増え方を抑えられた、と報告されています。フレームあたりの MSE で ConvLSTM を二桁ポイント規模で下回ったとされます。
- KTH（人の歩く・走る・手を振るなどの動作動画）: 評価指標は PSNR（Peak Signal-to-Noise Ratio、再構成画質。高いほど良い、単位 dB）と SSIM（Structural Similarity、構造的類似度。1に近いほど良い）。PredRNN は ConvLSTM より高い PSNR/SSIM を達成し、より鮮明で構造を保った予測フレームを生成したと報告されています。

改良版 PredRNN++（2018）は、同じ Moving MNIST と KTH で PredRNN をさらに上回る MSE/PSNR/SSIM を達成し、深い時空間経路でも勾配消失を抑えて安定に学習できることを示しました。TPAMI 拡張版（2022）は memory decoupling と reverse scheduled sampling を加え、Moving MNIST・KTH に加えてレーダー降水・交通流・BAIR ロボット操作などの実データを含む幅広いタスクで一貫した改善を報告しています。

アブレーションの要点: (1) 時空間メモリ M を外して C だけにすると ConvLSTM 相当に戻り、精度が落ちる（M の寄与の確認）。(2) M のジグザグ伝播を切って層内に閉じ込めると、層間の情報統合が失われ精度が低下する。(3) PredRNN++ では Gradient Highway を外すと深い構成で学習が不安定化する。(4) TPAMI 版では memory decoupling loss を外すと C と M が冗長な情報を重複保持し、長期予測の質が落ちる。これらが「層をまたぐ記憶」「勾配経路の工夫」「記憶の役割分担」の各効果を裏づけています。

## メリット・トレードオフ・限界

メリット
- 時空間メモリ M をジグザグに全層・全時刻へ流すことで、低層の細部と高層の大域構造を一体的に扱える。
- ConvLSTM より動画予測の精度が高い（Moving MNIST・KTH などで MSE/PSNR/SSIM で実証）。
- 空間構造を保ったまま時系列を扱えるので、映像・レーダー・BEV など時空間データに向く。
- 動画予測は正解が「実際の未来フレーム」なのでラベル付け不要。大量の生動画で学習できる。

トレードオフ・限界
- 記憶ストリームが2系統に増え、計算量とメモリ消費が ConvLSTM より大きい。
- 時空間メモリの経路が層数×時刻ぶん長く、勾配が消えやすい（PredRNN++ の Gradient Highway などで改良が必要だった）。
- 再帰生成のため未来予測がぼやけやすい問題は残る（決定論的に複数の可能性を平均してしまう）。後続研究で緩和されたが原理的な限界。
- 再帰構造ゆえ並列化しにくく、長い系列では学習が遅い。Transformer 系より学習スループットで不利になりうる。

未解決・研究上の課題: 長期予測になるほど誤差が蓄積し映像がぼやける問題（error accumulation）、複数の未来が起こりうる不確実性をどう表現するか（確率的世界モデル化）、Transformer や拡散モデルとどう統合するか、が継続的な研究テーマです。

## 発展トピック・研究の最前線
PredRNN 系の再帰的時空間予測は、その後の動画予測研究（E3D-LSTM＝3D畳み込みと記憶への注意、MIM＝Memory In Memory で非定常性を扱う、PhyDNet＝物理事前知識と残差を分離）や、Transformer ベースの動画予測（VideoGPT、MaskViT など）、拡散モデルによる世界モデル（DIAMOND など）へと発展しました。自動運転では「未来状態を映像/俯瞰グリッドレベルで予測して計画に活かす」世界モデルの系譜とつながり、PredRNN はその初期の代表例として位置づけられます。空間構造を潰してベクトル化してから時間を扱う [Conv1d 時間プーリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0469_conv1d-temporal-pooling) とは設計思想が対照的で、PredRNN は「空間そのものを予測する」方向に振り切っています。

## さらに学ぶための関連トピック
- [ConvLSTM](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0483_convlstm)
- [LSTM (Long Short-Term Memory)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0490_lstm_note)
- [Conv1d 時間プーリング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0469_conv1d-temporal-pooling)
- [Dilated Causal Convolution](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0484_dilated-causal-convolution)
- [GRU (Gated Recurrent Unit)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0486_gru_note)
