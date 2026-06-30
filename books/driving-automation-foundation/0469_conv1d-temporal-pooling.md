---
title: "Conv1d 時間プーリング"
---

# Conv1d 時間プーリング

## ひとことで言うと
映像や自車の動きが「1秒に10回（10Hz）」のような細かい間隔で記録されているとき、それをそのまま全部使うと「長くて薄っぺらい履歴」になってしまいます。Conv1d 時間プーリングは、1次元畳み込み（Conv1d）を使って「複数ステップを1ステップにまとめる（時間方向のダウンサンプリング）」と同時に「1ステップあたりの特徴量を増やす（チャンネルを濃くする）」ことで、時間的には粗く・特徴的には濃い履歴に作り変える手法です。固定のプーリング（max や average）と違い、まとめ方そのものをデータから学習できるのが最大の特徴です。

## 直感的な理解
動画を10Hz で記録すると、隣り合うフレームはほとんど同じです。車載カメラで0.1秒差の2枚を見比べても、人間にはまず区別がつきません。この冗長なフレームを全部モデルに渡すと、計算が無駄に重くなるだけでなく、判断が1フレームごとにブレやすくなります。これを「判断のチラつき（decision flicker）」と呼びます。例えば信号機が見えているのに、フレームごとの微妙なノイズで「進む/止まる」が小刻みに揺れたら、運転は不安定になります。

そこで発想を変えます。過去の履歴については時間の刻みを粗くして（10Hz→1Hz）チラつきを抑える代わりに、各ステップが持つ特徴量を増やして情報の密度を保つ。これが「coarser-in-time / richer-in-feature（時間は粗く、特徴は濃く）」という設計方針です。1秒分の10フレームを「その1秒に何が起きたか」を凝縮した1本の濃い特徴ベクトルにまとめるイメージです。情報を捨てるのではなく、時間軸から特徴軸へ移し替える、と捉えると理解しやすいでしょう。

## 基礎: 前提となる概念

サンプリングレート（Hz）とは1秒あたりの観測回数です。10Hz は0.1秒ごと、1Hz は1秒ごとに1点を意味します。10Hz から1Hz への変換は、10ステップを1ステップに減らす操作で、これをダウンサンプリング（downsampling）と呼びます。

プーリング（pooling）とは複数の値を1つにまとめる縮約操作です。最大値を取る max pooling、平均を取る average pooling が定番ですが、これらは「まとめ方」が固定で、学習されません。どの特徴を残すかは事前に決め打ちされています。

ストライド付き畳み込み（strided convolution）は、フィルタをずらす歩幅（stride）を1より大きくすることで出力を間引く畳み込みです。kernel_size と stride を同じ値にすると、入力を重ならない窓に区切ってそれぞれを1点に変換する「学習可能なプーリング」になります。固定のプーリングと違い、縮約の仕方（どの特徴を残し、どう重み付けて混ぜるか）をデータから学べます。

チャンネル（channel）は各時刻が持つ特徴量の本数です。10Hz の生履歴で各ステップが少数の特徴しか持たないとき、ダウンサンプリングと引き換えにチャンネルを増やすと、時間方向で失う情報を特徴方向で補えます。Conv1d の出力チャンネル数を増やすことで、これが実現できます。

GRU（Gated Recurrent Unit）は LSTM を簡略化した再帰型ネットで、系列を順に読んで1本の隠れ状態に要約するのに使われます。詳しくは [GRU (Gated Recurrent Unit)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0486_gru_note)。

## 仕組みを詳しく

典型的な実装は2段構成です。前段が畳み込みによる時間圧縮、後段が再帰型ネットによる系列要約です。

### ステップ1: 時間圧縮（非重複 Conv1d）
入力は `[B, T, input_dim]` の過去系列（B=バッチ, T=時間ステップ数, input_dim=各ステップの特徴量の数）で、古い順→新しい順に並んでいる前提です。例えば T=64 を10Hz で取ったなら6.4秒分の履歴です。

ここで kernel_size と stride を両方とも圧縮率（subsample ratio、例えば10）に等しくした Conv1d を使います。kernel=stride=10 とは「10ステップずつの重ならない窓を順に処理する」という意味です。各窓（10フレーム＝1秒分）が出力の1ステップに圧縮されます。これがダウンサンプリング（10Hz→1Hz）です。同時に出力チャンネルを hidden_dim にすることで特徴量を保持・拡張します。

T が圧縮率の倍数でないとき、先頭（最も古い）`T % ratio` ステップを切り捨てる（left-trim）実装が一般的です。理由は、窓を「現在」に揃え、最新フレームを必ず残すためです。例えば T=64、ratio=10 なら余り4を捨てて60ステップ＝6窓にし、最新の6.0秒分を使います。古い情報を切る方が、最新情報を切るより安全だという判断です。

形状の流れ（T=64, ratio=10, GELU 活性化と LayerNorm を付けた典型例）:

```
入力        [B, 64, input_dim]
left-trim → [B, 60, input_dim]            余り4ステップ(古い側)を捨てる
transpose → [B, input_dim, 60]            Conv1dはチャンネル軸が真ん中
Conv1d    → [B, hidden_dim, 6]            kernel=stride=10 で60→6
GELU      → [B, hidden_dim, 6]            非線形活性化
transpose → [B, 6, hidden_dim]
LayerNorm → [B, 6, hidden_dim]            約1Hz・6.0秒分の6ステップ
```

GELU（Gaussian Error Linear Unit、Hendrycks & Gimpel 2016）は ReLU の滑らかな変種で、Transformer 系でよく使われる活性化関数です。LayerNorm（レイヤー正規化、Ba et al. 2016）は各サンプルの特徴を正規化して学習を安定させる手法です。これらを Conv1d の後に挟むのは、圧縮直後の特徴のスケールを揃えて後段の学習を安定させるためです。

### ステップ2: 系列の要約（GRU）
圧縮後の低レート系列 `[B, T', hidden_dim]`（T'=6）を GRU に通し、最後の隠れ状態を取り出します。これが履歴全体を1本に要約した文脈ベクトル `[B, hidden_dim]` です。圧縮で時間を6ステップまで減らしてあるので、GRU が覚えるべき系列が短くなり、長距離記憶の負担が軽くなる、という分業が成り立ちます。畳み込みは「局所的な時間圧縮」、GRU は「圧縮後の系列の時間統合」と役割が分かれています。後段を GRU ではなく Transformer や平均プーリングにする変種もありますが、圧縮で系列が短い分どれも軽く済みます。

### 複数の入力源への適用
視覚特徴（カメラから抽出した特徴）と自車運動特徴（速度・操舵などの ego-motion 特徴）の両方を扱いたいとき、両者を特徴軸で結合してから一括圧縮し、最後に元の次元へ分割して返す設計がよく使われます。

```
joint = concat([visual_history, egomotion_history], dim=-1)  # [B, T, Dv+De]
ctx   = encoder(joint)                                       # [B, Dv+De]
v_ctx = ctx[:, :Dv]    # 視覚側の文脈
e_ctx = ctx[:, Dv:]    # 運動側の文脈
```

結合してから圧縮することで、視覚と運動の時間的な対応（この映像のときこう動いていた）を畳み込みが一緒に学べる利点があります。逆に別々に圧縮してから後で混ぜる設計もあり、どちらが良いかはタスク依存です。

## 手法の系譜と主要論文

- Springenberg, Dosovitskiy, Brox & Riedmiller (2015), "Striving for Simplicity: The All Convolutional Net"（ICLR workshop, arXiv:1412.6806）。画像認識で max pooling を strided convolution に置き換えても精度が落ちない（むしろ学習可能なダウンサンプリングになる）ことを示しました。「kernel=stride の畳み込みでプーリングを代替する」発想の理論的裏づけです。プーリングは固定の縮約ですが、strided conv は縮約の仕方自体を学習できるのが利点だと示しました。

- Tran, Bourdev, Fergus, Torresani & Paluri (2015), C3D（"Learning Spatiotemporal Features with 3D Convolutional Networks", ICCV, arXiv:1412.0767）。動画を時間軸でも畳み込む 3D 畳み込みを提案し、time stride で時間方向にダウンサンプリングすることが動画表現学習に有効だと示しました。1次元の時間プーリングと地続きの発想です。

- Bai, Kolter & Koltun (2018), TCN（[TCN (Temporal Convolutional Network)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0504_tcn-temporal-conv-net), arXiv:1803.01271）。時間方向の畳み込みで系列を扱う大枠の代表です。ただし Conv1d 時間プーリングは、TCN の中核である dilation（受容野を指数的に広げる工夫、[Dilated Causal Convolution](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0484_dilated-causal-convolution)）や深い residual スタックは使わず、1層の非重複畳み込み＋GRU 要約という軽量版にとどめる構成が一般的です。長距離依存は GRU 側に任せ、畳み込みは純粋に時間圧縮に限定する、という設計判断です。

- Hu et al. (2023), UniAD（"Planning-oriented Autonomous Driving", CVPR 2023 best paper, arXiv:2212.10156）。自動運転の end-to-end 計画で、過去の知覚特徴を要約してプランナーに条件付けする設計が採られています。「過去履歴を粗い時間解像度で要約して計画に渡す」という考え方の実例で、Conv1d 時間プーリングはその「過去要約」部分の最小構成と位置づけられます。

## 論文の実験結果(定量データ)

学習可能なダウンサンプリングの妥当性について、Springenberg et al. (2015) は CIFAR-10（10クラスの小画像分類、各クラス6000枚）で、max pooling を含む標準的な CNN と、pooling を全て strided conv に置き換えた "All-CNN" を比較しました。All-CNN は CIFAR-10 で当時の最良級にあたる誤り率（テスト誤り約9%前後、データ拡張ありで7%台と報告）を達成し、pooling を無くしても精度が落ちないこと、むしろシンプルな全畳み込み構成で十分なことを示しました。これは「縮約を固定せず学習させる」方針が少なくとも害にならない、という根拠になります。

C3D（Tran et al. 2015）は、UCF101（101種類の人間行動の動画分類ベンチマーク）で、当時の手作り特徴や2D-CNN ベースより高い分類精度（約85%前後と報告）を達成し、時間方向のダウンサンプリングを含む 3D 畳み込みが動画表現に有効だと定量的に示しました。

時間プーリングそのものの効果は、動画認識や自動運転計画の文脈で繰り返し報告されています。一般論として、高頻度のフレームをそのまま全部使うより、時間方向にダウンサンプリングして特徴を濃縮した方が、(1) 計算量とメモリが大幅に減る（10Hz→1Hz なら系列長が1/10、後段の自己注意なら計算量は系列長の2乗なので約1/100）、(2) 隣接フレームの冗長性が除かれて過学習が減る、(3) フレーム単位のノイズが平均化されて判断が安定する、という効果が観察されます。アブレーションとして、圧縮を外して10Hz の生履歴を直接 GRU に入れると、系列が10倍長くなって GRU の長距離記憶が破綻しやすくなり、学習も遅くなることが知られています。逆に圧縮率を上げすぎる（例えば数秒を1ステップに）と、秒以下の素早い変化（急ブレーキの瞬間など）が窓内に埋もれて捉えられなくなる、というトレードオフがあります。最適な圧縮率はタスクの時間スケールに依存します。

## メリット・トレードオフ・限界

メリット
- 高頻度の冗長な生履歴を低頻度に圧縮し、判断のチラつきを抑えられる。
- 時間を粗くする代わりに特徴を濃く保つので、情報量の損失を抑えられる。
- 縮約の仕方を Conv1d で学習できる（固定プーリングより柔軟。重み付けや特徴選択をデータに合わせられる）。
- 系列長が短くなるので、後段の GRU/Transformer の計算量・メモリが大幅に減る。
- 1層と軽量で、既存モデルに補助的な履歴エンコーダとして安全に足しやすい。

トレードオフ・限界
- 圧縮率の分だけ時間解像度が落ち、窓内の素早い変化が埋もれる。圧縮率はタスクの時間スケールに合わせる必要がある。
- dilation を使わないため、長距離の時間依存は畳み込みではなく後段の GRU に依存し、GRU の長距離記憶の限界をそのまま引き継ぐ。
- 因果性を畳み込み構造で明示的に強制しない実装が多い（窓内の未来も使う）。過去履歴のエンコードに限れば実用上問題は小さいが、リアルタイムのストリーミング処理に拡張する際は因果畳み込み（[Dilated Causal Convolution](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0484_dilated-causal-convolution)）への置き換えなど注意が要る。
- 圧縮率や次元はハイパーパラメータで、タスクごとの調整が要る。固定窓のため、窓の境界をまたぐイベントが2窓に分断されうる。

未解決・研究上の課題: 一定の圧縮率ではなく、シーンの変化の激しさに応じて適応的に時間解像度を変える（イベントが多いときは細かく、静的なときは粗く）アプローチが検討されています。動画 Transformer では、固定窓ではなく注意で重要フレームを動的に選ぶ token merging 系の研究がこの方向を進めています。

## 発展トピック・研究の最前線
時間圧縮は、動画 Transformer での token 削減（temporal token merging、ToMe など）や、世界モデルで観測を潜在状態に圧縮する流れと地続きです。固定 stride の畳み込みから、注意機構で重要フレームを選ぶ手法、可変長へ圧縮する手法へと発展しています。空間構造を潰して特徴ベクトルにしてから時間を扱う本手法とは対照的に、空間構造を保ったまま時間を扱いたい場合は [ConvLSTM](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0483_convlstm) や [PredRNN](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0494_predrnn) が代替候補になります。どちらを選ぶかは「下流タスクが空間配置を必要とするか（BEV 予測なら必要、軌道計画なら要約ベクトルで足りることが多い）」で決まります。

## さらに学ぶための関連トピック
- [TCN (Temporal Convolutional Network)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0504_tcn-temporal-conv-net)
- [Dilated Causal Convolution](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0484_dilated-causal-convolution)
- [WaveNet](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0517_wavenet)
- [GRU (Gated Recurrent Unit)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0486_gru_note)
- [ConvLSTM](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0483_convlstm)
- [PredRNN](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0494_predrnn)
