---
title: "Triton Inference Server"
---

# Triton Inference Server

## ひとことで言うと
Triton Inference Server は NVIDIA が作った「学習済みモデルを (主に) GPU 上で高速に推論させるための専用サーバ」です。PyTorch でも TensorFlow でも ONNX でも TensorRT でも、フレームワークを問わず同じサーバに載せられ、バラバラに来る問い合わせを自動でまとめて処理する「動的バッチング」や、1枚の GPU で複数モデルを同時に走らせる仕組みによって、GPU を遊ばせず効率よく使い切ります。多くの推論プラットフォームの「中身の高性能エンジン」として採用されています。

## 直感的な理解
推論を「ただ動かす」だけなら Python で `model(x)` と書けば済みます。実験ならそれで十分です。しかしそれを「GPU を無駄なく使って大量の問い合わせをさばくサーバ」にしようとすると、難しい問題が次々に出てきます。

問題1: GPU は「1件ずつ」処理すると非常にもったいない。GPU は何千もの計算を同時に行える並列プロセッサです。画像1枚だけを推論させると GPU の演算ユニットの大部分は遊んでいます。複数の画像を「まとめて (バッチで)」処理した方が1件あたりの時間が劇的に短くなります。でも問い合わせは別々のタイミングでバラバラに来ます。これをどうまとめるか。

問題2: モデルのフレームワークがバラバラ。あるモデルは PyTorch、別のは TensorFlow、変換後は ONNX や TensorRT。それぞれ別サーバを立てるのは管理が大変です。

問題3: 複数モデルを1台の GPU で動かしたい。GPU は高価なので、1枚に複数モデルを同居させてメモリと計算を共有したい。

問題4: 待ち時間とスループットのバランス。バッチをためすぎると、最初に来た問い合わせを長く待たせてしまう (レイテンシ悪化)。

Triton はこれらを「1つの高性能推論サーバ」としてまとめて解決します。NVIDIA が GPU の性能を最大限引き出すために開発したもので、もとは TensorRT Inference Server と呼ばれていました。

## 基礎: 前提となる概念
- **推論 (inference)**: 学習済みモデルに入力を与え出力を得ること。
- **レイテンシ (latency)**: 1件の問い合わせを投げてから結果が返るまでの時間。利用者の体感速度。p50 (中央値)、p99 (遅い側1%) のように分布で語ることが多い。
- **スループット (throughput)**: 単位時間あたりさばける問い合わせ数 (件/秒)。GPU が何枚必要か = コストを決める。
- **バッチ (batch)**: 複数の入力をまとめて1つの tensor にしたもの。GPU は1件ずつより一括処理が圧倒的に得意。
- **tensor の shape**: 多次元配列の形。例: 画像バッチ `[N, 3, H, W]` は「N 枚、3チャネル (RGB)、高さ H、幅 W」。
- **カーネル (kernel) と起動コスト**: GPU 上で動く計算の一単位を「カーネル」と呼ぶ。1回呼ぶたびに CPU から GPU へ指示を出す固定の起動コストがかかる。小さな入力で何度も呼ぶより、大きな入力でまとめて呼ぶほうがこのコストが割り勘になって効率が良い。バッチングが効く本質的な理由の一つ。
- **TorchScript / ONNX / TensorRT**: PyTorch モデルを「学習フレームワークから切り離した、配布・最適化しやすい形式」に変換したもの。TorchScript は PyTorch をシリアライズした形、ONNX (Open Neural Network Exchange) はフレームワーク中立の中間表現、TensorRT は NVIDIA GPU 向けに最適化された実行エンジン。
- **量子化 (quantization)**: 計算の精度を落として (FP32→FP16→INT8 など) 高速化・省メモリ化する手法。
- **層融合 (layer fusion)**: 連続する複数の演算 (畳み込み→バイアス加算→活性化など) を1つのカーネルにまとめ、メモリアクセスとカーネル起動コストを減らす最適化。

## 仕組みを詳しく

### マルチフレームワーク対応 (バックエンド方式)
Triton は「バックエンド」という仕組みで多様なモデル形式を扱います。PyTorch (TorchScript)、TensorFlow、ONNX Runtime、TensorRT、Python (任意の Python コードをバックエンドにできる)、さらに DALI (前処理)、FIL (決定木・勾配ブースティング)、TensorRT-LLM (大規模言語モデル) など。バックエンドは共有ライブラリとして差し込まれ、Triton 本体は「モデルのライフサイクル管理・リクエストのスケジューリング・バッチング」に専念します。

モデルは「モデルリポジトリ」という決まった構造のフォルダに置き、`config.pbtxt` という設定ファイルで「入力はこの名前・型・shape、出力はこの名前・型・shape」と宣言します。例:

```
model_repository/
└── example_policy/
    ├── config.pbtxt        # 入出力定義・バッチ設定・インスタンス数
    └── 1/                  # バージョン番号
        └── model.onnx
```

`config.pbtxt` の例 (抜粋):
```
max_batch_size: 16
input  [ { name: "images" data_type: TYPE_FP32 dims: [3, 224, 224] } ]
output [ { name: "scores" data_type: TYPE_FP32 dims: [1000] } ]
dynamic_batching { max_queue_delay_microseconds: 2000 }
instance_group [ { count: 2 kind: KIND_GPU } ]
```
ここで `max_batch_size: 16` は最大16件まとめてよい、`dynamic_batching` の `max_queue_delay_microseconds: 2000` は最大2msためる、`instance_group count: 2` は同じモデルを GPU 上に2インスタンス並べる、を意味します。バージョン番号フォルダを使うことで、新旧バージョンを並行に置き、ホットスワップ (無停止での差し替え) ができます。これは Olston らの TensorFlow-Serving が整理した実務要件をそのまま実装したものです。

### 動的バッチング (dynamic batching) — 目玉機能
バラバラに来る問い合わせを、ごく短い時間だけ待ち、その間に来たものを1つのバッチにまとめて GPU に投げます。数値例:

```
問い合わせが来るタイミング (個別処理だと GPU が遊ぶ):
  t=0.0ms  リクエストA
  t=0.5ms  リクエストB
  t=1.2ms  リクエストC

動的バッチングなし: A,B,C を別々に推論 → GPU を3回起動、各≈10ms → 合計≈30ms相当の GPU 時間
動的バッチングあり (最大2ms待つ): t=2.0ms に A,B,C をまとめて [batch=3] で1回推論 → ≈12ms
```
最大待ち時間 (`max_queue_delay_microseconds`) とバッチ上限 (`max_batch_size`) は設定でき、「レイテンシをどこまで許容してスループットを上げるか」を制御します。これが問題1と問題4への答えです。注意点として、待ち時間を伸ばすほどバッチは大きくなりスループットは上がりますが、ためている間のリクエストは待たされるためレイテンシが悪化します。この綱引き点を決めるのが運用者の仕事で、付属の `perf_analyzer` でプロファイルします。

### concurrent model execution (複数インスタンス・複数モデルの同時実行)
1枚の GPU 上で、同じモデルの複数インスタンスや異なるモデルを同時に走らせます。あるインスタンスがメモリ転送 (CPU↔GPU のデータコピー) で待っている間に別のインスタンスが計算する、という形で GPU の空き時間を埋め、利用率を上げます (問題3への答え)。内部的には CUDA stream (GPU 上の独立した実行キュー) を複数使ってカーネルを並行発行します。`instance_group` の `count` を増やすとインスタンスが増えますが、各インスタンスがモデルの重みを別々に持つためメモリ消費も増える、というトレードオフがあります。

### model ensembles (モデルの連結)
「前処理モデル → 本体モデル → 後処理モデル」を Triton 内部でパイプラインとして連結できます。クライアントは1回問い合わせるだけで内部の複数段が順に実行され、段間のデータがサーバ内 (理想的には GPU メモリ上) に留まる ため、ネットワーク往復と CPU↔GPU コピーが減って効率的です。例えば「画像デコード (DALI) → 物体検出 (TensorRT) → NMS 後処理 (Python)」を1本のエンセンブルにできます。

### sequence batching と stateful モデル
状態を持つモデル (RNN や、対話のように同じセッションの連続入力を扱うもの) 向けに、同じシーケンス ID の要求を同じインスタンスに固定して順序を保つ機能もあります。これがないと、あるセッションの2番目の入力が別インスタンスに振られて状態が失われてしまいます。

### TensorRT との組み合わせ
Triton はよく TensorRT と組み合わせます。TensorRT は層融合・FP16/INT8 量子化・カーネル自動チューニング (対象 GPU で最速のカーネル実装を実測で選ぶ) を行い、推論を数倍高速化・省メモリ化します。`PyTorch → ONNX → TensorRT エンジン` へ変換して Triton に載せる、という流れが典型です。変換は一度行えばよく、推論時はコンパイル済みエンジンを実行するだけなので速い、というのがポイントです。

### 推論プロトコルと外部プラットフォームとの連携
Triton は HTTP/REST と gRPC の両方、そして Open Inference Protocol (KServe v2) にネイティブ対応します。このため、Kubernetes 上の推論プラットフォーム ([KServe (Model Serving)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0524_kserve-model-serving) 等) のバックエンドとして差し込みやすく、「プラットフォーム = 入れ物と運用・オートスケール」「Triton = 中の高性能エンジン」という役割分担になります。

```
クライアント (評価ジョブ)
   │  HTTP / gRPC (Open Inference Protocol)
   ▼
推論プラットフォーム (エンドポイント公開・オートスケール・ルーティング)
   │
   ▼
Triton (動的バッチング・複数インスタンス・GPU 上で高速推論)
   │
   ▼
モデルリポジトリ (TorchScript / ONNX / TensorRT)
```

## 手法の系譜と主要論文
- **2017年 — 推論サービングの体系化**: Crankshaw ら "Clipper" (NSDI、UC Berkeley) が適応的バッチングでレイテンシ制約下のスループットを最大化する枠組みを提示。Triton の動的バッチングはこの考え方の高性能実装です。同年 Olston ら "TensorFlow-Serving" (NIPS ML Systems Workshop、arXiv:1712.06139) が、本番でモデルを安全に差し替えながら配信する仕組み (バージョニング、ホットスワップ、複数モデル同居) を示し、推論サーバの実務要件を整理しました。
- **2018年 — 整数演算による高速化**: Jacob ら "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (CVPR、Google) が、精度をほとんど落とさず INT8 整数演算で推論できることを示しました。Triton+TensorRT の INT8 量子化はこの系譜です。
- **2018-2019年 — Triton の登場**: NVIDIA が TensorRT Inference Server を公開、後に Triton Inference Server へ改称し、マルチフレームワーク・動的バッチング・並行実行を統合しました。
- **2022年 — 生成モデル向けの反復レベルスケジューリング**: Yu ら "Orca" (OSDI) が、Transformer の自己回帰生成では「リクエスト全体ではなく1トークン生成 (iteration) 単位でバッチを組み替える」continuous batching (iteration-level scheduling) を提案。長さの違う生成リクエストを混ぜてもGPUを遊ばせない手法です。
- **2023年 — KV キャッシュのメモリ管理**: Kwon ら "vLLM / PagedAttention" (SOSP) が、生成中に溜まる KV キャッシュ (各トークンの中間表現) を OS の仮想メモリのようにページ単位で管理し、メモリの無駄 (断片化) を激減させました。Triton は TensorRT-LLM バックエンドや vLLM 連携でこれらを取り込み、巨大言語モデルの推論 (in-flight batching = continuous batching、テンソル並列) に対応しています。

## 論文の実験結果 (定量データ)
Triton はプロダクトのためそれ自体の論文ベンチマークはありませんが、中核技術の定量効果が研究で示されています。

- **適応的バッチング (Clipper, NSDI 2017)**: バッチングなしと比べて、レイテンシ SLO (例: 20ms) を守りつつスループットを数倍に向上できることを報告。意味: 同じ GPU でさばける件数が数倍 = 必要 GPU 枚数とコストが数分の一になる。代償はバッチをためる分のレイテンシ増。
- **INT8 量子化 (Jacob et al., CVPR 2018)**: MobileNet などで、FP32 と比べて INT8 推論はモデルサイズを約 1/4、レイテンシを大幅短縮しつつ、ImageNet 分類精度の低下を数 % 以内 (適切なキャリブレーションでさらに小さく) に抑えられることを示しました。意味: 「精度をほぼ保ったまま速く・軽くできる」。代償はキャリブレーション (代表データで量子化の刻み幅を決める作業) と、タスクによってはわずかな精度低下。
- **TensorRT の層融合・FP16**: NVIDIA の公開ベンチマークでは、最適化前 (素のフレームワーク) と比べて TensorRT エンジンで GPU 推論スループットが数倍〜十数倍になるケースが報告されています (モデル・GPU 依存)。意味: 推論コストとレイテンシの両方を同時に改善できる。
- **continuous batching の効果 (Orca, OSDI 2022 / vLLM, SOSP 2023)**: LLM 推論で、リクエスト単位の素朴なバッチングと比べ、Orca は大幅なスループット向上を報告。vLLM は PagedAttention により従来比でスループットを数倍 (報告では最大で約 2〜4 倍規模、設定依存) に伸ばしつつ同等のレイテンシを維持したと報告しています。意味: 生成系では「メモリの使い方」と「いつバッチを組み直すか」が性能を支配する。
- **動的バッチングのスループット曲線 (一般的傾向)**: バッチサイズを 1→2→4→8→16 と上げると、スループットは増加し続けるがある点で頭打ちになり (GPU が飽和)、同時にレイテンシは単調増加します。実務では「許容レイテンシ上限を満たす最大のバッチ」を選びます。この曲線をプロファイルする `perf_analyzer`、設定を自動探索する `model_analyzer` というツールが付属します。

数値の意味の補足: ここで繰り返し出る「スループット」は GPU 1枚で何件/秒さばけるかで、これがコストを決めます。「レイテンシ」は1件の応答時間で、利用者体験を決めます。両者はバッチサイズで綱引きの関係にあり、Triton はこの綱引き点を設定で動かせる点に価値があります。

## メリット・トレードオフ・限界
メリット:
- 動的バッチングでバラバラの問い合わせをまとめ、GPU 利用率とスループットを大きく上げる。
- PyTorch / TensorFlow / ONNX / TensorRT などフレームワーク非依存で同じサーバに載せられる。
- 1枚の GPU で複数モデル・複数インスタンスを同時実行でき、高価な GPU を遊ばせない。
- HTTP / gRPC / Open Inference Protocol 対応で、推論プラットフォーム ([KServe (Model Serving)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0524_kserve-model-serving) 等) と組み合わせやすい。
- TensorRT と組み合わせると量子化・層融合でさらに高速・省メモリにできる。`perf_analyzer` / `model_analyzer` で設定を定量的に最適化できる。

トレードオフ・限界:
- 動的バッチングは個々の応答にわずかな待ち時間を加えるため、超低レイテンシが必須の用途では `max_queue_delay` を詰める調整が要る。
- モデルを Triton 用形式 (TorchScript / ONNX / TensorRT) に変換し `config.pbtxt` を書く手間がある。変換で挙動が微妙に変わる (演算子の非対応、数値差) こともあり、変換後は元モデルと出力を突き合わせる検証が要る。
- 複数モデル同居時は資源の奪い合いでレイテンシが不安定になりうる。MIG (GPU 分割) で隔離できるが設定は複雑。
- NVIDIA GPU 前提の最適化が中心で、他社アクセラレータでは恩恵が限定的。
- 未解決の課題として、LLM の効率的バッチング (可変長・長文脈)、マルチモデル間の公平なスケジューリング、量子化による精度劣化の自動検知などが研究・開発の最前線にある。

## 発展トピック・研究の最前線
- **LLM 推論最適化**: in-flight / continuous batching、PagedAttention 風の KV キャッシュ管理、テンソル並列・パイプライン並列での巨大モデル分散推論。prefill (プロンプト処理) と decode (1トークン生成) を分離してスケジュールする disaggregation も登場。
- **量子化の進化**: INT8 を超える INT4・FP8、活性化と重みで別精度を使う混合量子化、学習時に量子化を織り込む QAT (Quantization-Aware Training)。
- **コンパイラ最適化**: TensorRT 以外にも TVM、torch.compile/Inductor、XLA などグラフコンパイラが推論を最適化する流れ。
- **GPU 共有と隔離**: MIG / MPS による細粒度共有と、それに伴うスケジューリング・QoS 保証の研究。

## さらに学ぶための関連トピック
- [KServe (Model Serving)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0524_kserve-model-serving)
- [MLflow (Tracking + Model Registry)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1125_mlflow-tracking-registry)
- [Flyte (Pipeline Orchestration)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1107_flyte-pipeline-orchestration)
- [Kubeflow Training Operator (PyTorchJob)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0605_kubeflow-training-operator)
