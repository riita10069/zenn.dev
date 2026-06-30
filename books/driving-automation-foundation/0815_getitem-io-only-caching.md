---
title: "__getitem__をI/Oのみにする派生配列キャッシュ"
---

# __getitem__をI/Oのみにする派生配列キャッシュ

## ひとことで言うと
学習でデータを1件取り出すたびに呼ばれる `__getitem__` は、何百万回も実行されるホットパス(高頻度経路)です。そこで重い計算(速度や曲率の算出など)はデータセット作成時に1回だけ済ませて結果をためておき(キャッシュ)、`__getitem__` ではファイルを読むだけ(I/O だけ)に絞る最適化です。狙いは、GPU をデータ待ちで遊ばせないことです。

## 直感的な理解
料理にたとえます。注文(サンプル要求)が来るたびに、毎回ゼロから野菜を切り、出汁を取り直していたら厨房(データローダ)が追いつかず、客(GPU)は料理待ちで暇になります。そこで、下ごしらえ(出汁を取る、野菜を切る)は営業前にまとめて済ませて冷蔵庫(キャッシュ)に入れておき、注文が来たら盛り付けて出すだけにする。これがこのトピックの発想です。

機械学習では、GPU は非常に高速にミニバッチを処理します。1バッチの計算が数十ミリ秒で終わるのに、その1バッチ分のデータを用意するのに数百ミリ秒かかっていたら、GPU は大半の時間アイドル(遊休)になり、高価なアクセラレータが無駄になります。データ供給(input pipeline)がボトルネックにならないようにするのは、大規模学習の基本中の基本です。`__getitem__` から無駄な計算を追い出すのは、その第一歩です。

## 基礎: 前提となる概念

`__getitem__` と DataLoader。PyTorch の `Dataset` は `__getitem__(idx)`(idx 番目のサンプルを返す)を実装します。DataLoader はこれを並列のワーカープロセスから繰り返し呼び、結果をミニバッチにまとめて GPU に渡します。1エポック(全データを1周)あたりサンプル数だけ呼ばれ、それが複数エポック繰り返されるので、`__getitem__` は学習中に何百万〜何千万回も実行されます。ここでの 1 サンプルあたりのコストが、そのままデータ供給速度を決めます。

派生配列(derived array)。生のセンサーログは、たとえば「自車の位置と向きの列(ego ポーズ)」しか持っていないことが多い。一方モデルが欲しいのは「速度・加速度・向き・曲率」といった、そこから計算で導く量です。速度は位置の時間微分、曲率はおおよそ yaw_rate / speed、というように計算します。これら計算で作る量を派生配列と呼びます(→[egomotion信号の導出（速度・加速度・ヨーレート・曲率）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0813_egomotion-signal-derivation))。

メモ化(memoization)。「結果が入力に対して不変な計算は1回だけ行い、結果を保存して再利用する」という計算機科学の基本原則です。同じ入力に対して何度も同じ計算を繰り返すのを避けます。派生配列は1つの系列(シーン)に対して内容が決まっているので、メモ化の典型的な対象です。

GPU の利用率(GPU utilization)。GPU が実際に計算している時間の割合です。データ供給が遅いと、GPU は次のバッチを待ってアイドルし、利用率が下がります(たとえば 30% など)。理想は GPU をほぼ常時動かして 90% 以上に保つことです。

## 仕組みを詳しく

典型的な実装では、データセット構築時(`__init__` の中、有効サンプルの全列挙と同じタイミング → [履歴・未来ウィンドウによるサンプル列挙](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0217_dataset-sampling-window))に、系列ごとの派生物を辞書にキャッシュします。

```python
self._scene_egomotion:     dict[str, np.ndarray]   # (T, 4) 速度・加速度・yaw・曲率
self._scene_positions:     dict[str, np.ndarray]   # (T, 2) 系列内座標(地図用)
self._scene_camera_params: dict[str, torch.Tensor] # (V, 3, 4) 射影行列
```

構築時に、各系列について1回だけ次を行います。(1) ポーズを読み、微分や角度のアンラップを使って egomotion(速度・加速度・向き・曲率)と座標を計算してキャッシュ。(2) カメラ射影行列を計算してキャッシュ(これもフレームに依存しないので1回でよい → [カメラ射影行列 P = K_scaled @ T_ref_to_cam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0802_camera-projection-matrix))。

こうしておくと `__getitem__` は「読むだけ・引くだけ」になります。

```python
def __getitem__(self, idx):
    scene_id, frame_idx = self._samples[idx]      # 事前列挙したリストから取得

    # --- ここから先に重い計算は無い ---
    ego_xy  = self._scene_positions[scene_id][frame_idx]    # キャッシュ参照
    ego_yaw = self._scene_egomotion[scene_id][frame_idx, 2] # キャッシュ参照
    image   = load_image_frame(...)                         # JPEG を読む(I/O)
    bev_map = rasterize_map_tile(...)                       # 地図ラスタライズ
    history, target = slice_window(
        self._scene_egomotion[scene_id], frame_idx)         # キャッシュをスライスするだけ
    camera_params = self._scene_camera_params[scene_id]     # キャッシュ参照
    return Sample(...)
```

ポイントは、窓の切り出しが「再計算ではなくキャッシュ済み配列のスライス」になっている点です。微分(`np.gradient`)や角度のアンラップ(`np.unwrap`、-π〜π を周回しても連続に保つ処理)はすべて構築時に済んでいます。

before/after(概念)。

```
[before] __getitem__ ごとに:
   ポーズ全読み込み + np.gradient + np.unwrap + 射影行列計算 + JPEG読込 + 地図描画
   → 微分や行列計算が毎回走る(無駄)。1サンプルあたりが重い

[after] __getitem__ ごとに:
   キャッシュ参照 + 配列スライス + JPEG読込 + 地図描画
   → 重い計算は構築時に1回だけ。ホットパスは実質 I/O のみ
```

計算量の観点では、派生配列の計算回数が「サンプル数(数百万)」から「系列数(数千)」へ落ちます。同じ系列に属する隣接サンプルは窓が重複しているため、素朴な実装だと同じ微分を何百回も繰り返しますが、キャッシュでそれが消えます。

ローダ層のキャッシュも併用するのが定石です。たとえば、系列ごとのファイルローダ取得を LRU キャッシュ(Least Recently Used、最近使ったものを残す有限キャッシュ)で包めば、同じ系列を連続して読むときに開き直しを避けられます。ただし「全系列の生データを上限なしにキャッシュする」とメモリを食い潰すので、キャッシュ対象は派生済みの小さな配列に絞り、生データは都度 I/O で読む、というメモリ配慮が重要です。

なぜこれで安全か。キャッシュが正しさを壊さない条件は「キャッシュ対象が入力に対して不変であること」です。系列のセンサーデータは学習中に書き換わらないので、構築時に計算した派生配列は最後まで正しい。これがメモ化が成立する前提です。

## 手法の系譜と主要論文

これは特定論文の新手法というより、データローディング工学のベストプラクティスです。背景となる文献を挙げます。

Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (NeurIPS 2019, arXiv:1912.01703)。`Dataset` / `DataLoader` の設計思想として「`__getitem__` は1サンプルの取得に専念し、重い処理は避ける」「複数ワーカープロセスで I/O を並列化し、GPU 計算と重ねて隠す」ことが示されています。動機は、GPU を遊ばせないために input pipeline をボトルネックにしないことです。

Murray et al., "tf.data: A Machine Learning Data Processing Framework" (VLDB 2021, arXiv:2101.12127)。TensorFlow のデータパイプラインに関する論文で、prefetch(先読み)、cache(キャッシュ)、map のベクトル化・並列化といった最適化が学習スループットに与える影響を体系的に分析しています。キャッシュ可能な前処理を一度だけ実行して再利用することの効果が定量的に論じられており、本トピックの一般化にあたります。

Aizman et al., "High Performance I/O For Large Scale Deep Learning" (2019, arXiv:1910.05256, WebDataset の背景)。大規模学習では計算より I/O 自体がボトルネックになりやすいことを指摘し、サンプルをシャード(まとめたファイル)に固めて逐次・ストリーミングで読む手法を提案しました。本トピックが「計算を消して I/O だけにする」段階だとすれば、その先に「I/O 自体を速くする」工夫が続く関係です。

## 論文の実験結果(定量データ)

tf.data の論文では、最適化を入れないナイーブな入力パイプラインに対し、prefetch・並列 map・キャッシュを組み合わせることで、実運用のモデルで学習スループットが数倍に向上する事例が報告されています。とくに、毎ステップ繰り返される決定的な前処理をキャッシュすると、CPU 側の処理が削減され、エポックあたりの時間が大幅に短縮されることが示されています。

GPU 利用率の観点では、入力パイプラインがボトルネックのとき GPU 利用率が 50% 前後やそれ以下に落ち込む事例が広く知られています。データ供給を改善して GPU 利用率を 90% 以上に引き上げると、同じハードウェアで学習時間がほぼ半分になることもあります。これは「データ供給がボトルネックなら、モデルや GPU をいくら速くしても全体は速くならない(供給律速)」という性質を表します。具体的な数値感覚として、GPU が1バッチを 30ms で処理できるのに、データ準備に 1 バッチあたり 100ms かかっていれば、全体は 100ms 律速になり GPU 利用率は約 30% に沈みます。準備を 30ms 以下まで下げて初めて GPU が律速側に回り、利用率が 90% 超になります。`__getitem__` から微分や行列計算を追い出すのは、この準備時間を削る直接的な手段です。

FFCV(Leclerc et al., "FFCV: Accelerating Training by Removing Data Bottlenecks", CVPR 2023)は、データ供給を徹底的に最適化することで ImageNet 上の ResNet-50 学習を従来比で大幅に高速化(1台の8GPUマシンで数十分台の学習)できることを示し、入力パイプラインの最適化が壁時計時間に直結することを定量的に裏づけました。

キャッシュのアブレーション的な見方として、派生配列の計算を `__getitem__` に残したまま窓を1フレームずつずらすと、隣接サンプル間で同じ系列の微分を何度も計算し直すため、計算回数がサンプル数に比例して膨らみます。これを系列数まで落とすキャッシュは、重複の多いスライディングウィンドウ設計(→[履歴・未来ウィンドウによるサンプル列挙](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0217_dataset-sampling-window))と組み合わせるほど効果が大きくなります。

## メリット・トレードオフ・限界

メリット。ホットパスから重い計算(微分・角度アンラップ・行列計算)が消え、データ供給が速くなり GPU の待ち時間が減る。重複計算がなくなり計算量が「サンプル数」から「系列数」へ減る。妥当性チェックも構築時に集約され、学習中の例外が減る。

トレードオフ・限界。第一に、キャッシュがメモリを消費します。派生配列は (T,4) や (V,3,4) と小さいので通常は無視できますが、系列数が極端に多いと常駐メモリが効いてきます。第二に、構築時(`__init__`)が重くなります。全系列のポーズ読み込みと計算を起動時にまとめて行うため、初期化に時間がかかり、学習をすぐ始められない。第三に、画像(JPEG デコード)や地図ラスタライズは依然 `__getitem__` で行う I/O・描画なので、ここが次のボトルネック候補です(→[マップのラスタ化 (semantic raster)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0826_map-rasterization))。第四に、キャッシュが効くのは「入力に対して不変な計算」だけです。ランダムなデータ拡張(毎回違う結果が欲しい処理)はキャッシュできません。

研究・実務上の発展として、メモリに乗り切らない規模では、派生配列もディスクにシャード化して事前計算し(オフライン前処理)、学習時はそれをストリーミングで読む方式に移行します。これが WebDataset などの I/O 最適化につながります。

## 発展トピック・研究の最前線

データ供給の最適化は、モデルが大きくなるほど重要性を増しています。NVIDIA DALI(GPU 上で画像デコードと拡張を行うライブラリ)、FFCV(高速データローディング、Leclerc et al.)、tf.data の自動チューニング(AUTOTUNE)など、input pipeline 専用の高速化技術が研究・実装の両面で活発です。本トピックの「不変な計算は1回だけ」という原則は、これら全ての出発点であり続けています。

## さらに学ぶための関連トピック
- [履歴・未来ウィンドウによるサンプル列挙](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0217_dataset-sampling-window)
- [カメラ射影行列 P = K_scaled @ T_ref_to_cam](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0802_camera-projection-matrix)
- [egomotion信号の導出（速度・加速度・ヨーレート・曲率）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0813_egomotion-signal-derivation)
- [マップのラスタ化 (semantic raster)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0826_map-rasterization)
