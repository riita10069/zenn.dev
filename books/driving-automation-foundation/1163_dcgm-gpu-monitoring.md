---
title: "DCGM Exporter (GPU 監視)"
---

# DCGM Exporter (GPU 監視)

## ひとことで言うと
NVIDIA の GPU が「今どれくらい働いているか・何度になっているか・メモリをどれだけ使っているか」といった健康診断データを取り出し、監視システム(Prometheus など)が読み取れる形で公開する小さな常駐プログラムです。これを入れると「借りた GPU をちゃんと使い切れているか」をグラフで継続的に見られるようになります。

## 直感的な理解
GPU は1枚で数十万円〜数百万円、クラウドで借りても時間単価が高い「貴重品」です。貴重品を借りているなら「ちゃんと働かせているか」を知りたいのが自然です。ところが GPU の中の状態は、外から見ただけでは分かりません。学習が進んでいるように見えても、実は GPU の計算ユニットの大半が遊んでいて、データの読み込み待ちで止まっている——という事態は珍しくありません。

人間が手元のターミナルで `nvidia-smi`(NVIDIA 製の状態確認コマンド)を打てば一瞬の値は見えますが、それは「健康診断を1回だけ目視する」ようなもので、何台もある GPU を24時間追い続けるには使えません。DCGM Exporter は、その健康診断データを自動で測り続け、監視システムに流し込み続ける「常時装着の心電図」のような役割を果たします。

## 基礎: 前提となる概念

「メトリクス(metrics)」とは「ある瞬間の数値の計測値」です。例: GPU 利用率97%、温度68度、VRAM 使用量42GB。これを時刻とセットで連続記録したものが「時系列データ(time series)」で、時間順に並べると「3時間前からじわじわ利用率が落ちている」といった傾向が見えます。

「Prometheus(プロメテウス)」は、監視対象が公開する HTTP エンドポイントを定期的に取りに行って(スクレイプ、scrape)、メトリクスを時系列データベースに溜める監視ツールです。「exporter(エクスポータ)」とは、ある対象の状態を Prometheus が読める形式に変換して公開する小さなプログラムの総称です。CPU/メモリ用の node-exporter、Kubernetes 状態用の kube-state-metrics などがあり、GPU 用がこの DCGM Exporter です。

「VRAM(Video RAM)」は GPU 専用の高速メモリです。学習で扱うモデルの重み・活性値・勾配・最適化状態はすべてここに載るため、足りなくなると OOM(Out Of Memory、メモリ不足によるクラッシュ)で学習が止まります。

「サーマルスロットリング(thermal throttling)」は、GPU が熱くなりすぎたときに自分を守るため自動でクロック(動作周波数)を下げる挙動です。これが起きると見かけ上は動いているのに性能が落ちます。

「Tensor Core」は、NVIDIA GPU に搭載された行列積専用の演算ユニットです。FP16/BF16/TF32 などの低精度数値で行列演算を桁違いに速くします。混合精度(mixed precision)学習はこの Tensor Core を使うことで高速化・省メモリ化を実現しますが、コードや設定によっては Tensor Core が使われず通常の演算ユニットだけで動いてしまうことがあり、その場合は速度の恩恵を取りこぼします。

「DCGM」は "Data Center GPU Manager"(データセンター向け GPU 管理)の略で、NVIDIA が提供する GPU 監視・管理ライブラリです。DCGM Exporter は、DCGM が集めた GPU メトリクスを Prometheus 形式に変換して公開する橋渡し役です。

## 仕組みを詳しく

登場人物を整理します。

- DCGM: GPU ドライバの上で動き、GPU から利用率・温度・電力・メモリ・ECC エラー数・各種プロファイリング値を取得するライブラリ
- DCGM Exporter: DCGM を呼び出してメトリクスを取得し、HTTP エンドポイント(慣例的に `:9400/metrics`)で「テキスト形式の数値一覧」として公開する常駐プログラム
- Prometheus: そのエンドポイントを定期的に(例: 15秒ごと)スクレイプして時系列保存する監視ツール

DCGM Exporter がエンドポイントに出すデータは、次のような「メトリクス名{ラベル} 値」の行が並んだプレーンテキストです(簡略化した例)。

```
# GPU 利用率(0〜100 のパーセント)
DCGM_FI_DEV_GPU_UTIL{gpu="0",modelName="NVIDIA L40S",Hostname="gpu-node-1"} 97

# GPU 温度(摂氏)
DCGM_FI_DEV_GPU_TEMP{gpu="0",modelName="NVIDIA L40S"} 68

# 使用中の VRAM(MiB)
DCGM_FI_DEV_FB_USED{gpu="0"} 41984

# 空き VRAM(MiB)
DCGM_FI_DEV_FB_FREE{gpu="0"} 3776

# 消費電力(ワット)
DCGM_FI_DEV_POWER_USAGE{gpu="0"} 312

# Tensor Core パイプの稼働率(0.0〜1.0)
DCGM_FI_PROF_PIPE_TENSOR_ACTIVE{gpu="0"} 0.42
```

`DCGM_FI_DEV_GPU_UTIL` が「メトリクス名」、`{gpu="0",...}` が「ラベル(どの GPU か・どのホストか等の付帯情報)」、末尾の数字が「現在値」です。`FI` は Field Identifier、`DEV` は device(GPU 本体)、`PROF` は profiling(細かい稼働率計測)を表します。

Kubernetes 上では DCGM Exporter を「DaemonSet(デーモンセット)」として動かすのが定番です。DaemonSet とは「ラベルで絞った各ノードに必ず1個ずつ Pod を配置する」仕組みです。GPU ノードが1台増えれば、その新ノードにも自動で Exporter Pod が立ち上がるので、監視の取りこぼしが起きません。

代表的なメトリクスと、それで分かることを並べます。

- `DCGM_FI_DEV_GPU_UTIL`(GPU 利用率%): 学習が GPU を使い切れているか。低ければデータ読み込みや CPU 前処理がボトルネックの疑い。ただし「GPU が何か動いている割合」を測るだけで、計算ユニットを目一杯使っているかまでは表しません
- `DCGM_FI_DEV_FB_USED` / `FB_FREE`(VRAM 使用/空き): あとどれだけバッチサイズや解像度を上げられるか、OOM までの余裕
- `DCGM_FI_DEV_GPU_TEMP`(温度): サーマルスロットリングの兆候
- `DCGM_FI_DEV_POWER_USAGE`(電力): 電力リミットに当たって性能が頭打ちになっていないか
- `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`(Tensor Core 稼働率): 行列演算専用ユニットを実際に使えているか。混合精度学習が効いているかを見るのに有用
- `DCGM_FI_PROF_SM_ACTIVE` / `SM_OCCUPANCY`: SM(Streaming Multiprocessor、GPU の計算コア群)がどれだけ稼働・占有されているか。GPU_UTIL より精密な「実際の計算密度」の指標

最後の2つ(PROF 系)が研究的には特に有用です。GPU_UTIL が高くても、それは「カーネル(GPU 上の処理)が1つでも走っていた時間の割合」でしかなく、計算密度が低い(SM_ACTIVE が低い)場合があります。たとえば小さなカーネルが断続的に走るだけでも GPU_UTIL は100%近くに見えることがあります。SM_ACTIVE や TENSOR_ACTIVE まで見て初めて「本当に計算を詰め込めているか」が分かります。

この違いを数値例で具体化します。ある学習ジョブで `DCGM_FI_DEV_GPU_UTIL = 99` と出ていても、`DCGM_FI_PROF_SM_ACTIVE = 0.35`、`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE = 0.08` だったとします。これは「GPU は常時何かを動かしている(利用率99%)が、平均すると SM の35%しか働いておらず、Tensor Core に至っては8%しか火が入っていない」状態を意味します。原因の候補は、(a) バッチサイズが小さすぎて並列度が出ていない、(b) 混合精度が効いておらず FP32 経路で動いていて Tensor Core を使えていない、(c) 多数の小さなカーネル起動でオーバーヘッド律速になっている、などです。GPU_UTIL だけを見ていれば「フル稼働、問題なし」と誤判断しますが、PROF 系まで見れば「実は計算機を1/3も使えていない」と分かり、バッチサイズや精度設定を見直す根拠になります。研究現場では「GPU_UTIL=高 だが SM_ACTIVE=低」のギャップこそが最初に潰すべき無駄であることが多いです。

## 手法の系譜と主要論文

DCGM Exporter は製品ツールなので論文の主題ではありませんが、背後の「GPU 利用率が想像以上に低い」という問題は、機械学習インフラの実測研究で繰り返し報告され、監視の動機を裏づけています。

- Jeon ら "Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads"(USENIX ATC 2019、Microsoft の社内クラスタ Philly の解析)。実運用の GPU クラスタでディープラーニング学習ジョブの GPU 利用率がしばしば低く、ギャングスケジューリング(複数 GPU をまとめて割り当てる必要)による断片化、ジョブ間の干渉、データローディング待ちなどが原因で資源が無駄になっていることを大規模ログ解析で示しました。これは「利用率を継続監視しないと、高価な GPU を遊ばせていることに気づけない」という DCGM Exporter 導入の直接の動機です。

- Mohan ら "Analyzing and Mitigating Data Stalls in DNN Training"(VLDB 2021)。GPU が計算待ちで止まる「データストール(data stall、データ供給が間に合わず GPU が遊ぶ状態)」が学習時間の大きな割合を占めうることを定量化し、ストールを軽減するキャッシュ/プリフェッチ手法(MinIO キャッシュ等)を提案しました。低い GPU_UTIL は「GPU の計算能力不足」ではなく「データパイプラインの遅さ」のサインであることが多い、という気づきに直結します。

- Weng ら "MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters"(NSDI 2022、Alibaba PAI クラスタ、数千 GPU・約2か月分のトレース)。本番の機械学習クラスタでも GPU 利用率の中央値が低く(報告では多くのタスクの GPU メモリ・計算利用率がともに低位に偏る)、これが GPU 共有(複数ジョブで1枚を分け合う)の強い動機になることを示しました。低利用率は Jeon ら(2019)の Microsoft クラスタだけの問題ではなく、別事業者の別クラスタでも再現する普遍的現象であることを裏づけています。

- GPU プロファイリングの粒度に関しては、NVIDIA の Nsight Systems / Nsight Compute が単発の詳細解析を担い、DCGM の PROF メトリクスは「常時の軽量な稼働率モニタリング」を担う、という役割分担が研究・実務で定着しています。Nsight が「一度だけ顕微鏡で覗く」のに対し、DCGM Exporter は「24時間バイタルを記録し続ける」位置づけで、両者は競合ではなく補完です。

これらの含意は明確で、「GPU を買った/借りただけでは使い切れていない可能性が高く、監視して初めて改善できる」ということです。DCGM Exporter はその監視を低コストで実現する標準手段です。

## 論文の実験結果(定量データ)

- Jeon ら(ATC 2019)は、Philly クラスタの数万ジョブを解析し、ジョブの GPU 利用率分布が広く、多くのジョブが GPU を十分使い切れていないこと、待ち時間(キュー待ち+スケジューリング)がジョブ完了時間に大きく寄与することを報告しました。とくに分散ジョブではノード間配置の局所性(locality)が悪いと通信オーバーヘッドで利用率が落ちることを示しています。指標の意味: GPU 利用率は「確保した計算資源のうち実際に使えた割合」を測り、これが低いほど高価な GPU を遊ばせている=コスト効率が悪いことを意味します。

- Mohan ら(VLDB 2021)は、複数の代表的な学習ワークロードで、ストレージやデータ前処理がボトルネックのとき GPU が計算を止めて待つ時間(データストール)がエポック時間の相当割合に達しうることを示し、提案手法でその stall を大幅に削減して end-to-end 学習時間を改善できると報告しました。アブレーション的に「キャッシュを外す」「プリフェッチを外す」と stall が再び増え、GPU 利用率が落ちることが示されています。これは「GPU_UTIL を監視していれば、こうしたデータパイプライン由来の無駄を発見できる」ことの実証的根拠です。

これらの研究の含意は「GPU を確保しただけでは利用率は自動的には上がらず、監視して初めてボトルネック(データストール、断片化、通信)を特定でき、改善のループを回せる」ということです。

## メリット・トレードオフ・限界

メリット:
- GPU 専用メトリクス(利用率・温度・VRAM・電力・SM 稼働・Tensor Core 稼働)を NVIDIA 公式の信頼できる経路で取得できる
- DaemonSet で全 GPU ノードを自動カバーでき、ノード追加時の取りこぼしがない
- Prometheus / Grafana という業界標準と素直に連携でき、CPU/ネットワーク等のメトリクスと同じダッシュボードに統合できる
- 「気づきにくい低 GPU 利用率」を発見でき、データパイプライン改善やバッチサイズ調整の判断材料になる

トレードオフ・限界:
- 監視自体の運用コストが増える(Exporter のデプロイ、Prometheus のスクレイプ設定、Grafana ダッシュボード作成・保守)
- メトリクスはあくまで「症状」を示すだけで、原因特定・改善は人間(またはルール/異常検知)が担う必要がある
- PROF 系(SM_ACTIVE, TENSOR_ACTIVE 等)のプロファイリングメトリクスは、計測のために GPU 側でわずかなオーバーヘッドが乗る場合がある。また同時に Nsight 等の他のプロファイラを使うと衝突することがある
- メトリクスの保存先(時系列 DB)が肥大化するため、保持期間やストレージ設計が別途必要。GPU 台数 × メトリクス種類 × スクレイプ頻度でカーディナリティ(系列数)が増える
- GPU_UTIL の解釈には注意が必要で、高くても計算密度が低い場合がある(前述)。研究では SM_ACTIVE まで見るべき、という落とし穴がある

## 発展トピック・研究の最前線

- 利用率からの自動最適化: 監視で得た低利用率を起点に、バッチサイズ自動調整、データローダのワーカー数調整、GPU 共有(MPS/MIG)への振り分けを自動化する研究が進んでいます。観測(observe)→診断(diagnose)→是正(act)のループを自動化する MLOps の方向です。
- きめ細かい GPU 共有の監視: MIG(Multi-Instance GPU、1枚の GPU を複数の隔離インスタンスに分割)を使う場合、DCGM は MIG プロファイル単位でメトリクスを出せます。共有環境で「誰がどの分割をどれだけ使ったか」を測る需要が高まっています。
- 異常検知・SLO 監視: 温度・電力・ECC エラーの異常を検知して故障を予兆する、テールレイテンシや stall 率を SLO 化して監視する、といった応用。
- コスト連携: 利用率メトリクスは [Kubecost (コスト可視化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1165_kubecost) のようなコスト配賦ツールの入力にもなり、「低利用率=金額の無駄」を直接ドルで可視化する流れがあります。

## さらに学ぶための関連トピック
- [Prometheus + Grafana](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1169_prometheus-grafana)
- [Kubecost (コスト可視化)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1165_kubecost)
- [Warm GPU Node (pause Pod)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1157_warm-gpu-node)
- [NVIDIA Device Plugin](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1180_nvidia-device-plugin)
- [Bottlerocket OS](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1100_bottlerocket-os)
