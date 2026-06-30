---
title: "Deep Learning OSS Nvidia Driver AMI で GPU 環境を立ち上げる"
---

# Deep Learning OSS Nvidia Driver AMI で GPU 環境を立ち上げる

## ひとことで言うと

クラウド事業者が用意している「GPU を使う準備が全部済んだ OS イメージ(DLAMI)」を選んで仮想マシン(EC2 のような)を起動すると、NVIDIA ドライバや CUDA や PyTorch を自分でインストールせずに、ほぼそのまま GPU で機械学習を動かせます。環境構築でハマる時間をゼロに近づけ、かつ「検証済みのバージョンの組み合わせ」を固定することで実験の再現性を高める仕組みです。初学者が GPU の動かない原因(`torch.cuda.is_available()` が `False`)を半日かけて潰す、という定番の苦行を回避できます。

## 直感的な理解

GPU で深層学習を始めるのは、本来は「相性のシビアな部品を正しい順序で組み立てる」作業です。ドライバ、CUDA、cuDNN、深層学習フレームワークの4つが噛み合わないと、GPU が認識されない・起動時にドライバが読めずに固まる、といった事態になり、初心者が何時間も溶かす定番の落とし穴です。しかも厄介なことに、4つのどれか1つが少しズレただけで動かず、エラーメッセージも原因を直接は教えてくれないことが多い。

DLAMI(Deep Learning AMI)は、この組み立てが完成した状態の「設計図」を配る発想です。家を建てるのに基礎・配管・電気をゼロからやる代わりに、検査済みのモデルハウスをそのまま使うイメージ。起動した瞬間に「GPU が使える状態」になっているので、本題(モデルの実験)にすぐ取りかかれます。さらに「検査済みの組み合わせ」であることが、後で誰かが結果を再現するときの土台にもなります。

## 基礎: 前提となる概念

GPU で機械学習を動かすには、本来は次のソフトを相性の合うバージョンで揃える必要があります。

- **NVIDIA ドライバ**: OS が GPU というハードウェアを認識し制御するための部品。カーネルモジュールとして OS に組み込まれる。ドライバのバージョンは「対応する CUDA の上限」を決める。
- **CUDA(Compute Unified Device Architecture)**: NVIDIA GPU で汎用計算をするための土台。CUDA Toolkit は GPU 上でプログラム(カーネル)を走らせる仕組みとライブラリ群を提供する。
- **cuDNN(CUDA Deep Neural Network library)**: 畳み込み・プーリング・正規化など、深層学習特有の演算を GPU で高速に実行する専用ライブラリ。
- **深層学習フレームワーク(PyTorch など)**: 上記とバージョンが噛み合った、モデルを書くためのライブラリ。ビルド時にどの CUDA / cuDNN にリンクしたかが決まっている。

なぜ相性がシビアか。ドライバは「対応する CUDA の上限バージョン」を持ち、フレームワークは「ビルド時にリンクした CUDA/cuDNN のバージョン」を要求します。これがズレると、よくある失敗が次です。

```
torch.cuda.is_available()  →  False   (ドライバや CUDA がフレームワークと噛み合っていない)
起動時にカーネルパニック             (ドライバが OS カーネルと不整合)
実行時に cuDNN のバージョン不一致エラー
CUDA error: no kernel image is available for execution on the device
  (フレームワークが GPU の世代向けにビルドされていない=compute capability 不一致)
```

ここで重要な区別として、**ドライバは前方互換**(新しいドライバは古い CUDA を動かせる)を持つため、PyTorch が同梱する CUDA ランタイムは、ホストのドライバさえ十分新しければ動きます。つまり実際に手で合わせる必要があるのは主に「ドライバ ↔ フレームワークが要求する最低ドライバ版」と「フレームワーク ↔ GPU 世代(compute capability)」です。DLAMI はこれらを検証済みの組み合わせで固定して配ります。

用語を2つ。**AMI(Amazon Machine Image)** は「OS とソフトが入った、仮想マシンを起動するときのスナップショット(設計図)」のこと。**DLAMI** は、相性が検証済みのドライバ・CUDA・cuDNN・フレームワークがプリインストールされた AMI です。「OSS Nvidia Driver」という呼称は、NVIDIA がオープンソース化したカーネルモジュール版ドライバを使っていることを指します(従来のプロプライエタリ版に対し、新しい世代の GPU で推奨される)。

## 仕組みを詳しく

### ステップ1: AMI を見つける

クラウドのコンソールから GPU 向け DLAMI を名前(例: Deep Learning OSS Nvidia Driver AMI GPU PyTorch、Ubuntu 版)で検索して選ぶのが基本です。CLI から「最新版の AMI ID」を自動取得することもできます。

```bash
aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.* (Ubuntu 24.04)*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].{Name:Name, ImageId:ImageId}' \
  --output table
```

分解すると:

- `--owners amazon`: 事業者公式が出している AMI に絞る(出所不明の第三者 AMI を掴まないため。第三者 AMI はマルウェアや古い脆弱なソフトを含む恐れがある)。
- `--filters Name=name,Values=...`: 名前が「該当フレームワーク版、対象 OS」に一致するものだけ。
- `sort_by(@, &CreationDate) | [-1]`: 作成日順に並べて末尾(=最新)を1個取る。
- `[-1].{Name:..., ImageId:...}`: 名前と AMI の ID だけ表示。

これで `ami-xxxxxxxx` のような ID が得られ、起動コマンドに渡します。なぜ最新を自動取得するかというと、DLAMI は定期更新され、セキュリティ修正やドライバ更新が入るためです。ID を手でベタ書きすると古いまま固定されてしまいます。一方、厳密な再現性が必要な実験では、あえて ID を固定して同じ環境を再現する、という逆の判断もあり得ます(後述のトレードオフ)。

### ステップ2: GPU インスタンスを起動する

```bash
aws ec2 run-instances \
  --image-id <ami-id> \
  --instance-type <gpu-instance-type> \
  --subnet-id <private-subnet-id> \
  --iam-instance-profile Name=<ssm-instance-profile-name> \
  --no-associate-public-ip-address \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ml-dev}]'
```

注目すべき設計判断:

1. `--no-associate-public-ip-address`: パブリック IP を付けない。インターネットから直接アクセスできない(=攻撃面が小さい)。ではどう入るか。`--iam-instance-profile` でセッション管理エージェント(SSM のようなマネージド接続)の権限を付け、SSH の代わりにマネージドな接続経路で入る。インバウンドのポートを1つも開けずに済むのがセキュリティ上の利点で、鍵管理や踏み台ホストも不要になる。
2. `VolumeSize: 100`(GB): DLAMI 自体が CUDA・cuDNN・フレームワークなど大量のソフトを含む(数十 GB)ため、ディスクは大きめが必要。汎用 SSD(gp3 など)を 100GB 程度割り当てる。データセットも置くならさらに大きく。
3. `<gpu-instance-type>`: GPU の種類は VRAM 容量で選ぶ(後述)。

GPU インスタンスの種類は VRAM 容量で選びます。16GB クラスでは高解像度・大バッチの学習が乗らず、40GB クラス以上が要る、という判断は前段のメモリ計測(下記関連トピック)から導かれます。

### ステップ3: 動作確認

起動して接続したら、DLAMI 同梱の Python 環境を有効化(activate)し、GPU が見えているか確認します。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`True` と GPU 名が返れば、ドライバ・CUDA・cuDNN・フレームワークがすべて噛み合い、GPU 計算の準備ができたということです。さらに `nvidia-smi`(ドライバ付属のコマンド)で GPU 型番・VRAM・ドライバ版・対応 CUDA 版・現在のクロックや温度を確認できます。`torch.backends.cudnn.version()` で cuDNN 版も確認でき、これらをまとめてベンチマーク結果に残すのが再現性の作法です。

### コンテナとの使い分け

環境固定の手段には DLAMI のほかにコンテナ(Docker + NVIDIA Container Toolkit)があります。一般的な使い分けは次の通りです。

```
DLAMI:      とにかく早く GPU を1台立ち上げ、手で実験したいフェーズ向け。
            ホスト OS ごと固定するので「1台の開発機」に向く。
コンテナ:   学習・推論を本番運用・並列実行する基盤(Kubernetes など)向け。
            イメージを配布・バージョン管理でき、スケールや CI/CD と相性が良い。
            ホストにはドライバだけ入れ、CUDA/フレームワークはイメージ側で固定する。
```

両者は排他ではなく、現実には「DLAMI をホストにして、その上でコンテナを動かす」構成もよく使われます(DLAMI には NVIDIA Container Toolkit も同梱されることが多い)。開発の素早い試行は DLAMI、再現可能で配布したい本番は コンテナ、と用途で分けるのが研究コミュニティでも実務でも一般的です。

## 手法の系譜と主要論文

DLAMI はクラウドのインフラ製品で学術論文の主題ではありませんが、「環境の再現性(reproducibility)」という観点で機械学習研究と深く関わります。再現性は近年の機械学習研究で大きな問題として認識され、複数の調査論文が出ています。

- **Gundersen & Kjensmo, "State of the Art: Reproducibility in Artificial Intelligence"(AAAI 2018)**: 主要 AI 会議(IJCAI/AAAI)の論文を調査し、結果を再現するのに必要な情報(コード・データ・実験環境)がどれだけ揃っているかを定量化した。多くの論文で再現に必要な情報が欠落していること、特に実験環境(ハードウェア・ライブラリ版)の記載が乏しいことを示し、再現性危機を可視化した先駆的調査。
- **Pineau et al., "Improving Reproducibility in Machine Learning Research"(NeurIPS 2019 Reproducibility Program、後に JMLR 2021)**: 機械学習の結果が再現できない大きな原因の一つが「環境の違い(ライブラリ・ドライバ・乱数シードの差)」だと指摘し、再現性チェックリストや投稿時のコード・環境提出を制度化した。NeurIPS の投稿に再現性チェックリストを必須化した実際の制度設計につながった。検証済み環境を固定することは、この再現性問題への実務的対処の一つ。
- **NVIDIA CUDA Toolkit / Driver 互換性ドキュメント**: どのドライバがどの CUDA をサポートするかの対応表(compatibility matrix)を提供する一次情報。DLAMI はこの対応表に沿って検証済みの組み合わせを固定している。
- **環境凍結の実務文化(Docker / コンテナ、AMI、conda 環境エクスポート)**: 学術というより実務文化だが、「同じコードでも CUDA や cuDNN のバージョンが違うと、浮動小数点の丸めや非決定的なカーネル選択で計算結果や速度が変わりうる」という観測から、環境を凍結する習慣が定着した。深層学習の数値的再現性(同じシードでもビット単位では一致しない問題)に関する議論ともつながる。

なぜ重要か。論文の数値を他者が再現できなければ、その主張は検証不能になります。環境固定は「コードと結果の間にある見えない第三の変数(環境)」を潰す手段で、ベンチマークの比較可能性(下記関連トピック)を支える前提です。

## 論文の実験結果(定量データ)

このトピックは定量評価そのものではなく評価の土台ですが、再現性研究が示した定量的な現状を挙げます。

- **再現性の現状(Gundersen & Kjensmo, AAAI 2018)**: 調査対象の AI 論文のうち、実験を再現するために必要な要素(手法・データ・実験環境)がすべて揃っていた割合はごく一部にとどまった。特に「実験環境の記述」は最も欠落しやすい項目の一つで、ハードウェアやライブラリ版が書かれていない論文が多数を占めた。これが環境固定の重要性を裏付ける定量的根拠。
- **非決定性の影響**: 同じコード・同じシードでも、cuDNN のアルゴリズム選択が非決定的だと畳み込みの結果がビット単位で変わり、学習曲線が分岐することがある。`torch.use_deterministic_algorithms(True)` や `torch.backends.cudnn.deterministic=True` で抑えられるが速度が落ちるトレードオフがある。さらに環境(cuDNN 版)が違えば、決定的設定でも結果が変わりうる。つまり「シード固定」だけでは再現性は保証されず、環境固定とセットで初めて意味を持つ。
- **環境構築コストの定性データ**: ドライバ・CUDA・フレームワークを手で揃える作業は、相性ミスのたびにやり直しが発生し、初学者で半日〜数日溶かすことも珍しくない。DLAMI を使うと起動から数分で `cuda.is_available() == True` に到達でき、このコストがほぼゼロになる。

指標として明示的な数値は出にくい領域ですが、「再現できた割合」「環境構築に要した時間」「結果のビット一致性(同条件で何回回しても同じ値が出るか)」が評価軸になります。

## メリット・トレードオフ・限界

メリット

- ドライバ・CUDA・cuDNN・フレームワークのバージョン地獄を回避できる(相性検証済み)。
- 起動から数分で GPU 計算が動く。環境構築コストがほぼゼロ。
- 事業者公式提供で出所が信頼でき、定期的に更新される(セキュリティパッチ込み)。
- バージョンを固定するので、同じ AMI ID を使えば実験環境を再現しやすい。

トレードオフ・注意点

- イメージが巨大なため、ディスクを大きめ(例: 100GB)に取る必要があり、起動も少し遅い。
- 同梱フレームワークのバージョンが固定なので、別バージョンを使いたい場合は自分で仮想環境(conda / venv)を作り直す。すると DLAMI が保証した相性から外れ、自分で整合性を取る責任が戻ってくる。
- GPU インスタンスは高額。使い終わったら停止・終了しないと課金が続く。コスト管理が運用上の最重要点で、停止忘れが最大の出費要因になりがち。
- **AMI のバージョンドリフト**: 「最新の DLAMI」を毎回取ると、知らぬ間に CUDA / フレームワーク版が上がり、過去の結果と差が出ることがある。厳密な再現性が要る実験では AMI ID を固定するべきで、ここは「最新を取る利便性・セキュリティ」と「再現性」が正面から衝突する未解決のトレードオフ。実務では「開発は最新、論文/評価用は ID 固定」と使い分ける。
- **ベンダロックイン**: 特定クラウドの AMI に依存すると移植性が下がる。コンテナのほうが移植性は高く、別クラウドやオンプレに持っていける。

## 発展トピック・研究の最前線

- **コンテナベースの運用**: NVIDIA Container Toolkit でホストにはドライバだけ入れ、CUDA / フレームワークはイメージ側に固定する方式。Kubernetes 上の GPU スケジューリング、device plugin、オートスケールと組み合わせる運用が主流化している。イメージはレジストリでバージョン管理でき、再現性と配布性が AMI より高い。
- **イメージの provenance / SBOM**: イメージに含まれるソフトの一覧(Software Bill of Materials)を記録し、脆弱性追跡と再現性を両立する動き。サプライチェーン攻撃への対策としても重視される。
- **再現性メタデータの自動記録**: ベンチマーク結果に GPU 型番・ドライバ版・CUDA 版・cuDNN 版・フレームワーク版・コードのコミットハッシュ・乱数シードを一緒に残す習慣。実験管理ツール(MLflow など)がこれを自動収集する。これがないと後から「どの環境でいくつだった」が辿れない。
- **省コスト運用**: スポットインスタンス(中断ありの安価な GPU)、オンデマンドキャパシティ予約、warm node を1台維持して残りはスケールイン、といった GPU の高コストに対処する運用設計が研究・実務双方のテーマ。GPU 供給が逼迫する局面では、リージョンやインスタンス型を柔軟に変える運用も必要になる。

## さらに学ぶための関連トピック

- [VRAM の allocated と reserved の違い](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0705_torch-cuda-memory-allocated-vs-reserved)
- [GPU ウォームアップと cuda.synchronize による正確な計測](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0681_gpu-warmup-cuda-synchronize-timing)
- [レイテンシのp50/p99とジッタの読み方](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0688_latency-percentile-jitter)
- [推論速度ベンチマーク（FPS/レイテンシ計測）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0685_inference-speed-benchmark)
