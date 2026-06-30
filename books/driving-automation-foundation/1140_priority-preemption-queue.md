---
title: "優先度プリエンプション (research/production の2段階制御)"
---

# 優先度プリエンプション (research/production の2段階制御)

## ひとことで言うと
GPU が限られているのに複数の学習ジョブが順番待ちしている状況で、「本番(production)用の大事なジョブ」が「研究(research)用のお試しジョブ」を一時的に押しのけて先に GPU を使えるようにする仕組みです。ジョブに優先度の名札を付け、高優先度のジョブが来たら低優先度のジョブを退避(プリエンプト)させて資源を譲ります。GPU クラスタ向けのジョブキュー管理ツール(Kueue など)が、この2段階優先度制御を提供します。

## 直感的な理解
病院の待合室を思い浮かべてください。普段は来た順(先着順)で診察しますが、救急患者が運び込まれたら、軽症の人の順番を後回しにしてでも先に診ます。これがプリエンプション(preemption)の発想です。preemption とは「実行中・確保済みの資源を、より優先度の高い者のために取り上げる」ことです。

GPU クラスタでも同じことが起きます。研究者はハイパーパラメータ探索(sweep、設定を少しずつ変えた学習を何十個も投げて良い設定を探す作業)で大量のジョブを投げます。これは「研究(research)」用で、1個1個は急ぎません。一方で「本番に出すモデルを今すぐ再学習したい」という production ジョブが割り込みで発生することがあります。これは急ぎます。

もし全ジョブを「先に投げた順(FIFO、First-In-First-Out)」だけで処理すると、研究の sweep が GPU を占有している間、急ぎの production ジョブが延々と待たされます。逆に研究ジョブを全部止めると、production ジョブが来ない時間帯に GPU が遊んでしまいます。高価な GPU を遊ばせるのは大きな損失です。

そこで「重要度(優先度)」という軸を導入します。普段は研究ジョブが GPU を自由に使ってよいが、本番ジョブが来たら研究ジョブを一時退避させて本番を先に通す。本番が終わったら研究ジョブを再開する。これが優先度プリエンプションです。

## 基礎: 前提となる概念
前提知識を噛み砕きます。

GPU クラスタとは「複数の GPU マシンをまとめて1つの計算資源プールとして扱う仕組み」です。Kubernetes はそのプールを管理する代表的な基盤で、コンテナ化されたジョブ(Pod の集まり)をどのマシンに配置するかを決めます。

スケジューラ(scheduler)とは「待っているジョブのうち、どれを、いつ、どの資源に割り当てるかを決める交通整理役」です。Kubernetes 標準のスケジューラは Pod 単位で「空いている所へすぐ置く」発想ですが、深層学習のような長時間ジョブでは別の発想が要ります。

ここで Kueue のようなジョブキュー管理ツールが登場します。Kueue は「ジョブ全体に必要な資源が空くまでジョブを待たせ(queue)、空いたらまとめて許可(admit)する」というゲート制御を Kubernetes に足します。なぜジョブ全体でまとめて見るのか。深層学習ジョブは複数の Pod(分散学習なら複数 GPU)を全部同時に確保できないと意味がない(一部だけ動いても学習できない)ことが多いからです。これを gang scheduling(ギャングスケジューリング、関係する Pod 群を全か無かで配置する)と呼びます。

quota(クォータ、割り当て枠)とは「このキューが配ってよい資源の総量」です。GPU 枠を 1 にすれば、同時に GPU を使えるのは1ジョブ分だけ、という制約になります。

## 仕組みを詳しく
典型的な構成では、2段階の優先度クラスを定義します。

```yaml
kind: WorkloadPriorityClass
metadata: { name: research-low }
value: 1000
description: "研究/sweep ジョブのデフォルト優先度"
---
kind: WorkloadPriorityClass
metadata: { name: production-high }
value: 10000
description: "本番学習の高優先度 (research をプリエンプトする)"
```

WorkloadPriorityClass とは「ジョブ(Kueue の用語では Workload)に付ける優先度の名札」です。`value` が大きいほど優先されます。研究を 1000、本番を 10000 と10倍差を付けることで、本番が来たら必ず研究を押しのけられるようにします。

次にキューとプリエンプションの設定です。

```yaml
kind: ClusterQueue
metadata: { name: gpu-cq }
spec:
  resourceGroups:
    - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
      flavors:
        - name: gpu-flavor
          resources:
            - { name: "nvidia.com/gpu", nominalQuota: "1" }   # GPU 枠は全体で1
            - { name: "cpu",    nominalQuota: "14" }
            - { name: "memory", nominalQuota: "120Gi" }
  preemption:
    withinClusterQueue: LowerPriority   # 同キュー内で、より低優先度ジョブを退避してよい
    reclaimWithinCohort: Never
```

肝は `preemption.withinClusterQueue: LowerPriority` です。これは「同じキューの中で、自分より低い優先度のジョブを押しのけてよい(プリエンプトしてよい)」という意味です。

動作を時系列で追います(GPU 枠が1の場合)。

```
時刻 t0: 研究ジョブ R (research-low, value=1000) が GPU を確保して学習中。GPU 枠 1/1 使用。
時刻 t1: 本番ジョブ P (production-high, value=10000) が投入される。GPU 枠は空いていない。
         → Kueue は「P のほうが優先度が高い」と判断。
時刻 t2: Kueue が R をプリエンプト(退避)。R の Pod は止められ、キューへ戻され待機状態に。
         → GPU 枠が空く。
時刻 t3: Kueue が P を admit(許可)。P が GPU を確保して学習開始。
時刻 t4: P が完了して GPU 枠が空く。
         → 待機していた R が再び admit され、学習を再開(または再実行)。
```

`reclaimWithinCohort: Never` は「複数のキューが資源を貸し借りするグループ(cohort)をまたいで横取りはしない」設定です。単一キュー構成なら「キューをまたいだ取り合いはしない」と理解すれば十分です。

退避された研究ジョブがどう再開されるかが重要な注意点です。プリエンプトは Pod を一旦止めるので、対策なしだと「最初からやり直し」になります。これを避けるには研究ジョブ側で checkpoint(途中保存)を定期的に取り、再開時にそこから続けられるようにします。ここで永続ストレージ([オブジェクトストレージによるデータレイク (datasets/checkpoints/artifacts)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0835_s3-data-lake) の checkpoints 領域)が効いてきます。深層学習ジョブは「ミニバッチ単位で安全に中断・再開できる」という性質を持つので、プリエンプションと相性が良いのです。

## 手法の系譜と主要論文
クラスタスケジューリングと優先度プリエンプションは、システム分野で長く研究されてきたテーマです。

Verma らの「Large-scale cluster management at Google with Borg」(EuroSys 2015)は、Google 社内の大規模クラスタ管理システム Borg を解説した論文で、本トピックの源流の1つです。Borg は production(本番、応答性が重要なサービス)と batch(バッチ、低優先度の計算)のジョブを優先度バンドで区別し、低優先度ジョブをプリエンプトして高優先度ジョブにリソースを譲る仕組みを持ちます。設計の狙いは「高優先度の応答性を保ちつつ、空きリソースを低優先度ジョブで埋めて稼働率(utilization)を上げる」こと。トレードオフは、プリエンプトされた低優先度ジョブの再実行コストや、頻繁な退避によるスループット低下です。Kubernetes の優先度・プリエンプションの設計はこの系譜を継いでいます。

GPU 学習クラスタに特化した例として Xiao らの「Gandiva: Introspective Cluster Scheduling for Deep Learning」(OSDI 2018)があります。深層学習ジョブの特性 — checkpoint 可能、ミニバッチ境界で安全に中断できる、初期にだけ大量の試行が捨てられる(early feedback)— を積極的に利用し、ジョブのマイグレーション(別 GPU への移動)、タイムスライス(GPU を時分割して複数ジョブで共有)、packing(複数ジョブを1 GPU に詰める)を行って、優先度の高いジョブに素早く GPU を渡しつつ全体のスループットを上げる手法を提案しました。

その後 Tiresias(NSDI 2019)、Themis(NSDI 2020)、Pollux(OSDI 2021)などが、ジョブの残り時間予測、公平性(fairness)、スループットと公平性を同時最適化する適応的資源配分へと発展させました。Pollux は「バッチサイズと学習率をジョブの進み具合に応じて動的に調整しつつ資源を割り当てる」co-adaptive な手法で、goodput(有効スループット、無駄にならない学習進捗)という指標を導入した点が新しいところです。

## 論文の実験結果(定量データ)
Borg の論文では、production と batch を混在させプリエンプションで調停することで、クラスタの平均稼働率を高く保てることが報告されています。重要な観察は「production だけに合わせて資源を確保すると、ピークに備えた余剰がアイドル時に遊ぶ。その隙間を batch で埋め、必要なら退避させる」ことでクラスタ全体の効率が上がる点です。プリエンプションの発生率と再実行コストはトレードオフとして測定対象になります。

Gandiva の論文では、深層学習ジョブの待ち時間(特にハイパーパラメータ探索の早期フィードバックを得るまでの時間)を、introspective なスケジューリングで大幅に短縮できることを示しました。報告によると、複数ジョブを GPU で時分割・パッキングすることでクラスタ実効スループットがおよそ 26% 程度向上した、というのが代表的な数値です(ワークロード依存)。ここで効いている指標は「ジョブの平均完了時間(JCT, Job Completion Time)」と「GPU 利用率」で、両者を同時に改善できる点が貢献です。

Tiresias は、ジョブの実行時間が事前に分からない現実を踏まえ、優先度を動的に下げていく(Gittins index 近似や Least-Attained-Service)スケジューリングで、平均 JCT をベースラインに対して大きく短縮できることを示しました。Pollux は goodput を最適化することで、固定設定のスケジューラに比べ平均 JCT を 37〜50% 程度短縮できたと報告しています(クラスタとワークロード次第)。これらはいずれも「単純な FIFO や固定優先度より、ジョブ特性を見て適応的に資源を配ると、待ち時間が大幅に減る」という共通のメッセージを定量的に裏づけています。

## メリット・トレードオフ・限界
メリット
- GPU が限られていても、急ぎの本番ジョブを研究ジョブより先に通せる。
- 普段は研究ジョブで GPU を埋めておけるので稼働率が上がる(GPU を遊ばせない)。
- 優先度はジョブに名札(WorkloadPriorityClass)を付けるだけで、ジョブのコードを変えなくてよい。
- gang scheduling と組み合わせれば、分散学習の「一部 Pod だけ起動して詰まる」事故も防げる。

トレードオフ・限界
- プリエンプトされた研究ジョブは中断され、checkpoint がないと最初からやり直しになる。再実行コストが無駄になりうる。
- 本番ジョブが頻繁に来ると研究ジョブが進まない「飢餓(starvation)」が起きうる。優先度設計と投入頻度のバランスが要る。
- 2段階(research/production)とシンプルなぶん、締め切り考慮や公平性保証のような細かい制御はできない。
- 優先度はあくまで「順番」の制御であって、GPU 台数不足そのものは解決しない。同時実行数は物理的な枠で頭打ちになる。
- プリエンプションのタイミング次第で、ほぼ完了直前のジョブを殺してしまう無駄が起きうる(チェックポイント間隔の設計が重要)。

研究上の未解決課題としては、再実行コスト・公平性・締め切り・エネルギー効率を同時に最適化する汎用スケジューラの設計があります。また、ジョブの残り時間を正確に予測できないことが多くの手法の根本的な難しさで、予測誤差に頑健なスケジューリングが今も研究の焦点です。

## 発展トピック・研究の最前線
- ジョブ実行時間予測に基づく適応スケジューリング(Tiresias, Pollux)と、予測不要で頑健に動かす手法の対比。
- GPU 共有(MPS / MIG / 時分割)で1枚の GPU を複数ジョブで安全に共有し、プリエンプションの粒度を細かくする方向。
- 公平性(fairness)とスループットのトレードオフを定式化した手法(Themis の finish-time fairness など)。
- 大規模分散学習における弾力的(elastic)学習: 利用可能な GPU 数の増減に合わせてジョブが動的にスケールし、プリエンプションを「殺す」のではなく「縮小する」で吸収する。
- スポットインスタンス(突然回収されうる安価な計算資源)上での学習をプリエンプション耐性のある checkpoint 戦略で安全に回す研究。

## さらに学ぶための関連トピック
- [オブジェクトストレージによるデータレイク (datasets/checkpoints/artifacts)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0835_s3-data-lake)
- [Ingest Adapter (データソース取込みの共通インタフェース)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0816_ingest-adapter-protocol)
- [Activation Recomputation (selective gradient checkpointing)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0155_activation-recomputation)
- [CPU/NVMe Offload](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0164_cpu-offload)
