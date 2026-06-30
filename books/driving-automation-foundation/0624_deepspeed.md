---
title: "DeepSpeed"
---

# DeepSpeed

## ひとことで言うと
DeepSpeed（ディープスピード）は、Microsoft が公開した「大きなニューラルネットワークを、限られた台数の GPU でも学習できるようにする道具箱（ライブラリ）」です。中心となるアイデアは ZeRO（ゼロ）と呼ばれるメモリ節約の仕組みで、複数の GPU が「同じものを重複して抱えている無駄」を取り除き、互いに分担して持ち合うことで 1 枚あたりのメモリ使用量を劇的に減らします。これにより、本来なら数十枚必要だった巨大モデルを、ずっと少ない枚数で学習できるようになります。さらに、足りない分を CPU メモリや SSD に逃がす機能（offload）や、計算そのものを速くする最適化カーネルも束ねています。一言でいえば「巨大モデルを現実的な台数で回すための、メモリと通信と計算の総合最適化スイート」です。

## 直感的な理解
4 人で同じ分厚い辞書を 1 冊ずつ全員が抱えて持ち歩く状況を想像してください。重い辞書を 4 冊ぶん全員が持っているのは明らかに無駄です。もし「あなたは A〜F の章、私は G〜M の章」と分担して 1 人 1 冊の 1/4 だけ持ち、必要なページが来たら持っている人に聞けば、1 人あたりの荷物は 4 分の 1 で済みます。

GPU を使ったモデル学習でも、まったく同じ無駄が起きています。素朴な「データ並列」という方式では、GPU 全員がモデルの重み・勾配・オプティマイザの内部状態という巨大な荷物を丸ごとコピーして抱えます。GPU を 8 枚に増やしても、8 枚それぞれが同じ巨大な荷物を抱えたままなので、1 枚に乗らない巨大モデルは何枚 GPU を足しても乗りません。ここが分散学習で多くの人がつまずく直感の落とし穴です。「GPU を増やせばメモリも増えるはず」と思いがちですが、素朴なデータ並列ではメモリは増えず、計算スループット（さばける枚数）だけが増えます。ZeRO は「全員が同じものを持つのをやめ、分担して持ち合おう」という発想で、この壁を壊し、GPU を増やすと本当に 1 枚あたりのメモリが減るようにします。

## 基礎: 前提となる概念
深く理解するために、まず「学習で何がメモリを食うのか」を分解します。

ニューラルネットワークの学習は次を繰り返します。(1) データを入力して予測を出す（順伝播 forward）、(2) 予測と正解のズレ（損失 loss）を計算する、(3) 各パラメータを少し変えると損失がどう変わるかという「勾配 gradient」を計算する（逆伝播 backward）、(4) 勾配を使ってパラメータを少し更新する（オプティマイザ step）。

このとき GPU メモリを占有するのは大きく次の 4 種類です。

- パラメータ（重み）本体: モデルが学習する数値そのもの。
- 勾配: 各パラメータに対する損失の傾き。パラメータと同じ個数だけある。
- オプティマイザ状態: Adam（アダム、最もよく使われる更新アルゴリズム）は各パラメータについて「過去の勾配の移動平均（1 次モーメント、momentum）」と「勾配の 2 乗の移動平均（2 次モーメント、variance）」の 2 つを保持する。
- 活性化（activation）: 順伝播の途中結果。逆伝播のときに使うので保持される。バッチサイズ・系列長・層数に比例して膨らむ。

混合精度学習（計算は 16bit の半精度 float16/bfloat16 で速く行い、更新の正確さのために 32bit のパラメータのコピーを別途持つ手法）で Adam を使う典型構成では、1 パラメータあたりのメモリは次のように積み上がります。これは ZeRO 論文（Rajbhandari ら 2020）が「モデル状態（model states）」と呼んで定式化した内訳そのものです。

```
パラメータ本体 (fp16)        : 2 バイト
勾配 (fp16)                  : 2 バイト
fp32 パラメータのコピー       : 4 バイト  ┐
Adam 1次モーメント (fp32)     : 4 バイト  ├ オプティマイザ状態 = 12 バイト
Adam 2次モーメント (fp32)     : 4 バイト  ┘
----------------------------------------
合計                         : 16 バイト/パラメータ（活性化は別）
```

パラメータ 75 億個（7.5B）のモデルなら、活性化を除いても 16 × 7.5e9 ≈ 120GB。45GB の GPU 1 枚には到底乗りません。注目すべきは、最大の犯人が「オプティマイザ状態（12 バイト）」であって、パラメータ本体（2 バイト）ではないことです。素朴な直感だと「重みが重い」と思いがちですが、実は Adam の内部状態とその fp32 コピーが全体の 4 分の 3 を占めます。ここが ZeRO の攻めどころになります。

ここで「データ並列（Data Parallelism, DP）」を確認します。同じモデルを全 GPU にコピーし、各 GPU が違うミニバッチを処理し、計算した勾配を全 GPU で平均（all-reduce、[NCCL (集合通信)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0643_nccl-collective-comm) 参照）して、全員が同じ更新を適用する方式です。実装が簡単でスケールしやすい一方、上記の 16 バイト/パラメータを全 GPU が重複して持つため、メモリ効率が極端に悪いという欠点があります。対照的に「モデル並列（テンソル並列 [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0659_tensor-parallelism) / パイプライン並列 [Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0646_pipeline-parallelism)）」はモデル自体を割って各 GPU に分散するためメモリは節約できますが、通信が重く実装が難しい。ZeRO の動機はまさに「データ並列の単純さ・計算効率を保ったまま、モデル並列のメモリ効率を手に入れる」ことでした。

## 仕組みを詳しく

### ZeRO の 3 段階
ZeRO（Zero Redundancy Optimizer、冗長性ゼロのオプティマイザ）は、データ並列の「重複保持」を段階的に取り除きます。GPU が N 枚あるとします。各段階で「何を 1/N に分割（shard、シャード化）するか」が異なります。

- ZeRO Stage 1 (P_os): オプティマイザ状態（12 バイト/パラメータ）を N 分割。各 GPU は 1/N だけ持つ。最大の犯人を分割するので効果が大きい。
- ZeRO Stage 2 (P_os+g): 上に加えて勾配（2 バイト）も N 分割。
- ZeRO Stage 3 (P_os+g+p): さらにパラメータ本体（2 バイト）も N 分割。各 GPU はモデルの一部しか常時保持しない。これは PyTorch の FSDP（後述）と同じ発想。

メモリの目安を式で書きます。混合精度 Adam の 1 パラメータあたり 16 バイトのうち、分割される量が段階で変わります（記号 N は GPU 枚数）。

```
方式            1パラメータあたりメモリ
素朴なDP        16 バイト（全部重複）
Stage 1         4 + 12/N バイト（fp16重み2+勾配2 は重複、状態12を分割）
Stage 2         2 + (2+12)/N バイト（fp16重み2 だけ重複）
Stage 3         16/N バイト（全部分割）
```

具体例: 120GB のモデルを GPU 8 枚で動かす場合、素朴な DP では各 GPU 120GB（不可能）。Stage 1 では 4 + 12/8 = 5.5 バイト/param ≒ 41GB。Stage 2 では 2 + 14/8 = 3.75 バイト/param ≒ 28GB。Stage 3 では理論上 16/8 = 2 バイト/param ≒ 15GB まで下がり、45GB GPU に余裕で乗ります。N（枚数）を増やすほど 1 枚あたりが軽くなるのが ZeRO の本質です。N=64 の Stage 3 なら 120/64 ≒ 1.9GB まで縮みます。

ただしタダではありません。Stage 3 では、ある層を計算する瞬間にはその層の全パラメータが手元に必要です。分担して持っているので、計算直前に他の GPU から「この層のパラメータを集める（all-gather）」通信が走り、計算が終わると手放します（free）。逆伝播でも各層のパラメータを再び all-gather し、勾配を集約する reduce-scatter が走ります。つまりメモリを「追加の通信」と引き換えに節約しています。通信量の理論を押さえると、Stage 1/2 のデータ並列はステップあたり約 2×(モデルサイズ) の通信（all-reduce 相当）ですが、Stage 3 は順伝播の all-gather + 逆伝播の all-gather + reduce-scatter で約 3×(モデルサイズ) と通信量が 1.5 倍に増えます。ここが本質的なトレードオフであり、GPU 間ネットワークが細いと Stage 3 の速度が頭打ちになる根本原因です。

### 活性化メモリと checkpointing の併用
ZeRO が削るのは上記「モデル状態」の部分で、活性化（activation）は別問題です。系列長の長い Transformer ではむしろ活性化がメモリ支配的になることもあります。DeepSpeed は activation checkpointing（順伝播の途中結果を一部だけ残し、逆伝播時に必要になったら再計算する手法。メモリと計算を交換する）を ZeRO と組み合わせられます。ZeRO 論文では活性化分割（partitioned activation checkpointing）も提案され、活性化もテンソル並列の軸で分割して重複を除けることを示しました。

### ZeRO-Offload と ZeRO-Infinity
ZeRO だけでも N を増やせばメモリは減りますが、GPU が少ないとそもそも限界があります。そこで DeepSpeed は「GPU メモリに乗り切らない分を、より大きいが遅い記憶階層に逃がす（offload）」機能を持ちます。

- ZeRO-Offload（Ren ら 2021）: オプティマイザ状態と fp32 パラメータコピーを CPU の主記憶（RAM、GPU メモリより桁違いに大きいが遅い）に置き、Adam の更新計算自体も CPU で行う。GPU は順伝播・逆伝播に専念できる。賢い点は「計算グラフのどの部分を CPU/GPU どちらに割り当てるか」を最小通信量の観点で定式化し、forward/backward は GPU、parameter update は CPU、という分担が通信最小であることを理論的に導いたこと。さらに CPU 側に最適化した高速 Adam 実装（CPUAdam、SIMD/マルチスレッド利用）を用意し、CPU 更新が律速にならないよう工夫した。GPU 1 枚でも数十億パラメータ級を学習可能にした。
- ZeRO-Infinity（Rajbhandari ら 2021）: さらに NVMe（高速 SSD）まで使い、GPU メモリ → CPU RAM → NVMe という記憶階層を総動員する。理論上は GPU 1 枚でも兆パラメータ級を視野に入れる。鍵は「帯域中心設計（bandwidth-centric partitioning）」と積極的な prefetch（次に要るデータを先読みして転送を計算の裏に隠す）、転送の重複（overlap）で、NVMe の遅さを実効的に隠す。

記憶階層の速度差を直感的に押さえておきます。GPU メモリ（HBM）は毎秒 1〜3 TB、CPU RAM は数十〜百数十 GB/s、NVMe は数 GB/s 程度です。下に逃がすほど容量は増えるが転送が遅くなる、というのが offload のトレードオフの根です。NVMe は HBM より 2〜3 桁遅いため、prefetch で隠しきれないと GPU が待たされ TFLOPS が落ちます。

### ZeRO++ — 通信を削る方向の進化
通信が Stage 3 のボトルネックになる問題に対し、ZeRO++（Wang ら 2023, arXiv:2306.10209）は 3 つの通信圧縮を提案しました。(1) qwZ: all-gather するパラメータを int8 に量子化して通信量を半減、(2) hpZ: パラメータをノード内に二次的に複製して逆伝播の all-gather をノード内（高速 NVLink）に閉じ込め、ノード間通信を消す、(3) qgZ: 勾配の reduce-scatter を量子化通信に置き換える。これにより低帯域環境や小バッチ時の通信量を最大 4 分の 1 に削減し、スループットを大きく改善したと報告しています。これは「ZeRO は通信が重い」という弱点への直接的な回答です。

### 道具箱としての DeepSpeed
DeepSpeed は ZeRO だけのライブラリではありません。

- 3D 並列の統合: データ並列 ＋ パイプライン並列（[Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0646_pipeline-parallelism)）＋ テンソル並列（[Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0659_tensor-parallelism)）を組み合わせて、巨大モデルを多次元に分割する。Megatron-DeepSpeed として大規模言語モデルの学習に使われた。各軸の使い分けは「テンソル並列は通信が重いのでノード内 NVLink に閉じる、パイプライン並列はノード間、データ並列＝ZeRO で外側」というのが定石。
- 最適化カーネル: 複数の演算を 1 つにまとめた fused Adam、効率的な Attention 実装、transformer kernel などで計算自体を速くする。
- 大規模推論（DeepSpeed-Inference / DeepSpeed-MII）: 学習済み巨大モデルを少ない GPU で動かす最適化（テンソル並列推論、KV キャッシュ管理）。
- 導入のしやすさ: 既存の PyTorch コードに対し `model_engine, optimizer, _, _ = deepspeed.initialize(model=model, ...)` でモデルとオプティマイザを包み、Stage や offload の設定を JSON（ds_config）で渡すだけで導入できる。コードの大改修が要らないのが普及した一因。

### FSDP との関係
PyTorch 本体には FSDP（Fully Sharded Data Parallel）という、ZeRO Stage 3 と発想がほぼ同じ機能が標準で入っています（パラメータ・勾配・オプティマイザ状態を分割保持し、計算直前に all-gather）。そのため DeepSpeed は「FSDP の代替または補完」として語られます。選択基準は、CPU/NVMe offload や 3D 並列の作り込み、推論最適化まで欲しいなら DeepSpeed、PyTorch 標準で完結させたく追加依存を避けたいなら FSDP、というのが一般的な目安です。Hugging Face Accelerate などの上位ライブラリはどちらも切り替えて使えるようになっています。

## 手法の系譜と主要論文
- ZeRO（Rajbhandari ら, "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", SC 2020, arXiv:1910.02054, Microsoft）: データ並列の冗長保持を分割で取り除く中核アイデア。動機は「データ並列はスケールしやすいがメモリ効率が悪く、モデル並列は通信が重く実装が難しい。両者の良いとこ取りをしたい」。新規性は、メモリ削減を実現しつつデータ並列の単純さと計算効率を保った点。
- DeepSpeed システム論文（Rasley ら, "DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters", KDD 2020）: ZeRO を含む学習システム全体を実用ライブラリとして提示。
- ZeRO-Offload（Ren ら, USENIX ATC 2021, arXiv:2101.06840）: 計算と状態を CPU に賢く配分（CPU には更新、GPU には順逆伝播）し、転送と計算を重ねて隠す。動機は「潤沢な GPU を持たない研究者に大規模学習を開放する（democratizing）」。
- ZeRO-Infinity（Rajbhandari ら, SC 2021, arXiv:2104.07857）: NVMe まで階層化し、帯域中心分割・prefetch・重複転送で速度低下を緩和。
- ZeRO++（Wang ら, 2023, arXiv:2306.10209）: 量子化通信とノード内複製で Stage 3 の通信量を削減。低帯域・小バッチ環境を救う。
- PyTorch FSDP（Zhao ら, VLDB 2023, arXiv:2304.11277, Meta）: Stage 3 と同等の分割発想を PyTorch コアに統合し、標準機能化した。
- Megatron-LM（Shoeybi ら, arXiv:1909.08053; Narayanan ら, SC 2021, arXiv:2104.04473）: テンソル並列・パイプライン並列の系譜で、DeepSpeed と組み合わせた Megatron-DeepSpeed が超大規模 LLM 学習（Megatron-Turing NLG 530B など）の標準構成になった。

系譜を一言でまとめると、「データ並列のメモリ無駄を削る（ZeRO 2020）→ 足りない分を CPU/NVMe に逃がす（Offload/Infinity 2021）→ 増えた通信を量子化で削る（ZeRO++ 2023）」という、メモリ・通信のトレードオフを順に潰してきた流れです。並行して PyTorch がその核を標準機能（FSDP）に取り込み、Megatron 系のモデル並列と組み合わさって超大規模 LLM 学習のデファクト構成が固まりました。

## 論文の実験結果（定量データ）
ここで指標の意味を補足します。大規模学習では「学習できる最大モデルサイズ」「GPU あたりのスループット（TFLOPS、1 秒あたり浮動小数点演算数。GPU の理論ピークに対しどれだけ引き出せたか）」「スケーリング効率（GPU を N 倍にしたとき速度が何倍になるか）」が主要指標です。

- ZeRO 論文（SC 2020）の報告: ZeRO により、当時のデータ並列が扱えた数十億パラメータ級から、100B（1000 億）超のモデルを学習可能にした。400 GPU（V100）で 100B パラメータのモデルを学習し、GPU あたり約 38 TFLOPS（V100 のピーク約 125 TFLOPS に対しおよそ 3 割）、ほぼ線形（super-linear に近い）なスケーリングを示したと報告。Stage を上げるほど扱えるモデルが大きくなることをメモリ内訳の理論計算とともに示した。アブレーション的な観点では、Stage 1 だけでもオプティマイザ状態が最大の犯人なので約 4 倍のメモリ削減が効く一方、Stage 3 まで進めるとパラメータも分割され通信が 1.5 倍に増えるという段階的トレードオフが定量化されている。
- ZeRO-Offload 論文（ATC 2021）の報告: 単一の V100 GPU（32GB）で 13B パラメータのモデルを学習可能にした。これは offload なしの同条件（約 1.4B が上限）に対し約 10 倍。複数 GPU でも 128 GPU で 70B 級まで拡張。「CPU に状態を置くと遅くなる」懸念に対し、計算配分と通信のオーバーラップ、および CPUAdam により単一 GPU で 40 TFLOPS 前後を維持したと報告。これは offload のナイーブ実装（GPU が CPU の更新完了を待つ）に対する大きな改善で、PyTorch 単体に比べ単一 GPU で最大約 10 倍のスループットを報告。
- ZeRO-Infinity 論文（SC 2021）の報告: 単一 DGX-2 ノード（16 GPU）で 1 兆（1T）パラメータ級、512 GPU で 32T パラメータ級の学習が原理的に可能と示した。NVMe の遅さは prefetch と帯域重ね合わせで隠し、512 GPU で 25 PetaFLOPS 級（ピークの約 4 割の実効効率）の総スループットを報告。GPU 数に対しほぼ超線形のスーパースケーリングも観測。
- ZeRO++ 論文（2023）の報告: 低帯域クラスタや小グローバルバッチの設定で、通信量を最大 4 分の 1 に削減し、ZeRO-3 比で最大約 2.4 倍のスループット向上を報告。通信律速の現実環境で効くことを示した。
- PyTorch FSDP 論文（VLDB 2023）の報告: 175B パラメータの GPT 規模モデルを多数 GPU でほぼ線形にスケールさせ、A100 クラスタで GPU あたり高い利用率（実装やモデルにより 100〜180 TFLOPS 級、A100 ピーク 312 TFLOPS に対し 3〜6 割）を達成。DeepSpeed ZeRO-3 と同等の機能・性能を PyTorch ネイティブで実現できることを示した。

数値の意味を初心者向けに補うと、TFLOPS が大きいほど「買った GPU の馬力を無駄なく引き出せている」ことを表します（ピークの 3〜5 割引き出せれば大規模学習としては良好な部類）。スケーリング効率が線形（8 枚で 8 倍）に近いほど、台数を増やした投資が素直に速度に変わるということです。Offload はメモリは稼げても TFLOPS が下がりやすく、これが「メモリと速度のトレードオフ」の正体です。

## メリット・トレードオフ・限界
メリット
- ZeRO によりデータ並列の冗長保持を取り除き、1 GPU あたりメモリを大幅削減。GPU を増やすほど 1 枚あたりが軽くなる（素朴な DP にはない性質）。
- ZeRO-Offload / Infinity により、GPU が少なく（極端には 1 枚でも）大規模モデルを学習可能。研究室レベルの計算資源を底上げする。
- 3D 並列・効率カーネル・推論最適化まで束ねた道具箱で、JSON 設定中心の低い導入コスト。
- 巨大モデル学習の事実上の標準ツールの 1 つで、知見・ドキュメントが豊富。

トレードオフ・限界
- Stage を上げるほど all-gather / reduce-scatter の通信が増え（Stage 3 は約 1.5 倍）、GPU 間ネットワークが遅いと速度が頭打ち。通信が同期点を作るためストラグラー（遅い 1 台）に弱い。
- Offload は転送の遅さで TFLOPS が下がる。メモリと速度のトレードオフが激しく、NVMe offload は特に遅い。
- 設定項目が多く、最適な Stage・offload 量・bucket サイズ・通信オーバーラップのチューニングに経験が要る。設定ミスで OOM や速度激減が起きやすい。
- 小規模で 1 枚に乗るモデルには不要で、導入が複雑さを増やすだけになりうる。
- PyTorch 標準 FSDP と機能が重複し、エコシステムの選択判断が必要。

研究上の未解決課題としては、通信と計算のオーバーラップをいかに自動で最適化するか、異種環境（GPU と CPU/NVMe の帯域差が大きい構成）での自動配分、そして MoE（Mixture of Experts）など疎なモデルでのシャーディング戦略、低精度（FP8）学習との統合などが活発に研究されています。

## 発展トピック・研究の最前線
- 3D 並列の自動探索（どの軸をどれだけ分割すれば最速かを自動決定する Alpa などの研究）。
- 活性化メモリ削減（activation checkpointing / recomputation、selective recompute と ZeRO の併用）。
- 量子化を絡めた学習（8bit Adam、FP8 学習）でオプティマイザ状態のバイト数自体を削る方向。ZeRO の 12 バイト/param を半分以下にできる。
- MoE 学習向けの専用シャーディング（DeepSpeed-MoE）。expert parallelism と ZeRO の組み合わせ。
- 推論側の最適化（KV キャッシュのシャーディング、テンソル並列推論、continuous batching）。
- 通信圧縮の進化（ZeRO++ の量子化通信を低帯域クラウド学習へ展開）。

## さらに学ぶための関連トピック
- [Tensor Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0659_tensor-parallelism)
- [Pipeline Parallelism](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0646_pipeline-parallelism)
- [NCCL (集合通信)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0643_nccl-collective-comm)
- [Spot Instance 学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1181_spot-instance-training)
- [Model Registry (Stage 遷移)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1168_model-registry-staging)
