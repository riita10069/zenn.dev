---
title: "Discriminative / Layer-wise Learning Rates"
---

# Discriminative / Layer-wise Learning Rates

## ひとことで言うと
Discriminative Learning Rates (識別的学習率) とは、ニューラルネットワークの「層」ごとに違う学習率を割り当てる手法です。学習率 (learning rate) とは「1回の更新で重みをどれだけ動かすか」を決める数字です。事前学習済みの下の方の層 (汎用知識を持つ) はそっと小さく動かし、新しく付けた上の方の層 (新タスクに適応させたい) は大きく動かします。転移学習で「下層の汎用知識を守りつつ上層を速く適応させる」ための、シンプルで効果の大きい道具です。

## 直感的な理解
古い建物をリフォームする場面を想像してください。基礎や柱 (土台) はしっかりしているので触らず、内装や家具 (使い勝手に直結する部分) だけを大きく作り替えます。土台を強く揺らせば建物全体が崩れますが、内装は思い切って変えてよい。ニューラルネットワークの fine-tuning も同じで、土台にあたる下層は慎重に、内装にあたる上層は大胆に動かしたいのです。

転移学習で事前学習済みバックボーンを流用するとき、素朴には「全層に同じ学習率をかけて fine-tuning する」やり方が考えられます。ところがこれには問題があります。学習率を大きくすると、新しく付けた上層 (出力に近い層) はタスクに速く適応しますが、同時に下層 (入力に近い層) の重みも大きく書き換わります。下層は「エッジやテクスチャを検出する」という、大規模事前学習で苦労して獲得した汎用知識を持っており、これを大きく動かすと汎用知識が壊れます (破滅的忘却 = catastrophic forgetting に近い現象。新しいことを学ぶ過程で以前学んだことを急速に忘れる)。逆に忘却を防ごうと全層で学習率を小さくすると、上層が新タスクに十分適応できず学習が遅くなります。

ここに本質的なジレンマがあります。「下層は守りたい、上層は速く変えたい」。両方を1つの学習率で満たすことはできません。Yosinski et al. (2014) が示したとおり下層ほど汎用的で転移しやすく、上層ほどタスク特化なので、層によって最適な動かし方が違うのです。そこで「層ごとに学習率を変える」が自然な解決策になります。

## 基礎: 前提となる概念
- 学習率: 重み更新の歩幅。大きいほど速く動くが不安定・忘却のリスク、小さいほど慎重だが遅い。更新式は概念的に「新しい重み = 古い重み − 学習率 × 勾配」。
- パラメータグループ (parameter group): 最適化器に渡す重みのまとまり。グループごとに学習率などを別々に設定できる。
- 下層 / 上層: 入力に近い側が下層 (low-level, 汎用)、出力に近い側が上層 (high-level, タスク特化)。
- 減衰率 (decay factor): 1つ下の層に進むごとに学習率を何倍に縮めるかの係数。0.65 や 1/2.6 などが使われる。1.0 なら一律学習率と同じ、0 に近いほど下層をほぼ凍結する。
- スケジュール (schedule): 学習の進行に応じて学習率を時間方向に変える計画 (ウォームアップ・減衰など)。識別的学習率は「層方向」の差、スケジュールは「時間方向」の変化で、両者は直交し併用できる。

## 仕組みを詳しく
考え方はとてもシンプルです。ネットワークを下から上へグループに分け、各グループに別々の学習率を設定します。

例として、ネットワークが「バックボーン下層・バックボーン上層・新しいヘッド (出力層)」の3グループに分かれているとします。基準学習率を `lr = 1e-3` とすると、

```
新しい head        : lr = 1e-3      (大きく動かす。まだ何も学んでいないので)
backbone 上層      : lr = 1e-4      (中くらい)
backbone 下層      : lr = 1e-5      (小さく。汎用知識を守る)
```

のように、下に行くほど学習率を小さくします。ULMFiT (Howard & Ruder 2018) では、ある層の学習率を `lr` としたとき、その1つ下の層の学習率を `lr / 2.6` のように一定比で割っていく式を提案しました (この比はテキスト分類タスクで経験的に良かった値)。

PyTorch での実装は、最適化器に「パラメータグループ」を渡すだけです。

```python
optimizer = torch.optim.AdamW([
    {"params": backbone_low.parameters(),  "lr": 1e-5},
    {"params": backbone_high.parameters(), "lr": 1e-4},
    {"params": head.parameters(),          "lr": 1e-3},
])
```

before/after で言うと、before は「全層 lr=1e-3 の一律」、after は「層ごとに 1e-5 / 1e-4 / 1e-3 と差をつける」だけ。コード量はわずかですが効果は大きく変わります。

### Layer-wise LR Decay (LLRD) との関係
似た用語に [Layer-wise LR Decay (LLRD)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0395_layer-wise-lr-decay) (LLRD) があります。両者は「下層ほど学習率を小さくする」点で同じ思想ですが、ニュアンスが異なります。
- Discriminative LR: 数グループに粗く分け、グループごとに学習率を指定する。ULMFiT 由来。
- LLRD: 隣り合う層ごとに一定の減衰率 (例: 0.65) を掛け、最下層に向かって連続的・指数的に学習率を小さくする。最上層を第 N 層、層番号を i (i=1 が最下層) とすると `lr_i = lr_top × decay^(N − i)` のように層番号 i の関数で与える。Transformer の fine-tuning (BERT・BEiT・ELECTRA など) で標準的に使われる。

両者はほぼ地続きで、LLRD は Discriminative LR を「全層に滑らかに」適用した精密版と捉えられます。なぜ指数減衰が自然か――層を1つ下るたびに「より汎用的で守るべき度合い」が一定の割合で増す、という近似に対応するからです。Transformer は同じ構造のブロックが積み重なっているため、層番号の関数で滑らかに減衰させやすいという事情もあります。

## 手法の系譜と主要論文
- Yosinski et al., "How transferable are features in deep neural networks?", NeurIPS 2014。「下層は汎用・上層は特化」を実験で示し、層ごとに扱いを変えるべきという発想の根拠を与えた (直接 discriminative LR を提案したわけではない)。
- Howard & Ruder, "ULMFiT", ACL 2018。discriminative fine-tuning を明示的に提案し、`1/2.6` 減衰式を導入。slanted triangular learning rates (学習率を最初急上昇させてから緩やかに下げる時間方向のスケジュール) と [Gradual Unfreezing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0388_gradual-unfreezing) をセットで使い、安定した転移を実現した。
- Clark et al., "ELECTRA", ICLR 2020。事前学習済み Transformer の fine-tuning で layer-wise LR decay を採用し、下流タスクの安定性・精度を底上げ。LLRD が NLP の標準作法になる流れを後押し。
- Bao et al., "BEiT", ICLR 2022 / He et al., "MAE", CVPR 2022。画像 Transformer の自己教師あり事前学習後の fine-tuning で LLRD (減衰 0.65〜0.9 程度) を標準採用。その後の視覚 Transformer のレシピでも定番化した。
- Lee et al., "Surgical Fine-Tuning", ICLR 2023。「どの層を動かすべきか」は分布シフトの種類で変わると示し、層選択を識別的学習率の延長線で精密化した。
- 関連手法として LARS / LAMB (Layer-wise Adaptive Rate Scaling; You et al. 2020 ほか) があり、これは転移目的ではなく大バッチ学習の安定化のために層ごとに学習率を適応させるもの。同じ「層ごとに学習率を変える」系譜の、別動機の発展。

## 論文の実験結果(定量データ)
ULMFiT (2018) は IMDb・AG News・TREC など複数のテキスト分類で、誤り率 (error rate, 低いほど良い) を当時の最良手法より相対で18〜24%程度削減と報告。アブレーションでは、discriminative fine-tuning・slanted triangular LR・gradual unfreezing をそれぞれ外すと精度が落ち、3つを組み合わせたときが最良でした。特に discriminative fine-tuning を外すと一律学習率になり、下層の汎用知識が壊れて精度・安定性が悪化することを示しています。これは「層ごとに学習率を変えること自体が効いている」ことの直接的な証拠です。指標の意味としては、error rate の相対18%削減は「間違いが約2割減る」ことを意味し、層ごとの工夫だけでこの差が出る点が重要です。

画像・言語の Transformer fine-tuning では、LLRD の減衰率がハイパーパラメータとして効きます。報告例として、BEiT/MAE 系のレシピでは減衰率 0.65〜0.85 あたりが ImageNet fine-tuning で良好とされ、減衰なし (= 一律学習率) に比べて top-1 accuracy が数十分の一〜1ポイント弱改善する場合があると報告されています。減衰を強くしすぎる (下層をほぼ凍結) と適応不足、弱すぎる (一律に近い) と忘却、という両端の間に最適点があるのが典型的な振る舞いです。値が小さく見えますが、上位を競う ImageNet では 0.5 ポイントの差が順位を分けるため軽視できません。

Lee et al. (2023) の Surgical Fine-Tuning は、分布シフトが入力側 (画素レベルのノイズ・ぼけ) のときは下層だけ、出力側 (ラベル定義の変化) のときは上層だけを動かすのが最良で、全層一律 fine-tuning に対して数ポイント改善する場合があると報告。識別的学習率の「層ごとに動かす強さを変える」発想を、シフトの種類に応じて極端化したものと位置づけられます。

## メリット・トレードオフ・限界
メリット
- 事前学習済みの汎用知識 (下層) を壊さずに守れる。
- 新タスク特化が必要な上層は速く適応できる。
- 実装が容易 (最適化器のパラメータグループに学習率を指定するだけ)。
- fine-tuning の安定性を高め、小データでのばらつき (シード間分散) を抑える。

トレードオフ・限界
- 層の分け方・減衰比というハイパーパラメータが増え、調整コストがかかる。
- 効果はタスク依存で、必ず精度が上がる保証はない。
- 層境界の決め方が恣意的になりがち (どこを「下層」と区切るか)。Transformer のように層が均質なら LLRD で滑らかに扱えるが、異種構造 (CNN と Transformer の混成など) では設計が難しい。
- 分布シフトの向きを誤ると逆効果: 出力側のシフトなのに下層を強く動かすと、汎用特徴を無駄に壊しうる (Lee et al. 2023)。
- 未解決の課題: 最適な減衰率やグループ分けを自動で決める方法は確立していない。アーキテクチャや下流タスクごとに探索が必要なのが現状。

## 発展トピック・研究の最前線
- LLRD の自動調整: 減衰率を学習可能パラメータ化したり、メタ学習で決める試み。
- 層選択的微調整: BitFit (バイアスのみ更新; Zaken et al. 2022) や surgical fine-tuning (分布シフトの種類に応じて更新する層を選ぶ; Lee et al. 2023) など、「どの層をどれだけ動かすか」をより細かく制御する研究。
- PEFT との接続: LoRA・アダプタなどパラメータ効率的微調整も「層ごとに適応の強さを変える」という意味で同じ問題意識の延長にある。どの層に LoRA を挿すか・ランクをどう割り当てるかは識別的学習率と同型の問いになる。

## さらに学ぶための関連トピック
- [Transfer Learning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0364_transfer-learning)
- [Layer-wise LR Decay (LLRD)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0395_layer-wise-lr-decay)
- [Gradual Unfreezing](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0388_gradual-unfreezing)
- [Fine-tuning](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0384_fine-tuning)
- [Frozen Backbone](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0355_frozen-backbone)
