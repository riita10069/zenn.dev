---
title: "online encoderとfrozenターゲットエンコーダ"
---

# online encoderとfrozenターゲットエンコーダ

## ひとことで言うと
自己教師あり学習で「未来」や「別視点」「別の拡張画像」を予測するとき、ネットワークを2つの役割に分けるのが標準レシピです。学習して毎ステップ更新される online encoder(オンラインエンコーダ、予測する側)と、更新を止めた target encoder(ターゲットエンコーダ、正解を作る側)です。この非対称な2本立てが、表現が無意味な定数に潰れる崩壊(collapse、崩壊)を防ぎます。本ノートでは、なぜ非対称性が崩壊を防ぐのか、ターゲットの更新方法に3つの流派(完全凍結・EMA・stop-gradient のみ)があること、それぞれの長所と短所を、前提知識ゼロの読者から研究者レベルまで段階的に解説します。

## 直感的な理解
弓道の練習を思い浮かべてください。的(まと)に矢を当てる練習をするには、的が動かないことが大前提です。もし的が「矢が来る方向へ自分から寄っていく」性質を持っていたら、どこに撃っても当たってしまい、上達(つまり本物の照準合わせ)は起きません。

自己教師あり学習で「2つの埋め込みを近づける」のは、これとよく似ています。予測する側(online、射手)が、正解を作る側(target、的)に近づこうとする。ここで的(target)も自由に動けると、的のほうから予測に寄っていって「どこを撃っても当たる」状態、つまり全入力が同じ1点に潰れる崩壊が起きます。これを防ぐには、的を動かさない(=ターゲット側への勾配を止める)。すると射手は本物の照準を学ぶしかなくなります。online は学習で動く射手、target は固定された的、というのがこのレシピの本質です。重要なのは「両方を同時に動かせると、楽をして潰れる」点で、わざと片方を動きにくくする非対称性が崩壊を防ぎます。

## 基礎: 前提となる概念
- 表現崩壊(collapse): 自己教師あり学習でネットワークが「全入力を同じ定数ベクトルに写す」退化解に陥ること。どの2つも完全一致するので損失は0になるが、特徴は何の情報も持たない。最も警戒される失敗モードです。
- 対比学習(contrastive learning): 「同じものの2つの見え方」を近づけ、「別物(ネガティブサンプル)」を遠ざけて崩壊を防ぐ方式。引き離す相手がいるので潰れない。ただし大量のネガティブと巨大バッチが必要で扱いが面倒、という弱点があります。
- 非対称設計(asymmetric architecture): online と target を別扱いし、online 側にだけ予測器(predictor)を付け、target 側へは勾配を流さない設計。ネガティブなしで崩壊を防ぐ鍵です。
- stop-gradient / detach: ターゲット側への勾配(更新の方向)を遮断する操作。`detach()` や `torch.no_grad()` で実現します。崩壊回避の中核操作です。
- EMA(指数移動平均): ターゲットの重みを online の重みでゆっくり追従させる更新。`θ_target ← τ·θ_target + (1−τ)·θ_online`。τ が1に近いほどターゲットはゆっくり動きます。τ=1 なら完全凍結、τ=0 なら即座に online と同一(=崩壊しやすい)になります。
- 予測器(predictor): online 側にだけ付ける小さなネット(数層の MLP など)。online の出力をさらに変換してから target と比べます。online と target の出力空間を意図的にずらすことで、非対称性を強める役割があります。

## 仕組みを詳しく
2つのエンコーダの役割を整理します。

- online encoder: 入力(過去・コンテキスト・拡張画像A)を符号化し、予測器を通して「target はこう出力するはず」と予測する。勾配が流れ、毎ステップ更新される。
- target encoder: 入力(未来・別視点・拡張画像B)を符号化して正解を作る。勾配を流さない(stop-gradient / detach)。更新方法は流派により異なる。

target encoder の更新方法には主に3流派あります。

1. 完全凍結(frozen, pretrained のまま): target を一切動かさない。最も単純で、ターゲット空間が完全に固定。同じ凍結 backbone を online のエンコード部分と共有する構成では、online と target の差は「backbone の上に乗る小さな予測ヘッドだけ」という特殊形になります(関連: [frozen backboneによるJEPAターゲット](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0024_frozen-backbone-jepa-target))。
2. EMA(指数移動平均): `θ_target ← τ·θ_target + (1−τ)·θ_online`(τ≈0.99〜0.999)。online がゆっくり target に染み込み、target は「online の遅延コピー」になる。少しずつ良くなるが急には動かない。BYOL の方式。
3. stop-gradient のみ(重み共有): online と target が同じ重みを共有し、target 側の出力で勾配だけ切る。最も軽い(別ネットを持たない)。ただしターゲットが online と同じ速さで動く(非定常)ため、predictor と stop-gradient の正確な配置が必須。SimSiam の方式。

擬似コードで全体像を示します。

```python
# online: 予測する側(学習)
z_online = online_encoder(x_a)          # 例: 過去フレーム / 拡張A
p = predictor(z_online)                 # target を予測

# target: 正解を作る側(更新しない)
with torch.no_grad():
    z_target = target_encoder(x_b)       # 例: 未来フレーム / 拡張B
    z_target = z_target.detach()         # 勾配を切る = 崩壊防止の要

loss = mse(p, z_target)
loss.backward()                          # online と predictor だけ更新

# target の更新(EMA を使う場合のみ)
with torch.no_grad():
    for tp, op in zip(target_encoder.parameters(), online_encoder.parameters()):
        tp.data = tau * tp.data + (1 - tau) * op.data
```

数値例で「なぜ stop-gradient が効くか」を見ます。仮に target にも勾配を流したとすると、最適化器は「p と z_target の両方を同時に小さな同じ値へ動かす」のが一番楽だと気づき、両者を0へ押し下げて崩壊します(loss = MSE(0,0) = 0)。`detach()` で target への勾配を切ると、最適化器は z_target を動かせず、与えられた非自明な z_target に p を合わせるしかなくなります。動く的を消して固定の的にする、というのが本質です。

自動運転のワールドモデル(WAM)では、このレシピが「online = 過去から未来を予測する小さなヘッド群」「target = 凍結 backbone が出す未来特徴」という形で採用されます。さらに過去も未来も同じ凍結 backbone で符号化するため、online encoder の backbone 部分と target encoder が実体としては同じ凍結ネットになる(差は上に乗る予測ヘッドだけ)、という最も単純な特殊形になります(関連: [World Action Model (WAM) の設計](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0030_world-action-model-design))。

実装上は、完全凍結と EMA を切り替え可能にし、デフォルトでは補助損失を opt-in(明示的に有効化したときだけ動く)にして、既存の挙動を壊さない設計にするのが安全です。これにより、WAM を入れても運転本体の学習結果が変わらないことを保証しつつ、効果検証ができます。

## 手法の系譜と主要論文
- SimCLR(Chen et al., ICML 2020, arXiv:2002.05709)。対比学習の代表。同じ画像の2拡張を近づけ、バッチ内の他画像をネガティブとして遠ざける。崩壊は「遠ざける相手がいる」ことで防ぐが、性能はバッチサイズに強く依存(大きいほど良い)し、数千規模のバッチを要した。
- MoCo(He et al., CVPR 2020, "Momentum Contrast", arXiv:1911.05722)。対比学習側だが、momentum encoder(=EMA で更新するターゲット)とキュー(過去のネガティブを溜めておく仕組み)を導入し、巨大バッチなしで大量のネガティブを扱えるようにした。EMA ターゲットの有用性を早期に示した研究で、後の BYOL の伏線になります。
- BYOL(Grill et al., NeurIPS 2020, "Bootstrap Your Own Latent", arXiv:2006.07733)。online/target の2本立て + target を EMA で更新 + predictor を online 側だけに付ける、という非対称性の組み合わせで、ネガティブなしの崩壊回避を実現。提案理由は「対比学習のネガティブ依存をなくしたい」。効果は「ネガティブなしで当時最高水準」。トレードオフは「EMA 係数 τ や predictor の有無に性能が敏感」。
- SimSiam(Chen & He, CVPR 2021, "Exploring Simple Siamese Representation Learning", arXiv:2011.10566)。EMA すら使わず、重み共有 + stop-gradient + predictor だけで崩壊を防げると示した。提案理由は「BYOL の EMA は本当に必須か?を検証」。効果は「stop-gradient こそが崩壊防止の核心だと実証」。トレードオフは「ターゲットが非定常で、predictor と stop-gradient の正確な配置を誤ると崩壊する」。
- I-JEPA(Assran et al., CVPR 2023, arXiv:2301.08243)。画像内のマスク予測で online(context encoder + predictor)と target(EMA で更新する target encoder)を分離。特徴空間で予測することで、ピクセル再構成より計算効率を高めた。
- JEPA 枠組み(LeCun, 2022)。target encoder を非学習にする要件を一般原理として提示。online/target 非対称性を「崩壊しない学習則」の中心に据えました。

これらに共通するのは「online は学習、target は勾配を切る(凍結 or EMA or stop-grad)」という非対称性です。流派の違いは「target をどれだけ、どう動かすか」の一点に集約されます。完全凍結(τ=1)からEMA(0.99〜0.999)、即時追従(τ=0、崩壊しやすい)まで、τ を動かすと連続的に流派がつながっていると見ることもできます。

## 論文の実験結果(定量データ)
ターゲット側を切り離す非対称性の効果は、アブレーションで最も鮮明に出ます。指標の意味とともに整理します。

- BYOL は ImageNet 線形評価(凍結特徴の上に線形分類器を学習し、その精度で表現の良さを測る標準手順)で約74.3%のトップ1精度を達成し、ResNet-50 でネガティブを使う SimCLR を上回りました。決定的なアブレーションは「ターゲットを online と同一にして stop-gradient/EMA を外すと、表現が崩壊して精度が無意味な水準(数%以下)まで落ちる」というもの。非対称性が「成立条件」であることを示しています。
- SimSiam は、stop-gradient を外した瞬間に損失が一気にゼロ付近へ落ち、同時に出力の標準偏差が崩壊(全サンプルが同じベクトルへ収束し、ばらつきが消える)する様子を学習曲線で示しました。stop-gradient ありでは ImageNet 線形評価で約68〜71%(設定による)。EMA を使わなくても stop-gradient だけで崩壊が防げる、という主張の直接的な定量証拠です。また predictor を外すと同様に崩壊することも示し、「stop-gradient + predictor」の組がともに必要だと明らかにしました。
- MoCo は EMA(momentum)係数 m に強く依存し、m=0.999 付近で良好、m=0(EMA なし=ターゲットも即更新)では精度が大きく劣化することを示しました。ターゲットをゆっくり動かすことの重要性を、係数を振った曲線で定量化しています。
- I-JEPA は ImageNet で、ピクセル再構成の MAE と同等以上の線形評価精度を、より少ない事前学習計算量(GPU 時間)で達成。特徴空間 + EMA ターゲットの効率を示しました。

読み解きのまとめ: 「online と target の更新速度に差をつける(target を凍結 or 遅延させる)」ことが、自己教師あり学習が崩壊せずに成立する鍵です。完全凍結はターゲットの質が事前学習に固定される代わりに最も単純で確実。EMA は緩やかに改善するが τ の調整が要る。stop-gradient のみは最軽量だが predictor と配置を誤ると崩壊リスクが残る、というトレードオフがアブレーションから読み取れます。

## メリット・トレードオフ・限界
メリット

- ネガティブサンプルなしで崩壊を防げる(対比学習の面倒なネガティブ収集と巨大バッチが不要)
- 完全凍結ならターゲットが完全固定で実装が最も単純、EMA なら緩やかに改善する両刀になる
- online 側だけ更新するので学習が安定しやすい

トレードオフ・限界

- 完全凍結はターゲットの質が事前学習に固定され、下流ドメインに追従しない
- EMA は τ の調整が必要で、大きすぎると更新が遅すぎ、小さすぎると非定常で崩壊リスクが上がる
- stop-gradient のみ(重み共有)はターゲットが動くため、predictor と stop-grad の正確な配置を誤ると崩壊しうる
- いずれの方式も detach / stop-gradient の付け忘れが致命的(静かに崩壊する)
- 研究上の未解決課題: どの流派が最良かはタスク・データ規模に依存し、理論的決着はついていない。特にデータが少ない/ドメイン特化の場合に、完全凍結と EMA のどちらが有利かは経験則に頼っている

## 発展トピック・研究の最前線
- 崩壊回避の別アプローチ: VICReg(Bardes et al., 2022)や Barlow Twins(Zbontar et al., 2021)は、非対称な stop-gradient ではなく「特徴の分散を一定以上に保つ」「特徴次元間の相関を消す」正則化で崩壊を防ぎます。online/target の非対称性に頼らない設計として対照的で、なぜ崩壊しないかの理解を別角度から深めます。
- 理論的理解: SimSiam の解析以降、「stop-gradient は予測器を介した一種の期待値最大化(EM, Expectation-Maximization)的な交互最適化と解釈できる」という理論研究が進み、なぜ崩壊しないのかの数理的理解が深まっています。
- 動画・時系列 JEPA: V-JEPA(2024)は動画の時空間マスク予測で online/EMA ターゲットを使い、自動運転の未来予測と直結する系列版の最前線です。
- 自動運転特有の課題: ターゲットを完全凍結するか、運転本体の生きた backbone を stop-gradient で共有するか、という選択は、ターゲットの新鮮さ(ドメイン追従)と学習干渉(運転本体を乱さない)のトレードオフとして現在も議論されています。

## さらに学ぶための関連トピック
- [frozen backboneによるJEPAターゲット](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0024_frozen-backbone-jepa-target)
- [World Action Model (WAM) の設計](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0030_world-action-model-design)
- [特徴再構成ロス(未来特徴予測の補助損失)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0555_feature-reconstruction-loss)
- [特徴空間で未来を予測するヘッド(JEPA 風 future-state head)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0011_future-state-jepa-loss)
