---
title: "Pre-Norm Transformer ブロック構成"
---

# Pre-Norm Transformer ブロック構成

## ひとことで言うと

Transformer というニューラルネットワークでは「注意機構（Attention）」と「全結合層（FFN）」という2つの計算ブロックを繰り返し積み重ねます。その2ブロックそれぞれの「入口の手前」に、数値を整える正規化（LayerNorm）を置く作り方を Pre-Norm と呼びます。逆に「出口の後」に置くのが Post-Norm です。Pre-Norm は学習が安定し、深いネットワークでも壊れにくくなる、という地味だけど確実に効く工夫です。

## 直感的な理解

深いネットワークを学習させると、層を通るたびに途中の数値（中間表現）が膨らんだり潰れたりしがちです。膨らみすぎると学習の手がかりである勾配（パラメータをどっちにどれだけ動かすかの微分）が爆発し、潰れると勾配が消えて学習が止まります。これを勾配爆発・勾配消失と呼びます。

たとえると、何十人もが一列に並んで伝言ゲームをする状況です。各人が前の人の言葉を少し変えて次へ渡すと、終端では元の意味が跡形もなく崩れることがあります。これを防ぐには「各人の手前で言葉を一度整える（正規化する）」か、あるいは「元の伝言メモを最後まで持ち回って参照できるようにする（残差接続）」のが有効です。Pre-Norm は前者を、Transformer の残差接続は後者を担い、両者を組み合わせることで深い層でも情報と勾配がきれいに流れるようにします。

正規化を「ブロックの後」に置くか「前」に置くかという一見ささいな差が、学習の安定性に大きく効きます。これが Pre-Norm の話の核心です。

## 基礎: 前提となる概念

- LayerNorm（レイヤー正規化）= ある1サンプルの特徴ベクトル（例えば長さ256の数値の並び）について、その256個の平均が0、標準偏差が1になるよう引き算と割り算をする操作です。さらに学習可能なスケール係数 γ とシフト係数 β を掛け足しして、ネットワークが必要なら正規化の効果を弱められるようにします。バッチ方向ではなく特徴方向に正規化するので、バッチサイズに依存しないのが特徴です（Ba et al., 2016, arXiv:1607.06450）。
- 残差接続（residual connection）= サブブロックの出力を入力にそのまま足し戻す配線です。`x = x + F(x)` の形をとります。F が何も学習していない初期状態でも入力 x が素通りでき、勾配が深い層まで届きやすくなります。He らの ResNet で導入され、深層学習を実用化した最重要技術の一つです。
- 勾配爆発 / 勾配消失 = 誤差逆伝播で勾配が層を遡るたびに大きくなりすぎる（爆発）、または小さくなりすぎて消える（消失）現象です。どちらも学習を妨げます。
- ウォームアップ（learning-rate warmup）= 学習開始直後だけ学習率を小さくしておき、徐々に上げていく手当てです。Post-Norm の Transformer はこれがないと初期に発散しやすいことが知られていました。

## 仕組みを詳しく

1つの Transformer ブロックは、ふつう次の2つのサブブロックでできています。

- 注意サブブロック: 系列内のどの要素に注目するかを計算する
- FFN サブブロック: 各要素を独立に2層の全結合層で変換する

両方に残差接続が付きます。Post-Norm と Pre-Norm の違いを式で並べると一目瞭然です。x を入力として、

Post-Norm（元祖 Transformer）:
```
x = LayerNorm( x + Attention(x) )
x = LayerNorm( x + FFN(x) )
```

Pre-Norm:
```
x = x + Attention( LayerNorm(x) )
x = x + FFN( LayerNorm(x) )
```

Pre-Norm では、残差接続の「足し戻す側」の x には LayerNorm がかかりません。サブブロックの中に入るときだけ正規化された値を渡します。この差が決定的に効きます。Pre-Norm では、入力 x から最終出力までを残差接続だけでつなぐ「正規化を一度も通らない素通り経路」が層の数だけ確保されます。勾配はこの素通り経路を通って深い層まで減衰せず届くので、勾配消失が起きにくくなります。

ASCII 図で対比します。

```
Post-Norm:   x ─┬─→[Attn]─→(+)─→[LN]─→ ...
                └──────────┘            （足し戻した直後に正規化が挟まる）

Pre-Norm:    x ─┬─→[LN]→[Attn]─→(+)─→ ...
                └────────────────┘      （素通り経路に正規化が一切ない）
```

具体的な数値例で考えます。多視点特徴の融合に Transformer ブロックを使う場合、トークン列 `[B, N, 256]`（B=バッチ、N=系列長やグリッドのセル数、256=特徴次元）に対して、LayerNorm は最後の次元256について平均0・分散1へそろえます。pre-norm を使うアテンション融合の典型的な後処理は、

```
x_norm = LayerNorm(x)                  # Attention の手前で正規化
attn_out = MultiheadAttention(x_norm, x_norm, x_norm)
x = x + attn_out                       # 正規化していない x に足し戻す
x = x + FFN( LayerNorm(x) )            # FFN も手前で正規化、残差加算
```

という形になります。複数カメラの特徴のように「スケールが揃っておらず数値が暴れやすい」入力を安定して融合するうえで、この pre-norm 構成は相性が良いとされています。

## 手法の系譜と主要論文

- Vaswani et al., "Attention Is All You Need"（NeurIPS 2017、arXiv:1706.03762）。Transformer の原型で、Post-Norm 構成を採用し、学習率ウォームアップを併用していました。Post-Norm は表現力の面では問題ないものの、深層化と学習安定性の両立が難しいことが後に指摘されます。

- Wang et al., "Learning Deep Transformer Models for Machine Translation"（ACL 2019、arXiv:1906.01787）。Pre-Norm が数十層という深い Transformer を安定して学習できることを機械翻訳で実証し、深層化に向く構成として広めました。Post-Norm では深くすると学習が崩れる一方、Pre-Norm では崩れずに性能が伸ばせることを示しました。

- Xiong et al., "On Layer Normalization in the Transformer Architecture"（ICML 2020、arXiv:2002.04745）。Pre-Norm と Post-Norm を理論と実験の両面から比較した代表的論文です。提案は単純で「LayerNorm を残差ブロックの内側（手前）に移す」こと。鍵は、Post-Norm では学習開始時に出力層付近の勾配の期待値が非常に大きくなり、これがウォームアップを必須にしている原因だと理論的に示した点です。Pre-Norm に変えると初期勾配が層をまたいでよく振る舞い、ウォームアップなしでも安定して学習率も上げられると報告しました。

- 補足: 現在の大規模言語モデル（GPT 系）や ViT など多くの実用モデルは Pre-Norm を標準採用しています。さらに最近は RMSNorm（平均引きを省いた軽量版正規化）や、Attention と FFN の出力をスケールするゲート、DeepNorm（残差を係数倍して深層 Post-Norm を安定化する手法）など、Pre/Post の良いとこ取りを狙う派生も提案されています。

## 論文の実験結果（定量データ）

- Xiong et al.（2020）は IWSLT/WMT の機械翻訳タスクで、Post-Norm はウォームアップ段数や学習率に敏感で、設定を誤ると BLEU（翻訳一致度、高いほど良い）が大きく崩れる一方、Pre-Norm はウォームアップを外しても安定して収束し、より大きな学習率を使えると報告しました。結果として「Pre-Norm はハイパーパラメータ調整の負担を減らし、収束を速める」ことが定量的に示されています。一方で、十分にチューニングできた場合の最終 BLEU は Post-Norm の方が僅かに高くなりうることも観測され、「安定性（Pre-Norm）と最終性能の上限（Post-Norm）のトレードオフ」が浮かび上がりました。

- Wang et al.（2019）は、Post-Norm では学習が破綻する深さ（例えば20層以上）でも Pre-Norm なら安定して学習でき、WMT 英独翻訳で深い Pre-Norm モデルが浅いモデルを BLEU で上回ることを示しました。これは「Pre-Norm が深層化という新しい性能の伸びしろを開いた」ことを意味します。

- 数値の意味の補足: BLEU は機械翻訳で出力文と参照訳の n-gram（連続する単語の並び）一致度を測る0〜100の指標で、1〜2ポイントの差でも実用上は有意とされます。Pre-Norm の利得は「同じ計算量でより深く積めるようになり、その深さで BLEU が伸びる」点と「調整失敗による大崩れを防ぐ」点の両方にあります。

## メリット・トレードオフ・限界

メリット
- 学習が安定し、勾配爆発・消失が起きにくい。深い層を積みやすい。
- 学習率ウォームアップなしでも発散しにくく、ハイパーパラメータ調整が楽。
- 残差の素通り経路があるため、初期状態でも入力がそのまま伝わり学習の立ち上がりが速い。

トレードオフ・限界
- 十分にチューニングした Post-Norm に比べ、最終精度がごくわずかに劣る場合がある。各層の出力を毎回正規化する Post-Norm は表現を強く制約でき、丁寧に調整すれば僅かに高い精度が出ることがあるためです。
- Pre-Norm は LayerNorm を残差の外に置くため、層を重ねるほど残差経路上の数値スケールが単調に増える傾向があり、最終出力の直前に追加の LayerNorm を入れる対処が要ることがあります。
- 未解決の論点として、Pre/Post の最終性能差をどう埋めるか（DeepNorm などの提案はあるが決定打はない）、正規化を完全に取り除いて初期化や残差スケールだけで安定化できるか（normalizer-free な研究）が研究の最前線として残っています。

## 発展トピック・研究の最前線

正規化の置き方をめぐる研究は今も活発です。DeepNorm（Wang et al., 2022）は残差を係数倍することで Post-Norm のまま1000層規模の Transformer を安定学習できることを示しました。RMSNorm は平均引きを省くことで計算を軽くしつつ性能を保ち、大規模言語モデルで広く使われています。さらに「正規化なし」を目指す方向（適切な初期化と残差スケーリングで安定化する NFNet 系の発想を Transformer に持ち込む研究）もあります。多視点知覚や運転モデルの文脈では、Attention 自体の解説（[Multi-Head Attention融合](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0312_multihead-attention-fusion)）や、それを使ったカメラ融合（[CrossAttentionViewFusion（カメラ間アテンション融合）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0952_cross-camera-attention-fusion)）と合わせて理解すると、なぜ pre-norm が選ばれるのかが腑に落ちます。

## さらに学ぶための関連トピック

- [CrossAttentionViewFusion（カメラ間アテンション融合）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0952_cross-camera-attention-fusion)
- [Multi-Head Attention融合](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0312_multihead-attention-fusion)
- [Layer Normalization](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0132_layernorm)
- [残差接続(skip connection)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0358_residual-connection)
- [BEV Spatial Cross-Attention（鳥瞰図への空間クロスアテンション融合）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0932_bev-spatial-cross-attention)
