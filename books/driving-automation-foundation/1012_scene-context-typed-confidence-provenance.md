---
title: "型付き SceneContext（信頼度・出所つきシーン文脈）"
---

# 型付き SceneContext（信頼度・出所つきシーン文脈）

## ひとことで言うと
自動運転モデルが「この場面はこういう状況だ」と理解した結果を、ただの数字の配列(テンソル)として渡すのではなく、「どれくらい自信があるか(confidence)」と「誰がその情報を出したか(provenance, 出どころ)」という付帯情報をつけた、決まった形のデータ構造(型)として持ち運ぶ設計です。経路を計画する部品(プランナ)に、ラベルの貼られた整理箱に入れて渡すイメージです。これにより、後段が「この推定はあまり信用できない」「これは地図由来だから信頼できる」と区別して扱えるようになります。

## 直感的な理解
ある人が「あそこに歩行者がいると思う」とあなたに伝えたとします。情報の価値は、それだけでは決まりません。「絶対そう」と言ったのか「自信は半々」と言ったのかで、あなたの行動は変わるはずです。さらに「測量済みの地図に書いてある」のか「ぼんやり遠くを見た印象」なのかでも、信じる度合いが変わります。

機械でも同じです。前段のネットワークが「歩行者がいる、確率 51%」と思っていても、生のテンソルだけを後段に渡すと、その「51%」という弱さと「どこから来た推定か」という出どころが消えてしまいます。安全に関わるシステムでは、推定そのものより「その推定をどこまで信じてよいか」が重要な場面が多くあります。型付き SceneContext は、この「自信」と「出どころ」を構造的に運ぶ仕組みです。

## 基礎: 前提となる概念
順に噛み砕きます。

- テンソル(tensor): 多次元の数値配列。たとえば `[B, 256]` という形は「B 個の場面それぞれに、長さ 256 の特徴ベクトルが一本ずつ」を意味します。B はバッチサイズ(同時に処理する場面の数)、256 は特徴量の次元数です。
- ロジット(logit): ソフトマックスにかける前の生のスコア。大小関係はあるが確率にはなっていない数値の並び。
- ソフトマックス(softmax): いくつかのロジットを「合計が 1 になる確率」に変換する関数。`[1.2, 2.5, 0.3]` のような生スコアを `[0.20, 0.73, 0.07]` のような確率に直します。
- confidence(信頼度): ここでは「最も支持されているクラスの確率」を指します。softmax の結果の最大値です。0〜1 のスカラ値で、1 に近いほど自信が強い。
- provenance(出所・由来): その情報がどこから来たかの記録。たとえば「HD マップ由来」「VLM(画像を見て言葉で説明する AI)由来」など。
- 型(type)/ dataclass: プログラムで「決まったフィールドを持つ構造体」を定義する仕組み。Python の `dataclass` は、フィールド名と型を宣言するだけでその構造体を作れます。生テンソルを裸で渡す代わりに、名前のついた箱に入れて渡せます。
- プランナ(planner): 理解結果を受けて、これからどう走るか(軌跡や操作)を決める後段の部品。

なぜ型で運ぶのか。生テンソルを直接渡すと、後段は「この 256 個の数字が何を意味するか」を暗黙の前提として持つことになります。モデル構成を変えて新しい出力(たとえば「交差点かどうか」)を足すと、入力の形が変わり、後段のコードも書き換える羽目になります。型で包めば、フィールドを追加するだけで済み、既存コードを壊しません。これが「拡張に強い設計」の核心です。

## 仕組みを詳しく
典型的な実装では、Python の `dataclass` で内側と外側の二層の型を定義します。

内側は、個別の推論結果を入れる箱です。たとえば因果推論ヘッド(後述)の出力をまとめる構造体は、概念的に次のようになります。

```python
@dataclass
class CausalReasoningContext:
    reasoning_latent: torch.Tensor      # [B, latent_dim] 後段を条件づける連続ベクトル
    causal_class_logits: torch.Tensor   # [B, num_classes] クラス分類の生スコア
    confidence: torch.Tensor            # [B] スカラ確率(0〜1の自信度)
    provenance: str = "vlm_causal_head" # この推論の出どころ
```

各フィールドの意味を初心者向けに説明します。

- `reasoning_latent` は「考えた結果を圧縮した特徴ベクトル」。形は `[B, latent_dim]`。これを後段に足し込むと、後段の判断を条件づけられます。
- `causal_class_logits` は分類の生スコア。形は `[B, num_classes]`。たとえば num_classes=5 なら「交差点・歩行者・信号・障害物・通常走行」の 5 種に対応するスコア。
- `confidence` は `[B]`、場面ごとに 1 個のスカラ。softmax の結果が `[0.05, 0.51, 0.10, 0.04, 0.30]` なら confidence は 0.51。
- `provenance` は文字列。どの仕組みが出した推論かの記録。

外側は、プランナに渡す包括的な箱です。

```python
@dataclass
class SceneContext:
    causal_reasoning: Optional[CausalReasoningContext] = None
    # 将来: hd_map_info, route_command, ... を足せる
```

`Optional` で初期値が `None` なのが重要です。これは「中身は空でもよい」という意味で、特定のヘッドを使わない構成では `SceneContext()` という空の箱だけ渡せます。つまり、この仕組みを導入してもデフォルトの挙動を一切変えずに済みます。将来、HD マップ情報やルート指示(「次の交差点で右折」)も、このクラスにフィールドを足すだけで同じ通り道に載せられます。

confidence の作り方を数値例で示します。

```python
confidence = F.softmax(decision_logits, dim=-1).max(dim=-1).values
```

`decision_logits` が `[1.2, 2.5, 0.3, -0.4, 1.0]` なら、softmax で概ね `[0.13, 0.49, 0.05, 0.03, 0.30]` になり、max を取ると 0.49。これが「最も支持されるクラス(この例では 2 番目)の確率 = 自信度」です。

処理の流れを段階で書くと次の通りです。

1. 前段(理解を担うヘッド)が文脈ベクトル `[B, D]` を受け取る。
2. クラスのロジット `[B, num_classes]` と reasoning_latent `[B, latent_dim]` を計算する。
3. softmax + max で confidence `[B]` を作る。
4. これらを内側の箱(例: CausalReasoningContext)に詰め、さらに外側の SceneContext に入れる。
5. その SceneContext をプランナに渡す。プランナは confidence や provenance を見て、推定を割り引いたり信頼したりして使う。

ここで設計上の含意を一つ。confidence を「持ち運ぶ」だけでは十分ではなく、その confidence が現実の正答率と一致している(較正されている, calibrated)ことが望まれます。softmax の最大値は往々にして過信(実際の正答率より高い確率を出す)するため、後段がそのまま信じると危険です。これは次節の研究系譜と密接に関係します。

## 手法の系譜と主要論文
「シーン理解を型と信頼度つきで運ぶ」という発想は、いくつかの研究の合流点にあります。

不確実性を陽に持つ重要性は、Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (NeurIPS 2017) が定式化しました。彼らは、予測の不確実性を二種類に分けました。aleatoric uncertainty(データ自体のばらつき・ノイズ由来、入力をいくら集めても消えない)と、epistemic uncertainty(モデルの知識不足由来、データを増やせば減る)です。両者を分けて推定すると、難しいサンプルでモデルが「自信が無い」と表明できるようになります。効果は、自信の無い予測に低い重みを与えられること。トレードオフは、不確実性を出すための追加出力や複数回サンプリングの計算が要ること。confidence を softmax 最大値で近似する素朴な方法は、この厳密なベイズ的不確実性の簡易版に当たります。

confidence が現実とずれる(過信する)問題は、Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017) が広く知らしめました。深いネットワークほど、出力する確率が実際の正答率より高く出る(miscalibration)ことを実測で示し、temperature scaling という後処理で較正できることを提案しました。これは「confidence を運ぶなら、その confidence を信用できるよう較正せよ」という、本トピックの裏側の必須課題を指します。

「視覚から意味・安全に関わる中間表現を抽出してプランナに渡す」流れは、Shao et al., "InterFuser: Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer" (CoRL 2022) に近い発想があります。InterFuser は、周辺物体や交通標識の状態という「解釈可能な安全表現」を中間に明示的に出力し、それを安全制御に使いました。動機は、ブラックボックスの end-to-end 出力だけでは安全性を検証しにくいこと。効果は、人間が中間表現を点検でき、安全制約をかけやすいこと。トレードオフは、中間表現のためのアノテーションや設計コスト。

言語・記号レベルのシーン理解を運転に持ち込む点では、Sima et al., "DriveLM: Driving with Graph Visual Question Answering" (2023, arXiv:2312.14150、後に ECCV 2024) が、運転シーンを質問応答のグラフとして表現し、知覚・予測・計画を言語で接続しました。動機は人間に近い段階的推論。効果は説明可能性と汎化。トレードオフは VLM 由来の出力がノイズを含み誤答すること。型付き文脈が provenance と confidence を持つのは、まさにこのノイズを後段で割り引くためです。

## 論文の実験結果(定量データ)
本トピックは設計パターンであり単一のベンチマークを持ちませんが、構成要素の効果は定量的に測られています。

不確実性推定の効果(Kendall & Gal, NeurIPS 2017): セマンティックセグメンテーション(CamVid, NYUv2)と深度推定で、aleatoric と epistemic を同時にモデル化すると、しないベースラインより精度が向上したと報告されました。たとえば深度推定では、不確実性を出力に組み込むことで誤差(RMSE 等)が改善し、加えて「不確実性が高いと予測した領域ほど実際の誤差も大きい」という相関が示され、confidence が意味を持つことが裏づけられました。

較正(Guo et al., ICML 2017): 較正の質を測る指標として ECE(Expected Calibration Error, 期待較正誤差)があります。これは「モデルが確率 p と言ったサンプル群の、実際の正答率が p からどれだけずれるか」の平均です。論文では、素のニューラルネットの ECE が数〜十数パーセントに達する(つまり大きく過信する)のに対し、temperature scaling 一つで ECE を 1% 前後まで下げられると示しました。アブレーションとして、temperature scaling は他の複雑な較正手法に匹敵する性能を最小のパラメータ(温度スカラ 1 個)で達成する点が強調されています。これは「confidence をそのまま信じると危険、較正が要る」ことの定量的証拠です。

解釈可能中間表現(InterFuser, CoRL 2022): CARLA の運転ベンチマーク(Town05, Longest6)で、中間の安全表現を出力する構成が、出力しないブラックボックス end-to-end より運転スコア(Driving Score)と違反率で優れたと報告されました。中間表現を介すことで、衝突や信号無視といった違反が減ったことがアブレーションで示されています。

## メリット・トレードオフ・限界
メリット
- プランナのインターフェースを固定したまま、新しい理解結果を後から足せる(拡張に強い)。
- confidence により、後段が推定の弱さを割り引いて使える。
- provenance により、信頼できる地図由来と当てになりにくい推測由来を区別して扱える。
- 空の箱を渡せるので、既存のデフォルト挙動を壊さずに段階導入できる。

トレードオフ・限界
- 型を一枚かませる分、コードと概念がわずかに増える(生テンソルを直接渡すより一手間多い)。
- confidence が softmax 最大値である限り、過信を完全には防げない。較正(calibration)が別途必要で、これを怠ると「高い confidence なのに間違い」という最も危険な失敗を見逃す。
- フィールドを増やしすぎると、どの構成でどれが埋まるのかが分かりにくくなり、結局ドキュメントが要る。
- provenance を文字列で持つだけでは、後段がそれを正しく解釈・活用する保証はない。出所ごとの信頼度を体系的に重みづける枠組みは別途設計が要る。

未解決の課題として、複数の出所(地図・VLM・検出器)からの矛盾する情報を、confidence と provenance を使ってどう統合・調停するか(センサー融合の意思決定版)は、研究上の open problem です。

## 発展トピック・研究の最前線
- 不確実性の較正手法の発展: temperature scaling を超えて、分布シフト下でも崩れにくい較正(Dirichlet calibration, focal calibration)、回帰タスクの較正(quantile/conformal prediction)。
- conformal prediction: 統計的保証つきで「予測区間」を出す枠組み。confidence を「点推定 + 区間」に格上げし、安全マージンの設計に直結させる流れ。
- 中間表現の標準化: 知覚・予測・計画をつなぐ中間表現を、クエリ(query)や言語(VQA グラフ)として共通化する研究(UniAD, DriveLM)。型付き文脈はこの工学的実装形と言える。
- evidential deep learning: 一回の forward で aleatoric/epistemic 双方を出す手法。複数回サンプリングのコストを避けつつ不確実性を運ぶ方向。

## さらに学ぶための関連トピック
- [System 2 因果推論ヘッド（オプショナル）](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1049_system2-causal-reasoning-head)
- [ニューラルネットの不確実性とキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0794_uncertainty-calibration-neural-nets)
- [解釈可能な中間表現](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1076_interpretable-intermediate-representations)
- [Python dataclassによる型付き構造](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1099_python-dataclass-typed-structures)
- [System 1 / System 2 二重過程の自動運転モデル](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1087_system1-system2-dual-process)
