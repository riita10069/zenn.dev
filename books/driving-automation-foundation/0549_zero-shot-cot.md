---
title: "Zero-shot CoT (Let's think step by step)"
---

# Zero-shot CoT (Let's think step by step)

## ひとことで言うと
大規模言語モデル（LLM）に難しい問題を解かせるとき、お手本を一切見せず、質問の後ろに「Let's think step by step（順を追って考えよう）」というたった一文を付け足すだけで、モデルが自発的に途中の考え方を書き出し、正答率が大きく上がるという、驚くほど単純なテクニックです。お手本を用意する手間がゼロのまま Chain-of-Thought（[Chain-of-Thought (CoT) プロンプティング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0529_chain-of-thought-prompting)、思考の連鎖）を引き出せます。

## 直感的な理解
通常の Few-shot 版の CoT は強力ですが、「途中の推論つきの模範解答」を人間が手で書いて用意する必要がありました。しかも算数には算数用、常識推論には常識用、というようにタスクごとに別々のお手本を作らないと十分に効きません。さらにお手本の選び方・順番・書き方しだいで成績がぶれるという厄介さもあり、新しい種類のタスクへ展開するたびにコストと当たり外れがついて回りました。

Zero-shot CoT の発想は「お手本を真似させなくても、モデルに『考えろ』と指示するだけで推論を始めさせられないか」というものです。直感の核心は、大規模モデルは事前学習の段階で、人間が書いた膨大な解説文・教科書・Q&A を読んでおり、その中に「Let's think step by step」のような前置きの後に段階的説明が続くパターンを無数に見ている、という点にあります。だからこの一文を呼び水（trigger）として与えると、モデルは「この後は段階的に説明する流れだ」という出力分布へ自然に入り、推論を書き始める。能力そのものを新たに作るのではなく、すでにモデルの中に潜在している「段階的に説明する振る舞い」を引き出すスイッチを押している、というイメージです。著者らはこれを「LLM は明示的に引き出されていないだけで、汎用の推論能力を持つ zero-shot reasoner だ」と表現しました。

## 基礎: 前提となる概念
- ゼロショット（zero-shot）: お手本を1つも見せずにタスクを解かせること。「ショット」はお手本の数で、zero-shot は0個、few-shot は数個を指します。Zero-shot CoT は「お手本0個でも CoT を起こす」ことを目指す技術です。なお紛らわしいですが、ここでの zero-shot は「お手本ゼロ」であって「指示文ゼロ」ではありません。指示文（呼び水）は与えます。
- 呼び水（trigger phrase）: 推論を誘発するために質問の後ろに置く決まり文句。Zero-shot CoT における「Let's think step by step」がこれにあたります。タスクの種類を問わず使い回せる汎用の一文である点が肝です。
- 2段階プロンプト（two-stage prompting）: 「考えさせる段階」と「答えを取り出す段階」を分けて、モデルを2回呼ぶ方式。後述するとおり、これが Zero-shot CoT の実装上の特徴です。
- 答え抽出（answer extraction）: モデルが書いた長い推論文から、最終的な数値や選択肢だけをきれいに取り出す処理。自由文に答えが埋もれていると自動採点しにくいため、専用の段階を設けます。
- 標準ゼロショット（standard zero-shot）: 呼び水なしで「答えは？」とだけ聞く素朴な比較対象。Zero-shot CoT の効果は、この標準ゼロショットに対する伸びで測られます。

## 仕組みを詳しく
Zero-shot CoT は2段階のプロンプトから成り立ちます。これが通常の一発質問応答との大きな違いです。

第1段階（推論を引き出す、reasoning extraction）:

```
Q: ジャグラーが16個のボールをジャグリングできます。
   半分はゴルフボールで、そのうち半分が青いです。青いゴルフボールは何個？
A: Let's think step by step.
```

`A:` の後にいきなり答えを書かせるのではなく `Let's think step by step.` を置きます。するとモデルはこれに続けて推論文を生成します。

```
   ボールは全部で16個あります。半分がゴルフボールなので 16 ÷ 2 = 8個。
   そのうち半分が青いので 8 ÷ 2 = 4個です。
```

第2段階（答えを抜き出す、answer extraction）:

第1段階で生成された推論文の後ろに、答えを取り出すための一文（例: `Therefore, the answer (arabic numerals) is`、日本語なら「したがって、答えは」）を付け、もう一度モデルに続きを生成させます。

```
（上の推論文）...青いゴルフボールは4個です。
Therefore, the answer (arabic numerals) is
→ 4
```

なぜ2段階に分けるか。第1段階は「考えること」に専念させ、第2段階で「考えた結果から数値や選択肢だけをきれいに取り出す」役割分担をしているからです。1段階だと推論と最終答えが文章中で混ざり、プログラムで自動抽出するのが難しくなります。実装上は、第1段階の出力をそのまま第2段階のプロンプトに連結して2回目の生成を行います。第2段階の抽出用フレーズもタスク形式（数値か、選択肢か、Yes/No か）に応じて少しだけ替えると安定します。

ポイントは、お手本（例題と模範解答のペア）が一つもないことです。`Let's think step by step.` という汎用の指示文だけで、算数・常識・論理など種類を問わず推論を誘発できます。Few-shot CoT がタスクごとにお手本を作る必要があったのに対し、この一文を使い回せるのが最大の利点です。Few-shot CoT との関係を整理すると、Zero-shot CoT は「お手本 r1, r2, … を見せて推論分布へ誘導する」代わりに「指示文 t によって同じ推論分布へ誘導する」もの、と言えます（記号: r はお手本の推論、t は呼び水）。

## 手法の系譜と主要論文
- 土台: Wei et al. 2022「Chain-of-Thought Prompting」(NeurIPS 2022, arXiv:2201.11903) が Few-shot 版 CoT を提案。お手本に推論を書き込んで真似させる方式（[Chain-of-Thought (CoT) プロンプティング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0529_chain-of-thought-prompting)）。

- 提案: Kojima, Gu, Reid, Matsuo, Iwasawa 2022「Large Language Models are Zero-Shot Reasoners」(NeurIPS 2022, arXiv:2205.11916)。新規性は、手作りお手本を完全に不要にし、汎用の一文で推論を誘発した点。著者らは「大規模モデルには明示的に引き出されていないだけで汎用の推論能力が眠っている」と主張しました。Matsuo・Iwasawa は東京大学の研究者で、産学の共同成果です。

- 自動化: Zhang et al. 2023「Automatic Chain of Thought Prompting」(Auto-CoT, ICLR 2023, arXiv:2210.03493) は、Zero-shot CoT で自動生成した推論をクラスタリングして代表例を選び、それを Few-shot のお手本として自動構築しました。質問を意味ベクトルでクラスタに分け、各クラスタから代表質問を1つ取り、それに Zero-shot CoT で推論を付けてお手本にする、という流れです。人手でのお手本作成を完全に自動化し、Few-shot CoT の精度を手作りに近づけた点が貢献です。「Zero-shot CoT を Few-shot CoT のお手本製造機として使う」という、両者を橋渡しする位置づけです。

- 呼び水の自動探索: Zhou et al. 2023「APE: Large Language Models Are Human-Level Prompt Engineers」(ICLR 2023, arXiv:2211.01910) は、呼び水を人手で探す代わりに LLM 自身に候補文を生成・評価させ、`Let's think step by step.` を上回る指示文（"Let's work this out in a step by step way to be sure we have the right answer." など）を自動発見しました。後の OPRO (Yang et al. 2023, arXiv:2309.03409) も同様に指示文を最適化問題として扱います。呼び水が「魔法の言葉」ではなく探索可能な設計変数であることを示した流れです。

- 後続の頑健化として、生成した推論を複数本サンプリングして多数決する [Self-Consistency](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0544_self-consistency) と組み合わせる使い方が一般的になりました。Zero-shot CoT で推論を出し、それを複数サンプリングして多数決、という組み合わせは実務で定番です。

## 論文の実験結果（定量データ）
Kojima et al. 2022 は、当時の InstructGPT（text-davinci-002, 175B）などを用いて、呼び水なしの標準 zero-shot（単に答えを聞く）と Zero-shot CoT を比較しました。指標はいずれも正答率（最終答えが正解と一致した割合）です。

- MultiArith（複数演算を要する算数文章題）: 約17.7% → 約78.7%。一文を足すだけで4倍以上に跳ね上がりました。
- GSM8K（小学校算数文章題）: 約10.4% → 約40.7%。約4倍の改善です。
- AQuA（選択式の代数文章題）: 約22.4% → 約33.5%。
- 記号推論（Last Letter Concatenation、各単語の最後の文字を連結する等）、Coin Flip（コイン反転）、日付計算、常識推論ベンチマークでも、呼び水なし標準 zero-shot を一貫して大きく上回りました。

アブレーションの要点。著者らは呼び水の文言を多数比較し、「Let's think step by step」が最も効果的だったと報告しています。「Let's solve this problem by splitting it into steps.」や「Let's think about this logically.」「First,」など意味の近い言い換えでも効きますが効果に差があり、一方で無関係・誤誘導的な前置き（例: 感情を煽る文や「とにかく答えだけ言え」系）では効果が出ない、あるいは下がりました。つまり「段階的に考えよ」という意味内容そのものが効いているのであって、ただ文字を増やせばよいわけではない、という結論です。また、効果はモデル規模に強く依存し、小規模モデルではほとんど改善しませんでした（[Chain-of-Thought (CoT) プロンプティング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0529_chain-of-thought-prompting) 同様の創発的性質）。2段階構成についても、推論抽出と答え抽出を分けることで自動採点の精度が安定すると報告されています。

数値の意味を補足すると、17.7%→78.7% は「ほとんど解けなかった問題群の大半が解けるようになった」ことを意味し、しかもタスク固有のお手本を一切作っていない点で、「汎用の推論誘発」が一文で可能だと示した結果です。ただし MultiArith のような一部タスクでは Zero-shot CoT が Few-shot CoT に肉薄する一方、GSM8K では依然 Few-shot CoT（手作りお手本つき）のほうが高く、手本ありが優位な場面も残ります。APE による呼び水最適化では、MultiArith・GSM8K でさらに数ポイント上積みされ、文言の選択が無視できない効きを持つことが定量的に確認されました。

## メリット・トレードオフ・限界
メリット:
- お手本を一切用意しなくてよい。`Let's think step by step.` の一文を全タスクに使い回せる。
- 実装が極めて簡単（プロンプトに1行足すだけ）。エンジニアリングコストがほぼゼロ。
- 新しいタスクへ即適用でき、タスクごとのお手本作りやチューニングが要らない。お手本の選び方による成績のぶれも避けられる。

トレードオフと限界:
- 手作りお手本を持つ Few-shot CoT に精度で劣る場面がある（特に複雑な算数や、出力形式の制御が必要なタスク）。
- 呼び水の文言に性能が敏感（プロンプト感度）。最適な一文は言語・モデル・タスクで変わりうるため、移植時に再探索が要ることがある。
- 2段階（推論→抽出）でモデルを2回呼ぶため、コストと遅延が増える。
- 大規模モデルでのみ有効で、小モデルでは効きにくい。
- もっともらしいが誤った推論を生む危険は CoT と同様に残り、生成説明の忠実性（本当に判断根拠か）は保証されない。

研究上の論点として、なぜ特定の一文が効くのかの理論的説明はまだ完全ではなく、事前学習コーパス中の「段階的説明パターン」の分布に依存するという経験的理解にとどまっています。また指示チューニング済みモデルでは呼び水なしでも推論を出すことが増え、Zero-shot CoT の「呼び水としての効き」がモデル世代によって相対的に薄れる傾向も観測されています（推論傾向がすでに学習で焼き込まれているため）。

## 発展トピック・研究の最前線
- Auto-CoT のように Zero-shot CoT を「お手本自動生成器」として使い、Few-shot CoT を無人で組み立てる方向。
- Self-Consistency や検証器（verifier）と組み合わせて頑健化する方向（[Self-Consistency](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0544_self-consistency)）。
- 指示チューニング（instruction tuning）や RLHF で「段階的に考える」傾向をモデルに焼き込み、呼び水なしでも推論が出るようにする方向。近年の推論特化モデルはこの延長で、推論時計算（test-time compute）を増やすことを学習目標に取り込んでいる。
- 言語・ドメイン横断での最適な呼び水探索（APE・OPRO のような自動プロンプト最適化）。呼び水をハイパーパラメータとして探索可能にする流れ。
- マルチモーダルへの展開: 画像つき質問でも `Let's think step by step` 相当の誘発が有効か、視覚的根拠をどう挟むか、という方向。自動運転の説明生成（[Reason2Drive](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0782_reason2drive)）はこの周辺に位置する。

## さらに学ぶための関連トピック
- [Chain-of-Thought (CoT) プロンプティング](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0529_chain-of-thought-prompting)
- [Self-Consistency](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0544_self-consistency)
- [Tree of Thoughts (ToT)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0547_tree-of-thoughts)
- [ReAct (Reasoning + Acting)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0543_react-reasoning-acting)
- [Reason2Drive](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0782_reason2drive)
