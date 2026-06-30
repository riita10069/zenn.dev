---
title: "Agent-Driver"
---

# Agent-Driver

## ひとことで言うと
Agent-Driver は、大規模言語モデル(LLM)を「賢い運転手の頭脳」として中心に据え、その頭脳が「道具(ツール)」「記憶(メモリ)」「推論エンジン」を呼び出しながら運転判断を組み立てる仕組みです。前身の GPT-Driver が「LLM に軌道を一気に書かせる」だけだったのに対し、Agent-Driver は LLM を1回答えるだけの装置ではなく、複数の機能を使い分ける「エージェント(自律的に行動を選ぶ主体)」として動かす点が新しいのです。人間ドライバーの認知(知覚し・思い出し・考えて・行動する一連の働き)を LLM エージェントとして模倣した、と言えます。

## 直感的な理解
人間が運転するとき、周囲のすべてを同じ重さで見ているわけではありません。「今、注意すべき相手」に視線を集中し(能動的な注意)、「前にも似た交差点を通ったな」と過去の経験を思い出し(記憶)、「あの車が止まっているから自分はこうしよう」と筋道立てて考えます(推論)。この一連の働きが運転の認知です。

前身の GPT-Driver([GPT-Driver](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0538_gpt-driver))は、知覚結果をすべてテキストにして LLM に渡し、軌道を一気に出力させる方式でした。シンプルですが、2つの問題がありました。1つ目は、情報を一度に全部テキストで渡すとプロンプトが長くなりすぎ、本当に重要な物体(すぐ前の車)が膨大な背景情報に埋もれること。2つ目は、その場限りの判断しかできず、過去の経験を蓄えて引き出す仕組みがないことです。

Agent-Driver は、人間の認知に倣ってこれを解こうとしました。「最初から全部渡す」のではなく「必要な情報を必要なときに取りに行く」、そして「過去の似たシーンを思い出して参考にする」。LLM を、受け身の生成器から、能動的に道具と記憶を使うエージェントへと格上げしたのです。これは汎用 AI 研究で「LLM エージェント」と呼ばれる潮流を、運転に持ち込んだ試みでもあります。

## 基礎: 前提となる概念
用語を順に噛み砕きます。

LLM エージェント(LLM agent)とは、LLM を司令塔にして、外部の道具や情報源を呼び出しながら多段で行動する仕組みです。LLM が「次に何をすべきか」を決め、その指示で外部の関数を実行し、結果を見てまた次を決める、というループを回します。1回の入力に対し1回出力する素朴な使い方との違いがここにあります。

ツール使用(tool use / function calling)とは、LLM がプログラムの関数(ツール)を呼び出して、自分にはできない計算やデータ取得を外部に任せる技術です。たとえば「最も近い物体を取得する関数」を LLM が選んで実行し、結果を受け取ります。LLM は正確な数値計算が苦手なので、計算は外部関数に任せるのが定石です。

RAG(Retrieval-Augmented Generation、検索拡張生成)とは、LLM が答える前に、外部の知識ベースから関連情報を検索して取り込み、それを踏まえて回答する手法です。Agent-Driver では「今と似た過去の運転シーン」を検索して参考にする部分がこれに当たります。

ベクトル検索(vector search)とは、対象(ここでは運転シーン)を数値の列(ベクトル、埋め込み)に変換し、ベクトル空間で近いもの(似ているもの)を探す技術です。「似たシーン」を効率よく見つけるのに使います。

Chain-of-Thought(CoT)とは、答えの前に途中の考えを文章で書かせる手法です。Agent-Driver の推論エンジンは、CoT で「どの物体に注意するか → 危険はあるか → 高レベル行動 → 軌道」と段階を踏みます。

few-shot 学習とは、ごく少数の例だけで新しいタスクに適応できること、という意味です。Agent-Driver は経験メモリと言語事前知識のおかげで、少量データでも妥当に振る舞えると報告されています。

## 仕組みを詳しく
Agent-Driver は LLM を司令塔に置き、その周りに3つの部品を配置します。

1. ツールライブラリ(tool library): LLM が必要に応じて呼び出せる関数の道具箱です。「自車周りで最も近い物体を取得」「ある物体の未来の動きを予測」「地図上の車線情報を取得」「占有グリッド(空き/塞がりの格子)を取得」といった関数が並びます。LLM は「今は前方車両の予測が知りたい」と判断したら、その関数だけを呼びます。最初から全情報をテキストで渡すのではなく、必要な情報を必要なときに取りに行く(神経科学でいう能動的な情報収集)形になり、重要物体が長いプロンプトに埋もれにくくなります。

2. 認知メモリ(cognitive memory): 過去の運転知識を貯める記憶で、2種類あります。
   - コモンセンスメモリ(常識記憶): 「交差点では一時停止する」「歩行者優先」のような交通ルールや運転の常識を文章で蓄えたもの。
   - 経験メモリ(experience memory): 過去に遭遇したシーンと、そのとき取った良い判断のペアを貯めたもの。新しいシーンに出会うと、シーンをベクトル化して「今と似た過去シーン」を検索し、その判断を参考にします。これは RAG の運転版です。

3. 推論エンジン(reasoning engine): 集めた情報と思い出した経験をもとに、LLM が CoT で段階的に結論を出す部分です。「どの物体に注意すべきか」→「危険はあるか」→「高レベル行動方針(減速・車線変更など)」→「最終的な未来軌道の座標列」という順に進みます。さらに、出した計画を自己検証(self-reflection)して、危険があれば修正する経路も持ちます。

処理の流れを数値例で追うと:

```
ステップ1(知覚をテキスト化):
  "自車速度 8.0 m/s。物体: 前方車 A(前 15m, 6 m/s)、歩行者 B(右 4m)"
        │
ステップ2(ツール呼び出し):
  LLM が「A の未来予測を取得」「前方の車線情報を取得」関数を選んで実行
        │
ステップ3(メモリ検索):
  経験メモリから「前方に減速車、右に歩行者」という似た過去シーンを取り出す
        │
ステップ4(CoT 推論):
  "A が減速中で右に歩行者 → 車線変更は危険。緩やかに減速し車間を保つ"
        │
ステップ5(出力 + 自己検証):
  3秒ぶんの軌道座標 (x1,y1)...(x6,y6) を生成し、衝突がないか点検
```

このように、Agent-Driver は LLM を「1問1答の生成器」ではなく「道具と記憶を使いながら多段で考えるエージェント」に拡張しています。GPT-Driver との決定的な違いは、情報を一括で渡さず能動的に取りに行く点と、過去の経験を蓄えて再利用できる点です。

## 手法の系譜と主要論文
Agent-Driver は、汎用の LLM エージェント研究を運転に応用した代表例です。

思想の源流は、LLM をエージェント化する一般的潮流にあります。ReAct (Yao et al., ICLR 2023, arXiv:2210.03629) は、推論(Reasoning)と行動(Acting)を交互に行わせ、考えながら外部ツールを使う枠組みを示しました。Toolformer (Schick et al., NeurIPS 2023, arXiv:2302.04761) は、LLM に API 呼び出しを自分で学習させる手法を提案しました。Reflexion (Shinn et al., NeurIPS 2023) は、失敗を言語で振り返って次に活かす自己反省の枠組みを示しました。Agent-Driver は、これらの「LLM がツールを使い、経験を貯め、振り返って考える」発想を運転判断に持ち込んだものと位置づけられます。

直接の前身は GPT-Driver (Mao et al., 2023, arXiv:2310.01415) で、Agent-Driver は同じ著者グループによる発展形です([GPT-Driver](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0538_gpt-driver))。GPT-Driver の「全情報を一括テキスト化」「経験を蓄えられない」という限界を、ツール・メモリ・推論エンジンで克服しました。

比較ベースラインには、前身 GPT-Driver に加え、知覚・予測・計画を統合した end-to-end 手法 UniAD (Hu et al., CVPR 2023, Best Paper) が用いられます([UniAD (Planning-oriented統合E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1090_uniad_note))。

Agent-Driver (Mao, Ye, Qian, Pavone, Wang, 2023, arXiv:2311.10813, "A Language Agent for Autonomous Driving", COLM 2024) の新規性は、ツールライブラリ・認知メモリ(常識+経験)・CoT 推論エンジンを LLM に統合し、人間の認知プロセスを模した運転フレームワークを構築した点にあります。後続では DiLu([DiLu](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0531_dilu_note))が知識駆動・経験蓄積をさらに強調し、LanguageMPC([LanguageMPC](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0540_languagempc))が言語判断と数理制御を橋渡しするなど、エージェント路線が枝分かれしていきます。

## 論文の実験結果(定量データ)
評価は nuScenes の planning タスクで行われました。指標は GPT-Driver と同じく、L2 誤差(計画軌道と人間の実軌道の距離のずれ、メートル単位、小さいほど良い)と衝突率(計画軌道が他物体とぶつかる割合、パーセント、小さいほど安全)です。

Agent-Driver は、前身の GPT-Driver、および end-to-end ベースラインの UniAD を上回る L2 誤差と衝突率の低減を報告しました。報告された平均 L2 誤差はおよそ 0.3m 前後の水準で GPT-Driver より小さく、平均衝突率も低減したとされます。すなわち、LLM を単に軌道生成に使うより、ツールで能動的に情報を集め、経験メモリで過去の判断を参照し、多段で推論させる方が、計画精度も安全性も高まったということです。これは「LLM を生成器からエージェントへ格上げする」設計が定量的に有効だったことを示します。

データ効率に関する知見も重要です。経験メモリと言語事前知識のおかげで、Agent-Driver は少数のデータでも適応できる(few-shot 学習が効く)ことが示されました。報告では、訓練データの 1〜数% 程度の少量でも、フルデータで学んだ専用モデルに迫る計画性能を出せたとされます。新しい状況でも、似た過去シーンを引っ張り出して参考にできるため、ゼロから学び直す必要が減ります。

アブレーション(各部品を抜く分析)の示唆として、3つの部品それぞれが性能に寄与することが報告されています。ツールライブラリを外して全情報を一括で渡すと重要物体が埋もれて性能が落ち、経験メモリを外すと過去の判断を活かせず汎化が下がり、CoT 推論を浅くすると多段判断の質が落ちる、という形です。とりわけ経験メモリは諸刃の剣で、質の良い経験を貯めれば判断が改善する一方、不適切な経験を貯めると悪い判断を再生産しうるという依存性が指摘されています。なお、これらの数値も nuScenes の open-loop 指標(直進バイアスのショートカットが存在する)の上で読む必要があり、閉ループでの優位は別途検証が要ります。

## メリット・トレードオフ・限界
メリット

- 必要な情報だけをツールで取りに行くため、重要物体が長いプロンプトに埋もれにくい。
- 経験メモリで過去の判断を再利用でき、少数データでも適応しやすい(few-shot)。
- 多段の CoT で判断根拠が説明可能になり、解釈性が高い。
- 前身 GPT-Driver や end-to-end の UniAD より高い計画精度・低い衝突率を達成。

トレードオフ・限界

- ツール呼び出しやメモリ検索で LLM を複数回呼ぶため、1回出力の GPT-Driver よりさらに遅く、計算・API コストが高い。高頻度の車載リアルタイムには厳しい。
- 部品(ツール/メモリ/推論)が多く、システムが複雑で保守が大変。各部品の不具合が全体に波及しうる。
- 経験メモリの質に判断が依存する。不適切な経験を貯めると、似た状況で悪い判断を再生産する恐れがある(memory poisoning に類する問題)。
- 知覚をテキスト化する段階での情報損失は GPT-Driver と同様に残り、ツールが返す結果も元の知覚精度に縛られる。
- 評価は主にオフライン(open-loop)で、実車の閉ループ運用・長期安全性の実証はこれから。

研究上の未解決課題としては、複数回の LLM 呼び出しを軽量化して実時間に乗せること、経験メモリの質を保証・更新する仕組み(悪い経験のフィルタリング)、そして閉ループでの安全性検証が挙げられます。

## 発展トピック・研究の最前線
Agent-Driver が示した「運転を LLM エージェントで解く」流れは、後続研究で多様に展開しています。DiLu([DiLu](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0531_dilu_note))は、常識推論と経験の蓄積を強調し、運転を知識駆動で行うエージェントを提案しました。LanguageMPC([LanguageMPC](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0540_languagempc))は、LLM の判断を低レベル制御(MPC、モデル予測制御)と結びつけ、言語の高レベル判断と数理最適化の精密さを橋渡しします。

研究全体の力点は、「賢いが遅いエージェント」をいかに実用速度にするかへ移っています。複数回の LLM 呼び出しを減らす、賢いエージェントの判断を軽量モデルへ蒸留する、といった軽量化が主戦場です。また、経験メモリを安全に運用するため、悪い経験をフィルタする仕組みや、エージェントの判断を別モデルで検証する仕組み(運転版の自己検証)も研究が進んでいます。知覚のテキスト化による情報損失を避ける VLM 方向([DriveVLM](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0535_drivevlm) [OmniDrive](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0542_omnidrive))や、推論を構造的に評価する Graph VQA([DriveLM (Graph VQA)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0534_drivelm))とも合流しつつあり、エージェント・VLM・基盤モデルの境界はしだいに溶けつつあります。

## さらに学ぶための関連トピック
- [GPT-Driver](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0538_gpt-driver)
- [LanguageMPC](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0540_languagempc)
- [DiLu](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0531_dilu_note)
- [OmniDrive](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0542_omnidrive)
- [DriveVLM](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0535_drivevlm)
- [DriveLM (Graph VQA)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0534_drivelm)
- [UniAD (Planning-oriented統合E2E)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1090_uniad_note)
