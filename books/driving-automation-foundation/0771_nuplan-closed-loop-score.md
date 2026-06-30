---
title: "nuPlan Closed-Loop Score (CLS)"
---

# nuPlan Closed-Loop Score (CLS)

## ひとことで言うと
nuPlan Closed-Loop Score(CLS)とは、自動運転の計画(planning)を実際に車を動かして評価し、「どれだけ進めたか」「ぶつからなかったか」「乗り心地は良いか」「交通ルールを守ったか」を一つの総合点(0〜1)にまとめた、nuPlan ベンチマークの代表スコアです。安全に関わる重大違反は乗算ペナルティとして強く効き、その他の良し悪しは重み付き加点として効く、という二段構えの設計が特徴です。

## 直感的な理解
自動運転モデルの良し悪しを一つの数字で言いたい、というニーズがあります。しかし「安全に運転できたか」は一面では測れません。たとえば次のような状況です。

- ぶつからなかったが、怖くて 1 メートルも進まなかった。これは安全だが役立たずです。
- たくさん進んだが、何度も急ブレーキで同乗者が気持ち悪い。これは進むが不快です。
- スムーズだったが、信号無視・車線逸脱をしていた。これは危険です。

つまり「進行度」「安全(衝突回避)」「快適性」「規則遵守」を全部バランスよく見ないと、本当に良い運転かは判断できません。これらを一つのスコアに統合したのが nuPlan の Closed-Loop Score です。とくに「いくら進んでも、ぶつかったら台無し」という価値観を、乗算ペナルティという仕組みで明確に表現しているのが設計の肝です。

nuPlan は実走行データに基づく大規模な計画ベンチマークで、ボストン・ピッツバーグ・ラスベガス・シンガポールの実走行データ(約 1300 時間)を使います。2023 年のチャレンジで広く使われ、計画手法の標準的な評価基盤になりました。

「クローズドループ(closed-loop、閉ループ)」なので、モデルの計画で実際にシミュレータの車を動かし、その結果が次の入力になります(詳しくは [クローズドループ評価 (Closed-Loop)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0745_closed-loop-evaluation))。だから誤差の累積や分布シフトも反映されます。録画上で 1 ステップ予測を比べるだけの [オープンループ評価 (Open-Loop)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0695_open-loop-evaluation) とは、ここが本質的に違います。

## 基礎: 前提となる概念
計画(planning)とは、自車がこれからたどるべき軌跡を決めることです。知覚(perception)が「周りに何があるか」を把握し、予測(prediction)が「他車がどう動くか」を見込んだうえで、計画が「では自分はどう動くか」を決めます。CLS はこの計画の良し悪しを測ります。

コントローラとシミュレーションの関係も前提です。CLS では、モデルが出した計画(軌跡)を LQR コントローラ(目標軌跡を追従する制御器)で実行してシミュレーションを進めます。つまり「計画を出すモデル」と「計画を実行する制御器」と「世界を進めるシミュレータ」の三者が協調して動きます。

IDM(Intelligent Driver Model)も知っておくと良い概念です。これは前の車との車間・速度差から加減速を決める、古典的で頑健な追従モデルです。nuPlan の反応モードでは、他のエージェントがこの IDM で自車に反応して動きます。後述する PDM もこの IDM を計画の中核に使います。

## 仕組みを詳しく
nuPlan には 2 つのクローズドループモードがあります。非反応(non-reactive)モードでは、他のエージェント(車・歩行者)はログどおりに動き、自車だけをモデルで動かします。反応(reactive)モードでは、他のエージェントが IDM で自車に反応して動くので、よりリアルです。どちらも自車の計画を LQR コントローラで実行してシミュレーションします。

CLS は複数のサブ指標を組み合わせます。大きく分けて 2 種類です。

乗算ペナルティ(これに引っかかると、スコアが 0 に近づく重大違反):
- 衝突なし(no at-fault collision): 自分の責任でぶつかったらほぼ 0 点。
- 走行可能領域順守(drivable area compliance): 道路・車線を大きく外れたら 0 点。
- 走行方向順守(driving direction compliance): 逆走などをしないこと。

重み付き加点(これらは部分点として効く):
- 進行度(ego progress): 目標ルートに沿ってどれだけ進めたか。
- Time-to-collision(TTC): 衝突までの余裕時間。安全マージン。
- 快適性(comfort): 横加速度・縦/横ジャーク・ヨーレートが nuPlan の閾値内か([Comfort Violation Rate](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0670_comfort-violation-rate)、[横加速度 (Lateral Acceleration)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0689_lateral-acceleration) 参照)。
- 速度制限順守(speed limit compliance)。

計算の形は、概念的には次のとおりです(正確な係数は devkit のバージョンで変わります)。

```
CLS = (乗算ペナルティの積) × (重み付き加点の合計)
```

乗算ペナルティは「衝突したか」「道路を外れたか」のような重大事項で、一つでもアウトなら全体が大きく下がります。重み付き加点は、その上で「どれだけ上手に・快適に・速く進めたか」を 0〜1 で評価します。最終的に各シナリオで 0〜1 のスコアが出て、全シナリオの平均が報告されます。1.0 に近いほど「人間のように安全・快適・スムーズに走り切った」を意味します。

数値例で感覚をつかみます。あるシナリオで、衝突なし(ペナルティ 1.0)・道路順守(1.0)で、進行度 0.9・TTC 良好・快適 0.8・速度順守 1.0 だったとします。加点の重み付き平均が例えば 0.88 になると、CLS ≈ 1.0 × 0.88 = 0.88 です。逆に途中で軽く接触すると衝突ペナルティが 0 近くになり、他がどれだけ良くても CLS ≈ 0 まで落ちます。この「一発アウト」設計が、安全最優先という価値観を数式に焼き込んでいます。

なお nuPlan は、参考のためオープンループスコア(OLS)も同時に報告します。OLS と CLS が同じ手法で大きく食い違うことがあり、これが [Open-Loop/Closed-Loop ギャップ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0776_open-loop-closed-loop-gap) を実データで観測する場になっています。

## 手法の系譜と主要論文
Caesar ら(nuPlan, Motional, 2021)は「世界初の大規模クローズドループ計画ベンチマーク」を掲げて公開しました。動機は、それまでの計画評価がオープンループ(録画上の L2 誤差)中心で、実運用性能と相関しない([Open-Loop/Closed-Loop ギャップ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0776_open-loop-closed-loop-gap))という問題意識です。衝突・進行度・快適性・規則遵守を統合した CLS により、計画手法を実運用に近い形で公平に比較できるようになりました。

Dauner ら(2023, "Parting with Misconceptions about Learning-based Vehicle Motion Planning", PDM, CoRL)は、nuPlan Challenge 2023 で優勝した PDM(Predictive Driver Model)を提案しました。中心アイデアは意外なほど単純で、複数の候補軌跡(IDM で異なる目標速度・横ずらしを与えて生成)をシミュレーションで先読みし、最も CLS が高くなる軌跡を選ぶ、というものです。学習をほとんど使わないこの手法が、当時の凝った学習ベース手法を上回りました。

Dauner ら(2024, "NAVSIM", NeurIPS)は、nuPlan の CLS の思想(乗算ペナルティ × 重み付き加点)を継承しつつ、より軽量な非反応 4 秒シミュレーションで計算する PDMS を提案しました。nuPlan CLS が「フルクローズドループで重い」のに対し、PDMS は「軽くてオープンループより相関が良い中間」を狙ったもので、両者は思想を共有する親戚関係にあります(詳しくは [クローズドループ評価 (Closed-Loop)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0745_closed-loop-evaluation))。

## 論文の実験結果(定量データ)
nuPlan Challenge 2023 の最も重要な実験的知見は、ルールベースの PDM-Closed が学習ベース手法を closed-loop score で上回ったことです。報告によれば、PDM-Closed は非反応クローズドループでおよそ 0.9 前後の高い CLS を達成し、当時の代表的な学習ベース計画器(例えば模倣学習ベースの手法)を明確に上回りました。一方、これらの学習ベース手法はオープンループスコアでは PDM に匹敵、あるいは上回る場合もあり、「オープンループで強い手法とクローズドループで強い手法が一致しない」ことがベンチマーク全体で観測されました。

この対比の意味を噛み砕きます。CLS は 0〜1 のスコアで、1.0 が「人間のように安全・快適に走り切った」状態です。PDM-Closed が約 0.9 を出すというのは、ほとんどのシナリオで重大違反なく走り切れた、ということです。学習ベース手法が CLS で劣るのは、オープンループでは人間軌跡を上手に真似られても、いざ自分で走らせると分布シフトで少しずつズレ、乗算ペナルティ(衝突・道路逸脱)に引っかかってスコアを大きく落とすからです。これは、CLS という評価軸そのものの性質(累積誤差・規則遵守を厳しく見る)が手法選択に影響する好例です。

アブレーション的に見ると、PDM の強さは「シミュレーションで候補を先読みして選ぶ」点にあります。先読み(候補を CLS シミュレータで評価して選ぶ)を外すと性能が落ちることが示され、クローズドループ評価軸では「評価指標をそのまま最適化に使う」ことが効く、という知見が得られました。これは、評価指標が単なる採点ではなく、設計と最適化の指針にもなることを示しています。

## メリット・トレードオフ・限界
メリットとして、進行度・安全・快適性・規則遵守を一つの数字に統合でき、手法を実運用に近い形で公平に比較できます。クローズドループなので誤差累積・分布シフトを反映し、オープンループ L2 より実性能との相関が高い。乗算ペナルティ設計により「ぶつかったら台無し」という安全最優先の価値観を明確に表現できます。

トレードオフと限界もあります。フルクローズドループは計算が重く、シナリオ準備・実行コストが高い。LQR コントローラや IDM 他車モデルといったシミュレーション前提が現実と異なる(sim-to-real gap)。乗算ペナルティで一度の重大違反が全体を 0 近くまで落とすため、部分的な改善が見えにくいことがあります。さらに、単一スコアに集約するため、どの要素(進行度か快適性か)で点を落としたかは内訳を別途見る必要があります。反応モードでも IDM の挙動が現実の人間ほど多様でない、という限界も残ります。

研究上の未解決課題は、(1) シミュレーション前提を現実に近づける(他車モデルの多様化、高忠実度シミュレーション)、(2) 乗算ペナルティの「一発アウト」と部分点の見やすさのバランス、(3) このスコアが本当に実車安全性と相関するかの検証、です。

## 発展トピック・研究の最前線
nuPlan CLS の思想は、より軽量な proxy(NAVSIM の PDMS)へ受け継がれ、エンドツーエンド運転の標準評価として広がりつつあります。最近は、CLS のような統合スコアを学習目的に組み込む(微分可能な近似を作る、あるいは PDM のように評価器を選択に使う)研究や、反応エージェントをデータ駆動で現実的にモデル化する研究が進んでいます。また、CLS が苦手とする「長尾の稀シナリオ」を重点的に評価するシナリオマイニングや、快適性閾値([Comfort Violation Rate](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0670_comfort-violation-rate)、[横加速度 (Lateral Acceleration)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0689_lateral-acceleration))の妥当性を人間の主観評価と突き合わせる研究も、評価の信頼性を高める方向として注目されています。

## さらに学ぶための関連トピック
- [クローズドループ評価 (Closed-Loop)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0745_closed-loop-evaluation)
- [オープンループ評価 (Open-Loop)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0695_open-loop-evaluation)
- [Open-Loop/Closed-Loop ギャップ](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0776_open-loop-closed-loop-gap)
- [Comfort Violation Rate](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0670_comfort-violation-rate)
- [横加速度 (Lateral Acceleration)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0689_lateral-acceleration)
