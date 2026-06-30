---
title: "FDE (Final Displacement Error)"
---

# FDE (Final Displacement Error)

## ひとことで言うと
予測した進路の「一番最後の地点」だけを取り出して、それが実際の最後の地点からどれだけ離れているかを測る指標です。ADE が全時刻の平均だったのに対し、FDE は終点 1 点だけを見ます。「N 秒後に車や人がどこにいると予測したか」がどれだけ当たっていたかを表し、長期予測の信頼性を厳しく評価する標準指標です。

## 直感的な理解
ADE(全時刻の平均誤差、[ADE (Average Displacement Error)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0663_ade-average-displacement-error))には弱点があります。「平均してしまう」ことです。

具体例。予測軌跡が「最初の数秒はぴったり正解に沿っているが、最後だけ大きく曲がって外れる」場合を考えます。

```
時刻 0.0〜3.0s: ほぼ正解と一致 (誤差ほぼ0)
時刻 6.4s:      正解は直進、予測は右に5mずれた
```

このとき、最初の大量の「誤差ほぼ 0」が平均を引き下げるので、ADE は小さく出ます。「平均的にはよく当たっている」という評価です。しかし運転の安全という観点では、終点で 5m もずれているのは大問題で、隣の車線や歩道に突っ込んでいるかもしれません。

つまり「長期予測がどれだけ信頼できるか」「遠い未来のズレ」を、平均で薄めずに直接見たい、という動機があります。これに応えるのが FDE です。終点だけを見ることで、予測の「最終到達点の正確さ」をピンポイントで評価できます。予測時間が長くなるほど誤差は累積して大きくなりやすいので、終点はモデルにとって最も難しい部分で、FDE はその一番難しいところを点数化します。

## 基礎: 前提となる概念
- 軌跡(trajectory): 時刻ごとの位置を並べたもの。最終時刻の位置が FDE の対象。
- 予測ホライズン(prediction horizon): 何秒先まで予測するか。FDE はこの最終時刻のズレを測るので、ホライズンが長いほど FDE は大きくなりやすい。
- 誤差の累積(error accumulation): 制御量(加速度・曲率)を積分して位置を作る場合、初期のわずかな誤差が後ろの時刻ほど積み上がる。終点が最も誤差を被る。
- ユークリッド距離(Euclidean distance): 2 点間の直線距離。FDE は終点同士のこの距離。
- 目標地点(goal / endpoint): 軌跡の最終到達点。これを先に予測してから途中を埋める設計があり、FDE と直結する。
- ゲート(gate, 関所): モデルを採用してよいかの合否判定。FDE はしきい値判定に使いやすく、安全上の最低ラインを引くのに向く。

## 仕組みを詳しく
FDE の計算は非常にシンプルです。

ステップ 1: 予測軌跡と正解軌跡の、それぞれ最後の時刻の位置を取り出す。
ステップ 2: その 2 点のユークリッド距離を計算する。

```
FDE = || pred_T - gt_T ||
```

T は最終時刻、`pred_T` は予測の終点、`gt_T` は正解の終点です。

数値例:

```
最終時刻の予測位置: (12.0, 1.5)
最終時刻の正解位置: (12.0, 0.0)
FDE = sqrt((12-12)^2 + (1.5-0)^2) = sqrt(2.25) = 1.5 m
```

「N 秒後の到達点が 1.5 メートルずれていた」という意味です。

ADE との違いを並べると分かりやすいです。同じ 64 時刻の軌跡に対して:

- ADE: 64 個の各時刻の誤差を全部平均する。
- FDE: 64 個のうち最後の 1 個だけを使う。

実装も終点を取るだけです。

```python
errors = np.linalg.norm(pred_xy - gt_xy, axis=1)  # (T,) 各時刻の距離
fde = errors[-1]                                   # 最後の要素 = FDE
```

`errors[-1]` が配列の最後の要素、つまり最終時刻の誤差です。たとえば 10Hz(0.1 秒ごと)で 64 時刻、合計 6.4 秒先まで予測するなら、この最終時刻は 6.4 秒後にあたり、指標名は `FDE@6.4s` になります。何秒後を見るかで指標名が変わるため、ホライズンを明記するのが慣例です。

FDE の使われ方で重要なのが「ゲート(関所)」としての利用です。FDE は終点 1 点なので解釈が明快で、安全上の最低ラインをしきい値で引きやすい。たとえば「ADE@3s ≤ 2.0m かつ FDE@6.4s ≤ 5.0m なら合格」のように複数指標と組み合わせ、長期予測が一定以上ぶれるモデルを自動で弾く運用が一般的です。ベンチマークでは Miss Rate(終点誤差が閾値、例 2m を超えたサンプルの割合)の基礎にも FDE が使われ、「大外れ判定」の中核になっています([Miss Rate (MR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0763_miss-rate))。

多峰(複数本予測)への拡張も ADE と同じで、K 本のうち終点が最も近い 1 本で FDE を取る minFDE_k が標準です。`minFDE_k = min_{i∈1..K} || pred_i,T - gt_T ||` と書きます。これは「もっともらしい候補を 1 本でも出せていれば終点を評価する」という考え方です([minFDE_k (best-of-K FDE)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0762_min-fde-k))。

## 手法の系譜と主要論文
Social LSTM(Alahi et al., Stanford, CVPR 2016)が ADE と同時に Final displacement error を導入しました。動機は、平均(ADE)だけでは最終到達点の正確さが見えないためです。歩行者予測では「最終的にどこへ着くか」が衝突回避や経路計画に直結するため、終点を別途評価する意味があります。ADE は小さいが FDE は大きい(途中は当たるが最後がぶれる)モデルを見分けられます。

Social GAN(Gupta et al., Stanford, CVPR 2018, arXiv:1803.10892)は FDE を minFDE(K 本中で終点が最良)として使いました。複数の未来がありうる状況では 1 本だけの FDE は過小評価になる、という問題意識からです([minFDE_k (best-of-K FDE)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0762_min-fde-k))。

Argoverse(Chang et al., Argo AI, CVPR 2019, arXiv:1911.02620)と nuScenes prediction(Caesar et al., CVPR 2020, arXiv:1903.11027)は車両予測ベンチマークで FDE / minFDE_k を公式指標化しました。Argoverse は終点誤差が閾値(例 2m)を超えたかで Miss Rate も定義しており、FDE が大外れ判定の基礎になっています。これらのリーダーボードでは minFDE_k がしばしば主たる順位決定指標として使われ、終点精度の競争を促しました。

ここから「終点を先に決める」という設計思想が生まれます。TNT(Target-driveN Trajectory prediction, Zhao et al., CoRL 2020, arXiv:2008.08294)は、まず候補となる目標地点(target)を多数サンプリングし、各目標へ至る軌跡を生成してから絞り込む、という段階的な手法を提案しました。これは「終点予測の質が全体精度を決める」という観察を直接設計に反映したもので、FDE を強く意識した枠組みです。続く DenseTNT(Gu et al., ICCV 2021, arXiv:2108.09640)は、離散的な目標候補の代わりに密な目標確率分布を予測し、アンカー(あらかじめ決めた候補点)に依存しない目標推定を可能にして、minFDE をさらに改善しました。

## 論文の実験結果(定量データ)
Social LSTM の ETH/UCY(過去 3.2 秒観測 → 未来 4.8 秒予測)では、線形外挿や素の LSTM に比べ Social LSTM が FDE を改善しました。歩行者データでは終点誤差がおよそ 1m 前後の水準で、相互作用モデル化により最終到達点の予測がより正確になることが示されました。

Social GAN では K=20 本サンプリングして minFDE を取ることで、同じ ETH/UCY で minFDE をさらに下げ(サブシーンにより 0.5m 台まで)、多峰サンプリングが終点予測に効くことを定量化しました。アブレーションでは多峰生成を外すと minFDE が悪化することが示されています。

Argoverse の未来 3 秒予測では、上位手法の minFDE_6 はおよそ 1.1〜1.4m、minFDE_1 はおよそ 3〜4m 程度という水準が報告されました。ここから分かる重要な事実は 2 つです。第一に、終点誤差(FDE/minFDE)は ADE/minADE よりも常に大きく、長期予測の難しさを如実に映すこと。第二に、許す本数 K を増やすほど minFDE は単調に下がるため、K を揃えないと比較が成立しないこと。アブレーションでは、ADE と同様に車線中心線などの地図文脈を入力すると FDE が大きく改善し、終点の不確実性を減らすうえで地図情報が決定的だと示されました。

TNT / DenseTNT 系では、目標地点を明示的に予測する設計が minFDE をさらに下げることが示されました。DenseTNT は Argoverse で当時の最良水準の minFDE を達成し、特に「アンカーに依存せず密な目標分布を学ぶ」ことで、終点の多様な可能性(交差点での複数の行き先など)を取りこぼしにくくなる効果が報告されています。アブレーションでは、目標予測の段階を外して直接軌跡を回帰すると minFDE が悪化することが示され、終点を先に決める設計の有効性が定量化されました。

## メリット・トレードオフ・限界
メリット
- 長期予測のズレを平均で薄めずに直接見られる。
- 計算が極めて単純で、終点 1 点なので解釈が明快。
- 合格/不合格のしきい値判定(ゲート)や Miss Rate 定義に使いやすく、安全上の最低ラインを引きやすい。
- ADE とセットで見ると「途中は良いが終点が悪い」などの傾向を切り分けられる。

トレードオフ・限界
- 終点 1 点しか見ないので、途中で大きく蛇行しても終点さえ合えば良い評価になる。ADE と併用して補う必要がある。
- 複数のもっともらしい未来(多峰性)を 1 本の FDE では扱えない。これを補うのが [minFDE_k (best-of-K FDE)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0762_min-fde-k)。ただし K を増やせば下がるので K を揃えない比較は無意味。
- 確率の付け方を評価しない。minFDE は最良 1 本だけ見るため、確率の質には別指標(NLL、mAP)が要る。
- 衝突や車線逸脱といった安全そのものは測れず、終点の幾何的なズレのみを見る。
- ホライズン依存が強く、何秒後の FDE かを明記しないと比較できない(FDE@3s と FDE@6.4s は別物)。

## 発展トピック・研究の最前線
FDE は「終点の不確実性」を測るため、確率予測(複数候補とその確率)を扱う研究で重要度が増しています。目標地点(goal/endpoint)を先に予測してから軌跡を埋める「goal-conditioned」手法(TNT, DenseTNT など)は、まさに FDE を直接最適化する設計で、終点予測の質が全体精度を決めるという観察に基づきます。また minADE/minFDE が良くてもクローズドループ(予測を車に反映して走らせる)で破綻する乖離が知られ、nuPlan などの閉ループ評価で「終点が合うこと」と「安全に走れること」のギャップが議論されています。さらに、複数候補の確率を物体検出のように評価する mAP 系指標(Waymo Open Motion など)が併用され、「終点が近いか」だけでなく「もっともらしい終点に高い確率を割り当てられているか」まで測る方向に進んでいます。FDE は依然として軌跡予測の主要指標ですが、それ単体では安全性を保証しない、という限界の認識が研究の前提になっています。

## さらに学ぶための関連トピック
- [ADE (Average Displacement Error)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0663_ade-average-displacement-error)
- [マルチホライズンADE (ADE@1s/2s/3s)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0691_multi-horizon-ade)
- [minFDE_k (best-of-K FDE)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0762_min-fde-k)
- [Miss Rate (MR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0763_miss-rate)
- [クローズドループ評価 (Closed-Loop)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0745_closed-loop-evaluation)
