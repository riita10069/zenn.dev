---
title: "インスタンスセグメンテーション"
---

# インスタンスセグメンテーション

## ひとことで言うと

インスタンスセグメンテーション(instance segmentation)とは、画像に写る物体の一つひとつを、ピクセル単位の輪郭(マスク)で塗り分け、かつそれぞれを別個の物体として区別するタスクです。たとえば3人の歩行者が写っていたら、人物領域を画素レベルで切り抜くだけでなく「これは人A、これは人B、これは人C」と個体まで分けます。自動運転では、各歩行者・各車両の正確な形と境界を知ることが、距離推定や進路予測の前提になります。

## 直感的な理解

似た用語を整理すると一気に分かります。

- 画像分類(image classification): 画像全体に1ラベル。「この写真には猫がいる」。
- 物体検出(object detection): 物体を四角い枠(バウンディングボックス)で囲む。「ここに猫、あそこに犬」。
- セマンティックセグメンテーション(semantic segmentation): 全画素にクラスを塗る。ただし同じクラスの物体同士は区別しない。「これらの画素はすべて人」。
- インスタンスセグメンテーション: 画素を塗り、かつ個体を区別する。「この画素群は人A、あの画素群は人B」。
- パノプティックセグメンテーション(panoptic): 数えられる物体(人・車=things)は個体区別し、数えにくい背景(空・道路=stuff)はまとめて塗る。両者の統合版。

```
分類:        [画像全体 → "人がいる"]
検出:        [□人 □人 □人]  (枠だけ、画素は塗らない)
セマンティック: [人人人で塗る が 3人の区別なし]
インスタンス:  [人A 人B 人C をそれぞれ別の色で塗る]
```

四角い枠より画素マスクのほうが、重なった物体や複雑な形を正確に表せます。これがインスタンスセグメンテーションの価値です。

## 基礎: 前提となる概念

評価指標は mAP(mean Average Precision、平均適合率の平均)が標準です。理解の鍵は IoU(Intersection over Union、交差面積比)。予測マスクと正解マスクについて、重なり面積を和集合面積で割った値です。

```
IoU = (予測 ∩ 正解) / (予測 ∪ 正解)
```

完全一致なら1、まったく重ならなければ0。検出ではボックスの IoU、インスタンスセグメンテーションではマスクの IoU を使います。「IoU が 0.5 以上なら正解とみなす」のように閾値を設けて、適合率(precision、予測のうち当たりの割合)と再現率(recall、正解のうち当てた割合)を計算します。

AP は1クラスについて、閾値を変えながら適合率と再現率の曲線(PR曲線)を描き、その下の面積を取ったもの。COCO データセットでは IoU 閾値を 0.5 から 0.95 まで 0.05 刻みで変えて平均する、厳しい指標(AP@[.5:.95])を採用します。これを全クラスで平均したのが mAP です。マスクの mAP は mask AP、ボックスの mAP は box AP と呼んで区別します。

COCO(Common Objects in Context)は80クラス・約12万枚の標準ベンチマークで、この分野の事実上の物差しです。

## 仕組みを詳しく

### 二段階・検出ベース: Mask R-CNN

Mask R-CNN(He et al., ICCV 2017, arXiv:1703.06870)はこの分野の金字塔です。物体検出器 Faster R-CNN を拡張し、各物体に対してマスクを出す枝(branch)を足しました。流れは次の通り。

```
画像 → [バックボーン CNN(例 ResNet+FPN)] → 特徴マップ
     → [RPN: 物体がありそうな候補枠を提案]
     → 各候補枠について:
         ├─ クラス分類(これは人? 車?)
         ├─ 枠の微調整(box regression)
         └─ マスク予測(枠内を画素ごとに前景/背景に2値分類)
```

RPN(Region Proposal Network、領域提案ネットワーク)は「物体がありそうな場所」を大量に提案する小さなネット。FPN(Feature Pyramid Network)は大小さまざまな物体を捉えるため、複数解像度の特徴を組み合わせる仕組みです。

Mask R-CNN の重要な技術的貢献が RoIAlign です。従来手法(RoIPool)は候補枠の特徴を固定サイズに切り出すとき座標を整数に丸めていて、画素単位のマスクには致命的なズレを生みました。RoIAlign は丸めをやめ、双線形補間(bilinear interpolation、周囲の値から滑らかに中間値を計算)で正確に特徴をサンプリングします。この小さな変更がマスク精度を大きく押し上げました。

さらに、マスク予測とクラス分類を分離した点も巧妙です。マスク枝はクラスごとに独立な2値マスク(前景か背景か)を出し、「どのクラスか」はクラス分類枝に任せます。マスク予測自身がクラス間で競合しないので学習が安定します。

### 一段階・グリッドベース: SOLO

SOLO(Wang et al., "Segmenting Objects by Locations", ECCV 2020, arXiv:1912.04488)は、候補枠を介さず直接マスクを出す発想です。画像を S×S のグリッドに分け、各グリッドセルが「自分の中心にある物体のマスク」を直接予測します。物体の中心位置でインスタンスを区別するので、検出器が不要になりシンプルです。

### クエリベース: Mask2Former

Mask2Former(Cheng et al., CVPR 2022, arXiv:2112.01527)は Transformer の発想で問題を統一しました。学習可能な「クエリ(query)」と呼ぶベクトルを N 個用意し、各クエリが1つのインスタンス(または背景領域)を担当して、その「クラス」と「マスク」のペアを直接出力します。注目すべきは、セマンティック・インスタンス・パノプティックの3タスクを同一アーキテクチャで解けること。タスクごとに別設計を作っていた時代を終わらせた汎用フレームワークです。中核技術として、マスク領域だけに注意を集中させる masked attention を導入し、収束を速めました。

## 手法の系譜と主要論文

- 検出ベースの源流: R-CNN(Girshick et al. CVPR 2014)→ Fast/Faster R-CNN → Mask R-CNN(ICCV 2017, arXiv:1703.06870)。Mask R-CNN は ICCV 2017 の Marr Prize(最優秀論文賞)を受賞。
- 一段階・ボックスベースのマスク: YOLACT(Bolya et al. ICCV 2019)、リアルタイム志向。
- ボックスフリー: SOLO / SOLOv2(ECCV 2020 / NeurIPS 2020)。
- Transformer 統合: DETR(Carion et al. ECCV 2020, arXiv:2005.12872、検出をクエリ集合の予測として定式化)→ MaskFormer(NeurIPS 2021)→ Mask2Former(CVPR 2022, arXiv:2112.01527)。
- 基盤モデル: Segment Anything Model(SAM, Kirillov et al. ICCV 2023, arXiv:2304.02643)。点やボックスのプロンプトで任意物体をマスクする、クラス非依存の汎用セグメンタ(関連: [ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning))。

## 論文の実験結果(定量データ)

Mask R-CNN(ICCV 2017)は COCO の test-dev で、ResNet-101-FPN バックボーンを用いて mask AP がおよそ 35〜37% 程度と報告され、当時の従来手法(COCO 2015/2016 のセグメンテーション優勝手法)を上回りました。RoIAlign の効果のアブレーション(ある要素を抜いて効果を測る実験)では、RoIPool を RoIAlign に置き換えるだけで mask AP がおよそ 数ポイント(報告で約 3 ポイント前後、厳しい高 IoU 領域ではさらに大きく)向上したとされ、座標の丸めをやめることがいかに効いたかを定量的に示しました。

Mask2Former(CVPR 2022)は COCO のインスタンスセグメンテーションで、Swin-L など大型バックボーンを用いて mask AP がおよそ 50% 前後に達したと報告され、専用設計だった従来手法を上回りつつ、同じモデルでパノプティック(PQ 指標で高水準)・セマンティック(mIoU で高水準)も同時に達成しました。「3タスクを1つの設計で、しかも各タスク専用手法を超える」という結果が、統一アーキテクチャの説得力を高めました。

数値の読み方として、mask AP は box AP より一般に低く出ます。理由は、四角い枠を当てるより画素単位の輪郭を正確に当てるほうがはるかに難しいから。また COCO の AP@[.5:.95] は高 IoU(0.9 など輪郭がほぼ完璧)まで要求するため、見た目には良いマスクでも数値は伸びにくい厳しい指標である点を踏まえると、50% 前後という値の重みが分かります。

## メリット・トレードオフ・限界

メリット:
- 物体の正確な形・面積・境界が得られる。重なり・遮蔽(オクルージョン)に強く、距離や占有領域の推定に有用。
- 個体区別ができるので、追跡(トラッキング)や数え上げに直結(関連: [カルマンフィルタによる追跡](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0489_kalman-filter-tracking))。

トレードオフ・限界:
- アノテーションが超高コスト。1物体ずつ画素を塗る作業は、ボックスより桁違いに手間がかかる(関連: [能動学習(Active Learning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0874_active-learning))。
- 計算が重い。二段階法は精度は高いが遅く、車載リアルタイムには工夫が要る。
- 小物体・遮蔽・密集に弱い。遠くの歩行者群など、輪郭が曖昧な領域でマスク精度が落ちる。
- 境界のあいまいさ。正解マスク自体に人間の引き方のブレがあり、評価の上限を縛る。

## 発展トピック・研究の最前線

- プロンプト可能・基盤モデル: SAM のように、クラスを固定せず「指したものを塗る」汎用セグメンタ。ゼロショットで未知物体に対応(関連: [ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning))。
- パノプティック統合と動画: 動画インスタンスセグメンテーション(VIS)では、フレーム間で同一個体を追い続ける必要があり、追跡技術と融合します(関連: [カルマンフィルタによる追跡](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0489_kalman-filter-tracking))。
- 弱教師あり: ボックスや点だけ、あるいは画像レベルのラベルだけからマスクを学び、アノテーションコストを下げる研究。
- 3D・BEV への拡張: 自動運転では LiDAR 点群や鳥瞰図(Bird's-Eye-View)上でのインスタンス分割が重要に。

## さらに学ぶための関連トピック

- [画像分類](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0391_image-classification)
- [カルマンフィルタによる追跡](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0489_kalman-filter-tracking)
- [ゼロショット学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0107_zero-shot-learning)
- [能動学習(Active Learning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0874_active-learning)
- [損失関数の概観](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0593_loss-functions-overview)
