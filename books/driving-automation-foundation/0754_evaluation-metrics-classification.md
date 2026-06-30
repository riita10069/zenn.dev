---
title: "分類の評価指標"
---

# 分類の評価指標

## ひとことで言うと
分類の評価指標とは、「あるカテゴリかどうかを当てるモデル」が、どれくらい正しく当てられているかを数値で測るものさしです。accuracy(正解率)、precision(適合率)、recall(再現率)、F1、AUC などが代表で、それぞれ「どの種類の間違いをどれくらい重く見るか」が違います。

## 直感的な理解
たとえば「この画像に歩行者がいるか?」を判定するモデルを考えます。判定結果と実際の正解の組み合わせは4通りしかありません。これを混同行列(confusion matrix、予測と正解を縦横に並べた表)で整理します。

```
                  実際: いる(陽性)   実際: いない(陰性)
予測: いる(陽性)    TP (正しく検出)      FP (誤検出/空振り)
予測: いない(陰性)   FN (見逃し)         TN (正しく無し)
```

- TP(True Positive、真陽性): いるものを「いる」と当てた。
- FP(False Positive、偽陽性): いないのに「いる」と言った(空振り、誤報)。
- FN(False Negative、偽陰性): いるのに「いない」と見逃した。
- TN(True Negative、真陰性): いないものを「いない」と当てた。

自動運転では、歩行者の見逃し(FN)は事故に直結する最悪の間違いで、誤報(FP)は急ブレーキで乗り心地が悪くなる程度の間違いです。つまり間違いには軽重があるので、単一の数字「正解率」だけでは不十分なのです。各指標はこの4つの数の組み合わせから作ります。

## 基礎: 前提となる概念

二値分類(binary classification): 陽性(positive)か陰性(negative)かの2クラスを当てる問題。多クラスはこれを拡張します。

しきい値(threshold): 多くのモデルは「歩行者である確率0.0〜1.0」を出します。例えば「0.5以上なら陽性とみなす」という境界がしきい値です。しきい値を上げると陽性判定が慎重になり、FPが減る代わりにFNが増えます。指標の多くはこのしきい値に依存します。

クラス不均衡(class imbalance): 陽性と陰性の数が大きく偏っている状況。例えば全画像のうち歩行者ありが1%しかない場合、「常に歪んでいないと答える」だけで正解率99%になります。これが正解率の落とし穴です。

## 仕組みを詳しく

各指標を定義し、記号の意味を言葉で説明します。

正解率(Accuracy) = (TP + TN) / (TP + TN + FP + FN)
分子は正しく当てた総数、分母は全件数。全体の何割を当てたか。クラス不均衡に弱い。

適合率(Precision) = TP / (TP + FP)
「陽性と予測したもののうち、本当に陽性だった割合」。分母は陽性と予測した件数。これが高いと「誤報が少ない」。空振りのコストが高い場面(誤報で急ブレーキしたくない)で重視。

再現率(Recall、感度 Sensitivity とも) = TP / (TP + FN)
「実際に陽性のもののうち、正しく拾えた割合」。分母は実際に陽性の件数。これが高いと「見逃しが少ない」。見逃しが致命的な場面(歩行者・病気の検出)で重視。

precision と recall はふつう綱引き(トレードオフ)の関係です。しきい値を下げて何でも陽性と言えば recall は上がりますが precision は下がります。

F1スコア(F1 Score) = 2 × (Precision × Recall) / (Precision + Recall)
precision と recall の調和平均(逆数の平均の逆数)。両方が高くないと高くなりません。一方だけ高くてもダメ、というバランス指標。より一般のFβは β でrecall重視/precision重視を調整します(β>1でrecall重視)。

特異度(Specificity) = TN / (TN + FP)
実際に陰性のもののうち正しく陰性と言えた割合。

ROC曲線とAUC。ROC(Receiver Operating Characteristic)は、しきい値を0から1まで動かしながら、横軸に偽陽性率 FPR = FP/(FP+TN)、縦軸に真陽性率 TPR = recall を取って描く曲線です。左上の角(FPR=0, TPR=1)が完璧。AUC(Area Under the Curve、曲線の下の面積)はこの曲線の下の面積で、0.5(でたらめ)〜1.0(完璧)を取ります。AUCの直感的な意味は「ランダムに選んだ陽性1件と陰性1件を比べたとき、陽性の方に高いスコアを付ける確率」です。しきい値に依存しない総合力を一つの数で表せるのが利点です。

PR曲線(Precision-Recall曲線): 横軸recall、縦軸precisionの曲線。その下の面積をAP(Average Precision、平均適合率)と呼びます。クラス不均衡が激しい時はROC-AUCより実態を反映しやすいとされます(陽性が極端に少ないと、ROCのFPRは大きなTNに薄められて楽観的になるため)。物体検出のmAP(mean Average Precision)はこのAPをクラスとIoUしきい値で平均したもので、検出ベンチマークの標準です。

多クラスへの拡張: クラスごとにprecision/recallを計算し、平均します。マクロ平均(macro average)は各クラスを同じ重みで平均、マイクロ平均(micro average)は全件をまとめて計算(多数クラスの影響が大きい)、加重平均(weighted)はクラスの件数で重み付けします。

## 手法の系譜と主要論文

これらの指標の起源は情報検索と統計に遡ります。precision/recall は情報検索分野(Cleverdon, 1960年代)で文書検索の評価に使われ始めました。ROC分析は第二次大戦のレーダー信号検出理論(信号検出理論、Signal Detection Theory)が源流で、後に医学診断・機械学習へ広がりました。

機械学習での体系的整理として、Fawcett (2006, "An introduction to ROC analysis", Pattern Recognition Letters) はROC/AUCの実務的な解説の定番です。

Powers (2011, "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness and Correlation", Journal of Machine Learning Technologies) は、precision/recall/F値が陰性側(TN)を無視するなどの偏りを批判し、informedness(Youden's J = TPR + TNR − 1)やmarkedness、Matthews相関係数 MCC といった、混同行列の4要素を全て使うバランスの良い指標を整理しました。

MCC(Matthews Correlation Coefficient)は近年、クラス不均衡下で最も信頼できる単一指標としてしばしば推奨されます(Chicco & Jurman, 2020, BMC Genomics)。これは予測と正解の相関係数で −1〜+1 を取り、4要素すべてが良くないと高くなりません。

実装面では scikit-learn の sklearn.metrics(accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score など)が事実上の標準で、研究・実務で広く使われています。

## 論文の実験結果(定量データ)

指標そのものは「ものさし」なので固有のベンチマーク数値は持ちませんが、なぜ指標選択が重要かを示す具体例を挙げます。

不均衡の落とし穴(具体数値): 陽性1%、陰性99%のデータで「全件を陰性と予測」するダミー分類器を考えます。正解率 = 99%(高く見える)。しかし recall = 0/(0+陽性件数) = 0%、precision は定義できない(陽性予測ゼロ)、F1 = 0、ROC-AUC = 0.5(でたらめ)。正解率だけ見ると優秀に見え、F1とAUCを見ると無能だと分かります。これが「正解率だけ報告してはいけない」理由の定量的根拠です。

Chicco & Jurman (2020) は、複数のシミュレーション・実データで、F1やaccuracyが高くてもMCCが低い(つまり実際は性能が悪い)ケースを示し、MCCの方が混同行列の全情報を反映するため信頼できると結論づけました。彼らはMCCが高いときのみ、4つのセル(TP/TN/FP/FN)がすべて良好であることを定量的に示しています。

物体検出の文脈では、COCOデータセットの標準指標がmAP(IoUしきい値0.5から0.95を0.05刻みで平均したAP)です。例えば近年の検出モデルがCOCO mAP 50前後を達成、といった数値が比較に使われ、単一のmAP値で多数モデルを順位付けします。AP/mAPがしきい値に依存しない総合指標であることが、こうした横並び比較を可能にしています。

## メリット・トレードオフ・限界
- accuracy: 直感的で計算も簡単。だがクラス不均衡に致命的に弱い。
- precision/recall: 間違いの種類を区別できる。だが片方だけ報告すると恣意的(しきい値で操作可能)。必ずペアで、できればしきい値も併記すべき。
- F1: バランスを一つの数で見られる。だがprecisionとrecallを同等に扱うため、見逃しと誤報のコストが違う応用(運転の歩行者検出など)では不適切なことがある。Fβで重み調整が必要。
- AUC: しきい値非依存で総合力を表す。だが極端な不均衡では楽観的になりがちで、その場合はPR-AUC/APを併用。また「実運用で使う特定のしきい値での性能」は表さない。
- 一般的な限界: どの指標も「予測と正解ラベルの一致」しか測らず、誤りのコスト(歩行者見逃し=死亡事故)を自動では反映しません。安全クリティカル分野では、指標値だけでなく失敗事例の質的分析が必須です。

## 発展トピック・研究の最前線
- 較正(calibration): モデルが出す確率0.8が「本当に80%の頻度で当たる」かを測るECE(Expected Calibration Error)。意思決定には予測値だけでなく確率の信頼性が重要。
- 不均衡対策の指標: balanced accuracy(クラスごとrecallの平均)、MCC、PR-AUC。
- 安全クリティカル評価: コスト重み付き指標、ロングテール(希少だが重大なケース)に着目した評価。自動運転では検出の指標が運転性能に直結しない問題が指摘される(エンドツーエンドの項参照)。
- 多クラス・多ラベルでの集約: macro/micro/weightedの使い分けと報告の透明性。

## さらに学ぶための関連トピック
- [エンドツーエンド運転](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1065_end-to-end-driving)
- [敵対的生成ネットワーク(GAN)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0237_generative-adversarial-networks)
- [Frenet座標系による軌道計画](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0176_frenet-frame-planning)
