---
title: "SOTIF(意図機能の安全性)"
---

# SOTIF(意図機能の安全性)

## ひとことで言うと

SOTIF(Safety Of The Intended Functionality、意図機能の安全性)とは、システムが「壊れていない」のに、性能の限界や予見できる誤用のせいで危険を生むリスクを扱う安全の考え方です。国際規格 ISO 21448 にまとめられています。従来の安全規格(ISO 26262)が「部品の故障」に注目したのに対し、SOTIF は「故障なんてしていないが、センサやAIが想定外の状況をうまく扱えず事故になる」という、自動運転やADAS(先進運転支援システム)特有の新しいリスクに焦点を当てます。

## 直感的な理解

カメラとAIで前方の歩行者を検出して自動ブレーキをかける車を考えます。ハードもソフトも仕様通りに動いている。それでも、強い逆光でカメラが白飛びして歩行者を見落とす、あるいは道路に描かれた人の絵を本物と誤認して急ブレーキする、といったことが起こり得ます。

ここで重要なのは、何も「壊れていない」ことです。回路も焼けていないしソフトもクラッシュしていない。仕様の範囲内で正しく動いているのに、その仕様自体が現実の多様さをカバーしきれていないために危険が出る。これが SOTIF の扱う領域です。AIは学習データにない状況で予測できないふるまいをするため、この問題は機械学習を使うシステムで特に深刻になります。

## 基礎: 前提となる概念

2つの規格の役割分担を押さえます。

- ISO 26262(Functional Safety、機能安全): 電気・電子システムの「故障(malfunction)」に起因するリスクを扱う。たとえばセンサ断線、計算機のビット反転など。「壊れたときに安全側に倒れる」設計を求めます。
- ISO 21448(SOTIF): 「故障していない」状態での、性能限界(performance limitation)と予見可能な誤用(reasonably foreseeable misuse)に起因するリスクを扱う。両者は補完関係で、自動運転の安全には両方が要ります。

SOTIF の中心概念がシナリオの4象限分類です。あらゆる運転状況(シナリオ)を「既知か未知か」「安全か危険か」の2軸で4つに分けます。

```
              安全(safe)        危険(hazardous)
  既知 known   領域1 既知安全      領域2 既知危険
  未知 unknown 領域3 未知安全      領域4 未知危険
```

- 領域1(既知・安全): 想定済みで問題ない状況。理想はここ。
- 領域2(既知・危険): 危険と分かっている状況。対策を設計して領域1へ移す。
- 領域3(未知・安全): 想定していなかったが偶然問題なかった状況。
- 領域4(未知・危険): 想定外でしかも危険な状況。最も恐ろしく、SOTIF活動の目的はこの領域4を見つけ出し、領域2(既知・危険)へ、最終的に領域1(既知・安全)へ移していくことです。

SOTIF の開発活動は要するに「領域2と領域4を可能な限り小さくする」プロセスだと言えます。

## 仕組みを詳しく

SOTIF プロセスの典型的な流れは次の通りです。

1. 機能と意図動作の定義: システムが何をする想定かを明確にする。設計が曖昧だとそもそも限界を議論できません。
2. ハザード特定と評価: 性能限界がどんな危険につながるかを洗い出す。たとえば「霧でLiDARの測距精度が落ちる」「逆光でカメラが歩行者を見落とす」。
3. トリガリング条件(triggering condition)の特定: 危険を引き起こす具体的な入力・環境条件を見つける。逆光・降雨・夜間・珍しい服装の歩行者・路面の反射などがトリガになります。これが SOTIF 分析の核心です。
4. 対策の設計: 性能向上(より良いセンサやモデル)、冗長化(複数センサの融合)、動作制限(性能が出ない状況では速度を落とす・人間に運転を戻す)など。
5. 検証・妥当性確認(V&V): シミュレーション、実走行、シナリオベース試験、統計的評価で残留リスクが十分小さいことを示す。

機械学習システムへの当てはめが SOTIF の難所です。AIの性能限界は明示的なバグではなく、学習データの偏りや分布外入力(OOD: out-of-distribution)として現れます。たとえば学習データに夜間の自転車が少なければ、夜間の自転車検出が静かに劣化する。これは「故障」ではなく「性能限界」なので ISO 26262 では捉えきれず、まさに SOTIF の領域です。

このため SOTIF と機械学習の安全技術は密接につながります。分布外検知や異常検知で「自分がいま苦手な状況にいる」ことを検知し(関連: [異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection))、不確実性推定で予測の信頼度を測り(関連: [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods)、[モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration))、信頼度が低いときは安全側に倒す(減速・停止・人間への移譲)。能動学習でトリガリング条件に当たるレアシーンを優先収集する、といった具合に、各機械学習技術が SOTIF の対策手段として位置づきます(関連: [能動学習(Active Learning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0874_active-learning))。

## 手法の系譜と主要規格

- ISO 26262(2011 初版、2018 第2版): 自動車の機能安全。ASIL(Automotive Safety Integrity Level、A〜D の安全度水準)という概念を導入し、故障のリスクに応じた厳しさを定める。SOTIF の前提となる土台。
- ISO/PAS 21448(2019): SOTIF の最初の公開仕様(PAS: Publicly Available Specification)。当初は運転支援(ADAS、SAEレベル1〜2程度)を主な対象としていました。
- ISO 21448(2022 正式版): 正式な国際規格として発行。より高度な自動運転も視野に拡張されました。
- UL 4600(2020, Underwriters Laboratories): 自動運転車の「安全性の主張(safety case)」を、運転手がいない前提で体系的に組み立てる規格。SOTIF と相補的に参照されます。
- 関連枠組み: シナリオベースの安全評価では PEGASUS プロジェクト(独)や、責任の所在を数式で表す RSS(Responsibility-Sensitive Safety、Shalev-Shwartz et al. 2017, arXiv:1708.06374)などが議論されます。

これらは論文というより規格・標準文書の系譜である点が、純粋な機械学習トピックとの違いです。

## 評価の考え方(定量的側面)

SOTIF には機械学習論文のような単一のベンチマーク数値はありませんが、定量的な議論は不可欠です。中心は「残留リスクが受容可能なほど小さいことをどう数値で示すか」です。

よく引かれる困難として、自動運転の安全を統計的に証明するには膨大な走行距離が必要という試算があります。RAND研究所の報告(Kalra & Paddock, 2016)は、人間ドライバーの死亡事故率(報告でおよそ1億マイルに1件規模)と比べて自動運転が同等以上に安全だと統計的に示すには、車両群でおよそ数億〜数十億マイル規模の走行が必要になり得ると論じました。これは実走行だけでは現実的でないことを意味し、シミュレーションやシナリオベース試験、SOTIF的なトリガリング条件の体系的洗い出しが不可欠である根拠になっています。

性能面の定量評価としては、トリガリング条件ごとの検出率・誤検出率(知覚の指標、関連: [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation))、不確実性の校正の良さ(予測確率と実際の正解率の一致度、関連: [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration))、分布外検知の AUROC(関連: [異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection))などが、SOTIF の主張を支える証拠として使われます。これらの数値が「領域4(未知・危険)をどれだけ潰せたか」を間接的に裏づける形になります。

## メリット・トレードオフ・限界

意義:
- 故障以外のリスクを正面から扱う枠組みを与え、AIを含む現代の運転システムの安全議論の共通言語になっている。
- シナリオの4象限という直感的な整理で、組織横断の議論をしやすくする。

トレードオフ・限界:
- 「未知の未知(領域4)」を完全に列挙することは原理的に不可能。SOTIF はリスクを十分小さくする努力であって、ゼロを保証するものではありません。
- トリガリング条件の網羅性に終わりがない。コストと安全のトレードオフが常につきまといます。
- 機械学習の説明性・検証可能性の限界。AIがなぜその判断をしたか説明しづらく、規格が求める証拠づくりが難しい。
- 「受容可能なリスク」の社会的合意が定まりにくい。どこまで安全なら許されるかは技術だけでは決まりません。

## 発展トピック・研究の最前線

- シナリオベース検証の自動化: シミュレーションで危険シナリオを自動生成し、領域4を効率的に探索する研究。
- 運転設計領域(ODD: Operational Design Domain)の明確化: システムが安全に動作する条件(天候・道路・速度域)を明示し、その外では作動しない設計。SOTIF対策の実務的な要。
- ランタイム監視と安全モニタ: 走行中にモデルの不確実性や分布外度を監視し、危険時にフォールバック動作へ移す(関連: [異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection)、[アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods))。
- 規格の進化: 高度自動運転やAI特化の安全保証(AIセーフティ標準、ISO/TR 4804 → ISO 8800 など)への展開。

## さらに学ぶための関連トピック

- [異常検知](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0734_anomaly-detection)
- [アンサンブル学習](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0175_ensemble-methods)
- [モデルのキャリブレーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0764_model-calibration)
- [能動学習(Active Learning)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0874_active-learning)
- [インスタンスセグメンテーション](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0392_instance-segmentation)
