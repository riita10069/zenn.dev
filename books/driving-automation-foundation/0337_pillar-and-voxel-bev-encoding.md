---
title: "Pillar/Voxel BEVエンコード"
---

# Pillar/Voxel BEVエンコード

## ひとことで言うと
Pillar/Voxel BEVエンコードは、LiDARが出す「順番もまばらさもバラバラな3次元の点の集まり(点群)」を、立方体のマス目(voxel)や地面から垂直に伸びた柱(pillar)に区切って整え、最終的に上から見下ろした地図(BEV)状の特徴に変換する、LiDAR側の基盤となる前処理・特徴化の手法です。これにより、画像用に作られた畳み込みニューラルネットワークがそのまま点群に使えるようになります。多くの3D検出器やカメラ-LiDAR融合手法のLiDAR枝は、この変換から始まります。

## 直感的な理解
点群は、空中にばらまかれた無数の小さな砂粒のようなものです。砂粒には決まった並び順がなく、物体の表面にだけ集まり、近くは密で遠くはまばらです。一方、画像処理で使うニューラルネットワークは「方眼紙のように規則正しく並んだマス目(画素)」を前提に作られています。砂粒の集まりをそのまま方眼紙用の道具で扱うことはできません。

そこで、空間に方眼(格子)をかぶせ、各マスに入った砂粒をひとまとめにして「このマスにはこういう特徴の砂が入っている」という1つの代表値に置き換えます。すると砂粒の集まりが方眼紙の上のデータに変わり、画像と同じ道具で処理できるようになります。これがvoxel化・pillar化の本質です。立方体のマスで区切るのがvoxel、地面のマスだけで区切って高さ方向は1本の柱にまとめるのがpillarです。voxelは高さ情報を残す代わりに重く、pillarは高さを潰す代わりに軽い、というのが核心的なトレードオフです。

## 基礎: 前提となる概念
点群の扱いにくさを3点で押さえます。

順序がない(unordered/permutation invariance)。点に決まった並び順がなく、画素のように「左上から右下へ」と並んでいません。同じ物体でも点を並べる順番は無数にありえます。畳み込みは「隣の画素」という空間的な隣接関係を前提にするため、順序のない点には直接使えません。点群を扱う関数は、点の並べ替えに対して出力が変わらない(順序不変)必要があります。

まばら(sparse)。空間の大半は点がなく、物体の表面にだけ点が集まります。±50m四方を0.1m刻みのvoxelにすると数千万マスになりますが、点が入るのはその数%程度です。

密度が不均一。LiDARの近くは点が密で、遠くは疎です。同じ物体でも距離によって点の数が大きく変わります(数m先の車に数千点、50m先の車に数十点)。

これらの性質のため、点群を「規則的な格子表現」に変換する必要があります。格子にすれば後段で普通の2D/3D畳み込みが使え、最終的にBEV特徴([BEVグリッドとpc_range設計](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0927_bev-grid-and-pc-range) と同じ鳥瞰の格子)にできます。BEVにすると、カメラ側のBEVと同じ土俵で融合でき([BEVFusion (カメラ+LiDAR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0325_bevfusion) や [TransFusion](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0345_transfusion) の前提)、地面に平行な物体配置を自然に扱えます。

PointNet(Qi et al., CVPR 2017, arXiv:1612.00593)。順序のない点集合を直接扱う先駆的なネットワークです。各点を独立に多層パーセプトロン(MLP、全結合層を重ねた小さなネットワーク)で処理し、最後にmax pooling(各特徴次元で全点の最大値を取る集約)で1つのベクトルにまとめます。max poolingは点の順序に依存しない(対称関数)ため、順序のない点集合を扱えます。pillar方式やvoxel方式の点エンコーダは、このPointNetの簡易版を各柱・各voxelの中で使います。

## 仕組みを詳しく
2つの代表方式を分けて説明します。

### Voxel方式(VoxelNet / SECOND)
第1に、空間を3次元の小さな立方体(voxel、ボクセル)に区切ります。例えば前後左右±50m・上下数mの空間を0.1m刻みの立方体に分割すると、格子はおよそ1000×1000×40のサイズになります。

第2に、各voxelに入った点をまとめ、その中の点集合から特徴ベクトルを学習します。VoxelNetではこれをVFE(Voxel Feature Encoding)層と呼び、PointNet風に各点をMLPで処理してvoxel内でmax poolingし集約します。各点は座標(x,y,z,反射強度r)に加え、voxel内の点の重心からのずれなどを付けた拡張特徴で表します。点が入っていないvoxelは空のままです。点が多すぎるvoxelは最大T点にサンプリングします。

第3に、これで「3次元格子に1ベクトルずつ」の規則的な表現になるので、3D畳み込みで特徴を抽出します。

第4に、高さ(z)方向を畳み込みやreshapeでつぶして、BEV特徴 [C, BEV_H, BEV_W] にし、2D検出ヘッドへ渡します。

VoxelNetの課題は3D畳み込みが重いことです。空のvoxelにも計算が走る素朴な3D convは遅く、メモリも食います。SECOND(Yan et al., 2018)は「ほとんどのvoxelが空」という性質を使い、点のあるvoxelだけ計算するsparse convolution(疎な畳み込み、submanifold sparse conv)を導入して大幅に高速化しました。空のvoxelを計算しないため、メモリと計算が劇的に減ります。今日のLiDAR検出器(CenterPointなど)のバックボーンはほぼこのsparse conv系です。

### Pillar方式(PointPillars)
第1に、高さ方向(z)には区切らず、地面のxy平面だけを格子に区切ります。各セルは地面から空まで伸びた柱(pillar)になります。例えばxyを0.16m刻みにすると、格子はおよそ432×496程度です。

第2に、各pillarに入った点を最大N点(例えば100点)拾い、各点を拡張した特徴にします。具体的には、点の座標(x,y,z,反射強度)に加え、pillar内の点の算術平均からの差(xc,yc,zc)、pillar幾何中心からの差(xp,yp)などを連結して、各点を例えば9次元程度のベクトルにします。これにより各点が「自分が柱の中でどこにいるか」の文脈を持ちます。空のpillarや点不足のpillarはゼロ埋めし、計算は点のあるpillarだけに集約します。

第3に、簡易なPointNet(各点を独立にMLPで処理し、pillar内でmax poolingして1ベクトルにまとめる)でpillarごとの特徴を得ます。

第4に、pillarをxy格子に並べ直すと「擬似画像(pseudo-image)」になります。shape [C, H, W] = [64, 496, 432] のような2次元特徴マップです。点群が画像と同じ形に化けるのがポイントです。

第5に、あとは普通の2D CNN(画像用と同じバックボーン+FPN)で処理してBEV特徴を出し、検出ヘッドへ渡します。

PointPillarsの最大の利点は、3D畳み込みを完全に避け2D畳み込みだけで済むため非常に速いことです(論文では当時60FPS超、最速設定で100FPS超を報告)。代わりに高さ方向を1柱にまとめてしまうため、上下に重なる物体や高さ方向の細かい構造の表現力はvoxelより落ちます。

数値例での対比を示します。

```
voxel方式:  点群 → (X, Y, Z, C) の4次元テンソル → 3D sparse conv → 高さをつぶす → BEV [C,H,W]
            高さ情報を保てるが3D convが要る（sparse化で軽量化）

pillar方式: 点群 → 最初に高さをつぶす → 擬似画像 (C, H, W) → 2D conv → BEV [C,H,W]
            軽くて速いが高さ方向の表現力が落ちる
```

格子解像度の選び方が精度と速度を直接左右します。刻みを細かくすると小物体を取りこぼしにくくなりますが、格子数が増えてメモリと計算が膨らみます。逆に粗くすると速いが、歩行者のような小さく薄い対象を1セルに丸め込んで見落としやすくなります。

## 手法の系譜と主要論文
点群検出は、手作り特徴から学習特徴へ、そして重い3D畳み込みから軽い2D畳み込みへと進化しました。

PointNet / PointNet++(Qi et al., CVPR 2017 / NeurIPS 2017)。順序のない点集合を直接扱う考え方を確立しました。voxel/pillar内の点エンコーダの理論的な土台です。

VoxelNet(Zhou & Tuzel, CVPR 2018, arXiv:1711.06396)。「VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection」。提案はVFE層でvoxelごとに点集合を学習的に特徴化し、3D convで検出するエンドツーエンドのパイプラインです。動機は、それまで手作りだった点群特徴(占有率や密度などの統計量)を学習に置き換えることでした。効果は精度向上ですが、3D convが重く遅いのが難点でした。

SECOND(Yan, Mao, Li, Sensors 2018)。「Sparsely Embedded Convolutional Detection」。提案はsparse convolutionで空voxelを飛ばすことと、向き推定を安定させるロス(sine-error loss)です。動機は、VoxelNetの3D convの大半が空voxelの無駄計算だったことです。効果は大幅な高速化と省メモリで、トレードオフはsparse conv実装が複雑なことです。

PointPillars(Lang, Vora, Caesar, et al., CVPR 2019, arXiv:1812.05784)。「PointPillars: Fast Encoders for Object Detection from Point Clouds」。提案は高さをつぶしてpillarで擬似画像化し、2D convのみで処理することです。動機は、3D convを避けて推論を高速・省メモリにすること(自動運転の実時間要求)でした。効果は当時最速級で、トレードオフは高さ方向の表現力が落ち、薄い物体や立体的な構造で不利になりうることです。

これらはカメラ-LiDAR融合([BEVFusion (カメラ+LiDAR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0325_bevfusion), [TransFusion](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0345_transfusion))のLiDAR枝として今も標準的に使われ、共通BEV表現の出発点になっています。系譜の根にはPointNet(Qi et al., CVPR 2017)があり、順序のない点集合を扱う考え方を提供しました。後段の検出ヘッドとしては、アンカー不要で物体中心を直接予測するCenterPoint(Yin et al., CVPR 2021)が事実上の標準になり、voxel/pillarバックボーンと組み合わされます。

## 論文の実験結果(定量データ)
評価は主にKITTI(Geiger et al., CVPR 2012、自動運転検出の古典的ベンチマーク)とnuScenes、Waymoで行われます。指標を噛み砕きます。
- AP(Average Precision、平均適合率)。検出の正確さ。KITTIではEasy/Moderate/Hardの難易度別(物体の大きさ・遮蔽・切れ具合で区分)に、3D APとBEV APを報告します。高いほど良い。
- FPS(Frames Per Second、秒間処理フレーム数)。推論速度。高いほど速く、自動運転の実時間要求(LiDARは通常10Hz回転なので10FPS以上)に直結します。

VoxelNetはKITTIで、当時の手作り特徴ベースの手法を上回る3D AP(車クラスのModerateでおよそ65%前後)を達成し、学習特徴の有効性を示しました。ただし推論は数FPS程度(およそ4FPS)と遅く、実時間には届きませんでした。

SECONDはsparse convolutionにより、VoxelNetと同等以上の精度(車Moderate 3D APでおよそ76%前後)を保ちながら20FPS以上へ高速化しました。空voxelを計算しない効果が、精度を落とさず速度だけを大きく改善することを定量的に示した点が重要です。これにより3D convが実用圏に入りました。

PointPillarsはKITTI車クラスでModerate 3D APおよそ74〜77%前後を報告しつつ、推論速度は60FPS超(論文では最速設定で100FPS超)と、当時の検出器の中で群を抜く速さでした。これは「2D convだけで済む」設計の効果を端的に示します。アブレーションでは、各点に付加した文脈特徴(pillar中心や重心からの相対位置など)を外すと精度が落ち、これらの幾何的な手がかりが効いていることが確認されています。pillar解像度を粗くすると速度は上がるが小物体(歩行者・自転車)のAPが顕著に下がる、という解像度トレードオフも報告されています。

精度と速度のトレードオフをまとめると、voxel+3D sparse conv系は高さ情報を保てるぶん精度面で有利になりやすく(特に向き推定や薄い物体)、pillar系は速度で圧倒的に有利です。どちらを選ぶかは、車載の計算予算と必要な精度のバランスで決まります。融合手法のLiDAR枝としては、速度重視ならpillar、精度重視ならsparse conv系のvoxelが選ばれる傾向があります。nuScenes/Waymoのような大規模・多クラスでは精度要求が高く、sparse voxel系が主流です。

## メリット・トレードオフ・限界
メリット。順序のないまばらな点群を、CNNが扱える規則的な格子(最終的にBEV)に変換できます。BEV化によりカメラBEVと同座標で融合できます。pillarは2D convのみで非常に高速、voxelは高さ情報を保持でき、sparse convで空領域を省けます。どちらも学習で特徴を獲得するため、手作り特徴より表現力が高いです。

トレードオフと限界。格子の刻み(解像度)を細かくすると精度は上がりますが、メモリと計算が増えます。逆に粗くすると小物体を取りこぼします。pillarは高さ方向の表現力が低く(上下に重なる物体や薄い障害物に弱い)、voxel+3D convは軽量化しても点ベース手法より細部の幾何が粗くなりがちです。そして格子化の際に量子化誤差(点を格子に丸める誤差)が必然的に生じます。同じ格子内の複数点の細かな違いは、1つの代表ベクトルに集約された時点で失われます。遠方の疎な点群では1セルに数点しか入らず、特徴が不安定になります。

未解決の研究課題としては、量子化誤差を避けつつ高速に保つ点ベース・voxelベースのハイブリッド設計、遠方の疎な点群での精度向上、悪天候(雨・雪・霧による偽点やノイズ)への頑健化、そして格子解像度を入力に応じて動的に変える適応的な離散化が挙げられます。

## 発展トピック・研究の最前線
pillar/voxel離散化は今も使われ続けていますが、研究は次の方向に進んでいます。

第1に、点ベースと格子ベースのハイブリッドです。PV-RCNN(Shi et al., CVPR 2020)のように、voxelの効率(粗い特徴を高速抽出)と点の細かさ(keypointで精密に集約)を組み合わせ、量子化誤差を抑えつつ速度を保つ設計が広く使われています。第2に、Transformerの導入です。voxelやpillarの特徴にアテンションをかけ、遠く離れた領域の関係も捉えるVoxel Transformer系や、点を直接トークン化するSST/FlatFormerのような設計が研究されています。第3に、時間融合です。複数フレームの点群を自車運動で整列して重ね、速度や遮蔽の手がかりを得る方向で、これは [BEVFusion (カメラ+LiDAR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0325_bevfusion) や [TransFusion](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0345_transfusion) の時間拡張とも合流します。第4に、レンジビュー(LiDARの生の距離画像)やBEVとの併用など、表現の使い分け・統合も活発です。

いずれの方向でも、pillar/voxelエンコードは「点群をネットワークが扱える形に整える」最初の一歩として基盤的な役割を保ち続けており、マルチモーダル融合の共通BEV表現の出発点として欠かせない要素です。

## さらに学ぶための関連トピック
- [BEVFusion (カメラ+LiDAR)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0325_bevfusion)
- [TransFusion](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0345_transfusion)
- [BEVグリッドとpc_range設計](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0927_bev-grid-and-pc-range)
- [IPM (逆透視投影)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0281_ipm-inverse-perspective-mapping)
- [Sparse4D](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0295_sparse4d)
