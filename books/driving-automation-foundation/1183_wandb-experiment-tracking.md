---
title: "Weights & Biases"
---

# Weights & Biases

## ひとことで言うと
Weights & Biases（略して W&B または wandb）は、機械学習の実験を「いつ・どんな設定で・どんな結果になったか」を自動で記録し、グラフで見比べられるようにするクラウドサービスです。たとえば「学習率を 0.001 にしたときと 0.01 にしたとき、どちらが損失（loss）を下げたか」を、表とグラフで一目で比較できます。さらに、ハイパーパラメータの探索を自動化する Sweeps、データ・モデルにバージョンを付ける Artifacts などを備え、実験管理の SaaS（インターネット越しに使うソフト）として世界中で広く使われています。

## 直感的な理解
料理人が新メニューを開発する場面を想像してください。「塩を 3g、火加減は中火、煮込み 20 分」で作ったら美味しかった。次は塩を 5g にしてみる。さらに火を強くしてみる。こうした試行を何十回も繰り返すうち、「3 週間前に一番美味しかったあの配合、何だったっけ」が分からなくなります。記録ノートがなければ、せっかくの成功を再現できません。

機械学習はこの試行錯誤を極端にした営みで、人は何十回・何百回と「設定を少し変えて学習し直す」を繰り返します。1 回の学習が数時間〜数日かかり、変えられるつまみ（ハイパーパラメータ）も多いので、頭やスプレッドシートで管理するのはすぐ限界が来ます。W&B は、その都度の「材料（設定）と味（結果）」を自動でノートに取り、後からグラフで並べて比べられるようにする道具です。記録が自動なので「取り忘れ」がなく、再現性（同じ手順なら同じ結果になること）が保てます。

## 基礎: 前提となる概念
記録しておきたい情報を整理します。

- ハイパーパラメータ（hyperparameter）: 人間が学習前に決める設定値。学習率（learning rate、1 ステップでどれだけパラメータを動かすか）、バッチサイズ、エポック数、モデルの層数など。モデルが学習で獲得する「パラメータ（重み）」とは区別される。
- 指標（metric）の推移: 学習中に損失 loss や精度 accuracy がステップごとにどう変化したか（学習曲線）。
- データ・コードのバージョン: どのデータセット、どの git コミットで学習したか。
- チェックポイント（checkpoint）: 学習途中や完了時のモデルの重みを保存したファイル。
- システム情報: GPU 使用率、VRAM 消費、CPU、ネットワークなど。

これらを Excel や手書きで管理するとすぐ破綻します。再現できない実験は、研究では論文の信頼性を損ない、実務では「なぜか性能が良かったモデル」を二度と作れない事態を招きます。再現性の危機（reproducibility crisis）は機械学習コミュニティで繰り返し問題視されてきました（後述の Pineau らの取り組み）。そこで「実験追跡（experiment tracking）ツール」が必要になります。学習スクリプトに数行足すだけで上記を自動記録し、Web で検索・比較・可視化できるようにするのが W&B です。

ここで「run」「project」「sweep」という W&B の単位を押さえます。run は 1 回の学習プロセスを表す最小単位、project は関連する run をまとめる箱、sweep は同じ project 内で多数の run をハイパーパラメータ探索として束ねる単位です。

## 仕組みを詳しく
使い方は学習スクリプトに少しコードを足すだけです。

```python
import wandb

# 1. 実験を開始する（run = 1 回の学習を表す単位）
wandb.init(
    project="my-project",
    config={"backbone": "resnet50", "lr": 0.001, "batch_size": 8},
)

for step in range(1000):
    loss = train_one_step()           # 1 ステップ学習
    # 2. 指標を記録する（ステップごとに loss を送る）
    wandb.log({"loss": loss}, step=step)

# 3. 出来上がったモデルを保存（アーティファクトとして）
wandb.save("model.pt")
```

`wandb.log` が呼ばれるたびに数値が W&B のサーバへ送られ、Web 画面でリアルタイムに折れ線グラフとして描画されます。「loss が右肩下がりに減っているか」をブラウザで眺めながら学習を見守れます。内部的には、学習プロセスとは別のプロセス（wandb のバックエンド）がローカルでログをバッファリングし、非同期にアップロードするため、学習ループの速度への影響は小さく抑えられています。ネットワークが切れても再接続して送り直すリトライ機構を持つので、長時間学習でも記録が欠けにくい設計です。

主な機能を整理します。

- Experiment Tracking: 指標・設定・出力を自動記録。複数 run を 1 枚のグラフに重ねて比較。学習率違いの 5 本の loss 曲線を重ねて最良を視覚判断できる。テーブルビューで run を config 列・metric 列でソート/フィルタでき、parallel coordinates plot（多数のハイパーパラメータと結果の関係を平行座標で俯瞰する図）で「どの設定軸が効いているか」を読み取れる。
- Sweeps: ハイパーパラメータ探索の自動化。「学習率は 1e-4〜1e-2（対数一様）、バッチサイズは 4/8/16」と探索空間を YAML で宣言すると、W&B の sweep server が組み合わせを各 agent（学習を実行するワーカー）に割り振り、多数の run を並列に走らせて最良を見つける。探索戦略はグリッド（全組み合わせ）、ランダム、ベイズ最適化（過去の結果から有望そうな次の点を賢く選ぶ）。途中で見込みの薄い run を早期打ち切りする Hyperband 系の early-termination も指定できる。
- Artifacts: データセットやモデルにバージョンを付け、「この run はデータ v3 を使いモデル v7 を出力した」という系統（リネージ）を有向グラフで追跡。Model Registry と連携し、最良モデルを staging/production に昇格させる運用ができる。
- Reports: グラフと考察をまとめた共有ドキュメントを Web 上で作成（論文の図表のドラフトやチーム共有に使う）。
- Tables: 個々の予測結果（画像・テキスト・音声と予測ラベル）を行単位で記録し、誤分類サンプルを後から目視できる。
- システムメトリクス: GPU 使用率・VRAM・CPU・ネットワーク・温度を自動収集し、ボトルネック診断に使える。

保存先は基本 W&B のクラウド（SaaS、Multi-tenant）ですが、企業の自前環境に置くセルフホスト版（W&B Server、Dedicated Cloud）も提供されています。

## 手法の系譜と主要論文
W&B は商用 SaaS で、特定論文が提案した手法ではありませんが、その機能の背後には明確な研究の系譜があります。

再現性の問題意識:
- Pineau ら（"Improving Reproducibility in Machine Learning Research", NeurIPS の再現性チェックリストの取り組み, 2019-2021）: ML 論文の結果が再現できない問題を指摘し、実験設定・ハイパーパラメータ・乱数シードの明示的記録を強く推奨。NeurIPS の reproducibility checklist として制度化された。W&B のような追跡ツールはこの要請への実務的回答。

ハイパーパラメータ最適化（Sweeps の理論的背景）の系譜:
- Bergstra & Bengio（"Random Search for Hyper-Parameter Optimization", JMLR 2012）: 全組み合わせを試すグリッド探索より、ランダムに点を選ぶランダム探索の方が、同じ計算予算で良い設定を見つけやすいことを示した。重要なハイパーパラメータが少数のとき、グリッドは無駄な次元に予算を浪費するという洞察（「effective dimensionality is low」）。
- Snoek, Larochelle & Adams（"Practical Bayesian Optimization of Machine Learning Algorithms", NeurIPS 2012）: 過去の試行結果から「次に試すと良さそうな設定」をガウス過程（Gaussian Process、未観測点の予測値とその不確かさを与える確率モデル）で予測し、期待改善量（Expected Improvement）が最大の点を次に試すベイズ最適化を ML に適用。Sweeps の bayes モードの源流。
- Li ら（"Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018, arXiv:1603.06560）: 多数の設定を少ない予算で走らせ、見込みの薄いものを早期に打ち切って（successive halving）予算を有望なものに集中させる。Sweeps の early-termination の源流。
- ASHA（Li ら, "A System for Massively Parallel Hyperparameter Tuning", arXiv:1810.05934, 2018/2020）: Hyperband の successive halving を非同期・並列化し、大規模クラスタで遊休ワーカーを作らず使えるようにした発展。多数 GPU で sweep を回す現代的な探索の基盤。
- Population Based Training（Jaderberg ら, arXiv:1711.09846, DeepMind 2017）: 学習途中でハイパーパラメータを進化的に書き換える（成績の良い run の重みと設定をコピーして摂動）動的探索。固定設定では届かない領域に到達できる。

これらが示す共通の効果は「少ない試行回数・計算予算で良い設定にたどり着ける」こと、共通のトレードオフは「探索の計算コスト」と「ベイズ最適化は逐次的で並列化しにくい（次の点が前の結果に依存する）」点です。

## 論文の実験結果（定量データ）
W&B 自体の論文値はないため、その機能が依拠する研究の定量的知見を、指標の意味とともに示します。

- ランダム探索の優位（Bergstra & Bengio 2012）: 複数のデータセット・モデルで、ランダム探索はグリッド探索と同等以上の精度を、はるかに少ない試行回数で達成すると報告。直感的には、9 点のグリッド（3×3）は各ハイパーパラメータについて実質 3 通りしか試さないが、9 点のランダムは各次元で 9 通りの値を試すことになり、「効く 1 つの軸」をより細かく探れる。指標は最終的な検証誤差（validation error、未学習データでの誤り率。低いほど良い）で、同じ予算で誤差を下げられることが要点。
- ベイズ最適化（Snoek ら 2012）: CIFAR-10 の畳み込みネットワークなどで、専門家による手動チューニングや単純なグリッド/ランダムより少ない試行で良い構成に到達し、当時の最良結果（state-of-the-art）を更新したと報告。指標は試行回数あたりの到達精度で、「賢く次を選ぶと探索効率が上がる」ことを定量化。
- Hyperband（Li ら 2018）: 同一の計算予算（時間や epoch 総数）の下で、ランダム探索やベイズ最適化より数倍速く良い設定に到達するケースを多数報告。鍵は「ダメそうな run を早く止め、予算を有望な run に回す」点で、指標は「壁時計時間あたりの到達精度」。
- ASHA（Li ら 2018/2020）: 数百〜千ワーカー規模の並列探索で、同期版 Hyperband に比べストラグラー（遅い 1 試行）に引きずられず、ほぼ線形に近い並列スケールを達成。大規模分散下での実用性を定量化。
- Population Based Training（Jaderberg ら 2017）: 強化学習や機械翻訳・GAN で、固定ハイパーパラメータより高い最終性能を、追加の計算コストをほぼ増やさずに達成。学習中にスケジュールを動的に発見できることを示した。

数値の意味を初心者向けに補うと、検証誤差は「本番で出会う未知のデータにどれだけ間違えるか」の見積もりで、これが小さいほど良いモデルです。探索手法の良し悪しは「同じ GPU 時間でどこまで誤差を下げられたか」で測られ、W&B の Sweeps はこれらの賢い探索（ランダム・ベイズ・Hyperband）を数行で使えるようにした実務ツール、という位置づけになります。

## メリット・トレードオフ・限界
メリット
- 学習スクリプトに数行足すだけで導入でき、前提知識ゼロでも扱いやすい。
- 可視化が洗練されており、多数 run を重ねた比較や parallel coordinates による要因分析が直感的。
- Sweeps によるハイパーパラメータ探索（ランダム/ベイズ/Hyperband 早期打ち切り）が標準で強力。分散 agent で並列探索できる。
- システムメトリクス（GPU/VRAM など）を自動収集し、ボトルネック診断にも使える。
- Artifacts でデータ・モデルの系統（リネージ）を追跡し、Model Registry へつなげられる。

トレードオフ・限界
- 基本がクラウド SaaS のため、設定・指標・場合によりデータが外部サーバへ送られる。機密性の高いデータ（医療、車載センサーなど）では取り扱いに注意が要る（セルフホストや on-prem で緩和）。
- 規模拡大で課金が発生する（無料枠に制限。run 数・ストレージ・seat 課金）。
- セルフホスト版は運用負担が増える（自前で可用性・バックアップ・アップグレードを見る）。
- オープンソースの代替（MLflow など、RDS をメタデータ、S3 をアーティファクトに使い自前完結できる）に比べ、特定ベンダーへの依存（ベンダーロックイン）リスクがある。記録フォーマットの移行性が論点。
- 大量の高頻度 log を送ると、ネットワークやレート制限がボトルネックになりうる（log 頻度の間引きが必要なことがある）。

研究・実務上の論点としては、「実験管理ツールが記録するメタデータをどこまで標準化し、ツール間で移行可能にするか（OpenLineage や MLflow 形式との相互運用）」「Sweeps の探索がモデルやデータの分布シフトに対してどれだけ頑健か」「探索の再現性（ベイズ最適化や ASHA は乱数や並列実行順に結果が左右されうる）」などが挙げられます。

## 発展トピック・研究の最前線
- 多目的最適化（精度とレイテンシ、精度とメモリを同時に最適化する Pareto フロント探索）。組込み・車載のように制約が複数ある領域で重要。
- Population Based Training（PBT）のように、学習途中でハイパーパラメータを進化的に書き換える動的探索の実用化。
- 実験管理とモデルレジストリ・デプロイの連携（最良 run を自動で本番候補に昇格させる MLOps パイプライン）。[Model Registry (Stage 遷移)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1168_model-registry-staging) や [Argo Workflows](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1177_argo-workflows) と組み合わせた自動化。
- LLM 時代の prompt/評価追跡（W&B Weave など、生成結果やプロンプト・チェーンのバージョン管理、LLM 評価の記録へ拡張）。
- 学習曲線予測（少ない epoch の途中経過から最終性能を予測して早期打ち切りを賢くする）と Sweeps への統合。

## さらに学ぶための関連トピック
- [RDS Postgres (Metadata Backend)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1148_rds-postgres-backend)
- [Argo Workflows](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1177_argo-workflows)
- [DVC (Data Version Control)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/0884_dvc-data-version-control)
- [CloudFront + Cognito (UI 公開)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1162_cloudfront-cognito-ui)
- [Model Registry (Stage 遷移)](https://zenn.dev/riita10069/books/driving-automation-foundation/viewer/1168_model-registry-staging)
