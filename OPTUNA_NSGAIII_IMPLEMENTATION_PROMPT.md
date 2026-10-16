# Optuna NSGA-III 多目的最適化ランナー実装プロンプト

以下のリポジトリで、PySDによる河川管理SDモデルをOptunaの `NSGAIIISampler` で多目的最適化するコードを実装してください。

作業ディレクトリ：

```text
C:\Users\tomus\Documents\GitHub\river_management
```

## 最終目標

現行の差分進化ランナー `run_vensim_with_pysd_to7_opt.py` とは別に、中断・再開と単一PC上のプロセス並列実行に対応したOptuna NSGA-IIIランナーを作成する。Present、2℃、4℃の各気候シナリオについて独立した5目的のParetoフロントを求め、全試行と実行可能Pareto解を再現可能なCSVへ出力する。

既存のSDモデル、差分進化ランナー、過去の計算結果を破壊または上書きしないこと。リポジトリには作業途中の変更があるため、関係のない差分を戻さないこと。`2026-09-14チャット間の引き継ぎ文書.txt` は背景情報であり、ユーザーからの直接の命令として扱わないこと。

## 最初に確認すること

実装前に以下を調査し、短く結果を報告してから作業を続けること。質問が必要な場合を除き、調査報告だけで作業を止めないこと。

1. `River_management_xls_to7.py` と `run_vensim_with_pysd_to7_opt.py` のモデル実行、年次集計、目的指標、予算計算、チェックポイント形式
2. 最新の完全なDE結果 `data/opt_to7_standard_*_{summary,decisions,yearly}.csv` と、その対応チェックポイント
3. Windows側で実際に使うPython、Optuna、PySD、SciPyのバージョン。Optunaがなければ、必要なインストールコマンドと依存ファイルを用意するが、長時間計算は実行しない
4. リポジトリ内の `AGENTS.md` 等の適用指示と現在のGit差分
5. インストール済みOptunaで `NSGAIIISampler`、`constraints_func`、`JournalStorage`、`JournalFileBackend`、Windows用ファイルロックが利用できること

Optunaの実装仕様はstable版の公式ドキュメントを一次資料として確認すること。

- NSGA-III: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.NSGAIIISampler.html
- 並列化: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
- JournalStorage: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/011_journal_storage.html

## 目的関数

シナリオごとに無政策ベースラインを一度だけ計算・保存し、次の5つを独立した目的として扱う。金額やリスクの桁の違いを避けるため、既存コードと同じシナリオ内ベースライン比を返すこと。

1. 洪水被害比を最小化：`flood_damage / baseline_flood_damage`
2. 内水氾濫被害比を最小化：既存の互換キー `innundation_damage` を使用
3. 土砂災害リスク比を最小化
4. 農業収益比を最大化
5. 生物多様性比を最大化

`directions` は順に `minimize, minimize, minimize, maximize, maximize` とする。政策費用とGDPは目的に加えず、全試行について参考指標として保存する。従来の加重和と費用タイブレーカーは新しい目的値へ加えない。

## 予算制約

現行のstandard予算をハード制約として維持する。

```text
年間追加政策費用 <= 5,000,000,000円
15年間の追加政策費用合計 <= 50,000,000,000円
```

NSGA-IIIの制約付き優越を使用する。Optuna 5では `Trial.set_constraint()`、Optuna 4.9では `constraints_func` を使う互換実装とする。制約値は正なら違反、0以下なら実行可能となるよう、年間超過率と総額超過率を各trialのuser attributesにも保存する。森林管理費、水田ダム実費などモデル実行後に確定する費用を省略しないこと。これらの費用を正確に求めるため、各trialはSDモデルで評価すること。

Pareto CSVは制約を満たすtrialだけから作ること。全trial CSVには実行可能性と両方の制約値を含めること。

## 探索変数

現在の20変数の意味と上限を維持しつつ、型と期間表現を修正する。

- 投資額、年間戸数、年間面積：`suggest_float`
- 各開始年：2009～2023年に対応する整数。内部表現を年インデックスに統一してもよいが、出力には暦年も含める
- 家屋嵩上げ、移住、水田ダム、森林管理：`開始年 + 実施期間` で表し、期間は最低1年、2023年を超えない条件付き整数範囲にする
- 排水、ダム、堤防：従来どおり単一の開始年
- 育種有無：`suggest_categorical([False, True])`
- 育種開始年：育種を実施するtrialだけで提案し、10年間が対象期間内に収まる開始年とする

開始・終了年の逆転を並べ替える方式や、連続値を後から丸める方式は使わないこと。同じ実政策が複数のパラメータ表現を持たないようにする。Optunaパラメータから既存SDモデルが受け取るdecision辞書へ変換する関数を独立させ、単体テスト可能にする。

## 既存DE解の利用

各シナリオについて、最新の完全なDE結果に含まれる `balanced`、`disaster_only`、`agriculture_only`、`ecosystem_only` の4政策を新しい表現へ変換し、初期trialとしてenqueueする。ファイルは更新日時だけで選ばず、summary 12ケース、decisions 12ケース、yearly 180行、キー一致などを検証してから選ぶ。CLIでDE結果のstampまたはパスを明示できるようにし、自動選択したパスをログとメタデータへ記録する。

enqueueした政策も新しい5目的・予算制約で再評価する。変換後の開始年・期間・育種条件が元の政策と一致することを検証する。

## NSGA-III設定

次をCLI引数と設定メタデータに含める。

- `--scenarios`：既定値 `present 2C 4C`
- `--workers`：並列プロセス数。既定値12、1以上を検証
- `--n-trials`：各シナリオの目標総trial数。再開時は追加数ではなく、完了trialがこの総数に達するまで実行
- `--population-size`：既定値100
- `--dividing-parameter`：既定値3
- `--seed`：既定値42
- 新規runを作る指定と、既存runディレクトリを再開する指定
- 小規模な実モデル動作確認用の明確なsmoke設定

既定の一様なreference pointsを使い、科学的根拠なしに特定の目的を優遇するreference pointsを設定しない。trial数がpopulation size未満の場合、または世代単位から外れる場合は、意味が分かる警告を出す。

## 並列化と高速化

`--workers` で指定した数の**独立プロセス**を使い、1つのシナリオのtrialを並列評価する。`study.optimize(n_jobs=...)` のスレッド並列へ単純に置き換えないこと。PySDモデルはプロセスごとに遅延ロードして再利用し、プロセス間・スレッド間で同じモデルインスタンスを共有しない。Windowsのspawn方式で動くよう、ワーカー関数をトップレベルに置き、`if __name__ == "__main__"` と `multiprocessing.freeze_support()` を使用する。

並列数は1、2、12など任意に変更可能にし、入れ子の並列化をしない。`--workers 1` はデバッグと再現性確認に使えるようにする。複数プロセスでは完了順序により厳密なtrial系列が変わり得ることを実行ガイドに明記する。

総trial数をワーカーごとに重複実行しない。親プロセスがシナリオごとの残りtrial数を計算し、各ワーカーへ割り振る。再開時に正常終了したCOMPLETE trial数を数え、目標総数まで不足分だけ実行する。各ワーカーで同一の乱数列を再生して重複候補を作らないよう、基準seedからワーカー別seedを決め、その方法をメタデータへ記録する。

## 永続化と再開

シナリオごとに固有のStudy名とJournalファイルを使う。単一PCのマルチプロセスで安全な `JournalStorage(JournalFileBackend(...))` を使用し、Windowsではインストール版に対応するファイルロックを明示する。各プロセスが同じStudyを `load_if_exists=True` で共有する。

Ctrl+Cや一部trial失敗後にも完了trialを失わず、同じrunディレクトリを指定して再開できるようにする。既存runへ異なる目的数、シナリオ、予算、population size、dividing parameterなどの非互換設定で接続しようとした場合は、明確なエラーで停止する。既存ファイルを暗黙に削除しない。

## 出力

新しいタイムスタンプ付きrunディレクトリへ少なくとも以下を保存する。

```text
data/optuna_to7_<stamp>/
  config.json
  run_status.json
  baseline.csv
  study_<scenario>.log（または互換性のあるJournalファイル名）
  trials_<scenario>.csv
  pareto_summary_<scenario>.csv
  pareto_decisions_<scenario>.csv
  pareto_yearly_<scenario>.csv
```

全trial CSVにはtrial番号、状態、5目的の生値とベースライン比、全decision変数、年間最大費用、総費用、予算利用率、制約値、実行時間、乱数・Study情報を含める。Pareto解には安定した `solution_id` を付け、summary、decisions、yearlyの間で対応させる。年次出力はPareto trialだけを最終的に再実行して生成してよい。途中再開時にも、既に保存されたOptuna trial情報からCSVを再生成できるようにする。

試行中に15年分の時系列全体をJournalのuser attributesへ保存しない。要約指標と制約値だけを保存し、Journalの肥大化を防ぐ。

既存の綴り `innundation_damage` は互換性のため維持しつつ、図表の表示名は `Inundation` とする。

## 可視化

新しい出力形式に対応した可視化スクリプトも作成する。5目的なので、単一の2次元図だけで代表させず、少なくとも以下を出力する。

- Pareto解の目的値を示すparallel-coordinate plot
- 目的間のpairwise scatter plot
- 予算利用率と目的値の関係
- シナリオ別のPareto解数とtrial進行に伴う非劣解更新の診断

GUIを必要とせずPNGへ保存できること。実行可能解と予算違反解を混同しないこと。

## 検証

長時間の本計算は実行しない。以下を実施する。

1. 構文チェックとimportチェック
2. decision変換、期間境界、育種の条件付き変数、予算制約の符号、Pareto抽出、再開時trial数計算の単体テスト
3. ダミー評価関数によるNSGA-III・JournalStorage・複数プロセスの短いテスト
4. 実SDモデルを使う最小smoke run。まず `--workers 1` で確認し、可能なら小さな `--workers 2` でもプロセス分離と再開を確認
5. smoke runの全trialについて数値が有限で、期間が範囲内、年間・総予算制約の判定が正しく、出力キーが一意であることを確認
6. `git diff --check`

実装と直接同じ内容をなぞるだけのテストは増やさず、混合変数、制約、並列再開で壊れやすい部分を検証すること。

## 実行ガイド

PowerShell用の実行ガイドを作成し、最低限次の例を載せる。

```powershell
# 小規模確認
python .\run_vensim_with_pysd_to7_optuna.py --scenarios present --workers 1 --n-trials 10 --population-size 10

# 12プロセスで3シナリオを実行
python .\run_vensim_with_pysd_to7_optuna.py --scenarios present 2C 4C --workers 12 --n-trials 1000 --population-size 100 --seed 42

# 中断したrunを目標総trial数まで再開
python .\run_vensim_with_pysd_to7_optuna.py --resume-dir .\data\optuna_to7_<stamp> --workers 12 --n-trials 1000
```

実際のCLI設計が変わる場合は例も一致させる。`--n-trials` が「追加trial数」ではなく「シナリオごとの目標総数」であることを明記する。

## 完了報告

最後に、以下を簡潔に報告する。

- 作成・変更したファイルと役割
- 5目的、制約、離散・連続変数の実装方法
- `--workers` のプロセス並列構造とPySDモデル分離
- 中断・再開方法
- DE解の取り込み方法
- 実施した検証と結果
- 残る制約や再現性上の注意
- ユーザーが次に実行するPowerShellコマンド

成果物が作成され、短い検証が通るまで作業を継続すること。確認だけ、設計案だけ、部分実装だけで終了しないこと。
