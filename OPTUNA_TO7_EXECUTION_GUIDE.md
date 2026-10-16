# to7 Optuna NSGA-III 実行ガイド

## 概要

`run_vensim_with_pysd_to7_optuna.py` は、河川管理SDモデルを次の5目的で最適化します。

1. 洪水被害比の最小化
2. 内水氾濫被害比の最小化
3. 土砂災害リスク比の最小化
4. 農業収益比の最大化
5. 生物多様性比の最大化

Present、2℃、4℃は別々のOptuna Studyとして扱います。年間追加政策費用50億円、15年間の追加政策費用500億円を制約とし、政策費用自体は目的関数に含めません。

Optuna 5では名前付き制約API、Optuna 4.9では互換コールバックを使用します。どちらも正の制約値を予算違反としてNSGA-IIIの選択へ反映します。

`NSGAIIISampler` はOptuna上でexperimental APIです。使用したOptunaの正確なバージョンは各runの `config.json` に保存されます。

年は整数、複数年政策は開始年と実施期間、育種は実施有無と条件付き開始年として探索します。既存のDE最良政策4件を各シナリオの初期trialに使用します。

## 環境確認

プロジェクトで通常使用しているWindows Python環境を有効にして、PowerShellで確認します。

```powershell
python --version
python -c "import pysd, numpy, pandas, scipy, matplotlib; print('model dependencies: OK')"
python -c "import optuna; print('Optuna', optuna.__version__)"
```

Optunaがない場合は追加します。

```powershell
python -m pip install -r .\requirements-optuna.txt
```

## 小規模確認

`--smoke` はPresentだけを10 trial、population size 10で実行します。長時間計算前の動作確認用です。

```powershell
python .\run_vensim_with_pysd_to7_optuna.py --smoke --workers 1
```

プロセス並列も確認する場合は次を使います。

```powershell
python .\run_vensim_with_pysd_to7_optuna.py --smoke --workers 2
```

`--smoke` では指定にかかわらず、シナリオ、trial数、population sizeが小規模設定へ変更されます。

## 本計算

3シナリオを12プロセスで各1,000 trialまで実行します。

```powershell
python .\run_vensim_with_pysd_to7_optuna.py `
  --scenarios present 2C 4C `
  --workers 12 `
  --n-trials 1000 `
  --population-size 100 `
  --dividing-parameter 3 `
  --seed 42
```

`--workers` は1以上の任意の整数に変更できます。各プロセスはPySDモデルを個別にロードして、そのプロセス内で再利用します。入れ子の並列化は行いません。

`--n-trials` は追加trial数ではなく、**各シナリオの正常終了trialの目標総数**です。NSGA-IIIの世代を完結させるため、通常は `--population-size` の倍数にします。

既定では `data` 内の最新かつ完全なDE結果を初期政策として使います。特定のDE結果を使う場合はstampを指定します。

```powershell
python .\run_vensim_with_pysd_to7_optuna.py `
  --scenarios present `
  --workers 12 `
  --n-trials 1000 `
  --de-results 260914_174448
```

DE解を使わない場合は `--no-de-seed` を指定します。

## 中断と再開

実行中のrunは `data\optuna_to7_<stamp>` に保存されます。Ctrl+Cで中断した場合も、COMPLETEになったtrialはJournalStorageに残ります。

同じ設定で、目標総数まで再開します。

```powershell
python .\run_vensim_with_pysd_to7_optuna.py `
  --resume-dir .\data\optuna_to7_<stamp> `
  --workers 12 `
  --n-trials 1000
```

再開時に `--population-size`、`--dividing-parameter`、`--seed`、`--scenarios` を省略すると、保存済み設定を使います。これらを明示して保存済み設定と食い違った場合は停止します。`--workers` と目標trial数は再開時に変更できます。

計算を増やす場合は、同じrunをより大きい目標総数で再開します。

```powershell
python .\run_vensim_with_pysd_to7_optuna.py `
  --resume-dir .\data\optuna_to7_<stamp> `
  --workers 12 `
  --n-trials 5000
```

複数プロセスではtrialの完了順序が実行ごとに変わる可能性があります。厳密なtrial系列の再現確認には `--workers 1` を使用します。

## 出力の再生成

JournalからCSVだけを再生成できます。SD最適化は追加実行されません。Pareto trialの年次一時ファイルがない場合に限り、その政策を再評価します。

```powershell
python .\run_vensim_with_pysd_to7_optuna.py `
  --resume-dir .\data\optuna_to7_<stamp> `
  --export-only
```

主な出力は次のとおりです。

```text
data\optuna_to7_<stamp>\
  config.json
  run_status.json
  baseline.csv
  baseline.json
  study_<scenario>.log
  trials_<scenario>.csv
  pareto_summary_<scenario>.csv
  pareto_decisions_<scenario>.csv
  pareto_yearly_<scenario>.csv
  trial_yearly\<scenario>\trial_*.csv
```

`trials_<scenario>.csv` には予算違反を含む全trialが入り、`pareto_*` には予算を満たすPareto解だけが入ります。

## 可視化

最新runを自動選択して図を作ります。

```powershell
python .\plot_optuna_to7_results.py
```

runを固定する場合は次のように指定します。

```powershell
python .\plot_optuna_to7_results.py `
  --run-dir .\data\optuna_to7_<stamp>
```

図は `figures\optuna_to7_<stamp>` に保存されます。シナリオごとに、Pareto parallel coordinates、目的間散布図、予算との関係、探索進行診断を作成します。

## テスト

第三者ライブラリを必要としない変換・制約テストは、WSL側でも実行できます。Optunaがインストールされた環境ではJournalStorageの2プロセステストも実行されます。

```powershell
python -m unittest -v .\test_optuna_to7.py
python -m py_compile .\run_vensim_with_pysd_to7_optuna.py .\plot_optuna_to7_results.py
```
