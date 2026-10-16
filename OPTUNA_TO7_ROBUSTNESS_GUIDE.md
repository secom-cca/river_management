# 全パレート解の気候横断・頑健性分析

対象は、Optuna NSGA-IIIで得た各気候シナリオのパレート解です。処理は2段階に分けています。

## 1. 全パレート解を3気候条件で評価

```powershell
python .\evaluate_optuna_to7_all_pareto.py `
  --run-dir .\data\optuna_to7_260915_102704 `
  --workers 12
```

既定の保存先は `analysis/optuna_to7_260915_102704/all_pareto_cross_scenario` です。同一気候の既存結果と、代表政策の気候横断評価を自動的に再利用します。中断後に同じコマンドを実行すると、未完了ケースだけを再開します。全件を計算し直す場合だけ `--force` を付けます。

主な出力は次のとおりです。

- `all_pareto_policies.csv`: 141パレート解の目的値と予算
- `all_pareto_decisions.csv`: 141パレート解の政策変数
- `cross_scenario_results.csv`: 141政策×3気候条件の423評価
- `cross_scenario_yearly.csv`: 423評価×15年の年次結果

## 2. 論文用の頑健性分析

```powershell
python .\analyze_optuna_to7_robustness.py `
  --run-dir .\data\optuna_to7_260915_102704
```

既定の保存先は `analysis/optuna_to7_260915_102704/paper_robustness` です。

分析には次が含まれます。

1. 全141政策の平均後悔値・最悪後悔値
2. 設計気候別の分布、上位10%の構成、Kruskal–Wallis検定とHolm補正済みのペア比較
3. 平均最良、最悪時最良、予算を考慮した頑健政策の選定
4. 政策変数と頑健性のSpearman相関、頑健上位10%との中央値比較
5. 3正規化方法×6目的重み×4気候確率、合計72条件の順位感度分析

基準分析は、各対象気候のパレート理想点・最悪点によるmin–max正規化、5目的等重み、3気候等確率です。`baseline_to_ideal` と `cross_policy_range` による正規化も感度分析に含めています。

「Budget robust」は、平均後悔値と最悪後悔値がともに全政策の上位25%に入る政策群から、総予算使用率が最小の政策を選びます。この範囲は `--robust-quantile` で変更できます。

現在の気候入力は各シナリオ1系列です。このコードが検証するのは現在・2℃・4℃の3シナリオ間の頑健性であり、複数の気象実現値に対する頑健性は含みません。
