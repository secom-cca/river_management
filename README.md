# River Management Model

河川管理・洪水リスク・農林業・地域経済を扱うシステムダイナミクスモデルです。
現行系統は `River_management_xls_to6.py` を、流域別YAML設定から実行します。

## セットアップ

Python 3.10以降を想定しています。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 流域設定

流域固有値は `config/basins/` に集約しています。

- `chikugo.yaml`: 現在の筑後川設定
- `template.yaml`: 別河川用テンプレート
- `README.md`: 入力列・単位・設定上の注意
- `basin_config.py`: 設定の読み込みと検証

YAMLには次を記述します。

- 計算開始年、年数、閏年の扱い
- 上下流の気象・観測流量ファイル
- 気候シナリオ別の降水倍率・気温上昇量
- 流域面積、ダム容量、計画高水流量、森林・水田・世帯・被害単価
- 較正済み水文パラメータ
- 感度分析範囲
- 水文較正の探索範囲

流域は環境変数 `RIVER_BASIN` で選択します。未指定時は `chikugo` です。

```bash
# 筑後川
python run_vensim_with_pysd_to7_para.py

# config/basins/new_basin.yaml を使用
RIVER_BASIN=new_basin python run_vensim_with_pysd_to7_para.py
```

YAMLの絶対パスを `RIVER_BASIN` に指定することもできます。

## 入力データ

筑後川設定は次の4ファイルを参照します。

```text
data/
├── jma_asakura_2009_2023.xlsx
├── jma_hita_2009_2023.xlsx
├── flow_senoshita_2009_2023_x100000_0to364_utf8.csv
└── flow_arase_2009_2023_x100000_0to364_utf8.csv
```

別河川でも、現在のPySDモデルと同じシート・列構成に整形してください。
気象Excelは `input` シート、観測流量CSVはA列が時刻、D列が流量です。
`data/` はGit管理外です。

## 別河川への適用手順

1. `config/basins/template.yaml` を `<流域キー>.yaml` としてコピーする
2. 上下流の気象・流量データを準備し、`inputs` を更新する
3. 面積、ダム・排水能力、計画高水流量、土地利用、世帯数などを入力する
4. `calibrate_hydrology.py` で水文パラメータを較正する
5. 較正期間と独立検証期間のNSEを確認する
6. 較正済みYAMLを採用してパラメータスタディ・施策最適化を実行する

河道形状が大きく異なる場合、YAML値だけでなく流量―水位関係や洪水・内水被害関数の
構造確認も必要です。現在の流量―水位関係は、基準水位と平方根係数をYAMLから設定できます。

## 水文パラメータの較正

```bash
RIVER_BASIN=new_basin python calibrate_hydrology.py \
  --maxiter 100 \
  --popsize 10 \
  --seed 42
```

9つの流出・浸透パラメータをSciPyのDifferential Evolutionで探索します。
既定では期間先頭70%を較正、残り30%を独立検証に使用し、上下流NSEを同時に評価します。
比率はYAMLの `hydrology_calibration.calibration_fraction` または
`--calibration-fraction` で変更できます。

最適化前に、現在のYAML値と入力データだけを確認できます。

```bash
RIVER_BASIN=new_basin python calibrate_hydrology.py --check-only
```

```text
results/calibration/<流域キー>/<日時>/
├── calibration_report.yaml
├── best_daily_flow.csv
└── <流域キー>_calibrated.yaml
```

元の設定は自動上書きしません。結果と検証値を確認してから、生成された
`<流域キー>_calibrated.yaml` を `config/basins/` に採用してください。

## パラメータスタディ

```bash
RIVER_BASIN=chikugo python run_vensim_with_pysd_to7_para.py
```

標本数・並列数などはスクリプト冒頭で設定します。

- `SAMPLE_MODE`: `one_at_a_time`、`random`、`lhs`
- `N_SAMPLES`: ランダム/LHSの標本数
- `RANDOM_SEED`: 乱数シード
- `SAVE_DAILY_OUTPUT`: 日次結果を保存するか
- `MAX_WORKERS`: 並列プロセス数

感度分析対象と `[下限, 上限, 基準値]` は流域YAMLの `sensitivity_bounds` で管理します。
日次出力は大きいため、不要なら `SAVE_DAILY_OUTPUT = False` にしてください。

```text
results/parameter_study/<流域キー>/
├── para_to7_all_scenarios_<日時>_parameter_sets.csv
├── para_to7_all_scenarios_<日時>_daily.csv
├── para_to7_all_scenarios_<日時>_yearly_summary.csv
└── para_to7_all_scenarios_<日時>_overall_summary.csv
```

## 施策最適化・可視化

```bash
RIVER_BASIN=chikugo python run_vensim_with_pysd_to7_opt.py
RIVER_BASIN=chikugo python plot_opt_to7_results.py
RIVER_BASIN=chikugo python plot_opt_to7_patterns.py
RIVER_BASIN=chikugo python visualize_parallel_coordinates.py
```

- 最適化結果: `results/optimization/<流域キー>/`
- 最適化図: `figures/opt_to7/<流域キー>/`
- 平行座標ビューアは流域別パラメータスタディ結果を優先し、旧 `data/` にもフォールバック

`run_vensim_with_pysd_to7_opt.py` が最適化するのはダム・堤防・排水・移転などの施策です。
水文パラメータの較正は `calibrate_hydrology.py` が担当します。

## その他のワークフロー

- `app.py`: `River_management_xls.py` を使う旧Streamlit系統
- `get_suimon_database.py`: 国土交通省WISの日流量取得
- `compute_nies_metrics.py`: NIES気候データの年次指標作成
- `src/`: AMeDAS・サンプルデータ・lookup作成
- `sde.config.js`, `config/`, `packages/`: SDEverywhere
- `archive/`: 旧版モデル・実行コード・Vensim較正生成物

研究資料の詳細説明:
[Google Docs](https://docs.google.com/document/d/116Xg9WkcorllC6vz6C-agFRKM3BDTerf/edit?usp=drive_link)
