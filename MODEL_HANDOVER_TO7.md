# 筑後川流域SDモデル: 引継ぎメモ

## 目的と対象範囲

- 筑後川流域を対象に、水文、水害、農業、森林、生物多様性、人口・経済、適応策の相互作用を扱うシステムダイナミクス（SD）モデル。
- 日次タイムステップで、2009年から2023年までの15年間を計算する。うるう年を含む。
- 将来気候は、現在気候、2℃上昇、4℃上昇の3シナリオを扱う。
- 目的は、気候変化下での適応策間のトレードオフ、被害軽減、農業収益、生物多様性への影響を分析すること。

## 主要ファイル

| ファイル | 用途 |
|---|---|
| `River_management_xls_to6.py` | 現在のPySDモデル本体。`to7`の分析・最適化もこのモデルを利用する。 |
| `run_vensim_with_pysd_to7_para.py` | 初期値を中心にLHSでパラメータ不確実性を分析するスクリプト。 |
| `plot_para_to7_uncertainty.py` | パラメータスタディ結果の年次不確実性バンド、レンジバー、相関ヒートマップを出力する。 |
| `run_vensim_with_pysd_to7_opt.py` | 初期値固定・シナリオ別の適応策最適化スクリプト。 |
| `run_vensim_with_pysd_to5_opt_time.py` | 旧最適化コード。現行分析では参照用であり、直接使わない。 |

`to6` はモデルの版番号、`to7` は主にパラメータスタディ・最適化スクリプトの分析版番号として使われている。現在の最適化スクリプトは `River_management_xls_to6.py` を直接ロードする。

## 気候・水文

### 気候シナリオ

| シナリオ | 降水係数 | 平均・最高・最低気温への加算 |
|---|---:|---:|
| `present` | 1.0 | 0.0℃ |
| `2C` | 1.1 | 1.3℃ |
| `4C` | 1.3 | 4.1℃ |

- 降水係数は、流入量・土砂災害リスク、および収量回帰に用いる7-9月降水量へ適用される。元データの `daily_precip_up/down` 自体はシナリオ間で同じ値になる。
- 上流の気象データはHita、下流の気象データはAsakuraを読む仕様であり、この対応は維持する。

### 水文パラメータ

- 流量について、上流・下流の流量データを用いて調整済み。NSEは概ね0.5。
- 現在のパラメータスタディと最適化では、以下を固定する。

```text
upstream_outflow_ratio = 0.272198
downstream_outflow_ratio = 0.240634
direct_discharge_ratio = 0.815126
upstream_percolation_ratio = 0.5
downstream_deep_percolation_ratio = 0.0263544
upstream_middle_flow_ratio = 0.443267
downstream_percolation_ratio = 0.5
downstream_middle_flow_ratio = 0.0326612
upstream_deep_percolation_ratio = 0.540584
```

## 農業サブモデル

### コメ収量と売上

- 年間の単位収量は、7-9月の降水量・日射量合計を用いる回帰式で計算する。

```text
yield_per_10a = 469.68 - 0.0571 * precip_jul_sep + 0.0462 * solar_jul_sep
```

- 単位は `kg/10a`。
- 年間総収量は、収穫日（10月1日）時点の水田面積を用いて算出する。

```text
yearly_total_crop_yield_kg = yield_per_10a * paddy_field_at_harvest * 10
```

- コメ売上は、品質補正を反映する。

```text
yearly_crop_revenue = yearly_total_crop_yield_kg * crop_price * crop_price_quality_factor
```

- `crop_price` の初期値は250円/kg。2009-2023年の再現期間に整合する生産者受取価格として扱う。
- `crop_production_cashflow` は、収穫日にその年の売上を一括計上する日次キャッシュフロー。15年収益の集計には、この変数の日次合計を用いる。

### 洪水による水田被害

- 浸水した水田は当年中に回復して収穫できるとはみなさない。
- 被害水田は翌年以降に回復を開始し、90%回復まで2年となる日次幾何回復率で回復する。回復期間は `paddy_field_recovery_years_to_90_percent` で変更できる。
- 水田・収量関係の診断用として、`paddy_field_at_harvest`、浸水日数、浸水イベント数、浸水水田面積日数などを出力している。

### コメ品質と品種改良

- `chalky_kernel_ratio` と `crop_price_quality_factor` により、高温による品質低下を価格へ反映する。
- 品種改良は `annual_breeding_investment` で実施する。
- 投資が途中で中断しても累積投資額は保持され、累計5億円に達すると成功する。
- 成功後は白未熟粒率を半分にする効果が永続する。
- 品種改良費は `municipality_cost` に含まれる。

## 森林サブモデル

### 基本構造

- `forest_area = upstream_area * forest_area_ratio`。植林によって森林の物理面積を増やす構造は使わない。
- 初期面積構成は、天然林40%、管理済み人工林30%、未管理人工林30%。
- `forest_function_coef` は面積加重平均で計算する。

```text
(天然林面積 * 1.0
 + 管理済み人工林面積 * 管理済み人工林係数
 + 未管理人工林面積 * 未管理人工林係数)
/ forest_area
```

- 未管理人工林係数の初期値は0.7。パラメータスタディでは0.5-0.9の範囲で扱った。
- `forest_function_coef` は、森林保水能力、森林の土砂災害抑制効果、生物多様性に反映する。
- CO2吸収には森林機能係数を適用しない。

### 森林管理施策

- `annual_forest_management_conversion_area` は、未管理人工林を年間何ha管理対象へ転換するかを表す。
- 新規管理面積は係数0.70で開始し、6つの回復段階を通じて0.75、0.80、0.85、0.90、0.95、1.0へ改善する。
- 既存の管理済み人工林は係数1.0。
- 管理費は37,200円/ha/年。これは資料に基づく値であり、48,000円へ戻す必要はない。
- `additional_forest_management_cost` は、初期時点以降に管理対象となった面積だけの年間追加費用。最適化の予算制約ではこの増分だけを使う。
- `number_of_planting_trees`、`tree_growth_time` 等は旧植林ロジックの名残であり、現行の森林機能には実質的に影響しない。最適化対象から除外する。

### 林業と生物多様性

- 林業売上は、管理済み人工林面積の2%を伐採するとして計算する。

```text
lumbering_area = managed_plantation_forest_area * 0.02
sales_of_forestry = lumbering_area * 1,010,000
```

- 生物多様性指標は、天然林、管理済み人工林、未管理人工林の機能係数を反映した面積を森林面積で除したもの。

## パラメータ不確実性分析

`run_vensim_with_pysd_to7_para.py` は、以下14パラメータをLHSで同時に振る。

```text
forest_area_ratio
paddy_field_ratio
ratio_of_paddy_field_in_risky_area
innundation_risky_area_ratio
flood_risky_area_ratio
recovery_ratio
crop_price
waterholding_capacity_of_forest_base
innundation_damage_per_resident
flood_damage_per_resident
gdp_per_resident
paddy_field_capacity_per_area
paddy_dam_capacity_per_area
unmanaged_plantation_forest_coef
```

土砂災害指標は、日田の100年確率日降水量 `322 mm/day` を固定基準とする連続指標である。

```text
forest_mitigation = forest_area_ratio * waterholding_capacity_of_forest_base * forest_function_coef
landslide_disaster_risk = MAX((scenario_adjusted_daily_precip_up - forest_mitigation) / 322, 0)
```

`erosion_control_of_forest` は独立パラメータではなく、`waterholding_capacity_of_forest_base` から導出する。球磨川流域の森林保水機能に関する国土交通省資料の `200–250 mm` を代理根拠とし、LHSでは `200–250 mm`、初期値 `225 mm` とする。保水容量を土砂災害の日雨量低減量へ読み替える点は、モデル上の仮定として明記する。

- 現在は試計算設定として `N_SAMPLES = 100`、`MAX_WORKERS = 15`。
- 各気候シナリオで基準ケース + LHS100サンプルを実行する。最終分析では`N_SAMPLES = 300`以上へ増やし、相関の安定性を確認する。
- 実行時刻を含むファイル名で、日次出力、年次サマリ、全期間サマリ、パラメータセットを`data/`へ保存する。
- 日次出力には降水、気温、日射、流量、河川水位、品質、収量、浸水、森林、経済などを含める。

## 適応策最適化

### スクリプト

- `run_vensim_with_pysd_to7_opt.py` を使用する。
- 初期値固定で、`present`、`2C`、`4C` をそれぞれ独立に最適化する。
- 差分進化法（SciPy）を使う。現在は探索確認用として `MAXITER = 20`、`POPSIZE = 8`、`MAX_WORKERS = 10`、`OPTIMIZATION_SEEDS = [42]`。
- 21変数なので、`POPSIZE = 8` は1世代あたり約168候補を評価する。最終分析では、`OPTIMIZATION_SEEDS = [42, 43, 44]`、`MAXITER`・`POPSIZE`の拡大による再現性確認を行う。

### 最適化変数

| 施策 | 最適化内容 |
|---|---|
| 排水 | 一時投資額、開始年 |
| ダム | 一時投資額、開始年 |
| 堤防 | 一時投資額、開始年 |
| 住宅嵩上げ | 年間件数、開始年、終了年 |
| 移転 | 年間件数、開始年、終了年 |
| 田んぼダム | 年額（上限50億円）、開始年、終了年。水田全体への導入で実投資を停止 |
| 品種改良 | 年額5,000万円で実施するか否か、開始年、終了年 |
| 森林管理 | 年間転換面積、開始年、終了年 |

排水・ダム・堤防は、最適化で年番号を実日数へ変換して開始日と365/366日の投資配分期間を渡す。うるう年でも投資総額と開始年がずれない。

### 目的関数

- モデルの金額は絶対額の精度に限界があるため、各シナリオの無対策基準ケースに対する相対比で評価する。
- 被害・土砂災害リスクは最小化、コメ売上・生物多様性は最大化する。
- 現在は重みがすべて1の `balanced` パターンのみ。目的ごとの重みを変えたパターンを追加すれば、Pareto前線の近似点を増やせる。
- 基準ケースの土砂災害リスクがゼロの場合、相対改善率が定義できないため、そのシナリオの土砂災害リスクは目的関数から除外され、出力のみ行う。

### 予算制約

- 費用を最小化目的には入れない。予算制約の範囲で効果を最大化する。
- `BUDGET_CASE` は以下から選択する。

| ケース | 年間上限 | 15年間累計上限 |
|---|---:|---:|
| `constrained` | 20億円 | 200億円 |
| `standard` | 50億円 | 500億円 |
| `expanded` | 100億円 | 1,000億円 |

- 予算は、モデル出力の`municipality_cost`を単純合計せず、施策ごとの一時費用・年額費用を最適化スクリプトで明示的に集計する。これはモデル内の費用変数に日額・年額・一時費用が混在しているため。
- 既存管理済み人工林の管理費は基礎費用として除外し、追加管理面積分のみ予算に含める。
- 年間・累計上限を超える候補には大きなペナルティを与える。

### 最適化の出力

実行時刻付きで以下を`data/`へ保存する。

```text
opt_to7_<budget_case>_<timestamp>_summary.csv
opt_to7_<budget_case>_<timestamp>_decisions.csv
opt_to7_<budget_case>_<timestamp>_yearly.csv
```

- `summary`: 15年間の結果、無対策比、予算消化率、最適化情報。
- `decisions`: 施策強度、開始年、終了年。
- `yearly`: 年別の被害、収益、生物多様性、森林面積、政策追加費用。

## 実行コマンド

PowerShellで、リポジトリ直下から実行する。

```powershell
python .\run_vensim_with_pysd_to7_para.py
python .\plot_para_to7_uncertainty.py
python .\run_vensim_with_pysd_to7_opt.py
```

## 検証状況と注意点

- 水文は上流・下流流量に対してNSE約0.5。
- 再現性確認では、水害被害額の決定係数は約0.76、コメ品質は約0.34、コメ収量は約0.41。品質・収量は符号・傾向の再現として解釈し、絶対額への過度な依存は避ける。
- 被害額は相関・傾向に基づく相対評価を主に用いる。
- 森林管理による面積移行と機能回復、排水投資の開始年は新規追加部分である。初回実行時に、`managed_plantation_forest_area`、`forest_function_coef`、`additional_forest_management_cost` の時系列が意図どおり変化するかを確認する。
- 最適化は現在、重み付き和による`balanced`解を1点出す。厳密なPareto最適化を行うには、複数の重みパターンを追加するか、NSGA-II等を導入する。
- `municipality_cost` はモデル全体の報告指標としては残すが、日次合計を政策予算と解釈しないこと。
- この引継ぎメモ作成環境にはPySDがないため、新規の森林管理・最適化スクリプトはPython構文チェック済みだが、PySDによる実行確認はWindowsの実行環境で行う必要がある。
