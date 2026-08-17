# Basin configuration guide

`template.yaml` をコピーして流域ごとの設定を作成します。

## 入力ファイルの共通形式

現行モデルはファイル名だけをYAMLから切り替え、シート・列構成は共通です。

### 上下流気象Excel

- シート名: `input`
- A列: モデル日（0始まり）
- B列: 日降水量（mm/day）
- C列: 日平均気温（℃）
- D列: 日最高気温（℃）
- E列: 日最低気温（℃）
- F列: 全天日射量（MJ/m²/day）

### 上下流観測流量CSV

- A列: モデル日（0始まり）
- D列: 観測流量（m³/day）
- 文字コード: UTF-8

計算期間と全入力系列の長さを一致させてください。先頭欠測はPySDが補間しますが、
較正前に `calibrate_hydrology.py --check-only` の警告を確認してください。

## 主な単位

- `initial_dam_capacity`: m³
- `initial_drainage_capacity`: m³/s
- `current_highwater_discharge`, `future_highwater_discharge`: m³/s
- `upstream_area`, `downstream_area`: ha
- `landslide_design_daily_precipitation`: mm/day
- `minimum_river_discharge`: m³/day
- `forest_area_ratio`, `paddy_field_ratio`, リスク比率: 0～1
- `crop_price`: 円/kg
- `*_damage_per_resident`, `gdp_per_resident`: 円/人/day
- `paddy_field_capacity_per_area`, `paddy_dam_capacity_per_area`: m³/ha
- 9つの流出・浸透パラメータ: 無次元比率

`river_water_level_base` と `river_water_level_discharge_coefficient` は、現在の
`水位 = base + sqrt(日流量) × coefficient` という簡略式に使われます。別河川では
観測されたH-Q関係に基づいて値を設定し、この式で不十分ならモデル式自体を変更してください。
