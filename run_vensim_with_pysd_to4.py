from pathlib import Path
import numpy as np
from pysd import read_vensim, load

# ---- 入力ファイル設定 ----
MODEL_MDL = Path("River_management_xls_to4.mdl")  # Vensim テキスト
MODEL_PY = Path("River_management_xls_to4.py")    # 変換済み PySD モデル

# ---- モデル読み込み（.py があれば優先、なければ .mdl を読み込んで自動変換）----
if MODEL_PY.exists():
    model = load(MODEL_PY.as_posix())
elif MODEL_MDL.exists():
    model = read_vensim(MODEL_MDL.as_posix())
else:
    raise FileNotFoundError("River_management_xls_to4.py も .mdl も見つかりません。")

# ---- シミュレーション設定（Vensim の CONTROL に合わせる）----
CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
USE_LEAP_YEARS = True  # River_management_xls_to4.py と合わせる


def _is_leap_year(year):
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


if USE_LEAP_YEARS:
    year_lengths = [
        366 if _is_leap_year(CALENDAR_START_YEAR + i) else 365
        for i in range(CALENDAR_NUM_YEARS)
    ]
else:
    year_lengths = [365] * CALENDAR_NUM_YEARS

total_days = int(np.sum(year_lengths))
time = list(range(0, total_days, 1))

# ---- パラメータ上書き（必要な場合のみ）----
# opt_mode を切り替えると最適化済みパラメータを自動適用
#opt_mode = "opt_riv_dis_down"  # None or "opt_riv_dis_down" #2026/01/07 以降、基本使わない. 
#opt_mode = "opt_riv_dis_up"  # None or "opt_riv_dis_up" #2026/01/07. 
opt_mode = "opt_riv_dis_up_down"  # None or "opt_riv_dis_up" #2026/01/15. 


params = {
    "daily_precipitation_future_ratio": 1,
    "levee_investment_amount": 0,
    "dam_investment_amount": 0,
}

if opt_mode == "opt_riv_dis_down":
    params.update(
        {
            "upstream_outflow_ratio": 0.0270484,
            "downstream_outflow_ratio": 0.0446371,
            "direct_discharge_ratio": 0.99,
            "upstream_percolation_ratio": 0.303752,
            "downstream_deep_percolation_ratio": 0.251362,
            "upstream_middle_flow_ratio": 0.141604,
            "downstream_percolation_ratio": 0.191405,
            "downstream_middle_flow_ratio": 0.0620332,
            "waterholding_capacity_of_forest": 157.484,
            "upstream_deep_percolation_ratio": 0.922893,
        }
    )


if opt_mode == "opt_riv_dis_up":#2026/01/07追加
    params.update(
        {
            "upstream_outflow_ratio": 0.189252,
            "direct_discharge_ratio": 0.663718,
            "upstream_percolation_ratio": 0.499523,
            "upstream_middle_flow_ratio": 0.400464,
            "upstream_deep_percolation_ratio": 0.648598,
        }
    )

if opt_mode == "opt_riv_dis_up_down":#2026/01/15追加
    params.update(
        {
            "upstream_outflow_ratio": 0.187365,
            "downstream_outflow_ratio": 0.22335,
            "direct_discharge_ratio": 0.662729,
            "upstream_percolation_ratio": 0.5,
            "downstream_deep_percolation_ratio": 0.0268072,
            "upstream_middle_flow_ratio": 0.403377,
            "downstream_percolation_ratio": 0.5,
            "downstream_middle_flow_ratio": 0.0330995,
            "upstream_deep_percolation_ratio": 0.663594,
        }
    )


# ---- 取得したい出力（スネークケース）----
return_cols = [
    "day_of_year",
    "year_end_trigger", 
    "heat_stress_sample_trigger",  
    "daily_total_gdp",
    "dam_storage",
    "downstream_storage",
    "upstream_storage",
    "river_discharge_upstream",
    "river_discharge_downstream",
    "houses_damaged_by_inundation",
    "financial_damage_by_innundation",
    "financial_damage_by_flood",
    "daily_crop_production",
    "accumulated_crop_production_within_year",
    "yearly_crop_production",
    "yearly_crop_production_per_day",
    "crop_production_cashflow",
    "flow",
    "flow_down",
    "chalky_kernel_ratio",
    "heat_stress_at_heading_plus_20",
    "daily_ave_temp",
    "daily_min_temp",
    "daily_max_temp",
    "solar_radiation",
]

# ---- 実行 ----
res = model.run(
    params=params,
    return_timestamps=time,
    return_columns=return_cols,
)

# 年次GDP（合計）を年末ごとに格納
res = res.copy()
year_starts = np.cumsum([0] + year_lengths[:-1]).tolist()


def _year_index_from_day(day_number):
    idx = int(np.searchsorted(year_starts, day_number, side="right") - 1)
    return int(np.clip(idx, 0, len(year_lengths) - 1))


res["year"] = [ _year_index_from_day(day) for day in res.index ]
yearly_gdp_total = res.groupby("year")["daily_total_gdp"].sum()
res["yearly_gdp_total"] = 0.0
year_end_idx = res.groupby("year").tail(1).index
res.loc[year_end_idx, "yearly_gdp_total"] = yearly_gdp_total.values

print(res[return_cols].head())
print("\nYear-end values (last timestep):")
print(res.iloc[[-1]][["yearly_crop_production", "yearly_gdp_total"]])

#res.to_csv("data/simulation_output_to3.csv", index_label="day")
res.to_csv("data/simulation_output_to4_260414.csv", index_label="day")
