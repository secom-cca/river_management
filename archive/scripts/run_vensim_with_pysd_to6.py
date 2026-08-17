from pathlib import Path
import numpy as np
from pysd import read_vensim, load

# ---- 入力ファイル設定 ----
#MODEL_MDL = Path("River_management_xls_to6.mdl")  # Vensim テキスト
MODEL_PY = Path("River_management_xls_to6.py")    # 変換済み PySD モデル

# ---- モデル読み込み（.mdl を読み込んで自動変換）----
if MODEL_PY.exists():
    model = load(MODEL_PY.as_posix())
#elif MODEL_MDL.exists():
#    model = read_vensim(MODEL_MDL.as_posix())
else:
    raise FileNotFoundError(
        #"River_management_xls_to6.py も .mdl も見つかりません。"
        "River_management_xls_to6.py が見つかりません。"
    )

# ---- シミュレーション設定（Vensim の CONTROL に合わせる）----
CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
USE_LEAP_YEARS = True  # River_management_xls_to6.mdl と合わせる
#SCENARIO = "present"  # "present", "2C", "4C"
#SCENARIO = "2C"
SCENARIO = "4C"
RUN_DATE = "260421_2"

SCENARIO_TO_PRECIP_RATIO = {
    "present": 1.0,
    "2C": 1.1,
    "4C": 1.3,
}

SCENARIO_TO_TEMP_SHIFT = {
    "present": 0.0,
    "2C": 1.3,
    "4C": 4.1,
}

if SCENARIO not in SCENARIO_TO_PRECIP_RATIO:
    raise ValueError(f"Unknown scenario: {SCENARIO}")


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

# ---- パラメータ上書き（指定のパラメータ）----
params = {
    "daily_precipitation_future_ratio": SCENARIO_TO_PRECIP_RATIO[SCENARIO],
    "temperature_scenario_shift": SCENARIO_TO_TEMP_SHIFT[SCENARIO],
    "levee_investment_amount": 0,
    "dam_investment_amount": 0,
    
    # 以下, Vensimで最適化した水文系パラメータ (opt_result_chikugo_260415.txt)
    "upstream_outflow_ratio": 0.272198,
    "downstream_outflow_ratio": 0.240634,
    "direct_discharge_ratio": 0.815126,
    "upstream_percolation_ratio": 0.5,
    "downstream_deep_percolation_ratio": 0.0263544,
    "upstream_middle_flow_ratio": 0.443267,
    "downstream_percolation_ratio": 0.5,
    "downstream_middle_flow_ratio": 0.0326612,
    "upstream_deep_percolation_ratio": 0.540584,
}

# ---- 取得したい出力（スネークケース）----
return_cols = [
    "day_of_year",
    "harvest_trigger",
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
    "quality_adjusted_crop_price",
    "crop_production_cashflow",
    "accumulated_precipitation_jul_sep",
    "accumulated_solar_radiation_jul_sep",
    "accumulated_inundation_until_harvest",
    "paddy_field_at_harvest",
    "yield_per_10a",
    "yearly_total_crop_yield_kg",
    "yearly_crop_revenue",
    "accumulated_breeding_investment",
    "breeding_success",
    "flow",
    "flow_d",
    "chalky_kernel_ratio",
    "effective_chalky_kernel_ratio",
    "innundation_level",
    "heat_stress_at_heading_plus_20",
    "daily_ave_temp_up",
    "daily_min_temp_up",
    "daily_max_temp_up",
    "solar_radiation_up",
    "daily_ave_temp_down",
    "daily_min_temp_down",
    "daily_max_temp_down",
    "solar_radiation_down",
    "daily_precip_up",
    "daily_precip_down",
    "inundation_flag",
    "inundated_paddy_field_area",
    "accumulated_inundation_days_until_harvest",
    "accumulated_inundation_events_until_harvest",
    "accumulated_inundated_paddy_areadays_until_harvest",
    "forest_area",
    "natural_forest_area",
    "managed_plantation_forest_area",
    "unmanaged_plantation_forest_area",
    "managed_plantation_forest_coef",
    "unmanaged_plantation_forest_coef",
    "forest_function_coef",
    "forest_management_cost",
    "lumbering_area",
    "sales_of_forestry",
    "forest_area_storage_capacity",
    "co2_absorption",
    "landslide_disaster_risk",
    "biodiversity",
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

res["year"] = [_year_index_from_day(day) for day in res.index]
yearly_gdp_total = res.groupby("year")["daily_total_gdp"].sum()
res["yearly_gdp_total"] = 0.0
year_end_idx = res.groupby("year").tail(1).index
res.loc[year_end_idx, "yearly_gdp_total"] = yearly_gdp_total.values

yearly_summary_rows = []
for year, year_df in res.groupby("year"):
    harvest_rows = year_df[year_df["harvest_trigger"] > 0]
    harvest_day = int(harvest_rows.index[-1]) if not harvest_rows.empty else int(year_df.index[-1])
    harvest_row = res.loc[harvest_day]

    year_length = year_lengths[year]
    july_start = sum([31, 29 if year_length == 366 else 28, 31, 30, 31, 30])
    sep_end = sum([31, 29 if year_length == 366 else 28, 31, 30, 31, 30, 31, 31, 30])
    # Inundation: Jan-Sep (no reason to exclude pre-July); climate regression: Jul-Sep.
    jan_sep_df = year_df[year_df["day_of_year"] < sep_end]
    jul_sep_df = year_df[
        (year_df["day_of_year"] >= july_start) & (year_df["day_of_year"] < sep_end)
    ]

    yearly_summary_rows.append(
        {
            "year": year,
            "harvest_day": harvest_day,
            "paddy_field_at_harvest": harvest_row["paddy_field_at_harvest"],
            "accumulated_inundation_until_harvest": harvest_row["accumulated_inundation_until_harvest"],
            "innundation_level_at_harvest": harvest_row["innundation_level"],
            "innundation_level_jan_sep_sum": jan_sep_df["innundation_level"].sum(),
            "innundation_level_jan_sep_mean": jan_sep_df["innundation_level"].mean(),
            "innundation_level_jan_sep_max": jan_sep_df["innundation_level"].max(),
            "inundation_days_until_harvest": harvest_row["accumulated_inundation_days_until_harvest"],
            "inundation_events_until_harvest": harvest_row["accumulated_inundation_events_until_harvest"],
            "inundated_paddy_areadays_until_harvest": harvest_row["accumulated_inundated_paddy_areadays_until_harvest"],
            "accumulated_precipitation_jul_sep": harvest_row["accumulated_precipitation_jul_sep"],
            "accumulated_solar_radiation_jul_sep": harvest_row["accumulated_solar_radiation_jul_sep"],
            "yield_per_10a": harvest_row["yield_per_10a"],
            "yearly_total_crop_yield_kg": harvest_row["yearly_total_crop_yield_kg"],
            "yearly_crop_revenue": harvest_row["yearly_crop_revenue"],
            "accumulated_breeding_investment": harvest_row["accumulated_breeding_investment"],
            "breeding_success": harvest_row["breeding_success"],
            "forest_area": harvest_row["forest_area"],
            "natural_forest_area": harvest_row["natural_forest_area"],
            "managed_plantation_forest_area": harvest_row["managed_plantation_forest_area"],
            "unmanaged_plantation_forest_area": harvest_row["unmanaged_plantation_forest_area"],
            "managed_plantation_forest_coef": harvest_row["managed_plantation_forest_coef"],
            "unmanaged_plantation_forest_coef": harvest_row["unmanaged_plantation_forest_coef"],
            "forest_function_coef": harvest_row["forest_function_coef"],
            "forest_management_cost": harvest_row["forest_management_cost"],
            "lumbering_area": harvest_row["lumbering_area"],
            "sales_of_forestry": harvest_row["sales_of_forestry"],
            "forest_area_storage_capacity": harvest_row["forest_area_storage_capacity"],
            "co2_absorption": harvest_row["co2_absorption"],
            "landslide_disaster_risk": harvest_row["landslide_disaster_risk"],
            "biodiversity": harvest_row["biodiversity"],
        }
    )

yearly_summary = __import__("pandas").DataFrame(yearly_summary_rows)

print(res[return_cols].head())
print("\nYear-end values (last timestep):")
print(
    res.iloc[[-1]][
        [
            "yield_per_10a",
            "yearly_total_crop_yield_kg",
            "yearly_crop_revenue",
            "accumulated_inundation_until_harvest",
            "yearly_gdp_total",
        ]
    ]
)
print("\nYearly summary:")
print(yearly_summary)

res.to_csv(
    f"data/simulation_output_to6_{SCENARIO}_{RUN_DATE}_rice_forest.csv",
    index_label="day",
)
yearly_summary.to_csv(
    f"data/simulation_output_to6_{SCENARIO}_{RUN_DATE}_rice_forest_yearly.csv",
    index=False,
)
