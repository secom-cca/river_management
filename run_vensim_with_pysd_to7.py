from pathlib import Path
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from pysd import load


warnings.filterwarnings(
    "ignore",
    message=r"_ext_data_flow(_d)?",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"_ext_data_daily_precip_up",
    category=UserWarning,
)

# ---- 入力ファイル設定 ----
MODEL_PY = Path("River_management_xls_to7.py")

# ---- 変換済みPySDモデルの読み込み ----
if MODEL_PY.exists():
    model = load(MODEL_PY.as_posix())
else:
    raise FileNotFoundError(
        "River_management_xls_to7.py が見つかりません。"
    )

# ---- シミュレーション設定（Vensim の CONTROL に合わせる）----
CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
USE_LEAP_YEARS = True
#SCENARIO = "present"  # "present", "2C", "4C"
#SCENARIO = "2C"
SCENARIO = "4C"
RUN_DATE = datetime.now().strftime("%y%m%d_%H%M")

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
    "drainage_investment_amount": 0,
    "number_of_house_elevation": 0,
    "number_of_migration": 0,
    "annual_paddy_dam_investment": 0,
    "annual_breeding_investment": 0,
    "recovery_ratio": 0.9,
    "delay_of_recovery": 365,
    "initial_total_risk_households": 50_000,
    "risk_household_overlap_ratio": 0.5,
    "flood_damage_per_household": 30_000_000,
    "inundation_damage_per_household": 30_000_000,
    "average_household_size": 2.21,
    "daily_paddy_damage_fraction": 0.10,
    "paddy_damage_depth_threshold": 0.3,
    "paddy_field_recovery_years_to_90_percent": 2.0,
    "paddy_dam_cost_per_area": 1_500_000,
    "predischarge_capacity_ratio": 0.10,
    
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
    "dam_capacity",
    "dam_inflow",
    "predischarge_control",
    "predischarge_capacity",
    "predischarge_capacity_ratio",
    "dam_spill",
    "dam_outflow",
    "downstream_storage",
    "downstream_storage_capacity",
    "paddyfield_storage_capacity",
    "upstream_storage",
    "river_discharge_upstream",
    "river_discharge_downstream",
    "houses_damaged_by_inundation",
    "financial_damage_by_inundation",
    "financial_damage_by_flood",
    "flood_damage_ratio",
    "inundation_damage_ratio",
    "flood_event_max_damage_ratio",
    "inundation_event_max_damage_ratio",
    "combined_event_max_damage_ratio",
    "incremental_flood_damage_fraction",
    "incremental_inundation_damage_fraction",
    "incremental_combined_damage_fraction",
    "incremental_flood_damage_ratio",
    "incremental_inundation_damage_ratio",
    "flood_event_healthy_households",
    "inundation_event_healthy_households",
    "newly_damaged_flood_only_houses",
    "newly_damaged_inundation_only_houses",
    "newly_damaged_overlapping_houses",
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
    "damaging_paddy_inundation_flag",
    "physical_paddy_field",
    "current_year_damaged_paddy_field",
    "carryover_damaged_paddy_field",
    "damaged_paddy_field",
    "undamaged_risky_paddy_field",
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
    "flood_only_risk_houses",
    "innundation_only_risk_houses",
    "overlapping_risk_houses",
    "damaged_flood_only_houses",
    "damaged_inundation_only_houses",
    "damaged_overlapping_houses",
    "damaged_houses",
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
            "physical_paddy_field_at_harvest": harvest_row["physical_paddy_field"],
            "damaged_paddy_field_at_harvest": harvest_row["damaged_paddy_field"],
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
            "lumbering_area_year_sum": year_df["lumbering_area"].sum(),
            "sales_of_forestry_year_sum": year_df["sales_of_forestry"].sum(),
            "forest_area_storage_capacity": harvest_row["forest_area_storage_capacity"],
            "co2_absorption": harvest_row["co2_absorption"],
            "landslide_disaster_risk": harvest_row["landslide_disaster_risk"],
            "biodiversity": harvest_row["biodiversity"],
            "financial_damage_by_inundation_year_sum": year_df[
                "financial_damage_by_inundation"
            ].sum(),
            "financial_damage_by_flood_year_sum": year_df[
                "financial_damage_by_flood"
            ].sum(),
            "damaged_houses_at_year_end": year_df.iloc[-1]["damaged_houses"],
            "dam_storage_at_year_end": year_df.iloc[-1]["dam_storage"],
            "dam_storage_year_min": year_df["dam_storage"].min(),
            "dam_storage_year_max": year_df["dam_storage"].max(),
            "predischarge_control_year_sum": year_df["predischarge_control"].sum(),
            "predischarge_capacity_year_mean": year_df[
                "predischarge_capacity"
            ].mean(),
            "dam_spill_year_sum": year_df["dam_spill"].sum(),
        }
    )

yearly_summary = pd.DataFrame(yearly_summary_rows)

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

Path("data").mkdir(exist_ok=True)
res.to_csv(
    f"data/simulation_output_to7_{SCENARIO}_{RUN_DATE}_rice_forest.csv",
    index_label="day",
)
yearly_summary.to_csv(
    f"data/simulation_output_to7_{SCENARIO}_{RUN_DATE}_rice_forest_yearly.csv",
    index=False,
)
