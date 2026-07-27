from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import numpy as np
import pandas as pd
from pysd import load


# PySD reports a known missing initial point in the validated observed-flow
# files. The model interpolates it; suppress only this repeated warning.
warnings.filterwarnings(
    "ignore",
    message=r"_ext_data_flow(_d)?",
    category=UserWarning,
)


# ---- Input model ----
MODEL_PY = Path("River_management_xls_to6.py")
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        if MODEL_PY.exists():
            _MODEL = load(MODEL_PY.as_posix())
        else:
            raise FileNotFoundError("River_management_xls_to6.py が見つかりません。")
    return _MODEL


# ---- Simulation settings ----
CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
USE_LEAP_YEARS = True
SCENARIOS = ["present", "2C", "4C"]
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M")

SAMPLE_MODE = "lhs"  # "one_at_a_time", "random", or "lhs"
N_SAMPLES = 100
RANDOM_SEED = 42
SAVE_DAILY_OUTPUT = True
MAX_WORKERS = 15 #=10でCPU負荷50%ぐらい。

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

for scenario in SCENARIOS:
    if scenario not in SCENARIO_TO_PRECIP_RATIO:
        raise ValueError(f"Unknown scenario: {scenario}")


ASSUMPTION_BOUNDS = {
    "forest_area_ratio": (0.85, 0.95, 0.92),
    "paddy_field_ratio": (0.10, 0.20, 0.15),
    "ratio_of_paddy_field_in_risky_area": (0.05, 0.15, 0.10),
    "innundation_risky_area_ratio": (0.5, 0.8, 0.8),
    "flood_risky_area_ratio": (0.5, 0.8, 0.8),
    "recovery_ratio": (0.5, 1.0, 0.9),
    "crop_price": (200, 400, 250),
    # This sampled value also determines erosion_control_of_forest in the model.
    "waterholding_capacity_of_forest_base": (200, 250, 225),
    "innundation_damage_per_resident": (5e6 / 365, 1.5e7 / 365, 1e7 / 365),
    "flood_damage_per_resident": (5e6 / 365, 1.5e7 / 365, 1e7 / 365),
    "gdp_per_resident": (2.5e6 / 365, 3.5e6 / 365, 3.0e6 / 365),
    "paddy_field_capacity_per_area": (400, 600, 500),
    "paddy_dam_capacity_per_area": (1000, 2000, 1500),
    "unmanaged_plantation_forest_coef": (0.5, 0.9, 0.7),
}


def fixed_params_for_scenario(scenario):
    return {
        "daily_precipitation_future_ratio": SCENARIO_TO_PRECIP_RATIO[scenario],
        "temperature_scenario_shift": SCENARIO_TO_TEMP_SHIFT[scenario],
        "levee_investment_amount": 0,
        "dam_investment_amount": 0,
        "drainage_investment_amount": 0,
        "number_of_house_elevation": 0,
        "number_of_migration": 0,
        "annual_paddy_dam_investment": 0,
        "annual_breeding_investment": 0,
        # Calibrated hydrological parameters. Keep fixed in this assumption study.
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


RETURN_COLS = [
    "day_of_year",
    "harvest_trigger",
    "year_end_trigger",
    "daily_total_gdp",
    "daily_precip_up",
    "daily_precip_down",
    "daily_precipitation_future_ratio",
    "daily_ave_temp_up",
    "daily_min_temp_up",
    "daily_max_temp_up",
    "daily_ave_temp_down",
    "daily_min_temp_down",
    "daily_max_temp_down",
    "solar_radiation_up",
    "solar_radiation_down",
    "flow",
    "flow_d",
    "river_discharge_upstream",
    "river_discharge_downstream",
    "river_water_level_upstream",
    "river_water_level_downstream",
    "flood_water_level",
    "heat_stress_at_heading_plus_20",
    "accumulated_heat_stress",
    "financial_damage_by_innundation",
    "financial_damage_by_flood",
    "daily_crop_production",
    "accumulated_crop_production_within_year",
    "yearly_crop_production",
    "yearly_crop_production_per_day",
    "accumulated_precipitation_jul_sep",
    "accumulated_inundation_until_harvest",
    "paddy_field_at_harvest",
    "yield_per_10a",
    "yearly_total_crop_yield_kg",
    "yearly_crop_revenue",
    "chalky_kernel_ratio",
    "effective_chalky_kernel_ratio",
    "crop_price_quality_factor",
    "quality_adjusted_crop_price",
    "innundation_level",
    "inundation_flag",
    "accumulated_inundation_days_until_harvest",
    "accumulated_inundation_events_until_harvest",
    "accumulated_inundated_paddy_areadays_until_harvest",
    "current_year_damaged_paddy_field",
    "carryover_damaged_paddy_field",
    "damaged_paddy_field",
    "paddy_field_recovery_years_to_90_percent",
    "paddy_field_recovery_rate_per_day",
    "forest_area",
    "forest_function_coef",
    "forest_management_cost",
    "sales_of_forestry",
    "forest_area_storage_capacity",
    "co2_absorption",
    "scenario_adjusted_daily_precip_up",
    "forest_mitigation",
    "landslide_design_daily_precipitation",
    "landslide_disaster_risk",
    "biodiversity",
    "houses_in_flood_risk",
    "houses_in_inundation_risk",
    "flood_only_risk_houses",
    "innundation_only_risk_houses",
    "overlapping_risk_houses",
    "houses_in_nonrisky_area",
    "elevated_houses",
    "municipality_cost",
]


def _is_leap_year(year):
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


if USE_LEAP_YEARS:
    YEAR_LENGTHS = [
        366 if _is_leap_year(CALENDAR_START_YEAR + i) else 365
        for i in range(CALENDAR_NUM_YEARS)
    ]
else:
    YEAR_LENGTHS = [365] * CALENDAR_NUM_YEARS

YEAR_STARTS = np.cumsum([0] + YEAR_LENGTHS[:-1]).tolist()
TOTAL_DAYS = int(np.sum(YEAR_LENGTHS))
TIME = list(range(0, TOTAL_DAYS, 1))


def _year_index_from_day(day_number):
    idx = int(np.searchsorted(YEAR_STARTS, day_number, side="right") - 1)
    return int(np.clip(idx, 0, len(YEAR_LENGTHS) - 1))


def _base_assumption_params():
    return {name: spec[2] for name, spec in ASSUMPTION_BOUNDS.items()}


def _latin_hypercube_samples(rng, sample_count, parameter_names):
    lhs = np.empty((sample_count, len(parameter_names)))
    for col_index, _name in enumerate(parameter_names):
        cutpoints = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        lhs[:, col_index] = rng.permutation(cutpoints)
    return lhs


def build_parameter_sets():
    base = _base_assumption_params()
    cases = [{"case_id": "base", "varied_parameter": "base", "level": "base", **base}]

    if SAMPLE_MODE == "one_at_a_time":
        for name, (low, high, _initial) in ASSUMPTION_BOUNDS.items():
            low_case = dict(base)
            low_case[name] = low
            cases.append(
                {
                    "case_id": f"{name}_low",
                    "varied_parameter": name,
                    "level": "low",
                    **low_case,
                }
            )

            high_case = dict(base)
            high_case[name] = high
            cases.append(
                {
                    "case_id": f"{name}_high",
                    "varied_parameter": name,
                    "level": "high",
                    **high_case,
                }
            )
    elif SAMPLE_MODE in {"random", "lhs"}:
        rng = np.random.default_rng(RANDOM_SEED)
        names = list(ASSUMPTION_BOUNDS)

        if SAMPLE_MODE == "lhs":
            unit_samples = _latin_hypercube_samples(rng, N_SAMPLES, names)
        else:
            unit_samples = rng.random((N_SAMPLES, len(names)))

        for sample_index in range(N_SAMPLES):
            sample = {}
            for param_index, name in enumerate(names):
                low, high, _initial = ASSUMPTION_BOUNDS[name]
                sample[name] = float(low + unit_samples[sample_index, param_index] * (high - low))
            cases.append(
                {
                    "case_id": f"{SAMPLE_MODE}_{sample_index:04d}",
                    "varied_parameter": f"{SAMPLE_MODE}_all",
                    "level": SAMPLE_MODE,
                    **sample,
                }
            )
    else:
        raise ValueError(f"Unknown SAMPLE_MODE: {SAMPLE_MODE}")

    return pd.DataFrame(cases)


def run_case(case, scenario):
    model = get_model()
    params = fixed_params_for_scenario(scenario)
    params.update({name: case[name] for name in ASSUMPTION_BOUNDS})

    res = model.run(
        params=params,
        return_timestamps=TIME,
        return_columns=RETURN_COLS,
        initial_condition="original",
    )
    res = res.copy()
    res.insert(0, "scenario", scenario)
    res.insert(1, "case_id", case["case_id"])
    res.insert(2, "varied_parameter", case["varied_parameter"])
    res.insert(3, "level", case["level"])
    res["year"] = [_year_index_from_day(day) for day in res.index]
    # Raw precipitation remains observational input. This derived column shows
    # the scenario-adjusted precipitation used by the crop-yield regression.
    res["daily_precip_down_scenario_adjusted"] = (
        res["daily_precip_down"] * res["daily_precipitation_future_ratio"]
    )

    yearly_gdp_total = res.groupby("year")["daily_total_gdp"].sum()
    res["yearly_gdp_total"] = 0.0
    year_end_idx = res.groupby("year").tail(1).index
    res.loc[year_end_idx, "yearly_gdp_total"] = yearly_gdp_total.values

    yearly_rows = []
    for year, year_df in res.groupby("year"):
        harvest_rows = year_df[year_df["harvest_trigger"] > 0]
        harvest_day = (
            int(harvest_rows.index[-1]) if not harvest_rows.empty else int(year_df.index[-1])
        )
        harvest_row = res.loc[harvest_day]
        year_end_row = year_df.iloc[-1]
        year_length = YEAR_LENGTHS[year]
        sep_end = sum([31, 29 if year_length == 366 else 28, 31, 30, 31, 30, 31, 31, 30])
        jan_sep_df = year_df[year_df["day_of_year"] < sep_end]

        yearly_rows.append(
            {
                "scenario": scenario,
                "case_id": case["case_id"],
                "varied_parameter": case["varied_parameter"],
                "level": case["level"],
                "year": year,
                "harvest_day": harvest_day,
                "paddy_field_at_harvest": harvest_row["paddy_field_at_harvest"],
                "current_year_damaged_paddy_field_at_harvest": harvest_row[
                    "current_year_damaged_paddy_field"
                ],
                "carryover_damaged_paddy_field_at_harvest": harvest_row[
                    "carryover_damaged_paddy_field"
                ],
                "damaged_paddy_field_at_harvest": harvest_row[
                    "damaged_paddy_field"
                ],
                "current_year_damaged_paddy_field_at_year_end": year_end_row[
                    "current_year_damaged_paddy_field"
                ],
                "carryover_damaged_paddy_field_at_year_end": year_end_row[
                    "carryover_damaged_paddy_field"
                ],
                "damaged_paddy_field_at_year_end": year_end_row[
                    "damaged_paddy_field"
                ],
                "paddy_field_recovery_years_to_90_percent": harvest_row[
                    "paddy_field_recovery_years_to_90_percent"
                ],
                "paddy_field_recovery_rate_per_day": harvest_row[
                    "paddy_field_recovery_rate_per_day"
                ],
                "accumulated_inundation_until_harvest": harvest_row[
                    "accumulated_inundation_until_harvest"
                ],
                "daily_precip_up_at_harvest": harvest_row["daily_precip_up"],
                "daily_precip_down_at_harvest": harvest_row["daily_precip_down"],
                "daily_precipitation_future_ratio": harvest_row[
                    "daily_precipitation_future_ratio"
                ],
                "daily_precip_down_scenario_adjusted_at_harvest": harvest_row[
                    "daily_precip_down_scenario_adjusted"
                ],
                "accumulated_precipitation_jul_sep_at_harvest": harvest_row[
                    "accumulated_precipitation_jul_sep"
                ],
                "daily_ave_temp_up_at_harvest": harvest_row["daily_ave_temp_up"],
                "daily_min_temp_up_at_harvest": harvest_row["daily_min_temp_up"],
                "daily_max_temp_up_at_harvest": harvest_row["daily_max_temp_up"],
                "daily_ave_temp_down_at_harvest": harvest_row["daily_ave_temp_down"],
                "daily_min_temp_down_at_harvest": harvest_row["daily_min_temp_down"],
                "daily_max_temp_down_at_harvest": harvest_row["daily_max_temp_down"],
                "solar_radiation_up_at_harvest": harvest_row["solar_radiation_up"],
                "solar_radiation_down_at_harvest": harvest_row["solar_radiation_down"],
                "flow_at_harvest": harvest_row["flow"],
                "flow_d_at_harvest": harvest_row["flow_d"],
                "river_discharge_upstream_at_harvest": harvest_row[
                    "river_discharge_upstream"
                ],
                "river_discharge_downstream_at_harvest": harvest_row[
                    "river_discharge_downstream"
                ],
                "river_water_level_upstream_at_harvest": harvest_row[
                    "river_water_level_upstream"
                ],
                "river_water_level_downstream_at_harvest": harvest_row[
                    "river_water_level_downstream"
                ],
                "flood_water_level_at_harvest": harvest_row["flood_water_level"],
                "daily_crop_production_at_harvest": harvest_row["daily_crop_production"],
                "accumulated_crop_production_within_year_at_harvest": harvest_row[
                    "accumulated_crop_production_within_year"
                ],
                "yearly_crop_production_at_harvest": harvest_row["yearly_crop_production"],
                "yearly_crop_production_per_day_at_harvest": harvest_row[
                    "yearly_crop_production_per_day"
                ],
                "chalky_kernel_ratio_at_harvest": harvest_row["chalky_kernel_ratio"],
                "effective_chalky_kernel_ratio_at_harvest": harvest_row[
                    "effective_chalky_kernel_ratio"
                ],
                "crop_price_quality_factor_at_harvest": harvest_row[
                    "crop_price_quality_factor"
                ],
                "quality_adjusted_crop_price_at_harvest": harvest_row[
                    "quality_adjusted_crop_price"
                ],
                "heat_stress_at_heading_plus_20_at_harvest": harvest_row[
                    "heat_stress_at_heading_plus_20"
                ],
                "accumulated_heat_stress_at_harvest": harvest_row[
                    "accumulated_heat_stress"
                ],
                "innundation_level_at_harvest": harvest_row["innundation_level"],
                "innundation_level_jan_sep_sum": jan_sep_df["innundation_level"].sum(),
                "innundation_level_jan_sep_mean": jan_sep_df["innundation_level"].mean(),
                "innundation_level_jan_sep_max": jan_sep_df["innundation_level"].max(),
                "daily_precip_up_year_sum": year_df["daily_precip_up"].sum(),
                "daily_precip_down_year_sum": year_df["daily_precip_down"].sum(),
                "daily_precip_down_scenario_adjusted_year_sum": year_df[
                    "daily_precip_down_scenario_adjusted"
                ].sum(),
                "daily_precip_up_year_mean": year_df["daily_precip_up"].mean(),
                "daily_precip_down_year_mean": year_df["daily_precip_down"].mean(),
                "daily_precip_up_year_max": year_df["daily_precip_up"].max(),
                "scenario_adjusted_daily_precip_up_year_max": year_df[
                    "scenario_adjusted_daily_precip_up"
                ].max(),
                "daily_precip_down_year_max": year_df["daily_precip_down"].max(),
                "daily_ave_temp_up_year_mean": year_df["daily_ave_temp_up"].mean(),
                "daily_ave_temp_down_year_mean": year_df["daily_ave_temp_down"].mean(),
                "daily_ave_temp_up_year_max": year_df["daily_ave_temp_up"].max(),
                "daily_ave_temp_down_year_max": year_df["daily_ave_temp_down"].max(),
                "daily_ave_temp_up_year_min": year_df["daily_ave_temp_up"].min(),
                "daily_ave_temp_down_year_min": year_df["daily_ave_temp_down"].min(),
                "solar_radiation_up_year_sum": year_df["solar_radiation_up"].sum(),
                "solar_radiation_down_year_sum": year_df["solar_radiation_down"].sum(),
                "solar_radiation_up_year_mean": year_df["solar_radiation_up"].mean(),
                "solar_radiation_down_year_mean": year_df["solar_radiation_down"].mean(),
                "river_discharge_upstream_year_sum": year_df["river_discharge_upstream"].sum(),
                "river_discharge_downstream_year_sum": year_df["river_discharge_downstream"].sum(),
                "river_discharge_upstream_year_mean": year_df["river_discharge_upstream"].mean(),
                "river_discharge_downstream_year_mean": year_df["river_discharge_downstream"].mean(),
                "river_discharge_upstream_year_max": year_df["river_discharge_upstream"].max(),
                "river_discharge_downstream_year_max": year_df["river_discharge_downstream"].max(),
                "inundation_days_until_harvest": harvest_row[
                    "accumulated_inundation_days_until_harvest"
                ],
                "inundation_events_until_harvest": harvest_row[
                    "accumulated_inundation_events_until_harvest"
                ],
                "inundated_paddy_areadays_until_harvest": harvest_row[
                    "accumulated_inundated_paddy_areadays_until_harvest"
                ],
                "yield_per_10a": harvest_row["yield_per_10a"],
                "yearly_total_crop_yield_kg": harvest_row["yearly_total_crop_yield_kg"],
                "yearly_crop_revenue": harvest_row["yearly_crop_revenue"],
                "forest_area": harvest_row["forest_area"],
                "forest_function_coef": harvest_row["forest_function_coef"],
                "forest_management_cost": harvest_row["forest_management_cost"],
                "sales_of_forestry": harvest_row["sales_of_forestry"],
                "forest_area_storage_capacity": harvest_row["forest_area_storage_capacity"],
                "co2_absorption": harvest_row["co2_absorption"],
                "landslide_disaster_risk_at_harvest": harvest_row[
                    "landslide_disaster_risk"
                ],
                "landslide_disaster_risk_year_sum": year_df[
                    "landslide_disaster_risk"
                ].sum(),
                "landslide_disaster_risk_year_max": year_df[
                    "landslide_disaster_risk"
                ].max(),
                "forest_mitigation_year_mean": year_df["forest_mitigation"].mean(),
                "landslide_design_daily_precipitation": harvest_row[
                    "landslide_design_daily_precipitation"
                ],
                "biodiversity": harvest_row["biodiversity"],
                "financial_damage_by_innundation_year_sum": year_df[
                    "financial_damage_by_innundation"
                ].sum(),
                "financial_damage_by_flood_year_sum": year_df[
                    "financial_damage_by_flood"
                ].sum(),
                "municipality_cost_year_sum": year_df["municipality_cost"].sum(),
                "yearly_gdp_total": year_end_row["yearly_gdp_total"],
            }
        )

    yearly_summary = pd.DataFrame(yearly_rows)
    return res, yearly_summary


def run_task(task):
    scenario, case = task
    return scenario, case["case_id"], *run_case(case, scenario)


def build_overall_summary(yearly_summary):
    metric_aggs = {
        "paddy_field_at_harvest": "mean",
        "damaged_paddy_field_at_harvest": "mean",
        "damaged_paddy_field_at_year_end": "mean",
        "paddy_field_recovery_years_to_90_percent": "mean",
        "paddy_field_recovery_rate_per_day": "mean",
        "daily_precipitation_future_ratio": "mean",
        "daily_precip_up_year_sum": "mean",
        "daily_precip_down_year_sum": "mean",
        "daily_precip_down_scenario_adjusted_year_sum": "mean",
        "daily_precip_up_year_mean": "mean",
        "daily_precip_down_year_mean": "mean",
        "river_discharge_upstream_year_sum": "mean",
        "river_discharge_downstream_year_sum": "mean",
        "river_discharge_upstream_year_mean": "mean",
        "river_discharge_downstream_year_mean": "mean",
        "river_discharge_upstream_year_max": "mean",
        "river_discharge_downstream_year_max": "mean",
        "scenario_adjusted_daily_precip_up_year_max": "mean",
        "daily_ave_temp_up_year_mean": "mean",
        "daily_ave_temp_down_year_mean": "mean",
        "daily_ave_temp_up_year_max": "mean",
        "daily_ave_temp_down_year_max": "mean",
        "daily_ave_temp_up_year_min": "mean",
        "daily_ave_temp_down_year_min": "mean",
        "solar_radiation_up_year_sum": "mean",
        "solar_radiation_down_year_sum": "mean",
        "solar_radiation_up_year_mean": "mean",
        "solar_radiation_down_year_mean": "mean",
        "daily_crop_production_at_harvest": "mean",
        "accumulated_precipitation_jul_sep_at_harvest": "mean",
        "yearly_crop_production_at_harvest": "mean",
        "yearly_crop_production_per_day_at_harvest": "mean",
        "yield_per_10a": "mean",
        "yearly_total_crop_yield_kg": "mean",
        "yearly_crop_revenue": "mean",
        "chalky_kernel_ratio_at_harvest": "mean",
        "effective_chalky_kernel_ratio_at_harvest": "mean",
        "crop_price_quality_factor_at_harvest": "mean",
        "quality_adjusted_crop_price_at_harvest": "mean",
        "heat_stress_at_heading_plus_20_at_harvest": "mean",
        "accumulated_heat_stress_at_harvest": "mean",
        "biodiversity": "mean",
        "forest_area_storage_capacity": "mean",
        "landslide_disaster_risk_year_sum": "sum",
        "landslide_disaster_risk_year_max": "max",
        "forest_mitigation_year_mean": "mean",
        "landslide_design_daily_precipitation": "mean",
        "financial_damage_by_innundation_year_sum": "sum",
        "financial_damage_by_flood_year_sum": "sum",
        "municipality_cost_year_sum": "sum",
        "yearly_gdp_total": "sum",
    }
    return (
        yearly_summary.groupby(
            ["scenario", "case_id", "varied_parameter", "level"],
            as_index=False,
        )
        .agg(metric_aggs)
        .sort_values(["scenario", "varied_parameter", "level", "case_id"])
    )


def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    parameter_sets = build_parameter_sets()
    parameter_sets.to_csv(
        output_dir / f"para_to7_all_scenarios_{RUN_STAMP}_parameter_sets.csv",
        index=False,
    )

    all_daily = []
    all_yearly = []
    total_runs = len(SCENARIOS) * len(parameter_sets)
    tasks = [
        (scenario, case.to_dict())
        for scenario in SCENARIOS
        for _case_index, case in parameter_sets.iterrows()
    ]

    completed = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_task, task) for task in tasks]
        for future in as_completed(futures):
            scenario, case_id, daily_result, yearly_summary = future.result()
            completed += 1
            print(f"[{completed}/{total_runs}] completed {scenario} / {case_id}")
            all_yearly.append(yearly_summary)

            if SAVE_DAILY_OUTPUT:
                all_daily.append(daily_result.reset_index().rename(columns={"index": "day"}))

    if SAVE_DAILY_OUTPUT:
        daily_all = pd.concat(all_daily, ignore_index=True)
        daily_all.to_csv(
            output_dir / f"para_to7_all_scenarios_{RUN_STAMP}_daily.csv",
            index=False,
        )

    yearly_all = pd.concat(all_yearly, ignore_index=True)
    yearly_all.to_csv(
        output_dir / f"para_to7_all_scenarios_{RUN_STAMP}_yearly_summary.csv",
        index=False,
    )

    overall_summary = build_overall_summary(yearly_all)
    overall_summary.to_csv(
        output_dir / f"para_to7_all_scenarios_{RUN_STAMP}_overall_summary.csv",
        index=False,
    )

    print("\nParameter study completed.")
    print(f"Scenarios: {len(SCENARIOS)}")
    print(f"Cases per scenario: {len(parameter_sets)}")
    print(f"Runs: {total_runs}")
    print(overall_summary.head())


if __name__ == "__main__":
    main()
