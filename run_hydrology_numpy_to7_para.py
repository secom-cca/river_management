"""LHS uncertainty study for assumptions that enter the NumPy hydrology."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from River_management_xls_to7_numpy import (
    TOTAL_DAYS,
    load_weather,
    simulate_hydrology_batch,
)
from hydrology_numpy_to7_common import (
    HYDROLOGY_ASSUMPTION_BOUNDS,
    SCENARIOS,
    WARMUP_DAYS,
    YEAR_INDEX,
    YEAR_LENGTHS,
    YEAR_STARTS,
    latin_hypercube_samples,
    load_observed_flows,
    nse,
    parameters_for_scenario,
)


N_SAMPLES = 100
RANDOM_SEED = 42
SAVE_DAILY_OUTPUT = True
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = Path("data/hydrology_numpy_para_to7") / RUN_STAMP

DAILY_OUTPUT_COLUMNS = [
    "daily_precip_up",
    "daily_precip_down",
    "scenario_adjusted_daily_precip_up",
    "upstream_storage",
    "upstream_storage_capacity",
    "river_discharge_upstream",
    "dam_storage",
    "dam_capacity",
    "dam_inflow",
    "predischarge_control",
    "dam_spill",
    "downstream_storage",
    "downstream_storage_capacity",
    "paddyfield_storage_capacity",
    "drainage_capacity",
    "river_discharge_downstream",
    "inside_water_innundation_level",
    "innundation_level",
]


def build_parameter_sets():
    names = list(HYDROLOGY_ASSUMPTION_BOUNDS)
    base = {name: HYDROLOGY_ASSUMPTION_BOUNDS[name][2] for name in names}
    rows = [{"case_id": "base", "sample_index": -1, **base}]
    rng = np.random.default_rng(RANDOM_SEED)
    samples = latin_hypercube_samples(rng, N_SAMPLES, names)
    for sample_index, unit_values in enumerate(samples):
        values = {}
        for name, unit_value in zip(names, unit_values):
            low, high, _base = HYDROLOGY_ASSUMPTION_BOUNDS[name]
            values[name] = low + unit_value * (high - low)
        rows.append(
            {
                "case_id": f"lhs_{sample_index:04d}",
                "sample_index": sample_index,
                **values,
            }
        )
    return pd.DataFrame(rows)


def yearly_rows(scenario, cases, result):
    rows = []
    for case_index, case in cases.iterrows():
        for year, (start, length) in enumerate(zip(YEAR_STARTS, YEAR_LENGTHS)):
            section = slice(int(start), int(start + length))
            up = result["river_discharge_upstream"][case_index, section]
            down = result["river_discharge_downstream"][case_index, section]
            inundation = result["innundation_level"][case_index, section]
            inside_water = result[
                "inside_water_innundation_level"
            ][case_index, section]
            rows.append(
                {
                    "scenario": scenario,
                    "case_id": case["case_id"],
                    "year": year,
                    "calendar_year": 2009 + year,
                    "river_discharge_upstream_year_sum": up.sum(),
                    "river_discharge_upstream_year_mean": up.mean(),
                    "river_discharge_upstream_year_max": up.max(),
                    "river_discharge_downstream_year_sum": down.sum(),
                    "river_discharge_downstream_year_mean": down.mean(),
                    "river_discharge_downstream_year_max": down.max(),
                    "dam_storage_year_mean": result["dam_storage"][
                        case_index, section
                    ].mean(),
                    "dam_storage_year_max": result["dam_storage"][
                        case_index, section
                    ].max(),
                    "dam_spill_year_sum": result["dam_spill"][
                        case_index, section
                    ].sum(),
                    "innundation_level_year_sum": inundation.sum(),
                    "innundation_level_year_max": inundation.max(),
                    "inside_water_innundation_days": int(
                        np.count_nonzero(inside_water > 0)
                    ),
                }
            )
    return rows


def overall_rows(scenario, cases, result, observed):
    rows = []
    warm = slice(WARMUP_DAYS, None)
    for case_index, case in cases.iterrows():
        rows.append(
            {
                "scenario": scenario,
                "case_id": case["case_id"],
                "nse_upstream": nse(
                    observed["flow"][warm],
                    result["river_discharge_upstream"][case_index, warm],
                ),
                "nse_downstream": nse(
                    observed["flow_d"][warm],
                    result["river_discharge_downstream"][case_index, warm],
                ),
                "river_discharge_upstream_max": result[
                    "river_discharge_upstream"
                ][case_index].max(),
                "river_discharge_downstream_max": result[
                    "river_discharge_downstream"
                ][case_index].max(),
                "dam_spill_total": result["dam_spill"][case_index].sum(),
                "inside_water_innundation_days": int(
                    np.count_nonzero(
                        result["inside_water_innundation_level"][case_index] > 0
                    )
                ),
                "innundation_level_max": result["innundation_level"][
                    case_index
                ].max(),
            }
        )
    return rows


def daily_frame(scenario, cases, result):
    case_count = len(cases)
    frame = pd.DataFrame(
        {
            "scenario": np.repeat(scenario, case_count * TOTAL_DAYS),
            "case_id": np.repeat(cases["case_id"].to_numpy(), TOTAL_DAYS),
            "time": np.tile(np.arange(TOTAL_DAYS, dtype=int), case_count),
            "year": np.tile(YEAR_INDEX, case_count),
        }
    )
    for name in DAILY_OUTPUT_COLUMNS:
        frame[name] = result[name].reshape(-1)
    return frame


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    cases = build_parameter_sets()
    cases.to_csv(OUTPUT_DIR / "parameter_sets.csv", index=False)

    weather = load_weather(number_of_days=TOTAL_DAYS)
    observed = load_observed_flows()
    yearly = []
    overall = []
    timing = []
    daily_path = OUTPUT_DIR / "hydrology_daily.csv"

    for scenario_index, scenario in enumerate(SCENARIOS):
        parameter_sets = []
        for _index, case in cases.iterrows():
            overrides = {
                name: case[name] for name in HYDROLOGY_ASSUMPTION_BOUNDS
            }
            parameter_sets.append(parameters_for_scenario(scenario, overrides))

        started = perf_counter()
        result = simulate_hydrology_batch(
            parameter_sets,
            weather=weather,
            output_columns=DAILY_OUTPUT_COLUMNS,
        )
        elapsed = perf_counter() - started
        timing.append(
            {
                "scenario": scenario,
                "cases": len(cases),
                "days_per_case": TOTAL_DAYS,
                "calculation_seconds": elapsed,
            }
        )
        yearly.extend(yearly_rows(scenario, cases, result))
        overall.extend(overall_rows(scenario, cases, result, observed))

        if SAVE_DAILY_OUTPUT:
            daily_frame(scenario, cases, result).to_csv(
                daily_path,
                index=False,
                mode="w" if scenario_index == 0 else "a",
                header=scenario_index == 0,
            )
        print(
            f"{scenario}: {len(cases)} cases x {TOTAL_DAYS} days "
            f"completed in {elapsed:.3f} seconds."
        )

    yearly_frame = pd.DataFrame(yearly)
    overall_frame = pd.DataFrame(overall).merge(cases, on="case_id", how="left")
    yearly_frame.to_csv(OUTPUT_DIR / "yearly_summary.csv", index=False)
    overall_frame.to_csv(OUTPUT_DIR / "overall_summary.csv", index=False)
    pd.DataFrame(timing).to_csv(OUTPUT_DIR / "timing.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
