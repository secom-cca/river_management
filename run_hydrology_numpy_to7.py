"""Run the standalone NumPy to7 hydrology model for all climate scenarios."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from River_management_xls_to7_numpy import (
    HYDROLOGY_OUTPUT_COLUMNS,
    TOTAL_DAYS,
    load_weather,
    simulate_hydrology_batch,
)
from hydrology_numpy_to7_common import (
    SCENARIOS,
    YEAR_INDEX,
    YEAR_STARTS,
    load_observed_flows,
    parameters_for_scenario,
    performance_summary,
)


RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = Path("data/hydrology_numpy_to7") / RUN_STAMP


def annual_summary(daily):
    rows = []
    for (scenario, year), frame in daily.groupby(["scenario", "year"], sort=True):
        for location, observed_column, simulated_column in (
            ("upstream", "flow", "river_discharge_upstream"),
            ("downstream", "flow_d", "river_discharge_downstream"),
        ):
            observed = frame[observed_column]
            simulated = frame[simulated_column]
            rows.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "location": location,
                    "observed_sum": observed.sum(),
                    "simulated_sum": simulated.sum(),
                    "observed_mean": observed.mean(),
                    "simulated_mean": simulated.mean(),
                    "observed_max": observed.max(),
                    "simulated_max": simulated.max(),
                }
            )
    return pd.DataFrame(rows)


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    weather = load_weather(number_of_days=TOTAL_DAYS)
    observed = load_observed_flows()
    scenario_names = list(SCENARIOS)
    parameter_sets = [parameters_for_scenario(name) for name in scenario_names]

    started = perf_counter()
    results = simulate_hydrology_batch(
        parameter_sets,
        weather=weather,
        output_columns=HYDROLOGY_OUTPUT_COLUMNS,
    )
    elapsed = perf_counter() - started

    frames = []
    summaries = []
    time = np.arange(TOTAL_DAYS, dtype=int)
    for case_index, scenario in enumerate(scenario_names):
        frame = pd.DataFrame(
            {name: values[case_index] for name, values in results.items()}
        )
        frame.insert(0, "time", time)
        frame.insert(1, "year", YEAR_INDEX)
        frame.insert(2, "calendar_year", 2009 + YEAR_INDEX)
        frame.insert(3, "scenario", scenario)
        frame["flow"] = observed["flow"]
        frame["flow_d"] = observed["flow_d"]
        frames.append(frame)
        summaries.extend(
            [
                {
                    "scenario": scenario,
                    **performance_summary(
                        observed["flow"],
                        results["river_discharge_upstream"][case_index],
                        "upstream",
                    ),
                },
                {
                    "scenario": scenario,
                    **performance_summary(
                        observed["flow_d"],
                        results["river_discharge_downstream"][case_index],
                        "downstream",
                    ),
                },
            ]
        )

    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(OUTPUT_DIR / "hydrology_daily.csv", index=False)
    annual_summary(daily).to_csv(
        OUTPUT_DIR / "hydrology_annual_summary.csv", index=False
    )
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "nse_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_stamp": RUN_STAMP,
                "engine": "NumPy batch",
                "scenarios": ",".join(scenario_names),
                "days_per_scenario": TOTAL_DAYS,
                "calculation_seconds": elapsed,
                "first_year_start_day": int(YEAR_STARTS[0]),
            }
        ]
    ).to_csv(OUTPUT_DIR / "run_metadata.csv", index=False)

    print(f"NumPy hydrology completed in {elapsed:.3f} seconds.")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
