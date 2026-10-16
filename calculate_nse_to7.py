"""Run the current to7 base model and calculate upstream/downstream NSE."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_vensim_with_pysd_to7_para import (
    ASSUMPTION_BOUNDS,
    TIME,
    _year_index_from_day,
    fixed_params_for_scenario,
    get_model,
)


# Initial storage is set from an approximately 90-day residence-time assumption.
WARMUP_DAYS = 90
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = Path("data/nse_to7") / RUN_STAMP
FLOW_COLUMNS = [
    "flow",
    "river_discharge_upstream",
    "flow_d",
    "river_discharge_downstream",
]


def performance_summary(frame, location, observed_column, simulated_column):
    paired = frame[[observed_column, simulated_column]].apply(
        pd.to_numeric, errors="coerce"
    )
    paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
    observed = paired[observed_column].to_numpy(dtype=float)
    simulated = paired[simulated_column].to_numpy(dtype=float)
    if len(observed) < 2:
        raise ValueError(f"Insufficient valid data for {location} NSE.")

    error = simulated - observed
    observed_variance_sum = np.sum((observed - observed.mean()) ** 2)
    if observed_variance_sum <= 0:
        raise ValueError(f"Observed flow has no variance for {location} NSE.")

    return {
        "location": location,
        "warmup_days_excluded": WARMUP_DAYS,
        "nse": 1 - np.sum(error**2) / observed_variance_sum,
        "rmse": np.sqrt(np.mean(error**2)),
        "pbias_percent": 100 * error.sum() / observed.sum(),
        "pearson_r": np.corrcoef(observed, simulated)[0, 1],
        "valid_days": len(observed),
        "skipped_days": len(frame) - len(observed),
        "observed_mean": observed.mean(),
        "simulated_mean": simulated.mean(),
    }


def main():
    params = fixed_params_for_scenario("present")
    params.update({name: values[2] for name, values in ASSUMPTION_BOUNDS.items()})

    print("Running current to7 present/base simulation...")
    result = get_model().run(
        params=params,
        return_timestamps=TIME,
        return_columns=FLOW_COLUMNS,
        initial_condition="original",
    )
    flows = result.copy()
    flows.index.name = "time"
    flows = flows.reset_index()
    flows.insert(1, "year", [_year_index_from_day(day) for day in flows["time"]])
    flows.insert(2, "scenario", "present")
    flows.insert(3, "case_id", "base")

    evaluated = flows[flows["time"] >= WARMUP_DAYS]
    summaries = pd.DataFrame(
        [
            performance_summary(
                evaluated, "upstream", "flow", "river_discharge_upstream"
            ),
            performance_summary(
                evaluated, "downstream", "flow_d", "river_discharge_downstream"
            ),
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    flows.to_csv(OUTPUT_DIR / "present_base_nse_flows.csv", index=False)
    summaries.to_csv(OUTPUT_DIR / "nse_summary.csv", index=False)

    for summary in summaries.to_dict("records"):
        print(
            f"{summary['location']}: NSE={summary['nse']:.4f}, "
            f"RMSE={summary['rmse']:.2f}, PBIAS={summary['pbias_percent']:.2f}%, "
            f"r={summary['pearson_r']:.4f}, valid days={summary['valid_days']}"
        )
    print(f"Saved NSE results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
