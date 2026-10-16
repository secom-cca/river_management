"""Calculate current to7 upstream/downstream NSE with NumPy hydrology."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from River_management_xls_to7_numpy import (
    TOTAL_DAYS,
    load_weather,
    simulate_hydrology,
)
from hydrology_numpy_to7_common import (
    YEAR_INDEX,
    load_observed_flows,
    parameters_for_scenario,
    performance_summary,
)


RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = Path("data/nse_to7_numpy") / RUN_STAMP


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    weather = load_weather(number_of_days=TOTAL_DAYS)
    observed = load_observed_flows()
    started = perf_counter()
    result = simulate_hydrology(
        weather=weather,
        params=parameters_for_scenario("present"),
    )
    elapsed = perf_counter() - started

    flows = pd.DataFrame(
        {
            "time": np.arange(TOTAL_DAYS, dtype=int),
            "year": YEAR_INDEX,
            "scenario": "present",
            "case_id": "base",
            "flow": observed["flow"],
            "river_discharge_upstream": result["river_discharge_upstream"].to_numpy(),
            "flow_d": observed["flow_d"],
            "river_discharge_downstream": result[
                "river_discharge_downstream"
            ].to_numpy(),
        }
    )
    summaries = pd.DataFrame(
        [
            performance_summary(
                flows["flow"], flows["river_discharge_upstream"], "upstream"
            ),
            performance_summary(
                flows["flow_d"], flows["river_discharge_downstream"], "downstream"
            ),
        ]
    )
    summaries.insert(0, "engine", "NumPy")
    summaries["calculation_seconds"] = elapsed

    flows.to_csv(OUTPUT_DIR / "present_base_nse_flows.csv", index=False)
    summaries.to_csv(OUTPUT_DIR / "nse_summary.csv", index=False)
    print(summaries.to_string(index=False))
    print(f"Saved NSE results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
