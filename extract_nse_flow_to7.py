"""Extract present/base flow series for NSE recalculation from a to7 daily CSV."""

from pathlib import Path

import pandas as pd


RUN_DATE = "260723_1430"
DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / f"para_to7_all_scenarios_{RUN_DATE}_daily.csv"
OUTPUT_CSV = DATA_DIR / f"para_to7_{RUN_DATE}_present_base_nse_flows.csv"
CHUNK_SIZE = 100_000

USECOLS = [
    "time",
    "year",
    "scenario",
    "case_id",
    "flow",
    "river_discharge_upstream",
    "flow_d",
    "river_discharge_downstream",
]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Daily parameter-study CSV not found: {INPUT_CSV}")

    rows_written = 0
    for chunk in pd.read_csv(INPUT_CSV, usecols=USECOLS, chunksize=CHUNK_SIZE):
        selected = chunk[(chunk["scenario"] == "present") & (chunk["case_id"] == "base")]
        if selected.empty:
            continue

        selected.to_csv(
            OUTPUT_CSV,
            mode="a" if rows_written else "w",
            header=rows_written == 0,
            index=False,
        )
        rows_written += len(selected)

    if rows_written == 0:
        raise ValueError("No rows found for scenario='present' and case_id='base'.")

    print(f"Extracted {rows_written:,} rows to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
