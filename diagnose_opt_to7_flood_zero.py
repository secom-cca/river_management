"""Diagnose why flood damage is zero in a completed to7 optimisation run."""

from pathlib import Path

import pandas as pd

import run_vensim_with_pysd_to7_opt as opt


# Set explicitly to reproduce a prior run, or leave None for the newest run.
RUN_STAMP = "260723_2017"
DATA_DIR = Path("data")

FLOOD_COLUMNS = [
    "river_discharge_downstream",
    "discharge_allowance",
    "flood_water_amount",
    "flood_water_level",
    "flood_damage_ratio",
    "houses_in_flood_risk",
    "houses_damaged_by_flood",
    "financial_damage_by_flood",
    "levee_investment",
    "levee_level_increase",
]


def input_paths():
    if RUN_STAMP is None:
        candidates = list(DATA_DIR.glob("opt_to7_*_*_decisions.csv"))
        if not candidates:
            raise FileNotFoundError("No optimisation decisions CSV was found in data/.")
        decisions_path = max(candidates, key=lambda path: path.stat().st_mtime)
    else:
        matches = list(DATA_DIR.glob(f"opt_to7_*_{RUN_STAMP}_decisions.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one decisions CSV for run stamp {RUN_STAMP}.")
        decisions_path = matches[0]
    stem = decisions_path.name[: -len("_decisions.csv")]
    return stem, decisions_path


def decision_from_row(row):
    vector = [float(row[name]) for name in opt.DECISION_BOUNDS]
    return opt.decode_decision_vector(vector)


def main():
    stem, decisions_path = input_paths()
    decisions = pd.read_csv(decisions_path, encoding="utf-8-sig")

    # Reuse the optimisation runner's sequential-year setup, but capture the
    # daily flood pathway variables needed to test the zero-damage condition.
    original_columns = list(opt.RETURN_COLS)
    opt.RETURN_COLS[:] = list(dict.fromkeys(original_columns + FLOOD_COLUMNS))
    daily_frames = []
    for _, row in decisions.iterrows():
        scenario = row["scenario"]
        results = opt.run_model_yearly(scenario, decision_from_row(row))
        for year_index, result in enumerate(results):
            frame = result.reset_index(names="time")
            frame.insert(0, "year", opt.CALENDAR_START_YEAR + year_index)
            frame.insert(0, "scenario", scenario)
            daily_frames.append(frame)

    daily = pd.concat(daily_frames, ignore_index=True)
    annual = (
        daily.groupby(["scenario", "year"], as_index=False)
        .agg(
            maximum_river_discharge_downstream=("river_discharge_downstream", "max"),
            minimum_discharge_allowance=("discharge_allowance", "min"),
            maximum_flood_water_amount=("flood_water_amount", "max"),
            maximum_flood_water_level=("flood_water_level", "max"),
            maximum_flood_damage_ratio=("flood_damage_ratio", "max"),
            flood_damage_days=("flood_damage_ratio", lambda values: (values > 0).sum()),
            annual_flood_damage=("financial_damage_by_flood", "sum"),
        )
    )
    summary = (
        daily.groupby("scenario", as_index=False)
        .agg(
            maximum_flood_water_level=("flood_water_level", "max"),
            maximum_flood_damage_ratio=("flood_damage_ratio", "max"),
            flood_damage_days=("flood_damage_ratio", lambda values: (values > 0).sum()),
            total_flood_damage=("financial_damage_by_flood", "sum"),
        )
    )

    daily.to_csv(DATA_DIR / f"{stem}_flood_diagnostics_daily.csv", index=False, encoding="utf-8-sig")
    annual.to_csv(DATA_DIR / f"{stem}_flood_diagnostics_annual.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(DATA_DIR / f"{stem}_flood_diagnostics_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Saved flood diagnostics to: {DATA_DIR / stem}")


if __name__ == "__main__":
    main()
