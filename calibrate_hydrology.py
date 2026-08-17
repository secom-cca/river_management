"""Calibrate basin hydrology parameters against upstream/downstream flow.

The first part of the configured period is used for calibration and the final
part for independent validation. A calibrated copy of the basin YAML is written
to results/; the source config is never overwritten automatically.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pysd import load
from scipy.optimize import differential_evolution

from basin_config import ACTIVE_BASIN_ENV, BASE_DIR, load_basin_config, scenario_maps


MODEL_PY = BASE_DIR / "River_management_xls_to6.py"
RETURN_COLUMNS = [
    "river_discharge_upstream",
    "river_discharge_downstream",
    "flow",
    "flow_d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--basin",
        default=os.environ.get(ACTIVE_BASIN_ENV, "chikugo"),
        help="config/basins/<name>.yaml のname、またはYAMLパス",
    )
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="現在のYAML値を1回評価し、最適化せずに終了",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=None,
        help="期間先頭から較正に使う割合。未指定時はYAML値（既定0.7）",
    )
    return parser.parse_args()


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def total_days(config: dict) -> int:
    calendar = config["calendar"]
    if not calendar["use_leap_years"]:
        return 365 * int(calendar["num_years"])
    return sum(
        366 if is_leap_year(int(calendar["start_year"]) + offset) else 365
        for offset in range(int(calendar["num_years"]))
    )


def nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(simulated)
    observed = observed[mask]
    simulated = simulated[mask]
    if observed.size < 2:
        return float("nan")
    denominator = np.sum((observed - observed.mean()) ** 2)
    if denominator <= 0:
        return float("nan")
    return float(1 - np.sum((simulated - observed) ** 2) / denominator)


def score_period(frame: pd.DataFrame, start: int, stop: int) -> dict[str, float]:
    part = frame.iloc[start:stop]
    return {
        "upstream_nse": nse(
            part["flow"].to_numpy(),
            part["river_discharge_upstream"].to_numpy(),
        ),
        "downstream_nse": nse(
            part["flow_d"].to_numpy(),
            part["river_discharge_downstream"].to_numpy(),
        ),
    }


def main() -> None:
    args = parse_args()
    # River_management_xls_to6.py reads this environment variable at load time.
    os.environ[ACTIVE_BASIN_ENV] = args.basin
    config = load_basin_config(args.basin)
    calibration = config["hydrology_calibration"]
    fraction = float(
        args.calibration_fraction
        if args.calibration_fraction is not None
        else calibration.get("calibration_fraction", 0.7)
    )
    if not 0 < fraction < 1:
        raise ValueError("calibration_fraction は0より大きく1未満にしてください")

    missing_inputs = []
    for value in config["inputs"].values():
        path = Path(value)
        resolved = path if path.is_absolute() else BASE_DIR / path
        if not resolved.exists():
            missing_inputs.append(value)
    if missing_inputs:
        raise FileNotFoundError("入力データが見つかりません: " + ", ".join(missing_inputs))

    names = list(calibration["bounds"])
    bounds = [tuple(map(float, calibration["bounds"][name])) for name in names]
    weights = calibration.get(
        "objective_weights", {"upstream_nse": 1.0, "downstream_nse": 1.0}
    )
    days = total_days(config)
    split_index = int(days * fraction)
    timestamps = list(range(days))
    precip_map, temperature_map = scenario_maps(config)
    present = "present" if "present" in precip_map else next(iter(precip_map))

    base_params = dict(config["model_parameters"])
    base_params.update(
        {
            "daily_precipitation_future_ratio": precip_map[present],
            "temperature_scenario_shift": temperature_map[present],
            "levee_investment_amount": 0,
            "dam_investment_amount": 0,
            "drainage_investment_amount": 0,
            "number_of_house_elevation": 0,
            "number_of_migration": 0,
            "annual_paddy_dam_investment": 0,
            "annual_breeding_investment": 0,
        }
    )

    model = load(MODEL_PY.as_posix())
    evaluations = 0

    def run_vector(vector: np.ndarray) -> pd.DataFrame:
        params = base_params.copy()
        params.update(dict(zip(names, map(float, vector))))
        return model.run(
            params=params,
            return_timestamps=timestamps,
            return_columns=RETURN_COLUMNS,
            initial_condition="original",
        )

    def objective(vector: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            metrics = score_period(run_vector(vector), 0, split_index)
            losses = []
            for metric_name in ("upstream_nse", "downstream_nse"):
                value = metrics[metric_name]
                if not np.isfinite(value):
                    return 1e12
                losses.append(float(weights.get(metric_name, 1.0)) * (1 - value))
            score = float(sum(losses))
        except Exception as exc:  # Optimizer candidates can make the model unstable.
            print(f"evaluation {evaluations}: failed ({exc})")
            return 1e12
        print(
            f"evaluation {evaluations}: objective={score:.6g}, "
            f"upstream NSE={metrics['upstream_nse']:.4f}, "
            f"downstream NSE={metrics['downstream_nse']:.4f}"
        )
        return score

    if args.check_only:
        current = np.array([base_params[name] for name in names], dtype=float)
        frame = run_vector(current)
        check_metrics = {
            "calibration": score_period(frame, 0, split_index),
            "validation": score_period(frame, split_index, days),
            "full_period": score_period(frame, 0, days),
        }
        print(yaml.safe_dump(check_metrics, allow_unicode=True, sort_keys=False))
        return

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=True,
        updating="immediate",
        workers=1,
    )

    best_parameters = dict(zip(names, map(float, result.x)))
    best_frame = run_vector(result.x)
    metrics = {
        "calibration": score_period(best_frame, 0, split_index),
        "validation": score_period(best_frame, split_index, days),
        "full_period": score_period(best_frame, 0, days),
    }

    stamp = datetime.now().strftime("%y%m%d_%H%M")
    output_dir = BASE_DIR / "results" / "calibration" / config["key"] / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    best_frame.to_csv(output_dir / "best_daily_flow.csv", index_label="day")

    report = {
        "basin": config["key"],
        "basin_name": config["name"],
        "source_config": str(config["_path"]),
        "calibration_fraction": fraction,
        "split_day": split_index,
        "optimizer": {
            "success": bool(result.success),
            "message": str(result.message),
            "objective": float(result.fun),
            "evaluations": int(result.nfev),
            "iterations": int(result.nit),
            "seed": args.seed,
        },
        "metrics": metrics,
        "best_parameters": best_parameters,
    }
    with (output_dir / "calibration_report.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(report, stream, allow_unicode=True, sort_keys=False)

    calibrated_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    calibrated_config["model_parameters"].update(best_parameters)
    with (output_dir / f"{config['key']}_calibrated.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(calibrated_config, stream, allow_unicode=True, sort_keys=False)

    print(f"Calibration finished: {output_dir}")
    print(yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
