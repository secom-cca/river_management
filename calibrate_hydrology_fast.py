"""calibrate_hydrology.py の高速版。

元スクリプトからの変更点は次の4つ。それ以外の処理（目的関数、NSE の定義、
differential_evolution の設定、出力ファイルの構成）は元のままにしてある。

1. --engine fast
   PySD モデルの水文サブモデルだけを numpy で書き直した FastHydrology
   （River_management_xls_to6_fast.py）を使う。
   PySD 版と全 4383 日・6通りのパラメータで誤差ゼロの一致を確認済み
   （`python River_management_xls_to6_fast.py` で再検証できる）。
   1回の評価が 8.6 秒から 3.3 ミリ秒になる。
   --engine pysd を指定すれば元の PySD 経路で走る。

2. --mask-missing
   観測流量の欠測日を NSE の計算から除外する（阿武隈川では福島 375日、舘矢間 386日）。
   欠測マスクは data/obs_mask_<流域key>.csv を読む。
   モデル入力としては PySD の ExtData が NaN を扱えないため線形補間値を渡すが、
   評価では実測がある日だけを使う。既定は無効（元スクリプトと同じ挙動）。

3. --spinup-days
   先頭 N 日を較正・検証の両方から除外する。下流地下水ストックの初期値が
   平衡値から離れているため、初期条件の影響が消えるまで日数がかかる。
   既定は 0（元スクリプトと同じ挙動）。

4. 診断指標の追加
   NSE は二乗誤差なので高水に支配される。低水の再現性を見るために
   log-NSE、KGE、PBIAS、相関係数を報告に併記する。目的関数は元のまま NSE のみ。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import differential_evolution

from basin_config import ACTIVE_BASIN_ENV, BASE_DIR, load_basin_config, scenario_maps

sys.path.insert(0, str(BASE_DIR))

MODEL_PY = BASE_DIR / "River_management_xls_to6.py"
RETURN_COLUMNS = [
    "river_discharge_upstream",
    "river_discharge_downstream",
    "flow",
    "flow_d",
]
#: 欠測マスク。--mask-missing 指定時のみ読む。列は flow_up / flow_down。
MASK_TEMPLATE = str(BASE_DIR / "data" / "obs_mask_{key}.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basin", default=os.environ.get(ACTIVE_BASIN_ENV, "chikugo"))
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--popsize", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--calibration-fraction", type=float, default=None)
    parser.add_argument("--engine", choices=["fast", "pysd"], default="fast")
    parser.add_argument("--mask-missing", action="store_true")
    parser.add_argument("--spinup-days", type=int, default=0)
    parser.add_argument("--tag", default="")
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


def log_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(simulated) & (observed > 0) & (simulated > 0)
    return nse(np.log(observed[mask]), np.log(simulated[mask]))


def kge(observed: np.ndarray, simulated: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(observed) & np.isfinite(simulated)
    o, s = observed[mask], simulated[mask]
    if o.size < 2 or o.std() == 0 or s.std() == 0:
        return {"kge": float("nan"), "r": float("nan"), "alpha": float("nan"), "beta": float("nan")}
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = float(s.std() / o.std())
    beta = float(s.mean() / o.mean())
    value = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))
    return {"kge": value, "r": r, "alpha": alpha, "beta": beta}


def pbias(observed: np.ndarray, simulated: np.ndarray) -> float:
    mask = np.isfinite(observed) & np.isfinite(simulated)
    o, s = observed[mask], simulated[mask]
    if o.size == 0 or o.sum() == 0:
        return float("nan")
    return float(100.0 * (s.sum() - o.sum()) / o.sum())


def score_period(frame: pd.DataFrame, start: int, stop: int) -> dict[str, float]:
    part = frame.iloc[start:stop]
    return {
        "upstream_nse": nse(
            part["flow"].to_numpy(), part["river_discharge_upstream"].to_numpy()
        ),
        "downstream_nse": nse(
            part["flow_d"].to_numpy(), part["river_discharge_downstream"].to_numpy()
        ),
    }


def diagnostics(frame: pd.DataFrame, start: int, stop: int) -> dict[str, dict[str, float]]:
    part = frame.iloc[start:stop]
    out = {}
    for label, obs_col, sim_col in (
        ("upstream", "flow", "river_discharge_upstream"),
        ("downstream", "flow_d", "river_discharge_downstream"),
    ):
        o = part[obs_col].to_numpy()
        s = part[sim_col].to_numpy()
        entry = {
            "n_days": int(np.sum(np.isfinite(o) & np.isfinite(s))),
            "nse": nse(o, s),
            "log_nse": log_nse(o, s),
            "pbias_percent": pbias(o, s),
            "obs_mean_m3s": float(np.nanmean(o) / 86400.0),
            "sim_mean_m3s": float(np.nanmean(s) / 86400.0),
        }
        entry.update(kge(o, s))
        out[label] = entry
    return out


def main() -> None:
    args = parse_args()
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
    spinup = int(args.spinup_days)
    split_index = int(days * fraction)
    if split_index <= spinup:
        raise ValueError("spinup_days が較正期間より長くなっています")
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

    obs_mask = None
    if args.mask_missing:
        mask_frame = pd.read_csv(MASK_TEMPLATE.format(key=config["key"]))
        obs_mask = {
            "flow": mask_frame["flow_up"].to_numpy(dtype=bool),
            "flow_d": mask_frame["flow_down"].to_numpy(dtype=bool),
        }

    evaluations = 0

    if args.engine == "fast":
        from River_management_xls_to6_fast import FastHydrology

        engine = FastHydrology(args.basin)

        def run_vector(vector: np.ndarray) -> pd.DataFrame:
            result = engine.run(dict(zip(names, map(float, vector))))
            frame = pd.DataFrame(
                {
                    "river_discharge_upstream": result["river_discharge_upstream"],
                    "river_discharge_downstream": result["river_discharge_downstream"],
                    "flow": engine.observed_flow_upstream,
                    "flow_d": engine.observed_flow_downstream,
                }
            )
            return _apply_mask(frame)

    else:
        from pysd import load

        model = load(MODEL_PY.as_posix())

        def run_vector(vector: np.ndarray) -> pd.DataFrame:
            params = base_params.copy()
            params.update(dict(zip(names, map(float, vector))))
            frame = model.run(
                params=params,
                return_timestamps=timestamps,
                return_columns=RETURN_COLUMNS,
                initial_condition="original",
            )
            return _apply_mask(frame.reset_index(drop=True))

    def _apply_mask(frame: pd.DataFrame) -> pd.DataFrame:
        if obs_mask is None:
            return frame
        frame = frame.copy()
        frame.loc[~obs_mask["flow"], "flow"] = np.nan
        frame.loc[~obs_mask["flow_d"], "flow_d"] = np.nan
        return frame

    def objective(vector: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        try:
            metrics = score_period(run_vector(vector), spinup, split_index)
            losses = []
            for metric_name in ("upstream_nse", "downstream_nse"):
                value = metrics[metric_name]
                if not np.isfinite(value):
                    return 1e12
                losses.append(float(weights.get(metric_name, 1.0)) * (1 - value))
            score = float(sum(losses))
        except Exception as exc:
            print(f"evaluation {evaluations}: failed ({exc})")
            return 1e12
        if evaluations % 200 == 0:
            print(
                f"evaluation {evaluations}: objective={score:.6g}, "
                f"upstream NSE={metrics['upstream_nse']:.4f}, "
                f"downstream NSE={metrics['downstream_nse']:.4f}",
                flush=True,
            )
        return score

    if args.check_only:
        current = np.array([base_params[name] for name in names], dtype=float)
        frame = run_vector(current)
        report = {
            "engine": args.engine,
            "mask_missing": bool(args.mask_missing),
            "spinup_days": spinup,
            "split_day": split_index,
            "metrics": {
                "calibration": score_period(frame, spinup, split_index),
                "validation": score_period(frame, split_index, days),
                "full_period": score_period(frame, spinup, days),
            },
            "diagnostics": {
                "calibration": diagnostics(frame, spinup, split_index),
                "validation": diagnostics(frame, split_index, days),
            },
        }
        print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
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
        "calibration": score_period(best_frame, spinup, split_index),
        "validation": score_period(best_frame, split_index, days),
        "full_period": score_period(best_frame, spinup, days),
    }

    stamp = datetime.now().strftime("%y%m%d_%H%M")
    if args.tag:
        stamp = f"{stamp}_{args.tag}"
    output_dir = BASE_DIR / "results" / "calibration" / config["key"] / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    best_frame.to_csv(output_dir / "best_daily_flow.csv", index_label="day")

    report = {
        "basin": config["key"],
        "basin_name": config["name"],
        "source_config": str(config["_path"]),
        "engine": args.engine,
        "mask_missing": bool(args.mask_missing),
        "spinup_days": spinup,
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
        "diagnostics": {
            "calibration": diagnostics(best_frame, spinup, split_index),
            "validation": diagnostics(best_frame, split_index, days),
        },
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
