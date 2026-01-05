from pathlib import Path
import csv
import functools
import math

import numpy as np
from pysd import read_vensim, load

try:
    from scipy.optimize import differential_evolution
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---- 入力ファイル設定 ----
MODEL_MDL = Path("River_management_xls_to3.mdl")  # Vensim テキスト
MODEL_PY = Path("River_management_xls_to3.py")    # 変換済み PySD モデル

_MODEL = None


def get_model():
    """
    マルチプロセス対応のため、モデルは遅延ロードする。
    """
    global _MODEL
    if _MODEL is None:
        if MODEL_PY.exists():
            _MODEL = load(MODEL_PY.as_posix())
        elif MODEL_MDL.exists():
            _MODEL = read_vensim(MODEL_MDL.as_posix())
        else:
            raise FileNotFoundError(
                "River_management_xls_to3.py も .mdl も見つかりません。"
            )
    return _MODEL

# ---- シミュレーション設定 ----
SIM_YEARS = 1  # 複数年評価したい場合は増やす
time = list(range(0, 365 * SIM_YEARS, 1))

# ---- 進捗表示設定 ----
PROGRESS_EVERY = 5  # 何イテレーションごとに表示するか

# ---- 最適化対象の適応策（スネークケース）----
ADAPTATION_BOUNDS = {
    "drainage_investment_amount": (0, 10_000_000_000),
    "dam_investment_amount": (0, 10_000_000_000),
    "levee_investment_amount": (0, 10_000_000_000),
    "number_of_house_elevation": (0, 1000),
    "number_of_migration": (0, 1000),
    "number_of_planting_trees": (0, 100_000),
    "annual_paddy_dam_investment": (0, 10_000_000_000),
}

# ---- 目的指標 ----
INDICATOR_CONFIG = {
    "financial_damage_by_flood": {"goal": "minimize"},
    "financial_damage_by_innundation": {"goal": "minimize"},
    "landslide_disaster_risk": {"goal": "minimize"},
    "crop_production_cashflow": {"goal": "target", "target": 1_000_000_000_000},
    "biodiversity": {"goal": "target", "target": 1.0},
    "municipality_cost": {"goal": "minimize"},
    "yearly_gdp_total": {"goal": "maximize"},
}

# 最適化に使う指標（目的関数に含めるもの）
SELECTED_INDICATORS = [
    #"financial_damage_by_flood",
    #"financial_damage_by_innundation",
    #"landslide_disaster_risk",
    #"crop_production_cashflow",
    #"biodiversity",
    #"municipality_cost",
    "yearly_gdp_total",
]

# 出力する指標（CSVに書き出すもの）
OUTPUT_INDICATORS = list(INDICATOR_CONFIG.keys())

WEIGHTS = {name: 1.0 for name in SELECTED_INDICATORS}

# ---- ベースパラメータ ----
BASE_PARAMS = {
    "daily_precipitation_future_ratio": 1,
    "levee_investment_amount": 0,
    "dam_investment_amount": 0,
}

# ---- 取得したい出力 ----
RETURN_COLS = [
    "financial_damage_by_flood",
    "financial_damage_by_innundation",
    "landslide_disaster_risk",
    "crop_production_cashflow",
    "biodiversity",
    "municipality_cost",
    "daily_total_gdp",
]


def run_model(params):
    return get_model().run(
        params=params,
        return_timestamps=time,
        return_columns=RETURN_COLS,
        initial_condition="original",
    )


def aggregate_metrics(res):
    df = res.copy()
    df["year"] = (df.index // 365).astype(int)

    metrics = {}
    metrics["financial_damage_by_flood"] = (
        df.groupby("year")["financial_damage_by_flood"].sum().sum()
    )
    metrics["financial_damage_by_innundation"] = (
        df.groupby("year")["financial_damage_by_innundation"].sum().sum()
    )
    metrics["landslide_disaster_risk"] = (
        df.groupby("year")["landslide_disaster_risk"].sum().sum()
    )
    metrics["crop_production_cashflow"] = (
        df.groupby("year")["crop_production_cashflow"].sum().sum()
    )
    metrics["biodiversity"] = df["biodiversity"].mean()
    metrics["municipality_cost"] = (
        df.groupby("year")["municipality_cost"].sum().sum()
    )
    metrics["yearly_gdp_total"] = df.groupby("year")["daily_total_gdp"].sum().sum()
    return metrics


def make_scales(base_metrics):
    scales = {}
    for name in SELECTED_INDICATORS:
        base = abs(base_metrics.get(name, 0))
        scales[name] = max(base, 1.0)
    return scales


def objective_from_metrics(metrics, scales):
    total = 0.0
    for name in SELECTED_INDICATORS:
        cfg = INDICATOR_CONFIG[name]
        weight = WEIGHTS.get(name, 1.0)
        goal = cfg["goal"]
        scale = scales.get(name, 1.0)
        value = metrics.get(name, 0.0)

        if goal == "minimize":
            loss = value / scale
        elif goal == "target":
            target = cfg.get("target", 0.0)
            loss = abs(value - target) / scale
        elif goal == "maximize":
            loss = -value / scale
        else:
            raise ValueError(f"Unknown goal: {goal}")

        total += weight * loss
    return total


def pack_params(x):
    return dict(zip(ADAPTATION_BOUNDS.keys(), x))


def build_params(x):
    params = BASE_PARAMS.copy()
    params.update(pack_params(x))
    return params


def objective_with_scales(x, scales):
    params = build_params(x)
    res = run_model(params)
    metrics = aggregate_metrics(res)
    return objective_from_metrics(metrics, scales)


def optimize():
    base_res = run_model(BASE_PARAMS)
    base_metrics = aggregate_metrics(base_res)
    scales = make_scales(base_metrics)

    bounds = list(ADAPTATION_BOUNDS.values())
    iteration_state = {"iter": 0}

    objective = functools.partial(objective_with_scales, scales=scales)

    def progress_callback(xk, convergence):
        iteration_state["iter"] += 1
        if iteration_state["iter"] % PROGRESS_EVERY == 0:
            current_score = objective(xk)
            best_params = pack_params(xk)
            params_str = ", ".join(
                f"{k}={v:.6g}" for k, v in best_params.items()
            )
            print(
                f"[iter {iteration_state['iter']}] best_score={current_score:.6g} "
                f"convergence={convergence:.6g} params: {params_str}"
            )
        return False

    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "scipy が見つかりません。scipy を入れるか、簡易探索版に切り替えてください。"
        )

    result = differential_evolution(
        objective,
        bounds,
        maxiter=60,
        popsize=15,
        polish=True,
        seed=42,
        callback=progress_callback,
        workers=8,
        updating="deferred",
    )

    best_params = build_params(result.x)
    best_res = run_model(best_params)
    best_metrics = aggregate_metrics(best_res)
    return result, best_params, best_metrics


if __name__ == "__main__":
    result, best_params, best_metrics = optimize()
    print("Best score:", result.fun)
    print(f"Iterations (nit): {result.nit}")
    print(f"Function evaluations (nfev): {result.nfev}")
    print("Best params:")
    for k, v in best_params.items():
        if k in ADAPTATION_BOUNDS:
            print(f"  {k}: {v}")
    print("Metrics:")
    for k in SELECTED_INDICATORS:
        print(f"  {k}: {best_metrics.get(k)}")

    # ---- CSV出力 ----
    output_path = Path("data/opt_result_to3_gdp.csv")
    rows = []
    rows.append(["score", result.fun])
    rows.append(["sim_years", SIM_YEARS])
    rows.append([])
    rows.append(["parameter", "value"])
    for key in ADAPTATION_BOUNDS.keys():
        rows.append([key, best_params.get(key)])
    rows.append([])
    rows.append(["indicator", "value"])
    for key in OUTPUT_INDICATORS:
        rows.append([key, best_metrics.get(key)])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
