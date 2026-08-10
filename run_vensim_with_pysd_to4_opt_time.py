from pathlib import Path
import csv
import functools

import numpy as np
from pysd import read_vensim, load

try:
    from scipy.optimize import differential_evolution
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ---- 入力ファイル設定 ----
MODEL_MDL = Path("River_management_xls_to4.mdl")  # Vensim テキスト
MODEL_PY = Path("River_management_xls_to4.py")    # 変換済み PySD モデル

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
                "River_management_xls_to4.py も .mdl も見つかりません。"
            )
    return _MODEL


# ---- シミュレーション設定 ----
CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
USE_LEAP_YEARS = True  # River_management_xls_to4.py と合わせる


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


def _year_index_from_day(day_number):
    idx = int(np.searchsorted(YEAR_STARTS, day_number, side="right") - 1)
    return int(np.clip(idx, 0, len(YEAR_LENGTHS) - 1))


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

# 出力する指標（CSVに書き出すもの）
OUTPUT_INDICATORS = list(INDICATOR_CONFIG.keys())

# ---- 最適化パターン ----
PATTERNS = [
    # {
    #     "name": "dis",
    #     "indicators": [
    #         "financial_damage_by_flood",
    #         "financial_damage_by_innundation",
    #         "landslide_disaster_risk",
    #     ],
    # },
    # {
    #     "name": "crop",
    #     "indicators": [
    #         "crop_production_cashflow",
    #     ],
    # },
    # {
    #     "name": "bio",
    #     "indicators": [
    #         "biodiversity",
    #     ],
    # },
    # {
    #     "name": "money",
    #     "indicators": [
    #         "municipality_cost",
    #         "yearly_gdp_total",
    #     ],
    # },
    # {
    #     "name": "all_but_money",
    #     "indicators": [
    #         "financial_damage_by_flood",
    #         "financial_damage_by_innundation",
    #         "landslide_disaster_risk",
    #         "crop_production_cashflow",
    #         "biodiversity",
    #     ],
    # },
    {
        "name": "all",
        "indicators": [
            "financial_damage_by_flood",
            "financial_damage_by_innundation",
            "landslide_disaster_risk",
            "crop_production_cashflow",
            "biodiversity",
            "municipality_cost",
            "yearly_gdp_total",
        ],
    },
]

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


def _year_time_range(year_index):
    start = YEAR_STARTS[year_index]
    length = YEAR_LENGTHS[year_index]
    return list(range(start, start + length, 1))


def run_model_yearly(params_by_year):
    model = get_model()
    results = []
    for year_index, params in enumerate(params_by_year):
        initial = "original" if year_index == 0 else "current"
        res = model.run(
            params=params,
            return_timestamps=_year_time_range(year_index),
            return_columns=RETURN_COLS,
            initial_condition=initial,
        )
        results.append(res)
    return results


def aggregate_metrics_from_yearly(results):
    metrics = {
        "financial_damage_by_flood": 0.0,
        "financial_damage_by_innundation": 0.0,
        "landslide_disaster_risk": 0.0,
        "crop_production_cashflow": 0.0,
        "municipality_cost": 0.0,
        "yearly_gdp_total": 0.0,
    }
    biodiversity_sum = 0.0
    total_days = 0

    for res in results:
        metrics["financial_damage_by_flood"] += res["financial_damage_by_flood"].sum()
        metrics["financial_damage_by_innundation"] += res[
            "financial_damage_by_innundation"
        ].sum()
        metrics["landslide_disaster_risk"] += res["landslide_disaster_risk"].sum()
        metrics["crop_production_cashflow"] += res["crop_production_cashflow"].sum()
        metrics["municipality_cost"] += res["municipality_cost"].sum()
        metrics["yearly_gdp_total"] += res["daily_total_gdp"].sum()
        biodiversity_sum += res["biodiversity"].sum()
        total_days += len(res.index)

    metrics["biodiversity"] = biodiversity_sum / total_days if total_days else 0.0
    return metrics


def make_scales(base_metrics, selected_indicators):
    scales = {}
    for name in selected_indicators:
        base = abs(base_metrics.get(name, 0))
        scales[name] = max(base, 1.0)
    return scales


def objective_from_metrics(metrics, scales, selected_indicators, weights):
    total = 0.0
    for name in selected_indicators:
        cfg = INDICATOR_CONFIG[name]
        weight = weights.get(name, 1.0)
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


def pack_params_by_year(x):
    params = []
    param_names = list(ADAPTATION_BOUNDS.keys())
    offset = 0
    for _ in range(CALENDAR_NUM_YEARS):
        year_values = x[offset: offset + len(param_names)]
        params.append(dict(zip(param_names, year_values)))
        offset += len(param_names)
    return params


def build_params_by_year(x):
    params = []
    for year_params in pack_params_by_year(x):
        merged = BASE_PARAMS.copy()
        merged.update(year_params)
        params.append(merged)
    return params


def objective_with_scales(x, scales, selected_indicators, weights):
    params_by_year = build_params_by_year(x)
    results = run_model_yearly(params_by_year)
    metrics = aggregate_metrics_from_yearly(results)
    return objective_from_metrics(metrics, scales, selected_indicators, weights)


def optimize(selected_indicators, weights):
    base_params_by_year = [BASE_PARAMS.copy() for _ in range(CALENDAR_NUM_YEARS)]
    base_results = run_model_yearly(base_params_by_year)
    base_metrics = aggregate_metrics_from_yearly(base_results)
    scales = make_scales(base_metrics, selected_indicators)

    bounds = list(ADAPTATION_BOUNDS.values()) * CALENDAR_NUM_YEARS
    iteration_state = {"iter": 0}

    objective = functools.partial(
        objective_with_scales,
        scales=scales,
        selected_indicators=selected_indicators,
        weights=weights,
    )

    def progress_callback(xk, convergence):
        iteration_state["iter"] += 1
        if iteration_state["iter"] % PROGRESS_EVERY == 0:
            current_score = objective(xk)
            print(
                f"[iter {iteration_state['iter']}] best_score={current_score:.6g} "
                f"convergence={convergence:.6g}"
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

    best_params_by_year = build_params_by_year(result.x)
    best_results = run_model_yearly(best_params_by_year)
    best_metrics = aggregate_metrics_from_yearly(best_results)
    return result, best_params_by_year, best_metrics


if __name__ == "__main__":
    combined_rows = []
    combined_rows.append(
        [
            "pattern",
            "score",
            "calendar_start_year",
            "calendar_num_years",
            "use_leap_years",
            "year",
            "parameter",
            "value",
            "indicator",
            "indicator_value",
        ]
    )

    for pattern in PATTERNS:
        selected_indicators = pattern["indicators"]
        weights = {name: 1.0 for name in selected_indicators}
        print(f"\n=== Pattern: {pattern['name']} ===")
        result, best_params_by_year, best_metrics = optimize(
            selected_indicators, weights
        )
        print("Best score:", result.fun)
        print(f"Iterations (nit): {result.nit}")
        print(f"Function evaluations (nfev): {result.nfev}")
        print("Best params by year:")
        for year_index, year_params in enumerate(best_params_by_year):
            year = CALENDAR_START_YEAR + year_index
            print(f"  {year}:")
            for k in ADAPTATION_BOUNDS.keys():
                print(f"    {k}: {year_params.get(k)}")
        print("Metrics:")
        for k in selected_indicators:
            print(f"  {k}: {best_metrics.get(k)}")

        for year_index, year_params in enumerate(best_params_by_year):
            year = CALENDAR_START_YEAR + year_index
            for key in ADAPTATION_BOUNDS.keys():
                combined_rows.append(
                    [
                        pattern["name"],
                        result.fun,
                        CALENDAR_START_YEAR,
                        CALENDAR_NUM_YEARS,
                        USE_LEAP_YEARS,
                        year,
                        key,
                        year_params.get(key),
                        "",
                        "",
                    ]
                )

        for key in OUTPUT_INDICATORS:
            combined_rows.append(
                [
                    pattern["name"],
                    result.fun,
                    CALENDAR_START_YEAR,
                    CALENDAR_NUM_YEARS,
                    USE_LEAP_YEARS,
                    "",
                    "",
                    "",
                    key,
                    best_metrics.get(key),
                ]
            )

    # ---- CSV出力 ----
    output_path = Path("data/opt_result_to4_time_all.csv")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(combined_rows)

    # 指標ごとのCSV出力
    for indicator in OUTPUT_INDICATORS:
        indicator_rows = [combined_rows[0]]
        for row in combined_rows[1:]:
            if row[8] == indicator:
                indicator_rows.append(row)
        indicator_path = Path(f"data/opt_result_to4_time_{indicator}.csv")
        with indicator_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(indicator_rows)
