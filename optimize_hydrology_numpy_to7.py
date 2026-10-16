"""Trial calibration of the nine to7 hydrology ratios using NumPy batches."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from River_management_xls_to7_numpy import (
    TOTAL_DAYS,
    load_weather,
    simulate_hydrology_batch,
)
from hydrology_numpy_to7_common import (
    CALIBRATED_HYDROLOGY_PARAMETERS,
    WARMUP_DAYS,
    load_observed_flows,
    parameters_for_scenario,
)


# Trial settings. Increase MAXITER and use multiple RANDOM_SEED values before
# treating a solution as a final calibration result.
MAXITER = 20
POPSIZE = 8
TOLERANCE = 0.001
RANDOM_SEED = 42
POLISH = False
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = Path("data/hydrology_numpy_calibration_to7") / RUN_STAMP

CALIBRATION_BOUNDS = {
    # These reproduce the 0.01--0.99 ranges in the existing Vensim .voc file.
    "upstream_outflow_ratio": (0.01, 0.99),
    "downstream_outflow_ratio": (0.01, 0.99),
    "direct_discharge_ratio": (0.01, 0.99),
    "upstream_percolation_ratio": (0.01, 0.99),
    "downstream_deep_percolation_ratio": (0.01, 0.99),
    "upstream_middle_flow_ratio": (0.01, 0.99),
    "downstream_percolation_ratio": (0.01, 0.99),
    "downstream_middle_flow_ratio": (0.01, 0.99),
    "upstream_deep_percolation_ratio": (0.01, 0.99),
}

_WEATHER = None
_OBSERVED = None
_HISTORY = []
_STARTED = None


def calibration_inputs():
    global _WEATHER, _OBSERVED
    if _WEATHER is None:
        _WEATHER = load_weather(number_of_days=TOTAL_DAYS)
    if _OBSERVED is None:
        _OBSERVED = load_observed_flows()
    return _WEATHER, _OBSERVED


def objective(candidate_matrix):
    """Return equal-weight upstream/downstream NSE loss for a candidate batch."""
    names = list(CALIBRATION_BOUNDS)
    candidates = np.asarray(candidate_matrix, dtype=float)
    single = candidates.ndim == 1
    if single:
        candidates = candidates.reshape(1, -1)
    elif candidates.shape[0] == len(names):
        # scipy passes vectorized candidates as (parameters, candidates).
        candidates = candidates.T
    if candidates.shape[1] != len(names):
        raise ValueError(f"Unexpected candidate shape: {candidates.shape}")

    parameter_sets = []
    for row in candidates:
        overrides = dict(zip(names, row))
        parameter_sets.append(parameters_for_scenario("present", overrides))
    weather, observed = calibration_inputs()
    result = simulate_hydrology_batch(
        parameter_sets,
        weather=weather,
        output_columns=[
            "river_discharge_upstream",
            "river_discharge_downstream",
        ],
    )

    warm = slice(WARMUP_DAYS, None)
    observed_up = observed["flow"][warm]
    observed_down = observed["flow_d"][warm]
    denominator_up = np.sum((observed_up - observed_up.mean()) ** 2)
    denominator_down = np.sum((observed_down - observed_down.mean()) ** 2)
    nse_up = 1.0 - np.sum(
        (result["river_discharge_upstream"][:, warm] - observed_up) ** 2,
        axis=1,
    ) / denominator_up
    nse_down = 1.0 - np.sum(
        (result["river_discharge_downstream"][:, warm] - observed_down) ** 2,
        axis=1,
    ) / denominator_down
    losses = ((1.0 - nse_up) + (1.0 - nse_down)) / 2.0
    losses[~np.isfinite(losses)] = 1e12
    return float(losses[0]) if single else losses


def progress(best_parameters, convergence):
    generation = len(_HISTORY) + 1
    best_objective = objective(best_parameters)
    elapsed_minutes = (perf_counter() - _STARTED) / 60.0
    _HISTORY.append(
        {
            "generation": generation,
            "best_objective": best_objective,
            "convergence": convergence,
            "elapsed_minutes": elapsed_minutes,
        }
    )
    print(
        f"generation={generation}/{MAXITER}, objective={best_objective:.6g}, "
        f"convergence={convergence:.4g}, elapsed={elapsed_minutes:.1f} min"
    )
    return False


def result_metrics(parameters):
    weather, observed = calibration_inputs()
    result = simulate_hydrology_batch(
        [parameters_for_scenario("present", parameters)],
        weather=weather,
        output_columns=[
            "river_discharge_upstream",
            "river_discharge_downstream",
        ],
    )
    warm = slice(WARMUP_DAYS, None)
    rows = []
    for location, observed_name, simulated_name in (
        ("upstream", "flow", "river_discharge_upstream"),
        ("downstream", "flow_d", "river_discharge_downstream"),
    ):
        obs = observed[observed_name][warm]
        sim = result[simulated_name][0, warm]
        denominator = np.sum((obs - obs.mean()) ** 2)
        rows.append(
            {
                "location": location,
                "nse": 1.0 - np.sum((sim - obs) ** 2) / denominator,
                "rmse": np.sqrt(np.mean((sim - obs) ** 2)),
                "pbias_percent": 100.0 * np.sum(sim - obs) / np.sum(obs),
            }
        )
    return rows


def main():
    global _STARTED
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    calibration_inputs()

    names = list(CALIBRATION_BOUNDS)
    current = np.asarray([CALIBRATED_HYDROLOGY_PARAMETERS[name] for name in names])
    _STARTED = perf_counter()
    result = differential_evolution(
        objective,
        bounds=list(CALIBRATION_BOUNDS.values()),
        strategy="best1bin",
        maxiter=MAXITER,
        popsize=POPSIZE,
        tol=TOLERANCE,
        seed=RANDOM_SEED,
        polish=POLISH,
        updating="deferred",
        workers=1,
        vectorized=True,
        callback=progress,
        x0=current,
    )
    elapsed = perf_counter() - _STARTED
    optimized = dict(zip(names, result.x))

    parameter_frame = pd.DataFrame(
        {
            "parameter": names,
            "previous_calibrated_value": current,
            "optimized_value": result.x,
            "lower_bound": [CALIBRATION_BOUNDS[name][0] for name in names],
            "upper_bound": [CALIBRATION_BOUNDS[name][1] for name in names],
        }
    )
    metric_frame = pd.DataFrame(
        [
            {"parameter_set": "previous", **row}
            for row in result_metrics(CALIBRATED_HYDROLOGY_PARAMETERS)
        ]
        + [
            {"parameter_set": "optimized", **row}
            for row in result_metrics(optimized)
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "success": result.success,
                "message": str(result.message),
                "objective": result.fun,
                "generations": result.nit,
                "evaluations_reported_by_scipy": result.nfev,
                "candidate_evaluations": (
                    POPSIZE * len(names) * (result.nit + 1)
                ),
                "population_size": POPSIZE * len(names),
                "candidate_vectorization": True,
                "elapsed_seconds": elapsed,
                "seed": RANDOM_SEED,
            }
        ]
    )
    parameter_frame.to_csv(OUTPUT_DIR / "optimized_parameters.csv", index=False)
    metric_frame.to_csv(OUTPUT_DIR / "nse_comparison.csv", index=False)
    pd.DataFrame(_HISTORY).to_csv(OUTPUT_DIR / "generation_history.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "optimization_summary.csv", index=False)
    print(metric_frame.to_string(index=False))
    print(f"Saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
