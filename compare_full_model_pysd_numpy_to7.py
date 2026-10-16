"""Compare all parameter-study outputs for PySD and NumPy hydrology."""

from datetime import datetime
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from pysd import load

from numpy_hydrology_adapter_to7 import NumpyHydrologyAdapter
from run_vensim_with_pysd_to7_para import (
    ASSUMPTION_BOUNDS,
    RETURN_COLS,
    TIME,
    fixed_params_for_scenario,
)


warnings.filterwarnings(
    "ignore", message=r"_ext_data_flow(_d)?", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=r"_ext_data_daily_precip_up", category=UserWarning
)


MODEL_PATH = Path("River_management_xls_to7.py")
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = Path("data/full_model_numpy_validation") / RUN_STAMP


def main():
    params = fixed_params_for_scenario("present")
    params.update({name: values[2] for name, values in ASSUMPTION_BOUNDS.items()})

    reference_model = load(MODEL_PATH.as_posix())
    numpy_model = NumpyHydrologyAdapter(MODEL_PATH, capacity_mode="static")

    started = perf_counter()
    reference = reference_model.run(
        params=params,
        return_timestamps=TIME,
        return_columns=RETURN_COLS,
        initial_condition="original",
    )
    reference_seconds = perf_counter() - started

    started = perf_counter()
    candidate = numpy_model.run(
        params=params,
        return_timestamps=TIME,
        return_columns=RETURN_COLS,
        initial_condition="original",
    )
    numpy_seconds = perf_counter() - started

    rows = []
    for column in RETURN_COLS:
        reference_values = pd.to_numeric(reference[column], errors="coerce").to_numpy()
        candidate_values = pd.to_numeric(candidate[column], errors="coerce").to_numpy()
        valid = np.isfinite(reference_values) & np.isfinite(candidate_values)
        difference = candidate_values[valid] - reference_values[valid]
        scale = np.maximum(np.abs(reference_values[valid]), 1.0)
        rows.append(
            {
                "variable": column,
                "valid_values": int(valid.sum()),
                "max_abs_error": float(np.max(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference**2))),
                "max_relative_error": float(np.max(np.abs(difference) / scale)),
                "allclose_rtol_1e-10_atol_1e-6": bool(
                    np.allclose(
                        candidate_values[valid],
                        reference_values[valid],
                        rtol=1e-10,
                        atol=1e-6,
                    )
                ),
            }
        )

    comparison = pd.DataFrame(rows)
    timing = pd.DataFrame(
        [
            {
                "pysd_run_seconds": reference_seconds,
                "numpy_hydrology_hybrid_seconds": numpy_seconds,
                "full_model_speedup": reference_seconds / numpy_seconds,
            }
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    comparison.to_csv(OUTPUT_DIR / "variable_comparison.csv", index=False)
    timing.to_csv(OUTPUT_DIR / "timing_comparison.csv", index=False)

    failures = comparison[~comparison["allclose_rtol_1e-10_atol_1e-6"]]
    print(comparison.nlargest(10, "max_abs_error").to_string(index=False))
    print("\nTiming:")
    print(timing.to_string(index=False))
    print(f"\nFailed comparisons: {len(failures)} / {len(comparison)}")
    print(f"Saved: {OUTPUT_DIR}")
    if not failures.empty:
        raise AssertionError(
            "The hybrid full-model result differs from the PySD reference."
        )


if __name__ == "__main__":
    main()
