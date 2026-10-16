"""Shared configuration and diagnostics for the to7 NumPy hydrology runners."""

from pathlib import Path

import numpy as np
import pandas as pd

from River_management_xls_to7_numpy import DEFAULT_PARAMETERS, TOTAL_DAYS


CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
WARMUP_DAYS = 90

SCENARIOS = {
    "present": {
        "daily_precipitation_future_ratio": 1.0,
        "temperature_scenario_shift": 0.0,
    },
    "2C": {
        "daily_precipitation_future_ratio": 1.1,
        "temperature_scenario_shift": 1.3,
    },
    "4C": {
        "daily_precipitation_future_ratio": 1.3,
        "temperature_scenario_shift": 4.1,
    },
}

CALIBRATED_HYDROLOGY_PARAMETERS = {
    "upstream_outflow_ratio": 0.272198,
    "downstream_outflow_ratio": 0.240634,
    "direct_discharge_ratio": 0.815126,
    "upstream_percolation_ratio": 0.5,
    "downstream_deep_percolation_ratio": 0.0263544,
    "upstream_middle_flow_ratio": 0.443267,
    "downstream_percolation_ratio": 0.5,
    "downstream_middle_flow_ratio": 0.0326612,
    "upstream_deep_percolation_ratio": 0.540584,
}

# Only assumptions that enter the hydrology equations are included here. The
# crop-price, GDP and damaged-paddy assumptions in the full-model LHS do not.
HYDROLOGY_ASSUMPTION_BOUNDS = {
    "forest_area_ratio": (0.85, 0.95, 0.92),
    "paddy_field_ratio": (0.10, 0.20, 0.15),
    "innundation_risky_area_ratio": (0.5, 0.8, 0.8),
    "waterholding_capacity_of_forest_base": (200, 250, 225),
    "paddy_field_capacity_per_area": (400, 600, 500),
    "paddy_dam_capacity_per_area": (1000, 2000, 1500),
    "unmanaged_plantation_forest_coef": (0.5, 0.9, 0.7),
}


def _is_leap_year(year):
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


YEAR_LENGTHS = np.asarray(
    [
        366 if _is_leap_year(CALENDAR_START_YEAR + offset) else 365
        for offset in range(CALENDAR_NUM_YEARS)
    ],
    dtype=int,
)
YEAR_STARTS = np.cumsum(np.r_[0, YEAR_LENGTHS[:-1]])
YEAR_INDEX = np.repeat(np.arange(CALENDAR_NUM_YEARS), YEAR_LENGTHS)

if len(YEAR_INDEX) != TOTAL_DAYS:
    raise RuntimeError("NumPy hydrology calendar is inconsistent with TOTAL_DAYS.")


def parameters_for_scenario(scenario, overrides=None):
    """Return the calibrated base parameters for one climate scenario."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {scenario}")
    parameters = DEFAULT_PARAMETERS.copy()
    parameters.update(CALIBRATED_HYDROLOGY_PARAMETERS)
    parameters.update(SCENARIOS[scenario])
    if overrides:
        parameters.update(overrides)
    return parameters


def _read_observed_column(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "flow" not in frame:
        raise KeyError(f"Column 'flow' was not found in {path}.")
    values = pd.to_numeric(frame["flow"], errors="coerce")
    values = values.interpolate(limit_direction="both")
    array = values.to_numpy(dtype=float)
    if len(array) < TOTAL_DAYS:
        raise ValueError(
            f"Observed flow in {path} has {len(array)} rows; "
            f"{TOTAL_DAYS} are required."
        )
    return array[:TOTAL_DAYS]


def load_observed_flows(model_root=None):
    """Load observed Arase and Senoshita flows used by the PySD model."""
    root = Path(model_root or Path(__file__).parent).resolve()
    return {
        "flow": _read_observed_column(
            root / "data" / "flow_arase_2009_2023_x100000_0to364_utf8.csv"
        ),
        "flow_d": _read_observed_column(
            root / "data" / "flow_senoshita_2009_2023_x100000_0to364_utf8.csv"
        ),
    }


def nse(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(simulated)
    observed = observed[valid]
    simulated = simulated[valid]
    denominator = np.sum((observed - observed.mean()) ** 2)
    if len(observed) < 2 or denominator <= 0:
        return np.nan
    return float(1.0 - np.sum((simulated - observed) ** 2) / denominator)


def performance_summary(observed, simulated, location, warmup_days=WARMUP_DAYS):
    """Calculate NSE and supporting diagnostics after the warm-up period."""
    observed = np.asarray(observed, dtype=float)[warmup_days:]
    simulated = np.asarray(simulated, dtype=float)[warmup_days:]
    valid = np.isfinite(observed) & np.isfinite(simulated)
    observed = observed[valid]
    simulated = simulated[valid]
    if len(observed) < 2:
        raise ValueError(f"Insufficient valid data for {location} diagnostics.")
    error = simulated - observed
    return {
        "location": location,
        "warmup_days_excluded": warmup_days,
        "nse": nse(observed, simulated),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pbias_percent": float(100.0 * error.sum() / observed.sum()),
        "pearson_r": float(np.corrcoef(observed, simulated)[0, 1]),
        "valid_days": len(observed),
        "skipped_days": TOTAL_DAYS - warmup_days - len(observed),
        "observed_mean": float(observed.mean()),
        "simulated_mean": float(simulated.mean()),
    }


def latin_hypercube_samples(rng, sample_count, parameter_names):
    """Generate a dependency-free Latin hypercube on the unit interval."""
    samples = np.empty((sample_count, len(parameter_names)))
    for column, _name in enumerate(parameter_names):
        strata = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        samples[:, column] = rng.permutation(strata)
    return samples
