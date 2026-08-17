"""Multi-objective adaptation optimisation for the to7 study configuration.

The physical and socioeconomic assumptions are fixed at their base values. Each
climate scenario is optimised separately. Outcomes are evaluated relative to
the no-additional-policy baseline of the same scenario, rather than by treating
the modelled damage totals as observed absolute monetary values.

Flood and inner-flood damages remain additive when they occur on the same day.
This represents compounded housing damage for households exposed to both hazards;
the model deliberately does not apply an overlap discount to these loss terms.

Policy cost is not a primary objective. A 0.001-weighted total-budget term is
used only to break otherwise equivalent solutions in favour of lower cost.
"""

from datetime import datetime
from pathlib import Path
import functools
import json
import warnings

import numpy as np
import pandas as pd
from pysd import load

from basin_config import load_active_basin, scenario_maps

try:
    from scipy.optimize import differential_evolution
except ImportError as exc:
    raise RuntimeError("This script requires scipy: pip install scipy") from exc


# PySD emits this warning for the first point of the two validated observed-flow
# CSVs even though their flow column is populated. Observed flow is not used by
# the optimisation objective, and all other external-data warnings stay visible.
warnings.filterwarnings(
    "ignore",
    # `filterwarnings` uses a start-anchored match. Matching this internal
    # element name works reliably in Windows spawned worker processes.
    message=r"_ext_data_flow(_d)?",
    category=UserWarning,
)
# The model deliberately interpolates the initial daily precipitation point
# from the Hita weather workbook. Suppress only that known one-cell warning.
warnings.filterwarnings(
    "ignore",
    message=r"_ext_data_daily_precip_up",
    category=UserWarning,
)


# ---- Model and simulation settings ----
BASE_DIR = Path(__file__).resolve().parent
MODEL_PY = BASE_DIR / "River_management_xls_to6.py"
BASIN_CONFIG = load_active_basin()
BASIN_KEY = BASIN_CONFIG["key"]
OUTPUT_DIR = BASE_DIR / "results" / "optimization" / BASIN_KEY
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M")

CALENDAR_START_YEAR = int(BASIN_CONFIG["calendar"]["start_year"])
CALENDAR_NUM_YEARS = int(BASIN_CONFIG["calendar"]["num_years"])
USE_LEAP_YEARS = bool(BASIN_CONFIG["calendar"]["use_leap_years"])

#SCENARIOS = ["present"]
SCENARIOS = [
    name for name in ("2C", "4C") if name in BASIN_CONFIG["climate_scenarios"]
]
if not SCENARIOS:
    SCENARIOS = list(BASIN_CONFIG["climate_scenarios"])

# Present-climate exploration across all objective patterns. Restore all climate
# scenarios after confirming convergence and policy-pattern behaviour.
MAXITER = 120
POPSIZE = 8
# Require a tighter population-objective spread before declaring convergence.
# This lets the small policy-cost tie-breaker distinguish near-equal solutions.
DE_TOLERANCE = 0.001

# Keep one seed for an exploratory run. For final analysis, for example use
# [42, 43, 44] and compare both objective values and policy decisions.
OPTIMIZATION_SEEDS = [42]

# Print each completed DE generation so long PySD runs visibly advance in the
# PowerShell console. Set this higher only when console output is excessive.
PROGRESS_EVERY = 1
MAX_WORKERS = 12 #10でCPU消費50%弱
BREEDING_ANNUAL_COST = 50_000_000
BREEDING_DURATION_YEARS = 10
# Set to a checkpoint directory to seed a new run from its final populations.
# This starts a new DE run; SciPy does not support an exact generation-by-
# generation continuation with its public API.
RESUME_CHECKPOINT_DIR = None

# The policy budget excludes management of the initially managed plantation.
BUDGET_CASE = "standard"
BUDGET_CASES = {
    "constrained": {"annual": 2_000_000_000, "total": 20_000_000_000},
    "standard": {"annual": 5_000_000_000, "total": 50_000_000_000},
    "expanded": {"annual": 10_000_000_000, "total": 100_000_000_000},
}
# Objective values are order-one ratios, so a large value makes the budget a
# practical hard constraint rather than a sixth objective.
BUDGET_PENALTY = 1_000_000.0
# A secondary tie-breaker only. Even moving from zero to full budget use changes
# the objective by at most 0.001, so it cannot replace a material improvement
# in the primary outcome ratios.
COST_TIEBREAKER_WEIGHT = 1e-3

SCENARIO_TO_PRECIP_RATIO, SCENARIO_TO_TEMP_SHIFT = scenario_maps(BASIN_CONFIG)

# Regional, socioeconomic, and calibrated hydrological values are maintained in
# one basin config and shared with the parameter-study and calibration scripts.
FIXED_ASSUMPTIONS = dict(BASIN_CONFIG["model_parameters"])

# A compact policy representation avoids optimising a separate value for every
# year. Start/end years are rounded to calendar-year indices when evaluated.
DECISION_BOUNDS = {
    "drainage_investment_amount": (0, 5_000_000_000),
    "drainage_start_year": (0, CALENDAR_NUM_YEARS - 1),
    "dam_investment_amount": (0, 5_000_000_000),
    "dam_start_year": (0, CALENDAR_NUM_YEARS - 1),
    "levee_investment_amount": (0, 5_000_000_000),
    "levee_start_year": (0, CALENDAR_NUM_YEARS - 1),
    "house_elevation_per_year": (0, 1000),
    "house_elevation_start_year": (0, CALENDAR_NUM_YEARS - 1),
    "house_elevation_end_year": (0, CALENDAR_NUM_YEARS - 1),
    "migration_per_year": (0, 1000),
    "migration_start_year": (0, CALENDAR_NUM_YEARS - 1),
    "migration_end_year": (0, CALENDAR_NUM_YEARS - 1),
    "annual_paddy_dam_investment": (0, 5_000_000_000),
    "paddy_dam_start_year": (0, CALENDAR_NUM_YEARS - 1),
    "paddy_dam_end_year": (0, CALENDAR_NUM_YEARS - 1),
    "breeding_enabled": (0, 1),
    # Breeding is either not undertaken or funded continuously for ten years.
    "breeding_start_year": (0, CALENDAR_NUM_YEARS - BREEDING_DURATION_YEARS),
    "annual_forest_management_conversion_area": (0, 3_000),
    "forest_management_start_year": (0, CALENDAR_NUM_YEARS - 1),
    "forest_management_end_year": (0, CALENDAR_NUM_YEARS - 1),
}

# A weighted-sum run is a reproducible initial approximation of a Pareto search.
# Additional weight patterns can be appended to obtain more points on the front.
OBJECTIVE_PATTERNS = {
    "balanced": {
        "flood_damage": 1.0,
        "innundation_damage": 1.0,
        "landslide_risk": 1.0,
        "crop_revenue": 1.0,
        "biodiversity": 1.0,
    },
    "disaster_only": {
        "flood_damage": 1.0,
        "innundation_damage": 1.0,
        "landslide_risk": 1.0,
        "crop_revenue": 0.0,
        "biodiversity": 0.0,
    },
    "agriculture_only": {
        "flood_damage": 0.0,
        "innundation_damage": 0.0,
        "landslide_risk": 0.0,
        "crop_revenue": 1.0,
        "biodiversity": 0.0,
    },
    "ecosystem_only": {
        "flood_damage": 0.0,
        "innundation_damage": 0.0,
        "landslide_risk": 0.0,
        "crop_revenue": 0.0,
        "biodiversity": 1.0,
    },
}

RETURN_COLS = [
    "financial_damage_by_flood",
    "financial_damage_by_innundation",
    "landslide_disaster_risk",
    "crop_production_cashflow",
    "biodiversity",
    "daily_total_gdp",
    "forest_function_coef",
    "forest_mitigation",
    "scenario_adjusted_daily_precip_up",
    "landslide_design_daily_precipitation",
    "houses_in_flood_risk",
    "houses_in_inundation_risk",
    "flood_only_risk_houses",
    "innundation_only_risk_houses",
    "overlapping_risk_houses",
    "houses_in_nonrisky_area",
    "elevated_houses",
    "managed_plantation_forest_area",
    "unmanaged_plantation_forest_area",
    "additional_forest_management_cost",
    "paddy_dam_investment",
    "paddy_dam_area",
    "paddy_dam_ratio",
    "breeding_investment",
    "accumulated_breeding_investment",
    "breeding_success",
]

UNIT_COST_OF_ELEVATION = 3_000_000
UNIT_COST_OF_MIGRATION = 3_000_000

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        if not MODEL_PY.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PY}")
        _MODEL = load(MODEL_PY.as_posix())
    return _MODEL


def load_resume_population(scenario, pattern_name, seed):
    """Load a prior final population as the initial population for a new run."""
    if RESUME_CHECKPOINT_DIR is None:
        return None
    path = Path(RESUME_CHECKPOINT_DIR) / (
        f"population_{scenario}_{pattern_name}_seed{seed}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Resume population not found: {path}")
    population = pd.read_csv(path, encoding="utf-8-sig")
    missing = [name for name in DECISION_BOUNDS if name not in population.columns]
    if missing:
        raise ValueError(f"Resume population is missing decision columns: {missing}")
    return population[list(DECISION_BOUNDS)].to_numpy(dtype=float)


def _is_leap_year(year):
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


YEAR_LENGTHS = [
    366 if USE_LEAP_YEARS and _is_leap_year(CALENDAR_START_YEAR + index) else 365
    for index in range(CALENDAR_NUM_YEARS)
]
YEAR_STARTS = np.cumsum([0] + YEAR_LENGTHS[:-1]).tolist()


def _year_timestamps(year_index):
    start = YEAR_STARTS[year_index]
    return list(range(start, start + YEAR_LENGTHS[year_index]))


def _year_start_day(year_index):
    return YEAR_STARTS[year_index]


def _round_year(value):
    return int(np.clip(np.rint(value), 0, CALENDAR_NUM_YEARS - 1))


def decode_decision_vector(vector):
    decision = dict(zip(DECISION_BOUNDS, vector))
    for name in DECISION_BOUNDS:
        if name.endswith("start_year") or name.endswith("end_year"):
            decision[name] = _round_year(decision[name])
    decision["breeding_enabled"] = int(decision["breeding_enabled"] >= 0.5)

    # A policy period is always non-empty. This also makes the encoded schedule
    # insensitive to the order in which DE proposes the two endpoints.
    for prefix in (
        "house_elevation",
        "migration",
        "paddy_dam",
        "forest_management",
    ):
        start_key = f"{prefix}_start_year"
        end_key = f"{prefix}_end_year"
        decision[start_key], decision[end_key] = sorted(
            (decision[start_key], decision[end_key])
        )
    # The end year is derived for reporting and plotting, not optimised.
    decision["breeding_end_year"] = (
        decision["breeding_start_year"] + BREEDING_DURATION_YEARS - 1
    )
    return decision


def _is_active(decision, prefix, year_index):
    if prefix == "breeding":
        return decision["breeding_enabled"] and (
            decision["breeding_start_year"]
            <= year_index
            <= decision["breeding_end_year"]
        )
    return decision[f"{prefix}_start_year"] <= year_index <= decision[
        f"{prefix}_end_year"
    ]


def fixed_params_for_scenario(scenario):
    if scenario not in SCENARIO_TO_PRECIP_RATIO:
        raise ValueError(f"Unknown scenario: {scenario}")
    params = FIXED_ASSUMPTIONS.copy()
    params.update(
        {
            "daily_precipitation_future_ratio": SCENARIO_TO_PRECIP_RATIO[scenario],
            "temperature_scenario_shift": SCENARIO_TO_TEMP_SHIFT[scenario],
            # Base policy state. Decision-specific values are inserted below.
            "drainage_investment_amount": 0,
            "drainage_investment_start_time": 0,
            "drainage_investment_duration": 365,
            "dam_investment_amount": 0,
            "dam_investment_start_time": 0,
            "dam_investment_duration": 365,
            "levee_investment_amount": 0,
            "levee_investment_start_time": 0,
            "levee_investment_duration": 365,
            "number_of_house_elevation": 0,
            "number_of_migration": 0,
            "annual_paddy_dam_investment": 0,
            "annual_breeding_investment": 0,
            "annual_forest_management_conversion_area": 0,
        }
    )
    return params


def policy_params_for_year(decision, year_index):
    """Return the model parameter values for one calendar year."""
    params = {
        "drainage_investment_amount": decision["drainage_investment_amount"],
        "drainage_investment_start_time": _year_start_day(
            decision["drainage_start_year"]
        ),
        "drainage_investment_duration": YEAR_LENGTHS[
            decision["drainage_start_year"]
        ],
        "dam_investment_amount": decision["dam_investment_amount"],
        "dam_investment_start_time": _year_start_day(decision["dam_start_year"]),
        "dam_investment_duration": YEAR_LENGTHS[decision["dam_start_year"]],
        "levee_investment_amount": decision["levee_investment_amount"],
        "levee_investment_start_time": _year_start_day(
            decision["levee_start_year"]
        ),
        "levee_investment_duration": YEAR_LENGTHS[
            decision["levee_start_year"]
        ],
        "number_of_house_elevation": (
            decision["house_elevation_per_year"]
            if _is_active(decision, "house_elevation", year_index)
            else 0
        ),
        "number_of_migration": (
            decision["migration_per_year"]
            if _is_active(decision, "migration", year_index)
            else 0
        ),
        "annual_paddy_dam_investment": (
            decision["annual_paddy_dam_investment"]
            if _is_active(decision, "paddy_dam", year_index)
            else 0
        ),
        "annual_breeding_investment": (
            BREEDING_ANNUAL_COST
            if decision["breeding_enabled"]
            and _is_active(decision, "breeding", year_index)
            else 0
        ),
        "annual_forest_management_conversion_area": (
            decision["annual_forest_management_conversion_area"]
            if _is_active(decision, "forest_management", year_index)
            else 0
        ),
    }
    return params


def policy_cost_before_forest(decision, year_index):
    """Annual policy cost with units explicitly handled as Yen/year."""
    cost = 0.0
    if year_index == decision["drainage_start_year"]:
        cost += decision["drainage_investment_amount"]
    if year_index == decision["dam_start_year"]:
        cost += decision["dam_investment_amount"]
    if year_index == decision["levee_start_year"]:
        cost += decision["levee_investment_amount"]
    if _is_active(decision, "house_elevation", year_index):
        cost += decision["house_elevation_per_year"] * UNIT_COST_OF_ELEVATION
    if _is_active(decision, "migration", year_index):
        cost += decision["migration_per_year"] * UNIT_COST_OF_MIGRATION
    if decision["breeding_enabled"] and _is_active(decision, "breeding", year_index):
        cost += BREEDING_ANNUAL_COST
    return cost


def run_model_yearly(scenario, decision):
    """Run sequential years so annual policy schedules retain model stocks."""
    model = get_model()
    base_params = fixed_params_for_scenario(scenario)
    results = []

    for year_index in range(CALENDAR_NUM_YEARS):
        params = base_params.copy()
        params.update(policy_params_for_year(decision, year_index))
        result = model.run(
            params=params,
            return_timestamps=_year_timestamps(year_index),
            return_columns=RETURN_COLS,
            initial_condition="original" if year_index == 0 else "current",
        )
        results.append(result)
    return results


def aggregate_results(results, decision):
    metrics = {
        "flood_damage": 0.0,
        "innundation_damage": 0.0,
        "landslide_risk": 0.0,
        "crop_revenue": 0.0,
        "biodiversity": 0.0,
        "daily_total_gdp": 0.0,
    }
    yearly_rows = []

    for year_index, result in enumerate(results):
        forest_cost = float(result["additional_forest_management_cost"].mean())
        paddy_dam_cost = float(result["paddy_dam_investment"].sum())
        breeding_cost = float(result["breeding_investment"].sum())
        policy_cost = (
            policy_cost_before_forest(decision, year_index)
            + paddy_dam_cost
            + forest_cost
        )
        row = {
            "year_index": year_index,
            "year": CALENDAR_START_YEAR + year_index,
            "flood_damage": float(result["financial_damage_by_flood"].sum()),
            "innundation_damage": float(result["financial_damage_by_innundation"].sum()),
            "landslide_risk": float(result["landslide_disaster_risk"].sum()),
            "maximum_landslide_risk": float(result["landslide_disaster_risk"].max()),
            "maximum_adjusted_daily_precip_up": float(
                result["scenario_adjusted_daily_precip_up"].max()
            ),
            "forest_mitigation": float(result["forest_mitigation"].mean()),
            # Crop cashflow is booked only at harvest, so its annual sum is revenue.
            "crop_revenue": float(result["crop_production_cashflow"].sum()),
            "biodiversity": float(result["biodiversity"].mean()),
            "daily_total_gdp": float(result["daily_total_gdp"].sum()),
            "forest_function_coef": float(result["forest_function_coef"].mean()),
            "houses_in_flood_risk": float(result["houses_in_flood_risk"].iloc[-1]),
            "houses_in_inundation_risk": float(
                result["houses_in_inundation_risk"].iloc[-1]
            ),
            "flood_only_risk_houses": float(
                result["flood_only_risk_houses"].iloc[-1]
            ),
            "innundation_only_risk_houses": float(
                result["innundation_only_risk_houses"].iloc[-1]
            ),
            "overlapping_risk_houses": float(
                result["overlapping_risk_houses"].iloc[-1]
            ),
            "houses_in_nonrisky_area": float(
                result["houses_in_nonrisky_area"].iloc[-1]
            ),
            "elevated_houses": float(result["elevated_houses"].iloc[-1]),
            "managed_plantation_forest_area": float(
                result["managed_plantation_forest_area"].iloc[-1]
            ),
            "unmanaged_plantation_forest_area": float(
                result["unmanaged_plantation_forest_area"].iloc[-1]
            ),
            "additional_forest_management_cost": forest_cost,
            "paddy_dam_actual_cost": paddy_dam_cost,
            "paddy_dam_area": float(result["paddy_dam_area"].iloc[-1]),
            "paddy_dam_ratio": float(result["paddy_dam_ratio"].iloc[-1]),
            "breeding_actual_cost": breeding_cost,
            "accumulated_breeding_investment": float(
                result["accumulated_breeding_investment"].iloc[-1]
            ),
            "breeding_success": float(result["breeding_success"].iloc[-1]),
            "policy_additional_cost": policy_cost,
        }
        yearly_rows.append(row)
        for metric in ("flood_damage", "innundation_damage", "landslide_risk", "crop_revenue", "daily_total_gdp"):
            metrics[metric] += row[metric]
        metrics["biodiversity"] += row["biodiversity"]

    metrics["biodiversity"] /= CALENDAR_NUM_YEARS
    metrics["policy_additional_cost"] = sum(
        row["policy_additional_cost"] for row in yearly_rows
    )
    metrics["max_annual_policy_additional_cost"] = max(
        row["policy_additional_cost"] for row in yearly_rows
    )
    return metrics, pd.DataFrame(yearly_rows)


def evaluate_policy(scenario, decision):
    results = run_model_yearly(scenario, decision)
    return aggregate_results(results, decision)


def relative_objective(metrics, baseline, weights):
    """Return a dimensionless objective based on within-scenario ratios."""
    score = 0.0
    for name in ("flood_damage", "innundation_damage", "landslide_risk"):
        base = baseline[name]
        # If the no-policy simulation has no event, a relative damage reduction
        # cannot be identified. It is reported but omitted from this objective.
        if base > 1e-12:
            score += weights[name] * metrics[name] / base
    for name in ("crop_revenue", "biodiversity"):
        base = baseline[name]
        if abs(base) > 1e-12:
            score -= weights[name] * metrics[name] / base
    return score


def budget_penalty(metrics):
    limits = BUDGET_CASES[BUDGET_CASE]
    annual_excess = max(
        0.0,
        metrics["max_annual_policy_additional_cost"] / limits["annual"] - 1.0,
    )
    total_excess = max(
        0.0,
        metrics["policy_additional_cost"] / limits["total"] - 1.0,
    )
    if annual_excess > 0 or total_excess > 0:
        # Treat the approved policy budget as a hard feasibility constraint.
        return BUDGET_PENALTY * (1 + annual_excess + total_excess)
    return 0.0


def cost_tiebreaker(metrics):
    """Prefer lower-cost policies only when primary outcomes are effectively tied."""
    total_limit = BUDGET_CASES[BUDGET_CASE]["total"]
    return COST_TIEBREAKER_WEIGHT * metrics["policy_additional_cost"] / total_limit


def objective_for_vector(vector, scenario, baseline, weights):
    """Pickleable objective used by scipy worker processes on Windows."""
    decision = decode_decision_vector(vector)
    metrics, _ = evaluate_policy(scenario, decision)
    return (
        relative_objective(metrics, baseline, weights)
        + cost_tiebreaker(metrics)
        + budget_penalty(metrics)
    )


def optimise_scenario(scenario, pattern_name, weights, seed):
    started_at = datetime.now()
    print(f"{scenario}/{pattern_name}: calculating no-policy baseline...")
    zero_decision = decode_decision_vector(
        [0.0 for _ in DECISION_BOUNDS]
    )
    baseline, _ = evaluate_policy(scenario, zero_decision)
    print(
        f"{scenario}/{pattern_name}: baseline complete; starting differential "
        f"evolution (up to {MAXITER} generations, population={POPSIZE * len(DECISION_BOUNDS)}, "
        f"workers={MAX_WORKERS}, tol={DE_TOLERANCE})."
    )
    evaluations = {"count": 0}

    objective = functools.partial(
        objective_for_vector,
        scenario=scenario,
        baseline=baseline,
        weights=weights,
    )

    def progress(_vector, convergence):
        evaluations["count"] += 1
        if evaluations["count"] % PROGRESS_EVERY == 0:
            elapsed_minutes = (datetime.now() - started_at).total_seconds() / 60
            print(
                f"{scenario}/{pattern_name}: generation={evaluations['count']}/"
                f"{MAXITER} ({evaluations['count'] / MAXITER:.0%}), "
                f"convergence={convergence:.4g}, elapsed={elapsed_minutes:.1f} min"
            )
        return False

    initial_population = load_resume_population(scenario, pattern_name, seed)
    if initial_population is not None:
        print(
            f"{scenario}/{pattern_name}: resuming from {len(initial_population)} "
            "saved population members."
        )

    de_kwargs = {
        "bounds": list(DECISION_BOUNDS.values()),
        "maxiter": MAXITER,
        "popsize": POPSIZE,
        "tol": DE_TOLERANCE,
        "seed": seed,
        "polish": False,
        "callback": progress,
        # Deferred updating is required when scipy evaluates candidates in
        # parallel. Each worker lazily loads and reuses its own PySD model.
        "updating": "deferred",
        "workers": MAX_WORKERS,
    }
    if initial_population is not None:
        de_kwargs["init"] = initial_population

    result = differential_evolution(
        objective,
        **de_kwargs,
    )
    decision = decode_decision_vector(result.x)
    metrics, yearly = evaluate_policy(scenario, decision)
    elapsed_minutes = (datetime.now() - started_at).total_seconds() / 60
    print(
        f"{scenario}/{pattern_name}: complete; generations={result.nit}, "
        f"evaluations={result.nfev}, objective={result.fun:.6g}, "
        f"elapsed={elapsed_minutes:.1f} min"
    )
    return result, decision, baseline, metrics, yearly


def _comparison_row(scenario, pattern_name, seed, result, baseline, metrics):
    row = {
        "basin": BASIN_KEY,
        "scenario": scenario,
        "pattern": pattern_name,
        "seed": seed,
        "budget_case": BUDGET_CASE,
        "objective_score": result.fun,
        "function_evaluations": result.nfev,
        "iterations": result.nit,
        # SciPy's explicit termination status distinguishes convergence from
        # a run that merely reached MAXITER.
        "optimizer_success": result.success,
        "optimizer_message": result.message,
    }
    for name, value in metrics.items():
        row[name] = value
        base = baseline.get(name)
        if base is not None:
            row[f"baseline_{name}"] = base
            if abs(base) > 1e-12:
                row[f"{name}_vs_baseline_ratio"] = value / base
    limits = BUDGET_CASES[BUDGET_CASE]
    row["annual_budget_utilisation"] = (
        metrics["max_annual_policy_additional_cost"] / limits["annual"]
    )
    row["total_budget_utilisation"] = (
        metrics["policy_additional_cost"] / limits["total"]
    )
    row["cost_tiebreaker"] = cost_tiebreaker(metrics)
    return row


def save_checkpoint(
    checkpoint_dir, summary_rows, decision_rows, yearly_frames, optimiser_runs
):
    """Persist completed patterns and their final DE populations for later reuse."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        checkpoint_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(decision_rows).to_csv(
        checkpoint_dir / "decisions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(yearly_frames, ignore_index=True).to_csv(
        checkpoint_dir / "yearly.csv", index=False, encoding="utf-8-sig"
    )

    state_rows = []
    for scenario, pattern_name, seed, result in optimiser_runs:
        population = np.asarray(result.population, dtype=float)
        population_frame = pd.DataFrame(population, columns=list(DECISION_BOUNDS))
        population_frame.insert(0, "candidate_index", np.arange(len(population_frame)))
        population_frame.insert(1, "objective_energy", result.population_energies)
        population_frame.to_csv(
            checkpoint_dir / f"population_{scenario}_{pattern_name}_seed{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        state_rows.append(
            {
                "basin": BASIN_KEY,
                "scenario": scenario,
                "pattern": pattern_name,
                "seed": seed,
                "objective_score": float(result.fun),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                **{
                    f"best_raw_{name}": float(value)
                    for name, value in zip(DECISION_BOUNDS, result.x)
                },
            }
        )
    pd.DataFrame(state_rows).to_csv(
        checkpoint_dir / "optimiser_state.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "basin": BASIN_KEY,
        "basin_config": str(BASIN_CONFIG["_path"]),
        "decision_variable_order": list(DECISION_BOUNDS),
        "population_size": POPSIZE * len(DECISION_BOUNDS),
        "maxiter": MAXITER,
        "tol": DE_TOLERANCE,
        "seeds": OPTIMIZATION_SEEDS,
        "scenarios": SCENARIOS,
        "objective_patterns": OBJECTIVE_PATTERNS,
    }
    (checkpoint_dir / "resume_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"opt_to7_{BUDGET_CASE}_{RUN_STAMP}"
    checkpoint_dir = OUTPUT_DIR / "checkpoints" / stem
    summary_rows = []
    decision_rows = []
    yearly_frames = []
    optimiser_runs = []

    for scenario in SCENARIOS:
        for pattern_name, weights in OBJECTIVE_PATTERNS.items():
            for seed in OPTIMIZATION_SEEDS:
                print(
                    f"\n=== {scenario}: {pattern_name} / {BUDGET_CASE} "
                    f"budget / seed={seed} ==="
                )
                result, decision, baseline, metrics, yearly = optimise_scenario(
                    scenario, pattern_name, weights, seed
                )
                optimiser_runs.append((scenario, pattern_name, seed, result))
                summary_rows.append(
                    _comparison_row(
                        scenario, pattern_name, seed, result, baseline, metrics
                    )
                )
                decision_rows.append(
                    {
                        "basin": BASIN_KEY,
                        "scenario": scenario,
                        "pattern": pattern_name,
                        "seed": seed,
                        **decision,
                    }
                )
                yearly.insert(0, "seed", seed)
                yearly.insert(0, "pattern", pattern_name)
                yearly.insert(0, "scenario", scenario)
                yearly.insert(0, "basin", BASIN_KEY)
                yearly_frames.append(yearly)
                save_checkpoint(
                    checkpoint_dir,
                    summary_rows,
                    decision_rows,
                    yearly_frames,
                    optimiser_runs,
                )
                print(f"Checkpoint saved: {checkpoint_dir}")

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / f"{stem}_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(decision_rows).to_csv(
        OUTPUT_DIR / f"{stem}_decisions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(yearly_frames, ignore_index=True).to_csv(
        OUTPUT_DIR / f"{stem}_yearly.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Saved: {OUTPUT_DIR / stem}")


if __name__ == "__main__":
    main()
