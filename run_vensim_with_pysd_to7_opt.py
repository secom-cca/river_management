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
from breeding_response_to7 import model_configuration, model_parameters

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
MODEL_PY = Path("River_management_xls_to7.py")
OUTPUT_DIR = Path("data")
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")

CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
USE_LEAP_YEARS = True

# Use a subset such as ["present"] for an exploratory run. Because the to7
# damage structure changed, prior to6-based optimisation results are not
# directly comparable and all three scenarios should be rerun for analysis.
SCENARIOS = ["present", "2C", "4C"]

# Coarse first-stage search across every scenario and objective pattern. Resume
# promising cases from their checkpoints with larger values for final analysis.
MAXITER = 60
POPSIZE = 5
# Use a tighter tolerance for the continuation so small outcome differences and
# the 0.001 policy-cost tie-breaker do not trigger another coarse early stop.
DE_TOLERANCE = 0.001

# Keep one seed for an exploratory run. For final analysis, for example use
# [42, 43, 44] and compare both objective values and policy decisions.
OPTIMIZATION_SEEDS = [42]

# Print each completed DE generation so long PySD runs visibly advance in the
# PowerShell console. Set this higher only when console output is excessive.
PROGRESS_EVERY = 1
# Keep part of the workstation's logical CPU capacity available for the OS.
MAX_WORKERS = 12
BREEDING_ANNUAL_COST = 50_000_000
BREEDING_DURATION_YEARS = 10
# Continue only the cases that ended with optimizer_success=False. Their saved
# Historical checkpoints use the old calendar/breeding model. Start a new batch
# after this model revision; only compatible new checkpoints can be continued.
CONTINUE_FROM_CHECKPOINT_DIR = None
# Leave this as None for the first continuation run. If that run is interrupted
# after completing one or more cases, set this to its new checkpoint directory
# and rerun while keeping CONTINUE_FROM_CHECKPOINT_DIR unchanged.
RESUME_CHECKPOINT_DIR = None
# The original random-generator state was not saved. Use a distinct,
# reproducible stream for the continuation instead of replaying seed 42.
CONTINUATION_RNG_SEED_OFFSET = 1_000

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

SCENARIO_TO_PRECIP_RATIO = {"present": 1.0, "2C": 1.1, "4C": 1.3}
SCENARIO_TO_TEMP_SHIFT = {"present": 0.0, "2C": 1.3, "4C": 4.1}

# Fixed values are the central values of the to7 parameter study. Hydrological
# parameters remain at the separately calibrated values.
FIXED_ASSUMPTIONS = {
    "forest_area_ratio": 0.92,
    "paddy_field_ratio": 0.15,
    "ratio_of_paddy_field_in_risky_area": 0.10,
    "innundation_risky_area_ratio": 0.8,
    "flood_risky_area_ratio": 0.8,
    "recovery_ratio": 0.9,
    "delay_of_recovery": 365,
    "initial_total_risk_households": 50_000,
    "risk_household_overlap_ratio": 0.5,
    "crop_price": 250,
    "waterholding_capacity_of_forest_base": 225,
    "landslide_design_daily_precipitation": 322,
    "inundation_damage_per_household": 30_000_000,
    "flood_damage_per_household": 30_000_000,
    "average_household_size": 2.21,
    "gdp_per_resident": 3e6 / 365,
    "paddy_field_capacity_per_area": 500,
    "paddy_dam_capacity_per_area": 1500,
    "paddy_dam_cost_per_area": 1_500_000,
    "daily_paddy_damage_fraction": 0.10,
    "paddy_damage_depth_threshold": 0.3,
    "paddy_field_recovery_years_to_90_percent": 2.0,
    "predischarge_capacity_ratio": 0.10,
    "unmanaged_plantation_forest_coef": 0.7,
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
    "financial_damage_by_inundation",
    "flood_event_healthy_households",
    "inundation_event_healthy_households",
    "landslide_disaster_risk",
    "crop_production_cashflow",
    "biodiversity",
    "daily_total_gdp",
    "forest_function_coef",
    "forest_mitigation",
    "scenario_adjusted_daily_precip_up",
    "landslide_design_daily_precipitation",
    "dam_storage",
    "dam_capacity",
    "predischarge_control",
    "predischarge_capacity",
    "predischarge_capacity_ratio",
    "dam_spill",
    "dam_outflow",
    "houses_in_flood_risk",
    "houses_in_inundation_risk",
    "flood_only_risk_houses",
    "innundation_only_risk_houses",
    "overlapping_risk_houses",
    "damaged_flood_only_houses",
    "damaged_inundation_only_houses",
    "damaged_overlapping_houses",
    "damaged_houses",
    "houses_in_nonrisky_area",
    "elevated_houses",
    "managed_plantation_forest_area",
    "unmanaged_plantation_forest_area",
    "additional_forest_management_cost",
    "paddy_dam_investment",
    "paddy_dam_area",
    "paddy_dam_ratio",
    "physical_paddy_field",
    "damaged_paddy_field",
    "paddyfield_storage_capacity",
    "breeding_investment",
    "accumulated_breeding_investment",
    "breeding_success",
    "breeding_effect_ratio",
    "chalky_kernel_ratio",
    "effective_chalky_kernel_ratio",
    "crop_price_quality_factor",
    "harvest_trigger",
]

# Observe stocks after the final integration step of each calendar year.
# Daily flows still have exactly 365/366 samples, excluding next January 1.
YEAR_END_COLUMNS = [
    "dam_storage", "dam_capacity", "houses_in_flood_risk",
    "houses_in_inundation_risk", "flood_only_risk_houses",
    "innundation_only_risk_houses", "overlapping_risk_houses",
    "damaged_flood_only_houses", "damaged_inundation_only_houses",
    "damaged_overlapping_houses", "damaged_houses", "houses_in_nonrisky_area",
    "elevated_houses", "managed_plantation_forest_area",
    "unmanaged_plantation_forest_area", "paddy_dam_area", "paddy_dam_ratio",
    "physical_paddy_field", "damaged_paddy_field", "paddyfield_storage_capacity",
    "accumulated_breeding_investment", "breeding_success", "breeding_effect_ratio",
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
    """Load a saved population from the active batch or continuation source."""
    directories = []
    if RESUME_CHECKPOINT_DIR is not None:
        directories.append(Path(RESUME_CHECKPOINT_DIR))
    if CONTINUE_FROM_CHECKPOINT_DIR is not None:
        source = Path(CONTINUE_FROM_CHECKPOINT_DIR)
        if source not in directories:
            directories.append(source)

    filename = f"population_{scenario}_{pattern_name}_seed{seed}.csv"
    for directory in directories:
        path = directory / filename
        if not path.exists():
            continue
        population = pd.read_csv(path, encoding="utf-8-sig")
        missing = [name for name in DECISION_BOUNDS if name not in population.columns]
        if missing:
            raise ValueError(
                f"Resume population is missing decision columns: {missing}"
            )
        values = population[list(DECISION_BOUNDS)].to_numpy(dtype=float)
        if values.shape != (POPSIZE * len(DECISION_BOUNDS), len(DECISION_BOUNDS)):
            raise ValueError(
                f"Unexpected saved population shape {values.shape} in {path}; "
                f"expected {(POPSIZE * len(DECISION_BOUNDS), len(DECISION_BOUNDS))}."
            )
        return values
    return None


def _case_key(row):
    return str(row["scenario"]), str(row["pattern"]), int(row["seed"])


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _configured_case_keys():
    return {
        (scenario, pattern_name, int(seed))
        for scenario in SCENARIOS
        for pattern_name in OBJECTIVE_PATTERNS
        for seed in OPTIMIZATION_SEEDS
    }


def _filter_records(records, selected_keys):
    return [row for row in records if _case_key(row) in selected_keys]


def _filter_yearly_frame(frame, selected_keys):
    keys = frame.apply(_case_key, axis=1)
    return frame[keys.isin(selected_keys)].copy()


def _order_output_frame(frame):
    """Keep scenario/pattern/seed/year order stable in combined output files."""
    frame = frame.copy()
    scenario_order = {name: index for index, name in enumerate(SCENARIOS)}
    pattern_order = {
        name: index for index, name in enumerate(OBJECTIVE_PATTERNS)
    }
    frame["_scenario_order"] = frame["scenario"].map(scenario_order)
    frame["_pattern_order"] = frame["pattern"].map(pattern_order)
    sort_columns = ["_scenario_order", "_pattern_order", "seed"]
    if "year" in frame.columns:
        sort_columns.append("year")
    frame = frame.sort_values(sort_columns).drop(
        columns=["_scenario_order", "_pattern_order"]
    )
    return frame.reset_index(drop=True)


def load_completed_checkpoint(checkpoint_dir):
    """Load and validate completed cases from an interrupted optimisation batch."""
    metadata_path = checkpoint_dir / "resume_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if metadata.get("model_configuration") != model_configuration():
        raise ValueError("Checkpoint uses a different calendar/breeding model; start a new optimisation batch.")
    table_names = ("summary", "decisions", "yearly", "optimiser_state")
    paths = {name: checkpoint_dir / f"{name}.csv" for name in table_names}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Resume checkpoint is missing required files: " + ", ".join(missing)
        )

    summary = pd.read_csv(paths["summary"], encoding="utf-8-sig")
    decisions = pd.read_csv(paths["decisions"], encoding="utf-8-sig")
    yearly = pd.read_csv(paths["yearly"], encoding="utf-8-sig")
    optimiser_state = pd.read_csv(
        paths["optimiser_state"], encoding="utf-8-sig"
    )

    summary_keys = {_case_key(row) for row in summary.to_dict("records")}
    decision_keys = {_case_key(row) for row in decisions.to_dict("records")}
    yearly_keys = {_case_key(row) for row in yearly.to_dict("records")}
    state_keys = {_case_key(row) for row in optimiser_state.to_dict("records")}
    if not summary_keys or not (
        summary_keys == decision_keys == yearly_keys == state_keys
    ):
        raise ValueError(
            "Resume checkpoint tables do not contain the same completed cases."
        )

    for scenario, pattern_name, seed in summary_keys:
        population_path = checkpoint_dir / (
            f"population_{scenario}_{pattern_name}_seed{seed}.csv"
        )
        if not population_path.exists():
            raise FileNotFoundError(
                f"Completed case is missing its population: {population_path}"
            )

    return {
        "summary_rows": summary.to_dict("records"),
        "decision_rows": decisions.to_dict("records"),
        "yearly_frames": [yearly],
        "state_rows": optimiser_state.to_dict("records"),
        "completed_keys": summary_keys,
    }


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
    params.update(model_parameters())
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


def run_model_yearly(scenario, decision, *, model_params=None, return_columns=None):
    """Run sequential years so annual policy schedules retain model stocks."""
    model = get_model()
    base_params = fixed_params_for_scenario(scenario)
    if model_params:
        base_params.update(model_params)
    results = []

    for year_index in range(CALENDAR_NUM_YEARS):
        params = base_params.copy()
        params.update(policy_params_for_year(decision, year_index))
        result = model.run(
            params=params,
            return_timestamps=_year_timestamps(year_index),
            return_columns=RETURN_COLS if return_columns is None else return_columns,
            final_time=YEAR_STARTS[year_index] + YEAR_LENGTHS[year_index],
            initial_condition="original" if year_index == 0 else "current",
        )
        result.attrs["year_end_state"] = {
            name: float(getattr(model.components, name)())
            for name in YEAR_END_COLUMNS
        }
        result.attrs["year_end_time"] = float(model.time())
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
        year_end = result.attrs.get("year_end_state", {})

        def at_year_end(name):
            return float(year_end[name]) if name in year_end else float(result[name].iloc[-1])

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
            # Keep the historical summary key for output compatibility, while
            # reading the corrected model component name.
            "innundation_damage": float(result["financial_damage_by_inundation"].sum()),
            "landslide_risk": float(result["landslide_disaster_risk"].sum()),
            "maximum_landslide_risk": float(result["landslide_disaster_risk"].max()),
            "maximum_adjusted_daily_precip_up": float(
                result["scenario_adjusted_daily_precip_up"].max()
            ),
            "forest_mitigation": float(result["forest_mitigation"].mean()),
            "dam_storage_at_year_end": at_year_end("dam_storage"),
            "dam_storage_year_min": float(result["dam_storage"].min()),
            "dam_storage_year_max": float(result["dam_storage"].max()),
            "dam_capacity_at_year_end": at_year_end("dam_capacity"),
            "predischarge_control_year_sum": float(
                result["predischarge_control"].sum()
            ),
            "predischarge_capacity_year_mean": float(
                result["predischarge_capacity"].mean()
            ),
            "dam_spill_year_sum": float(result["dam_spill"].sum()),
            "dam_outflow_year_sum": float(result["dam_outflow"].sum()),
            # Crop cashflow is booked only at harvest, so its annual sum is revenue.
            "crop_revenue": float(result["crop_production_cashflow"].sum()),
            "biodiversity": float(result["biodiversity"].mean()),
            "daily_total_gdp": float(result["daily_total_gdp"].sum()),
            "forest_function_coef": float(result["forest_function_coef"].mean()),
            "houses_in_flood_risk": at_year_end("houses_in_flood_risk"),
            "houses_in_inundation_risk": at_year_end("houses_in_inundation_risk"),
            "flood_only_risk_houses": at_year_end("flood_only_risk_houses"),
            "innundation_only_risk_houses": at_year_end("innundation_only_risk_houses"),
            "overlapping_risk_houses": at_year_end("overlapping_risk_houses"),
            "damaged_flood_only_houses": at_year_end("damaged_flood_only_houses"),
            "damaged_inundation_only_houses": at_year_end("damaged_inundation_only_houses"),
            "damaged_overlapping_houses": at_year_end("damaged_overlapping_houses"),
            "damaged_houses": at_year_end("damaged_houses"),
            "houses_in_nonrisky_area": at_year_end("houses_in_nonrisky_area"),
            "elevated_houses": at_year_end("elevated_houses"),
            "managed_plantation_forest_area": at_year_end("managed_plantation_forest_area"),
            "unmanaged_plantation_forest_area": at_year_end("unmanaged_plantation_forest_area"),
            "additional_forest_management_cost": forest_cost,
            "paddy_dam_actual_cost": paddy_dam_cost,
            "paddy_dam_area": at_year_end("paddy_dam_area"),
            "paddy_dam_ratio": at_year_end("paddy_dam_ratio"),
            "physical_paddy_field": at_year_end("physical_paddy_field"),
            "damaged_paddy_field": at_year_end("damaged_paddy_field"),
            "paddyfield_storage_capacity": at_year_end("paddyfield_storage_capacity"),
            "breeding_actual_cost": breeding_cost,
            "accumulated_breeding_investment": at_year_end("accumulated_breeding_investment"),
            "breeding_success": at_year_end("breeding_success"),
            "breeding_effect_ratio": at_year_end("breeding_effect_ratio"),
            "year_end_time": result.attrs.get("year_end_time", float(result.index[-1])),
            "policy_additional_cost": policy_cost,
        }
        harvest = result[result["harvest_trigger"] > 0.5]
        if len(harvest) != 1:
            raise ValueError("Expected exactly one harvest per calendar year")
        for name in (
            "accumulated_breeding_investment", "breeding_effect_ratio",
            "chalky_kernel_ratio", "effective_chalky_kernel_ratio", "crop_price_quality_factor",
        ):
            row[name + "_at_harvest"] = float(harvest[name].iloc[0])
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


def evaluate_policy(scenario, decision, *, model_params=None):
    results = run_model_yearly(scenario, decision, model_params=model_params)
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

    de_seed = (
        seed + CONTINUATION_RNG_SEED_OFFSET
        if CONTINUE_FROM_CHECKPOINT_DIR is not None
        else seed
    )
    de_kwargs = {
        "bounds": list(DECISION_BOUNDS.values()),
        "maxiter": MAXITER,
        "popsize": POPSIZE,
        "tol": DE_TOLERANCE,
        "seed": de_seed,
        "polish": False,
        "callback": progress,
        # Deferred updating is required when scipy evaluates candidates in
        # parallel. Each worker lazily loads and reuses its own PySD model.
        "updating": "deferred",
        "workers": MAX_WORKERS,
    }
    if initial_population is not None:
        de_kwargs["init"] = initial_population
        print(
            f"{scenario}/{pattern_name}: continuation RNG seed={de_seed}."
        )

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
    checkpoint_dir,
    summary_rows,
    decision_rows,
    yearly_frames,
    optimiser_runs,
    resumed_state_rows=None,
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

    state_rows = list(resumed_state_rows or [])
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
        "model_configuration": model_configuration(),
        "decision_variable_order": list(DECISION_BOUNDS),
        "population_size": POPSIZE * len(DECISION_BOUNDS),
        "maxiter": MAXITER,
        "tol": DE_TOLERANCE,
        "seeds": OPTIMIZATION_SEEDS,
        "scenarios": SCENARIOS,
        "objective_patterns": OBJECTIVE_PATTERNS,
        "continue_from_checkpoint": (
            str(CONTINUE_FROM_CHECKPOINT_DIR)
            if CONTINUE_FROM_CHECKPOINT_DIR is not None
            else None
        ),
        "continuation_rng_seed_offset": (
            CONTINUATION_RNG_SEED_OFFSET
            if CONTINUE_FROM_CHECKPOINT_DIR is not None
            else None
        ),
    }
    (checkpoint_dir / "resume_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    configured_keys = _configured_case_keys()
    continuation_source = None
    continuation_target_keys = configured_keys
    retained_source_keys = set()
    source_summary_by_key = {}
    source_state_by_key = {}

    if CONTINUE_FROM_CHECKPOINT_DIR is not None:
        source_dir = Path(CONTINUE_FROM_CHECKPOINT_DIR)
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Continuation source checkpoint not found: {source_dir}"
            )
        if (
            RESUME_CHECKPOINT_DIR is not None
            and Path(RESUME_CHECKPOINT_DIR).resolve() == source_dir.resolve()
        ):
            raise ValueError(
                "RESUME_CHECKPOINT_DIR must be the new continuation checkpoint, "
                "not CONTINUE_FROM_CHECKPOINT_DIR."
            )
        continuation_source = load_completed_checkpoint(source_dir)
        source_keys = continuation_source["completed_keys"]
        missing_source_keys = configured_keys - source_keys
        if missing_source_keys:
            raise ValueError(
                "Continuation source does not contain every configured case: "
                + ", ".join(map(str, sorted(missing_source_keys)))
            )
        source_summary_by_key = {
            _case_key(row): row for row in continuation_source["summary_rows"]
        }
        source_state_by_key = {
            _case_key(row): row for row in continuation_source["state_rows"]
        }
        continuation_target_keys = {
            key
            for key in configured_keys
            if not _as_bool(source_state_by_key[key]["optimizer_success"])
        }
        if not continuation_target_keys:
            raise ValueError(
                "Continuation source has no configured optimizer_success=False cases."
            )
        retained_source_keys = configured_keys - continuation_target_keys
        print(
            f"Continuation source: {source_dir}; extending "
            f"{len(continuation_target_keys)} unconverged case(s) by up to "
            f"{MAXITER} generations and retaining {len(retained_source_keys)} "
            "converged case(s)."
        )

    if RESUME_CHECKPOINT_DIR is None:
        stem = f"opt_to7_{BUDGET_CASE}_{RUN_STAMP}"
        checkpoint_dir = OUTPUT_DIR / "checkpoints" / stem
    else:
        checkpoint_dir = Path(RESUME_CHECKPOINT_DIR)
        stem = checkpoint_dir.name
        expected_prefix = f"opt_to7_{BUDGET_CASE}_"
        if not stem.startswith(expected_prefix):
            raise ValueError(
                f"Resume checkpoint '{stem}' does not match budget case "
                f"'{BUDGET_CASE}'."
            )

    final_paths = [
        OUTPUT_DIR / f"{stem}_summary.csv",
        OUTPUT_DIR / f"{stem}_decisions.csv",
        OUTPUT_DIR / f"{stem}_yearly.csv",
    ]

    if RESUME_CHECKPOINT_DIR is None:
        existing = [path for path in final_paths if path.exists()]
        if checkpoint_dir.exists():
            existing.append(checkpoint_dir)
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing optimisation output: "
                + ", ".join(str(path) for path in existing)
            )
        summary_rows = []
        decision_rows = []
        yearly_frames = []
        resumed_state_rows = []
        completed_keys = set()
    else:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Resume checkpoint directory not found: {checkpoint_dir}"
            )
        completed = load_completed_checkpoint(checkpoint_dir)
        summary_rows = completed["summary_rows"]
        decision_rows = completed["decision_rows"]
        yearly_frames = completed["yearly_frames"]
        resumed_state_rows = completed["state_rows"]
        completed_keys = completed["completed_keys"]
        if CONTINUE_FROM_CHECKPOINT_DIR is not None and not completed_keys.issubset(
            continuation_target_keys
        ):
            raise ValueError(
                "The resume checkpoint contains cases outside this continuation set."
            )
        print(
            f"Resuming {stem}: loaded {len(completed_keys)} completed case(s)."
        )

    optimiser_runs = []

    for scenario in SCENARIOS:
        for pattern_name, weights in OBJECTIVE_PATTERNS.items():
            for seed in OPTIMIZATION_SEEDS:
                case_key = (scenario, pattern_name, int(seed))
                if (
                    CONTINUE_FROM_CHECKPOINT_DIR is not None
                    and case_key not in continuation_target_keys
                ):
                    print(
                        f"Retaining previously converged case: {scenario}/"
                        f"{pattern_name}/seed={seed}"
                    )
                    continue
                if case_key in completed_keys:
                    print(
                        f"Skipping completed case: {scenario}/{pattern_name}/"
                        f"seed={seed}"
                    )
                    continue
                print(
                    f"\n=== {scenario}: {pattern_name} / {BUDGET_CASE} "
                    f"budget / seed={seed} ==="
                )
                result, decision, baseline, metrics, yearly = optimise_scenario(
                    scenario, pattern_name, weights, seed
                )
                optimiser_runs.append((scenario, pattern_name, seed, result))
                comparison_row = _comparison_row(
                    scenario, pattern_name, seed, result, baseline, metrics
                )
                if CONTINUE_FROM_CHECKPOINT_DIR is not None:
                    source_summary = source_summary_by_key[case_key]
                    source_state = source_state_by_key[case_key]
                    comparison_row.update(
                        {
                            "continuation_applied": True,
                            "previous_objective_score": float(
                                source_summary["objective_score"]
                            ),
                            "additional_iterations": int(result.nit),
                            "cumulative_iterations": int(
                                source_state["iterations"]
                            )
                            + int(result.nit),
                        }
                    )
                summary_rows.append(comparison_row)
                decision_rows.append(
                    {
                        "scenario": scenario,
                        "pattern": pattern_name,
                        "seed": seed,
                        **decision,
                    }
                )
                yearly.insert(0, "seed", seed)
                yearly.insert(0, "pattern", pattern_name)
                yearly.insert(0, "scenario", scenario)
                yearly_frames.append(yearly)
                save_checkpoint(
                    checkpoint_dir,
                    summary_rows,
                    decision_rows,
                    yearly_frames,
                    optimiser_runs,
                    resumed_state_rows,
                )
                print(f"Checkpoint saved: {checkpoint_dir}")

    if CONTINUE_FROM_CHECKPOINT_DIR is not None:
        continued_keys = {_case_key(row) for row in summary_rows}
        if continued_keys != continuation_target_keys:
            missing = continuation_target_keys - continued_keys
            raise RuntimeError(
                "Continuation ended without every target case: "
                + ", ".join(map(str, sorted(missing)))
            )

        retained_summary_rows = []
        for row in _filter_records(
            continuation_source["summary_rows"], retained_source_keys
        ):
            annotated = dict(row)
            key = _case_key(row)
            annotated.update(
                {
                    "continuation_applied": False,
                    "previous_objective_score": float(row["objective_score"]),
                    "additional_iterations": 0,
                    "cumulative_iterations": int(
                        source_state_by_key[key]["iterations"]
                    ),
                }
            )
            retained_summary_rows.append(annotated)
        final_summary_rows = retained_summary_rows + summary_rows
        final_decision_rows = _filter_records(
            continuation_source["decision_rows"], retained_source_keys
        ) + decision_rows
        retained_yearly = _filter_yearly_frame(
            continuation_source["yearly_frames"][0], retained_source_keys
        )
        final_yearly_frames = [retained_yearly] + yearly_frames
    else:
        final_summary_rows = summary_rows
        final_decision_rows = decision_rows
        final_yearly_frames = yearly_frames

    _order_output_frame(pd.DataFrame(final_summary_rows)).to_csv(
        final_paths[0], index=False, encoding="utf-8-sig"
    )
    _order_output_frame(pd.DataFrame(final_decision_rows)).to_csv(
        final_paths[1], index=False, encoding="utf-8-sig"
    )
    _order_output_frame(pd.concat(final_yearly_frames, ignore_index=True)).to_csv(
        final_paths[2], index=False, encoding="utf-8-sig"
    )
    print(f"Saved: {OUTPUT_DIR / stem}")


if __name__ == "__main__":
    main()
