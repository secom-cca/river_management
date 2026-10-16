"""Constrained five-objective optimisation of the to7 PySD model with NSGA-III.

The existing differential-evolution runner remains the source of model and
aggregation semantics. This module changes only the optimiser-facing policy
representation: years are integers, intervals are start plus duration, and
breeding is conditional on a Boolean choice.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from breeding_response_to7 import model_configuration


SCHEMA_VERSION = 2
CALENDAR_START_YEAR = 2009
CALENDAR_NUM_YEARS = 15
CALENDAR_END_YEAR = CALENDAR_START_YEAR + CALENDAR_NUM_YEARS - 1
BREEDING_DURATION_YEARS = 10
SCENARIO_ORDER = ("present", "2C", "4C")
OBJECTIVE_NAMES = (
    "flood_damage_ratio",
    "innundation_damage_ratio",
    "landslide_risk_ratio",
    "crop_revenue_ratio",
    "biodiversity_ratio",
)
DIRECTIONS = ("minimize", "minimize", "minimize", "maximize", "maximize")
RAW_METRICS = (
    "flood_damage",
    "innundation_damage",
    "landslide_risk",
    "crop_revenue",
    "biodiversity",
    "daily_total_gdp",
    "policy_additional_cost",
    "max_annual_policy_additional_cost",
)
ANNUAL_BUDGET = 5_000_000_000.0
TOTAL_BUDGET = 50_000_000_000.0
INTERVAL_PREFIXES = (
    "house_elevation",
    "migration",
    "paddy_dam",
    "forest_management",
)
ONE_TIME_PREFIXES = ("drainage", "dam", "levee")
CONTINUOUS_BOUNDS = {
    "drainage_investment_amount": (0.0, 5_000_000_000.0),
    "dam_investment_amount": (0.0, 5_000_000_000.0),
    "levee_investment_amount": (0.0, 5_000_000_000.0),
    "house_elevation_per_year": (0.0, 1_000.0),
    "migration_per_year": (0.0, 1_000.0),
    "annual_paddy_dam_investment": (0.0, 5_000_000_000.0),
    "annual_forest_management_conversion_area": (0.0, 3_000.0),
}
BASELINE_TO_METRIC = {
    "flood_damage_ratio": "flood_damage",
    "innundation_damage_ratio": "innundation_damage",
    "landslide_risk_ratio": "landslide_risk",
    "crop_revenue_ratio": "crop_revenue",
    "biodiversity_ratio": "biodiversity",
}


@dataclass(frozen=True)
class WorkerSpec:
    run_dir: str
    scenario: str
    study_name: str
    journal_path: str
    baseline: dict[str, float]
    n_trials: int
    population_size: int
    dividing_parameter: int
    seed: int
    worker_index: int


def _import_optuna():
    try:
        return importlib.import_module("optuna")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Optuna is required. Install requirements-optuna.txt in the "
            "Python environment used to run this script."
        ) from exc


def _import_core():
    return importlib.import_module("run_vensim_with_pysd_to7_opt")


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value cannot be stored: {value}")
        return value
    if hasattr(value, "item"):
        return _json_scalar(value.item())
    return str(value)


def policy_params_to_decision(params: Mapping[str, Any]) -> dict[str, Any]:
    """Convert the unique mixed-variable representation to a core decision."""
    decision: dict[str, Any] = {}
    for name in CONTINUOUS_BOUNDS:
        value = float(params[name])
        lower, upper = CONTINUOUS_BOUNDS[name]
        if not lower <= value <= upper:
            raise ValueError(f"{name}={value} is outside [{lower}, {upper}].")
        decision[name] = value

    for prefix in ONE_TIME_PREFIXES:
        year = int(params[f"{prefix}_start_year"])
        _validate_year_index(year, f"{prefix}_start_year")
        decision[f"{prefix}_start_year"] = year

    for prefix in INTERVAL_PREFIXES:
        start = int(params[f"{prefix}_start_year"])
        duration = int(params[f"{prefix}_duration_years"])
        _validate_year_index(start, f"{prefix}_start_year")
        if duration < 1 or start + duration > CALENDAR_NUM_YEARS:
            raise ValueError(
                f"{prefix} interval start={start}, duration={duration} "
                f"must finish by index {CALENDAR_NUM_YEARS - 1}."
            )
        decision[f"{prefix}_start_year"] = start
        decision[f"{prefix}_end_year"] = start + duration - 1

    breeding_enabled = bool(params["breeding_enabled"])
    decision["breeding_enabled"] = int(breeding_enabled)
    if breeding_enabled:
        breeding_start = int(params["breeding_start_year"])
        if not 0 <= breeding_start <= CALENDAR_NUM_YEARS - BREEDING_DURATION_YEARS:
            raise ValueError("breeding_start_year does not fit the ten-year period.")
    else:
        if "breeding_start_year" in params:
            raise ValueError(
                "breeding_start_year must be absent when breeding is disabled."
            )
        breeding_start = 0
    decision["breeding_start_year"] = breeding_start
    decision["breeding_end_year"] = breeding_start + BREEDING_DURATION_YEARS - 1
    return decision


def _validate_year_index(value: int, name: str) -> None:
    if not 0 <= value < CALENDAR_NUM_YEARS:
        raise ValueError(f"{name}={value} is outside 0..{CALENDAR_NUM_YEARS - 1}.")


def decision_to_policy_params(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a decoded DE decision into the unique Optuna representation."""
    params = {name: float(decision[name]) for name in CONTINUOUS_BOUNDS}
    for prefix in ONE_TIME_PREFIXES:
        params[f"{prefix}_start_year"] = int(decision[f"{prefix}_start_year"])
    for prefix in INTERVAL_PREFIXES:
        start = int(decision[f"{prefix}_start_year"])
        end = int(decision[f"{prefix}_end_year"])
        if end < start:
            raise ValueError(f"DE decision has an inverted {prefix} interval.")
        params[f"{prefix}_start_year"] = start
        params[f"{prefix}_duration_years"] = end - start + 1
    enabled = bool(int(float(decision["breeding_enabled"])))
    params["breeding_enabled"] = enabled
    if enabled:
        params["breeding_start_year"] = int(decision["breeding_start_year"])
    policy_params_to_decision(params)
    return params


def suggest_policy(trial: Any) -> dict[str, Any]:
    """Suggest one policy without continuous rounding or interval aliases."""
    params: dict[str, Any] = {}
    for name, (lower, upper) in CONTINUOUS_BOUNDS.items():
        params[name] = trial.suggest_float(name, lower, upper)
    for prefix in ONE_TIME_PREFIXES:
        name = f"{prefix}_start_year"
        params[name] = trial.suggest_int(name, 0, CALENDAR_NUM_YEARS - 1)
    for prefix in INTERVAL_PREFIXES:
        start_name = f"{prefix}_start_year"
        duration_name = f"{prefix}_duration_years"
        start = trial.suggest_int(start_name, 0, CALENDAR_NUM_YEARS - 1)
        params[start_name] = start
        params[duration_name] = trial.suggest_int(
            duration_name, 1, CALENDAR_NUM_YEARS - start
        )
    enabled = trial.suggest_categorical("breeding_enabled", [False, True])
    params["breeding_enabled"] = enabled
    if enabled:
        params["breeding_start_year"] = trial.suggest_int(
            "breeding_start_year",
            0,
            CALENDAR_NUM_YEARS - BREEDING_DURATION_YEARS,
        )
    return params


def objective_ratios(
    metrics: Mapping[str, float], baseline: Mapping[str, float]
) -> tuple[float, ...]:
    values = []
    for objective_name in OBJECTIVE_NAMES:
        metric_name = BASELINE_TO_METRIC[objective_name]
        base = float(baseline[metric_name])
        if abs(base) <= 1e-12:
            raise ValueError(f"Baseline {metric_name} is zero; ratio is undefined.")
        ratio = float(metrics[metric_name]) / base
        if not math.isfinite(ratio):
            raise ValueError(f"Non-finite objective ratio for {metric_name}.")
        values.append(ratio)
    return tuple(values)


def budget_constraints(metrics: Mapping[str, float]) -> tuple[float, float]:
    """Return normalized violations; positive values violate the constraint."""
    values = (
        float(metrics["max_annual_policy_additional_cost"]) / ANNUAL_BUDGET - 1.0,
        float(metrics["policy_additional_cost"]) / TOTAL_BUDGET - 1.0,
    )
    return tuple(0.0 if abs(value) <= 1e-12 else value for value in values)


def constraints_func(frozen_trial: Any) -> tuple[float, float]:
    attrs = frozen_trial.user_attrs
    return (
        float(attrs["annual_budget_constraint"]),
        float(attrs["total_budget_constraint"]),
    )


def set_trial_constraints(trial: Any, annual: float, total: float) -> None:
    """Use Optuna 5 named constraints; Optuna 4 reads mirrored user attrs."""
    if hasattr(trial, "set_constraint"):
        trial.set_constraint("annual_budget", annual)
        trial.set_constraint("total_budget", total)


def is_feasible_attrs(attrs: Mapping[str, Any], tolerance: float = 1e-12) -> bool:
    try:
        return (
            float(attrs["annual_budget_constraint"]) <= tolerance
            and float(attrs["total_budget_constraint"]) <= tolerance
        )
    except (KeyError, TypeError, ValueError):
        return False


def split_trial_quota(total: int, workers: int) -> list[int]:
    if total < 0 or workers < 1:
        raise ValueError("total must be nonnegative and workers must be positive.")
    base, remainder = divmod(total, workers)
    return [base + (index < remainder) for index in range(workers)]


def _study_storage(optuna: Any, journal_path: str):
    journal = optuna.storages.journal
    lock = None
    if os.name == "nt" and hasattr(journal, "JournalFileOpenLock"):
        lock = journal.JournalFileOpenLock(journal_path)
    backend = journal.JournalFileBackend(journal_path, lock_obj=lock)
    return optuna.storages.JournalStorage(backend)


def _make_sampler(optuna: Any, population_size: int, dividing_parameter: int, seed: int):
    sampler_kwargs = {
        "population_size": population_size,
        "dividing_parameter": dividing_parameter,
        "seed": seed,
    }
    # Optuna 5 records named constraints through Trial.set_constraint().
    # Optuna 4.9 needs the sampler callback that reads the mirrored user attrs.
    if not hasattr(optuna.trial.Trial, "set_constraint"):
        sampler_kwargs["constraints_func"] = constraints_func
    with warnings.catch_warnings():
        experimental = getattr(optuna.exceptions, "ExperimentalWarning", Warning)
        warnings.simplefilter("ignore", experimental)
        return optuna.samplers.NSGAIIISampler(**sampler_kwargs)


def _study(
    optuna: Any,
    study_name: str,
    journal_path: str,
    population_size: int,
    dividing_parameter: int,
    seed: int,
):
    storage = _study_storage(optuna, journal_path)
    sampler = _make_sampler(optuna, population_size, dividing_parameter, seed)
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        directions=list(DIRECTIONS),
        load_if_exists=True,
    )


def _atomic_dataframe_csv(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _trial_yearly_path(run_dir: Path, scenario: str, trial_number: int) -> Path:
    return run_dir / "trial_yearly" / scenario / f"trial_{trial_number:08d}.csv"


def _objective(
    trial: Any,
    scenario: str,
    baseline: Mapping[str, float],
    run_dir: str,
    study_name: str,
    worker_index: int,
    worker_seed: int,
):
    started = time.perf_counter()
    params = suggest_policy(trial)
    decision = policy_params_to_decision(params)
    core = _import_core()
    metrics, yearly = core.evaluate_policy(scenario, decision)
    ratios = objective_ratios(metrics, baseline)
    annual_constraint, total_constraint = budget_constraints(metrics)
    set_trial_constraints(trial, annual_constraint, total_constraint)

    attrs: dict[str, Any] = {
        "scenario": scenario,
        "annual_budget_constraint": annual_constraint,
        "total_budget_constraint": total_constraint,
        "annual_budget_utilisation": annual_constraint + 1.0,
        "total_budget_utilisation": total_constraint + 1.0,
        "feasible": annual_constraint <= 1e-12 and total_constraint <= 1e-12,
        "elapsed_seconds": time.perf_counter() - started,
        "worker_pid": os.getpid(),
        "worker_index": worker_index,
        "worker_seed": worker_seed,
        "study_name": study_name,
    }
    attrs.update({name: float(value) for name, value in zip(OBJECTIVE_NAMES, ratios)})
    attrs.update({name: float(metrics[name]) for name in RAW_METRICS})
    for prefix in INTERVAL_PREFIXES:
        attrs[f"{prefix}_end_year"] = int(decision[f"{prefix}_end_year"])
    attrs["breeding_end_year"] = (
        int(decision["breeding_end_year"]) if decision["breeding_enabled"] else None
    )
    for name, value in attrs.items():
        trial.set_user_attr(name, _json_scalar(value))

    yearly = yearly.copy()
    yearly.insert(0, "trial_number", trial.number)
    yearly.insert(0, "scenario", scenario)
    _atomic_dataframe_csv(
        yearly,
        _trial_yearly_path(Path(run_dir), scenario, trial.number),
    )
    return ratios


def _worker_run(spec: WorkerSpec) -> int:
    optuna = _import_optuna()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = _study(
        optuna,
        spec.study_name,
        spec.journal_path,
        spec.population_size,
        spec.dividing_parameter,
        spec.seed,
    )

    def objective(trial: Any):
        return _objective(
            trial,
            spec.scenario,
            spec.baseline,
            spec.run_dir,
            spec.study_name,
            spec.worker_index,
            spec.seed,
        )

    def progress(_study: Any, trial: Any) -> None:
        if trial.state.name == "COMPLETE":
            print(
                f"{spec.scenario}: trial={trial.number} complete "
                f"worker={spec.worker_index} pid={os.getpid()}",
                flush=True,
            )

    study.optimize(objective, n_trials=spec.n_trials, callbacks=[progress])
    return spec.n_trials


def _completed_trials(study: Any) -> list[Any]:
    return [trial for trial in study.trials if trial.state.name == "COMPLETE"]


def _failed_trials(study: Any) -> list[Any]:
    return [trial for trial in study.trials if trial.state.name == "FAIL"]


def _run_parallel(specs: Sequence[WorkerSpec]) -> None:
    active = [spec for spec in specs if spec.n_trials > 0]
    if not active:
        return
    if len(active) == 1:
        _worker_run(active[0])
        return
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(processes=len(active))
    try:
        pool.map(_worker_run, active)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def _latest_complete_de_results(data_dir: Path) -> tuple[str, Path, Path, Path]:
    candidates: list[tuple[float, str, Path, Path, Path]] = []
    for summary in data_dir.glob("opt_to7_standard_*_summary.csv"):
        stem = summary.name[: -len("_summary.csv")]
        decisions = data_dir / f"{stem}_decisions.csv"
        yearly = data_dir / f"{stem}_yearly.csv"
        if not decisions.exists() or not yearly.exists():
            continue
        try:
            _validate_de_files(summary, decisions, yearly)
        except (ValueError, OSError):
            continue
        candidates.append((summary.stat().st_mtime, stem, summary, decisions, yearly))
    if not candidates:
        raise FileNotFoundError("No complete 12-case DE output was found in data/.")
    _, stem, summary, decisions, yearly = max(candidates, key=lambda item: item[0])
    return stem, summary, decisions, yearly


def _de_results_for_stamp(data_dir: Path, stamp_or_path: str):
    supplied = Path(stamp_or_path)
    if supplied.exists():
        if supplied.name.endswith("_summary.csv"):
            summary = supplied
            stem = supplied.name[: -len("_summary.csv")]
            root = supplied.parent
        elif supplied.is_dir():
            matches = list(supplied.glob("opt_to7_standard_*_summary.csv"))
            if len(matches) != 1:
                raise ValueError("DE directory must contain exactly one matching summary CSV.")
            summary = matches[0]
            stem = summary.name[: -len("_summary.csv")]
            root = supplied
        else:
            raise ValueError("DE source path must be a summary CSV or directory.")
    else:
        matches = list(data_dir.glob(f"opt_to7_standard_*_{stamp_or_path}_summary.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Could not identify one DE run for {stamp_or_path!r}.")
        summary = matches[0]
        stem = summary.name[: -len("_summary.csv")]
        root = data_dir
    decisions = root / f"{stem}_decisions.csv"
    yearly = root / f"{stem}_yearly.csv"
    _validate_de_files(summary, decisions, yearly)
    return stem, summary, decisions, yearly


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_de_files(summary: Path, decisions: Path, yearly: Path) -> None:
    if not decisions.exists() or not yearly.exists():
        raise ValueError("DE summary, decisions, and yearly files must all exist.")
    summary_rows = _read_csv_rows(summary)
    decision_rows = _read_csv_rows(decisions)
    yearly_rows = _read_csv_rows(yearly)
    expected = {
        (scenario, pattern, "42")
        for scenario in SCENARIO_ORDER
        for pattern in (
            "balanced",
            "disaster_only",
            "agriculture_only",
            "ecosystem_only",
        )
    }
    key = lambda row: (row["scenario"], row["pattern"], str(row["seed"]))
    summary_keys = {key(row) for row in summary_rows}
    decision_keys = {key(row) for row in decision_rows}
    yearly_keys = {key(row) for row in yearly_rows}
    if len(summary_rows) != 12 or len(decision_rows) != 12 or len(yearly_rows) != 180:
        raise ValueError("DE files are not the expected 12 cases x 15 years.")
    if not summary_keys == decision_keys == yearly_keys == expected:
        raise ValueError("DE file keys are incomplete or inconsistent.")
    counts: dict[tuple[str, str, str], set[str]] = {}
    for row in yearly_rows:
        counts.setdefault(key(row), set()).add(row["year"])
    if any(len(years) != CALENDAR_NUM_YEARS for years in counts.values()):
        raise ValueError("Each DE case must contain 15 distinct years.")


def _de_seed_params(decisions_path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped = {scenario: [] for scenario in SCENARIO_ORDER}
    rows = _read_csv_rows(decisions_path)
    pattern_order = {
        "balanced": 0,
        "disaster_only": 1,
        "agriculture_only": 2,
        "ecosystem_only": 3,
    }
    rows.sort(key=lambda row: (SCENARIO_ORDER.index(row["scenario"]), pattern_order[row["pattern"]]))
    for row in rows:
        decision: dict[str, Any] = {}
        for name in CONTINUOUS_BOUNDS:
            decision[name] = float(row[name])
        for prefix in ONE_TIME_PREFIXES:
            decision[f"{prefix}_start_year"] = int(float(row[f"{prefix}_start_year"]))
        for prefix in INTERVAL_PREFIXES:
            decision[f"{prefix}_start_year"] = int(float(row[f"{prefix}_start_year"]))
            decision[f"{prefix}_end_year"] = int(float(row[f"{prefix}_end_year"]))
        decision["breeding_enabled"] = int(float(row["breeding_enabled"]))
        decision["breeding_start_year"] = int(float(row["breeding_start_year"]))
        grouped[row["scenario"]].append(decision_to_policy_params(decision))
    return grouped


def _zero_decision() -> dict[str, Any]:
    params: dict[str, Any] = {name: 0.0 for name in CONTINUOUS_BOUNDS}
    for prefix in ONE_TIME_PREFIXES:
        params[f"{prefix}_start_year"] = 0
    for prefix in INTERVAL_PREFIXES:
        params[f"{prefix}_start_year"] = 0
        params[f"{prefix}_duration_years"] = 1
    params["breeding_enabled"] = False
    return policy_params_to_decision(params)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("optuna", "pysd", "numpy", "pandas", "scipy", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _write_baselines(run_dir: Path, baselines: Mapping[str, Mapping[str, float]]) -> None:
    _atomic_json(run_dir / "baseline.json", baselines)
    fieldnames = ["scenario", *RAW_METRICS]
    with (run_dir / "baseline.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, metrics in baselines.items():
            writer.writerow({"scenario": scenario, **{name: metrics[name] for name in RAW_METRICS}})


def _load_baselines(run_dir: Path) -> dict[str, dict[str, float]]:
    raw = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    return {
        scenario: {name: float(value) for name, value in metrics.items()}
        for scenario, metrics in raw.items()
    }


def _new_config(args: argparse.Namespace, run_dir: Path, de_source: str | None):
    return {
        "schema_version": SCHEMA_VERSION,
        "model_configuration": model_configuration(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "scenarios": list(args.scenarios),
        "objectives": list(OBJECTIVE_NAMES),
        "directions": list(DIRECTIONS),
        "annual_budget": ANNUAL_BUDGET,
        "total_budget": TOTAL_BUDGET,
        "sampler": "NSGAIIISampler",
        "population_size": args.population_size,
        "dividing_parameter": args.dividing_parameter,
        "seed": args.seed,
        "worker_seed_rule": "seed + completed_trials_before_round + worker_index + 1",
        "de_seed_source": de_source,
        "packages": _package_versions(),
    }


def _validate_resume_config(config: Mapping[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "model_configuration": model_configuration(),
        "scenarios": list(args.scenarios),
        "objectives": list(OBJECTIVE_NAMES),
        "directions": list(DIRECTIONS),
        "annual_budget": ANNUAL_BUDGET,
        "total_budget": TOTAL_BUDGET,
        "sampler": "NSGAIIISampler",
        "population_size": args.population_size,
        "dividing_parameter": args.dividing_parameter,
        "seed": args.seed,
    }
    mismatches = [name for name, value in expected.items() if config.get(name) != value]
    if mismatches:
        raise ValueError(
            "Resume configuration is incompatible for: " + ", ".join(mismatches)
            + ". Changed models require a new run and a newly calculated baseline."
        )


def _trial_row(trial: Any, scenario: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": scenario,
        "trial_number": trial.number,
        "trial_state": trial.state.name,
        "datetime_start": trial.datetime_start,
        "datetime_complete": trial.datetime_complete,
        "duration_seconds": (
            trial.duration.total_seconds() if trial.duration is not None else None
        ),
    }
    if trial.values is not None:
        row.update({name: value for name, value in zip(OBJECTIVE_NAMES, trial.values)})
    row.update({f"param_{name}": value for name, value in trial.params.items()})
    row.update(trial.user_attrs)
    row["feasible"] = trial.state.name == "COMPLETE" and is_feasible_attrs(trial.user_attrs)
    return row


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _json_scalar(row.get(name)) for name in fields})


def _pareto_trials(study: Any) -> list[Any]:
    return sorted(
        (
            trial
            for trial in study.best_trials
            if trial.state.name == "COMPLETE" and is_feasible_attrs(trial.user_attrs)
        ),
        key=lambda trial: trial.number,
    )


def _decision_output(trial: Any, scenario: str) -> dict[str, Any]:
    decision = policy_params_to_decision(trial.params)
    row: dict[str, Any] = {
        "solution_id": f"{scenario}_trial_{trial.number:08d}",
        "scenario": scenario,
        "trial_number": trial.number,
    }
    row.update(decision)
    for prefix in (*ONE_TIME_PREFIXES, *INTERVAL_PREFIXES):
        start = decision[f"{prefix}_start_year"]
        row[f"{prefix}_start_calendar_year"] = CALENDAR_START_YEAR + start
        if prefix in INTERVAL_PREFIXES:
            end = decision[f"{prefix}_end_year"]
            row[f"{prefix}_end_calendar_year"] = CALENDAR_START_YEAR + end
            row[f"{prefix}_duration_years"] = end - start + 1
    if decision["breeding_enabled"]:
        row["breeding_start_calendar_year"] = CALENDAR_START_YEAR + decision["breeding_start_year"]
        row["breeding_end_calendar_year"] = CALENDAR_START_YEAR + decision["breeding_end_year"]
    else:
        row["breeding_start_year"] = None
        row["breeding_end_year"] = None
        row["breeding_start_calendar_year"] = None
        row["breeding_end_calendar_year"] = None
    return row


def export_study(study: Any, run_dir: Path, scenario: str) -> dict[str, int]:
    trials = sorted(study.trials, key=lambda trial: trial.number)
    trial_rows = [_trial_row(trial, scenario) for trial in trials]
    _write_rows(run_dir / f"trials_{scenario}.csv", trial_rows)

    pareto = _pareto_trials(study)
    summary_rows = []
    decision_rows = []
    yearly_frames = []
    for trial in pareto:
        solution_id = f"{scenario}_trial_{trial.number:08d}"
        summary = _trial_row(trial, scenario)
        summary["solution_id"] = solution_id
        summary_rows.append(summary)
        decision_rows.append(_decision_output(trial, scenario))
        yearly_path = _trial_yearly_path(run_dir, scenario, trial.number)
        if not yearly_path.exists():
            core = _import_core()
            decision = policy_params_to_decision(trial.params)
            _, yearly = core.evaluate_policy(scenario, decision)
            yearly.insert(0, "trial_number", trial.number)
            yearly.insert(0, "scenario", scenario)
            _atomic_dataframe_csv(yearly, yearly_path)
        pandas = importlib.import_module("pandas")
        yearly = pandas.read_csv(yearly_path)
        yearly.insert(0, "solution_id", solution_id)
        yearly_frames.append(yearly)
    _write_rows(run_dir / f"pareto_summary_{scenario}.csv", summary_rows)
    _write_rows(run_dir / f"pareto_decisions_{scenario}.csv", decision_rows)
    yearly_output = run_dir / f"pareto_yearly_{scenario}.csv"
    if yearly_frames:
        pandas = importlib.import_module("pandas")
        _atomic_dataframe_csv(pandas.concat(yearly_frames, ignore_index=True), yearly_output)
    else:
        yearly_output.write_text("", encoding="utf-8-sig")
    return {
        "all": len(trials),
        "complete": len(_completed_trials(study)),
        "failed": len(_failed_trials(study)),
        "pareto": len(pareto),
    }


def _prepare_new_run(args: argparse.Namespace, run_dir: Path):
    optuna = _import_optuna()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "trial_yearly").mkdir()

    de_source = None
    seed_params: dict[str, list[dict[str, Any]]] = {
        scenario: [] for scenario in SCENARIO_ORDER
    }
    if not args.no_de_seed:
        if args.de_results:
            stem, _, decisions, _ = _de_results_for_stamp(Path("data"), args.de_results)
        else:
            stem, _, decisions, _ = _latest_complete_de_results(Path("data"))
        de_source = str(decisions)
        seed_params = _de_seed_params(decisions)
        print(f"DE seed source: {stem}")

    core = _import_core()
    baselines: dict[str, dict[str, float]] = {}
    zero = _zero_decision()
    for scenario in args.scenarios:
        print(f"{scenario}: calculating no-policy baseline...", flush=True)
        metrics, _ = core.evaluate_policy(scenario, zero)
        baselines[scenario] = {name: float(metrics[name]) for name in RAW_METRICS}
    _write_baselines(run_dir, baselines)
    config = _new_config(args, run_dir, de_source)
    _atomic_json(run_dir / "config.json", config)

    for scenario in args.scenarios:
        study_name = f"to7_nsga3_{run_dir.name}_{scenario}"
        journal_path = str((run_dir / f"study_{scenario}.log").resolve())
        study = _study(
            optuna,
            study_name,
            journal_path,
            args.population_size,
            args.dividing_parameter,
            args.seed,
        )
        for params in seed_params[scenario]:
            study.enqueue_trial(params, skip_if_exists=True)
    return config, baselines


def _load_resume(args: argparse.Namespace, run_dir: Path):
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if args.scenarios_were_default:
        args.scenarios = list(config["scenarios"])
    if args.population_was_default:
        args.population_size = int(config["population_size"])
    if args.dividing_was_default:
        args.dividing_parameter = int(config["dividing_parameter"])
    if args.seed_was_default:
        args.seed = int(config["seed"])
    _validate_resume_config(config, args)
    return config, _load_baselines(run_dir)


def _run_scenario(
    args: argparse.Namespace,
    baselines: Mapping[str, Mapping[str, float]],
    run_dir: Path,
    scenario: str,
) -> dict[str, int]:
    optuna = _import_optuna()
    study_name = f"to7_nsga3_{run_dir.name}_{scenario}"
    journal_path = str((run_dir / f"study_{scenario}.log").resolve())
    study = _study(
        optuna,
        study_name,
        journal_path,
        args.population_size,
        args.dividing_parameter,
        args.seed,
    )
    if args.export_only:
        return export_study(study, run_dir, scenario)

    stalled_rounds = 0
    while len(_completed_trials(study)) < args.n_trials:
        complete_before = len(_completed_trials(study))
        remaining = args.n_trials - complete_before
        quotas = split_trial_quota(remaining, min(args.workers, remaining))
        specs = [
            WorkerSpec(
                run_dir=str(run_dir.resolve()),
                scenario=scenario,
                study_name=study_name,
                journal_path=journal_path,
                baseline=dict(baselines[scenario]),
                n_trials=quota,
                population_size=args.population_size,
                dividing_parameter=args.dividing_parameter,
                seed=args.seed + complete_before + worker_index + 1,
                worker_index=worker_index,
            )
            for worker_index, quota in enumerate(quotas)
        ]
        print(
            f"{scenario}: {complete_before}/{args.n_trials} complete; "
            f"running {remaining} trials with {len(specs)} processes.",
            flush=True,
        )
        _run_parallel(specs)
        study = _study(
            optuna,
            study_name,
            journal_path,
            args.population_size,
            args.dividing_parameter,
            args.seed,
        )
        complete_after = len(_completed_trials(study))
        if complete_after == complete_before:
            stalled_rounds += 1
            if stalled_rounds >= 2:
                raise RuntimeError(
                    f"{scenario}: no COMPLETE trials were produced in two rounds."
                )
        else:
            stalled_rounds = 0
        export_study(study, run_dir, scenario)
    return export_study(study, run_dir, scenario)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_ORDER, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1_000,
        help="Target COMPLETE trial count per scenario, including resumed trials.",
    )
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--dividing-parameter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--de-results",
        help="DE timestamp, summary CSV, or directory. Defaults to latest complete run.",
    )
    parser.add_argument("--no-de-seed", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    args.scenarios_were_default = args.scenarios is None
    args.population_was_default = args.population_size is None
    args.dividing_was_default = args.dividing_parameter is None
    args.seed_was_default = args.seed is None
    if args.scenarios is None:
        args.scenarios = list(SCENARIO_ORDER)
    if args.population_size is None:
        args.population_size = 100
    if args.dividing_parameter is None:
        args.dividing_parameter = 3
    if args.seed is None:
        args.seed = 42
    if args.smoke:
        if args.resume_dir:
            parser.error("--smoke cannot be combined with --resume-dir.")
        args.scenarios = ["present"]
        args.n_trials = 10
        args.population_size = 10
    if args.workers < 1:
        parser.error("--workers must be at least 1.")
    if args.n_trials < 1 and not args.export_only:
        parser.error("--n-trials must be at least 1.")
    if args.population_size < 2:
        parser.error("--population-size must be at least 2.")
    if args.dividing_parameter < 1:
        parser.error("--dividing-parameter must be at least 1.")
    if args.export_only and not args.resume_dir:
        parser.error("--export-only requires --resume-dir.")
    if args.resume_dir and (args.de_results or args.no_de_seed):
        parser.error("DE seed options apply only to a new run.")
    return args


def warn_trial_population(args: argparse.Namespace) -> None:
    if args.export_only:
        return
    if args.n_trials < args.population_size:
        print(
            "Warning: n-trials is smaller than population-size; this is suitable "
            "only for a smoke test.",
            file=sys.stderr,
        )
    elif args.n_trials % args.population_size:
        print(
            "Warning: n-trials is not a multiple of population-size; the final "
            "NSGA-III generation will be incomplete.",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    multiprocessing.freeze_support()
    _import_optuna()

    if args.resume_dir:
        run_dir = args.resume_dir.resolve()
        _, baselines = _load_resume(args, run_dir)
    else:
        stamp = datetime.now().strftime("%y%m%d_%H%M%S")
        run_dir = (args.output_root / f"optuna_to7_{stamp}").resolve()
        _, baselines = _prepare_new_run(args, run_dir)

    warn_trial_population(args)
    status_path = run_dir / "run_status.json"
    preserved_state = None
    if args.export_only and status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        preserved_state = status.get("state")
        status["last_action"] = "export_only"
    else:
        status = {
            "run_dir": str(run_dir),
            "target_complete_trials_per_scenario": args.n_trials,
            "workers": args.workers,
            "state": "running",
            "scenarios": {},
        }
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    if not args.export_only:
        status["target_complete_trials_per_scenario"] = args.n_trials
        status["workers"] = args.workers
        status["state"] = "running"
    _atomic_json(run_dir / "run_status.json", status)
    try:
        for scenario in args.scenarios:
            counts = _run_scenario(args, baselines, run_dir, scenario)
            status["scenarios"][scenario] = counts
            status["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _atomic_json(run_dir / "run_status.json", status)
    except KeyboardInterrupt:
        status["state"] = "interrupted"
        status["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(run_dir / "run_status.json", status)
        print(f"Interrupted safely. Resume directory: {run_dir}", file=sys.stderr)
        return 130
    except BaseException as exc:
        status["state"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(run_dir / "run_status.json", status)
        raise
    status["state"] = preserved_state or ("exported" if args.export_only else "complete")
    status["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_json(run_dir / "run_status.json", status)
    print(f"Saved Optuna run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
