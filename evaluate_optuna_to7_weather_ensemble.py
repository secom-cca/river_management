"""Evaluate fixed Optuna policies under shared annual weather permutations.

Default: 50 random permutations plus a separately reported original-order
control, present/2C/4C, and the existing representative policies. Raw case
caches are written atomically after every completed model evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd

from breeding_response_to7 import model_configuration
from evaluate_optuna_to7_all_pareto import load_pareto
from evaluate_optuna_to7_cross_scenario import OBJECTIVES, RAW_METRICS, SCENARIO_ORDER, decision_from_row, latest_run
from weather_ensemble_to7 import DATES, YEARS, evaluate_weather_policy, extract_weather, permute_weather, year_permutations


_SOURCE = None
_ORDERS = {}
_WEATHER_CACHE = {}
BASELINE_ID = "no_policy"
SCHEMA = 1
WEATHER_BOOKS = ("data/jma_hita_2009_2023.xlsx", "data/jma_asakura_2009_2023.xlsx")


def atomic_json(path: Path, value: dict | list) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature() -> dict:
    return {"model_configuration": model_configuration(),
            "weather_books": {name: file_hash(Path(name)) for name in WEATHER_BOOKS},
            "implementation": {name: file_hash(Path(name)) for name in
                ("weather_ensemble_to7.py", "evaluate_optuna_to7_weather_ensemble.py")}}


def worker_init(source_path: str, sequences: list[dict]) -> None:
    global _SOURCE, _ORDERS, _WEATHER_CACHE
    _SOURCE = pd.read_csv(source_path, index_col="date", parse_dates=["date"], float_precision="round_trip")
    _ORDERS = {row["sequence_id"]: row["source_years"] for row in sequences}
    _WEATHER_CACHE = {}


def worker(task: dict) -> dict:
    sequence = task["sequence_id"]
    if sequence not in _WEATHER_CACHE:
        if len(_WEATHER_CACHE) >= 2:
            _WEATHER_CACHE.clear()
        _WEATHER_CACHE[sequence] = permute_weather(_SOURCE, _ORDERS[sequence])[0]
    started = time.perf_counter()
    metrics, yearly = evaluate_weather_policy(task["target_scenario"], task["decision"], _WEATHER_CACHE[sequence])
    return {"key": list(case_key(task)), "metrics": {name: float(metrics[name]) for name in RAW_METRICS},
            "yearly": yearly.to_dict("records"), "elapsed_seconds": time.perf_counter() - started}


def case_key(task: dict) -> tuple[str, str, str]:
    return task["sequence_id"], task["target_scenario"], task["source_solution_id"]


def case_path(output: Path, key: tuple) -> Path:
    if any(not str(part).replace("_", "").isalnum() for part in key):
        raise ValueError("Unsafe case identifier.")
    return output / "cases" / ("__".join(key) + ".json")


def read_case(output: Path, key: tuple) -> dict | None:
    path = case_path(output, key)
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["key"] != list(key) or any(not math.isfinite(float(record["metrics"][k])) for k in RAW_METRICS):
        raise ValueError(f"Invalid cached model metrics: {path}")
    years = record["yearly"]
    if len(years) != 15 or {int(row["year"]) for row in years} != set(YEARS):
        raise ValueError(f"Invalid cached annual records: {path}")
    return record


def select_policies(run: Path, selection: str, identifiers: list[str] | None):
    policies, decisions = load_pareto(run)
    if identifiers:
        selected = policies.loc[policies["solution_id"].isin(identifiers)].copy()
        if len(selected) != len(set(identifiers)):
            raise ValueError("Requested policy IDs are absent from the source Pareto exports.")
    elif selection == "all":
        selected = policies.copy()
    elif selection == "candidates":
        path = Path("analysis") / run.name / "paper_robustness" / "final_candidates.csv"
        chosen = pd.read_csv(path)["source_solution_id"]
        selected = policies.loc[policies["solution_id"].isin(chosen)].copy()
    else:
        from analyze_optuna_to7_representatives import select_representatives
        selected = pd.concat([select_representatives(policies.loc[policies["scenario"].eq(s)])
                              for s in SCENARIO_ORDER], ignore_index=True)
    if selected.empty:
        raise ValueError("No policy was selected.")
    selected = selected.reset_index(drop=True)
    subset = decisions.set_index("solution_id").loc[selected["solution_id"]].reset_index()
    return selected, subset


def validate_identity(output: Path, run: Path, policies: pd.DataFrame) -> pd.DataFrame:
    reference = json.loads((run / "baseline.json").read_text(encoding="utf-8"))
    rows = []
    checks = [(BASELINE_ID, s, reference[s]) for s in SCENARIO_ORDER]
    checks += [(row["solution_id"], row["scenario"], row) for row in policies.to_dict("records")]
    for solution, scenario, expected in checks:
        record = read_case(output, ("original", scenario, solution))
        if record is None:
            raise RuntimeError("Original-order validation is not complete.")
        for name in RAW_METRICS:
            old, new = float(expected[name]), float(record["metrics"][name])
            ok = bool(np.isclose(old, new, rtol=1e-9, atol=1e-5))
            rows.append({"solution_id": solution, "scenario": scenario, "metric": name,
                         "saved_value": old, "replayed_value": new, "difference": new - old, "passed": ok})
    result = pd.DataFrame(rows)
    result.to_csv(output / "identity_validation.csv", index=False, encoding="utf-8-sig")
    if not result["passed"].all():
        raise RuntimeError("Original-order results differ from saved results. See identity_validation.csv.")
    return result


def export_results(output: Path, sequences: list[dict], policies: pd.DataFrame, config: dict) -> None:
    from run_vensim_with_pysd_to7_optuna import BASELINE_TO_METRIC
    baseline_rows, policy_rows, annual_rows = [], [], []
    for sequence in sequences:
        for scenario in SCENARIO_ORDER:
            base = read_case(output, (sequence["sequence_id"], scenario, BASELINE_ID))
            if base is None:
                raise RuntimeError("Missing baseline case.")
            base_metrics = base["metrics"]
            baseline_rows.append({"sequence_id": sequence["sequence_id"], "is_original": sequence["is_original"],
                                  "target_scenario": scenario, **base_metrics})
            for policy in policies.to_dict("records"):
                record = read_case(output, (sequence["sequence_id"], scenario, policy["solution_id"]))
                if record is None:
                    raise RuntimeError("Missing policy case.")
                metrics = record["metrics"]
                keys = {"sequence_id": sequence["sequence_id"], "is_original": sequence["is_original"],
                        "source_solution_id": policy["solution_id"], "source_scenario": policy["scenario"],
                        "target_scenario": scenario}
                row = {**keys, **metrics, "elapsed_seconds": record["elapsed_seconds"]}
                for objective in OBJECTIVES:
                    metric = BASELINE_TO_METRIC[objective]
                    denominator = base_metrics[metric]
                    row[objective] = metrics[metric] / denominator if abs(denominator) > 1e-12 else np.nan
                    row[f"baseline_{metric}"] = denominator
                    row[f"ratio_defined_{metric}"] = abs(denominator) > 1e-12
                row["annual_budget_utilisation"] = metrics["max_annual_policy_additional_cost"] / config["annual_budget"]
                row["total_budget_utilisation"] = metrics["policy_additional_cost"] / config["total_budget"]
                row["feasible"] = bool(row["annual_budget_utilisation"] <= 1 + 1e-12 and
                                       row["total_budget_utilisation"] <= 1 + 1e-12)
                policy_rows.append(row)
                annual_rows.extend([{**keys, **r} for r in record["yearly"]])
    pd.DataFrame(baseline_rows).to_csv(output / "baselines.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(policy_rows).to_csv(output / "ensemble_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual_rows).to_csv(output / "ensemble_yearly.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--selection", choices=["representatives", "candidates", "all"])
    parser.add_argument("--policy-ids", nargs="+")
    parser.add_argument("--n-sequences", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--smoke", action="store_true", help="One permutation + original, two policies, all three climates.")
    args = parser.parse_args()
    if args.workers < 1 or (args.n_sequences is not None and args.n_sequences < 1):
        parser.error("Workers and random-sequence counts must be positive.")
    if args.resume_dir and (args.output_dir or args.smoke):
        parser.error("--resume-dir cannot be combined with --output-dir or --smoke.")

    if args.resume_dir:
        output = args.resume_dir.resolve()
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if config["schema_version"] != SCHEMA or config["signature"] != signature():
            raise ValueError("Model, weather inputs or ensemble implementation changed; create a new ensemble run.")
        run = Path(config["source_run"])
        for name in ("selection", "n_sequences", "seed"):
            if getattr(args, name) is not None and getattr(args, name) != config[name]:
                raise ValueError(f"Resume option differs from the saved ensemble: {name}")
        if args.run_dir and args.run_dir.resolve() != run:
            raise ValueError("Resume source run differs.")
        if args.policy_ids and args.policy_ids != config["requested_policy_ids"]:
            raise ValueError("Resume policy selection differs.")
        for name, checksum in config["snapshot_sha256"].items():
            if file_hash(output / name) != checksum:
                raise ValueError(f"Saved ensemble input changed: {name}")
        policies = pd.read_csv(output / "selected_policies.csv")
        decisions = pd.read_csv(output / "selected_decisions.csv")
        sequences = json.loads((output / "sequences.json").read_text(encoding="utf-8"))
    else:
        run = (args.run_dir or latest_run(Path("data"))).resolve()
        source_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if source_config.get("model_configuration") != model_configuration():
            raise ValueError("Source policies were optimized with a different model; use the corrected-model run.")
        status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
        if status.get("state") != "complete" or set(source_config["scenarios"]) != set(SCENARIO_ORDER):
            raise ValueError("Expected a completed present/2C/4C optimization run.")
        selection = args.selection or "representatives"
        policies, decisions = select_policies(run, selection, args.policy_ids)
        count = 1 if args.smoke else args.n_sequences or 50
        if args.smoke:
            policies = policies.iloc[:2].copy()
            decisions = decisions.loc[decisions["solution_id"].isin(policies["solution_id"])].copy()
        sequences = year_permutations(count, args.seed if args.seed is not None else 42)
        from run_vensim_with_pysd_to7_opt import get_model
        source = extract_weather(get_model())
        stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%y%m%d_%H%M%S")
        output = (args.output_dir or Path("analysis") / run.name /
                  f"weather_ensemble_{stamp}{'_smoke' if args.smoke else ''}").resolve()
        output.mkdir(parents=True, exist_ok=False)
        (output / "cases").mkdir()
        source.to_csv(output / "source_weather.csv", index_label="date", encoding="utf-8-sig")
        policies.to_csv(output / "selected_policies.csv", index=False, encoding="utf-8-sig")
        decisions.to_csv(output / "selected_decisions.csv", index=False, encoding="utf-8-sig")
        atomic_json(output / "sequences.json", sequences)
        mapping, audits = [], []
        for sequence in sequences:
            _, audit = permute_weather(source, sequence["source_years"])
            audits.append({"sequence_id": sequence["sequence_id"], **audit})
            mapping.extend([{"sequence_id": sequence["sequence_id"], "target_year": t, "source_year": s}
                            for t, s in zip(YEARS, sequence["source_years"])])
        pd.DataFrame(mapping).to_csv(output / "year_mapping.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(audits).to_csv(output / "calendar_audit.csv", index=False, encoding="utf-8-sig")
        config = {"schema_version": SCHEMA, "source_run": str(run), "signature": signature(),
                  "selection": selection, "requested_policy_ids": args.policy_ids,
                  "n_sequences": count, "seed": args.seed if args.seed is not None else 42,
                  "annual_budget": source_config["annual_budget"], "total_budget": source_config["total_budget"],
                  "original_excluded_from_ensemble_statistics": True,
                  "calendar_method": "month-day mapping; Feb29 insertion is mean of adjacent days; source Feb29 dropped in nonleap target",
                  "snapshot_sha256": {name: file_hash(output / name) for name in
                      ("source_weather.csv", "selected_policies.csv", "selected_decisions.csv", "sequences.json")}}
        atomic_json(output / "config.json", config)

    decision_map = {row["solution_id"]: decision_from_row(row) for row in decisions.to_dict("records")}
    from run_vensim_with_pysd_to7_optuna import _zero_decision
    zero = _zero_decision()
    baseline_tasks = [{"sequence_id": seq["sequence_id"], "target_scenario": s,
                       "source_solution_id": BASELINE_ID, "decision": zero}
                      for seq in sequences for s in SCENARIO_ORDER]
    policy_tasks = [{"sequence_id": seq["sequence_id"], "target_scenario": s,
                     "source_solution_id": row["solution_id"], "decision": decision_map[row["solution_id"]]}
                    for seq in sequences for s in SCENARIO_ORDER for row in policies.to_dict("records")]
    expected_count = len(baseline_tasks) + len(policy_tasks)
    state = {"state": "running", "policy_count": len(policies), "random_sequence_count": config["n_sequences"],
             "expected_model_cases": expected_count, "workers": args.workers}
    atomic_json(output / "run_status.json", state)
    print(f"Ensemble directory: {output}", flush=True)
    print(f"{len(policies)} policies; {config['n_sequences']} random sequences + original; {expected_count} model cases.", flush=True)
    pool = None
    completed_now = 0
    try:
        if args.workers > 1:
            pool = multiprocessing.get_context("spawn").Pool(args.workers, initializer=worker_init,
                         initargs=(str(output / "source_weather.csv"), sequences))
        else:
            worker_init(str(output / "source_weather.csv"), sequences)

        def evaluate(tasks: list[dict], label: str) -> None:
            nonlocal completed_now
            pending = [task for task in tasks if read_case(output, case_key(task)) is None]
            print(f"{label}: {len(pending)} pending / {len(tasks)} cases.", flush=True)
            iterator = pool.imap_unordered(worker, pending) if pool else map(worker, pending)
            for number, record in enumerate(iterator, start=1):
                atomic_json(case_path(output, tuple(record["key"])), record)
                completed_now += 1
                if number % 5 == 0 or number == len(pending):
                    print(f"{label}: completed {number}/{len(pending)}.", flush=True)
                    state["new_cases_completed"] = completed_now
                    atomic_json(output / "run_status.json", state)

        evaluate([task for task in baseline_tasks if task["sequence_id"] == "original"], "Original baselines")
        own_scenario = dict(zip(policies["solution_id"], policies["scenario"]))
        evaluate([task for task in policy_tasks if task["sequence_id"] == "original" and
                  task["target_scenario"] == own_scenario[task["source_solution_id"]]], "Identity policies")
        validation = validate_identity(output, run, policies)
        print(f"Original-order validation passed: {len(validation)} metric checks.", flush=True)
        evaluate(baseline_tasks, "Ensemble baselines")
        evaluate(policy_tasks, "Policy evaluations")
        if pool:
            pool.close()
            pool.join()
            pool = None
        export_results(output, sequences, policies, config)
        state["state"] = "complete"
        state["completed_model_cases"] = expected_count
        atomic_json(output / "run_status.json", state)
    except BaseException as exc:
        if pool:
            pool.terminate()
            pool.join()
        state["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        atomic_json(output / "run_status.json", state)
        raise

    from analyze_optuna_to7_weather_ensemble import analyze
    analyze(output)
    print(f"Saved weather-ensemble evaluation and plots: {output}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
