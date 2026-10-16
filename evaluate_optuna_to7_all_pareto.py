"""Cross-evaluate every Optuna Pareto policy under every climate scenario.

The script reuses the source-scenario Pareto exports and, when available, the
earlier representative-policy cross evaluation.  Only missing policy/climate
combinations are simulated.  Results are checkpointed and can be resumed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from evaluate_optuna_to7_cross_scenario import (
    EvaluationTask,
    OBJECTIVES,
    RAW_METRICS,
    SCENARIO_ORDER,
    decision_from_row,
    evaluate_task,
    latest_run,
)


RESULT_COLUMNS = (
    "source_solution_id",
    "source_scenario",
    "target_scenario",
    *OBJECTIVES,
    *RAW_METRICS,
    "annual_budget_constraint",
    "total_budget_constraint",
    "annual_budget_utilisation",
    "total_budget_utilisation",
    "feasible",
)
YEARLY_KEY_COLUMNS = ("source_solution_id", "source_scenario", "target_scenario")


def evaluation_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["source_solution_id"]), str(row["target_scenario"])


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def load_pareto(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    decisions = []
    for scenario in SCENARIO_ORDER:
        summary_path = run_dir / f"pareto_summary_{scenario}.csv"
        decision_path = run_dir / f"pareto_decisions_{scenario}.csv"
        if not summary_path.exists() or not decision_path.exists():
            raise FileNotFoundError(f"Missing Pareto exports for {scenario}: {run_dir}")
        summaries.append(pd.read_csv(summary_path))
        decisions.append(pd.read_csv(decision_path))
    policies = pd.concat(summaries, ignore_index=True)
    policy_decisions = pd.concat(decisions, ignore_index=True)
    if policies["solution_id"].duplicated().any():
        raise ValueError("Pareto solution_id values must be unique.")
    if set(policies["solution_id"]) != set(policy_decisions["solution_id"]):
        raise ValueError("Pareto summaries and decisions contain different policies.")
    scenario_by_id = policies.set_index("solution_id")["scenario"]
    mapped = policy_decisions["solution_id"].map(scenario_by_id)
    if not mapped.equals(policy_decisions["scenario"]):
        raise ValueError("Scenario labels disagree between Pareto summaries and decisions.")
    scenario_rank = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    policies["scenario_order"] = policies["scenario"].map(scenario_rank)
    policy_decisions["scenario_order"] = policy_decisions["scenario"].map(scenario_rank)
    policies = policies.sort_values(["scenario_order", "trial_number"]).drop(
        columns="scenario_order"
    ).reset_index(drop=True)
    policy_decisions = policy_decisions.sort_values(
        ["scenario_order", "trial_number"]
    ).drop(columns="scenario_order").reset_index(drop=True)
    return policies, policy_decisions


def seed_source_scenario_results(
    run_dir: Path, policies: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    yearly_frames = []
    for scenario in SCENARIO_ORDER:
        subset = policies[policies["scenario"] == scenario]
        for _, source in subset.iterrows():
            row = {
                "source_solution_id": source["solution_id"],
                "source_scenario": scenario,
                "target_scenario": scenario,
            }
            for name in RESULT_COLUMNS[3:]:
                row[name] = source[name]
            rows.append(row)

        yearly = pd.read_csv(run_dir / f"pareto_yearly_{scenario}.csv")
        yearly = yearly[yearly["solution_id"].isin(set(subset["solution_id"]))].copy()
        yearly = yearly.rename(columns={"solution_id": "source_solution_id"})
        yearly["source_scenario"] = scenario
        yearly["target_scenario"] = scenario
        yearly = yearly.drop(columns=["scenario", "trial_number"], errors="ignore")
        first = list(YEARLY_KEY_COLUMNS)
        yearly = yearly[first + [name for name in yearly.columns if name not in first]]
        yearly_frames.append(yearly)
    return pd.DataFrame(rows), pd.concat(yearly_frames, ignore_index=True)


def _read_reusable(
    path: Path,
    expected_ids: set[str],
    required_columns: Sequence[str],
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=required_columns)
    frame = pd.read_csv(path)
    missing = set(required_columns) - set(frame.columns)
    if missing:
        return pd.DataFrame(columns=required_columns)
    frame = frame[frame["source_solution_id"].isin(expected_ids)].copy()
    return frame.loc[:, list(required_columns)]


def _deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.drop_duplicates(
        subset=["source_solution_id", "target_scenario"], keep="last"
    ).reset_index(drop=True)


def _ordered_results(
    frame: pd.DataFrame, policy_order: Mapping[str, int]
) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["policy_order"] = ordered["source_solution_id"].map(policy_order)
    ordered["target_order"] = ordered["target_scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    return ordered.sort_values(["policy_order", "target_order"]).drop(
        columns=["policy_order", "target_order"]
    ).reset_index(drop=True)


def _ordered_yearly(
    frame: pd.DataFrame, policy_order: Mapping[str, int]
) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["policy_order"] = ordered["source_solution_id"].map(policy_order)
    ordered["target_order"] = ordered["target_scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    return ordered.sort_values(["policy_order", "target_order", "year"]).drop(
        columns=["policy_order", "target_order"]
    ).reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument(
        "--force", action="store_true", help="Discard prior all-Pareto cross results."
    )
    parser.add_argument(
        "--no-reuse-representatives",
        action="store_true",
        help="Do not reuse the earlier representative-policy cross evaluation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be at least 1.")
    run_dir = (args.run_dir or latest_run(Path("data"))).resolve()
    output_dir = (
        args.output_dir
        or Path("analysis") / run_dir.name / "all_pareto_cross_scenario"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policies, decisions = load_pareto(run_dir)
    expected_ids = set(policies["solution_id"])
    policy_order = {
        solution_id: index for index, solution_id in enumerate(policies["solution_id"])
    }
    expected_keys = {
        (solution_id, target)
        for solution_id in policies["solution_id"]
        for target in SCENARIO_ORDER
    }

    source_results, source_yearly = seed_source_scenario_results(run_dir, policies)
    results_path = output_dir / "cross_scenario_results.csv"
    yearly_path = output_dir / "cross_scenario_yearly.csv"
    result_frames = [source_results]
    yearly_frames = [source_yearly]
    reused_representative_rows = 0

    if not args.force:
        prior_results = _read_reusable(results_path, expected_ids, RESULT_COLUMNS)
        prior_yearly = _read_reusable(
            yearly_path,
            expected_ids,
            [*YEARLY_KEY_COLUMNS, "year"],
        )
        if not prior_results.empty:
            result_frames.append(prior_results)
        if not prior_yearly.empty:
            full_prior_yearly = pd.read_csv(yearly_path)
            full_prior_yearly = full_prior_yearly[
                full_prior_yearly["source_solution_id"].isin(expected_ids)
            ].copy()
            yearly_frames.append(full_prior_yearly)

        if not args.no_reuse_representatives:
            representative_dir = Path("analysis") / run_dir.name / "cross_scenario"
            representative_results = _read_reusable(
                representative_dir / "cross_scenario_results.csv",
                expected_ids,
                RESULT_COLUMNS,
            )
            if not representative_results.empty:
                result_frames.append(representative_results)
                reused_representative_rows = len(representative_results)
            representative_yearly_path = representative_dir / "cross_scenario_yearly.csv"
            if representative_yearly_path.exists():
                representative_yearly = pd.read_csv(representative_yearly_path)
                representative_yearly = representative_yearly[
                    representative_yearly["source_solution_id"].isin(expected_ids)
                ].copy()
                if not representative_yearly.empty:
                    yearly_frames.append(representative_yearly)

    results = _deduplicate(pd.concat(result_frames, ignore_index=True))
    yearly = pd.concat(yearly_frames, ignore_index=True)
    yearly = yearly.drop_duplicates(
        subset=["source_solution_id", "target_scenario", "year"], keep="last"
    ).reset_index(drop=True)
    completed = {evaluation_key(row) for _, row in results.iterrows()}

    baselines = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    tasks = []
    for _, row in decisions.iterrows():
        decision = decision_from_row(row)
        for target_scenario in SCENARIO_ORDER:
            key = (str(row["solution_id"]), target_scenario)
            if key in completed:
                continue
            tasks.append(
                EvaluationTask(
                    source_solution_id=key[0],
                    source_scenario=str(row["scenario"]),
                    selection_roles="all_pareto",
                    selection_label=key[0],
                    target_scenario=target_scenario,
                    decision=decision,
                    baseline={
                        name: float(value)
                        for name, value in baselines[target_scenario].items()
                    },
                )
            )

    def save_progress() -> None:
        _atomic_csv(_ordered_results(results, policy_order), results_path)
        _atomic_csv(_ordered_yearly(yearly, policy_order), yearly_path)

    print(
        f"Loaded {len(policies)} Pareto policies; "
        f"reused {len(completed)}/{len(expected_keys)} combinations "
        f"({reused_representative_rows} representative rows read).",
        flush=True,
    )
    if tasks:
        print(
            f"Evaluating {len(tasks)} missing combinations with {args.workers} workers.",
            flush=True,
        )
        pool = None
        if args.workers == 1:
            iterator = map(evaluate_task, tasks)
        else:
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=min(args.workers, len(tasks)))
            iterator = pool.imap_unordered(evaluate_task, tasks)
        try:
            for number, (row, yearly_rows) in enumerate(iterator, start=1):
                row = {name: row[name] for name in RESULT_COLUMNS}
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                new_yearly = pd.DataFrame(yearly_rows).drop(
                    columns=["selection_roles", "selection_label"], errors="ignore"
                )
                yearly = pd.concat([yearly, new_yearly], ignore_index=True)
                if number % args.checkpoint_every == 0 or number == len(tasks):
                    save_progress()
                print(
                    f"Completed {number}/{len(tasks)}: "
                    f"{row['source_solution_id']} -> {row['target_scenario']}",
                    flush=True,
                )
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            save_progress()
            raise
        else:
            if pool is not None:
                pool.close()
                pool.join()

    results = _deduplicate(results)
    yearly = yearly.drop_duplicates(
        subset=["source_solution_id", "target_scenario", "year"], keep="last"
    )
    if {evaluation_key(row) for _, row in results.iterrows()} != expected_keys:
        raise RuntimeError(
            f"Cross-evaluation is incomplete: {len(results)}/{len(expected_keys)} combinations."
        )
    expected_yearly_rows = len(expected_keys) * 15
    if len(yearly) != expected_yearly_rows:
        raise RuntimeError(
            f"Yearly results are incomplete: {len(yearly)}/{expected_yearly_rows} rows."
        )
    save_progress()
    _atomic_csv(policies, output_dir / "all_pareto_policies.csv")
    _atomic_csv(decisions, output_dir / "all_pareto_decisions.csv")
    metadata = {
        "source_run_dir": str(run_dir),
        "policy_count": len(policies),
        "evaluation_count": len(results),
        "yearly_row_count": len(yearly),
        "source_scenarios": list(SCENARIO_ORDER),
        "target_scenarios": list(SCENARIO_ORDER),
        "reused_representative_rows": reused_representative_rows,
    }
    (output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved all-Pareto cross evaluation: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
