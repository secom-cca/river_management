"""Replay saved policies to measure model changes without running optimisation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from breeding_response_to7 import model_configuration
from evaluate_optuna_to7_cross_scenario import (
    EvaluationTask, OBJECTIVES, RAW_METRICS, SCENARIO_ORDER,
    decision_from_row, evaluate_task,
)
import run_vensim_with_pysd_to7_opt as core
import run_vensim_with_pysd_to7_optuna as optimiser


def select_policies(decisions, candidates, selection):
    """Include old final candidates, every breeding start, and a no-breeding control."""
    if selection == "all":
        return decisions.copy().assign(comparison_role="all_pareto")
    selected = {}
    for row in candidates.to_dict("records"):
        selected[row["solution_id"]] = "final_" + str(row["candidate_roles"])
    covered = {
        int(float(row.breeding_start_year))
        for row in decisions[decisions.solution_id.isin(selected)].itertuples()
        if int(row.breeding_enabled)
    }
    # Rotate design climates for broad coverage; prefer earlier recorded trials.
    for start, preferred in zip(range(6), ("present", "2C", "4C", "4C", "present", "2C")):
        if start in covered:
            continue
        pool = decisions[(decisions.breeding_enabled == 1) & (decisions.breeding_start_year == start)]
        preferred_pool = pool[pool.scenario == preferred]
        if not preferred_pool.empty:
            pool = preferred_pool
        if pool.empty:
            raise ValueError(f"No saved Pareto policy for breeding start {2009 + start}")
        row = pool.sort_values("trial_number").iloc[0]
        selected[row.solution_id] = f"breeding_start_{2009 + start}"
    pool = decisions[decisions.breeding_enabled == 0]
    if pool.empty:
        raise ValueError("No saved policy without breeding")
    row = pool.sort_values(["scenario", "trial_number"]).iloc[0]
    selected[row.solution_id] = "no_breeding_control"
    output = decisions[decisions.solution_id.isin(selected)].copy()
    output["comparison_role"] = output.solution_id.map(selected)
    return output.sort_values(["scenario", "trial_number"]).reset_index(drop=True)


def relative_change(old, new):
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    return np.divide(100 * (new - old), np.abs(old), out=np.full_like(old, np.nan), where=np.abs(old) > 1e-12)


def plot_changes(comparison, decisions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = decisions.solution_id.tolist()
    labels = []
    for row in decisions.itertuples():
        breeding = f"B{2009 + int(row.breeding_start_year)}" if int(row.breeding_enabled) else "No breeding"
        labels.append(f"{row.comparison_role} ({row.scenario}, {breeding})")
    fields = [name + "_improvement_change_pp" for name in OBJECTIVES]
    span = max(0.001, float(comparison[fields].abs().to_numpy().max()))
    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(ids) * 0.44)), sharey=True)
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        data = comparison[comparison.target_scenario == scenario].set_index("source_solution_id").loc[ids, fields].to_numpy()
        im = ax.imshow(data, cmap="RdYlGn", vmin=-span, vmax=span, aspect="auto")
        ax.set_title(scenario)
        ax.set_xticks(range(5), ["Flood", "Inundation", "Landslide", "Crop", "Biodiversity"], rotation=45, ha="right")
        ax.set_yticks(range(len(ids)), labels)
        if len(ids) <= 20:
            for i in range(len(ids)):
                for j in range(5):
                    ax.text(j, i, f"{data[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.suptitle("Fixed policies: change in baseline-relative improvement\nPositive values mean improved outcomes; no new optimisation")
    fig.subplots_adjust(left=0.28, right=0.92, bottom=0.23, top=0.83, wspace=0.1)
    cax = fig.add_axes([0.94, 0.25, 0.015, 0.55])
    fig.colorbar(im, cax=cax, label="Change (percentage points)")
    fig.savefig(output / "metric_changes.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "metric_changes.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/optuna_to7_260915_102704"))
    parser.add_argument("--selection", choices=["quick", "all"], default="quick")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    configuration = model_configuration()
    if configuration["breeding_mode"] != "threshold":
        raise ValueError("This comparison requires the adopted threshold model")
    started = time.perf_counter()
    source = Path("analysis") / args.run_dir.name
    evaluations = source / "all_pareto_cross_scenario"
    all_decisions = pd.read_csv(evaluations / "all_pareto_decisions.csv")
    candidates = pd.read_csv(source / "paper_robustness" / "final_candidate_decisions.csv") if args.selection == "quick" else pd.DataFrame()
    decisions = select_policies(all_decisions, candidates, args.selection)
    old_results = pd.read_csv(evaluations / "cross_scenario_results.csv")
    old_yearly = pd.read_csv(evaluations / "cross_scenario_yearly.csv")
    ids = set(decisions.solution_id)
    old_results = old_results[old_results.source_solution_id.isin(ids)].copy()
    old_yearly = old_yearly[old_yearly.source_solution_id.isin(ids)].copy()
    keys = ["source_solution_id", "target_scenario"]
    if old_results.duplicated(keys).any() or len(old_results) != len(ids) * 3:
        raise ValueError("Saved comparisons must have every selected policy under all three climates")
    output = args.output_dir or source / datetime.now().strftime("revision_comparison_%y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    decisions.to_csv(output / "selected_decisions.csv", index=False, encoding="utf-8-sig")
    old_baselines = json.loads((args.run_dir / "baseline.json").read_text(encoding="utf-8"))
    baselines, baseline_rows = {}, []
    for scenario in SCENARIO_ORDER:
        print(f"{scenario}: recalculating no-policy baseline", flush=True)
        metrics, _ = core.evaluate_policy(scenario, optimiser._zero_decision())
        baselines[scenario] = metrics
        for name in RAW_METRICS:
            old, new = old_baselines[scenario][name], metrics[name]
            baseline_rows.append({"scenario": scenario, "metric": name, "old": old, "new": new, "difference": new - old, "change_pct": float(relative_change(old, new))})
    pd.DataFrame(baseline_rows).to_csv(output / "baseline_comparison.csv", index=False, encoding="utf-8-sig")
    tasks = []
    for row in decisions.to_dict("records"):
        for target in SCENARIO_ORDER:
            tasks.append(EvaluationTask(
                source_solution_id=row["solution_id"], source_scenario=row["scenario"],
                selection_roles=row["comparison_role"], selection_label=row["comparison_role"],
                target_scenario=target, decision=decision_from_row(row), baseline=baselines[target],
            ))
    new_rows, new_years = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            row, yearly = future.result()
            if len(yearly) != 15 or not np.isfinite([row[name] for name in RAW_METRICS]).all():
                raise ValueError("Invalid replay result")
            new_rows.append(row)
            new_years.extend(yearly)
            print(f"{len(new_rows)}/{len(tasks)}: {row['source_solution_id']} / {row['target_scenario']}", flush=True)
    new_results = pd.DataFrame(new_rows)
    new_yearly = pd.DataFrame(new_years)
    comparison = old_results.merge(new_results, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    for name in RAW_METRICS:
        comparison[name + "_difference"] = comparison[name + "_new"] - comparison[name + "_old"]
        comparison[name + "_change_pct"] = relative_change(comparison[name + "_old"], comparison[name + "_new"])
    for index, name in enumerate(OBJECTIVES):
        sign = -1 if index < 3 else 1
        comparison[name + "_improvement_change_pp"] = sign * 100 * (comparison[name + "_new"] - comparison[name + "_old"])
    for version, frame in (("old", old_yearly), ("new", new_yearly)):
        final = frame.sort_values("year").groupby(keys, as_index=False).tail(1)
        final = final[keys + ["accumulated_breeding_investment", "breeding_success"]].rename(columns={
            "accumulated_breeding_investment": "final_breeding_investment_" + version,
            "breeding_success": "final_breeding_success_" + version,
        })
        comparison = comparison.merge(final, on=keys, validate="one_to_one")
    comparison["feasibility_changed"] = comparison.feasible_old != comparison.feasible_new
    comparison["breeding_attainment_changed"] = comparison.final_breeding_success_old != comparison.final_breeding_success_new
    comparison = comparison.merge(decisions[["solution_id", "comparison_role", "breeding_enabled", "breeding_start_calendar_year"]], left_on="source_solution_id", right_on="solution_id", validate="many_to_one")
    comparison = comparison.sort_values(keys)
    comparison.to_csv(output / "policy_comparison.csv", index=False, encoding="utf-8-sig")
    year_keys = keys + ["year"]
    yearly_comparison = old_yearly.merge(new_yearly, on=year_keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(yearly_comparison) != len(tasks) * 15:
        raise ValueError("Missing year in before/after comparison")
    yearly_comparison.to_csv(output / "yearly_comparison.csv", index=False, encoding="utf-8-sig")
    plot_changes(comparison, decisions, output)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_run": str(args.run_dir.resolve()), "saved_evaluation_dir": str(evaluations.resolve()),
        "source_run_config": json.loads((args.run_dir / "config.json").read_text(encoding="utf-8")),
        "model_configuration": configuration, "selection": args.selection,
        "policy_count": len(decisions), "policy_replays": len(tasks), "baseline_runs": 3,
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Policies and all scenario/physical assumptions are held fixed; no new optimisation.",
            "Old values come from the saved evaluation, not a rerun of a historical source tree.",
            "Old yearly stocks were sampled before December 31 integration; new stocks include it.",
            "Improvement changes use each model's own no-policy baseline; absolute changes are reported separately.",
            "A small fixed-policy sample cannot establish a new Pareto front or the best design climate.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
