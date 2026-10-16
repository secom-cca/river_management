"""Cross-evaluate representative Optuna policies under every climate scenario."""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_ORDER = ("present", "2C", "4C")
SCENARIO_LABELS = {"present": "Present", "2C": "2℃", "4C": "4℃"}
OBJECTIVES = (
    "flood_damage_ratio",
    "innundation_damage_ratio",
    "landslide_risk_ratio",
    "crop_revenue_ratio",
    "biodiversity_ratio",
)
OBJECTIVE_LABELS = ("Flood", "Inundation", "Landslide", "Crop", "Biodiversity")
MINIMIZE = np.array([True, True, True, False, False])
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
CONTINUOUS_DECISIONS = (
    "drainage_investment_amount",
    "dam_investment_amount",
    "levee_investment_amount",
    "house_elevation_per_year",
    "migration_per_year",
    "annual_paddy_dam_investment",
    "annual_forest_management_conversion_area",
)
ONE_TIME_PREFIXES = ("drainage", "dam", "levee")
INTERVAL_PREFIXES = ("house_elevation", "migration", "paddy_dam", "forest_management")


@dataclass(frozen=True)
class EvaluationTask:
    source_solution_id: str
    source_scenario: str
    selection_roles: str
    selection_label: str
    target_scenario: str
    decision: dict[str, Any]
    baseline: dict[str, float]


def latest_run(data_dir: Path) -> Path:
    candidates = [
        path
        for path in data_dir.glob("optuna_to7_*")
        if path.is_dir() and (path / "config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No Optuna to7 run directory was found.")
    return max(candidates, key=lambda path: (path / "config.json").stat().st_mtime)


def decision_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    decision = {name: float(row[name]) for name in CONTINUOUS_DECISIONS}
    for prefix in ONE_TIME_PREFIXES:
        decision[f"{prefix}_start_year"] = int(float(row[f"{prefix}_start_year"]))
    for prefix in INTERVAL_PREFIXES:
        decision[f"{prefix}_start_year"] = int(float(row[f"{prefix}_start_year"]))
        decision[f"{prefix}_end_year"] = int(float(row[f"{prefix}_end_year"]))
    enabled = bool(int(float(row["breeding_enabled"])))
    decision["breeding_enabled"] = int(enabled)
    if enabled:
        decision["breeding_start_year"] = int(float(row["breeding_start_year"]))
        decision["breeding_end_year"] = int(float(row["breeding_end_year"]))
    else:
        decision["breeding_start_year"] = 0
        decision["breeding_end_year"] = 9
    return decision


def evaluation_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["source_solution_id"]), str(row["target_scenario"])


def evaluate_task(task: EvaluationTask) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    core = importlib.import_module("run_vensim_with_pysd_to7_opt")
    optuna_runner = importlib.import_module("run_vensim_with_pysd_to7_optuna")
    metrics, yearly = core.evaluate_policy(task.target_scenario, task.decision)
    ratios = optuna_runner.objective_ratios(metrics, task.baseline)
    annual_constraint, total_constraint = optuna_runner.budget_constraints(metrics)
    row: dict[str, Any] = {
        "source_solution_id": task.source_solution_id,
        "source_scenario": task.source_scenario,
        "selection_roles": task.selection_roles,
        "selection_label": task.selection_label,
        "target_scenario": task.target_scenario,
        **{name: float(value) for name, value in zip(OBJECTIVES, ratios)},
        **{name: float(metrics[name]) for name in RAW_METRICS},
        "annual_budget_constraint": float(annual_constraint),
        "total_budget_constraint": float(total_constraint),
        "annual_budget_utilisation": float(annual_constraint + 1.0),
        "total_budget_utilisation": float(total_constraint + 1.0),
        "feasible": bool(annual_constraint <= 1e-12 and total_constraint <= 1e-12),
    }
    yearly = yearly.copy()
    yearly.insert(0, "target_scenario", task.target_scenario)
    yearly.insert(0, "source_scenario", task.source_scenario)
    yearly.insert(0, "source_solution_id", task.source_solution_id)
    records = yearly.to_dict(orient="records")
    return row, records


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _ordered_results(frame: pd.DataFrame, policy_order: Mapping[str, int]) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["policy_order"] = ordered["source_solution_id"].map(policy_order)
    ordered["target_order"] = ordered["target_scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    return ordered.sort_values(["policy_order", "target_order"]).drop(
        columns=["policy_order", "target_order"]
    ).reset_index(drop=True)


def add_regrets(results: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    output = results.copy()
    for scenario in SCENARIO_ORDER:
        pareto = pd.read_csv(run_dir / f"pareto_summary_{scenario}.csv")
        values = pareto.loc[:, OBJECTIVES].to_numpy(dtype=float)
        ideal = np.where(MINIMIZE, values.min(axis=0), values.max(axis=0))
        nadir = np.where(MINIMIZE, values.max(axis=0), values.min(axis=0))
        span = np.abs(nadir - ideal)
        span[span <= 1e-15] = 1.0
        mask = output["target_scenario"] == scenario
        target_values = output.loc[mask, OBJECTIVES].to_numpy(dtype=float)
        regret = np.where(
            MINIMIZE,
            (target_values - ideal) / span,
            (ideal - target_values) / span,
        )
        regret = np.maximum(regret, 0.0)
        for index, objective in enumerate(OBJECTIVES):
            output.loc[mask, f"normalized_regret_{objective}"] = regret[:, index]
    regret_columns = [f"normalized_regret_{name}" for name in OBJECTIVES]
    output["mean_normalized_regret"] = output[regret_columns].mean(axis=1)
    output["worst_normalized_regret"] = output[regret_columns].max(axis=1)
    return output


def summarize_robustness(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for solution_id, group in results.groupby("source_solution_id", sort=False):
        first = group.iloc[0]
        feasible = group["feasible"].astype(str).str.lower().eq("true")
        row: dict[str, Any] = {
            "source_solution_id": solution_id,
            "source_scenario": first["source_scenario"],
            "selection_roles": first["selection_roles"],
            "selection_label": first["selection_label"],
            "evaluated_scenarios": len(group),
            "feasible_scenarios": int(feasible.sum()),
            "feasibility_rate": float(feasible.mean()),
            "mean_normalized_regret": float(group["mean_normalized_regret"].mean()),
            "worst_normalized_regret": float(group["worst_normalized_regret"].max()),
            "mean_total_budget_utilisation": float(group["total_budget_utilisation"].mean()),
            "max_total_budget_utilisation": float(group["total_budget_utilisation"].max()),
            "max_annual_budget_utilisation": float(group["annual_budget_utilisation"].max()),
        }
        for objective, minimize in zip(OBJECTIVES, MINIMIZE):
            row[f"mean_{objective}"] = float(group[objective].mean())
            row[f"worst_{objective}"] = float(
                group[objective].max() if minimize else group[objective].min()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _improvement_matrix(results: pd.DataFrame, policy_ids: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    columns = []
    blocks = []
    for scenario in SCENARIO_ORDER:
        subset = results[results["target_scenario"] == scenario].set_index("source_solution_id")
        values = subset.loc[list(policy_ids), OBJECTIVES].to_numpy(dtype=float)
        improvements = np.empty_like(values)
        improvements[:, :3] = (1.0 - values[:, :3]) * 100
        improvements[:, 3:] = (values[:, 3:] - 1.0) * 100
        blocks.append(improvements)
        columns.extend(
            f"{SCENARIO_LABELS[scenario]}\n{label}" for label in OBJECTIVE_LABELS
        )
    return np.hstack(blocks), columns


def plot_cross_scenario(results: pd.DataFrame, policies: pd.DataFrame, output: Path) -> None:
    policy_ids = policies["solution_id"].tolist()
    values, columns = _improvement_matrix(results, policy_ids)
    colours = np.zeros_like(values)
    objective_count = len(OBJECTIVES)
    for scenario_index in range(len(SCENARIO_ORDER)):
        for objective_index in range(objective_count):
            col = scenario_index * objective_count + objective_index
            low = values[:, col].min()
            span = values[:, col].max() - low
            colours[:, col] = 0.5 if span <= 1e-15 else (values[:, col] - low) / span
    height = max(6.0, 0.5 * len(policies) + 2.2)
    fig, ax = plt.subplots(figsize=(19, height))
    image = ax.imshow(colours, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for row in range(len(policies)):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.1f}%", ha="center", va="center", fontsize=7)
    labels = [
        f"{SCENARIO_LABELS.get(row['scenario'], row['scenario'])} | {row['selection_label']}"
        for _, row in policies.iterrows()
    ]
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(range(len(policies)), labels)
    for boundary in (4.5, 9.5):
        ax.axvline(boundary, color="white", linewidth=2.5)
    ax.set_title("Cross-climate performance (annotations are improvement from each target baseline)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Relative desirability within each target and objective")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mean_vs_worst(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    colors = summary["feasibility_rate"].to_numpy(dtype=float)
    sizes = 45 + 90 * np.clip(summary["max_total_budget_utilisation"].to_numpy(dtype=float), 0, 1.5)
    scatter = ax.scatter(
        summary["mean_normalized_regret"],
        summary["worst_normalized_regret"],
        c=colors,
        s=sizes,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )
    for _, row in summary.iterrows():
        label = f"{SCENARIO_LABELS.get(row['source_scenario'], row['source_scenario'])} | {row['selection_label']}"
        ax.annotate(
            label,
            (row["mean_normalized_regret"], row["worst_normalized_regret"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set_xlabel("Mean normalized regret across climates (lower is better)")
    ax.set_ylabel("Worst normalized regret across climates (lower is better)")
    ax.grid(alpha=0.25)
    ax.set_title("Average versus worst-case cross-climate performance")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Budget-feasible climate fraction")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--representatives-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="Discard existing cross-evaluation rows.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    run_dir = (args.run_dir or latest_run(Path("data"))).resolve()
    representatives_dir = (
        args.representatives_dir
        or Path("analysis") / run_dir.name / "representatives"
    ).resolve()
    output_dir = (
        args.output_dir or Path("analysis") / run_dir.name / "cross_scenario"
    ).resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    policies = pd.read_csv(representatives_dir / "representative_policies.csv")
    decisions = pd.read_csv(representatives_dir / "representative_decisions.csv")
    decisions = decisions.merge(
        policies[["solution_id", "selection_roles", "selection_label"]],
        on="solution_id",
        how="inner",
        suffixes=("", "_policy"),
        validate="one_to_one",
    )
    for name in ("selection_roles", "selection_label"):
        alternate = f"{name}_policy"
        if alternate in decisions:
            decisions[name] = decisions[alternate]
            decisions = decisions.drop(columns=alternate)
    baselines = json.loads((run_dir / "baseline.json").read_text(encoding="utf-8"))
    policy_order = {solution_id: index for index, solution_id in enumerate(policies["solution_id"])}
    expected_keys = {
        (row["solution_id"], scenario)
        for _, row in policies.iterrows()
        for scenario in SCENARIO_ORDER
    }

    results_path = output_dir / "cross_scenario_results.csv"
    yearly_path = output_dir / "cross_scenario_yearly.csv"
    if not args.force and results_path.exists():
        results = pd.read_csv(results_path)
        results = results[
            results.apply(lambda row: evaluation_key(row) in expected_keys, axis=1)
        ].copy()
    else:
        results = pd.DataFrame()
    if not args.force and yearly_path.exists():
        yearly = pd.read_csv(yearly_path)
        yearly = yearly[
            yearly.apply(lambda row: evaluation_key(row) in expected_keys, axis=1)
        ].copy()
    else:
        yearly = pd.DataFrame()
    completed = set() if results.empty else {evaluation_key(row) for _, row in results.iterrows()}

    tasks = []
    for _, row in decisions.iterrows():
        decision = decision_from_row(row)
        for target_scenario in SCENARIO_ORDER:
            key = (row["solution_id"], target_scenario)
            if key in completed:
                continue
            tasks.append(
                EvaluationTask(
                    source_solution_id=str(row["solution_id"]),
                    source_scenario=str(row["scenario"]),
                    selection_roles=str(row["selection_roles"]),
                    selection_label=str(row["selection_label"]),
                    target_scenario=target_scenario,
                    decision=decision,
                    baseline={name: float(value) for name, value in baselines[target_scenario].items()},
                )
            )

    def save_progress() -> None:
        if results.empty:
            return
        ordered_results = _ordered_results(results, policy_order)
        _atomic_csv(ordered_results, results_path)
        if not yearly.empty:
            ordered_yearly = yearly.copy()
            ordered_yearly["policy_order"] = ordered_yearly["source_solution_id"].map(policy_order)
            ordered_yearly["target_order"] = ordered_yearly["target_scenario"].map(
                {name: index for index, name in enumerate(SCENARIO_ORDER)}
            )
            ordered_yearly = ordered_yearly.sort_values(
                ["policy_order", "target_order", "year"]
            ).drop(columns=["policy_order", "target_order"])
            _atomic_csv(ordered_yearly, yearly_path)

    if tasks:
        print(f"Evaluating {len(tasks)} policy-climate combinations with {args.workers} workers.")
        if args.workers == 1:
            iterator = map(evaluate_task, tasks)
            pool = None
        else:
            context = multiprocessing.get_context("spawn")
            pool = context.Pool(processes=min(args.workers, len(tasks)))
            iterator = pool.imap_unordered(evaluate_task, tasks)
        try:
            for number, (row, yearly_rows) in enumerate(iterator, start=1):
                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                yearly = pd.concat([yearly, pd.DataFrame(yearly_rows)], ignore_index=True)
                save_progress()
                print(f"Completed {number}/{len(tasks)}: {row['source_solution_id']} -> {row['target_scenario']}")
        except BaseException:
            if pool is not None:
                pool.terminate()
                pool.join()
            raise
        else:
            if pool is not None:
                pool.close()
                pool.join()
    save_progress()

    if len(results) != len(expected_keys):
        raise RuntimeError(
            f"Cross-evaluation is incomplete: {len(results)}/{len(expected_keys)} combinations."
        )
    results = add_regrets(_ordered_results(results, policy_order), run_dir)
    _atomic_csv(results, results_path)
    summary = summarize_robustness(results)
    summary["policy_order"] = summary["source_solution_id"].map(policy_order)
    summary = summary.sort_values("policy_order").drop(columns="policy_order").reset_index(drop=True)
    _atomic_csv(summary, output_dir / "robustness_summary.csv")
    plot_cross_scenario(results, policies, figure_dir / "04_cross_scenario_performance.png")
    plot_mean_vs_worst(summary, figure_dir / "05_mean_vs_worst.png")
    metadata = {
        "source_run_dir": str(run_dir),
        "representatives_dir": str(representatives_dir),
        "policy_count": len(policies),
        "evaluation_count": len(results),
        "target_scenarios": list(SCENARIO_ORDER),
    }
    (output_dir / "cross_scenario_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved cross-scenario analysis: {output_dir}")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
