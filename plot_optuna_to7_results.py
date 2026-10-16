"""Create Pareto and progress diagnostics for a to7 Optuna NSGA-III run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OBJECTIVES = [
    "flood_damage_ratio",
    "innundation_damage_ratio",
    "landslide_risk_ratio",
    "crop_revenue_ratio",
    "biodiversity_ratio",
]
LABELS = [
    "Flood damage ↓",
    "Inundation damage ↓",
    "Landslide risk ↓",
    "Crop revenue ↑",
    "Biodiversity ↑",
]
SCENARIO_LABELS = {"present": "Present", "2C": "2℃", "4C": "4℃"}


def latest_run(data_dir: Path) -> Path:
    candidates = [
        path
        for path in data_dir.glob("optuna_to7_*")
        if path.is_dir() and (path / "config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No Optuna to7 run directory was found.")
    return max(candidates, key=lambda path: (path / "config.json").stat().st_mtime)


def read_nonempty_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 3:
        return pd.DataFrame()
    return pd.read_csv(path)


def validate_frame(frame: pd.DataFrame, path: Path, require_solution: bool = False) -> None:
    required = set(OBJECTIVES)
    if require_solution:
        required.add("solution_id")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    values = frame[OBJECTIVES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite objective values.")


def _as_minimization(values: np.ndarray) -> np.ndarray:
    converted = values.copy()
    converted[:, 3:] *= -1
    return converted


def incremental_front_sizes(values: np.ndarray, step: int) -> tuple[list[int], list[int]]:
    minimization = _as_minimization(values)
    front: list[np.ndarray] = []
    xs: list[int] = []
    sizes: list[int] = []
    for count, point in enumerate(minimization, start=1):
        dominated = any(
            np.all(existing <= point) and np.any(existing < point)
            for existing in front
        )
        if not dominated:
            front = [
                existing
                for existing in front
                if not (np.all(point <= existing) and np.any(point < existing))
            ]
            front.append(point)
        if count % step == 0 or count == len(minimization):
            xs.append(count)
            sizes.append(len(front))
    return xs, sizes


def parallel_coordinates(pareto: pd.DataFrame, output: Path, title: str) -> None:
    values = pareto[OBJECTIVES].to_numpy(dtype=float)
    favourable = values.copy()
    favourable[:, :3] *= -1
    low = favourable.min(axis=0)
    span = favourable.max(axis=0) - low
    span[span == 0] = 1.0
    normalized = (favourable - low) / span
    budget = pareto["total_budget_utilisation"].to_numpy(dtype=float)
    norm = plt.Normalize(budget.min(), max(budget.max(), budget.min() + 1e-12))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for row, use in zip(normalized, budget):
        ax.plot(range(len(OBJECTIVES)), row, color=cmap(norm(use)), alpha=0.45)
    ax.set_xticks(range(len(OBJECTIVES)), LABELS, rotation=15, ha="right")
    ax.set_ylabel("Normalized favourable outcome within Pareto set")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    colorbar.set_label("Total budget utilisation")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pairwise_scatter(pareto: pd.DataFrame, output: Path, title: str) -> None:
    size = len(OBJECTIVES)
    fig, axes = plt.subplots(size, size, figsize=(13, 13))
    colors = pareto["total_budget_utilisation"].to_numpy(dtype=float)
    for row in range(size):
        for col in range(size):
            ax = axes[row, col]
            if row == col:
                ax.hist(pareto[OBJECTIVES[col]], bins=min(20, max(5, len(pareto) // 3)), color="#4c78a8")
            else:
                ax.scatter(
                    pareto[OBJECTIVES[col]],
                    pareto[OBJECTIVES[row]],
                    c=colors,
                    cmap="viridis",
                    s=14,
                    alpha=0.7,
                )
            if row == size - 1:
                ax.set_xlabel(LABELS[col], fontsize=8, rotation=12)
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(LABELS[row], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.grid(alpha=0.15)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def budget_relationships(trials: pd.DataFrame, pareto: pd.DataFrame, output: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    pareto_numbers = set(pareto["trial_number"].astype(int))
    on_front = trials["trial_number"].astype(int).isin(pareto_numbers)
    for index, (objective, label) in enumerate(zip(OBJECTIVES, LABELS)):
        ax = axes.flat[index]
        ax.scatter(
            trials.loc[~on_front, "total_budget_utilisation"],
            trials.loc[~on_front, objective],
            color="#b8b8b8",
            s=10,
            alpha=0.35,
            label="Feasible non-Pareto trial",
        )
        ax.scatter(
            trials.loc[on_front, "total_budget_utilisation"],
            trials.loc[on_front, objective],
            color="#d62728",
            s=18,
            alpha=0.75,
            label="Final Pareto",
        )
        ax.set_xlabel("Total budget utilisation")
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.95, 0.08), frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def progress_diagnostic(trials: pd.DataFrame, population_size: int, output: Path, title: str) -> None:
    ordered = trials.sort_values("trial_number")
    xs, sizes = incremental_front_sizes(
        ordered[OBJECTIVES].to_numpy(dtype=float), max(1, population_size)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(xs, sizes, marker="o", markersize=3)
    axes[0].set_xlabel("Cumulative feasible trials")
    axes[0].set_ylabel("Current non-dominated set size")
    axes[0].grid(alpha=0.25)
    generations = ordered["trial_number"].to_numpy(dtype=int) // population_size
    axes[1].scatter(
        generations,
        ordered["duration_seconds"],
        s=9,
        alpha=0.35,
        color="#4c78a8",
    )
    axes[1].set_xlabel("Approximate NSGA-III generation")
    axes[1].set_ylabel("Trial duration (s)")
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pareto_count_plot(counts: dict[str, tuple[int, int]], output: Path) -> None:
    scenarios = list(counts)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(scenarios))
    pareto = [counts[scenario][0] for scenario in scenarios]
    complete = [counts[scenario][1] for scenario in scenarios]
    bars = ax.bar(x, pareto, color="#4c78a8")
    ax.bar_label(bars, labels=[f"{p}/{c}" for p, c in zip(pareto, complete)])
    ax.set_xticks(x, [SCENARIO_LABELS.get(s, s) for s in scenarios])
    ax.set_ylabel("Feasible Pareto solutions")
    ax.set_title("Pareto solutions / feasible completed trials")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--scenarios", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (args.run_dir or latest_run(Path("data"))).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    scenarios = args.scenarios or config["scenarios"]
    unknown = set(scenarios) - set(config["scenarios"])
    if unknown:
        raise ValueError(f"Scenarios are not present in this run: {sorted(unknown)}")
    output_root = (args.output_dir or Path("figures") / run_dir.name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    counts = {}
    for scenario in scenarios:
        trials_path = run_dir / f"trials_{scenario}.csv"
        pareto_path = run_dir / f"pareto_summary_{scenario}.csv"
        trials = read_nonempty_csv(trials_path)
        pareto = read_nonempty_csv(pareto_path)
        if trials.empty or pareto.empty:
            print(f"Skipping {scenario}: exported trials or Pareto results are empty.")
            continue
        trials = trials[
            (trials["trial_state"] == "COMPLETE")
            & trials["feasible"].astype(str).str.lower().eq("true")
        ].copy()
        validate_frame(trials, trials_path)
        validate_frame(pareto, pareto_path, require_solution=True)
        scenario_dir = output_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        label = SCENARIO_LABELS.get(scenario, scenario)
        parallel_coordinates(
            pareto,
            scenario_dir / "01_pareto_parallel_coordinates.png",
            f"{label}: feasible Pareto policies",
        )
        pairwise_scatter(
            pareto,
            scenario_dir / "02_pareto_pairwise_scatter.png",
            f"{label}: pairwise Pareto objectives",
        )
        budget_relationships(
            trials,
            pareto,
            scenario_dir / "03_budget_relationships.png",
            f"{label}: objectives and total budget use",
        )
        progress_diagnostic(
            trials,
            int(config["population_size"]),
            scenario_dir / "04_search_progress.png",
            f"{label}: search progress and trial duration",
        )
        counts[scenario] = (len(pareto), len(trials))
    if counts:
        pareto_count_plot(counts, output_root / "00_pareto_counts.png")
    print(f"Saved figures: {output_root}")


if __name__ == "__main__":
    main()
