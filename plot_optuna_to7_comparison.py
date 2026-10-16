"""Compare archived, re-evaluated, and newly optimized present policies."""

from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_optuna_to7_results import OBJECTIVES, validate_frame


METRICS = ["Flood", "Inundation", "Landslide", "Crop", "Biodiversity"]
GROUPS = ["Previous / original model", "Previous / corrected model", "New / corrected model"]
COLORS = ["#808080", "#e69f00", "#0072b2"]
MARKERS = ["x", "^", "o"]


def feasible_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["feasible"].astype(str).str.lower().eq("true")].copy()


def improvements(frame: pd.DataFrame) -> np.ndarray:
    values = frame[OBJECTIVES].to_numpy(dtype=float)
    return (values - 1) * np.array([-100, -100, -100, 100, 100])


def save_figure(fig: plt.Figure, directory: Path, name: str) -> None:
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def legend_handles() -> list[Line2D]:
    return [Line2D([], [], color=color, marker=marker, linestyle="none", label=group)
            for group, color, marker in zip(GROUPS, COLORS, MARKERS)]


def outcome_plot(frames: list[pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    for j, ax in enumerate(axes.flat[:5]):
        for i, (frame, color) in enumerate(zip(frames, COLORS)):
            y = improvements(frame)[:, j]
            # Deterministic offsets display ties without changing the outcome.
            offsets = np.linspace(-0.13, 0.13, len(y))
            ax.boxplot([y], positions=[i], widths=0.44, showfliers=False,
                       patch_artist=True,
                       boxprops={"facecolor": color, "alpha": 0.18},
                       medianprops={"color": color, "linewidth": 2})
            ax.scatter(i + offsets, y, s=15, color=color, alpha=0.55)
        ax.set_xticks(range(3), ["Old", "Old re-evaluated", "New"], fontsize=9)
        ax.set_title(METRICS[j])
        ax.set_ylabel("Improvement vs no policy (%) — higher is better")
        ax.grid(axis="y", alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=legend_handles(), loc="center", frameon=False)
    axes.flat[-1].text(0.03, 0.15,
                       "All archived policies are shown after re-evaluation;\n"
                       "some are no longer Pareto policies.\n"
                       "Distributions describe policy sets, not uncertainty.",
                       transform=axes.flat[-1].transAxes, fontsize=9)
    fig.suptitle("Present: outcome distributions of archived and new Pareto policies")
    fig.tight_layout()
    save_figure(fig, output, "01_outcome_distributions")


def tradeoff_plot(frames: list[pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
    for ax, (a, b) in zip(axes.flat, itertools.combinations(range(5), 2)):
        for frame, color, marker in zip(frames, COLORS, MARKERS):
            v = improvements(frame)
            ax.scatter(v[:, a], v[:, b], color=color, marker=marker,
                       s=23, alpha=0.65, linewidths=0.8)
        ax.set_xlabel(f"{METRICS[a]} improvement (%)")
        ax.set_ylabel(f"{METRICS[b]} improvement (%)")
        ax.grid(alpha=0.2)
    fig.suptitle("Present: policy tradeoffs (upper right is better)")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    save_figure(fig, output, "02_tradeoffs")


def budget_plot(frames: list[pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6))
    for j, ax in enumerate(axes.flat[:5]):
        for frame, color, marker in zip(frames, COLORS, MARKERS):
            ax.scatter(frame["total_budget_utilisation"] * 100,
                       improvements(frame)[:, j], color=color,
                       marker=marker, s=23, alpha=0.65)
        ax.set_xlabel("Total budget utilisation (%)")
        ax.set_ylabel(f"{METRICS[j]} improvement (%)")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=legend_handles(), loc="center", frameon=False)
    fig.suptitle("Present: budget use and outcomes")
    fig.tight_layout()
    save_figure(fig, output, "03_budget_outcomes")


def common_regrets(frames: list[pd.DataFrame]) -> list[np.ndarray]:
    direction = np.array([1, 1, 1, -1, -1])
    # One common reference for all groups, based only on corrected evaluations.
    pool = pd.concat(frames[1:], ignore_index=True)[OBJECTIVES].to_numpy(float) * direction
    ideal = pool.min(axis=0)
    span = pool.max(axis=0) - ideal
    span[span <= 1e-15] = 1.0
    return [np.maximum((f[OBJECTIVES].to_numpy(float) * direction - ideal) / span, 0)
            for f in frames]


def regret_plot(regrets: list[np.ndarray], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.8))
    for r, group, color, marker in zip(regrets, GROUPS, COLORS, MARKERS):
        ax.scatter(r.mean(axis=1), r.max(axis=1), label=group,
                   color=color, marker=marker, s=35, alpha=0.7)
    ax.set_xlabel("Mean regret across 5 objectives — lower is better")
    ax.set_ylabel("Worst objective regret — lower is better")
    ax.set_title("Present only: regret with a shared reference range")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.text(0.5, 0.01, "Reference: pooled corrected old + new policies; equal objective weights.\n"
             "This is not a cross-climate robustness assessment.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, output, "04_common_reference_regret")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run-dir", type=Path, default=Path("data/optuna_to7_260915_102704"))
    parser.add_argument("--new-run-dir", type=Path, default=Path("data/optuna_to7_261009_105336"))
    parser.add_argument("--revision-dir", type=Path, default=Path(
        "analysis/optuna_to7_260915_102704/revision_comparison_261008_170210"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    new_config = json.loads((args.new_run_dir / "config.json").read_text(encoding="utf-8"))
    revision_manifest = json.loads((args.revision_dir / "manifest.json").read_text(encoding="utf-8"))
    if (not new_config.get("model_configuration") or
            new_config["model_configuration"] != revision_manifest.get("model_configuration")):
        raise ValueError("Archived re-evaluations and the new run use different model configurations.")
    old = pd.read_csv(args.old_run_dir / "pareto_summary_present.csv")
    new = pd.read_csv(args.new_run_dir / "pareto_summary_present.csv")
    comparison = pd.read_csv(args.revision_dir / "policy_comparison.csv")
    selected = comparison.loc[
        comparison["source_scenario_old"].eq("present") & comparison["target_scenario"].eq("present")
    ].copy()
    if selected["source_solution_id"].duplicated().any():
        raise ValueError("Duplicate archived present policies in revision comparison.")
    if set(selected["source_solution_id"]) != set(old["solution_id"]):
        raise ValueError("Revision comparison does not cover exactly the archived present policies.")
    original = selected.set_index("source_solution_id").loc[old["solution_id"]]
    if not np.allclose(original[[c + "_old" for c in OBJECTIVES]].to_numpy(float),
                       old[OBJECTIVES].to_numpy(float), rtol=1e-10, atol=1e-12):
        raise ValueError("Revision comparison was not generated from these old objectives.")
    # Compare outcomes on the same no-policy baseline.
    for run in (args.old_run_dir, args.new_run_dir):
        baseline = pd.read_csv(run / "baseline.csv")
        values = baseline.loc[baseline["scenario"].eq("present"),
                              [c.removesuffix("_ratio") for c in OBJECTIVES]].to_numpy(float)
        if run == args.old_run_dir:
            old_baseline = values
        elif not np.allclose(old_baseline, values, rtol=1e-10, atol=1e-12):
            raise ValueError("No-policy baselines differ; a common baseline is required.")
    corrected = selected.drop(columns=["solution_id"], errors="ignore").rename(columns={**{c + "_new": c for c in OBJECTIVES},
                                          "feasible_new": "feasible",
                                          "total_budget_utilisation_new": "total_budget_utilisation",
                                          "source_solution_id": "solution_id"})
    frames = [feasible_rows(f) for f in (old, corrected, new)]
    for frame in frames:
        validate_frame(frame, Path("comparison input"), require_solution=True)
        if frame.empty or frame["solution_id"].duplicated().any():
            raise ValueError("Expected nonempty policy sets with unique solution IDs.")
    output = args.output_dir or Path("figures") / (
        f"optuna_to7_present_comparison_{datetime.now():%y%m%d_%H%M%S}")
    output.mkdir(parents=True, exist_ok=False)
    regrets = common_regrets(frames)
    records = []
    summary = []
    for frame, group, r in zip(frames, GROUPS, regrets):
        values = improvements(frame)
        saved = frame[["solution_id", *OBJECTIVES, "total_budget_utilisation"]].copy()
        saved["group"] = group
        saved["mean_regret_shared_reference"] = r.mean(axis=1)
        saved["worst_regret_shared_reference"] = r.max(axis=1)
        for j, metric in enumerate(METRICS):
            saved[f"{metric.lower()}_improvement_pct"] = values[:, j]
            summary.append({"group": group, "policies": len(frame), "metric": metric,
                            "best_improvement_pct": values[:, j].max(),
                            "median_improvement_pct": np.median(values[:, j]),
                            "worst_improvement_pct": values[:, j].min()})
        records.append(saved)
    pd.concat(records, ignore_index=True).to_csv(output / "comparison_policies.csv", index=False,
                                               encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(output / "outcome_summary.csv", index=False, encoding="utf-8-sig")
    outcome_plot(frames, output)
    tradeoff_plot(frames, output)
    budget_plot(frames, output)
    regret_plot(regrets, output)
    manifest = {"scenario": "present", "old_run": str(args.old_run_dir.resolve()),
                "new_run": str(args.new_run_dir.resolve()),
                "revision_comparison": str(args.revision_dir.resolve()),
                "policy_counts": dict(zip(GROUPS, map(len, frames))),
                "regret_reference": "pooled corrected archived and newly optimized policies",
                "notes": ["Re-evaluated archived policies are not filtered to a new Pareto front.",
                          "Policy distributions are not statistical uncertainty intervals.",
                          "Shared-reference regrets differ from separate-run normalization."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Comparison plots and CSVs saved to {output.resolve()}")
    print(json.dumps(manifest["policy_counts"]))


if __name__ == "__main__":
    main()
