"""Plot completed optimisation rows from an in-progress checkpoint."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set this to the in-progress run to inspect. Checkpoint files are updated after
# each completed scenario/pattern/seed combination, so this is safe while a run
# is continuing (rerun the script after later checkpoints are written).
RUN_STAMP = "260726_1459"
CHECKPOINT_DIR = Path("data/checkpoints") / f"opt_to7_standard_{RUN_STAMP}"
OUTPUT_DIR = Path("figures/opt_to7") / RUN_STAMP / "checkpoint"
PATTERN_ORDER = ["balanced", "disaster_only", "agriculture_only", "ecosystem_only"]
PATTERN_LABELS = {
    "balanced": "Balanced",
    "disaster_only": "Disaster only",
    "agriculture_only": "Agriculture only",
    "ecosystem_only": "Ecosystem only",
}
PATTERN_COLORS = {
    "balanced": "#4c78a8",
    "disaster_only": "#e45756",
    "agriculture_only": "#59a14f",
    "ecosystem_only": "#7a5195",
}


def ordered(frame):
    frame = frame.copy()
    frame["pattern"] = pd.Categorical(frame["pattern"], PATTERN_ORDER, ordered=True)
    return frame.sort_values(["scenario", "pattern"])


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(summary):
    metrics = [
        ("flood_damage_vs_baseline_ratio", "Flood damage reduction"),
        ("innundation_damage_vs_baseline_ratio", "Innundation damage reduction"),
        ("landslide_risk_vs_baseline_ratio", "Landslide risk reduction"),
        ("crop_revenue_vs_baseline_ratio", "Crop revenue increase"),
        ("biodiversity_vs_baseline_ratio", "Biodiversity increase"),
    ]
    scenarios = list(summary["scenario"].drop_duplicates())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 4.2 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        rows = ordered(summary[summary["scenario"] == scenario])
        x = np.arange(len(metrics))
        width = 0.75 / len(rows)
        for index, (_, row) in enumerate(rows.iterrows()):
            values = []
            for column, label in metrics:
                ratio = float(row[column])
                values.append((1 - ratio) * 100 if "reduction" in label else (ratio - 1) * 100)
            ax.bar(
                x + (index - (len(rows) - 1) / 2) * width,
                values,
                width=width,
                color=PATTERN_COLORS[str(row["pattern"])],
                label=PATTERN_LABELS[str(row["pattern"])],
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{scenario}: completed objective patterns", loc="left")
        ax.set_ylabel("Change from baseline (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xticks(x, [label for _, label in metrics], rotation=18, ha="right")
    fig.suptitle("Checkpoint optimisation outcomes", y=1.01)
    save(fig, "01_checkpoint_outcomes.png")


def plot_budget(summary):
    scenarios = list(summary["scenario"].drop_duplicates())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 3.8 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        rows = ordered(summary[summary["scenario"] == scenario])
        x = np.arange(len(rows))
        ax.bar(x, rows["total_budget_utilisation"].astype(float) * 100,
               color=[PATTERN_COLORS[str(value)] for value in rows["pattern"]])
        ax.axhline(100, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{scenario}: total policy budget use", loc="left")
        ax.set_ylabel("Budget utilisation (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x, [PATTERN_LABELS[str(value)] for value in rows["pattern"]], rotation=25, ha="right")
    fig.suptitle("Checkpoint budget diagnostics", y=1.01)
    save(fig, "02_checkpoint_budget.png")


def main():
    summary_path = CHECKPOINT_DIR / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Checkpoint summary not found: {summary_path}")
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty:
        raise ValueError("The checkpoint has no completed optimisation rows yet.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_outcomes(summary)
    plot_budget(summary)
    print(f"Read completed rows from: {summary_path}")
    print(f"Saved checkpoint figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
