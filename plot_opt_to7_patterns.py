"""Plot policy-pattern comparisons for a completed to7 optimisation trial."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basin_config import load_active_basin


# Set explicitly, for example "260724_1719", to reproduce a past run.
RUN_STAMP = None
BASE_DIR = Path(__file__).resolve().parent
BASIN_KEY = load_active_basin()["key"]
PRIMARY_DATA_DIR = BASE_DIR / "results" / "optimization" / BASIN_KEY
DATA_DIR = (
    PRIMARY_DATA_DIR
    if list(PRIMARY_DATA_DIR.glob("opt_to7_*_summary.csv"))
    else BASE_DIR / "data"
)
FIG_DIR = BASE_DIR / "figures" / "opt_to7" / BASIN_KEY
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


def input_paths():
    if RUN_STAMP is None:
        summaries = list(DATA_DIR.glob("opt_to7_*_*_summary.csv"))
        if not summaries:
            raise FileNotFoundError("No optimisation summary CSV was found in data/.")
        summary = max(summaries, key=lambda path: path.stat().st_mtime)
    else:
        matches = list(DATA_DIR.glob(f"opt_to7_*_{RUN_STAMP}_summary.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one summary CSV for run stamp {RUN_STAMP}.")
        summary = matches[0]

    stem = summary.name[: -len("_summary.csv")]
    decisions = DATA_DIR / f"{stem}_decisions.csv"
    yearly = DATA_DIR / f"{stem}_yearly.csv"
    missing = [path.name for path in (decisions, yearly) if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete optimisation output set: " + ", ".join(missing))
    return stem, summary, decisions, yearly


def ordered(frame):
    return frame.set_index("pattern").reindex(PATTERN_ORDER).dropna(how="all").reset_index()


def save(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_changes(summary, output_dir, scenario):
    metrics = [
        ("flood_damage_vs_baseline_ratio", "Flood damage reduction"),
        ("innundation_damage_vs_baseline_ratio", "Innundation damage reduction"),
        ("landslide_risk_vs_baseline_ratio", "Landslide risk reduction"),
        ("crop_revenue_vs_baseline_ratio", "Crop revenue increase"),
        ("biodiversity_vs_baseline_ratio", "Biodiversity increase"),
    ]
    data = ordered(summary)
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.8), sharey=False)

    for ax, (column, title) in zip(axes, metrics):
        favourable_change = []
        for _, row in data.iterrows():
            ratio = float(row[column])
            favourable_change.append((1 - ratio) * 100 if "reduction" in title else (ratio - 1) * 100)
        ax.bar(
            np.arange(len(data)),
            favourable_change,
            color=[PATTERN_COLORS[pattern] for pattern in data["pattern"]],
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(data)), [PATTERN_LABELS[p] for p in data["pattern"]], rotation=45, ha="right")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Change from baseline (%)")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"{scenario}: optimisation outcomes by objective pattern", y=1.03)
    save(fig, output_dir, "01_pattern_outcome_changes.png")


def plot_budget_and_gdp(summary, output_dir, scenario):
    data = ordered(summary)
    x = np.arange(len(data))
    labels = [PATTERN_LABELS[pattern] for pattern in data["pattern"]]
    colors = [PATTERN_COLORS[pattern] for pattern in data["pattern"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    axes[0].bar(x, data["total_budget_utilisation"].astype(float) * 100, color=colors)
    axes[0].axhline(100, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("Total budget utilisation (%)")
    axes[0].set_title("Total budget use")
    axes[0].grid(axis="y", alpha=0.25)

    gdp_change = (data["daily_total_gdp_vs_baseline_ratio"].astype(float) - 1) * 100
    axes[1].bar(x, gdp_change, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Change from baseline (%)")
    axes[1].set_title("Total GDP diagnostic")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"{scenario}: budget use and GDP diagnostic by objective pattern", y=1.02)
    save(fig, output_dir, "02_pattern_budget_and_gdp.png")


def plot_policy_schedules(decisions, output_dir, scenario):
    policies = [
        ("Drainage", "drainage_start_year", None, "drainage_investment_amount"),
        ("Dam", "dam_start_year", None, "dam_investment_amount"),
        ("Levee", "levee_start_year", None, "levee_investment_amount"),
        ("House elevation", "house_elevation_start_year", "house_elevation_end_year", "house_elevation_per_year"),
        ("Migration", "migration_start_year", "migration_end_year", "migration_per_year"),
        ("Paddy dam", "paddy_dam_start_year", "paddy_dam_end_year", "annual_paddy_dam_investment"),
        ("Breeding", "breeding_start_year", "breeding_end_year", "breeding_enabled"),
        ("Forest management", "forest_management_start_year", "forest_management_end_year", "annual_forest_management_conversion_area"),
    ]
    data = ordered(decisions)
    fig, axes = plt.subplots(len(data), 1, figsize=(11, 10), sharex=True, sharey=True)
    if len(data) == 1:
        axes = [axes]

    for ax, (_, decision) in zip(axes, data.iterrows()):
        pattern = decision["pattern"]
        for position, (label, start_col, end_col, value_col) in enumerate(policies):
            value = float(decision[value_col])
            if label == "Breeding":
                active = int(value) == 1
            else:
                active = value > 0
            if not active:
                continue

            start = int(decision[start_col])
            color = PATTERN_COLORS[pattern]
            if end_col is None:
                ax.scatter(start, position, color=color, marker="D", s=55, zorder=3)
            else:
                end = int(decision[end_col])
                ax.barh(position, end - start + 1, left=start, height=0.52, color=color, alpha=0.82)

        ax.set_yticks(range(len(policies)), [policy[0] for policy in policies])
        ax.set_title(PATTERN_LABELS[pattern], loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()

    axes[-1].set_xlabel("Model year index (0 = 2009)")
    fig.suptitle(f"{scenario}: policy timing by objective pattern", y=0.995)
    save(fig, output_dir, "03_pattern_policy_schedules.png")


def main():
    stem, summary_path, decisions_path, yearly_path = input_paths()
    run_stamp = "_".join(stem.rsplit("_", 2)[-2:])
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    decisions = pd.read_csv(decisions_path, encoding="utf-8-sig")
    _ = pd.read_csv(yearly_path, encoding="utf-8-sig")

    scenarios = list(summary["scenario"].drop_duplicates())
    for scenario in scenarios:
        scenario_summary = summary[summary["scenario"] == scenario].copy()
        scenario_decisions = decisions[decisions["scenario"] == scenario].copy()
        output_dir = FIG_DIR / run_stamp / "patterns" / scenario
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_outcome_changes(scenario_summary, output_dir, scenario)
        plot_budget_and_gdp(scenario_summary, output_dir, scenario)
        plot_policy_schedules(scenario_decisions, output_dir, scenario)
        print(f"Saved {scenario} pattern-comparison figures to: {output_dir}")


if __name__ == "__main__":
    main()
