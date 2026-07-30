"""Create six diagnostic figures for a completed to7 optimisation run."""

from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set explicitly, for example "260723_2017", to reproduce a past run.
# Leave as None to use the latest complete optimisation output set.
RUN_STAMP = None
DATA_DIR = Path("data")
FIG_DIR = Path("figures/opt_to7")
SCENARIO_ORDER = ["present", "2C", "4C"]
SCENARIO_LABELS = {"present": "Present", "2C": "2℃", "4C": "4℃"}
SCENARIO_COLORS = {"present": "#1f77b4", "2C": "#ff7f0e", "4C": "#d62728"}
# A single scenario-comparison figure must use one objective pattern. The
# optimisation CSV has one row for each scenario/pattern combination.
OBJECTIVE_PATTERN = "balanced"
ACTIVE_SCENARIOS = []
BUDGET_LIMIT = 5_000_000_000


def input_paths():
    if RUN_STAMP is None:
        candidates = list(DATA_DIR.glob("opt_to7_*_*_summary.csv"))
        if not candidates:
            raise FileNotFoundError("No optimisation summary CSV was found in data/.")
        summary = max(candidates, key=lambda path: path.stat().st_mtime)
        suffix = "_summary.csv"
        stem = summary.name[: -len(suffix)]
        run_stamp = "_".join(stem.rsplit("_", 2)[-2:])
    else:
        run_stamp = RUN_STAMP
        matches = list(DATA_DIR.glob(f"opt_to7_*_{run_stamp}_summary.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one summary CSV for run stamp {run_stamp}.")
        summary = matches[0]
        stem = summary.name[: -len("_summary.csv")]

    decisions = DATA_DIR / f"{stem}_decisions.csv"
    yearly = DATA_DIR / f"{stem}_yearly.csv"
    missing = [path.name for path in (summary, decisions, yearly) if not path.exists()]
    if missing:
        raise FileNotFoundError("Incomplete optimisation output set: " + ", ".join(missing))
    return run_stamp, summary, decisions, yearly


def ordered(frame):
    return frame.set_index("scenario").reindex(ACTIVE_SCENARIOS).reset_index()


def save(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(output_dir / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_objective_changes(summary, output_dir):
    """Five objectives as favourable changes from the no-policy baseline."""
    metrics = [
        ("flood_damage_vs_baseline_ratio", "Flood damage reduction"),
        ("innundation_damage_vs_baseline_ratio", "Innundation damage reduction"),
        ("landslide_risk_vs_baseline_ratio", "Landslide risk reduction"),
        ("crop_revenue_vs_baseline_ratio", "Crop revenue increase"),
        ("biodiversity_vs_baseline_ratio", "Biodiversity increase"),
    ]
    data = ordered(summary)
    x = np.arange(len(metrics))
    width = 0.75 / len(ACTIVE_SCENARIOS)
    fig, ax = plt.subplots(figsize=(11, 5.4))

    for index, scenario in enumerate(ACTIVE_SCENARIOS):
        row = data[data["scenario"] == scenario].iloc[0]
        values = []
        for column, label in metrics:
            ratio = float(row[column])
            values.append((1 - ratio) * 100 if "reduction" in label else (ratio - 1) * 100)
        ax.bar(
            x + (index - (len(ACTIVE_SCENARIOS) - 1) / 2) * width,
            values,
            width=width,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, [label for _, label in metrics], rotation=18, ha="right")
    ax.set_ylabel("Improvement from no-policy baseline (%)")
    ax.set_title("Optimised outcomes relative to the no-policy baseline")
    ax.legend(title="Climate scenario", frameon=False)
    save(fig, output_dir, "01_objective_changes.png")


def plot_annual_budget(yearly, output_dir):
    """Annual policy costs separated into known cost categories."""
    fig, axes = plt.subplots(len(ACTIVE_SCENARIOS), 1, figsize=(11, 3 * len(ACTIVE_SCENARIOS)), sharex=True, sharey=True)
    if len(ACTIVE_SCENARIOS) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, ACTIVE_SCENARIOS):
        data = yearly[yearly["scenario"] == scenario].copy()
        data = data.sort_values("year")
        breeding = data["breeding_actual_cost"].to_numpy(dtype=float)
        paddy = data["paddy_dam_actual_cost"].to_numpy(dtype=float)
        forest = data["additional_forest_management_cost"].to_numpy(dtype=float)
        total = data["policy_additional_cost"].to_numpy(dtype=float)
        other = np.maximum(total - breeding - paddy - forest, 0)
        years = data["year"].to_numpy()
        scale = 1e8
        ax.bar(
            years,
            other / scale,
            label="Drainage, dam, levee, elevation and migration",
            color="#6c757d",
        )
        ax.bar(years, paddy / scale, bottom=other / scale, label="Paddy dam", color="#4c9f70")
        ax.bar(years, forest / scale, bottom=(other + paddy) / scale, label="Forest management", color="#2f6f4e")
        ax.bar(years, breeding / scale, bottom=(other + paddy + forest) / scale, label="Breeding", color="#9b59b6")
        ax.axhline(BUDGET_LIMIT / scale, color="black", linestyle="--", linewidth=1, label="Annual budget limit")
        ax.set_title(SCENARIO_LABELS[scenario], loc="left", fontsize=11)
        ax.set_ylabel("Annual cost (100 M Yen)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.06), frameon=False)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Annual policy portfolio and budget limit", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / "02_annual_policy_budget.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_policy_schedule(decisions, output_dir):
    """Gantt-style view of each selected intervention schedule."""
    rows = [
        ("Drainage", "drainage_start_year", None, "drainage_investment_amount", "#4c78a8"),
        ("Dam", "dam_start_year", None, "dam_investment_amount", "#f58518"),
        ("Levee", "levee_start_year", None, "levee_investment_amount", "#e45756"),
        ("House elevation", "house_elevation_start_year", "house_elevation_end_year", "house_elevation_per_year", "#72b7b2"),
        ("Migration", "migration_start_year", "migration_end_year", "migration_per_year", "#54a24b"),
        ("Paddy dam", "paddy_dam_start_year", "paddy_dam_end_year", "annual_paddy_dam_investment", "#59a14f"),
        ("Breeding", "breeding_start_year", "breeding_end_year", "breeding_enabled", "#b279a2"),
        ("Forest management", "forest_management_start_year", "forest_management_end_year", "annual_forest_management_conversion_area", "#2f6f4e"),
    ]
    fig, axes = plt.subplots(len(ACTIVE_SCENARIOS), 1, figsize=(11, 3.4 * len(ACTIVE_SCENARIOS)), sharex=True)
    if len(ACTIVE_SCENARIOS) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, ACTIVE_SCENARIOS):
        decision = decisions[decisions["scenario"] == scenario].iloc[0]
        for position, (label, start_col, end_col, value_col, color) in enumerate(rows):
            start = int(decision[start_col])
            value = float(decision[value_col])
            active = value > 0
            if label == "Breeding":
                active = int(value) == 1
            if not active:
                continue
            if end_col is None:
                ax.scatter(start, position, color=color, marker="D", s=65, zorder=3)
            else:
                end = int(decision[end_col])
                ax.barh(position, end - start + 1, left=start, height=0.52, color=color, alpha=0.85)
                if label == "Breeding":
                    text = "50 M Yen/year"
                elif "investment" in value_col:
                    text = f"{value / 1e9:.1f} B Yen/year"
                elif "area" in value_col:
                    text = f"{value:,.0f} ha/year"
                else:
                    text = f"{value:,.0f} households/year"
                ax.text(start + 0.1, position, text, va="center", fontsize=8)
        ax.set_yticks(range(len(rows)), [row[0] for row in rows])
        ax.set_title(SCENARIO_LABELS[scenario], loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()
    axes[-1].set_xlabel("Model year index (0 = 2009)")
    fig.legend(
        handles=[
            Line2D([], [], marker="D", linestyle="None", color="#555555", label="One-time infrastructure investment"),
            Patch(color="#999999", label="Active implementation period"),
        ],
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        frameon=False,
    )
    fig.suptitle("Optimised policy schedules", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "03_policy_schedule.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_outcomes(yearly, output_dir):
    """Optimised annual trajectories; baseline annual trajectories were not saved."""
    metrics = [
        ("flood_damage", "Flood damage", 1e8),
        ("innundation_damage", "Innundation damage", 1e8),
        ("landslide_risk", "Landslide risk", 1),
        ("crop_revenue", "Crop revenue", 1e9),
        ("biodiversity", "Biodiversity", 1),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12.5, 14), sharex=True)
    for ax, (column, label, scale) in zip(axes, metrics):
        for scenario in ACTIVE_SCENARIOS:
            data = yearly[yearly["scenario"] == scenario].sort_values("year")
            ax.plot(data["year"], data[column] / scale, color=SCENARIO_COLORS[scenario], linewidth=2, label=SCENARIO_LABELS[scenario])
        suffix = " (100 M Yen)" if scale == 1e8 else " (B Yen)" if scale == 1e9 else ""
        ax.set_ylabel(label + suffix)
        ax.grid(alpha=0.25)
    axes[0].legend(title="Climate scenario", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42), frameon=False)
    axes[-1].set_xlabel("Year")
    fig.suptitle("Annual outcomes for optimised policy portfolios", y=0.995)
    # Reserve room for long outcome labels and separate the five panels.
    fig.tight_layout(rect=[0.22, 0.02, 0.99, 0.96], h_pad=1.2)
    fig.savefig(output_dir / "04_yearly_optimised_outcomes.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_policy_mechanisms(yearly, output_dir):
    """Show the principal state variables affected by selected policies."""
    metrics = [
        ("paddy_dam_ratio", "Paddy dam ratio", 1),
        ("forest_function_coef", "Forest function coefficient", 1),
        ("managed_plantation_forest_area", "Managed plantation forest area (ha)", 1),
        ("unmanaged_plantation_forest_area", "Unmanaged plantation forest area (ha)", 1),
        ("accumulated_breeding_investment", "Cumulative breeding investment (100 M Yen)", 1e8),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12.5, 14), sharex=True)
    for ax, (column, label, scale) in zip(axes, metrics):
        for scenario in ACTIVE_SCENARIOS:
            data = yearly[yearly["scenario"] == scenario].sort_values("year")
            ax.plot(data["year"], data[column] / scale, color=SCENARIO_COLORS[scenario], linewidth=2, label=SCENARIO_LABELS[scenario])
            if column == "accumulated_breeding_investment":
                success = data["breeding_success"].astype(float) > 0
                ax.scatter(data.loc[success, "year"], data.loc[success, column] / scale, color=SCENARIO_COLORS[scenario], marker="*", s=60, zorder=3)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].legend(title="Climate scenario", ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.42), frameon=False)
    axes[-1].legend(
        handles=[
            Line2D(
                [],
                [],
                marker="*",
                linestyle="None",
                color="#555555",
                markersize=10,
                label="Breeding success (cumulative investment >= 500 M Yen)",
            )
        ],
        loc="upper left",
        frameon=False,
    )
    axes[-1].set_xlabel("Year")
    fig.suptitle("Policy mechanisms in the optimised portfolios", y=0.995)
    # Long state-variable labels need a wide left margin and panel spacing.
    fig.tight_layout(rect=[0.22, 0.02, 0.99, 0.96], h_pad=1.2)
    fig.savefig(output_dir / "05_policy_mechanisms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cost_gdp_tradeoff(summary, output_dir):
    """Diagnostic trade-off: costs are constrained, not an optimisation objective."""
    data = ordered(summary)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for scenario in ACTIVE_SCENARIOS:
        row = data[data["scenario"] == scenario].iloc[0]
        x = float(row["total_budget_utilisation"]) * 100
        y = (float(row["daily_total_gdp_vs_baseline_ratio"]) - 1) * 100
        ax.scatter(x, y, s=110, color=SCENARIO_COLORS[scenario], label=SCENARIO_LABELS[scenario], zorder=3)
        ax.annotate(SCENARIO_LABELS[scenario], (x, y), xytext=(6, 6), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Total budget utilisation (%)")
    ax.set_ylabel("Change in total GDP from baseline (%)")
    ax.set_title("Diagnostic cost and GDP trade-off")
    ax.text(0.02, 0.02, "GDP is not included in the optimisation objective.", transform=ax.transAxes, fontsize=9)
    ax.grid(alpha=0.25)
    save(fig, output_dir, "06_cost_gdp_tradeoff.png")


def main():
    global ACTIVE_SCENARIOS
    run_stamp, summary_path, decisions_path, yearly_path = input_paths()
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    decisions = pd.read_csv(decisions_path, encoding="utf-8-sig")
    yearly = pd.read_csv(yearly_path, encoding="utf-8-sig")

    summary = summary[summary["pattern"] == OBJECTIVE_PATTERN].copy()
    decisions = decisions[decisions["pattern"] == OBJECTIVE_PATTERN].copy()
    yearly = yearly[yearly["pattern"] == OBJECTIVE_PATTERN].copy()
    if summary.empty:
        raise ValueError(f"Objective pattern not found: {OBJECTIVE_PATTERN}")
    ACTIVE_SCENARIOS = [
        scenario for scenario in SCENARIO_ORDER if scenario in set(summary["scenario"])
    ]
    output_dir = FIG_DIR / run_stamp / OBJECTIVE_PATTERN
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_objective_changes(summary, output_dir)
    plot_annual_budget(yearly, output_dir)
    plot_policy_schedule(decisions, output_dir)
    plot_yearly_outcomes(yearly, output_dir)
    plot_policy_mechanisms(yearly, output_dir)
    plot_cost_gdp_tradeoff(summary, output_dir)
    print(
        f"Saved six figures for pattern '{OBJECTIVE_PATTERN}' and scenarios "
        f"{ACTIVE_SCENARIOS} to: {output_dir}"
    )


if __name__ == "__main__":
    main()
