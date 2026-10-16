"""Plot policy-pattern comparisons for a completed to7 optimisation trial."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Select the latest complete output set. Set an explicit stamp to reproduce a
# past run after a newer continuation batch has finished.
RUN_STAMP = None
DATA_DIR = Path("data")
FIG_DIR = Path("figures/opt_to7")
CALENDAR_START_YEAR = 2009
SCENARIO_ORDER = ["present", "2C", "4C"]
SCENARIO_LABELS = {"present": "Present", "2C": "2℃", "4C": "4℃"}
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
        summaries = []
        for path in DATA_DIR.glob("opt_to7_*_*_summary.csv"):
            stem = path.name[: -len("_summary.csv")]
            if all(
                (DATA_DIR / f"{stem}_{suffix}.csv").exists()
                for suffix in ("decisions", "yearly")
            ):
                summaries.append(path)
        if not summaries:
            raise FileNotFoundError(
                "No complete optimisation output set was found in data/."
            )
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


def validate_inputs(summary, decisions, yearly):
    """Reject incomplete or ambiguous optimisation output before plotting."""
    key_columns = ["scenario", "pattern", "seed"]
    expected_keys = {
        (scenario, pattern, 42)
        for scenario in SCENARIO_ORDER
        for pattern in PATTERN_ORDER
    }
    summary_keys = set(summary[key_columns].itertuples(index=False, name=None))
    decision_keys = set(decisions[key_columns].itertuples(index=False, name=None))
    yearly_keys = set(yearly[key_columns].drop_duplicates().itertuples(index=False, name=None))
    if summary.duplicated(key_columns).any() or decisions.duplicated(key_columns).any():
        raise ValueError("Summary or decisions contains duplicate scenario/pattern/seed rows.")
    if summary_keys != expected_keys:
        raise ValueError(
            "The selected run is not the expected complete 12-case optimisation trial."
        )
    if not (summary_keys == decision_keys == yearly_keys):
        raise ValueError("Summary, decisions, and yearly CSV case keys do not match.")
    year_counts = yearly.groupby(key_columns)["year"].nunique()
    if not (year_counts == 15).all():
        raise ValueError("Every optimisation case must contain 15 calendar years.")


def succeeded(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def objective_score(row, pattern):
    score = float(row["cost_tiebreaker"])
    if pattern in {"balanced", "disaster_only"}:
        score += sum(
            float(row[column])
            for column in (
                "flood_damage_vs_baseline_ratio",
                "innundation_damage_vs_baseline_ratio",
                "landslide_risk_vs_baseline_ratio",
            )
        )
    if pattern == "balanced":
        score -= float(row["crop_revenue_vs_baseline_ratio"])
        score -= float(row["biodiversity_vs_baseline_ratio"])
    elif pattern == "agriculture_only":
        score -= float(row["crop_revenue_vs_baseline_ratio"])
    elif pattern == "ecosystem_only":
        score -= float(row["biodiversity_vs_baseline_ratio"])
    return score


def mark_better_stored_candidates(summary):
    """Flag named trial results beaten by another already saved policy candidate."""
    summary = summary.copy()
    summary["better_stored_candidate"] = ""
    for scenario in SCENARIO_ORDER:
        scenario_rows = summary[summary["scenario"] == scenario]
        for pattern in PATTERN_ORDER:
            own = scenario_rows[scenario_rows["pattern"] == pattern].iloc[0]
            ranked = sorted(
                (objective_score(row, pattern), row["pattern"])
                for _, row in scenario_rows.iterrows()
            )
            if ranked[0][0] < objective_score(own, pattern) - 1e-10:
                mask = (summary["scenario"] == scenario) & (
                    summary["pattern"] == pattern
                )
                summary.loc[mask, "better_stored_candidate"] = ranked[0][1]
    return summary


def display_pattern(row):
    label = PATTERN_LABELS[row["pattern"]]
    if not succeeded(row["optimizer_success"]):
        label += "†"
    if row.get("better_stored_candidate", ""):
        label += "‡"
    return label


def add_trial_notes(fig, data):
    notes = []
    if not all(succeeded(value) for value in data["optimizer_success"]):
        notes.append("† Reached the 60-generation limit; trial solution.")
    if "better_stored_candidate" in data and (
        data["better_stored_candidate"].astype(str) != ""
    ).any():
        notes.append("‡ Another stored pattern candidate scores better for this objective.")
    if notes:
        fig.text(0.01, 0.01, "  ".join(notes), fontsize=8)


def ordered(frame):
    return frame.set_index("pattern").reindex(PATTERN_ORDER).dropna(how="all").reset_index()


def save(fig, output_dir, name):
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_dir / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_outcome_changes(summary, output_dir, scenario):
    metrics = [
        ("flood_damage_vs_baseline_ratio", "Flood damage reduction"),
        ("innundation_damage_vs_baseline_ratio", "Inundation damage reduction"),
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
        ax.set_xticks(
            np.arange(len(data)),
            [display_pattern(row) for _, row in data.iterrows()],
            rotation=45,
            ha="right",
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Change from baseline (%)")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"{SCENARIO_LABELS[scenario]}: trial outcomes by objective pattern", y=1.03
    )
    add_trial_notes(fig, data)
    save(fig, output_dir, "01_pattern_outcome_changes.png")


def plot_budget_and_gdp(summary, output_dir, scenario):
    data = ordered(summary)
    x = np.arange(len(data))
    labels = [display_pattern(row) for _, row in data.iterrows()]
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

    fig.suptitle(
        f"{SCENARIO_LABELS[scenario]}: budget use and GDP diagnostic by objective pattern",
        y=1.02,
    )
    add_trial_notes(fig, data)
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

            start = CALENDAR_START_YEAR + int(decision[start_col])
            color = PATTERN_COLORS[pattern]
            if end_col is None:
                ax.scatter(start, position, color=color, marker="D", s=55, zorder=3)
            else:
                end = CALENDAR_START_YEAR + int(decision[end_col])
                ax.barh(position, end - start + 1, left=start, height=0.52, color=color, alpha=0.82)

        ax.set_yticks(range(len(policies)), [policy[0] for policy in policies])
        ax.set_title(display_pattern(decision), loc="left", fontsize=11)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()

    axes[-1].set_xlabel("Year")
    fig.suptitle(
        f"{SCENARIO_LABELS[scenario]}: policy timing by objective pattern", y=0.995
    )
    add_trial_notes(fig, data)
    save(fig, output_dir, "03_pattern_policy_schedules.png")


def main():
    stem, summary_path, decisions_path, yearly_path = input_paths()
    run_stamp = "_".join(stem.rsplit("_", 2)[-2:])
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    decisions = pd.read_csv(decisions_path, encoding="utf-8-sig")
    yearly = pd.read_csv(yearly_path, encoding="utf-8-sig")
    validate_inputs(summary, decisions, yearly)
    summary = mark_better_stored_candidates(summary)
    status = summary[
        [
            "scenario",
            "pattern",
            "seed",
            "optimizer_success",
            "better_stored_candidate",
        ]
    ]
    decisions = decisions.merge(
        status, on=["scenario", "pattern", "seed"], validate="one_to_one"
    )

    scenarios = [
        scenario for scenario in SCENARIO_ORDER if scenario in set(summary["scenario"])
    ]
    failed = summary[~summary["optimizer_success"].map(succeeded)]
    if not failed.empty:
        print(
            "Note: cases marked with † reached the generation limit and are trial "
            "solutions, not confirmed optima."
        )
    for _, row in summary[summary["better_stored_candidate"] != ""].iterrows():
        print(
            f"Note: {row['scenario']}/{row['pattern']} is marked ‡ because the "
            f"stored {row['better_stored_candidate']} candidate scores better for "
            "that objective; the plotted result was not substituted."
        )
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
