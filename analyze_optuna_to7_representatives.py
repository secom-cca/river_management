"""Select and visualise representative policies from an Optuna Pareto set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

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
MINIMIZE = np.array([True, True, True, False, False])
ROLE_ORDER = (
    "flood_priority",
    "inundation_priority",
    "ecosystem_priority",
    "agriculture_priority",
    "low_budget",
    "balanced",
)
ROLE_LABELS = {
    "flood_priority": "Flood",
    "inundation_priority": "Inundation",
    "ecosystem_priority": "Ecosystem",
    "agriculture_priority": "Agriculture",
    "low_budget": "Low budget",
    "balanced": "Balanced",
}
PERFORMANCE_COLUMNS = (
    "flood_improvement_pct",
    "inundation_improvement_pct",
    "landslide_improvement_pct",
    "crop_improvement_pct",
    "biodiversity_improvement_pct",
    "annual_budget_pct",
    "total_budget_pct",
)
PERFORMANCE_LABELS = (
    "Flood\nimprovement",
    "Inundation\nimprovement",
    "Landslide\nimprovement",
    "Crop revenue\nimprovement",
    "Biodiversity\nimprovement",
    "Annual budget\nuse",
    "Total budget\nuse",
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
CONTINUOUS_LABELS = (
    "Drainage\n(billion yen)",
    "Dam\n(billion yen)",
    "Levee\n(billion yen)",
    "House elevation\n(houses/year)",
    "Migration\n(houses/year)",
    "Paddy dam\n(billion yen/year)",
    "Forest management\n(ha/year)",
)
CONTINUOUS_SCALES = np.array([1e9, 1e9, 1e9, 1.0, 1.0, 1e9, 1.0])
CONTINUOUS_BOUNDS = np.array([5.0, 5.0, 5.0, 1000.0, 1000.0, 5.0, 3000.0])
SCHEDULE_PREFIXES = (
    "drainage",
    "dam",
    "levee",
    "house_elevation",
    "migration",
    "paddy_dam",
    "forest_management",
    "breeding",
)
SCHEDULE_LABELS = (
    "Drainage",
    "Dam",
    "Levee",
    "House elevation",
    "Migration",
    "Paddy dam",
    "Forest management",
    "Breeding",
)
YEARLY_COLUMNS = (
    "policy_additional_cost",
    "flood_damage",
    "innundation_damage",
    "landslide_risk",
    "crop_revenue",
    "biodiversity",
)
YEARLY_LABELS = (
    "Policy cost (billion yen)",
    "Flood damage (billion yen)",
    "Inundation damage (billion yen)",
    "Landslide risk",
    "Crop revenue (billion yen)",
    "Biodiversity",
)
YEARLY_SCALES = (1e9, 1e9, 1e9, 1.0, 1e9, 1.0)


def latest_run(data_dir: Path) -> Path:
    candidates = [
        path
        for path in data_dir.glob("optuna_to7_*")
        if path.is_dir() and (path / "config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No Optuna to7 run directory was found.")
    return max(candidates, key=lambda path: (path / "config.json").stat().st_mtime)


def objective_losses(frame: pd.DataFrame) -> pd.DataFrame:
    """Return per-objective losses scaled from zero (best) to one (worst)."""
    values = frame.loc[:, OBJECTIVES].to_numpy(dtype=float)
    oriented = values.copy()
    oriented[:, ~MINIMIZE] *= -1.0
    low = oriented.min(axis=0)
    span = oriented.max(axis=0) - low
    span[span <= 1e-15] = 1.0
    losses = (oriented - low) / span
    return pd.DataFrame(
        losses,
        index=frame.index,
        columns=[f"normalized_loss_{name}" for name in OBJECTIVES],
    )


def select_representatives(pareto: pd.DataFrame) -> pd.DataFrame:
    """Select six interpretable roles and merge roles assigned to one solution."""
    required = {
        "solution_id",
        "scenario",
        "total_budget_utilisation",
        *OBJECTIVES,
    }
    missing = required - set(pareto.columns)
    if missing:
        raise ValueError(f"Pareto summary is missing columns: {sorted(missing)}")
    if pareto.empty:
        raise ValueError("Pareto summary is empty.")
    values = pareto.loc[:, OBJECTIVES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Pareto summary contains non-finite objective values.")

    losses = objective_losses(pareto)
    work = pd.concat([pareto.copy(), losses], axis=1)
    loss_columns = list(losses.columns)
    work["maximum_normalized_regret"] = work[loss_columns].max(axis=1)
    ecosystem_columns = [
        "normalized_loss_landslide_risk_ratio",
        "normalized_loss_biodiversity_ratio",
    ]
    work["ecosystem_normalized_regret"] = work[ecosystem_columns].max(axis=1)

    chosen = {
        "flood_priority": work["flood_damage_ratio"].idxmin(),
        "inundation_priority": work["innundation_damage_ratio"].idxmin(),
        "ecosystem_priority": work["ecosystem_normalized_regret"].idxmin(),
        "agriculture_priority": work["crop_revenue_ratio"].idxmax(),
        "low_budget": work["total_budget_utilisation"].idxmin(),
        "balanced": work["maximum_normalized_regret"].idxmin(),
    }
    roles_by_index: dict[int, list[str]] = {}
    for role in ROLE_ORDER:
        roles_by_index.setdefault(chosen[role], []).append(role)

    rows = []
    for index, roles in roles_by_index.items():
        row = work.loc[index].copy()
        row["selection_roles"] = ";".join(roles)
        row["selection_label"] = "/".join(ROLE_LABELS[role] for role in roles)
        row["selection_order"] = min(ROLE_ORDER.index(role) for role in roles)
        rows.append(row)
    selected = pd.DataFrame(rows).sort_values("selection_order").reset_index(drop=True)
    return selected


def add_improvement_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["flood_improvement_pct"] = (1.0 - output["flood_damage_ratio"]) * 100
    output["inundation_improvement_pct"] = (
        1.0 - output["innundation_damage_ratio"]
    ) * 100
    output["landslide_improvement_pct"] = (
        1.0 - output["landslide_risk_ratio"]
    ) * 100
    output["crop_improvement_pct"] = (output["crop_revenue_ratio"] - 1.0) * 100
    output["biodiversity_improvement_pct"] = (
        output["biodiversity_ratio"] - 1.0
    ) * 100
    output["annual_budget_pct"] = output["annual_budget_utilisation"] * 100
    output["total_budget_pct"] = output["total_budget_utilisation"] * 100
    return output


def policy_label(row: pd.Series) -> str:
    return f"{SCENARIO_LABELS.get(row['scenario'], row['scenario'])} | {row['selection_label']}"


def _favourable_colours(values: np.ndarray) -> np.ndarray:
    favourable = values.copy()
    favourable[:, -2:] *= -1.0
    low = favourable.min(axis=0)
    span = favourable.max(axis=0) - low
    span[span <= 1e-15] = 1.0
    return (favourable - low) / span


def plot_performance(selected: pd.DataFrame, output: Path) -> None:
    values = selected.loc[:, PERFORMANCE_COLUMNS].to_numpy(dtype=float)
    colours = _favourable_colours(values)
    height = max(5.2, 0.48 * len(selected) + 2.1)
    fig, ax = plt.subplots(figsize=(13.5, height))
    image = ax.imshow(colours, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for row in range(len(selected)):
        for col in range(len(PERFORMANCE_COLUMNS)):
            suffix = "%"
            ax.text(
                col,
                row,
                f"{values[row, col]:.1f}{suffix}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    ax.set_xticks(range(len(PERFORMANCE_COLUMNS)), PERFORMANCE_LABELS)
    ax.set_yticks(range(len(selected)), [policy_label(row) for _, row in selected.iterrows()])
    ax.set_title("Representative policy performance (annotations are baseline-relative percentages)")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative desirability within selected policies")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _schedule_annotation(row: pd.Series, prefix: str) -> tuple[str, float]:
    if prefix == "breeding" and not bool(int(float(row["breeding_enabled"]))):
        return "Off", np.nan
    start = row.get(f"{prefix}_start_calendar_year")
    if pd.isna(start):
        start_index = row.get(f"{prefix}_start_year")
        if pd.isna(start_index):
            return "Off", np.nan
        start = 2009 + int(float(start_index))
    start = int(float(start))
    end = row.get(f"{prefix}_end_calendar_year")
    if pd.isna(end):
        return str(start), float(start)
    end = int(float(end))
    return (str(start) if start == end else f"{start}–{end}"), float(start)


def plot_policy_design(decisions: pd.DataFrame, output: Path) -> None:
    raw = decisions.loc[:, CONTINUOUS_DECISIONS].to_numpy(dtype=float) / CONTINUOUS_SCALES
    normalized = np.clip(raw / CONTINUOUS_BOUNDS, 0, 1)
    schedule_values = np.full((len(decisions), len(SCHEDULE_PREFIXES)), np.nan)
    schedule_text: list[list[str]] = []
    for row_index, (_, row) in enumerate(decisions.iterrows()):
        text_row = []
        for col_index, prefix in enumerate(SCHEDULE_PREFIXES):
            annotation, start = _schedule_annotation(row, prefix)
            text_row.append(annotation)
            schedule_values[row_index, col_index] = start
        schedule_text.append(text_row)

    height = max(6.0, 0.5 * len(decisions) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(20, height), gridspec_kw={"width_ratios": [1, 1.15]})
    left = axes[0].imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row in range(len(decisions)):
        for col in range(len(CONTINUOUS_DECISIONS)):
            axes[0].text(col, row, f"{raw[row, col]:.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_xticks(range(len(CONTINUOUS_LABELS)), CONTINUOUS_LABELS, rotation=25, ha="right")
    axes[0].set_yticks(range(len(decisions)), [policy_label(row) for _, row in decisions.iterrows()])
    axes[0].set_title("Policy magnitudes")
    cbar_left = fig.colorbar(left, ax=axes[0], fraction=0.03, pad=0.02)
    cbar_left.set_label("Fraction of decision upper bound")

    masked = np.ma.masked_invalid(schedule_values)
    right = axes[1].imshow(masked, cmap="viridis", vmin=2009, vmax=2023, aspect="auto")
    axes[1].set_facecolor("#eeeeee")
    for row in range(len(decisions)):
        for col in range(len(SCHEDULE_PREFIXES)):
            axes[1].text(col, row, schedule_text[row][col], ha="center", va="center", fontsize=7)
    axes[1].set_xticks(range(len(SCHEDULE_LABELS)), SCHEDULE_LABELS, rotation=25, ha="right")
    axes[1].set_yticks(range(len(decisions)), [policy_label(row) for _, row in decisions.iterrows()])
    axes[1].set_title("Implementation period (start–end calendar year)")
    cbar_right = fig.colorbar(right, ax=axes[1], fraction=0.03, pad=0.02)
    cbar_right.set_label("Start calendar year")
    fig.suptitle("Representative policy design", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_yearly(yearly: pd.DataFrame, selected: pd.DataFrame, output: Path) -> None:
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in set(selected["scenario"])]
    fig, axes = plt.subplots(
        len(scenarios),
        len(YEARLY_COLUMNS),
        figsize=(23, 4.2 * len(scenarios)),
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10")
    for row_index, scenario in enumerate(scenarios):
        policies = selected[selected["scenario"] == scenario].reset_index(drop=True)
        for policy_index, (_, policy) in enumerate(policies.iterrows()):
            subset = yearly[yearly["solution_id"] == policy["solution_id"]].sort_values("year")
            if len(subset) != 15:
                raise ValueError(f"{policy['solution_id']} has {len(subset)} yearly rows; expected 15.")
            for col_index, (column, scale) in enumerate(zip(YEARLY_COLUMNS, YEARLY_SCALES)):
                axes[row_index, col_index].plot(
                    subset["year"],
                    subset[column] / scale,
                    color=cmap(policy_index),
                    linewidth=1.7,
                    label=policy["selection_label"],
                )
        for col_index, label in enumerate(YEARLY_LABELS):
            ax = axes[row_index, col_index]
            ax.set_title(label)
            ax.grid(alpha=0.22)
            ax.tick_params(axis="x", rotation=45)
            if col_index == 0:
                ax.set_ylabel(SCENARIO_LABELS.get(scenario, scenario))
        axes[row_index, -1].legend(loc="best", fontsize=7, frameon=False)
    fig.suptitle("Annual outcomes of representative policies", y=1.005)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIO_ORDER)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = (args.run_dir or latest_run(Path("data"))).resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    scenarios = args.scenarios or list(config["scenarios"])
    output_dir = (
        args.output_dir or Path("analysis") / run_dir.name / "representatives"
    ).resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    selected_frames = []
    decision_frames = []
    yearly_frames = []
    for scenario in scenarios:
        summary_path = run_dir / f"pareto_summary_{scenario}.csv"
        decisions_path = run_dir / f"pareto_decisions_{scenario}.csv"
        yearly_path = run_dir / f"pareto_yearly_{scenario}.csv"
        summary = pd.read_csv(summary_path)
        decisions = pd.read_csv(decisions_path)
        yearly = pd.read_csv(yearly_path)
        selected = select_representatives(summary)
        selected_frames.append(selected)
        identifiers = set(selected["solution_id"])
        selected_decisions = decisions[decisions["solution_id"].isin(identifiers)].copy()
        selected_yearly = yearly[yearly["solution_id"].isin(identifiers)].copy()
        if len(selected_decisions) != len(selected):
            raise ValueError(f"Decision rows do not match selected policies for {scenario}.")
        decision_frames.append(selected_decisions)
        yearly_frames.append(selected_yearly)

    selected = add_improvement_columns(pd.concat(selected_frames, ignore_index=True))
    selected["scenario_order"] = selected["scenario"].map(
        {name: index for index, name in enumerate(SCENARIO_ORDER)}
    )
    selected = selected.sort_values(["scenario_order", "selection_order"]).reset_index(drop=True)
    decisions = pd.concat(decision_frames, ignore_index=True)
    decisions = decisions.merge(
        selected[["solution_id", "selection_roles", "selection_label", "selection_order", "scenario_order"]],
        on="solution_id",
        how="inner",
        validate="one_to_one",
    ).sort_values(["scenario_order", "selection_order"]).reset_index(drop=True)
    yearly = pd.concat(yearly_frames, ignore_index=True)
    yearly = yearly.merge(
        selected[["solution_id", "selection_roles", "selection_label", "selection_order", "scenario_order"]],
        on="solution_id",
        how="inner",
        validate="many_to_one",
    ).sort_values(["scenario_order", "selection_order", "year"]).reset_index(drop=True)

    _atomic_csv(selected.drop(columns="scenario_order"), output_dir / "representative_policies.csv")
    _atomic_csv(decisions.drop(columns="scenario_order"), output_dir / "representative_decisions.csv")
    _atomic_csv(yearly.drop(columns="scenario_order"), output_dir / "representative_yearly.csv")
    metadata = {
        "source_run_dir": str(run_dir),
        "scenarios": scenarios,
        "selection_roles": list(ROLE_ORDER),
        "normalization": "Per-scenario Pareto min-max after converting all objectives to losses.",
        "representative_count": len(selected),
    }
    (output_dir / "selection_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_performance(selected, figure_dir / "01_representative_performance.png")
    plot_policy_design(decisions, figure_dir / "02_policy_design.png")
    plot_yearly(yearly, selected, figure_dir / "03_yearly_outcomes.png")
    print(f"Selected {len(selected)} unique representative policies.")
    print(f"Saved representative analysis: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
