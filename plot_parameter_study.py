"""Create static PNG figures from a completed parameter-study CSV set."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from basin_config import ACTIVE_BASIN_ENV, BASE_DIR, load_basin_config


PREFIX = "para_to7_all_scenarios_"
SUFFIXES = ("parameter_sets", "overall_summary", "yearly_summary")
SCENARIO_PRIORITY = ["present", "2C", "4C"]

METRIC_LABELS = {
    "yearly_gdp_total": "Total GDP",
    "yearly_crop_revenue": "Crop revenue",
    "yield_per_10a": "Rice yield per 10a",
    "damaged_paddy_field_at_year_end": "Damaged paddy field",
    "river_discharge_downstream_year_max": "Maximum downstream discharge",
    "landslide_disaster_risk_year_sum": "Landslide risk",
    "financial_damage_by_innundation_year_sum": "Inner-flood damage",
    "financial_damage_by_flood_year_sum": "River-flood damage",
    "municipality_cost_year_sum": "Municipality cost",
    "biodiversity": "Biodiversity",
    "forest_area_storage_capacity": "Forest storage capacity",
}

OVERALL_METRIC_PRIORITY = [
    "yearly_gdp_total",
    "yearly_crop_revenue",
    "yield_per_10a",
    "damaged_paddy_field_at_year_end",
    "river_discharge_downstream_year_max",
    "landslide_disaster_risk_year_sum",
    "financial_damage_by_innundation_year_sum",
    "financial_damage_by_flood_year_sum",
    "municipality_cost_year_sum",
    "biodiversity",
    "forest_area_storage_capacity",
]

YEARLY_METRIC_PRIORITY = [
    "yearly_gdp_total",
    "yearly_crop_revenue",
    "yield_per_10a",
    "damaged_paddy_field_at_year_end",
    "river_discharge_downstream_year_max",
    "financial_damage_by_flood_year_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--basin",
        default=os.environ.get(ACTIVE_BASIN_ENV, "chikugo"),
        help="流域キー、または流域YAMLのパス",
    )
    parser.add_argument("--run-stamp", help="例: 260727_1604。未指定時は最新一式")
    parser.add_argument("--data-dir", type=Path, help="CSVディレクトリを明示")
    parser.add_argument("--output-dir", type=Path, help="PNG出力先を明示")
    parser.add_argument("--target", help="上位パラメータ散布図の目的指標")
    parser.add_argument("--scenario", help="散布図に使うシナリオ")
    parser.add_argument("--top-parameters", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def scenario_order(values) -> list[str]:
    found = [str(value) for value in pd.Series(values).dropna().unique()]
    return [name for name in SCENARIO_PRIORITY if name in found] + sorted(
        set(found) - set(SCENARIO_PRIORITY)
    )


def _paths_for_stamp(data_dir: Path, stamp: str) -> dict[str, Path] | None:
    paths = {
        suffix: data_dir / f"{PREFIX}{stamp}_{suffix}.csv" for suffix in SUFFIXES
    }
    return paths if all(path.exists() for path in paths.values()) else None


def find_csv_set(
    basin_key: str, run_stamp: str | None, explicit_dir: Path | None
) -> tuple[str, dict[str, Path]]:
    directories = (
        [explicit_dir.expanduser().resolve()]
        if explicit_dir
        else [
            BASE_DIR / "results" / "parameter_study" / basin_key,
            BASE_DIR / "data",  # Output location used before basin configs.
        ]
    )
    if run_stamp:
        for data_dir in directories:
            paths = _paths_for_stamp(data_dir, run_stamp)
            if paths:
                return run_stamp, paths
        raise FileNotFoundError(f"実行日時 {run_stamp!r} の完全なCSV一式がありません")

    candidates = []
    pattern = re.compile(rf"^{PREFIX}(.+)_overall_summary\.csv$")
    for directory_index, data_dir in enumerate(directories):
        if not data_dir.exists():
            continue
        for overall_path in data_dir.glob(f"{PREFIX}*_overall_summary.csv"):
            match = pattern.match(overall_path.name)
            if not match:
                continue
            stamp = match.group(1)
            paths = _paths_for_stamp(data_dir, stamp)
            if paths:
                candidates.append(
                    (directory_index, overall_path.stat().st_mtime, stamp, paths)
                )
    if not candidates:
        searched = ", ".join(str(path) for path in directories)
        raise FileNotFoundError(f"完全なパラメータスタディCSV一式がありません: {searched}")

    # Prefer the new basin-specific directory, then the newest complete run.
    best_directory = min(item[0] for item in candidates)
    eligible = [item for item in candidates if item[0] == best_directory]
    _, _, stamp, paths = max(eligible, key=lambda item: item[1])
    return stamp, paths


def metric_label(name: str) -> str:
    return METRIC_LABELS.get(name, name.replace("_", " "))


def available_metrics(frame: pd.DataFrame, priority: list[str], limit: int) -> list[str]:
    return [name for name in priority if name in frame.columns][:limit]


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scenario_distributions(
    overall: pd.DataFrame, metrics: list[str], scenarios: list[str], output_dir: Path, dpi: int
) -> Path | None:
    if not metrics:
        return None
    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.8 * rows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        values = [
            overall.loc[overall["scenario"].astype(str) == scenario, metric].dropna()
            for scenario in scenarios
        ]
        ax.boxplot(values, tick_labels=scenarios, showfliers=False)
        ax.set_title(metric_label(metric))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle("Parameter-study outcome distributions by climate scenario", y=1.01)
    fig.tight_layout()
    return save_figure(fig, output_dir, "01_scenario_outcome_distributions.png", dpi)


def calculate_sensitivity(
    merged: pd.DataFrame,
    parameter_names: list[str],
    metrics: list[str],
    scenarios: list[str],
) -> pd.DataFrame:
    result = pd.DataFrame(index=parameter_names)
    samples = merged[merged["case_id"].astype(str) != "base"]
    for scenario in scenarios:
        part = samples[samples["scenario"].astype(str) == scenario]
        for metric in metrics:
            column = f"{scenario} | {metric}"
            correlations = {}
            for parameter in parameter_names:
                valid = part[[parameter, metric]].dropna()
                correlations[parameter] = (
                    valid[parameter].rank().corr(valid[metric].rank())
                    if len(valid) >= 3
                    and valid[parameter].nunique() > 1
                    and valid[metric].nunique() > 1
                    else np.nan
                )
            result[column] = pd.Series(correlations)
    return result


def plot_sensitivity_heatmap(
    sensitivity: pd.DataFrame, output_dir: Path, dpi: int
) -> Path | None:
    if sensitivity.empty or sensitivity.dropna(how="all").empty:
        return None
    matrix = sensitivity.to_numpy(dtype=float)
    width = max(12, 0.75 * len(sensitivity.columns))
    height = max(7, 0.42 * len(sensitivity.index))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(sensitivity.columns)))
    ax.set_xticklabels(
        [column.replace(" | ", "\n") for column in sensitivity.columns],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(sensitivity.index)))
    ax.set_yticklabels([name.replace("_", " ") for name in sensitivity.index], fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if np.isfinite(value) and abs(value) >= 0.35:
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Spearman rank correlation")
    ax.set_title("Sensitivity of outcomes to sampled assumptions")
    fig.tight_layout()
    return save_figure(fig, output_dir, "02_sensitivity_spearman_heatmap.png", dpi)


def plot_top_parameter_scatter(
    merged: pd.DataFrame,
    sensitivity: pd.DataFrame,
    target: str,
    scenario: str,
    top_n: int,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    sensitivity_column = f"{scenario} | {target}"
    if target not in merged or sensitivity_column not in sensitivity:
        return None
    ranked = sensitivity[sensitivity_column].abs().dropna().nlargest(top_n)
    if ranked.empty:
        return None
    part = merged[
        (merged["scenario"].astype(str) == scenario)
        & (merged["case_id"].astype(str) != "base")
    ]
    cols = 2
    rows = int(np.ceil(len(ranked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    for ax, parameter in zip(axes.flat, ranked.index):
        valid = part[[parameter, target]].dropna()
        ax.scatter(valid[parameter], valid[target], s=18, alpha=0.6)
        if len(valid) >= 3 and valid[parameter].nunique() > 1:
            slope, intercept = np.polyfit(valid[parameter], valid[target], 1)
            xline = np.linspace(valid[parameter].min(), valid[parameter].max(), 100)
            ax.plot(xline, slope * xline + intercept, color="tab:red", linewidth=1.5)
        correlation = sensitivity.loc[parameter, sensitivity_column]
        ax.set_title(f"{parameter.replace('_', ' ')}\nSpearman r = {correlation:.3f}")
        ax.set_xlabel(parameter)
        ax.set_ylabel(metric_label(target))
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(ranked) :]:
        ax.set_visible(False)
    fig.suptitle(f"Top parameter relationships: {scenario} / {metric_label(target)}", y=1.01)
    fig.tight_layout()
    return save_figure(fig, output_dir, "03_top_parameter_scatter.png", dpi)


def plot_yearly_trajectories(
    yearly: pd.DataFrame,
    metrics: list[str],
    scenarios: list[str],
    start_year: int,
    output_dir: Path,
    dpi: int,
) -> Path | None:
    if not metrics or "year" not in yearly:
        return None
    cols = 2
    rows = int(np.ceil(len(metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows), squeeze=False)
    numeric_year = pd.to_numeric(yearly["year"], errors="coerce")
    calendar_year = numeric_year + start_year if numeric_year.max() < start_year else numeric_year
    work = yearly.assign(_calendar_year=calendar_year)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(scenarios), 1)))
    for ax, metric in zip(axes.flat, metrics):
        for color, scenario in zip(colors, scenarios):
            part = work[work["scenario"].astype(str) == scenario]
            grouped = part.groupby("_calendar_year")[metric]
            median = grouped.median()
            low = grouped.quantile(0.1)
            high = grouped.quantile(0.9)
            x = median.index.to_numpy(dtype=float)
            ax.plot(x, median.to_numpy(), label=scenario, color=color)
            ax.fill_between(x, low.to_numpy(), high.to_numpy(), color=color, alpha=0.15)
        ax.set_title(metric_label(metric))
        ax.set_xlabel("Year")
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    for ax in axes.flat[len(metrics) :]:
        ax.set_visible(False)
    axes.flat[0].legend(title="Scenario")
    fig.suptitle("Annual median and 10–90% parameter-study range", y=1.01)
    fig.tight_layout()
    return save_figure(fig, output_dir, "04_yearly_outcome_trajectories.png", dpi)


def plot_scenario_changes(
    overall: pd.DataFrame, metrics: list[str], scenarios: list[str], output_dir: Path, dpi: int
) -> Path | None:
    if "present" not in scenarios or len(scenarios) < 2 or not metrics:
        return None
    medians = overall.groupby(overall["scenario"].astype(str))[metrics].median()
    reference = medians.loc["present"].replace(0, np.nan)
    comparison = medians.drop(index="present").div(reference).sub(1).mul(100)
    if comparison.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(11, len(metrics) * 1.3), 6))
    x = np.arange(len(metrics))
    width = 0.8 / len(comparison)
    for index, (scenario, row) in enumerate(comparison.iterrows()):
        offset = (index - (len(comparison) - 1) / 2) * width
        ax.bar(x + offset, row.to_numpy(), width=width, label=scenario)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_label(metric) for metric in metrics], rotation=35, ha="right")
    ax.set_ylabel("Median change from present (%)")
    ax.set_title("Climate-scenario change across sampled assumptions")
    ax.legend(title="Scenario")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return save_figure(fig, output_dir, "05_scenario_change_from_present.png", dpi)


def plot_parameter_distributions(
    parameter_sets: pd.DataFrame,
    bounds: dict,
    parameter_names: list[str],
    output_dir: Path,
    dpi: int,
) -> Path | None:
    samples = parameter_sets[parameter_sets["case_id"].astype(str) != "base"]
    normalized = []
    labels = []
    bases = []
    for name in parameter_names:
        low, high, base = map(float, bounds[name])
        if high <= low:
            continue
        values = (pd.to_numeric(samples[name], errors="coerce") - low) / (high - low)
        normalized.append(values.dropna().clip(0, 1))
        labels.append(name.replace("_", " "))
        bases.append((base - low) / (high - low))
    if not normalized:
        return None
    fig, ax = plt.subplots(figsize=(11, max(6, 0.45 * len(labels))))
    positions = np.arange(1, len(labels) + 1)
    ax.boxplot(
        normalized,
        orientation="horizontal",
        tick_labels=labels,
        positions=positions,
        showfliers=False,
    )
    ax.scatter(bases, positions, color="tab:red", marker="|", s=180, label="Base value")
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Position within configured [low, high] range")
    ax.set_title("Sampled parameter coverage")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return save_figure(fig, output_dir, "06_parameter_sampling_coverage.png", dpi)


def main() -> None:
    args = parse_args()
    config = load_basin_config(args.basin)
    basin_key = config["key"]
    stamp, paths = find_csv_set(basin_key, args.run_stamp, args.data_dir)
    parameter_sets = pd.read_csv(paths["parameter_sets"])
    overall = pd.read_csv(paths["overall_summary"])
    yearly = pd.read_csv(paths["yearly_summary"])

    for frame in (parameter_sets, overall, yearly):
        if "basin" not in frame:
            frame.insert(0, "basin", basin_key)

    parameter_names = [
        name
        for name in config["sensitivity_bounds"]
        if name in parameter_sets.columns
    ]
    if not parameter_names:
        raise ValueError("parameter_sets CSVにYAMLの感度分析パラメータがありません")
    join_keys = ["basin", "case_id"]
    merged = overall.merge(
        parameter_sets[join_keys + parameter_names],
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    scenarios = scenario_order(overall["scenario"])
    overall_metrics = available_metrics(overall, OVERALL_METRIC_PRIORITY, 6)
    sensitivity_metrics = available_metrics(overall, OVERALL_METRIC_PRIORITY, 6)
    yearly_metrics = available_metrics(yearly, YEARLY_METRIC_PRIORITY, 4)
    target = args.target or (
        "yearly_gdp_total" if "yearly_gdp_total" in overall else overall_metrics[0]
    )
    if target not in overall:
        raise ValueError(f"目的指標がoverall_summaryにありません: {target}")
    scenario = args.scenario or ("present" if "present" in scenarios else scenarios[0])
    if scenario not in scenarios:
        raise ValueError(f"シナリオがCSVにありません: {scenario}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else BASE_DIR / "figures" / "parameter_study" / basin_key / stamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = calculate_sensitivity(
        merged, parameter_names, sensitivity_metrics, scenarios
    )
    sensitivity.to_csv(output_dir / "sensitivity_spearman.csv", index_label="parameter")

    generated = []
    plot_calls = [
        plot_scenario_distributions(overall, overall_metrics, scenarios, output_dir, args.dpi),
        plot_sensitivity_heatmap(sensitivity, output_dir, args.dpi),
        plot_top_parameter_scatter(
            merged,
            sensitivity,
            target,
            scenario,
            max(1, args.top_parameters),
            output_dir,
            args.dpi,
        ),
        plot_yearly_trajectories(
            yearly,
            yearly_metrics,
            scenarios,
            int(config["calendar"]["start_year"]),
            output_dir,
            args.dpi,
        ),
        plot_scenario_changes(overall, overall_metrics, scenarios, output_dir, args.dpi),
        plot_parameter_distributions(
            parameter_sets,
            config["sensitivity_bounds"],
            parameter_names,
            output_dir,
            args.dpi,
        ),
    ]
    generated.extend(path.name for path in plot_calls if path is not None)
    manifest = {
        "basin": basin_key,
        "run_stamp": stamp,
        "source_csv": {name: str(path) for name, path in paths.items()},
        "target_metric": target,
        "scatter_scenario": scenario,
        "figures": generated,
        "sensitivity_table": "sensitivity_spearman.csv",
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(generated)} figures to: {output_dir}")
    for filename in generated:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
