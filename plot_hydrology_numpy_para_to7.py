"""Plot uncertainty and sensitivity from the NumPy hydrology LHS output."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hydrology_numpy_to7_common import HYDROLOGY_ASSUMPTION_BOUNDS, SCENARIOS


RUN_STAMP = None
DATA_ROOT = Path("data/hydrology_numpy_para_to7")
FIGURE_ROOT = Path("figures/hydrology_numpy_para_to7")
SCENARIO_COLORS = {"present": "#1f77b4", "2C": "#ff7f0e", "4C": "#d62728"}
SCENARIO_LABELS = {
    "present": "Present",
    "2C": "2\u2103",
    "4C": "4\u2103",
}


def select_run():
    if RUN_STAMP:
        folder = DATA_ROOT / RUN_STAMP
    else:
        candidates = [
            path.parent for path in DATA_ROOT.glob("*/overall_summary.csv")
            if (path.parent / "yearly_summary.csv").exists()
        ]
        if not candidates:
            raise FileNotFoundError("No complete NumPy hydrology LHS result was found.")
        folder = max(candidates, key=lambda path: path.stat().st_mtime)
    return folder


def plot_yearly_bands(yearly, output_dir):
    metrics = [
        ("river_discharge_upstream_year_max", "Upstream annual maximum flow"),
        ("river_discharge_downstream_year_max", "Downstream annual maximum flow"),
        ("innundation_level_year_max", "Annual maximum inundation level"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 9), sharex=True)
    for axis, (metric, title) in zip(axes, metrics):
        for scenario in SCENARIOS:
            subset = yearly[yearly["scenario"] == scenario]
            lhs = subset[subset["case_id"] != "base"]
            grouped = lhs.groupby("calendar_year")[metric]
            quantiles = grouped.quantile([0.05, 0.5, 0.95]).unstack()
            base = subset[subset["case_id"] == "base"].set_index("calendar_year")
            color = SCENARIO_COLORS[scenario]
            axis.fill_between(
                quantiles.index,
                quantiles[0.05],
                quantiles[0.95],
                color=color,
                alpha=0.14,
            )
            axis.plot(
                quantiles.index,
                quantiles[0.5],
                color=color,
                label=f"{SCENARIO_LABELS[scenario]} LHS median",
            )
            axis.plot(
                base.index,
                base[metric],
                color=color,
                linestyle="--",
                linewidth=1.0,
            )
        axis.set_title(title, loc="left")
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Calendar year")
    fig.suptitle("NumPy hydrology uncertainty: 5th-95th percentile")
    fig.tight_layout()
    fig.savefig(output_dir / "01_yearly_uncertainty_bands.png", dpi=180)
    plt.close(fig)


def plot_nse_ranges(overall, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for axis, metric, title in (
        (axes[0], "nse_upstream", "Upstream NSE"),
        (axes[1], "nse_downstream", "Downstream NSE"),
    ):
        values = [
            overall[
                (overall["scenario"] == scenario)
                & (overall["case_id"] != "base")
            ][metric]
            for scenario in SCENARIOS
        ]
        boxes = axis.boxplot(values, patch_artist=True, tick_labels=[
            SCENARIO_LABELS[name] for name in SCENARIOS
        ])
        for patch, scenario in zip(boxes["boxes"], SCENARIOS):
            patch.set_facecolor(SCENARIO_COLORS[scenario])
            patch.set_alpha(0.45)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("NSE range across hydrology assumptions")
    fig.tight_layout()
    fig.savefig(output_dir / "02_nse_ranges.png", dpi=180)
    plt.close(fig)


def plot_correlations(overall, output_dir):
    parameters = list(HYDROLOGY_ASSUMPTION_BOUNDS)
    outcomes = [
        "nse_upstream",
        "nse_downstream",
        "river_discharge_upstream_max",
        "river_discharge_downstream_max",
        "inside_water_innundation_days",
        "innundation_level_max",
    ]
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(17, 6), constrained_layout=True)
    image = None
    for axis, scenario in zip(axes, SCENARIOS):
        subset = overall[
            (overall["scenario"] == scenario) & (overall["case_id"] != "base")
        ]
        matrix = subset[parameters + outcomes].corr().loc[parameters, outcomes]
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        axis.set_xticks(np.arange(len(outcomes)), [name.replace("_", " ") for name in outcomes], rotation=55, ha="right")
        axis.set_yticks(np.arange(len(parameters)), [name.replace("_", " ") for name in parameters])
        axis.set_title(SCENARIO_LABELS[scenario])
    fig.colorbar(image, ax=axes, label="Pearson correlation", shrink=0.75)
    fig.suptitle("Hydrology parameter-outcome correlation")
    fig.savefig(output_dir / "03_parameter_outcome_correlations.png", dpi=180)
    plt.close(fig)


def main():
    data_dir = select_run()
    output_dir = FIGURE_ROOT / data_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    yearly = pd.read_csv(data_dir / "yearly_summary.csv")
    overall = pd.read_csv(data_dir / "overall_summary.csv")
    plot_yearly_bands(yearly, output_dir)
    plot_nse_ranges(overall, output_dir)
    plot_correlations(overall, output_dir)
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
