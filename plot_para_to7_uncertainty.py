from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set a timestamp such as "260723_1200" to select a specific run.
# Leave as None to use the latest complete parameter-study output set.
RUN_DATE = None
DATA_DIR = Path("data")
FIG_DIR = Path("figures/para_to7")


def _input_paths():
    if RUN_DATE is not None:
        run_date = RUN_DATE
        overall = DATA_DIR / f"para_to7_all_scenarios_{run_date}_overall_summary.csv"
    else:
        candidates = list(DATA_DIR.glob("para_to7_all_scenarios_*_overall_summary.csv"))
        if not candidates:
            raise FileNotFoundError("No parameter-study overall summary CSV was found in data/.")
        overall = max(candidates, key=lambda path: path.stat().st_mtime)
        prefix = "para_to7_all_scenarios_"
        suffix = "_overall_summary.csv"
        run_date = overall.name[len(prefix) : -len(suffix)]

    yearly = DATA_DIR / f"para_to7_all_scenarios_{run_date}_yearly_summary.csv"
    parameters = DATA_DIR / f"para_to7_all_scenarios_{run_date}_parameter_sets.csv"
    missing = [path for path in (yearly, overall, parameters) if not path.exists()]
    if missing:
        missing_names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Incomplete parameter-study output set: {missing_names}")
    return run_date, yearly, overall, parameters

SCENARIO_ORDER = ["present", "2C", "4C"]
SCENARIO_COLORS = {
    "present": "#1f77b4",
    "2C": "#ff7f0e",
    "4C": "#d62728",
}

SCENARIO_LABELS = {
    "present": "Present",
    "2C": "2℃",
    "4C": "4℃",
}

YEARLY_METRICS = [
    "daily_precip_up_year_sum",
    "daily_precip_down_year_sum",
    "daily_precip_up_year_mean",
    "daily_precip_down_year_mean",
    "daily_precip_up_year_max",
    "daily_precip_down_year_max",
    "river_discharge_upstream_year_sum",
    "river_discharge_downstream_year_sum",
    "river_discharge_upstream_year_mean",
    "river_discharge_downstream_year_mean",
    "river_discharge_upstream_year_max",
    "river_discharge_downstream_year_max",
    "daily_ave_temp_up_year_mean",
    "daily_ave_temp_down_year_mean",
    "daily_ave_temp_up_year_max",
    "daily_ave_temp_down_year_max",
    "daily_ave_temp_up_year_min",
    "daily_ave_temp_down_year_min",
    "solar_radiation_up_year_sum",
    "solar_radiation_down_year_sum",
    "solar_radiation_up_year_mean",
    "solar_radiation_down_year_mean",
    "daily_crop_production_at_harvest",
    "accumulated_crop_production_within_year_at_harvest",
    "yearly_crop_production_at_harvest",
    "yearly_crop_production_per_day_at_harvest",
    "chalky_kernel_ratio_at_harvest",
    "effective_chalky_kernel_ratio_at_harvest",
    "crop_price_quality_factor_at_harvest",
    "quality_adjusted_crop_price_at_harvest",
    "heat_stress_at_heading_plus_20_at_harvest",
    "accumulated_heat_stress_at_harvest",
    "yearly_gdp_total",
    "yearly_crop_revenue",
    "yearly_total_crop_yield_kg",
    "yield_per_10a",
    "financial_damage_by_innundation_year_sum",
    "financial_damage_by_flood_year_sum",
    "paddy_field_at_harvest",
    "biodiversity",
    "forest_area_storage_capacity",
    "landslide_disaster_risk_year_sum",
    "landslide_disaster_risk_year_max",
    "scenario_adjusted_daily_precip_up_year_max",
    "forest_mitigation_year_mean",
]

OVERALL_METRICS = [
    "daily_precip_up_year_sum",
    "daily_precip_down_year_sum",
    "daily_precip_up_year_mean",
    "daily_precip_down_year_mean",
    "river_discharge_upstream_year_sum",
    "river_discharge_downstream_year_sum",
    "river_discharge_upstream_year_mean",
    "river_discharge_downstream_year_mean",
    "river_discharge_upstream_year_max",
    "river_discharge_downstream_year_max",
    "daily_ave_temp_up_year_mean",
    "daily_ave_temp_down_year_mean",
    "solar_radiation_up_year_sum",
    "solar_radiation_down_year_sum",
    "daily_crop_production_at_harvest",
    "accumulated_crop_production_within_year_at_harvest",
    "yearly_crop_production_at_harvest",
    "yearly_crop_production_per_day_at_harvest",
    "chalky_kernel_ratio_at_harvest",
    "effective_chalky_kernel_ratio_at_harvest",
    "crop_price_quality_factor_at_harvest",
    "quality_adjusted_crop_price_at_harvest",
    "heat_stress_at_heading_plus_20_at_harvest",
    "accumulated_heat_stress_at_harvest",
    "yearly_gdp_total",
    "yearly_crop_revenue",
    "yearly_total_crop_yield_kg",
    "financial_damage_by_innundation_year_sum",
    "financial_damage_by_flood_year_sum",
    "biodiversity",
    "forest_area_storage_capacity",
    "landslide_disaster_risk_year_sum",
    "landslide_disaster_risk_year_max",
]

PARAMETER_COLUMNS = [
    "forest_area_ratio",
    "paddy_field_ratio",
    "ratio_of_paddy_field_in_risky_area",
    "innundation_risky_area_ratio",
    "flood_risky_area_ratio",
    "recovery_ratio",
    "crop_price",
    "waterholding_capacity_of_forest_base",
    "innundation_damage_per_resident",
    "flood_damage_per_resident",
    "gdp_per_resident",
    "paddy_field_capacity_per_area",
    "paddy_dam_capacity_per_area",
    "unmanaged_plantation_forest_coef",
]


def _ensure_output_dir(run_date):
    run_fig_dir = FIG_DIR / run_date
    run_fig_dir.mkdir(parents=True, exist_ok=True)
    return run_fig_dir


def _clean_metric_name(metric):
    text = metric.replace("_", " ").lower()
    return text[:1].upper() + text[1:] if text else text


def _lhs_only(df):
    return df[df["level"].isin(["lhs", "random"])].copy()


def _base_only(df):
    return df[df["case_id"] == "base"].copy()


def plot_yearly_uncertainty_bands(yearly, output_dir):
    lhs = _lhs_only(yearly)
    base = _base_only(yearly)

    for metric in YEARLY_METRICS:
        if metric not in yearly.columns:
            continue

        fig, axes = plt.subplots(
            nrows=len(SCENARIO_ORDER),
            ncols=1,
            figsize=(10, 8),
            sharex=True,
            sharey=True,
        )
        if len(SCENARIO_ORDER) == 1:
            axes = [axes]

        for ax, scenario in zip(axes, SCENARIO_ORDER):
            scen_lhs = lhs[lhs["scenario"] == scenario]
            if scen_lhs.empty:
                ax.set_visible(False)
                continue

            quantiles = (
                scen_lhs.groupby("year")[metric]
                .quantile([0.05, 0.5, 0.95])
                .unstack()
                .reset_index()
            )
            years = quantiles["year"].to_numpy()
            lower = quantiles[0.05].to_numpy(dtype=float)
            median = quantiles[0.5].to_numpy(dtype=float)
            upper = quantiles[0.95].to_numpy(dtype=float)
            color = SCENARIO_COLORS.get(scenario)

            ax.fill_between(years, lower, upper, color=color, alpha=0.18, linewidth=0)
            ax.plot(years, median, color=color, linewidth=2)

            scen_base = base[base["scenario"] == scenario]
            if not scen_base.empty:
                ax.plot(
                    scen_base["year"].to_numpy(),
                    scen_base[metric].to_numpy(dtype=float),
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                )

            ax.set_title(SCENARIO_LABELS.get(scenario, scenario), loc="left", fontsize=11)
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("Year index")
        fig.supylabel(_clean_metric_name(metric))
        fig.suptitle(f"Yearly uncertainty band: {_clean_metric_name(metric)}", y=0.985)
        legend_handles = [
            Patch(facecolor="gray", alpha=0.18, label="LHS 5-95% range"),
            Line2D([0], [0], color="black", linewidth=2, label="LHS median"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.4, label="Base case"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            fontsize=8,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(output_dir / f"yearly_band_{metric}.png", dpi=200)
        plt.close(fig)


def plot_overall_range_bars(overall, output_dir):
    lhs = _lhs_only(overall)
    base = _base_only(overall)

    for metric in OVERALL_METRICS:
        if metric not in overall.columns:
            continue

        fig, ax = plt.subplots(figsize=(8, 4.5))
        y_positions = np.arange(len(SCENARIO_ORDER))

        for row_index, scenario in enumerate(SCENARIO_ORDER):
            scen_lhs = lhs[lhs["scenario"] == scenario]
            if scen_lhs.empty:
                continue

            values = scen_lhs[metric].dropna().to_numpy(dtype=float)
            q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
            color = SCENARIO_COLORS.get(scenario)

            ax.hlines(row_index, q05, q95, color=color, linewidth=8, alpha=0.35)
            ax.plot(q50, row_index, marker="o", color=color, markersize=6, label=None)

            scen_base = base[base["scenario"] == scenario]
            if not scen_base.empty:
                ax.plot(
                    scen_base[metric].iloc[0],
                    row_index,
                    marker="D",
                    color=color,
                    markeredgecolor="black",
                    markersize=5,
                )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([SCENARIO_LABELS.get(label, label) for label in SCENARIO_ORDER])
        ax.set_title(f"Overall uncertainty range: {_clean_metric_name(metric)}")
        ax.set_xlabel(_clean_metric_name(metric))
        ax.grid(True, axis="x", alpha=0.25)
        legend_handles = [
            Patch(facecolor="gray", alpha=0.35, label="LHS 5-95% range"),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                color="black",
                label="LHS median",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="None",
                color="black",
                markerfacecolor="white",
                label="Base case",
            ),
        ]
        ax.legend(handles=legend_handles, loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"overall_range_{metric}.png", dpi=200)
        plt.close(fig)


def plot_correlation_heatmaps(overall, parameter_sets, output_dir):
    lhs_overall = _lhs_only(overall)
    lhs_params = _lhs_only(parameter_sets)
    merged = lhs_overall.merge(lhs_params, on=["case_id", "varied_parameter", "level"], how="left")

    metric_cols = [metric for metric in OVERALL_METRICS if metric in merged.columns]
    param_cols = [param for param in PARAMETER_COLUMNS if param in merged.columns]

    for scenario in SCENARIO_ORDER:
        scen = merged[merged["scenario"] == scenario]
        if scen.empty:
            continue

        corr = pd.DataFrame(index=param_cols, columns=metric_cols, dtype=float)
        for param in param_cols:
            for metric in metric_cols:
                corr.loc[param, metric] = scen[param].corr(scen[metric])

        fig_width = max(8, len(metric_cols) * 1.1)
        fig_height = max(7, len(param_cols) * 0.35)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")

        ax.set_xticks(np.arange(len(metric_cols)))
        ax.set_xticklabels([_clean_metric_name(col) for col in metric_cols], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(param_cols)))
        ax.set_yticklabels([_clean_metric_name(col) for col in param_cols])
        ax.set_title(f"Parameter-outcome correlation: {SCENARIO_LABELS.get(scenario, scenario)}")

        for i in range(len(param_cols)):
            for j in range(len(metric_cols)):
                value = corr.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)

        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Pearson correlation")
        fig.tight_layout()
        fig.savefig(output_dir / f"correlation_heatmap_{scenario}.png", dpi=200)
        plt.close(fig)


def main():
    run_date, yearly_csv, overall_csv, parameter_csv = _input_paths()
    output_dir = _ensure_output_dir(run_date)
    yearly = pd.read_csv(yearly_csv)
    overall = pd.read_csv(overall_csv)
    parameter_sets = pd.read_csv(parameter_csv)

    plot_yearly_uncertainty_bands(yearly, output_dir)
    plot_overall_range_bars(overall, output_dir)
    plot_correlation_heatmaps(overall, parameter_sets, output_dir)

    print(f"Read parameter study: {run_date}")
    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
