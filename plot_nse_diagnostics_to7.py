"""Create flow-reproduction diagnostics for the present/base to7 simulation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_DATE = "260723_1430"
DATA_DIR = Path("data")
INPUT_CSV = DATA_DIR / f"para_to7_{RUN_DATE}_present_base_nse_flows.csv"
OUTPUT_DIR = Path("figures/nse_to7") / RUN_DATE
WARMUP_DAYS = 90
CALENDAR_START = "2009-01-01"


def paired_data(frame, observed_column, simulated_column):
    paired = frame[["time", "year", observed_column, simulated_column]].copy()
    paired = paired.apply(pd.to_numeric, errors="coerce").dropna()
    return paired[np.isfinite(paired[observed_column]) & np.isfinite(paired[simulated_column])]


def performance_summary(paired, location, observed_column, simulated_column):
    observed = paired[observed_column].to_numpy(dtype=float)
    simulated = paired[simulated_column].to_numpy(dtype=float)
    error = simulated - observed
    denominator = np.sum((observed - observed.mean()) ** 2)

    return {
        "location": location,
        "valid_days": len(paired),
        "nse": 1 - np.sum(error**2) / denominator,
        "rmse": np.sqrt(np.mean(error**2)),
        "pbias_percent": 100 * error.sum() / observed.sum(),
        "pearson_r": np.corrcoef(observed, simulated)[0, 1],
        "observed_mean": observed.mean(),
        "simulated_mean": simulated.mean(),
    }


def annual_diagnostics(paired, location, observed_column, simulated_column):
    rows = []
    for year, year_data in paired.groupby("year", sort=True):
        observed_total = year_data[observed_column].sum()
        simulated_total = year_data[simulated_column].sum()
        observed_peak_index = year_data[observed_column].idxmax()
        simulated_peak_index = year_data[simulated_column].idxmax()
        observed_peak = year_data.loc[observed_peak_index]
        simulated_peak = year_data.loc[simulated_peak_index]

        rows.append(
            {
                "location": location,
                "year": int(year),
                "observed_year_sum": observed_total,
                "simulated_year_sum": simulated_total,
                "annual_pbias_percent": 100 * (simulated_total - observed_total) / observed_total,
                "observed_peak_flow": observed_peak[observed_column],
                "simulated_peak_flow": simulated_peak[simulated_column],
                "peak_flow_error_percent": 100
                * (simulated_peak[simulated_column] - observed_peak[observed_column])
                / observed_peak[observed_column],
                "observed_peak_time": int(observed_peak["time"]),
                "simulated_peak_time": int(simulated_peak["time"]),
                "peak_timing_error_days": int(simulated_peak["time"] - observed_peak["time"]),
            }
        )
    return pd.DataFrame(rows)


def plot_hydrographs(datasets):
    fig, axes = plt.subplots(len(datasets), 1, figsize=(13, 7), sharex=True)
    for ax, location, paired, observed_column, simulated_column in zip(axes, *zip(*datasets)):
        ax.plot(paired["date"], paired[observed_column], color="#333333", linewidth=0.7, label="Observed")
        ax.plot(paired["date"], paired[simulated_column], color="#1f77b4", linewidth=0.7, label="Model")
        ax.set_title(location.capitalize(), loc="left")
        ax.set_ylabel("Flow")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Daily flow reproduction after {WARMUP_DAYS}-day warm-up")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_hydrograph.png", dpi=200)
    plt.close(fig)


def plot_scatter(datasets):
    fig, axes = plt.subplots(1, len(datasets), figsize=(10, 4.5))
    for ax, location, paired, observed_column, simulated_column in zip(axes, *zip(*datasets)):
        observed = paired[observed_column].to_numpy(dtype=float)
        simulated = paired[simulated_column].to_numpy(dtype=float)
        lower = min(observed.min(), simulated.min())
        upper = max(observed.max(), simulated.max())
        ax.scatter(observed, simulated, s=4, alpha=0.18, color="#1f77b4", edgecolors="none")
        ax.plot([lower, upper], [lower, upper], color="#d62728", linestyle="--", label="1:1")
        ax.set_title(location.capitalize())
        ax.set_xlabel("Observed flow")
        ax.set_ylabel("Model flow")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Observed versus model daily flow")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "observed_vs_model_scatter.png", dpi=200)
    plt.close(fig)


def plot_annual_diagnostics(annual):
    locations = annual["location"].unique()
    fig, axes = plt.subplots(len(locations), 2, figsize=(13, 6), sharex="col")
    if len(locations) == 1:
        axes = np.array([axes])

    for row_index, location in enumerate(locations):
        data = annual[annual["location"] == location]
        years = data["year"].to_numpy()
        sum_ax, peak_ax = axes[row_index]
        sum_ax.plot(years, data["observed_year_sum"], marker="o", color="#333333", label="Observed")
        sum_ax.plot(years, data["simulated_year_sum"], marker="o", color="#1f77b4", label="Model")
        sum_ax.set_title(f"{location.capitalize()}: annual flow sum", loc="left")
        sum_ax.grid(alpha=0.25)
        sum_ax.legend()

        peak_ax.plot(years, data["observed_peak_flow"], marker="o", color="#333333", label="Observed")
        peak_ax.plot(years, data["simulated_peak_flow"], marker="o", color="#d62728", label="Model")
        peak_ax.set_title(f"{location.capitalize()}: annual maximum flow", loc="left")
        peak_ax.grid(alpha=0.25)
        peak_ax.legend()

    axes[-1, 0].set_xlabel("Year")
    axes[-1, 1].set_xlabel("Year")
    fig.suptitle("Annual flow diagnostics")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "annual_flow_and_peak.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(len(locations), 1, figsize=(11, 5.5), sharex=True)
    if len(locations) == 1:
        axes = [axes]
    for ax, location in zip(axes, locations):
        data = annual[annual["location"] == location]
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar(data["year"], data["peak_timing_error_days"], color="#9467bd", width=0.7)
        ax.set_title(f"{location.capitalize()}: annual peak timing error", loc="left")
        ax.set_ylabel("Model - observed days")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Year")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "annual_peak_timing_error.png", dpi=200)
    plt.close(fig)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"NSE flow CSV not found: {INPUT_CSV}. Run extract_nse_flow_to7.py first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    flows = pd.read_csv(INPUT_CSV)
    flows["time"] = pd.to_numeric(flows["time"], errors="coerce")
    flows = flows[flows["time"] >= WARMUP_DAYS].copy()
    flows["date"] = pd.to_datetime(CALENDAR_START) + pd.to_timedelta(flows["time"], unit="D")

    specifications = [
        ("upstream", "flow", "river_discharge_upstream"),
        ("downstream", "flow_d", "river_discharge_downstream"),
    ]
    datasets = []
    summaries = []
    annual_tables = []
    for location, observed_column, simulated_column in specifications:
        paired = paired_data(flows, observed_column, simulated_column)
        paired["date"] = pd.to_datetime(CALENDAR_START) + pd.to_timedelta(paired["time"], unit="D")
        datasets.append((location, paired, observed_column, simulated_column))
        summaries.append(performance_summary(paired, location, observed_column, simulated_column))
        annual_tables.append(annual_diagnostics(paired, location, observed_column, simulated_column))

    summary = pd.DataFrame(summaries)
    annual = pd.concat(annual_tables, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "nse_summary.csv", index=False)
    annual.to_csv(OUTPUT_DIR / "annual_flow_diagnostics.csv", index=False)

    plot_hydrographs(datasets)
    plot_scatter(datasets)
    plot_annual_diagnostics(annual)
    print(f"Saved NSE diagnostics to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
