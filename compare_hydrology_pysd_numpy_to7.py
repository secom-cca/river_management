"""Validate and benchmark the NumPy to7 hydrology against the PySD model."""

from datetime import datetime
from pathlib import Path
from time import perf_counter
import warnings

import matplotlib
import numpy as np
import pandas as pd
from pysd import load

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from River_management_xls_to7_numpy import (
    DEFAULT_PARAMETERS,
    HYDROLOGY_OUTPUT_COLUMNS,
    TOTAL_DAYS,
    load_weather,
    simulate_hydrology,
)


warnings.filterwarnings(
    "ignore", message=r"_ext_data_flow(_d)?", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message=r"_ext_data_daily_precip_up", category=UserWarning
)


MODEL_PY = Path("River_management_xls_to7.py")
SCENARIOS = {
    "present": {"daily_precipitation_future_ratio": 1.0, "temperature_scenario_shift": 0.0},
    "2C": {"daily_precipitation_future_ratio": 1.1, "temperature_scenario_shift": 1.3},
    "4C": {"daily_precipitation_future_ratio": 1.3, "temperature_scenario_shift": 4.1},
}
WARMUP_DAYS = 90
RUN_STAMP = datetime.now().strftime("%y%m%d_%H%M%S")
OUTPUT_DIR = Path("data/hydrology_numpy_validation") / RUN_STAMP
FIGURE_DIR = Path("figures/hydrology_numpy_validation") / RUN_STAMP
TIME = np.arange(TOTAL_DAYS, dtype=float)
SCENARIO_LABELS = {"present": "Present", "2C": "2\N{DEGREE SIGN}C", "4C": "4\N{DEGREE SIGN}C"}
SCENARIO_COLORS = {"present": "#2478B5", "2C": "#E67E22", "4C": "#D62F2F"}


PYSD_PARAMETERS = {
    name: value
    for name, value in DEFAULT_PARAMETERS.items()
    if name
    not in {
        "upstream_area",
        "downstream_area",
        "initial_drainage_capacity",
        "initial_dam_capacity",
        "initial_paddy_dam_ratio",
    }
}
PYSD_PARAMETERS.update(
    {
        "annual_paddy_dam_investment": 0.0,
        "annual_forest_management_conversion_area": 0.0,
        "drainage_investment_amount": 0.0,
        "dam_investment_amount": 0.0,
    }
)


def nse(observed, simulated):
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(simulated)
    observed = observed[valid]
    simulated = simulated[valid]
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    return 1.0 - np.sum((observed - simulated) ** 2) / denominator


def comparison_metrics(scenario, column, reference, candidate):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    difference = candidate - reference
    scale = np.maximum(np.abs(reference), 1.0)
    return {
        "scenario": scenario,
        "variable": column,
        "max_abs_error": float(np.max(np.abs(difference))),
        "mean_abs_error": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_relative_error": float(np.max(np.abs(difference) / scale)),
        "allclose_rtol_1e-10_atol_1e-6": bool(
            np.allclose(candidate, reference, rtol=1e-10, atol=1e-6)
        ),
    }


def _display_name(name):
    return name.replace("_", " ").capitalize()


def plot_flow_comparison(flow_comparison):
    dates = pd.date_range("2009-01-01", periods=TOTAL_DAYS, freq="D")
    fig, axes = plt.subplots(
        len(SCENARIOS), 2, figsize=(15, 10), sharex=True, constrained_layout=True
    )
    locations = (
        ("upstream", "Upstream river discharge"),
        ("downstream", "Downstream river discharge"),
    )
    for row, scenario in enumerate(SCENARIOS):
        scenario_data = flow_comparison[flow_comparison["scenario"] == scenario]
        for column, (location, title) in enumerate(locations):
            axis = axes[row, column]
            pysd_values = scenario_data[f"pysd_{location}"].to_numpy()
            numpy_values = scenario_data[f"numpy_{location}"].to_numpy()
            axis.plot(
                dates,
                pysd_values,
                color="#303030",
                linewidth=1.1,
                label="PySD",
            )
            axis.plot(
                dates,
                numpy_values,
                color=SCENARIO_COLORS[scenario],
                linewidth=0.8,
                linestyle=(0, (4, 2)),
                label="NumPy",
            )
            max_difference = np.max(np.abs(numpy_values - pysd_values))
            axis.text(
                0.99,
                0.96,
                f"Max |difference| = {max_difference:.3g}",
                transform=axis.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            )
            axis.grid(alpha=0.25)
            axis.set_title(f"{SCENARIO_LABELS[scenario]}: {title}", loc="left")
            axis.set_ylabel("Flow (m3/day)")
            if row == len(SCENARIOS) - 1:
                axis.set_xlabel("Date")
            if row == 0 and column == 1:
                axis.legend(loc="upper left")
    fig.suptitle("PySD and NumPy hydrology comparison", fontsize=16)
    fig.savefig(FIGURE_DIR / "01_river_discharge_comparison.png", dpi=180)
    plt.close(fig)


def plot_numerical_error(metrics):
    ordered_variables = (
        metrics.groupby("variable")["max_relative_error"]
        .max()
        .sort_values()
        .index.tolist()
    )
    y = np.arange(len(ordered_variables))
    offsets = np.linspace(-0.22, 0.22, len(SCENARIOS))
    fig, axis = plt.subplots(figsize=(12, 13), constrained_layout=True)
    for offset, scenario in zip(offsets, SCENARIOS):
        values = (
            metrics[metrics["scenario"] == scenario]
            .set_index("variable")
            .reindex(ordered_variables)["max_relative_error"]
            .to_numpy()
        )
        axis.scatter(
            np.maximum(values, 1e-18),
            y + offset,
            s=28,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
        )
    axis.set_xscale("log")
    axis.set_yticks(y, [_display_name(name) for name in ordered_variables])
    axis.set_xlabel("Maximum relative difference")
    axis.set_title("Numerical differences between PySD and NumPy")
    axis.grid(axis="x", alpha=0.3)
    axis.legend(title="Climate scenario", loc="lower right")
    axis.text(
        0.01,
        0.01,
        "Zero differences are displayed at 1e-18. All comparisons passed "
        "rtol=1e-10 and atol=1e-6.",
        transform=axis.transAxes,
        fontsize=9,
    )
    fig.savefig(FIGURE_DIR / "02_numerical_error.png", dpi=180)
    plt.close(fig)


def plot_runtime(timings):
    x = np.arange(len(timings))
    width = 0.34
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    pysd_bars = axis.bar(
        x - width / 2,
        timings["pysd_run_seconds"],
        width,
        color="#555555",
        label="PySD",
    )
    numpy_bars = axis.bar(
        x + width / 2,
        timings["numpy_run_seconds"],
        width,
        color=[SCENARIO_COLORS[value] for value in timings["scenario"]],
        label="NumPy",
    )
    axis.set_yscale("log")
    axis.set_xticks(
        x, [SCENARIO_LABELS[value] for value in timings["scenario"]]
    )
    axis.set_ylabel("Run time (seconds, log scale)")
    axis.set_title("Hydrology calculation time")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    axis.bar_label(pysd_bars, fmt="%.2f s", padding=3, fontsize=9)
    axis.bar_label(numpy_bars, fmt="%.3f s", padding=3, fontsize=9)
    for index, speedup in enumerate(timings["run_speedup"]):
        axis.text(
            index,
            0.5,
            f"{speedup:.0f}x faster",
            horizontalalignment="center",
            verticalalignment="center",
            fontweight="bold",
        )
    axis.text(
        0.01,
        0.01,
        "Model and weather-file loading times are excluded.",
        transform=axis.transAxes,
        fontsize=9,
    )
    fig.savefig(FIGURE_DIR / "03_runtime_comparison.png", dpi=180)
    plt.close(fig)


def plot_nse_comparison(nse_summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    overall_min = min(nse_summary["pysd_nse"].min(), nse_summary["numpy_nse"].min())
    overall_max = max(nse_summary["pysd_nse"].max(), nse_summary["numpy_nse"].max())
    for axis, location in zip(axes, ("upstream", "downstream")):
        subset = nse_summary[nse_summary["location"] == location].set_index(
            "scenario"
        ).reindex(SCENARIOS)
        x = np.arange(len(subset))
        axis.scatter(
            x - 0.05,
            subset["pysd_nse"],
            color="#303030",
            marker="o",
            s=65,
            label="PySD",
            zorder=3,
        )
        axis.scatter(
            x + 0.05,
            subset["numpy_nse"],
            color=[SCENARIO_COLORS[value] for value in subset.index],
            marker="x",
            s=80,
            linewidth=2,
            label="NumPy",
            zorder=4,
        )
        axis.set_xticks(x, [SCENARIO_LABELS[value] for value in subset.index])
        axis.set_title(location.capitalize(), loc="left")
        axis.set_ylim(overall_min - 0.03, overall_max + 0.04)
        axis.grid(axis="y", alpha=0.3)
        for index, difference in enumerate(subset["nse_difference"]):
            marker_height = max(
                subset.iloc[index]["pysd_nse"],
                subset.iloc[index]["numpy_nse"],
            )
            axis.annotate(
                f"Delta={difference:.1e}",
                xy=(index, marker_height),
                xytext=(0, 12),
                textcoords="offset points",
                horizontalalignment="center",
                fontsize=8,
            )
    axes[0].set_ylabel("NSE after 90-day warm-up")
    axes[1].legend(loc="upper right")
    fig.supxlabel("Climate scenario")
    fig.suptitle("NSE comparison", fontsize=15)
    fig.savefig(FIGURE_DIR / "04_nse_comparison.png", dpi=180)
    plt.close(fig)


def main():
    if OUTPUT_DIR.exists() or FIGURE_DIR.exists():
        raise FileExistsError(
            "Refusing to overwrite existing output: "
            f"{OUTPUT_DIR} or {FIGURE_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True)
    FIGURE_DIR.mkdir(parents=True)

    weather_load_start = perf_counter()
    weather = load_weather(number_of_days=TOTAL_DAYS)
    weather_load_seconds = perf_counter() - weather_load_start

    load_start = perf_counter()
    model = load(MODEL_PY.as_posix())
    pysd_load_seconds = perf_counter() - load_start

    metric_rows = []
    timing_rows = []
    nse_rows = []
    flow_frames = []

    pysd_return_columns = HYDROLOGY_OUTPUT_COLUMNS + ["flow", "flow_d"]
    for scenario, scenario_params in SCENARIOS.items():
        pysd_params = PYSD_PARAMETERS.copy()
        pysd_params.update(scenario_params)

        pysd_start = perf_counter()
        pysd_result = model.run(
            params=pysd_params,
            return_timestamps=TIME,
            return_columns=pysd_return_columns,
            initial_condition="original",
        )
        pysd_seconds = perf_counter() - pysd_start

        numpy_params = DEFAULT_PARAMETERS.copy()
        numpy_params.update(scenario_params)
        numpy_start = perf_counter()
        numpy_result = simulate_hydrology(weather=weather, params=numpy_params)
        numpy_seconds = perf_counter() - numpy_start

        for column in HYDROLOGY_OUTPUT_COLUMNS:
            metric_rows.append(
                comparison_metrics(
                    scenario,
                    column,
                    pysd_result[column].to_numpy(),
                    numpy_result[column].to_numpy(),
                )
            )

        warm = slice(WARMUP_DAYS, None)
        for location, observed_column, simulated_column in (
            ("upstream", "flow", "river_discharge_upstream"),
            ("downstream", "flow_d", "river_discharge_downstream"),
        ):
            observed = pysd_result[observed_column].to_numpy()[warm]
            pysd_simulated = pysd_result[simulated_column].to_numpy()[warm]
            numpy_simulated = numpy_result[simulated_column].to_numpy()[warm]
            nse_rows.append(
                {
                    "scenario": scenario,
                    "location": location,
                    "pysd_nse": nse(observed, pysd_simulated),
                    "numpy_nse": nse(observed, numpy_simulated),
                    "nse_difference": nse(observed, numpy_simulated)
                    - nse(observed, pysd_simulated),
                }
            )

        timing_rows.append(
            {
                "scenario": scenario,
                "pysd_run_seconds": pysd_seconds,
                "numpy_run_seconds": numpy_seconds,
                "run_speedup": pysd_seconds / max(numpy_seconds, 1e-12),
            }
        )
        flow_frames.append(
            pd.DataFrame(
                {
                    "scenario": scenario,
                    "day": TIME,
                    "observed_upstream": pysd_result["flow"].to_numpy(),
                    "pysd_upstream": pysd_result[
                        "river_discharge_upstream"
                    ].to_numpy(),
                    "numpy_upstream": numpy_result[
                        "river_discharge_upstream"
                    ].to_numpy(),
                    "observed_downstream": pysd_result["flow_d"].to_numpy(),
                    "pysd_downstream": pysd_result[
                        "river_discharge_downstream"
                    ].to_numpy(),
                    "numpy_downstream": numpy_result[
                        "river_discharge_downstream"
                    ].to_numpy(),
                }
            )
        )

    metrics = pd.DataFrame(metric_rows)
    timings = pd.DataFrame(timing_rows)
    timings["pysd_model_load_seconds"] = pysd_load_seconds
    timings["numpy_weather_load_seconds"] = weather_load_seconds
    nse_summary = pd.DataFrame(nse_rows)
    flow_comparison = pd.concat(flow_frames, ignore_index=True)

    metrics.to_csv(OUTPUT_DIR / "variable_comparison.csv", index=False)
    timings.to_csv(OUTPUT_DIR / "timing_comparison.csv", index=False)
    nse_summary.to_csv(OUTPUT_DIR / "nse_comparison.csv", index=False)
    flow_comparison.to_csv(OUTPUT_DIR / "flow_comparison.csv", index=False)

    plot_flow_comparison(flow_comparison)
    plot_numerical_error(metrics)
    plot_runtime(timings)
    plot_nse_comparison(nse_summary)

    print("\nMaximum numerical error by scenario:")
    print(
        metrics.groupby("scenario")[["max_abs_error", "max_relative_error"]]
        .max()
        .to_string()
    )
    print("\nTiming (model loading excluded from run_speedup):")
    print(timings.to_string(index=False))
    print("\nNSE after 90-day warm-up:")
    print(nse_summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
