"""Validate breeding funding and compare crop responses without optimisation."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import run_vensim_with_pysd_to7_opt as core
from run_vensim_with_pysd_to7_optuna import _zero_decision
from breeding_response_to7 import (
    MIDPOINT_YEN, SCALE_YEN, MAX_REDUCTION, TOLERANCE_YEN,
    logistic_effect_ratio, model_configuration, model_parameters,
)


def _case(task):
    scenario, mode, start_year, response_params = task
    decision = _zero_decision()
    if start_year is not None:
        start_index = start_year - core.CALENDAR_START_YEAR
        decision.update(
            breeding_enabled=1, breeding_start_year=start_index,
            breeding_end_year=start_index + core.BREEDING_DURATION_YEARS - 1,
        )
    results = core.run_model_yearly(scenario, decision, model_params=response_params)
    metrics, yearly = core.aggregate_results(results, decision)
    label = "no_breeding" if start_year is None else str(start_year)
    metadata = {"scenario": scenario, "mode": mode, "start_year": label}
    daily_frames = []
    maximum_error = 0.0
    for year_index, result in enumerate(results):
        calendar_year = core.CALENDAR_START_YEAR + year_index
        active_years = 0 if start_year is None else max(0, min(10, calendar_year - start_year + 1))
        expected = active_years * core.BREEDING_ANNUAL_COST
        observed = result.attrs["year_end_state"]["accumulated_breeding_investment"]
        error = abs(observed - expected)
        maximum_error = max(maximum_error, error)
        if error > TOLERANCE_YEN:
            raise AssertionError(f"{task}, {calendar_year}: investment error {error} yen")
        active = start_year is not None and start_year <= calendar_year < start_year + 10
        annual_expected = core.BREEDING_ANNUAL_COST if active else 0.0
        if abs(result["breeding_investment"].sum() - annual_expected) > TOLERANCE_YEN:
            raise AssertionError(f"{task}: annual cashflow does not match schedule")
        if len(result) != core.YEAR_LENGTHS[year_index]:
            raise AssertionError(f"{task}: duplicate or missing calendar day")
        boundary = core.YEAR_STARTS[year_index] + core.YEAR_LENGTHS[year_index]
        if result.attrs["year_end_time"] != boundary:
            raise AssertionError(f"{task}: integration stopped before year end")
        success = result.attrs["year_end_state"]["breeding_success"]
        if success != int(active_years == 10):
            raise AssertionError(f"{task}: incorrect target attainment at {calendar_year}")
        end_effect = result.attrs["year_end_state"]["breeding_effect_ratio"]
        expected_effect = success if mode == "threshold" else logistic_effect_ratio(
            observed, response_params["breeding_logistic_midpoint"], response_params["breeding_logistic_scale"]
        )
        if not np.isclose(end_effect, expected_effect, rtol=0, atol=1e-12):
            raise AssertionError(f"{task}: incorrect effect progress")
        expected_quality = result["chalky_kernel_ratio"] * (
            1 - response_params["breeding_max_quality_reduction"] * result["breeding_effect_ratio"]
        )
        if not np.allclose(result["effective_chalky_kernel_ratio"], expected_quality, rtol=0, atol=1e-10):
            raise AssertionError(f"{task}: quality response is inconsistent")
        harvest = result[result["harvest_trigger"] > 0.5]
        if len(harvest) != 1:
            raise AssertionError(f"{task}: expected one harvest per year")
        for name in ("chalky_kernel_ratio", "effective_chalky_kernel_ratio", "crop_price_quality_factor"):
            yearly.loc[year_index, name + "_at_harvest"] = float(harvest[name].iloc[0])
        daily = result[[
            "breeding_investment", "accumulated_breeding_investment", "breeding_success", "breeding_effect_ratio",
            "chalky_kernel_ratio", "effective_chalky_kernel_ratio", "crop_price_quality_factor",
            "harvest_trigger", "crop_production_cashflow",
        ]].copy()
        daily.insert(0, "date", pd.to_datetime("2009-01-01") + pd.to_timedelta(result.index, unit="D"))
        daily.insert(1, "year", calendar_year)
        daily_frames.append(daily)
    daily = pd.concat(daily_frames, ignore_index=True)
    for name, value in metadata.items():
        yearly.insert(0, name, value)
        daily.insert(0, name, value)
    summary = {
        **metadata, **metrics,
        "final_investment_yen": float(yearly["accumulated_breeding_investment"].iloc[-1]),
        "maximum_investment_error_yen": maximum_error,
        "target_attained": bool(yearly["breeding_success"].iloc[-1]),
        "final_effect_ratio": float(yearly["breeding_effect_ratio"].iloc[-1]),
        "first_target_year_end": (
            int(yearly.loc[yearly["breeding_success"] > 0.5, "year"].iloc[0])
            if yearly["breeding_success"].max() > 0.5 else None
        ),
        "first_effect_harvest_year": (
            int(yearly.loc[yearly["breeding_effect_ratio_at_harvest"] > 1e-12, "year"].iloc[0])
            if yearly["breeding_effect_ratio_at_harvest"].max() > 1e-12 else None
        ),
        "passed": True,
    }
    return summary, yearly, daily


def _save_figure(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")


def _plot(yearly, summary, args, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"threshold": ("#377eb8", "--"), "logistic": ("#e66101", "-")}
    # Funding rows overlap by design; quality and revenue use harvest-time values.
    columns = [
        ("accumulated_breeding_investment", 1e-8, "Investment at year end (100M yen)"),
        ("breeding_effect_ratio_at_harvest", 100, "Effect progress at harvest (%)"),
        ("effective_chalky_kernel_ratio_at_harvest", 1, "Chalky kernels at harvest (%)"),
        ("crop_revenue", 1e-9, "Annual crop revenue (billion yen)"),
    ]
    for start in sorted({min(args.start_years), max(args.start_years)}):
        fig, axes = plt.subplots(4, len(args.scenarios), figsize=(5 * len(args.scenarios), 11), squeeze=False, sharex=True)
        for column_index, scenario in enumerate(args.scenarios):
            baseline = yearly[(yearly.scenario == scenario) & (yearly.start_year == "no_breeding")].drop_duplicates("year").sort_values("year")
            for row_index, (name, factor, label) in enumerate(columns):
                ax = axes[row_index, column_index]
                ax.plot(baseline.year, baseline[name] * factor, color="0.5", linestyle=":", label="No breeding")
                for mode in args.modes:
                    data = yearly[(yearly.scenario == scenario) & (yearly.start_year == str(start)) & (yearly["mode"] == mode)].sort_values("year")
                    color, line = styles[mode]
                    ax.plot(data.year, data[name] * factor, color=color, linestyle=line, label=mode.capitalize())
                ax.set_ylabel(label)
                ax.grid(alpha=0.2)
                if row_index == 0:
                    ax.set_title(scenario)
                if row_index == 3:
                    ax.set_xlabel("Calendar year")
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(f"Breeding only; funding begins {start}\nFunding: 50M yen/year for 10 years; investment curves overlap", fontsize=13)
        fig.tight_layout()
        _save_figure(fig, output, f"breeding_comparison_{start}")
        plt.close(fig)

    fig, axes = plt.subplots(1, len(args.scenarios), figsize=(5 * len(args.scenarios), 4), squeeze=False)
    for i, scenario in enumerate(args.scenarios):
        ax = axes[0, i]
        for mode in args.modes:
            data = summary[(summary.scenario == scenario) & (summary["mode"] == mode) & (summary.start_year != "no_breeding")].copy()
            data["start_year_number"] = data.start_year.astype(int)
            data = data.sort_values("start_year_number")
            color, line = styles[mode]
            ax.plot(data.start_year_number, data.crop_revenue_change_pct, color=color, linestyle=line, marker="o", label=mode.capitalize())
        ax.set(title=scenario, xlabel="Funding start year", ylabel="15-year crop revenue gain (%)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    _save_figure(fig, output, "start_year_sensitivity")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    investment = np.linspace(0, 600e6, 1001)
    for mode in args.modes:
        progress = (
            (investment >= 500e6 - TOLERANCE_YEN).astype(float) if mode == "threshold"
            else [logistic_effect_ratio(value, args.midpoint_yen, args.scale_yen) for value in investment]
        )
        color, line = styles[mode]
        ax.plot(investment / 1e8, np.asarray(progress) * 100, color=color, linestyle=line, label=mode.capitalize())
    ax.axvline(5, color="0.5", linestyle=":", label="10-year investment target")
    ax.set(xlabel="Cumulative investment (100M yen)", ylabel="Effect progress (%)", ylim=(-2, 102))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output, "investment_effect_curve")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+", choices=["present", "2C", "4C"], default=["present", "2C", "4C"])
    parser.add_argument("--modes", nargs="+", choices=["threshold", "logistic"], default=["threshold"])
    parser.add_argument("--start-years", nargs="+", type=int, choices=range(2009, 2015), default=list(range(2009, 2015)))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--midpoint-yen", type=float, default=MIDPOINT_YEN)
    parser.add_argument("--scale-yen", type=float, default=SCALE_YEN)
    parser.add_argument("--max-reduction", type=float, default=MAX_REDUCTION)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    try:
        responses = {mode: model_parameters(mode, midpoint=args.midpoint_yen, scale=args.scale_yen, max_reduction=args.max_reduction) for mode in args.modes}
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir or Path("analysis") / datetime.now().strftime("breeding_to7_%y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    # Record the configuration at launch, before parallel calculations begin.
    configuration = model_configuration()
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ["River_management_xls_to7.py", "run_vensim_with_pysd_to7_opt.py", "breeding_response_to7.py", Path(__file__).name]}
    tasks = [(scenario, mode, start, responses[mode]) for scenario in args.scenarios for mode in args.modes for start in [None, *args.start_years]]
    summaries, years, days = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(_case, task): task for task in tasks}
        for future in as_completed(futures):
            summary, yearly, daily = future.result()
            summaries.append(summary)
            years.append(yearly)
            days.append(daily)
            print(f"{len(summaries)}/{len(tasks)}: {futures[future][:3]} passed", flush=True)
    summary = pd.DataFrame(summaries).sort_values(["scenario", "mode", "start_year"])
    yearly = pd.concat(years, ignore_index=True).sort_values(["scenario", "mode", "start_year", "year"])
    daily = pd.concat(days, ignore_index=True).sort_values(["scenario", "mode", "start_year", "date"])
    for scenario in args.scenarios:
        baseline_rows = summary[(summary.scenario == scenario) & (summary.start_year == "no_breeding")]
        if not np.allclose(baseline_rows.crop_revenue, baseline_rows.crop_revenue.iloc[0], rtol=0, atol=0.01):
            raise AssertionError("No-breeding baseline differs between response modes")
        for mode in args.modes:
            baseline = summary[(summary.scenario == scenario) & (summary["mode"] == mode) & (summary.start_year == "no_breeding")].crop_revenue.iloc[0]
            selected = (summary.scenario == scenario) & (summary["mode"] == mode)
            summary.loc[selected, "crop_revenue_change_yen"] = summary.loc[selected, "crop_revenue"] - baseline
            summary.loc[selected, "crop_revenue_change_pct"] = 100 * (summary.loc[selected, "crop_revenue"] / baseline - 1)
    for name, frame in [("case_summary", summary), ("yearly", yearly), ("daily", daily)]:
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scenarios": args.scenarios, "modes": args.modes, "start_years": args.start_years,
        "annual_investment_yen": core.BREEDING_ANNUAL_COST, "duration_years": 10,
        "year_end_convention": "after December 31 integration; next January 1 00:00",
        "sources_sha256": source_hashes,
        "all_cases_passed": bool(summary["passed"].all()),
        "model_configuration": configuration,
        "response_parameters": responses,
        "interpretation": "Progress is an assumed deterministic response, not success probability. Crop quality price bins are unchanged. Only breeding differs; all other additional policies are zero.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(yearly, summary, args, output)
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
