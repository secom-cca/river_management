"""Explore narrative policy types from archived decisions and spending, without model runs.

Primary types use a transparent, provisional spending-share threshold.
Forest scale and breeding are separate policy-content tags; objective outcomes
are deliberately not used to assign types or select illustrative policies.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIOS = ["present", "2C", "4C"]
TYPES = ["river_drainage", "housing", "agriculture", "ecosystem", "mixed"]
TYPE_JA = dict(zip(TYPES, ["河川・排水施設重視", "住宅対策重視", "農業関連対策重視",
                           "森林整備重視", "複合型"]))
TYPE_EN = dict(zip(TYPES, ["River / drainage", "Housing", "Agriculture-related", "Forest", "Mixed"]))
COLORS = ["#0072b2", "#d55e00", "#e69f00", "#009e73", "#888888"]
SECTOR_KEYS = TYPES[:-1]
COMPONENTS = ["drainage", "dam", "levee", "elevation", "migration", "paddy_dam", "breeding", "forest"]
COMPONENT_LABELS = ["Drainage", "Dam", "Levee", "House elevation", "Migration", "Paddy dam", "Breeding", "Forest"]
COMPONENT_COLORS = ["#56b4e9", "#0072b2", "#332288", "#f4a582", "#d55e00", "#e69f00", "#cc79a7", "#009e73"]
SCHEDULE_PREFIXES = dict(zip(COMPONENTS, ["drainage", "dam", "levee", "house_elevation",
                                         "migration", "paddy_dam", "breeding", "forest_management"]))
SECTOR_COMPONENTS = {"river_drainage": ["drainage", "dam", "levee"],
                     "housing": ["elevation", "migration"],
                     "agriculture": ["paddy_dam", "breeding"], "ecosystem": ["forest"]}
PERIODS = ["early", "middle", "late"]
PERIOD_JA = {"early": "初期", "middle": "中期", "late": "後期"}
PERIOD_COLORS = ["#56b4e9", "#aaaaaa", "#d55e00"]
SECTOR_ACTION_JA = {"river_drainage": "河川・排水施設", "housing": "住宅対策",
                     "agriculture": "農業関連対策", "ecosystem": "森林整備"}


def period(year_index: float) -> str:
    return PERIODS[min(2, int(year_index // 5))]


def save(fig: plt.Figure, output: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_policies(run: Path) -> pd.DataFrame:
    policies = []
    for scenario in SCENARIOS:
        frame = pd.read_csv(run / f"pareto_summary_{scenario}.csv")
        yearly = pd.read_csv(run / f"pareto_yearly_{scenario}.csv")
        if frame["solution_id"].duplicated().any():
            raise ValueError("Duplicate policy IDs.")
        years = yearly.groupby("solution_id")["year"].agg(["size", "nunique"])
        if set(years.index) != set(frame["solution_id"]) or not (years.eq(15).all().all()):
            raise ValueError("Expected exactly 15 distinct archived annual rows per policy.")
        totals = yearly.groupby("solution_id")[["paddy_dam_actual_cost", "breeding_actual_cost",
                                                "additional_forest_management_cost"]].sum()
        frame = frame.join(totals, on="solution_id")
        for name in ("drainage", "dam", "levee"):
            frame[f"cost_{name}"] = frame[f"param_{name}_investment_amount"]
        for name, output_name in (("house_elevation", "elevation"), ("migration", "migration")):
            frame[f"cost_{output_name}"] = (frame[f"param_{name}_per_year"] *
                                               frame[f"param_{name}_duration_years"] * 3_000_000)
        frame["cost_paddy_dam"] = frame["paddy_dam_actual_cost"]
        frame["cost_breeding"] = frame["breeding_actual_cost"]
        frame["cost_forest"] = frame["additional_forest_management_cost"]
        costs = frame[[f"cost_{k}" for k in COMPONENTS]].to_numpy(float)
        if not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("Invalid archived policy costs.")
        total = costs.sum(axis=1)
        if (total <= 0).any() or not np.allclose(total, frame["policy_additional_cost"], rtol=1e-10, atol=1e-4):
            raise ValueError("Reconstructed costs do not match saved policy cost.")
        for key in COMPONENTS:
            frame[f"share_{key}"] = frame[f"cost_{key}"] / total
        frame["share_river_drainage"] = frame[["share_drainage", "share_dam", "share_levee"]].sum(axis=1)
        frame["share_housing"] = frame[["share_elevation", "share_migration"]].sum(axis=1)
        frame["share_agriculture"] = frame[["share_paddy_dam", "share_breeding"]].sum(axis=1)
        frame["share_ecosystem"] = frame["share_forest"]
        frame["forest_planned_area_ha"] = (frame["param_annual_forest_management_conversion_area"] *
                                             frame["param_forest_management_duration_years"])
        frame["forest_planned_upper_bound_fraction"] = frame["forest_planned_area_ha"] / 45_000
        frame["forest_large_scale_tag"] = frame["forest_planned_upper_bound_fraction"] >= 0.5
        frame["breeding_tag"] = frame["param_breeding_enabled"].astype(str).str.lower().eq("true")
        frame["breeding_start_calendar_year"] = np.where(
            frame["breeding_tag"], frame["param_breeding_start_year"] + 2009, np.nan)
        frame["forest_start_calendar_year"] = frame["param_forest_management_start_year"] + 2009
        phase_costs = yearly.assign(period=yearly["year_index"].astype(int).floordiv(5)).pivot_table(
            index="solution_id", columns="period", values="policy_additional_cost", aggfunc="sum")
        for i, name in enumerate(PERIODS):
            frame[f"spending_share_{name}"] = frame["solution_id"].map(phase_costs[i]) / total
        weighted_year = yearly.assign(weighted=yearly["year_index"] * yearly["policy_additional_cost"])
        frame["spending_weighted_mean_calendar_year"] = (
            frame["solution_id"].map(weighted_year.groupby("solution_id")["weighted"].sum()) / total + 2009)
        policies.append(frame)
    return pd.concat(policies, ignore_index=True)


def classify(frame: pd.DataFrame, threshold: float) -> pd.Series:
    shares = frame[[f"share_{k}" for k in SECTOR_KEYS]]
    strongest = shares.idxmax(axis=1).str.removeprefix("share_")
    return strongest.where(shares.max(axis=1) >= threshold, "mixed")


def counts(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    work = frame.assign(primary_type=classify(frame, threshold))
    return pd.crosstab(work["scenario"], work["primary_type"]).reindex(
        index=SCENARIOS, columns=TYPES, fill_value=0)


def add_timing(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = frame.copy()
    actions = []
    for row in frame.to_dict("records"):
        for component in COMPONENTS:
            cost = float(row[f"cost_{component}"])
            if cost <= 0:
                continue
            prefix = SCHEDULE_PREFIXES[component]
            start = int(row[f"param_{prefix}_start_year"])
            duration = (1 if component in ("drainage", "dam", "levee") else
                        10 if component == "breeding" else int(row[f"param_{prefix}_duration_years"]))
            if not 0 <= start <= start + duration - 1 < 15:
                raise ValueError("Policy schedule lies outside the evaluation horizon.")
            actions.append({"solution_id": row["solution_id"], "scenario": row["scenario"],
                            "component": component, "start_year_index": start,
                            "start_calendar_year": start + 2009, "duration_years": duration,
                            "end_calendar_year": start + duration - 1 + 2009,
                            "start_period": period(start), "cost_yen": cost,
                            "share_of_total_cost": row[f"share_{component}"]})
    timeline = pd.DataFrame(actions)
    indexed = timeline.set_index(["solution_id", "component"])
    narratives = []
    for index, row in output.iterrows():
        for sector, components in SECTOR_COMPONENTS.items():
            selected = timeline.loc[timeline["solution_id"].eq(row["solution_id"]) &
                                    timeline["component"].isin(components)]
            mean_start = np.average(selected["start_year_index"], weights=selected["cost_yen"])
            output.loc[index, f"{sector}_weighted_start_calendar_year"] = mean_start + 2009
            output.loc[index, f"{sector}_start_period"] = period(mean_start)
        main_sector = max(SECTOR_KEYS, key=lambda k: row[f"share_{k}"])
        main_period = output.loc[index, f"{main_sector}_start_period"]
        output.loc[index, "main_sector"] = main_sector
        output.loc[index, "main_sector_start_period"] = main_period
        phrases = [row["primary_narrative_ja"],
                   f"主力の{SECTOR_ACTION_JA[main_sector]}を{PERIOD_JA[main_period]}開始"]
        # The largest two individual cost components convey concrete schedules.
        main_actions = timeline.loc[timeline["solution_id"].eq(row["solution_id"])].nlargest(2, "cost_yen")
        action_names = dict(zip(COMPONENTS, ["排水施設", "ダム", "堤防", "かさ上げ", "移転",
                                            "田んぼダム", "育種", "森林整備"]))
        for action in main_actions.itertuples():
            phrases.append(f"{action_names[action.component]}は{action.start_calendar_year}年開始・"
                           f"{action.duration_years}年間")
        river_period = output.loc[index, "river_drainage_start_period"]
        if main_sector != "river_drainage":
            phrases.append(f"河川・排水施設は{PERIOD_JA[river_period]}開始中心")
        if "forest" not in set(main_actions["component"]):
            forest = indexed.loc[(row["solution_id"], "forest")]
            phrases.append(f"森林整備は{int(forest.start_calendar_year)}年開始・{int(forest.duration_years)}年間")
        if row["breeding_tag"]:
            phrases.append(f"育種は{int(row['breeding_start_calendar_year'])}年開始")
        else:
            phrases.append("育種なし")
        narratives.append("、".join(phrases))
    output["timing_narrative_ja"] = narratives
    return output, timeline


def start_period_summary(timeline: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scenario in SCENARIOS:
        for component in COMPONENTS:
            selected = timeline.loc[timeline["scenario"].eq(scenario) & timeline["component"].eq(component)]
            record = {"scenario": scenario, "component": component, "active_policy_count": len(selected)}
            for name in PERIODS:
                record[f"{name}_start_count"] = int(selected["start_period"].eq(name).sum())
            records.append(record)
    return pd.DataFrame(records)


def start_period_plot(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)
    for ax, scenario in zip(axes, SCENARIOS):
        rows = summary.loc[summary["scenario"].eq(scenario)].set_index("component").loc[COMPONENTS]
        base = np.zeros(8)
        for name, color in zip(PERIODS, PERIOD_COLORS):
            values = rows[f"{name}_start_count"].to_numpy() / rows["active_policy_count"].to_numpy() * 100
            ax.barh(range(8), values, left=base, color=color)
            for j, value in enumerate(values):
                if value >= 10:
                    ax.text(base[j] + value / 2, j, f"{value:.0f}%", ha="center", va="center", fontsize=9)
            base += values
        labels = [f"{label} (n={n})" for label, n in zip(COMPONENT_LABELS, rows["active_policy_count"])]
        ax.set_yticks(range(8), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(scenario)
        ax.set_xlabel("Policies by scheduled start period (%)")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in PERIOD_COLORS],
               ["Early: 2009–2013", "Middle: 2014–2018", "Late: 2019–2023"],
               loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Scheduled starts of active interventions — each policy counts once")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    save(fig, output, "03_intervention_start_periods")


def schedule_plot(examples: pd.DataFrame, timeline: pd.DataFrame, output: Path) -> None:
    columns = 2
    rows = math.ceil(len(examples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(14, rows * 3.8), squeeze=False)
    for ax, (_, policy) in zip(axes.flat, examples.iterrows()):
        actions = timeline.loc[timeline["solution_id"].eq(policy["solution_id"])]
        for action in actions.itertuples():
            position = COMPONENTS.index(action.component)
            ax.barh(position, action.duration_years, left=action.start_calendar_year,
                    color=COMPONENT_COLORS[position], height=0.65)
        ax.set_yticks(range(8), COMPONENT_LABELS, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(2008.7, 2024.3)
        ax.set_xticks([2009, 2014, 2019, 2024])
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(f"{policy['scenario']} | t{int(policy['trial_number']):05d} | "
                     f"{TYPE_EN[policy['primary_type']]} / {policy['main_sector_start_period']} start", fontsize=10)
    for ax in axes.flat[len(examples):]:
        ax.axis("off")
    fig.suptitle("Illustrative policy schedules: implementation windows from archived decisions")
    fig.text(0.5, 0.01, "Forest bars show conversion windows; forest costs can continue afterwards.\n"
             "Fixed schedules do not represent disaster-triggered responses. Breeding effects are not shown.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.065, 1, 0.97))
    save(fig, output, "04_illustrative_schedules")


def count_plot(frame: pd.DataFrame, output: Path, threshold: float) -> None:
    summary = counts(frame, threshold)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    base = np.zeros(3)
    for kind, color in zip(TYPES, COLORS):
        v = summary[kind].to_numpy()
        axes[0].bar(SCENARIOS, v, bottom=base, color=color, label=TYPE_EN[kind])
        for i, n in enumerate(v):
            if n:
                axes[0].text(i, base[i] + n / 2, str(n), ha="center", va="center", fontsize=10)
        base += v
    axes[0].set_title(f"Primary type: at least {threshold:.0%} of policy spending")
    axes[0].set_ylabel("Archived Pareto policies")
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS],
               [TYPE_EN[k] for k in TYPES], loc="lower center", ncol=5, frameon=False, fontsize=9)
    mean = frame.groupby("scenario")[[f"share_{k}" for k in SECTOR_KEYS]].mean().reindex(SCENARIOS)
    base = np.zeros(3)
    for key, color in zip(SECTOR_KEYS, COLORS):
        v = mean[f"share_{key}"].to_numpy() * 100
        axes[1].bar(SCENARIOS, v, bottom=base, color=color)
        for i, n in enumerate(v):
            if n >= 5:
                axes[1].text(i, base[i] + n / 2, f"{n:.1f}%", ha="center", va="center", fontsize=10)
        base += v
    axes[1].set_title("Mean spending shares (each policy has equal weight)")
    axes[1].set_ylabel("Share of total policy cost (%)")
    fig.suptitle("Exploratory narratives: policy content, not objective performance")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, output, "01_narrative_overview")


def portfolio_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 14), sharex=True)
    for ax, scenario in zip(axes, SCENARIOS):
        group = frame.loc[frame["scenario"].eq(scenario)].copy()
        group["type_order"] = group["primary_type"].map(dict(zip(TYPES, range(5))))
        group = group.sort_values(["type_order", "share_housing", "trial_number"])
        positions = np.arange(len(group))
        base = np.zeros(len(group))
        for key, color in zip(COMPONENTS, COMPONENT_COLORS):
            v = group[f"share_{key}"].to_numpy() * 100
            ax.barh(positions, v, left=base, height=0.85, color=color)
            base += v
        ax.set_yticks(positions, [f"t{int(n):05d}" for n in group["trial_number"]], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of total policy cost (%)")
        ax.set_title(f"{scenario}: {len(group)} archived policies")
        ax.grid(axis="x", alpha=0.15)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in COMPONENT_COLORS], COMPONENT_LABELS,
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Individual policy portfolios — sorted by primary type, then housing share")
    fig.tight_layout(rect=(0, 0.055, 1, 0.97))
    save(fig, output, "02_individual_portfolios")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/optuna_to7_260915_102704"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dominant-share", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.5 <= args.dominant_share <= 1:
        raise ValueError("Use a dominant share between 0.5 and 1.")
    frame = load_policies(args.run_dir)
    frame["primary_type"] = classify(frame, args.dominant_share)
    frame["primary_narrative_ja"] = frame["primary_type"].map(TYPE_JA)
    frame["narrative_ja"] = [
        f"{r.primary_narrative_ja}、森林整備{r.forest_start_calendar_year:.0f}年開始・"
        f"{r.param_forest_management_duration_years:.0f}年間"
        + ("（計画面積が探索上限の50%以上）" if r.forest_large_scale_tag else "")
        + (f"、育種{r.breeding_start_calendar_year:.0f}年開始を併用" if r.breeding_tag else "、育種なし")
        for r in frame.itertuples()]
    frame, timeline = add_timing(frame)
    timing_summary = start_period_summary(timeline)
    output = args.output_dir or Path("analysis") / args.run_dir.name / f"narratives_{datetime.now():%y%m%d_%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "policy_narratives.csv", index=False, encoding="utf-8-sig")
    timeline.to_csv(output / "policy_action_timeline.csv", index=False, encoding="utf-8-sig")
    timing_summary.to_csv(output / "intervention_start_periods.csv", index=False, encoding="utf-8-sig")
    counts(frame, args.dominant_share).to_csv(output / "narrative_counts.csv", encoding="utf-8-sig")
    sensitivity = pd.concat([counts(frame, t).reset_index().assign(dominant_share=t)
                             for t in sorted(set([0.5, 0.6, 0.7, args.dominant_share]))], ignore_index=True)
    sensitivity.to_csv(output / "classification_thresholds.csv", index=False, encoding="utf-8-sig")
    examples = []
    features = [f"share_{k}" for k in COMPONENTS] + ["forest_planned_upper_bound_fraction", "breeding_tag"]
    for _key, group in frame.groupby(["scenario", "primary_type"], sort=False):
        values = group[features].astype(float).copy()
        for key in SECTOR_KEYS:
            values[f"{key}_onset"] = (group[f"{key}_weighted_start_calendar_year"] - 2009) / 14
        distance = (values - values.median()).abs().sum(axis=1)
        examples.append(group.loc[distance.idxmin()])
    examples_frame = pd.DataFrame(examples)
    examples_frame.to_csv(output / "illustrative_policies.csv", index=False, encoding="utf-8-sig")
    count_plot(frame, output, args.dominant_share)
    portfolio_plot(frame, output)
    start_period_plot(timing_summary, output)
    schedule_plot(examples_frame, timeline, output)
    manifest = {"run_dir": str(args.run_dir.resolve()), "policy_count": len(frame),
                "dominant_share": args.dominant_share,
                "cost_grouping": {"river_drainage": ["drainage", "dam", "levee"],
                                  "housing": ["elevation", "migration"],
                                  "agriculture": ["paddy_dam", "breeding"], "ecosystem": ["forest"]},
                "forest_tag": "planned area >= 22500 ha (50% of 3000 ha/year x 15 years search upper bound)",
                "timing_periods": {"early": [2009, 2013], "middle": [2014, 2018], "late": [2019, 2023]},
                "sector_onset": "component start years weighted by component total cost; does not measure expenditure timing",
                "spending_timing": "annual saved policy costs, including recurring forest costs",
                "illustrative_selection": "nearest to component-wise median of spending shares, planned forest intensity, breeding flag and sector start times; no outcomes used",
                "notes": ["Exploratory analyst-defined labels, not inferred decision-maker intentions.",
                          "Agriculture-related spending includes paddy dams, which also provide flood mitigation.",
                          "Forest management is the modeled ecosystem intervention; this is not a complete ecological strategy.",
                          "Breeding tags describe investment decisions, not success in the archived threshold model.",
                          "Planned forest area differs from actual converted area and ecological recovery.",
                          "Start times are fixed in advance, not triggered by observed disasters.",
                          "Start times and spending periods are separate; early starts can entail long-duration spending.",
                          "Sector-weighted starts can conceal different component starts; use individual schedules as well.",
                          "No model evaluations or robustness rankings were performed."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {output.resolve()}")
    print(counts(frame, args.dominant_share).to_string())


if __name__ == "__main__":
    main()
