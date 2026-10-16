"""Paper-oriented robustness analysis for all cross-evaluated Pareto policies.

Outputs cover five questions: all-policy cross-climate performance, differences
by design climate, final robust candidates, policy-design mechanisms, and rank
sensitivity to normalization, objective weights, and climate probabilities.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from evaluate_optuna_to7_cross_scenario import latest_run
from run_vensim_with_pysd_to7_optuna import CONTINUOUS_BOUNDS


SCENARIO_ORDER = ("present", "2C", "4C")
SCENARIO_LABELS = {"present": "Present", "2C": "2℃", "4C": "4℃"}
SCENARIO_COLORS = {"present": "#4C78A8", "2C": "#F58518", "4C": "#54A24B"}
OBJECTIVES = (
    "flood_damage_ratio",
    "innundation_damage_ratio",
    "landslide_risk_ratio",
    "crop_revenue_ratio",
    "biodiversity_ratio",
)
OBJECTIVE_LABELS = ("Flood", "Inundation", "Landslide", "Crop", "Biodiversity")
MINIMIZE = np.array([True, True, True, False, False])
REGRET_COLUMNS = tuple(f"regret_{name}" for name in OBJECTIVES)

NORMALIZATIONS = ("pareto_range", "baseline_to_ideal", "cross_policy_range")
OBJECTIVE_WEIGHT_SCHEMES: dict[str, np.ndarray] = {
    "equal": np.ones(5),
    "flood_x2": np.array([2, 1, 1, 1, 1], dtype=float),
    "inundation_x2": np.array([1, 2, 1, 1, 1], dtype=float),
    "landslide_x2": np.array([1, 1, 2, 1, 1], dtype=float),
    "crop_x2": np.array([1, 1, 1, 2, 1], dtype=float),
    "biodiversity_x2": np.array([1, 1, 1, 1, 2], dtype=float),
}
CLIMATE_PROBABILITY_SCHEMES: dict[str, np.ndarray] = {
    "equal": np.array([1, 1, 1], dtype=float),
    "present_heavy": np.array([0.60, 0.25, 0.15]),
    "2C_heavy": np.array([0.20, 0.60, 0.20]),
    "4C_heavy": np.array([0.15, 0.25, 0.60]),
}

MECHANISM_FEATURES = (
    "drainage_investment_amount",
    "dam_investment_amount",
    "levee_investment_amount",
    "house_elevation_per_year",
    "migration_per_year",
    "annual_paddy_dam_investment",
    "annual_forest_management_conversion_area",
    "drainage_start_year",
    "dam_start_year",
    "levee_start_year",
    "house_elevation_start_year",
    "house_elevation_duration_years",
    "migration_start_year",
    "migration_duration_years",
    "paddy_dam_start_year",
    "paddy_dam_duration_years",
    "forest_management_start_year",
    "forest_management_duration_years",
    "breeding_enabled",
    "breeding_start_year",
)
FEATURE_LABELS = {
    "drainage_investment_amount": "Drainage amount",
    "dam_investment_amount": "Dam amount",
    "levee_investment_amount": "Levee amount",
    "house_elevation_per_year": "House elevation/year",
    "migration_per_year": "Migration/year",
    "annual_paddy_dam_investment": "Paddy dam/year",
    "annual_forest_management_conversion_area": "Forest management/year",
    "drainage_start_year": "Drainage start",
    "dam_start_year": "Dam start",
    "levee_start_year": "Levee start",
    "house_elevation_start_year": "House elevation start",
    "house_elevation_duration_years": "House elevation duration",
    "migration_start_year": "Migration start",
    "migration_duration_years": "Migration duration",
    "paddy_dam_start_year": "Paddy dam start",
    "paddy_dam_duration_years": "Paddy dam duration",
    "forest_management_start_year": "Forest management start",
    "forest_management_duration_years": "Forest management duration",
    "breeding_enabled": "Breeding enabled",
    "breeding_start_year": "Breeding start",
}
FEATURE_RANGES = {
    **{name: upper - lower for name, (lower, upper) in CONTINUOUS_BOUNDS.items()},
    "drainage_start_year": 14,
    "dam_start_year": 14,
    "levee_start_year": 14,
    "house_elevation_start_year": 14,
    "house_elevation_duration_years": 15,
    "migration_start_year": 14,
    "migration_duration_years": 15,
    "paddy_dam_start_year": 14,
    "paddy_dam_duration_years": 15,
    "forest_management_start_year": 14,
    "forest_management_duration_years": 15,
    "breeding_enabled": 1,
    "breeding_start_year": 5,
}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _losses(values: np.ndarray) -> np.ndarray:
    return np.where(MINIMIZE, values, -values)


def add_regrets(
    results: pd.DataFrame,
    policies: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Add target-specific normalized objective regrets to long results."""
    if method not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization: {method}")
    output = results.copy()
    for target in SCENARIO_ORDER:
        mask = output["target_scenario"] == target
        target_values = output.loc[mask, OBJECTIVES].to_numpy(dtype=float)
        target_losses = _losses(target_values)
        source_pareto = policies[policies["scenario"] == target]
        pareto_losses = _losses(source_pareto.loc[:, OBJECTIVES].to_numpy(dtype=float))
        ideal = pareto_losses.min(axis=0)
        if method == "pareto_range":
            reference_nadir = pareto_losses.max(axis=0)
        elif method == "baseline_to_ideal":
            reference_nadir = _losses(np.ones((1, len(OBJECTIVES))))[0]
        else:
            ideal = target_losses.min(axis=0)
            reference_nadir = target_losses.max(axis=0)
        span = reference_nadir - ideal
        span[span <= 1e-15] = 1.0
        regret = np.maximum((target_losses - ideal) / span, 0.0)
        for index, column in enumerate(REGRET_COLUMNS):
            output.loc[mask, column] = regret[:, index]
    return output


def score_policies(
    regrets: pd.DataFrame,
    objective_weights: Sequence[float],
    climate_probabilities: Sequence[float],
) -> pd.DataFrame:
    """Calculate weighted mean and cell-wise worst regret for each policy."""
    objective_weights = np.asarray(objective_weights, dtype=float)
    climate_probabilities = np.asarray(climate_probabilities, dtype=float)
    if objective_weights.shape != (len(OBJECTIVES),) or np.any(objective_weights < 0):
        raise ValueError("objective_weights must contain five non-negative values.")
    if climate_probabilities.shape != (len(SCENARIO_ORDER),) or np.any(
        climate_probabilities < 0
    ):
        raise ValueError("climate_probabilities must contain three non-negative values.")
    if objective_weights.sum() <= 0 or climate_probabilities.sum() <= 0:
        raise ValueError("Weights and probabilities must have positive sums.")
    objective_mean_weights = objective_weights / objective_weights.sum()
    objective_worst_weights = objective_weights / objective_weights.mean()
    climate_probabilities = climate_probabilities / climate_probabilities.sum()
    climate_map = dict(zip(SCENARIO_ORDER, climate_probabilities))

    rows = []
    for solution_id, group in regrets.groupby("source_solution_id", sort=False):
        group = group.set_index("target_scenario").loc[list(SCENARIO_ORDER)]
        matrix = group.loc[:, REGRET_COLUMNS].to_numpy(dtype=float)
        climate_means = matrix @ objective_mean_weights
        weighted_mean = float(
            sum(climate_map[name] * climate_means[index] for index, name in enumerate(SCENARIO_ORDER))
        )
        active_climates = climate_probabilities > 0
        weighted_worst = float(
            (matrix[active_climates] * objective_worst_weights).max()
        )
        feasible = group["feasible"].astype(str).str.lower().eq("true")
        rows.append(
            {
                "source_solution_id": solution_id,
                "source_scenario": group.iloc[0]["source_scenario"],
                "evaluated_scenarios": len(group),
                "feasible_scenarios": int(feasible.sum()),
                "feasibility_rate": float(feasible.mean()),
                "mean_regret": weighted_mean,
                "worst_regret": weighted_worst,
                "mean_total_budget_utilisation": float(
                    group["total_budget_utilisation"].mean()
                ),
                "max_annual_budget_utilisation": float(
                    group["annual_budget_utilisation"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity(
    results: pd.DataFrame, policies: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    regret_frames = {
        method: add_regrets(results, policies, method) for method in NORMALIZATIONS
    }
    frames = []
    for normalization, regrets in regret_frames.items():
        for objective_scheme, objective_weights in OBJECTIVE_WEIGHT_SCHEMES.items():
            for climate_scheme, probabilities in CLIMATE_PROBABILITY_SCHEMES.items():
                scores = score_policies(regrets, objective_weights, probabilities)
                scores["normalization"] = normalization
                scores["objective_weight_scheme"] = objective_scheme
                scores["climate_probability_scheme"] = climate_scheme
                scores["variant_id"] = (
                    normalization + "__" + objective_scheme + "__" + climate_scheme
                )
                scores["mean_rank"] = scores["mean_regret"].rank(
                    method="min", ascending=True
                )
                scores["worst_rank"] = scores["worst_regret"].rank(
                    method="min", ascending=True
                )
                frames.append(scores)
    sensitivity = pd.concat(frames, ignore_index=True)
    return sensitivity, regret_frames


def add_physical_summary(base: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    output = base.copy()
    aggregate_rows = []
    for solution_id, group in results.groupby("source_solution_id", sort=False):
        row: dict[str, Any] = {"source_solution_id": solution_id}
        for objective, minimize in zip(OBJECTIVES, MINIMIZE):
            row[f"mean_{objective}"] = float(group[objective].mean())
            row[f"worst_{objective}"] = float(
                group[objective].max() if minimize else group[objective].min()
            )
        aggregate_rows.append(row)
    return output.merge(pd.DataFrame(aggregate_rows), on="source_solution_id", validate="one_to_one")


def mark_robust_top_decile(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    count = len(output)
    output["mean_rank_pct"] = output["mean_regret"].rank(method="min") / count
    output["worst_rank_pct"] = output["worst_regret"].rank(method="min") / count
    output["robust_rank_score"] = output[["mean_rank_pct", "worst_rank_pct"]].max(axis=1)
    top_count = max(1, math.ceil(count * 0.10))
    top_ids = set(output.nsmallest(top_count, "robust_rank_score")["source_solution_id"])
    output["robust_top_decile"] = output["source_solution_id"].isin(top_ids)
    return output


def select_candidates(
    summary: pd.DataFrame, robust_quantile: float
) -> pd.DataFrame:
    feasible = summary[summary["feasibility_rate"] >= 1.0 - 1e-12].copy()
    if feasible.empty:
        raise ValueError("No policy is feasible in every target scenario.")
    role_rows = [
        ("mean_best", feasible.loc[feasible["mean_regret"].idxmin()]),
        ("worst_best", feasible.loc[feasible["worst_regret"].idxmin()]),
    ]
    mean_limit = float(feasible["mean_regret"].quantile(robust_quantile))
    worst_limit = float(feasible["worst_regret"].quantile(robust_quantile))
    robust_pool = feasible[
        (feasible["mean_regret"] <= mean_limit)
        & (feasible["worst_regret"] <= worst_limit)
    ]
    if robust_pool.empty:
        robust_pool = feasible
    already_selected = {row["source_solution_id"] for _, row in role_rows}
    unique_pool = robust_pool[~robust_pool["source_solution_id"].isin(already_selected)]
    if unique_pool.empty:
        unique_pool = robust_pool
    budget_row = unique_pool.loc[unique_pool["mean_total_budget_utilisation"].idxmin()]
    role_rows.append(("budget_robust", budget_row))

    roles_by_id: dict[str, list[str]] = {}
    rows_by_id: dict[str, pd.Series] = {}
    for role, row in role_rows:
        solution_id = str(row["source_solution_id"])
        roles_by_id.setdefault(solution_id, []).append(role)
        rows_by_id[solution_id] = row
    candidates = []
    for order, (solution_id, roles) in enumerate(roles_by_id.items()):
        row = rows_by_id[solution_id].to_dict()
        row["candidate_roles"] = ";".join(roles)
        row["candidate_label"] = "/".join(
            {
                "mean_best": "Mean best",
                "worst_best": "Worst-case best",
                "budget_robust": "Budget robust",
            }[role]
            for role in roles
        )
        row["candidate_order"] = order
        row["robust_pool_mean_limit"] = mean_limit
        row["robust_pool_worst_limit"] = worst_limit
        candidates.append(row)
    return pd.DataFrame(candidates)


def candidate_performance_table(
    candidates: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    identifiers = set(candidates["source_solution_id"])
    selected = results[results["source_solution_id"].isin(identifiers)].copy()
    selected = selected.merge(
        candidates[
            ["source_solution_id", "candidate_roles", "candidate_label", "candidate_order"]
        ],
        on="source_solution_id",
        validate="many_to_one",
    )
    for objective, minimize in zip(OBJECTIVES, MINIMIZE):
        label = objective.removesuffix("_ratio").replace("innundation", "inundation")
        selected[f"{label}_improvement_pct"] = (
            (1.0 - selected[objective]) * 100
            if minimize
            else (selected[objective] - 1.0) * 100
        )
    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    selected["target_order"] = selected["target_scenario"].map(order)
    return selected.sort_values(["candidate_order", "target_order"]).drop(
        columns="target_order"
    ).reset_index(drop=True)


def design_climate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    top_mean_limit = summary["mean_regret"].quantile(0.10)
    top_worst_limit = summary["worst_regret"].quantile(0.10)
    for scenario in SCENARIO_ORDER:
        group = summary[summary["source_scenario"] == scenario]
        rows.append(
            {
                "source_scenario": scenario,
                "policy_count": len(group),
                "all_climate_feasible_count": int((group["feasibility_rate"] == 1).sum()),
                "all_climate_feasible_rate": float((group["feasibility_rate"] == 1).mean()),
                "mean_regret_mean": float(group["mean_regret"].mean()),
                "mean_regret_median": float(group["mean_regret"].median()),
                "mean_regret_q25": float(group["mean_regret"].quantile(0.25)),
                "mean_regret_q75": float(group["mean_regret"].quantile(0.75)),
                "worst_regret_mean": float(group["worst_regret"].mean()),
                "worst_regret_median": float(group["worst_regret"].median()),
                "worst_regret_q25": float(group["worst_regret"].quantile(0.25)),
                "worst_regret_q75": float(group["worst_regret"].quantile(0.75)),
                "top_10pct_mean_count": int((group["mean_regret"] <= top_mean_limit).sum()),
                "top_10pct_worst_count": int((group["worst_regret"] <= top_worst_limit).sum()),
                "robust_top_decile_count": int(group["robust_top_decile"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(values, np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if not len(valid):
        return adjusted
    ordered = valid[np.argsort(values[valid])]
    running = 0.0
    count = len(ordered)
    for rank, index in enumerate(ordered):
        candidate = min(1.0, values[index] * (count - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def design_climate_tests(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pair_rows = []
    for metric in ("mean_regret", "worst_regret"):
        groups = [
            summary.loc[summary["source_scenario"] == scenario, metric].to_numpy()
            for scenario in SCENARIO_ORDER
        ]
        statistic, pvalue = stats.kruskal(*groups)
        rows.append(
            {
                "metric": metric,
                "test": "Kruskal-Wallis",
                "group_a": "all",
                "group_b": "all",
                "statistic": statistic,
                "p_value": pvalue,
                "p_holm": pvalue,
            }
        )
        for left_index in range(len(SCENARIO_ORDER)):
            for right_index in range(left_index + 1, len(SCENARIO_ORDER)):
                left = SCENARIO_ORDER[left_index]
                right = SCENARIO_ORDER[right_index]
                statistic, pvalue = stats.mannwhitneyu(
                    groups[left_index], groups[right_index], alternative="two-sided"
                )
                pair_rows.append(
                    {
                        "metric": metric,
                        "test": "Mann-Whitney U",
                        "group_a": left,
                        "group_b": right,
                        "statistic": statistic,
                        "p_value": pvalue,
                    }
                )
    if pair_rows:
        adjusted = _holm_adjust([row["p_value"] for row in pair_rows])
        for row, p_holm in zip(pair_rows, adjusted):
            row["p_holm"] = p_holm
        rows.extend(pair_rows)
    return pd.DataFrame(rows)


def mechanism_analysis(
    decisions: pd.DataFrame, summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = decisions.merge(
        summary[
            [
                "source_solution_id",
                "mean_regret",
                "worst_regret",
                "robust_top_decile",
            ]
        ],
        left_on="solution_id",
        right_on="source_solution_id",
        validate="one_to_one",
    )
    correlations = []
    comparisons = []
    for feature in MECHANISM_FEATURES:
        values = pd.to_numeric(merged[feature], errors="coerce")
        for outcome in ("mean_regret", "worst_regret"):
            valid = values.notna() & merged[outcome].notna()
            if valid.sum() >= 3 and values[valid].nunique() > 1:
                result = stats.spearmanr(values[valid], merged.loc[valid, outcome])
                rho, pvalue = float(result.statistic), float(result.pvalue)
            else:
                rho, pvalue = np.nan, np.nan
            correlations.append(
                {
                    "feature": feature,
                    "feature_label": FEATURE_LABELS[feature],
                    "outcome": outcome,
                    "n": int(valid.sum()),
                    "spearman_rho": rho,
                    "p_value": pvalue,
                }
            )
        robust = values[merged["robust_top_decile"] & values.notna()]
        other = values[~merged["robust_top_decile"] & values.notna()]
        if len(robust) and len(other):
            statistic, pvalue = stats.mannwhitneyu(
                robust, other, alternative="two-sided"
            )
        else:
            statistic, pvalue = np.nan, np.nan
        difference = float(robust.median() - other.median())
        comparisons.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "robust_n": len(robust),
                "other_n": len(other),
                "robust_median": float(robust.median()),
                "other_median": float(other.median()),
                "median_difference": difference,
                "normalized_median_difference": difference / FEATURE_RANGES[feature],
                "mann_whitney_u": statistic,
                "p_value": pvalue,
            }
        )
    correlations_frame = pd.DataFrame(correlations)
    correlations_frame["p_holm"] = _holm_adjust(correlations_frame["p_value"])
    comparisons_frame = pd.DataFrame(comparisons)
    comparisons_frame["p_holm"] = _holm_adjust(comparisons_frame["p_value"])
    return correlations_frame, comparisons_frame


def sensitivity_summaries(
    sensitivity: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_count = sensitivity["variant_id"].nunique()
    best_rows = []
    for variant_id, group in sensitivity.groupby("variant_id", sort=False):
        mean_best = group.loc[group["mean_regret"].idxmin()]
        worst_best = group.loc[group["worst_regret"].idxmin()]
        best_rows.extend(
            [
                {
                    "variant_id": variant_id,
                    "criterion": "mean_best",
                    "source_solution_id": mean_best["source_solution_id"],
                    "source_scenario": mean_best["source_scenario"],
                },
                {
                    "variant_id": variant_id,
                    "criterion": "worst_best",
                    "source_solution_id": worst_best["source_solution_id"],
                    "source_scenario": worst_best["source_scenario"],
                },
            ]
        )
    best = pd.DataFrame(best_rows)
    source = (
        best.groupby(["criterion", "source_scenario"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    source["variant_count"] = variant_count
    source["selection_fraction"] = source["selection_count"] / variant_count

    top_count = max(1, math.ceil(sensitivity["source_solution_id"].nunique() * 0.10))
    top_frames = []
    for _, group in sensitivity.groupby("variant_id", sort=False):
        marked = group.copy()
        marked["robust_rank"] = marked[["mean_rank", "worst_rank"]].max(axis=1)
        top_frames.append(marked.nsmallest(top_count, "robust_rank"))
    top = pd.concat(top_frames, ignore_index=True)
    stability = (
        top.groupby(["source_solution_id", "source_scenario"], as_index=False)
        .size()
        .rename(columns={"size": "top_decile_count"})
    )
    stability["variant_count"] = variant_count
    stability["top_decile_fraction"] = stability["top_decile_count"] / variant_count
    stability = stability.merge(
        candidates[["source_solution_id", "candidate_label"]],
        on="source_solution_id",
        how="left",
    ).sort_values("top_decile_fraction", ascending=False)
    return source, stability


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_source_pareto(policies: pd.DataFrame, output: Path) -> None:
    values = policies.loc[:, OBJECTIVES].to_numpy(dtype=float)
    improvements = np.where(MINIMIZE, (1 - values) * 100, (values - 1) * 100)
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(18, 4.8))
    rng = np.random.default_rng(42)
    for objective_index, ax in enumerate(axes):
        groups = []
        for scenario_index, scenario in enumerate(SCENARIO_ORDER):
            mask = policies["scenario"].eq(scenario).to_numpy()
            group = improvements[mask, objective_index]
            groups.append(group)
            jitter = rng.normal(0, 0.045, len(group))
            ax.scatter(
                np.full(len(group), scenario_index + 1) + jitter,
                group,
                s=15,
                alpha=0.55,
                color=SCENARIO_COLORS[scenario],
            )
        ax.boxplot(groups, tick_labels=[SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
        ax.set_title(OBJECTIVE_LABELS[objective_index])
        ax.set_ylabel("Improvement from source-climate baseline (%)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Source-scenario Pareto objective distributions", y=1.02)
    _save_figure(fig, output)


def plot_design_climate_distributions(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    rng = np.random.default_rng(42)
    for ax, metric, title in zip(
        axes,
        ("mean_regret", "worst_regret"),
        ("Mean regret across climates", "Worst objective-climate regret"),
    ):
        groups = []
        for index, scenario in enumerate(SCENARIO_ORDER):
            values = summary.loc[summary["source_scenario"] == scenario, metric].to_numpy()
            groups.append(values)
            ax.scatter(
                np.full(len(values), index + 1) + rng.normal(0, 0.05, len(values)),
                values,
                s=18,
                alpha=0.6,
                color=SCENARIO_COLORS[scenario],
            )
        ax.boxplot(groups, tick_labels=[SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
        ax.set_title(title)
        ax.set_ylabel("Normalized regret (lower is better)")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Robustness distributions by policy design climate", y=1.02)
    _save_figure(fig, output)


def plot_all_policy_scatter(
    summary: pd.DataFrame, candidates: pd.DataFrame, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    for scenario in SCENARIO_ORDER:
        group = summary[summary["source_scenario"] == scenario]
        ax.scatter(
            group["mean_regret"],
            group["worst_regret"],
            s=42,
            alpha=0.7,
            label=f"{SCENARIO_LABELS[scenario]} (n={len(group)})",
            color=SCENARIO_COLORS[scenario],
            edgecolor="white",
            linewidth=0.35,
        )
    for _, candidate in candidates.iterrows():
        ax.scatter(
            candidate["mean_regret"],
            candidate["worst_regret"],
            marker="*",
            s=260,
            color=SCENARIO_COLORS[candidate["source_scenario"]],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            candidate["candidate_label"],
            (candidate["mean_regret"], candidate["worst_regret"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean normalized regret across climates (lower is better)")
    ax.set_ylabel("Worst normalized regret across climates (lower is better)")
    ax.set_title("All Pareto policies: average versus worst-case performance")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    _save_figure(fig, output)


def plot_candidate_performance(performance: pd.DataFrame, output: Path) -> None:
    candidates = performance.sort_values("candidate_order").drop_duplicates(
        "source_solution_id"
    )
    matrix_rows = []
    labels = []
    for _, candidate in candidates.iterrows():
        group = performance[
            performance["source_solution_id"] == candidate["source_solution_id"]
        ].set_index("target_scenario").loc[list(SCENARIO_ORDER)]
        values = group.loc[:, OBJECTIVES].to_numpy(dtype=float)
        improvement = np.where(MINIMIZE, (1 - values) * 100, (values - 1) * 100)
        matrix_rows.append(improvement.ravel())
        labels.append(
            f"{candidate['candidate_label']}\n"
            f"({SCENARIO_LABELS[candidate['source_scenario']]}, {candidate['source_solution_id']})"
        )
    matrix = np.asarray(matrix_rows)
    colours = np.zeros_like(matrix)
    for column in range(matrix.shape[1]):
        low, high = matrix[:, column].min(), matrix[:, column].max()
        colours[:, column] = 0.5 if high <= low else (matrix[:, column] - low) / (high - low)
    fig, ax = plt.subplots(figsize=(18, max(3.6, 1.05 * len(labels) + 2)))
    image = ax.imshow(colours, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:.1f}%", ha="center", va="center", fontsize=8)
    columns = [
        f"{SCENARIO_LABELS[scenario]}\n{objective}"
        for scenario in SCENARIO_ORDER
        for objective in OBJECTIVE_LABELS
    ]
    ax.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for boundary in (4.5, 9.5):
        ax.axvline(boundary, color="white", linewidth=2.5)
    ax.set_title("Final robust candidates (annotations are baseline-relative improvements)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    colorbar.set_label("Relative desirability within each target/objective")
    _save_figure(fig, output)


def plot_sensitivity(source: pd.DataFrame, output: Path) -> None:
    pivot = source.pivot(index="source_scenario", columns="criterion", values="selection_fraction")
    pivot = pivot.reindex(SCENARIO_ORDER).fillna(0)
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.bar(x - width / 2, pivot.get("mean_best", 0), width, label="Mean-regret winner")
    ax.bar(x + width / 2, pivot.get("worst_best", 0), width, label="Worst-regret winner")
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of 72 sensitivity variants")
    ax.set_title("Design climate of the best policy under sensitivity variants")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    _save_figure(fig, output)


def plot_mechanisms(comparison: pd.DataFrame, output: Path) -> None:
    shown = comparison.reindex(
        comparison["normalized_median_difference"].abs().sort_values(ascending=False).index
    ).head(12)
    shown = shown.sort_values("normalized_median_difference")
    colors = np.where(shown["normalized_median_difference"] >= 0, "#4C78A8", "#E45756")
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(
        shown["feature_label"], shown["normalized_median_difference"], color=colors
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Top-decile median minus other-policy median (fraction of decision range)")
    ax.set_title("Decision-variable median differences: robust top decile vs others")
    ax.text(
        0.99,
        0.02,
        "Descriptive comparison; corrected p-values are reported in the CSV",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    ax.grid(axis="x", alpha=0.22)
    _save_figure(fig, output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--robust-quantile",
        type=float,
        default=0.25,
        help="Quantile cutoff on both regret measures for the budget-robust pool.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0 < args.robust_quantile <= 1:
        raise ValueError("--robust-quantile must be in (0, 1].")
    run_dir = (args.run_dir or latest_run(Path("data"))).resolve()
    evaluation_dir = (
        args.evaluation_dir
        or Path("analysis") / run_dir.name / "all_pareto_cross_scenario"
    ).resolve()
    output_dir = (
        args.output_dir or Path("analysis") / run_dir.name / "paper_robustness"
    ).resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    policies = pd.read_csv(evaluation_dir / "all_pareto_policies.csv")
    decisions = pd.read_csv(evaluation_dir / "all_pareto_decisions.csv")
    results = pd.read_csv(evaluation_dir / "cross_scenario_results.csv")
    yearly = pd.read_csv(evaluation_dir / "cross_scenario_yearly.csv")
    expected = len(policies) * len(SCENARIO_ORDER)
    if len(results) != expected:
        raise ValueError(f"Expected {expected} cross evaluations; found {len(results)}.")

    sensitivity, regret_frames = build_sensitivity(results, policies)
    base = sensitivity[
        (sensitivity["normalization"] == "pareto_range")
        & (sensitivity["objective_weight_scheme"] == "equal")
        & (sensitivity["climate_probability_scheme"] == "equal")
    ].copy()
    base = add_physical_summary(base, results)
    base = mark_robust_top_decile(base)
    candidates = select_candidates(base, args.robust_quantile)
    performance = candidate_performance_table(candidates, results)
    climate_summary = design_climate_summary(base)
    climate_tests = design_climate_tests(base)
    correlations, mechanism_comparison = mechanism_analysis(decisions, base)
    sensitivity_source, sensitivity_stability = sensitivity_summaries(
        sensitivity, candidates
    )

    candidate_ids = set(candidates["source_solution_id"])
    candidate_decisions = decisions[decisions["solution_id"].isin(candidate_ids)].merge(
        candidates[
            ["source_solution_id", "candidate_roles", "candidate_label", "candidate_order"]
        ],
        left_on="solution_id",
        right_on="source_solution_id",
        validate="one_to_one",
    ).sort_values("candidate_order")
    candidate_yearly = yearly[yearly["source_solution_id"].isin(candidate_ids)].merge(
        candidates[
            ["source_solution_id", "candidate_roles", "candidate_label", "candidate_order"]
        ],
        on="source_solution_id",
        validate="many_to_one",
    ).sort_values(["candidate_order", "target_scenario", "year"])

    _atomic_csv(base.sort_values("mean_regret"), output_dir / "all_policy_robustness.csv")
    _atomic_csv(
        regret_frames["pareto_range"], output_dir / "cross_scenario_regrets.csv"
    )
    _atomic_csv(climate_summary, output_dir / "design_climate_summary.csv")
    _atomic_csv(climate_tests, output_dir / "design_climate_tests.csv")
    _atomic_csv(candidates, output_dir / "final_candidates.csv")
    _atomic_csv(performance, output_dir / "final_candidate_performance.csv")
    _atomic_csv(candidate_decisions, output_dir / "final_candidate_decisions.csv")
    _atomic_csv(candidate_yearly, output_dir / "final_candidate_yearly.csv")
    _atomic_csv(correlations, output_dir / "mechanism_correlations.csv")
    _atomic_csv(mechanism_comparison, output_dir / "mechanism_top_decile_comparison.csv")
    _atomic_csv(sensitivity, output_dir / "sensitivity_scores.csv")
    _atomic_csv(sensitivity_source, output_dir / "sensitivity_source_summary.csv")
    _atomic_csv(sensitivity_stability, output_dir / "sensitivity_policy_stability.csv")

    plot_source_pareto(policies, figure_dir / "01_source_pareto_objectives.png")
    plot_design_climate_distributions(
        base, figure_dir / "02_design_climate_robustness.png"
    )
    plot_all_policy_scatter(base, candidates, figure_dir / "03_all_policy_robustness.png")
    plot_candidate_performance(
        performance, figure_dir / "04_final_candidate_performance.png"
    )
    # Reuse the established policy-design visual with candidate labels.
    from analyze_optuna_to7_representatives import plot_policy_design

    design_plot = candidate_decisions.copy()
    design_plot["selection_label"] = design_plot["candidate_label"]
    plot_policy_design(design_plot, figure_dir / "05_final_candidate_design.png")
    plot_sensitivity(sensitivity_source, figure_dir / "06_sensitivity_stability.png")
    plot_mechanisms(mechanism_comparison, figure_dir / "07_robust_policy_mechanisms.png")

    metadata = {
        "source_run_dir": str(run_dir),
        "evaluation_dir": str(evaluation_dir),
        "policy_count": len(policies),
        "cross_evaluation_count": len(results),
        "base_normalization": "pareto_range",
        "base_objective_weights": "equal",
        "base_climate_probabilities": "equal",
        "normalization_methods": list(NORMALIZATIONS),
        "objective_weight_schemes": {
            key: value.tolist() for key, value in OBJECTIVE_WEIGHT_SCHEMES.items()
        },
        "climate_probability_schemes": {
            key: (value / value.sum()).tolist()
            for key, value in CLIMATE_PROBABILITY_SCHEMES.items()
        },
        "sensitivity_variant_count": sensitivity["variant_id"].nunique(),
        "robust_top_decile_definition": (
            "Lowest 10% by the maximum of mean-regret and worst-regret percentile ranks."
        ),
        "budget_robust_pool_quantile": args.robust_quantile,
        "weather_realization_scope": (
            "One deterministic weather trajectory per climate scenario; "
            "no weather-ensemble uncertainty is included."
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved paper robustness analysis: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
