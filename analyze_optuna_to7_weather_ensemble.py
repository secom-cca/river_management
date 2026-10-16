"""Analyze saved weather-ensemble results without rerunning the SD model."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import hashlib
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from analyze_optuna_to7_robustness import NORMALIZATIONS, OBJECTIVE_WEIGHT_SCHEMES, CLIMATE_PROBABILITY_SCHEMES
from evaluate_optuna_to7_cross_scenario import OBJECTIVES, SCENARIO_ORDER, OBJECTIVE_LABELS


DIRECTION = np.array([1, 1, 1, -1, -1])
REGRETS = [f"regret_{name}" for name in OBJECTIVES]
COLORS = {"present": "#0072b2", "2C": "#e69f00", "4C": "#009e73"}


def save(fig: plt.Figure, folder: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def normalized_regrets(results: pd.DataFrame, run: Path, method: str):
    output = results.copy()
    settings = {}
    for scenario in SCENARIO_ORDER:
        mask = output["target_scenario"].eq(scenario)
        losses = output.loc[mask, OBJECTIVES].to_numpy(float) * DIRECTION
        source = pd.read_csv(run / f"pareto_summary_{scenario}.csv")[list(OBJECTIVES)].to_numpy(float) * DIRECTION
        ideal = source.min(axis=0)
        if method == "pareto_range":
            nadir = source.max(axis=0)
        elif method == "baseline_to_ideal":
            nadir = DIRECTION.copy()
        elif method == "cross_policy_range":
            ideal, nadir = losses.min(axis=0), losses.max(axis=0)
        else:
            raise ValueError("Unknown normalization method.")
        span = nadir - ideal
        fallback = span <= 1e-15
        span[fallback] = 1.
        output.loc[mask, REGRETS] = np.maximum((losses - ideal) / span, 0.)
        settings[scenario] = {"ideal_losses": ideal.tolist(), "nadir_losses": nadir.tolist(),
                              "scales": span.tolist(), "fallback_objectives":
                              [name for name, flag in zip(OBJECTIVES, fallback) if flag]}
    return output, settings


def sensitivity_scores(results: pd.DataFrame, policies: pd.DataFrame, sequences: list[str], run: Path):
    ids = policies["solution_id"].tolist()
    ordered_keys = pd.MultiIndex.from_tuples(list(itertools.product(ids, sequences, SCENARIO_ORDER)),
                      names=["source_solution_id", "sequence_id", "target_scenario"])
    feasibility = results.groupby("source_solution_id")["feasible"].mean().reindex(ids)
    frames, normalizations = [], {}
    source_by_id = policies.set_index("solution_id")["scenario"]
    for method in NORMALIZATIONS:
        regrets, normalizations[method] = normalized_regrets(results, run, method)
        matrix = regrets.set_index(list(ordered_keys.names)).reindex(ordered_keys)[REGRETS].to_numpy(float)
        if not np.isfinite(matrix).all():
            raise ValueError("Regret matrix has missing or non-finite cells.")
        cube = matrix.reshape(len(ids), len(sequences), 3, 5)
        for objective_name, ow in OBJECTIVE_WEIGHT_SCHEMES.items():
            for climate_name, cw in CLIMATE_PROBABILITY_SCHEMES.items():
                means = np.einsum("pnco,o,c->pn", cube, ow / ow.sum(), cw / cw.sum())
                worst_by_sequence = (cube[:, :, cw > 0, :] * (ow / ow.mean())).max(axis=(2, 3))
                frame = pd.DataFrame({"source_solution_id": ids, "source_scenario": source_by_id.loc[ids].to_numpy(),
                    "mean_regret": means.mean(axis=1), "regret_q90": np.quantile(means, .9, axis=1),
                    "worst_regret": worst_by_sequence.max(axis=1),
                    "worst_regret_q90": np.quantile(worst_by_sequence, .9, axis=1),
                    "feasibility_rate": feasibility.to_numpy(), "normalization": method,
                    "objective_weight_scheme": objective_name, "climate_weight_scheme": climate_name,
                    "variant_id": f"{method}__{objective_name}__{climate_name}"})
                frame["mean_rank"] = frame["mean_regret"].rank(method="min")
                frame["worst_rank"] = frame["worst_regret"].rank(method="min")
                frames.append(frame)
    return pd.concat(frames, ignore_index=True), normalizations


def improvement_plot(random: pd.DataFrame, original: pd.DataFrame, policies: pd.DataFrame, output: Path) -> None:
    ids = policies["solution_id"].tolist()
    labels = [f"P{i+1:02d}" for i in range(len(ids))]
    fig, axes = plt.subplots(3, 5, figsize=(18, max(9, len(ids) * .48 + 4)))
    for row, scenario in enumerate(SCENARIO_ORDER):
        for col, objective in enumerate(OBJECTIVES):
            ax = axes[row, col]
            selected = random.loc[random["target_scenario"].eq(scenario)]
            sign = -1 if col < 3 else 1
            values = [(selected.loc[selected["source_solution_id"].eq(policy), objective].dropna().to_numpy() - 1) * sign * 100
                      for policy in ids]
            ax.boxplot(values, orientation="horizontal", tick_labels=labels, whis=(10, 90), showfliers=False,
                       patch_artist=True, boxprops={"facecolor": COLORS[scenario], "alpha": .3},
                       medianprops={"color": COLORS[scenario], "linewidth": 1.5})
            historical = original.loc[original["target_scenario"].eq(scenario)].set_index("source_solution_id")
            dots = (historical.reindex(ids)[objective].to_numpy(float) - 1) * sign * 100
            ax.scatter(dots, np.arange(1, len(ids) + 1), color="#333333", marker="D", s=13, zorder=3)
            ax.invert_yaxis()
            ax.axvline(0, color="#777777", linewidth=.7)
            ax.grid(axis="x", alpha=.2)
            ax.set_title(f"{scenario} | {OBJECTIVE_LABELS[col]}")
            ax.set_xlabel("Improvement vs matching no-policy baseline (%)")
            if col != 0:
                ax.set_yticklabels([])
    fig.suptitle(f"Weather-order sensitivity — {random['sequence_id'].nunique()} permutations; higher improvements are better")
    fig.text(.5, .008, "Boxes: 25–75%; whiskers: 10–90% of sampled permutations; diamonds: original order.\n"
             "These are ensemble distributions, not confidence intervals. Policy IDs are in policy_labels.csv.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .96))
    save(fig, output, "01_improvement_distributions")


def robustness_plot(scores: pd.DataFrame, labels: dict[str, str], output: Path, n_sequences: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for scenario in SCENARIO_ORDER:
        group = scores.loc[scores["source_scenario"].eq(scenario)]
        if not group.empty:
            ax.scatter(group["mean_regret"], group["worst_regret"], color=COLORS[scenario], label=scenario, s=40)
    annotate = scores if len(scores) <= 25 else scores.nsmallest(3, "mean_regret")
    for row in annotate.itertuples():
        ax.annotate(labels[row.source_solution_id], (row.mean_regret, row.worst_regret),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean reference regret across permutations, climates and objectives")
    ax.set_ylabel("Maximum reference regret among sampled cases")
    ax.grid(alpha=.2)
    ax.legend(title="Policy design climate", frameon=False)
    ax.set_title(f"Fixed-policy ensemble: {n_sequences} permutations, average versus sampled worst")
    fig.text(.5, .01, "Equal objective/climate weights; original Pareto reference ranges; lower is better.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    save(fig, output, "02_mean_vs_worst_regret")


def winners_table(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in sensitivity.groupby("variant_id", sort=False):
        group = group.loc[group["feasibility_rate"] >= 1 - 1e-12]
        if group.empty:
            continue
        for criterion in ("mean_regret", "worst_regret"):
            best = group[criterion].min()
            winners = group.loc[np.isclose(group[criterion], best, atol=1e-12, rtol=0)]
            for row in winners.itertuples():
                rows.append({"variant_id": variant, "criterion": criterion,
                             "normalization": row.normalization, "objective_weight_scheme": row.objective_weight_scheme,
                             "climate_weight_scheme": row.climate_weight_scheme, "source_solution_id": row.source_solution_id,
                             "source_scenario": row.source_scenario, "tie_weight": 1 / len(winners)})
    return pd.DataFrame(rows)


def sensitivity_plot(winners: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, criterion in zip(axes, ("mean_regret", "worst_regret")):
        group = winners.loc[winners["criterion"].eq(criterion)]
        if criterion == "worst_regret":
            # Climate weights are positive in all schemes and do not change the maximum.
            group = group.loc[group["climate_weight_scheme"].eq("equal")]
        count = group["variant_id"].nunique()
        totals = group.groupby("source_scenario")["tie_weight"].sum().reindex(SCENARIO_ORDER, fill_value=0)
        ax.bar(SCENARIO_ORDER, totals, color=[COLORS[s] for s in SCENARIO_ORDER])
        ax.set_title(f"{criterion}: {count} distinct settings")
        ax.set_ylabel("Best-policy selections (ties split equally)")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Evaluation-preference sensitivity after weather-ensemble evaluation")
    fig.tight_layout()
    save(fig, output, "03_sensitivity_winners")


def selection_heatmaps(ensemble_dir: Path) -> None:
    """Show individual winners, preserving ties, using saved analysis only."""
    winners = pd.read_csv(ensemble_dir / "sensitivity_winners.csv")
    policies = pd.read_csv(ensemble_dir / "policy_labels.csv")
    decisions = pd.read_csv(ensemble_dir / "selected_decisions.csv")
    figures = ensemble_dir / "figures"
    figures.mkdir(exist_ok=True)
    policy_index = policies.set_index("solution_id")
    summary = policies[["solution_id", "plot_label", "scenario"]].copy()
    cmap = ListedColormap(["#ffffff", "#f2cb77", "#22659b"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
    legend = [Patch(facecolor=color, edgecolor="#aaaaaa", label=label)
              for color, label in zip(cmap.colors, ("Not selected", "Joint best (tie)", "Sole best"))]
    objective_labels = ["Equal", "Flood x2", "Inundation x2", "Landslide x2", "Crop x2", "Biodiversity x2"]
    climate_labels = ["Equal", "Present", "2C", "4C"]
    normalization_labels = {"pareto_range": "Original Pareto range",
                            "baseline_to_ideal": "No-policy to ideal",
                            "cross_policy_range": "All-policy ensemble range"}

    for criterion in ("mean_regret", "worst_regret"):
        group = winners.loc[winners["criterion"].eq(criterion)].copy()
        climates = list(CLIMATE_PROBABILITY_SCHEMES) if criterion == "mean_regret" else ["equal"]
        if criterion == "worst_regret":
            group = group.loc[group["climate_weight_scheme"].eq("equal")]
        if group.empty:
            continue
        if group.duplicated(["variant_id", "source_solution_id"]).any():
            raise ValueError("Duplicate policy selections in saved winners.")
        totals = group.groupby("variant_id")["tie_weight"].sum().to_numpy()
        if not np.allclose(totals, 1., atol=1e-12, rtol=0):
            raise ValueError("Tie weights do not sum to one in saved winners.")
        variants = [f"{method}__{objective}__{climate}"
                    for method in NORMALIZATIONS for objective in OBJECTIVE_WEIGHT_SCHEMES for climate in climates]
        # A tied winner has the same selection status regardless of tie size.
        # Its fractional share is kept separately in the export, not as intensity.
        group["status"] = np.where(group["tie_weight"].eq(1.), 2, 1)
        matrix = group.pivot(index="source_solution_id", columns="variant_id", values="status")
        matrix = matrix.reindex(columns=variants, fill_value=0).fillna(0).astype(int)
        sole = group.loc[group["status"].eq(2)].groupby("source_solution_id").size()
        total = group.groupby("source_solution_id").size()
        shares = group.groupby("source_solution_id")["tie_weight"].sum()
        ordered_ids = sorted(matrix.index, key=lambda sid: (-sole.get(sid, 0), -shares.get(sid, 0), sid))
        matrix = matrix.reindex(ordered_ids)
        export = matrix.copy()
        export.insert(0, "plot_label", policy_index.loc[ordered_ids, "plot_label"].to_numpy())
        export.insert(1, "source_scenario", policy_index.loc[ordered_ids, "scenario"].to_numpy())
        export.to_csv(ensemble_dir / f"selection_matrix_{criterion}.csv", encoding="utf-8-sig")
        summary[f"{criterion}_sole_best_count"] = summary["solution_id"].map(sole).fillna(0).astype(int)
        summary[f"{criterion}_joint_best_count"] = (summary["solution_id"].map(total).fillna(0)
                                                   - summary[f"{criterion}_sole_best_count"]).astype(int)
        summary[f"{criterion}_tie_split_count"] = summary["solution_id"].map(shares).fillna(0)

        if criterion == "mean_regret":
            # Twenty-four columns per panel retain legible climate labels.
            fig, axes = plt.subplots(3, 1, figsize=(16, max(7.5, len(matrix) * .7 + 5.5)))
            for ax, method in zip(axes, NORMALIZATIONS):
                columns = [variant for variant in variants if variant.startswith(method + "__")]
                values = matrix.loc[:, columns].to_numpy()
                ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
                ax.set_yticks(range(len(matrix)),
                             [f"{policy_index.loc[sid, 'plot_label']} | {sid}" for sid in matrix.index], fontsize=9)
                ax.set_xticks(range(24), climate_labels * 6, fontsize=8)
                ax.set_xlabel("Climate weight: Equal / Present-heavy / 2C-heavy / 4C-heavy", fontsize=9)
                for j, label in enumerate(objective_labels):
                    ax.text(j * 4 + 1.5, 1.025, label, transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=9)
                    if j:
                        ax.axvline(j * 4 - .5, color="#888888", linewidth=1)
                ax.set_title(normalization_labels[method], pad=34, loc="left", fontsize=11)
                ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
                ax.set_yticks(np.arange(-.5, len(matrix), 1), minor=True)
                ax.grid(which="minor", color="#cccccc", linewidth=.35)
                ax.tick_params(which="minor", bottom=False, left=False)
            fig.suptitle("Individual policy selection — mean regret, 72 evaluation settings", fontsize=14)
            fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(.5, .02), frameon=False)
            fig.text(.5, .005, "Only policies selected at least once are shown. Cells indicate selection, not regret magnitude or probability.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .085, 1, .945), h_pad=2.6)
            save(fig, figures, "04_mean_policy_selection_heatmap")
        else:
            # Separate climate panels keep all tied policies legible in one figure.
            sizes = [int(policy_index.loc[matrix.index, "scenario"].eq(s).sum()) for s in SCENARIO_ORDER]
            fig, axes = plt.subplots(1, 3, figsize=(19, max(8, max(sizes) * .23 + 4)))
            for ax, scenario in zip(axes, SCENARIO_ORDER):
                ids = [sid for sid in matrix.index if policy_index.loc[sid, "scenario"] == scenario]
                ax.set_title(f"Policy design climate: {scenario}", fontsize=11, color=COLORS[scenario], pad=48)
                if not ids:
                    ax.text(.5, .5, "No selected policies", transform=ax.transAxes, ha="center")
                    ax.set_axis_off()
                    continue
                values = matrix.loc[ids].to_numpy()
                ax.imshow(values, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
                ax.set_ylim(max(sizes) - .5, -.5)
                ax.set_yticks(range(len(ids)), [policy_index.loc[sid, "plot_label"] for sid in ids], fontsize=8)
                ax.set_xticks(range(18), objective_labels * 3, rotation=90, fontsize=8)
                for j, method in enumerate(NORMALIZATIONS):
                    ax.text(j * 6 + 2.5, 1.025, ["Pareto", "Baseline", "All-policy"][j],
                            transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
                    if j:
                        ax.axvline(j * 6 - .5, color="#888888", linewidth=1)
                counts = (matrix.loc[:, variants] > 0).sum(axis=0).to_numpy()
                for j, count in enumerate(counts):
                    ax.text(j, -1.25, str(count), ha="center", va="bottom", fontsize=7, clip_on=False)
                ax.set_xticks(np.arange(-.5, 18, 1), minor=True)
                ax.set_yticks(np.arange(-.5, len(ids), 1), minor=True)
                ax.grid(which="minor", color="#cccccc", linewidth=.3)
                ax.tick_params(which="minor", bottom=False, left=False)
            fig.suptitle("Individual policy selection — maximum regret, 18 distinct settings", fontsize=14)
            fig.legend(handles=legend, loc="lower center", ncol=3, bbox_to_anchor=(.5, .025), frameon=False)
            fig.text(.5, .006, "Numbers above columns: total co-winners across all design climates. Yellow cells are ties, not weaker selections.\n"
                     "Climate weights do not change this maximum. Policy IDs and decisions are in sensitivity_selected_policies.csv.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, .075, 1, .94), w_pad=2.5)
            save(fig, figures, "05_worst_policy_selection_heatmap")

    details = summary.merge(decisions.drop(columns="scenario"), on="solution_id", validate="one_to_one")
    details.to_csv(ensemble_dir / "sensitivity_selected_policies.csv", index=False, encoding="utf-8-sig")
    sole_columns = [name for name in summary if name.endswith("_sole_best_count")]
    details.loc[details[sole_columns].sum(axis=1).gt(0)].to_csv(
        ensemble_dir / "sensitivity_unique_winner_policies.csv", index=False, encoding="utf-8-sig")
    print(f"Saved policy selection heatmaps and policy decisions: {ensemble_dir}")


def analyze(ensemble_dir: Path) -> None:
    config = json.loads((ensemble_dir / "config.json").read_text(encoding="utf-8"))
    results = pd.read_csv(ensemble_dir / "ensemble_results.csv")
    policies = pd.read_csv(ensemble_dir / "selected_policies.csv")
    original_mask = results["is_original"].astype(str).str.lower().eq("true")
    original, random = results.loc[original_mask].copy(), results.loc[~original_mask].copy()
    if random.duplicated(["source_solution_id", "sequence_id", "target_scenario"]).any():
        raise ValueError("Duplicate ensemble cases.")
    expected = len(policies) * config["n_sequences"] * 3
    if len(random) != expected or len(original) != len(policies) * 3:
        raise ValueError("Ensemble results are incomplete.")
    random["feasible"] = random["feasible"].astype(str).str.lower().eq("true")
    labels = dict(zip(policies["solution_id"], [f"P{i+1:02d}" for i in range(len(policies))]))
    policies.assign(plot_label=policies["solution_id"].map(labels)).to_csv(
        ensemble_dir / "policy_labels.csv", index=False, encoding="utf-8-sig")
    records = []
    for (policy, scenario), group in random.groupby(["source_solution_id", "target_scenario"], sort=False):
        historical = original.loc[original["source_solution_id"].eq(policy) & original["target_scenario"].eq(scenario)].iloc[0]
        for j, objective in enumerate(OBJECTIVES):
            sign = -1 if j < 3 else 1
            v = (group[objective].dropna() - 1) * sign * 100
            records.append({"source_solution_id": policy, "plot_label": labels[policy], "target_scenario": scenario,
                            "objective": objective, "defined_sequence_count": len(v), "sequence_count": len(group),
                            "mean_improvement_pct": v.mean(), "median_improvement_pct": v.median(),
                            "p10_improvement_pct": v.quantile(.1), "minimum_improvement_pct": v.min(),
                            "maximum_improvement_pct": v.max(), "original_improvement_pct":
                            (historical[objective] - 1) * sign * 100,
                            "budget_feasibility_rate": group["feasible"].mean()})
    pd.DataFrame(records).to_csv(ensemble_dir / "improvement_summary.csv", index=False, encoding="utf-8-sig")
    figures = ensemble_dir / "figures"
    figures.mkdir(exist_ok=True)
    improvement_plot(random, original, policies, figures)
    finite = np.isfinite(random[list(OBJECTIVES)].to_numpy(float)).all()
    metadata = {"random_sequences": config["n_sequences"], "policy_count": len(policies),
                "original_excluded_from_statistics": True,
                "all_ratios_defined": bool(finite),
                "notes": ["Common year permutations are shared by every policy and climate.",
                          "Percentiles describe sampled order uncertainty conditional on the observed years and climate shifts.",
                          "Leap-day insertion/removal can slightly change precipitation totals; see calendar_audit.csv.",
                          "Reference regret is not regret relative to an optimum re-searched for each weather realization."]}
    if finite:
        reference_dir = ensemble_dir / "reference_pareto"
        reference_dir.mkdir(exist_ok=True)
        for scenario in SCENARIO_ORDER:
            destination = reference_dir / f"pareto_summary_{scenario}.csv"
            if not destination.exists():
                shutil.copy2(Path(config["source_run"]) / destination.name, destination)
        reference_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in reference_dir.glob("pareto_summary_*.csv")}
        hash_path = reference_dir / "sha256.json"
        if hash_path.exists() and json.loads(hash_path.read_text()) != reference_hashes:
            raise ValueError("Saved Pareto normalization references changed.")
        hash_path.write_text(json.dumps(reference_hashes, indent=2), encoding="utf-8")
        sequence_ids = sorted(random["sequence_id"].unique())
        sensitivity, normalization = sensitivity_scores(random, policies, sequence_ids, reference_dir)
        sensitivity.to_csv(ensemble_dir / "sensitivity_scores.csv", index=False, encoding="utf-8-sig")
        base = sensitivity.loc[sensitivity["variant_id"].eq("pareto_range__equal__equal")].copy()
        base["plot_label"] = base["source_solution_id"].map(labels)
        base.to_csv(ensemble_dir / "policy_robustness.csv", index=False, encoding="utf-8-sig")
        winners = winners_table(sensitivity)
        winners.to_csv(ensemble_dir / "sensitivity_winners.csv", index=False, encoding="utf-8-sig")
        (ensemble_dir / "regret_normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
        robustness_plot(base, labels, figures, config["n_sequences"])
        if not winners.empty:
            sensitivity_plot(winners, figures)
            selection_heatmaps(ensemble_dir)
        metadata.update({"mean_sensitivity_settings": 72, "distinct_worst_sensitivity_settings": 18,
            "reference_pareto_sha256": reference_hashes,
            "pareto_range_reference": "Fixed source optimization Pareto ranges for each target climate (original weather order).",
            "cross_policy_range_reference": "Fixed pool of selected policies and all random permutations, separately per target climate."})
    else:
        metadata["regret_analysis_status"] = "Skipped: some matching baseline ratios are undefined. Raw metrics remain in ensemble_results.csv."
        print(metadata["regret_analysis_status"])
    (ensemble_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved ensemble analysis: {ensemble_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--selection-plots-only", action="store_true",
                        help="Draw policy selection heatmaps from saved winner tables; do not recompute scores.")
    args = parser.parse_args()
    if args.selection_plots_only:
        selection_heatmaps(args.ensemble_dir.resolve())
    else:
        analyze(args.ensemble_dir.resolve())
