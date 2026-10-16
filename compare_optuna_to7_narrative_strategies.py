"""Four fixed, budget-feasible narrative strategies; no objective-based tuning.

Default: original weather, present/2C/4C. --expand-ensemble adds the exact
50 saved permutations, retaining original-order controls separately. All
decisions are frozen before any SD evaluation and reused across climates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_optuna_to7_policy_contributions import costs_by_year
from analyze_optuna_to7_weather_selection_stability import fixed_regrets
from compare_optuna_to7_gray_infrastructure_pareto import dominates, losses
from evaluate_optuna_to7_cross_scenario import (
    OBJECTIVES, RAW_METRICS, SCENARIO_ORDER, decision_from_row,
)
from evaluate_optuna_to7_weather_ensemble import (
    atomic_json, case_path, read_case, signature, worker, worker_init,
)
from plot_optuna_to7_selected_policy_narratives import (
    COLORS, COMPONENTS, LABELS, POLICIES, configure_fonts, save,
)

TOTAL, ANNUAL = 50e9, 5e9
FOREST_COST = 37_200.
HOUSING_COST = 3e6
BREEDING_TOTAL = .5e9
METRICS = list(RAW_METRICS[:5])
SIGNS = np.array([1., 1., 1., -1., -1.])
SECTORS = ("disaster", "ecosystem", "agriculture")
SECTOR_LABELS = ["災害対策", "生態系", "農業・農地"]
SECTOR_COMPONENTS = {
    "disaster": ("drainage", "dam", "levee", "elevation", "migration"),
    "ecosystem": ("forest",), "agriculture": ("paddy_dam", "breeding"),
}
STRATEGIES = {
    "S1_disaster": {"label": "S1 災害対策優先", "shares": [.6, .2, .2]},
    "S2_ecosystem": {"label": "S2 生態系優先", "shares": [.2, .6, .2]},
    "S3_agriculture": {"label": "S3 農業・農地対策優先", "shares": [.2, .2, .6]},
    "S4_equal": {"label": "S4 均等配分", "shares": [1/3, 1/3, 1/3]},
}
INPUTS = ["config.json", "source_weather.csv", "sequences.json", "baselines.csv",
          "selected_decisions.csv", "policy_labels.csv", "ensemble_results.csv",
          "regret_normalization.json", "sensitivity_scores.csv",
          "review_original_vs_ensemble.csv"]


def build_design(shares):
    """Convert sector envelopes into one fixed feasible original decision.

    No model outcomes enter this calculation. Forest cost reserves use each
    year's END managed area, an upper bound on its annual mean charge, and
    include maintenance through 2023. Unspent envelopes are not reassigned.
    The three one-time investments are phased in a fixed 2009/2010/2011 order.
    Five disaster actions have equal lifetime budget settings, reduced equally
    to meet the original bounds and conservative annual spending limits.
    """
    from run_vensim_with_pysd_to7_optuna import _zero_decision
    values = np.asarray(shares, float)
    if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError("Expected three nonnegative shares summing to one")
    frames = dict(zip(SECTORS, (TOTAL * values).tolist()))
    if frames["agriculture"] < BREEDING_TOTAL:
        raise ValueError("The composite agricultural strategy needs the fixed breeding cost")
    forest_year_weights = np.minimum(np.arange(1, 16), 10)
    forest_area = min(3000., frames["ecosystem"] / (FOREST_COST * forest_year_weights.sum()))
    forest_upper = forest_area * FOREST_COST * forest_year_weights
    paddy = min(5e9, (frames["agriculture"] - BREEDING_TOTAL) / 10)
    other_upper = forest_upper + np.array([paddy + BREEDING_TOTAL / 10] * 10 + [0.] * 5)
    # Housing each receives t/10 annually for ten years; each structure gets t
    # once. Fixed order is an implementation convention, not a priority result.
    coefficients = np.array([1.2] * 3 + [.2] * 7 + [0.] * 5)
    active = coefficients > 0
    if (other_upper > ANNUAL).any():
        raise ValueError("Non-disaster schedules exceed the annual ceiling")
    equal_amount = min(frames["disaster"] / 5, 5e9, 1000 * HOUSING_COST * 10,
                       float(np.min((ANNUAL - 1. - other_upper[active]) / coefficients[active])))
    if equal_amount < 0:
        raise ValueError("No feasible equal disaster allocation")
    d = _zero_decision()
    for index, action in enumerate(("drainage", "dam", "levee")):
        d[f"{action}_investment_amount"] = equal_amount
        d[f"{action}_start_year"] = index
    for prefix in ("house_elevation", "migration", "paddy_dam", "forest_management"):
        d[f"{prefix}_start_year"] = 0
        d[f"{prefix}_end_year"] = 9
    d["house_elevation_per_year"] = equal_amount / (10 * HOUSING_COST)
    d["migration_per_year"] = equal_amount / (10 * HOUSING_COST)
    d["annual_forest_management_conversion_area"] = forest_area
    d["annual_paddy_dam_investment"] = paddy
    d.update(breeding_enabled=1, breeding_start_year=0, breeding_end_year=9)
    upper = other_upper + coefficients * equal_amount
    reserved = {"disaster": equal_amount * 5, "ecosystem": float(forest_upper.sum()),
                "agriculture": paddy * 10 + BREEDING_TOTAL}
    validate_decision(d)
    if upper.max() > ANNUAL or upper.sum() > TOTAL + .01:
        raise ValueError("Conservative budget guard failed")
    return {"target_shares": values.tolist(), "sector_envelope_yen": frames,
            "sector_reserved_upper_yen": reserved,
            "equal_disaster_action_amount_yen": equal_amount,
            "annual_spending_upper_yen": upper.tolist(),
            "annual_forest_cost_upper_yen": forest_upper.tolist(), "decision": d}


def validate_decision(d):
    from run_vensim_with_pysd_to7_optuna import CONTINUOUS_BOUNDS, decision_to_policy_params, policy_params_to_decision
    for key, (low, high) in CONTINUOUS_BOUNDS.items():
        if not low <= d[key] <= high:
            raise ValueError(f"Decision outside optimizer bounds: {key}")
    replay = policy_params_to_decision(decision_to_policy_params(d))
    if replay != d:
        raise ValueError("Decision does not round-trip through optimizer representation")


def strategy_worker(task):
    """Use the existing evaluator; check reset after each original-order case."""
    result = worker(task)
    if task["sequence_id"] == "original":
        from run_vensim_with_pysd_to7_optuna import _zero_decision
        control = dict(task, source_solution_id="no_policy", decision=_zero_decision())
        result["post_policy_baseline"] = worker(control)["metrics"]
    return result


def csv(rows, out, name):
    pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")


def validate_record(record, design, ensemble):
    decision = design["decision"]
    costs = {a: costs_by_year(record, decision, a) for a in COMPONENTS}
    matrix = np.array([costs[a] for a in COMPONENTS])
    years = record["yearly"]
    if not np.isfinite(matrix).all() or (matrix < -.001).any():
        raise ValueError("Invalid component expenditures")
    np.testing.assert_allclose(matrix.sum(axis=0), [r["policy_additional_cost"] for r in years], rtol=1e-10, atol=.001)
    np.testing.assert_allclose([matrix.sum(), matrix.sum(axis=0).max()],
        [record["metrics"]["policy_additional_cost"], record["metrics"]["max_annual_policy_additional_cost"]], rtol=1e-10, atol=.001)
    if np.any(matrix.sum(axis=0) > np.array(design["annual_spending_upper_yen"]) + .01):
        raise ValueError("Realized cost exceeded conservative annual reserve")
    if np.any(matrix.sum(axis=0) > ANNUAL + .01) or matrix.sum() > TOTAL + .01:
        raise ValueError("Budget exceeded")
    for sector, components in SECTOR_COMPONENTS.items():
        actual = sum(sum(costs[a]) for a in components)
        if actual > design["sector_envelope_yen"][sector] + .01:
            raise ValueError(f"Sector envelope exceeded: {sector}")
    # No new conversions or investments after 2018; forest maintenance persists.
    for action in COMPONENTS:
        if action != "forest" and max(costs[action][10:]) > .001:
            raise ValueError("New expenditure outside the agreed ten years")
    np.testing.assert_allclose([sum(costs[a]) for a in SECTOR_COMPONENTS["disaster"]],
                               design["equal_disaster_action_amount_yen"], rtol=1e-10, atol=.001)
    checks = []
    if record["key"][0] == "original":
        expected = read_case(ensemble, ("original", record["key"][1], "no_policy"))["metrics"]
        for metric in RAW_METRICS:
            value = record["post_policy_baseline"][metric]
            ok = bool(np.isclose(value, expected[metric], rtol=1e-10, atol=1e-5))
            checks.append({"case_id": "__".join(record["key"]), "metric": metric,
                           "difference": value - expected[metric], "passed": ok})
    return costs, checks


def export(ensemble, out, designs, sequences, records, reference_records):
    """Use saved per-sequence baselines and frozen original Pareto scales."""
    baseline = pd.read_csv(ensemble / "baselines.csv").set_index(["sequence_id", "target_scenario"])
    source = pd.read_csv(ensemble / "ensemble_results.csv")
    ids = [sid for sid, _ in POLICIES]
    sequence_ids = [s["sequence_id"] for s in sequences]
    labels_frame = pd.read_csv(ensemble / "policy_labels.csv").set_index("solution_id")
    labels = {k: v["label"] for k, v in STRATEGIES.items()}
    labels.update({sid: labels_frame.loc[sid, "plot_label"] for sid in ids})
    labels["no_policy"] = "追加対策なし"
    order = [*STRATEGIES, *ids, "no_policy"]
    selected = source.loc[source.source_solution_id.isin(ids) & source.sequence_id.isin(sequence_ids)].copy()
    selected["policy_id"] = selected.source_solution_id
    selected["policy_kind"] = "optimized_reference"
    result_rows, annual, checks = [], [], []
    budgets = []
    for record in records.values():
        sequence, climate, sid = record["key"]
        costs, identity = validate_record(record, designs[sid], ensemble)
        checks.extend(identity)
        row = {"sequence_id": sequence, "target_scenario": climate,
               "is_original": sequence == "original", "policy_id": sid,
               "policy_kind": "narrative_strategy", **record["metrics"]}
        result_rows.append(row)
        for i, y in enumerate(record["yearly"]):
            annual.append({"policy_id": sid, "sequence_id": sequence, "target_scenario": climate,
                           **y, **{f"cost_{a}_yen": costs[a][i] for a in COMPONENTS}})
        budget = {"policy_id": sid, "sequence_id": sequence, "target_scenario": climate,
                  "is_original": sequence == "original", "actual_total_yen": sum(sum(v) for v in costs.values())}
        budget["unused_total_yen"] = TOTAL - budget["actual_total_yen"]
        for a in COMPONENTS:
            budget[f"cost_{a}_yen"] = sum(costs[a])
        for s, components in SECTOR_COMPONENTS.items():
            spent = sum(sum(costs[a]) for a in components)
            budget[f"envelope_{s}_yen"] = designs[sid]["sector_envelope_yen"][s]
            budget[f"actual_{s}_yen"] = spent
            budget[f"unused_{s}_yen"] = budget[f"envelope_{s}_yen"] - spent
            budget[f"actual_{s}_share_of_spending"] = spent / budget["actual_total_yen"]
        budgets.append(budget)
    for seq in sequence_ids:
        for climate in SCENARIO_ORDER:
            b = baseline.loc[(seq, climate)]
            result_rows.append({"sequence_id": seq, "target_scenario": climate,
                               "is_original": seq == "original", "policy_id": "no_policy",
                               "policy_kind": "baseline", **{m: float(b[m]) for m in RAW_METRICS}})
    frame = pd.concat([pd.DataFrame(result_rows), selected], ignore_index=True)
    if frame.duplicated(["policy_id", "sequence_id", "target_scenario"]).any() or len(frame) != len(order) * len(sequences) * 3:
        raise ValueError("Missing or duplicated comparison cases")
    for metric, objective, sign in zip(METRICS, OBJECTIVES, SIGNS):
        b = np.array([baseline.loc[(s, c), metric] for s, c in zip(frame.sequence_id, frame.target_scenario)])
        if (b <= 0).any():
            raise ValueError("Undefined baseline ratio")
        frame[objective] = frame[metric].to_numpy(float) / b
        frame[f"improvement_{metric}_pct"] = sign * (1 - frame[objective]) * 100
    frame["feasible"] = (frame.policy_additional_cost <= TOTAL + .01) & (frame.max_annual_policy_additional_cost <= ANNUAL + .01)
    if not frame.feasible.all():
        raise ValueError("Infeasible comparison case")
    settings = json.loads((ensemble / "regret_normalization.json").read_text())["pareto_range"]
    regrets = fixed_regrets(frame, settings)
    for j, metric in enumerate(METRICS):
        frame[f"regret_{metric}"] = regrets[:, j]
    frame["mean_reference_regret"] = regrets.mean(axis=1)
    frame["max_component_reference_regret"] = regrets.max(axis=1)
    frame["policy_label"] = frame.policy_id.map(labels)
    csv(frame, out, "comparison_cases.csv")
    csv(annual, out, "strategy_annual_results.csv")
    csv(budgets, out, "strategy_budget_cases.csv")
    # Independently replay P91 in all three climates to check worker/input identity.
    for r in reference_records:
        seq, c, sid = r["key"]
        expected = read_case(ensemble, (seq, c, sid))["metrics"]
        for m in RAW_METRICS:
            value = r["metrics"][m]
            checks.append({"case_id": "__".join(r["key"]), "metric": m,
                           "difference": value - expected[m],
                           "passed": bool(np.isclose(value, expected[m], rtol=1e-10, atol=1e-5))})
    if not checks or not all(c["passed"] for c in checks):
        raise ValueError("Identity validation failed")
    csv(checks, out, "identity_validation.csv")
    scores, performance, budget_summary = [], [], []
    for scope, mask in (("original", frame.sequence_id.eq("original")),
                        ("ensemble", frame.sequence_id.ne("original"))):
        part = frame.loc[mask]
        if part.empty:
            continue
        for sid in order:
            g = part.loc[part.policy_id.eq(sid)]
            scores.append({"scope": scope, "policy_id": sid, "policy_label": labels[sid],
                           "sequence_count": g.sequence_id.nunique(),
                           "mean_regret": float(g.mean_reference_regret.mean()),
                           "worst_regret": float(g.max_component_reference_regret.max()),
                           "mean_actual_cost_yen": float(g.policy_additional_cost.mean()),
                           "min_actual_cost_yen": float(g.policy_additional_cost.min()),
                           "max_actual_cost_yen": float(g.policy_additional_cost.max()),
                           "all_feasible": bool(g.feasible.all())})
            for climate in SCENARIO_ORDER:
                h = g.loc[g.target_scenario.eq(climate)]
                for metric in METRICS:
                    v = h[f"improvement_{metric}_pct"]
                    performance.append({"scope": scope, "policy_id": sid, "target_scenario": climate,
                                        "metric": metric, "sequence_count": len(h),
                                        "mean_improvement_pct": float(v.mean()),
                                        "p10_improvement_pct": float(v.quantile(.1)),
                                        "p90_improvement_pct": float(v.quantile(.9)),
                                        "min_improvement_pct": float(v.min()), "max_improvement_pct": float(v.max())})
        b = pd.DataFrame(budgets)
        b = b.loc[b.is_original if scope == "original" else ~b.is_original]
        for (sid, c), h in b.groupby(["policy_id", "target_scenario"], sort=False):
            row = {"scope": scope, "policy_id": sid, "target_scenario": c, "sequence_count": len(h)}
            for column in h.select_dtypes(include="number").columns:
                row[f"mean_{column}"] = float(h[column].mean())
            budget_summary.append(row)
    score_frame = pd.DataFrame(scores)
    for scope in score_frame.scope.unique():
        known = score_frame.loc[score_frame.scope.eq(scope) & score_frame.policy_id.isin(ids)].set_index("policy_id")
        if scope == "original":
            expected = pd.read_csv(ensemble / "review_original_vs_ensemble.csv").set_index("source_solution_id")
            columns = ["original_mean_regret", "original_worst_regret"]
        else:
            expected = pd.read_csv(ensemble / "sensitivity_scores.csv")
            expected = expected.loc[expected.variant_id.eq("pareto_range__equal__equal")].set_index("source_solution_id")
            columns = ["mean_regret", "worst_regret"]
        np.testing.assert_allclose(known.loc[ids, ["mean_regret", "worst_regret"]], expected.loc[ids, columns], rtol=1e-10, atol=1e-12)
    csv(score_frame, out, "policy_scores.csv")
    csv(performance, out, "performance_summary.csv")
    csv(budget_summary, out, "strategy_budget_summary.csv")
    # Original-order objective dominance is unweighted, within each target climate.
    pairs = []
    original = frame.loc[frame.sequence_id.eq("original")]
    for c in SCENARIO_ORDER:
        b = original.loc[original.policy_id.eq("no_policy") & original.target_scenario.eq(c)].iloc[0]
        for sid in STRATEGIES:
            candidate = original.loc[original.policy_id.eq(sid) & original.target_scenario.eq(c)].iloc[0]
            for ref in ids:
                r = original.loc[original.policy_id.eq(ref) & original.target_scenario.eq(c)].iloc[0]
                pairs.append({"strategy_id": sid, "target_scenario": c, "reference_policy_id": ref,
                              "reference_label": labels[ref],
                              "reference_dominates": dominates(losses(r, b), losses(candidate, b)),
                              "strategy_dominates": dominates(losses(candidate, b), losses(r, b)),
                              **{f"reference_advantage_{m}_pp": float(r[f"improvement_{m}_pct"] - candidate[f"improvement_{m}_pct"]) for m in METRICS}})
    csv(pairs, out, "original_objective_comparisons.csv")
    plot_budgets(pd.DataFrame(budgets), designs, out)
    plot_performance(frame, order, labels, out)
    report(out, score_frame, pd.DataFrame(budgets), pd.DataFrame(pairs), len(sequences) - 1)
    return {"comparison_case_count": len(frame), "new_policy_case_count": len(records),
            "identity_check_count": len(checks), "all_identity_passed": True,
            "max_identity_absolute_difference": max(abs(r["difference"]) for r in checks),
            "all_cases_budget_feasible": True, "all_sector_envelopes_respected": True,
            "all_disaster_actions_equal_cost": True, "existing_scores_reproduced": True,
            "original_excluded_from_ensemble_statistics": True}


def plot_budgets(budgets, designs, out):
    configure_fonts()
    # Actual costs shown for original present; other climates/orders remain in CSV.
    b = budgets.loc[budgets.is_original & budgets.target_scenario.eq("present")].set_index("policy_id").loc[list(STRATEGIES)]
    x = np.arange(4)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    colors = ["#d55e00", "#009e73", "#e69f00"]
    for offset, kind, alpha in ((-.19, "envelope", .35), (.19, "actual", 1.)):
        bottom = np.zeros(4)
        for sector, color, label in zip(SECTORS, colors, SECTOR_LABELS):
            values = b[f"{kind}_{sector}_yen"].to_numpy() / 1e8
            axes[0].bar(x + offset, values, width=.35, bottom=bottom, color=color, alpha=alpha,
                        label=label if kind == "actual" else None)
            bottom += values
        for i, v in enumerate(bottom):
            axes[0].text(i + offset, v + 4, f"{v:.1f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 550)
    axes[0].set_title("分野別の配分枠（左・淡色）と実支出（右・濃色）")
    axes[0].legend(ncol=3)
    bottom = np.zeros(4)
    for action, label, color in zip(COMPONENTS, LABELS, COLORS):
        v = b[f"cost_{action}_yen"].to_numpy() / 1e8
        axes[1].bar(x, v, bottom=bottom, color=color, label=label)
        bottom += v
    axes[1].set_title("対策別の実支出：維持管理費を含む2009～2023年の合計")
    axes[1].legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(.5, -.15))
    for ax in axes:
        ax.set_xticks(x, [v["label"] for v in STRATEGIES.values()])
        ax.set_ylabel("追加政策費用（億円）")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("固定した4戦略の予算配分と実支出（基本系列・present）")
    fig.text(.5, .01, "配分枠の未使用分は再配分しない。新規対策は2009～2018年、森林管理費は2023年まで計上。\n"
             "排水・ダム・堤防の投資年は2009・2010・2011年。各気候・各系列の実支出はCSVに保存。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    save(fig, out, "01_budget_allocation")


def plot_performance(frame, order, labels, out):
    configure_fonts()
    random = frame.loc[frame.sequence_id.ne("original")]
    original = frame.loc[frame.sequence_id.eq("original")]
    names = ["洪水被害", "内水被害", "土砂災害リスク", "作物収益", "生物多様性"]
    fig, axes = plt.subplots(3, 5, figsize=(19, 11), sharex="col", sharey=True)
    short = [k.split("_")[0] if k in STRATEGIES else labels[k] for k in order]
    for i, c in enumerate(SCENARIO_ORDER):
        for j, (metric, name) in enumerate(zip(METRICS, names)):
            ax = axes[i, j]
            col = f"improvement_{metric}_pct"
            if len(random):
                values = [random.loc[random.policy_id.eq(s) & random.target_scenario.eq(c), col].to_numpy() for s in order]
                ax.boxplot(values, vert=False, positions=np.arange(len(order)), widths=.5,
                           whis=(10, 90), showfliers=False, patch_artist=True,
                           boxprops={"facecolor": "#c6dbef"}, medianprops={"color": "#333333"})
            h = original.loc[original.target_scenario.eq(c)].set_index("policy_id").loc[order]
            ax.scatter(h[col], range(len(order)), c=["#0072b2"] * 4 + ["#666666"] * 6, marker="D", s=23, zorder=3)
            ax.set_yticks(range(len(order)), short)
            ax.set_ylim(len(order) - .5, -.5)
            ax.axvline(0, color="gray", lw=.7)
            ax.grid(axis="x", alpha=.2)
            ax.set_title(f"{c}：{name}")
            if j == 0:
                ax.set_ylabel("固定戦略／既存候補政策")
            if i == 2:
                ax.set_xlabel("対応する無対策比の改善率（%・右ほど良い）")
    scope = "箱：50系列の25～75%、ひげ：10～90%。菱形：基本系列。分布は信頼区間ではない。" if len(random) else "菱形：基本気象系列のみ。並べ替え系列での頑健性は未評価。"
    fig.suptitle("4つの説明しやすい戦略と、既存の5候補政策の成果比較")
    fig.text(.5, .012, "S1 災害対策優先／S2 生態系優先／S3 農業・農地対策優先／S4 均等配分\n" + scope + "\n"
             "配分は予算枠であり実支出は一致しない。成果と費用は併せて解釈する。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, .96))
    save(fig, out, "02_performance_comparison")


def report(out, scores, budgets, pairs, nrandom):
    text = ["説明しやすい4戦略と既存候補政策の比較", "",
            "設定：災害／生態系／農業・農地の配分枠は60/20/20、20/60/20、20/20/60、1/3ずつ。",
            "3分野への枠は上限であり実支出比率ではない。使わない枠の再配分はしない。",
            "災害5対策は同額。排水2009年・ダム2010年・堤防2011年に一括投資し、住宅対策は2009～2018年。",
            "災害5対策は、他分野の支出を確保した後の年間上限と各対策上限に合わせ同率縮小する。",
            "森林は2009～2018年に一定面積を整備（最大3000ha/年）。年末面積から2023年までの管理費上限を予約。",
            "品種改良は年間0.5億円×10年＝5億円。農業分野の残り枠は10年均等の田んぼダム投資。",
            "年間上限50億円、総額上限500億円。未使用枠や対策量の上限を理由に実支出は異なる。",
            "各戦略はモデル成果を見る前に確定し、3気候と全気象系列で共通。モデル式・単価・遅延は変更なし。", "",
            f"気象：基本系列に加えて既存の{nrandom}並べ替え系列。基本系列は系列統計に含めない。",
            "平均後悔値は5指標×3気候×対象系列の等重み平均。最大後悔値はその全成分中の最大値。",
            "元の気候別Pareto理想点・範囲を固定。負の後悔値は0、1を超える値は切り詰めない。",
            "各系列で最適化し直した解との差ではなく、固定参照範囲からの後悔値である。", ""]
    for scope in scores.scope.unique():
        text.append(f"{scope}：平均後悔値が小さい順（費用は3気候・対象系列の平均）")
        for r in scores.loc[scores.scope.eq(scope)].sort_values("mean_regret").itertuples():
            text.append(f"  {r.policy_label}: 平均 {r.mean_regret:.6f}、最大 {r.worst_regret:.6f}、費用 {r.mean_actual_cost_yen/1e8:.2f}億円")
        text.append("")
    text.append("基本系列の支配関係（既存の5候補に限定、費用は目的に含めない）")
    for (sid, c), group in pairs.groupby(["strategy_id", "target_scenario"], sort=False):
        text.append(f"  {STRATEGIES[sid]['label']}／{c}: 既存候補に支配される件数 {int(group.reference_dominates.sum())}/5、既存候補を支配する件数 {int(group.strategy_dominates.sum())}/5")
    text.extend(["", "解釈上の範囲：", "・任意の固定配分と実施ルールによる比較。すべてのナラティブ戦略や全パレート解の優劣は証明しない。",
                 "・費用が異なるため、成果差だけで費用対効果の優越とはしない。生態系の優先枠が大きくても整備量上限で未使用になる。",
                 "・品種改良は全4戦略で同じ開始年・規模であり、その効果自体は戦略間の差にならない。",
                 "・森林や田んぼダムには防災効果もある。分野分類は主な政策目的を表し効果を分離しない。",
                 "・全候補は元の単発投資・連続実施期間という政策表現の範囲内。再最適化・費用目的の追加はしていない。", "",
                 "図：01_budget_allocation（枠と実支出）、02_performance_comparison（5指標と気象系列分布）。",
                 "表：strategy_design.json／policy_decisions.csv、strategy_budget_summary.csv、policy_scores.csv、performance_summary.csv。"])
    (out / "RESULTS_INTERPRETATION.txt").write_text("\n".join(text) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expand-ensemble", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    ensemble = args.ensemble_dir.resolve()
    out = (args.output_dir or ensemble / "narrative_strategies").resolve()
    if out == ensemble or ensemble in out.parents and out.name in ("cases", "reference_pareto"):
        raise ValueError("Use a separate comparison directory")
    source_config = json.loads((ensemble / "config.json").read_text(encoding="utf-8"))
    if signature() != source_config["signature"]:
        raise ValueError("Model/weather signature changed")
    paths = [ensemble / name for name in INPUTS]
    input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    designs = {sid: build_design(spec["shares"]) for sid, spec in STRATEGIES.items()}
    config = {"schema": 1, "model_signature": signature(), "input_sha256": input_hashes,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "designs": designs, "notes": ["Fixed rules before evaluation; no objective-based tuning.",
                  "Envelopes, not guaranteed expenditure shares; no unused-budget reassignment.",
                  "Original optimizer representation and bounds; fixed one-time years 2009/2010/2011."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)
    if (out / "config.json").exists():
        if json.loads((out / "config.json").read_text()) != config:
            raise ValueError("Inputs/design/code changed: choose another output directory")
    else:
        atomic_json(out / "config.json", config)
    atomic_json(out / "strategy_design.json", designs)
    csv([{"policy_id": sid, "label": STRATEGIES[sid]["label"], **spec["decision"]} for sid, spec in designs.items()], out, "policy_decisions.csv")
    all_sequences = json.loads((ensemble / "sequences.json").read_text())
    if len(all_sequences) != 51 or sum(s["sequence_id"] == "original" for s in all_sequences) != 1:
        raise ValueError("Expected exactly the original and 50 saved permutations")
    sequences = all_sequences if args.expand_ensemble else [s for s in all_sequences if s["sequence_id"] == "original"]
    tasks = [{"sequence_id": seq["sequence_id"], "target_scenario": c,
              "source_solution_id": sid, "decision": designs[sid]["decision"]}
             for seq in sequences for c in SCENARIO_ORDER for sid in STRATEGIES]
    reference_id = "2C_trial_00010367"
    reference_decisions = pd.read_csv(ensemble / "selected_decisions.csv").set_index("solution_id")
    reference_tasks = [{"sequence_id": "original", "target_scenario": c, "source_solution_id": reference_id,
                        "decision": decision_from_row(reference_decisions.loc[reference_id])} for c in SCENARIO_ORDER]
    all_tasks = tasks + reference_tasks
    records = {}
    for task in all_tasks:
        key = (task["sequence_id"], task["target_scenario"], task["source_solution_id"])
        r = read_case(out, key)
        if r is not None:
            records[key] = r
    pending = [t for t in all_tasks if (t["sequence_id"], t["target_scenario"], t["source_solution_id"]) not in records]
    if args.analyze_only and pending:
        raise ValueError("Saved cases incomplete for requested scope")
    started = time.perf_counter()
    state = {"state": "running", "scope": "ensemble" if args.expand_ensemble else "original",
             "total_cases": len(all_tasks), "completed_cases": len(records), "workers": args.workers}
    atomic_json(out / "run_status.json", state)
    if pending:
        with multiprocessing.get_context("spawn").Pool(args.workers, initializer=worker_init,
            initargs=(str(ensemble / "source_weather.csv"), all_sequences)) as pool:
            for r in pool.imap_unordered(strategy_worker, pending):
                key = tuple(r["key"])
                records[key] = r
                atomic_json(case_path(out, key), r)
                state.update(completed_cases=len(records), elapsed_seconds=time.perf_counter() - started)
                atomic_json(out / "run_status.json", state)
                if len(records) % 10 == 0 or len(records) == len(all_tasks):
                    print(f"Narrative strategies: {len(records)}/{len(all_tasks)}, {state['elapsed_seconds']:.0f}s", flush=True)
    new_records = {key: r for key, r in records.items() if key[2] in STRATEGIES}
    references = [r for key, r in records.items() if key[2] == reference_id]
    validation = export(ensemble, out, designs, sequences, new_records, references)
    if signature() != source_config["signature"] or any(hashlib.sha256(p.read_bytes()).hexdigest() != input_hashes[str(p)] for p in paths):
        raise ValueError("Source changed during evaluation")
    validation["source_hashes_unchanged"] = True
    atomic_json(out / "validation_summary.json", validation)
    state.update(state="complete", elapsed_seconds=time.perf_counter() - started, **validation)
    atomic_json(out / "run_status.json", state)
    print(f"Saved: {out}", flush=True)


if __name__ == "__main__":
    main()
