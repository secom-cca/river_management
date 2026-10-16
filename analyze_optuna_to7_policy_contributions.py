"""Paired one-action-at-a-time ablations of five fixed to7 policies.

By default assess only the original weather order. --screen-only includes
three diagnostic permutations; --expand-ensemble explicitly extends selected
pairs to all saved random orders. Original diagnostics are never pooled into
random statistics. No model equations or other decisions change.
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

from evaluate_optuna_to7_weather_ensemble import (
    atomic_json, case_path, read_case, signature, worker, worker_init,
)
from evaluate_optuna_to7_cross_scenario import (
    OBJECTIVES, RAW_METRICS, SCENARIO_ORDER, decision_from_row,
)
from analyze_optuna_to7_weather_ensemble import normalized_regrets, REGRETS
from plot_optuna_to7_selected_policy_narratives import (
    POLICIES, COMPONENTS, LABELS, configure_fonts, save,
)

METRICS = list(RAW_METRICS[:5])
METRIC_LABELS = ["洪水被害", "内水氾濫被害", "土砂災害リスク", "作物収益", "生物多様性"]
DIRECTION = np.array([1., 1., 1., -1., -1.])
REMOVE_KEYS = dict(zip(COMPONENTS, [
    "drainage_investment_amount", "dam_investment_amount", "levee_investment_amount",
    "house_elevation_per_year", "migration_per_year", "annual_paddy_dam_investment",
    "breeding_enabled", "annual_forest_management_conversion_area",
]))
RTOL, ATOL = 1e-10, 1e-6


def remove_action(decision: dict, action: str) -> dict:
    """Preserve schedules and every other action, including baseline capacity."""
    changed = decision.copy()
    changed[REMOVE_KEYS[action]] = 0
    return changed


def effect_values(with_metrics: dict, without_metrics: dict, baseline: dict) -> dict:
    """Positive contribution means the removed action improved an outcome."""
    out = {}
    for metric, sign in zip(METRICS, DIRECTION):
        a, b = float(with_metrics[metric]), float(without_metrics[metric])
        denominator = float(baseline[metric])
        delta = b - a
        out[f"with_{metric}"] = a
        out[f"without_{metric}"] = b
        out[f"delta_{metric}_without_minus_with"] = delta
        out[f"changed_{metric}"] = not bool(np.isclose(a, b, rtol=RTOL, atol=ATOL))
        out[f"contribution_{metric}_baseline_pp"] = (
            sign * delta / denominator * 100 if abs(denominator) > 1e-12 else np.nan
        )
    return out


def costs_by_year(record: dict, decision: dict, action: str) -> list[float]:
    values = []
    for row in record["yearly"]:
        i = int(row["year_index"])
        if action in ("drainage", "dam", "levee"):
            cost = decision[f"{action}_investment_amount"] if i == decision[f"{action}_start_year"] else 0.
        elif action in ("elevation", "migration"):
            prefix = "house_elevation" if action == "elevation" else "migration"
            cost = decision[f"{prefix}_per_year"] * 3_000_000 if decision[f"{prefix}_start_year"] <= i <= decision[f"{prefix}_end_year"] else 0.
        elif action == "breeding":
            # Budget bookkeeping uses exact calendar-year allocations.
            cost = 50_000_000 if decision["breeding_enabled"] and decision["breeding_start_year"] <= i <= decision["breeding_end_year"] else 0.
        else:
            cost = row["paddy_dam_actual_cost" if action == "paddy_dam" else "additional_forest_management_cost"]
        values.append(float(cost))
    return values


def paired_row(ensemble: Path, record: dict, action: str, label: str,
               decision: dict, config: dict, evidence: str) -> dict:
    key = tuple(record["key"])
    existing = read_case(ensemble, key)
    baseline = read_case(ensemble, (key[0], key[1], "no_policy"))
    if existing is None or baseline is None:
        raise ValueError(f"Missing paired reference: {key}")
    removed = costs_by_year(existing, decision, action)
    original_years = sorted(existing["yearly"], key=lambda r: r["year"])
    without_years = sorted(record["yearly"], key=lambda r: r["year"])
    for old, new, saving in zip(original_years, without_years, removed):
        if not np.isclose(old["policy_additional_cost"] - new["policy_additional_cost"], saving, rtol=1e-9, atol=.01):
            raise ValueError(f"Unexpected annual cost change: {key}, {action}, {old['year']}")
    saving = existing["metrics"]["policy_additional_cost"] - record["metrics"]["policy_additional_cost"]
    if not np.isclose(saving, sum(removed), rtol=1e-9, atol=.01):
        raise ValueError(f"Unexpected aggregate cost change: {key}, {action}")
    feasible = (record["metrics"]["policy_additional_cost"] <= config["total_budget"] + .01
                and record["metrics"]["max_annual_policy_additional_cost"] <= config["annual_budget"] + .01)
    if not feasible:
        raise ValueError(f"Ablation violates budget: {key}, {action}")
    return {"solution_id": key[2], "plot_label": label, "sequence_id": key[0],
            "target_scenario": key[1], "action": action, "action_label": LABELS[COMPONENTS.index(action)],
            "is_original": key[0] == "original", "evidence_scope": evidence,
            "removed_cost_yen": sum(removed), "cost_saving_yen": saving,
            "without_feasible": feasible,
            **effect_values(existing["metrics"], record["metrics"], baseline["metrics"])}


def choose_screening(ensemble: Path, ids: list[str], run: Path) -> tuple[list[str], dict]:
    frame = pd.read_csv(ensemble / "ensemble_results.csv")
    random = frame.loc[~frame["is_original"].astype(str).str.lower().eq("true")]
    regret, _ = normalized_regrets(random, run, "pareto_range")
    selected = regret.loc[regret["source_solution_id"].isin(ids)].copy()
    mean = selected.groupby("sequence_id")[REGRETS].mean().mean(axis=1)
    typical = (mean - mean.median()).abs().idxmin()
    adverse = selected.groupby("sequence_id")[REGRETS].max().max(axis=1).idxmax()
    # Existing annual diagnosis identifies perm_0004 as a crop-floor example.
    crop_floor = "perm_0004"
    if crop_floor not in set(random["sequence_id"]):
        raise ValueError("Expected saved crop-floor diagnostic order perm_0004.")
    sequences = list(dict.fromkeys(["original", typical, adverse, crop_floor]))
    return sequences, {"typical": typical, "adverse": adverse, "crop_floor": crop_floor,
                       "selection": "Typical closest to median selected-policy mean regret; adverse largest selected-policy component regret; fixed crop-floor example."}


def regret_breakdown(ensemble: Path, output: Path, ids: list[str], labels: dict, run: Path, basic: bool = False) -> None:
    frame = pd.read_csv(ensemble / "ensemble_results.csv")
    is_original = frame["is_original"].astype(str).str.lower().eq("true")
    frame = frame.loc[is_original if basic else ~is_original]
    regrets, _ = normalized_regrets(frame, run, "pareto_range")
    selected = regrets.loc[regrets["source_solution_id"].isin(ids)].copy()
    selected["plot_label"] = selected["source_solution_id"].map(labels)
    detailed = selected.groupby(["source_solution_id", "plot_label", "target_scenario"], sort=False)[REGRETS].mean().reset_index()
    detailed.to_csv(output / "regret_components_by_climate.csv", index=False, encoding="utf-8-sig")
    summary = selected.groupby("source_solution_id")[REGRETS].mean().reindex(ids)
    if basic:
        reference = pd.read_csv(ensemble / "review_original_vs_ensemble.csv").set_index("source_solution_id")
        reference = reference.rename(columns={"original_mean_regret": "mean_regret", "original_worst_regret": "worst_regret"})
    else:
        reference = pd.read_csv(ensemble / "sensitivity_scores.csv")
        reference = reference.loc[reference["variant_id"].eq("pareto_range__equal__equal")].set_index("source_solution_id")
    if not np.allclose(summary.mean(axis=1), reference.loc[ids, "mean_regret"], rtol=1e-10, atol=1e-12):
        raise ValueError("Regret decomposition does not reproduce saved mean scores.")
    summary.insert(0, "plot_label", [labels[sid] for sid in ids])
    summary["mean_regret"] = reference.loc[ids, "mean_regret"]
    summary["worst_regret"] = reference.loc[ids, "worst_regret"]
    summary.reset_index().to_csv(output / "policy_regret_components.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(ids))
    for metric, name, color in zip(REGRETS, METRIC_LABELS, ["#0072b2", "#56b4e9", "#e69f00", "#cc79a7", "#009e73"]):
        values = summary[metric].to_numpy() / 5
        ax.bar(summary["plot_label"], values, bottom=bottom, label=name, color=color)
        bottom += values
    ax.set_ylabel("平均後悔値への寄与（小さいほど良い）")
    ax.set_title("5政策の平均後悔値：5指標別の内訳" + ("（元の気象順序）" if basic else ""))
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -.13))
    ax.grid(axis="y", alpha=.2)
    scope = "元の気象順序・3気候・5指標" if basic else "50系列・3気候・5指標"
    fig.text(.5, .01, f"元の気候別Pareto範囲で固定正規化。{scope}を等重みで平均。積み上げ合計＝既存の平均後悔値。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .16, 1, 1))
    save(fig, output, "03_regret_components")


def summarize(rows: list[dict], output: Path, ids: list[str], labels: dict, screening: list[str], basic: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "action_effect_cases.csv", index=False, encoding="utf-8-sig")
    summary = []
    for (sid, action, scenario), group in frame.groupby(["solution_id", "action", "target_scenario"], sort=False):
        # Completed 50-series data take precedence; otherwise show diagnostics only.
        random = group.loc[~group["is_original"]]
        assessed = group if basic else random
        row = {"solution_id": sid, "plot_label": labels[sid], "action": action,
               "action_label": LABELS[COMPONENTS.index(action)], "target_scenario": scenario,
               "random_sequence_count": len(random),
               "evidence_scope": "original_only" if basic else "full_ensemble" if len(random) == 50 else "screening_only",
               "mean_cost_saving_yen": assessed["cost_saving_yen"].mean(),
               "all_without_feasible": bool(group["without_feasible"].all())}
        for metric in METRICS:
            col = f"contribution_{metric}_baseline_pp"
            row[f"original_{col}"] = float(group.loc[group["is_original"], col].iloc[0])
            row[f"changed_count_{metric}"] = int(assessed[f"changed_{metric}"].sum())
            for stat, value in (("mean", assessed[col].mean()), ("min", assessed[col].min()), ("max", assessed[col].max())):
                row[f"{stat}_{col}"] = value
        summary.append(row)
    result = pd.DataFrame(summary)
    result.to_csv(output / "action_effect_summary.csv", index=False, encoding="utf-8-sig")
    for fig_name, full in (("01_screening_contributions", False), ("02_ensemble_contributions", True)):
        if basic and full:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(19, 13), sharey=True)
        for ax, climate in zip(axes, SCENARIO_ORDER):
            values, ticks, sizes = [], [], []
            for sid in ids:
                for action in COMPONENTS:
                    group = frame.loc[frame["solution_id"].eq(sid) & frame["action"].eq(action) & frame["target_scenario"].eq(climate)]
                    if full:
                        group = group.loc[~group["is_original"]]
                        count = len(group)
                        values.append(group[[f"contribution_{m}_baseline_pp" for m in METRICS]].mean().to_numpy() if count == 50 else np.full(5, np.nan))
                    else:
                        group = group.loc[group["sequence_id"].eq("original")]
                        count = len(group)
                        values.append(group[[f"contribution_{m}_baseline_pp" for m in METRICS]].iloc[0].to_numpy())
                    ticks.append(f"{labels[sid]} {LABELS[COMPONENTS.index(action)]}")
                    sizes.append(count)
            data = np.array(values)
            limit = max(float(np.nanmax(np.abs(data))), .01) if np.isfinite(data).any() else .01
            # Common color scale is assigned below across all three panels.
            ax._contribution_data = data
            ax.set_title(f"評価気候：{climate}")
            ax.set_xticks(range(5), METRIC_LABELS, rotation=35, ha="right")
            ax.set_yticks(range(len(ticks)), ticks)
            for y in range(8, len(ticks), 8):
                ax.axhline(y - .5, color="black", linewidth=.8)
        limit = max([.01, *(float(np.nanmax(np.abs(ax._contribution_data))) for ax in axes if np.isfinite(ax._contribution_data).any())])
        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#dddddd")
        for ax in axes:
            data = ax._contribution_data
            im = ax.imshow(data, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
            for y, row in enumerate(data):
                for x, value in enumerate(row):
                    if np.isfinite(value):
                        ax.text(x, y, f"{value:.2f}" if abs(value) >= .005 else "0.00", ha="center", va="center", fontsize=7,
                                color="white" if abs(value) > limit * .55 else "black")
        cax = fig.add_axes([.925, .23, .012, .54])
        fig.colorbar(im, cax=cax, label="対策の寄与（無対策比のパーセントポイント）")
        title = "元の気象順序：一つの対策を除いたときの成果差" if not full else "50並べ替え系列：対策の平均寄与（検証を拡張した組のみ）"
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(left=.13, bottom=.12, top=.94, right=.9, wspace=.15)
        fig.text(.5, .02, "正：対策ありの方が良い。負：対策ありの方が悪い。灰色：50系列での検証対象外。\n"
                 "他の対策と実施時期は固定。対策間の相互作用があるため、寄与は加算できない。0.00は表示上の丸めを含む。", ha="center", fontsize=10)
        save(fig, output, "01_original_contributions" if basic else fig_name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--extension-pp", type=float, default=.01,
                        help="Expand a policy/action pair when any screening outcome reaches this baseline-relative percentage-point magnitude.")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--expand-ensemble", action="store_true", help="Explicitly opt in to 50-series expansion after screening.")
    parser.add_argument("--reuse-dir", type=Path, help="Reuse compatible completed ablation cases from an earlier diagnostic run.")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.extension_pp <= 0:
        parser.error("workers and extension-pp must be positive")
    if args.screen_only and args.expand_ensemble:
        parser.error("screen-only and expand-ensemble are mutually exclusive")
    basic = not (args.screen_only or args.expand_ensemble)
    ensemble = args.ensemble_dir.resolve()
    output = (args.output_dir or ensemble / "selected_policy_narratives" /
              ("policy_contributions_original" if basic else "policy_contributions_expanded")).resolve()
    config = json.loads((ensemble / "config.json").read_text(encoding="utf-8"))
    if signature() != config["signature"]:
        raise ValueError("Model/weather revision differs from the source ensemble")
    run = ensemble / "reference_pareto"
    ids = [sid for sid, _ in POLICIES]
    labels = pd.read_csv(ensemble / "policy_labels.csv").set_index("solution_id")["plot_label"].to_dict()
    decision_frame = pd.read_csv(ensemble / "selected_decisions.csv").set_index("solution_id")
    decisions = {sid: decision_from_row(decision_frame.loc[sid].to_dict()) for sid in ids}
    sequences = json.loads((ensemble / "sequences.json").read_text(encoding="utf-8"))
    random_ids = [s["sequence_id"] for s in sequences if not s["is_original"]]
    if len(random_ids) != 50:
        raise ValueError("This analysis expects the completed 50-sequence ensemble")
    screening, selection = (["original"], {"selection": "Original 2009-2023 weather order only"}) if basic else choose_screening(ensemble, ids, run)
    output.mkdir(parents=True, exist_ok=True)
    inputs = [ensemble / n for n in ("config.json", "source_weather.csv", "sequences.json", "selected_decisions.csv", "policy_labels.csv", "sensitivity_scores.csv", "review_original_vs_ensemble.csv")]
    inputs += list(run.glob("pareto_summary_*.csv"))
    plan = {"schema": 1, "policy_ids": ids, "signature": signature(), "screening_sequences": screening,
            "screening_selection": selection, "extension_pp": args.extension_pp,
            "action_remove_keys": REMOVE_KEYS,
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text(encoding="utf-8")) != plan:
            raise ValueError("Analysis configuration/code changed; use a new output directory")
    else:
        atomic_json(output / "config.json", plan)
    reuse = args.reuse_dir.resolve() if args.reuse_dir else None
    if reuse:
        prior = json.loads((reuse / "config.json").read_text(encoding="utf-8"))
        if prior["signature"] != plan["signature"] or prior["action_remove_keys"] != REMOVE_KEYS:
            raise ValueError("Reuse model/weather/action definitions differ")
        if any(plan["input_sha256"].get(k) != v for k, v in prior["input_sha256"].items()):
            raise ValueError("Reuse source snapshots differ")
        atomic_json(output / "cache_reuse_provenance.json", {"source_directory": str(reuse), "source_config": prior})
    for action in ["replay", *COMPONENTS]:
        (output / action / "cases").mkdir(parents=True, exist_ok=True)
    configure_fonts()
    regret_breakdown(ensemble, output, ids, labels, run, basic=basic)
    state = {"state": "running", "workers": args.workers, "screening_sequences": screening}
    started = time.perf_counter()
    pool = None

    def task(seq, climate, sid, action=None):
        return {"sequence_id": seq, "target_scenario": climate, "source_solution_id": sid,
                "decision": remove_action(decisions[sid], action) if action else decisions[sid]}

    def run_tasks(action, tasks):
        nonlocal pool
        folder = output / action
        pending = []
        for t in tasks:
            key = (t["sequence_id"], t["target_scenario"], t["source_solution_id"])
            if read_case(folder, key) is not None:
                continue
            # Existing levee cases contain exactly the same change and weather.
            old = read_case(reuse / action, key) if reuse else None
            if old is None and action == "levee" and key[0] != "original":
                old = read_case(ensemble / "selected_policy_narratives" / "levee_ensemble_check" / "without_levee", key)
            if old is not None:
                if action != "replay":
                    paired_row(ensemble, old, action, labels[key[2]], decisions[key[2]], config, "reused_case")
                atomic_json(case_path(folder, key), old)
            else:
                pending.append(t)
        print(f"{state['stage']}/{action}: {len(pending)} new / {len(tasks)} required", flush=True)
        if pending and args.analyze_only:
            raise ValueError("Requested saved ablations are incomplete")
        if pending and pool is None:
            pool = multiprocessing.get_context("spawn").Pool(args.workers, initializer=worker_init,
                initargs=(str(ensemble / "source_weather.csv"), sequences))
        for n, record in enumerate(pool.imap_unordered(worker, pending) if pending else [], 1):
            key = tuple(record["key"])
            if action != "replay":
                paired_row(ensemble, record, action, labels[key[2]], decisions[key[2]], config, state["stage"])
            atomic_json(case_path(folder, key), record)
            if n % 20 == 0 or n == len(pending):
                state.update({"action": action, "completed_action_cases": len(tasks) - len(pending) + n,
                              "required_action_cases": len(tasks), "elapsed_seconds": time.perf_counter() - started})
                atomic_json(output / "run_status.json", state)
                print(f"{action}: {state['completed_action_cases']}/{len(tasks)}; {state['elapsed_seconds']:.0f}s", flush=True)

    rows = []
    try:
        state["stage"] = "identity"
        controls = [task("original", climate, sid) for sid in ids for climate in SCENARIO_ORDER]
        run_tasks("replay", controls)
        validation = []
        for t in controls:
            key = (t["sequence_id"], t["target_scenario"], t["source_solution_id"])
            a, b = read_case(ensemble, key), read_case(output / "replay", key)
            for metric in RAW_METRICS:
                ok = bool(np.isclose(a["metrics"][metric], b["metrics"][metric], rtol=1e-9, atol=.01))
                validation.append({"solution_id": key[2], "target_scenario": key[1], "metric": metric,
                                   "difference": b["metrics"][metric] - a["metrics"][metric], "passed": ok})
        pd.DataFrame(validation).to_csv(output / "identity_validation.csv", index=False, encoding="utf-8-sig")
        if not all(r["passed"] for r in validation):
            raise ValueError("Replayed policies do not match source metrics")
        state["stage"] = "screening"
        for action in COMPONENTS:
            run_tasks(action, [task(seq, climate, sid, action) for seq in screening for climate in SCENARIO_ORDER for sid in ids])
        extensions = []
        for action in COMPONENTS:
            for sid in ids:
                pairs = [paired_row(ensemble, read_case(output / action, (seq, climate, sid)), action,
                                   labels[sid], decisions[sid], config, "screening") for seq in screening for climate in SCENARIO_ORDER]
                maximum = max(abs(r[f"contribution_{m}_baseline_pp"]) for r in pairs for m in METRICS)
                extend = args.expand_ensemble and (maximum >= args.extension_pp or action == "levee")
                extensions.append({"solution_id": sid, "plot_label": labels[sid], "action": action,
                                   "screening_max_abs_contribution_pp": maximum, "extend_to_50": extend,
                                   "reason": "existing full levee validation" if action == "levee" else "screening threshold"})
        pd.DataFrame(extensions).to_csv(output / "extension_selection.csv", index=False, encoding="utf-8-sig")
        state["stage"] = "full_ensemble"
        if args.expand_ensemble:
            for action in COMPONENTS:
                selected_ids = [r["solution_id"] for r in extensions if r["action"] == action and r["extend_to_50"]]
                run_tasks(action, [task(seq, climate, sid, action) for seq in random_ids for climate in SCENARIO_ORDER for sid in selected_ids])
        for selection_row in extensions:
            action, sid = selection_row["action"], selection_row["solution_id"]
            seqs = ["original", *random_ids] if selection_row["extend_to_50"] else screening
            evidence = "original_only" if basic else "full_ensemble" if len(seqs) == 51 else "screening_only"
            for seq in seqs:
                for climate in SCENARIO_ORDER:
                    record = read_case(output / action, (seq, climate, sid))
                    rows.append(paired_row(ensemble, record, action, labels[sid], decisions[sid], config, evidence))
        summary = summarize(rows, output, ids, labels, screening, basic=basic)
        metadata = {**plan, "paired_cases": len(rows), "identity_metric_checks": len(validation),
                    "expanded_policy_action_pairs": int(sum(r["extend_to_50"] for r in extensions)) if not args.screen_only else 0,
                    "notes": ["Positive contribution means action improved outcome, expressed relative to matched no-policy outcome.",
                              "Screening-only zero effects do not establish zero effects in untested orders.",
                              "All action removals preserve other decisions and baseline infrastructure; no budget reallocation or reoptimization.",
                              "Contributions are conditional marginal differences, not additive attribution or Shapley values.",
                              "Mean regret uses fixed source Pareto reference and equal objective/climate weights.",
                              "Original order only." if basic else "Original order excluded from random-ensemble means."]}
        atomic_json(output / "analysis_metadata.json", metadata)
        state.update({"state": "screening_complete" if args.screen_only else "complete", "stage": "analysis",
                      "paired_cases": len(rows), "elapsed_seconds": time.perf_counter() - started})
        atomic_json(output / "run_status.json", state)
        print(f"Saved policy contribution analysis: {output}", flush=True)
    except BaseException as exc:
        state.update({"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": repr(exc)})
        atomic_json(output / "run_status.json", state)
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
