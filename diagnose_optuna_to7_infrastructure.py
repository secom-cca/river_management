"""Trace infrastructure capacity, daily hydraulics and damage on original weather.

Uses temporary parameter overrides, never edits the model or source policies.
Capacity-disable and tenfold conversion cases are mechanism stress tests, not
calibrated scenarios or recommended policies. Each altered baseline is paired
with the same altered baseline's no-policy result.
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

from evaluate_optuna_to7_cross_scenario import RAW_METRICS, SCENARIO_ORDER, decision_from_row
from evaluate_optuna_to7_weather_ensemble import atomic_json, read_case, signature
from analyze_optuna_to7_policy_contributions import METRICS, DIRECTION, remove_action
from plot_optuna_to7_selected_policy_narratives import configure_fonts, save
from weather_ensemble_to7 import weather_override

POLICIES = {"no_policy": "無対策", "2C_trial_00008685": "P75", "2C_trial_00009646": "P85"}
ACTIONS = ("drainage", "dam", "levee", "paddy_dam")
ACTION_LABELS = dict(zip(ACTIONS, ["排水", "ダム", "堤防", "田んぼダム"]))
RESET_NAMES = ("current_highwater_discharge", "initial_dam_capacity", "initial_drainage_capacity",
               "initial_paddy_dam_ratio", "capacity_per_dam_investment", "level_per_levee_investment",
               "cost_for_unit_drainage_increase", "dam_construction_time", "levee_construction_time",
               "direct_discharge_ratio")
DAILY = ["river_discharge_upstream", "river_discharge_downstream", "discharge_allowance",
         "flood_water_amount", "flood_water_level", "innundation_level", "flood_damage_ratio",
         "inundation_damage_ratio", "drainage_capacity", "drainage", "downstream_storage",
         "downstream_storage_capacity", "downstream_outflow", "downstream_percolation",
         "evaporation_downstream", "dam_capacity", "dam_storage", "dam_inflow", "dam_outflow",
         "dam_spill", "predischarge_control", "upstream_outflow", "middle_flow", "paddy_dam_area",
         "paddy_dam_ratio", "paddyfield_storage_capacity", "overlapping_risk_houses",
         "flood_event_healthy_households", "inundation_event_healthy_households",
         "houses_damaged_by_flood", "houses_damaged_by_inundation",
         "financial_damage_by_flood", "financial_damage_by_inundation"]
_SOURCE = _OUTPUT = _DEFAULTS = None


def worker_init(source: str, output: str):
    global _SOURCE, _OUTPUT, _DEFAULTS
    import run_vensim_with_pysd_to7_opt as core
    _SOURCE = pd.read_csv(source, index_col="date", parse_dates=["date"], float_precision="round_trip")
    _OUTPUT = Path(output)
    model = core.get_model()
    _DEFAULTS = {n: float(core.FIXED_ASSUMPTIONS.get(n, getattr(model.components, n)())) for n in RESET_NAMES}


def profile_path(output, case_id):
    return output / "daily" / f"{case_id}.npz"


def worker(task):
    import run_vensim_with_pysd_to7_opt as core
    started = time.perf_counter()
    # PySD's component overrides persist between runs: explicitly restore every
    # varied constant before each task, including calibrated direct routing.
    params = {**_DEFAULTS, **task["parameters"]}
    with weather_override(core.get_model(), _SOURCE):
        results = core.run_model_yearly(task["scenario"], task["decision"], model_params=params,
                                       return_columns=list(dict.fromkeys([*core.RETURN_COLS, *DAILY])))
        metrics, yearly = core.aggregate_results(results, task["decision"])
    daily = pd.concat([r[DAILY] for r in results])
    arrays = {n: daily[n].to_numpy(float) for n in DAILY}
    arrays["time"] = daily.index.to_numpy(float)
    tmp = profile_path(_OUTPUT, task["case_id"]).with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(profile_path(_OUTPUT, task["case_id"]))
    diagnostic = []
    for i, r in enumerate(results):
        demand = np.maximum(r["downstream_storage"] - r["evaporation_downstream"]
                            - r["downstream_percolation"] - r["downstream_outflow"], 0)
        cap = r["drainage_capacity"] * 86400
        diagnostic.append({"year": 2009 + i,
            "drainage_capacity_m3s_max": float(r["drainage_capacity"].max()),
            "drainage_required_m3s_max": float(demand.max() / 86400),
            "drainage_capacity_binding_days": int((demand > cap + 1e-6).sum()),
            "dam_capacity_m3_max": float(r["dam_capacity"].max()),
            "dam_storage_ratio_max": float((r["dam_storage"] / r["dam_capacity"].clip(lower=1e-12)).max()),
            "dam_spill_days": int((r["dam_spill"] > 1e-6).sum()),
            "levee_peak_capacity_m3s_max": float(r["discharge_allowance"].max() * core.get_model().components.peaktomean_flow_ratio() / 86400),
            "paddy_dam_coverage_max": float(r["paddy_dam_ratio"].max()),
            "paddy_storage_capacity_m3_max": float(r["paddyfield_storage_capacity"].max()),
            "river_discharge_m3s_max": float(r["river_discharge_downstream"].max() / 86400),
            "flood_depth_m_max": float(r["flood_water_level"].max()),
            "inundation_depth_m_max": float(r["innundation_level"].max()),
            "flood_damage_days": int((r["flood_damage_ratio"] > 0).sum()),
            "inundation_damage_days": int((r["inundation_damage_ratio"] > 0).sum())})
    return {"case_id": task["case_id"], "metrics": {n: float(metrics[n]) for n in RAW_METRICS},
            "yearly": yearly.to_dict("records"), "diagnostic": diagnostic,
            "effective_parameters": params, "elapsed_seconds": time.perf_counter() - started}


def isolated_decision(zero, action, start, amount=5e9, end=None):
    out = zero.copy()
    if action == "paddy_dam":
        out.update(annual_paddy_dam_investment=amount, paddy_dam_start_year=start,
                   paddy_dam_end_year=start if end is None else end)
    else:
        out[f"{action}_investment_amount"] = amount
        out[f"{action}_start_year"] = start
    return out


def build_tasks(decisions):
    tasks = []
    zero = decisions["no_policy"]
    def add(variant, family, sid, decision, parameters=None, action="", value=None, compare="reference", baseline="reference"):
        for climate in SCENARIO_ORDER:
            case = f"{variant}__{sid}__{climate}"
            tasks.append({"case_id": case, "variant": variant, "family": family,
                          "policy_id": sid, "policy_label": POLICIES[sid], "scenario": climate,
                          "decision": decision, "parameters": parameters or {}, "action": action,
                          "value": value, "comparison_case": f"{compare}__{sid}__{climate}",
                          "baseline_case": f"{baseline}__no_policy__{climate}"})
    for sid, decision in decisions.items():
        add("reference", "reference", sid, decision)
        if sid != "no_policy":
            for action in ACTIONS:
                add(f"remove_{action}", "ablation", sid, remove_action(decision, action), action=action)
    for action in ACTIONS:
        for start, tag in ((0, "early"), (12, "late")):
            add(f"isolated_{action}_{tag}", "investment", "no_policy", isolated_decision(zero, action, start), action=action, value=2009 + start)
    add("paddy_full_early", "investment", "no_policy", isolated_decision(zero, "paddy_dam", 0, end=5), action="paddy_dam", value="full_coverage")
    defaults = {"drainage": ("initial_drainage_capacity", 50.),
                "dam": ("initial_dam_capacity", 74200000.),
                "levee": ("current_highwater_discharge", 5200.)}
    for action, (parameter, nominal) in defaults.items():
        for factor in (0., .8, 1.2):
            variant = f"capacity_{action}_{int(factor*100):03d}"
            for sid, decision in decisions.items():
                add(variant, "capacity", sid, decision, {parameter: nominal * factor}, action, factor, baseline=variant)
    for ratio in (0., 1.):
        variant = f"capacity_paddy_{int(ratio*100):03d}"
        for sid, decision in decisions.items():
            add(variant, "capacity", sid, decision, {"initial_paddy_dam_ratio": ratio}, "paddy_dam", ratio, baseline=variant)
    conversion = [
        ("drainage", "cost10", {"cost_for_unit_drainage_increase": 1e7}),
        ("dam", "conversion10", {"capacity_per_dam_investment": .002}),
        ("dam", "delay1", {"dam_construction_time": 1.}),
        ("levee", "conversion10", {"level_per_levee_investment": .01728}),
        ("levee", "delay1", {"levee_construction_time": 1.}),
    ]
    for action, name, params in conversion:
        variant = f"stress_{action}_{name}"
        add(variant, "stress_baseline", "no_policy", zero, params, action, name, baseline=variant)
        add(variant + "_invest", "conversion_delay", "no_policy", isolated_decision(zero, action, 0),
            params, action, name, compare=f"isolated_{action}_early", baseline=variant)
    # Repeat reference after interventions to test restoration of overrides.
    add("reference_after", "identity_after", "no_policy", zero)
    return tasks


def profile(output, case):
    with np.load(profile_path(output, case)) as archive:
        return {k: archive[k] for k in archive.files}


def export(output, tasks, records, source_config):
    rows, annual = [], []
    for task in tasks:
        r = records[task["case_id"]]
        ref = records[task["comparison_case"]]
        baseline = records[task["baseline_case"]]
        d, a = profile(output, task["case_id"]), profile(output, task["comparison_case"])
        row = {k: task[k] for k in ("case_id", "variant", "family", "policy_id", "policy_label", "scenario", "action", "value", "comparison_case", "baseline_case")}
        row.update(r["metrics"])
        row["feasible"] = r["metrics"]["policy_additional_cost"] <= source_config["total_budget"] + .01 and r["metrics"]["max_annual_policy_additional_cost"] <= source_config["annual_budget"] + .01
        for metric, sign in zip(METRICS, DIRECTION):
            row[f"delta_{metric}"] = r["metrics"][metric] - ref["metrics"][metric]
            denom = baseline["metrics"][metric]
            row[f"improvement_{metric}_matched_baseline_pp"] = sign * (denom - r["metrics"][metric]) / denom * 100 if abs(denom) > 1e-12 else np.nan
        for col in ("river_discharge_downstream", "flood_water_level", "innundation_level", "drainage", "dam_outflow", "discharge_allowance", "paddyfield_storage_capacity"):
            delta = d[col] - a[col]
            row[f"max_abs_delta_{col}"] = float(np.max(np.abs(delta)))
        for col in ("flood_damage_ratio", "inundation_damage_ratio"):
            row[f"changed_days_{col}"] = int((np.abs(d[col] - a[col]) > 1e-12).sum())
        for col in ("flood_event_healthy_households", "inundation_event_healthy_households"):
            row[f"max_abs_delta_{col}"] = float(np.max(np.abs(d[col] - a[col])))
        row["max_flood_depth_m"] = float(d["flood_water_level"].max())
        row["max_inundation_depth_m"] = float(d["innundation_level"].max())
        row["drainage_binding_days"] = sum(y["drainage_capacity_binding_days"] for y in r["diagnostic"])
        row["max_required_drainage_m3s"] = max(y["drainage_required_m3s_max"] for y in r["diagnostic"])
        rows.append(row)
        annual.extend([{**{k: row[k] for k in ("case_id", "variant", "family", "policy_label", "scenario", "action")}, **y} for y in r["diagnostic"]])
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "case_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(output / "annual_hydraulic_diagnostics.csv", index=False, encoding="utf-8-sig")
    configure_fonts()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, climate in enumerate(SCENARIO_ORDER):
        chosen = frame.loc[frame["scenario"].eq(climate) & frame["family"].eq("investment")]
        labels = [f"{ACTION_LABELS[r.action]} {'2009' if r.value == 2009 else '2021' if r.value == 2021 else '2009–14全整備'}" for r in chosen.itertuples()]
        for i, metric in enumerate(("flood_damage", "innundation_damage")):
            ax = axes[i, j]
            ax.barh(range(len(chosen)), chosen[f"improvement_{metric}_matched_baseline_pp"], color="#0072b2")
            ax.set_yticks(range(len(chosen)), labels)
            ax.invert_yaxis()
            ax.axvline(0, color="black", lw=.6)
            ax.set_title(f"{climate}：{'洪水' if i == 0 else '内水'}")
            ax.set_xlabel("無対策比改善率（%）")
            ax.grid(axis="x", alpha=.2)
    fig.suptitle("他の追加対策なし：50億円の単独投資と実施時期", fontsize=15)
    fig.text(.5, .01, "2009/2021：50億円を1年投資。田んぼダム全整備：50億円/年を2009–14に投入、全整備後は支出停止。\n予算制約を別途照合。元の気象順序のみ。各指標は同じ設定の無対策と比較。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .95))
    save(fig, output, "01_isolated_investment")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for j, climate in enumerate(SCENARIO_ORDER):
        for i, metric in enumerate(("flood_damage", "innundation_damage")):
            ax = axes[i, j]
            ref = frame.loc[frame["variant"].eq("reference") & frame["policy_id"].eq("no_policy") & frame["scenario"].eq(climate), metric].iloc[0]
            for action in ("drainage", "dam", "levee"):
                part = frame.loc[frame["family"].eq("capacity") & frame["policy_id"].eq("no_policy") & frame["scenario"].eq(climate) & frame["action"].eq(action)].copy()
                x = [float(v) for v in part["value"]] + [1.]
                y = part[metric].to_list() + [ref]
                order = np.argsort(x)
                ax.plot(np.array(x)[order], np.array(y)[order] / 1e12, marker="o", label=ACTION_LABELS[action])
            ax.set_title(f"{climate}：{'洪水' if i == 0 else '内水'}")
            ax.set_xlabel("既存能力の倍率（0は能力除去の診断）")
            ax.set_ylabel("15年累積被害（兆円）")
            ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("既存能力を変えたときの被害：無対策の感度", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, output, "02_baseline_capacity")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1: parser.error("workers must be positive")
    ensemble = args.ensemble_dir.resolve()
    output = (args.output_dir or ensemble / ("infrastructure_diagnostics_original_smoke" if args.smoke else "infrastructure_diagnostics_original")).resolve()
    source_config = json.loads((ensemble / "config.json").read_text(encoding="utf-8"))
    if signature() != source_config["signature"]: raise ValueError("Source model/weather differs")
    from run_vensim_with_pysd_to7_optuna import _zero_decision
    decisions = {"no_policy": _zero_decision()}
    selected = pd.read_csv(ensemble / "selected_decisions.csv").set_index("solution_id")
    decisions.update({sid: decision_from_row(selected.loc[sid].to_dict()) for sid in POLICIES if sid != "no_policy"})
    tasks = build_tasks(decisions)
    if args.smoke:
        # Sequential default -> zero dam -> default checks override leakage.
        tasks = [next(t for t in tasks if t["variant"] == v and t["policy_id"] == "no_policy" and t["scenario"] == "4C") for v in ("reference", "capacity_dam_000", "reference_after")]
    output.mkdir(parents=True, exist_ok=True)
    for name in ("cases", "daily"): (output / name).mkdir(exist_ok=True)
    plan = {"schema": 1, "model_signature": signature(), "tasks": tasks,
            "input_sha256": {n: hashlib.sha256((ensemble / n).read_bytes()).hexdigest() for n in ("config.json", "selected_decisions.csv", "source_weather.csv")},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "notes": ["Original weather only; no optimization or model edits.",
                      "0-capacity and 10x conversion are mechanism stress tests, not calibrated confidence intervals.",
                      "Dam initial water stock is 60% of capacity and changes with initial capacity.",
                      "Altered baseline comparisons use the correspondingly altered no-policy denominator."]}
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text(encoding="utf-8")) != plan: raise ValueError("Inputs/code changed: use new output directory")
    else: atomic_json(output / "config.json", plan)
    records = {}
    for task in tasks:
        p = output / "cases" / f"{task['case_id']}.json"
        if p.exists() and profile_path(output, task["case_id"]).exists():
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["case_id"] != task["case_id"] or not np.isfinite(list(r["metrics"].values())).all(): raise ValueError("Invalid cached case")
            records[task["case_id"]] = r
    pending = [t for t in tasks if t["case_id"] not in records]
    if args.analyze_only and pending: raise ValueError("Incomplete saved cases")
    state = {"state": "running", "expected_cases": len(tasks), "completed_cases": len(records)}
    atomic_json(output / "run_status.json", state)
    started = time.perf_counter()
    if pending:
        with multiprocessing.get_context("spawn").Pool(1 if args.smoke else args.workers, initializer=worker_init,
                    initargs=(str(ensemble / "source_weather.csv"), str(output))) as pool:
            for r in pool.imap_unordered(worker, pending):
                records[r["case_id"]] = r
                atomic_json(output / "cases" / f"{r['case_id']}.json", r)
                state.update(completed_cases=len(records), elapsed_seconds=time.perf_counter() - started)
                if len(records) % 10 == 0 or len(records) == len(tasks):
                    atomic_json(output / "run_status.json", state)
                    print(f"Infrastructure diagnostics: {len(records)}/{len(tasks)}; {state['elapsed_seconds']:.0f}s", flush=True)
    validation = []
    for task in tasks:
        if task["family"] not in ("reference", "identity_after"): continue
        original = read_case(ensemble, ("original", task["scenario"], task["policy_id"]))
        r = records[task["case_id"]]
        for metric in RAW_METRICS:
            ok = bool(np.isclose(r["metrics"][metric], original["metrics"][metric], rtol=1e-9, atol=.01))
            validation.append({"case_id": task["case_id"], "metric": metric, "difference": r["metrics"][metric] - original["metrics"][metric], "passed": ok})
    pd.DataFrame(validation).to_csv(output / "identity_validation.csv", index=False, encoding="utf-8-sig")
    if not all(r["passed"] for r in validation): raise ValueError("Reference/repeated reference differs from saved source")
    if not args.smoke: export(output, tasks, records, source_config)
    state.update(state="complete", identity_checks=len(validation), elapsed_seconds=time.perf_counter() - started)
    atomic_json(output / "run_status.json", state)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
