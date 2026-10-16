"""Paired ablation of levee investment for five policies and saved weather orders.

Reuse original ensemble cases as the with-levee reference. Only the levee
investment amount is set to zero; other decisions, weather and climate match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_optuna_to7_weather_ensemble import worker_init, worker, atomic_json, case_path, read_case, signature
from evaluate_optuna_to7_cross_scenario import decision_from_row, SCENARIO_ORDER, RAW_METRICS
from plot_optuna_to7_selected_policy_narratives import POLICIES, configure_fonts


OBJECTIVE_METRICS = ["flood_damage", "innundation_damage", "landslide_risk", "crop_revenue", "biodiversity"]
RTOL, ATOL = 1e-10, 1e-6


def paired_row(ensemble: Path, record: dict, label: str, investment: float, config: dict) -> dict:
    key = tuple(record["key"])
    existing = read_case(ensemble, key)
    baseline = read_case(ensemble, (key[0], key[1], "no_policy"))
    if existing is None or baseline is None:
        raise ValueError(f"Missing source reference: {key}")
    with_levee, without = existing["metrics"], record["metrics"]
    saving = with_levee["policy_additional_cost"] - without["policy_additional_cost"]
    if not np.isclose(saving, investment, rtol=1e-9, atol=.01):
        raise ValueError(f"Cost saving does not match removed levee investment: {key}")
    result = {"sequence_id": key[0], "target_scenario": key[1], "solution_id": key[2],
              "plot_label": label, "removed_levee_investment_yen": investment, "cost_saving_yen": saving,
              "with_levee_cost_yen": with_levee["policy_additional_cost"],
              "without_levee_cost_yen": without["policy_additional_cost"],
              "without_levee_feasible": without["policy_additional_cost"] <= config["total_budget"] + .01
              and without["max_annual_policy_additional_cost"] <= config["annual_budget"] + .01}
    for metric in OBJECTIVE_METRICS:
        a, b = with_levee[metric], without[metric]
        delta = b - a
        result[f"with_levee_{metric}"] = a
        result[f"without_levee_{metric}"] = b
        result[f"delta_{metric}_without_minus_with"] = delta
        result[f"changed_{metric}"] = not np.isclose(a, b, rtol=RTOL, atol=ATOL)
        denominator = baseline["metrics"][metric]
        result[f"delta_{metric}_baseline_pp"] = delta / denominator * 100 if denominator else np.nan
    return result


def summarize(ensemble: Path, output: Path, policies: pd.DataFrame, tasks: list[dict], config: dict) -> None:
    by_id = policies.set_index("solution_id")
    rows = []
    for task in tasks:
        key = (task["sequence_id"], task["target_scenario"], task["source_solution_id"])
        record = read_case(output / "without_levee", key)
        if record is None:
            raise ValueError(f"Incomplete ablation: {key}")
        policy = by_id.loc[key[2]]
        rows.append(paired_row(ensemble, record, policy["plot_label"], float(policy["param_levee_investment_amount"]), config))
    results = pd.DataFrame(rows)
    results.to_csv(output / "levee_effect_cases.csv", index=False, encoding="utf-8-sig")
    summary = []
    for (sid, scenario), group in results.groupby(["solution_id", "target_scenario"], sort=False):
        row = {"solution_id": sid, "plot_label": group["plot_label"].iloc[0], "target_scenario": scenario,
               "sequence_count": len(group), "removed_levee_investment_yen": group["removed_levee_investment_yen"].iloc[0],
               "cost_saving_yen": group["cost_saving_yen"].mean(),
               "without_levee_feasibility_rate": group["without_levee_feasible"].mean()}
        for metric in OBJECTIVE_METRICS:
            delta = group[f"delta_{metric}_without_minus_with"]
            pp = group[f"delta_{metric}_baseline_pp"]
            row[f"changed_count_{metric}"] = int(group[f"changed_{metric}"].sum())
            row[f"mean_delta_{metric}"] = delta.mean()
            row[f"min_delta_{metric}"] = delta.min()
            row[f"max_delta_{metric}"] = delta.max()
            row[f"mean_delta_{metric}_baseline_pp"] = pp.mean()
            row[f"min_delta_{metric}_baseline_pp"] = pp.min()
            row[f"max_delta_{metric}_baseline_pp"] = pp.max()
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(output / "levee_effect_summary.csv", index=False, encoding="utf-8-sig")
    configure_fonts()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, sharex=True)
    extent = max(float(results["delta_flood_damage_baseline_pp"].abs().max()) * 1.2, .01)
    ids = policies["solution_id"].tolist()
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        for i, sid in enumerate(ids):
            group = results.loc[results["solution_id"].eq(sid) & results["target_scenario"].eq(scenario)]
            x = group["delta_flood_damage_baseline_pp"].to_numpy()
            jitter = np.linspace(-.14, .14, len(x))
            ax.scatter(x, i + jitter, color="#0072b2", alpha=.45, s=12)
            ax.scatter([x.mean()], [i], color="#d55e00", marker="D", s=35, zorder=3)
            ax.text(.98, i, f"{group['changed_flood_damage'].sum()}/{len(group)}", transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=9)
        ax.set_title(f"評価気候：{scenario}")
        ax.axvline(0, color="#555555", linewidth=.8)
        ax.set_xlim(-extent, extent)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_yticks(range(len(ids)), policies["plot_label"])
        ax.set_xlabel("堤防による洪水被害削減の上乗せ\n（無対策比改善率の差：パーセントポイント）")
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("5政策・50系列：堤防投資あり／なしの比較", fontsize=15)
    fig.text(.5, .015, "正の値：堤防ありの方が洪水被害が小さい。青点：各系列、赤い菱形：平均。右端の数字：差が確認された系列数。\n"
             "既存の堤防投資額・開始年の寄与を検証。他の政策変数は固定し、最適化は再実行していない。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .93))
    for ext in ("png", "pdf"):
        fig.savefig(output / f"levee_flood_effect.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    metadata = {"policy_count": len(ids), "paired_cases": len(results), "random_sequences": results["sequence_id"].nunique(),
                "comparison": "Saved existing policy versus same policy with levee_investment_amount=0",
                "change_detection_rtol": RTOL, "change_detection_atol": ATOL,
                "changed_cases_by_objective": {m: int(results[f"changed_{m}"].sum()) for m in OBJECTIVE_METRICS},
                "notes": ["Original order is validated separately and excluded from these statistics.",
                          "This tests the existing levee amount and timing; it is not an optimized no-levee strategy.",
                          "Effect-free cases do not establish that levees are ineffective at other amounts or timings."]}
    atomic_json(output / "analysis_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Summarize saved paired cases without executing the model.")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    ensemble = args.ensemble_dir.resolve()
    output = (args.output_dir or ensemble / "selected_policy_narratives" / "levee_ensemble_check").resolve()
    config = json.loads((ensemble / "config.json").read_text(encoding="utf-8"))
    if signature() != config["signature"]:
        raise ValueError("Model or weather implementation differs from the saved ensemble.")
    sequences = json.loads((ensemble / "sequences.json").read_text(encoding="utf-8"))
    labels = pd.read_csv(ensemble / "policy_labels.csv").set_index("solution_id")
    decisions = pd.read_csv(ensemble / "selected_decisions.csv").set_index("solution_id")
    ids = [sid for sid, _reason in POLICIES]
    policies = labels.loc[ids].reset_index()
    inputs = {name: hashlib.sha256((ensemble / name).read_bytes()).hexdigest()
              for name in ("source_weather.csv", "sequences.json", "selected_decisions.csv", "policy_labels.csv", "config.json")}
    check_config = {"source_ensemble": str(ensemble), "policy_ids": ids, "input_sha256": inputs,
                    "comparison": "levee_investment_amount=0; all other decisions unchanged", "schema": 1}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        if json.loads((output / "config.json").read_text(encoding="utf-8")) != check_config:
            raise ValueError("Existing levee check uses different inputs.")
    else:
        atomic_json(output / "config.json", check_config)
    for folder in ("with_levee_replay", "without_levee"):
        (output / folder / "cases").mkdir(parents=True, exist_ok=True)
    control, tasks = [], []
    for sequence in sequences:
        for sid in ids:
            decision = decision_from_row(decisions.loc[sid].to_dict())
            if sequence["is_original"]:
                control.append({"sequence_id": "original", "target_scenario": decisions.loc[sid, "scenario"],
                                "source_solution_id": sid, "decision": decision})
            else:
                decision["levee_investment_amount"] = 0.
                for scenario in SCENARIO_ORDER:
                    tasks.append({"sequence_id": sequence["sequence_id"], "target_scenario": scenario,
                                  "source_solution_id": sid, "decision": decision})
    if args.analyze_only:
        summarize(ensemble, output, policies, tasks, config)
        return
    state = {"state": "running", "expected_ablation_cases": len(tasks), "workers": args.workers}
    atomic_json(output / "run_status.json", state)
    started = time.perf_counter()
    pool = multiprocessing.get_context("spawn").Pool(args.workers, initializer=worker_init,
               initargs=(str(ensemble / "source_weather.csv"), sequences))
    try:
        for stage, work in (("with_levee_replay", control), ("without_levee", tasks)):
            pending = [t for t in work if read_case(output / stage, (t["sequence_id"], t["target_scenario"], t["source_solution_id"])) is None]
            state["stage"], state["completed_stage_cases"] = stage, len(work) - len(pending)
            print(f"{stage}: {len(pending)} pending / {len(work)}", flush=True)
            atomic_json(output / "run_status.json", state)
            for n, record in enumerate(pool.imap_unordered(worker, pending), 1):
                atomic_json(case_path(output / stage, tuple(record["key"])), record)
                if stage == "with_levee_replay":
                    original = read_case(ensemble, tuple(record["key"]))
                    if original is None or any(not np.isclose(record["metrics"][m], original["metrics"][m], rtol=1e-9, atol=.01) for m in RAW_METRICS):
                        raise ValueError("Replayed original policy does not match the saved result.")
                else:
                    sid = record["key"][2]
                    paired_row(ensemble, record, labels.loc[sid, "plot_label"], float(labels.loc[sid, "param_levee_investment_amount"]), config)
                state["completed_stage_cases"] = len(work) - len(pending) + n
                state["elapsed_seconds"] = time.perf_counter() - started
                if n % 10 == 0 or n == len(pending):
                    atomic_json(output / "run_status.json", state)
                    print(f"{stage}: {state['completed_stage_cases']}/{len(work)}; {state['elapsed_seconds']:.0f}s", flush=True)
            if stage == "with_levee_replay":
                for t in control:
                    key = (t["sequence_id"], t["target_scenario"], t["source_solution_id"])
                    replay, original = read_case(output / stage, key), read_case(ensemble, key)
                    if replay is None or original is None or any(not np.isclose(replay["metrics"][m], original["metrics"][m], rtol=1e-9, atol=.01) for m in RAW_METRICS):
                        raise ValueError("Cached replay does not match the original policy.")
        pool.close()
        pool.join()
    except BaseException as exc:
        pool.terminate()
        pool.join()
        state.update({"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": repr(exc)})
        atomic_json(output / "run_status.json", state)
        raise
    summarize(ensemble, output, policies, tasks, config)
    state.update({"state": "complete", "completed_ablation_cases": len(tasks), "identity_metric_checks": len(control) * len(RAW_METRICS),
                  "elapsed_seconds": time.perf_counter() - started})
    atomic_json(output / "run_status.json", state)
    print(f"Saved paired levee ensemble check: {output}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
