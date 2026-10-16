"""Plot fixed-reference hypervolume histories from exported Optuna trial CSVs.

No model runs or new optimization trials are performed. The trial-count axis
includes infeasible, failed, and unfinished attempts; only feasible COMPLETE
trials enter the front. Default sampling is every 100 attempts, including zero
and the final attempt. Scaling is fixed throughout each scenario's history.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

from plot_optuna_to7_results import OBJECTIVES, SCENARIO_LABELS, latest_run


MINIMIZATION_SIGNS = np.array([1, 1, 1, -1, -1], dtype=float)
SCALING_SCHEMA = 1
COLORS = {"present": "#0072b2", "2C": "#e69f00", "4C": "#d55e00"}


def exact_hypervolume(front: np.ndarray, reference: np.ndarray) -> float:
    # Optuna's internal calculator is isolated here because it is not a public API.
    try:
        from optuna._hypervolume import compute_hypervolume
    except ImportError as exc:
        raise RuntimeError("This script requires Optuna's compute_hypervolume (verified with 5.0.0).") from exc
    relevant = front[np.all(front <= reference, axis=1)]
    return float(compute_hypervolume(relevant, reference, assume_pareto=True))


def make_scaling(feasible_ratios: np.ndarray, reference_value: float) -> dict:
    losses = feasible_ratios * MINIMIZATION_SIGNS
    ideal = losses.min(axis=0)
    baseline = MINIMIZATION_SIGNS.copy()  # All original baseline-relative ratios are 1.
    gain = baseline - ideal
    # Preserve directions for objectives without an observed baseline improvement.
    fallback = gain <= 1e-15
    scale = np.where(fallback, 1.0, gain)
    return {"ideal_losses": ideal.tolist(), "baseline_losses": baseline.tolist(),
            "scales": scale.tolist(), "fallback_scale_objectives":
            [name for name, flag in zip(OBJECTIVES, fallback) if flag],
            "reference_point": [reference_value] * len(OBJECTIVES)}


def validate_scaling(scaling: dict) -> None:
    for name in ("ideal_losses", "baseline_losses", "scales", "reference_point"):
        values = np.asarray(scaling[name], dtype=float)
        if values.shape != (len(OBJECTIVES),) or not np.isfinite(values).all():
            raise ValueError(f"Invalid scaling field: {name}")
    if np.any(np.asarray(scaling["scales"]) <= 0):
        raise ValueError("All normalization scales must be positive.")


def add_to_front(front: np.ndarray, point: np.ndarray) -> np.ndarray:
    # Equality is also excluded: count distinct objective vectors, not policies.
    if np.any(np.all(front <= point, axis=1)):
        return front
    removed = np.all(point <= front, axis=1) & np.any(point < front, axis=1)
    return np.vstack([front[~removed], point])


def history_from_losses(losses: np.ndarray, eligible: np.ndarray, completed: np.ndarray,
                        reference: np.ndarray, step: int) -> pd.DataFrame:
    """Process consecutive attempted trials in order, retaining the feasible front."""
    if step < 1 or len(losses) == 0:
        raise ValueError("History requires at least one trial and a positive step.")
    if len(eligible) != len(losses) or len(completed) != len(losses):
        raise ValueError("Trial masks have incompatible lengths.")
    if np.any(eligible & ~completed) or not np.isfinite(losses[eligible]).all():
        raise ValueError("Eligible trials must be complete with finite objectives.")
    front = np.empty((0, losses.shape[1]), dtype=float)
    rows = [{"trial_count": 0, "complete_count": 0, "feasible_count": 0,
             "nondominated_vectors": 0, "reference_included_vectors": 0, "hypervolume": 0.0}]
    n_complete = n_feasible = 0
    cached_hv = 0.0
    changed = False
    for count, (point, keep, complete) in enumerate(zip(losses, eligible, completed), start=1):
        n_complete += int(complete)
        n_feasible += int(keep)
        if keep:
            updated = add_to_front(front, point)
            changed = changed or updated is not front
            front = updated
        if count % step == 0 or count % 3000 == 0 or count == len(losses):
            if changed:
                cached_hv = exact_hypervolume(front, reference)
                changed = False
            rows.append({"trial_count": count, "complete_count": n_complete,
                         "feasible_count": n_feasible, "nondominated_vectors": len(front),
                         "reference_included_vectors": int(np.all(front <= reference, axis=1).sum()),
                         "hypervolume": cached_hv})
    result = pd.DataFrame(rows)
    if np.any(np.diff(result["hypervolume"]) < -1e-10):
        raise RuntimeError("Hypervolume unexpectedly decreased with fixed scaling and reference.")
    result["hypervolume_change"] = result["hypervolume"].diff()
    previous = result["hypervolume"].shift(1)
    result["hypervolume_change_pct"] = result["hypervolume_change"] / previous.where(previous > 0) * 100
    return result


def read_trials(path: Path, max_trials: int | None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"trial_number", "trial_state", "feasible", *OBJECTIVES}
    if required - set(frame):
        raise ValueError(f"{path}: missing columns {sorted(required - set(frame))}")
    numbers = pd.to_numeric(frame["trial_number"], errors="raise").to_numpy(float)
    if not np.isfinite(numbers).all() or np.any(numbers < 0) or np.any(numbers != np.floor(numbers)):
        raise ValueError(f"{path}: invalid trial numbers.")
    if frame["trial_number"].duplicated().any() or frame.empty:
        raise ValueError(f"{path}: expected nonempty trials with unique numbers.")
    frame["trial_number"] = numbers.astype(int)
    frame = frame.sort_values("trial_number")
    available = int(frame["trial_number"].max()) + 1
    limit = max_trials if max_trials is not None else available
    if limit > available:
        raise ValueError(f"{path}: requested {limit} attempts, but only {available} are exported.")
    frame = frame.loc[frame["trial_number"] < limit].reset_index(drop=True)
    if not np.array_equal(frame["trial_number"].to_numpy(), np.arange(limit)):
        raise ValueError(f"{path}: missing trial numbers within the requested history.")
    completed = frame["trial_state"].eq("COMPLETE").to_numpy()
    eligible = completed & frame["feasible"].astype(str).str.lower().eq("true").to_numpy()
    if not eligible.any():
        raise ValueError(f"{path}: no feasible COMPLETE trials in the selected range.")
    return frame, eligible, completed


def draw_history(ax: plt.Axes, history: pd.DataFrame, scenario: str, step: int) -> None:
    x = history["trial_count"].to_numpy()
    y = history["hypervolume"].to_numpy()
    color = COLORS.get(scenario, "#0072b2")
    ax.plot(x, y, color=color, linewidth=1.7)
    for count in (3000, 6000, 9000, 12000):
        if count <= x[-1]:
            ax.axvline(count, color="#aaaaaa", linestyle=":", linewidth=0.7)
    ax.scatter(x[-1], y[-1], color=color, s=25, zorder=3)
    ax.text(0.98, 0.04, f"Final HV = {y[-1]:.6f}", transform=ax.transAxes,
            ha="right", fontsize=9, color=color)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Cumulative attempted trials (including infeasible trials)")
    ax.set_ylabel("Hypervolume (higher is better)")
    ax.set_title(f"{SCENARIO_LABELS.get(scenario, scenario)} — sampled every {step} trials")
    ax.ticklabel_format(axis="x", style="plain")
    ax.grid(alpha=0.2)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIO_LABELS))
    parser.add_argument("--max-trials", type=int, help="Cutoff in total attempts, not just feasible trials.")
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--reference-value", type=float, default=None, help="Default: 1.1 on each normalized loss axis.")
    parser.add_argument("--normalization-file", type=Path, help="Reuse a saved normalization.json for extended histories.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.step < 1 or (args.max_trials is not None and args.max_trials < 1):
        parser.error("--step and --max-trials must be positive.")
    if args.reference_value is not None and (not np.isfinite(args.reference_value) or args.reference_value <= 1):
        parser.error("--reference-value must be finite and greater than 1.")
    run = (args.run_dir or latest_run(Path("data"))).resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    scenarios = args.scenarios or config["scenarios"]
    if len(scenarios) != len(set(scenarios)) or set(scenarios) - set(config["scenarios"]):
        parser.error("Selected scenarios must be unique and included in this run.")
    if config["objectives"] != OBJECTIVES or config["directions"] != ["minimize"] * 3 + ["maximize"] * 2:
        raise ValueError("Run objectives or directions differ from this script's five-objective definition.")
    loaded = None
    if args.normalization_file:
        loaded = json.loads(args.normalization_file.read_text(encoding="utf-8"))
        if (loaded.get("schema_version") != SCALING_SCHEMA or loaded.get("objectives") != OBJECTIVES or
                loaded.get("model_configuration") != config.get("model_configuration")):
            raise ValueError("Saved normalization has incompatible objectives or model configuration.")

    histories = {}
    scales = {}
    for scenario in scenarios:
        frame, eligible, completed = read_trials(run / f"trials_{scenario}.csv", args.max_trials)
        ratios = frame[OBJECTIVES].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(ratios[eligible]).all():
            raise ValueError(f"{scenario}: feasible trials have non-finite objective values.")
        if loaded is not None:
            scaling = loaded["scenarios"][scenario]
            validate_scaling(scaling)
            if args.reference_value is not None and not np.allclose(scaling["reference_point"], args.reference_value):
                raise ValueError("Requested reference point differs from the saved normalization.")
        else:
            scaling = make_scaling(ratios[eligible], args.reference_value or 1.1)
            scaling["source_trial_count"] = len(frame)
            scaling["source_feasible_count"] = int(eligible.sum())
        normalized = (ratios * MINIMIZATION_SIGNS - scaling["ideal_losses"]) / scaling["scales"]
        history = history_from_losses(normalized, eligible, completed,
                                      np.asarray(scaling["reference_point"]), args.step)
        history.insert(0, "scenario", scenario)
        histories[scenario] = history
        scales[scenario] = scaling
        print(f"{scenario}: {len(frame)} attempts, {int(eligible.sum())} feasible; "
              f"final HV {history.iloc[-1]['hypervolume']:.8f}", flush=True)

    output = (args.output_dir or Path("figures") / run.name / "hypervolume").resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(histories.values(), ignore_index=True)
    combined.to_csv(output / "hypervolume_history.csv", index=False, encoding="utf-8-sig")
    checkpoints = combined.loc[combined["trial_count"].mod(3000).eq(0) |
                               combined.groupby("scenario")["trial_count"].transform("max").eq(combined["trial_count"])].copy()
    previous_checkpoint = checkpoints.groupby("scenario")["hypervolume"].shift(1)
    checkpoints["checkpoint_change_pct"] = (
        checkpoints["hypervolume"] / previous_checkpoint.where(previous_checkpoint > 0) - 1) * 100
    checkpoints.to_csv(output / "hypervolume_checkpoints.csv", index=False, encoding="utf-8-sig")
    normalization = {"schema_version": SCALING_SCHEMA, "objectives": OBJECTIVES,
                     "model_configuration": config.get("model_configuration"),
                     "scenarios": scales, "scaling_source_run":
                     loaded.get("scaling_source_run") if loaded is not None else str(run),
                     "scaling_source_max_trials":
                     loaded.get("scaling_source_max_trials") if loaded is not None else args.max_trials}
    (output / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")
    for scenario, history in histories.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        draw_history(ax, history, scenario, args.step)
        fig.text(0.5, 0.01, "Fixed per-scenario scaling and reference; objective vectors from feasible COMPLETE trials.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        save_figure(fig, output, f"hypervolume_{scenario}")
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(10, 3.6 * len(scenarios)), squeeze=False)
    for ax, scenario in zip(axes.flat, scenarios):
        draw_history(ax, histories[scenario], scenario, args.step)
    fig.suptitle("Hypervolume histories — fixed scaling within each climate scenario")
    fig.text(0.5, 0.006, "Scenario scales differ: compare search progress, not climate outcomes or robustness.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_figure(fig, output, "hypervolume_all_scenarios")
    metadata = {"source_run": str(run), "scenarios": scenarios, "sampling_step": args.step,
                "max_trials_requested": args.max_trials, "normalization_file_reused":
                str(args.normalization_file.resolve()) if args.normalization_file else None,
                "optuna_version": optuna.__version__, "numpy_version": np.__version__,
                "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
                "notes": ["Trial-count axis includes every attempt; only feasible COMPLETE trials enter the front.",
                          "Trial-number order is used, not wall-clock completion order in the parallel run.",
                          "Zero, the final attempt and every 3000-attempt checkpoint are always included.",
                          "Minimization losses: first three ratios unchanged, crop and biodiversity ratios negated.",
                          "New scaling: observed feasible ideal -> 0, no-policy ratio -> 1, fixed throughout the history.",
                          "No observed improvement: use scale=1 and record the affected objective explicitly.",
                          "Vectors outside the reference box contribute zero and are excluded from HV computation.",
                          "Distinct objective vectors are counted; duplicate policies need not have distinct outcomes.",
                          "For extended histories, reuse normalization.json to retain the original scale and reference."]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved hypervolume plots and CSVs: {output}")


if __name__ == "__main__":
    main()
