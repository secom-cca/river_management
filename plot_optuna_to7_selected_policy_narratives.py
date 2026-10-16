"""Plot budget allocation, spending timing and schedules for five robust candidates.

Only saved decisions and original-order case outputs are read; no SD runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


POLICIES = [
    ("2C_trial_00008685", "平均後悔値：58条件で最良"),
    ("2C_trial_00010367", "平均後悔値：標準設定で最良"),
    ("2C_trial_00005623", "最大後悔値：7条件で単独最良"),
    ("2C_trial_00009646", "最大後悔値：2条件で単独最良"),
    ("4C_trial_00010656", "元順序の平均最良・今回平均3位"),
]
COMPONENTS = ["drainage", "dam", "levee", "elevation", "migration", "paddy_dam", "breeding", "forest"]
LABELS = ["排水施設", "ダム", "堤防", "住宅かさ上げ", "住宅移転", "田んぼダム", "品種改良", "森林管理"]
COLORS = ["#56b4e9", "#0072b2", "#332288", "#f4a582", "#d55e00", "#e69f00", "#cc79a7", "#009e73"]
PREFIXES = dict(zip(COMPONENTS, ["drainage", "dam", "levee", "house_elevation", "migration",
                               "paddy_dam", "breeding", "forest_management"]))
YEARS = list(range(2009, 2024))
PERIODS = ["初期：2009–2013", "中期：2014–2018", "後期：2019–2023"]
PERIOD_COLORS = ["#6baed6", "#bdbdbd", "#9e9ac8"]
UNIT_HOUSING_COST = 3_000_000  # Current to7 model: yen per elevated/relocated house.


def configure_fonts() -> None:
    paths = [Path("C:/Windows/Fonts/meiryo.ttc"), Path("/mnt/c/Windows/Fonts/meiryo.ttc")]
    for path in paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=str(path)).get_name()
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    plt.rcParams.update({"font.size": 10, "axes.unicode_minus": False, "pdf.fonttype": 42})


def save(fig: plt.Figure, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_inputs(ensemble: Path):
    labels = pd.read_csv(ensemble / "policy_labels.csv").set_index("solution_id")
    decisions = pd.read_csv(ensemble / "selected_decisions.csv").set_index("solution_id")
    config = json.loads((ensemble / "config.json").read_text(encoding="utf-8"))
    budgets, annual, schedules, source_files = [], [], [], []
    for sid, reason in POLICIES:
        if sid not in decisions.index or sid not in labels.index:
            raise ValueError(f"The requested policy is not in this ensemble: {sid}")
        decision, policy = decisions.loc[sid], labels.loc[sid]
        label, scenario = policy["plot_label"], decision["scenario"]
        path = ensemble / "cases" / f"original__{scenario}__{sid}.json"
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["key"] != ["original", scenario, sid]:
            raise ValueError(f"Case identity mismatch: {path}")
        source_files.append(path)
        if len(case["yearly"]) != 15 or sorted(int(v["year"]) for v in case["yearly"]) != YEARS:
            raise ValueError("Each policy must have 15 distinct annual outputs.")
        windows = {}
        for component in COMPONENTS:
            prefix = PREFIXES[component]
            start = int(float(decision[f"{prefix}_start_calendar_year"]))
            end = start if component in ("drainage", "dam", "levee") else int(float(decision[f"{prefix}_end_calendar_year"]))
            if not 2009 <= start <= end <= 2023:
                raise ValueError("A decision window is outside the evaluation horizon.")
            windows[component] = (start, end)
        records = []
        for row in sorted(case["yearly"], key=lambda v: v["year"]):
            year = int(row["year"])
            costs = {}
            for component in ("drainage", "dam", "levee"):
                costs[component] = float(decision[f"{component}_investment_amount"]) if year == windows[component][0] else 0.
            for component in ("elevation", "migration"):
                prefix = PREFIXES[component]
                start, end = windows[component]
                costs[component] = float(decision[f"{prefix}_per_year"]) * UNIT_HOUSING_COST if start <= year <= end else 0.
            costs.update({"paddy_dam": float(row["paddy_dam_actual_cost"]),
                          "breeding": float(row["breeding_actual_cost"]),
                          "forest": float(row["additional_forest_management_cost"])})
            if not np.isfinite(list(costs.values())).all() or min(costs.values()) < 0:
                raise ValueError("Invalid saved policy costs.")
            reconstructed = sum(costs.values())
            if not np.isclose(reconstructed, row["policy_additional_cost"], rtol=1e-10, atol=1e-4):
                raise ValueError(f"Annual cost decomposition mismatch: {sid}, {year}")
            record = {"solution_id": sid, "plot_label": label, "scenario": scenario,
                      "year": year, "period": PERIODS[(year - 2009) // 5],
                      **{f"cost_{k}_yen": costs[k] for k in COMPONENTS}, "total_cost_yen": reconstructed}
            records.append(record)
        annual.extend(records)
        total = sum(v["total_cost_yen"] for v in records)
        if total <= 0 or not np.isclose(total, case["metrics"]["policy_additional_cost"], rtol=1e-10, atol=1e-4):
            raise ValueError("Total saved cost does not match its annual decomposition.")
        if not np.isclose(total, float(policy["policy_additional_cost"]), rtol=1e-10, atol=1e-4):
            raise ValueError("Costs differ between the ensemble control and the source optimization.")
        budget = {"solution_id": sid, "plot_label": label, "scenario": scenario, "selection_reason": reason,
                  "total_cost_yen": total, "total_budget_utilisation": total / config["total_budget"],
                  "max_annual_cost_yen": max(v["total_cost_yen"] for v in records)}
        for component in COMPONENTS:
            cost = sum(v[f"cost_{component}_yen"] for v in records)
            budget[f"cost_{component}_yen"] = cost
            budget[f"share_{component}_pct"] = cost / total * 100
            start, end = windows[component]
            paid_years = [v["year"] for v in records if v[f"cost_{component}_yen"] > 1e-4]
            prefix = PREFIXES[component]
            if component in ("drainage", "dam", "levee"):
                setting, unit = float(decision[f"{component}_investment_amount"]), "円（一括投資）"
            elif component in ("elevation", "migration"):
                setting, unit = float(decision[f"{prefix}_per_year"]), "戸／年（設定値）"
            elif component == "paddy_dam":
                setting, unit = float(decision["annual_paddy_dam_investment"]), "円／年（設定値）"
            elif component == "forest":
                setting, unit = float(decision["annual_forest_management_conversion_area"]), "ha／年（新規整備の設定値）"
            else:
                setting, unit = cost / (end - start + 1), "円／年（実支出の平均）"
            schedules.append({"solution_id": sid, "plot_label": label, "scenario": scenario,
                              "component": component, "component_label": LABELS[COMPONENTS.index(component)],
                              "planned_start_year": start, "planned_end_year": end,
                              "planned_duration_years": end - start + 1,
                              "first_spending_year": min(paid_years) if paid_years else np.nan,
                              "last_spending_year": max(paid_years) if paid_years else np.nan,
                              "years_with_spending": len(paid_years), "annual_or_oneoff_setting": setting,
                              "setting_unit": unit, "total_cost_yen": cost, "share_of_cost_pct": cost / total * 100,
                              "period_definition": "新規整備期間（管理費は終了後も継続）" if component == "forest"
                              else "一括投資年（工事期間ではない）" if component in ("drainage", "dam", "levee")
                              else "固定された実施・投資期間"})
        for j, name in enumerate(("early", "middle", "late")):
            budget[f"spending_share_{name}_pct"] = sum(v["total_cost_yen"] for v in records if (v["year"] - 2009) // 5 == j) / total * 100
        budgets.append(budget)
    return pd.DataFrame(budgets), pd.DataFrame(annual), pd.DataFrame(schedules), config, source_files


def budget_plot(budgets: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1, 1.1]})
    y = np.arange(len(budgets))
    left = np.zeros(len(budgets))
    for component, name, color in zip(COMPONENTS, LABELS, COLORS):
        v = budgets[f"share_{component}_pct"].to_numpy()
        axes[0].barh(y, v, left=left, color=color, label=name, height=.65)
        for i, width in enumerate(v):
            if width >= 7:
                axes[0].text(left[i] + width / 2, i, f"{width:.1f}%", ha="center", va="center", fontsize=10)
        left += v
    for i, row in enumerate(budgets.itertuples()):
        axes[0].text(101.5, i, f"{row.total_cost_yen / 1e8:.1f}億円\n予算使用率 {row.total_budget_utilisation:.1%}", va="center", fontsize=9)
    axes[0].set_yticks(y, budgets["plot_label"])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 120)
    axes[0].set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_xlabel("各政策の追加費用総額に対する配分（%）")
    axes[0].set_title("① 何に予算を割くか", loc="left", fontsize=15)
    axes[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.04), frameon=False, fontsize=9)
    matrix = budgets[[f"share_{k}_pct" for k in COMPONENTS]].to_numpy()
    axes[1].imshow(matrix, cmap="Blues", vmin=0, vmax=65, aspect="auto")
    axes[1].set_xticks(range(8), LABELS)
    axes[1].set_yticks(y, budgets["plot_label"])
    for i in range(len(budgets)):
        for j in range(8):
            axes[1].text(j, i, f"{matrix[i, j]:.2f}%", ha="center", va="center",
                         color="white" if matrix[i, j] > 35 else "black", fontsize=10)
    axes[1].set_title("小規模な対策も含めた配分の内訳", loc="left", pad=12)
    fig.suptitle("5政策の予算配分：2009–2023年の追加政策費用", fontsize=16, y=1.02)
    fig.text(.5, .008, "元の気象順序・各政策の設計気候の保存済み費用を使用。森林は新規に管理した人工林の追加管理費。\n"
             "配分率は総予算上限に対する割合ではなく、その政策の実支出総額に対する割合。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .97), h_pad=2.3)
    save(fig, output, "01_budget_allocation")


def spending_plot(budgets: pd.DataFrame, annual: pd.DataFrame, config: dict, output: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    for ax, row in zip(axes.flat, budgets.itertuples()):
        selected = annual.loc[annual["solution_id"].eq(row.solution_id)].sort_values("year")
        bottom = np.zeros(15)
        for component, name, color in zip(COMPONENTS, LABELS, COLORS):
            value = selected[f"cost_{component}_yen"].to_numpy() / 1e8
            ax.bar(YEARS, value, bottom=bottom, color=color, width=.85)
            bottom += value
        ax.axhline(config["annual_budget"] / 1e8, color="#555555", linestyle="--", linewidth=.8)
        ax.set_ylim(0, config["annual_budget"] / 1e8 * 1.08)
        ax.set_xlim(2008.4, 2023.6)
        ax.set_xticks([2009, 2012, 2015, 2018, 2021, 2023])
        ax.set_ylabel("年次追加費用（億円）")
        ax.set_title(f"{row.plot_label}：{row.selection_reason}", fontsize=10)
        ax.grid(axis="y", alpha=.2)
    ax = axes.flat[-1]
    left = np.zeros(5)
    for name, label, color in zip(("early", "middle", "late"), PERIODS, PERIOD_COLORS):
        values = budgets[f"spending_share_{name}_pct"].to_numpy()
        ax.barh(range(5), values, left=left, color=color, label=label)
        for i, v in enumerate(values):
            ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=9)
        left += values
    ax.set_yticks(range(5), budgets["plot_label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("実支出の初期・中期・後期への配分（%）")
    ax.set_title("支出時期の比較", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=3, fontsize=8, frameon=False)
    fig.suptitle("② 何を先行させ、何を後半に実施するか：実際の年次支出", fontsize=16)
    fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(COLORS, LABELS)],
               loc="lower center", ncol=4, bbox_to_anchor=(.5, .04), frameon=False, fontsize=10)
    fig.text(.5, .006, "破線：年間予算上限。森林の支出は新規整備が終わった後も継続する管理費を含む。\n"
             "開始年は固定された計画であり、災害発生に応じて投資を開始する政策ではない。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .95), h_pad=2.3)
    save(fig, output, "02_annual_spending_timing")


def schedule_plot(budgets: pd.DataFrame, schedules: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(14, 17), sharex=True)
    for ax, policy in zip(axes, budgets.itertuples()):
        for j in range(3):
            ax.axvspan(2009 + 5*j, 2014 + 5*j, color=PERIOD_COLORS[j], alpha=.10)
        actions = schedules.loc[schedules["solution_id"].eq(policy.solution_id)].set_index("component")
        for j, component in enumerate(COMPONENTS):
            action = actions.loc[component]
            start, end, duration = int(action["planned_start_year"]), int(action["planned_end_year"]), int(action["planned_duration_years"])
            if action["total_cost_yen"] <= 0:
                continue
            ax.barh(j, duration, left=start, height=.62, color=COLORS[j], edgecolor="white")
            text = (f"{start}（一括）" if component in ("drainage", "dam", "levee")
                    else f"{start}–{end}（{duration}年）" if duration > 1 else f"{start}（1年）")
            if duration >= 4:
                ax.text(start + duration/2, j, text, ha="center", va="center", fontsize=9,
                        color="white" if component in ("migration", "forest") else "black")
            else:
                x = end + 1.12 if end < 2021 else start - .12
                ax.text(x, j, text, ha="left" if end < 2021 else "right", va="center", fontsize=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .5})
            if component == "forest" and action["last_spending_year"] > end:
                tail = int(action["last_spending_year"]) - end
                ax.barh(j, tail, left=end+1, height=.62, facecolor="none", edgecolor=COLORS[j], hatch="///", linewidth=.8)
        ax.set_yticks(range(8), LABELS[:-1] + ["森林（新規整備）"], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"{policy.plot_label} | {policy.solution_id}", loc="left", fontsize=11)
        ax.set_xlim(2008.8, 2024.3)
        ax.set_xticks(YEARS)
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("③ どの対策を長く継続するか：固定された実施・投資期間", fontsize=16)
    fig.legend(handles=[Patch(facecolor="#777777", label="塗りつぶし：実施・投資期間"),
                        Patch(facecolor="none", edgecolor=COLORS[-1], hatch="///", label="斜線：森林の追加管理費のみ継続")],
               loc="lower center", ncol=2, bbox_to_anchor=(.5, .035), frameon=False, fontsize=10)
    fig.text(.5, .006, "森林の塗りつぶしは未管理人工林を新規に整備する期間。終了後も管理費は発生する。\n"
             "ダム・堤防・排水施設の1年幅はモデル上の一括投資年であり、実際の工事期間ではない。\n"
             "棒は対策効果の持続期間を表さない。", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .073, 1, .965), h_pad=1.5)
    save(fig, output, "03_implementation_duration")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    ensemble = args.ensemble_dir.resolve()
    output = (args.output_dir or ensemble / "selected_policy_narratives").resolve()
    configure_fonts()
    budgets, annual, schedules, config, case_files = load_inputs(ensemble)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("policy_budget_shares", budgets), ("policy_annual_spending", annual),
                        ("policy_action_schedule", schedules)):
        frame.to_csv(output / f"{name}.csv", index=False, encoding="utf-8-sig")
    budget_plot(budgets, output)
    spending_plot(budgets, annual, config, output)
    schedule_plot(budgets, schedules, output)
    inputs = [ensemble / name for name in ("policy_labels.csv", "selected_decisions.csv", "config.json")] + case_files
    metadata = {"ensemble_dir": str(ensemble), "policy_ids": [sid for sid, _ in POLICIES],
                "policy_labels": budgets["plot_label"].tolist(), "source_cases": "Original weather order, each policy's design climate",
                "years": YEARS, "periods": PERIODS, "unit_housing_cost_yen": UNIT_HOUSING_COST,
                "annual_budget_yen": config["annual_budget"], "total_budget_yen": config["total_budget"],
                "annual_cost_checks": len(annual), "total_cost_checks": len(budgets),
                "input_sha256": {str(f.relative_to(ensemble)): hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},
                "notes": ["Only additional policy costs are included, not baseline expenditure or total social costs.",
                          "Forest conversion periods and continuing forest management expenditure are distinct.",
                          "Schedules are fixed calendar plans, not disaster-triggered actions.",
                          "One-off infrastructure investment years are not construction durations.",
                          "Policy content does not prove the causal contribution of each intervention to performance."]}
    (output / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {len(annual)} annual and {len(budgets)} total cost decompositions. Saved plots: {output}")


if __name__ == "__main__":
    main()
