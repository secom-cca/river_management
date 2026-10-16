"""Explain saved original-weather infrastructure tests without rerunning SD."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_optuna_to7_infrastructure import profile
from plot_optuna_to7_selected_policy_narratives import configure_fonts, save
from weather_ensemble_to7 import DATES


def damage_ratio(depth):
    return np.select([depth < .1, depth < .3, depth < .5, depth < 1,
                      depth < 2, depth < 3], [0., .047, .189, .253, .406, .592], default=.8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.diagnostic_dir.resolve()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    if status["state"] != "complete": raise ValueError("Diagnostics incomplete")
    for name, expected in config["model_signature"]["model_configuration"]["source_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model changed: {name}")
    frame = pd.read_csv(out / "case_summary.csv")
    indexed = frame.set_index("case_id")
    mechanism, crossings, exposure = [], [], []
    capacity_columns = {"drainage": "drainage_capacity", "dam": "dam_capacity",
                        "levee": "discharge_allowance", "paddy_dam": "paddyfield_storage_capacity"}
    for task in config["tasks"]:
        if task["family"] not in ("investment", "conversion_delay", "ablation"): continue
        cid = task["case_id"]
        row = indexed.loc[cid]
        d = profile(out, cid)
        refid = task["baseline_case"] if task["family"] != "ablation" else task["comparison_case"]
        ref = profile(out, refid)
        # Confirm the plotted lookup matches actual saved SD outputs.
        for depth, ratio in (("flood_water_level", "flood_damage_ratio"), ("innundation_level", "inundation_damage_ratio")):
            if not np.array_equal(damage_ratio(d[depth]), d[ratio]):
                raise ValueError("Diagnostic lookup differs from model")
            indices = np.flatnonzero(d[ratio] != ref[ratio])
            for i in indices:
                crossings.append({"case_id": cid, "reference_case": refid, "hazard": ratio,
                    "date": DATES[int(d['time'][i])].date().isoformat(),
                    "reference_depth_m": ref[depth][i], "changed_depth_m": d[depth][i],
                    "reference_damage_ratio": ref[ratio][i], "changed_damage_ratio": d[ratio][i]})
        cap = capacity_columns[task["action"]]
        mechanism.append({"case_id": cid, "scenario": task["scenario"], "action": task["action"],
            "variant": task["variant"], "reference_case": refid,
            "capacity_column": cap, "final_reference_capacity": ref[cap][-1],
            "final_changed_capacity": d[cap][-1], "final_capacity_change": d[cap][-1] - ref[cap][-1],
            "max_abs_flood_depth_change_mm": np.abs(d['flood_water_level']-ref['flood_water_level']).max()*1000,
            "max_abs_inundation_depth_change_mm": np.abs(d['innundation_level']-ref['innundation_level']).max()*1000,
            "changed_flood_damage_ratio_days": int(np.sum(d['flood_damage_ratio'] != ref['flood_damage_ratio'])),
            "changed_inundation_damage_ratio_days": int(np.sum(d['inundation_damage_ratio'] != ref['inundation_damage_ratio'])),
            "delta_flood_damage_yen": row['flood_damage']-indexed.loc[refid, 'flood_damage'],
            "delta_inundation_damage_yen": row['innundation_damage']-indexed.loc[refid, 'innundation_damage']})
        if task["action"] == "paddy_dam":
            identical = all(np.array_equal(d[k],ref[k]) for k in
                            ('river_discharge_downstream', 'flood_water_level', 'flood_damage_ratio'))
            exposure.append({"case_id": cid, "reference_case": refid, "flood_hydraulics_and_damage_ratio_identical": identical,
                "max_abs_flood_event_healthy_households_change": np.abs(d['flood_event_healthy_households']-ref['flood_event_healthy_households']).max(),
                "delta_flood_damage_yen": row['flood_damage']-indexed.loc[refid,'flood_damage'],
                "delta_inundation_damage_yen": row['innundation_damage']-indexed.loc[refid,'innundation_damage'],
                "delta_sum_flood_inundation_damage_yen": (row['flood_damage']+row['innundation_damage'])-(indexed.loc[refid,'flood_damage']+indexed.loc[refid,'innundation_damage'])})
    pd.DataFrame(mechanism).to_csv(out/'investment_mechanism_trace.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(crossings).to_csv(out/'damage_threshold_crossings.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(exposure).to_csv(out/'paddy_shared_exposure_check.csv',index=False,encoding='utf-8-sig')
    sensitivity = []
    for row in frame.loc[frame.family.eq('capacity')].itertuples():
        ref = indexed.loc[row.comparison_case]
        sensitivity.append({'case_id':row.case_id, 'policy_label':row.policy_label, 'scenario':row.scenario,
            'action':row.action, 'capacity_factor_or_paddy_coverage':row.value,
            **{f'{m}_change_vs_nominal_percent':(getattr(row,m)/ref[m]-1)*100
               for m in ('flood_damage','innundation_damage')}})
    pd.DataFrame(sensitivity).to_csv(out/'capacity_sensitivity_vs_nominal.csv',index=False,encoding='utf-8-sig')

    configure_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    examples = [('4C','levee','flood_water_level','flood_damage_ratio','stress_levee_conversion10_invest'),
                ('present','drainage','innundation_level','inundation_damage_ratio','isolated_drainage_early')]
    point_rows = []
    for ax, (climate, action, depth, ratio, variant) in zip(axes, examples):
        ref = profile(out, f'reference__no_policy__{climate}')
        changed = profile(out, f'{variant}__no_policy__{climate}')
        candidates = np.flatnonzero(ref[ratio] != changed[ratio])
        i = candidates[np.argmax(ref[ratio][candidates]-changed[ratio][candidates])]
        samples = [('追加なし',ref,'#444444')]
        if action == 'levee': samples.append(('50億円・標準効率',profile(out,f'isolated_levee_early__no_policy__{climate}'),'#0072b2'))
        samples.append(('50億円・効率10倍（診断）' if action=='levee' else '50億円・2009年開始',changed,'#d55e00'))
        lo = max(0,min(a[depth][i] for _,a,_ in samples)-.02)
        hi = max(a[depth][i] for _,a,_ in samples)+.02
        x = np.linspace(lo,hi,2000)
        ax.plot(x*100,damage_ratio(x)*100,color='#aaaaaa',drawstyle='steps-post')
        for label,a,color in samples:
            ax.scatter(a[depth][i]*100,a[ratio][i]*100,color=color,s=65,label=label,zorder=3)
            point_rows.append({'scenario':climate,'date':DATES[int(ref['time'][i])].date().isoformat(),
                               'label':label,'depth_cm':a[depth][i]*100,'damage_ratio_percent':a[ratio][i]*100})
        for threshold in (.1,.3,.5,1.,2.,3.):
            if lo < threshold < hi: ax.axvline(threshold*100,color='gray',ls=':',lw=.8)
        ax.set_title(f"{climate}・{'洪水／堤防' if action=='levee' else '内水／排水'}：{DATES[int(ref['time'][i])].date()}")
        ax.set_xlabel('モデルの浸水深（cm）'); ax.set_ylabel('被害率（%）')
        ax.legend(fontsize=9); ax.grid(alpha=.2)
    fig.suptitle('水深が変わっても、区分境界を越えなければ被害率は同じ',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93))
    save(fig,out,'03_damage_threshold_examples')
    pd.DataFrame(point_rows).to_csv(out/'damage_threshold_examples.csv',index=False,encoding='utf-8-sig')
    print(f'Saved mechanism tables and threshold figure: {out}')


if __name__ == '__main__':
    main()
