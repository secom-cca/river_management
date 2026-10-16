"""Original-weather comparison: existing facilities plus 50bn yen over 10 years.

Each infrastructure receives 5bn yen in each calendar year 2009–2018.
Stocks and construction-delay pipelines continue through 2023. This is an
explicit multi-year diagnostic policy, outside the optimizer's one-time
decision representation. Model equations and unit costs are unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnose_optuna_to7_infrastructure import DAILY
from evaluate_optuna_to7_cross_scenario import RAW_METRICS, SCENARIO_ORDER
from evaluate_optuna_to7_weather_ensemble import atomic_json, read_case, signature
from plot_optuna_to7_selected_policy_narratives import configure_fonts, save
from weather_ensemble_to7 import weather_override

ACTIONS = ('no_policy','levee','drainage','dam')
LABELS = {'no_policy':'追加対策なし','levee':'堤防・河道改修のみ','drainage':'排水のみ','dam':'ダムのみ'}
COLORS = {'levee':'#0072b2','drainage':'#e69f00','dam':'#009e73'}
ANNUAL_INVESTMENT = 5e9
INVESTMENT_YEARS = tuple(range(2009,2019))
METRICS = ['flood_damage','innundation_damage','landslide_risk','crop_revenue','biodiversity']
SIGNS = [1,1,1,-1,-1]
_SOURCE = _OUTPUT = None


def scheduled_cost(action, year):
    if action not in ACTIONS: raise ValueError('Unknown action')
    return ANNUAL_INVESTMENT if action != 'no_policy' and year in INVESTMENT_YEARS else 0.


def worker_init(source, output):
    global _SOURCE, _OUTPUT
    _SOURCE = pd.read_csv(source,index_col='date',parse_dates=['date'],float_precision='round_trip')
    _OUTPUT = Path(output)


def run_scheduled(scenario, action):
    import run_vensim_with_pysd_to7_opt as core
    from run_vensim_with_pysd_to7_optuna import _zero_decision
    model = core.get_model()
    zero = _zero_decision()
    results = []
    investments = [f'{a}_investment' for a in ACTIONS[1:]]
    columns = list(dict.fromkeys([*core.RETURN_COLS,*DAILY,*investments]))
    with weather_override(model,_SOURCE):
        for yi in range(core.CALENDAR_NUM_YEARS):
            year = core.CALENDAR_START_YEAR+yi
            # Reissue the chosen amount over this calendar year's days;
            # all other policies remain zero, with persistent physical stocks.
            decision = zero.copy()
            if action != 'no_policy':
                decision[f'{action}_investment_amount'] = scheduled_cost(action,year)
                decision[f'{action}_start_year'] = yi
            params = core.fixed_params_for_scenario(scenario)
            params.update(core.policy_params_for_year(decision,yi))
            result = model.run(params=params,return_timestamps=core._year_timestamps(yi),
                return_columns=columns,final_time=core.YEAR_STARTS[yi]+core.YEAR_LENGTHS[yi],
                initial_condition='original' if yi==0 else 'current')
            result.attrs['year_end_state'] = {name:float(getattr(model.components,name)())
                for name in [*core.YEAR_END_COLUMNS,'drainage_capacity','discharge_allowance']}
            result.attrs['year_end_time'] = float(model.time())
            actual = {a:float(result[f'{a}_investment'].sum()) for a in ACTIONS[1:]}
            expected = {a:scheduled_cost(action,year) if a==action else 0. for a in ACTIONS[1:]}
            for a in ACTIONS[1:]:
                if not np.isclose(actual[a],expected[a],rtol=1e-12,atol=.001):
                    raise ValueError(f'Calendar expenditure mismatch: {action}/{scenario}/{year}/{a}')
            result.attrs['scheduled_infrastructure_costs'] = actual
            results.append(result)
    # Reuse the exact existing objective aggregation; replace the cost column
    # that otherwise describes a single original optimizer decision.
    metrics, yearly = core.aggregate_results(results,zero)
    for yi,result in enumerate(results):
        costs = result.attrs['scheduled_infrastructure_costs']
        for a,cost in costs.items(): yearly.loc[yi,f'{a}_actual_cost'] = cost
        yearly.loc[yi,'policy_additional_cost'] = sum(costs.values())
        yearly.loc[yi,'drainage_capacity_at_year_end_m3s'] = result.attrs['year_end_state']['drainage_capacity']
        yearly.loc[yi,'levee_peak_capacity_at_year_end_m3s'] = result.attrs['year_end_state']['discharge_allowance']*model.components.peaktomean_flow_ratio()/86400
        yearly.loc[yi,'maximum_flood_depth_m'] = result.flood_water_level.max()
        yearly.loc[yi,'maximum_inundation_depth_m'] = result.innundation_level.max()
        yearly.loc[yi,'flood_positive_damage_ratio_days'] = (result.flood_damage_ratio>0).sum()
        yearly.loc[yi,'inundation_positive_damage_ratio_days'] = (result.inundation_damage_ratio>0).sum()
    metrics['policy_additional_cost'] = float(yearly.policy_additional_cost.sum())
    metrics['max_annual_policy_additional_cost'] = float(yearly.policy_additional_cost.max())
    target = 0 if action=='no_policy' else ANNUAL_INVESTMENT*len(INVESTMENT_YEARS)
    if not np.isclose(metrics['policy_additional_cost'],target,rtol=1e-12,atol=.01): raise ValueError('Total expenditure mismatch')
    daily = pd.concat([r[DAILY+investments] for r in results])
    arrays = {n:daily[n].to_numpy(float) for n in daily.columns}; arrays['time'] = daily.index.to_numpy(float)
    for key,value in arrays.items():
        if not np.isfinite(value).all(): raise ValueError(f'Nonfinite daily {key}')
    # Existing assets must remain present at every policy's initial state.
    for col,nominal in [('drainage_capacity',50.),('dam_capacity',74200000.),('discharge_allowance',5200*86400/4)]:
        if not np.isclose(arrays[col][0],nominal,rtol=1e-12,atol=1e-6): raise ValueError('Existing initial capacity changed')
    return metrics,yearly,arrays


def worker(task):
    action,scenario,tag = task
    started=time.perf_counter()
    metrics,yearly,arrays = run_scheduled(scenario,action)
    cid = f'{tag}__{action}__{scenario}'
    with (_OUTPUT/'daily'/f'{cid}.npz').open('wb') as f: np.savez_compressed(f,**arrays)
    return {'case_id':cid,'action':action,'scenario':scenario,'tag':tag,'metrics':metrics,
            'yearly':yearly.to_dict('records'),'elapsed_seconds':time.perf_counter()-started}


def read_profile(out,cid):
    with np.load(out/'daily'/f'{cid}.npz') as z:return {k:z[k] for k in z.files}


def export(out,records):
    rows,annual,crossings = [],[],[]
    from run_vensim_with_pysd_to7_opt import CALENDAR_START_YEAR
    dates = pd.date_range(f'{CALENDAR_START_YEAR}-01-01','2023-12-31')
    for climate in SCENARIO_ORDER:
        base = records[f'comparison__no_policy__{climate}']
        b = read_profile(out,base['case_id'])
        for action in ACTIONS:
            r = records[f'comparison__{action}__{climate}'];d=read_profile(out,r['case_id'])
            row = {'case_id':r['case_id'],'action':action,'label':LABELS[action],'scenario':climate,**r['metrics']}
            row['budget_feasible'] = row['policy_additional_cost']<=50e9+.01 and row['max_annual_policy_additional_cost']<=5e9+.01
            for metric,sign in zip(METRICS,SIGNS):
                row[f'improvement_{metric}_pct'] = sign*(base['metrics'][metric]-r['metrics'][metric])/base['metrics'][metric]*100
            last=r['yearly'][-1]
            row['final_drainage_capacity_m3s']=last['drainage_capacity_at_year_end_m3s']
            row['final_levee_peak_capacity_m3s']=last['levee_peak_capacity_at_year_end_m3s']
            row['final_dam_capacity_m3']=last['dam_capacity_at_year_end']
            for col in ['flood_water_level','innundation_level']:
                row[f'max_abs_change_{col}_mm']=float(np.abs(d[col]-b[col]).max()*1000)
                row[f'changed_days_{col}']=int((np.abs(d[col]-b[col])>1e-12).sum())
            for depth,ratio in [('flood_water_level','flood_damage_ratio'),('innundation_level','inundation_damage_ratio')]:
                changed=np.flatnonzero(np.abs(d[ratio]-b[ratio])>1e-12)
                row[f'changed_days_{ratio}']=len(changed)
                for i in changed:
                    crossings.append({'case_id':r['case_id'],'action':action,'scenario':climate,'hazard':ratio,
                        'date':dates[int(d['time'][i])].date().isoformat(),'baseline_depth_m':b[depth][i],
                        'policy_depth_m':d[depth][i],'baseline_damage_ratio':b[ratio][i],'policy_damage_ratio':d[ratio][i]})
            row['sum_flood_inundation_damage_yen']=r['metrics']['flood_damage']+r['metrics']['innundation_damage']
            row['improvement_sum_flood_inundation_pct']=(1-row['sum_flood_inundation_damage_yen']/(base['metrics']['flood_damage']+base['metrics']['innundation_damage']))*100
            rows.append(row)
            annual.extend([{'case_id':r['case_id'],'action':action,'scenario':climate,**y} for y in r['yearly']])
    frame=pd.DataFrame(rows);years=pd.DataFrame(annual)
    frame.to_csv(out/'comparison_summary.csv',index=False,encoding='utf-8-sig')
    years.to_csv(out/'annual_results.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(crossings,columns=['case_id','action','scenario','hazard','date','baseline_depth_m','policy_depth_m','baseline_damage_ratio','policy_damage_ratio']).to_csv(out/'damage_threshold_changes.csv',index=False,encoding='utf-8-sig')
    if not frame.budget_feasible.all():raise ValueError('Budget exceeded')
    # Capacity development is investment driven and must match across climates.
    for action in ACTIONS:
        y=years.loc[years.action.eq(action)]
        for column in ['drainage_capacity_at_year_end_m3s','levee_peak_capacity_at_year_end_m3s','dam_capacity_at_year_end']:
            wide=y.pivot(index='year',columns='scenario',values=column)
            np.testing.assert_allclose(wide['present'],wide['2C'],rtol=1e-12,atol=1e-6)
            np.testing.assert_allclose(wide['present'],wide['4C'],rtol=1e-12,atol=1e-6)
    drainage=years.loc[years.action.eq('drainage') & years.scenario.eq('present')]
    np.testing.assert_allclose(drainage.drainage_capacity_at_year_end_m3s,50+50*np.minimum(np.arange(1,16),10),rtol=1e-12,atol=1e-8)
    configure_fonts()
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for j,climate in enumerate(SCENARIO_ORDER):
        part=frame.loc[frame.scenario.eq(climate)]
        for i,metric in enumerate(['flood_damage','innundation_damage']):
            ax=axes[i,j]
            ax.bar(range(4),part[f'improvement_{metric}_pct'],color=['#aaaaaa',*[COLORS[a] for a in ACTIONS[1:]]])
            ax.set_xticks(range(4),['追加なし','堤防','排水','ダム']);ax.axhline(0,color='black',lw=.7)
            ax.set_title(f'{climate}・{"洪水" if i==0 else "内水"}');ax.set_ylabel('追加なし比の被害改善率（%）');ax.grid(axis='y',alpha=.2)
            for x,v in enumerate(part[f'improvement_{metric}_pct']): ax.annotate(f'{v:.2f}',(x,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9,xytext=(0,3 if v>=0 else -3),textcoords='offset points')
            values=part[f'improvement_{metric}_pct']
            low,high=min(0.,float(values.min())),max(0.,float(values.max()))
            span=max(high-low,.5)
            ax.set_ylim(low-.15*span,high+.2*span)
    fig.suptitle('既存施設を残した継続投資：年間50億円×10年、2009–2023年の累積被害',fontsize=14)
    fig.tight_layout(rect=(0,.07,1,.94));fig.text(.5,.01,'各対策だけに2009–2018年の総額500億円を投入。2019年以降の新規支出は0。\n単価・既存能力・建設遅延・被害関数は現モデルのまま。各パネルの縦軸目盛に注意。',ha='center',fontsize=9)
    save(fig,out,'01_damage_improvements')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    configs=[('levee','levee_peak_capacity_at_year_end_m3s',5200,1000,1,'ピーク流下能力（m³/s）'),
             ('drainage','drainage_capacity_at_year_end_m3s',50,500,1,'排水能力（m³/s）'),
             ('dam','dam_capacity_at_year_end',74200000,10000000,1e6,'ダム容量（百万m³）')]
    capacity=[]
    for ax,(action,col,initial,planned,scale,ylabel) in zip(axes,configs):
        y=years.loc[years.action.eq(action)&years.scenario.eq('present')].sort_values('year')
        ax.plot([2008,*y.year],[initial/scale,*(y[col]/scale)],marker='o',color=COLORS[action],label='実現能力（年末）')
        ax.axhline(initial/scale,color='gray',ls=':',label='既存能力')
        ax.axhline((initial+planned)/scale,color='black',ls='--',label='全投資の効果実現後')
        ax.axvspan(2008.5,2018.5,color=COLORS[action],alpha=.08,label='新規投資期間')
        ax.set_ylabel(ylabel);ax.set_xlabel('年（2008位置は投資開始前）');ax.set_title(LABELS[action]);ax.grid(alpha=.2);ax.legend(fontsize=8)
        realized=float(y[col].iloc[-1]-initial)
        if not -1e-6<=realized<=planned+1e-5:raise ValueError('Invalid final capacity increase')
        capacity.append({'action':action,'initial_capacity':initial,'planned_additional_capacity':planned,
            'realized_additional_capacity_at_end_2023':realized,'realized_fraction_at_end_2023':realized/planned,
            'unit':'m3' if action=='dam' else 'm3/s','plot_axis_label':ylabel,'scale_for_plot':scale})
    pd.DataFrame(capacity).to_csv(out/'capacity_realization.csv',index=False,encoding='utf-8-sig')
    fig.suptitle('10年間の投資と、その後の建設遅延による能力の実現',fontsize=14)
    fig.tight_layout(rect=(0,.05,1,.92));fig.text(.5,.01,'単価一定のモデル上の計画量。3気候で能力推移が一致することを検証し、presentを表示。',ha='center',fontsize=9)
    save(fig,out,'02_capacity_realization')
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    ensemble=args.ensemble_dir.resolve();out=(args.output_dir or ensemble/'gray_infrastructure_10years_original').resolve()
    if out==ensemble:raise ValueError('Use a separate output directory')
    source=json.loads((ensemble/'config.json').read_text(encoding='utf-8'))
    if signature()!=source['signature']:raise ValueError('Model/weather signature changed')
    hashes={n:hashlib.sha256((ensemble/n).read_bytes()).hexdigest() for n in ['config.json','source_weather.csv','baselines.csv']}
    tasks=[(a,c,'comparison') for a in ACTIONS for c in SCENARIO_ORDER]
    tasks += [('no_policy',c,'restoration') for c in SCENARIO_ORDER]
    plan={'schema':1,'model_signature':signature(),'input_sha256':hashes,'tasks':tasks,
        'annual_investment_yen':ANNUAL_INVESTMENT,'investment_years':INVESTMENT_YEARS,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes':['Existing assets retained; original weather and present/2C/4C only.',
            'Explicit recurring policy outside one-time optimizer representation; no reoptimization.',
            'Unit costs, linear capacity conversion and construction delays unchanged.',
            'Actual daily spending replaces the original one-time decision cost aggregation.']}
    # Normalize tuples before round-trip comparison of cached metadata.
    plan=json.loads(json.dumps(plan))
    out.mkdir(parents=True,exist_ok=True)
    for name in ['cases','daily']:(out/name).mkdir(exist_ok=True)
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text(encoding='utf-8'))!=plan:raise ValueError('Code/inputs changed: choose a new output directory')
    else:atomic_json(out/'config.json',plan)
    records={}
    for action,scenario,tag in tasks:
        cid=f'{tag}__{action}__{scenario}';p=out/'cases'/f'{cid}.json'
        if p.exists() and (out/'daily'/f'{cid}.npz').exists():records[cid]=json.loads(p.read_text(encoding='utf-8'))
    pending=[t for t in tasks if f'{t[2]}__{t[0]}__{t[1]}' not in records]
    if args.analyze_only and pending:raise ValueError('Saved cases incomplete')
    state={'state':'running','comparison_cases':12,'restoration_cases':3,'completed_cases':len(records)}
    atomic_json(out/'run_status.json',state);started=time.perf_counter()
    if pending:
        with multiprocessing.get_context('spawn').Pool(args.workers,initializer=worker_init,
            initargs=(str(ensemble/'source_weather.csv'),str(out))) as pool:
            for r in pool.imap_unordered(worker,pending):
                records[r['case_id']]=r;atomic_json(out/'cases'/f"{r['case_id']}.json",r)
                state.update(completed_cases=len(records),elapsed_seconds=time.perf_counter()-started)
                atomic_json(out/'run_status.json',state)
                print(f"Recurring infrastructure: {len(records)}/{len(tasks)}; {state['elapsed_seconds']:.0f}s",flush=True)
    validation=[]
    for climate in SCENARIO_ORDER:
        original=read_case(ensemble,('original',climate,'no_policy'))
        for tag in ['comparison','restoration']:
            r=records[f'{tag}__no_policy__{climate}']
            for metric in RAW_METRICS:
                delta=r['metrics'][metric]-original['metrics'][metric]
                ok=bool(np.isclose(r['metrics'][metric],original['metrics'][metric],rtol=1e-9,atol=.01))
                validation.append({'case_id':r['case_id'],'metric':metric,'difference':delta,'passed':ok})
    pd.DataFrame(validation).to_csv(out/'identity_validation.csv',index=False,encoding='utf-8-sig')
    if not all(r['passed'] for r in validation):raise ValueError('Baseline identity failed')
    export(out,records)
    if signature()!=source['signature'] or any(hashlib.sha256((ensemble/n).read_bytes()).hexdigest()!=h for n,h in hashes.items()):raise ValueError('Source changed')
    state.update(state='complete',identity_checks=len(validation),elapsed_seconds=time.perf_counter()-started)
    atomic_json(out/'run_status.json',state);print(f'Saved: {out}',flush=True)


if __name__=='__main__':main()
