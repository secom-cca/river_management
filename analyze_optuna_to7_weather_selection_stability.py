"""Selection stability and adverse weather-order diagnostics from saved cases.

No SD runs: common subsets of existing permutations, fixed Pareto scaling,
equal climate/objective weights, and separate original-order controls.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_optuna_to7_weather_ensemble import DIRECTION
from evaluate_optuna_to7_cross_scenario import OBJECTIVES, SCENARIO_ORDER
from plot_optuna_to7_selected_policy_narratives import POLICIES, configure_fonts, save
from weather_ensemble_to7 import permute_weather

METRIC_LABELS = ['洪水', '内水', '土砂災害', '作物収益', '生物多様性']
RAW = ['flood_damage', 'innundation_damage', 'landslide_risk', 'crop_revenue', 'biodiversity']
TIE_ATOL = 1e-12
COLORS = ['#0072b2', '#e69f00', '#009e73', '#cc79a7', '#666666']
INPUTS = ['config.json', 'ensemble_results.csv', 'ensemble_yearly.csv', 'policy_robustness.csv',
          'selected_policies.csv', 'selected_decisions.csv', 'policy_labels.csv',
          'regret_normalization.json', 'year_mapping.csv', 'source_weather.csv']


def fixed_regrets(frame, settings):
    """Apply saved scales; never fit ranges to an extracted subset."""
    values = np.empty((len(frame), 5))
    for scenario in SCENARIO_ORDER:
        mask = frame.target_scenario.eq(scenario).to_numpy()
        s = settings[scenario]
        loss = frame.loc[mask, list(OBJECTIVES)].to_numpy(float) * DIRECTION
        values[mask] = np.maximum((loss - np.array(s['ideal_losses'])) / np.array(s['scales']), 0.)
    if not np.isfinite(values).all(): raise ValueError('Undefined/nonfinite reference regret')
    return values


def selection_masks(scores, eligible, top_k=3):
    """Inclusive ties, consistent with existing absolute 1e-12 winner tolerance."""
    s = np.where(eligible, scores, np.inf)
    if eligible.sum() < top_k: raise ValueError('Too few feasible candidates')
    winner = eligible & np.isclose(s, s.min(), atol=TIE_ATOL, rtol=0)
    cutoff = np.sort(s[eligible])[top_k-1]
    top = eligible & (s <= cutoff + TIE_ATOL)
    ranks = np.where(eligible, 1 + (s[None, :] < s[:, None] - TIE_ATOL).sum(axis=1), np.nan)
    return winner, top, ranks


def resample_selection(scores, eligible, sizes, repetitions, seed):
    """One permutation-index subset shared by every candidate and climate."""
    rng = np.random.default_rng(seed)
    output, draws = [], []
    for size in sizes:
        if not 1 <= size <= scores.shape[1]: raise ValueError('Invalid subset size')
        count = 1 if size == scores.shape[1] else repetitions
        wins, tops, ranks = [], [], []
        for repetition in range(count):
            indices = np.arange(scores.shape[1]) if count == 1 and size == scores.shape[1] else np.sort(rng.choice(scores.shape[1],size,replace=False))
            w, t, r = selection_masks(scores[:, indices].mean(axis=1), eligible)
            wins.append(w); tops.append(t); ranks.append(r)
            draws.append({'sequence_count':size,'repetition':repetition,'sequence_indices':indices.tolist()})
        rank = np.array(ranks)
        for policy_index in range(scores.shape[0]):
            output.append({'policy_index':policy_index,'sequence_count':size,'draw_count':count,
                'eligible':bool(eligible[policy_index]),
                'best_frequency':float(np.mean(np.array(wins)[:,policy_index])),
                'top3_frequency':float(np.mean(np.array(tops)[:,policy_index])),
                'rank_median':float(np.nanmedian(rank[:,policy_index])) if eligible[policy_index] else np.nan,
                'rank_p10':float(np.nanquantile(rank[:,policy_index],.1)) if eligible[policy_index] else np.nan,
                'rank_p90':float(np.nanquantile(rank[:,policy_index],.9)) if eligible[policy_index] else np.nan})
    return pd.DataFrame(output), draws


def csv(frame, out, name):
    frame.to_csv(out / name,index=False,encoding='utf-8-sig')


def diagnose_years(ensemble, out, weak, labels, decisions):
    # All information below comes from existing annual outputs / weather snapshot.
    yearly = pd.read_csv(ensemble/'ensemble_yearly.csv')
    mapping = pd.read_csv(ensemble/'year_mapping.csv')
    weather = pd.read_csv(ensemble/'source_weather.csv',index_col='date',parse_dates=['date'],float_precision='round_trip')
    keys = weak[['source_solution_id','sequence_id','target_scenario']].drop_duplicates()
    original_keys = keys[['source_solution_id','target_scenario']].drop_duplicates().assign(sequence_id='original')
    selected_keys = pd.concat([keys,original_keys],ignore_index=True).drop_duplicates()
    annual = yearly.merge(selected_keys,on=['source_solution_id','sequence_id','target_scenario'],how='inner',validate='many_to_one')
    if len(annual) != len(selected_keys)*15: raise ValueError('Missing weak-case annual rows')
    weather_rows = []
    for seq in selected_keys.sequence_id.unique():
        order = mapping.loc[mapping.sequence_id.eq(seq)].sort_values('target_year').source_year.astype(int).tolist()
        moved, _ = permute_weather(weather,order)
        for year, part in moved.groupby(moved.index.year):
            warm = part.loc[part.index.month.isin([7,8,9])]
            weather_rows.append({'sequence_id':seq,'year':year,
                'maximum_daily_precip_up_unadjusted':part.daily_precip_up.max(),
                'maximum_daily_precip_down_unadjusted':part.daily_precip_down.max(),
                'jas_mean_temperature_down_unadjusted':warm.daily_ave_temp_down.mean()})
    annual = annual.merge(mapping.rename(columns={'target_year':'year'}),on=['sequence_id','year'],validate='many_to_one')
    annual = annual.merge(pd.DataFrame(weather_rows),on=['sequence_id','year'],validate='many_to_one')
    annual['plot_label'] = annual.source_solution_id.map(labels)
    annual['jas_mean_temperature_down_adjusted'] = annual.jas_mean_temperature_down_unadjusted + annual.target_scenario.map({'present':0.,'2C':1.3,'4C':4.1})
    csv(annual,out,'weak_case_annual.csv')
    # Diagnose crop quality and timing even if flood dominates the aggregate score.
    details = []
    for row in weak.itertuples():
        d = annual.loc[annual.source_solution_id.eq(row.source_solution_id) & annual.sequence_id.eq(row.sequence_id) & annual.target_scenario.eq(row.target_scenario)].sort_values('year')
        first = d.iloc[0]; rain = d.loc[d.maximum_adjusted_daily_precip_up.idxmax()]
        completed = d.breeding_effect_ratio_at_harvest > 0
        details.append({**row._asdict(),
            'first_source_year':int(first.source_year),
            'largest_rain_source_year':int(rain.source_year),'largest_rain_target_year':int(rain.year),
            'first_year_adjusted_max_precip_up_mm':float(first.maximum_adjusted_daily_precip_up),
            'share_flood_damage_first_3years':d.iloc[:3].flood_damage.sum()/d.flood_damage.sum() if d.flood_damage.sum()>0 else np.nan,
            'share_inundation_damage_first_3years':d.iloc[:3].innundation_damage.sum()/d.innundation_damage.sum() if d.innundation_damage.sum()>0 else np.nan,
            'first_harvest_with_breeding_effect':int(d.loc[completed,'year'].min()) if completed.any() else np.nan,
            'years_quality_factor_at_floor':int(np.isclose(d.crop_price_quality_factor_at_harvest,.71,atol=1e-12,rtol=0).sum()),
            'years_at_floor_after_breeding':int((completed & np.isclose(d.crop_price_quality_factor_at_harvest,.71,atol=1e-12,rtol=0)).sum())})
    detail = pd.DataFrame(details); csv(detail,out,'weak_case_summary.csv')
    # Five mean-adverse examples, one per candidate, selected automatically.
    examples = weak.loc[weak.criterion.eq('mean') & weak.adverse_rank.eq(1)]
    shared = len(examples[['sequence_id','target_scenario','dominant_objective_index']].drop_duplicates()) == 1
    if shared:
        fig, grid = plt.subplots(3,2,figsize=(16,11))
        panels = grid.ravel()
    else:
        fig, axes = plt.subplots(len(examples),2,figsize=(16,3*len(examples)),squeeze=False)
    for (i,row) in enumerate(examples.itertuples()):
        d = annual.loc[annual.source_solution_id.eq(row.source_solution_id)&annual.sequence_id.eq(row.sequence_id)&annual.target_scenario.eq(row.target_scenario)].sort_values('year')
        decision = decisions.loc[row.source_solution_id]
        if not shared or i == 0:
            ax = panels[0] if shared else axes[i,0]
            if row.dominant_objective_index == 3:
                ax.plot(d.year,d.jas_mean_temperature_down_adjusted,color='#d55e00',marker='o')
                ax.set_ylabel('7–9月平均気温（℃）\n下流、作物の熱ストレス自体ではない')
            else:
                ax.bar(d.year,d.maximum_adjusted_daily_precip_up,color='#56b4e9')
                ax.set_ylabel('上流・年最大日降水（mm）\nモデル内の気候補正後')
            ax.set_xticks(d.year,[str(int(x)) for x in d.source_year],rotation=45)
            ax.set_xlabel('配置された元の気象年（位置は2009→2023年）')
            ax.set_title(f'{"5政策に共通" if shared else labels[row.source_solution_id]}：{row.sequence_id}／{row.target_scenario}')
            ax.grid(axis='y',alpha=.2)
        ax = panels[i+1] if shared else axes[i,1]
        j = row.dominant_objective_index
        metric = RAW[j]
        scale = 1e8 if j in (0,1,3) else 1.
        ax.bar(d.year,d[metric]/scale,color='#0072b2',label='不利な系列')
        control = annual.loc[annual.source_solution_id.eq(row.source_solution_id)&annual.sequence_id.eq('original')&annual.target_scenario.eq(row.target_scenario)].sort_values('year')
        ax.plot(control.year,control[metric]/scale,color='#333333',marker='.',label='元順序')
        ax.set_ylabel(f'年次{METRIC_LABELS[j]}'+('（億円）' if scale>1 else '（モデル指標）'))
        windows = [('house_elevation','#cc79a7','嵩上げ実施'),('migration','#e69f00','移転実施')] if j<2 else [('breeding','#cc79a7','育種投資')] if j==3 else [('forest_management','#009e73','森林管理実施')]
        for prefix,color,label in windows:
            ax.axvspan(decision[f'{prefix}_start_calendar_year']-.45,decision[f'{prefix}_end_calendar_year']+.45,alpha=.09,color=color,label=label)
        if j==3:
            ready = d.loc[d.breeding_effect_ratio_at_harvest>0,'year']
            if len(ready): ax.axvline(ready.min(),color='#d55e00',ls=':',label='育種効果が初めて適用された収穫年')
        ax.set_xticks(d.year,d.year.astype(int).astype(str),rotation=45); ax.set_xlabel('政策実施年'); ax.grid(axis='y',alpha=.2)
        ax.set_title(f'{labels[row.source_solution_id]}：平均スコアで最大の寄与 {row.target_scenario}・{METRIC_LABELS[j]}')
        ax.legend(fontsize=8,ncol=2)
    fig.suptitle('候補ごとの平均スコアが最も不利な系列：気象配置と年次成果',fontsize=15)
    fig.tight_layout(rect=(0,.02,1,.96))
    fig.text(.5,.005,'着色は投資・対策実施期間。年次成果との対応は記述的診断であり、時期の因果効果を分離した実験ではない。',ha='center',fontsize=9)
    save(fig,out,'03_adverse_weather_timing')
    return detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--repetitions',type=int,default=500)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.repetitions < 1: parser.error('repetitions must be positive')
    ensemble = args.ensemble_dir.resolve()
    out = (args.output_dir or ensemble/'weather_selection_stability').resolve()
    if out == ensemble or ensemble in out.parents and out.name in ('cases','figures'): raise ValueError('Use a separate analysis output directory')
    hashes = {n:hashlib.sha256((ensemble/n).read_bytes()).hexdigest() for n in INPUTS}
    out.mkdir(parents=True,exist_ok=True)
    settings = json.loads((ensemble/'regret_normalization.json').read_text(encoding='utf-8'))['pareto_range']
    policies = pd.read_csv(ensemble/'policy_labels.csv')
    ids = policies.solution_id.tolist(); labels = dict(zip(ids,policies.plot_label))
    decisions = pd.read_csv(ensemble/'selected_decisions.csv').set_index('solution_id')
    results = pd.read_csv(ensemble/'ensemble_results.csv')
    seqs = sorted(results.loc[results.sequence_id.ne('original'),'sequence_id'].unique())
    if len(seqs) != 50: raise ValueError('This analysis expects the completed 50-permutation ensemble')
    keys = ['source_solution_id','sequence_id','target_scenario']
    if results.duplicated(keys).any(): raise ValueError('Duplicate saved cases')
    results[[f'regret_{j}' for j in range(5)]] = fixed_regrets(results,settings)
    index = pd.MultiIndex.from_tuples(list(itertools.product(ids,seqs,SCENARIO_ORDER)),names=keys)
    random = results.set_index(keys).reindex(index)
    cube = random[[f'regret_{j}' for j in range(5)]].to_numpy().reshape(len(ids),50,3,5)
    if not np.isfinite(cube).all(): raise ValueError('Missing policy/sequence/climate case')
    eligible = random.feasible.astype(str).str.lower().eq('true').to_numpy().reshape(len(ids),50,3).all(axis=(1,2))
    scores = cube.mean(axis=(2,3)); worst = cube.max(axis=(2,3))
    base = pd.read_csv(ensemble/'policy_robustness.csv').set_index('source_solution_id').loc[ids]
    np.testing.assert_allclose(scores.mean(axis=1),base.mean_regret,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(worst.max(axis=1),base.worst_regret,rtol=1e-12,atol=1e-12)
    sizes = [10,20,30,40,50]
    frequency, draws = resample_selection(scores,eligible,sizes,args.repetitions,args.seed)
    frequency['source_solution_id'] = frequency.policy_index.map(dict(enumerate(ids)))
    frequency['plot_label'] = frequency.source_solution_id.map(labels)
    csv(frequency,out,'selection_frequencies.csv')
    draw_path = out/'resampled_subsets.json'
    draw_path.write_text(json.dumps({'sequence_ids':seqs,'draws':draws},indent=2),encoding='utf-8')
    _, _, full_rank = selection_masks(scores.mean(axis=1),eligible)
    seq_records = []
    for p,sid in enumerate(ids):
        for n,seq in enumerate(seqs):
            c,j = np.unravel_index(cube[p,n].argmax(),(3,5))
            seq_records.append({'source_solution_id':sid,'plot_label':labels[sid],'sequence_id':seq,
                'mean_score':scores[p,n],'maximum_component_regret':worst[p,n],
                'dominant_climate':SCENARIO_ORDER[c],'dominant_objective':METRIC_LABELS[j]})
    csv(pd.DataFrame(seq_records),out,'sequence_scores.csv')
    originals = results.loc[results.sequence_id.eq('original')].set_index(keys).sort_index()
    original_scores = []
    for sid in ids:
        original_scores.append(originals.loc[(sid,'original'),[f'regret_{j}' for j in range(5)]].to_numpy().mean())
    _, _, original_ranks = selection_masks(np.array(original_scores),eligible)
    candidates = [sid for sid,_ in POLICIES]
    summary = base.reset_index().copy()
    summary['full_mean_rank_inclusive_tolerance'] = full_rank
    summary['original_mean_score'] = original_scores; summary['original_mean_rank'] = original_ranks
    summary['is_narrative_candidate'] = summary.source_solution_id.isin(candidates)
    for size in sizes:
        f = frequency.loc[frequency.sequence_count.eq(size)].set_index('source_solution_id')
        for metric in ('best_frequency','top3_frequency','rank_median','rank_p10','rank_p90'):
            summary[f'{metric}_n{size}'] = summary.source_solution_id.map(f[metric])
    csv(summary.sort_values('mean_regret'),out,'policy_stability_summary.csv')
    csv(summary.loc[summary.is_narrative_candidate].sort_values('mean_regret'),out,'candidate_summary.csv')
    paired = []
    for a,b in itertools.combinations(candidates,2):
        x,y = ids.index(a),ids.index(b)
        difference = scores[x]-scores[y]
        for n,seq in enumerate(seqs):
            paired.append({'policy_a':labels[a],'policy_b':labels[b],'sequence_id':seq,
                'score_a':scores[x,n],'score_b':scores[y,n],'difference_a_minus_b':difference[n]})
    paired = pd.DataFrame(paired); csv(paired,out,'paired_sequence_differences.csv')
    pair_rows = []
    for (a,b),part in paired.groupby(['policy_a','policy_b'],sort=False):
        d = part.difference_a_minus_b
        pair_rows.append({'policy_a':a,'policy_b':b,'mean_difference_a_minus_b':d.mean(),
            'sequences_a_better':int((d < -TIE_ATOL).sum()),'sequences_b_better':int((d>TIE_ATOL).sum()),
            'tied_sequences':int((d.abs()<=TIE_ATOL).sum()),'difference_p10':d.quantile(.1),'difference_p90':d.quantile(.9)})
    csv(pd.DataFrame(pair_rows),out,'paired_comparison_summary.csv')

    configure_fonts()
    display = list(dict.fromkeys([*base.nsmallest(8,'mean_regret').index,*candidates]))
    fig,axes = plt.subplots(1,2,figsize=(14,6))
    for sid in display:
        f = frequency.loc[frequency.source_solution_id.eq(sid)]
        for ax,metric in zip(axes,['best_frequency','top3_frequency']):
            ax.plot(f.sequence_count,f[metric]*100,marker='o',label=labels[sid],lw=1.6)
    for ax,title in zip(axes,['同点を含む最良','同点を含む上位3政策']):
        ax.set_title(title);ax.set_xlabel('再抽出した系列数');ax.set_ylabel('再抽出で選ばれた割合（%）')
        ax.set_xticks(sizes);ax.set_ylim(-3,103);ax.grid(alpha=.2)
    axes[1].legend(ncol=3,fontsize=9)
    fig.suptitle('全143政策の選定安定性：固定Pareto正規化・等重み',fontsize=15)
    fig.tight_layout(rect=(0,.1,1,.94));fig.text(.5,.02,f'10–40系列：各{args.repetitions}組を重複なしで抽出。50系列：全標本1組。線は上位8政策と5候補。\n元順序は除外。頻度は保存50系列内の安定性であり、将来確率ではない。',ha='center',fontsize=9)
    save(fig,out,'01_selection_stability')
    leader = base.mean_regret.idxmin()
    fig,axes = plt.subplots(2,1,figsize=(14,9),gridspec_kw={'height_ratios':[2,1]})
    for sid,color in zip(candidates,COLORS):
        p = ids.index(sid)
        axes[0].plot(range(1,51),scores[p],marker='.',lw=.8,color=color,label=f'{labels[sid]}（全143政策中の平均順位{int(full_rank[p])}）')
    a = next(sid for sid in candidates if labels[sid]=='P75')
    b = next(sid for sid in candidates if labels[sid]=='P91')
    difference = scores[ids.index(a)]-scores[ids.index(b)]
    axes[1].plot(range(1,51),difference,marker='o',lw=1,color='#0072b2',ms=3)
    axes[1].fill_between(range(1,51),difference,0,where=difference>0,color='#e69f00',alpha=.2,label='P91が有利')
    axes[1].fill_between(range(1,51),difference,0,where=difference<0,color='#0072b2',alpha=.2,label='P75が有利')
    axes[1].axhline(difference.mean(),color='#d55e00',ls=':',label='50系列での平均差')
    axes[0].set_ylabel('系列ごとの平均スコア（低いほど良好）');axes[0].legend(ncol=3,fontsize=9)
    axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('P75 − P91 のスコア差\n負値はP75が有利');axes[1].legend(ncol=3,fontsize=9)
    axes[1].set_title(f'近い2候補を拡大：P75が有利 {int((difference < -TIE_ATOL).sum())}系列／P91が有利 {int((difference > TIE_ATOL).sum())}系列',fontsize=10)
    for ax in axes: ax.grid(alpha=.2);ax.set_xlim(.5,50.5);ax.set_xlabel('保存済み並べ替え系列番号（番号に良悪の順序はない）')
    fig.suptitle('主要5候補を同じ気象系列で比較：3気候×5指標の平均',fontsize=15)
    fig.tight_layout(rect=(0,.03,1,.94));save(fig,out,'02_paired_policy_scores')
    weak = []
    for sid in candidates:
        p = ids.index(sid)
        for criterion,values,count in [('mean',scores[p],5),('maximum',worst[p],1)]:
            # Deterministic sequence ID breaks display ties only; all max ties
            # are exported below separately and not treated as a unique maximum.
            order = np.argsort(-values,kind='stable')[:count]
            for rank,n in enumerate(order,1):
                c,j = np.unravel_index(cube[p,n].argmax(),(3,5))
                weak.append({'source_solution_id':sid,'plot_label':labels[sid],'criterion':criterion,
                    'adverse_rank':rank,'sequence_id':seqs[n],'target_scenario':SCENARIO_ORDER[c],
                    'dominant_objective_index':int(j),'dominant_objective':METRIC_LABELS[j],
                    'mean_score':scores[p,n],'maximum_component_regret':worst[p,n],
                    'dominant_component_regret':cube[p,n,c,j],
                    'dominant_component_share_of_mean':cube[p,n,c,j]/cube[p,n].sum() if cube[p,n].sum()>0 else np.nan,
                    'dominant_component_tie_count':int(np.isclose(cube[p,n],cube[p,n,c,j],atol=TIE_ATOL,rtol=0).sum()),
                    'adverse_score_tie_count':int(np.isclose(values,values[n],atol=TIE_ATOL,rtol=0).sum())})
    weak = pd.DataFrame(weak);csv(weak,out,'adverse_cases.csv')
    components = []
    for row in weak.itertuples():
        p,n = ids.index(row.source_solution_id),seqs.index(row.sequence_id)
        for c,j in itertools.product(range(3),range(5)):
            value = cube[p,n,c,j]
            components.append({'source_solution_id':row.source_solution_id,'plot_label':row.plot_label,
                'criterion':row.criterion,'adverse_rank':row.adverse_rank,'sequence_id':row.sequence_id,
                'target_scenario':SCENARIO_ORDER[c],'objective':METRIC_LABELS[j],
                'component_regret':value,'contribution_to_mean_score':value/15,
                'delta_vs_policy_50sequence_mean':value-cube[p,:,c,j].mean()})
    csv(pd.DataFrame(components),out,'adverse_regret_components.csv')
    maxima = []
    for sid in candidates:
        p=ids.index(sid);m=worst[p].max()
        for n,c,j in np.argwhere(np.isclose(cube[p],m,atol=TIE_ATOL,rtol=0)):
            maxima.append({'source_solution_id':sid,'plot_label':labels[sid],'sequence_id':seqs[n],
                'target_scenario':SCENARIO_ORDER[c],'objective':METRIC_LABELS[j],'regret':m})
    csv(pd.DataFrame(maxima),out,'all_maximum_drivers.csv')
    print('Selection and paired-score analysis complete; diagnosing saved annual cases.',flush=True)
    diagnose_years(ensemble,out,weak,labels,decisions)
    after = {n:hashlib.sha256((ensemble/n).read_bytes()).hexdigest() for n in INPUTS}
    if hashes != after: raise ValueError('Source files changed during analysis')
    metadata = {'schema':1,'policy_count':len(ids),'sequence_count':len(seqs),'feasible_policy_count':int(eligible.sum()),
        'repetitions_per_partial_subset':args.repetitions,'full_subset_draw_count':1,'seed':args.seed,
        'normalization':'saved pareto_range; fixed across every draw','weights':'equal objectives and equal climates',
        'tie_absolute_tolerance':TIE_ATOL,'original_excluded':True,'sd_model_runs':0,
        'mean_and_maximum_match_existing':True,'source_sha256':hashes,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'notes':['Selection frequencies are conditional on saved 50 permutations, not future probabilities.',
                 'Inclusive top-three ties may select more than three candidates.',
                 'Mean-score adverse sequences and maximum-component drivers are different criteria.',
                 'Annual weather/response alignment is descriptive, not a causal intervention test.',
                 'Weak weather charts show the largest contributing climate/objective in the adverse mean-score case.']}
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved: {out}',flush=True)


if __name__ == '__main__':
    main()
