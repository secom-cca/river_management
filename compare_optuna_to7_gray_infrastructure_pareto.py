"""Check original-weather five-objective dominance of recurring gray policies.

Uses only saved outputs. Comparisons always share the target climate; costs
and GDP are reported but are not added to the original five objectives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

METRICS = ('flood_damage','innundation_damage','landslide_risk','crop_revenue','biodiversity')
SIGNS = (1,1,1,-1,-1)
CLIMATES = ('present','2C','4C')


def read_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows: raise ValueError(f'No rows for {path}')
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def losses(row, baseline):
    scales=[float(baseline[m]) for m in METRICS]
    if not all(v>0 for v in scales):raise ValueError('Positive baseline scales required')
    return tuple(s*float(row[m])/v for m,s,v in zip(METRICS,SIGNS,scales))


def dominates(a, b, tolerance=1e-9):
    """a is no worse in all minimized losses and strictly better in at least one."""
    if len(a)!=len(METRICS) or len(b)!=len(METRICS): raise ValueError('Expected five objectives')
    return all(x<=y+tolerance for x,y in zip(a,b)) and any(x<y-tolerance for x,y in zip(a,b))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ensemble-dir',type=Path,required=True)
    parser.add_argument('--comparison-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    ensemble=args.ensemble_dir.resolve()
    comparison=(args.comparison_dir or ensemble/'gray_infrastructure_10years_original').resolve()
    out=(args.output_dir or comparison/'pareto_dominance').resolve()
    if out in (ensemble,comparison):raise ValueError('Use a separate output directory')
    paths=[comparison/'comparison_summary.csv',comparison/'config.json',ensemble/'config.json',
        ensemble/'ensemble_results.csv',ensemble/'policy_labels.csv',ensemble/'reference_pareto'/'sha256.json']
    paths += [ensemble/'reference_pareto'/f'pareto_summary_{c}.csv' for c in CLIMATES]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    config=json.loads((comparison/'config.json').read_text(encoding='utf-8'))
    source=json.loads((ensemble/'config.json').read_text(encoding='utf-8'))
    if config['model_signature']!=source['signature']:raise ValueError('Different source model signatures')
    snapshot_hashes=json.loads((ensemble/'reference_pareto'/'sha256.json').read_text(encoding='utf-8'))
    for c in CLIMATES:
        p=ensemble/'reference_pareto'/f'pareto_summary_{c}.csv'
        if hashlib.sha256(p.read_bytes()).hexdigest()!=snapshot_hashes[p.name]:raise ValueError('Pareto snapshot changed')
    gray=read_rows(comparison/'comparison_summary.csv')
    original=[r for r in read_rows(ensemble/'ensemble_results.csv') if r['sequence_id']=='original']
    labels={r['solution_id']:r['plot_label'] for r in read_rows(ensemble/'policy_labels.csv')}
    for r in original:
        if r['feasible'].lower()!='true':raise ValueError('Infeasible comparison policy')
    by_key={(r['source_solution_id'],r['target_scenario']):r for r in original}
    if len(by_key)!=len(original) or len(original)!=len(labels)*3:raise ValueError('Incomplete original-order cross evaluation')
    out.mkdir(parents=True,exist_ok=True)
    summary,pairs,witnesses,stability=[],[],[],[]
    for c in CLIMATES:
        front=read_rows(ensemble/'reference_pareto'/f'pareto_summary_{c}.csv')
        if any(r['scenario']!=c or r['feasible'].lower()!='true' for r in front):raise ValueError('Invalid source front')
        # Source-scenario exports and original-order replay must agree.
        for r in front:
            replay=by_key[(r['solution_id'],c)]
            for m in METRICS:
                if abs(float(r[m])-float(replay[m]))>max(1e-8,abs(float(r[m]))*1e-12):raise ValueError('Source/replay objective mismatch')
        baseline=next(r for r in gray if r['scenario']==c and r['action']=='no_policy')
        for policy in [r for r in gray if r['scenario']==c]:
            candidate_loss=losses(policy,baseline)
            for scope,refs in [('same_climate_front',front),('all_design_climates_same_target',[r for r in original if r['target_scenario']==c])]:
                relations=[]
                for ref in refs:
                    sid=ref.get('solution_id',ref.get('source_solution_id'))
                    ref_loss=losses(ref,baseline)
                    dom=dominates(ref_loss,candidate_loss)
                    reverse=dominates(candidate_loss,ref_loss)
                    equal=all(abs(a-b)<=1e-9 for a,b in zip(ref_loss,candidate_loss))
                    relation='reference_dominates' if dom else 'gray_dominates' if reverse else 'equivalent' if equal else 'tradeoff'
                    row={'gray_case_id':policy['case_id'],'gray_action':policy['action'],'target_scenario':c,
                        'comparison_scope':scope,'reference_solution_id':sid,'reference_label':labels[sid],
                        'reference_design_climate':ref.get('scenario',ref.get('source_scenario')),
                        'relation':relation,'reference_cost_yen':float(ref['policy_additional_cost']),
                        'gray_cost_yen':float(policy['policy_additional_cost'])}
                    for m,s in zip(METRICS,SIGNS):
                        row[f'gray_{m}']=float(policy[m]);row[f'reference_{m}']=float(ref[m])
                        row[f'reference_advantage_{m}_baseline_pp']=s*(float(policy[m])-float(ref[m]))/float(baseline[m])*100
                    pairs.append(row);relations.append((ref,row))
                dominators=[(r,p) for r,p in relations if p['relation']=='reference_dominates']
                witness=min(dominators,key=lambda x:(float(x[0]['policy_additional_cost']),x[1]['reference_solution_id']))[1] if dominators else None
                result={'gray_case_id':policy['case_id'],'gray_action':policy['action'],'target_scenario':c,
                    'comparison_scope':scope,'reference_policy_count':len(refs),'dominating_reference_count':len(dominators),
                    'gray_dominates_reference_count':sum(p['relation']=='gray_dominates' for _,p in relations),
                    'equivalent_reference_count':sum(p['relation']=='equivalent' for _,p in relations),
                    'tradeoff_reference_count':sum(p['relation']=='tradeoff' for _,p in relations),
                    'is_dominated':bool(dominators),'gray_cost_yen':float(policy['policy_additional_cost']),
                    'witness_label':witness['reference_label'] if witness else '',
                    'witness_solution_id':witness['reference_solution_id'] if witness else '',
                    'witness_cost_yen':witness['reference_cost_yen'] if witness else ''}
                summary.append(result)
                if witness:
                    witnesses.append(witness)
                    if any(witness[f'reference_advantage_{m}_baseline_pp']< -1e-7 for m in METRICS):raise ValueError('Invalid witness')
                # Report floating-point exact counts separately; diagnose any
                # classifications that depend on a small numerical tolerance.
                for tolerance in [0.,1e-12,1e-10,1e-9]:
                    count=sum(dominates(losses(r,baseline),candidate_loss,tolerance) for r in refs)
                    stability.append({'gray_case_id':policy['case_id'],'comparison_scope':scope,
                        'tolerance_in_baseline_ratio':tolerance,'dominating_reference_count':count,'is_dominated':count>0})
    write_rows(out/'dominance_summary.csv',summary)
    write_rows(out/'pairwise_dominance.csv',pairs)
    write_rows(out/'dominating_policy_examples.csv',witnesses)
    write_rows(out/'numerical_tolerance_check.csv',stability)
    if hashes!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}:raise ValueError('Inputs changed')
    grouped={}
    for r in stability:grouped.setdefault((r['gray_case_id'],r['comparison_scope']),set()).add(r['is_dominated'])
    if any(len(s)!=1 for s in grouped.values()):raise ValueError('Dominance classification depends on tolerance')
    metadata={'schema':1,'objectives':list(METRICS),'minimize':[True,True,True,False,False],
        'tolerance_in_baseline_ratio':1e-9,'normalization':'Positive within-climate no-policy scaling; no clipping or objective weights.',
        'notes':['Each reference and gray policy is evaluated in the same target climate and original weather order.',
                 'Primary comparison is each climate-specific exported front; secondary comparison is all 143 policies cross-evaluated in that climate.',
                 'A dominating witness is no worse in all five objectives and better in at least one.',
                 'Costs and GDP are not sixth objectives. Least cost selects an explanatory witness only.',
                 'Gray recurring policies extend the original one-time decision representation, while retaining annual/total budget ceilings.',
                 'Machine-exact and tolerance checks give identical dominated/not-dominated classifications.'],
        'sd_model_runs':0,'sources_unchanged':True,'input_sha256':hashes,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'analysis_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in summary:
        if r['comparison_scope']=='same_climate_front':
            print(r['target_scenario'],r['gray_action'],f"dominated by {r['dominating_reference_count']}/{r['reference_policy_count']}; reverse {r['gray_dominates_reference_count']}; example {r['witness_label']}")
    print(f'Saved: {out}')


if __name__=='__main__':main()
