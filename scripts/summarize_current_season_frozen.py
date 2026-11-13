"""Score a locked season reconstruction, retaining rejected/raw predictions."""
import json,hashlib,random
from collections import defaultdict
from datetime import date
from pathlib import Path
import reevaluate_models_v2 as scoring

OUT=Path(__file__).resolve().parents[1]/'reports/season_frozen_audit_20261112'


def bootstrap(days,values):
    groups=defaultdict(list)
    for d,v in zip(days,values):groups[d].append(v)
    if len(groups)<2:return None
    bins=[(sum(v),len(v)) for v in groups.values()];rng=random.Random(20261112);samples=[]
    for _ in range(4000):
        chosen=rng.choices(bins,k=len(bins));samples.append(sum(v[0] for v in chosen)/sum(v[1] for v in chosen))
    return [scoring.q(samples,.025),scoring.q(samples,.975)]


def compare(rows,old,new,target,index):
    values=[]
    for r in rows:
        a,b=old(r),new(r)
        if a is not None and b is not None:values.append(dict(day=r['day'],y=r['actual'][index],old=a,new=b))
    result=scoring.compare(values,target) if values else dict(samples=0)
    result['missing_pairs']=len(rows)-len(values)
    return result


def main():
    assert not (OUT/'results.json').exists()
    complete=json.loads((OUT/'complete.json').read_text())
    assert hashlib.sha256((OUT/'rows.jsonl').read_bytes()).hexdigest()==complete['rows_sha256']
    rows=[]
    for line in (OUT/'rows.jsonl').read_text().splitlines():
        w=json.loads(line);assert scoring.digest(w['payload'])==w['sha256'];rows.append(w['payload'])
    assert len(rows)==complete['samples']
    good=[r for r in rows if 'failure' not in r]
    scoring.boot=bootstrap
    comparisons={};fields=['win_prob_home','predicted_spread','predicted_total']
    for i,target in enumerate(('win','spread','total')):
        old=lambda r:r['baseline']['models']['mlp']['raw'].get(fields[i]) if r['baseline']['models']['mlp']['raw'] else None
        new=lambda r:r['candidate']['models']['mlp']['raw'].get(fields[i]) if r['candidate']['models']['mlp']['raw'] else None
        comparisons[target]=compare(good,old,new,target,i)
    totals={}
    for mode in ('direct','residual'):
        selected=[r for r in good if r.get('total_shadows',{}).get('candidates')]
        totals[mode]=compare(selected,lambda r:r['baseline']['models']['xgboost']['raw']['predicted_total'],
            lambda r:next(c['raw_predicted_total'] for c in r['total_shadows']['candidates'] if c['strategy']==mode),'total',2)
    win_rows=[r for r in good if r.get('win_shadow')]
    win=compare(win_rows,lambda r:r['baseline']['models']['xgboost']['raw']['win_prob_home'],
        lambda r:r['win_shadow']['win_prob_home'],'win',0)
    xgb={t:scoring.stats([r['actual'][i] for r in good],
        [r['baseline']['models']['xgboost']['raw'][fields[i]] for r in good],t)
        for i,t in enumerate(('win','spread','total'))}
    slices={};start=min(date.fromisoformat(r['day']) for r in good)
    for name,predicate in [('first_7_days',lambda r:(date.fromisoformat(r['day'])-start).days<7),
        ('missing_previous_stats',lambda r:any('incomplete' in w.lower() for w in r['candidate']['warnings'])),
        ('high_total_240',lambda r:r['actual'][2]>=240)]:
        selected=[r for r in good if predicate(r)]
        if selected:
            slices[name]=dict(samples=len(selected),total=compare(selected,
                lambda r:r['baseline']['models']['mlp']['raw']['predicted_total'],
                lambda r:r['candidate']['models']['mlp']['raw']['predicted_total'],'total',2))
    quality=dict(samples=len(rows),successful=len(good),days=len({r['day'] for r in rows}),
        day_range=[min(r['day'] for r in rows),max(r['day'] for r in rows)],failures=complete['failures'],
        incomplete_target_stats=sum(not r['target_stats_complete'] for r in rows),
        verified_injury_team_reports=sum(v is not None for r in good for v in r['injuries'].values()),
        complete_injury_team_reports=sum(v is not None and v['complete'] for r in good for v in r['injuries'].values()),
        injury_team_opportunities=2*len(good),
        guard_rejections={s:sum(r[s]['models']['mlp']['guard_rejected'] for r in good) for s in ('baseline','candidate')},
        nonfinite_rows=sum(bool(r['nonfinite_paths']) for r in good),
        exact_tipoff_records=sum(r['time_precision']=='exact' for r in good))
    passed=not quality['failures'] and all(v['repair_noninferiority_screen_pass'] and not v['missing_pairs'] for v in comparisons.values())
    result=dict(quality=quality,repair_three_heads=comparisons,xgb=xgb,total_shadows=totals,win_shadow=win,slices=slices,
        A_repair_retrospective_safety_pass=passed,authentic_pregame=False,independent_new_test=False,
        candidate_search=False,promotion=False,production_changes=False,
        conclusion='Historical fixed-repair safety evidence, not 200 fabricated pregame forecasts; unknown injury coverage and reconstructed arrival times limit interpretation')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(quality=quality,repair={k:dict(old=v['baseline'],new=v['candidate'],ci=v['paired_day_95ci']) for k,v in comparisons.items()},
        shadows={k:dict(samples=v['samples'],ci=v.get('paired_day_95ci'),baseline=v.get('baseline'),candidate=v.get('candidate')) for k,v in totals.items()},pass_repair=passed),indent=2))


if __name__=='__main__':main()
