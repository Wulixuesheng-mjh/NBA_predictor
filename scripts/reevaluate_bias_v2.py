"""Paired absolute-mean-bias check; kept separate from MAE comparisons."""
import random,statistics
from collections import defaultdict
from reevaluate_models_v2 import ROOT,OUT,read,save,q


def absolute_bias_check(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['day']].append((r['old']-r['y'],r['new']-r['y']))
    old=statistics.fmean(a for v in groups.values() for a,b in v)
    new=statistics.fmean(b for v in groups.values() for a,b in v)
    ci=None
    if len(groups)>=2:
        rng=random.Random(20261110);values=[];bins=list(groups.values())
        for _ in range(4000):
            sampled=[r for group in rng.choices(bins,k=len(bins)) for r in group]
            values.append(abs(statistics.fmean(b for a,b in sampled))-abs(statistics.fmean(a for a,b in sampled)))
        ci=[q(values,.025),q(values,.975)]
    return dict(samples=len(rows),days=len(groups),baseline_bias=old,candidate_bias=new,
        absolute_bias_difference=abs(new)-abs(old),paired_day_95ci=ci,
        directional_bias_screen_pass=bool(ci and ci[1]<0),real_formal_gate_pass=False)


def main():
    rows=read(ROOT/'reports/mlp_constant_repair_20261110/workspace_snapshot_validation.json')['per_game']
    repair=absolute_bias_check([dict(day=r['source_day'],y=r['actual_total'],old=r['before'][2],new=r['after'][2]) for r in rows])
    records=read(OUT/'totals_records.json')['rows'];live={}
    for mode in ('direct','residual'):
        rows=[]
        for r in records:
            g=r['current_game']
            if g['status']=='Final':
                totals=r['payload']['totals'];rows.append(dict(day=g['source_day'],y=g['home_score']+g['away_score'],old=totals['baseline'],new=totals[mode]))
        live[mode]=absolute_bias_check(rows)
    save('absolute_bias_review.json',dict(repair_reconstructed=repair,live_total_shadows=live,
        caveat='very few live game days; descriptive intervals are not sufficient proof of population calibration'))


if __name__=='__main__':main()
