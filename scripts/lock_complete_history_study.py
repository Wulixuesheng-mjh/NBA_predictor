"""Lock development choices against the original frozen cohort before audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from ml.src.evaluation import classification_metrics, regression_metrics, paired_day_bootstrap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(reference, rows, target):
    if [(r['nba_game_id'],r['actual']) for r in rows] != [(r['nba_game_id'],r['actual']) for r in reference]:
        raise ValueError('Development cohort or labels differ')
    index={'win':0,'spread':1,'total':2}[target]
    y=np.array([r['actual'][index] for r in rows])
    baseline=np.array([r['prediction'][index] for r in reference])
    candidate=np.array([r['prediction'][index] for r in rows])
    a=(baseline-y)**2 if target=='win' else abs(baseline-y)
    b=(candidate-y)**2 if target=='win' else abs(candidate-y)
    metric=classification_metrics if target=='win' else regression_metrics
    before,after=metric(y,baseline),metric(y,candidate)
    improvement=float(1-b.mean()/a.mean())
    ci=paired_day_bootstrap(reference,a,b)
    worst=0.;slices={}
    masks={label:np.array([bool(r.get(label,False)) for r in reference]) for label in
        ('early_season','major_injury','incomplete_injury_report','injury_net_unavailable')}
    masks.update({kind:np.array([r.get('game_type')==kind for r in reference]) for kind in
                  ('regular','playoffs','play_in')})
    for label,mask in masks.items():
        if not mask.any(): continue
        change=float(b[mask].mean()/a[mask].mean()-1)
        slices[label]=dict(samples=int(mask.sum()),change=change)
        if mask.sum()>=100: worst=max(worst,change)
    rmse_ok=target=='win' or after['rmse']<=before['rmse']*1.01
    return dict(baseline=before,candidate=after,development_improvement=improvement,
        development_loss_difference_95ci=ci,worst_eligible_slice_change=worst,rmse_gate=bool(rmse_ok),
        slices=slices,development_gate_pass=bool(improvement>=.02 and ci[1]<0 and worst<=.05 and rmse_ok),
        eligible_for_shadow=False,audit_improvement=None)


def main(args):
    root=Path(args.directory);output=root/'development_lock.json'
    if output.exists(): raise ValueError('Completed development lock is immutable')
    reports={m:json.loads((root/f'development_{m}.json').read_text()) for m in ('direct','residual')}
    reference=json.loads(Path(args.reference).read_text())['predictions']['xgboost:T0']
    candidates={f'{mode}:{name}':rows for mode,report in reports.items() for name,rows in report['predictions'].items()}
    for mode,report in reports.items():
        if not report['metrics'] or report['protocol']['total_strategy']!=mode:
            raise ValueError('Incomplete or incompatible development report')
        if report['protocol']['feature_cache_key']!='bbeb026e149ba7b9' or report['protocol']['injuries']!='none':
            raise ValueError('Unplanned study inputs')
        if report['protocol'].get('cohort_reference_sha256')!=sha(args.reference):
            raise ValueError('Frozen reference checksum differs')
    selected={}
    gates={}
    for target,index in (('win',0),('spread',1),('total',2)):
        choices={k:v for k,v in candidates.items() if target=='total' or k.startswith('direct:')}
        def loss(rows):
            errors=np.array([r['prediction'][index]-r['actual'][index] for r in rows])
            return np.mean(errors**2 if target=='win' else abs(errors))
        selected[target]=min(choices,key=lambda k:loss(choices[k]))
        gates[target]=assess(reference,choices[selected[target]],target)
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),selected=selected,
        reference_sha256=sha(args.reference),source_sha256={m:sha(root/f'development_{m}.json') for m in reports},
        executor_sha256=sha(__file__),release_gates=gates,audit_not_executed=True,promotion=False,
        candidates={k:dict(total_mae=float(np.mean([abs(r['prediction'][2]-r['actual'][2]) for r in rows])),
            total_bias=float(np.mean([r['prediction'][2]-r['actual'][2] for r in rows]))) for k,rows in candidates.items()})
    with output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--reference',required=True)
    main(p.parse_args())
