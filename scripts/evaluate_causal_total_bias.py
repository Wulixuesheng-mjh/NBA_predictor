"""One prespecified additive-bias experiment; no scale inflation or audit tuning."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from ml.src.evaluation import nba_day, paired_day_bootstrap, regression_metrics


def correct(rows, seed=()):
    history=list(seed)
    months=defaultdict(list)
    for row in rows:
        months[nba_day(row)[:7]].append(row)
    corrected=[]
    for month,values in sorted(months.items()):
        assert all(day[:7]<month for day,_ in history)
        errors=[error for _,error in history[-1000:]]
        offset=-sum(errors)/(len(errors)+200) if len(errors)>=200 else 0.0
        offset=float(np.clip(offset,-5,5))
        for row in values:
            corrected.append(dict(**row,raw_total=row['prediction'][2],
                corrected_total=row['prediction'][2]+offset,offset=offset,
                calibration_history=len(errors),calibration_until=history[-1][0] if history else None))
        history.extend((nba_day(row),row['prediction'][2]-row['actual'][2]) for row in values)
    return corrected,history


def summarize(rows):
    y=np.array([r['actual'][2] for r in rows])
    raw=np.array([r['raw_total'] for r in rows])
    predicted=np.array([r['corrected_total'] for r in rows])
    before,after=regression_metrics(y,raw),regression_metrics(y,predicted)
    return dict(samples=len(rows),raw=before,corrected=after,
        mae_improvement=(before['mae']-after['mae'])/before['mae'],
        difference_95ci=paired_day_bootstrap(rows,abs(raw-y),abs(predicted-y)),
        months=[dict(month=month,offset=next(r['offset'] for r in rows if nba_day(r)[:7]==month),
                     history_until=next(r['calibration_until'] for r in rows if nba_day(r)[:7]==month))
                for month in sorted({nba_day(r)[:7] for r in rows})])


def main(args):
    root=Path(args.directory)
    lock=json.loads((root/'total_selection_lock.json').read_text())
    output=dict(protocol=dict(minimum_past_oos=200,window=1000,prior_strength=200,max_absolute_offset=5,
        monthly_fixed_offset=True,uses_training_in_sample_errors=False,audit_retuning=False),candidates={})
    for mode,key in lock['selected_by_mode'].items():
        dev_path=root/f'development_{mode}.json'
        assert hashlib.sha256(dev_path.read_bytes()).hexdigest()==lock['source_sha256'][mode]
        dev=json.loads(dev_path.read_text())['predictions']['xgboost:'+key.split(':')[1]]
        audit=json.loads((root/f'audit_{mode}.json').read_text())['predictions']['xgboost:'+key.split(':')[1]]
        d,history=correct(dev)
        a,_=correct(audit,history)
        output['candidates'][key]=dict(development=summarize(d),audit=summarize(a),
            records=dict(development=d,audit=a))
    output['decision']=dict(promote=False,production_bias_correction_enabled=False,
        reason='Research evidence only; original release gates remain required')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:{p:v[p] for p in ('development','audit')}
                      for k,v in output['candidates'].items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
