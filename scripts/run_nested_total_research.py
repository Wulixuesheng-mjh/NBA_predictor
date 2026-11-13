"""One prespecified adaptive residual rule, selected only inside past folds."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from ml.src.evaluation import complete_day_split, monthly_folds, nba_day, paired_day_bootstrap, regression_metrics
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.training_policy import policy_mask_and_weights, subset_arrays


PARAMETERS = dict(n_estimators=1600, learning_rate=.03, min_child_weight=20,
    reg_alpha=1., reg_lambda=10., subsample=.9, colsample_bytree=.9, gamma=0.,
    tree_method='hist', n_jobs=2, random_state=42)
CHOICES = [dict(objective=objective,max_depth=depth,policy=policy)
           for objective,depth,policy in itertools.product(
               ('reg:squarederror','reg:absoluteerror'),(2,3),('T0','T4'))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def score(reference,rows):
    assert [(r['nba_game_id'],r['actual']) for r in reference]==[(r['nba_game_id'],r['actual']) for r in rows]
    y=np.array([r['actual'][2] for r in rows]); p=np.array([r['prediction'][2] for r in rows])
    a=np.array([abs(r['prediction'][2]-r['actual'][2]) for r in reference]); b=abs(p-y)
    slices={}
    worst=0.
    for label,predicate in (
        ('early_season',lambda r:r['early_season']),
        ('major_injury',lambda r:r.get('major_injury',False)),
        ('incomplete_injury',lambda r:r.get('incomplete_injury_report',False)),
        ('playoffs',lambda r:r['game_type']=='playoffs'),
        ('regular',lambda r:r['game_type']=='regular')):
        keep=np.array([predicate(r) for r in rows],dtype=bool)
        if keep.any():
            change=float(b[keep].mean()/a[keep].mean()-1)
            slices[label]=dict(samples=int(keep.sum()),mae=float(b[keep].mean()),change=change)
            if keep.sum()>=100:
                worst=max(worst,change)
    rmse=regression_metrics(y,p)['rmse']; base_rmse=regression_metrics(y,np.array([r['prediction'][2] for r in reference]))['rmse']
    ci=paired_day_bootstrap(rows,a,b)
    improvement=float((a.mean()-b.mean())/a.mean())
    return dict(metrics=regression_metrics(y,p),improvement=improvement,difference_95ci=ci,
        worst_slice_degradation=worst,rmse_gate=bool(rmse<=base_rmse*1.01),slices=slices,
        development_gate_pass=bool(improvement>=.02 and ci[1]<0 and worst<=.05 and rmse<=base_rmse*1.01))


def fit_fold(past,test):
    split=complete_day_split(past.sample_info)
    selection=subset_arrays(past,np.arange(len(past.X))<split)
    val=past.X[split:]; vi=FEATURE_NAMES.index('shrunk_expected_total')
    actual_val=past.y_total[split:]; residual_val=actual_val-val[:,vi]
    baseline_mae=float(np.mean(abs(residual_val)))
    trials=[]
    for choice in CHOICES:
        mask,w,weight_report=policy_mask_and_weights(selection.sample_info,choice['policy'])
        train=subset_arrays(selection,mask)
        model=xgb.XGBRegressor(**PARAMETERS,objective=choice['objective'],max_depth=choice['max_depth'],
                               eval_metric='mae',early_stopping_rounds=80)
        model.fit(train.X,train.y_total-train.X[:,vi],sample_weight=w,
                  eval_set=[(val,residual_val)],sample_weight_eval_set=[np.ones(len(val))],verbose=False)
        mae=float(np.mean(abs(model.predict(val)-residual_val)))
        trials.append(dict(**choice,mae=mae,rounds=int(model.best_iteration)+1,
                           selection_weights=weight_report))
    chosen=min(trials,key=lambda r:(r['mae'],r['max_depth'],r['objective'],r['policy']))
    if baseline_mae<=chosen['mae']:
        prediction=test.X[:,vi].copy()
        fitted=dict(strategy='unadjusted_baseline',rounds=0)
    else:
        mask,w,weight_report=policy_mask_and_weights(past.sample_info,chosen['policy'])
        train=subset_arrays(past,mask)
        params=PARAMETERS|dict(n_estimators=chosen['rounds'],max_depth=chosen['max_depth'],objective=chosen['objective'])
        model=xgb.XGBRegressor(**params,eval_metric='mae')
        model.fit(train.X,train.y_total-train.X[:,vi],sample_weight=w,verbose=False)
        prediction=model.predict(test.X)+test.X[:,vi]
        fitted=dict(strategy='residual',**chosen,full_fit_weights=weight_report)
    assert np.isfinite(prediction).all()
    return prediction,dict(chosen=fitted,trials=trials,baseline_validation_mae=baseline_mae,
        inner_fit_until=nba_day(selection.sample_info[-1]),
        validation_from=nba_day(past.sample_info[split]),validation_until=nba_day(past.sample_info[-1]),
        full_fit_until=nba_day(past.sample_info[-1]))


def main(args):
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=True)
    source=Path(args.reference)/f'{args.phase}_direct.json'
    reference=json.loads(source.read_text())['predictions']['xgboost:T0']
    ref_by_id={r['nba_game_id']:r for r in reference}
    data=DatasetCache().load(args.feature_cache_key,len(FEATURE_NAMES))
    if data is None:
        raise ValueError('Missing pinned causal cache')
    disabled=[f'{s}_{n}' for s in ('home','away') for n in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES]
    data.X[:,[FEATURE_NAMES.index(n) for n in disabled]]=0
    assert np.all(data.X[:,FEATURE_NAMES.index('shrunk_expected_total')]>0)
    protocol=dict(rule='past-only nested residual selection',feature_cache_key=args.feature_cache_key,
        choices=CHOICES,params=PARAMETERS,stopping_metric='mae',early_stopping_rounds=80,
        inner_validation='last 15% complete days, common across choices',sample_weights='mean one within each actual fitting window',
        refit_rounds='best_iteration+1, no minimum',fallback='baseline if it wins past validation',
        disabled_features=disabled,source_code_sha256=sha(__file__),xgboost_version=xgb.__version__,
        audit_is_pristine=False,production_promotion=False)
    protocol['test_cohort']='exact fixed reference games; no extra early months'
    protocol_path=folder/'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())==protocol
    else:
        with protocol_path.open('x') as f:
            json.dump(protocol,f,indent=2)
    if args.phase=='audit':
        lock=json.loads((folder/'development_lock.json').read_text())
        assert lock['protocol_sha256']==sha(protocol_path) and lock['development_sha256']==sha(folder/'development.json')
        assert lock['development_gate_pass'], 'Audit blocked: development did not meet gates'
    path=folder/f'{args.phase}.json'
    report=dict(protocol=protocol,reference_sha256=sha(source),folds=[],predictions=[])
    if path.exists():
        if not args.resume:
            raise ValueError('Use explicit --resume for an identical incomplete protocol')
        report=json.loads(path.read_text())
        assert report['protocol']==protocol and report['reference_sha256']==sha(source)
        if 'summary' in report:
            raise ValueError('Completed report is immutable')
    seasons={'2023-24','2024-25'} if args.phase=='development' else {'2025-26'}
    for month,train_mask,test_mask in monthly_folds(data.sample_info,seasons):
        test_mask &= np.array([row['nba_game_id'] in ref_by_id for row in data.sample_info])
        if not test_mask.any():
            continue
        if any(f['month']==month for f in report['folds']):
            continue
        past=subset_arrays(data,train_mask);test=subset_arrays(data,test_mask)
        assert max(map(nba_day,past.sample_info))<min(map(nba_day,test.sample_info))
        prediction,details=fit_fold(past,test)
        for index,(info,p) in enumerate(zip(test.sample_info,prediction)):
            original=ref_by_id[info['nba_game_id']]
            assert original['fold']==month
            assert original['actual'][2] == float(test.y_total[index])
            report['predictions'].append(original|dict(prediction=[None,None,float(p)],
                reference_baseline_total=original.get('baseline_total'),
                baseline_total=float(test.X[index,FEATURE_NAMES.index('shrunk_expected_total')])))
        report['folds'].append(dict(month=month,**details))
        write(path,report)
        print(json.dumps(dict(phase=args.phase,month=month,samples=len(report['predictions']),
            chosen=details['chosen'],elapsed_at=datetime.now(timezone.utc).isoformat())),flush=True)
    report['summary']=score(reference,report['predictions'])
    write(path,report)
    if args.phase=='development':
        lock_path=folder/'development_lock.json'
        lock=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(protocol_path),
                  development_sha256=sha(path),development_gate_pass=report['summary']['development_gate_pass'])
        with lock_path.open('x') as f:
            json.dump(lock,f,indent=2)
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--feature-cache-key',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['development','audit'],required=True);p.add_argument('--resume',action='store_true')
    main(p.parse_args())
