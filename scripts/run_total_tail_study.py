"""One locked shallow residual candidate; development only, never promotion.

The actual-total tail is a postgame diagnostic, never a fitting weight or input.
Run lock before diagnose or develop; completed outputs cannot be overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from ml.src.config import ml_settings
from ml.src.evaluation import complete_day_split, monthly_folds, nba_day, paired_day_bootstrap, regression_metrics
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.training_policy import normalize_fit_weights, policy_mask_and_weights, subset_arrays


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def protocol(root):
    refs = dict(original=root/'totals_time_v9_20261029/development_direct.json',
                direct=root/'complete_history_20261104/development_direct.json',
                residual=root/'complete_history_20261104/development_residual.json')
    parameters = XGBoostTrainer()._common_params() | dict(n_jobs=2)
    return dict(kind='prespecified_total_development_study', targets=['total'],
        feature_cache_key='bbeb026e149ba7b9', feature_dim=213, feature_semantics=9,
        feature_cache_sha256='f1a859fe94ab439f2edf68f824456f67e4df7ee904a83293b47b7380ccaafbdb',
        references={name: dict(path=str(path), sha256=sha(path)) for name,path in refs.items()},
        source_sha256=sha(__file__), xgboost_version=xgb.__version__,
        cohort='exact original 1516 development games, labels, folds and slice flags',
        development_seasons=['2023-24','2024-25'], audit_used=False,
        candidate=dict(name='residual_T1_depth3', policy='T1', max_depth=3,
            parameters=parameters, objective='reg:squarederror', stopping_metric='rmse',
            early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
            intervention='only max_depth 5 -> 3; same residual baseline and T1 window'),
        stopping='last 15% complete NBA days within policy window; best_iteration+1; full past refit',
        weights='mean one at each actual fit; test metrics unweighted',
        disabled_features=[f'{side}_{name}' for side in ('home','away')
            for name in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES],
        diagnostic_slices=dict(early_season='frozen reference flag',
            high_actual_total='actual total >=250; postgame conditioning only',
            high_pregame_baseline='shrunk pace/efficiency baseline >=240'),
        constraints='no manual-win record, no market-target fitting, no tail oversampling or posthoc score offset',
        gates=dict(mae_improvement_min=.02, rmse_degradation_max=.01,
            slice_degradation_max=.05, slice_min_samples=100, paired_day_ci_upper_lt=0),
        selection='one candidate versus original pipeline, also report current residual T1; no adaptive grid',
        failure='retain current research result; new experiments require new prespecification',
        historical_audit='not executed here; 2025-26 is previously studied, not pristine',
        formal_pregame_games_required=200, production_promotion=False)


def metrics(reference, rows):
    assert [(r['nba_game_id'],r['actual']) for r in reference] == [(r['nba_game_id'],r['actual']) for r in rows]
    actual=np.array([r['actual'][2] for r in rows])
    predicted=np.array([r['prediction'][2] for r in rows])
    old=np.array([r['prediction'][2] for r in reference])
    before,after=abs(old-actual),abs(predicted-actual)
    slices={}; worst=0.
    for name,mask,gate in (
        ('early_season',np.array([bool(r.get('early_season')) for r in rows]),True),
        ('playoffs',np.array([r.get('game_type')=='playoffs' for r in rows]),True),
        ('major_injury',np.array([bool(r.get('major_injury')) for r in rows]),True),
        ('incomplete_stats',np.array([not r.get('team_stats_complete',False) for r in rows]),True),
        ('high_actual_total',actual>=250,False),
        ('high_pregame_baseline',np.array([r['baseline_total']>=240 for r in rows]),True)):
        if not mask.any():
            slices[name]=dict(samples=0); continue
        change=float(after[mask].mean()/before[mask].mean()-1)
        slices[name]=dict(samples=int(mask.sum()),metrics=regression_metrics(actual[mask],predicted[mask]),
            reference_metrics=regression_metrics(actual[mask],old[mask]),mae_change=change,
            postgame_conditioned=name=='high_actual_total')
        if gate and mask.sum()>=100: worst=max(worst,change)
    result=regression_metrics(actual,predicted); original=regression_metrics(actual,old)
    ci=paired_day_bootstrap(rows,before,after)
    improvement=float(1-after.mean()/before.mean())
    return dict(samples=len(rows),metrics=result,reference_metrics=original,
        improvement=improvement,difference_95ci=ci,slices=slices,worst_gate_slice_degradation=worst,
        development_gate_pass=bool(improvement>=.02 and ci[1]<0 and worst<=.05
            and result['rmse']<=original['rmse']*1.01))


def fit(past,test,p,depth):
    mask,weights,weight_report=policy_mask_and_weights(past.sample_info,'T1')
    chosen=subset_arrays(past,mask); split=complete_day_split(chosen.sample_info)
    baseline=FEATURE_NAMES.index('shrunk_expected_total')
    assert np.all(chosen.X[:,baseline]>0) and np.all(test.X[:,baseline]>0)
    c=p['candidate']; params=c['parameters']|dict(max_depth=depth)
    model=xgb.XGBRegressor(**params,objective=c['objective'],eval_metric=c['stopping_metric'],
        early_stopping_rounds=c['early_stopping_rounds'])
    residual=chosen.y_total-chosen.X[:,baseline]
    model.fit(chosen.X[:split],residual[:split],
        sample_weight=normalize_fit_weights(weights[:split],split),
        eval_set=[(chosen.X[:split],residual[:split]),(chosen.X[split:],residual[split:])],
        sample_weight_eval_set=[normalize_fit_weights(weights[:split],split),np.ones(len(chosen.X)-split)],
        verbose=False)
    rounds=int(model.best_iteration)+1
    final=xgb.XGBRegressor(**(params|dict(n_estimators=rounds)),
        objective=c['objective'],eval_metric=c['stopping_metric'])
    final.fit(chosen.X,residual,sample_weight=normalize_fit_weights(weights,len(chosen.X)),verbose=False)
    return final.predict(test.X)+test.X[:,baseline],dict(rounds=rounds,weights=weight_report,
        selection_fit_until=nba_day(chosen.sample_info[split-1]),
        validation_from=nba_day(chosen.sample_info[split]),fit_until=nba_day(chosen.sample_info[-1]))


def main(args):
    root=Path(ml_settings.models_dir); folder=root/'total_tail_study_20261105'
    folder.mkdir(parents=True,exist_ok=True); file=folder/'protocol.json'
    p=protocol(root)
    if args.action=='lock':
        if file.exists(): assert json.loads(file.read_text())['protocol']==p
        else: publish(file,dict(locked_at=datetime.now(timezone.utc).isoformat(),protocol=p))
        print(json.dumps(dict(status='LOCKED',protocol_sha256=sha(file))));return
    assert json.loads(file.read_text())['protocol']==p, 'Locked code/inputs/parameters changed'
    original=json.loads(Path(p['references']['original']['path']).read_text())['predictions']['xgboost:T0']
    residual=json.loads(Path(p['references']['residual']['path']).read_text())['predictions']['xgboost:T1']
    if args.action=='diagnose':
        direct=json.loads(Path(p['references']['direct']['path']).read_text())['predictions']['xgboost:T0']
        baseline=[r|dict(prediction=[None,None,r['baseline_total']]) for r in residual]
        result=dict(protocol_sha256=sha(file),fit_models=False,audit_used=False,
            tail_conditioning_caveat='Mean forecasts regress toward central outcomes; conditional high-actual underprediction alone does not prove unconditional bias.',
            comparisons={name:metrics(original,rows) for name,rows in
                (('direct_T0',direct),('residual_T1',residual),('pace_efficiency_baseline',baseline))})
        publish(folder/'diagnosis.json',result)
        print(json.dumps({name:value['metrics'] for name,value in result['comparisons'].items()}));return
    cache=DatasetCache(); assert sha(cache.path_for(p['feature_cache_key']))==p['feature_cache_sha256']
    data=cache.load(p['feature_cache_key'],len(FEATURE_NAMES)); assert data is not None
    data.X[:,[FEATURE_NAMES.index(name) for name in p['disabled_features']]]=0
    assert not (folder/'development.json').exists(), 'Completed study cannot be overwritten'
    lookup={r['nba_game_id']:r for r in original}; prior={r['nba_game_id']:r for r in residual}
    result=dict(protocol_sha256=sha(file),predictions=[],folds=[],audit_used=False)
    for month,train_mask,test_mask in monthly_folds(data.sample_info,set(p['development_seasons'])):
        test_mask &= np.array([r['nba_game_id'] in lookup for r in data.sample_info])
        if not test_mask.any(): continue
        past,test=subset_arrays(data,train_mask),subset_arrays(data,test_mask)
        assert max(map(nba_day,past.sample_info))<min(map(nba_day,test.sample_info))
        if not result['folds']:
            control,_=fit(past,test,p,5)
            difference=float(np.max(abs(control-np.array([prior[r['nba_game_id']]['prediction'][2] for r in test.sample_info]))))
            assert difference<1e-4, ('Total-only executor does not reproduce fixed control',difference)
            result['control_reproduction_max_difference']=difference
        prediction,detail=fit(past,test,p,3)
        for i,(info,predicted) in enumerate(zip(test.sample_info,prediction)):
            ref=lookup[info['nba_game_id']]
            assert ref['fold']==month and ref['actual'][2]==float(test.y_total[i])
            result['predictions'].append(ref|dict(prediction=[None,None,float(predicted)]))
        result['folds'].append(dict(month=month,**detail))
        print(json.dumps(dict(month=month,samples=len(result['predictions']),**detail)),flush=True)
    result['summary']=metrics(original,result['predictions'])
    result['vs_current_residual_T1']=metrics(residual,result['predictions'])
    result['promotion']=False
    publish(folder/'development.json',result)
    print(json.dumps(result['summary']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['lock','diagnose','develop'])
    main(parser.parse_args())
