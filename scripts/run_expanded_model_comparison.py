"""Frozen-rule full-two-season development comparison, no production weights.

Prior fixed-cohort failures remain failures; a larger cohort cannot promote a
previous candidate automatically. All compared rules refit only past data.
"""
import argparse,fcntl,hashlib,json,os,shutil,tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from ml.src.config import ml_settings
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.models.mlp_model import MLPTrainer
from ml.src.training_policy import policy_mask_and_weights,subset_arrays,normalize_fit_weights
from ml.src.evaluation import monthly_folds,complete_day_split,nba_day,regression_metrics,paired_day_bootstrap

STUDY='expanded_model_comparison_20261105_v1'
CACHE='4e14a9886bed1bc1'
CACHE_SHA='fca08d72c70946addc13ca6186fade3b625b4e859b7d46b0e6dbcf7cff81e59a'
MODELS=('xgb_direct_T0','xgb_residual_T1','xgb_median_T1','mlp_shared_T1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def publish(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def cohort(data):
    rows=[]
    for i,info in enumerate(data.sample_info):
        if info['season'] not in ('2023-24','2024-25'):continue
        rows.append(dict(**info,index=i,early_season=bool(
            data.X[i,FEATURE_NAMES.index('home_season_games_played')]<10 or
            data.X[i,FEATURE_NAMES.index('away_season_games_played')]<10),
            actual_total=float(data.y_total[i])))
    return rows


def protocol(root,data,rows):
    source=Path(__file__).resolve().parents[1]
    files=[Path(__file__)]+[source/p for p in ('ml/src/models/mlp_model.py','ml/src/models/xgboost_model.py',
        'ml/src/config.py','ml/src/training_policy.py','ml/src/evaluation.py','ml/src/features/feature_builder.py',
        'ml/src/features/dataset_cache.py','ml/src/features/player_features.py','ml/src/features/injury_net_features.py')]
    return dict(kind='expanded_common_development_cohort',study=STUDY,feature_cache_key=CACHE,
        cache_sha256=CACHE_SHA,feature_names=FEATURE_NAMES,feature_semantics=9,
        source_hashes={str(p.relative_to(source)):sha(p) for p in files},
        versions=dict(torch=torch.__version__,xgboost=xgb.__version__,numpy=np.__version__),
        cohort_rule='all eligible cached games in both development seasons; no outcome filtering',
        cohort_sha256=digest(rows),samples=len(rows),models=list(MODELS),
        xgb_params=XGBoostTrainer()._common_params()|dict(n_jobs=2,tree_method='hist'),
        xgb_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
        mlp_settings={k:v for k,v in ml_settings.model_dump().items() if k.startswith('mlp_') or k=='random_seed'},
        mlp_rule='existing stabilized 213-dim shared network, input normalization none, constant-column mask, train/serve clipping, full refit',
        mlp_loss_weights=[.3,.35,.35],mlp_loss_weight_search=False,mlp_total_only_search=False,
        reason='production numerical instability diagnosed first; evaluate corrected existing shared rule before adding loss/structure candidates',
        disabled_features=[f'{s}_{n}' for s in ('home','away') for n in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES],
        folds='monthly NBA days; last 15% complete past days inside policy window choose stop, full allowed past refit',
        policies=dict(xgb_direct_T0='T0',xgb_residual_T1='T1',xgb_median_T1='T1',mlp_shared_T1='T1'),
        metrics='unweighted raw total MAE,RMSE,bias,p90,underprediction and guard counts; no dropping bad model outputs',
        slices=['each full season','early season within each season','actual total >=250 diagnostic only','playoffs'],
        paired_bootstrap='2000 resamples of NBA days, seed42',
        descriptive_gates=dict(mae_improvement_min=.02,rmse_degradation_max=.01,slice_min=100,slice_degradation_max=.05,ci_upper_lt=0),
        prior_fixed_cohort='1516 games; preserve all failed gates and report exact overlapping predictions separately',
        original_model_weights_used_for_historical_accuracy=False,live_records_used=False,audit_used=False,
        promotion=False,expanded_results_cannot_override_prior_failed_gates=True)


def fit_xgb(past,test,name,p,destination):
    policy=p['policies'][name];mask,w,weights=policy_mask_and_weights(past.sample_info,policy)
    chosen=subset_arrays(past,mask);split=complete_day_split(chosen.sample_info)
    index=FEATURE_NAMES.index('shrunk_expected_total');residual=name!='xgb_direct_T0'
    target=chosen.y_total-chosen.X[:,index] if residual else chosen.y_total
    extra=dict(objective='reg:quantileerror',quantile_alpha=.5,eval_metric='quantile') if name=='xgb_median_T1' else dict(objective='reg:squarederror',eval_metric='rmse')
    model=xgb.XGBRegressor(**p['xgb_params'],**extra,early_stopping_rounds=p['xgb_stopping_rounds'])
    sw=normalize_fit_weights(w[:split],split)
    model.fit(chosen.X[:split],target[:split],sample_weight=sw,
        eval_set=[(chosen.X[:split],target[:split]),(chosen.X[split:],target[split:])],
        sample_weight_eval_set=[sw,np.ones(len(chosen.X)-split)],verbose=False)
    rounds=int(model.best_iteration)+1
    full=xgb.XGBRegressor(**(p['xgb_params']|dict(n_estimators=rounds)),**extra)
    full.fit(chosen.X,target,sample_weight=normalize_fit_weights(w,len(w)),verbose=False)
    file=destination/f'{name}.json';full.save_model(file)
    prediction=full.predict(test.X)+(test.X[:,index] if residual else 0)
    return prediction,dict(rounds=rounds,weights=weights,selection_fit_until=nba_day(chosen.sample_info[split-1]),
        validation_from=nba_day(chosen.sample_info[split]),full_fit_until=nba_day(chosen.sample_info[-1]),
        artifacts={file.name:sha(file)})


def fit_mlp(past,test,p,destination,single_indexes):
    mask,w,weights=policy_mask_and_weights(past.sample_info,'T1')
    chosen=subset_arrays(past,mask);split=complete_day_split(chosen.sample_info)
    model=MLPTrainer(models_dir=str(destination/'mlp'),version='research_only',device='cpu')
    selected=model.train(chosen.X[:split],chosen.y_win[:split],chosen.y_spread[:split],chosen.y_total[:split],
        chosen.X[split:],chosen.y_win[split:],chosen.y_spread[split:],chosen.y_total[split:],
        sample_weight=normalize_fit_weights(w[:split],split))
    model.refit_full(chosen.X,chosen.y_win,chosen.y_spread,chosen.y_total,
        sample_weight=w,best_epoch=selected['best_epoch'],lr_schedule=selected['lr_schedule'])
    batch=model.predict(test.X);batch=[batch] if not isinstance(batch,list) else batch
    values=np.array([r.predicted_total for r in batch])
    single=[model.predict(test.X[i:i+1]).predicted_total for i in single_indexes]
    difference=float(np.max(abs(np.asarray(single)-values[single_indexes]))) if len(single_indexes) else 0.
    if difference>1e-3:raise ValueError(('MLP single/batch mismatch',difference))
    folder=destination/'mlp';folder.mkdir(exist_ok=True)
    torch.save(model.model.state_dict(),folder/'model.pt')
    np.savez_compressed(folder/'scaler.npz',mean=model._scaler_mean,std=model._scaler_std,
        active_features=model._active_features,y_total_mean=model._y_total_mean,y_total_std=model._y_total_std,
        y_spread_mean=model._y_spread_mean,y_spread_std=model._y_spread_std)
    metadata=dict(kind='research_shared_mlp',input_dim=len(FEATURE_NAMES),input_normalization='none',
        feature_names=FEATURE_NAMES,loss_weights=[.3,.35,.35],not_release_package=True)
    publish(folder/'metadata.json',metadata)
    return values,dict(best_epoch=selected['best_epoch'],lr_schedule=selected['lr_schedule'],weights=weights,
        selection_fit_until=nba_day(chosen.sample_info[split-1]),validation_from=nba_day(chosen.sample_info[split]),
        full_fit_until=nba_day(chosen.sample_info[-1]),single_batch_samples=len(single_indexes),
        max_single_batch_difference=difference,inactive_columns=int(np.sum(~model._active_features)),
        artifacts={str(f.relative_to(destination)):sha(f) for f in folder.iterdir() if f.is_file()})


def describe(rows):
    y=np.array([r['actual_total'] for r in rows]);v=np.array([r['prediction'] for r in rows])
    return dict(samples=len(rows),**regression_metrics(y,v),underprediction_rate=float(np.mean(v<y)),
        total_guard_rejections=int(np.sum((v<150)|(v>300)))) if len(rows) else dict(samples=0)


def run(root,folder,p,data,rows):
    if (folder/'development.json').exists():raise ValueError('Completed experiment immutable')
    locked=json.loads((folder/'protocol.json').read_text())
    if locked['protocol']!=p:raise ValueError('Protocol/code/cache/config changed')
    indexes={r['nba_game_id']:r for r in rows};data.X=data.X.copy()
    data.X[:,[FEATURE_NAMES.index(n) for n in p['disabled_features']]]=0
    parity_ids={rows[i]['nba_game_id'] for i in np.linspace(0,len(rows)-1,100,dtype=int)}
    result=dict(protocol_sha256=sha(folder/'protocol.json'),predictions={m:[] for m in MODELS},folds=[],audit_used=False,promoted=False)
    for month,train_mask,test_mask in monthly_folds(data.sample_info,{'2023-24','2024-25'}):
        test_mask &= np.array([r['nba_game_id'] in indexes for r in data.sample_info])
        if not test_mask.any():continue
        past,test=subset_arrays(data,train_mask),subset_arrays(data,test_mask)
        if max(map(nba_day,past.sample_info))>=min(map(nba_day,test.sample_info)):raise ValueError('Future row in fit')
        dest=folder/month
        if dest.exists():
            saved=json.loads((dest/'fold.json').read_text())
            if saved['protocol_sha256']!=result['protocol_sha256']:raise ValueError('Wrong fold protocol')
            for detail in saved['models'].values():
                for file,expected in detail['artifacts'].items():
                    if sha(dest/file)!=expected:raise ValueError('Changed model artifact')
        else:
            work=Path(tempfile.mkdtemp(prefix=f'.{month}_',dir=folder))
            try:
                saved=dict(month=month,protocol_sha256=result['protocol_sha256'],models={},predictions={})
                for name in MODELS:
                    single=np.array([i for i,r in enumerate(test.sample_info) if r['nba_game_id'] in parity_ids],dtype=int)
                    if name=='mlp_shared_T1':prediction,detail=fit_mlp(past,test,p,work,single)
                    else:prediction,detail=fit_xgb(past,test,name,p,work)
                    if not np.isfinite(prediction).all():raise ValueError('Nonfinite prediction; do not drop affected games')
                    saved['models'][name]=detail
                    saved['predictions'][name]=[indexes[info['nba_game_id']]|dict(prediction=float(value),fold=month)
                        for info,value in zip(test.sample_info,prediction)]
                publish(work/'fold.json',saved);os.rename(work,dest)
            finally:
                if work.exists():shutil.rmtree(work)
        for name in MODELS:result['predictions'][name].extend(saved['predictions'][name])
        result['folds'].append({k:v for k,v in saved.items() if k!='predictions'})
        print(json.dumps(dict(month=month,games=len(result['predictions'][MODELS[0]]),models=4)),flush=True)
    expected=[r['nba_game_id'] for r in rows]
    for name,values in result['predictions'].items():
        if [r['nba_game_id'] for r in values]!=expected:raise ValueError('Incomplete or mismatched expanded cohort')
    result['metrics']={m:describe(v) for m,v in result['predictions'].items()}
    result['slices']={m:{season:dict(all=describe([r for r in v if r['season']==season]),
        early=describe([r for r in v if r['season']==season and r['early_season']]))
        for season in ('2023-24','2024-25')} for m,v in result['predictions'].items()}
    baseline=result['predictions']['xgb_direct_T0'];y=np.array([r['actual_total'] for r in baseline]);a=abs(np.array([r['prediction'] for r in baseline])-y)
    result['comparisons']={}
    for m,v in result['predictions'].items():
        b=abs(np.array([r['prediction'] for r in v])-y)
        improvement=float(1-b.mean()/a.mean());ci=paired_day_bootstrap(v,a,b)
        worst=0.
        for season in ('2023-24','2024-25'):
            for early in (False,True):
                mask=np.array([r['season']==season and (r['early_season'] if early else True) for r in v])
                if mask.sum()>=100:worst=max(worst,float(b[mask].mean()/a[mask].mean()-1))
        rmse_ok=result['metrics'][m]['rmse']<=result['metrics']['xgb_direct_T0']['rmse']*1.01
        result['comparisons'][m]=dict(improvement=improvement,difference_95ci=ci,
            worst_season_or_early_slice_degradation=worst,
            expanded_descriptive_gate=bool(improvement>=.02 and ci[1]<0 and rmse_ok and worst<=.05),
            promotion_allowed=False)
    result['tail_diagnostics']={m:describe([r for r in v if r['actual_total']>=250]) for m,v in result['predictions'].items()}
    old=json.loads((root/'total_distribution_20261105_v1/development.json').read_text())
    original_ids={r['nba_game_id'] for r in old['predictions']}
    result['original_1516_overlap']={m:describe([r for r in v if r['nba_game_id'] in original_ids]) for m,v in result['predictions'].items()}
    prior={r['nba_game_id']:r['prediction'][2] for r in old['predictions']}
    overlap=[r for r in result['predictions']['xgb_median_T1'] if r['nba_game_id'] in prior]
    result['old_median_overlap_max_difference']=max(abs(r['prediction']-prior[r['nba_game_id']]) for r in overlap)
    if result['old_median_overlap_max_difference']>1e-4:raise ValueError('Frozen median rule did not reproduce old cohort')
    result['previous_failed_gate_preserved']=not old['summary']['development_gate_pass']
    publish(folder/'development.json',result)
    print(json.dumps(dict(metrics=result['metrics'],comparisons=result['comparisons'],slices=result['slices'])),flush=True)


def main(args):
    torch.set_num_threads(2)
    root=Path(ml_settings.models_dir);folder=root/STUDY;folder.mkdir(parents=True,exist_ok=True)
    cache=DatasetCache()
    if sha(cache.path_for(CACHE))!=CACHE_SHA:raise ValueError('Pinned cache changed')
    data=cache.load(CACHE,len(FEATURE_NAMES));rows=cohort(data)
    if len(rows)!=2631 or ml_settings.mlp_input_normalization!='none':raise ValueError('Unexpected cohort or MLP normalization')
    p=protocol(root,data,rows)
    with (folder/'.study.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='lock':
            if (folder/'protocol.json').exists():
                if json.loads((folder/'protocol.json').read_text())['protocol']!=p:raise ValueError('Immutable protocol changed')
            else:
                publish(folder/'cohort.json',rows)
                publish(folder/'protocol.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),protocol=p))
            print(json.dumps(dict(status='LOCKED',samples=len(rows),protocol_sha256=sha(folder/'protocol.json'))))
        else:run(root,folder,p,data,rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','develop']);main(parser.parse_args())
