"""One locked forecast-season anchor intervention across five frozen T1 rules."""
import argparse,fcntl,json,os,shutil,tempfile
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ml.src.config import ml_settings
from ml.src.forecast_window import forecast_mask_and_weights
from ml.src.training_policy import policy_mask_and_weights,subset_arrays
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.evaluation import monthly_folds,nba_day
from scripts import run_expanded_model_comparison as shared
from scripts import run_total_only_mlp_study as focused

STUDY='forecast_season_anchor_20261105_v1'
RULES=('xgb_residual_T1','xgb_median_T1','mlp_shared_T1','mlp_total_direct_T1','mlp_total_residual_T1')


def source_for(name):
    return 'total_only_mlp_20261105_v1' if name.startswith('mlp_total_') else 'expanded_model_comparison_20261105_v1'


def fold_context(data,test):
    target=min(nba_day(r) for r in test.sample_info);seasons={r['season'] for r in test.sample_info}
    if len(seasons)!=1:raise ValueError('Fold crosses recorded NBA seasons')
    return dict(forecast_day=target,forecast_season=next(iter(seasons)),cutoff_day=str(date.fromisoformat(target)-timedelta(days=1)))


def load(root):
    cache=DatasetCache()
    if shared.sha(cache.path_for(shared.CACHE))!=shared.CACHE_SHA:raise ValueError('Pinned cache changed')
    data=cache.load(shared.CACHE,len(FEATURE_NAMES));rows=shared.cohort(data)
    sources={s:json.loads((root/s/'development.json').read_text()) for s in {source_for(n) for n in RULES}}
    return data,rows,sources


def changes(data,rows):
    ids={r['nba_game_id'] for r in rows};report=[]
    for month,past_mask,test_mask in monthly_folds(data.sample_info,{'2023-24','2024-25'}):
        test_mask &= np.array([r['nba_game_id'] in ids for r in data.sample_info])
        if not test_mask.any():continue
        past,test=subset_arrays(data,past_mask),subset_arrays(data,test_mask);ctx=fold_context(data,test)
        old_mask,old_w,old=policy_mask_and_weights(past.sample_info,'T1')
        new_mask,new_w,new=forecast_mask_and_weights(past.sample_info,'T1',**ctx)
        same_mask=np.array_equal(old_mask,new_mask)
        different=not same_mask or not np.allclose(old_w,new_w,rtol=0,atol=1e-12) if same_mask else True
        report.append(dict(month=month,changed=different,test_games=len(test.X),context=ctx,
            old_window=old,new_window=new,old_mask_sha256=shared.digest(old_mask.tolist()),new_mask_sha256=shared.digest(new_mask.tolist())))
    return report


def protocol(root,data,rows):
    source=Path(__file__).resolve().parents[1]
    files=[Path(__file__)]+[source/p for p in ('ml/src/forecast_window.py','ml/src/training_policy.py',
        'ml/src/config.py','ml/src/models/mlp_model.py','ml/src/models/xgboost_model.py','ml/src/evaluation.py',
        'scripts/run_expanded_model_comparison.py','scripts/run_total_only_mlp_study.py')]
    references={str(root/s/name):shared.sha(root/s/name) for s in {source_for(n) for n in RULES} for name in ('protocol.json','development.json')}
    return dict(study=STUDY,kind='forecast_season_window_anchor_intervention',cache_key=shared.CACHE,
        versions=dict(torch=torch.__version__,numpy=np.__version__,xgboost=shared.xgb.__version__),
        cache_sha256=shared.CACHE_SHA,feature_names=FEATURE_NAMES,cohort_sha256=shared.digest(rows),samples=2631,
        source_hashes={str(p.relative_to(source)):shared.sha(p) for p in files},references=references,
        rules=list(RULES),intervention='T1 membership from explicit target season; age from explicit prior NBA cutoff day',
        cutoff='first target NBA source day minus one day, not server date and not latest fitted game',
        fit_range='actual selected game dates saved separately from information cutoff',
        model_parameters='exactly prior frozen shared and total-only rules; no architecture or loss changes',
        legacy_policy_function_changed=False,live_records_used=False,calibration='none',audit_used=False,
        reuse='reuse old points/artifacts only if selected rows and weights are identical; retrain changed folds only',
        changes=changes(data,rows),metrics='all common raw predictions; per-season early, opening-month, tail diagnostics',
        gates=dict(primary='unchanged xgb_direct_T0',improvement_min=.02,rmse_degradation_max=.01,
            slice_min=100,slice_degradation_max=.05,ci_upper_lt=0),
        previous_failures_preserved=True,production_promotion=False)


def run(root,folder,p,data,rows,sources):
    if json.loads((folder/'protocol.json').read_text())['protocol']!=p:raise ValueError('Frozen anchor protocol changed')
    if (folder/'development.json').exists():raise ValueError('Completed study immutable')
    parent=json.loads((root/'expanded_model_comparison_20261105_v1/protocol.json').read_text())['protocol']
    data.X=data.X.copy();data.X[:,[FEATURE_NAMES.index(n) for n in parent['disabled_features']]]=0
    ids={r['nba_game_id']:r for r in rows};parity={rows[i]['nba_game_id'] for i in np.linspace(0,len(rows)-1,100,dtype=int)}
    planned={f['month']:f for f in p['changes']}
    result=dict(protocol_sha256=shared.sha(folder/'protocol.json'),predictions={n:[] for n in RULES},folds=[],promoted=False,audit_used=False)
    for month,pm,tm in monthly_folds(data.sample_info,{'2023-24','2024-25'}):
        tm &= np.array([r['nba_game_id'] in ids for r in data.sample_info])
        if not tm.any():continue
        past,test=subset_arrays(data,pm),subset_arrays(data,tm);plan=planned[month];ctx=plan['context']
        def anchored(info,policy,**kwargs):
            if kwargs:raise ValueError('Unexpected legacy cutoff override')
            return forecast_mask_and_weights(info,policy,**ctx)
        if not plan['changed']:
            detail={};predictions={}
            for name in RULES:
                old=sources[source_for(name)];fold=next(f for f in old['folds'] if f['month']==month)
                for rel,expected in fold['models'][name]['artifacts'].items():
                    if shared.sha(root/source_for(name)/month/rel)!=expected:raise ValueError('Reused artifact changed')
                detail[name]=dict(fold['models'][name],artifact_root=source_for(name)+'/'+month,
                    reused=True,window_anchor=plan['new_window'])
                predictions[name]=[dict(r) for r in old['predictions'][name] if r['fold']==month]
        else:
            destination=folder/month
            if destination.exists():
                saved=json.loads((destination/'fold.json').read_text())
                if saved['protocol_sha256']!=result['protocol_sha256']:raise ValueError('Wrong saved fold')
                detail,predictions=saved['models'],saved['predictions']
                for d in detail.values():
                    for f,h in d['artifacts'].items():
                        if shared.sha(destination/f)!=h:raise ValueError('Saved artifact changed')
            else:
                work=Path(tempfile.mkdtemp(prefix=f'.{month}_',dir=folder));detail={};predictions={}
                try:
                    with patch.object(shared,'policy_mask_and_weights',anchored),patch.object(focused,'policy_mask_and_weights',anchored):
                        for name in RULES:
                            single=np.array([i for i,r in enumerate(test.sample_info) if r['nba_game_id'] in parity],dtype=int)
                            if name.startswith('xgb_'):values,d=shared.fit_xgb(past,test,name,parent,work)
                            elif name=='mlp_shared_T1':values,d=shared.fit_mlp(past,test,parent,work,single)
                            else:values,d=focused.fit(past,test,name,work,single)
                            if not np.isfinite(values).all():raise ValueError('Nonfinite raw output')
                            detail[name]=dict(d,artifact_root=STUDY+'/'+month,reused=False,window_anchor=plan['new_window'])
                            predictions[name]=[ids[r['nba_game_id']]|dict(prediction=float(v),fold=month,window_policy='forecast_season_v1')
                                for r,v in zip(test.sample_info,values)]
                    shared.publish(work/'fold.json',dict(protocol_sha256=result['protocol_sha256'],month=month,models=detail,predictions=predictions))
                    os.rename(work,destination)
                finally:
                    if work.exists():shutil.rmtree(work)
        for name in RULES:
            if [(r['nba_game_id'],r['actual_total']) for r in predictions[name]]!=[(r['nba_game_id'],ids[r['nba_game_id']]['actual_total']) for r in test.sample_info]:
                raise ValueError('Mismatched target cohort')
            result['predictions'][name].extend(predictions[name])
        result['folds'].append(dict(month=month,changed=plan['changed'],models=detail))
        print(json.dumps(dict(month=month,retrained=plan['changed'],games=len(result['predictions'][RULES[0]]))),flush=True)
    baseline=sources['expanded_model_comparison_20261105_v1']['predictions']['xgb_direct_T0']
    result['metrics']={n:shared.describe(r) for n,r in result['predictions'].items()};result['comparisons']={};result['slices']={}
    for name,candidate in result['predictions'].items():
        original=sources[source_for(name)]['predictions'][name]
        comparisons={}
        for label,reference in (('vs_same_rule_old_anchor',original),('vs_xgb_direct',baseline)):
            comparison=focused.compare(candidate,reference)
            mask=np.array([r['fold'].endswith('-10') for r in candidate])
            a=np.array([abs(r['prediction']-r['actual_total']) for r in reference])[mask]
            b=np.array([abs(r['prediction']-r['actual_total']) for r in candidate])[mask]
            change=float(b.mean()/a.mean()-1)
            comparison['opening_month_mae_change']=change
            if mask.sum()>=100 and change>.05:comparison['descriptive_gate']=False
            comparisons[label]=comparison
        result['comparisons'][name]=comparisons
        result['slices'][name]={s:dict(all=shared.describe([r for r in candidate if r['season']==s]),
            early=shared.describe([r for r in candidate if r['season']==s and r['early_season']])) for s in ('2023-24','2024-25')}
        result['slices'][name]['opening_month']=shared.describe([r for r in candidate if r['fold'].endswith('-10')])
        result['slices'][name]['high_actual_total']=shared.describe([r for r in candidate if r['actual_total']>=250])
    shared.publish(folder/'development.json',result)
    print(json.dumps(dict(metrics=result['metrics'],comparisons=result['comparisons'])),flush=True)


def main(args):
    torch.set_num_threads(2);root=Path(ml_settings.models_dir);folder=root/STUDY;folder.mkdir(parents=True,exist_ok=True)
    data,rows,sources=load(root);p=protocol(root,data,rows)
    if len(rows)!=2631:raise ValueError('Unexpected cohort')
    with (folder/'.study.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='lock':
            if (folder/'protocol.json').exists():
                if json.loads((folder/'protocol.json').read_text())['protocol']!=p:raise ValueError('Immutable protocol changed')
            else:shared.publish(folder/'protocol.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),protocol=p))
            print(json.dumps(dict(status='LOCKED',protocol_sha256=shared.sha(folder/'protocol.json'),changed_folds=[r['month'] for r in p['changes'] if r['changed']])))
        else:run(root,folder,p,data,rows,sources)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','develop']);main(parser.parse_args())
