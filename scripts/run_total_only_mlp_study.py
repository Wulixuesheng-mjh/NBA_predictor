"""Research-only total target ablation, preserving frozen shared trainer code.

The network detaches unused heads; validation and safety checks address only
the trained target. A scoped factory override is confined to this offline job.
"""
import argparse,fcntl,json,os,shutil,tempfile
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ml.src.models import mlp_model
from ml.src.models.mlp_model import MLPTrainer,NBA_MLP
from ml.src.config import ml_settings
from ml.src.training_policy import policy_mask_and_weights,subset_arrays,normalize_fit_weights
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.dataset_cache import DatasetCache
from ml.src.evaluation import monthly_folds,complete_day_split,nba_day,paired_day_bootstrap
from scripts.run_expanded_model_comparison import CACHE,CACHE_SHA,cohort,sha,publish,digest,describe

STUDY='total_only_mlp_20261105_v1'
CHOICES=('mlp_total_direct_T1','mlp_total_residual_T1')


class TotalOnlyNetwork(NBA_MLP):
    def forward(self,x):
        win,spread,total=super().forward(x)
        return win.detach(),spread.detach(),total


class TotalOnlyTrainer(MLPTrainer):
    def train(self,*args,**kwargs):
        with patch.object(mlp_model,'NBA_MLP',TotalOnlyNetwork):
            return super().train(*args,**kwargs)

    def _compute_val_loss(self,val_data,win_criterion,spread_criterion,total_criterion):
        self.model.eval()
        with torch.no_grad():
            Xv,_,_,yt=val_data
            total=self.model(Xv)[2]
            return float((.35*total_criterion(total,yt)).mean().item())

    def _run_sanity_check(self,X_val):
        values=self.predict(X_val);values=values if isinstance(values,list) else [values]
        total=np.array([v.predicted_total for v in values])+self.sanity_baseline
        if (not np.isfinite(total).all() or total.min()<50 or total.max()>500 or
            np.quantile(total,.01)<180 or np.quantile(total,.99)>350):
            raise ValueError('Trained total head failed reconstructed-point safety check')


def fit(past,test,name,dest,single):
    mask,w,weights=policy_mask_and_weights(past.sample_info,'T1')
    chosen=subset_arrays(past,mask);split=complete_day_split(chosen.sample_info)
    b=FEATURE_NAMES.index('shrunk_expected_total');residual=name.endswith('residual_T1')
    y=chosen.y_total-chosen.X[:,b] if residual else chosen.y_total
    trainer=TotalOnlyTrainer(models_dir=str(dest/name),version='research_total_only',device='cpu')
    trainer.sanity_baseline=chosen.X[split:,b] if residual else np.zeros(len(chosen.X)-split)
    selected=trainer.train(chosen.X[:split],chosen.y_win[:split],chosen.y_spread[:split],y[:split],
        chosen.X[split:],chosen.y_win[split:],chosen.y_spread[split:],y[split:],
        sample_weight=normalize_fit_weights(w[:split],split))
    trainer.refit_full(chosen.X,chosen.y_win,chosen.y_spread,y,sample_weight=w,
        best_epoch=selected['best_epoch'],lr_schedule=selected['lr_schedule'])
    raw=trainer.predict(test.X);raw=raw if isinstance(raw,list) else [raw]
    value=np.array([r.predicted_total for r in raw])+(test.X[:,b] if residual else 0)
    alone=np.array([trainer.predict(test.X[i:i+1]).predicted_total+(test.X[i,b] if residual else 0) for i in single])
    difference=float(np.max(abs(alone-value[single]))) if len(single) else 0.
    if difference>1e-3 or not np.isfinite(value).all():raise ValueError('Invalid trained total output/parity')
    folder=dest/name;folder.mkdir()
    torch.save(trainer.model.state_dict(),folder/'model.pt')
    np.savez_compressed(folder/'scaler.npz',mean=trainer._scaler_mean,std=trainer._scaler_std,
        active_features=trainer._active_features,y_total_mean=trainer._y_total_mean,y_total_std=trainer._y_total_std)
    publish(folder/'metadata.json',dict(targets=['total'],input_dim=len(FEATURE_NAMES),feature_names=FEATURE_NAMES,
        input_normalization='none',total_mode='residual' if residual else 'direct',
        effective_gradient_loss_weights=[0,0,.35],auxiliary_heads_untrained=True,not_release_package=True))
    detail=dict(best_epoch=selected['best_epoch'],lr_schedule=selected['lr_schedule'],weights=weights,
        selection_fit_until=nba_day(chosen.sample_info[split-1]),validation_from=nba_day(chosen.sample_info[split]),
        full_fit_until=nba_day(chosen.sample_info[-1]),single_batch_samples=len(single),max_single_batch_difference=difference,
        artifacts={str(f.relative_to(dest)):sha(f) for f in folder.iterdir() if f.is_file()})
    return value,detail


def protocol(root,rows):
    source=Path(__file__).resolve().parents[1]
    files=[Path(__file__),source/'scripts/run_expanded_model_comparison.py',source/'ml/src/models/mlp_model.py',
        source/'ml/src/config.py',source/'ml/src/training_policy.py',source/'ml/src/evaluation.py']
    reference=root/'expanded_model_comparison_20261105_v1'
    return dict(study=STUDY,kind='total_only_mlp_target_ablation',cache_key=CACHE,cache_sha256=CACHE_SHA,
        versions=dict(torch=torch.__version__,numpy=np.__version__),
        cohort_sha256=digest(rows),samples=2631,feature_names=FEATURE_NAMES,
        source_hashes={str(p.relative_to(source)):sha(p) for p in files},
        shared_comparison_protocol_sha256=sha(reference/'protocol.json'),shared_report_sha256=sha(reference/'development.json'),
        choices=list(CHOICES),effective_gradient_weights=[0,0,.35],reference_gradient_weights=[.3,.35,.35],
        implementation='detach auxiliary head outputs in offline subclass; total-only validation and reconstructed-total safety; core source unchanged',
        normalization='existing none input BN, constant-feature mask, matched clipping and full-refit scaler',
        stopping='same T1 selected window, last 15% complete NBA days; total Huber validation only; exact best epoch/LR full refit',
        mlp_settings={k:v for k,v in ml_settings.model_dump().items() if k.startswith('mlp_') or k=='random_seed'},
        policy_semantics='unchanged existing T1 anchored to latest fitted game; no window change during target ablation',
        hypothesis='numerically stabilized shared MLP remains worse; isolate total objective and residual target',
        raw_accuracy='all 2631 raw outputs, no trimming rejected cases',
        metrics=['MAE','RMSE','bias','underprediction','tail error','per-season early error'],
        gates=dict(primary_comparator='xgb_direct_T0',improvement_min=.02,rmse_degradation_max=.01,
            slice_min=100,slice_degradation_max=.05,paired_day_ci_upper_lt=0),
        secondary_comparator='mlp_shared_T1; beating MLP alone does not justify deployment',
        scope='finite two-candidate study; no additional weight grid',audit_used=False,live_records_used=False,
        production_promotion=False,unused_heads_must_never_be_exposed=True)


def compare(rows,reference):
    if [(r['nba_game_id'],r['actual_total']) for r in rows]!=[(r['nba_game_id'],r['actual_total']) for r in reference]:
        raise ValueError('Unpaired cohort')
    y=np.array([r['actual_total'] for r in rows]);a=abs(np.array([r['prediction'] for r in reference])-y)
    b=abs(np.array([r['prediction'] for r in rows])-y);improvement=float(1-b.mean()/a.mean())
    ci=paired_day_bootstrap(rows,a,b);worst=0.
    for season in ('2023-24','2024-25'):
        for early in (False,True):
            mask=np.array([r['season']==season and (r['early_season'] if early else True) for r in rows])
            if mask.sum()>=100:worst=max(worst,float(b[mask].mean()/a[mask].mean()-1))
    return dict(improvement=improvement,difference_95ci=ci,worst_slice_degradation=worst,
        descriptive_gate=bool(improvement>=.02 and ci[1]<0 and worst<=.05 and describe(rows)['rmse']<=describe(reference)['rmse']*1.01),
        promotion_allowed=False)


def run(root,folder,p,data,rows):
    if json.loads((folder/'protocol.json').read_text())['protocol']!=p:raise ValueError('Locked inputs changed')
    if (folder/'development.json').exists():raise ValueError('Completed study immutable')
    parent=json.loads((root/'expanded_model_comparison_20261105_v1/protocol.json').read_text())['protocol']
    reference=json.loads((root/'expanded_model_comparison_20261105_v1/development.json').read_text())
    data.X=data.X.copy();data.X[:,[FEATURE_NAMES.index(n) for n in parent['disabled_features']]]=0
    lookup={r['nba_game_id']:r for r in rows};parity={rows[i]['nba_game_id'] for i in np.linspace(0,len(rows)-1,100,dtype=int)}
    result=dict(protocol_sha256=sha(folder/'protocol.json'),predictions={m:[] for m in CHOICES},folds=[],promoted=False,audit_used=False)
    for month,train_mask,test_mask in monthly_folds(data.sample_info,{'2023-24','2024-25'}):
        test_mask &= np.array([r['nba_game_id'] in lookup for r in data.sample_info])
        if not test_mask.any():continue
        past,test=subset_arrays(data,train_mask),subset_arrays(data,test_mask)
        if max(map(nba_day,past.sample_info))>=min(map(nba_day,test.sample_info)):raise ValueError('Future fitting row')
        dest=folder/month
        if dest.exists():
            saved=json.loads((dest/'fold.json').read_text())
            if saved['protocol_sha256']!=result['protocol_sha256']:raise ValueError('Wrong fold protocol')
            for detail in saved['models'].values():
                for file,value in detail['artifacts'].items():
                    if sha(dest/file)!=value:raise ValueError('Changed model artifact')
        else:
            work=Path(tempfile.mkdtemp(prefix=f'.{month}_',dir=folder))
            try:
                saved=dict(month=month,protocol_sha256=result['protocol_sha256'],models={},predictions={})
                single=np.array([i for i,r in enumerate(test.sample_info) if r['nba_game_id'] in parity],dtype=int)
                for name in CHOICES:
                    prediction,detail=fit(past,test,name,work,single)
                    saved['models'][name]=detail
                    saved['predictions'][name]=[lookup[info['nba_game_id']]|dict(prediction=float(v),fold=month)
                        for info,v in zip(test.sample_info,prediction)]
                publish(work/'fold.json',saved);os.rename(work,dest)
            finally:
                if work.exists():shutil.rmtree(work)
        for name in CHOICES:result['predictions'][name].extend(saved['predictions'][name])
        result['folds'].append({k:v for k,v in saved.items() if k!='predictions'})
        print(json.dumps(dict(month=month,games=len(result['predictions'][CHOICES[0]]),models=2)),flush=True)
    result['metrics']={m:describe(v) for m,v in result['predictions'].items()}
    result['comparisons']={m:{base:compare(v,reference['predictions'][base]) for base in ('mlp_shared_T1','xgb_direct_T0')}
        for m,v in result['predictions'].items()}
    result['slices']={m:{season:dict(all=describe([r for r in v if r['season']==season]),
        early=describe([r for r in v if r['season']==season and r['early_season']])) for season in ('2023-24','2024-25')}
        for m,v in result['predictions'].items()}
    result['tail_diagnostics']={m:describe([r for r in v if r['actual_total']>=250]) for m,v in result['predictions'].items()}
    publish(folder/'development.json',result)
    print(json.dumps(dict(metrics=result['metrics'],comparisons=result['comparisons'],slices=result['slices'])),flush=True)


def main(args):
    torch.set_num_threads(2);root=Path(ml_settings.models_dir);folder=root/STUDY;folder.mkdir(parents=True,exist_ok=True)
    cache=DatasetCache()
    if sha(cache.path_for(CACHE))!=CACHE_SHA:raise ValueError('Cache changed')
    data=cache.load(CACHE,len(FEATURE_NAMES));rows=cohort(data)
    if len(rows)!=2631 or ml_settings.mlp_input_normalization!='none':raise ValueError('Unexpected cohort/normalization')
    p=protocol(root,rows)
    with (folder/'.study.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action=='lock':
            if (folder/'protocol.json').exists():
                if json.loads((folder/'protocol.json').read_text())['protocol']!=p:raise ValueError('Immutable protocol changed')
            else:publish(folder/'protocol.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),protocol=p))
            print(json.dumps(dict(status='LOCKED',protocol_sha256=sha(folder/'protocol.json'))))
        else:run(root,folder,p,data,rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','develop']);main(parser.parse_args())
