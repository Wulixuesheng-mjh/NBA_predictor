"""Prespecified low-complexity residual benchmark; lock on development, audit once."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.training_policy import policy_mask_and_weights,subset_arrays
from ml.src.evaluation import monthly_folds,complete_day_split,regression_metrics,paired_day_bootstrap

NAMES=['shrunk_expected_total','expected_pace','home_avg_pts','away_avg_pts',
       'home_avg_pts_allowed','away_avg_pts_allowed','home_avg_total_last3','away_avg_total_last3',
       'home_bayes_off_rtg','away_bayes_off_rtg','home_bayes_def_rtg','away_bayes_def_rtg',
       'home_avg_fg3a_rate','away_avg_fg3a_rate','home_trend_pts','away_trend_pts']


def evaluate(data,policies,seasons):
    indices=[FEATURE_NAMES.index(n) for n in NAMES];bi=FEATURE_NAMES.index('shrunk_expected_total')
    outputs={p:[] for p in policies};folds=[]
    for month,train,test in monthly_folds(data.sample_info,seasons):
        past=subset_arrays(data,train)
        for policy in policies:
            mask,w,weights=policy_mask_and_weights(past.sample_info,policy)
            chosen=subset_arrays(past,mask);split=complete_day_split(chosen.sample_info)
            X=chosen.X[:,indices];base=chosen.X[:,bi];y=chosen.y_total-base
            selection_w=w[:split]/w[:split].mean()
            scaler=StandardScaler().fit(X[:split],sample_weight=selection_w)
            trials=[]
            for alpha in (10.,100.,1000.):
                fit=Ridge(alpha=alpha).fit(scaler.transform(X[:split]),y[:split],sample_weight=selection_w)
                mae=np.mean(abs(fit.predict(scaler.transform(X[split:]))+base[split:]-chosen.y_total[split:]))
                trials.append((float(mae),alpha))
            alpha=min(trials)[1]
            scaler=StandardScaler().fit(X,sample_weight=w)
            model=Ridge(alpha=alpha).fit(scaler.transform(X),y,sample_weight=w)
            prediction=model.predict(scaler.transform(data.X[test][:,indices]))+data.X[test,bi]
            info=[r for r,keep in zip(data.sample_info,test) if keep]
            actual=data.y_total[test]
            outputs[policy].extend(dict(**r,predicted_total=float(p),actual_total=float(a),fold=month)
                                   for r,p,a in zip(info,prediction,actual))
            folds.append(dict(month=month,policy=policy,alpha=alpha,weights=weights,
                coefficients=model.coef_.tolist(),intercept=float(model.intercept_),
                scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist()))
    return dict(predictions=outputs,folds=folds,metrics={p:regression_metrics(
        np.array([r['actual_total'] for r in rows]),np.array([r['predicted_total'] for r in rows]))
        for p,rows in outputs.items()})


def main(args):
    path=Path(args.output)
    if path.exists():
        raise ValueError('Do not overwrite a completed benchmark or retune after audit')
    data=DatasetCache().load(args.feature_cache_key,len(FEATURE_NAMES))
    assert data is not None and np.all(data.X[:,FEATURE_NAMES.index('shrunk_expected_total')]>0)
    report=dict(protocol=dict(feature_cache_key=args.feature_cache_key,feature_names=NAMES,
        objective='ridge residual around shrunk pace-efficiency baseline',
        alphas=[10,100,1000],policies=['T0','T4','T5'],standardization='past training only',
        monthly_fixed=True,audit_retuning=False,promote=False))
    report['development']=evaluate(data,['T0','T4','T5'],{'2023-24','2024-25'})
    selected=min(report['development']['metrics'],key=lambda p:report['development']['metrics'][p]['mae'])
    report['selected_on_development']=selected
    path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix('.development.lock.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    report['audit']=evaluate(data,[selected],{'2025-26'})
    comparison={}
    for phase in ('development','audit'):
        direct=json.loads((Path(args.reference)/f'{phase}_direct.json').read_text())['predictions']['xgboost:T0']
        rows=report[phase]['predictions'][selected]
        assert [(r['nba_game_id'],r['actual_total']) for r in rows]==[(r['nba_game_id'],r['actual'][2]) for r in direct]
        a=np.array([abs(r['prediction'][2]-r['actual'][2]) for r in direct])
        b=np.array([abs(r['predicted_total']-r['actual_total']) for r in rows])
        comparison[phase]=dict(improvement=float((a.mean()-b.mean())/a.mean()),
            difference_95ci=paired_day_bootstrap(rows,a,b))
    report['comparison_vs_direct_T0']=comparison
    report['decision']=dict(promote=False,requires_full_slice_and_engineering_gates=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(selected=selected,development=report['development']['metrics'],
                         audit=report['audit']['metrics'],comparisons=comparison),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature-cache-key',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
