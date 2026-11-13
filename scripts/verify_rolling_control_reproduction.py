"""Reproduce one frozen fold before using the runner for a data intervention."""
import argparse
import json
from pathlib import Path
import numpy as np
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.evaluation import monthly_folds,complete_day_split
from ml.src.training_policy import subset_arrays
from ml.src.models.xgboost_model import XGBoostTrainer


def main(args):
    source=json.loads(Path(args.reference).read_text());ref=source['predictions']['xgboost:T0']
    month=ref[0]['fold'];expected=[r for r in ref if r['fold']==month]
    data=DatasetCache().load(source['protocol']['feature_cache_key'],len(FEATURE_NAMES))
    assert data is not None
    disabled=[f'{s}_{n}' for s in ('home','away') for n in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES]
    data.X[:,[FEATURE_NAMES.index(n) for n in disabled]]=0
    _,train,test=next(f for f in monthly_folds(data.sample_info,{'2023-24','2024-25'}) if f[0]==month)
    past,holdout=subset_arrays(data,train),subset_arrays(data,test);split=complete_day_split(past.sample_info)
    model=XGBoostTrainer();params=model._common_params()
    assert params==source['protocol']['xgboost_params']
    model._common_params=lambda:params|dict(n_jobs=2)
    model.train(past.X[:split],past.y_win[:split],past.y_spread[:split],past.y_total[:split],
        past.X[split:],past.y_win[split:],past.y_spread[split:],past.y_total[split:],
        sample_weight=np.ones(split),total_strategy='direct')
    model.refit_full(past.X,past.y_win,past.y_spread,past.y_total,sample_weight=np.ones(len(past.X)))
    pred=model.predict(holdout.X)
    actual=np.array([[p.win_prob_home,p.predicted_spread,p.predicted_total] for p in pred])
    target=np.array([r['prediction'] for r in expected])
    assert [r['nba_game_id'] for r in holdout.sample_info]==[r['nba_game_id'] for r in expected]
    maximum=np.abs(actual-target).max(axis=0)
    assert maximum[0]<=1e-5 and np.all(maximum[1:]<=1e-3),maximum
    result=dict(status='PASS',month=month,samples=len(expected),max_difference=maximum.tolist(),
                no_files_saved=True,production_unchanged=True)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
