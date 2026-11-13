"""An isolated win-only candidate. Never adapts production spread/total heads."""
import hashlib
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from ml.src.evaluation import complete_day_split, nba_day
from ml.src.training_policy import policy_mask_and_weights, subset_arrays, normalize_fit_weights
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.config import ml_settings


DISABLED=[f'{s}_{n}' for s in ('home','away') for n in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES]


def prepare_win_input(X, manifest):
    if (manifest.get('kind')!='win_only_shadow' or manifest.get('targets')!=['win']
            or manifest.get('feature_names')!=FEATURE_NAMES
            or manifest.get('feature_semantic_version')!=CACHE_FORMAT_VERSION
            or manifest.get('disabled_features')!=DISABLED):
        raise ValueError('Win-only shadow preprocessing mismatch')
    value=np.asarray(X,dtype=np.float64)
    if value.ndim==1: value=value.reshape(1,-1)
    if value.ndim!=2 or value.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(value).all():
        raise ValueError('Invalid win-only input')
    value=value.copy();value[:,[FEATURE_NAMES.index(n) for n in DISABLED]]=0
    return value


def schema():
    return dict(kind='win_only_shadow',targets=['win'],feature_names=FEATURE_NAMES,
        feature_semantic_version=CACHE_FORMAT_VERSION,disabled_features=DISABLED)


def fit_win_policy(data, policy='T5', params=None):
    mask,weights,report=policy_mask_and_weights(data.sample_info,policy)
    chosen=subset_arrays(data,mask)
    X=prepare_win_input(chosen.X,schema());split=complete_day_split(chosen.sample_info)
    common=params or (XGBoostTrainer()._common_params()|dict(n_jobs=2))
    model=xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',
        early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,**common)
    w=normalize_fit_weights(weights[:split],split)
    model.fit(X[:split],chosen.y_win[:split],sample_weight=w,
        eval_set=[(X[:split],chosen.y_win[:split]),(X[split:],chosen.y_win[split:])],
        sample_weight_eval_set=[w,np.ones(len(X)-split)],verbose=False)
    rounds=int(model.best_iteration)+1
    full=xgb.XGBClassifier(objective='binary:logistic',eval_metric='logloss',
        **(common|dict(n_estimators=rounds)))
    full.fit(X,chosen.y_win,sample_weight=normalize_fit_weights(weights,len(X)),verbose=False)
    details=dict(policy=policy,weights=report,best_rounds=rounds,parameters=common,
        fit_from=nba_day(chosen.sample_info[0]),fit_until=nba_day(chosen.sample_info[-1]),
        selection_fit_until=nba_day(chosen.sample_info[split-1]),
        validation_from=nba_day(chosen.sample_info[split]),validation_until=nba_day(chosen.sample_info[-1]),
        samples=len(X),sample_ids_sha256=hashlib.sha256(json.dumps(
            [r['nba_game_id'] for r in chosen.sample_info]).encode()).hexdigest())
    return full,details


def predict_win_bundle(folder, X, *, target_day, as_of):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if as_of.tzinfo is None or str(target_day)<=manifest['training_data_until']:
        raise ValueError('Win shadow needs aware as_of and an out-of-training target day')
    path=folder/'win_model.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['model_sha256']:
        raise ValueError('Win shadow artifact changed')
    value=prepare_win_input(X,manifest)
    model=xgb.XGBClassifier();model.load_model(path)
    if model.n_features_in_!=len(FEATURE_NAMES): raise ValueError('Win shadow model dimension mismatch')
    probabilities=model.predict_proba(value)[:,1]
    if not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1)):
        raise ValueError('Invalid shadow probabilities')
    return [dict(model_version=manifest['version'],win_prob_home=float(p),win_prob_away=float(1-p),
                 status='diagnostic_only',acceptance_run=False) for p in probabilities]
