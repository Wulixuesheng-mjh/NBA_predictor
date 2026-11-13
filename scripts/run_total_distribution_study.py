"""Locked development-only conditional residual quantiles; no release mutation."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import xgboost as xgb
from ml.src.config import ml_settings
from ml.src.evaluation import complete_day_split, monthly_folds, nba_day
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.training_policy import normalize_fit_weights, policy_mask_and_weights, subset_arrays
from scripts.run_total_tail_study import metrics

QUANTILES = (.05, .10, .50, .90, .95)
STUDY = 'total_distribution_20261105_v1'
CACHE = '4e14a9886bed1bc1'
CACHE_SHA = 'fca08d72c70946addc13ca6186fade3b625b4e859b7d46b0e6dbcf7cff81e59a'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def ordered_quantiles(raw):
    """Project tails around an unchanged independently fitted median."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 5 or not np.isfinite(raw).all():
        raise ValueError('Five finite ordered quantile columns required')
    median = raw[:, 2:3]
    return np.concatenate((np.minimum(np.sort(raw[:, :2], axis=1), median),
        median, np.maximum(np.sort(raw[:, 3:], axis=1), median)), axis=1)


def interval_metrics(actual, raw):
    actual, raw = np.asarray(actual, dtype=float), np.asarray(raw, dtype=float)
    q = ordered_quantiles(raw)
    if actual.ndim != 1 or len(actual) != len(q) or not len(actual) or not np.isfinite(actual).all():
        raise ValueError('Nonempty matching finite labels required')
    intervals = {}
    for coverage, left, right in ((.8, 1, 3), (.9, 0, 4)):
        lower, upper = q[:, left], q[:, right]
        alpha = 1 - coverage
        score = upper-lower + 2/alpha*np.maximum(lower-actual, 0) + 2/alpha*np.maximum(actual-upper, 0)
        intervals[str(coverage)] = dict(nominal_coverage=coverage,
            empirical_coverage=float(np.mean((actual >= lower) & (actual <= upper))),
            mean_width=float(np.mean(upper-lower)), interval_score=float(score.mean()))
    return dict(samples=len(actual), calibrated=False,
        scope='historical reconstructed OOS quantiles; not certified real-pregame intervals',
        raw_crossing_rate=float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1))),
        projected_crossing_rate=float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1))),
        pinball={str(a):float(np.maximum(a*(actual-q[:, i]), (a-1)*(actual-q[:, i])).mean())
            for i,a in enumerate(QUANTILES)}, intervals=intervals)


def protocol(root):
    source = Path(__file__).resolve().parents[1]
    files = [Path(__file__), source/'scripts/run_total_tail_study.py'] + [source/p for p in (
        'ml/src/config.py', 'ml/src/evaluation.py', 'ml/src/training_policy.py',
        'ml/src/models/xgboost_model.py', 'ml/src/features/dataset_cache.py',
        'ml/src/features/feature_builder.py', 'ml/src/features/player_features.py',
        'ml/src/features/injury_net_features.py')]
    references = dict(original=root/'totals_time_v9_20261029/development_direct.json',
        residual=root/'complete_history_20261104/development_residual.json')
    return dict(study=STUDY, kind='total_only_conditional_residual_quantile_study',
        feature_cache_key=CACHE, feature_cache_sha256=CACHE_SHA, feature_dim=213, feature_semantics=9,
        feature_names=FEATURE_NAMES,
        source_hashes={str(p.relative_to(source)):sha(p) for p in files},
        references={k:dict(path=str(p),sha256=sha(p)) for k,p in references.items()},
        xgboost_version=xgb.__version__, numpy_version=np.__version__,
        cohort='exact frozen original 1516 development games, labels, slice flags and monthly folds',
        development_seasons=['2023-24','2024-25'], audit_used=False,
        candidate=dict(name='quantile_residual_T1', policy='T1', quantiles=list(QUANTILES),
            objective='reg:quantileerror', eval_metric='quantile',
            parameters=XGBoostTrainer()._common_params() | dict(n_jobs=2,tree_method='hist'),
            early_stopping_rounds=ml_settings.xgboost_early_stopping_rounds,
            point_output='independently fitted q0.5 only; upper quantiles never replace point prediction'),
        stopping='per quantile, last 15% complete past NBA days within T1; exact best_iteration+1 full past refit',
        sample_weights='mean one inside each actual fit; validation and outer evaluation unweighted',
        residual_baseline='existing 48-minute shrunk pace/efficiency baseline; target remains full-game total',
        disabled_features=[f'{side}_{name}' for side in ('home','away')
            for name in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES],
        monotonicity='sort lower and upper tails separately and clip at unchanged q0.5',
        diagnostics=['frozen early-season flag','actual total >=250 (postgame only)',
            'pregame baseline >=240 (retain empty slices)','playoffs','major injury'],
        uncertainty='nominal uncalibrated 80/90 intervals, coverage, width, pinball and interval score; no in-fit calibration',
        gates=dict(mae_improvement_min=.02,rmse_degradation_max=.01,
            important_slice_degradation_max=.05,important_slice_min_samples=100,paired_day_ci_upper_lt=0),
        decision='one structural candidate, primary MAE versus original; also compare retained residual T1',
        no_adaptive_search=True, no_tail_oversampling=True, no_point_offset=True,
        live_observations_used_for_training=False, market_used_for_training=False,
        controls='reproduce frozen squared-error residual T1 on first outer fold to max abs 1e-4',
        on_failure='retain prior result, no audit or deployment; new mechanism needs new locked protocol',
        on_success='only eligible for separately locked historical audit; audit not pristine',
        formal_pregame_games_required=200, production_promotion=False)


def fit_one(past, test, p, alpha, model_path=None):
    mask, weights, report = policy_mask_and_weights(past.sample_info, 'T1')
    chosen = subset_arrays(past, mask)
    split = complete_day_split(chosen.sample_info)
    index = FEATURE_NAMES.index('shrunk_expected_total')
    if not np.all(chosen.X[:, index] > 0) or not np.all(test.X[:, index] > 0):
        raise ValueError('Baseline unavailable; cannot compare on a changed cohort')
    y = chosen.y_total-chosen.X[:, index]
    candidate = p['candidate']
    extra = dict(objective='reg:squarederror',eval_metric='rmse') if alpha is None else dict(
        objective=candidate['objective'],eval_metric=candidate['eval_metric'],quantile_alpha=alpha)
    params = candidate['parameters']
    selecting = xgb.XGBRegressor(**params,**extra,
        early_stopping_rounds=candidate['early_stopping_rounds'])
    inner_weights = normalize_fit_weights(weights[:split], split)
    selecting.fit(chosen.X[:split], y[:split], sample_weight=inner_weights,
        eval_set=[(chosen.X[:split],y[:split]),(chosen.X[split:],y[split:])],
        sample_weight_eval_set=[inner_weights,np.ones(len(chosen.X)-split)],verbose=False)
    rounds = int(selecting.best_iteration)+1
    full = xgb.XGBRegressor(**(params | dict(n_estimators=rounds)),**extra)
    full.fit(chosen.X,y,sample_weight=normalize_fit_weights(weights,len(chosen.X)),verbose=False)
    values = full.predict(test.X)+test.X[:,index]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction')
    detail = dict(rounds=rounds,weights=report,fit_from=nba_day(chosen.sample_info[0]),
        selection_fit_until=nba_day(chosen.sample_info[split-1]),
        validation_from=nba_day(chosen.sample_info[split]),validation_until=nba_day(chosen.sample_info[-1]),
        full_fit_until=nba_day(chosen.sample_info[-1]))
    if model_path is not None:
        full.save_model(model_path)
        detail |= dict(model_file=Path(model_path).name, model_sha256=sha(model_path))
    return values, detail


def distribution_summaries(rows):
    actual = np.array([r['actual'][2] for r in rows])
    raw = np.array([r['raw_total_quantiles'] for r in rows])
    subsets = dict(early_season=np.array([bool(r.get('early_season')) for r in rows]),
        high_actual_total=actual>=250,
        high_pregame_baseline=np.array([r['baseline_total']>=240 for r in rows]),
        playoffs=np.array([r.get('game_type')=='playoffs' for r in rows]))
    return dict(all=interval_metrics(actual,raw),
        slices={name:interval_metrics(actual[mask],raw[mask]) if mask.any() else dict(samples=0)
            for name,mask in subsets.items()})


def develop(root, folder, p):
    file = folder/'protocol.json'
    if json.loads(file.read_text())['protocol'] != p:
        raise ValueError('Code, inputs or parameters changed after locking')
    if (folder/'development.json').exists():
        raise ValueError('Completed study cannot be overwritten')
    cache = DatasetCache()
    if sha(cache.path_for(CACHE)) != CACHE_SHA:
        raise ValueError('Pinned feature cache changed')
    data = cache.load(CACHE,len(FEATURE_NAMES))
    if data is None:
        raise ValueError('Missing compatible pinned cache')
    data.X = data.X.copy()
    data.X[:,[FEATURE_NAMES.index(n) for n in p['disabled_features']]] = 0
    original = json.loads(Path(p['references']['original']['path']).read_text())['predictions']['xgboost:T0']
    retained = json.loads(Path(p['references']['residual']['path']).read_text())['predictions']['xgboost:T1']
    if len(original) != 1516:
        raise ValueError('Unexpected development cohort')
    lookup = {r['nba_game_id']:r for r in original}
    prior = {r['nba_game_id']:r for r in retained}
    result = dict(protocol_sha256=sha(file),predictions=[],folds=[],audit_used=False,promoted=False)
    for month,train_mask,test_mask in monthly_folds(data.sample_info,set(p['development_seasons'])):
        test_mask &= np.array([r['nba_game_id'] in lookup for r in data.sample_info])
        if not test_mask.any():
            continue
        past,test = subset_arrays(data,train_mask),subset_arrays(data,test_mask)
        if max(map(nba_day,past.sample_info))>=min(map(nba_day,test.sample_info)):
            raise ValueError('Future row in fitting window')
        destination = folder/month
        if destination.exists():
            saved = json.loads((destination/'fold.json').read_text())
            if saved['protocol_sha256'] != sha(file):
                raise ValueError('Fold belongs to another protocol')
            for detail in saved['models'].values():
                if sha(destination/detail['model_file']) != detail['model_sha256']:
                    raise ValueError('Saved model changed')
        else:
            work = Path(tempfile.mkdtemp(prefix=f'.{month}_',dir=folder))
            try:
                saved = dict(month=month,protocol_sha256=sha(file),models={},predictions=[])
                if not result['folds']:
                    control,_ = fit_one(past,test,p,None)
                    error = float(np.max(abs(control-np.array([prior[r['nba_game_id']]['prediction'][2] for r in test.sample_info]))))
                    if error >= 1e-4:
                        raise ValueError(('Squared-error control failed reproduction',error))
                    saved['control_reproduction_max_difference'] = error
                values = []
                for alpha in QUANTILES:
                    prediction,detail = fit_one(past,test,p,alpha,work/f'q_{alpha}.json')
                    saved['models'][str(alpha)] = detail
                    values.append(prediction)
                raw = np.column_stack(values)
                ordered = ordered_quantiles(raw)
                for i,info in enumerate(test.sample_info):
                    ref = lookup[info['nba_game_id']]
                    if ref['fold']!=month or ref['actual'][2]!=float(test.y_total[i]):
                        raise ValueError('Fold/label differs from frozen reference')
                    saved['predictions'].append(ref | dict(prediction=[None,None,float(ordered[i,2])],
                        raw_total_quantiles=raw[i].tolist(),total_quantiles=ordered[i].tolist()))
                publish_once(work/'fold.json',saved)
                os.rename(work,destination)
            finally:
                if work.exists():
                    shutil.rmtree(work)
        expected = [(r['nba_game_id'],float(test.y_total[i])) for i,r in enumerate(test.sample_info)]
        if [(r['nba_game_id'],r['actual'][2]) for r in saved['predictions']] != expected:
            raise ValueError('Resumed fold cohort/labels changed')
        result['predictions'].extend(saved['predictions'])
        result['folds'].append({k:v for k,v in saved.items() if k!='predictions'})
        print(json.dumps(dict(month=month,samples=len(result['predictions']),quantiles=5)),flush=True)
    result['summary'] = metrics(original,result['predictions'])
    result['vs_retained_residual_T1'] = metrics(retained,result['predictions'])
    result['distribution'] = distribution_summaries(result['predictions'])
    publish_once(folder/'development.json',result)
    print(json.dumps(dict(summary=result['summary'],distribution=result['distribution']['all'])),flush=True)


@contextmanager
def exclusive_study(folder):
    with (folder/'.research.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def main(args):
    root = Path(ml_settings.models_dir)
    folder = root/STUDY
    folder.mkdir(parents=True,exist_ok=True)
    with exclusive_study(folder):
        p = protocol(root)
        if args.action == 'lock':
            file = folder/'protocol.json'
            if file.exists():
                if json.loads(file.read_text())['protocol'] != p:
                    raise ValueError('Existing protocol is immutable')
            else:
                if sha(DatasetCache().path_for(CACHE)) != CACHE_SHA:
                    raise ValueError('Pinned cache checksum mismatch')
                publish_once(file,dict(locked_at_utc=datetime.now(timezone.utc).isoformat(),protocol=p))
            print(json.dumps(dict(status='LOCKED',protocol_sha256=sha(file))))
        else:
            develop(root,folder,p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['lock','develop'])
    main(parser.parse_args())
