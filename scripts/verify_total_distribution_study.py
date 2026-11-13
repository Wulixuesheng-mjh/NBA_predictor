"""Replay every locked OOS quantile using saved research model artifacts."""
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from ml.src.config import ml_settings
from ml.src.evaluation import nba_day
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from scripts.run_total_distribution_study import STUDY, protocol, sha, publish_once, ordered_quantiles


def main():
    root=Path(ml_settings.models_dir); folder=root/STUDY
    active=sha(root/'active.json')
    locked=json.loads((folder/'protocol.json').read_text())
    p=locked['protocol']
    if protocol(root)!=p:
        raise ValueError('Locked source, references or parameters changed')
    result=json.loads((folder/'development.json').read_text())
    protocol_sha=sha(folder/'protocol.json')
    if result['protocol_sha256']!=protocol_sha:
        raise ValueError('Development report from another protocol')
    cache=DatasetCache()
    if sha(cache.path_for(p['feature_cache_key']))!=p['feature_cache_sha256']:
        raise ValueError('Pinned cache changed')
    data=cache.load(p['feature_cache_key'],len(FEATURE_NAMES))
    indexes={r['nba_game_id']:i for i,r in enumerate(data.sample_info)}
    disabled=[FEATURE_NAMES.index(n) for n in p['disabled_features']]
    baseline=FEATURE_NAMES.index('shrunk_expected_total')
    rows=result['predictions']
    if len(rows)!=1516 or len({r['nba_game_id'] for r in rows})!=len(rows):
        raise ValueError('Unexpected or duplicated OOS cohort')
    single_ids={rows[i]['nba_game_id'] for i in np.linspace(0,len(rows)-1,100,dtype=int)}
    maximum=0.; parity=0.; models=0; replayed=0
    for fold in result['folds']:
        month=fold['month']; group=[r for r in rows if r['fold']==month]
        saved=json.loads((folder/month/'fold.json').read_text())
        if saved['predictions']!=group or saved['protocol_sha256']!=protocol_sha:
            raise ValueError('Fold checkpoint differs from completed report')
        index=[indexes[r['nba_game_id']] for r in group]
        X=data.X[index].copy(); X[:,disabled]=0
        if [r['actual'][2] for r in group]!=data.y_total[index].tolist():
            raise ValueError('Outer labels differ from frozen cache')
        selected=[i for i,r in enumerate(group) if r['nba_game_id'] in single_ids]
        values=[]
        for name,detail in fold['models'].items():
            file=folder/month/detail['model_file']
            if sha(file)!=detail['model_sha256'] or saved['models'][name]!=detail:
                raise ValueError('Saved quantile artifact changed')
            if detail['full_fit_until']>=min(nba_day(r) for r in group):
                raise ValueError('Outer label inside model fitting range')
            model=xgb.XGBRegressor(n_jobs=1); model.load_model(file)
            prediction=model.predict(X)+X[:,baseline]
            if len(selected):
                single=np.array([model.predict(X[i:i+1])[0]+X[i,baseline] for i in selected])
                maximum=max(maximum,float(np.max(abs(single-prediction[selected]))))
            values.append(prediction); models+=1
        raw=np.column_stack(values)
        expected=np.asarray([r['raw_total_quantiles'] for r in group])
        processed=ordered_quantiles(raw)
        maximum=max(maximum,float(np.max(abs(raw-expected))),
            float(np.max(abs(processed-np.asarray([r['total_quantiles'] for r in group])))))
        if not np.array_equal(processed[:,2],np.asarray([r['prediction'][2] for r in group])):
            raise ValueError('Tail projection changed point prediction')
        parity+=len(selected); replayed+=len(group)
    if maximum>1e-5 or parity!=100:
        raise ValueError(('Model replay or single/batch parity failed',maximum,parity))
    if sha(root/'active.json')!=active:
        raise ValueError('Primary changed during research verification')
    report=dict(status='PASS',study=STUDY,protocol_sha256=protocol_sha,
        out_of_sample_games_replayed=replayed,models_verified=models,
        single_batch_parity_games=parity,max_score_difference=maximum,
        median_preserved=True,feature_dim=len(FEATURE_NAMES),primary_unchanged=True,
        formal_acceptance_added=0,promoted=False,
        scope='research artifact reproducibility; not calibration or production release approval')
    publish_once(folder/'validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
