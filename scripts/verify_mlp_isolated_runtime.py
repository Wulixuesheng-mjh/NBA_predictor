"""Run new total contract inside the original ONNX-only serving image."""
import json
from pathlib import Path
import numpy as np
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.dataset_cache import DatasetCache
from ml.src.mlp_input_contract import IsolatedMLPTotal,file_sha


def main():
    root=Path(ml_settings.models_dir);before=file_sha(root/'active.json')
    folder=root/'isolated_mlp_contracts/mlp_semantics_isolation_20261105_v1'
    runtime=IsolatedMLPTotal(folder);p=json.loads((folder/'engineering_protocol.json').read_text())
    cache=DatasetCache();key=p['probe_cache_key']
    if file_sha(cache.path_for(key))!=p['probe_cache_sha256']:raise ValueError('Probe cache changed')
    data=cache.load(key,len(FEATURE_NAMES));index={r['nba_game_id']:i for i,r in enumerate(data.sample_info)}
    ids=p['probe_ids'];X=data.X[[index[i] for i in ids]].astype(float)
    args=dict(feature_names=FEATURE_NAMES,semantic_version=9,input_stage='raw_feature_builder')
    batch=runtime.predict_total(X,**args);values=np.array([r['predicted_total'] for r in batch])
    singles=np.array([runtime.predict_total(x,**args)[0]['predicted_total'] for x in X])
    fit=json.loads((folder/'source_fit.json').read_text());study,month,rule=fit['source_model'].split('/')
    source=root/study/month/'fold.json'
    if file_sha(source)!=p['source_fold_sha256']:raise ValueError('Source fold changed')
    expected=json.loads(source.read_text())['predictions'][rule];positions={i:n for n,i in enumerate(ids)}
    error=max(abs(values[positions[r['nba_game_id']]]-r['prediction']) for r in expected)
    if error>1e-3 or float(abs(values-singles).max())>1e-3:raise ValueError('ONNX-only runtime replay failed')
    assert all(r['targets']==['total'] and r['status']=='isolated_research_only' for r in batch)
    assert before==file_sha(root/'active.json')
    print(json.dumps(dict(status='PASS',scope='read_only_ONNX_only_isolated_runtime',samples=len(X),
        source_fold_replayed=len(expected),source_fold_max_difference=float(error),
        single_batch_difference=float(abs(values-singles).max()),untrained_heads_exposed=False,
        primary_unchanged=True,promoted=False,formal_acceptance_added=0),indent=2))


if __name__=='__main__':main()
