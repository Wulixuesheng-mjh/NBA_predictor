"""Prepare a gated, win-only offline bundle with frozen-fold reproduction."""
import argparse
from datetime import datetime, timezone, date, timedelta
import hashlib,json
from pathlib import Path
import numpy as np
from app.services.release_gate_service import check_offline_gate
from ml.src.config import ml_settings
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.training_policy import subset_arrays
from ml.src.win_shadow import fit_win_policy, prepare_win_input, predict_win_bundle, schema


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    root=Path(ml_settings.models_dir);study=root/'complete_history_20261104'
    active=(root/'active.json').read_bytes()
    evidence=json.loads((study/'audit_summary.json').read_text())
    check_offline_gate(evidence,['release_gates','win'],'win')
    lock=json.loads((study/'development_lock.json').read_text())
    if sha(study/'development_lock.json')!=evidence['protocol']['development_lock_sha256']:
        raise ValueError('Development lock changed')
    if lock['selected']['win']!='direct:xgboost:T5': raise ValueError('Unplanned win policy')
    source=study/'development_direct.json'
    if sha(source)!=lock['source_sha256']['direct']: raise ValueError('Development source changed')
    data=DatasetCache().load('bbeb026e149ba7b9',len(FEATURE_NAMES))
    rows=json.loads(source.read_text())['predictions']['xgboost:T5']
    month=rows[0]['fold'];reference=[r for r in rows if r['fold']==month]
    first=min(r.get('source_game_date',r['game_date'])[:10] for r in reference)
    past=subset_arrays(data,np.array([r.get('source_game_date',r['game_date'])[:10]<first for r in data.sample_info]))
    model,details=fit_win_policy(past)
    index={r['nba_game_id']:i for i,r in enumerate(data.sample_info)}
    sample=data.X[[index[r['nba_game_id']] for r in reference]]
    reproduced=model.predict_proba(prepare_win_input(sample,schema()))[:,1]
    error=float(np.max(abs(reproduced-np.array([r['prediction'][0] for r in reference]))))
    if error>1e-5: raise ValueError(f'Shadow fit differs from locked research: {error}')
    folder=root/'win_shadows'/args.version
    folder.mkdir(parents=True,exist_ok=False)
    model,details=fit_win_policy(data)
    path=folder/'win_model.json';model.save_model(path)
    manifest=schema()|dict(version=args.version,training_data_until=details['fit_until'],fit=details,
        feature_cache_key='bbeb026e149ba7b9',cache_sha256=sha(DatasetCache().path_for('bbeb026e149ba7b9')),
        model_sha256=sha(path),audit_evidence_sha256=sha(study/'audit_summary.json'),
        role='offline_win_candidate',production_promotion=False,
        source_sha256=sha(Path(__file__).resolve().parents[1]/'ml/src/win_shadow.py'))
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    sample=data.X[np.linspace(0,len(data.X)-1,100,dtype=int)]
    now=datetime.now(timezone.utc)
    target=(date.fromisoformat(details['fit_until'])+timedelta(days=1)).isoformat()
    batch=predict_win_bundle(folder,sample,target_day=target,as_of=now)
    single=[predict_win_bundle(folder,x,target_day=target,as_of=now)[0] for x in sample]
    difference=max(abs(a['win_prob_home']-b['win_prob_home']) for a,b in zip(batch,single))
    if difference>1e-5: raise ValueError('Win shadow single/batch mismatch')
    assert (root/'active.json').read_bytes()==active
    report=dict(status='PASS',version=args.version,parity_samples=100,batch_single_error=difference,
        locked_fold_samples=len(reference),locked_fold_error=error,fit=details,active_unchanged=True,
        online_exposed=False,formal_acceptance_started=False,counts_as_real_observations=False)
    (folder/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);main(p.parse_args())
