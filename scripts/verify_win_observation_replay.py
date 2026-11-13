"""Read-only replay of real win records using their pinned weights and inputs.

Run inside the frozen win worker image with this helper mounted read-only.
No historical feature queries, score writes, clock changes, or new forecasts.
"""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from app.services.win_observation_service import checked_protocol, digest, sha
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.inference_snapshot import validate_snapshot
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.predictor import Predictor
from ml.src.win_shadow import prepare_win_input, predict_win_bundle


def main():
    root=Path(ml_settings.models_dir)
    folder=root/'win_observations/win_T5_20261104_tminus60_v1'
    p,manifest=checked_protocol(root,folder)
    active=sha(root/'active.json')
    originals={path:sha(path) for path in sorted((folder/'records').glob('*.json'))}
    old=XGBoostTrainer();old.load(p['baseline']['xgboost_version'])
    adapter=Predictor(xgb_trainer=old,release_manifest=p['baseline'])
    results=[]
    for path in originals:
        entry=json.loads(path.read_text());row=entry['payload']
        if entry['payload_sha256']!=digest(row) or row['protocol_sha256']!=digest(p):
            raise ValueError('Observation checksum or protocol mismatch')
        if row['kind']!='win_only_diagnostic' or row['acceptance_run']:
            raise ValueError('Unexpected record kind')
        start=datetime.fromisoformat(row['scheduled_start_utc'])
        as_of=datetime.fromisoformat(row['as_of']);created=datetime.fromisoformat(row['created_at'])
        if any(t.tzinfo is None for t in (start,as_of,created)):
            raise ValueError('Missing precise timestamp')
        lead=(start-as_of).total_seconds()/60
        if not 50<=lead<=60 or created>=start:
            raise ValueError('Record not created in real pregame window')
        if row['feature_names']!=FEATURE_NAMES or set(row['win_probabilities'])!={'baseline','candidate'}:
            raise ValueError('Input order or output target mismatch')
        raw=np.asarray(row['builder_vector'],dtype=float).reshape(1,-1)
        baseline=validate_snapshot(row['baseline_input'],model_name='xgboost',
            model_version=p['baseline']['xgboost_version'],manifest=p['baseline'])
        candidate=validate_snapshot(row['candidate_input'],model_name='xgboost_win',
            model_version=p['version'],manifest=manifest)
        prepared=adapter._model_features(adapter._prepare_release_features(raw),old.model_win.n_features_in_)
        if not np.array_equal(prepared,baseline) or not np.array_equal(prepare_win_input(raw,manifest),candidate):
            raise ValueError('Saved raw-to-processed transformation cannot be reproduced')
        predicted=dict(baseline=float(old.model_win.predict_proba(baseline)[0,1]),
            candidate=predict_win_bundle(root/'win_shadows'/p['version'],raw,
                target_day=row['source_game_date'],as_of=as_of)[0]['win_prob_home'])
        difference=max(abs(predicted[name]-row['win_probabilities'][name]) for name in predicted)
        if difference>1e-7:
            raise ValueError('Pinned prediction does not replay')
        results.append(dict(game_id=row['game_id'],as_of=row['as_of'],lead_minutes=lead,
            max_probability_difference=difference,baseline_dim=baseline.shape[1],candidate_dim=candidate.shape[1],
            unknown_injury_sides=[side for side,report in row['injury_snapshots'].items() if report is None]))
    if active!=sha(root/'active.json') or any(sha(path)!=value for path,value in originals.items()):
        raise ValueError('Production selector or original observation changed during replay')
    checked_protocol(root,folder)
    print(json.dumps(dict(status='PASS',verified=len(results),records=results,source_records_unchanged=True,
        primary_unchanged=True,new_forecasts_added=0,formal_acceptance_added=0,
        scope='exact input/weight replay; not feature-query reconstruction or precision approval'),indent=2))


if __name__=='__main__':
    main()
