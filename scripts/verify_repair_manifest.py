"""Smoke-load the inactive repair manifest without changing active.json."""
import json,hashlib
from pathlib import Path
import numpy as np
from ml.src.predictor import load_predictor
from ml.src.features.feature_builder import FeatureVector
from ml.src.legacy_mlp_compatibility import CONTRACT_ID

root=Path('/app/ml/saved_models');before=(root/'active.json').read_bytes()
folder=Path('/out');manifest=json.loads((folder/'candidate_manifest.json').read_text())
predictor=load_predictor(release_manifest=manifest)
assert predictor.release_manifest==manifest
row=json.loads(Path('/evidence/total_root_cause_20261109/frozen_runtime_diagnosis/reconstructed_rows.jsonl').read_text().splitlines()[0])
result=predictor.predict(FeatureVector(features=np.array(row['feature_vector']),home_games_available=10,away_games_available=10))
assert abs(result.raw_outputs['xgboost']['predicted_total']-row['xgb_raw'])<1e-3
assert abs(result.raw_outputs['mlp']['predicted_total']-row['centered_total'])<1e-3
assert result.data_quality['mlp_input_compatibility']==CONTRACT_ID and result.recommended=='xgboost'
assert before==(root/'active.json').read_bytes()
out=folder/'candidate_manifest_smoke.json'
if out.exists():raise ValueError('Immutable evidence exists')
out.write_text(json.dumps(dict(status='PASS',manifest_version=manifest['version'],
    compatibility=CONTRACT_ID,mlp_output_version=result.mlp.model_version,
    active_sha256=hashlib.sha256(before).hexdigest(),primary_unchanged=True),indent=2),encoding='utf-8')
print('PASS: inactive repair manifest loads, XGB unchanged, compatibility trace preserved')
