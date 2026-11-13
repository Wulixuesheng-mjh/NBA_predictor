"""A legacy baseline can gain observed hashes, never fictional training ranges."""
import hashlib
import json
import numpy as np
import pytest
from ml.src.release_provenance import archive_active_manifest


def artifacts(root):
    old=dict(version='old',xgboost_version='old',mlp_version='old',feature_dim=166,
             training_data_until='2026-10-21',calibration_enabled=False)
    (root/'active.json').write_text(json.dumps(old))
    xgb=root/'xgboost/old';mlp=root/'mlp/old';xgb.mkdir(parents=True);mlp.mkdir(parents=True)
    for name in ('win','spread','total'):
        (xgb/f'{name}_model.json').write_text(json.dumps({'learner':{'learner_model_param':{'num_feature':'166'}}}))
    (xgb/'metadata.json').write_text('{}');(mlp/'metadata.json').write_text('{}')
    (mlp/'model.onnx').write_bytes(b'opaque-model-test')
    np.savez(mlp/'scaler.npz',mean=np.zeros(166),std=np.ones(166))
    return old


def test_archive_preserves_active_and_unknown_fit_metadata(tmp_path):
    old=artifacts(tmp_path);before=(tmp_path/'active.json').read_bytes()
    archive=archive_active_manifest(tmp_path,legacy_semantic_version=5)
    assert archive['origin_active_manifest']==old
    assert archive['origin_active_sha256']==hashlib.sha256(before).hexdigest()
    assert archive['feature_semantic_version']==5 and len(archive['feature_names'])==166
    assert len(archive['file_hashes']['xgboost'])==4 and len(archive['file_hashes']['mlp'])==3
    assert 'fit_range' not in archive and 'validation_range' not in archive
    assert (tmp_path/'active.json').read_bytes()==before


def test_unknown_legacy_semantics_or_different_dimensions_are_rejected(tmp_path):
    artifacts(tmp_path)
    with pytest.raises(ValueError,match='explicitly verified'):
        archive_active_manifest(tmp_path)
    path=tmp_path/'xgboost/old/total_model.json'
    path.write_text(json.dumps({'learner':{'learner_model_param':{'num_feature':'213'}}}))
    with pytest.raises(ValueError,match='dimension'):
        archive_active_manifest(tmp_path,legacy_semantic_version=5)


def test_incomplete_baseline_does_not_acquire_a_valid_manifest(tmp_path):
    artifacts(tmp_path);(tmp_path/'mlp/old/scaler.npz').unlink()
    with pytest.raises(ValueError,match='Incomplete'):
        archive_active_manifest(tmp_path,legacy_semantic_version=5)
