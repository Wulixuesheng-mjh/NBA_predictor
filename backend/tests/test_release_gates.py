import hashlib
import json
from pathlib import Path
import importlib.util
import pytest
from app.services.release_gate_service import check_offline_gate,referenced_report,validate_offline_evidence,check_changed_targets


def test_failed_offline_improvement_cannot_be_overridden_by_pass_flag():
    report={'gates':{'candidate':dict(eligible_for_shadow=True,rmse_gate=True,
        worst_slice_degradation=0,development={'mae_improvement':.0055,'loss_difference_95ci':[-.2,-.01]},
        audit={'mae_improvement':.01})}}
    with pytest.raises(ValueError,match='offline'):
        check_offline_gate(report,['gates','candidate'],'total')
    with pytest.raises(ValueError,match='target mismatch'):
        check_offline_gate(report,['gates','candidate'],'win')


def test_evidence_hash_and_target_coverage_are_required(tmp_path):
    path=tmp_path/'report.json';path.write_text('{}')
    reference={'report':'report.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    assert referenced_report(tmp_path,reference)=={}
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checksum'):
        referenced_report(tmp_path,reference)
    with pytest.raises(ValueError,match='every changed target'):
        validate_offline_evidence(tmp_path,{}, {'total':reference})


def test_promotion_without_evidence_and_training_activation_leave_primary_unchanged(tmp_path):
    source=Path('/app/scripts/promote_model_release.py')
    if not source.exists():
        source=Path(__file__).resolve().parents[2]/'scripts/promote_model_release.py'
    spec=importlib.util.spec_from_file_location('gated_promote',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    active=tmp_path/'active.json';active.write_text('{"version":"old"}')
    before=active.read_bytes()
    with pytest.raises(ValueError,match='requires'):
        module.promote(tmp_path,'candidate')
    from ml.src.trainer import Trainer
    with pytest.raises(ValueError,match='Direct activation'):
        Trainer._write_release_manifest('candidate',None,None,None,activate=True)
    assert active.read_bytes()==before


def test_changed_total_cannot_be_hidden_by_declaring_only_win(tmp_path):
    folder=tmp_path/'xgboost'/'old';folder.mkdir(parents=True)
    hashes={}
    for target in ('win','spread','total'):
        path=folder/f'{target}_model.json';path.write_text(target)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    active=dict(xgboost_version='old',feature_schema_sha256='same',feature_semantic_version=9)
    candidate=dict(feature_schema_sha256='same',feature_semantic_version=9,file_hashes={'xgboost':dict(hashes)})
    check_changed_targets(tmp_path,active,candidate,['win'])
    candidate['file_hashes']['xgboost']['total_model.json']='changed'
    with pytest.raises(ValueError,match='total.*omitted'):
        check_changed_targets(tmp_path,active,candidate,['win'])


def test_changed_mask_or_calibration_cannot_bypass_full_target_gates(tmp_path):
    folder=tmp_path/'xgboost'/'old';folder.mkdir(parents=True)
    hashes={}
    for target in ('win','spread','total'):
        path=folder/f'{target}_model.json';path.write_text(target)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    active=dict(xgboost_version='old',feature_schema_sha256='same',feature_semantic_version=9,
                calibration_enabled=False)
    candidate={**active,'file_hashes':{'xgboost':hashes},'disabled_features':['home_injury_firepower_loss']}
    with pytest.raises(ValueError,match='preprocessing'):
        check_changed_targets(tmp_path,active,candidate,['total'])
    candidate['disabled_features']=[];candidate['calibration_enabled']=True
    with pytest.raises(ValueError,match='preprocessing'):
        check_changed_targets(tmp_path,active,candidate,['total'])
