from pathlib import Path
import numpy as np
import pytest
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src import mlp_input_contract as contract


def settings():
    c=dict(kind=contract.KIND,role='isolated_research_only',publishable=False,
        feature_semantic_version=9,feature_names=FEATURE_NAMES,targets=['total'],total_mode='direct',
        input_stage='raw_feature_builder',graph_input_stage='standardized_clipped_active_features',
        graph_output_stage='standardized_total',input_normalization='none',clip_sigma=5,
        disabled_features=[],runtime_code_sha256=contract.file_sha(contract.__file__))
    n=len(FEATURE_NAMES)
    s=dict(mean=np.zeros(n),std=np.ones(n),active_features=np.ones(n,dtype=bool),y_total_mean=np.array(225.),y_total_std=np.array(10.))
    return c,s


def test_legacy_metadata_cannot_acquire_new_schema_by_defaults():
    with pytest.raises(ValueError):contract.validate_metadata(dict(input_dim=166,has_onnx=True))
    with pytest.raises(ValueError):contract.validate_contract(dict(version='20261022_050952',feature_dim=166))


@pytest.mark.parametrize('change',[dict(semantic_version=5),dict(input_stage='standardized_clipped_active_features'),dict(feature_names=list(reversed(FEATURE_NAMES)))])
def test_new_input_requires_correct_stage_version_and_order(change):
    c,s=settings();args=dict(feature_names=FEATURE_NAMES,semantic_version=9,input_stage='raw_feature_builder');args.update(change)
    with pytest.raises(ValueError):contract.prepare_input(c,np.ones((1,len(FEATURE_NAMES))),s,**args)


def test_input_never_pads_or_truncates_and_keeps_source_unchanged():
    c,s=settings();args=dict(feature_names=FEATURE_NAMES,semantic_version=9,input_stage='raw_feature_builder')
    with pytest.raises(ValueError):contract.prepare_input(c,np.zeros((1,166)),s,**args)
    X=np.ones((1,len(FEATURE_NAMES)))*100;before=X.copy();s['active_features'][0]=False
    result=contract.prepare_input(c,X,s,**args)
    assert result[0,0]==0 and result[0,1]==5 and np.array_equal(X,before)


def test_missing_scaler_fields_and_zero_scales_are_rejected():
    _,s=settings();s.pop('active_features')
    with pytest.raises(ValueError):contract.validate_scaler(s)
    _,s=settings();s['std'][0]=0
    with pytest.raises(ValueError):contract.validate_scaler(s)


def test_direct_and_residual_target_units_cannot_be_mixed():
    c,s=settings();assert contract.decode_total(c,[.5],s).tolist()==[230.]
    with pytest.raises(ValueError):contract.decode_total(c,[.5],s,[225.])
    c['total_mode']='residual';s['y_total_mean']=np.array(0.)
    with pytest.raises(ValueError):contract.decode_total(c,[.5],s)
    assert contract.decode_total(c,[.5],s,[225.]).tolist()==[230.]
