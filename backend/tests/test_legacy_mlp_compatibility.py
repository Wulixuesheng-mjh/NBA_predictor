from dataclasses import dataclass
import hashlib,json
from types import SimpleNamespace
import numpy as np
import pytest
from ml.src import legacy_mlp_compatibility as repair


def test_restore_only_demonstrated_coordinates_without_mutation():
    X=np.arange(332,dtype=float).reshape(2,166);before=X.copy()
    fixed=repair.restore_constants(X)
    assert np.array_equal(X,before)
    assert np.all(fixed[:,[17,50]]==0)
    keep=[i for i in range(166) if i not in (17,50)]
    assert np.array_equal(fixed[:,keep],before[:,keep])
    assert np.array_equal(repair.restore_constants(fixed),fixed)
    assert repair.restore_constants(X[0]).shape==(166,)
    assert repair.restore_constants(X.astype('float32')).dtype==np.dtype('float32')


@pytest.mark.parametrize('X',[np.zeros(213),np.zeros(165),np.full(166,np.nan),np.zeros((1,1,166))])
def test_wrong_input_does_not_adapt_or_fill(X):
    with pytest.raises(ValueError):repair.restore_constants(X)


def fixture(tmp_path,monkeypatch):
    folder=tmp_path/repair.VERSION;folder.mkdir()
    (folder/'model.onnx').write_bytes(b'fixed test artifact')
    monkeypatch.setattr(repair,'FILES',{'model.onnx':hashlib.sha256(b'fixed test artifact').hexdigest()})
    path=tmp_path/'contract.json';path.write_text(json.dumps(repair.contract_spec()))
    @dataclass
    class Result:
        predicted_total:float
        model_version:str=repair.VERSION
    class Model:
        models_dir=str(tmp_path);version=repair.VERSION
        _scaler_mean=np.zeros(166);_scaler_std=np.ones(166)
        def predict(self,X):
            self.received=X
            return Result(float(X.sum())) if X.ndim==1 else [Result(float(r.sum())) for r in X]
    return Model(),path,folder


def test_wrapper_explicit_contract_traces_and_single_batch_consistency(tmp_path,monkeypatch):
    base,path,_=fixture(tmp_path,monkeypatch);wrapped=repair.CompatibleLegacyMLP(base,path)
    X=np.ones((2,166));result=wrapped.predict(X)
    assert all(r.predicted_total==164 and repair.CONTRACT_ID in r.model_version for r in result)
    assert wrapped.predict(X[0]).predicted_total==result[0].predicted_total
    assert np.all(X==1) and repair.CONTRACT_ID in wrapped.version


def test_other_models_and_tampered_artifacts_fail_closed(tmp_path,monkeypatch):
    base,path,folder=fixture(tmp_path,monkeypatch)
    base.version='other'
    with pytest.raises(ValueError):repair.CompatibleLegacyMLP(base,path)
    base.version=repair.VERSION;(folder/'model.onnx').write_bytes(b'changed')
    with pytest.raises(ValueError):repair.CompatibleLegacyMLP(base,path)


def test_modified_contract_or_scaler_cannot_activate(tmp_path,monkeypatch):
    base,path,_=fixture(tmp_path,monkeypatch)
    contract=repair.contract_spec();contract['coordinates']=[18,50]
    path.write_text(json.dumps(contract))
    with pytest.raises(ValueError):repair.CompatibleLegacyMLP(base,path)
    path.write_text(json.dumps(repair.contract_spec()));base._scaler_mean[17]=1
    with pytest.raises(ValueError):repair.CompatibleLegacyMLP(base,path)
