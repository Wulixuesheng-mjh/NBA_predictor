"""Explicit compatibility for one pinned legacy bundle, before its scaler/graph.

Only the two demonstrated constant training coordinates are restored. The
original trainer, weights, target decoding and unrelated models stay intact.
"""
import hashlib
import json
from pathlib import Path

VERSION = '20261022_050952'
CONTRACT_ID = 'legacy_mlp_1022_constant_ast_v1'
CACHE_SHA = 'cf0ab97b56dcd161440354abdd5a4a6a4cf46d4edf4788fa9c4110ea281d6805'
FILES = {
    'model.onnx': 'dc26ed0ffdeae42920c115e3b501da7b8f0c52ea28db3f4b8f50b3f1ddad4e03',
    'model.onnx.data': '42a2c8f86004394f8cbeec14026839fcd478023111038aaca1ca9e669b22f652',
    'model.pt': '9a353c296265effa7b1be8ca78d0d0901cfd815be87fd4ee49e2c5231200560d',
    'scaler.npz': '9a9d554dde697650f7ae9dcaa3de733411b51b50b50e772786570c28cfa51167',
    'metadata.json': '8004f0a3c0081f418a219b80e5bd5efdb66003aaf1c1f7cbebc47627add84e64',
}


def contract_spec():
    return dict(id=CONTRACT_ID,model_version=VERSION,input_dim=166,
        input_stage='legacy_raw_before_scaler',coordinates=[17,50],training_values=[0.0,0.0],
        source_cache_sha256=CACHE_SHA,artifact_sha256=dict(FILES),
        applies_to=['win','spread','total'],precision_approved=False)


def restore_constants(X):
    import numpy as np
    value=np.asarray(X)
    if (value.dtype.kind not in 'fiub' or value.ndim not in (1,2)
        or value.shape[-1]!=166 or not np.isfinite(value).all()):
        raise ValueError('Legacy compatibility requires exact finite raw 166-column input')
    value=value.copy();value[..., [17,50]]=0.0
    return value


class CompatibleLegacyMLP:
    def __init__(self,base,contract_path):
        import numpy as np
        contract=contract_path if isinstance(contract_path,dict) else json.loads(Path(contract_path).read_text(encoding='utf-8'))
        if contract!=contract_spec():raise ValueError('Unrecognized legacy MLP compatibility contract')
        if base.version!=VERSION:raise ValueError('Compatibility is restricted to the pinned 1022 MLP')
        folder=Path(base.models_dir)/VERSION
        for name,expected in FILES.items():
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('Legacy MLP artifact changed: '+name)
        if (base._scaler_mean is None or base._scaler_std is None
            or np.asarray(base._scaler_mean).shape!=(166,) or np.asarray(base._scaler_std).shape!=(166,)
            or not np.array_equal(base._scaler_mean[[17,50]],[0.,0.])
            or not np.array_equal(base._scaler_std[[17,50]],[1.,1.])):
            raise ValueError('Legacy coordinate/scaler semantics changed')
        self.base=base
        self.input_compatibility_id=CONTRACT_ID
        self.input_compatibility_contract=contract

    @property
    def version(self):return VERSION+'+'+CONTRACT_ID

    def __getattr__(self,name):return getattr(self.base,name)

    def predict(self,X):
        result=self.base.predict(restore_constants(X))
        for prediction in result if isinstance(result,list) else [result]:
            prediction.model_version=VERSION+'+'+CONTRACT_ID
        return result
