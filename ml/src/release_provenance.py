"""Archive the existing production bundle without inventing legacy fit metadata."""
import hashlib
import json
from pathlib import Path

import numpy as np


def archive_active_manifest(root, *, legacy_semantic_version=None):
    from ml.src.features.feature_builder import FEATURE_NAMES
    root=Path(root).resolve(); original=(root/'active.json').read_bytes()
    active=json.loads(original); result=dict(active)
    dim=active.get('feature_dim')
    if not isinstance(dim,int) or not 0<dim<=len(FEATURE_NAMES):
        raise ValueError('Unknown legacy feature dimension')
    semantic=active.get('feature_semantic_version')
    if semantic is None:
        if legacy_semantic_version!=5 or dim!=166:
            raise ValueError('Legacy 166-column semantic version must be explicitly verified')
        semantic=5
    names=active.get('feature_names',FEATURE_NAMES[:dim])
    if names!=FEATURE_NAMES[:dim]:
        raise ValueError('Unknown legacy feature order')
    hashes={}
    for group in ('xgboost','mlp'):
        version=active.get(f'{group}_version')
        if version is None: continue
        if Path(version).name!=version:
            raise ValueError('Invalid pinned model version')
        folder=(root/group/version).resolve()
        if not folder.is_relative_to(root): raise ValueError('Model path outside root')
        required=({'win_model.json','spread_model.json','total_model.json','metadata.json'} if group=='xgboost'
                  else {'model.onnx','scaler.npz','metadata.json'})
        if any(not (folder/name).is_file() for name in required):
            raise ValueError('Incomplete production baseline bundle')
        if group=='xgboost':
            for name in ('win','spread','total'):
                model=json.loads((folder/f'{name}_model.json').read_text())
                if int(model['learner']['learner_model_param']['num_feature'])!=dim:
                    raise ValueError('Baseline model input dimension mismatch')
        else:
            with np.load(folder/'scaler.npz',allow_pickle=False) as scaler:
                if len(scaler['mean'])!=dim or len(scaler['std'])!=dim:
                    raise ValueError('Baseline MLP scaler dimension mismatch')
        hashes[group]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())
                       if p.is_file() and p.suffix in ('.json','.onnx','.npz','.pt')}
        for name,expected in active.get('file_hashes',{}).get(group,{}).items():
            if hashes[group].get(name)!=expected:
                raise ValueError('Existing production checksum changed')
    if not hashes.get('xgboost'):
        raise ValueError('Production XGB baseline is absent')
    result.update(feature_names=names, feature_semantic_version=semantic, file_hashes=hashes,
        origin_active_sha256=hashlib.sha256(original).hexdigest(), origin_active_manifest=active,
        provenance_note='Observed artifact hashes and verified schema; unknown legacy fit/validation ranges remain unknown')
    if (root/'active.json').read_bytes()!=original:
        raise ValueError('Production selector changed during archival')
    return result
