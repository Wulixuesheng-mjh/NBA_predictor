"""Strict isolated total-only MLP contract; never adapts legacy 166 inputs.

This research interface is deliberately separate from the frozen legacy
diagnostic runtime and from the normal three-target production loader.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION

KIND='isolated_mlp_total_contract_v1'


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_metadata(meta):
    if meta.get('input_dim')!=len(FEATURE_NAMES) or meta.get('feature_names')!=FEATURE_NAMES:
        raise ValueError('No legacy schema adaptation: exact declared 213-column order required')
    if meta.get('input_normalization')!='none' or meta.get('targets')!=['total']:
        raise ValueError('An explicit stabilized total-only model is required')
    if meta.get('total_mode') not in ('direct','residual'):raise ValueError('Unknown total target units')


def validate_contract(c):
    if c.get('kind')!=KIND or c.get('role')!='isolated_research_only' or c.get('publishable') is not False:
        raise ValueError('Not an isolated research contract; legacy models require their frozen runtime')
    if c.get('feature_semantic_version')!=CACHE_FORMAT_VERSION or c.get('feature_names')!=FEATURE_NAMES:
        raise ValueError('Feature semantic version/order mismatch')
    if c.get('targets')!=['total'] or c.get('total_mode') not in ('direct','residual'):
        raise ValueError('Only declared total target may be consumed')
    if (c.get('input_stage')!='raw_feature_builder' or c.get('graph_input_stage')!='standardized_clipped_active_features'
        or c.get('graph_output_stage')!='standardized_total' or c.get('input_normalization')!='none'
        or c.get('clip_sigma')!=5):raise ValueError('Unknown preprocessing stages')
    if len(set(c.get('disabled_features',[])))!=len(c.get('disabled_features',[])) or any(n not in FEATURE_NAMES for n in c.get('disabled_features',[])):
        raise ValueError('Unknown or duplicated disabled feature')
    if c.get('runtime_code_sha256')!=file_sha(__file__):raise ValueError('Contract runtime changed')


def validate_scaler(scale):
    required=('mean','std','active_features','y_total_mean','y_total_std')
    if any(k not in scale for k in required):raise ValueError('Scaler fields absent; no legacy defaults permitted')
    dim=len(FEATURE_NAMES)
    if (np.asarray(scale['mean']).shape!=(dim,) or np.asarray(scale['std']).shape!=(dim,)
        or np.asarray(scale['active_features']).shape!=(dim,) or np.asarray(scale['active_features']).dtype!=np.dtype(bool)):
        raise ValueError('Scaler dimensions or active mask mismatch')
    if not np.isfinite(scale['mean']).all() or not np.isfinite(scale['std']).all() or np.any(scale['std']<=0):raise ValueError('Invalid feature scaler')
    for k in ('y_total_mean','y_total_std'):
        if np.asarray(scale[k]).size!=1 or not np.isfinite(scale[k]).all():raise ValueError('Invalid target scaler')
    if float(scale['y_total_std'])<=0:raise ValueError('Invalid total target scale')


def prepare_input(c,X,scale,*,feature_names,semantic_version,input_stage):
    validate_contract(c);validate_scaler(scale)
    if feature_names!=c['feature_names'] or semantic_version!=c['feature_semantic_version'] or input_stage!=c['input_stage']:
        raise ValueError('Caller input contract mismatch; do not convert or standardize twice')
    value=np.asarray(X,dtype=np.float64)
    if value.ndim==1:value=value.reshape(1,-1)
    if value.ndim!=2 or value.shape[1]!=len(FEATURE_NAMES) or not len(value) or not np.isfinite(value).all():
        raise ValueError('Exact finite model input required; no padding/truncation')
    value=value.copy();value[:,[FEATURE_NAMES.index(n) for n in c['disabled_features']]]=0
    value=np.clip((value-scale['mean'])/scale['std'],-5,5)
    value[:,~scale['active_features']]=0
    return value.astype('float32')


def decode_total(c,normalized,scale,baseline=None):
    values=np.asarray(normalized,dtype=np.float64).reshape(-1)*float(scale['y_total_std'])+float(scale['y_total_mean'])
    if c['total_mode']=='residual':
        if baseline is None:raise ValueError('Residual target requires its explicit positive baseline')
        b=np.asarray(baseline,dtype=float).reshape(-1)
        if b.shape!=values.shape or not np.isfinite(b).all() or np.any(b<=0):raise ValueError('Invalid residual baseline')
        values+=b
    elif baseline is not None:raise ValueError('Direct total cannot add a baseline a second time')
    if not np.isfinite(values).all():raise ValueError('Nonfinite total output')
    return values


class IsolatedMLPTotal:
    def __init__(self,folder):
        self.folder=Path(folder).resolve();self.contract=json.loads((self.folder/'contract.json').read_text())
        c=self.contract;validate_contract(c)
        if not isinstance(c.get('file_hashes'),dict) or not {'model.onnx','scaler.npz','source_metadata.json'}.issubset(c['file_hashes']):raise ValueError('Incomplete contract artifacts')
        for name,expected in c['file_hashes'].items():
            if Path(name).name!=name or '/' in name or '\\' in name:raise ValueError('Artifact path must stay in bundle')
            if file_sha(self.folder/name)!=expected:raise ValueError('Contract artifact checksum mismatch')
        metadata=json.loads((self.folder/'source_metadata.json').read_text());validate_metadata(metadata)
        if metadata['total_mode']!=c['total_mode']:raise ValueError('Source target units differ from contract')
        with np.load(self.folder/'scaler.npz',allow_pickle=False) as f:self.scale={k:f[k] for k in f.files}
        validate_scaler(self.scale)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(self.folder/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        inputs,outputs=self.session.get_inputs(),self.session.get_outputs()
        if len(inputs)!=1 or inputs[0].name!='features' or inputs[0].shape[-1]!=len(FEATURE_NAMES):raise ValueError('Graph input does not match contract')
        if len(outputs)!=1 or outputs[0].name!='total_z':raise ValueError('Graph must expose only trained total output')

    def predict_total(self,X,*,feature_names,semantic_version,input_stage,baseline=None):
        prepared=prepare_input(self.contract,X,self.scale,feature_names=feature_names,semantic_version=semantic_version,input_stage=input_stage)
        output=self.session.run(['total_z'],{'features':prepared})[0]
        values=decode_total(self.contract,output,self.scale,baseline)
        return [dict(model_id=self.contract['model_id'],predicted_total=float(v),
            usable=bool(150<=v<=300),status='isolated_research_only',targets=['total'],
            calibration_applied=False,total_uncertainty_status='not_calibrated') for v in values]
