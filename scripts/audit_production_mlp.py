"""Offline numerical audit of original production MLP code and artifacts.

Probe labels are never used: historical inputs here are compatibility probes,
not previously recorded forecasts or accuracy/selection evidence.
"""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decode(outputs,scaler):
    raw=np.column_stack([np.asarray(a).reshape(-1) for a in outputs]).astype(float)
    raw[:,1]=raw[:,1]*float(scaler['y_spread_std'])+float(scaler['y_spread_mean'])
    raw[:,2]=raw[:,2]*float(scaler['y_total_std'])+float(scaler['y_total_mean'])
    return raw


def accepted(values):
    return np.isfinite(values).all(axis=1)&(values[:,0]>=0)&(values[:,0]<=1)&(abs(values[:,1])<=45)&(values[:,2]>=150)&(values[:,2]<=300)


def main(args):
    root=Path(args.root);active=sha(root/'active.json')
    manifest=json.loads((root/'release_baselines/20261022_050952_9824b224.json').read_text())
    folder=root/'mlp'/manifest['mlp_version']; names=FEATURE_NAMES[:166]
    data=DatasetCache(root/'feature_cache').load('4e14a9886bed1bc1',len(FEATURE_NAMES))
    indexes=set(np.linspace(0,len(data.X)-1,100,dtype=int).tolist())
    indexes.update(range(max(0,len(data.X)-107),len(data.X)))
    opener=np.flatnonzero((data.X[:,FEATURE_NAMES.index('home_season_games_played')]==0)|
        (data.X[:,FEATURE_NAMES.index('away_season_games_played')]==0))
    indexes.update(opener[np.linspace(0,len(opener)-1,min(20,len(opener)),dtype=int)].tolist())
    indexes=sorted(indexes); raw=data.X[indexes].astype(float).copy()
    raw[:,FEATURE_NAMES.index('expected_total')]/=2
    X=raw[:,:166]
    spec=importlib.util.spec_from_file_location('original_production_mlp',args.source)
    production=importlib.util.module_from_spec(spec);sys.modules[spec.name]=production;spec.loader.exec_module(production)
    meta=json.loads((folder/'metadata.json').read_text())
    net=production.NBA_MLP(input_dim=meta['input_dim'])
    state=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(state);net.eval();torch.set_num_threads(2)
    scaler=np.load(folder/'scaler.npz'); z=(X-scaler['mean'])/scaler['std']
    clipped=np.clip(z,-5,5).astype('float32')
    session=ort.InferenceSession(str(folder/'model.onnx'),providers=['CPUExecutionProvider'])
    onnx_raw=session.run(None,{session.get_inputs()[0].name:clipped})
    with torch.no_grad(): pt=[o.numpy() for o in net(torch.from_numpy(clipped))]
    onnx_values,pt_values=decode(onnx_raw,scaler),decode(pt,scaler)
    difference=abs(onnx_values-pt_values)
    rms=net.input_bn.running_mean.detach().numpy();var=net.input_bn.running_var.detach().numpy()
    weight=net.input_bn.weight.detach().numpy();bias=net.input_bn.bias.detach().numpy()
    amplification=abs(weight)/np.sqrt(var+net.input_bn.eps)
    critical=(var<1e-6)&(amplification>100)
    normalized=(clipped-rms)/np.sqrt(var+net.input_bn.eps)*weight+bias
    # Diagnostic intervention only: center nearly constant BN coordinates.
    counterfactual=clipped.copy();counterfactual[:,critical]=rms[critical]
    with torch.no_grad(): cf=decode([o.numpy() for o in net(torch.from_numpy(counterfactual))],scaler)
    selected=np.linspace(0,len(indexes)-1,100,dtype=int)
    single=np.vstack([decode(session.run(None,{session.get_inputs()[0].name:clipped[i:i+1]}),scaler)[0] for i in selected])
    single_error=abs(single-onnx_values[selected]).max(axis=0)
    passed=accepted(onnx_values)
    feature_rows=[]
    for i,name in enumerate(names):
        feature_rows.append(dict(name=name,scaler_mean=float(scaler['mean'][i]),scaler_std=float(scaler['std'][i]),
            max_abs_z=float(abs(z[:,i]).max()),clipped_rows=int(np.sum(abs(z[:,i])>5)),
            bn_running_mean=float(rms[i]),bn_running_var=float(var[i]),bn_amplification=float(amplification[i]),
            max_abs_after_bn=float(abs(normalized[:,i]).max()),critical=bool(critical[i])))
    rows=[]
    for j,index in enumerate(indexes):
        info=data.sample_info[index]
        rows.append(dict(game=info['nba_game_id'],day=str(info.get('source_game_date') or info['game_date'])[:10],
            kind='recomputed_input_numerical_probe_not_saved_forecast',onnx=onnx_values[j].tolist(),
            pytorch=pt_values[j].tolist(),guard_accepted=bool(passed[j]),clipped_features=int(np.sum(abs(z[j])>5)),
            max_abs_bn=float(abs(normalized[j]).max()),centered_critical_bn_counterfactual=cf[j].tolist()))
    all_error=difference.max(axis=0)
    report=dict(kind='offline_numerical_audit_no_accuracy_claim',version=manifest['mlp_version'],samples=len(indexes),
        source_sha256=sha(args.source),source_file=args.source,
        artifacts={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        feature_order='legacy 166 prefix; original training metadata does not independently store feature names',
        feature_names=names,onnx_inputs=[dict(name=i.name,shape=i.shape) for i in session.get_inputs()],
        onnx_outputs=[o.name for o in session.get_outputs()],
        scaler_keys=scaler.files,scaler_finite=bool(np.isfinite(z).all()),
        target_scalars={k:float(scaler[k]) for k in scaler.files if np.asarray(scaler[k]).size==1},
        eval_mode=not net.training,dropout_training_flags=[m.training for m in net.modules() if isinstance(m,torch.nn.Dropout)],
        pytorch_onnx_max_difference=all_error.tolist(),
        pytorch_onnx_tolerances=[1e-5,1e-3,1e-3],
        pytorch_onnx_pass=bool(np.all(all_error<=np.array([1e-5,1e-3,1e-3]))),
        accepted_only_engine_max_difference=difference[passed].max(axis=0).tolist() if passed.any() else None,
        single_batch_samples=100,single_batch_max_difference=single_error.tolist(),
        guard_rejections=int(np.sum(~passed)),critical_bn_features=[names[i] for i in np.flatnonzero(critical)],
        counterfactual_guard_rejections=int(np.sum(~accepted(cf))),counterfactual_not_a_model_fix=True,
        features=sorted(feature_rows,key=lambda r:r['max_abs_after_bn'],reverse=True),rows=rows,
        labels_used=False,primary_unchanged=sha(root/'active.json')==active)
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('features','rows','feature_names','artifacts')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/app/ml/saved_models')
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);main(p.parse_args())
