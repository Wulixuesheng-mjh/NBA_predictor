"""Export a quarantined typed total graph; engineering proof, not promotion."""
import json,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import onnx
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.dataset_cache import DatasetCache
from ml.src.models.mlp_model import NBA_MLP
from ml.src import mlp_input_contract as contract
from scripts.run_expanded_model_comparison import sha,publish,CACHE,CACHE_SHA

ID='mlp_semantics_isolation_20261105_v1'
SOURCE_STUDY='forecast_season_anchor_20261105_v1'
MONTH='2024-10'
RULE='mlp_total_direct_T1'


class TotalGraph(torch.nn.Module):
    def __init__(self,model):super().__init__();self.net=model
    def forward(self,x):
        x=self.net.input_bn(x)
        x=self.net.layer1(x);x=self.net.layer2(x);x=self.net.layer3(x);x=self.net.layer4(x)
        return self.net.total_head(x)


def main():
    torch.set_num_threads(2);root=Path(ml_settings.models_dir)
    source=root/SOURCE_STUDY/MONTH/RULE;folder=root/'isolated_mlp_contracts'/ID
    if folder.exists():raise ValueError('Isolation package already exists; never overwrite')
    meta=json.loads((source/'metadata.json').read_text());contract.validate_metadata(meta)
    fold=json.loads((root/SOURCE_STUDY/MONTH/'fold.json').read_text())
    detail=fold['models'][RULE]
    for name,expected in detail['artifacts'].items():
        if sha(root/SOURCE_STUDY/MONTH/name)!=expected:raise ValueError('Research source artifact changed')
    data=DatasetCache().load(CACHE,len(FEATURE_NAMES))
    if sha(DatasetCache().path_for(CACHE))!=CACHE_SHA:raise ValueError('Probe cache changed')
    parent=json.loads((root/'expanded_model_comparison_20261105_v1/protocol.json').read_text())['protocol']
    selected=set(np.linspace(0,len(data.X)-1,100,dtype=int).tolist())
    def col(name):return data.X[:,FEATURE_NAMES.index(name)]
    masks=dict(normal=(col('home_season_games_played')>=10)&(col('away_season_games_played')>=10)&
        (col('home_injury_net_available')>0)&(col('away_injury_net_available')>0),
        missing_report=(col('home_injury_net_available')==0)|(col('away_injury_net_available')==0),
        season_opener=(col('home_season_games_played')==0)|(col('away_season_games_played')==0))
    for label,mask in masks.items():
        available=np.flatnonzero(mask)
        if len(available)<20:raise ValueError('Insufficient engineering probes: '+label)
        selected.update(available[np.linspace(0,len(available)-1,20,dtype=int)].tolist())
    index={r['nba_game_id']:i for i,r in enumerate(data.sample_info)}
    recorded=fold['predictions'][RULE]
    selected.update(index[r['nba_game_id']] for r in recorded)
    selected=sorted(selected);raw=data.X[selected].astype(float)
    folder.mkdir(parents=True)
    # Save engineering protocol before graph export or parity measurement.
    publish(folder/'engineering_protocol.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),
        source_case=dict(study=SOURCE_STUDY,month=MONTH,rule=RULE,selection_reason='fixed explicit total-only source, not accuracy selection'),
        source_protocol_sha256=sha(root/SOURCE_STUDY/'protocol.json'),source_fold_sha256=sha(root/SOURCE_STUDY/MONTH/'fold.json'),
        source_artifacts=detail['artifacts'],probe_cache_key=CACHE,probe_cache_sha256=CACHE_SHA,
        probe_ids=[data.sample_info[i]['nba_game_id'] for i in selected],
        tolerances=dict(total_absolute=1e-3),legacy_166_expected='reject',targets=['total'],
        role='engineering_only',precision_approved=False,production_promotion=False))
    shutil.copyfile(source/'scaler.npz',folder/'scaler.npz')
    shutil.copyfile(source/'metadata.json',folder/'source_metadata.json')
    publish(folder/'source_fit.json',dict(source_model=f'{SOURCE_STUDY}/{MONTH}/{RULE}',fit=detail,
        feature_semantic_version=9,source_mode='historical_reconstruction',precision_approved=False))
    state=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    if any(k.startswith('input_bn.') for k in state):raise ValueError('Old input BN graph must not enter new contract')
    net=NBA_MLP(input_dim=len(FEATURE_NAMES),input_normalization='none');net.load_state_dict(state);net.eval()
    graph=TotalGraph(net);graph.eval()
    torch.onnx.export(graph,torch.zeros((1,len(FEATURE_NAMES))),str(folder/'model.onnx'),
        dynamo=False,external_data=False,export_params=True,opset_version=17,do_constant_folding=True,
        input_names=['features'],output_names=['total_z'],dynamic_axes={'features':{0:'batch'},'total_z':{0:'batch'}})
    proto=onnx.load(str(folder/'model.onnx'));onnx.checker.check_model(proto)
    if len(proto.graph.output)!=1 or any('win_head' in i.name or 'spread_head' in i.name for i in proto.graph.initializer):
        raise ValueError('Untrained auxiliary heads leaked into isolated graph')
    c=dict(kind=contract.KIND,role='isolated_research_only',publishable=False,model_id=ID,
        feature_semantic_version=9,feature_names=meta['feature_names'],targets=['total'],total_mode=meta['total_mode'],
        input_stage='raw_feature_builder',graph_input_stage='standardized_clipped_active_features',
        graph_output_stage='standardized_total',input_normalization='none',clip_sigma=5,
        disabled_features=parent['disabled_features'],runtime_code_sha256=contract.file_sha(contract.__file__),
        source_fit_until=detail['full_fit_until'],information_cutoff_day=detail['window_anchor']['information_cutoff_day'],
        source_study=SOURCE_STUDY,source_protocol_sha256=sha(root/SOURCE_STUDY/'protocol.json'),precision_approved=False,
        file_hashes={name:sha(folder/name) for name in ('model.onnx','scaler.npz','source_metadata.json','source_fit.json')})
    publish(folder/'contract.json',c)
    runtime=contract.IsolatedMLPTotal(folder)
    # Independently reproduce original training/inference preprocessing.
    scale=np.load(source/'scaler.npz');prepared=raw.copy()
    prepared[:,[FEATURE_NAMES.index(n) for n in parent['disabled_features']]]=0
    prepared=np.clip((prepared-scale['mean'])/scale['std'],-5,5);prepared[:,~scale['active_features']]=0
    with torch.no_grad():reference=net(torch.tensor(prepared,dtype=torch.float32))[2].numpy().reshape(-1)
    expected=np.asarray([float(v*float(scale['y_total_std'])+float(scale['y_total_mean'])) for v in reference])
    kwargs=dict(feature_names=FEATURE_NAMES,semantic_version=9,input_stage='raw_feature_builder')
    batch=runtime.predict_total(raw,**kwargs);values=np.array([r['predicted_total'] for r in batch])
    alone=np.array([runtime.predict_total(raw[i],**kwargs)[0]['predicted_total'] for i in range(len(raw))])
    locations={data.sample_info[i]['nba_game_id']:j for j,i in enumerate(selected)}
    saved_error=max(abs(values[locations[r['nba_game_id']]]-r['prediction']) for r in recorded)
    error=float(abs(values-expected).max());single_error=float(abs(alone-values).max())
    if max(error,single_error,saved_error)>1e-3:raise ValueError('Isolated graph does not reproduce trained totals')
    rejects=[]
    for label,args in (('legacy_166',dict(feature_names=FEATURE_NAMES[:166],semantic_version=5,input_stage='raw_feature_builder')),
        ('wrong_order',dict(feature_names=list(reversed(FEATURE_NAMES)),semantic_version=9,input_stage='raw_feature_builder')),
        ('double_scaling',dict(feature_names=FEATURE_NAMES,semantic_version=9,input_stage='standardized_clipped_active_features'))):
        try:runtime.predict_total(raw[:1,:166] if label=='legacy_166' else raw[:1],**args)
        except ValueError:rejects.append(label)
        else:raise ValueError('Unsafe input accepted: '+label)
    assert all(set(r).isdisjoint({'win_prob_home','predicted_spread'}) for r in batch)
    report=dict(status='PASS',model_id=ID,role='engineering_only',precision_approved=False,publishable=False,
        samples=len(raw),probe_slices={k:int(mask[selected].sum()) for k,mask in masks.items()},
        torch_onnx_max_total_difference=error,single_batch_max_difference=single_error,
        source_fold_replayed=len(recorded),source_fold_max_difference=float(saved_error),
        rejected_inputs=rejects,graph_outputs=[o.name for o in proto.graph.output],untrained_heads_exposed=False,
        source_labels_used_for_accuracy=False,formal_acceptance_added=0,production_promotion=False)
    publish(folder/'engineering_validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
