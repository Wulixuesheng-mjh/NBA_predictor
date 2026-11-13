"""Exact total-only replay for expanded shared and focused research artifacts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.features.dataset_cache import DatasetCache
from ml.src.models.mlp_model import NBA_MLP
from ml.src.evaluation import nba_day
from scripts.run_expanded_model_comparison import sha,publish


def neural_values(folder,X,residual_baseline=None):
    meta=json.loads((folder/'metadata.json').read_text());scale=np.load(folder/'scaler.npz')
    if meta['feature_names']!=FEATURE_NAMES or meta['input_normalization']!='none':raise ValueError('Research input schema mismatch')
    net=NBA_MLP(input_dim=len(FEATURE_NAMES),input_normalization='none')
    net.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True));net.eval()
    scaled=np.clip((X-scale['mean'])/scale['std'],-5,5)
    scaled[:,~scale['active_features']]=0
    with torch.no_grad():raw=net(torch.tensor(scaled,dtype=torch.float32))[2].numpy().reshape(-1)
    # Match trainer.predict scalar de-standardization, consuming only total.
    values=np.array([float(v*float(scale['y_total_std'])+float(scale['y_total_mean'])) for v in raw])
    if meta.get('total_mode')=='residual':
        if residual_baseline is None:raise ValueError('Missing residual baseline')
        values+=residual_baseline
    return values


def main(args):
    torch.set_num_threads(2);root=Path(ml_settings.models_dir);folder=root/args.study
    active=sha(root/'active.json');source=Path(__file__).resolve().parents[1]
    wrapped=json.loads((folder/'protocol.json').read_text());p=wrapped['protocol']
    for relative,expected in p['source_hashes'].items():
        if sha(source/relative)!=expected:raise ValueError('Research source changed: '+relative)
    result=json.loads((folder/'development.json').read_text())
    if result['protocol_sha256']!=sha(folder/'protocol.json'):raise ValueError('Wrong completed protocol')
    cache=DatasetCache();key=p.get('feature_cache_key') or p['cache_key']
    if sha(cache.path_for(key))!=p['cache_sha256']:raise ValueError('Changed pinned cache')
    data=cache.load(key,len(FEATURE_NAMES));indexes={r['nba_game_id']:i for i,r in enumerate(data.sample_info)}
    parent=json.loads((root/'expanded_model_comparison_20261105_v1/protocol.json').read_text())['protocol']
    disabled=[FEATURE_NAMES.index(n) for n in parent['disabled_features']]
    b=FEATURE_NAMES.index('shrunk_expected_total');reports={};artifacts=0
    for name,rows in result['predictions'].items():
        if len(rows)!=2631 or len({r['nba_game_id'] for r in rows})!=2631:raise ValueError('Bad full cohort')
        singles={rows[i]['nba_game_id'] for i in np.linspace(0,len(rows)-1,100,dtype=int)}
        maximum=0.;parity=0;replayed=0
        for fold in result['folds']:
            month=fold['month'];group=[r for r in rows if r['fold']==month];detail=fold['models'][name]
            for file,expected in detail['artifacts'].items():
                if sha(folder/month/file)!=expected:raise ValueError('Model artifact changed')
            artifacts+=len(detail['artifacts'])
            if detail['full_fit_until']>=min(nba_day(r) for r in group):raise ValueError('Outer labels in fitting range')
            loc=[indexes[r['nba_game_id']] for r in group];X=data.X[loc].copy();X[:,disabled]=0
            if data.y_total[loc].tolist()!=[r['actual_total'] for r in group]:raise ValueError('Labels changed')
            selected=[i for i,r in enumerate(group) if r['nba_game_id'] in singles]
            if name.startswith('xgb_'):
                net=xgb.XGBRegressor(n_jobs=1);net.load_model(folder/month/f'{name}.json')
                values=net.predict(X)+(X[:,b] if name!='xgb_direct_T0' else 0)
                alone=np.array([net.predict(X[i:i+1])[0]+(X[i,b] if name!='xgb_direct_T0' else 0) for i in selected])
            else:
                mf=folder/month/('mlp' if name=='mlp_shared_T1' else name)
                values=neural_values(mf,X,X[:,b])
                alone=np.array([neural_values(mf,X[i:i+1],X[i:i+1,b])[0] for i in selected])
            maximum=max(maximum,float(np.max(abs(values-np.array([r['prediction'] for r in group])))))
            if len(selected):maximum=max(maximum,float(np.max(abs(alone-values[selected]))))
            replayed+=len(group);parity+=len(selected)
        if maximum>1e-3 or parity!=100:raise ValueError((name,'Research replay/parity failure',maximum,parity))
        reports[name]=dict(replayed=replayed,single_batch_samples=parity,max_score_difference=maximum)
    if sha(root/'active.json')!=active:raise ValueError('Primary changed')
    report=dict(status='PASS',study=args.study,models=reports,artifact_files_verified=artifacts,
        trained_targets_only=True,core_source_unchanged=True,primary_unchanged=True,
        promoted=False,formal_acceptance_added=0,scope='research total inference replay; not ONNX or release qualification')
    publish(folder/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);main(p.parse_args())
