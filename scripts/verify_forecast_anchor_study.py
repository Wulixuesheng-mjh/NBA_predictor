"""Replay all anchored and reused fold artifacts; verify the mask decisions."""
import json
from pathlib import Path
import numpy as np
import torch
import xgboost as xgb
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.training_policy import subset_arrays
from ml.src.evaluation import monthly_folds,nba_day
from scripts import run_forecast_anchor_study as study
from scripts.verify_expanded_research_models import neural_values


def main():
    torch.set_num_threads(2);root=Path(ml_settings.models_dir);folder=root/study.STUDY
    data,rows,_=study.load(root)
    p=json.loads((folder/'protocol.json').read_text())['protocol']
    if study.protocol(root,data,rows)!=p:raise ValueError('Anchor protocol/source changed')
    report=json.loads((folder/'development.json').read_text())
    if report['protocol_sha256']!=study.shared.sha(folder/'protocol.json'):raise ValueError('Wrong report protocol')
    ids={r['nba_game_id']:i for i,r in enumerate(data.sample_info)};wanted={r['nba_game_id'] for r in rows}
    selected={rows[i]['nba_game_id'] for i in np.linspace(0,len(rows)-1,100,dtype=int)}
    parent=json.loads((root/'expanded_model_comparison_20261105_v1/protocol.json').read_text())['protocol']
    disabled=[FEATURE_NAMES.index(n) for n in parent['disabled_features']];b=FEATURE_NAMES.index('shrunk_expected_total')
    max_error={n:0. for n in study.RULES};counts={n:0 for n in study.RULES};parity={n:0 for n in study.RULES}
    folds={r['month']:r for r in report['folds']}
    for month,pm,tm in monthly_folds(data.sample_info,{'2023-24','2024-25'}):
        tm &= np.array([r['nba_game_id'] in wanted for r in data.sample_info])
        if not tm.any():continue
        past,test=subset_arrays(data,pm),subset_arrays(data,tm);ctx=study.fold_context(data,test)
        _,_,expected_window=study.forecast_mask_and_weights(past.sample_info,'T1',**ctx)
        X=test.X.copy();X[:,disabled]=0
        single=[i for i,r in enumerate(test.sample_info) if r['nba_game_id'] in selected]
        for name in study.RULES:
            detail=folds[month]['models'][name]
            if detail['window_anchor']!=expected_window:raise ValueError('Window metadata mismatch')
            artifact_root=(root/detail['artifact_root']).resolve()
            if not artifact_root.is_relative_to(root.resolve()):raise ValueError('Artifact escaped model root')
            for file,expected in detail['artifacts'].items():
                if study.shared.sha(artifact_root/file)!=expected:raise ValueError('Artifact changed')
            values=[r for r in report['predictions'][name] if r['fold']==month]
            if [(r['nba_game_id'],r['actual_total']) for r in values]!=[(r['nba_game_id'],float(test.y_total[i])) for i,r in enumerate(test.sample_info)]:raise ValueError('Changed cohort/labels')
            if detail['full_fit_until']>=min(nba_day(r) for r in values):raise ValueError('Training sees target day')
            if name.startswith('xgb_'):
                model=xgb.XGBRegressor(n_jobs=1);model.load_model(artifact_root/f'{name}.json')
                predicted=model.predict(X)+X[:,b]
                singles=np.array([model.predict(X[i:i+1])[0]+X[i,b] for i in single])
            else:
                model_folder=artifact_root/('mlp' if name=='mlp_shared_T1' else name)
                predicted=neural_values(model_folder,X,X[:,b])
                singles=np.array([neural_values(model_folder,X[i:i+1],X[i:i+1,b])[0] for i in single])
            error=float(np.max(abs(predicted-np.array([r['prediction'] for r in values]))))
            if single:error=max(error,float(np.max(abs(singles-predicted[single]))))
            max_error[name]=max(max_error[name],error);counts[name]+=len(values);parity[name]+=len(single)
    if any(n!=2631 for n in counts.values()) or any(n!=100 for n in parity.values()) or max(max_error.values())>1e-3:raise ValueError('Anchor replay/parity failed')
    result=dict(status='PASS',models={n:dict(replayed=counts[n],single_batch_samples=parity[n],max_score_difference=max_error[n]) for n in study.RULES},
        mask_decisions_verified=True,retrained_folds=[f['month'] for f in report['folds'] if f['changed']],
        reused_folds=sum(not f['changed'] for f in report['folds']),promoted=False,formal_acceptance_added=0)
    study.shared.publish(folder/'validation.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
