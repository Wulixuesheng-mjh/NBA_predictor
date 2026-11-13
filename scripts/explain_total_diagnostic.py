"""Explain total-regression errors in score units, not win-classifier SHAP."""
import argparse
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.predictor import Predictor
from ml.src.features.feature_builder import FEATURE_NAMES


def main(args):
    report=json.loads(Path(args.report).read_text())
    bundle=Path(args.bundle)
    old=XGBoostTrainer();old.load(report['active_manifest']['xgboost_version'])
    adapter=Predictor(xgb_trainer=old,release_manifest=report['active_manifest'])
    models={'old_compatible':old}
    for mode in ('direct','residual'):
        model=XGBoostTrainer(models_dir=str(bundle/'models'));model.load(mode)
        models[mode]=model
    selected=sorted(report['rows'],key=lambda r:r['totals']['old_compatible']-r['actual_total'])[:5]
    output=[]
    for row in selected:
        explanations={}
        for name,model in models.items():
            X=np.asarray(row['feature_vector'],dtype=np.float64).reshape(1,-1)
            if name=='old_compatible':
                X=adapter._adapt_features(adapter._prepare_release_features(X),old.model_total.n_features_in_)
            else:
                disabled=report['candidate_manifests'][name]['disabled_features']
                X[:,[FEATURE_NAMES.index(n) for n in disabled]]=0
            contribution=model.model_total.get_booster().predict(xgb.DMatrix(X),pred_contribs=True)[0]
            baseline=float(X[0,FEATURE_NAMES.index('shrunk_expected_total')]) if model._total_mode=='residual' else 0
            assert abs(float(contribution.sum())+baseline-row['totals'][name])<1e-3
            order=np.argsort(contribution[:-1])
            def values(indexes):
                return [dict(feature=FEATURE_NAMES[i],value=float(X[0,i]),points=float(contribution[i]))
                        for i in indexes]
            explanations[name]=dict(target='total_points',base_value=float(contribution[-1]),
                added_pace_efficiency_baseline=baseline,negative=values(order[:5]),
                positive=values(order[-5:][::-1]))
        output.append(dict(game_id=row['game_id'],day=row['source_game_date'],home=row['home'],away=row['away'],
                           actual_total=row['actual_total'],totals=row['totals'],explanations=explanations))
    Path(args.output).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
