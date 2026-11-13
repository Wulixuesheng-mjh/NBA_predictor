"""Independent replay of saved root-cause numerical evidence; no DB writes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from ml.src.predictor import load_predictor
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FEATURE_NAMES


def main(args):
    folder=Path(args.source);out=Path(args.output)
    if out.exists():raise ValueError('Do not overwrite verification')
    p=json.loads((folder/'protocol.json').read_text());report=json.loads((folder/'report.json').read_text())
    rows=[json.loads(line) for line in (folder/'reconstructed_rows.jsonl').read_text().splitlines()]
    assert len(rows)==len(p['cohort'])==len({r['game_id'] for r in rows})==132
    assert [r['game_id'] for r in rows]==p['cohort']
    assert all('2026-10-21'<r['source_day']<='2026-11-08' for r in rows)
    root=Path(ml_settings.models_dir);active=(root/'active.json').read_bytes()
    assert hashlib.sha256(active).hexdigest()==p['active_sha256']
    predictor=load_predictor(release_manifest=p['production']);xgb=predictor.xgb_trainer;mlp=predictor.mlp_trainer
    for name,expected in p['xgb_files'].items():assert hashlib.sha256((root/'xgboost'/p['production']['xgboost_version']/name).read_bytes()).hexdigest()==expected
    X=np.array([r['processed_vector'] for r in rows]);raw=np.array([r['feature_vector'] for r in rows])
    adapted=predictor._adapt_features(predictor._prepare_release_features(raw),166)
    assert np.array_equal(adapted,X)
    xb=xgb._predict_totals(X);mb=mlp.predict(X)
    mlp_total=np.array([r.predicted_total for r in mb])
    xerr=float(max(abs(xb-np.array([r['xgb_raw'] for r in rows]))))
    merr=float(max(abs(mlp_total-np.array([r['mlp_raw'] for r in rows]))))
    assert max(xerr,merr)<1e-3
    for key in report['metrics']:
        available=[r for r in rows if r.get(key) is not None]
        y=np.array([r['actual_total'] for r in available]);pred=np.array([r[key] for r in available])
        assert len(available)==report['metrics'][key]['samples']
        assert abs(float(np.mean(pred-y))-report['metrics'][key]['bias'])<1e-9
        assert abs(float(np.mean(abs(pred-y)))-report['metrics'][key]['mae'])<1e-9
    assert not any(r['calibration_applied'] for r in rows)
    rounding={m:float(max(abs(r[m+'_raw']-r[m+'_final']) for r in rows if r[m+'_final'] is not None)) for m in ('xgb','mlp')}
    assert max(rounding.values())<=.25
    assert all(abs(sum(r['error_decomposition'].values())-(r['actual_total']-r['xgb_raw']))<1e-5 for r in rows)
    # Both actual shifted coordinates suffice; the other four predeclared BN coordinates are zero already.
    z=np.clip((X-mlp._scaler_mean)/mlp._scaler_std,-5,5).astype('float32')
    if mlp._active_features is not None:z[:,~mlp._active_features]=0
    ast=[FEATURE_NAMES.index(n) for n in ('home_avg_ast_pct','away_avg_ast_pct')]
    z[:,ast]=0
    total=mlp._predict_onnx(z)[2].reshape(-1)*mlp._y_total_std+mlp._y_total_mean
    centering_difference=float(max(abs(total-np.array([r['centered_total'] for r in rows]))))
    assert centering_difference<1e-3
    zero_ast=X.copy();zero_ast[:,ast]=0
    xgb_zero_difference=float(max(abs(xgb._predict_totals(zero_ast)-xb)))
    assert xgb_zero_difference==0
    counts=xgb.model_total.get_booster().get_score(importance_type='weight')
    split_count={FEATURE_NAMES[i]:int(counts.get('f'+str(i),0)) for i in ast}
    newer=[r for r in rows if r['source_day']>='2026-11-05']
    assert len(newer)==25
    newer_metrics={key:dict(bias=float(np.mean([r[key]-r['actual_total'] for r in newer])),
        mae=float(np.mean([abs(r[key]-r['actual_total']) for r in newer])))
        for key in ('xgb_raw','mlp_raw','centered_total')}
    assert (root/'active.json').read_bytes()==active
    result=dict(status='PASS',cohort=132,source_days=17,newer_batch_samples=25,newer_metrics=newer_metrics,
        replay_max_error=dict(xgb=xerr,mlp=merr),rounding_max_shift=rounding,
        ast_only_vs_six_channel_centering_max_difference=centering_difference,
        xgb_zero_ast_negative_control_max_difference=xgb_zero_difference,xgb_ast_split_count=split_count,
        mlp_active_feature_mask_present=mlp._active_features is not None,
        target_scaler=dict(mean=mlp._y_total_mean,std=mlp._y_total_std),
        all_decompositions_verified=True,all_metrics_recomputed=True,
        reconstructed_not_pregame_accuracy=True,trained=False,promoted=False,primary_unchanged=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
