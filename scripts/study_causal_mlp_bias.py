"""Prespecified past-only additive bias study on stable rolling MLP rules.

Historical reconstruction only. Residuals are grouped by the frozen learning
rule across rolling refits, never inserted into production's per-bundle store.
"""
import argparse,hashlib,json
from datetime import date,datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from ml.src.config import ml_settings
from ml.src.evaluation import regression_metrics

STUDY='causal_mlp_bias_20261105_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def publish(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
def describe(rows):
    if not rows:return dict(samples=0)
    y=np.array([r['actual_total'] for r in rows]);v=np.array([r['prediction'] for r in rows])
    return dict(samples=len(rows),**regression_metrics(y,v),underprediction_rate=float(np.mean(v<y)),
        total_guard_rejections=int(np.sum((v<150)|(v>300))))


def apply_past_bias(rows,minimum=200,maximum=1000,lag_days=7):
    if len({r['nba_game_id'] for r in rows})!=len(rows):raise ValueError('Duplicate forecasts')
    result=[]
    for month in sorted({r['fold'] for r in rows}):
        cutoff=date.fromisoformat(month+'-01')-timedelta(days=lag_days)
        prior=[r for r in rows if r['fold']<month and date.fromisoformat(str(r.get('source_game_date') or r['game_date'])[:10])<=cutoff]
        prior=sorted(prior,key=lambda r:(str(r.get('source_game_date') or r['game_date']),r['nba_game_id']))[-maximum:]
        residuals=np.asarray([r['actual_total']-r['prediction'] for r in prior],dtype=float)
        if not np.isfinite(residuals).all():raise ValueError('Nonfinite prior residual; do not silently drop failures')
        enabled=len(prior)>=minimum
        offset=float(residuals.mean()) if enabled else 0.
        provenance=dict(enabled=enabled,prior_count=len(prior),offset=offset,cutoff=str(cutoff),
            prior_ids_sha256=digest([r['nba_game_id'] for r in prior]),
            latest_prior_day=max((str(r.get('source_game_date') or r['game_date'])[:10] for r in prior),default=None))
        for row in rows:
            if row['fold']==month:
                result.append(row|dict(uncorrected_prediction=row['prediction'],prediction=row['prediction']+offset,
                    calibration=provenance,calibration_kind='historical_rule_level_past_residual_mean'))
    lookup={r['nba_game_id']:r for r in result}
    return [lookup[r['nba_game_id']] for r in rows]


def inputs(root):
    shared=root/'expanded_model_comparison_20261105_v1/development.json'
    focused=root/'total_only_mlp_20261105_v1/development.json'
    a=json.loads(shared.read_text());b=json.loads(focused.read_text())
    rows=dict(mlp_shared_T1=a['predictions']['mlp_shared_T1'],mlp_total_direct_T1=b['predictions']['mlp_total_direct_T1'])
    for name,report in (('mlp_shared_T1',a),('mlp_total_direct_T1',b)):
        if any(report['slices'][name][season]['all']['bias']>=0 for season in ('2023-24','2024-25')):
            raise ValueError('Stable historical negative bias not confirmed in both seasons')
    return rows,a['predictions']['xgb_direct_T0'],[shared,focused]


def protocol(root):
    rows,_,files=inputs(root);source=Path(__file__).resolve().parents[1]
    return dict(study=STUDY,kind='historical_rule_level_causal_additive_bias',
        source_hashes={str(p.relative_to(source)):sha(p) for p in (Path(__file__),source/'scripts/run_expanded_model_comparison.py',
            source/'scripts/run_total_only_mlp_study.py',source/'ml/src/evaluation.py')},
        references={str(p):sha(p) for p in files},bases=list(rows),samples=2631,
        rule='one monthly fixed offset = mean of previous same-rule rolling OOS residuals',
        eligibility='both full development seasons have negative mean bias before correction',
        minimum_residuals=200,maximum_residuals=1000,lag_days_before_month=7,
        only_prior_months=True,offset_shrinkage=False,offset_cap=None,
        target='actual minus uncorrected prediction; never recursively learn from corrected residuals',
        evaluation='all original games retained, including warmup with zero correction; no model/threshold search',
        residual_scope='same frozen learning rule across monthly versions; not production single-bundle residual certification',
        labels='reconstructed historical NBA days, no claim of original source-publication timestamps',
        gates=dict(primary_comparator='xgb_direct_T0',improvement_min=.02,rmse_degradation_max=.01,
            slice_min=100,slice_degradation_max=.05,paired_day_ci_upper_lt=0),
        live_forecasts_used=False,apply_to_broken_1022_mlp=False,production_store_written=False,
        audit_used=False,production_promotion=False)


def main(args):
    root=Path(ml_settings.models_dir);folder=root/STUDY;folder.mkdir(parents=True,exist_ok=True)
    p=protocol(root)
    if args.action=='lock':
        if (folder/'protocol.json').exists():
            if json.loads((folder/'protocol.json').read_text())['protocol']!=p:raise ValueError('Immutable bias protocol changed')
        else:publish(folder/'protocol.json',dict(locked_at=datetime.now(timezone.utc).isoformat(),protocol=p))
        print(json.dumps(dict(status='LOCKED',protocol_sha256=sha(folder/'protocol.json'))));return
    if json.loads((folder/'protocol.json').read_text())['protocol']!=p:raise ValueError('Locked inputs changed')
    if (folder/'development.json').exists():raise ValueError('Completed bias study immutable')
    from scripts.run_total_only_mlp_study import compare
    bases,xgb,_=inputs(root);result=dict(protocol_sha256=sha(folder/'protocol.json'),predictions={},metrics={},comparisons={},slices={},promoted=False)
    for name,rows in bases.items():
        corrected=apply_past_bias(rows);result['predictions'][name]=corrected
        result['metrics'][name]=describe(corrected)
        result['comparisons'][name]=dict(vs_uncorrected=compare(corrected,rows),vs_xgb=compare(corrected,xgb))
        result['slices'][name]={season:dict(all=describe([r for r in corrected if r['season']==season]),
            early=describe([r for r in corrected if r['season']==season and r['early_season']])) for season in ('2023-24','2024-25')}
        result.setdefault('enabled_games',{})[name]=sum(r['calibration']['enabled'] for r in corrected)
    publish(folder/'development.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','develop']);main(parser.parse_args())
