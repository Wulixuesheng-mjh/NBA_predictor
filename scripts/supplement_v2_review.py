"""Make the original locked benchmark explicit alongside the same-cache control."""
import json,hashlib
from pathlib import Path
from reevaluate_models_v2 import ROOT,OUT,read,save,sha,vector_comparison,stats


def main():
    reference=ROOT/'reports/totals_time_20261029/development_direct.json'
    assert sha(reference)=='1e9ed59bf959915db931ab126868b7e0a8989031deef7fce6cec05841043e096'
    save('supplement_protocol.json',dict(purpose='show both named controls, not switch baselines to obtain a pass',
        original_locked_reference_sha256=sha(reference),same_cache_control='complete_history direct:T0',
        selection='existing locked T5 win, T4 spread, T1 total plus already archived median total',
        no_retraining=True,no_promotion=True,manual_records='earliest saved repair snapshot only; one game, never formal acceptance'))
    ref=read(reference)['predictions']['xgboost:T0']
    folder=ROOT/'reports/complete_history_20261104'
    direct=read(folder/'development_direct.json')['predictions'];res=read(folder/'development_residual.json')['predictions']
    old={name:vector_comparison(ref,source[key],target) for name,source,key,target in
        [('win_T5',direct,'xgboost:T5','win'),('spread_T4',direct,'xgboost:T4','spread'),('total_T1_residual',res,'xgboost:T1','total')]}
    distribution=read(ROOT/'reports/total_distribution_20261105/development.json')['predictions']
    old['total_median']=vector_comparison(ref,distribution,'total')
    context=read(OUT/'context.json');games={g['id']:g for g in context['games']}
    records=[]
    for filename in ('same_input_comparison.json','user_requested_fixed_prediction_1030.json'):
        file=ROOT/'reports/live_feedback_20261110'/filename;data=read(file);r=data['payload']
        assert hashlib.sha256(json.dumps(r,sort_keys=True,allow_nan=False).encode()).hexdigest()==data['sha256']
        assert r['as_of']<r['scheduled_start']
        records.append((file,r))
    records.sort(key=lambda item:item[1]['as_of']);file,r=records[0];g=games[r['game_id']]
    assert g['status']=='Final' and g['score_complete']
    y=[int(g['home_score']>g['away_score']),g['home_score']-g['away_score'],g['home_score']+g['away_score']]
    manual=dict(kind='genuine_manual_pregame_diagnostic_not_formal',game_id=g['id'],
        source_day=g['source_day'],as_of=r['as_of'],source_file=str(file.relative_to(ROOT)),snapshots=2,unique_games=1,
        actual=dict(home_score=g['home_score'],away_score=g['away_score']),metrics={})
    for target,index,key in [('win',0,'win_prob_home'),('spread',1,'predicted_spread'),('total',2,'predicted_total')]:
        manual['metrics'][target]={label:stats([y[index]],[r[label]['raw_outputs']['mlp'][key]],target) for label in ('before','after')}
    save('supplement_results.json',dict(original_locked_reference=old,manual_repair_diagnostic=manual,
        retrospective_V2_review_only=True,old_reports_unchanged=True,
        explanation='Historical reference and same-cache T0 differ; neither is literal fixed serving 1022 test accuracy. State baseline in every claim.'))
    print(json.dumps(dict(old_reference={k:dict(pass_historical=v['historical_development_screen'],ci=v['paired_day_95ci'],
        improvement=v['relative_improvement']) for k,v in old.items()},manual=manual),indent=2))


if __name__=='__main__':main()
