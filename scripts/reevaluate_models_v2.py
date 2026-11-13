"""Retrospective V2 evidence review. No fits, parameter selection or promotion."""
import hashlib,json,math,random,statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/v2_reevaluation_20261110'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(p):return hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(name,v):
    p=OUT/name
    if p.exists():raise ValueError('Never overwrite re-review evidence')
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def q(v,p):
    a=sorted(v);i=(len(a)-1)*p;k=int(i);return a[k]+(a[min(k+1,len(a)-1)]-a[k])*(i-k)
def boot(days,values):
    groups=defaultdict(list)
    for d,v in zip(days,values):groups[d].append(v)
    if len(groups)<2:return None
    bins=[(sum(v),len(v)) for v in groups.values()];rng=random.Random(20261110);output=[]
    for _ in range(4000):
        sampled=rng.choices(bins,k=len(bins));output.append(sum(v[0] for v in sampled)/sum(v[1] for v in sampled))
    return [q(output,.025),q(output,.975)]
def stats(y,p,target):
    if not y:return dict(samples=0)
    e=[b-a for a,b in zip(y,p)]
    if target=='win':
        return dict(samples=len(y),brier=statistics.fmean(v*v for v in e),
            logloss=statistics.fmean(-a*math.log(max(1e-15,min(1-1e-15,b)))-(1-a)*math.log(max(1e-15,min(1-1e-15,1-b))) for a,b in zip(y,p)),
            accuracy=statistics.fmean((b>=.5)==bool(a) for a,b in zip(y,p)))
    ae=list(map(abs,e));return dict(samples=len(y),mae=statistics.fmean(ae),rmse=math.sqrt(statistics.fmean(v*v for v in e)),
        bias=statistics.fmean(e),median_ae=statistics.median(ae),p90_ae=q(ae,.9),underprediction_rate=statistics.fmean(v<0 for v in e))
def compare(rows,target):
    y=[r['y'] for r in rows];a=[r['old'] for r in rows];b=[r['new'] for r in rows];days=[r['day'] for r in rows]
    la=[(v-u)**2 if target=='win' else abs(v-u) for u,v in zip(y,a)]
    lb=[(v-u)**2 if target=='win' else abs(v-u) for u,v in zip(y,b)]
    delta=[v-u for u,v in zip(la,lb)];ci=boot(days,delta)
    old=stats(y,a,target);new=stats(y,b,target)
    return dict(samples=len(rows),days=len(set(days)),baseline=old,candidate=new,mean_loss_difference=statistics.fmean(delta),
        paired_day_95ci=ci,relative_improvement=1-statistics.fmean(lb)/statistics.fmean(la),
        primary_directional_screen_pass=ci is not None and ci[1]<0,
        repair_noninferiority_screen_pass=ci is not None and ci[1]<=.01*statistics.fmean(la),
        rmse_safety=None if target=='win' else new['rmse']<=old['rmse']*1.01)
def vector_comparison(reference,candidate,target):
    position={'win':0,'spread':1,'total':2}[target]
    a={r['nba_game_id']:r for r in reference};assert len(a)==len(reference)==len(candidate)
    rows=[]
    for r in candidate:
        old=a[r['nba_game_id']];assert r['actual']==old['actual'] and r['source_game_date']==old['source_game_date']
        rows.append(dict(day=r['source_game_date'],y=r['actual'][position],old=old['prediction'][position],new=r['prediction'][position]))
    summary=compare(rows,target);slices={}
    for name,mask in [('early_season',lambda r:r.get('early_season')),('major_injury',lambda r:r.get('major_injury')),
        ('regular',lambda r:r.get('game_type')=='regular'),('playoffs',lambda r:r.get('game_type')=='playoffs')]:
        selected=[row for row,original in zip(rows,candidate) if mask(original)]
        if selected:
            value=compare(selected,target);value['eligible']=len(selected)>=100
            value['within_5pct']=value['candidate'].get('brier',value['candidate'].get('mae'))<=value['baseline'].get('brier',value['baseline'].get('mae'))*1.05
            slices[name]=value
    summary['slices']=slices
    summary['important_slice_safety']=all(v['within_5pct'] for v in slices.values() if v['eligible'])
    summary['historical_development_screen']=summary['primary_directional_screen_pass'] and summary['important_slice_safety'] and summary['rmse_safety'] is not False
    return summary


def main():
    inputs=['docs/model-admission-v2-proposal-2026-11-10.md',
        'reports/complete_history_20261104/development_direct.json','reports/complete_history_20261104/development_residual.json',
        'reports/complete_history_20261104/audit_summary.json','reports/totals_time_20261029/summary.json',
        'reports/expanded_model_comparison_20261105/development.json','reports/forecast_season_anchor_20261105/development.json',
        'reports/mlp_constant_repair_20261110/workspace_snapshot_validation.json','reports/mlp_preflight_20261110/api_database_preflight.json',
        'reports/injury_and_gates_20261110/manual_trace.json']
    save('protocol.json',dict(rule='V2 retrospective re-review, not automatic acceptance or prospective registration',
        inputs={name:sha(ROOT/name) for name in inputs},comparison_selection='previously locked T5 win, T4 spread, T1 residual total; no new search',
        draws=4000,seed=20261110,group='NBA source day',formal_min=200,minimum_days=20,
        primary_ci_upper_lt=0,rmse_ratio_max=1.01,slice_ratio_max=1.05,slice_min=100,
        old_failures_not_overwritten=True,training=False,promotion=False))
    base=ROOT/'reports/complete_history_20261104'
    direct=read(base/'development_direct.json')['predictions'];res=read(base/'development_residual.json')['predictions']
    history={name:vector_comparison(direct['xgboost:T0'],source[key],target) for name,source,key,target in
        [('win_T5',direct,'xgboost:T5','win'),('spread_T4',direct,'xgboost:T4','spread'),('total_T1_residual',res,'xgboost:T1','total')]}
    expanded=read(ROOT/'reports/expanded_model_comparison_20261105/development.json')
    expanded_out={}
    for name in ('xgb_residual_T1','xgb_median_T1'):
        reference=expanded['predictions']['xgb_direct_T0'];candidate=expanded['predictions'][name]
        lookup={r['nba_game_id']:r for r in reference};assert len(candidate)==len(reference)
        rows=[dict(day=r['source_game_date'],y=r['actual_total'],new=r['prediction'],old=lookup[r['nba_game_id']]['prediction']) for r in candidate]
        expanded_out[name]=compare(rows,'total')
    old_shadow=read(ROOT/'reports/totals_time_20261029/summary.json')['gates']
    shadow_development={k:dict(original_gate=g,primary_directional_screen_pass=g['development']['loss_difference_95ci'][1]<0,
        reason='2% removed; a CI crossing zero still fails directional evidence') for k,g in old_shadow.items()}
    context=read(OUT/'context.json');games={r['id']:r for r in context['games']}
    source=read(OUT/'totals_records.json');live=[]
    for record in source['rows']:
        assert digest(record['payload'])==record['payload_hash']
        assert record['as_of']<record['captured_start'] and record['created_at']<record['captured_start']
        g=record['current_game']
        if g['status']=='Final':
            live.append(dict(day=g['source_day'],actual=g['home_score']+g['away_score'],totals=record['payload']['totals']))
    total_live={mode:compare([dict(day=r['day'],y=r['actual'],old=r['totals']['baseline'],new=r['totals'][mode]) for r in live],'total') for mode in ('direct','residual')}
    win_live=[];paired_live=[]
    for dirname,out,kind in [('win_evidence',win_live,'win'),('paired_evidence',paired_live,'paired')]:
        for file in sorted((OUT/dirname/'records').glob('*.json')):
            data=read(file);r=data['payload'];expected=data.get('payload_sha256',data.get('sha256'))
            assert digest(r)==expected
            assert r['as_of']<r['scheduled_start_utc'] and r['created_at']<r['scheduled_start_utc']
            g=games[r['game_id']]
            if g['status']=='Final' and g['score_complete']:
                if kind=='win':out.append(dict(day=g['source_day'],y=int(g['home_score']>g['away_score']),old=r['win_probabilities']['baseline'],new=r['win_probabilities']['candidate']))
                else:out.append(dict(game=g,record=r))
    paired_metrics={}
    for name in ('xgboost','mlp'):
        rows=[r for r in paired_live if r['record']['models'][name]['raw'] is not None]
        paired_metrics[name]=dict(raw=stats([r['game']['home_score']+r['game']['away_score'] for r in rows],
            [r['record']['models'][name]['raw']['predicted_total'] for r in rows],'total'),
            guard_rejected=sum(r['record']['models'][name]['guard_rejected'] for r in rows))
    repaired=read(ROOT/'reports/mlp_constant_repair_20261110/workspace_snapshot_validation.json')
    repair_metrics={};source_rows=repaired['per_game']
    for target,index in [('win',0),('spread',1),('total',2)]:
        rows=[]
        for r in source_rows:
            g=games[r['game_id']];assert g['home_score']+g['away_score']==r['actual_total']
            y=[int(g['home_score']>g['away_score']),g['home_score']-g['away_score'],r['actual_total']][index]
            rows.append(dict(day=r['source_day'],y=y,old=r['before'][index],new=r['after'][index]))
        repair_metrics[target]=compare(rows,target)
    rollout=read(OUT/'rollout.json')
    results=dict(checked_at=context['checked_at'],historical_selected=history,expanded_total=expanded_out,
        live_shadow_historical_rule_evidence=shadow_development,live_totals=total_live,
        live_win=compare(win_live,'win') if win_live else dict(samples=0),live_production_pair=paired_metrics,
        repair_reconstructed_three_heads=repair_metrics,repair_engineering=read(ROOT/'reports/mlp_preflight_20261110/api_database_preflight.json'),
        global_data_chain='FAIL: manually entered Out records lack target game and eligible snapshots; no matching persisted user prediction',
        formal_runs=rollout['formal_runs'],formal_forecasts=rollout['formal_forecasts'],
        minimum_formal_gate_pass=False,probability_interval_certification='INSUFFICIENT: no 200 real version-matched calibrated residuals',
        limitations=['new review of old chosen candidates is not pristine independent testing',
            'historical research reference is not fixed serving 1022 weights',
            'T1 historical evidence must not be attributed to live residual T4',
            'no deleted tail games, no new fits, no retrospective pregame forecasts'],promotion_allowed=False)
    save('results.json',results)
    print(json.dumps(dict(history={k:{s:v[s] for s in ('relative_improvement','paired_day_95ci','historical_development_screen')} for k,v in history.items()},
        live_totals={k:{s:v[s] for s in ('samples','days','baseline','candidate','paired_day_95ci')} for k,v in total_live.items()},
        live_win=results['live_win'],repair=repair_metrics,formal_forecasts=rollout['formal_forecasts']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
