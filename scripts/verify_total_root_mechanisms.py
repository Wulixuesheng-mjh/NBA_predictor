"""Follow-up fixed-channel and exact-as_of input drift diagnosis. No repairs."""
import argparse,asyncio,json,hashlib
from pathlib import Path
import numpy as np
from sqlalchemy import select,text
from app.database import async_session_factory
from app.models.game import Game
from app.models.team_game_stats import TeamGameStats
from ml.src.config import ml_settings
from ml.src.predictor import load_predictor
from ml.src.features.feature_builder import FeatureBuilder,FEATURE_NAMES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


async def main(args):
    out=Path(args.output);source=Path(args.source)
    if out.exists():raise ValueError('Never overwrite evidence')
    out.mkdir(parents=True)
    rows=[json.loads(line) for line in (source/'reconstructed_rows.jsonl').read_text().splitlines()]
    authentic=json.loads((source/'authentic_records.json').read_text())['rows']
    root=Path(ml_settings.models_dir);active=(root/'active.json').read_bytes()
    predictor=load_predictor(release_manifest=json.loads(active));mlp=predictor.mlp_trainer;xgb=predictor.xgb_trainer
    audit=json.loads(Path(args.mlp_audit).read_text());critical=[r for r in audit['features'] if r['critical']]
    legacy=root/'feature_cache/train_99ec73146ff0f31e.npz'
    assert sha(legacy)=='cf0ab97b56dcd161440354abdd5a4a6a4cf46d4edf4788fa9c4110ea281d6805'
    groups={'home_team':[i for i,n in enumerate(FEATURE_NAMES[:166]) if n.startswith('home_') and i<33],
        'away_team':list(range(33,66)), 'remaining_features':list(range(66,166))}
    protocol=dict(kind='fixed_mechanism_followup_no_training',source_sha256={p.name:sha(p) for p in
        (source/'protocol.json',source/'reconstructed_rows.jsonl',source/'authentic_records.json')},
        hypothesis_bn='compare predetermined critical channels individually and joint; hold all other inputs and weights fixed',
        hypothesis_drift='exact stored as_of with frozen code/current DB; quantify input drift, not authentic forecasts',
        channel_names=[r['name'] for r in critical],drift_groups=groups,legacy_cache_sha256=sha(legacy),
        dates_not_selected_by_error=True,holdout_description='Nov5-8 batch reported separately; no tuning',
        precision_approval=False,repair=False,training=False,promotion=False)
    save(out/'protocol.json',protocol)
    with np.load(legacy,allow_pickle=False) as data:train=data['X'];cache_meta=json.loads(str(data['meta']))
    X=np.array([r['processed_vector'] for r in rows]);Z=np.clip((X-mlp._scaler_mean)/mlp._scaler_std,-5,5).astype('float32')
    rawz=(X-mlp._scaler_mean)/mlp._scaler_std
    if mlp._active_features is not None:Z[:,~mlp._active_features]=0
    def total(z):
        return mlp._predict_onnx(z)[2].reshape(-1)*mlp._y_total_std+mlp._y_total_mean
    original=total(Z);recorded=np.array([r['mlp_raw'] for r in rows])
    assert max(abs(original-recorded))<1e-3
    effects={};individual=[]
    for feature in critical:
        i=FEATURE_NAMES.index(feature['name']);probe=Z.copy();probe[:,i]=feature['bn_running_mean'];values=total(probe)
        shift=values-original;effects[feature['name']]=dict(mean_shift=float(shift.mean()),min_shift=float(shift.min()),max_shift=float(shift.max()),
            train_min=float(train[:,i].min()),train_max=float(train[:,i].max()),train_std=float(train[:,i].std()),
            current_min=float(X[:,i].min()),current_max=float(X[:,i].max()),current_clipped=int((abs(rawz[:,i])>5).sum()),
            scaler_mean=float(mlp._scaler_mean[i]),scaler_std=float(mlp._scaler_std[i]),
            bn_running_mean=feature['bn_running_mean'],bn_running_var=feature['bn_running_var'],bn_amplification=feature['bn_amplification'])
        individual.append(values)
    probe=Z.copy()
    for feature in critical:probe[:,FEATURE_NAMES.index(feature['name'])]=feature['bn_running_mean']
    together=total(probe)
    # Noncritical negative control: exact copy through ONNX and original preprocessing.
    assert float(abs(total(Z.copy())-original).max())==0
    save(out/'channel_isolation.json',dict(samples=len(rows),legacy_cache_meta=cache_meta,feature_names_provenance='legacy compatibility prefix; names absent from original cache metadata',
        individual=effects,individual_shifts_not_additive=True,all_channel_mean_shift=float((together-original).mean()),
        copy_negative_control_max_difference=0,source_total_max_difference=float(max(abs(original-recorded))),
        per_game=[dict(game_id=r['game_id'],source_day=r['source_day'],raw=float(original[j]),centered=float(together[j]),
            individual={f['name']:float(v[j]) for f,v in zip(critical,individual)}) for j,r in enumerate(rows)]))
    drift=[]
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await session.execute(text('SET TRANSACTION READ ONLY'))
        builder=FeatureBuilder()
        from datetime import datetime
        for record in authentic:
            g=await session.get(Game,record['game_id']);as_of=datetime.fromisoformat(record['as_of'])
            fv=await builder.build_features(g.home_team_id,g.away_team_id,g.scheduled_start_utc,
                as_of=as_of,target_game_id=g.id,use_historical=False,session=session)
            current=predictor._adapt_features(predictor._prepare_release_features(fv.features.reshape(1,-1)),166)
            stored=np.array(record['processed_features']).reshape(1,-1)
            old=float(xgb._predict_totals(stored)[0]);new=float(xgb._predict_totals(current)[0])
            differences=[dict(feature=FEATURE_NAMES[i],stored=float(stored[0,i]),current=float(current[0,i]))
                for i in range(166) if abs(stored[0,i]-current[0,i])>1e-6]
            changes={}
            for label,indices in groups.items():
                xx=stored.copy();xx[:,indices]=current[:,indices]
                changes[label]=float(xgb._predict_totals(xx)[0])-old
            latest=[]
            for side,tid in [('home',g.home_team_id),('away',g.away_team_id)]:
                candidates=(await session.scalars(select(Game).where(Game.status=='Final',Game.game_date<as_of,
                    (Game.home_team_id==tid)|(Game.away_team_id==tid)).order_by(Game.game_date.desc()).limit(10))).all()
                latest.append(dict(side=side,games=[dict(id=p.id,nba_game_id=p.nba_game_id,day=str(p.source_game_date),
                    team_complete=p.team_stats_complete,player_complete=p.player_stats_complete) for p in candidates]))
            drift.append(dict(game_id=g.id,as_of=record['as_of'],kind='exact_as_of_reconstruction_with_current_database',
                actual_total=record['actual_total'],stored_total=old,current_rebuilt_total=new,shift=new-old,
                differences=differences,group_one_at_a_time_shifts=changes,groups_not_additive=True,
                stored_warnings=record['warnings'],current_warnings=fv.data_quality_warnings,
                current_recent_games=latest,current_processed_features=current[0].tolist()))
    save(out/'authentic_input_drift.json',dict(rows=drift,mean_shift=float(np.mean([r['shift'] for r in drift])),
        original_mean_error=float(np.mean([r['stored_total']-r['actual_total'] for r in drift])),
        current_reconstruction_mean_error=float(np.mean([r['current_rebuilt_total']-r['actual_total'] for r in drift])),
        cause_scope='input-vector drift established; individual data revisions not fully replayed',
        authentic_records_modified=False))
    assert (root/'active.json').read_bytes()==active
    print(json.dumps(dict(status='PASS',channel_effects=effects,drift=[{k:r[k] for k in ('game_id','shift','group_one_at_a_time_shifts')} for r in drift],
        primary_unchanged=True,trained=False,promoted=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mlp-audit',required=True);asyncio.run(main(p.parse_args()))
