"""Independent fixed-repair real-time safety cohort; never changes active.json."""
import argparse,asyncio,fcntl,json,logging
import numpy as np
from datetime import datetime,timedelta,timezone
from pathlib import Path
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FeatureBuilder,FEATURE_NAMES
from ml.src.legacy_mlp_compatibility import contract_spec,restore_constants
from ml.src.predictor import load_predictor
from scripts.paired_models_worker import sha,digest,publish,safe_values,paired_outputs,in_window,runtime_sources

RUN='mlp_repair_safety_20261112_v1'


def protocol(root):
    active=json.loads((root/'active.json').read_text())
    assert active['mlp_version']==active['xgboost_version']=='20261022_050952'
    hashes={m:{p.name:sha(p) for p in (root/m/active[m+'_version']).iterdir() if p.is_file()} for m in ('xgboost','mlp')}
    sources=runtime_sources();sources['scripts/repaired_mlp_safety_worker.py']=sha(__file__)
    return dict(run=RUN,kind='independent_A_repair_safety',active=active,active_sha256=sha(root/'active.json'),
        models=hashes,contract=contract_spec(),source_hashes=sources,raw_feature_names=FEATURE_NAMES,
        capture='actual server time T-60..T-50; one game once; no backfill',lookback=10,
        sample_plan='first captures reaching >=200 games and >=20 NBA source days, regardless of outcomes; then stop capture',
        primary_losses={'win':'Brier','spread':'MAE','total':'MAE'},
        final_test='paired NBA-day bootstrap95% upper loss difference <= 1% baseline loss, all three heads',
        review='counts/quality only until locked final sample settles; no interim tuning',
        injury_policy='same eligible snapshots/as_of for both; preserve unknown and partial masks',
        promotion=False,training=False,replaces_old_protocol=False)


def checked(root,folder):
    w=json.loads((folder/'protocol.json').read_text());p=w['protocol']
    if w['sha256']!=digest(p) or p!=protocol(root):raise ValueError('Frozen safety runtime/model changed')
    return p


def records(folder,p):
    rows=[]
    for file in sorted((folder/'records').glob('*.json')):
        w=json.loads(file.read_text());r=w['payload']
        assert w['sha256']==digest(r) and r['protocol_sha256']==digest(p)
        assert r['as_of']<r['scheduled_start_utc'] and r['created_at']<r['scheduled_start_utc']
        rows.append(r)
    return rows


def full(rows):return len(rows)>=200 and len({r['source_game_date'] for r in rows})>=20


async def capture(root,folder):
    from app.services.injury_snapshot_reader import latest_injury_snapshot
    p=checked(root,folder)
    with (folder/'records'/'.capture.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        rows=records(folder,p)
        if full(rows):return 0
        now=datetime.now(timezone.utc);count=0
        async with async_session_factory() as db:
            await db.connection(execution_options={'isolation_level':'REPEATABLE READ'})
            await db.execute(__import__('sqlalchemy').text('SET TRANSACTION READ ONLY'))
            games=list((await db.scalars(select(Game).where(Game.status=='Scheduled',Game.time_precision=='exact',
                Game.game_type.in_(('regular','play_in','playoffs')),Game.scheduled_start_utc>=now+timedelta(minutes=50),
                Game.scheduled_start_utc<=now+timedelta(minutes=60)).order_by(Game.scheduled_start_utc,Game.id))).all())
            if not games:return 0
            baseline=load_predictor(release_manifest=p['active'])
            release={**p['active'],'mlp_input_compatibility':p['contract']}
            candidate=load_predictor(release_manifest=release)
            for game in games:
                file=folder/'records'/f'{game.id}.json';as_of=datetime.now(timezone.utc)
                if full(rows):break
                if file.exists() or not in_window(game,as_of) or game.source_game_date is None:continue
                if str(game.source_game_date)<=p['active']['training_data_until']:continue
                fv=await FeatureBuilder().build_features(game.home_team_id,game.away_team_id,game.scheduled_start_utc,
                    session=db,as_of=as_of,target_game_id=game.id,use_historical=False,lookback_games=10)
                old=paired_outputs(baseline,fv);new=paired_outputs(candidate,fv)
                effective=restore_constants(new['processed_input'])
                trainer=candidate.mlp_trainer
                z=(effective-trainer._scaler_mean)/trainer._scaler_std
                new['mlp_scaled_clipped_input']=np.clip(z,-5,5).astype('float32').tolist()
                new['mlp_scaling']=dict(clipped_count=int(np.sum(abs(z)>5)),
                    clipped_names=[FEATURE_NAMES[i] for i in np.flatnonzero(abs(z)>5)],max_abs_z=float(abs(z).max()))
                if old['models']['xgboost']!=new['models']['xgboost']:raise ValueError('Unrelated XGB output changed')
                injuries={}
                for side,tid in [('home',game.home_team_id),('away',game.away_team_id)]:
                    report=await latest_injury_snapshot(db,game.id,tid,as_of)
                    injuries[side]=None if report is None else dict(id=report.id,complete=report.is_complete,
                        entries=report.entries_json,published_at=report.published_at.isoformat(),observed_at=report.observed_at.isoformat())
                payload=dict(kind=p['kind'],protocol_sha256=digest(p),game_id=game.id,
                    source_game_date=str(game.source_game_date),scheduled_start_utc=game.scheduled_start_utc.isoformat(),
                    as_of=as_of.isoformat(),created_at=datetime.now(timezone.utc).isoformat(),builder_vector=fv.features.tolist(),
                    injuries=injuries,baseline=old,candidate=new,
                    candidate_effective_mlp_input=restore_constants(new['processed_input']).tolist())
                invalid=[];payload=safe_values(payload,invalid);payload['nonfinite_paths']=invalid
                checked(root,folder)
                if datetime.now(timezone.utc)>=game.scheduled_start_utc:continue
                if publish(file,dict(payload=payload,sha256=digest(payload))):rows.append(payload);count+=1
        return count


async def status(root,folder):
    p=checked(root,folder);rows=records(folder,p);settled=0
    async with async_session_factory() as db:
        for r in rows:
            game=await db.get(Game,r['game_id'])
            if game and game.status=='Final' and game.score_complete:
                actual=dict(win=int(game.home_score>game.away_score),spread=game.home_score-game.away_score,
                    total=game.home_score+game.away_score)
                outcomes=folder/'records'/'outcomes';outcomes.mkdir(exist_ok=True)
                file=outcomes/f"{game.id}.json"
                if file.exists():
                    stored=json.loads(file.read_text());assert stored['sha256']==digest(stored['payload'])
                    assert stored['payload']['actual']==actual,'Authoritative score revised; retain old settlement and investigate'
                else:
                    errors={}
                    for side in ('baseline','candidate'):
                        raw=r[side]['models']['mlp']['raw'] or {}
                        errors[side]={key:raw.get(field)-actual[key] if raw.get(field) is not None else None
                            for key,field in [('win','win_prob_home'),('spread','predicted_spread'),('total','predicted_total')]}
                    payload=dict(game_id=game.id,prediction_sha256=digest(r),actual=actual,errors=errors,
                        settled_at=datetime.now(timezone.utc).isoformat(),source='database Final score_complete')
                    publish(file,dict(payload=payload,sha256=digest(payload)))
                settled+=1
    return dict(run=RUN,captured=len(rows),nba_days=len({r['source_game_date'] for r in rows}),settled=settled,
        cohort_locked=full(rows),ready_for_final_analysis=full(rows) and settled==len(rows),
        guard_rejections={side:sum(r[side]['models']['mlp']['guard_rejected'] for r in rows) for side in ('baseline','candidate')},
        integrity_errors=0,production_promoted=False,legacy_formal_database_rows_added=0)


async def main(action):
    root=Path(ml_settings.models_dir);folder=root/'mlp_repair_safety'/RUN
    if action=='lock':
        folder.mkdir(parents=True,exist_ok=True);(folder/'records').mkdir(exist_ok=True)
        if (folder/'protocol.json').exists():checked(root,folder)
        else:
            p=protocol(root);assert publish(folder/'protocol.json',dict(protocol=p,sha256=digest(p),locked_at=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(status='LOCKED',run=RUN)));return
    if action=='status':print(json.dumps(await status(root,folder)));return
    while True:
        try:
            count=await capture(root,folder)
            await status(root,folder)
            if count:print(json.dumps(dict(captured=count)),flush=True)
        except Exception:
            if action=='once':raise
            logging.exception('Safety capture failed; no backfill')
        if action=='once':return
        await asyncio.sleep(300)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['lock','once','status','serve'])
    asyncio.run(main(parser.parse_args().action))
