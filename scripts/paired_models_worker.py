"""Separate real-time diagnostic observer of the unchanged production models."""
import argparse,asyncio,fcntl,hashlib,json,logging,math,os,tempfile
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FeatureBuilder,FEATURE_NAMES
from ml.src.predictor import load_predictor

RUN='paired_1022_20261105_tminus60_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def publish(path,value):
    path=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
            temporary=Path(f.name);json.dump(value,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        os.link(temporary,path);return True
    except FileExistsError:return False
    finally:
        if temporary:temporary.unlink(missing_ok=True)


def safe_values(value,paths=None,path=''):
    paths=[] if paths is None else paths
    if isinstance(value,dict):return {k:safe_values(v,paths,path+'.'+str(k)) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe_values(v,paths,path+f'[{i}]') for i,v in enumerate(value)]
    if isinstance(value,(float,np.floating)) and not math.isfinite(value):paths.append(path);return None
    return value


def in_window(game,now):
    return game.status=='Scheduled' and game.time_precision=='exact' and game.scheduled_start_utc is not None and 50<=(game.scheduled_start_utc-now).total_seconds()/60<=60


def runtime_sources():
    root=Path(__file__).resolve().parents[1]
    files=[Path(__file__)]+list((root/'ml/src').rglob('*.py'))+list((root/'app').rglob('*.py'))
    return {str(p.relative_to(root)):sha(p) for p in sorted(files)}


def current_protocol(root):
    import onnxruntime
    active=json.loads((root/'active.json').read_text())
    artifacts={}
    for model,version in (('xgboost',active['xgboost_version']),('mlp',active['mlp_version'])):
        folder=root/model/version
        artifacts[model]={p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    return dict(kind='production_xgb_mlp_paired_diagnostic',acceptance_run=False,
        run=RUN,active=active,active_sha256=sha(root/'active.json'),model_artifacts=artifacts,
        source_hashes=runtime_sources(),onnxruntime_version=onnxruntime.__version__,
        raw_feature_names=FEATURE_NAMES,legacy_processed_feature_names=FEATURE_NAMES[:166],
        capture='first actual server-time T-60..T-50; never backfill',lookback=10,
        confidence='legacy confidence saved as diagnostic only; not calibrated total uncertainty',
        evaluation='same-as_of raw paired outputs, guard rejections, usable final subset reported separately',
        selection_on_results=False,promotion=False)


def checked(root,folder):
    wrapped=json.loads((folder/'protocol.json').read_text());p=wrapped['protocol']
    if wrapped['sha256']!=digest(p) or current_protocol(root)!=p:raise ValueError('Paired protocol/model/runtime changed')
    return p


def lock(root,folder):
    folder.mkdir(parents=True,exist_ok=True);(folder/'records').mkdir(exist_ok=True)
    p=current_protocol(root);wrapped=dict(protocol=p,sha256=digest(p),locked_at=datetime.now(timezone.utc).isoformat())
    if (folder/'protocol.json').exists():
        checked(root,folder)
    elif not publish(folder/'protocol.json',wrapped):raise ValueError('Concurrent protocol publication')
    return p


def observations(folder,p):
    rows=[]
    for file in sorted((folder/'records').glob('*.json')):
        entry=json.loads(file.read_text());r=entry['payload']
        if entry['sha256']!=digest(r) or r['protocol_sha256']!=digest(p):raise ValueError('Paired record integrity error')
        rows.append(r)
    return rows


def paired_outputs(predictor,fv):
    result=predictor.predict(fv,apply_calibration=False)
    prepared=predictor._prepare_release_features(fv.features.reshape(1,-1))
    X=predictor._adapt_features(prepared,166)
    mlp=predictor.mlp_trainer
    scaling=None;clipped=None
    if mlp is not None and mlp._scaler_mean is not None and mlp._scaler_std is not None:
        z=(X-mlp._scaler_mean)/mlp._scaler_std
        clipped=np.clip(z,-5,5).astype('float32')
        scaling=dict(clipped_count=int(np.sum(abs(z)>5)),
            clipped_names=[FEATURE_NAMES[i] for i in np.flatnonzero(abs(z[0])>5)],max_abs_z=float(abs(z).max()))
    model_rows={}
    for name in ('xgboost','mlp'):
        final=getattr(result,name)
        model_rows[name]=dict(raw=result.raw_outputs.get(name),final=asdict(final) if final is not None else None,
            guard_rejected=result.raw_outputs.get(name) is not None and final is None,
            inference_failed=result.raw_outputs.get(name) is None,
            total_uncertainty_status='unvalidated_legacy_confidence')
    return dict(models=model_rows,processed_feature_names=FEATURE_NAMES[:166],processed_input=X[0].tolist(),
        mlp_scaled_clipped_input=clipped[0].tolist() if clipped is not None else None,mlp_scaling=scaling,
        warnings=result.data_quality['warnings'],calibration_applied=result.calibration_applied)


async def capture(root,folder):
    from app.services.injury_snapshot_reader import latest_injury_snapshot
    p=checked(root,folder)
    with (folder/'records'/'.capture.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return 0
        observations(folder,p);now=datetime.now(timezone.utc);count=0
        async with async_session_factory() as session:
            await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
            games=list((await session.scalars(select(Game).where(Game.status=='Scheduled',Game.time_precision=='exact',
                Game.game_type.in_(('regular','play_in','playoffs')),Game.scheduled_start_utc>=now+timedelta(minutes=50),
                Game.scheduled_start_utc<=now+timedelta(minutes=60)).order_by(Game.id))).all())
            if not games:return 0
            predictor=load_predictor(release_manifest=p['active'])
            for game in games:
                file=folder/'records'/f'{game.id}.json';as_of=datetime.now(timezone.utc)
                if file.exists() or not in_window(game,as_of):continue
                if game.source_game_date is None or str(game.source_game_date)<=p['active']['training_data_until']:continue
                fv=await FeatureBuilder().build_features(game.home_team_id,game.away_team_id,game.scheduled_start_utc,
                    session=session,as_of=as_of,target_game_id=game.id,use_historical=False,lookback_games=10)
                outputs=paired_outputs(predictor,fv)
                injuries={}
                for side,tid in (('home',game.home_team_id),('away',game.away_team_id)):
                    report=await latest_injury_snapshot(session,game.id,tid,as_of,reconstructed=False)
                    injuries[side]=None if report is None else dict(id=report.id,sha256=report.source_sha256,
                        complete=report.is_complete,published_at=report.published_at.isoformat(),observed_at=report.observed_at.isoformat())
                payload=dict(kind=p['kind'],acceptance_run=False,protocol_sha256=digest(p),game_id=game.id,
                    source_game_date=str(game.source_game_date),scheduled_start_utc=game.scheduled_start_utc.isoformat(),
                    as_of=as_of.isoformat(),created_at=datetime.now(timezone.utc).isoformat(),
                    builder_vector=fv.features.tolist(),injury_snapshots=injuries,**outputs)
                nonfinite=[];payload=safe_values(payload,nonfinite);payload['nonfinite_paths']=nonfinite
                checked(root,folder)
                if datetime.now(timezone.utc)>=game.scheduled_start_utc:continue
                count+=int(publish(file,dict(payload=payload,sha256=digest(payload))))
        return count


def errors(rows,name,which):
    values=[]
    for r,g in rows:
        output=r['models'][name][which]
        if output and isinstance(output.get('predicted_total'),(float,int)):
            values.append(output['predicted_total']-g.home_score-g.away_score)
    if not values:return dict(samples=0)
    e=np.array(values)
    return dict(samples=len(e),mae=float(abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),
        bias=float(e.mean()),underprediction_rate=float(np.mean(e<0)))


async def status(root,folder):
    p=checked(root,folder);rows=observations(folder,p);settled=[]
    async with async_session_factory() as session:
        for r in rows:
            game=await session.get(Game,r['game_id'])
            if game and game.status=='Final' and game.score_complete:settled.append((r,game))
    paired=[(r,g) for r,g in settled if all(isinstance((r['models'][m]['raw'] or {}).get('predicted_total'),(int,float)) for m in ('xgboost','mlp'))]
    return dict(kind=p['kind'],acceptance_run=False,captured=len(rows),settled=len(settled),
        same_as_of_paired_settled=len(paired),integrity_errors=0,
        rejections={m:sum(r['models'][m]['guard_rejected'] for r in rows) for m in ('xgboost','mlp')},
        inference_failures={m:sum(r['models'][m]['inference_failed'] for r in rows) for m in ('xgboost','mlp')},
        paired_raw_metrics={m:errors(paired,m,'raw') for m in ('xgboost','mlp')},
        final_usable_metrics={m:errors(settled,m,'final') for m in ('xgboost','mlp')})


async def main(args):
    root=Path(ml_settings.models_dir);folder=root/'paired_model_observations'/RUN
    if args.action=='lock':lock(root,folder);print(json.dumps(dict(status='LOCKED',protocol_sha256=sha(folder/'protocol.json'))));return
    if args.action=='status':print(json.dumps(await status(root,folder)));return
    while True:
        try:
            count=await capture(root,folder)
            if count:print(json.dumps(dict(captured=count,acceptance_run=False)),flush=True)
        except Exception:
            if args.action=='once':raise
            logging.exception('Paired capture failed; no backfill or promotion')
        if args.action=='once':return
        await asyncio.sleep(300)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['lock','status','once','serve']);asyncio.run(main(p.parse_args()))
