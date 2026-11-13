"""Independent append-only win diagnostics, isolated from totals and acceptance."""
import asyncio,hashlib,json,os,tempfile,fcntl
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.services.prospective_service import digest,feature_code_hash,verify_bundle
from ml.src.config import ml_settings


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish_once(path,payload):
    """Publish a complete fsynced file atomically; an existing record never changes."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as stream:
            temporary=Path(stream.name);json.dump(payload,stream,allow_nan=False)
            stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
        return True
    except FileExistsError:
        return False
    finally:
        if temporary: temporary.unlink(missing_ok=True)


def in_window(game,now):
    return (game.status=='Scheduled' and game.time_precision=='exact' and game.scheduled_start_utc is not None
        and 50<=(game.scheduled_start_utc-now).total_seconds()/60<=60)


def code_hash():
    from ml.src import win_shadow
    worker=Path(win_shadow.__file__).resolve().parents[2]/'scripts/win_shadow_worker.py'
    return digest(dict(features=feature_code_hash(),service=sha(__file__),win_inference=sha(win_shadow.__file__),worker=sha(worker)))


def lock_run(root,name,version):
    from app.services.release_gate_service import check_offline_gate
    root=Path(root).resolve()
    if Path(name).name!=name or Path(version).name!=version: raise ValueError('Invalid run/version')
    candidate=root/'win_shadows'/version
    manifest=json.loads((candidate/'manifest.json').read_text())
    validation=json.loads((candidate/'validation.json').read_text())
    study=root/'complete_history_20261104'/'audit_summary.json'
    if sha(study)!=manifest['audit_evidence_sha256']: raise ValueError('Audit evidence changed')
    check_offline_gate(json.loads(study.read_text()),['release_gates','win'],'win')
    if (validation.get('status')!='PASS' or validation.get('version')!=version
            or validation.get('parity_samples',0)<100 or validation.get('locked_fold_error',1)>1e-5):
        raise ValueError('Win engineering evidence incomplete')
    active=(root/'active.json').read_bytes()
    baseline=json.loads((root/'release_baselines/20261022_050952_9824b224.json').read_text())
    if baseline['origin_active_sha256']!=hashlib.sha256(active).hexdigest(): raise ValueError('Baseline changed')
    verify_bundle(root,baseline)
    if sha(candidate/'win_model.json')!=manifest['model_sha256']: raise ValueError('Candidate model changed')
    protocol=dict(kind='win_only_diagnostic',acceptance_run=False,targets=['win'],version=version,
        baseline=baseline,active_sha256=sha(root/'active.json'),manifest_sha256=sha(candidate/'manifest.json'),
        validation_sha256=sha(candidate/'validation.json'),code_hash=code_hash(),lead_minutes=60,
        window_minutes=10,lookback=10,rule='first actual server-time T-60..T-50 observation; never backfill')
    folder=root/'win_observations'/name
    file=folder/'protocol.json';payload=dict(protocol=protocol,sha256=digest(protocol))
    if not publish_once(file,payload) and json.loads(file.read_text())!=payload:
        raise ValueError('Existing win protocol is immutable; use a new name')
    (folder/'records').mkdir(exist_ok=True)
    return folder


def checked_protocol(root,folder):
    value=json.loads((folder/'protocol.json').read_text());p=value['protocol']
    candidate=root/'win_shadows'/p['version']
    if (value['sha256']!=digest(p) or p['kind']!='win_only_diagnostic' or p['acceptance_run']
        or p['code_hash']!=code_hash() or sha(root/'active.json')!=p['active_sha256']
        or sha(candidate/'manifest.json')!=p['manifest_sha256']
        or sha(candidate/'validation.json')!=p['validation_sha256']):
        raise ValueError('Win protocol or runtime changed')
    verify_bundle(root,p['baseline'])
    manifest=json.loads((candidate/'manifest.json').read_text())
    if sha(candidate/'win_model.json')!=manifest['model_sha256']: raise ValueError('Win model changed')
    return p,manifest


async def capture_due(root,folder):
    from ml.src.features.feature_builder import FeatureBuilder,FEATURE_NAMES
    from ml.src.models.xgboost_model import XGBoostTrainer
    from ml.src.predictor import Predictor
    from ml.src.win_shadow import predict_win_bundle,prepare_win_input
    from ml.src.inference_snapshot import input_snapshot
    from app.services.injury_snapshot_reader import latest_injury_snapshot
    root,folder=Path(root),Path(folder);p,manifest=checked_protocol(root,folder)
    count=0
    with (folder/'records'/'.capture.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return 0
        for path in (folder/'records').glob('*.json'):
            prior=json.loads(path.read_text())
            if prior['payload_sha256']!=digest(prior['payload']) or prior['payload'].get('protocol_sha256')!=digest(p):
                raise ValueError('Win observation integrity check failed')
        async with async_session_factory() as session:
            if session.bind.dialect.name=='postgresql':
                await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
            now=datetime.now(timezone.utc)
            games=list((await session.scalars(select(Game).where(Game.status=='Scheduled',Game.time_precision=='exact',
                Game.game_type.in_(('regular','play_in','playoffs')),
                Game.scheduled_start_utc>=now+timedelta(minutes=50),
                Game.scheduled_start_utc<=now+timedelta(minutes=60)).order_by(Game.scheduled_start_utc,Game.id))).all())
            for game in games:
                path=folder/'records'/f'{game.id}.json'
                if path.exists():
                    old=json.loads(path.read_text())
                    if old['payload_sha256']!=digest(old['payload']) or old['payload'].get('protocol_sha256')!=digest(p):
                        raise ValueError('Existing observation changed')
                    continue
                as_of=datetime.now(timezone.utc)
                if not in_window(game,as_of): continue
                day=str(game.source_game_date or game.scheduled_start_utc.date())
                if any(day<=m['training_data_until'] for m in (manifest,p['baseline'])): continue
                fv=await FeatureBuilder().build_features(game.home_team_id,game.away_team_id,game.scheduled_start_utc,
                    session=session,as_of=as_of,target_game_id=game.id,use_historical=False,lookback_games=10)
                candidate=predict_win_bundle(root/'win_shadows'/p['version'],fv.features,target_day=day,as_of=as_of)[0]
                old=XGBoostTrainer();old.load(p['baseline']['xgboost_version'])
                adapter=Predictor(xgb_trainer=old,release_manifest=p['baseline'])
                X=adapter._model_features(adapter._prepare_release_features(fv.features.reshape(1,-1)),int(old.model_win.n_features_in_))
                baseline=float(old.model_win.predict_proba(X)[0,1])
                injuries={}
                for side,tid in (('home',game.home_team_id),('away',game.away_team_id)):
                    report=await latest_injury_snapshot(session,game.id,tid,as_of,reconstructed=False)
                    injuries[side]=None if report is None else dict(id=report.id,sha256=report.source_sha256,
                        published_at=report.published_at.isoformat(),observed_at=report.observed_at.isoformat(),complete=report.is_complete)
                payload=dict(kind=p['kind'],acceptance_run=False,protocol_sha256=digest(p),game_id=game.id,
                    as_of=as_of.isoformat(),created_at=datetime.now(timezone.utc).isoformat(),
                    scheduled_start_utc=game.scheduled_start_utc.isoformat(),source_game_date=day,
                    win_probabilities=dict(baseline=baseline,candidate=candidate['win_prob_home']),
                    builder_vector=fv.features.tolist(),feature_names=FEATURE_NAMES,injury_snapshots=injuries,
                    baseline_input=adapter._input_snapshot(X,'xgboost',old.version),
                    candidate_input=input_snapshot(prepare_win_input(fv.features,manifest)[0],FEATURE_NAMES,
                        model_name='xgboost_win',model_version=p['version'],manifest=manifest),warnings=fv.data_quality_warnings)
                checked_protocol(root,folder)
                if datetime.now(timezone.utc)>=game.scheduled_start_utc: continue
                count+=int(publish_once(path,dict(payload=payload,payload_sha256=digest(payload))))
    return count


async def status(root,folder):
    from ml.src.evaluation import classification_metrics
    root,folder=Path(root),Path(folder);p,_=checked_protocol(root,folder)
    values=[];errors=0
    async with async_session_factory() as session:
        for path in sorted((folder/'records').glob('*.json')):
            value=json.loads(path.read_text());row=value['payload']
            if value['payload_sha256']!=digest(row) or row['protocol_sha256']!=digest(p): errors+=1;continue
            game=await session.get(Game,row['game_id'])
            if game and game.status=='Final' and game.score_complete:
                values.append((float(game.home_score>game.away_score),row['win_probabilities']))
    return dict(kind=p['kind'],acceptance_run=False,version=p['version'],captured=len(list((folder/'records').glob('*.json'))),
        settled=len(values),integrity_errors=errors,metrics={name:classification_metrics(
            np.array([v[0] for v in values]),np.array([v[1][name] for v in values]))
            for name in ('baseline','candidate')} if values else {})
