"""Locked T-minus-60 diagnostic capture and read-only score settlement."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from sqlalchemy import select,text
from app.models.game import Game
from app.models.prediction import Prediction
from app.models.shadow_observation import ShadowObservationRun,ShadowObservation
from app.services.prospective_service import digest,feature_code_hash,assert_pregame


def model_root():
    from ml.src.config import ml_settings
    return Path(ml_settings.models_dir).resolve()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def service_hash():
    from app.services import shadow_total_service
    return digest({p.name:sha(p) for p in (Path(__file__),Path(shadow_total_service.__file__))})


def in_window(game,as_of):
    if game.status!='Scheduled' or game.time_precision!='exact' or not game.scheduled_start_utc:
        return False
    lead=(game.scheduled_start_utc-as_of).total_seconds()/60
    return 50<=lead<=60


def _verify_protocol(run):
    root=model_root();p=run.protocol
    if digest(p)!=run.protocol_hash or p['feature_code_hash']!=feature_code_hash() or p['service_hash']!=service_hash():
        return False
    if sha(root/'active.json')!=p['active_sha256'] or sha(root/'shadow_totals.json')!=p['catalog_sha256']:
        return False
    folder=root/'xgboost'/p['baseline']['xgboost_version']
    return all(sha(folder/name)==expected for name,expected in p['baseline_files'].items())


def verify_protocol(run):
    try:
        return _verify_protocol(run)
    except (OSError,KeyError,ValueError):
        return False


async def lock_diagnostic_run(session):
    root=model_root()
    active=json.loads((root/'active.json').read_text())
    catalog=json.loads((root/'shadow_totals.json').read_text())
    assert catalog['status']=='diagnostic_shadow' and not catalog['primary_promotion']
    folder=root/'xgboost'/active['xgboost_version']
    protocol=dict(purpose='diagnostic_only',catalog_version=catalog['version'],
        catalog_sha256=sha(root/'shadow_totals.json'),active_sha256=sha(root/'active.json'),
        baseline=active,baseline_files={p.name:sha(p) for p in folder.glob('*.json')},
        feature_code_hash=feature_code_hash(),service_hash=service_hash(),
        capture_lead_minutes=60,capture_window_minutes=10,lookback_games=10,
        rule='First actual server-time snapshot in T-60..T-50; no backdating or retuning',
        acceptance_run=False)
    name=catalog['version']+'_tminus60'
    existing=await session.scalar(select(ShadowObservationRun).where(ShadowObservationRun.name==name))
    if existing:
        if existing.protocol_hash!=digest(protocol):
            raise ValueError('Diagnostic protocol is immutable; use a new version')
        return existing
    run=ShadowObservationRun(name=name,protocol=protocol,protocol_hash=digest(protocol),enabled=True)
    session.add(run);await session.flush()
    return run


async def capture_one(session,run,game):
    from ml.src.features.feature_builder import FEATURE_NAMES,FeatureBuilder
    from ml.src.models.xgboost_model import XGBoostTrainer
    from ml.src.predictor import Predictor
    from app.services.shadow_total_service import shadow_totals
    from app.services.injury_snapshot_reader import latest_injury_snapshot
    if not verify_protocol(run):
        raise ValueError('Frozen diagnostic configuration changed; capture stopped')
    if session.bind.dialect.name=='postgresql':
        await session.execute(text('SELECT pg_advisory_xact_lock(:key)'),
            {'key':(83<<48)|(run.id<<32)|game.id})
    existing=await session.scalar(select(ShadowObservation).where(
        ShadowObservation.run_id==run.id,ShadowObservation.game_id==game.id))
    if existing:
        return False
    as_of=datetime.now(timezone.utc)
    if not in_window(game,as_of):
        return False
    baseline=run.protocol['baseline']
    fv=await FeatureBuilder().build_features(game.home_team_id,game.away_team_id,
        game.scheduled_start_utc,session=session,as_of=as_of,target_game_id=game.id,use_historical=False)
    shadow=shadow_totals(fv.features,as_of=as_of,lookback=10,
        target_day=str(game.source_game_date),root=model_root())
    if not shadow or shadow['status']!='diagnostic_shadow':
        raise ValueError('Pinned shadow models unavailable')
    assert_pregame(game,as_of,[baseline,*[dict(training_data_until=c['training_data_until'])
        for c in shadow['candidates']]])
    old=XGBoostTrainer(models_dir=str(model_root()/'xgboost'));old.load(baseline['xgboost_version'])
    adapter=Predictor(xgb_trainer=old,release_manifest=baseline)
    X=adapter._adapt_features(adapter._prepare_release_features(fv.features.reshape(1,-1)),old.model_total.n_features_in_)
    totals={'baseline':float(old._predict_totals(X)[0])}
    totals.update({c['strategy']:c['raw_predicted_total'] for c in shadow['candidates']})
    injuries={}
    for side,tid in [('home',game.home_team_id),('away',game.away_team_id)]:
        report=await latest_injury_snapshot(session,game.id,tid,as_of,reconstructed=False)
        injuries[side]=None if report is None else dict(id=report.id,complete=report.is_complete,
            published_at=report.published_at.isoformat(),observed_at=report.observed_at.isoformat(),
            sha256=report.source_sha256,entries=report.entries_json)
    market=await session.scalar(select(Prediction).where(Prediction.game_id==game.id,
        Prediction.prediction_mode=='pregame',Prediction.as_of<=as_of,
        Prediction.market_total.is_not(None)).order_by(Prediction.as_of.desc(),Prediction.id.desc()).limit(1))
    payload=dict(purpose='diagnostic_only',acceptance_run=False,protocol_hash=run.protocol_hash,
        source_game_date=str(game.source_game_date),totals=totals,shadow=shadow,
        baseline_feature_names=FEATURE_NAMES[:X.shape[1]],baseline_processed_vector=X[0].tolist(),
        warnings=fv.data_quality_warnings,injury_snapshots=injuries,
        market=None if market is None else dict(source='manual_prediction_input',
            observed_at=market.as_of.isoformat(),total=market.market_total))
    assert_pregame(game,as_of,[baseline])
    if not verify_protocol(run):
        raise ValueError('Configuration changed while computing diagnostic observation')
    session.add(ShadowObservation(run_id=run.id,game_id=game.id,as_of=as_of,
        scheduled_start_utc=game.scheduled_start_utc,payload=payload,payload_hash=digest(payload)))
    await session.flush()
    return True


async def capture_due(session):
    now=datetime.now(timezone.utc);count=0
    runs=(await session.execute(select(ShadowObservationRun).where(ShadowObservationRun.enabled.is_(True)))).scalars().all()
    for run in runs:
        if not verify_protocol(run):
            continue
        prior=(await session.execute(select(ShadowObservation).where(ShadowObservation.run_id==run.id))).scalars().all()
        if any(r.payload_hash!=digest(r.payload) or r.payload.get('protocol_hash')!=run.protocol_hash for r in prior):
            continue
        games=(await session.execute(select(Game).where(Game.status=='Scheduled',Game.time_precision=='exact',
            Game.game_type.in_(('regular','play_in','playoffs')),
            Game.scheduled_start_utc>=now+timedelta(minutes=50),
            Game.scheduled_start_utc<=now+timedelta(minutes=60)).order_by(Game.scheduled_start_utc,Game.id))).scalars().all()
        for game in games:
            count+=int(await capture_one(session,run,game))
    return count


async def progress(session,run):
    from ml.src.evaluation import regression_metrics
    pairs=(await session.execute(select(ShadowObservation,Game).join(Game,Game.id==ShadowObservation.game_id)
        .where(ShadowObservation.run_id==run.id).order_by(Game.source_game_date,Game.id))).all()
    valid=[(r,g) for r,g in pairs if r.payload_hash==digest(r.payload) and
        r.payload.get('protocol_hash')==run.protocol_hash]
    finished=[(r,g) for r,g in valid if g.status=='Final' and g.score_complete and
        g.home_score is not None and g.away_score is not None]
    report=dict(name=run.name,version=run.protocol['catalog_version'],purpose='diagnostic_only',
        acceptance_run=False,captured=len(pairs),settled=len(finished),awaiting=len(valid)-len(finished),
        integrity_errors=len(pairs)-len(valid),configuration_matches=verify_protocol(run),
        enabled=run.enabled,metrics={},market_samples=0)
    if finished:
        actual=np.array([g.home_score+g.away_score for _,g in finished])
        for mode in ('baseline','direct','residual'):
            pred=np.array([r.payload['totals'][mode] for r,_ in finished])
            report['metrics'][mode]=regression_metrics(actual,pred)
        market=[(r,g) for r,g in finished if r.payload.get('market')]
        report['market_samples']=len(market)
        report['over_under']={}
        for mode in ('baseline','direct','residual'):
            wins=pushes=ties=0
            for r,g in market:
                line=r.payload['market']['total'];actual=g.home_score+g.away_score
                predicted=r.payload['totals'][mode]
                pushes+=int(actual==line);ties+=int(predicted==line)
                wins+=int(actual!=line and predicted!=line and (actual>line)==(predicted>line))
            report['over_under'][mode]=dict(correct=wins,push=pushes,no_direction=ties,
                eligible=sum(g.home_score+g.away_score!=r.payload['market']['total'] and
                             r.payload['totals'][mode]!=r.payload['market']['total'] for r,g in market))
    return report
