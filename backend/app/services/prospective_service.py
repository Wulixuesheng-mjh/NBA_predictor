"""Locked, paired pregame capture. Never backfill records after tipoff."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from sqlalchemy import select, text

from app.models.game import Game
from app.models.prospective import ProspectiveForecast, ProspectiveRun


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_bundle(root, manifest):
    for group, key in (('xgboost', 'xgboost_version'), ('mlp', 'mlp_version')):
        version = manifest.get(key)
        if not version:
            continue
        hashes = manifest.get('file_hashes', {}).get(group, {})
        if not hashes:
            raise ValueError(f'Missing pinned file hashes: {group}')
        for name, expected in hashes.items():
            path = (root/group/version/name).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError('Model bundle path missing or outside model root')
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Model file changed: {group}/{version}/{name}')


def feature_code_hash():
    from ml.src.features import feature_builder
    folder = Path(feature_builder.__file__).parent
    files = sorted(folder.glob('*.py')) + [folder.parent/'predictor.py',
        folder.parent/'models/xgboost_model.py',folder.parent/'models/mlp_model.py',
        folder.parent/'inference_snapshot.py', folder.parent/'uncertainty.py']
    from app.services import injury_snapshot_reader
    from app.collector import player_aliases
    files += [Path(injury_snapshot_reader.__file__),Path(player_aliases.__file__)]
    return digest({f'{p.parent.name}/{p.name}':hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def validate_record_inputs(record, protocol):
    """Certify each model's actual boundary input, not just the raw builder row."""
    from ml.src.features.feature_builder import FEATURE_NAMES
    from ml.src.inference_snapshot import validate_snapshot
    from ml.src.predictor import Predictor
    if record.feature_names != FEATURE_NAMES:
        raise ValueError('Acceptance builder feature order differs')
    raw = np.asarray(record.feature_vector, dtype=np.float64).reshape(1, -1)
    for label in ('baseline', 'candidate'):
        manifest = protocol[label]
        output = getattr(record, f'{label}_output')
        if output.get('release_manifest') != manifest:
            raise ValueError('Acceptance model release provenance differs')
        prepared = Predictor(release_manifest=manifest)._prepare_release_features(raw)
        for group in ('xgboost', 'mlp'):
            version = manifest.get(f'{group}_version')
            if version is None:
                continue
            snapshot = output.get('processed_inputs', {}).get(group)
            value = validate_snapshot(snapshot, model_name=group, model_version=version, manifest=manifest)
            dim = manifest.get('feature_dim', 166)
            if snapshot['feature_names'] != FEATURE_NAMES[:dim] or value.shape != (1, dim):
                raise ValueError('Acceptance model input schema differs')
            if not np.array_equal(value, prepared[:, :dim]):
                raise ValueError('Acceptance model input differs from release preprocessing')


async def lock_run(session, *, name, baseline, candidate, gate_evidence):
    """Call only after the candidate/rules are fixed; no automatic promotion."""
    from ml.src.config import ml_settings
    from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION
    from ml.src.features.feature_builder import FEATURE_NAMES
    root = Path(ml_settings.models_dir)
    if candidate.get('feature_semantic_version') != CACHE_FORMAT_VERSION:
        raise ValueError('Candidate must be trained with current feature semantics')
    if candidate.get('feature_names') != FEATURE_NAMES:
        raise ValueError('Candidate feature order is incompatible')
    if not gate_evidence.get('rules_locked') or not gate_evidence.get('coverage_report_sha256'):
        raise ValueError('Missing locked rules or injury coverage evidence')
    from app.services.release_gate_service import validate_offline_evidence
    validate_offline_evidence(root,candidate,gate_evidence.get('offline_gates'))
    active_bytes=(root/'active.json').read_bytes()
    active=json.loads(active_bytes)
    if baseline.get('origin_active_manifest', baseline)!=active:
        raise ValueError('Formal baseline differs from current production')
    active_sha256=hashlib.sha256(active_bytes).hexdigest()
    if baseline.get('origin_active_sha256',active_sha256)!=active_sha256:
        raise ValueError('Formal baseline source selector checksum differs')
    for manifest in (baseline, candidate):
        verify_bundle(root, manifest)
        if manifest.get('calibration_enabled'):
            raise ValueError('Prospective capture requires uncalibrated frozen releases')
        if not manifest.get('training_data_until'):
            raise ValueError('Unknown model training cutoff')
    protocol = dict(baseline=baseline, candidate=candidate, gate_evidence=gate_evidence,
        feature_semantic_version=CACHE_FORMAT_VERSION, feature_code_hash=feature_code_hash(),
        lookback_games=10, capture_lead_minutes=60, capture_window_minutes=10,
        clock='server_utc', report_every=50, interim_retuning=False,
        primary_model='xgboost', active_manifest_sha256=active_sha256, enabled=True)
    existing = await session.scalar(select(ProspectiveRun).where(ProspectiveRun.name == name))
    if existing:
        if existing.protocol_hash != digest(protocol):
            raise ValueError('An existing protocol cannot be overwritten')
        return existing
    run = ProspectiveRun(name=name, protocol=protocol, protocol_hash=digest(protocol), target_games=200)
    session.add(run)
    await session.flush()
    return run


def assert_pregame(game, as_of, manifests, now=None):
    now = now or datetime.now(timezone.utc)
    start = game.scheduled_start_utc
    if game.status == 'Final' or game.time_precision != 'exact' or start is None:
        raise ValueError('Exact future tipoff is required')
    if as_of > now or now >= start or as_of >= start or (now-as_of).total_seconds() > 120:
        raise ValueError('Past/future/backdated capture is forbidden')
    target_day = str(game.source_game_date or start.date())
    for manifest in manifests:
        if str(manifest['training_data_until'])[:10] >= target_day:
            raise ValueError('Target game is not beyond the model training cutoff')


async def capture_game(session, run, game):
    from ml.src.config import ml_settings
    from ml.src.features.feature_builder import FEATURE_NAMES, FeatureBuilder
    from ml.src.predictor import load_predictor
    from app.services.injury_snapshot_reader import latest_injury_snapshot
    protocol = run.protocol
    if digest(protocol) != run.protocol_hash or feature_code_hash() != protocol['feature_code_hash']:
        raise ValueError('Protocol or feature code changed after locking')
    if hashlib.sha256((Path(ml_settings.models_dir)/'active.json').read_bytes()).hexdigest()!=protocol.get('active_manifest_sha256'):
        raise ValueError('Production baseline changed after protocol locking')
    # Serialize duplicate workers in PostgreSQL, releasing with transaction.
    await session.execute(text('SELECT pg_advisory_xact_lock(:key)'),
                          {'key': (run.id << 32) | game.id})
    existing = await session.scalar(select(ProspectiveForecast).where(
        ProspectiveForecast.run_id == run.id, ProspectiveForecast.game_id == game.id))
    if existing:
        return existing
    as_of = datetime.now(timezone.utc)
    manifests = [protocol['baseline'], protocol['candidate']]
    assert_pregame(game, as_of, manifests)
    fv = await FeatureBuilder().build_features(game.home_team_id, game.away_team_id,
        game.scheduled_start_utc, lookback_games=protocol['lookback_games'], session=session,
        target_game_id=game.id, as_of=as_of, use_historical=False)
    outputs = []
    for manifest in manifests:
        verify_bundle(Path(ml_settings.models_dir), manifest)
        predictor = load_predictor(release_manifest=manifest)
        if not predictor.xgb_trainer or not predictor.xgb_trainer.is_trained:
            raise ValueError('Pinned XGB model is unavailable')
        if manifest.get('mlp_version') and predictor.mlp_trainer is None:
            raise ValueError('Pinned MLP model is unavailable')
        result = predictor.predict(fv, apply_calibration=False)
        output = result.to_dict()
        if output.get('xgboost') is None:
            raise ValueError('Primary model rejected its output')
        output['raw_outputs'] = result.raw_outputs
        if 'xgboost' not in result.processed_inputs:
            raise ValueError('Processed primary model input is absent')
        output['processed_inputs'] = result.processed_inputs
        output['release_manifest'] = manifest
        outputs.append(output)
    assert_pregame(game, as_of, manifests)
    reports = {}
    for side, team_id in (('home',game.home_team_id),('away',game.away_team_id)):
        report = await latest_injury_snapshot(session, game.id, team_id, as_of)
        reports[side] = None if report is None else dict(id=report.id,
            published_at=report.published_at.isoformat(), observed_at=report.observed_at.isoformat(),
            sha256=report.source_sha256, complete=report.is_complete,
            raw_complete=getattr(report,'raw_is_complete',report.is_complete),
            identity_resolved_names=getattr(report,'identity_resolved_names',[]))
    vector = np.asarray(fv.features, dtype=np.float64).tolist()
    feature_hash = digest(dict(names=FEATURE_NAMES, values=vector))
    row = ProspectiveForecast(run_id=run.id, game_id=game.id, as_of=as_of,
        scheduled_start_utc=game.scheduled_start_utc, feature_names=FEATURE_NAMES,
        feature_vector=vector, feature_hash=feature_hash, baseline_output=outputs[0],
        candidate_output=outputs[1], data_quality=dict(warnings=fv.data_quality_warnings,
        injury_snapshots=reports, mode='pregame', protocol_hash=run.protocol_hash), market_snapshot=None)
    validate_record_inputs(row, protocol)
    session.add(row)
    await session.flush()
    return row


async def capture_due(session):
    """No rows/rules are invented when no final candidate has been locked."""
    now = datetime.now(timezone.utc)
    runs = list((await session.execute(select(ProspectiveRun))).scalars())
    captured = 0
    for run in runs:
        if not run.protocol.get('enabled'):
            continue
        count = len((await session.execute(select(ProspectiveForecast.id).where(
            ProspectiveForecast.run_id == run.id))).all())
        if count >= run.target_games:
            continue
        lead, window = run.protocol['capture_lead_minutes'], run.protocol['capture_window_minutes']
        games = (await session.execute(select(Game).where(
            Game.time_precision == 'exact', Game.status != 'Final',
            Game.scheduled_start_utc >= now + timedelta(minutes=lead-window),
            Game.scheduled_start_utc <= now + timedelta(minutes=lead),
            Game.game_type.in_(('regular','play_in','playoffs')),
        ).order_by(Game.scheduled_start_utc, Game.id))).scalars().all()
        for game in games[:max(0,run.target_games-count)]:
            existing = await session.scalar(select(ProspectiveForecast.id).where(
                ProspectiveForecast.run_id == run.id, ProspectiveForecast.game_id == game.id))
            if existing is None:
                await capture_game(session, run, game)
                captured += 1
    return captured


async def progress_report(session, run):
    from ml.src.evaluation import classification_metrics, regression_metrics, paired_day_bootstrap
    pairs = (await session.execute(select(ProspectiveForecast, Game).join(
        Game, Game.id == ProspectiveForecast.game_id,
    ).where(ProspectiveForecast.run_id == run.id).order_by(
        Game.source_game_date, Game.game_date, Game.id,
    ))).all()
    completed = [(row,game) for row,game in pairs if game.status == 'Final'
        and game.score_complete and game.home_score is not None and game.away_score is not None]
    report = dict(run=run.name, protocol_hash=run.protocol_hash, captured=len(pairs),
        completed=len(completed), target=run.target_games, next_report=min(200,50*(len(completed)//50+1)),
        interim_retuning=False, promotion_performed=False, metrics={})
    if not completed:
        return report
    actual = np.array([[float(g.home_score>g.away_score), g.home_score-g.away_score,
                        g.home_score+g.away_score] for _,g in completed])
    losses = {}
    for name in ('baseline','candidate'):
        outputs = [getattr(row, name+'_output')['raw_outputs']['xgboost'] for row,_ in completed]
        pred = np.array([[o['win_prob_home'],o['predicted_spread'],o['predicted_total']] for o in outputs])
        report['metrics'][name] = dict(win=classification_metrics(actual[:,0],pred[:,0]),
            spread=regression_metrics(actual[:,1],pred[:,1]), total=regression_metrics(actual[:,2],pred[:,2]))
        losses[name] = np.column_stack(((pred[:,0]-actual[:,0])**2,
            np.abs(pred[:,1]-actual[:,1]), np.abs(pred[:,2]-actual[:,2])))
    info = [dict(source_game_date=str(g.source_game_date or g.game_date.date()),
                 game_date=g.game_date.isoformat()) for _,g in completed]
    report['paired_loss_difference_95ci'] = {target: paired_day_bootstrap(info,
        losses['baseline'][:,i], losses['candidate'][:,i]) for i,target in enumerate(('win','spread','total'))}
    return report
