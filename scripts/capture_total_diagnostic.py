"""Append-only real-time pregame observations, separate from 200-game acceptance."""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.services.injury_snapshot_reader import latest_injury_snapshot
from app.services.prospective_service import assert_pregame, feature_code_hash
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FeatureBuilder, FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.predictor import Predictor
from diagnose_current_totals import sha


async def main(args):
    root=Path(ml_settings.models_dir)
    active_path=root/'active.json'
    active_bytes=active_path.read_bytes()
    active=json.loads(active_bytes)
    assert active['version']=='20261022_050952'
    bundle=Path(args.bundle)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    manifests,models={},{}
    for mode in ('direct','residual'):
        manifest=json.loads((bundle/f'{mode}_manifest.json').read_text())
        assert manifest['feature_names']==FEATURE_NAMES and not manifest['eligible_for_200_game_acceptance']
        for name,expected in manifest['file_hashes'].items():
            assert sha(bundle/'models'/mode/name)==expected
        model=XGBoostTrainer(models_dir=str(bundle/'models'));model.load(mode)
        manifests[mode],models[mode]=manifest,model
    old=XGBoostTrainer();old.load(active['xgboost_version'])
    adapter=Predictor(xgb_trainer=old,release_manifest=active)
    old_folder=root/'xgboost'/active['xgboost_version']
    old_hashes={p.name:sha(p) for p in old_folder.glob('*.json')}
    now=datetime.now(timezone.utc)
    captured=[]
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        games=(await session.execute(select(Game).where(Game.status=='Scheduled',
            Game.time_precision=='exact',Game.scheduled_start_utc>now,
            Game.scheduled_start_utc<=now+timedelta(hours=args.horizon_hours),
            Game.game_type.in_(('regular','play_in','playoffs'))).order_by(Game.scheduled_start_utc))).scalars().all()
        for game in games:
            path=output/f'{game.id}.json'
            if path.exists():
                continue
            as_of=datetime.now(timezone.utc)
            assert_pregame(game,as_of,[active,*manifests.values()])
            fv=await FeatureBuilder().build_features(game.home_team_id,game.away_team_id,
                game.scheduled_start_utc,session=session,as_of=as_of,target_game_id=game.id,use_historical=False)
            X=fv.features.reshape(1,-1)
            oldX=adapter._adapt_features(adapter._prepare_release_features(X),old.model_total.n_features_in_)
            vectors={'old_compatible':oldX[0].tolist()}
            totals={'old_compatible':float(old._predict_totals(oldX)[0])}
            for mode,model in models.items():
                value=X.copy()
                value[:,[FEATURE_NAMES.index(n) for n in manifests[mode]['disabled_features']]]=0
                vectors[mode]=value[0].tolist()
                totals[mode]=float(model._predict_totals(value)[0])
            injuries={}
            for side,tid in [('home',game.home_team_id),('away',game.away_team_id)]:
                report=await latest_injury_snapshot(session,game.id,tid,as_of,reconstructed=False)
                injuries[side]=None if report is None else dict(id=report.id,
                    published_at=report.published_at.isoformat(),observed_at=report.observed_at.isoformat(),
                    complete=report.is_complete,entries=report.entries_json,
                    raw_complete=getattr(report,'raw_is_complete',report.is_complete))
            assert_pregame(game,as_of,[active,*manifests.values()])
            assert active_path.read_bytes()==active_bytes
            record=dict(mode='diagnostic_pregame_shadow',acceptance_run=False,game_id=game.id,
                nba_game_id=game.nba_game_id,as_of=as_of.isoformat(),
                scheduled_start_utc=game.scheduled_start_utc.isoformat(),source_game_date=str(game.source_game_date),
                home_team_id=game.home_team_id,away_team_id=game.away_team_id,
                lead_minutes=(game.scheduled_start_utc-as_of).total_seconds()/60,
                totals=totals,baseline=float(X[0,FEATURE_NAMES.index('shrunk_expected_total')]),
                warnings=fv.data_quality_warnings,injury_snapshots=injuries,
                feature_names={k:FEATURE_NAMES[:len(v)] for k,v in vectors.items()},
                processed_vectors=vectors,feature_code_sha256=feature_code_hash(),
                baseline_manifest=active,old_file_hashes=old_hashes,candidate_manifests=manifests,
                plausibility={k:150<=v<=300 for k,v in totals.items()},
                rules='Frozen models; first snapshot per game; no retrospective capture or interim retuning')
            record['record_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
            with path.open('x',encoding='utf-8') as handle:
                json.dump(record,handle,indent=2)
            captured.append(game.id)
    print(json.dumps(dict(captured=len(captured),game_ids=captured,
        acceptance_games_added=0,active_unchanged=active_path.read_bytes()==active_bytes)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--horizon-hours',type=float,default=36)
    asyncio.run(main(parser.parse_args()))
