"""One immutable actual-time win-only diagnostic; never formal T-60 acceptance."""
import asyncio,argparse,json,hashlib
from datetime import datetime,timezone,timedelta
from pathlib import Path
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team
from ml.src.config import ml_settings
from ml.src.features.feature_builder import FeatureBuilder,FEATURE_NAMES
from ml.src.win_shadow import prepare_win_input,predict_win_bundle
from ml.src.inference_snapshot import input_snapshot,checksum


async def main(args):
    folder=Path(ml_settings.models_dir)/'win_shadows'/args.version
    manifest=json.loads((folder/'manifest.json').read_text())
    validation=json.loads((folder/'validation.json').read_text())
    if validation.get('status')!='PASS' or validation.get('version')!=args.version:
        raise ValueError('Win candidate has not passed engineering validation')
    now=datetime.now(timezone.utc)
    async with async_session_factory() as session:
        game=await session.scalar(select(Game).where(Game.status=='Scheduled',Game.time_precision=='exact',
            Game.scheduled_start_utc>now+timedelta(hours=2),
            Game.game_type.in_(('regular','play_in','playoffs'))).order_by(Game.scheduled_start_utc,Game.id).limit(1))
        if game is None: raise ValueError('No exact future game available')
        start=game.scheduled_start_utc
        target_day=str(game.source_game_date or start.date())
        if target_day<=manifest['training_data_until']: raise ValueError('Target is within training')
        fv=await FeatureBuilder().build_features(game.home_team_id,game.away_team_id,start,
            session=session,target_game_id=game.id,as_of=now,use_historical=False,lookback_games=10)
        output=predict_win_bundle(folder,fv.features,target_day=target_day,as_of=now)[0]
        names={t.id:t.full_name for t in (await session.scalars(select(Team))).all()}
        if datetime.now(timezone.utc)>=start: raise ValueError('Game started during capture')
        processed=prepare_win_input(fv.features,manifest)
        record=dict(mode='manual_diagnostic',acceptance_run=False,capture_lead_rule='actual server time',
            game_id=game.id,home=names[game.home_team_id],away=names[game.away_team_id],
            as_of=now.isoformat(),scheduled_start_utc=start.isoformat(),source_game_date=target_day,
            manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
            training_data_until=manifest['training_data_until'],output=output,
            feature_names=FEATURE_NAMES,builder_vector=fv.features.tolist(),
            processed_input=input_snapshot(processed[0],FEATURE_NAMES,model_name='xgboost_win',
                model_version=args.version,manifest=manifest),data_quality=dict(warnings=fv.data_quality_warnings,
                home_games_available=fv.home_games_available,away_games_available=fv.away_games_available))
    path=folder/'manual_diagnostics'/f'{record["game_id"]}_{now.strftime("%Y%m%dT%H%M%S%f")}.json'
    path.parent.mkdir(exist_ok=True)
    with path.open('x') as stream: json.dump(record|dict(payload_sha256=checksum(record)),stream,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k not in ('builder_vector','processed_input','feature_names')}
        |dict(path=str(path)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);asyncio.run(main(p.parse_args()))
