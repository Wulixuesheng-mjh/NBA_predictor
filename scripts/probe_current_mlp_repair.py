"""Same-input live diagnostic, not production activation or formal acceptance."""
import argparse,asyncio,json,importlib.util,sys,hashlib
from datetime import datetime,timezone
from pathlib import Path
from sqlalchemy import text
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team
from ml.src.features.feature_builder import FeatureBuilder
from ml.src.predictor import load_predictor


async def main(args):
    spec=importlib.util.spec_from_file_location('mlp_fix_probe','/tmp/mlp_fix_probe.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    baseline=load_predictor();fixed=load_predictor()
    fixed.mlp_trainer=module.CompatibleLegacyMLP(fixed.mlp_trainer,'/tmp/mlp_fix_contract.json')
    as_of=datetime.now(timezone.utc)
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await session.execute(text('SET TRANSACTION READ ONLY'))
        game=await session.get(Game,32798)
        home=await session.get(Team,game.home_team_id);away=await session.get(Team,game.away_team_id)
        assert (home.abbreviation,away.abbreviation)==('GSW','NOP')
        if game.scheduled_start_utc<=as_of:raise ValueError('Not pregame; do not backdate')
        fv=await FeatureBuilder().build_features(game.home_team_id,game.away_team_id,game.scheduled_start_utc,
            session=session,as_of=as_of,target_game_id=game.id,use_historical=False)
        before=baseline.predict(fv);after=fixed.predict(fv)
        assert before.xgboost==after.xgboost
    record=dict(kind='manual_same_input_pregame_diagnostic_not_acceptance',game_id=game.id,
        home=home.abbreviation,away=away.abbreviation,
        as_of=as_of.isoformat(),scheduled_start=game.scheduled_start_utc.isoformat(),
        before=before.to_dict(),after=after.to_dict(),raw_features=fv.features.tolist(),
        feature_names=fv.feature_names,contract=module.contract_spec(),production_activated=False,database_writes=0)
    data=json.dumps(record,sort_keys=True,allow_nan=False)
    output=Path(args.output)
    if output.exists():raise ValueError('Preserve prior diagnostic')
    output.write_text(json.dumps(dict(payload=record,sha256=hashlib.sha256(data.encode()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(dict(as_of=record['as_of'],game_id=game.id,before=record['before']['mlp'],after=record['after']['mlp'],
        xgb_unchanged=True,production_activated=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='/tmp/current_gsw_nop_mlp_comparison_20261110.json')
    asyncio.run(main(parser.parse_args()))
