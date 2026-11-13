"""Read-only explanation of live-source writes during a pinned cache build."""
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
from sqlalchemy import select, func
from app.database import async_session_factory, Base
import app.models  # register source tables
from app.models.game import Game
from app.models.data_revision import DataRevision
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES


async def main(args):
    since=datetime.fromisoformat(args.since)
    sources=('games','team_game_stats','game_player_stats','players','player_injuries',
        'teams','injury_reports','roster_snapshots','box_score_snapshots','historical_player_injuries')
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        key=await DatasetCache().key(session,feature_names=FEATURE_NAMES,seasons=None,
            lookback_games=10,min_games_before=5,use_historical=True)
        changes={}
        for name in sources:
            table=Base.metadata.tables[name]
            field=next((table.c[n] for n in ('updated_at','observed_at','created_at') if n in table.c),None)
            if field is not None:
                changes[name]=await session.scalar(select(func.count()).select_from(table).where(field>=since))
        games=list((await session.scalars(select(Game).where(Game.updated_at>=since).order_by(Game.id))).all())
        revision=await session.scalar(select(func.max(DataRevision.revision)))
    report=dict(pinned_key=args.cache_key,current_key=key,key_matches=key==args.cache_key,
        current_revision=revision,checked_changes_since=since.isoformat(),source_tables_changed_rows=changes,
        changed_games=[dict(id=g.id,status=g.status,source_game_date=str(g.source_game_date),
            updated_at=g.updated_at.isoformat(),team_complete=g.team_stats_complete,
            player_complete=g.player_stats_complete) for g in games],
        writes=0,interpretation='Pinned cache provenance is not relabelled to a later live revision.')
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache-key',required=True)
    parser.add_argument('--since',required=True);parser.add_argument('--output',required=True)
    asyncio.run(main(parser.parse_args()))
