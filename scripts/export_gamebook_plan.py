"""Read-only manifest of incomplete development-season official NBA games."""
import asyncio
import argparse
import json
from pathlib import Path
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team


async def main(args):
    async with async_session_factory() as session:
        teams = {t.id:t for t in (await session.execute(select(Team))).scalars()}
        games = (await session.execute(select(Game).where(
            Game.season.in_(args.seasons), Game.status=='Final',
            Game.game_type.in_(('regular','play_in','playoffs')),
            Game.player_stats_complete.is_(False),Game.score_complete.is_(True),
        ).order_by(Game.game_date))).scalars().all()
        result = []
        for game in games:
            if args.game_ids and game.id not in args.game_ids:
                continue
            day = str(game.source_game_date or game.game_date.date()).replace('-','')
            home,away = teams[game.home_team_id],teams[game.away_team_id]
            filename = f'{day}_{away.abbreviation}{home.abbreviation}_book.pdf'
            result.append(dict(game_id=game.id,nba_game_id=game.nba_game_id,season=game.season,
                game_date=str(game.source_game_date or game.game_date.date()),
                home_team_id=home.id,away_team_id=away.id,home_name=home.full_name,away_name=away.full_name,
                home_score=game.home_score,away_score=game.away_score,filename=filename,
                url=f'https://statsdmz.nba.com/pdfs/{day}/{filename}'))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(games=len(result),path=str(path))))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', nargs='+', default=['2023-24','2024-25'])
    parser.add_argument('--game-ids', nargs='+', type=int)
    parser.add_argument('--output', default='/app/ml/saved_models/data_quality/dev_gamebook_plan.json')
    asyncio.run(main(parser.parse_args()))
