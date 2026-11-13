"""Transactional acceptance for stable source IDs and NBA-day rest."""
import asyncio
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import func, select
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team
from app.collector.balldontlie_collector import BallDontLieCollector
from ml.src.features.context_features import ContextFeatures


async def main():
    async with async_session_factory() as session:
        try:
            game = (await session.execute(select(Game).where(
                Game.nba_game_id.like('bdl-%'), Game.time_precision == 'exact',
            ).limit(1))).scalar_one()
            original_day, start = game.source_game_date, game.scheduled_start_utc
            home, away = await session.get(Team, game.home_team_id), await session.get(Team, game.away_team_id)
            count_before = await session.scalar(select(func.count(Game.id)))
            game.source_game_date = original_day - timedelta(days=1)
            await session.flush()
            payload = dict(id=int(game.nba_game_id.split('-')[1]), date=str(original_day),
                datetime=start.isoformat(), season=int(game.season[:4]), status=game.status,
                home_team={'abbreviation':home.abbreviation}, visitor_team={'abbreviation':away.abbreviation},
                home_team_score=game.home_score, visitor_team_score=game.away_score)
            collector = BallDontLieCollector(api_key='unused-test')
            await collector._upsert_game(session, payload,
                {home.abbreviation:home.id, away.abbreviation:away.id}, game.status)
            await session.flush()
            assert game.source_game_date == original_day and game.game_date == start
            assert await session.scalar(select(func.count(Game.id))) == count_before
            prior = Game(nba_game_id='test-'+uuid4().hex[:12], season='2039-40',
                source_game_date=date(2040,1,1), game_date=datetime(2040,1,2,4,tzinfo=timezone.utc),
                scheduled_start_utc=datetime(2040,1,2,4,tzinfo=timezone.utc), time_precision='exact',
                game_type='regular', status='Final', home_team_id=home.id, away_team_id=away.id,
                home_score=110, away_score=100)
            session.add(prior)
            await session.flush()
            rest = await ContextFeatures._compute_rest_days(session, home.id,
                datetime(2040,1,3,0,tzinfo=timezone.utc), target_game_day=date(2040,1,2))
            assert rest == 1, 'Consecutive NBA days less than 24 hours apart must be back-to-back'
        finally:
            await session.rollback()
    print('PASS: stable source ID, date/time repair, no duplicate game, NBA-day back-to-back')


if __name__ == '__main__':
    asyncio.run(main())
