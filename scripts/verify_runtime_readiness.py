"""Read-only production baseline smoke; no ledger or cohort writes."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, select
from app.database import async_session_factory
from app.models.game import Game
from app.models.prediction import Prediction
from app.services.prediction_service import PredictionService
from app.collector.kaggle_collector import KaggleCollector


async def main():
    folder = await KaggleCollector()._ensure_dataset()
    async with async_session_factory() as session:
        before = await session.scalar(select(func.count(Prediction.id)))
        game = (await session.execute(select(Game).where(Game.status == 'Scheduled',
            Game.time_precision == 'exact', Game.scheduled_start_utc > datetime.now(timezone.utc),
        ).order_by(Game.scheduled_start_utc).limit(1))).scalar_one()
        output = await PredictionService.generate_prediction(session,game_id=game.id,
            prediction_mode='pregame',persist=False)
        assert output.get('xgboost'), output.get('data_quality')
        past = (await session.execute(select(Game).where(Game.status == 'Final',
            Game.time_precision == 'exact').limit(1))).scalar_one()
        try:
            await PredictionService.generate_prediction(session,game_id=past.id,
                as_of=past.scheduled_start_utc-timedelta(minutes=30),prediction_mode='pregame',persist=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Postgame backdated ledger write was accepted')
        assert await session.scalar(select(func.count(Prediction.id))) == before
        print(json.dumps(dict(status='PASS',source_dir=folder, game_id=game.id,
            xgboost=output['xgboost'], mlp=output.get('mlp'), data_quality=output.get('data_quality'),
            ledger_unchanged=True),indent=2,default=str))


if __name__ == '__main__':
    asyncio.run(main())
