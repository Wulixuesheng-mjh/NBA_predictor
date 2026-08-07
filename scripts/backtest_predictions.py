"""Generate backtest predictions for completed games.

Runs inside the Docker backend container. For each completed game with scores,
builds features using data from BEFORE the game date, generates predictions,
and evaluates them against actual results.

Usage:
  docker exec nba-backend python scripts/backtest_predictions.py --limit 100
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure ml and app modules are importable
# __file__ = /app/scripts/backtest_predictions.py → parent.parent = /app
_APP_DIR = str(Path(__file__).parent.parent)
sys.path.insert(0, _APP_DIR)           # /app → ml.src.*, app.*
sys.path.insert(0, _APP_DIR + "/ml")   # /app/ml → src.*


async def main(limit: int = 100, skip_existing: bool = True):
    """Run backtest on completed games."""
    from sqlalchemy import select, and_, func
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

    from app.config import settings as backend_settings
    from app.models.game import Game
    from app.models.team import Team
    from app.models.prediction import Prediction
    from app.models.prediction_results import PredictionResult

    from ml.src.predictor import load_predictor
    from ml.src.features.feature_builder import FeatureBuilder

    # Connect to the same database as backend
    engine = create_async_engine(backend_settings.database_url, echo=False)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        # 1. Get predictor
        print("Loading predictor...")
        predictor = load_predictor()
        if predictor is None or predictor.xgb_trainer is None:
            print("ERROR: Could not load predictor / XGBoost model")
            return
        print(f"  XGBoost: v{predictor.xgb_trainer.version}")
        print(f"  MLP: {'available' if predictor.mlp_trainer else 'not available'}")

        # 2. Find completed games with scores, ordered by date (oldest first for
        #    realistic backtest: early-season games get smaller lookback windows)
        subq = (
            select(Game.id)
            .where(
                and_(
                    Game.status == "Final",
                    Game.home_score.isnot(None),
                    Game.away_score.isnot(None),
                )
            )
            .order_by(Game.game_date.desc())
            .limit(limit * 3)  # oversample to allow for failures
        )
        if skip_existing:
            # Exclude games that already have predictions
            existing_subq = select(Prediction.game_id).distinct()
            subq = subq.where(Game.id.not_in(existing_subq))

        subq = subq.order_by(Game.game_date.asc())  # oldest first for realistic backtest
        subq = subq.limit(limit)

        result = await session.execute(subq)
        game_ids = [row[0] for row in result.all()]
        print(f"\nFound {len(game_ids)} games to process (limit={limit}, skip_existing={skip_existing})")

        if not game_ids:
            print("No games to process.")
            return

        # 3. Process each game
        builder = FeatureBuilder()
        success = 0
        skipped = 0
        failed = 0

        t_start = time.monotonic()
        for i, game_id in enumerate(game_ids):
            try:
                # Get game info
                game_result = await session.execute(
                    select(Game).where(Game.id == game_id)
                )
                game = game_result.scalar_one()

                # Skip if game_date is too early (no prior data for lookback)
                home_team = await session.execute(
                    select(Team).where(Team.id == game.home_team_id)
                )
                away_team = await session.execute(
                    select(Team).where(Team.id == game.away_team_id)
                )
                home = home_team.scalar_one_or_none()
                away = away_team.scalar_one_or_none()

                if home is None or away is None:
                    skipped += 1
                    continue

                # Build features as of game date (uses data from BEFORE the game)
                try:
                    fv = await builder.build_features(
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        game_date=game.game_date,
                        lookback_games=10,
                        session=session,
                    )
                except ValueError as e:
                    # Not enough historical data for this game
                    if "not enough" in str(e).lower() or "no games" in str(e).lower():
                        skipped += 1
                        continue
                    raise

                if fv is None:
                    skipped += 1
                    continue

                # Generate prediction
                prediction = predictor.predict(fv, apply_calibration=True)
                xgb = prediction.xgboost
                if xgb is None:
                    skipped += 1
                    continue

                # Actual results
                actual_spread = (game.home_score or 0) - (game.away_score or 0)
                actual_total = (game.home_score or 0) + (game.away_score or 0)
                actual_winner = "home" if (game.home_score or 0) > (game.away_score or 0) else "away"

                # Evaluate
                win_correct = (
                    (xgb.win_prob_home > 0.5 and actual_winner == "home")
                    or (xgb.win_prob_home <= 0.5 and actual_winner == "away")
                )
                spread_correct = None
                if xgb.predicted_spread is not None:
                    # "Correct" if predicted spread direction matches actual
                    spread_correct = (
                        (xgb.predicted_spread > 0 and actual_spread > 0)
                        or (xgb.predicted_spread < 0 and actual_spread < 0)
                    )

                total_correct = None
                if xgb.predicted_total is not None:
                    total_correct = (
                        (xgb.predicted_total > 228.5 and actual_total > 228.5)
                        or (xgb.predicted_total <= 228.5 and actual_total <= 228.5)
                    )

                # Store prediction
                pred_record = Prediction(
                    game_id=game.id,
                    model_name="xgboost",
                    model_version=xgb.model_version,
                    predicted_at=game.game_date.replace(
                        hour=12, minute=0, second=0, tzinfo=timezone.utc
                    ) if game.game_date else datetime.now(timezone.utc),
                    win_prob_home=xgb.win_prob_home,
                    win_prob_away=xgb.win_prob_away,
                    predicted_spread=xgb.predicted_spread,
                    spread_confidence=xgb.spread_confidence,
                    predicted_total=xgb.predicted_total,
                    total_confidence=xgb.total_confidence,
                    calibrated=True,
                    lookback_games=10,
                )
                session.add(pred_record)
                await session.flush()

                # Store evaluation — use game date + 1 day (results available next day)
                eval_date = (
                    game.game_date.replace(hour=8, minute=0, second=0, tzinfo=timezone.utc)
                    + timedelta(days=1)
                    if game.game_date
                    else datetime.now(timezone.utc)
                )

                eval_record = PredictionResult(
                    prediction_id=pred_record.id,
                    actual_spread=actual_spread,
                    actual_total=actual_total,
                    actual_winner=actual_winner,
                    spread_correct=spread_correct,
                    total_correct=total_correct,
                    win_correct=win_correct,
                    evaluated_at=eval_date,
                )
                session.add(eval_record)

                success += 1

                # Commit in batches
                if success % 20 == 0:
                    await session.commit()
                    elapsed = time.monotonic() - t_start
                    rate = success / elapsed if elapsed > 0 else 0
                    print(
                        f"  [{success}/{len(game_ids)}] "
                        f"{home.abbreviation} vs {away.abbreviation} "
                        f"({game.game_date.date()}) — "
                        f"win={xgb.win_prob_home:.0%}H "
                        f"spread={xgb.predicted_spread:+.1f} (actual={actual_spread:+.0f}) "
                        f"total={xgb.predicted_total:.0f} (actual={actual_total:.0f}) "
                        f"| {rate:.1f} games/s"
                    )

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  [FAIL] game {game_id}: {type(e).__name__}: {e}")
                continue

        # Final commit
        await session.commit()

        elapsed = time.monotonic() - t_start
        print(f"\n{'='*60}")
        print(f"Backtest complete in {elapsed:.1f}s")
        print(f"  Success: {success}")
        print(f"  Skipped: {skipped}")
        print(f"  Failed:  {failed}")
        print(f"  Rate:    {success/elapsed:.1f} games/s" if elapsed > 0 else "")
        print(f"{'='*60}")

        # Show summary stats using raw SQL
        from sqlalchemy import text as sql_text
        r = await session.execute(sql_text(
            "SELECT COUNT(*) as cnt, "
            "AVG(CASE WHEN pr.win_correct THEN 1.0 ELSE 0.0 END) as win_acc, "
            "AVG(ABS(pr.actual_spread - p.predicted_spread)) as spread_mae, "
            "AVG(ABS(pr.actual_total - p.predicted_total)) as total_mae "
            "FROM prediction_results pr "
            "JOIN predictions p ON pr.prediction_id = p.id"
        ))
        row = r.one()
        total_eval, win_acc, spread_mae, total_mae = row
        print(f"\n  Total evaluated: {total_eval}")
        print(f"  Win accuracy:    {win_acc:.1%}" if win_acc is not None else "  Win accuracy: N/A")
        print(f"  Spread MAE:      {spread_mae:.1f}" if spread_mae is not None else "  Spread MAE: N/A")
        print(f"  Total MAE:       {total_mae:.1f}" if total_mae is not None else "  Total MAE: N/A")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate backtest predictions")
    parser.add_argument("--limit", type=int, default=100, help="Max games to process")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip games that already have predictions")
    args = parser.parse_args()
    asyncio.run(main(limit=args.limit, skip_existing=args.skip_existing))
