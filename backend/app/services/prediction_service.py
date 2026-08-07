"""Prediction service — integrates ML module for predictions."""

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.prediction import Prediction
from app.models.prediction_results import PredictionResult
from app.models.game import Game
from app.models.team import Team
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Service for generating and retrieving predictions."""

    @staticmethod
    def _get_predictor():
        """Lazy-load the ML Predictor.

        Returns a Predictor instance or None if ML module is unavailable.
        """
        try:
            ml_path = str(Path(__file__).parent.parent.parent / "ml")
            if ml_path not in sys.path:
                sys.path.insert(0, ml_path)

            from ml.src.predictor import load_predictor
            return load_predictor()
        except ImportError as e:
            logger.warning("ML module not available: %s", e)
            return None
        except Exception as e:
            logger.error("Failed to load predictor: %s", e)
            return None

    @staticmethod
    async def generate_prediction(
        session: AsyncSession,
        home_team_id: int,
        away_team_id: int,
        lookback_games: int = 10,
    ) -> dict[str, Any]:
        """Generate a prediction for a matchup.

        Args:
            session: Database session.
            home_team_id: Home team internal ID.
            away_team_id: Away team internal ID.
            lookback_games: Number of recent games to use for features (5-15).

        Returns:
            Dict with prediction results or error info.
        """
        # Validate teams exist
        home_team = await session.execute(
            select(Team).where(Team.id == home_team_id)
        )
        home_team = home_team.scalar_one_or_none()
        if home_team is None:
            raise ValueError(f"Team with id {home_team_id} not found")

        away_team = await session.execute(
            select(Team).where(Team.id == away_team_id)
        )
        away_team = away_team.scalar_one_or_none()
        if away_team is None:
            raise ValueError(f"Team with id {away_team_id} not found")

        if home_team_id == away_team_id:
            raise ValueError("Home and away teams must be different")

        # Load predictor
        predictor = PredictionService._get_predictor()
        if predictor is None:
            return {
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_name": home_team.full_name,
                "away_team_name": away_team.full_name,
                "home_team_abbr": home_team.abbreviation,
                "away_team_abbr": away_team.abbreviation,
                "lookback_games": lookback_games,
                "actual_games_home": 0,
                "actual_games_away": 0,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
                "xgboost": None,
                "mlp": None,
                "consensus": None,
                "recommended": "unavailable",
                "data_quality": {
                    "home_games_available": 0,
                    "away_games_available": 0,
                    "warnings": ["ML module not available — models may not be trained yet"],
                },
                "calibration_applied": False,
            }

        # Build features and predict
        try:
            from ml.src.features.feature_builder import FeatureBuilder

            builder = FeatureBuilder()
            game_date = datetime.now(timezone.utc)
            t0 = time.monotonic()

            fv = await builder.build_features(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                game_date=game_date,
                lookback_games=lookback_games,
                session=session,
            )

            prediction = predictor.predict(fv, apply_calibration=True)
            result = prediction.to_dict()

            # SHAP explanation (best-effort, optional)
            try:
                from ml.src.features.feature_builder import FEATURE_NAMES, FEATURE_DISPLAY_NAMES
                explanation_raw = predictor.explain_prediction(
                    X=fv.features.reshape(1, -1),
                    feature_names=FEATURE_NAMES,
                    top_k=5,
                    display_names=FEATURE_DISPLAY_NAMES,
                )
                if explanation_raw:
                    result["explanation"] = {
                        "top_features": explanation_raw,
                        "explained_model": "xgboost_win",
                    }
                else:
                    result["explanation"] = None
            except Exception:
                logger.debug("SHAP explanation skipped", exc_info=True)
                result["explanation"] = None

            duration_ms = round((time.monotonic() - t0) * 1000, 1)
            logger.info(
                "Prediction generated successfully",
                extra={
                    "home_team": home_team.abbreviation,
                    "away_team": away_team.abbreviation,
                    "lookback": lookback_games,
                    "duration_ms": duration_ms,
                    "xgboost_available": result.get("xgboost") is not None,
                    "mlp_available": result.get("mlp") is not None,
                },
            )
        except Exception as e:
            logger.error(
                "Prediction generation failed",
                exc_info=True,
                extra={
                    "home_team_id": home_team_id,
                    "away_team_id": away_team_id,
                    "error_type": type(e).__name__,
                },
            )
            raise

        # Enrich with team names
        result["home_team_name"] = home_team.full_name
        result["away_team_name"] = away_team.full_name
        result["home_team_abbr"] = home_team.abbreviation
        result["away_team_abbr"] = away_team.abbreviation

        # Store prediction in database (always, even without a matching scheduled game)
        try:
            # Look up a matching scheduled game (best-effort, for result evaluation later)
            game_query = (
                select(Game)
                .where(
                    and_(
                        Game.home_team_id == home_team_id,
                        Game.away_team_id == away_team_id,
                        Game.status == "Scheduled",
                    )
                )
                .order_by(Game.game_date.asc())
                .limit(1)
            )
            game_result = await session.execute(game_query)
            game = game_result.scalar_one_or_none()

            # Save XGBoost prediction
            if result.get("xgboost"):
                xgb = result["xgboost"]
                pred_record = Prediction(
                    game_id=game.id if game is not None else None,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    model_name="xgboost",
                    model_version=xgb.get("model_version", "unknown"),
                    predicted_at=datetime.now(timezone.utc),
                    win_prob_home=xgb["win_prob_home"],
                    win_prob_away=xgb["win_prob_away"],
                    predicted_spread=xgb.get("predicted_spread"),
                    spread_confidence=xgb.get("spread_confidence"),
                    predicted_total=xgb.get("predicted_total"),
                    total_confidence=xgb.get("total_confidence"),
                    calibrated=result.get("calibration_applied", False),
                    features_json=fv.raw_features if hasattr(fv, 'raw_features') else None,
                    lookback_games=lookback_games,
                )
                session.add(pred_record)
                await session.commit()
                logger.info("Prediction #%d saved to history", pred_record.id)
        except Exception as e:
            logger.warning("Could not store prediction in DB: %s", e)

        return result

    @staticmethod
    async def list_predictions(
        session: AsyncSession,
        game_id: int | None = None,
        model_name: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[dict], int]:
        """List historical predictions with team names.

        Predictions without a linked game use the stored home_team_id/away_team_id
        for team name lookups (common during offseason when no schedule exists).
        """
        query = select(Prediction)
        count_query = select(func.count(Prediction.id))

        if game_id is not None:
            query = query.where(Prediction.game_id == game_id)
            count_query = count_query.where(Prediction.game_id == game_id)

        if model_name:
            query = query.where(Prediction.model_name == model_name)
            count_query = count_query.where(Prediction.model_name == model_name)

        query = query.order_by(desc(Prediction.predicted_at)).offset(offset).limit(limit)

        result = await session.execute(query)
        predictions = result.scalars().all()

        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0

        # Collect all team IDs for efficient batch lookup
        team_ids: set[int] = set()
        game_ids: set[int] = set()
        for p in predictions:
            if p.game_id:
                game_ids.add(p.game_id)
            if p.home_team_id:
                team_ids.add(p.home_team_id)
            if p.away_team_id:
                team_ids.add(p.away_team_id)

        # Batch lookup teams
        team_map: dict[int, tuple[str | None, str | None]] = {}
        if team_ids:
            team_query = select(Team.id, Team.full_name, Team.abbreviation).where(
                Team.id.in_(team_ids)
            )
            team_result = await session.execute(team_query)
            for row in team_result.all():
                team_map[row.id] = (row.full_name, row.abbreviation)

        # Batch lookup games
        game_map: dict[int, Game] = {}
        if game_ids:
            game_query = select(Game).where(Game.id.in_(game_ids))
            game_result = await session.execute(game_query)
            for g in game_result.scalars().all():
                game_map[g.id] = g

        # Build response items
        items = []
        for pred in predictions:
            # Resolve team names: prefer linked game, fallback to stored team IDs
            if pred.game_id and pred.game_id in game_map:
                g = game_map[pred.game_id]
                ht = team_map.get(g.home_team_id, (None, None))
                at = team_map.get(g.away_team_id, (None, None))
                home_name = ht[0]
                away_name = at[0]
                game_date = g.game_date
            else:
                ht = team_map.get(pred.home_team_id, (None, None)) if pred.home_team_id else (None, None)
                at = team_map.get(pred.away_team_id, (None, None)) if pred.away_team_id else (None, None)
                home_name = ht[0]
                away_name = at[0]
                game_date = None

            # Check if result exists
            has_result_query = select(PredictionResult).where(
                PredictionResult.prediction_id == pred.id
            )
            has_result = await session.execute(has_result_query)
            has_result = has_result.scalar_one_or_none() is not None

            items.append({
                "id": pred.id,
                "game_id": pred.game_id,
                "home_team_id": pred.home_team_id,
                "away_team_id": pred.away_team_id,
                "model_name": pred.model_name,
                "model_version": pred.model_version,
                "predicted_at": pred.predicted_at,
                "win_prob_home": pred.win_prob_home,
                "predicted_spread": pred.predicted_spread,
                "predicted_total": pred.predicted_total,
                "lookback_games": pred.lookback_games,
                "home_team_name": home_name,
                "away_team_name": away_name,
                "game_date": game_date,
                "has_result": has_result,
            })

        return items, total

    @staticmethod
    async def get_prediction_detail(
        session: AsyncSession, prediction_id: int
    ) -> dict | None:
        """Get full prediction detail with result if available."""
        query = (
            select(Prediction)
            .where(Prediction.id == prediction_id)
        )
        result = await session.execute(query)
        pred = result.scalar_one_or_none()

        if pred is None:
            return None

        # Get related game and teams (game may be null for offseason predictions)
        game = None
        if pred.game_id:
            game_query = select(Game).where(Game.id == pred.game_id)
            game_result = await session.execute(game_query)
            game = game_result.scalar_one_or_none()

        home_name, away_name = None, None
        if game:
            home_t = await session.execute(select(Team.full_name).where(Team.id == game.home_team_id))
            away_t = await session.execute(select(Team.full_name).where(Team.id == game.away_team_id))
            home_name = home_t.scalar_one_or_none()
            away_name = away_t.scalar_one_or_none()
        else:
            # Fallback: resolve from stored team IDs
            if pred.home_team_id:
                home_t = await session.execute(select(Team.full_name).where(Team.id == pred.home_team_id))
                home_name = home_t.scalar_one_or_none()
            if pred.away_team_id:
                away_t = await session.execute(select(Team.full_name).where(Team.id == pred.away_team_id))
                away_name = away_t.scalar_one_or_none()

        # Get result if exists
        result_query = select(PredictionResult).where(
            PredictionResult.prediction_id == prediction_id
        )
        result_result = await session.execute(result_query)
        eval_result = result_result.scalar_one_or_none()

        return {
            "id": pred.id,
            "game_id": pred.game_id,
            "model_name": pred.model_name,
            "model_version": pred.model_version,
            "predicted_at": pred.predicted_at,
            "win_prob_home": pred.win_prob_home,
            "win_prob_away": pred.win_prob_away,
            "predicted_spread": pred.predicted_spread,
            "spread_confidence": pred.spread_confidence,
            "predicted_total": pred.predicted_total,
            "total_confidence": pred.total_confidence,
            "calibrated": pred.calibrated,
            "lookback_games": pred.lookback_games,
            "home_team_id": game.home_team_id if game else None,
            "away_team_id": game.away_team_id if game else None,
            "home_team_name": home_name,
            "away_team_name": away_name,
            "game_date": game.game_date if game else None,
            "has_result": eval_result is not None,
            "result": {
                "prediction_id": eval_result.prediction_id,
                "actual_spread": eval_result.actual_spread,
                "actual_total": eval_result.actual_total,
                "actual_winner": eval_result.actual_winner,
                "spread_correct": eval_result.spread_correct,
                "total_correct": eval_result.total_correct,
                "win_correct": eval_result.win_correct,
                "evaluated_at": eval_result.evaluated_at,
            } if eval_result else None,
        }
