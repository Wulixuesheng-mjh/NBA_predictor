"""Prediction service — integrates ML module for predictions."""

import logging
import sys
import time
import hashlib
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

# 模型总分与「节奏×效率」基线的允许偏差（分）；超出则标记为 divergent
BASELINE_TOLERANCE = 8.0


class PredictionService:
    """Service for generating and retrieving predictions."""

    @staticmethod
    def _get_predictor():
        """Lazy-load the ML Predictor.

        Returns a Predictor instance or None if ML module is unavailable.
        """
        try:
            ml_path = str(Path(__file__).parent.parent.parent.parent / "ml")
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
    def _build_market(
        home_odds: float | None,
        away_odds: float | None,
        model_home: float | None,
        model_source: str,
        edge_threshold: float = 0.03,
    ) -> dict[str, Any] | None:
        """Turn 体彩 decimal odds into implied probabilities and compare.

        Bookmaker odds carry a margin (overround), so the raw 1/odds values sum
        to more than 1. We normalise them to get the market's "fair"
        probability, then compare with the model's win probability: a positive
        edge means the model rates that side higher than the market does.

        Args:
            home_odds: Decimal odds for a home win (e.g. 1.85).
            away_odds: Decimal odds for an away win (e.g. 2.05).
            model_home: Model probability of a home win (0-1).
            model_source: Which model produced ``model_home``.
            edge_threshold: Minimum edge (probability points) to flag value.

        Returns:
            Dict for the API response, or None when inputs are unusable.
        """
        if home_odds is None or away_odds is None or model_home is None:
            return None
        if home_odds <= 1.0 or away_odds <= 1.0:
            return None

        inv_home = 1.0 / home_odds
        inv_away = 1.0 / away_odds
        overround = inv_home + inv_away
        implied_home = inv_home / overround
        implied_away = 1.0 - implied_home

        edge_home = model_home - implied_home
        edge_away = (1.0 - model_home) - implied_away

        if max(edge_home, edge_away) < edge_threshold:
            value_side = "none"
        else:
            value_side = "home" if edge_home >= edge_away else "away"

        return {
            "home_odds": round(home_odds, 3),
            "away_odds": round(away_odds, 3),
            "overround": round(overround, 4),
            "implied_home": round(implied_home, 4),
            "implied_away": round(implied_away, 4),
            "fair_home_odds": round(1.0 / implied_home, 3),
            "fair_away_odds": round(1.0 / implied_away, 3),
            "model_home": round(model_home, 4),
            "model_source": model_source,
            "edge_home": round(edge_home, 4),
            "edge_away": round(edge_away, 4),
            "value_side": value_side,
            "value_edge": round(max(edge_home, edge_away), 4),
        }

    @staticmethod
    async def generate_prediction(
        session: AsyncSession,
        home_team_id: int | None = None,
        away_team_id: int | None = None,
        game_id: int | None = None,
        as_of: datetime | None = None,
        prediction_mode: str = "hypothetical_preview",
        persist: bool = False,
        lookback_games: int = 10,
        spread_line: float | None = None,
        market_home_handicap: float | None = None,
        total_line: float | None = None,
        home_odds: float | None = None,
        away_odds: float | None = None,
    ) -> dict[str, Any]:
        """Generate a prediction for a matchup.

        Args:
            session: Database session.
            home_team_id: Home team internal ID.
            away_team_id: Away team internal ID.
            lookback_games: Number of recent games to use for features (5-15).
            spread_line: Optional 体彩让分盘口 (positive = home favored). When both
                spread_line and total_line are provided, the 盘口 binary
                (让分胜负 / 大小分) prediction is also returned.
            total_line: Optional 体彩大小分盘口 (expected total).
            home_odds: Optional 体彩主胜赔率 (decimal odds).
            away_odds: Optional 体彩客胜赔率 (decimal odds). When both odds are
                provided, a market-implied probability comparison is returned.

        Returns:
            Dict with prediction results or error info.
        """
        now = datetime.now(timezone.utc)
        as_of = as_of or now
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=timezone.utc)
        if prediction_mode not in {"pregame", "hypothetical_preview", "reconstructed_replay"}:
            raise ValueError("Unknown prediction mode")
        if spread_line is not None and market_home_handicap is not None:
            raise ValueError("spread_line and market_home_handicap cannot both be provided")
        # Legacy spread_line used a positive home-favorite number.  The new
        # market field preserves sportsbook notation (-6.5 = home lays 6.5).
        if market_home_handicap is None and spread_line is not None:
            market_home_handicap = -float(spread_line)

        target_game = await session.get(Game, game_id) if game_id is not None else None
        if game_id is not None and target_game is None:
            raise ValueError(f"Game with id {game_id} not found")
        if target_game is not None:
            if home_team_id is not None and home_team_id != target_game.home_team_id:
                raise ValueError("home_team_id does not match game_id")
            if away_team_id is not None and away_team_id != target_game.away_team_id:
                raise ValueError("away_team_id does not match game_id")
            home_team_id = target_game.home_team_id
            away_team_id = target_game.away_team_id
        if home_team_id is None or away_team_id is None:
            raise ValueError("Provide game_id or both team IDs")
        if prediction_mode == "pregame":
            if target_game is None:
                raise ValueError("Formal pregame predictions require game_id")
            if target_game.scheduled_start_utc is None or target_game.time_precision != "exact":
                raise ValueError("Formal pregame prediction requires an exact scheduled_start_utc")
            start = target_game.scheduled_start_utc
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if as_of >= start:
                raise ValueError("Formal pregame as_of must be earlier than tipoff")
            if now >= start or target_game.status == "Final":
                raise ValueError("A completed/started game requires reconstructed_replay; pregame backdating is forbidden")
            if as_of > now or (persist and (now - as_of).total_seconds() > 60):
                raise ValueError("Persisted pregame as_of must be the current request time")
        if persist and prediction_mode != "pregame":
            raise ValueError("Only formal pregame predictions may be persisted")

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
        if predictor is not None and prediction_mode == 'pregame':
            fitted_until = predictor.release_manifest.get('training_data_until')
            target_day = str(target_game.source_game_date or target_game.scheduled_start_utc.date())
            if fitted_until is None or str(fitted_until)[:10] >= target_day:
                raise ValueError('Formal pregame target must be beyond the model training cutoff')
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
                "cover_over": None,
                "market": None,
                "baseline": None,
            }

        # Build features and predict
        try:
            from ml.src.features.feature_builder import FeatureBuilder

            builder = FeatureBuilder()
            game_date = (
                target_game.scheduled_start_utc
                if target_game is not None and target_game.scheduled_start_utc is not None
                else target_game.game_date if target_game is not None else as_of
            )
            t0 = time.monotonic()

            fv = await builder.build_features(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                game_date=game_date,
                lookback_games=lookback_games,
                session=session,
                as_of=as_of,
                target_game_id=target_game.id if target_game is not None else None,
                use_historical=prediction_mode == "reconstructed_replay",
            )

            prediction = predictor.predict(fv, apply_calibration=True)
            result = prediction.to_dict()
            # Optional diagnostic comparison; a failed candidate cannot replace
            # the primary prediction or break the production response.
            try:
                from app.services.shadow_total_service import shadow_totals
                result['shadow_totals']=shadow_totals(fv.features,as_of=as_of,lookback=lookback_games,
                    target_day=str(target_game.source_game_date) if target_game and target_game.source_game_date else None)
            except Exception:
                logger.warning('Shadow totals unavailable',exc_info=True)
                result['shadow_totals']={'status':'unavailable','reason':'影子模型校验未通过','candidates':[]}

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

            # Market probabilities come from the regression point estimate and
            # rolling OOS residuals, so they respond monotonically to the
            # supplied line.  The old binary classifiers ignored the line.
            result["cover_over"] = None
            result["prediction_intervals"] = None
            point_source = "xgboost" if result.get("xgboost") else "mlp"
            point = result.get(point_source) or {}
            spread_residuals: list[float] = []
            total_residuals: list[float] = []
            try:
                from ml.src.config import ml_settings
                from ml.src.uncertainty import (
                    ResidualStore,
                    prediction_interval,
                    threshold_probabilities,
                )

                residual_store = ResidualStore(Path(ml_settings.models_dir) / "residuals.json")
                spread_residuals = residual_store.get(point_source, "spread", version=point.get('model_version'), as_of=as_of)
                total_residuals = residual_store.get(point_source, "total", version=point.get('model_version'), as_of=as_of)
                predicted_spread = point.get("predicted_spread")
                predicted_total = point.get("predicted_total")
                if predicted_spread is not None and predicted_total is not None:
                    result["prediction_intervals"] = {
                        "model": point_source,
                        "model_version": point.get('model_version'),
                        "spread_80": prediction_interval(predicted_spread, spread_residuals, 0.80),
                        "spread_90": prediction_interval(predicted_spread, spread_residuals, 0.90),
                        "total_80": prediction_interval(predicted_total, total_residuals, 0.80),
                        "total_90": prediction_interval(predicted_total, total_residuals, 0.90),
                        "spread_samples": len(spread_residuals),
                        "total_samples": len(total_residuals),
                        "calibrated": len(spread_residuals) >= 200 and len(total_residuals) >= 200,
                    }
            except Exception as e:
                logger.debug("Prediction intervals unavailable: %s", e)

            if market_home_handicap is not None and total_line is not None:
                try:
                    spread_threshold = -float(market_home_handicap)
                    cover = threshold_probabilities(
                        float(point["predicted_spread"]),
                        spread_threshold,
                        spread_residuals,
                        integer_outcome=True,
                        fallback_sigma=12.0,
                    )
                    over = threshold_probabilities(
                        float(point["predicted_total"]),
                        float(total_line),
                        total_residuals,
                        integer_outcome=True,
                        fallback_sigma=15.0,
                    )
                    result["cover_over"] = {
                        "model_version": point.get("model_version", "unknown"),
                        "spread_line": market_home_handicap,
                        "total_line": total_line,
                        "cover_prob_home": round(cover.win, 4),
                        "cover_prob_away": round(cover.lose, 4),
                        "cover_push_prob": round(cover.push, 4),
                        "cover_direction": "home" if cover.win >= cover.lose else "away",
                        "cover_direction_cn": "主队赢盘" if cover.win >= cover.lose else "客队赢盘",
                        "over_prob": round(over.win, 4),
                        "under_prob": round(over.lose, 4),
                        "total_push_prob": round(over.push, 4),
                        "over_direction": "over" if over.win >= over.lose else "under",
                        "over_direction_cn": "大分" if over.win >= over.lose else "小分",
                        "calibrated": cover.calibrated and over.calibrated,
                        "residual_samples": min(cover.sample_size, over.sample_size),
                        "explanation": None,
                    }
                except Exception as e:
                    logger.warning("Cover/over prediction unavailable: %s", e)
                    result["cover_over"] = None

            # 胜负赔率 vs 模型概率 — market reference for the same matchup
            result["market"] = None
            model_home: float | None = None
            model_source = "xgboost"
            if result.get("recommended") == "mlp" and result.get("mlp"):
                model_home = result["mlp"].get("win_prob_home")
                model_source = "mlp"
            elif result.get("xgboost"):
                model_home = result["xgboost"].get("win_prob_home")
            elif result.get("mlp"):
                model_home = result["mlp"].get("win_prob_home")
                model_source = "mlp"
            result["market"] = PredictionService._build_market(
                home_odds, away_odds, model_home, model_source
            )

            # 节奏 × 效率基线：独立于模型的期望总分，用于交叉校验
            result["baseline"] = None
            try:
                rf = getattr(fv, "raw_features", {}) or {}
                hp = float(rf.get("home_bayes_pace") or 0)
                ap = float(rf.get("away_bayes_pace") or 0)
                ho = float(rf.get("home_bayes_off_rtg") or 0)
                hd = float(rf.get("home_bayes_def_rtg") or 0)
                ao = float(rf.get("away_bayes_off_rtg") or 0)
                ad = float(rf.get("away_bayes_def_rtg") or 0)
                if min(hp, ap, ho, hd, ao, ad) > 0:
                    pace = (hp + ap) / 2.0
                    exp_home = pace * ((ho + ad) / 2.0) / 100.0
                    exp_away = pace * ((ao + hd) / 2.0) / 100.0
                    baseline_total = exp_home + exp_away
                    model_total = (result.get("xgboost") or {}).get("predicted_total")
                    diff = (
                        round(model_total - baseline_total, 1)
                        if model_total is not None
                        else None
                    )
                    result["baseline"] = {
                        "pace": round(pace, 1),
                        "expected_home": round(exp_home, 1),
                        "expected_away": round(exp_away, 1),
                        "baseline_total": round(baseline_total, 1),
                        "model_total": model_total,
                        "diff_vs_model": diff,
                        "status": (
                            "ok" if diff is None or abs(diff) <= BASELINE_TOLERANCE
                            else "divergent"
                        ),
                    }
            except Exception as e:
                logger.debug("Baseline computation skipped: %s", e)

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
        result["game_id"] = target_game.id if target_game is not None else None
        result["as_of"] = as_of.isoformat()
        result["prediction_mode"] = prediction_mode

        # Release provenance: which models answered, and how fresh the data is
        # Use the exact manifest captured while the model was loaded.  Reading
        # active.json again here could attach a newer release to an old model.
        release = dict(getattr(predictor, "release_manifest", {}) or {})
        result["model_release"] = {
            "xgboost_version": getattr(predictor.xgb_trainer, "version", None),
            "mlp_version": (
                getattr(predictor.mlp_trainer, "version", None)
                if predictor.mlp_trainer is not None
                else None
            ),
            "training_data_until": release.get("training_data_until"),
            "feature_dim": release.get("feature_dim"),
            "calibration_enabled": release.get("calibration_enabled"),
            "feature_semantic_version": release.get("feature_semantic_version"),
            "training_policy": release.get("training_policy"),
        }

        # Only a certified pregame request enters the immutable evaluation
        # ledger.  Preview and reconstructed replay remain pure calculations.
        if persist:
            try:
                from ml.src.features.feature_builder import FEATURE_NAMES

                ordered_vector = [float(value) for value in fv.features.tolist()]
                builder_snapshot = {"stage": "builder_output", "feature_names": FEATURE_NAMES,
                                    "feature_vector": ordered_vector}
                injury_snapshot = {
                    key: value for key, value in (fv.raw_features or {}).items()
                    if "injury" in key or "replacement_minutes" in key
                }
                records = []
                for model_name in ("xgboost", "mlp"):
                    output = result.get(model_name)
                    if not output:
                        continue
                    processed = prediction.processed_inputs.get(model_name)
                    if not processed:
                        raise ValueError(f'Missing processed input snapshot for {model_name}')
                    names, values = processed['feature_names'], processed['feature_vector']
                    vector_hash = hashlib.sha256(
                        ("|".join(names) + "|" + ",".join(f"{v:.12g}" for v in values)).encode()
                    ).hexdigest()
                    records.append(Prediction(
                    game_id=target_game.id,
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    model_name=model_name,
                    model_version=output.get("model_version", "unknown"),
                    predicted_at=now,
                    as_of=as_of,
                    prediction_mode=prediction_mode,
                    win_prob_home=output["win_prob_home"],
                    win_prob_away=output["win_prob_away"],
                    predicted_spread=output.get("predicted_spread"),
                    spread_confidence=output.get("spread_confidence"),
                    predicted_total=output.get("predicted_total"),
                    total_confidence=output.get("total_confidence"),
                    calibrated=result.get("calibration_applied", False) if model_name == "xgboost" else False,
                    calibration_factors={"calibration_applied": bool(result.get("calibration_applied", False))},
                    home_odds=home_odds,
                    away_odds=away_odds,
                    market_home_handicap=market_home_handicap,
                    market_total=total_line,
                    raw_output=(result.get("raw_outputs") or {}).get(model_name, output),
                    final_output=output,
                    baseline_output=result.get("baseline"),
                    injury_snapshot=injury_snapshot,
                    data_quality=result.get("data_quality"),
                    feature_names=names,
                    feature_vector=values,
                    feature_semantic_version=str(processed['feature_semantic_version']),
                    feature_hash=vector_hash,
                    features_json={**(fv.raw_features or {}), '_shadow_totals': result.get('shadow_totals'),
                        '_inference_snapshot': processed, '_builder_snapshot': builder_snapshot,
                        '_release_manifest': release},
                    lookback_games=lookback_games,
                ))
                session.add_all(records)
                await session.commit()
                logger.info("Saved %d formal pregame predictions", len(records))
            except Exception:
                await session.rollback()
                logger.exception("Formal pregame prediction could not be persisted")
                raise

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
