"""Prediction API endpoints — the core prediction interface.

POST /predictions  — Generate a new prediction for a matchup
GET  /predictions  — List historical predictions
GET  /predictions/export — Export predictions as CSV
GET  /predictions/{id} — Single prediction detail with result if available
"""

import csv
import io

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.prediction import Prediction
from app.services.prediction_service import PredictionService
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionHistoryListResponse,
    PredictionHistoryItem,
    PredictionDetailResponse,
    PredictionResultOutput,
)
from app.schemas.common import ErrorResponse

router = APIRouter()


@router.post(
    "/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def create_prediction(
    body: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Generate a new prediction for a matchup.

    This is the **core endpoint** of the application.

    Takes two team IDs and a lookback window (5-15 games), then:
    1. Builds ~71 features from recent games, injuries, H2H, and context
    2. Runs both XGBoost (primary) and MLP (auxiliary) models
    3. Returns predictions for win/loss, point spread, and total points
    4. Includes consensus analysis and data quality warnings

    Example request:
    ```json
    {
      "home_team_id": 1,
      "away_team_id": 2,
      "lookback_games": 10
    }
    ```
    """
    try:
        result = await PredictionService.generate_prediction(
            db,
            home_team_id=body.home_team_id,
            away_team_id=body.away_team_id,
            lookback_games=body.lookback_games,
        )
        return PredictionResponse(**result)
    except ValueError as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}. Ensure models are trained (run scripts/train_models.py).",
        )


@router.get(
    "/",
    response_model=PredictionHistoryListResponse,
)
async def list_predictions(
    game_id: int | None = Query(None, description="Filter by game ID"),
    model_name: str | None = Query(None, description="Filter by model (xgboost/mlp)"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List historical predictions with team names."""
    predictions, total = await PredictionService.list_predictions(
        db,
        game_id=game_id,
        model_name=model_name,
        offset=offset,
        limit=limit,
    )
    return PredictionHistoryListResponse(
        predictions=[PredictionHistoryItem(**p) for p in predictions],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/export")
async def export_predictions_csv(
    model_name: str | None = Query(None, description="Filter by model (xgboost/mlp)"),
    db: AsyncSession = Depends(get_db),
):
    """Export all predictions as CSV file."""
    predictions, _ = await PredictionService.list_predictions(
        db,
        model_name=model_name,
        offset=0,
        limit=10000,  # Export all
    )

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "ID", "日期", "主队", "客队", "模型", "模型版本",
        "主队胜率", "预测让分", "预测总分",
        "已评估", "预测时间",
    ])

    # Rows
    for p in predictions:
        writer.writerow([
            p["id"],
            p.get("game_date") or "",
            p.get("home_team_name") or "",
            p.get("away_team_name") or "",
            p.get("model_name") or "",
            p.get("model_version") or "",
            f"{p['win_prob_home']:.4f}",
            f"{p['predicted_spread']:.1f}" if p.get("predicted_spread") is not None else "",
            f"{p['predicted_total']:.1f}" if p.get("predicted_total") is not None else "",
            "是" if p.get("has_result") else "否",
            p.get("predicted_at") or "",
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8-sig",
        headers={
            "Content-Disposition": "attachment; filename=nba_predictions.csv"
        },
    )


@router.get("/accuracy-trends")
async def accuracy_trends(
    days: int = Query(default=30, ge=7, le=365, description="Days to look back"),
    window: int = Query(default=7, ge=3, le=30, description="Sliding window size in days"),
    db: AsyncSession = Depends(get_db),
):
    """Get sliding-window accuracy trends (public, no auth required)."""
    from datetime import datetime, timedelta, timezone
    from collections import defaultdict
    from app.models.prediction_results import PredictionResult
    from app.schemas.admin import (
        AccuracyTrendResponse,
        AccuracyTrendPoint,
        AccuracyTrendSummary,
    )

    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days + window)

    query = (
        select(
            PredictionResult.evaluated_at,
            PredictionResult.win_correct,
            PredictionResult.spread_correct,
            PredictionResult.total_correct,
            PredictionResult.actual_spread,
            PredictionResult.actual_total,
            Prediction.predicted_spread,
            Prediction.predicted_total,
        )
        .join(Prediction, PredictionResult.prediction_id == Prediction.id)
        .where(PredictionResult.evaluated_at >= cutoff_date)
        .order_by(PredictionResult.evaluated_at.asc())
    )
    result = await db.execute(query)
    rows = result.all()

    if not rows:
        return AccuracyTrendResponse(
            trends=[],
            overall=AccuracyTrendSummary(
                win_accuracy=0.0,
                spread_mae=None,
                total_mae=None,
                total_evaluated=0,
            ),
        )

    daily: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        date_str = row.evaluated_at.strftime("%Y-%m-%d") if row.evaluated_at else None
        if date_str is None:
            continue
        daily[date_str].append({
            "win_correct": row.win_correct,
            "spread_correct": row.spread_correct,
            "total_correct": row.total_correct,
            "predicted_spread": row.predicted_spread,
            "predicted_total": row.predicted_total,
            "actual_spread": row.actual_spread,
            "actual_total": row.actual_total,
        })

    trend_start = datetime.now(timezone.utc).date() - timedelta(days=days)
    all_dates = sorted(daily.keys())

    trends: list[AccuracyTrendPoint] = []
    current = trend_start

    while current <= datetime.now(timezone.utc).date():
        date_str = current.strftime("%Y-%m-%d")
        window_start = current - timedelta(days=window)
        window_samples: list[dict] = []
        for d in all_dates:
            d_date = datetime.strptime(d, "%Y-%m-%d").date()
            if window_start <= d_date <= current:
                window_samples.extend(daily[d])

        if window_samples:
            n = len(window_samples)
            win_acc = sum(1 for s in window_samples if s["win_correct"]) / n

            spread_errors = []
            for s in window_samples:
                if s["predicted_spread"] is not None and s["actual_spread"] is not None:
                    spread_errors.append(abs(s["predicted_spread"] - s["actual_spread"]))
            spread_mae = sum(spread_errors) / len(spread_errors) if spread_errors else None

            total_errors = []
            for s in window_samples:
                if s["predicted_total"] is not None and s["actual_total"] is not None:
                    total_errors.append(abs(s["predicted_total"] - s["actual_total"]))
            total_mae = sum(total_errors) / len(total_errors) if total_errors else None

            trends.append(AccuracyTrendPoint(
                date=date_str,
                win_accuracy=round(win_acc, 4),
                spread_mae=round(spread_mae, 2) if spread_mae is not None else None,
                total_mae=round(total_mae, 2) if total_mae is not None else None,
                sample_count=n,
                window_days=window,
            ))
        else:
            trends.append(AccuracyTrendPoint(
                date=date_str,
                win_accuracy=None,
                spread_mae=None,
                total_mae=None,
                sample_count=0,
                window_days=window,
            ))

        current += timedelta(days=1)

    all_samples = [s for samples in daily.values() for s in samples]
    total_n = len(all_samples)
    overall_win_acc = sum(1 for s in all_samples if s["win_correct"]) / total_n if total_n > 0 else 0.0

    overall_spread_errors = []
    for s in all_samples:
        if s["predicted_spread"] is not None and s["actual_spread"] is not None:
            overall_spread_errors.append(abs(s["predicted_spread"] - s["actual_spread"]))
    overall_spread_mae = sum(overall_spread_errors) / len(overall_spread_errors) if overall_spread_errors else None

    overall_total_errors = []
    for s in all_samples:
        if s["predicted_total"] is not None and s["actual_total"] is not None:
            overall_total_errors.append(abs(s["predicted_total"] - s["actual_total"]))
    overall_total_mae = sum(overall_total_errors) / len(overall_total_errors) if overall_total_errors else None

    return AccuracyTrendResponse(
        trends=trends,
        overall=AccuracyTrendSummary(
            win_accuracy=round(overall_win_acc, 4),
            spread_mae=round(overall_spread_mae, 2) if overall_spread_mae is not None else None,
            total_mae=round(overall_total_mae, 2) if overall_total_mae is not None else None,
            total_evaluated=total_n,
        ),
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionDetailResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_prediction(prediction_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single prediction detail, including actual result if game completed."""
    detail = await PredictionService.get_prediction_detail(db, prediction_id)
    if detail is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found")

    return PredictionDetailResponse(
        id=detail["id"],
        game_id=detail["game_id"],
        model_name=detail["model_name"],
        model_version=detail["model_version"],
        predicted_at=detail["predicted_at"],
        win_prob_home=detail["win_prob_home"],
        win_prob_away=detail["win_prob_away"],
        predicted_spread=detail["predicted_spread"],
        spread_confidence=detail["spread_confidence"],
        predicted_total=detail["predicted_total"],
        total_confidence=detail["total_confidence"],
        calibrated=detail["calibrated"],
        lookback_games=detail["lookback_games"],
        home_team_id=detail["home_team_id"],
        away_team_id=detail["away_team_id"],
        home_team_name=detail["home_team_name"],
        away_team_name=detail["away_team_name"],
        game_date=detail["game_date"],
        has_result=detail["has_result"],
        result=PredictionResultOutput(**detail["result"]) if detail["result"] else None,
    )

