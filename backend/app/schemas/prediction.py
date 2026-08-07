"""Prediction schemas — the core prediction request/response."""

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for generating a prediction."""

    home_team_id: int = Field(..., ge=1, description="Home team internal ID")
    away_team_id: int = Field(..., ge=1, description="Away team internal ID")
    lookback_games: int = Field(
        default=10,
        ge=5,
        le=15,
        description="Number of recent games to look back for team features",
    )


class ModelPredictionOutput(BaseModel):
    """Single model's prediction output."""

    model_version: str
    win_prob_home: float
    win_prob_away: float
    predicted_spread: float | None = None
    spread_confidence: float | None = None
    predicted_total: float | None = None
    total_confidence: float | None = None


class ConsensusOutput(BaseModel):
    """Agreement analysis between models."""

    win_agree: bool
    spread_direction_agree: bool
    total_agree: bool
    agreement_score: float


class DataQualityOutput(BaseModel):
    """Data quality assessment for this prediction."""

    home_games_available: int
    away_games_available: int
    warnings: list[str] = []


class ExplanationFeature(BaseModel):
    """Single feature's SHAP contribution."""

    feature: str
    display_name: str
    shap_value: float
    contribution: str  # "positive" | "negative"
    direction: str  # "favors_home" | "favors_away"


class ExplanationOutput(BaseModel):
    """SHAP explanation for a prediction."""

    top_features: list[ExplanationFeature]
    explained_model: str = "xgboost_win"


class PredictionResponse(BaseModel):
    """Complete prediction response."""

    home_team_id: int
    away_team_id: int
    home_team_name: str | None = None
    away_team_name: str | None = None
    home_team_abbr: str | None = None
    away_team_abbr: str | None = None
    lookback_games: int
    actual_games_home: int
    actual_games_away: int
    predicted_at: str

    xgboost: ModelPredictionOutput | None = None
    mlp: ModelPredictionOutput | None = None
    consensus: ConsensusOutput | None = None
    recommended: str  # "xgboost" | "mlp" | "unavailable"

    data_quality: DataQualityOutput
    calibration_applied: bool = False
    explanation: ExplanationOutput | None = None


class PredictionHistoryItem(BaseModel):
    """Single item in prediction history list."""

    id: int
    game_id: int | None = None
    home_team_id: int | None = None
    away_team_id: int | None = None
    model_name: str
    model_version: str
    predicted_at: datetime | None = None
    win_prob_home: float
    predicted_spread: float | None = None
    predicted_total: float | None = None
    lookback_games: int = 10
    home_team_name: str | None = None
    away_team_name: str | None = None
    game_date: datetime | None = None
    has_result: bool = False

    model_config = {"from_attributes": True}


class PredictionHistoryListResponse(BaseModel):
    """List of historical predictions."""

    predictions: list[PredictionHistoryItem]
    total: int
    offset: int = 0
    limit: int = 20


class PredictionResultOutput(BaseModel):
    """Actual result compared to prediction."""

    prediction_id: int
    actual_spread: float | None = None
    actual_total: int | None = None
    actual_winner: str | None = None
    spread_correct: bool | None = None
    total_correct: bool | None = None
    win_correct: bool | None = None
    evaluated_at: datetime | None = None

    model_config = {"from_attributes": True}


class PredictionDetailResponse(PredictionHistoryItem):
    """Full prediction detail with result if available."""

    win_prob_away: float
    spread_confidence: float | None = None
    total_confidence: float | None = None
    calibrated: bool = False
    home_team_id: int | None = None
    away_team_id: int | None = None
    result: PredictionResultOutput | None = None
