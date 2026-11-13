"""Prediction schemas — the core prediction request/response."""

from datetime import datetime

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PredictionRequest(BaseModel):
    """Request body for generating a prediction."""

    game_id: int | None = Field(default=None, ge=1)
    home_team_id: int | None = Field(default=None, ge=1, description="Home team internal ID")
    away_team_id: int | None = Field(default=None, ge=1, description="Away team internal ID")
    as_of: datetime | None = None
    mode: Literal["pregame", "hypothetical_preview", "reconstructed_replay"] = "hypothetical_preview"
    persist: bool = False
    lookback_games: int = Field(
        default=10,
        ge=5,
        le=15,
        description="Number of recent games to look back for team features",
    )
    spread_line: float | None = Field(
        default=None,
        description=(
            "体彩让分盘口 (positive = home favored, e.g. +3.5). "
            "Required together with total_line to enable 让分胜负/大小分 prediction."
        ),
    )
    market_home_handicap: float | None = Field(
        default=None,
        description="体彩原始符号；-6.5 表示主队让 6.5 分",
    )
    total_line: float | None = Field(
        default=None,
        description=(
            "体彩大小分盘口 (expected total, e.g. 220.5). "
            "Required together with spread_line to enable 让分胜负/大小分 prediction."
        ),
    )
    home_odds: float | None = Field(
        default=None,
        ge=1.01,
        le=100.0,
        description=(
            "体彩主胜赔率 (decimal odds, e.g. 1.85). "
            "Requires away_odds; enables the market comparison block."
        ),
    )
    away_odds: float | None = Field(
        default=None,
        ge=1.01,
        le=100.0,
        description=(
            "体彩客胜赔率 (decimal odds, e.g. 2.05). "
            "Requires home_odds; enables the market comparison block."
        ),
    )

    @model_validator(mode="after")
    def validate_request(self):
        if self.game_id is None and (self.home_team_id is None or self.away_team_id is None):
            raise ValueError("Provide game_id or both home_team_id and away_team_id")
        if self.spread_line is not None and self.market_home_handicap is not None:
            raise ValueError("spread_line and market_home_handicap cannot both be provided")
        if (self.home_odds is None) != (self.away_odds is None):
            raise ValueError("home_odds and away_odds must be provided together")
        return self


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


class ConfidenceOutput(BaseModel):
    """Overall prediction confidence assessment."""

    level: str  # "high" | "medium" | "low" | "not_recommended"
    score: float  # 0-100
    factors: dict[str, float] = {}  # factor_name → contribution
    label_cn: str  # Chinese display label


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


class CoverOverOutput(BaseModel):
    """盘口二分类预测 (让分胜负 + 大小分), driven by manually entered lines."""

    model_version: str
    spread_line: float | None = None
    total_line: float | None = None
    cover_prob_home: float  # P(home covers the spread)
    cover_prob_away: float  # P(away covers the spread)
    cover_direction: str  # "home" | "away"
    cover_direction_cn: str  # "主队赢盘" | "客队赢盘"
    over_prob: float  # P(game goes over the total)
    under_prob: float  # P(game goes under the total)
    cover_push_prob: float = 0.0
    total_push_prob: float = 0.0
    calibrated: bool = False
    residual_samples: int = 0
    over_direction: str  # "over" | "under"
    over_direction_cn: str  # "大分" | "小分"
    explanation: dict | None = None  # {"cover": [...], "over": [...]} SHAP contributions


class BaselineOutput(BaseModel):
    """节奏 × 效率基线：独立于模型的期望总分，用于交叉校验。"""

    pace: float
    expected_home: float
    expected_away: float
    baseline_total: float
    model_total: float | None = None
    diff_vs_model: float | None = None
    status: str  # "ok" | "divergent"


class MarketOddsOutput(BaseModel):
    """体彩胜负赔率 vs 模型概率 — market implied probability comparison."""

    home_odds: float
    away_odds: float
    overround: float  # 1/home_odds + 1/away_odds; > 1 = bookmaker margin
    implied_home: float  # margin-free implied probability of home win
    implied_away: float
    fair_home_odds: float  # 1 / implied_home
    fair_away_odds: float  # 1 / implied_away
    model_home: float  # model probability used for the comparison
    model_source: str  # "xgboost" | "mlp"
    edge_home: float  # model_home - implied_home (positive = model likes home more)
    edge_away: float
    value_side: str  # "home" | "away" | "none"
    value_edge: float  # |edge| of the favored side, in probability points


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
    game_id: int | None = None
    as_of: str | None = None
    prediction_mode: str = "hypothetical_preview"

    xgboost: ModelPredictionOutput | None = None
    mlp: ModelPredictionOutput | None = None
    consensus: ConsensusOutput | None = None
    recommended: str  # "xgboost" | "mlp" | "unavailable"

    data_quality: DataQualityOutput
    calibration_applied: bool = False
    confidence: ConfidenceOutput | None = None
    explanation: ExplanationOutput | None = None
    cover_over: CoverOverOutput | None = None
    market: MarketOddsOutput | None = None
    baseline: BaselineOutput | None = None
    # 发布包信息：模型版本、训练数据截止日、特征维度
    model_release: dict | None = None
    prediction_intervals: dict | None = None
    shadow_totals: dict | None = None


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
