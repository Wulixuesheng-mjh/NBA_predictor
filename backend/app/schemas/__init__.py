"""Pydantic schemas for request/response validation."""

from app.schemas.common import ApiResponse, PaginationMeta, ErrorResponse, ErrorDetail
from app.schemas.team import (
    TeamResponse, TeamListResponse, TeamDetailResponse,
    PlayerBrief, TeamRosterResponse,
)
from app.schemas.player import (
    PlayerResponse, PlayerListResponse, PlayerSeasonStats,
    PlayerInjuryBrief, PlayerDetailResponse,
)
from app.schemas.game import (
    GameResponse, GameListResponse, GameDetailResponse,
    TeamGameStatsBrief, PlayerGameStatsBrief, UpcomingGameResponse,
)
from app.schemas.injury import (
    InjuryCreate, InjuryUpdate, InjuryResponse,
    InjuryListResponse, ConfirmAllResponse,
)
from app.schemas.prediction import (
    PredictionRequest, PredictionResponse,
    ModelPredictionOutput, ConsensusOutput, DataQualityOutput,
    PredictionHistoryItem, PredictionHistoryListResponse,
    PredictionResultOutput, PredictionDetailResponse,
)
from app.schemas.admin import (
    CollectionTriggerResponse, CollectionLogItem,
    CollectionStatusResponse, SystemStatsResponse, ModelTrainResponse,
)
