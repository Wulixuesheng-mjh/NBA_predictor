"""Feature builder — orchestrates all feature computers into a unified vector.

This is the main entry point for feature computation. It:
1. Delegates to TeamFeatures, PlayerFeatures, H2HFeatures, ContextFeatures
2. Computes matchup interaction features from home/away team stats
3. Assembles the ~126-dimension feature vector in a fixed order
4. Handles adaptive lookback with early-season fallback
5. Provides dataset construction for model training
"""

import logging
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.team_features import TeamFeatures, TEAM_FEATURE_NAMES
from ml.src.features.player_features import PlayerFeatures, PLAYER_FEATURE_NAMES
from ml.src.features.player_form_features import PlayerFormFeatures, PLAYER_FORM_FEATURE_NAMES
from ml.src.features.h2h_features import H2HFeatures, H2H_FEATURE_NAMES
from ml.src.features.context_features import ContextFeatures, CONTEXT_FEATURE_NAMES
from ml.src.features.matchup_features import compute_matchup_features, MATCHUP_FEATURE_NAMES
from ml.src.features.position_features import PositionMatchFeatures, POSITION_MATCH_FEATURE_NAMES, POSITION_MATCH_DIFF_NAMES
from ml.src.features.bench_features import BenchFeatures, BENCH_FEATURE_NAMES
from ml.src.features.lineup_features import LineupFeatures, LINEUP_FEATURE_NAMES
from ml.src.features.opponent_adjusted_features import OpponentAdjustedFeatures, OPPONENT_ADJUSTED_FEATURE_NAMES
from ml.src.features.roster_continuity_features import RosterContinuityFeatures, ROSTER_CONTINUITY_FEATURE_NAMES
from ml.src.features.season_prior_features import SeasonPriorFeatures, SEASON_PRIOR_FEATURE_NAMES
from ml.src.features.injury_net_features import InjuryNetFeatures, INJURY_NET_FEATURE_NAMES

logger = logging.getLogger(__name__)

# ── Complete feature name list in fixed order ──

TEAM_FEATURE_EXTENSION_NAMES = ["avg_fg3a_rate", "season_games_played", "advanced_stats_coverage"]
ROSTER_FEATURE_EXTENSION_NAMES = ["roster_data_missing"]
TOTAL_BASELINE_FEATURE_NAMES = ["shrunk_expected_total"]
BASE_TEAM_FEATURE_NAMES = [name for name in TEAM_FEATURE_NAMES if name not in TEAM_FEATURE_EXTENSION_NAMES]
BASE_ROSTER_FEATURE_NAMES = [name for name in ROSTER_CONTINUITY_FEATURE_NAMES if name not in ROSTER_FEATURE_EXTENSION_NAMES]

FEATURE_NAMES = []
# Team features: home_* + away_*
for name in BASE_TEAM_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in BASE_TEAM_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
# Player/injury features: home_* + away_*
for name in PLAYER_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in PLAYER_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
# Player form features: home_* + away_*
for name in PLAYER_FORM_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in PLAYER_FORM_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
# H2H features
FEATURE_NAMES.extend(H2H_FEATURE_NAMES)
# Context features
FEATURE_NAMES.extend(CONTEXT_FEATURE_NAMES)
# Matchup interaction features (pace diff, off-vs-def, etc.)
FEATURE_NAMES.extend(MATCHUP_FEATURE_NAMES)
# Position match: per-team features get home/away prefix
for name in POSITION_MATCH_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in POSITION_MATCH_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
FEATURE_NAMES.extend(POSITION_MATCH_DIFF_NAMES)
# Bench features: home/away prefix
for name in BENCH_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in BENCH_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
# Lineup features: home/away prefix
for name in LINEUP_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in LINEUP_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
# Opponent-adjusted features: home/away prefix
for name in OPPONENT_ADJUSTED_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in OPPONENT_ADJUSTED_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
# Roster continuity features: home/away prefix
for name in BASE_ROSTER_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in BASE_ROSTER_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
# The historical 166-feature schema remains an exact prefix during the rolling
# deployment; all v6 additions are appended after it.
for name in TEAM_FEATURE_EXTENSION_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in TEAM_FEATURE_EXTENSION_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
for name in ROSTER_FEATURE_EXTENSION_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in ROSTER_FEATURE_EXTENSION_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
for name in SEASON_PRIOR_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in SEASON_PRIOR_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")
FEATURE_NAMES.extend(TOTAL_BASELINE_FEATURE_NAMES)
for name in INJURY_NET_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in INJURY_NET_FEATURE_NAMES:
    FEATURE_NAMES.append(f"away_{name}")

FEATURE_DIM = len(FEATURE_NAMES)

# ── Cover/over (盘口) reduced feature set ──────────────────────────────────
# Historical lines exist only for 2012-2019, where advanced stats are NULL and
# player box scores are zeroed (fga=0, player pts/reb/ast=0).  Features derived
# from those (shooting %, pace/off_rtg/def_rtg, and every player-stat feature)
# would be "0 in train / nonzero in predict", so the cover/over classifiers use
# only features reliable in EVERY season — basic box score, schedule, H2H,
# opponent-adjusted (pts-based), and over/under tendency.

COVER_OVER_SAFE_TEAM_FEATURES = [
    "win_rate", "avg_pts", "avg_pts_allowed",
    "avg_reb", "avg_ast", "avg_stl", "avg_blk",
    "trend_pts", "trend_net",
    "last5_vs_first5_pts_diff", "last5_vs_first5_net_diff",
    "win_rate_last5", "pts_std", "net_pts_std",
    "total_trend", "over_league_avg_ratio", "avg_total_last3",
    "games_available",
]

# The only matchup feature that survives: it derives from over_league_avg_ratio
# (pts-based), not from advanced stats.
COVER_OVER_SAFE_MATCHUP_FEATURES = ["combined_over_tendency"]

COVER_OVER_FEATURE_NAMES: list[str] = []
for _n in COVER_OVER_SAFE_TEAM_FEATURES:
    COVER_OVER_FEATURE_NAMES.append(f"home_{_n}")
for _n in COVER_OVER_SAFE_TEAM_FEATURES:
    COVER_OVER_FEATURE_NAMES.append(f"away_{_n}")
COVER_OVER_FEATURE_NAMES.extend(H2H_FEATURE_NAMES)
COVER_OVER_FEATURE_NAMES.extend(CONTEXT_FEATURE_NAMES)
COVER_OVER_FEATURE_NAMES.extend(COVER_OVER_SAFE_MATCHUP_FEATURES)
for _n in OPPONENT_ADJUSTED_FEATURE_NAMES:
    COVER_OVER_FEATURE_NAMES.append(f"home_{_n}")
for _n in OPPONENT_ADJUSTED_FEATURE_NAMES:
    COVER_OVER_FEATURE_NAMES.append(f"away_{_n}")

COVER_OVER_FEATURE_DIM = len(COVER_OVER_FEATURE_NAMES)

# ── Static feature caps ──────────────────────────────────────────────────
# Hand-set rather than data-derived so training and inference always agree
# without shipping percentile tables next to every model.
#   * rest_days / days_since_* — the off-season gap (up to 295 days) is not
#     "rest"; this was the single biggest MLP-destabilising outlier.
#   * *_ratio — ratio features explode when the denominator is near zero
#     (starters/bench scoring ratio reached 392 while its p99 was 11).
#   * *_pct — shooting percentages live in 0-1.
REST_DAYS_CAP = 14.0
RATIO_CAP = 50.0
PCT_CAP = 1.0


def feature_bounds(name: str) -> tuple[float, float]:
    """(lower, upper) hard bounds for one feature name."""
    if "rest_days" in name:
        if name.endswith("_diff"):
            return (-REST_DAYS_CAP, REST_DAYS_CAP)
        return (0.0, REST_DAYS_CAP)
    if "days_since" in name:
        return (0.0, REST_DAYS_CAP)
    if name.endswith("_ratio"):
        return (-RATIO_CAP, RATIO_CAP)
    if name.endswith("_pct"):
        return (-PCT_CAP, PCT_CAP)
    return (-1e6, 1e6)


def _bounds_arrays(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([feature_bounds(n)[0] for n in names], dtype=np.float64)
    upper = np.array([feature_bounds(n)[1] for n in names], dtype=np.float64)
    return lower, upper


_FEATURE_LOWER, _FEATURE_UPPER = _bounds_arrays(FEATURE_NAMES)
_COVER_LOWER, _COVER_UPPER = _bounds_arrays(COVER_OVER_FEATURE_NAMES)

# ── Chinese display name mapping for SHAP explanations ──

_DISPLAY_COMPONENTS: dict[str, str] = {
    # Team
    "win_rate": "胜率", "avg_pts": "场均得分", "avg_pts_allowed": "场均失分",
    "avg_reb": "场均篮板", "avg_ast": "场均助攻", "avg_stl": "场均抢断",
    "avg_blk": "场均盖帽", "fg_pct": "投篮命中率", "fg3_pct": "三分命中率",
    "ft_pct": "罚球命中率", "avg_ts_pct": "真实命中率",
    "avg_off_rtg": "进攻效率", "avg_def_rtg": "防守效率",
    "avg_net_rtg": "净效率", "avg_pace": "比赛节奏",
    "avg_tov_pct": "失误率", "avg_ftr": "罚球率", "avg_ast_pct": "助攻率",
    "trend_pts": "得分趋势", "trend_net": "净分趋势",
    "trend_off_rtg": "进攻效率趋势", "trend_def_rtg": "防守效率趋势",
    "last5_vs_first5_pts_diff": "近5场vs前5场得分差",
    "last5_vs_first5_net_diff": "近5场vs前5场净分差",
    "win_rate_last5": "近5场胜率", "pts_std": "得分稳定性",
    "net_pts_std": "净分稳定性", "games_available": "可用场次",
    "total_trend": "总分趋势", "over_league_avg_ratio": "超联盟均分比例",
    "avg_total_last3": "近3场均总分",
    # Player injury
    "injury_firepower_loss": "伤病火力损失",
    "total_minutes_share_lost": "损失上场时间占比",
    "key_player_out": "核心球员缺阵", "defensive_anchor_out": "防守支柱缺阵",
    "players_out_count": "缺阵球员数", "questionable_count": "出战成疑球员数",
    # Player form
    "top_scorer_ppg": "头号得分手场均分",
    "top_scorer_ppg_trend": "头号得分手得分趋势",
    "scoring_concentration": "得分集中度", "avg_plus_minus": "场均正负值",
    "top_scorer_plus_minus": "头号得分手正负值",
    "bench_scoring_pct": "替补得分占比",
    "starter_minutes_share": "首发上场时间占比",
    # H2H
    "home_win_count": "对战历史主胜次数", "avg_point_diff": "对战历史均分差",
    "avg_total_pts": "对战历史均总分",
    # Context
    "home_rest_days": "主队休息天数", "away_rest_days": "客队休息天数",
    "rest_days_diff": "休息天数差", "home_b2b": "主队背靠背",
    "away_b2b": "客队背靠背", "home_home_win_rate": "主队主场胜率",
    "away_away_win_rate": "客队客场胜率",
    "home_recent_home_win_rate": "主队近期主场胜率",
    "away_recent_away_win_rate": "客队近期客场胜率",
    "home_streak": "主队连胜场次", "away_streak": "客队连胜场次",
    "same_conference": "同分区", "same_division": "同赛区",
    "season_series_games": "赛季对战次数", "home_season_series_wins": "主队赛季对战胜场",
    # Matchup
    "pace_diff": "节奏差", "expected_pace": "预期节奏",
    "home_off_vs_away_def": "主攻vs客防", "away_off_vs_home_def": "客攻vs主防",
    "net_rtg_diff": "净效率差", "total_rating_diff": "综合实力差",
    "ts_diff": "真实命中率差", "tov_pct_diff": "失误率差",
    "ftr_diff": "罚球率差", "expected_total": "预期总分",
    "pace_product": "节奏乘积", "combined_over_tendency": "两队总分倾向",
    # Position match
    "frontcourt_scoring": "前场得分", "backcourt_scoring": "后场得分",
    "frontcourt_scoring_diff": "前场得分差", "backcourt_scoring_diff": "后场得分差",
    # Bench depth
    "bench_plus_minus_per_game": "替补正负值",
    "starters_vs_bench_scoring_ratio": "首发vs替补得分比",
    "sixth_man_scoring": "第六人得分",
    "rotation_depth": "轮换深度",
    # Lineup stability
    "lineup_stability": "阵容稳定性",
    # Opponent-adjusted
    "opp_adj_pts": "对手调整得分",
    "opp_adj_pts_allowed": "对手调整失分",
    "opp_adj_net_rtg": "对手调整净效率",
    "opp_adj_off_rtg": "对手调整进攻效率",
    "opp_adj_def_rtg": "对手调整防守效率",
    "sos_rating": "赛程强度",
    # Travel/fatigue (context)
    "travel_distance_category": "旅行距离等级",
    "home_travel_fatigue": "主队旅途疲劳",
    "away_travel_fatigue": "客队旅途疲劳",
    "home_days_since_home": "主队离家天数",
    # Roster continuity
    "roster_continuity": "阵容连续性",
    "core_retention": "核心球员留存率",
    "roster_turnover_rate": "阵容更替率",
    "minutes_continuity": "上场时间连续性",
}

_PREFIX_CN: dict[str, str] = {"home_": "主队", "away_": "客队"}


def _make_display_name(name: str) -> str:
    """Convert a feature name like 'home_win_rate' to '主队胜率'."""
    for prefix, cn_label in _PREFIX_CN.items():
        if name.startswith(prefix):
            base = name[len(prefix):]
            if base in _DISPLAY_COMPONENTS:
                return f"{cn_label}{_DISPLAY_COMPONENTS[base]}"
            return name  # fallback: raw name
    if name in _DISPLAY_COMPONENTS:
        return _DISPLAY_COMPONENTS[name]
    return name  # fallback: raw name


FEATURE_DISPLAY_NAMES: dict[str, str] = {
    name: _make_display_name(name) for name in FEATURE_NAMES
}

COVER_OVER_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    name: _make_display_name(name) for name in COVER_OVER_FEATURE_NAMES
}


@dataclass
class FeatureVector:
    """Complete feature vector with metadata."""

    features: np.ndarray  # shape: (FEATURE_DIM,)
    feature_names: list[str] = field(default_factory=lambda: FEATURE_NAMES.copy())
    home_team_id: int = 0
    away_team_id: int = 0
    game_date: datetime | None = None
    lookback_requested: int = 10
    home_games_available: int = 0
    away_games_available: int = 0
    data_quality_warnings: list[str] = field(default_factory=list)
    # Raw feature dict for debugging / storage
    raw_features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict (features as list)."""
        return {
            "features": self.features.tolist(),
            "feature_names": self.feature_names,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "game_date": self.game_date.isoformat() if self.game_date else None,
            "lookback_requested": self.lookback_requested,
            "home_games_available": self.home_games_available,
            "away_games_available": self.away_games_available,
            "data_quality_warnings": self.data_quality_warnings,
        }


@dataclass
class TrainingDataset:
    """Dataset for model training / backtesting."""

    X: np.ndarray  # shape: (n_samples, FEATURE_DIM)
    y_win: np.ndarray  # shape: (n_samples,) — 1=home win, 0=away win
    y_spread: np.ndarray  # shape: (n_samples,) — home_score - away_score
    y_total: np.ndarray  # shape: (n_samples,) — home_score + away_score
    sample_info: list[dict]  # metadata per sample
    cache_key: str | None = None


@dataclass
class CoverOverFeatureVector:
    """Reduced cross-season-safe feature vector for cover/over prediction."""

    features: np.ndarray  # shape: (COVER_OVER_FEATURE_DIM,)
    home_games_available: int = 0
    away_games_available: int = 0


@dataclass
class CoverOverDataset:
    """Dataset for cover/over binary classification (盘口).

    X is the reduced cross-season-safe feature subset (COVER_OVER_FEATURE_DIM).
    y_cover = 1 if (home - away) > spread_line else 0  (home covers the spread)
    y_over  = 1 if (home + away) > total_line else 0   (game goes over the total)
    """

    X: np.ndarray  # shape: (n_samples, COVER_OVER_FEATURE_DIM)
    y_cover: np.ndarray  # shape: (n_samples,) — 1=home covers, 0=away covers
    y_over: np.ndarray  # shape: (n_samples,) — 1=over, 0=under
    spread_line: np.ndarray  # shape: (n_samples,) — expected home margin
    total_line: np.ndarray  # shape: (n_samples,) — expected total
    sample_info: list[dict]  # metadata per sample


class FeatureBuilder:
    """Orchestrates feature computation across all feature modules."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory
        self.team_features = TeamFeatures(session_factory)
        self.player_features = PlayerFeatures(session_factory)
        self.player_form_features = PlayerFormFeatures(session_factory)
        self.h2h_features = H2HFeatures(session_factory)
        self.context_features = ContextFeatures(session_factory)
        self.position_features = PositionMatchFeatures(session_factory)
        self.bench_features = BenchFeatures(session_factory)
        self.lineup_features = LineupFeatures(session_factory)
        self.opponent_features = OpponentAdjustedFeatures(session_factory)
        self.roster_features = RosterContinuityFeatures(session_factory)
        self.season_prior_features = SeasonPriorFeatures(session_factory)
        self.injury_net_features = InjuryNetFeatures(session_factory)

    async def build_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        lookback_games: int = 10,
        session: AsyncSession | None = None,
        opponent_cache: dict[tuple[int, str], dict] | None = None,
        use_historical: bool = False,
        as_of: datetime | None = None,
        target_game_id: int | None = None,
    ) -> FeatureVector:
        """Build complete feature vector for a single matchup.

        Args:
            home_team_id: Home team internal ID.
            away_team_id: Away team internal ID.
            game_date: Scheduled game date (features use only data before this).
            lookback_games: Number of recent games for team stats (5-15).
            session: Optional existing session.
            use_historical: If True, read injury features from the historical
                snapshot table (training); if False, from the live table.

        Returns:
            FeatureVector with numpy array and metadata.
        """
        async def _build(sess: AsyncSession) -> FeatureVector:
            from app.models.game import Game

            warnings: list[str] = []
            target_game = await sess.get(Game, target_game_id) if target_game_id else None

            # Normalize game_date to datetime (callers may pass date)
            nonlocal game_date
            from datetime import datetime as _dt
            if not isinstance(game_date, _dt):
                game_date = _dt.combine(game_date, _dt.min.time())
            cutoff_date = as_of or game_date
            if not isinstance(cutoff_date, _dt):
                cutoff_date = _dt.combine(cutoff_date, _dt.min.time())

            # Determine actual lookback for each team (adaptive)
            home_lookback = lookback_games
            away_lookback = lookback_games

            # 1. Team features
            home_team_f = await self.team_features.compute(
                home_team_id, cutoff_date, home_lookback, session=sess
            )
            away_team_f = await self.team_features.compute(
                away_team_id, cutoff_date, away_lookback, session=sess
            )

            async def _latest_game_quality(team_id: int):
                return (await sess.execute(
                    select(
                        Game.nba_game_id,
                        Game.team_stats_complete,
                        Game.player_stats_complete,
                    )
                    .where(
                        Game.status == "Final",
                        Game.game_type.in_(("regular", "play_in", "playoffs")),
                        Game.score_complete.is_(True),
                        Game.game_date < cutoff_date,
                        or_(Game.home_team_id == team_id, Game.away_team_id == team_id),
                    )
                    .order_by(desc(Game.game_date))
                    .limit(1)
                )).one_or_none()

            home_latest_quality = await _latest_game_quality(home_team_id)
            away_latest_quality = await _latest_game_quality(away_team_id)
            for side, quality in (("Home", home_latest_quality), ("Away", away_latest_quality)):
                if quality is not None and not quality.team_stats_complete:
                    warnings.append(
                        f"{side} latest game {quality.nba_game_id} has incomplete team statistics"
                    )
                if quality is not None and not quality.player_stats_complete:
                    warnings.append(
                        f"{side} latest game {quality.nba_game_id} has incomplete player statistics"
                    )

            home_games = int(home_team_f.get("games_available", 0))
            away_games = int(away_team_f.get("games_available", 0))

            if home_games < lookback_games:
                warnings.append(
                    f"Home team only has {home_games}/{lookback_games} games available"
                )
            if away_games < lookback_games:
                warnings.append(
                    f"Away team only has {away_games}/{lookback_games} games available"
                )

            # 2. Player / injury features
            home_injury_f = await self.player_features.compute_injury_impact(
                home_team_id, cutoff_date, session=sess, use_historical=use_historical,
                target_game_id=target_game_id,
            )
            away_injury_f = await self.player_features.compute_injury_impact(
                away_team_id, cutoff_date, session=sess, use_historical=use_historical,
                target_game_id=target_game_id,
            )

            # 2b. Player form features (top scorer form, team depth)
            home_form_f = await self.player_form_features.compute(
                home_team_id, cutoff_date, lookback_games, session=sess
            )
            away_form_f = await self.player_form_features.compute(
                away_team_id, cutoff_date, lookback_games, session=sess
            )

            # 3. H2H features
            h2h_f = await self.h2h_features.compute(
                home_team_id, away_team_id, cutoff_date, session=sess
            )

            # 4. Context features
            context_f = await self.context_features.compute(
                home_team_id, away_team_id, cutoff_date, session=sess,
                target_game_day=target_game.source_game_date if target_game else None,
            )

            # 5. Matchup interaction features (derived from home/away team stats)
            matchup_f = compute_matchup_features(home_team_f, away_team_f)

            # 6. Position match features
            home_pos_f = await self.position_features.compute(
                home_team_id, cutoff_date, lookback_games, session=sess
            )
            away_pos_f = await self.position_features.compute(
                away_team_id, cutoff_date, lookback_games, session=sess
            )
            pos_diff_f = PositionMatchFeatures.compute_diff(home_pos_f, away_pos_f)

            # 7. Bench depth features
            home_bench_f = await self.bench_features.compute(
                home_team_id, cutoff_date, lookback_games, session=sess
            )
            away_bench_f = await self.bench_features.compute(
                away_team_id, cutoff_date, lookback_games, session=sess
            )

            # 8. Lineup stability features
            home_lineup_f = await self.lineup_features.compute(
                home_team_id, cutoff_date, session=sess
            )
            away_lineup_f = await self.lineup_features.compute(
                away_team_id, cutoff_date, session=sess
            )

            # 9. Opponent-adjusted features (cache speeds up training ~100x)
            home_opp_f = await self.opponent_features.compute(
                home_team_id, cutoff_date, lookback_games,
                session=sess, opponent_cache=opponent_cache,
            )
            away_opp_f = await self.opponent_features.compute(
                away_team_id, cutoff_date, lookback_games,
                session=sess, opponent_cache=opponent_cache,
            )

            # 10. Roster continuity features (season-over-season player overlap)
            home_roster_f = await self.roster_features.compute(
                home_team_id, cutoff_date, session=sess
            )
            away_roster_f = await self.roster_features.compute(
                away_team_id, cutoff_date, session=sess
            )

            # 11. Smoothed prior/current-season offense, defense and pace.
            home_prior_f = await self.season_prior_features.compute(
                home_team_id, cutoff_date, session=sess
            )
            away_prior_f = await self.season_prior_features.compute(
                away_team_id, cutoff_date, session=sess
            )
            prior_pace = (
                home_prior_f.get("bayes_pace", 0.0)
                + away_prior_f.get("bayes_pace", 0.0)
            ) / 2.0
            shrunk_expected_total = prior_pace * (
                home_prior_f.get("bayes_off_rtg", 0.0)
                + away_prior_f.get("bayes_def_rtg", 0.0)
                + away_prior_f.get("bayes_off_rtg", 0.0)
                + home_prior_f.get("bayes_def_rtg", 0.0)
            ) / 200.0

            # 12. Net rotation change after injury-driven minute redistribution.
            home_net_quality, away_net_quality = {}, {}
            home_injury_net_f = await self.injury_net_features.compute(
                home_team_id, cutoff_date, session=sess, use_historical=use_historical,
                target_game_id=target_game_id, diagnostics=home_net_quality,
            )
            away_injury_net_f = await self.injury_net_features.compute(
                away_team_id, cutoff_date, session=sess, use_historical=use_historical,
                target_game_id=target_game_id, diagnostics=away_net_quality,
            )
            for side, quality in (("Home", home_net_quality), ("Away", away_net_quality)):
                if quality.get('unavailable_reason'):
                    warnings.append(f"{side} injury net unavailable: {quality['unavailable_reason']}")

            # Assemble ordered feature vector
            raw_features: dict[str, float] = {}
            feature_list: list[float] = []

            # Home team features
            for name in BASE_TEAM_FEATURE_NAMES:
                val = home_team_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away team features
            for name in BASE_TEAM_FEATURE_NAMES:
                val = away_team_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            # Home injury features
            for name in PLAYER_FEATURE_NAMES:
                val = home_injury_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away injury features
            for name in PLAYER_FEATURE_NAMES:
                val = away_injury_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            # Home player form features
            for name in PLAYER_FORM_FEATURE_NAMES:
                val = home_form_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away player form features
            for name in PLAYER_FORM_FEATURE_NAMES:
                val = away_form_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            # H2H features
            for name in H2H_FEATURE_NAMES:
                val = h2h_f.get(name, 0.0)
                raw_features[name] = val
                feature_list.append(val)

            # Context features
            for name in CONTEXT_FEATURE_NAMES:
                val = context_f.get(name, 0.0)
                raw_features[name] = val
                feature_list.append(val)

            # Matchup interaction features
            for name in MATCHUP_FEATURE_NAMES:
                val = matchup_f.get(name, 0.0)
                raw_features[name] = val
                feature_list.append(val)

            # Home position match features
            for name in POSITION_MATCH_FEATURE_NAMES:
                val = home_pos_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away position match features
            for name in POSITION_MATCH_FEATURE_NAMES:
                val = away_pos_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            # Position match diff features
            for name in POSITION_MATCH_DIFF_NAMES:
                val = pos_diff_f.get(name, 0.0)
                raw_features[name] = val
                feature_list.append(val)

            # Home bench features
            for name in BENCH_FEATURE_NAMES:
                val = home_bench_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away bench features
            for name in BENCH_FEATURE_NAMES:
                val = away_bench_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            # Home lineup features
            for name in LINEUP_FEATURE_NAMES:
                val = home_lineup_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away lineup features
            for name in LINEUP_FEATURE_NAMES:
                val = away_lineup_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            # Home opponent-adjusted features
            for name in OPPONENT_ADJUSTED_FEATURE_NAMES:
                val = home_opp_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away opponent-adjusted features
            for name in OPPONENT_ADJUSTED_FEATURE_NAMES:
                val = away_opp_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            # Home roster continuity features
            for name in BASE_ROSTER_FEATURE_NAMES:
                val = home_roster_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away roster continuity features
            for name in BASE_ROSTER_FEATURE_NAMES:
                val = away_roster_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            for name in TEAM_FEATURE_EXTENSION_NAMES:
                val = home_team_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)
            for name in TEAM_FEATURE_EXTENSION_NAMES:
                val = away_team_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)
            for name in ROSTER_FEATURE_EXTENSION_NAMES:
                val = home_roster_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)
            for name in ROSTER_FEATURE_EXTENSION_NAMES:
                val = away_roster_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            for name in SEASON_PRIOR_FEATURE_NAMES:
                val = home_prior_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            for name in SEASON_PRIOR_FEATURE_NAMES:
                val = away_prior_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            raw_features["shrunk_expected_total"] = shrunk_expected_total
            feature_list.append(shrunk_expected_total)

            for name in INJURY_NET_FEATURE_NAMES:
                val = home_injury_net_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            for name in INJURY_NET_FEATURE_NAMES:
                val = away_injury_net_f.get(name, 0.0)
                raw_features[f"away_{name}"] = val
                feature_list.append(val)

            if home_team_f.get("advanced_stats_coverage", 0.0) < 0.8:
                warnings.append("Home recent advanced statistics are incomplete")
            if away_team_f.get("advanced_stats_coverage", 0.0) < 0.8:
                warnings.append("Away recent advanced statistics are incomplete")
            if home_roster_f.get("roster_data_missing", 0.0) > 0:
                warnings.append("Home roster continuity is unknown")
            if away_roster_f.get("roster_data_missing", 0.0) > 0:
                warnings.append("Away roster continuity is unknown")
            if home_injury_net_f.get("injury_net_available", 0.0) == 0:
                warnings.append("Home net injury replacement features unavailable")
            if away_injury_net_f.get("injury_net_available", 0.0) == 0:
                warnings.append("Away net injury replacement features unavailable")

            features = np.array(feature_list, dtype=np.float64)

            # Replace NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply static caps (identical at training and inference time)
            features = np.clip(features, _FEATURE_LOWER, _FEATURE_UPPER)

            return FeatureVector(
                features=features,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                game_date=game_date,
                lookback_requested=lookback_games,
                home_games_available=home_games,
                away_games_available=away_games,
                data_quality_warnings=warnings,
                raw_features=raw_features,
            )

        if session is not None:
            return await _build(session)
        else:
            async with self.session_factory() as sess:
                return await _build(sess)

    async def build_training_dataset(
        self,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        min_games_before: int = 10,
        session: AsyncSession | None = None,
        use_historical: bool = True,
        use_cache: bool = True,
    ) -> TrainingDataset:
        """Build a complete training dataset from historical games.

        For each completed game in the database (that has enough prior data),
        computes the feature vector AS OF the game date (no data leakage).

        Args:
            seasons: List of season strings to include (e.g. ["2024-25", "2025-26"]).
                     If None, uses all available seasons.
            lookback_games: Number of recent games for team features.
            min_games_before: Minimum prior games required for each team.
            session: Optional existing session.
            use_historical: Read injury features from the historical snapshot
                table (default True — the live table has no historical variance).
            use_cache: Reuse a cached `.npz` when the feature schema, build
                parameters and database fingerprint are unchanged. A new game,
                corrected score, roster move or injury edit invalidates it.

        Returns:
            TrainingDataset with X, y arrays and sample metadata.
        """
        from app.models.game import Game

        cache: DatasetCache | None = None
        cache_key: str | None = None

        async def _build(sess: AsyncSession) -> TrainingDataset:
            nonlocal cache, cache_key

            # ── Cache lookup (keyed by schema + parameters + DB fingerprint) ──
            if use_cache:
                cache = DatasetCache()
                cache_key = await cache.key(
                    sess,
                    feature_names=FEATURE_NAMES,
                    seasons=seasons,
                    lookback_games=lookback_games,
                    min_games_before=min_games_before,
                    use_historical=use_historical,
                )
                cached = cache.load(cache_key, FEATURE_DIM)
                if cached is not None:
                    return cached

            # ── Precompute opponent season stats (one pass, O(games)) ──
            # Without this cache, build_features() fires ~600K DB queries for
            # opponent stats (N+1 pattern).  With it, the entire lookup is O(1).
            logger.info("Precomputing opponent season stats cache...")
            opponent_cache = await OpponentAdjustedFeatures.precompute_opponent_cache(sess)

            # Query all completed games
            query = select(Game).where(
                Game.status == "Final",
                Game.score_complete.is_(True),
                Game.game_type.in_(["regular", "play_in", "playoffs"]),
            )
            if seasons:
                query = query.where(Game.season.in_(seasons))
            query = query.order_by(Game.game_date.asc())

            result = await sess.execute(query)
            games = result.scalars().all()
            games = sorted(games, key=lambda g: (
                g.source_game_date or g.game_date.date(), g.game_date, g.id,
            ))

            logger.info("Building dataset from %d completed games...", len(games))

            # Reject rows with too little complete prior team history before
            # computing the expensive player/roster feature groups. Counts use
            # strictly earlier dates, exactly like TeamFeatures' SQL query.
            prior_dates = defaultdict(list)
            complete_games = (await sess.execute(
                select(Game.home_team_id, Game.away_team_id, Game.game_date)
                .where(
                    Game.status == "Final", Game.team_stats_complete.is_(True),
                    Game.game_type.in_(("regular", "play_in", "playoffs")),
                ).order_by(Game.game_date.asc())
            )).all()
            for home_id, away_id, day in complete_games:
                prior_dates[home_id].append(day)
                prior_dates[away_id].append(day)
            games = [g for g in games if all(
                bisect_left(prior_dates[team_id], g.game_date) >= min_games_before
                for team_id in (g.home_team_id, g.away_team_id)
            )]
            logger.info("%d games have sufficient complete prior team history", len(games))

            X_list: list[np.ndarray] = []
            y_win_list: list[float] = []
            y_spread_list: list[float] = []
            y_total_list: list[float] = []
            sample_info: list[dict] = []
            failures = 0

            for i, game in enumerate(games):
                if game.home_score is None or game.away_score is None:
                    continue

                # Guard against corrupted rows that would poison the targets:
                # the Kaggle import produced a couple of 0-0 "Final" games and
                # one 36-33 exhibition game.
                game_total = game.home_score + game.away_score
                if (
                    game.home_score <= 0
                    or game.away_score <= 0
                    or game_total < 120
                    or game_total > 400
                ):
                    logger.debug(
                        "Skipping implausible final score %s-%s (%s)",
                        game.home_score,
                        game.away_score,
                        game.nba_game_id,
                    )
                    continue

                try:
                    fv = await self.build_features(
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        game_date=game.game_date,
                        lookback_games=lookback_games,
                        session=sess,
                        opponent_cache=opponent_cache,
                        use_historical=use_historical,
                        target_game_id=game.id,
                    )
                except Exception as e:
                    failures += 1
                    if failures <= 5:
                        logger.warning("Failed to build features for game %s: %s", game.nba_game_id, e)
                    continue

                # Skip if insufficient prior data
                if fv.home_games_available < min_games_before or fv.away_games_available < min_games_before:
                    continue

                X_list.append(fv.features)
                y_win_list.append(1.0 if (game.home_score > game.away_score) else 0.0)
                y_spread_list.append(float(game.home_score - game.away_score))
                y_total_list.append(float(game.home_score + game.away_score))
                sample_info.append({
                    "nba_game_id": game.nba_game_id,
                    "game_date": game.game_date.isoformat(),
                    "source_game_date": str(game.source_game_date or game.game_date.date()),
                    "time_precision": game.time_precision,
                    "feature_mode": "reconstructed_replay",
                    "season": game.season,
                    "game_type": game.game_type,
                    "score_complete": game.score_complete,
                    "team_stats_complete": game.team_stats_complete,
                    "player_stats_complete": game.player_stats_complete,
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                    "home_games_available": fv.home_games_available,
                    "away_games_available": fv.away_games_available,
                })

                # Progress logging: every 500 games or every 10% of total
                total_games = len(games)
                if (i + 1) % max(1, min(500, total_games // 10)) == 0:
                    logger.info(
                        "  Dataset progress: %d/%d games (%.0f%%), %d valid samples",
                        i + 1, total_games, (i + 1) / total_games * 100,
                        len(X_list),
                    )

            if not X_list:
                logger.warning("No valid training samples found!")
                return TrainingDataset(
                    X=np.array([]).reshape(0, FEATURE_DIM),
                    y_win=np.array([]),
                    y_spread=np.array([]),
                    y_total=np.array([]),
                    sample_info=[],
                )

            if failures:
                logger.warning("Feature build failed for %d/%d eligible games", failures, len(games))

            X = np.vstack(X_list)
            y_win = np.array(y_win_list)
            y_spread = np.array(y_spread_list)
            y_total = np.array(y_total_list)

            logger.info(
                "Built dataset: %d samples, %d features, "
                "win balance=%.2f, spread range=[%.1f, %.1f], total range=[%.0f, %.0f]",
                len(X_list),
                X.shape[1],
                y_win.mean(),
                y_spread.min(),
                y_spread.max(),
                y_total.min(),
                y_total.max(),
            )

            dataset = TrainingDataset(
                X=X,
                y_win=y_win,
                y_spread=y_spread,
                y_total=y_total,
                sample_info=sample_info,
                cache_key=cache_key,
            )

            if cache is not None and cache_key is not None:
                cache.save(cache_key, dataset)

            return dataset

        if session is not None:
            return await _build(session)
        else:
            async with self.session_factory() as sess:
                # One immutable DB snapshot across a potentially long build.
                # Concurrent score/stat imports must not mix revisions.
                await sess.connection(execution_options={'isolation_level':'REPEATABLE READ'})
                return await _build(sess)

    async def build_cover_over_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        lookback_games: int = 10,
        session: AsyncSession | None = None,
        opponent_cache: dict[tuple[int, str], dict] | None = None,
    ) -> CoverOverFeatureVector:
        """Build only the cross-season-safe feature subset for cover/over.

        Skips every player-stat-derived feature group (injury, player form,
        position, bench, lineup, roster continuity) and advanced stats, which
        are unreliable in the 2012-2019 era where betting lines live.  This is
        both faster and avoids a "0 in train / nonzero in predict" mismatch.
        """
        async def _build(sess: AsyncSession) -> CoverOverFeatureVector:
            home_team_f = await self.team_features.compute(
                home_team_id, game_date, lookback_games, session=sess
            )
            away_team_f = await self.team_features.compute(
                away_team_id, game_date, lookback_games, session=sess
            )
            h2h_f = await self.h2h_features.compute(
                home_team_id, away_team_id, game_date, session=sess
            )
            context_f = await self.context_features.compute(
                home_team_id, away_team_id, game_date, session=sess
            )
            matchup_f = compute_matchup_features(home_team_f, away_team_f)
            home_opp_f = await self.opponent_features.compute(
                home_team_id, game_date, lookback_games,
                session=sess, opponent_cache=opponent_cache,
            )
            away_opp_f = await self.opponent_features.compute(
                away_team_id, game_date, lookback_games,
                session=sess, opponent_cache=opponent_cache,
            )

            feature_list: list[float] = []
            for name in COVER_OVER_SAFE_TEAM_FEATURES:
                feature_list.append(home_team_f.get(name, 0.0))
            for name in COVER_OVER_SAFE_TEAM_FEATURES:
                feature_list.append(away_team_f.get(name, 0.0))
            for name in H2H_FEATURE_NAMES:
                feature_list.append(h2h_f.get(name, 0.0))
            for name in CONTEXT_FEATURE_NAMES:
                feature_list.append(context_f.get(name, 0.0))
            for name in COVER_OVER_SAFE_MATCHUP_FEATURES:
                feature_list.append(matchup_f.get(name, 0.0))
            for name in OPPONENT_ADJUSTED_FEATURE_NAMES:
                feature_list.append(home_opp_f.get(name, 0.0))
            for name in OPPONENT_ADJUSTED_FEATURE_NAMES:
                feature_list.append(away_opp_f.get(name, 0.0))

            features = np.nan_to_num(
                np.array(feature_list, dtype=np.float64),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            features = np.clip(features, _COVER_LOWER, _COVER_UPPER)
            return CoverOverFeatureVector(
                features=features,
                home_games_available=int(home_team_f.get("games_available", 0)),
                away_games_available=int(away_team_f.get("games_available", 0)),
            )

        if session is not None:
            return await _build(session)
        else:
            async with self.session_factory() as sess:
                return await _build(sess)

    async def build_cover_over_dataset(
        self,
        seasons: list[str] | None = None,
        lookback_games: int = 10,
        min_games_before: int = 10,
        session: AsyncSession | None = None,
    ) -> CoverOverDataset:
        """Build the cover/over training dataset from games that have a line.

        Joins games to betting_lines (only 2012-2019 have lines) and builds the
        reduced feature vector as of each game date (no data leakage).
        """
        from app.models.game import Game
        from app.models.betting_line import BettingLine

        async def _build(sess: AsyncSession) -> CoverOverDataset:
            opponent_cache = await OpponentAdjustedFeatures.precompute_opponent_cache(sess)

            query = (
                select(Game, BettingLine)
                .join(BettingLine, BettingLine.game_id == Game.id)
                .where(Game.status == "Final")
            )
            if seasons:
                query = query.where(Game.season.in_(seasons))
            query = query.order_by(Game.game_date.asc())

            result = await sess.execute(query)
            pairs = result.all()
            logger.info("Building cover/over dataset from %d lined games...", len(pairs))

            X_list: list[np.ndarray] = []
            y_cover_list: list[float] = []
            y_over_list: list[float] = []
            spread_list: list[float] = []
            total_list: list[float] = []
            sample_info: list[dict] = []

            for i, (game, line) in enumerate(pairs):
                if game.home_score is None or game.away_score is None:
                    continue
                if line.spread_line is None or line.total_line is None:
                    continue

                try:
                    fv = await self.build_cover_over_features(
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        game_date=game.game_date,
                        lookback_games=lookback_games,
                        session=sess,
                        opponent_cache=opponent_cache,
                    )
                except Exception as e:
                    logger.debug("Failed to build cover/over features for %s: %s", game.nba_game_id, e)
                    continue

                if fv.home_games_available < min_games_before or fv.away_games_available < min_games_before:
                    continue

                home = float(game.home_score)
                away = float(game.away_score)
                margin = home - away
                total = home + away

                X_list.append(fv.features)
                y_cover_list.append(1.0 if margin > line.spread_line else 0.0)
                y_over_list.append(1.0 if total > line.total_line else 0.0)
                spread_list.append(float(line.spread_line))
                total_list.append(float(line.total_line))
                sample_info.append({
                    "nba_game_id": game.nba_game_id,
                    "game_date": game.game_date.isoformat(),
                    "season": game.season,
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                    "home_score": home,
                    "away_score": away,
                    "spread_line": float(line.spread_line),
                    "total_line": float(line.total_line),
                    "cover": 1.0 if margin > line.spread_line else 0.0,
                    "over": 1.0 if total > line.total_line else 0.0,
                })

                if (i + 1) % 500 == 0:
                    logger.info(
                        "  Cover/over progress: %d/%d (%.0f%%), %d valid",
                        i + 1, len(pairs), (i + 1) / len(pairs) * 100, len(X_list),
                    )

            if not X_list:
                logger.warning("No valid cover/over training samples found!")
                return CoverOverDataset(
                    X=np.array([]).reshape(0, COVER_OVER_FEATURE_DIM),
                    y_cover=np.array([]),
                    y_over=np.array([]),
                    spread_line=np.array([]),
                    total_line=np.array([]),
                    sample_info=[],
                )

            X = np.vstack(X_list)
            y_cover = np.array(y_cover_list)
            y_over = np.array(y_over_list)
            spread_line = np.array(spread_list)
            total_line = np.array(total_list)

            logger.info(
                "Built cover/over dataset: %d samples, %d features, "
                "cover balance=%.2f, over balance=%.2f",
                len(X_list), X.shape[1], y_cover.mean(), y_over.mean(),
            )

            return CoverOverDataset(
                X=X,
                y_cover=y_cover,
                y_over=y_over,
                spread_line=spread_line,
                total_line=total_line,
                sample_info=sample_info,
            )

        if session is not None:
            return await _build(session)
        else:
            async with self.session_factory() as sess:
                return await _build(sess)
