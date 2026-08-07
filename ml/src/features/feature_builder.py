"""Feature builder — orchestrates all feature computers into a unified vector.

This is the main entry point for feature computation. It:
1. Delegates to TeamFeatures, PlayerFeatures, H2HFeatures, ContextFeatures
2. Computes matchup interaction features from home/away team stats
3. Assembles the ~126-dimension feature vector in a fixed order
4. Handles adaptive lookback with early-season fallback
5. Provides dataset construction for model training
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory
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

logger = logging.getLogger(__name__)

# ── Complete feature name list in fixed order ──

FEATURE_NAMES = []
# Team features: home_* + away_*
for name in TEAM_FEATURE_NAMES:
    FEATURE_NAMES.append(f"home_{name}")
for name in TEAM_FEATURE_NAMES:
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

FEATURE_DIM = len(FEATURE_NAMES)

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

    async def build_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        lookback_games: int = 10,
        session: AsyncSession | None = None,
        opponent_cache: dict[tuple[int, str], dict] | None = None,
    ) -> FeatureVector:
        """Build complete feature vector for a single matchup.

        Args:
            home_team_id: Home team internal ID.
            away_team_id: Away team internal ID.
            game_date: Scheduled game date (features use only data before this).
            lookback_games: Number of recent games for team stats (5-15).
            session: Optional existing session.

        Returns:
            FeatureVector with numpy array and metadata.
        """
        async def _build(sess: AsyncSession) -> FeatureVector:
            warnings: list[str] = []

            # Determine actual lookback for each team (adaptive)
            home_lookback = lookback_games
            away_lookback = lookback_games

            # 1. Team features
            home_team_f = await self.team_features.compute(
                home_team_id, game_date, home_lookback, session=sess
            )
            away_team_f = await self.team_features.compute(
                away_team_id, game_date, away_lookback, session=sess
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
                home_team_id, game_date, session=sess
            )
            away_injury_f = await self.player_features.compute_injury_impact(
                away_team_id, game_date, session=sess
            )

            # 2b. Player form features (top scorer form, team depth)
            home_form_f = await self.player_form_features.compute(
                home_team_id, game_date, lookback_games, session=sess
            )
            away_form_f = await self.player_form_features.compute(
                away_team_id, game_date, lookback_games, session=sess
            )

            # 3. H2H features
            h2h_f = await self.h2h_features.compute(
                home_team_id, away_team_id, game_date, session=sess
            )

            # 4. Context features
            context_f = await self.context_features.compute(
                home_team_id, away_team_id, game_date, session=sess
            )

            # 5. Matchup interaction features (derived from home/away team stats)
            matchup_f = compute_matchup_features(home_team_f, away_team_f)

            # 6. Position match features
            home_pos_f = await self.position_features.compute(
                home_team_id, game_date, lookback_games, session=sess
            )
            away_pos_f = await self.position_features.compute(
                away_team_id, game_date, lookback_games, session=sess
            )
            pos_diff_f = PositionMatchFeatures.compute_diff(home_pos_f, away_pos_f)

            # 7. Bench depth features
            home_bench_f = await self.bench_features.compute(
                home_team_id, game_date, lookback_games, session=sess
            )
            away_bench_f = await self.bench_features.compute(
                away_team_id, game_date, lookback_games, session=sess
            )

            # 8. Lineup stability features
            home_lineup_f = await self.lineup_features.compute(
                home_team_id, game_date, session=sess
            )
            away_lineup_f = await self.lineup_features.compute(
                away_team_id, game_date, session=sess
            )

            # 9. Opponent-adjusted features (cache speeds up training ~100x)
            home_opp_f = await self.opponent_features.compute(
                home_team_id, game_date, lookback_games,
                session=sess, opponent_cache=opponent_cache,
            )
            away_opp_f = await self.opponent_features.compute(
                away_team_id, game_date, lookback_games,
                session=sess, opponent_cache=opponent_cache,
            )

            # Assemble ordered feature vector
            raw_features: dict[str, float] = {}
            feature_list: list[float] = []

            # Home team features
            for name in TEAM_FEATURE_NAMES:
                val = home_team_f.get(name, 0.0)
                raw_features[f"home_{name}"] = val
                feature_list.append(val)

            # Away team features
            for name in TEAM_FEATURE_NAMES:
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

            features = np.array(feature_list, dtype=np.float64)

            # Replace NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

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

        Returns:
            TrainingDataset with X, y arrays and sample metadata.
        """
        from app.models.game import Game

        async def _build(sess: AsyncSession) -> TrainingDataset:
            # ── Precompute opponent season stats (one pass, O(games)) ──
            # Without this cache, build_features() fires ~600K DB queries for
            # opponent stats (N+1 pattern).  With it, the entire lookup is O(1).
            logger.info("Precomputing opponent season stats cache...")
            opponent_cache = await OpponentAdjustedFeatures.precompute_opponent_cache(sess)

            # Query all completed games
            query = select(Game).where(Game.status == "Final")
            if seasons:
                query = query.where(Game.season.in_(seasons))
            query = query.order_by(Game.game_date.asc())

            result = await sess.execute(query)
            games = result.scalars().all()

            logger.info("Building dataset from %d completed games...", len(games))

            X_list: list[np.ndarray] = []
            y_win_list: list[float] = []
            y_spread_list: list[float] = []
            y_total_list: list[float] = []
            sample_info: list[dict] = []

            for game in games:
                if game.home_score is None or game.away_score is None:
                    continue

                try:
                    fv = await self.build_features(
                        home_team_id=game.home_team_id,
                        away_team_id=game.away_team_id,
                        game_date=game.game_date,
                        lookback_games=lookback_games,
                        session=sess,
                        opponent_cache=opponent_cache,
                    )
                except Exception as e:
                    logger.debug("Failed to build features for game %s: %s", game.nba_game_id, e)
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
                    "home_team_id": game.home_team_id,
                    "away_team_id": game.away_team_id,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                    "home_games_available": fv.home_games_available,
                    "away_games_available": fv.away_games_available,
                })

            if not X_list:
                logger.warning("No valid training samples found!")
                return TrainingDataset(
                    X=np.array([]).reshape(0, FEATURE_DIM),
                    y_win=np.array([]),
                    y_spread=np.array([]),
                    y_total=np.array([]),
                    sample_info=[],
                )

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

            return TrainingDataset(
                X=X,
                y_win=y_win,
                y_spread=y_spread,
                y_total=y_total,
                sample_info=sample_info,
            )

        if session is not None:
            return await _build(session)
        else:
            async with self.session_factory() as sess:
                return await _build(sess)
