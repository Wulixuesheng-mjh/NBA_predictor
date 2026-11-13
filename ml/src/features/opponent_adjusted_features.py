"""Opponent-adjusted team statistics.

Raw team stats (avg_pts, avg_net_rtg, etc.) don't account for opponent quality.
A team averaging 120 PPG against weak defenses is overrated by raw stats alone.

This module adjusts team metrics by the quality of opponents faced:
  - opp_adj_pts:           Actual PPG / opponent average points allowed
  - opp_adj_pts_allowed:   Actual points allowed / opponent average PPG
  - opp_adj_net_rtg:       Net rating adjusted for opponent strength
  - opp_adj_off_rtg:       Offensive rating vs opponent average defensive rating
  - opp_adj_def_rtg:       Defensive rating vs opponent average offensive rating
  - sos_rating:            Average opponent win rate (strength of schedule)

Performance: during training, use OpponentStatsCache (built in one pass) which
answers "team stats as of date D" without touching games after D, so the offline
features match what a live prediction would see.

Leakage note: the previous implementation averaged the *whole season* for each
(team, season), which let a game played in November see stats from April — both
a train/serve mismatch and a direct leak of future results.
"""

import logging
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)


def previous_season(season: str) -> str | None:
    """Return the season before ``season`` ("2025-26" → "2024-25")."""
    if not season:
        return None
    if "-" in season:
        try:
            start = int(season.split("-")[0])
            return f"{start - 1}-{start % 100:02d}"
        except ValueError:
            return None
    try:
        return str(int(season) - 1)
    except ValueError:
        return None


OPPONENT_ADJUSTED_FEATURE_NAMES = [
    "opp_adj_pts",
    "opp_adj_pts_allowed",
    "opp_adj_net_rtg",
    "opp_adj_off_rtg",
    "opp_adj_def_rtg",
    "sos_rating",
]


class OpponentStatsCache:
    """As-of-date team stats: a lookup at date D only sees games before D.

    One pass over all completed games builds cumulative series per
    (team, season); ``get()`` binary-searches the date. This keeps offline
    feature building consistent with a live prediction and removes the
    "November game knows April stats" leak of the previous whole-season
    aggregate.
    """

    def __init__(self) -> None:
        self._dates: dict[tuple[int, str], list[datetime]] = defaultdict(list)
        self._cum: dict[tuple[int, str], list[tuple[float, float, int]]] = defaultdict(list)

    def add_game(
        self,
        team_id: int,
        season: str,
        game_date: datetime,
        pts: float,
        pts_allowed: float,
        win: bool,
    ) -> None:
        key = (team_id, season)
        prev = self._cum[key][-1] if self._cum[key] else (0.0, 0.0, 0)
        self._cum[key].append(
            (prev[0] + pts, prev[1] + pts_allowed, prev[2] + (1 if win else 0))
        )
        self._dates[key].append(game_date)

    @staticmethod
    def _stats(pts: float, allowed: float, wins: int, n: int) -> dict[str, float]:
        return {
            "avg_pts": pts / n,
            "avg_pts_allowed": allowed / n,
            "avg_off_rtg": pts / n,
            "avg_def_rtg": allowed / n,
            "win_rate": wins / n,
        }

    def get(self, team_id: int, season: str, before_date: datetime) -> dict[str, float] | None:
        key = (team_id, season)
        dates = self._dates.get(key)
        if dates:
            idx = bisect_left(dates, before_date)  # games strictly before the date
            if idx > 0:
                pts, allowed, wins = self._cum[key][idx - 1]
                return self._stats(pts, allowed, wins, idx)

        # Not enough current-season history (season opener): use last season
        prev = previous_season(season)
        if prev:
            prev_key = (team_id, prev)
            prev_dates = self._dates.get(prev_key)
            if prev_dates:
                pts, allowed, wins = self._cum[prev_key][-1]
                return self._stats(pts, allowed, wins, len(prev_dates))
        return None


class OpponentAdjustedFeatures:
    """Compute opponent-adjusted team statistics over a recent window."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    # ── Public API ──────────────────────────────────────────────────────

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        n_games: int = 10,
        session: AsyncSession | None = None,
        opponent_cache: "OpponentStatsCache | None" = None,
    ) -> dict[str, float]:
        """Compute opponent-adjusted stats for a team.

        For each of the team's last N games before `before_date`:
        1. Find the opponent
        2. Look up the opponent's season-average stats (from cache if provided)
        3. Compare team's performance to opponent's typical numbers

        Args:
            team_id: Internal team ID.
            before_date: Reference date (only games before this are used).
            n_games: Number of recent games to look back.
            session: Optional existing DB session.
            opponent_cache: Optional precomputed OpponentStatsCache. When
                provided, per-game opponent lookups are answered from memory
                (as-of-date, so no future leakage).

        Returns:
            Dict with opp_adj_* feature values.
        """
        async def _compute(sess: AsyncSession) -> dict[str, float]:
            from app.models.game import Game

            # ── Get the team's last N games ──
            team_games_query = (
                select(Game)
                .where(
                    and_(
                        Game.game_date < before_date,
                        Game.status == "Final",
                        Game.game_type.in_(("regular", "play_in", "playoffs")),
                        Game.score_complete.is_(True),
                        Game.home_score.isnot(None),
                        Game.away_score.isnot(None),
                        (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                    )
                )
                .order_by(Game.game_date.desc())
                .limit(n_games)
            )
            result = await sess.execute(team_games_query)
            team_games = list(result.scalars().all())

            if not team_games:
                return self._default_result()

            # ── Compute opponent quality per game ──
            opp_adj_pts_list: list[float] = []
            opp_adj_pts_allowed_list: list[float] = []
            opp_adj_net_rtg_list: list[float] = []
            opp_adj_off_rtg_list: list[float] = []
            opp_adj_def_rtg_list: list[float] = []
            opp_win_rates: list[float] = []

            for game in team_games:
                is_home = game.home_team_id == team_id
                opp_id = game.away_team_id if is_home else game.home_team_id

                team_pts = game.home_score if is_home else game.away_score
                opp_pts = game.away_score if is_home else game.home_score

                # ── Opponent season stats (cache or DB fallback) ──
                if opponent_cache is not None:
                    opp_stats = opponent_cache.get(
                        opp_id, game.season or "", game.game_date
                    )
                else:
                    opp_stats = await self._get_opponent_season_stats(
                        sess, opp_id, game.season or "", game.game_date
                    )

                if opp_stats is None:
                    continue

                opp_avg_pts_allowed = opp_stats["avg_pts_allowed"]
                opp_avg_pts = opp_stats["avg_pts"]
                opp_avg_off_rtg = opp_stats["avg_off_rtg"]
                opp_avg_def_rtg = opp_stats["avg_def_rtg"]
                opp_win_rate = opp_stats["win_rate"]

                # Opponent-adjusted ratios (>1 = better than opponent typically allows)
                if opp_avg_pts_allowed > 0:
                    opp_adj_pts_list.append(team_pts / opp_avg_pts_allowed)
                if opp_avg_pts > 0:
                    opp_adj_pts_allowed_list.append(opp_pts / opp_avg_pts)

                # Net rating adjustment
                raw_net = team_pts - opp_pts
                opp_avg_net = opp_avg_off_rtg - opp_avg_def_rtg
                opp_adj_net_rtg_list.append(raw_net - opp_avg_net)

                if opp_avg_def_rtg > 0:
                    opp_adj_off_rtg_list.append(team_pts / opp_avg_def_rtg)
                else:
                    opp_adj_off_rtg_list.append(1.0)

                if opp_avg_off_rtg > 0:
                    opp_adj_def_rtg_list.append(opp_pts / opp_avg_off_rtg)
                else:
                    opp_adj_def_rtg_list.append(1.0)

                opp_win_rates.append(opp_win_rate)

            # ── Average across the window ──
            def _mean(lst: list[float]) -> float:
                return sum(lst) / len(lst) if lst else 0.0

            return {
                "opp_adj_pts": round(_mean(opp_adj_pts_list), 4),
                "opp_adj_pts_allowed": round(_mean(opp_adj_pts_allowed_list), 4),
                "opp_adj_net_rtg": round(_mean(opp_adj_net_rtg_list), 2),
                "opp_adj_off_rtg": round(_mean(opp_adj_off_rtg_list), 4),
                "opp_adj_def_rtg": round(_mean(opp_adj_def_rtg_list), 4),
                "sos_rating": round(_mean(opp_win_rates), 4),
            }

        if session is not None:
            return await _compute(session)
        else:
            async with self.session_factory() as sess:
                return await _compute(sess)

    # ── Cache precomputation (bulk, for training) ───────────────────────

    @staticmethod
    async def precompute_opponent_cache(session: AsyncSession) -> "OpponentStatsCache":
        """Build the as-of-date team stats cache in ONE pass over all games.

        Returns:
            OpponentStatsCache — call ``.get(team_id, season, before_date)``
            for stats that only use games played before that date.
        """
        from app.models.game import Game

        query = (
            select(Game)
            .where(
                Game.status == "Final",
                Game.game_type.in_(("regular", "play_in", "playoffs")),
                Game.score_complete.is_(True),
                Game.home_score.isnot(None),
                Game.away_score.isnot(None),
            )
            .order_by(Game.game_date.asc())
        )
        result = await session.execute(query)
        all_games = result.scalars().all()

        cache = OpponentStatsCache()
        for g in all_games:
            season = g.season or "unknown"
            cache.add_game(
                team_id=g.home_team_id,
                season=season,
                game_date=g.game_date,
                pts=float(g.home_score),
                pts_allowed=float(g.away_score),
                win=g.home_score > g.away_score,
            )
            cache.add_game(
                team_id=g.away_team_id,
                season=season,
                game_date=g.game_date,
                pts=float(g.away_score),
                pts_allowed=float(g.home_score),
                win=g.away_score > g.home_score,
            )

        logger.info(
            "Precomputed opponent cache: %d (team, season) series from %d games",
            len(cache._cum),
            len(all_games),
        )
        return cache

    # ── Fallback: per-team lookup (used during single-game prediction) ─

    @staticmethod
    async def _get_opponent_season_stats(
        session: AsyncSession,
        team_id: int,
        season: str,
        before_date: datetime,
    ) -> dict | None:
        """Compute a team's season-to-date averages (as of before_date).

        Mirrors OpponentStatsCache.get(): current-season games before
        ``before_date``; if the team has not played yet this season, fall back
        to its final stats from the previous season (pre-season prior).

        Used as a fallback during single-game prediction when no precomputed
        cache is available.  For training, prefer precompute_opponent_cache().
        """
        from app.models.game import Game

        async def _aggregate(season_filter: str | None) -> dict | None:
            conditions = [
                Game.status == "Final",
                Game.game_type.in_(("regular", "play_in", "playoffs")),
                Game.score_complete.is_(True),
                Game.home_score.isnot(None),
                Game.away_score.isnot(None),
                (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
            ]
            if season_filter is not None:
                conditions.append(Game.season == season_filter)
            if season_filter is not None:
                conditions.append(Game.game_date < before_date)

            result = await session.execute(select(Game).where(and_(*conditions)))
            games = result.scalars().all()
            if not games:
                return None

            total_pts = 0.0
            total_pts_allowed = 0.0
            wins = 0
            for g in games:
                is_home = g.home_team_id == team_id
                pts = g.home_score if is_home else g.away_score
                pts_allowed = g.away_score if is_home else g.home_score
                total_pts += pts
                total_pts_allowed += pts_allowed
                if pts > pts_allowed:
                    wins += 1

            n = len(games)
            return {
                "avg_pts": total_pts / n,
                "avg_pts_allowed": total_pts_allowed / n,
                "avg_off_rtg": total_pts / n,
                "avg_def_rtg": total_pts_allowed / n,
                "win_rate": wins / n,
            }

        stats = await _aggregate(season)
        if stats is not None:
            return stats

        prev = previous_season(season)
        if prev:
            return await _aggregate(prev)
        return None

    @staticmethod
    def _default_result() -> dict[str, float]:
        return {
            "opp_adj_pts": 1.0,
            "opp_adj_pts_allowed": 1.0,
            "opp_adj_net_rtg": 0.0,
            "opp_adj_off_rtg": 0.0,
            "opp_adj_def_rtg": 0.0,
            "sos_rating": 0.5,
        }
