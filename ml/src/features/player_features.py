"""Player availability and injury impact features.

Computes firepower loss due to injured/absent players using the
minutes-weighted formula:

    firepower_loss = Σ(absent_player_avg_pts × avg_minutes / 48)

This properly weights player importance: a 20-ppg starter playing 36 min
counts more than a 20-ppg sixth man playing 20 min.

Injury priority for predictions (from plan):
    1. source='manual' (user manually added)
    2. confirmation='modified' (user reviewed and changed)
    3. confirmation='confirmed' (user confirmed)
    4. confirmation='pending' (auto-fetched, unreviewed)
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sqlalchemy import select, and_, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

PLAYER_FEATURE_NAMES = [
    "injury_firepower_loss",
    "total_minutes_share_lost",
    "key_player_out",
    "defensive_anchor_out",
    "players_out_count",
    "questionable_count",
]


@dataclass
class InjuryInfo:
    """Aggregated injury information for a player."""

    player_id: int
    injury_status: str  # Out / Doubtful / Questionable / Probable
    priority: int  # 1=manual, 2=modified, 3=confirmed, 4=pending
    season_ppg: float
    season_mpg: float  # minutes per game
    season_spg: float  # steals per game
    season_bpg: float  # blocks per game


class PlayerFeatures:
    """Computes player availability and injury impact features."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute_injury_impact(
        self,
        team_id: int,
        before_date: datetime,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute injury firepower loss for a team.

        Args:
            team_id: Internal team ID.
            before_date: Reference date for injury status.
            session: Optional existing session.

        Returns:
            Dict mapping feature_name -> float.
        """
        injuries = await self._query_active_injuries(team_id, before_date, session)
        return self._compute_from_injuries(injuries)

    async def _query_active_injuries(
        self,
        team_id: int,
        before_date: datetime,
        session: AsyncSession | None = None,
    ) -> list[InjuryInfo]:
        """Query active injuries for a team, with player season averages."""
        from app.models.player_injuries import PlayerInjury
        from app.models.player import Player
        from app.models.game_player_stats import GamePlayerStats
        from app.models.game import Game

        async def _query(sess: AsyncSession) -> list[InjuryInfo]:
            # Priority ordering: manual(1) > modified(2) > confirmed(3) > pending(4)
            priority_expr = case(
                (PlayerInjury.source == "manual", 1),
                (PlayerInjury.confirmation == "modified", 2),
                (PlayerInjury.confirmation == "confirmed", 3),
                else_=4,
            ).label("priority")

            # Get active injuries for players on this team
            query = (
                select(
                    PlayerInjury,
                    Player.nba_player_id,
                    priority_expr,
                )
                .join(Player, PlayerInjury.player_id == Player.id)
                .where(
                    and_(
                        Player.team_id == team_id,
                        PlayerInjury.is_active == True,
                        PlayerInjury.reported_date <= before_date,
                    )
                )
            )
            result = await sess.execute(query)
            injury_rows = result.all()

            if not injury_rows:
                return []

            # Get affected player IDs
            player_ids = [row.PlayerInjury.player_id for row in injury_rows]

            # Query season averages for these players (games before reference date)
            # We need per-game stats to compute averages
            player_stats_query = (
                select(
                    GamePlayerStats.player_id,
                    func.avg(GamePlayerStats.pts).label("avg_pts"),
                    func.avg(GamePlayerStats.stl).label("avg_stl"),
                    func.avg(GamePlayerStats.blk).label("avg_blk"),
                    func.count(GamePlayerStats.id).label("games_played"),
                )
                .join(Game, GamePlayerStats.game_id == Game.id)
                .where(
                    and_(
                        GamePlayerStats.player_id.in_(player_ids),
                        Game.game_date < before_date,
                        Game.status == "Final",
                    )
                )
                .group_by(GamePlayerStats.player_id)
            )
            stats_result = await sess.execute(player_stats_query)
            stats_map = {
                row.player_id: {
                    "ppg": float(row.avg_pts or 0),
                    "spg": float(row.avg_stl or 0),
                    "bpg": float(row.avg_blk or 0),
                    "games": row.games_played,
                }
                for row in stats_result.all()
            }

            # Also compute minutes per game (stored as string like "MM:SS" or float)
            minutes_query = (
                select(
                    GamePlayerStats.player_id,
                    GamePlayerStats.minutes,
                )
                .join(Game, GamePlayerStats.game_id == Game.id)
                .where(
                    and_(
                        GamePlayerStats.player_id.in_(player_ids),
                        Game.game_date < before_date,
                        Game.status == "Final",
                    )
                )
            )
            minutes_result = await sess.execute(minutes_query)
            minutes_rows = minutes_result.all()

            # Compute average minutes per game for each player
            minutes_data: dict[int, list[float]] = {}
            for row in minutes_rows:
                mins = _parse_minutes(row.minutes)
                if mins is not None:
                    minutes_data.setdefault(row.player_id, []).append(mins)

            injuries = []
            for inj_row in injury_rows:
                inj = inj_row.PlayerInjury
                pid = inj.player_id
                stats = stats_map.get(pid, {"ppg": 0.0, "spg": 0.0, "bpg": 0.0, "games": 0})
                mpg = np.mean(minutes_data.get(pid, [0.0])) if minutes_data.get(pid) else 0.0

                # Only count as "out" if status is Out or Doubtful
                # Questionable and Probable are tracked separately
                injuries.append(
                    InjuryInfo(
                        player_id=pid,
                        injury_status=inj.injury_status or "Out",
                        priority=int(inj_row.priority),
                        season_ppg=stats["ppg"],
                        season_mpg=mpg,
                        season_spg=stats["spg"],
                        season_bpg=stats["bpg"],
                    )
                )

            return injuries

        if session is not None:
            return await _query(session)
        else:
            async with self.session_factory() as sess:
                return await _query(sess)

    @staticmethod
    def _compute_from_injuries(injuries: list[InjuryInfo]) -> dict[str, float]:
        """Compute injury impact features from a list of injury records.

        Uses confirmation-priority dedup: for the same player, only the
        highest-priority (lowest number) injury record is used.
        """
        # Deduplicate: keep highest-priority record per player
        best_per_player: dict[int, InjuryInfo] = {}
        for inj in injuries:
            if inj.player_id not in best_per_player or inj.priority < best_per_player[inj.player_id].priority:
                best_per_player[inj.player_id] = inj

        # Only count players who are Out or Doubtful as "out"
        out_injuries = [
            inj
            for inj in best_per_player.values()
            if inj.injury_status.lower() in ("out", "doubtful")
        ]
        questionable_injuries = [
            inj
            for inj in best_per_player.values()
            if inj.injury_status.lower() == "questionable"
        ]

        # Firepower loss: Σ(ppg × mpg/48) for out/doubtful players
        firepower_loss = 0.0
        total_minutes_share_lost = 0.0
        key_player_out = 0.0
        defensive_anchor_out = 0.0

        for inj in out_injuries:
            # Minutes-weighted scoring loss
            minutes_share = inj.season_mpg / 48.0 if inj.season_mpg > 0 else 0.0
            weighted_loss = inj.season_ppg * minutes_share
            firepower_loss += weighted_loss
            total_minutes_share_lost += minutes_share

            # Key player: weighted loss > 10 (e.g., 15ppg × 36min/48 = 11.25)
            if weighted_loss > 10.0:
                key_player_out = 1.0

            # Defensive anchor: steals + blocks > 2.5 per game
            if inj.season_spg + inj.season_bpg > 2.5:
                defensive_anchor_out = 1.0

        features = {
            "injury_firepower_loss": firepower_loss,
            "total_minutes_share_lost": total_minutes_share_lost,
            "key_player_out": key_player_out,
            "defensive_anchor_out": defensive_anchor_out,
            "players_out_count": float(len(out_injuries)),
            "questionable_count": float(len(questionable_injuries)),
        }

        return features


def _parse_minutes(minutes_str: str | None) -> float | None:
    """Parse minutes from 'MM:SS' format or raw float.

    Returns float minutes or None if unparseable.
    """
    if minutes_str is None:
        return None
    if isinstance(minutes_str, (int, float)):
        return float(minutes_str)
    if ":" in str(minutes_str):
        try:
            parts = str(minutes_str).split(":")
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return None
    try:
        return float(minutes_str)
    except ValueError:
        return None
