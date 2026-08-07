"""Game context features.

Captures situational factors that affect game outcomes:
- Rest days (fatigue / preparation time)
- Back-to-back games (known disadvantage)
- Home/away performance splits
- Momentum (win/loss streaks)
"""

import logging
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import select, and_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

logger = logging.getLogger(__name__)

CONTEXT_FEATURE_NAMES = [
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_b2b",
    "away_b2b",
    "home_home_win_rate",
    "away_away_win_rate",
    "home_recent_home_win_rate",
    "away_recent_away_win_rate",
    "home_streak",
    "away_streak",
    "same_conference",
    "same_division",
    "season_series_games",
    "home_season_series_wins",
    # Travel & fatigue (v2)
    "travel_distance_category",
    "home_travel_fatigue",
    "away_travel_fatigue",
    "home_days_since_home",
]

# ── NBA Arena Coordinates (lat, lon) for travel distance ──

NBA_ARENA_COORDS: dict[str, tuple[float, float]] = {
    "ATL": (33.7573, -84.3963),   # State Farm Arena
    "BOS": (42.3662, -71.0621),   # TD Garden
    "BKN": (40.6826, -73.9754),   # Barclays Center
    "CHA": (35.2250, -80.8392),   # Spectrum Center
    "CHI": (41.8807, -87.6742),   # United Center
    "CLE": (41.4965, -81.6882),   # Rocket Mortgage FieldHouse
    "DAL": (32.7905, -96.8103),   # American Airlines Center
    "DEN": (39.7487, -105.0076),  # Ball Arena
    "DET": (42.3410, -83.0550),   # Little Caesars Arena
    "GSW": (37.7680, -122.3877),  # Chase Center
    "HOU": (29.7508, -95.3621),   # Toyota Center
    "IND": (39.7640, -86.1555),   # Gainbridge Fieldhouse
    "LAC": (34.0430, -118.2673),  # Intuit Dome (same city as LAL)
    "LAL": (34.0430, -118.2673),  # Crypto.com Arena
    "MEM": (35.1382, -90.0506),   # FedExForum
    "MIA": (25.7814, -80.1870),   # Kaseya Center
    "MIL": (43.0436, -87.9173),   # Fiserv Forum
    "MIN": (44.9795, -93.2762),   # Target Center
    "NOP": (29.9491, -90.0821),   # Smoothie King Center
    "NYK": (40.7505, -73.9934),   # Madison Square Garden
    "OKC": (35.4634, -97.5151),   # Paycom Center
    "ORL": (28.5392, -81.3839),   # Kia Center
    "PHI": (39.9014, -75.1720),   # Wells Fargo Center
    "PHX": (33.4458, -112.0712),  # Footprint Center
    "POR": (45.5316, -122.6668),  # Moda Center
    "SAC": (38.5802, -121.4997),  # Golden 1 Center
    "SAS": (29.4270, -98.4376),   # Frost Bank Center
    "TOR": (43.6435, -79.3791),   # Scotiabank Arena
    "UTA": (40.7683, -111.9011),  # Delta Center
    "WAS": (38.8981, -77.0208),   # Capital One Arena
}


class ContextFeatures:
    """Computes situational / context features for a matchup."""

    def __init__(self, session_factory=async_session_factory):
        self.session_factory = session_factory

    async def compute(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        """Compute all context features for a matchup.

        Args:
            home_team_id: Home team internal ID.
            away_team_id: Away team internal ID.
            game_date: Scheduled date of the game.
            session: Optional existing session.

        Returns:
            Dict mapping feature_name -> float.
        """
        async def _compute(sess: AsyncSession) -> dict[str, float]:
            # Rest days
            home_rest = await self._compute_rest_days(sess, home_team_id, game_date)
            away_rest = await self._compute_rest_days(sess, away_team_id, game_date)

            # Back-to-back
            home_b2b = 1.0 if home_rest == 0 else 0.0
            away_b2b = 1.0 if away_rest == 0 else 0.0

            # Home/Away win rates (season-long)
            home_home_wr = await self._compute_venue_win_rate(
                sess, home_team_id, game_date, is_home=True
            )
            away_away_wr = await self._compute_venue_win_rate(
                sess, away_team_id, game_date, is_home=False
            )

            # Recent home/away win rates (last 5 in that venue)
            home_recent_home_wr = await self._compute_venue_win_rate(
                sess, home_team_id, game_date, is_home=True, n_recent=5
            )
            away_recent_away_wr = await self._compute_venue_win_rate(
                sess, away_team_id, game_date, is_home=False, n_recent=5
            )

            # Streaks
            home_streak = await self._compute_streak(sess, home_team_id, game_date)
            away_streak = await self._compute_streak(sess, away_team_id, game_date)

            # Conference / Division matchup
            conf_div = await self._compute_conference_division(
                sess, home_team_id, away_team_id
            )

            # Season series (games between these teams this season)
            series = await self._compute_season_series(
                sess, home_team_id, away_team_id, game_date
            )

            # Travel & fatigue
            travel_dist = await self._compute_travel_distance(
                sess, home_team_id, away_team_id
            )
            home_fatigue = await self._compute_travel_fatigue(
                sess, home_team_id, game_date
            )
            away_fatigue = await self._compute_travel_fatigue(
                sess, away_team_id, game_date
            )
            home_days_since_home = await self._compute_days_since_home(
                sess, home_team_id, game_date
            )

            return {
                "home_rest_days": float(home_rest),
                "away_rest_days": float(away_rest),
                "rest_days_diff": float(home_rest - away_rest),
                "home_b2b": home_b2b,
                "away_b2b": away_b2b,
                "home_home_win_rate": home_home_wr,
                "away_away_win_rate": away_away_wr,
                "home_recent_home_win_rate": home_recent_home_wr,
                "away_recent_away_win_rate": away_recent_away_wr,
                "home_streak": home_streak,
                "away_streak": away_streak,
                "same_conference": conf_div["same_conference"],
                "same_division": conf_div["same_division"],
                "season_series_games": series["season_series_games"],
                "home_season_series_wins": series["home_season_series_wins"],
                "travel_distance_category": travel_dist,
                "home_travel_fatigue": home_fatigue,
                "away_travel_fatigue": away_fatigue,
                "home_days_since_home": home_days_since_home,
            }

        if session is not None:
            return await _compute(session)
        else:
            async with self.session_factory() as sess:
                return await _compute(sess)

    @staticmethod
    async def _compute_rest_days(
        session: AsyncSession, team_id: int, before_date: datetime
    ) -> int:
        """Compute days since the team's last game."""
        from app.models.game import Game

        query = (
            select(Game.game_date)
            .where(
                and_(
                    Game.status == "Final",
                    Game.game_date < before_date,
                    (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                )
            )
            .order_by(desc(Game.game_date))
            .limit(1)
        )
        result = await session.execute(query)
        last_game_date = result.scalar_one_or_none()

        if last_game_date is None:
            # No previous games this season — assume well-rested
            return 7

        # Handle timezone-aware vs naive datetime: make both naive
        bd = before_date
        if hasattr(bd, 'tzinfo') and bd.tzinfo is not None:
            bd = bd.replace(tzinfo=None)
        lg_date = last_game_date
        if hasattr(lg_date, 'tzinfo') and lg_date.tzinfo is not None:
            lg_date = lg_date.replace(tzinfo=None)

        delta = (bd - lg_date).days
        return max(0, delta)

    @staticmethod
    async def _compute_venue_win_rate(
        session: AsyncSession,
        team_id: int,
        before_date: datetime,
        is_home: bool,
        n_recent: int | None = None,
    ) -> float:
        """Compute win rate for a team in home or away games."""
        from app.models.game import Game

        if is_home:
            condition = Game.home_team_id == team_id
        else:
            condition = Game.away_team_id == team_id

        query = (
            select(Game)
            .where(
                and_(
                    condition,
                    Game.status == "Final",
                    Game.game_date < before_date,
                )
            )
            .order_by(desc(Game.game_date))
        )

        if n_recent is not None:
            query = query.limit(n_recent)

        result = await session.execute(query)
        games = result.scalars().all()

        if not games:
            return 0.5  # Neutral prior when no data

        wins = 0
        for g in games:
            if is_home:
                if (g.home_score or 0) > (g.away_score or 0):
                    wins += 1
            else:
                if (g.away_score or 0) > (g.home_score or 0):
                    wins += 1

        return wins / len(games)

    @staticmethod
    async def _compute_streak(
        session: AsyncSession, team_id: int, before_date: datetime
    ) -> float:
        """Compute current win/loss streak for a team.

        Positive = win streak, negative = losing streak.
        e.g., +3 means 3 consecutive wins, -2 means 2 consecutive losses.
        """
        from app.models.game import Game

        query = (
            select(Game)
            .where(
                and_(
                    Game.status == "Final",
                    Game.game_date < before_date,
                    (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                )
            )
            .order_by(desc(Game.game_date))
            .limit(20)  # Check up to last 20 games
        )
        result = await session.execute(query)
        games = result.scalars().all()

        if not games:
            return 0.0

        streak = 0
        for g in games:
            is_win = False
            if g.home_team_id == team_id:
                is_win = (g.home_score or 0) > (g.away_score or 0)
            else:
                is_win = (g.away_score or 0) > (g.home_score or 0)

            if streak == 0:
                streak = 1 if is_win else -1
            elif (streak > 0 and is_win) or (streak < 0 and not is_win):
                streak += 1 if is_win else -1
            else:
                break  # Streak ended

        return float(streak)

    @staticmethod
    async def _compute_travel_distance(
        session: AsyncSession, home_team_id: int, away_team_id: int
    ) -> float:
        """Compute travel distance category for the away team.

        Categories:
          0 = same city / arena (LAL vs LAC, NYK vs BKN)
          1 = same conference, short travel (< 500 mi)
          2 = cross-conference, medium travel (500-1500 mi)
          3 = coast-to-coast, long travel (> 1500 mi)
        """
        from app.models.team import Team

        query = select(Team).where(Team.id.in_([home_team_id, away_team_id]))
        result = await session.execute(query)
        teams = {t.id: t for t in result.scalars().all()}

        home_team = teams.get(home_team_id)
        away_team = teams.get(away_team_id)

        if not home_team or not away_team:
            return 1.0  # Default: moderate travel

        home_abbr = home_team.abbreviation or ""
        away_abbr = away_team.abbreviation or ""

        home_coords = NBA_ARENA_COORDS.get(home_abbr.upper())
        away_coords = NBA_ARENA_COORDS.get(away_abbr.upper())

        if not home_coords or not away_coords:
            return 1.0

        # Haversine distance in miles
        import math
        lat1, lon1 = math.radians(home_coords[0]), math.radians(home_coords[1])
        lat2, lon2 = math.radians(away_coords[0]), math.radians(away_coords[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        miles = 3959 * c  # Earth radius in miles

        if miles < 50:
            return 0.0  # Same city
        elif miles < 500:
            return 1.0  # Short travel
        elif miles < 1500:
            return 2.0  # Medium travel
        else:
            return 3.0  # Coast-to-coast

    @staticmethod
    async def _compute_travel_fatigue(
        session: AsyncSession, team_id: int, before_date: datetime
    ) -> float:
        """Check if team is on a road trip (consecutive away games).

        Returns the number of consecutive road games (capped at 5).
        Home games or rest reset the counter.
        """
        from app.models.game import Game

        query = (
            select(Game)
            .where(
                and_(
                    Game.status == "Final",
                    Game.game_date < before_date,
                    (Game.home_team_id == team_id) | (Game.away_team_id == team_id),
                )
            )
            .order_by(desc(Game.game_date))
            .limit(5)
        )
        result = await session.execute(query)
        games = result.scalars().all()

        fatigue = 0.0
        for g in games:
            if g.away_team_id == team_id:
                fatigue += 1.0
            else:
                break  # Home game breaks the road trip

        return fatigue

    @staticmethod
    async def _compute_days_since_home(
        session: AsyncSession, team_id: int, before_date: datetime
    ) -> float:
        """Days since the team last played a home game.

        High values = extended road trip effect (jet lag, fatigue).
        """
        from app.models.game import Game

        query = (
            select(Game.game_date)
            .where(
                and_(
                    Game.status == "Final",
                    Game.game_date < before_date,
                    Game.home_team_id == team_id,
                )
            )
            .order_by(desc(Game.game_date))
            .limit(1)
        )
        result = await session.execute(query)
        last_home_date = result.scalar_one_or_none()

        if last_home_date is None:
            return 14.0  # Season start — assume fresh

        bd = before_date
        if hasattr(bd, 'tzinfo') and bd.tzinfo is not None:
            bd = bd.replace(tzinfo=None)
        lh = last_home_date
        if hasattr(lh, 'tzinfo') and lh.tzinfo is not None:
            lh = lh.replace(tzinfo=None)

        return float(max(0, (bd - lh).days))

    @staticmethod
    async def _compute_conference_division(
        session: AsyncSession, home_team_id: int, away_team_id: int
    ) -> dict[str, float]:
        """Check if teams are in the same conference or division."""
        from app.models.team import Team

        query = select(Team).where(Team.id.in_([home_team_id, away_team_id]))
        result = await session.execute(query)
        teams = {t.id: t for t in result.scalars().all()}

        home = teams.get(home_team_id)
        away = teams.get(away_team_id)

        same_conf = 0.0
        same_div = 0.0
        if home and away and home.conference and away.conference:
            same_conf = 1.0 if home.conference == away.conference else 0.0
            same_div = 1.0 if home.division == away.division else 0.0

        return {"same_conference": same_conf, "same_division": same_div}

    @staticmethod
    async def _compute_season_series(
        session: AsyncSession,
        home_team_id: int,
        away_team_id: int,
        before_date: datetime,
    ) -> dict[str, float]:
        """Count prior meetings this season and home team wins in the series."""
        from app.models.game import Game
        from sqlalchemy import or_

        # Determine current season (use date() to avoid timezone issues in query)
        import datetime as dt
        if before_date.month >= 10:
            season_start = dt.datetime(before_date.year, 10, 1)
        else:
            season_start = dt.datetime(before_date.year - 1, 10, 1)

        query = (
            select(Game)
            .where(
                and_(
                    Game.status == "Final",
                    Game.game_date < before_date,
                    Game.game_date >= season_start,
                    or_(
                        and_(
                            Game.home_team_id == home_team_id,
                            Game.away_team_id == away_team_id,
                        ),
                        and_(
                            Game.home_team_id == away_team_id,
                            Game.away_team_id == home_team_id,
                        ),
                    ),
                )
            )
        )
        result = await session.execute(query)
        games = result.scalars().all()

        season_series_games = float(len(games))
        home_season_series_wins = 0.0
        for g in games:
            if g.home_team_id == home_team_id:
                if (g.home_score or 0) > (g.away_score or 0):
                    home_season_series_wins += 1.0
            else:
                # Home team was away in this past game
                if (g.away_score or 0) > (g.home_score or 0):
                    home_season_series_wins += 1.0

        return {
            "season_series_games": season_series_games,
            "home_season_series_wins": home_season_series_wins,
        }
