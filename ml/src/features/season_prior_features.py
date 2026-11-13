"""Bayesian early-season priors for offense, defense and pace."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ml.src.database import async_session_factory

SEASON_PRIOR_FEATURE_NAMES = [
    "prior_off_rtg",
    "prior_def_rtg",
    "prior_pace",
    "bayes_off_rtg",
    "bayes_def_rtg",
    "bayes_pace",
    "season_prior_missing",
]


def _season_for(day: datetime) -> str:
    year = day.year if day.month >= 10 else day.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def _previous(season: str) -> str:
    year = int(season[:4]) - 1
    return f"{year}-{str(year + 1)[-2:]}"


def shrink_prior(team_value: float | None, team_games: int, league_value: float, strength: int = 20) -> float:
    if team_value is None or team_games <= 0:
        return float(league_value)
    return float((team_games * team_value + strength * league_value) / (team_games + strength))


def blend_current(current_value: float | None, current_games: int, prior: float, strength: int = 10) -> float:
    if current_value is None or current_games <= 0:
        return float(prior)
    return float((current_games * current_value + strength * prior) / (current_games + strength))


class SeasonPriorFeatures:
    def __init__(self, session_factory=async_session_factory, current_strength: int = 10):
        self.session_factory = session_factory
        self.current_strength = current_strength

    async def compute(
        self,
        team_id: int,
        before_date: datetime,
        session: AsyncSession | None = None,
    ) -> dict[str, float]:
        from app.models.game import Game
        from app.models.team_game_stats import TeamGameStats

        async def _query(sess: AsyncSession) -> dict[str, float]:
            current = _season_for(before_date)
            previous = _previous(current)

            async def rows(season: str, only_team: int | None, cutoff: datetime | None):
                clauses = [
                    Game.status == "Final",
                    Game.team_stats_complete.is_(True),
                    Game.season == season,
                    Game.game_type == "regular",
                    TeamGameStats.off_rtg.isnot(None),
                    TeamGameStats.def_rtg.isnot(None),
                    TeamGameStats.pace.isnot(None),
                ]
                if only_team is not None:
                    clauses.append(TeamGameStats.team_id == only_team)
                if cutoff is not None:
                    clauses.append(Game.game_date < cutoff)
                result = await sess.execute(
                    select(
                        TeamGameStats.off_rtg,
                        TeamGameStats.def_rtg,
                        TeamGameStats.pace,
                        TeamGameStats.game_minutes,
                    )
                    .join(Game, TeamGameStats.game_id == Game.id)
                    .where(and_(*clauses))
                )
                return result.all()

            previous_team = await rows(previous, team_id, None)
            previous_league = await rows(previous, None, None)
            current_team = await rows(current, team_id, before_date)

            league_defaults = (112.0, 112.0, 99.0)

            def means(values, fallback):
                if not values:
                    return fallback
                # Weight efficiencies by estimated possessions.  Pace is
                # already normalized to 48 minutes, so pace * game_minutes / 48
                # recovers the possession exposure for regulation and OT.
                arr = np.asarray([row[:3] for row in values], dtype=np.float64)
                exposure = np.asarray([
                    max(float(row[2]) * float(row[3] or 48.0) / 48.0, 1.0)
                    for row in values
                ], dtype=np.float64)
                return tuple(float(v) for v in np.average(arr, axis=0, weights=exposure))

            league = means(previous_league, league_defaults)
            previous_mean = means(previous_team, (None, None, None))
            current_mean = means(current_team, (None, None, None))
            prior = tuple(
                shrink_prior(previous_mean[i], len(previous_team), league[i])
                for i in range(3)
            )
            bayes = tuple(
                blend_current(current_mean[i], len(current_team), prior[i], self.current_strength)
                for i in range(3)
            )
            return {
                "prior_off_rtg": prior[0],
                "prior_def_rtg": prior[1],
                "prior_pace": prior[2],
                "bayes_off_rtg": bayes[0],
                "bayes_def_rtg": bayes[1],
                "bayes_pace": bayes[2],
                "season_prior_missing": 1.0 if not previous_team else 0.0,
            }

        if session is not None:
            return await _query(session)
        async with self.session_factory() as sess:
            return await _query(sess)
