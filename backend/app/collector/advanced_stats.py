"""Derive advanced stats from basic box score data.

The primary data source (Kaggle) ships no advanced-stats file, and nba_api's
BoxScoreAdvancedV2/V3 endpoints are geo-blocked (and the traditional box score
endpoints are deprecated as of the 2025-26 season). So instead of collecting
advanced stats, we derive them from the basic box score we already have.

Formulas follow Basketball-Reference definitions. Offensive rebounds are
estimated because we only store total rebounds (no ORB/DRB split):

  Team Possessions = FGA + 0.44*FTA - OReb + TOV     (OReb ≈ OREB_RATE * REB)
  Pace    = 48 * ((Poss_H + Poss_A) / 2) / game_minutes
  ORtg    = 100 * PTS / Poss
  DRtg    = 100 * Opp_PTS / Opp_Poss
  TS%     = PTS / (2 * (FGA + 0.44 * FTA))
  eFG%    = (FGM + 0.5 * FG3M) / FGA
  TOV%    = TOV / (FGA + 0.44 * FTA + TOV)
  AST%    = AST / FGM   (team-level: fraction of made FGs that were assisted)

The derived columns map 1:1 to the nullable advanced-stat columns on
`TeamGameStats` (pace, off_rtg, def_rtg, ts_pct, efg_pct, tov_pct, ast_pct).
"""

from __future__ import annotations

# League-average offensive rebound rate (fraction of a team's rebounds that are
# offensive). ~23-26% historically; 0.25 is the standard round approximation.
OREB_RATE = 0.25

# Regulation game length in minutes.
REGULATION_MINUTES = 48.0


def team_possessions(
    fga: float, fta: float, tov: float, reb: float, oreb: float | None = None
) -> float:
    """Estimated possessions for one team from its box score."""
    offensive_rebounds = OREB_RATE * reb if oreb is None else oreb
    return fga + 0.44 * fta - offensive_rebounds + tov


def ts_pct(pts: float, fga: float, fta: float) -> float:
    """True shooting percentage."""
    denom = 2.0 * (fga + 0.44 * fta)
    return pts / denom if denom > 0 else 0.0


def efg_pct(fgm: float, fg3m: float, fga: float) -> float:
    """Effective field goal percentage."""
    return (fgm + 0.5 * fg3m) / fga if fga > 0 else 0.0


def tov_pct(tov: float, fga: float, fta: float) -> float:
    """Turnover percentage (per estimated possession)."""
    denom = fga + 0.44 * fta + tov
    return tov / denom if denom > 0 else 0.0


def ast_pct(ast: float, fgm: float) -> float:
    """Team assist percentage — fraction of made FGs that were assisted."""
    return ast / fgm if fgm > 0 else 0.0


def _null_adv() -> dict[str, float | None]:
    """Advanced-stat dict with all values NULL (for incomplete box scores)."""
    return {k: None for k in ("pace", "off_rtg", "def_rtg", "ts_pct", "efg_pct", "tov_pct", "ast_pct")}


def derive_game_advanced_stats(
    home, away, game_minutes: float = REGULATION_MINUTES
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Compute advanced stats for both teams of a single game.

    Args:
        home: object with numeric attrs ``pts, fgm, fga, fg3m, fta, tov, reb, ast``.
        away: same shape for the away team.
        game_minutes: minutes of game time (48 regulation, higher for overtime).

    Returns:
        ``(home_dict, away_dict)`` keyed by advanced-stat column names. ``pace``
        is game-level (identical for both teams); the rest are per-team. If
        either team lacks shooting data (fga <= 0, e.g. historical
        summary-only rows), all values are None — the possessions formula is
        invalid without attempts, so we must not write garbage.
    """
    if home.fga <= 0 or away.fga <= 0:
        return _null_adv(), _null_adv()

    h_poss = team_possessions(home.fga, home.fta, home.tov, home.reb, getattr(home, "oreb", None))
    a_poss = team_possessions(away.fga, away.fta, away.tov, away.reb, getattr(away, "oreb", None))
    avg_poss = (h_poss + a_poss) / 2.0
    pace = 48.0 * avg_poss / game_minutes if game_minutes > 0 else 100.0

    def build(row, own_poss, opp_poss, opp_pts):
        return {
            "pace": round(pace, 2),
            "off_rtg": round(100.0 * row.pts / own_poss, 2) if own_poss > 0 else 0.0,
            "def_rtg": round(100.0 * opp_pts / opp_poss, 2) if opp_poss > 0 else 0.0,
            "ts_pct": round(ts_pct(row.pts, row.fga, row.fta), 4),
            "efg_pct": round(efg_pct(row.fgm, row.fg3m, row.fga), 4),
            "tov_pct": round(tov_pct(row.tov, row.fga, row.fta), 4),
            "ast_pct": round(ast_pct(row.ast, row.fgm), 4),
        }

    return build(home, h_poss, a_poss, away.pts), build(away, a_poss, h_poss, home.pts)
