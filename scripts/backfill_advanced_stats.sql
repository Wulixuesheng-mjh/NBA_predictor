-- Backfill advanced stats into team_game_stats from basic box score data.
--
-- Derives pace, off_rtg, def_rtg, ts_pct, efg_pct, tov_pct, ast_pct from the
-- basic box score columns we already store. Neither the Kaggle dataset nor
-- nba_collector provides advanced stats, so we compute them (see
-- backend/app/collector/advanced_stats.py for the matching Python formulas).
--
--   Team Possessions = FGA + 0.44*FTA - OReb + TOV     (OReb ≈ 0.25 * REB)
--   pace    = (Poss_H + Poss_A) / 2                     (regulation, no OT scaling)
--   ORtg    = 100 * PTS / Poss
--   DRtg    = 100 * Opp_PTS / Opp_Poss
--   TS%     = PTS / (2 * (FGA + 0.44 * FTA))
--   eFG%    = (FGM + 0.5 * FG3M) / FGA
--   TOV%    = TOV / (FGA + 0.44 * FTA + TOV)
--   AST%    = AST / FGM
--
-- pace here is game-level (identical for both teams of a game) and uses the
-- 48-minute regulation assumption — overtime games get pace slightly
-- overestimated, which is negligible for model features (off_rtg/def_rtg and
-- the shooting percentages are exact regardless of overtime).
--
-- Run:
--   docker compose exec -T postgres psql -U nba_admin -d nba_predictor \
--     -v ON_ERROR_STOP=1 < scripts/backfill_advanced_stats.sql

WITH t AS (
    SELECT
        t1.id,
        t1.pts, t1.fgm, t1.fga, t1.fg3m, t1.fta, t1.tov, t1.reb, t1.ast,
        t2.pts AS opp_pts,
        t2.fga AS opp_fga,
        t1.fga + 0.44 * t1.fta - 0.25 * t1.reb + t1.tov AS poss,
        t2.fga + 0.44 * t2.fta - 0.25 * t2.reb + t2.tov AS opp_poss
    FROM team_game_stats t1
    JOIN team_game_stats t2
      ON t1.game_id = t2.game_id AND t1.team_id <> t2.team_id
)
UPDATE team_game_stats tgs
SET
    ts_pct   = ROUND(t.pts / NULLIF(2.0 * (t.fga + 0.44 * t.fta), 0), 4),
    efg_pct  = ROUND((t.fgm + 0.5 * t.fg3m) / NULLIF(t.fga, 0), 4),
    tov_pct  = ROUND(t.tov / NULLIF(t.fga + 0.44 * t.fta + t.tov, 0), 4),
    ast_pct  = ROUND(t.ast / NULLIF(t.fgm, 0), 4),
    pace     = ROUND((t.poss + t.opp_poss) / 2.0, 2),
    off_rtg  = ROUND(100.0 * t.pts / NULLIF(t.poss, 0), 2),
    def_rtg  = ROUND(100.0 * t.opp_pts / NULLIF(t.opp_poss, 0), 2)
FROM t
WHERE tgs.id = t.id
  -- Only rows with a complete box score: the possessions formula is invalid
  -- (negative) for historical summary-only rows where fga/fta/tov are 0.
  AND t.fga > 0
  AND t.opp_fga > 0;
