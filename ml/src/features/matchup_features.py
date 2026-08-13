"""Matchup interaction features (10 features) — derived from home/away team stats.

These features capture how the two teams' playing styles interact:
- Pace differential: Which team dictates the game tempo?
- Offense-vs-defense: Can the home offense score on the away defense? (and vice versa)
- Expected game total: Based on pace and offensive efficiency matchup.

All features are computed from already-calculated team feature dicts,
so they can be added after team features without additional DB queries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MATCHUP_FEATURE_NAMES = [
    # Pace & tempo
    "pace_diff",                # home pace - away pace (positive = home wants faster game)
    "expected_pace",            # (home_pace + away_pace) / 2 — estimated game tempo

    # Efficiency matchup: home offense vs away defense
    "home_off_vs_away_def",     # home_off_rtg - away_def_rtg (positive = home O has advantage)
    "away_off_vs_home_def",     # away_off_rtg - home_def_rtg (positive = away O has advantage)

    # Net rating & overall quality
    "net_rtg_diff",             # home_net_rtg - away_net_rtg (positive = home better overall)
    "total_rating_diff",        # (home_off + home_def) - (away_off + away_def) — total team strength

    # Shooting & efficiency differentials
    # NOTE: efg_diff removed — redundant with ts_diff (r=0.961, TS includes FTs)
    "ts_diff",                  # home_ts - away_ts (true shooting gap)
    "tov_pct_diff",             # away_tov - home_tov (positive = away turns over more)
    "ftr_diff",                 # home_ftr - away_ftr (free throw rate — aggressiveness gap)

    # Expected scoring
    "expected_total",           # theoretical total = expected_pace * (home_off + away_off) / 200

    # Total points (over/under) specific — Task A1
    "pace_product",             # home_pace * away_pace / 10000 — multiplicative tempo interaction
    "combined_over_tendency",   # (home_over_ratio + away_over_ratio) / 2 — both teams' over/under tendency
]


def compute_matchup_features(
    home_team_f: dict[str, float],
    away_team_f: dict[str, float],
) -> dict[str, float]:
    """Compute matchup interaction features from home and away team stat dicts.

    Args:
        home_team_f: Feature dict from TeamFeatures.compute() for the home team.
        away_team_f: Feature dict from TeamFeatures.compute() for the away team.

    Returns:
        Dict mapping MATCHUP_FEATURE_NAMES → float values.
    """
    home_pace = home_team_f.get("avg_pace", 0.0)
    away_pace = away_team_f.get("avg_pace", 0.0)

    home_off_rtg = home_team_f.get("avg_off_rtg", 0.0)
    home_def_rtg = home_team_f.get("avg_def_rtg", 0.0)
    away_off_rtg = away_team_f.get("avg_off_rtg", 0.0)
    away_def_rtg = away_team_f.get("avg_def_rtg", 0.0)

    home_net_rtg = home_team_f.get("avg_net_rtg", 0.0)
    away_net_rtg = away_team_f.get("avg_net_rtg", 0.0)

    home_ts = home_team_f.get("avg_ts_pct", 0.0)
    away_ts = away_team_f.get("avg_ts_pct", 0.0)

    home_tov = home_team_f.get("avg_tov_pct", 0.0)
    away_tov = away_team_f.get("avg_tov_pct", 0.0)

    home_ftr = home_team_f.get("avg_ftr", 0.0)
    away_ftr = away_team_f.get("avg_ftr", 0.0)

    # Pace differential
    pace_diff = home_pace - away_pace
    expected_pace = (home_pace + away_pace) / 2.0

    # Efficiency matchups
    home_off_vs_away_def = home_off_rtg - away_def_rtg
    away_off_vs_home_def = away_off_rtg - home_def_rtg

    # Net rating differentials
    net_rtg_diff = home_net_rtg - away_net_rtg

    # Total team strength difference
    # NOTE: This is mathematically equivalent to the old off_def_composite:
    #   (home_off - away_def) - (away_off - home_def) = (home_off + home_def) - (away_off + away_def)
    total_rating_diff = (home_off_rtg + home_def_rtg) - (away_off_rtg + away_def_rtg)

    # Shooting differentials
    ts_diff = home_ts - away_ts
    # NOTE: efg_diff removed — r=0.961 vs ts_diff (TS includes FTs, more comprehensive)
    tov_pct_diff = away_tov - home_tov  # positive = away turns over MORE (good for home)
    ftr_diff = home_ftr - away_ftr      # positive = home draws more fouls

    # Expected total points
    # Using the formula: Total ≈ Pace × (Off_Rtg_Home + Off_Rtg_Away) / 200
    # where Pace = possessions per 48 min, ORtg = points per 100 possessions
    # /200 because: (pts/100poss × poss/48min) / 2 → simplify to one number
    if expected_pace > 0:
        expected_total = expected_pace * (home_off_rtg + away_off_rtg) / 200.0
    else:
        expected_total = 0.0

    # ── Total-specific matchup features (Task A1) ──
    # Multiplicative pace interaction: two fast teams explode the total more
    # than a linear sum predicts. Normalize by 10000 (typical pace ~100²) → ~1.0.
    pace_product = (home_pace * away_pace) / 10000.0 if home_pace > 0 and away_pace > 0 else 0.0

    # Combined over/under tendency: both teams' recent over-league-average ratio.
    home_over_ratio = home_team_f.get("over_league_avg_ratio", 0.0)
    away_over_ratio = away_team_f.get("over_league_avg_ratio", 0.0)
    combined_over_tendency = (home_over_ratio + away_over_ratio) / 2.0

    features = {
        "pace_diff": pace_diff,
        "expected_pace": expected_pace,
        "home_off_vs_away_def": home_off_vs_away_def,
        "away_off_vs_home_def": away_off_vs_home_def,
        "net_rtg_diff": net_rtg_diff,
        "total_rating_diff": total_rating_diff,
        # NOTE: efg_diff removed — r=0.961 vs ts_diff
        "ts_diff": ts_diff,
        "tov_pct_diff": tov_pct_diff,
        "ftr_diff": ftr_diff,
        "expected_total": expected_total,
        "pace_product": pace_product,
        "combined_over_tendency": combined_over_tendency,
    }

    # Replace NaN/Inf (can happen when advanced stats are NULL for early season)
    for k in features:
        v = features[k]
        if v != v or v == float('inf') or v == float('-inf'):  # NaN or Inf check
            features[k] = 0.0

    return features
