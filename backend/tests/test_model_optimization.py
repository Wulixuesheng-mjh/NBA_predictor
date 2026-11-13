"""Focused tests for the model-optimization invariants."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app.schemas.prediction import PredictionRequest
from ml.src.evaluation import complete_day_split
from ml.src.features.injury_net_features import RotationPlayer, project_rotation, summarize_rotation
from ml.src.features.matchup_features import compute_matchup_features
from ml.src.features.roster_continuity_features import RosterContinuityFeatures
from ml.src.features.season_prior_features import blend_current, shrink_prior
from ml.src.training_policy import policy_mask_and_weights
from ml.src.uncertainty import prediction_interval, threshold_probabilities


def test_dnp_decimal_zero_does_not_dilute_player_production():
    from types import SimpleNamespace
    from ml.src.features.player_features import _aggregate_played_rows
    rows=[SimpleNamespace(player_id=1,minutes=m,pts=p,stl=s,blk=b) for m,p,s,b in (
        ('36.000000',20,2,1),('0.000000',0,0,0),('0:00',0,0,0),(None,0,0,0))]
    values,minutes=_aggregate_played_rows(rows)
    assert values[1]==dict(ppg=20,spg=2,bpg=1,games=1)
    assert minutes[1]==[36]


@pytest.mark.asyncio
async def test_identity_resolution_keeps_raw_report_and_unknown_submission(db_session):
    from types import SimpleNamespace
    from app.models.player import Player
    from app.services.injury_snapshot_reader import resolve_known_aliases
    player=Player(nba_player_id=1630231,full_name='Kenyon Martin Jr.')
    db_session.add(player)
    await db_session.flush()
    raw=[dict(player_id=None,player_name='Martin, KJ',status='Out')]
    snapshot=SimpleNamespace(entries_json=raw,is_complete=False,
        parse_issues=['unmatched_player:Martin, KJ'])
    resolved=await resolve_known_aliases(db_session,snapshot)
    assert resolved.is_complete and resolved.entries_json[0]['player_id']==player.id
    assert raw[0]['player_id'] is None and not resolved.raw_is_complete
    partial=SimpleNamespace(entries_json=raw,is_complete=False,
        parse_issues=['unmatched_player:Martin, KJ','not_yet_submitted'])
    assert not (await resolve_known_aliases(db_session,partial)).is_complete


def test_model_status_uses_published_release_instead_of_newest_candidate(tmp_path):
    from app.services.model_release_service import read_active_release
    (tmp_path/'active.json').write_text('{"version":"production-v1"}',encoding='utf-8')
    candidate = tmp_path/'xgboost/999999_candidate'
    candidate.mkdir(parents=True)
    (candidate/'win_model.json').write_text('{}')
    assert read_active_release(tmp_path)['version'] == 'production-v1'
    (tmp_path/'active.json').unlink()
    assert read_active_release(tmp_path) is None


@pytest.mark.parametrize("game_id,expected", [
    ("0012500001", "preseason"), ("12500001", "preseason"),
    ("0022500001", "regular"), ("22500001", "regular"),
    ("0042500001", "playoffs"), ("0052500001", "play_in"),
    ("0032500001", "exhibition"), ("0062500001", "cup_final"),
    ("bdl_12345", "unknown"),
])
def test_game_id_classification_handles_removed_leading_zeroes(game_id, expected):
    from app.collector.nba_collector import NBACollector
    from app.collector.kaggle_collector import KaggleCollector

    assert NBACollector._game_type(game_id) == expected
    assert KaggleCollector._game_type(game_id) == expected


def test_time_weight_is_mean_one_and_favors_recent_games():
    cutoff = datetime(2026, 6, 1, tzinfo=timezone.utc)
    rows = [
        {"game_date": (cutoff - timedelta(days=days)).isoformat()}
        for days in (730, 365, 0)
    ]
    mask, weights, report = policy_mask_and_weights(rows, "T4", cutoff=cutoff)
    assert mask.all()
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] < weights[1] < weights[2]
    assert report["effective_samples"] < 3


def test_complete_day_split_never_breaks_a_game_day():
    rows = [
        {"game_date": f"2026-01-{day:02d}T01:00:00+00:00"}
        for day in (1, 1, 1, 2, 2, 3, 3, 3, 4, 4)
    ]
    index = complete_day_split(rows, 0.25)
    assert index == 5


def test_expected_total_has_correct_points_per_100_scale():
    features = compute_matchup_features(
        {"avg_pace": 100, "avg_off_rtg": 112, "avg_def_rtg": 109},
        {"avg_pace": 100, "avg_off_rtg": 108, "avg_def_rtg": 111},
    )
    assert features["expected_total"] == pytest.approx(220.0)


def test_unknown_roster_is_not_encoded_as_full_continuity():
    features = RosterContinuityFeatures._compute_continuity({}, {1: 1000})
    assert features["roster_data_missing"] == 1.0
    assert features["roster_continuity"] == 0.5


def test_season_prior_smooths_zero_one_and_ten_games():
    prior = shrink_prior(115.0, 82, 112.0)
    zero = blend_current(None, 0, prior, 10)
    one = blend_current(125.0, 1, prior, 10)
    ten = blend_current(125.0, 10, prior, 10)
    assert zero == pytest.approx(prior)
    assert prior < one < ten < 125.0


def test_rotation_out_is_zero_and_team_minutes_sum_to_240():
    players = [
        RotationPlayer(i, "G", 30.0, 30.0, 0.0 if i == 0 else 1.0, 0.5, 0.05)
        for i in range(7)
    ] + [RotationPlayer(7, "C", 0.0, 20.0, 1.0, 0.4, 0.04)]
    projected = project_rotation(players)
    assert projected is not None
    assert projected[0] == 0.0
    assert sum(projected.values()) == pytest.approx(240.0, abs=1e-6)
    summary = summarize_rotation(players, projected)
    assert summary["injury_replacement_minutes"] > 0
    assert summary["injury_newly_lost_minutes"] == pytest.approx(30.0)


def test_long_term_absence_is_not_deducted_twice_and_return_can_be_positive():
    absent = RotationPlayer(1, "F", 0.0, 30.0, 0.0, 0.7, 0.04)
    healthy = [RotationPlayer(i, "G", 40.0, 40.0, 1.0, 0.4, 0.03) for i in range(2, 8)]
    projected = project_rotation([absent, *healthy])
    assert projected is not None
    assert summarize_rotation([absent, *healthy], projected)["injury_net_offense_delta"] == pytest.approx(0.0)

    returning = RotationPlayer(1, "F", 0.0, 30.0, 1.0, 0.8, 0.04)
    projected = project_rotation([returning, *healthy])
    assert projected is not None
    assert summarize_rotation([returning, *healthy], projected)["injury_net_offense_delta"] > 0


def test_market_probabilities_are_monotonic_and_integer_lines_have_pushes():
    residuals = [float(value) for value in range(-12, 13)] * 10
    easy = threshold_probabilities(7.0, 3.5, residuals, integer_outcome=True, fallback_sigma=12)
    hard = threshold_probabilities(7.0, 8.5, residuals, integer_outcome=True, fallback_sigma=12)
    integer = threshold_probabilities(7.0, 7.0, residuals, integer_outcome=True, fallback_sigma=12)
    assert easy.win > hard.win
    assert integer.push > 0
    assert prediction_interval(220.0, residuals, 0.8) is not None


def test_new_and_legacy_handicap_fields_are_mutually_exclusive():
    with pytest.raises(ValueError):
        PredictionRequest(
            home_team_id=1,
            away_team_id=2,
            spread_line=6.5,
            market_home_handicap=-6.5,
        )
