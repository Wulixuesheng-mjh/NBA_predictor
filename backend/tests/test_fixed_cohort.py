import importlib.util
from pathlib import Path
import pytest


def test_fixed_cohort_rejects_changed_labels_or_days_and_preserves_slices():
    path=Path('/app/scripts/run_rolling_evaluation.py')
    if not path.exists():
        path=Path(__file__).resolve().parents[2]/'scripts/run_rolling_evaluation.py'
    spec=importlib.util.spec_from_file_location('fixed_cohort',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    original=dict(nba_game_id='one',fold='2024-04',actual=[1.,5.,225.],
                  game_date='2024-04-01T23:00:00+00:00',major_injury=True,early_season=False)
    row=original|dict(major_injury=False,early_season=True,prediction=[.6,4.,224.])
    m.freeze_cohort_metadata(row,'2024-04',{'one':original})
    assert row['major_injury'] and not row['early_season'] and row['prediction'][2]==224
    assert original['major_injury'] is True
    with pytest.raises(AssertionError):
        m.freeze_cohort_metadata(row|dict(actual=[1.,5.,226.]),'2024-04',{'one':original})
    with pytest.raises(AssertionError):
        m.freeze_cohort_metadata(row|dict(game_date='2024-04-02T01:00:00+00:00'),'2024-04',{'one':original})
