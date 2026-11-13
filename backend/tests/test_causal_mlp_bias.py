import importlib.util
from pathlib import Path
import copy
import pytest


@pytest.fixture
def module():
    path=Path('/app/scripts/study_causal_mlp_bias.py')
    if not path.exists():path=Path(__file__).resolve().parents[2]/'scripts/study_causal_mlp_bias.py'
    spec=importlib.util.spec_from_file_location('causal_bias_test',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def rows(count):
    return [dict(nba_game_id=str(i),fold='2024-01',source_game_date='2024-01-10',prediction=220.,actual_total=225.) for i in range(count)]+[
        dict(nba_game_id='target',fold='2024-02',source_game_date='2024-02-01',prediction=220.,actual_total=300.),
        dict(nba_game_id='future',fold='2024-03',source_game_date='2024-03-01',prediction=220.,actual_total=999.)]


def test_bias_uses_past_uncorrected_errors_and_not_target_or_future_labels(module):
    original=rows(200);a=module.apply_past_bias(original)
    changed=copy.deepcopy(original);changed[-2]['actual_total']=-999;changed[-1]['actual_total']=-999
    b=module.apply_past_bias(changed)
    assert a[-2]['prediction']==b[-2]['prediction']==225.
    assert a[-2]['calibration']['prior_count']==200
    assert original[-2]['prediction']==220.,'Source forecasts must remain immutable'


def test_bias_warmup_keeps_unknown_calibration_disabled(module):
    result=module.apply_past_bias(rows(199))
    assert result[-2]['prediction']==220 and not result[-2]['calibration']['enabled']


def test_bias_respects_lag_and_rolling_capacity(module):
    original=rows(1005)
    original.append(dict(nba_game_id='too_recent',fold='2024-01',source_game_date='2024-01-30',prediction=0.,actual_total=10000.))
    target=next(r for r in module.apply_past_bias(original) if r['nba_game_id']=='target')
    assert target['prediction']==225 and target['calibration']['prior_count']==1000
    assert target['calibration']['latest_prior_day']=='2024-01-10'
