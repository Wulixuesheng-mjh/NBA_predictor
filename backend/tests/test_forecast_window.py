from datetime import datetime
import numpy as np
import pytest
from ml.src.forecast_window import forecast_mask_and_weights
from ml.src.training_policy import policy_mask_and_weights


def rows():
    return [dict(source_game_date=f'{year+1}-06-01',game_date=f'{year+1}-06-01',season=f'{year}-{str(year+1)[-2:]}') for year in (2020,2021,2022)]


def test_opener_keeps_two_completed_seasons_not_three():
    values=rows();old,_,_=policy_mask_and_weights(values,'T1')
    new,w,report=forecast_mask_and_weights(values,'T1',forecast_day='2023-10-24',forecast_season='2023-24',cutoff_day='2023-10-23')
    assert old.tolist()==[True,True,True] and new.tolist()==[False,True,True]
    assert w.tolist()==[1.,1.] and report['fit_until']=='2023-06-01'
    assert report['information_cutoff_day']=='2023-10-23'


def test_target_season_is_explicit_even_if_cutoff_is_september():
    mask,_,_=forecast_mask_and_weights(rows(),'T1',forecast_day='2023-10-01',forecast_season='2023-24',cutoff_day='2023-09-30')
    assert mask.tolist()==[False,True,True]


def test_windows_use_recorded_season_not_calendar_guess_for_delayed_season():
    values=[dict(source_game_date='2020-10-10',season='2019-20'),dict(source_game_date='2021-06-01',season='2020-21')]
    mask,_,_=forecast_mask_and_weights(values,'T1',forecast_day='2022-10-20',forecast_season='2022-23',cutoff_day='2022-10-19')
    assert mask.tolist()==[False,True]


def test_date_weights_have_mean_one_and_no_floor():
    _,w,report=forecast_mask_and_weights(rows(),'T4',forecast_day='2023-10-24',forecast_season='2023-24',cutoff_day='2023-10-23')
    assert w.mean()==pytest.approx(1.) and np.all(np.diff(w)>0)
    assert report['effective_samples']<3


@pytest.mark.parametrize('change',[{'source_game_date':'2023-10-24','season':'2023-24'}, {'source_game_date':'2023-10-23','season':'2024-25'}, {'season':'2022-23'}])
def test_bad_past_rows_fail_closed(change):
    with pytest.raises(ValueError):forecast_mask_and_weights([change],'T1',forecast_day='2023-10-24',forecast_season='2023-24',cutoff_day='2023-10-23')


def test_timestamps_and_nonprior_cutoffs_are_rejected():
    with pytest.raises(ValueError):forecast_mask_and_weights(rows(),'T1',forecast_day='2023-10-24',forecast_season='2023-24',cutoff_day=datetime(2023,10,23))
    with pytest.raises(ValueError):forecast_mask_and_weights(rows(),'T1',forecast_day='2023-10-24',forecast_season='2023-24',cutoff_day='2023-10-24')
