"""Explicit forecast-season windows; legacy policy implementation stays frozen.

Both dates are NBA source calendar days, never inferred from a UTC timestamp.
The caller supplies only past rows. Fit coverage is reported separately from
the information cutoff, so a June last game cannot redefine an October season.
"""
from datetime import date,datetime
from math import log
import re
import numpy as np
from ml.src.training_policy import POLICIES,TrainingPolicy


def nba_date(value):
    if isinstance(value,datetime):raise ValueError('Use an explicit NBA calendar day, not a timestamp')
    if isinstance(value,date):return value
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        raise ValueError('NBA day must be YYYY-MM-DD')
    return date.fromisoformat(value)


def season_year(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}',value):raise ValueError('Explicit NBA season required')
    year=int(value[:4])
    if int(value[-2:])!=(year+1)%100:raise ValueError('Invalid season suffix')
    return year


def forecast_mask_and_weights(sample_info,policy,*,forecast_day,forecast_season,cutoff_day):
    target,cutoff=nba_date(forecast_day),nba_date(cutoff_day)
    forecast_year=season_year(forecast_season)
    if target.year not in (forecast_year,forecast_year+1):raise ValueError('Target day does not match forecast season')
    if cutoff>=target:raise ValueError('Information cutoff must precede target NBA day')
    chosen=POLICIES[policy] if isinstance(policy,str) else policy
    if not isinstance(chosen,TrainingPolicy):raise ValueError('Unknown training policy')
    days=[];years=[]
    for row in sample_info:
        if not row.get('source_game_date'):raise ValueError('Training row lacks explicit NBA source day')
        day=nba_date(row['source_game_date']);year=season_year(row.get('season'))
        if day.year not in (year,year+1):raise ValueError('Training day and recorded season disagree')
        if day>cutoff or day>=target:raise ValueError('Future row in supplied past training data')
        if year>forecast_year:raise ValueError('Future season in supplied training data')
        days.append(day);years.append(year)
    earliest=None if chosen.completed_seasons is None else forecast_year-chosen.completed_seasons
    mask=np.array([earliest is None or year>=earliest for year in years],dtype=bool)
    kept=[d for d,yes in zip(days,mask) if yes]
    weights=np.ones(len(kept),dtype=float)
    if chosen.half_life_days is not None and kept:
        ages=np.array([(cutoff-d).days for d in kept],dtype=float)
        weights=np.exp(-log(2)*ages/chosen.half_life_days);weights/=weights.mean()
    by_season={}
    for row,w in zip([r for r,yes in zip(sample_info,mask) if yes],weights):
        bucket=by_season.setdefault(row['season'],dict(samples=0,weight=0.))
        bucket['samples']+=1;bucket['weight']+=float(w)
    report=dict(policy=chosen.name,anchor='explicit_forecast_season_v1',forecast_day=str(target),
        forecast_season=forecast_season,information_cutoff_day=str(cutoff),
        completed_seasons=chosen.completed_seasons,half_life_days=chosen.half_life_days,
        samples=len(kept),fit_from=str(min(kept)) if kept else None,fit_until=str(max(kept)) if kept else None,
        effective_samples=float(weights.sum()**2/(weights**2).sum()) if kept else 0.,by_season=by_season)
    return mask,weights,report
