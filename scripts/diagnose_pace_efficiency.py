"""Arithmetic diagnosis on fixed historical cohorts, never a deployable oracle."""
import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sqlalchemy import select, text
from app.database import async_session_factory
from app.models.game import Game
from app.models.team_game_stats import TeamGameStats
from ml.src.features.dataset_cache import DatasetCache
from ml.src.features.feature_builder import FEATURE_NAMES
from ml.src.evaluation import regression_metrics


def decompose_total(predicted_pace, predicted_efficiency, actual_pace, total, minutes):
    if min(predicted_pace, predicted_efficiency, actual_pace, minutes) <= 0:
        raise ValueError('Positive known pace, efficiency and duration required')
    regulation_total = total * 48.0 / minutes
    actual_efficiency = regulation_total * 100.0 / actual_pace
    dp, de = actual_pace-predicted_pace, actual_efficiency-predicted_efficiency
    baseline = predicted_pace*predicted_efficiency/100.0
    parts = dict(pace=dp*predicted_efficiency/100.0,
        efficiency=predicted_pace*de/100.0, interaction=dp*de/100.0,
        extra_duration=total-regulation_total)
    assert abs(sum(parts.values())-(total-baseline)) < 1e-8
    return dict(baseline=baseline, actual_pace=actual_pace,
                actual_efficiency=actual_efficiency, contributions=parts)


def summarize(rows):
    if not rows:
        return dict(samples=0)
    y=np.array([r['actual_total'] for r in rows]); b=np.array([r['baseline'] for r in rows])
    return dict(samples=len(rows), baseline=regression_metrics(y,b),
        pace=regression_metrics(np.array([r['actual_pace'] for r in rows]),
                               np.array([r['predicted_pace'] for r in rows])),
        efficiency=regression_metrics(np.array([r['actual_efficiency'] for r in rows]),
                                     np.array([r['predicted_efficiency'] for r in rows])),
        contributions={k:dict(mean=float(np.mean([r['contributions'][k] for r in rows])),
                              mean_absolute=float(np.mean([abs(r['contributions'][k]) for r in rows])))
                       for k in ('pace','efficiency','interaction','extra_duration')})


async def main(args):
    output=Path(args.output)
    if output.exists():
        raise ValueError('Do not overwrite completed diagnosis')
    data=DatasetCache().load(args.feature_cache_key,len(FEATURE_NAMES))
    assert data is not None
    lookup={r['nba_game_id']:i for i,r in enumerate(data.sample_info)}
    report=dict(protocol=dict(feature_cache_key=args.feature_cache_key, fit_models=False,
        oracle_values_are_postgame_diagnostics_only=True, selection_on_audit=False,
        missing_duration_or_pace_excluded=True, pace_is_box_score_estimate=True),phases={})
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        report['quality_by_season']=[dict(r._mapping) for r in (await session.execute(text('''
            SELECT season,count(*) AS games,
              count(*) FILTER (WHERE score_complete) AS score_complete,
              count(*) FILTER (WHERE team_stats_complete) AS team_complete,
              count(*) FILTER (WHERE player_stats_complete) AS player_complete
            FROM games WHERE status='Final' AND game_type IN ('regular','play_in','playoffs')
            GROUP BY season ORDER BY season
        '''))).all()]
        games=(await session.scalars(select(Game).where(Game.status=='Final'))).all()
        by_nba={g.nba_game_id:g for g in games}
        stats=defaultdict(list)
        for row in (await session.scalars(select(TeamGameStats))).all():
            stats[row.game_id].append(row)
        for phase in ('development','audit'):
            source=Path(args.reference)/f'{phase}_direct.json'
            original=json.loads(source.read_text())['predictions']['xgboost:T0']
            rows=[]; excluded=[]
            for ref in original:
                game=by_nba[ref['nba_game_id']]; pair=stats[game.id]
                X=data.X[lookup[ref['nba_game_id']]]
                values={name:float(X[FEATURE_NAMES.index(name)]) for name in (
                    'home_bayes_pace','away_bayes_pace','home_bayes_off_rtg','away_bayes_off_rtg',
                    'home_bayes_def_rtg','away_bayes_def_rtg','shrunk_expected_total')}
                assert abs(values['shrunk_expected_total']-ref['baseline_total'])<1e-3
                if not game.team_stats_complete or len(pair)!=2 or any(
                    t.game_minutes is None or t.pace is None or t.game_minutes<=0 or t.pace<=0 for t in pair):
                    excluded.append(dict(game_id=game.id,reason='unknown_duration_or_incomplete_stats'));continue
                assert abs(pair[0].game_minutes-pair[1].game_minutes)<.05
                p=(values['home_bayes_pace']+values['away_bayes_pace'])/2
                e=sum(values[k] for k in ('home_bayes_off_rtg','away_bayes_off_rtg',
                                          'home_bayes_def_rtg','away_bayes_def_rtg'))/2
                actual=game.home_score+game.away_score
                assert actual==ref['actual'][2]
                parts=decompose_total(p,e,sum(t.pace for t in pair)/2,actual,pair[0].game_minutes)
                assert abs(parts['baseline']-ref['baseline_total'])<1e-3
                rows.append(dict(game_id=game.id,nba_game_id=game.nba_game_id,
                    source_game_date=str(game.source_game_date),actual_total=actual,
                    predicted_pace=p,predicted_efficiency=e,minutes=pair[0].game_minutes,**parts))
            report['phases'][phase]=dict(reference_samples=len(original),excluded=excluded,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),all=summarize(rows),
                slices={label:summarize([r for r in rows if predicate(r)]) for label,predicate in (
                    ('total_ge_250',lambda r:r['actual_total']>=250),
                    ('total_le_210',lambda r:r['actual_total']<=210),
                    ('pace_ge_105',lambda r:r['actual_pace']>=105),
                    ('regulation',lambda r:r['minutes']<49),
                    ('overtime',lambda r:r['minutes']>=49))},rows=rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(quality_by_season=report['quality_by_season'],
        phases={k:{x:y for x,y in v.items() if x not in ('rows','excluded')}|{'excluded':len(v['excluded'])}
                for k,v in report['phases'].items()}),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--feature-cache-key',required=True)
    p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    asyncio.run(main(p.parse_args()))
