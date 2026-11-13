"""Settle immutable manual diagnostic files; never write an acceptance ledger."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.models.team_game_stats import TeamGameStats
from ml.src.evaluation import regression_metrics


def file_record(path):
    raw = path.read_bytes()
    record = json.loads(raw)
    expected = record.pop('record_sha256')
    assert hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest() == expected
    record['record_sha256'] = expected
    assert record['acceptance_run'] is False and record['mode'] == 'diagnostic_pregame_shadow'
    as_of = datetime.fromisoformat(record['as_of'])
    start = datetime.fromisoformat(record['scheduled_start_utc'])
    assert as_of.tzinfo and start.tzinfo and as_of < start
    manifests = [record['baseline_manifest'], *record['candidate_manifests'].values()]
    assert all(record['source_game_date'] > m['training_data_until'][:10] for m in manifests)
    return record, hashlib.sha256(raw).hexdigest()


async def main(folder, output):
    rows, rejected, hashes = [], [], {}
    seen = set()
    async with async_session_factory() as session:
        for path in sorted(Path(folder).glob('*.json')):
            try:
                record, file_hash = file_record(path)
                game = await session.get(Game, record['game_id'])
                assert game is not None and game.id not in seen
                assert game.home_team_id == record['home_team_id'] and game.away_team_id == record['away_team_id']
                assert str(game.source_game_date) == record['source_game_date']
                assert game.scheduled_start_utc and datetime.fromisoformat(record['as_of']) < game.scheduled_start_utc
            except (AssertionError, KeyError, ValueError) as exc:
                rejected.append(dict(file=path.name, reason=type(exc).__name__ + ': ' + str(exc)))
                continue
            seen.add(game.id)
            hashes[str(path)] = file_hash
            finished = game.status == 'Final' and game.score_complete and game.home_score is not None and game.away_score is not None
            frozen = dict(zip(record['feature_names']['direct'], record['processed_vectors']['direct']))
            stats = (await session.scalars(select(TeamGameStats).where(TeamGameStats.game_id == game.id))).all()
            actual_stats = [dict(side='home' if t.is_home else 'away', pts=t.pts,
                pace=t.pace, off_rtg=t.off_rtg, game_minutes=t.game_minutes,
                fg3a_rate=t.fg3a/t.fga if t.fga else None,
                fg3_pct=t.fg3m/t.fg3a if t.fg3a else None,
                fta_per_fga=t.fta/t.fga if t.fga else None)
                for t in stats] if game.team_stats_complete and len(stats) == 2 else []
            rows.append(dict(game_id=game.id, as_of=record['as_of'],
                source_game_date=str(game.source_game_date), status=game.status, settled=finished,
                scheduled_start_utc=game.scheduled_start_utc.isoformat(),
                home_score=game.home_score, away_score=game.away_score,
                actual_total=game.home_score + game.away_score if finished else None,
                totals=record['totals'], training_cutoffs={name: m['training_data_until'] for name, m in record['candidate_manifests'].items()},
                frozen_pregame_baseline=record['baseline'],
                frozen_pregame_features={name: frozen.get(name) for name in ('expected_pace',
                    'home_bayes_pace', 'away_bayes_pace', 'home_bayes_off_rtg', 'away_bayes_off_rtg',
                    'home_bayes_def_rtg', 'away_bayes_def_rtg', 'home_avg_fg3a_rate', 'away_avg_fg3a_rate')},
                actual_team_stats=actual_stats,
                team_stats_complete=game.team_stats_complete, player_stats_complete=game.player_stats_complete))
    settled = [r for r in rows if r['settled']]
    metrics = {}
    for mode in ('old_compatible', 'direct', 'residual'):
        if settled:
            metrics[mode] = regression_metrics(np.array([r['actual_total'] for r in settled]),
                                               np.array([r['totals'][mode] for r in settled]))
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == value for path, value in hashes.items())
    report = dict(mode='manual_diagnostic_settlement', acceptance_run=False,
        checked_at_utc=datetime.now(timezone.utc).isoformat(), captured=len(rows),
        settled=len(settled), pending=len(rows)-len(settled), rejected=rejected,
        metrics=metrics, rows=rows, input_sha256=hashes,
        records_unchanged=True, formal_acceptance_added=0,
        limitations='Small diagnostic sample; not a release gate, no interim retuning or market hit rates.')
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='/tmp/manual_diagnostic_settlement.json')
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output))
