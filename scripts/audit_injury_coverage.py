"""Measure eligible, complete and nonzero injury features without training."""
import asyncio
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

from sqlalchemy import select
from app.database import async_session_factory
from app.models.game import Game
from app.models.injury_report import InjuryReport
from app.services.injury_snapshot_reader import latest_injury_snapshot, snapshot_statuses
from ml.src.features.injury_net_features import InjuryNetFeatures
from ml.src.features.player_features import PlayerFeatures


async def main():
    reasons, rows = Counter(), []
    net, aggregate = InjuryNetFeatures(), PlayerFeatures()
    async with async_session_factory() as session:
        pairs = (await session.execute(select(InjuryReport.game_id, InjuryReport.team_id).distinct())).all()
        for game_id, team_id in pairs:
            game = await session.get(Game, game_id)
            cutoff = game.scheduled_start_utc or game.game_date
            cutoff -= timedelta(microseconds=1)
            report = await latest_injury_snapshot(session, game_id, team_id, cutoff, reconstructed=True)
            quality = {}
            features = await net.compute(team_id, cutoff, session=session, use_historical=True,
                                         target_game_id=game_id, diagnostics=quality)
            agg = await aggregate.compute_injury_impact(team_id, cutoff, session=session,
                        use_historical=True, target_game_id=game_id)
            reasons[quality.get('unavailable_reason', 'available')] += 1
            rows.append(dict(game_id=game_id, team_id=team_id, source_game_date=str(game.source_game_date),
                cutoff=cutoff.isoformat(), report_id=report.id if report else None,
                statuses=snapshot_statuses(report), quality=quality, net=features, aggregate=agg))
    result = dict(mode='reconstructed_replay', team_games=len(rows), reasons=dict(reasons),
        aggregate_nonzero=sum(r['aggregate']['injury_firepower_loss'] > 0 for r in rows),
        net_nonzero=sum(abs(r['net']['injury_net_offense_delta']) > 1e-8 for r in rows), rows=rows)
    path = Path('/app/ml/saved_models/data_quality/injury_coverage_20261028.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
