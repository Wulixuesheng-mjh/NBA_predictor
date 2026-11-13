"""Inventory authentic saved predictions; never infer missing MLP forecasts."""
import argparse,asyncio,json
from collections import Counter,defaultdict
from datetime import timezone
from pathlib import Path
import numpy as np
from sqlalchemy import select
from app.database import async_session_factory
from app.models.prediction import Prediction
from app.models.game import Game
from app.models.shadow_observation import ShadowObservation,ShadowObservationRun


def aware(value):
    return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


def summary(rows):
    if not rows:return dict(samples=0)
    error=np.array([r['predicted_total']-r['actual_total'] for r in rows])
    return dict(samples=len(rows),mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
        bias=float(error.mean()),underprediction_rate=float(np.mean(error<0)))


async def main(args):
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        preds=list((await session.scalars(select(Prediction).order_by(Prediction.id))).all())
        games={g.id:g for g in (await session.scalars(select(Game))).all()}
        excluded=Counter(); rows=[]; inventory=[]
        for p in preds:
            g=games.get(p.game_id);meta=p.features_json or {}
            inventory.append(dict(id=p.id,model=p.model_name,version=p.model_version,mode=p.prediction_mode,
                game_id=p.game_id,as_of=str(p.as_of),raw_keys=list((p.raw_output or {}).keys()),
                metadata_keys=list(meta.keys()),has_ordered_vector=p.feature_vector is not None))
            if not g or p.prediction_mode!='pregame' or not p.as_of or not p.created_at:
                excluded['unbound_preview_or_missing_time']+=1;continue
            if not g.scheduled_start_utc or g.time_precision!='exact':
                excluded['no_exact_start']+=1;continue
            if aware(p.as_of)>=aware(g.scheduled_start_utc) or aware(p.created_at)>=aware(g.scheduled_start_utc):
                excluded['not_created_before_tipoff']+=1;continue
            manifest=meta.get('_release_manifest') or {}
            cutoff=manifest.get('training_data_until')
            if not cutoff or not g.source_game_date or str(g.source_game_date)<=str(cutoff)[:10]:
                excluded['unverified_or_in_training_cutoff']+=1;continue
            if g.status!='Final' or not g.score_complete:
                excluded['awaiting_result']+=1;continue
            if p.predicted_total is None:excluded['no_stored_total']+=1;continue
            rows.append(dict(id=p.id,model=p.model_name,version=p.model_version,game_id=g.id,
                as_of=aware(p.as_of).isoformat(),predicted_total=p.predicted_total,
                actual_total=g.home_score+g.away_score,source='stored_prediction_not_replayed',
                exact_input_available=bool(meta.get('_inference_snapshot'))))
        pairs=defaultdict(dict)
        for r in rows:
            pairs[(r['game_id'],r['as_of'])].setdefault(r['model'],r)
        paired=[v for v in pairs.values() if 'xgboost' in v and 'mlp' in v]
        existing=[]
        for observation in (await session.scalars(select(ShadowObservation))).all():
            g=games[observation.game_id]
            if g.status=='Final' and g.score_complete:
                existing.append(dict(game_id=g.id,as_of=str(observation.as_of),
                    predicted_total=observation.payload['totals']['baseline'],actual_total=g.home_score+g.away_score,
                    source='existing_xgb_only_shadow_no_mlp'))
    report=dict(kind='read_only_saved_prediction_inventory',writes=0,
        inventory_counts=dict(Counter(f'{p.model_name}:{p.prediction_mode}' for p in preds)),
        excluded=dict(excluded),inventory=inventory,eligible_stored_rows=rows,
        exact_as_of_paired_games=len(paired),paired_metrics={m:summary([p[m] for p in paired]) for m in ('xgboost','mlp')},
        existing_xgb_only_shadow=dict(metrics=summary(existing),rows=existing),
        missing_mlp_backfilled=False,selection_on_live_results=False)
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inventory','eligible_stored_rows','existing_xgb_only_shadow')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
