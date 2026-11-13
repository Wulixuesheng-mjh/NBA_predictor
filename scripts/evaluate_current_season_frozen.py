"""Fixed model season audit, current corrected DB reconstruction, no fitting."""
import asyncio,json,hashlib,traceback
from pathlib import Path
from datetime import datetime,date,time,timedelta,timezone
from sqlalchemy import select,text,func
from app.database import async_session_factory
from app.models.game import Game
from app.services.injury_snapshot_reader import latest_injury_snapshot
from app.services.shadow_total_service import shadow_totals
from ml.src.config import ml_settings
from ml.src.predictor import load_predictor
from ml.src.features.feature_builder import FeatureBuilder,FEATURE_NAMES
from ml.src.features.opponent_adjusted_features import OpponentAdjustedFeatures
import numpy as np
import xgboost as xgb
from scripts.paired_models_worker import paired_outputs,safe_values,digest,sha
from ml.src.legacy_mlp_compatibility import contract_spec

OUT=Path('/out')


def predict_frozen_win(folder,features,target_day):
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['feature_names']==FEATURE_NAMES and manifest['targets']==['win']
    assert str(target_day)>manifest['training_data_until']
    assert sha(folder/'win_model.json')==manifest['model_sha256']
    value=np.asarray(features,dtype=np.float64).reshape(1,-1).copy()
    value[:,[FEATURE_NAMES.index(n) for n in manifest['disabled_features']]]=0
    model=xgb.XGBClassifier();model.load_model(folder/'win_model.json')
    probability=float(model.predict_proba(value)[0,1])
    assert np.isfinite(probability) and 0<=probability<=1
    return dict(model_version=manifest['version'],win_prob_home=probability)


async def main():
    root=Path(ml_settings.models_dir);active=json.loads((root/'active.json').read_text())
    assert active['training_data_until']=='2026-10-21'
    assert not (OUT/'protocol.json').exists(),'Do not overwrite completed season audit'
    async with async_session_factory() as db:
        await db.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await db.execute(text('SET TRANSACTION READ ONLY'))
        all_games=list((await db.scalars(select(Game).where(Game.season=='2026-27',
            Game.game_type.in_(('regular','play_in','playoffs'))).order_by(Game.source_game_date,Game.id))).all())
        cohort=[g for g in all_games if g.status=='Final' and g.score_complete and g.home_score is not None
            and g.away_score is not None and g.source_game_date and str(g.source_game_date)>active['training_data_until']
            and g.game_date<datetime.now(timezone.utc)]
        p=dict(kind='full_current_season_frozen_reconstruction',locked_at=datetime.now(timezone.utc).isoformat(),
            season='2026-27',cohort=[g.id for g in cohort],revision=await db.scalar(text('select revision from data_revisions')),
            active=active,active_sha256=sha(root/'active.json'),contract=contract_spec(),
            script_sha256=sha(__file__),feature_names=FEATURE_NAMES,
            cutoff='Conservative NBA source-day midnight, no target/same-source-day labels; reconstructed not authentic pregame',
            injury='publication-verified historical snapshots only, published<=cutoff; no present-day injury backfill',
            eligibility='all Final score-complete post-fit games, INCLUDING missing target statistics',
            excluded=dict(total_season_games=len(all_games),finished=sum(g.status=='Final' for g in all_games),
                in_fit=sum(g.status=='Final' and g.source_game_date is not None and str(g.source_game_date)<=active['training_data_until'] for g in all_games)),
            metrics='Brier, logloss, accuracy; MAE/RMSE/Bias/MedAE/P90, guards, failures; day-paired bootstrap4000 seed20261112',
            repair_gate='all three raw-head loss differences CI upper <= 1% old loss; no new silent errors; no fixed 200 requirement for retrospective safety screen',
            slices=['first 7 source-calendar days','missing previous statistics warning','actual total >=240 (diagnostic only)'],
            shadow_rule='fixed deployed packages only; evaluate only target days AFTER each model fit; no in-fit scoring',
            inference='historical corrected data availability/roster provenance cannot certify real historical receipt times',
            reused_history='previously diagnosed season, not a newly independent untouched test set',
            training=False,promotion=False,production_writes=0)
        (OUT/'protocol.json').write_text(json.dumps(p,indent=2));print(json.dumps(dict(locked=True,samples=len(cohort))),flush=True)
        baseline=load_predictor(release_manifest=active)
        fixed=load_predictor(release_manifest={**active,'mlp_input_compatibility':contract_spec()})
        cache=await OpponentAdjustedFeatures.precompute_opponent_cache(db)
        builder=FeatureBuilder();failures=0
        with (OUT/'rows.jsonl').open('x') as output:
            for n,g in enumerate(cohort):
                cutoff=min(g.game_date,datetime.combine(g.source_game_date,time.min,tzinfo=timezone.utc))
                row=dict(game_id=g.id,day=str(g.source_game_date),cutoff=cutoff.isoformat(),
                    actual=[int(g.home_score>g.away_score),g.home_score-g.away_score,g.home_score+g.away_score],
                    target_stats_complete=bool(g.team_stats_complete and g.player_stats_complete),
                    time_precision=g.time_precision)
                try:
                    assert g.game_date>=cutoff
                    leak=await db.scalar(select(func.count()).select_from(Game).where(Game.status=='Final',
                        Game.season=='2026-27',Game.source_game_date>=g.source_game_date,Game.game_date<cutoff))
                    assert leak==0,'Mixed date anchors would admit target/future source days'
                    fv=await builder.build_features(g.home_team_id,g.away_team_id,g.game_date,session=db,
                        as_of=cutoff,target_game_id=g.id,use_historical=True,opponent_cache=cache,lookback_games=10)
                    old=paired_outputs(baseline,fv);new=paired_outputs(fixed,fv)
                    assert old['models']['xgboost']==new['models']['xgboost'],'XGB changed during MLP fix'
                    row.update(baseline=old,candidate=new,builder_vector=fv.features.tolist(),injuries={})
                    for side,tid in [('home',g.home_team_id),('away',g.away_team_id)]:
                        r=await latest_injury_snapshot(db,g.id,tid,cutoff,reconstructed=True)
                        row['injuries'][side]=None if r is None else dict(id=r.id,complete=r.is_complete,
                            published_at=r.published_at.isoformat(),observed_at=r.observed_at.isoformat())
                    # The two total packages fit through 10/28; the win package through 10/29.
                    if str(g.source_game_date)>'2026-10-28':
                        row['total_shadows']=shadow_totals(fv.features,as_of=cutoff,lookback=10,target_day=str(g.source_game_date),root=root)
                    if str(g.source_game_date)>'2026-10-29':
                        row['win_shadow']=predict_frozen_win(root/'win_shadows/win_T5_20261104',fv.features,g.source_game_date)
                except Exception as exc:
                    failures+=1;row['failure']=str(exc);row['trace']=traceback.format_exc()
                paths=[];row=safe_values(row,paths);row['nonfinite_paths']=paths
                output.write(json.dumps(dict(payload=row,sha256=digest(row)),allow_nan=False)+'\n');output.flush()
                if n%10==0:print(json.dumps(dict(processed=n+1,total=len(cohort),failures=failures)),flush=True)
        assert sha(root/'active.json')==p['active_sha256']
        (OUT/'complete.json').write_text(json.dumps(dict(samples=len(cohort),failures=failures,
            rows_sha256=sha(OUT/'rows.jsonl'),production_unchanged=True,production_writes=0)))
        print(json.dumps(dict(done=True,samples=len(cohort),failures=failures)),flush=True)


if __name__=='__main__':asyncio.run(main())
