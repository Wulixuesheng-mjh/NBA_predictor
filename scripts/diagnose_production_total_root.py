"""Frozen serving-runtime diagnosis only. No fits, publishing or ledger writes."""
import argparse
import asyncio
from dataclasses import asdict
from datetime import date,timedelta,timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
from sqlalchemy import select,text
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team
from app.models.team_game_stats import TeamGameStats
from app.models.shadow_observation import ShadowObservation,ShadowObservationRun
from app.services.prospective_service import digest
from ml.src.config import ml_settings
from ml.src.predictor import load_predictor,Predictor
from ml.src.features.feature_builder import FeatureBuilder,FEATURE_NAMES
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.opponent_adjusted_features import OpponentAdjustedFeatures


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def summary(rows,key):
    valid=[r for r in rows if r.get(key) is not None]
    if not valid:return dict(samples=0,missing=len(rows))
    p=np.array([r[key] for r in valid]);a=np.array([r['actual_total'] for r in valid]);e=p-a
    groups=defaultdict(list)
    for r,err in zip(valid,e):groups[r['source_day']].append(float(err))
    days=sorted(groups);rng=np.random.default_rng(20261109)
    boot=[]
    for _ in range(2000):
        sample=rng.integers(len(days),size=len(days))
        boot.append(np.mean([v for i in sample for v in groups[days[i]]]))
    return dict(samples=len(valid),missing=len(rows)-len(valid),days=len(days),
        predicted_mean=float(p.mean()),actual_mean=float(a.mean()),bias=float(e.mean()),
        mae=float(abs(e).mean()),rmse=float(np.sqrt(np.mean(e**2))),
        underprediction_rate=float(np.mean(e<0)),bias_day_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist(),
        interval_caution='descriptive only; few days and reconstructed information limit inference')


def mlp_numeric(model,X,channels):
    z=(X-model._scaler_mean)/model._scaler_std
    clipped=np.clip(z,-5,5).astype('float32')
    if model._active_features is not None:clipped[:,~model._active_features]=0
    centered=clipped.copy()
    for row in channels: centered[:,FEATURE_NAMES.index(row['name'])]=row['bn_running_mean']
    win,spread,total=model._predict_onnx(centered)
    sp=float(spread[0,0]*model._y_spread_std+model._y_spread_mean)
    t=float(total[0,0]*model._y_total_std+model._y_total_mean)
    ok,reason=Predictor._is_plausible(SimpleNamespace(win_prob_home=float(win[0,0]),predicted_spread=sp,predicted_total=t))
    return dict(centered_total=t,centered_accepted=ok,centered_reason=reason,
        clipped_columns=int((abs(z)>5).sum()),max_abs_z=float(abs(z).max()))


async def main(args):
    out=Path(args.output)
    if out.exists():raise ValueError('Immutable diagnostic output already exists')
    out.mkdir(parents=True)
    root=Path(ml_settings.models_dir);active=(root/'active.json').read_bytes()
    manifest=json.loads(active);assert manifest['version']=='20261022_050952'
    predictor=load_predictor(release_manifest=manifest);xgb=predictor.xgb_trainer;mlp=predictor.mlp_trainer
    assert mlp is not None and mlp._onnx_session is not None,'Serving ONNX MLP unavailable'
    audit=json.loads(Path(args.mlp_audit).read_text())
    folder=root/'mlp'/manifest['mlp_version']
    assert all(sha(folder/name)==expected for name,expected in audit['artifacts'].items())
    channels=[r for r in audit['features'] if r['critical']]
    injuries=[FEATURE_NAMES.index(side+'_'+name) for side in ('home','away') for name in PLAYER_FEATURE_NAMES]
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        await session.execute(text('SET TRANSACTION READ ONLY'))
        revision=(await session.execute(text('SELECT revision FROM data_revisions'))).scalar_one()
        games=list((await session.scalars(select(Game).where(Game.status=='Final',Game.score_complete.is_(True),
            Game.team_stats_complete.is_(True),Game.player_stats_complete.is_(True),
            Game.game_type.in_(('regular','play_in','playoffs')),
            Game.source_game_date>date.fromisoformat(manifest['training_data_until']),
            Game.source_game_date<=date(2026,11,8)).order_by(Game.source_game_date,Game.id))).all())
        protocol=dict(kind='frozen_production_root_cause_diagnosis',production=manifest,revision=revision,
            active_sha256=hashlib.sha256(active).hexdigest(),cohort=[g.id for g in games],
            xgb_files={p.name:sha(p) for p in (root/'xgboost'/manifest['xgboost_version']).glob('*.json')},
            source_days=[str(g.source_game_date) for g in games],
            runtime_files={str(p.relative_to(Path('/app'))):sha(p) for p in Path('/app/ml/src').rglob('*.py')},
            mlp_audit_sha256=sha(args.mlp_audit),training=False,promotion=False,ledger_writes=False,
            hypotheses=dict(H1='postprocessing causes downward shift: falsified if final equals raw within .25 and calibration off',
                H2='injury features mechanically depress totals: fixed-input zeroing; no healthy assumption',
                H3='expected_total compatibility division causes shift: no-division numeric probe only; wrong train units may invalidate it',
                H4='legacy constant BN channels amplify distribution change: preset six-channel centering only; no deployment or precision claim'),
            negative_control='modify appended features only; legacy model outputs must not change',
            reconstruction='current corrected DB, frozen serving code, hypothetical T-60 cutoff, historical injury reader; not authentic forecasts',
            limits=['later-arriving stats cannot certify historical availability','legacy metadata lacks independently recorded feature names',
                'current player/roster provenance may not recreate original pregame state','interventions do not establish deployable accuracy gain'],
            bootstrap=dict(seed=20261109,draws=2000,group='NBA source day'))
        save(out/'protocol.json',protocol)
        actual_rows=[]
        observations=(await session.scalars(select(ShadowObservation).order_by(ShadowObservation.id))).all()
        for record in observations:
            game=await session.get(Game,record.game_id);run=await session.get(ShadowObservationRun,record.run_id)
            p=record.payload
            assert digest(p)==record.payload_hash and digest(run.protocol)==run.protocol_hash
            assert p['protocol_hash']==run.protocol_hash
            assert run.protocol['baseline']['version']==manifest['version']
            assert record.as_of<record.scheduled_start_utc and record.created_at<record.scheduled_start_utc
            assert str(game.source_game_date)>manifest['training_data_until']
            X=np.array(p['baseline_processed_vector'],dtype=float).reshape(1,-1)
            assert p['baseline_feature_names']==FEATURE_NAMES[:X.shape[1]]
            replay=float(xgb._predict_totals(X)[0]);stored=p['totals']['baseline']
            if abs(replay-stored)>1e-3:raise ValueError('Stored authentic XGB input failed replay')
            actual_rows.append(dict(game_id=game.id,source_day=str(game.source_game_date),as_of=record.as_of.isoformat(),
                created_at=record.created_at.isoformat(),start=record.scheduled_start_utc.isoformat(),
                actual_total=game.home_score+game.away_score,stored_total=stored,replayed_total=replay,
                processed_features=X[0].tolist(),warnings=p['warnings'],injury_snapshots=p['injury_snapshots'],
                mlp_stored=False,payload_hash=record.payload_hash))
        save(out/'authentic_records.json',dict(rows=actual_rows,metrics=summary(actual_rows,'stored_total')))
        names={t.id:t.abbreviation for t in (await session.scalars(select(Team))).all()}
        stats=defaultdict(list)
        for stat in (await session.scalars(select(TeamGameStats).where(TeamGameStats.game_id.in_([g.id for g in games])))).all():stats[stat.game_id].append(stat)
        builder=FeatureBuilder();cache=await OpponentAdjustedFeatures.precompute_opponent_cache(session)
        rows=[]
        for g in games:
            cutoff=g.scheduled_start_utc-timedelta(minutes=60) if g.scheduled_start_utc else g.game_date
            fv=await builder.build_features(g.home_team_id,g.away_team_id,g.game_date,session=session,
                opponent_cache=cache,use_historical=True,as_of=cutoff,target_game_id=g.id)
            result=predictor.predict(fv,apply_calibration=True)
            raw=result.raw_outputs
            X=predictor._adapt_features(predictor._prepare_release_features(fv.features.reshape(1,-1)),166)
            noinj=X.copy();noinj[:,injuries]=0
            scale_probe=X.copy();scale_probe[:,FEATURE_NAMES.index('expected_total')]*=2
            control=fv.features.copy();control[166:]+=123
            control=predictor._adapt_features(predictor._prepare_release_features(control.reshape(1,-1)),166)
            if not np.array_equal(control,X):raise ValueError('Appended feature negative control failed')
            def rawtot(m,xx):
                v=m.predict(xx);return float((v[0] if isinstance(v,list) else v).predicted_total)
            r=dict(game_id=g.id,source_day=str(g.source_game_date),home=names[g.home_team_id],away=names[g.away_team_id],
                as_of=cutoff.isoformat(),actual_total=g.home_score+g.away_score,kind='reconstructed_not_saved_forecast',
                feature_vector=fv.features.tolist(),feature_names=FEATURE_NAMES,processed_vector=X[0].tolist(),
                warnings=result.data_quality['warnings'],calibration_applied=result.calibration_applied,
                raw_outputs=raw,xgb_raw=raw['xgboost']['predicted_total'],mlp_raw=raw['mlp']['predicted_total'],
                xgb_final=None if result.xgboost is None else result.xgboost.predicted_total,
                mlp_final=None if result.mlp is None else result.mlp.predicted_total,
                xgb_zero_injury=rawtot(xgb,noinj),mlp_zero_injury=rawtot(mlp,noinj),
                xgb_skip_half=rawtot(xgb,scale_probe),mlp_skip_half=rawtot(mlp,scale_probe),
                injury_input_nonzero=bool(np.any(X[:,injuries]!=0)),**mlp_numeric(mlp,X,channels))
            f=dict(zip(FEATURE_NAMES,fv.features.tolist()))
            p=(f['home_avg_pace']+f['away_avg_pace'])/2
            eff=sum(f[n] for n in ('home_avg_off_rtg','away_avg_off_rtg','home_avg_def_rtg','away_avg_def_rtg'))/2
            r.update(rolling_baseline=p*eff/100,shrunk_baseline=f['shrunk_expected_total'],predicted_pace=p,
                predicted_efficiency=eff,expected_total_before_compatibility=f['expected_total'])
            pair=stats[g.id]
            if len(pair)==2 and all(t.pace and t.game_minutes for t in pair):
                duration=float(pair[0].game_minutes);actual_pace=float(np.mean([t.pace for t in pair]))
                actual_eff=r['actual_total']*48/duration*100/actual_pace
                dp=actual_pace-p;de=actual_eff-eff
                parts=dict(pace=dp*eff/100,efficiency=p*de/100,interaction=dp*de/100,
                    overtime=r['actual_total']*(1-48/duration),model_vs_baseline=r['rolling_baseline']-r['xgb_raw'])
                assert abs(sum(parts.values())-(r['actual_total']-r['xgb_raw']))<1e-5
                r.update(actual_pace=actual_pace,actual_efficiency=actual_eff,game_minutes=duration,error_decomposition=parts)
            rows.append(r)
            with (out/'reconstructed_rows.jsonl').open('a') as stream:stream.write(json.dumps(r,allow_nan=False)+'\n')
            if len(rows)%10==0:print(json.dumps(dict(completed=len(rows),planned=len(games))),flush=True)
        assert (root/'active.json').read_bytes()==active
        keys=['xgb_raw','xgb_final','mlp_raw','mlp_final','rolling_baseline','shrunk_baseline',
            'xgb_zero_injury','mlp_zero_injury','xgb_skip_half','mlp_skip_half','centered_total']
        result=dict(kind='reconstructed_diagnostic_not_historical_scores',samples=len(rows),protocol_sha256=sha(out/'protocol.json'),
            metrics={k:summary(rows,k) for k in keys},by_day={},injury_nonzero_games=sum(r['injury_input_nonzero'] for r in rows),
            calibration_applied=sum(r['calibration_applied'] for r in rows),
            source_sha256={k:sha(root/'xgboost'/manifest['xgboost_version']/k) for k in ('total.json','metadata.json')
                if (root/'xgboost'/manifest['xgboost_version']/k).exists()},
            mlp_centered_rejected=sum(not r['centered_accepted'] for r in rows),
            decomposition_mean={k:float(np.mean([r['error_decomposition'][k] for r in rows if 'error_decomposition' in r]))
                for k in ('pace','efficiency','interaction','overtime','model_vs_baseline')},
            primary_unchanged=True,ledger_writes=0,trained=False,promoted=False)
        for day in sorted({r['source_day'] for r in rows}):
            subset=[r for r in rows if r['source_day']==day]
            result['by_day'][day]={k:summary(subset,k) for k in ('xgb_raw','mlp_raw','rolling_baseline')}
        result['intervention_effects']={key:dict(mean_shift=float(np.mean([r[key]-r[base] for r in rows])),
            min_shift=float(min(r[key]-r[base] for r in rows)),max_shift=float(max(r[key]-r[base] for r in rows)))
            for key,base in [('xgb_zero_injury','xgb_raw'),('mlp_zero_injury','mlp_raw'),
                ('xgb_skip_half','xgb_raw'),('mlp_skip_half','mlp_raw'),('centered_total','mlp_raw')]}
        save(out/'report.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--mlp-audit',required=True)
    asyncio.run(main(parser.parse_args()))
