"""Read-only season replay and isolated shadow training; never publish or backfill a ledger."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from sqlalchemy import select, or_
from app.database import async_session_factory
from app.models.game import Game
from app.models.team import Team
from ml.src.config import ml_settings
from ml.src.evaluation import complete_day_split, nba_day, regression_metrics, paired_day_bootstrap
from ml.src.features.dataset_cache import DatasetCache, CACHE_FORMAT_VERSION
from ml.src.features.feature_builder import FeatureBuilder, FEATURE_NAMES
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.predictor import Predictor
from ml.src.training_policy import policy_mask_and_weights, subset_arrays


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, name):
    if not rows:
        return None
    actual = np.array([r['actual_total'] for r in rows])
    predicted = np.array([r['totals'][name] for r in rows])
    return dict(samples=len(rows), **regression_metrics(actual, predicted),
                under_count=int((predicted<actual).sum()), actual_mean=float(actual.mean()),
                prediction_mean=float(predicted.mean()))


async def main(args):
    root = Path(ml_settings.models_dir)
    active_path = root/'active.json'
    active_before = active_path.read_bytes()
    active = json.loads(active_before)
    assert active['version']=='20261022_050952', 'Reassess baseline if production changed'
    assert args.fit_until >= active['training_data_until']
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise ValueError('Diagnostic output already exists; choose a new directory')
    dataset = DatasetCache().load(args.feature_cache_key, len(FEATURE_NAMES))
    if dataset is None:
        raise ValueError('Missing pinned cache')
    past = subset_arrays(dataset, np.array([nba_day(r)<=args.fit_until for r in dataset.sample_info]))
    assert max(nba_day(r) for r in past.sample_info)<=args.fit_until
    disabled = [f'{side}_{name}' for side in ('home','away')
                for name in PLAYER_FEATURE_NAMES+INJURY_NET_FEATURE_NAMES]
    disabled_indexes = [FEATURE_NAMES.index(n) for n in disabled]
    past.X[:,disabled_indexes]=0
    old = XGBoostTrainer()
    old.load(active['xgboost_version'])
    adapter = Predictor(xgb_trainer=old, release_manifest=active)
    models, manifests = {}, {}
    baseline_index = FEATURE_NAMES.index('shrunk_expected_total')
    for mode, policy in [('direct','T5'),('residual','T4')]:
        if args.reuse:
            source=Path(args.reuse)
            manifest=json.loads((source/f'{mode}_manifest.json').read_text())
            assert manifest['training_data_until']==args.fit_until
            assert manifest['feature_names']==FEATURE_NAMES and manifest['disabled_features']==disabled
            for name,expected in manifest['file_hashes'].items():
                assert sha(source/'models'/mode/name)==expected
            model=XGBoostTrainer(models_dir=str(source/'models'))
            model.load(mode)
            models[mode],manifests[mode]=model,manifest
            continue
        mask, weights, weight_report = policy_mask_and_weights(past.sample_info,policy)
        chosen = subset_arrays(past,mask)
        split = complete_day_split(chosen.sample_info)
        model = XGBoostTrainer(models_dir=str(out/'models'), version=mode)
        baseline = chosen.X[:,baseline_index]
        model.train(chosen.X[:split],chosen.y_win[:split],chosen.y_spread[:split],chosen.y_total[:split],
                    chosen.X[split:],chosen.y_win[split:],chosen.y_spread[split:],chosen.y_total[split:],
                    sample_weight=weights[:split],total_strategy=mode,
                    total_baseline_train=baseline[:split] if mode=='residual' else None,
                    total_baseline_val=baseline[split:] if mode=='residual' else None,
                    total_baseline_feature_index=baseline_index if mode=='residual' else None)
        model.refit_full(chosen.X,chosen.y_win,chosen.y_spread,chosen.y_total,
                         sample_weight=weights,total_baseline=baseline if mode=='residual' else None)
        folder = Path(model.save())
        manifest = dict(version=mode,status='diagnostic_shadow_only',training_data_until=args.fit_until,
                        role='totals_only',calibration_enabled=False,lookback_games=10,
                        feature_dim=len(FEATURE_NAMES),feature_names=FEATURE_NAMES,
                        disabled_features=disabled,feature_semantic_version=CACHE_FORMAT_VERSION,
                        feature_cache_key=args.feature_cache_key, policy=weight_report,
                        fit_samples=len(chosen.X),fit_actual_until=max(map(nba_day,chosen.sample_info)),
                        selection_train_until=nba_day(chosen.sample_info[split-1]),
                        selection_validation_from=nba_day(chosen.sample_info[split]),
                        selection_validation_until=nba_day(chosen.sample_info[-1]),
                        fitted_rounds={n:int(m.n_estimators) for n,m in
                            [('win',model.model_win),('spread',model.model_spread),('total',model.model_total)]},
                        sample_sha256=hashlib.sha256(json.dumps(chosen.sample_info,sort_keys=True).encode()).hexdigest(),
                        file_hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
                        promote=False,eligible_for_200_game_acceptance=False)
        (out/f'{mode}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        # Real 100-row persistence + single/batch parity, including opener rows.
        sample=chosen.X[np.linspace(0,len(chosen.X)-1,100,dtype=int)]
        loaded=XGBoostTrainer(models_dir=str(out/'models'))
        loaded.load(mode)
        np.testing.assert_allclose(model._predict_totals(sample),loaded._predict_totals(sample),rtol=0,atol=1e-3)
        np.testing.assert_allclose(loaded._predict_totals(sample),
            [loaded._predict_totals(x.reshape(1,-1))[0] for x in sample],rtol=0,atol=1e-3)
        models[mode],manifests[mode]=loaded,manifest
    builder = FeatureBuilder()
    rows=[]
    async with async_session_factory() as session:
        await session.connection(execution_options={'isolation_level':'REPEATABLE READ'})
        teams={t.id:t.full_name for t in (await session.execute(select(Team))).scalars()}
        games=(await session.execute(select(Game).where(Game.season=='2026-27',Game.status=='Final',
            Game.score_complete.is_(True),Game.game_type.in_(('regular','play_in','playoffs')),
            Game.source_game_date>datetime.fromisoformat(args.fit_until).date())
            .order_by(Game.game_date))).scalars().all()
        if args.cohort_report:
            cohort=json.loads(Path(args.cohort_report).read_text())
            wanted={r['game_id'] for r in cohort['rows']}
            games=[g for g in games if g.id in wanted]
            assert {g.id for g in games}==wanted
        for game in games:
            assert str(game.source_game_date)>args.fit_until
            fv=await builder.build_features(game.home_team_id,game.away_team_id,game.game_date,
                session=session,use_historical=True,target_game_id=game.id)
            X=fv.features.reshape(1,-1)
            prepared=adapter._adapt_features(adapter._prepare_release_features(X),old.model_total.n_features_in_)
            totals={'old_compatible':float(old._predict_totals(prepared)[0]),
                    'baseline':float(X[0,baseline_index])}
            # Counterfactual input diagnostics, never presented as production predictions.
            no_injury=prepared.copy()
            no_injury[:,[i for i in disabled_indexes if i<no_injury.shape[1]]]=0
            totals['old_without_injuries']=float(old._predict_totals(no_injury)[0])
            unconverted=adapter._adapt_features(X,old.model_total.n_features_in_)
            totals['old_without_compat_conversion']=float(old._predict_totals(unconverted)[0])
            newX=X.copy();newX[:,disabled_indexes]=0
            for name,model in models.items():
                totals[name]=float(model._predict_totals(newX)[0])
            quality={}
            for side,tid in [('home',game.home_team_id),('away',game.away_team_id)]:
                history=(await session.execute(select(Game).where(Game.status=='Final',
                    Game.game_type.in_(('regular','play_in','playoffs')),
                    Game.source_game_date<game.source_game_date,
                    or_(Game.home_team_id==tid,Game.away_team_id==tid))
                    .order_by(Game.game_date.desc()).limit(10))).scalars().all()
                quality[side]=dict(recent_games=len(history),
                    missing_team_stats=sum(not h.team_stats_complete for h in history),
                    missing_player_stats=sum(not h.player_stats_complete for h in history),
                    latest_game_day=str(history[0].source_game_date) if history else None,
                    latest_team_stats_complete=history[0].team_stats_complete if history else False)
            gap=any(not q['latest_team_stats_complete'] for q in quality.values())
            row=dict(game_id=game.id,nba_game_id=game.nba_game_id,source_game_date=str(game.source_game_date),
                game_date=game.game_date.isoformat(),home=teams[game.home_team_id],away=teams[game.away_team_id],
                actual_total=game.home_score+game.away_score,totals=totals,
                latest_stats_gap=gap,quality=quality,warnings=fv.data_quality_warnings,
                feature_names=FEATURE_NAMES,feature_vector=fv.features.tolist(),
                feature_sha256=hashlib.sha256(np.asarray(fv.features,dtype='<f8').tobytes()).hexdigest(),
                mode='reconstructed_replay',exact_original_pregame_reproduction=False)
            rows.append(row)
    names=list(rows[0]['totals']) if rows else []
    summary={n:metrics(rows,n) for n in names}
    slices={label:{n:metrics([r for r in rows if r['latest_stats_gap']==flag],n) for n in names}
            for label,flag in [('latest_stats_complete',False),('latest_stats_gap',True)]}
    report=dict(mode='reconstructed_diagnostic_not_independent_test',fit_until=args.fit_until,
        generated_at=datetime.now(timezone.utc).isoformat(),reused_models=args.reuse,
        active_manifest=active,candidate_manifests=manifests,summary=summary,slices=slices,rows=rows,
        diagnostic_limitations=['Old release has no original ordered pregame vector; current compatibility reconstruction only',
            'Historical statistics publication timestamps are unknown; replay is not an archived real pregame forecast',
            'Official reconstructed injury inputs do not reproduce unarchived manual injury edits',
            'These recent games have already been inspected and cannot be a new independent acceptance test'],
        database_predictions_written=0,prospective_acceptance_started=False)
    assert active_path.read_bytes()==active_before
    report['active_manifest_unchanged']=True
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(summary=summary,slices=slices,rows=len(rows),active_unchanged=True),indent=2))


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature-cache-key',required=True)
    parser.add_argument('--fit-until',default='2026-10-21')
    parser.add_argument('--output',required=True)
    parser.add_argument('--reuse',help='Reuse frozen diagnostic models instead of refitting')
    parser.add_argument('--cohort-report',help='Use exactly the same games as a previous diagnostic')
    asyncio.run(main(parser.parse_args()))
