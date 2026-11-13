"""Monthly nested stop-selection/refit evaluation on the shared causal cache.

Development policies are selected only on 2023-24/2024-25 folds. Run audit
separately with the locked choices; 2025-26 is a historical audit, not pristine
unseen evidence. Predictions and fold boundaries are saved for day bootstrap.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ml.src.evaluation import (
    classification_metrics, complete_day_split, monthly_folds,
    paired_day_bootstrap, regression_metrics,
)
from ml.src.features.feature_builder import FEATURE_NAMES, FeatureBuilder
from ml.src.models.xgboost_model import XGBoostTrainer
from ml.src.config import ml_settings
from ml.src.features.player_features import PLAYER_FEATURE_NAMES
from ml.src.features.injury_net_features import INJURY_NET_FEATURE_NAMES
from ml.src.training_policy import POLICIES, policy_mask_and_weights, subset_arrays

LOG = logging.getLogger(__name__)


def summarize(rows):
    actual = np.asarray([r['actual'] for r in rows])
    pred = np.asarray([r['prediction'] for r in rows])
    return {
        'samples': len(rows),
        'win': classification_metrics(actual[:, 0], pred[:, 0]),
        'spread': regression_metrics(actual[:, 1], pred[:, 1]),
        'total': regression_metrics(actual[:, 2], pred[:, 2]),
        'implausible_output_count': int(np.sum(
            (np.abs(pred[:, 1]) > 45) | (pred[:, 2] < 150) | (pred[:, 2] > 300)
        )),
    }


def predictions(trainer, dataset):
    values = trainer.predict(dataset.X)
    if not isinstance(values, list):
        values = [values]
    return [
        {
            **info,
            'early_season': bool(
                dataset.X[i, FEATURE_NAMES.index('home_season_games_played')] < 10
                or dataset.X[i, FEATURE_NAMES.index('away_season_games_played')] < 10
            ),
            'actual': [float(w), float(s), float(t)],
            'baseline_total': float(dataset.X[i, FEATURE_NAMES.index('shrunk_expected_total')]),
            'prediction': [float(p.win_prob_home), float(p.predicted_spread), float(p.predicted_total)],
        }
        for i, (info, w, s, t, p) in enumerate(zip(
            dataset.sample_info, dataset.y_win, dataset.y_spread, dataset.y_total, values
        ))
    ]


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def freeze_cohort_metadata(row,month,cohort):
    row['fold']=month
    if cohort is not None:
        original=cohort[row['nba_game_id']]
        assert original['fold']==month and original['actual']==row['actual']
        assert str(original.get('source_game_date',original['game_date']))[:10]==str(row.get('source_game_date',row['game_date']))[:10]
        for flag in ('early_season','major_injury','incomplete_injury_report','injury_net_unavailable'):
            row[flag]=original.get(flag,False)


async def run(args):
    if args.feature_cache_key:
        from ml.src.features.dataset_cache import DatasetCache
        dataset = DatasetCache().load(args.feature_cache_key, len(FEATURE_NAMES))
        if dataset is None:
            raise ValueError('Pinned feature cache is missing or incompatible')
    else:
        dataset = await FeatureBuilder().build_training_dataset(min_games_before=5, use_cache=True)
    # Define slices from the original input before masking injury groups.
    # Otherwise the no-injury control would silently lose its injury slices.
    for i,info in enumerate(dataset.sample_info):
        info['major_injury']=any(dataset.X[i,FEATURE_NAMES.index(f'{side}_key_player_out')]>.5
                                  for side in ('home','away'))
        info['incomplete_injury_report']=any(dataset.X[i,FEATURE_NAMES.index(f'{side}_injury_report_complete')]<.5
                                             for side in ('home','away'))
        info['injury_net_unavailable']=any(dataset.X[i,FEATURE_NAMES.index(f'{side}_injury_net_available')]<.5
                                           for side in ('home','away'))
    # All ablations keep the ordered 213 columns, disabling only named groups.
    aggregate_names = {f'{side}_{name}' for side in ('home', 'away') for name in PLAYER_FEATURE_NAMES}
    net_names = {f'{side}_{name}' for side in ('home', 'away') for name in INJURY_NET_FEATURE_NAMES}
    for index, name in enumerate(FEATURE_NAMES):
        disable_prior = args.features == 'core' and (
            'prior_' in name or 'bayes_' in name or name == 'shrunk_expected_total'
        )
        disable_net = (args.features != 'full' or args.injuries in ('none', 'aggregate')) and name in net_names
        disable_aggregate = args.injuries in ('none', 'net') and name in aggregate_names
        if disable_prior or disable_net or disable_aggregate:
            dataset.X[:, index] = 0
    seasons = {'2023-24', '2024-25'} if args.phase == 'development' else {'2025-26'}
    cohort_path=getattr(args,'cohort_reference',None)
    cohort_rows=json.loads(Path(cohort_path).read_text())['predictions']['xgboost:T0'] if cohort_path else None
    cohort={r['nba_game_id']:r for r in cohort_rows} if cohort_rows is not None else None
    report = {
        'protocol': {
            'phase': args.phase, 'seasons': sorted(seasons), 'features': args.features,
            'monthly_fixed_model': True, 'inner_complete_day_validation_ratio': 0.15,
            'full_past_refit': True, 'validation_weighted': False,
            'weight_normalization': 'mean one among positive rows at each model fit boundary',
            'executor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_source_sha256': {str(p.relative_to(Path(__file__).resolve().parents[1])):
                hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path(__file__).resolve().parents[1]/'ml/src/models/xgboost_model.py',
                    Path(__file__).resolve().parents[1]/'ml/src/models/mlp_model.py',
                    Path(__file__).resolve().parents[1]/'ml/src/training_policy.py')},
            'audit_is_pristine_test': False, 'feature_dim': len(FEATURE_NAMES),
            'mlp_input_normalization': ml_settings.mlp_input_normalization,
            'mlp_training_params': {name:getattr(ml_settings,name) for name in (
                'mlp_epochs','mlp_batch_size','mlp_learning_rate','mlp_patience','mlp_dropout',
                'mlp_weight_decay','mlp_huber_delta_spread','mlp_huber_delta_total','random_seed')}
                if args.include_mlp else None,
            'mlp_policy_selection': 'logloss + spread_MAE/12 + total_MAE/18' if args.include_mlp else None,
            'feature_cache_key': args.feature_cache_key,
            'injuries': args.injuries,
            'nba_source_day_grouping': True,
            'slices_from_unmasked_features': True,
            'total_strategy': 'direct' if args.direct_total else args.total_strategy,
            'policies': args.policies,
            'xgboost_params': XGBoostTrainer()._common_params(),
            'fit_jobs_override': getattr(args,'fit_jobs',None),
            'cohort_reference_sha256': hashlib.sha256(Path(cohort_path).read_bytes()).hexdigest() if cohort_path else None,
            'cohort_slice_flags': 'frozen reference flags' if cohort_path else 'causal feature flags',
        },
        'folds': [], 'predictions': {}, 'metrics': {},
        'injury_coverage': {},
    }
    for season in sorted({row['season'] for row in dataset.sample_info}):
        mask = np.array([row['season'] == season for row in dataset.sample_info])
        report['injury_coverage'][season] = {group: int(sum(np.sum(np.abs(dataset.X[mask,
            FEATURE_NAMES.index(f'{side}_{feature}')]) > 1e-8) for side in ('home','away')))
            for group,feature in (('aggregate_nonzero','injury_firepower_loss'),
                                  ('net_nonzero','injury_net_offense_delta'))}
    if args.resume and Path(args.output).exists():
        previous = json.loads(Path(args.output).read_text(encoding='utf-8'))
        if previous['protocol'] != report['protocol']:
            raise ValueError('Cannot resume a report with a different evaluation protocol')
        report = previous
    baseline_index = FEATURE_NAMES.index('shrunk_expected_total')
    for month, train_mask, test_mask in monthly_folds(dataset.sample_info, seasons):
        if cohort is not None:
            test_mask &= np.array([row['nba_game_id'] in cohort for row in dataset.sample_info])
            if not test_mask.any():
                continue
        past = subset_arrays(dataset, train_mask)
        test = subset_arrays(dataset, test_mask)
        expected_names = [f'xgboost:{p}' for p in args.policies]
        if args.include_mlp:
            expected_names += [f'mlp:{p}' for p in args.policies if p != 'T0']
        expected_ids = [row['nba_game_id'] for row in test.sample_info]
        if all([
            row['nba_game_id'] for row in report['predictions'].get(name, [])
            if row['fold'] == month
        ] == expected_ids for name in expected_names):
            LOG.info('Resuming: %s already complete', month)
            continue
        # Atomic writes occur after each policy; discard an interrupted month
        # before recomputing it to avoid duplicate prediction rows.
        for name in list(report['predictions']):
            report['predictions'][name] = [r for r in report['predictions'][name] if r['fold'] != month]
        report['folds'] = [r for r in report['folds'] if r['month'] != month]
        seen = {}
        mlp_seen = {}
        for policy in args.policies:
            mask, weights, weight_report = policy_mask_and_weights(past.sample_info, policy)
            chosen = subset_arrays(past, mask)
            if len(chosen.X) < 1000:
                continue
            key = hashlib.sha256(mask.tobytes() + np.round(weights, 8).tobytes()).hexdigest()
            name = f'xgboost:{policy}'
            if key in seen:
                prior_name = seen[key]
                report['predictions'].setdefault(name, []).extend(
                    [dict(row) for row in report['predictions'][prior_name] if row['fold'] == month]
                )
                report['folds'].append({'month': month, 'model': name, 'duplicate_of': prior_name})
            else:
                seen[key] = name
                split = complete_day_split(chosen.sample_info)
                model = XGBoostTrainer(version=f'rolling-{month}-{policy}')
                if getattr(args,'fit_jobs',None) is not None:
                    original=model._common_params
                    jobs=args.fit_jobs
                    model._common_params=lambda original=original,jobs=jobs: original()|dict(n_jobs=jobs)
                LOG.info('%s %s: %d past / %d inner validation / %d test',
                         month, policy, split, len(chosen.X)-split, len(test.X))
                strategy = 'direct' if args.direct_total else args.total_strategy
                use_baseline = args.features != 'core' and strategy != 'direct'
                model.train(
                    chosen.X[:split], chosen.y_win[:split], chosen.y_spread[:split], chosen.y_total[:split],
                    chosen.X[split:], chosen.y_win[split:], chosen.y_spread[split:], chosen.y_total[split:],
                    sample_weight=weights[:split],
                    total_baseline_train=chosen.X[:split, baseline_index] if use_baseline else None,
                    total_baseline_val=chosen.X[split:, baseline_index] if use_baseline else None,
                    total_baseline_feature_index=baseline_index if use_baseline else None,
                    total_strategy=strategy,
                )
                model.refit_full(
                    chosen.X, chosen.y_win, chosen.y_spread, chosen.y_total,
                    sample_weight=weights,
                    total_baseline=chosen.X[:, baseline_index] if use_baseline else None,
                )
                rows = predictions(model, test)
                for row in rows:
                    freeze_cohort_metadata(row,month,cohort)
                report['predictions'].setdefault(name, []).extend(rows)
                report['folds'].append({
                    'month': month, 'model': name, 'weights': weight_report,
                    'selection_fit_until': chosen.sample_info[split-1]['game_date'],
                    'selection_validation_from': chosen.sample_info[split]['game_date'],
                    'fit_until': chosen.sample_info[-1]['game_date'],
                    'test_from': test.sample_info[0]['game_date'],
                    'test_until': test.sample_info[-1]['game_date'],
                    'total_mode': model._total_mode,
                })
            if args.include_mlp and policy != 'T0':
                from ml.src.models.mlp_model import MLPTrainer
                mlp_name = f'mlp:{policy}'
                duplicate_mlp = mlp_seen.get(key)
                # T0 is not an MLP candidate; train its equivalent T1 once.
                if duplicate_mlp is not None:
                    report['predictions'].setdefault(mlp_name, []).extend(
                        [dict(r) for r in report['predictions'][duplicate_mlp] if r['fold'] == month]
                    )
                else:
                    mlp_seen[key] = mlp_name
                    split = complete_day_split(chosen.sample_info)
                    mlp = MLPTrainer(version=f'rolling-{month}-{policy}', device='cpu')
                    metrics = mlp.train(
                        chosen.X[:split], chosen.y_win[:split], chosen.y_spread[:split], chosen.y_total[:split],
                        chosen.X[split:], chosen.y_win[split:], chosen.y_spread[split:], chosen.y_total[split:],
                        sample_weight=weights[:split],
                    )
                    mlp.refit_full(
                        chosen.X, chosen.y_win, chosen.y_spread, chosen.y_total,
                        sample_weight=weights, best_epoch=metrics['best_epoch'], lr_schedule=metrics['lr_schedule'],
                    )
                    rows = predictions(mlp, test)
                    for row in rows:
                        freeze_cohort_metadata(row,month,cohort)
                    report['predictions'].setdefault(mlp_name, []).extend(rows)
                    report['folds'].append({
                        'month': month, 'model': mlp_name, 'weights': weight_report,
                        'fit_until': chosen.sample_info[-1]['game_date'],
                        'test_from': test.sample_info[0]['game_date'],
                        'best_epoch': metrics['best_epoch'],
                    })
            write_report(Path(args.output), report)
    for name, rows in report['predictions'].items():
        entry = summarize(rows)
        slices = {}
        for label, predicate in (
            ('playoffs', lambda r: r.get('game_type') == 'playoffs'),
            ('incomplete_stats', lambda r: not r.get('team_stats_complete')),
            ('early_season', lambda r: r.get('early_season', False)),
            ('major_injury', lambda r: r.get('major_injury',False)),
            ('incomplete_injury_report', lambda r: r.get('incomplete_injury_report',False)),
            ('injury_net_unavailable', lambda r: r.get('injury_net_unavailable',False)),
        ):
            subset = [r for r in rows if predicate(r)]
            if subset:
                slices[label] = summarize(subset)
        entry['slices'] = slices
        report['metrics'][name] = entry
        if cohort_rows is not None:
            assert [(r['nba_game_id'],r['actual']) for r in rows]==[(r['nba_game_id'],r['actual']) for r in cohort_rows]
            comparisons={}
            for target,index in (('win',0),('spread',1),('total',2)):
                def target_loss(values):
                    return np.array([(r['prediction'][index]-r['actual'][index])**2 if index==0
                                     else abs(r['prediction'][index]-r['actual'][index]) for r in values])
                before,after=target_loss(cohort_rows),target_loss(rows)
                comparisons[target]=dict(improvement=float(1-after.mean()/before.mean()),
                    difference_95ci=paired_day_bootstrap(cohort_rows,before,after))
            entry['vs_original_pipeline']=comparisons
    if not report['metrics']:
        raise ValueError('No eligible rolling folds; need at least 1000 prior samples')
    xgb = {k:v for k,v in report['metrics'].items() if k.startswith('xgboost:')}
    policy_key = 'selected_policies' if args.phase == 'development' else 'descriptive_best_policies_do_not_select'
    report[policy_key] = {
        'xgboost_win': min(xgb, key=lambda k: xgb[k]['win']['brier']).split(':')[1],
        'xgboost_spread': min(xgb, key=lambda k: xgb[k]['spread']['mae']).split(':')[1],
        'xgboost_total': min(xgb, key=lambda k: xgb[k]['total']['mae']).split(':')[1],
    }
    mlp = {k:v for k,v in report['metrics'].items() if k.startswith('mlp:')}
    if mlp:
        # Fixed training-era scales, independent of the evaluated month.
        report[policy_key]['mlp_shared'] = min(mlp, key=lambda k: (
            mlp[k]['win']['logloss'] + mlp[k]['spread']['mae']/12 + mlp[k]['total']['mae']/18
        )).split(':')[1]
    reference = report['predictions'].get('xgboost:T0')
    if reference:
        for name, rows in report['predictions'].items():
            if [r['nba_game_id'] for r in rows] != [r['nba_game_id'] for r in reference]:
                continue
            intervals = {}
            for target, index in (('win', 0), ('spread', 1), ('total', 2)):
                def loss(values):
                    return np.array([
                        (r['prediction'][index]-r['actual'][index])**2 if index == 0
                        else abs(r['prediction'][index]-r['actual'][index]) for r in values
                    ])
                intervals[target] = paired_day_bootstrap(reference, loss(reference), loss(rows))
            report['metrics'][name]['loss_difference_95ci_vs_T0'] = intervals
    write_report(Path(args.output), report)
    print(json.dumps({policy_key: report[policy_key], 'metrics': report['metrics']}, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['development', 'audit'], default='development')
    parser.add_argument('--policies', nargs='+', choices=sorted(POLICIES), default=sorted(POLICIES))
    parser.add_argument('--features', choices=['core', 'prior', 'full'], default='full')
    parser.add_argument('--direct-total', action='store_true')
    parser.add_argument('--total-strategy', choices=['auto', 'direct', 'residual'], default='auto')
    parser.add_argument('--injuries', choices=['both', 'none', 'aggregate', 'net'], default='both')
    parser.add_argument('--include-mlp', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--feature-cache-key')
    parser.add_argument('--cohort-reference',help='Pin identical games, labels and slice flags for data interventions')
    parser.add_argument('--fit-jobs',type=int,help='CPU-only scheduling override, recorded separately from model hyperparameters')
    parser.add_argument('--output', default='/app/ml/saved_models/rolling_development.json')
    asyncio.run(run(parser.parse_args()))
