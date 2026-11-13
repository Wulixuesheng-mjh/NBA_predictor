"""Lock total candidates on development before reading historical audit results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from ml.src.evaluation import paired_day_bootstrap, regression_metrics


def load(root, phase, mode):
    return json.loads((root / f'{phase}_{mode}.json').read_text(encoding='utf-8'))


def rows(report, policy):
    return report['predictions'][f'xgboost:{policy}']


def compare(reference, candidate):
    assert [(r['nba_game_id'], r['actual']) for r in reference] == [
        (r['nba_game_id'], r['actual']) for r in candidate], 'Unmatched samples'
    a = np.array([abs(r['prediction'][2] - r['actual'][2]) for r in reference])
    b = np.array([abs(r['prediction'][2] - r['actual'][2]) for r in candidate])
    return dict(mae_improvement=float((a.mean()-b.mean())/a.mean()),
                loss_difference_95ci=paired_day_bootstrap(reference, a, b))


def phase_summary(reports):
    reference = rows(reports['direct'], 'T0')
    result = dict(candidates={}, baseline=regression_metrics(
        np.array([r['actual'][2] for r in reference]),
        np.array([r['baseline_total'] for r in reference])))
    for mode, report in reports.items():
        for key, metric in report['metrics'].items():
            policy = key.split(':')[1]
            sample = rows(report, policy)
            entry = dict(samples=metric['samples'], total=metric['total'],
                         slices={k:v['total'] | {'samples':v['samples']}
                                 for k,v in metric['slices'].items()},
                         vs_direct_T0=compare(reference, sample))
            if mode == 'residual' and f'xgboost:{policy}' in reports['direct']['predictions']:
                entry['vs_same_weight_direct'] = compare(rows(reports['direct'], policy), sample)
            entry['fold_weights'] = [dict(month=f['month'], **f['weights'])
                for f in report['folds'] if f.get('model') == key and 'weights' in f]
            entry['duplicate_folds'] = [dict(month=f['month'], duplicate_of=f['duplicate_of'])
                for f in report['folds'] if f.get('model') == key and 'duplicate_of' in f]
            result['candidates'][f'{mode}:{policy}'] = entry
    return result


def source_hash(root, phase, mode):
    return hashlib.sha256((root/f'{phase}_{mode}.json').read_bytes()).hexdigest()


def main(args):
    root = Path(args.directory)
    reports = {mode:load(root, 'development', mode) for mode in ('direct','residual')}
    for mode, report in reports.items():
        assert report['protocol']['total_strategy'] == mode
        assert report['protocol']['feature_cache_key'] == reports['direct']['protocol']['feature_cache_key']
        assert report['protocol']['injuries'] == 'none'
        assert report['metrics'], 'Incomplete development run'
    dev = phase_summary(reports)
    lock_path = root/'total_selection_lock.json'
    if args.lock:
        if lock_path.exists():
            raise ValueError('Selection already locked; do not overwrite after viewing audit')
        selected = {mode:min((k for k in dev['candidates'] if k.startswith(mode+':')),
                            key=lambda k: dev['candidates'][k]['total']['mae'])
                    for mode in reports}
        lock = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    selection_source='development MAE only', selected_by_mode=selected,
                    overall=min(selected.values(), key=lambda k:dev['candidates'][k]['total']['mae']),
                    source_sha256={m:source_hash(root,'development',m) for m in reports},
                    protocol=reports['direct']['protocol'],
                    thresholds=dict(development_improvement=.02, rmse_max_change=.01,
                                    eligible_slice_min_samples=100, slice_max_degradation=.05),
                    promote=False)
        with lock_path.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(lock,indent=2))
        (root/'development_summary.json').write_text(json.dumps(dev,indent=2),encoding='utf-8')
        print(json.dumps(lock,indent=2))
        return
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    assert all(source_hash(root,'development',m)==lock['source_sha256'][m] for m in reports)
    audit_reports={m:load(root,'audit',m) for m in reports}
    for mode, report in audit_reports.items():
        assert report['protocol']['total_strategy']==mode
        assert report['protocol']['feature_cache_key']==lock['protocol']['feature_cache_key']
        assert report['protocol']['injuries']=='none'
        assert report['metrics'], 'Incomplete audit run'
    audit=phase_summary(audit_reports)
    gates={}
    for mode,key in lock['selected_by_mode'].items():
        d, a = dev['candidates'][key], audit['candidates'][key]
        worst=0.0
        rmse_ok=True
        for phase in (dev,audit):
            base, candidate=phase['candidates']['direct:T0'],phase['candidates'][key]
            rmse_ok &= candidate['total']['rmse'] <= base['total']['rmse']*1.01
            for label, subset in base['slices'].items():
                if subset['samples']>=100:
                    other=candidate['slices'][label]
                    assert subset['samples']==other['samples']
                    worst=max(worst,other['mae']/subset['mae']-1)
        evidence = (d['vs_direct_T0']['mae_improvement']>=.02 and
                    a['vs_direct_T0']['mae_improvement']>0 and rmse_ok and worst<=.05 and
                    d['vs_direct_T0']['loss_difference_95ci'][1]<0)
        gates[key]=dict(eligible_for_shadow=bool(evidence), rmse_gate=bool(rmse_ok),
                       worst_slice_degradation=worst, development=d['vs_direct_T0'],
                       audit=a['vs_direct_T0'], prospective_still_required=True)
    result=dict(lock=lock,development=dev,audit=audit,gates=gates,
                decision=dict(promote=False,prospective_completed=0,
                    reason='Offline evidence only; no production release or prospective acceptance'))
    (root/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(selection=lock['selected_by_mode'],gates=gates),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--lock', action='store_true')
    main(parser.parse_args())
