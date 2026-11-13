"""Collect pinned-cache experiment evidence; do not infer release approval."""

import json
from pathlib import Path

from ml.src.config import ml_settings


def main():
    root = Path(ml_settings.models_dir)
    names = [
        'rolling_development.json', 'rolling_development_fixed.json', 'rolling_audit.json',
        'rolling_mlp_audit.json', 'ablations/E0_core_direct.json', 'ablations/E2_prior_direct.json',
        'ablations/E3_no_injury_direct.json', 'ablations/E3_net_only_direct.json',
    ]
    output = {}
    for name in names:
        if not (root / name).exists():
            continue
        report = json.loads((root / name).read_text())
        output[name] = {
            'complete': bool(report.get('metrics')),
            'selected_policies': report.get('selected_policies'),
            'last_month': report['folds'][-1]['month'] if report.get('folds') else None,
            'metrics': report.get('metrics', {}),
        }
        for model, metrics in output[name]['metrics'].items():
            rows = report['predictions'][model]
            metrics['output_ranges'] = {
                'spread_min': min(r['prediction'][1] for r in rows),
                'spread_max': max(r['prediction'][1] for r in rows),
                'total_min': min(r['prediction'][2] for r in rows),
                'total_max': max(r['prediction'][2] for r in rows),
            }
            metrics['invalid_outputs'] = sum(
                abs(r['prediction'][1]) > 45 or not 150 <= r['prediction'][2] <= 300
                for r in rows
            )
    path = root / 'experiment_summary_213.json'
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    compact = {}
    for name, entry in output.items():
        compact[name] = {**entry, 'metrics': {
            model: {
                'samples': metric['samples'], 'accuracy': metric['win']['accuracy'],
                'spread_mae': metric['spread']['mae'], 'total_mae': metric['total']['mae'],
                'total_bias': metric['total']['bias'], 'invalid': metric['invalid_outputs'],
            } for model, metric in entry['metrics'].items()
        }}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
