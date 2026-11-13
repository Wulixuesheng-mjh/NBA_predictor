"""Paired cross-ablation comparisons; do not select using the audit set."""
import argparse
import json
from pathlib import Path
import numpy as np
from ml.src.evaluation import paired_day_bootstrap


def main(args):
    root, output = Path(args.directory), {}
    phases = ('development',) if args.development_only else ('development','audit')
    for phase in phases:
        reports = {name:json.loads((root/f'{phase}_E3_{name}_direct.json').read_text(encoding='utf-8'))
                   for name in ('no_injury','aggregate','net_only')}
        reference = reports['no_injury']['predictions']['xgboost:T0']
        output[phase] = {'metrics':{},'comparisons':{},'exposure':{}}
        for name,report in reports.items():
            rows = report['predictions']['xgboost:T0']
            assert [(r['nba_game_id'],r['actual']) for r in rows] == [(r['nba_game_id'],r['actual']) for r in reference]
            output[phase]['metrics'][name] = report['metrics']['xgboost:T0']
            relevant = ('2023-24','2024-25') if phase == 'development' else ('2025-26',)
            output[phase]['exposure'][name] = {s:report['injury_coverage'][s] for s in relevant}
            if name == 'no_injury':
                continue
            comparison = {}
            for target,index in (('win',0),('spread',1),('total',2)):
                def loss(values):
                    err = np.array([r['prediction'][index]-r['actual'][index] for r in values])
                    return err**2 if index == 0 else np.abs(err)
                before, after = loss(reference), loss(rows)
                comparison[target] = dict(loss_improvement=float((before.mean()-after.mean())/before.mean()),
                    loss_difference_95ci=paired_day_bootstrap(reference,before,after))
            output[phase]['comparisons'][name] = comparison
    if args.development_only:
        output['decision'] = dict(promote=False, lock_final_candidate=False,
            reason='Development diagnostic only; matched audit and prospective evidence pending')
        path = root/'development_summary.json'
        path.write_text(json.dumps(output,indent=2),encoding='utf-8')
        print(json.dumps(output,indent=2))
        return
    development = output['development']['exposure']
    no_development_injuries = not any(v['aggregate_nonzero'] or v['net_nonzero']
        for seasons in development.values() for v in seasons.values())
    output['decision'] = dict(promote=False, lock_final_candidate=False,
        reason='No credible injury variation in development; audit-only coverage cannot select the injury strategy'
            if no_development_injuries else 'Review prespecified release gates and prospective evidence',
        audit_is_pristine_test=False, prospective_completed=0)
    output['release_gates']={}
    for name in ('aggregate','net_only'):
        gates={}
        feature='aggregate_nonzero' if name=='aggregate' else 'net_nonzero'
        mode_variation=sum(row[feature] for row in output['development']['exposure'][name].values())>0
        for target in ('win','spread','total'):
            metric='brier' if target=='win' else 'mae'
            gains={}
            worst_slice=0.0
            rmse_ok=True
            for phase in ('development','audit'):
                base=output[phase]['metrics']['no_injury']
                candidate=output[phase]['metrics'][name]
                gains[phase]=(base[target][metric]-candidate[target][metric])/base[target][metric]
                if target!='win':
                    rmse_ok &= candidate[target]['rmse'] <= base[target]['rmse']*1.01
                for label,subset in base.get('slices',{}).items():
                    other=candidate.get('slices',{}).get(label)
                    if other and subset['samples']>=100:
                        worst_slice=max(worst_slice,(other[target][metric]-subset[target][metric])/subset[target][metric])
            interval=output['development']['comparisons'][name][target]['loss_difference_95ci']
            passes=bool(mode_variation and gains['development']>=.02 and gains['audit']>0
                        and rmse_ok and worst_slice<=.05 and interval[1]<0)
            gates[target]=dict(eligible_for_shadow=passes,development_improvement=gains['development'],
                audit_improvement=gains['audit'],rmse_gate=bool(rmse_ok),worst_eligible_slice_change=worst_slice,
                development_loss_difference_95ci=interval,prospective_still_required=True)
            gates[target]['has_development_variation']=mode_variation
        output['release_gates'][name]=gates
    output['decision']['any_target_eligible_for_shadow']=any(
        gate['eligible_for_shadow'] for targets in output['release_gates'].values() for gate in targets.values())
    path = root/'summary.json'
    path.write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',default='/app/ml/saved_models/ablations_v8_20261028')
    parser.add_argument('--development-only', action='store_true')
    main(parser.parse_args())
