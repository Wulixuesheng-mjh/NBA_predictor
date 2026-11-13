"""Audit only targets that passed the immutable development lock; never retune."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from lock_complete_history_study import assess, sha


def main(args):
    root=Path(args.directory);lock_path=root/'development_lock.json'
    lock=json.loads(lock_path.read_text())
    for mode,digest in lock['source_sha256'].items():
        if sha(root/f'development_{mode}.json')!=digest: raise ValueError('Locked development report changed')
    allowed={t:lock['selected'][t] for t,g in lock['release_gates'].items() if g['development_gate_pass']}
    if not allowed: raise ValueError('No target passed development')
    reference=Path(args.reference)
    protocol=dict(allowed_targets=allowed,development_lock_sha256=sha(lock_path),
        audit_reference_sha256=sha(reference),selection='development only',executor_sha256=sha(__file__),promotion=False)
    with (root/'audit_protocol.json').open('x') as stream: json.dump(protocol,stream,indent=2)
    outputs={}
    for choice in sorted(set(allowed.values())):
        mode,model,policy=choice.split(':')
        if model!='xgboost': raise ValueError('Unexpected model choice')
        output=root/f'audit_{mode}_{policy}.json'
        if output.exists(): raise ValueError('Audit output is immutable')
        subprocess.run([sys.executable,str(Path(__file__).with_name('run_rolling_evaluation.py')),
            '--phase','audit','--features','full','--injuries','none','--total-strategy',mode,
            '--policies',policy,'--fit-jobs','2','--feature-cache-key','bbeb026e149ba7b9',
            '--cohort-reference',str(reference),'--output',str(output)],check=True)
        outputs[choice]=output
    gates=deepcopy(lock['release_gates'])
    reference_rows=json.loads(reference.read_text())['predictions']['xgboost:T0']
    for target,choice in allowed.items():
        mode,model,policy=choice.split(':')
        report=json.loads(outputs[choice].read_text())
        evidence=assess(reference_rows,report['predictions'][f'{model}:{policy}'],target)
        gates[target]['audit_improvement']=evidence['development_improvement']
        gates[target]['audit_metrics']=evidence['candidate']
        gates[target]['audit_difference_95ci']=evidence['development_loss_difference_95ci']
        gates[target]['worst_eligible_slice_change']=max(gates[target]['worst_eligible_slice_change'],evidence['worst_eligible_slice_change'])
        gates[target]['rmse_gate'] &= evidence['rmse_gate']
        gates[target]['eligible_for_shadow']=bool(gates[target]['development_gate_pass'] and
            evidence['development_improvement']>0 and gates[target]['rmse_gate'] and gates[target]['worst_eligible_slice_change']<=.05)
    result=dict(release_gates=gates,protocol=protocol,
        audit_sha256={k:sha(v) for k,v in outputs.items()},promotable_bundle=False,
        prospective_completed=0,promotion=False)
    with (root/'audit_summary.json').open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--reference',required=True)
    main(p.parse_args())
