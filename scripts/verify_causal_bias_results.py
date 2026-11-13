"""Replay every bias transform and verify last-fold labels cannot affect it."""
import json
from datetime import date
from pathlib import Path
from ml.src.config import ml_settings
from scripts import study_causal_mlp_bias as study


def main():
    root=Path(ml_settings.models_dir);folder=root/study.STUDY
    p=json.loads((folder/'protocol.json').read_text())['protocol']
    if study.protocol(root)!=p:raise ValueError('Bias protocol/source/inputs changed')
    result=json.loads((folder/'development.json').read_text());bases,_,_=study.inputs(root)
    if result['protocol_sha256']!=study.sha(folder/'protocol.json'):raise ValueError('Wrong result protocol')
    verified={}
    for name,source in bases.items():
        replay=study.apply_past_bias(source);saved=result['predictions'][name]
        if replay!=saved:raise ValueError('Bias transform does not replay exactly')
        final=max(r['fold'] for r in source)
        altered=[r|dict(actual_total=1000000.) if r['fold']==final else dict(r) for r in source]
        changed=study.apply_past_bias(altered)
        if [r['prediction'] for r in replay]!=[r['prediction'] for r in changed]:
            raise ValueError('Current/future labels leaked into correction')
        for r in saved:
            c=r['calibration']
            if c['prior_count']>1000 or (c['enabled'] and c['prior_count']<200):raise ValueError('Wrong residual window size')
            if c['latest_prior_day'] and date.fromisoformat(c['latest_prior_day'])>date.fromisoformat(c['cutoff']):raise ValueError('Lag violated')
            if not c['enabled'] and (c['offset']!=0 or r['prediction']!=r['uncorrected_prediction']):raise ValueError('Warmup is not uncorrected')
        verified[name]=dict(replayed=len(saved),enabled=sum(r['calibration']['enabled'] for r in saved),
            max_point_difference=0,current_and_future_labels_invariant=True)
    report=dict(status='PASS',kind='historical_causal_bias_verification',models=verified,
        production_store_written=False,promoted=False,formal_acceptance_added=0)
    study.publish(folder/'validation.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
