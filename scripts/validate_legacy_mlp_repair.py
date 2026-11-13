"""Fixed-weight compatibility regression, not model search or pregame scoring."""
import argparse,importlib.util,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ml.src.predictor import load_predictor
from ml.src.models.mlp_model import MLPTrainer
from ml.src.legacy_mlp_compatibility import CompatibleLegacyMLP,restore_constants,VERSION,FILES,CONTRACT_ID
from ml.src.features.feature_builder import FeatureVector,FEATURE_NAMES
from ml.src.features.dataset_cache import DatasetCache
from ml.src.config import ml_settings


def values(model,X):
    output=model.predict(X)
    if not isinstance(output,list):output=[output]
    return np.array([[r.win_prob_home,r.predicted_spread,r.predicted_total] for r in output])


def main(args):
    output=Path(args.output)
    if output.exists():raise ValueError('Never overwrite verification')
    root=Path(ml_settings.models_dir);active=(root/'active.json').read_bytes()
    source=Path(args.reports)/'total_root_cause_20261109/frozen_runtime_diagnosis'
    rows=[json.loads(line) for line in (source/'reconstructed_rows.jsonl').read_text().splitlines()]
    X=np.array([r['processed_vector'] for r in rows]);before=X.copy()
    original=load_predictor();fixed=load_predictor(mlp_compatibility_path=args.contract)
    replay=load_predictor(release_manifest=fixed.release_manifest)
    baseline=values(original.mlp_trainer,X);corrected=values(fixed.mlp_trainer,X)
    assert np.array_equal(X,before)
    assert np.array_equal(values(replay.mlp_trainer,X),corrected)
    assert replay.mlp_trainer.version==fixed.mlp_trainer.version
    expected=np.array([r['centered_total'] for r in rows])
    reference_error=float(max(abs(corrected[:,2]-expected)))
    assert reference_error<1e-3
    single=np.vstack([values(fixed.mlp_trainer,x) for x in X])
    single_error=abs(single-corrected).max(axis=0)
    assert np.all(single_error<=[1e-5,1e-3,1e-3])
    original_outputs=[r['mlp_raw'] for r in rows]
    assert max(abs(baseline[:,2]-original_outputs))<1e-3
    # Original observed training coordinates are zero: predictions must be exactly preserved.
    cache=root/'feature_cache/train_99ec73146ff0f31e.npz'
    with np.load(cache,allow_pickle=False) as data:train=data['X']
    train=train[np.linspace(0,len(train)-1,100,dtype=int)]
    unchanged=float(abs(values(original.mlp_trainer,train)-values(fixed.mlp_trainer,train)).max())
    assert unchanged==0
    controls=[];rejects=0;snapshot_replays=0
    for row in rows:
        fv=FeatureVector(features=np.array(row['feature_vector']),home_games_available=10,away_games_available=10)
        old=original.predict(fv);new=fixed.predict(fv)
        assert old.xgboost==new.xgboost and new.recommended=='xgboost'
        assert new.data_quality['mlp_input_compatibility']==CONTRACT_ID
        assert new.mlp is not None and CONTRACT_ID in new.mlp.model_version
        assert not new.calibration_applied
        if hasattr(new,'processed_inputs'):
            from scripts.replay_prediction import replay_processed_row
            snap=new.processed_inputs['mlp']
            saved=SimpleNamespace(features_json={'_release_manifest':fixed.release_manifest,'_inference_snapshot':snap},
                calibrated=False,model_name='mlp',model_version=new.mlp.model_version,
                feature_names=snap['feature_names'],feature_vector=snap['feature_vector'])
            replayed,_=replay_processed_row(saved,replay)
            assert abs(replayed.predicted_total-new.raw_outputs['mlp']['predicted_total'])<1e-3
            snapshot_replays+=1
        rejects+=int(old.mlp is None);controls.append(new.to_dict()['mlp'])
    # Existing 221 earlier probes report remaining failures; do not extend the repair to fit them.
    audit=json.loads((Path(args.reports)/'paired_mlp_20261105/production_mlp_audit.json').read_text())
    data=DatasetCache().load('4e14a9886bed1bc1',len(FEATURE_NAMES))
    index={r['nba_game_id']:i for i,r in enumerate(data.sample_info)}
    probes=data.X[[index[r['game']] for r in audit['rows']]].copy().astype(float)
    probes[:,FEATURE_NAMES.index('expected_total')]/=2;probes=probes[:,:166]
    def accepted(array):
        return np.isfinite(array).all(axis=1)&(array[:,0]>=0)&(array[:,0]<=1)&(abs(array[:,1])<=45)&(array[:,2]>=150)&(array[:,2]<=300)
    old_probe=values(original.mlp_trainer,probes);fixed_probe=values(fixed.mlp_trainer,probes)
    old_accepted=accepted(old_probe);new_accepted=accepted(fixed_probe)
    remaining=[dict(game=r['game'],day=r['day'],before=old_probe[i].tolist(),after=fixed_probe[i].tolist())
        for i,r in enumerate(audit['rows']) if not new_accepted[i]]
    # Stress the corrected coordinates only; other out-of-domain coordinates retain their guard behavior.
    stress=np.repeat(X[:1],5,axis=0);stress[:,17]=[0,.2,.7,5,1e20];stress[:,50]=[0,.7,.2,5,1e20]
    stress_fixed=values(fixed.mlp_trainer,stress)
    assert float(np.max(abs(stress_fixed-stress_fixed[0])))==0
    y=np.array([r['actual_total'] for r in rows])
    metrics={label:dict(total_bias=float((a[:,2]-y).mean()),total_mae=float(abs(a[:,2]-y).mean()),
        total_rmse=float(np.sqrt(np.mean((a[:,2]-y)**2))),
        win_mean=float(a[:,0].mean()),spread_mean=float(a[:,1].mean())) for label,a in [('before',baseline),('after',corrected)]}
    pytorch=None
    if args.original_torch_source:
        import torch
        torch.set_num_threads(2)
        spec=importlib.util.spec_from_file_location('legacy_torch_original',args.original_torch_source)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        trainer=MLPTrainer();trainer.load(VERSION);trainer._onnx_session=None;trainer.device='cpu'
        trainer.model=module.NBA_MLP(input_dim=166)
        trainer.model.load_state_dict(torch.load(root/'mlp'/VERSION/'model.pt',map_location='cpu',weights_only=True))
        trainer.model.eval();wrapped=CompatibleLegacyMLP(trainer,args.contract)
        pt=values(wrapped,X);diff=abs(pt-corrected).max(axis=0)
        assert np.all(diff<=[1e-5,1e-3,1e-3])
        pytorch=dict(samples=len(X),max_difference=diff.tolist(),eval_mode=not trainer.model.training)
    assert (root/'active.json').read_bytes()==active
    report=dict(status='PASS',id=CONTRACT_ID,weight_training=False,production_promoted=False,
        historical_pregame_accuracy=False,cohort=len(X),metrics=metrics,
        reference_probe_max_difference=reference_error,single_batch_max_difference=single_error.tolist(),
        original_training_samples=100,original_training_max_difference=unchanged,
        xgb_all_outputs_unchanged=True,prior_guard_rejects=rejects,repair_guard_rejects=0,
        all_mlp_heads_changed_max=abs(corrected-baseline).max(axis=0).tolist(),
        earlier_probes=dict(samples=len(probes),before_rejected=int((~old_accepted).sum()),after_rejected=len(remaining),
            newly_rejected=int((old_accepted&~new_accepted).sum()),remaining=remaining),
        coordinate_stress_samples=5,coordinate_stress_invariant=True,pytorch_onnx=pytorch,
        protocol_input_preserved=True,primary_unchanged=True,
        exact_processed_snapshot_replays=snapshot_replays,
        artifact_sha256=FILES,per_game=[dict(game_id=r['game_id'],source_day=r['source_day'],actual_total=r['actual_total'],
            before=baseline[i].tolist(),after=corrected[i].tolist(),final=controls[i]) for i,r in enumerate(rows)])
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='per_game' and k!='artifact_sha256'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--contract',required=True);p.add_argument('--reports',required=True)
    p.add_argument('--output',required=True);p.add_argument('--original-torch-source');main(p.parse_args())
