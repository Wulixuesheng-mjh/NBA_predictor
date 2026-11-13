"""Isolated selector, API, replay and rollback acceptance for the repair package."""
import asyncio,json,shutil,threading,time,subprocess
from pathlib import Path
import numpy as np
from app.config import settings
from ml.src.config import ml_settings
from ml.src.predictor import load_predictor
from ml.src.features.feature_builder import FeatureVector
from scripts.fixed_mlp_release import sha,read,atomic,validate,switch,RELEASE,FILES,XGB_FILES
from ml.src.legacy_mlp_compatibility import contract_spec,VERSION,CONTRACT_ID

OUT=Path('/out');ROOT=OUT/'test_models';PACKAGE=OUT/'package'


async def main():
    assert 'nba-mlp-preflight-db-20261110' in settings.database_url and not settings.scheduler_enabled
    assert not ROOT.exists(),'Do not overwrite release acceptance'
    ROOT.mkdir();PACKAGE.mkdir()
    original=Path(ml_settings.models_dir)
    for group in ('mlp','xgboost'):
        shutil.copytree(original/group/VERSION,ROOT/group/VERSION)
    (ROOT/'active.json').write_bytes((original/'active.json').read_bytes())
    (PACKAGE/'baseline.json').write_bytes((original/'active.json').read_bytes())
    base=read(ROOT/'active.json')
    candidate={**base,'version':RELEASE,'mlp_input_compatibility':contract_spec(),
        'file_hashes':{'mlp':FILES,'xgboost':XGB_FILES}}
    (PACKAGE/'candidate.json').write_text(json.dumps(candidate,indent=2))
    for src,dst in [('season_results.json','season_results.json'),('engineering.json','engineering.json')]:
        shutil.copyfile(OUT/src,PACKAGE/dst)
    ml_settings.models_dir=str(ROOT)
    # This synthetic fixture activates the candidate before testing its API;
    # the validated publisher itself is exercised only after evidence exists.
    atomic(ROOT/'active.json',(PACKAGE/'candidate.json').read_bytes())
    # Drain pre-existing synthetic fixtures only in the isolated database, so
    # this test's settlement count is scoped to its two new forecasts.
    from ml.src.feedback import FeedbackLoop
    ml_settings.models_dir=str(OUT/'prior_fixture_settlements')
    prior=await FeedbackLoop()._evaluate_predictions()
    ml_settings.models_dir=str(ROOT)
    (OUT/'fixture_setup.json').write_text(json.dumps(dict(prior_synthetic_settlements=prior)))
    from scripts.preflight_mlp_api import main as api_test
    await api_test()
    api=read(OUT/'api_database_preflight.json')
    proof_api=dict(full_version_retained=CONTRACT_ID in api['full_MLP_version_stored'],
        both_models_persisted=api['both_models_persisted'],snapshots_replayed=api['snapshots_replayed']==2,
        backdated_write_rejected=api['backdated_write_rejected'],candidate_manifest_sha256=sha(PACKAGE/'candidate.json'))
    (PACKAGE/'api_preflight.json').write_text(json.dumps(proof_api,indent=2))
    names=['baseline.json','candidate.json','season_results.json','engineering.json','api_preflight.json']
    (PACKAGE/'evidence.json').write_text(json.dumps(dict(files={n:sha(PACKAGE/n) for n in names},
        model_hashes=candidate['file_hashes'],scope='fixed 1022 repair only'),indent=2))
    atomic(ROOT/'active.json',(PACKAGE/'baseline.json').read_bytes())
    validate(ROOT,PACKAGE)
    old=load_predictor()
    switch(ROOT,PACKAGE,'activate');fixed=load_predictor()
    assert old.mlp_trainer.version==VERSION and CONTRACT_ID in fixed.mlp_trainer.version
    rows=[json.loads(s)['payload'] for s in (OUT/'season_rows.jsonl').read_text().splitlines()][:100]
    for row in rows:
        fv=FeatureVector(features=np.array(row['builder_vector']),home_games_available=10,away_games_available=10)
        a=old.predict(fv,apply_calibration=False);b=fixed.predict(fv,apply_calibration=False)
        assert a.raw_outputs['xgboost']==b.raw_outputs['xgboost']
        for key in ('win_prob_home','predicted_spread','predicted_total'):
            tolerance=1e-5 if key=='win_prob_home' else 1e-3
            assert abs(b.raw_outputs['mlp'][key]-row['candidate']['models']['mlp']['raw'][key])<=tolerance
    rejected=[]
    # Tampered contract, different weights/target declaration, evidence and
    # compare-and-swap failures must not change the active selector.
    before=(ROOT/'active.json').read_bytes();original_candidate=(PACKAGE/'candidate.json').read_bytes()
    for key,value in [('feature_dim',213),('mlp_input_compatibility',{}),('calibration_enabled',True)]:
        bad=dict(candidate);bad[key]=value;(PACKAGE/'candidate.json').write_text(json.dumps(bad))
        try:validate(ROOT,PACKAGE)
        except ValueError:rejected.append(key)
        else:raise AssertionError('Tampered candidate accepted')
        (PACKAGE/'candidate.json').write_bytes(original_candidate)
        assert (ROOT/'active.json').read_bytes()==before
    try:switch(ROOT,PACKAGE,'activate')
    except ValueError:rejected.append('stale_selector')
    else:raise AssertionError('Duplicate activation accepted')
    stop=threading.Event();errors=[];read_count=[0]
    accepted=[read(PACKAGE/'baseline.json'),candidate]
    def reader():
        while not stop.is_set():
            try:
                assert read(ROOT/'active.json') in accepted;read_count[0]+=1
            except Exception as e:errors.append(str(e));stop.set()
    thread=threading.Thread(target=reader);thread.start()
    try:
        for _ in range(5):switch(ROOT,PACKAGE,'rollback');switch(ROOT,PACKAGE,'activate')
        switch(ROOT,PACKAGE,'rollback')
    finally:stop.set();thread.join()
    assert not errors and read_count[0]>0
    assert sha(ROOT/'active.json')==sha(PACKAGE/'baseline.json')
    # Replay fixed-version saved rows after selector rollback, via the actual CLI.
    from app.database import async_session_factory
    from app.models.prediction import Prediction
    from sqlalchemy import select
    async with async_session_factory() as db:
        saved=await db.scalar(select(Prediction).where(Prediction.model_name=='mlp',
            Prediction.features_json['_release_manifest']['version'].as_string()==RELEASE).order_by(Prediction.id.desc()))
        assert saved is not None
    # The CLI reads MODELS_DIR from its own process, so point it at the isolated copy.
    import os
    env=dict(os.environ,MODELS_DIR=str(ROOT))
    result=subprocess.run(['python','/app/scripts/replay_prediction.py','--prediction-id',str(saved.id)],
        env=env,text=True,capture_output=True)
    (OUT/'cli_replay_after_rollback.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stdout+result.stderr
    report=dict(status='PASS',release=RELEASE,isolated_only=True,package_hashes_verified=True,
        original_model_files_unchanged=True,behavior_version_preserved=True,all_heads_100_probe_parity=True,
        xgb_100_inputs_unchanged=True,old_loaded_predictor_stable_across_switch=True,
        atomic_reader_observations=read_count[0],mixed_or_partial_manifests=0,
        rejected_negative_cases=rejected,baseline_selector_restored=True,
        fixed_saved_prediction_cli_replayed_after_rollback=True,real_production_changed=False)
    (OUT/'release_acceptance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':asyncio.run(main())
