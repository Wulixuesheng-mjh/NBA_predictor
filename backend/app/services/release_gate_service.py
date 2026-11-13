"""Fail-closed promotion evidence, including real database-backed acceptance."""
import hashlib
import json
from pathlib import Path
from sqlalchemy import select


def referenced_report(root,reference):
    path=(root/reference['report']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Release evidence outside model root or missing')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
        raise ValueError('Release evidence checksum changed')
    return json.loads(path.read_text(encoding='utf-8'))


def check_offline_gate(report,path,target):
    if not path or path[0] not in ('gates','release_gates'):
        raise ValueError('Unrecognized offline gate schema')
    if (path[0]=='gates' and target!='total') or (path[0]=='release_gates' and path[-1]!=target):
        raise ValueError('Offline gate target mismatch')
    gate=report
    for part in path:
        gate=gate[part]
    development=gate.get('development_improvement',gate.get('development',{}).get('mae_improvement',0))
    audit=gate.get('audit_improvement',gate.get('audit',{}).get('mae_improvement',0))
    interval=gate.get('development_loss_difference_95ci',gate.get('development',{}).get('loss_difference_95ci'))
    worst=gate.get('worst_eligible_slice_change',gate.get('worst_slice_degradation',1))
    if not (gate.get('eligible_for_shadow') and development>=.02 and audit>0 and
            gate.get('rmse_gate') and worst<=.05 and interval and interval[1]<0):
        raise ValueError(f'{target}: offline release gates not satisfied')


def validate_offline_evidence(root,candidate,references):
    targets=candidate.get('changed_targets',['win','spread','total'])
    if not targets or not set(targets)<=set(('win','spread','total')):
        raise ValueError('Changed prediction targets must be explicit')
    if not references or not set(targets)<=references.keys():
        raise ValueError('Passed offline gate reports required for every changed target')
    for target in targets:
        ref=references[target]
        check_offline_gate(referenced_report(root,ref),ref['gate_path'],target)
    return targets


def check_changed_targets(root,active,candidate,targets):
    """Declarations cannot hide changed heads or changed preprocessing."""
    all_targets={'win','spread','total'}
    preprocessing_changed = any(candidate.get(key) != active.get(key) for key in (
        'feature_schema_sha256', 'feature_semantic_version', 'feature_dim', 'feature_names',
        'calibration_enabled')) or set(candidate.get('disabled_features', [])) != set(active.get('disabled_features', []))
    if preprocessing_changed:
        if set(targets)!=all_targets:
            raise ValueError('Changed preprocessing requires gates for all prediction targets')
    for target in all_targets:
        filename=f'{target}_model.json'
        old=Path(root)/'xgboost'/active['xgboost_version']/filename
        previous=hashlib.sha256(old.read_bytes()).hexdigest() if old.is_file() else None
        current=candidate.get('file_hashes',{}).get('xgboost',{}).get(filename)
        if current!=previous and target not in targets:
            raise ValueError(f'Changed {target} model was omitted from release gates')
    if candidate.get('mlp_version')!=active.get('mlp_version') and set(targets)!=all_targets:
        raise ValueError('Changed auxiliary model requires full-target validation')


async def validate_promotion(root,version,evidence_path):
    from app.database import async_session_factory
    from app.models.prospective import ProspectiveRun,ProspectiveForecast
    from app.models.game import Game
    from app.services.prospective_service import digest,progress_report,feature_code_hash,validate_record_inputs
    from ml.src.features.dataset_cache import CACHE_FORMAT_VERSION
    from ml.src.features.feature_builder import FEATURE_NAMES
    root=Path(root).resolve()
    path=root/'candidates'/f'{version}.json'
    candidate=json.loads(path.read_text())
    proof=json.loads(Path(evidence_path).read_text())
    if proof['candidate_version']!=version or proof['candidate_manifest_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Evidence does not identify this exact candidate manifest')
    if candidate.get('feature_semantic_version')!=CACHE_FORMAT_VERSION or candidate.get('feature_names')!=FEATURE_NAMES:
        raise ValueError('Candidate feature preprocessing is incompatible')
    targets=validate_offline_evidence(root,candidate,proof.get('offline_gates'))
    check_changed_targets(root,json.loads((root/'active.json').read_text()),candidate,targets)
    engineering=referenced_report(root,proof['engineering'])
    if engineering.get('status')!='PASS' or engineering.get('version')!=version or engineering.get('parity_samples',0)<100:
        raise ValueError('Candidate engineering validation incomplete')
    if engineering.get('rejected_outputs_on_parity_samples',{}).get('xgboost',1)!=0:
        raise ValueError('Candidate has rejected validation predictions')
    if candidate.get('mlp_version') and engineering.get('mlp_pytorch_onnx_batch_single_parity') is not True:
        raise ValueError('MLP runtime parity evidence missing')
    async with async_session_factory() as session:
        run=await session.get(ProspectiveRun,proof['prospective_run_id'])
        if run is None or run.protocol_hash!=digest(run.protocol):
            raise ValueError('No matching locked formal acceptance protocol')
        if run.protocol.get('feature_code_hash')!=feature_code_hash():
            raise ValueError('Inference code differs from the locked acceptance protocol')
        if run.protocol['candidate']['version']!=version or run.protocol['candidate'].get('file_hashes')!=candidate.get('file_hashes'):
            raise ValueError('Acceptance candidate differs from promotion candidate')
        active=json.loads((root/'active.json').read_text())
        if run.protocol['baseline']['version']!=active['version']:
            raise ValueError('Production baseline changed since acceptance')
        if hashlib.sha256((root/'active.json').read_bytes()).hexdigest()!=run.protocol.get('active_manifest_sha256'):
            raise ValueError('Production selector differs from acceptance baseline')
        pairs=(await session.execute(select(ProspectiveForecast,Game).join(Game,
            Game.id==ProspectiveForecast.game_id).where(ProspectiveForecast.run_id==run.id))).all()
        completed=[(r,g) for r,g in pairs if g.status=='Final' and g.score_complete]
        if len(completed)<200:
            raise ValueError(f'Formal acceptance has {len(completed)}/200 settled games')
        for record,game in completed:
            if not (record.as_of<record.scheduled_start_utc and record.created_at<record.scheduled_start_utc):
                raise ValueError('Backfilled acceptance prediction is ineligible')
            if record.feature_hash!=digest(dict(names=record.feature_names,values=record.feature_vector)):
                raise ValueError('Acceptance feature snapshot changed')
            if record.data_quality.get('protocol_hash')!=run.protocol_hash:
                raise ValueError('Acceptance protocol provenance changed')
            validate_record_inputs(record, run.protocol)
            if str(game.source_game_date)<=candidate['training_data_until']:
                raise ValueError('In-training game included in acceptance')
        report=await progress_report(session,run)
        for target in targets:
            if report['paired_loss_difference_95ci'][target][0]>0:
                raise ValueError(f'{target}: acceptance demonstrates degradation')
    return dict(status='PASS',candidate=version,formal_completed=len(completed),targets=targets)
