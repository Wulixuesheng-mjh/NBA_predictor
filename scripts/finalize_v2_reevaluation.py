"""Verify and seal a read-only retrospective review, without fitting or promotion."""
import ast
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/v2_reevaluation_20261110'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(name, payload):
    path = OUT / name
    if path.exists():
        raise ValueError(f'Preserve existing evidence: {path}')
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    results = read(OUT / 'results.json')
    context = read(OUT / 'context.json')
    protocol = read(OUT / 'protocol.json')
    supplement = read(OUT / 'supplement_results.json')
    for name, expected in protocol['inputs'].items():
        assert sha(ROOT / name) == expected, name
    old_folders = ['mlp_constant_repair_20261110', 'mlp_preflight_20261110',
                   'complete_history_20261104', 'expanded_model_comparison_20261105',
                   'total_distribution_20261105']
    integrity = {}
    for folder in old_folders:
        base = ROOT / 'reports' / folder
        manifest = read(base / 'artifact_checksums.json')
        for name, value in manifest.items():
            expected = value['sha256'] if isinstance(value, dict) else value
            assert sha(base / name) == expected, str(base / name)
        integrity[folder] = {'checked_files': len(manifest), 'all_match': True,
                             'manifest_sha256': sha(base / 'artifact_checksums.json')}
    scripts = ['export_v2_evaluation_context.py', 'reevaluate_models_v2.py',
               'supplement_v2_review.py', 'reevaluate_bias_v2.py', 'finalize_v2_reevaluation.py']
    for name in scripts:
        ast.parse((ROOT / 'scripts' / name).read_text(encoding='utf-8-sig'))
    total_records = read(OUT / 'totals_records.json')['rows']
    assert len(total_records) == 7
    for row in total_records:
        assert digest(row['payload']) == row['payload_hash']
        assert datetime.fromisoformat(row['as_of']) < datetime.fromisoformat(row['captured_start'])
        assert datetime.fromisoformat(row['created_at']) < datetime.fromisoformat(row['captured_start'])
        assert not row['payload']['acceptance_run']
        for candidate in row['payload']['shadow']['candidates']:
            assert candidate['training_data_until'] < row['payload']['source_game_date']
    record_counts = {}
    for folder, count, cutoff in [('win_evidence', 4, '2026-10-29'),
                                  ('paired_evidence', 2, '2026-10-21')]:
        paths = list((OUT / folder / 'records').glob('*.json'))
        assert len(paths) == count
        for path in paths:
            record = read(path)
            row = record['payload']
            assert digest(row) == record.get('payload_sha256', record.get('sha256'))
            for field in ['as_of', 'created_at']:
                assert datetime.fromisoformat(row[field]) < datetime.fromisoformat(row['scheduled_start_utc'])
            assert cutoff < row['source_game_date']
            assert not row['acceptance_run']
        record_counts[folder] = count
    checks = read(OUT / 'live_shadow_hash_checks.json')
    assert sum(len(v) for v in checks.values()) == 10
    assert all(all(v.values()) for v in checks.values())
    for mode in ('direct', 'residual'):
        assert read(OUT / f'live_{mode}_manifest.json') == read(ROOT / f'reports/release_candidate_20261029/{mode}_manifest.json')
    assert read(OUT / 'live_win_engineering.json')['parity_samples'] >= 100
    assert read(ROOT / 'reports/release_candidate_20261029/package_validation.json')['parity_samples'] == 125
    selector = read(OUT / 'rollout.json')['active_sha256']
    assert (OUT / 'selector_end.txt').read_text(encoding='utf-8-sig').strip() == selector
    assert results['formal_runs'] == results['formal_forecasts'] == 0
    assert results['promotion_allowed'] is False and context['writes'] == 0
    assert len(context['injury']) == 2 and all(row['game_id'] is None for row in context['injury'])
    assert context['reports'][0]['count'] == 0
    assert supplement['manual_repair_diagnostic']['unique_games'] == 1
    assert supplement['manual_repair_diagnostic']['snapshots'] == 2
    assert all(v['historical_development_screen'] for v in supplement['original_locked_reference'].values())
    assert all(v['repair_noninferiority_screen_pass'] for v in results['repair_reconstructed_three_heads'].values())
    assert all(v['samples'] == 7 and v['days'] == 3 and v['paired_day_95ci'][1] > 0 for v in results['live_totals'].values())
    assert results['live_win']['samples'] == 4 and results['live_win']['days'] == 2
    save('verification.json', {'status': 'PASS', 'protocol_source_hashes': 'PASS',
         'old_artifacts': integrity, 'syntax_files': scripts,
         'true_record_counts': {'totals': 7, **record_counts, 'manual_repair_unique_games': 1},
         'time_training_cutoffs_payload_integrity': 'PASS', 'live_model_files_match': 10,
         'active_selector_unchanged': selector, 'production_writes': 0,
         'scope': 'artifact and arithmetic evidence validation, not model certification'})
    rows = [
        dict(model='production_xgb_1022', role='baseline', admission='NOT_RECERTIFIED',
             evidence='7 real total diagnostics; MAE 12.4016', blocker='shared input/persistence chain'),
        dict(model='production_mlp_1022', role='legacy', admission='FAIL_CORRECTNESS',
             evidence='2 real paired games, raw total MAE 26.8319, guard rejected 1',
             blocker='known constant-coordinate inference mismatch remains in active version'),
        dict(model='repaired_mlp_constant_ast_v1', role='A_fix', admission='INSUFFICIENT_REAL_SAFETY',
             engineering='PASS_ARCHIVED', reconstructed_three_head_safety='PASS',
             real_manual_games=1, blocker='no formal safety cohort; shared chain'),
        dict(model='live_total_direct_T5', role='B_upgrade', admission='NOT_QUALIFIED',
             engineering='PASS_ARCHIVED_AND_LIVE_HASH', development_direction='INSUFFICIENT',
             real_games=7, real_days=3, real_loss_direction='INSUFFICIENT'),
        dict(model='live_total_residual_T4', role='B_upgrade', admission='NOT_QUALIFIED',
             engineering='PASS_ARCHIVED_AND_LIVE_HASH', development_direction='INSUFFICIENT',
             real_games=7, real_days=3, real_loss_direction='INSUFFICIENT'),
        dict(model='win_T5', role='B_upgrade', admission='NOT_QUALIFIED', engineering='PASS_ARCHIVED',
             original_reference_development='PASS_RETROSPECTIVE', same_cache_direction='INSUFFICIENT',
             real_games=4, real_days=2, historical_audit='DIRECTION_POSITIVE_UNCERTAIN'),
        dict(model='research_spread_T4', role='B_upgrade', admission='NOT_QUALIFIED',
             original_reference_development='PASS_RETROSPECTIVE', same_cache_direction='INSUFFICIENT'),
        dict(model='research_total_T1_and_median', role='B_upgrade', admission='NOT_QUALIFIED',
             original_reference_development='PASS_RETROSPECTIVE', expanded_2631_direction='INSUFFICIENT')]
    save('gate_matrix.json', {'rules': 'V2 applied retrospectively at user request; publisher unchanged',
         'rows': rows, 'shared_functional_chain': 'FAIL', 'formal_games': 0, 'formal_days': 0,
         'required_formal_games': 200, 'required_formal_days': 20,
         'probability_intervals': 'NOT_CERTIFIED', 'promotion_allowed': False})
    save('continuation_checkpoint.json', {'stage': 'V2 read-only re-evaluation completed',
         'score_snapshot_utc': context['checked_at'], 'data_revision': context['revision'],
         'production_selector_sha256': selector, 'production_changed': False,
         'report': 'docs/model-v2-reevaluation-2026-11-10.md',
         'next': ['Fix demonstrated manual-injury game binding and prediction persistence with zero-omission end-to-end cases',
                  'Keep repaired MLP behavior contract frozen; use separate real paired safety protocol',
                  'Continue diagnostic shadows without renaming past records as formal observations',
                  'Do not promote, retrain or expand parameter search based on this retrospective review'],
         'formal_acceptance_started': False, 'promotion_allowed': False})
    archive_paths = [ROOT / 'scripts' / name for name in scripts]
    archive_paths += [ROOT / 'docs/model-admission-v2-proposal-2026-11-10.md',
                      ROOT / 'docs/model-v2-reevaluation-2026-11-10.md']
    with zipfile.ZipFile(OUT / 'review_sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in archive_paths:
            archive.write(path, str(path.relative_to(ROOT)))
    save('artifact_checksums.json', {str(path.relative_to(OUT)).replace('\\', '/'): sha(path)
         for path in sorted(OUT.rglob('*')) if path.is_file() and path.name != 'artifact_checksums.json'})
    for name, value in read(OUT / 'artifact_checksums.json').items():
        assert sha(OUT / name) == value
    print(json.dumps({'status': 'PASS', 'sealed_files': len(read(OUT / 'artifact_checksums.json')),
                      'old_artifact_files_verified': sum(v['checked_files'] for v in integrity.values()),
                      'promotion_allowed': False}))


if __name__ == '__main__':
    main()
