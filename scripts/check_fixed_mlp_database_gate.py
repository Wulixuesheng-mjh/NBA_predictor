"""Only isolated DB may execute selector commands; production is read-only check."""
import asyncio,json,subprocess
from pathlib import Path
from app.config import settings
from scripts.fixed_mlp_release import require_compatible_database,sha


def main():
    isolated='nba-mlp-preflight-db-20261110' in settings.database_url
    if isolated:
        assert not settings.scheduler_enabled
        root=Path('/out/test_models');before=sha(root/'active.json')
        for action in ('activate','rollback'):
            result=subprocess.run(['python','/app/scripts/fixed_mlp_release.py',action,
                '--models-dir',str(root),'--package','/out/package'],capture_output=True,text=True)
            assert result.returncode==0,result.stdout+result.stderr
        assert sha(root/'active.json')==before
        proof=dict(status='PASS',scope='isolated actual CLI activate/rollback',production_changed=False)
        name='guarded_cli_acceptance.json'
    else:
        # Never call switch or any selector mutation against production here.
        root=Path('/app/ml/saved_models');before=sha(root/'active.json')
        try:asyncio.run(require_compatible_database())
        except ValueError as exc:
            assert '010_mlp_compat_version' in str(exc)
        else:raise AssertionError('Expected production 009 schema to fail activation precheck')
        assert sha(root/'active.json')==before
        proof=dict(status='PASS',scope='production read-only schema precheck',
            premature_activation_blocked=True,active_sha256=before,production_changed=False)
        name='production_schema_guard.json'
    Path('/out',name).write_text(json.dumps(proof,indent=2));print(json.dumps(proof))


if __name__=='__main__':main()
