"""Additional negative acceptance, strictly on isolated models and database."""
import asyncio,json,subprocess
from pathlib import Path
from sqlalchemy import text
from app.config import settings
from app.database import async_session_factory
from scripts.fixed_mlp_release import validate,sha

OUT=Path('/out');ROOT=OUT/'test_models';PACKAGE=OUT/'package'


def main():
    assert 'nba-mlp-preflight-db-20261110' in settings.database_url and not settings.scheduler_enabled
    before=sha(ROOT/'active.json');rejected=[]
    for path,label in [(ROOT/'mlp/20261022_050952/metadata.json','model_tamper'),
                        (PACKAGE/'season_results.json','evidence_tamper')]:
        original=path.read_bytes()
        try:
            path.write_bytes(original+b' ')
            try:validate(ROOT,PACKAGE)
            except ValueError:rejected.append(label)
            else:raise AssertionError('Tampering accepted')
        finally:path.write_bytes(original)
        assert sha(ROOT/'active.json')==before
    validate(ROOT,PACKAGE)
    result=subprocess.run(['alembic','downgrade','009_shadow_observations'],capture_output=True,text=True)
    (OUT/'unsafe_schema_shrink.log').write_text(result.stdout+result.stderr)
    assert result.returncode!=0 and 'Cannot shrink model_version' in result.stdout+result.stderr
    async def check():
        async with async_session_factory() as db:
            revision=await db.scalar(text('select version_num from alembic_version'))
            width=await db.scalar(text("select character_maximum_length from information_schema.columns where table_name='predictions' and column_name='model_version'"))
            assert revision=='010_mlp_compat_version' and width==128
            return revision,width
    revision,width=asyncio.run(check())
    proof=dict(status='PASS',rejected=rejected,selector_unchanged=True,unsafe_schema_shrink_rejected=True,
        schema_retained=revision,width_retained=width,production_changed=False)
    (OUT/'negative_guards.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))


if __name__=='__main__':main()
