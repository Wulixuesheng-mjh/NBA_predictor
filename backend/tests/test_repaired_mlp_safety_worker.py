import importlib.util,json
from pathlib import Path
import pytest


@pytest.fixture
def worker():
    path=Path('/app/scripts/repaired_mlp_safety_worker.py')
    spec=importlib.util.spec_from_file_location('repair_safety_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_cohort_requires_both_count_and_day_diversity(worker):
    assert not worker.full([{'source_game_date':'2026-11-12'}]*200)
    rows=[{'source_game_date':str(i%20)} for i in range(199)]
    assert not worker.full(rows)
    assert worker.full(rows+[{'source_game_date':'0'}])


def test_records_refuse_overwrite_and_tampering(worker,tmp_path):
    (tmp_path/'records').mkdir();p={'frozen':True}
    payload=dict(protocol_sha256=worker.digest(p),as_of='2026-11-12T00:00:00Z',
        created_at='2026-11-12T00:01:00Z',scheduled_start_utc='2026-11-12T01:00:00Z')
    path=tmp_path/'records/1.json'
    assert worker.publish(path,dict(payload=payload,sha256=worker.digest(payload)))
    assert not worker.publish(path,{'replacement':True})
    assert len(worker.records(tmp_path,p))==1
    record=json.loads(path.read_text());record['payload']['as_of']='2026-11-12T02:00:00Z'
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError):worker.records(tmp_path,p)


def test_late_created_record_rejected_even_with_matching_hash(worker,tmp_path):
    (tmp_path/'records').mkdir();p={'frozen':True}
    payload=dict(protocol_sha256=worker.digest(p),as_of='2026-11-12T00:00:00Z',
        created_at='2026-11-12T02:00:00Z',scheduled_start_utc='2026-11-12T01:00:00Z')
    worker.publish(tmp_path/'records/1.json',dict(payload=payload,sha256=worker.digest(payload)))
    with pytest.raises(AssertionError):worker.records(tmp_path,p)
