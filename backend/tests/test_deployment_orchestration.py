from types import SimpleNamespace
import pytest,httpx
from fastapi import HTTPException


def test_only_legacy_capture_job_is_disabled(monkeypatch):
    import app.scheduler.scheduler as scheduler
    jobs=[]
    fake=SimpleNamespace(add_job=lambda *a,**kw:jobs.append(kw['id']),start=lambda:None)
    monkeypatch.setattr(scheduler,'get_scheduler',lambda:fake)
    monkeypatch.setenv('SHADOW_CAPTURE_ENABLED','false')
    scheduler.start_scheduler()
    assert 'shadow_observation_capture' not in jobs
    assert {'kaggle_collection','roster_sync','nightly_retraining','prospective_capture'}<=set(jobs)


@pytest.mark.asyncio
async def test_shadow_status_comes_from_frozen_observer(monkeypatch):
    from app.api.predictions import shadow_status
    expected=dict(purpose='diagnostic_only',acceptance_run=False,runs=[{'configuration_matches':True}])
    class Client:
        def __init__(self,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def get(self,url):
            assert url=='http://total-observer:8010/status'
            return httpx.Response(200,json=expected,request=httpx.Request('GET',url))
    monkeypatch.setenv('LEGACY_TOTAL_STATUS_URL','http://total-observer:8010/status')
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    assert await shadow_status(None)==expected


@pytest.mark.asyncio
async def test_unavailable_observer_is_not_reported_healthy(monkeypatch):
    from app.api.predictions import shadow_status
    class Client:
        def __init__(self,**kw):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*a):pass
        async def get(self,url):raise httpx.ConnectError('offline')
    monkeypatch.setenv('LEGACY_TOTAL_STATUS_URL','http://total-observer:8010/status')
    monkeypatch.setattr(httpx,'AsyncClient',Client)
    with pytest.raises(HTTPException) as exc:await shadow_status(None)
    assert exc.value.status_code==503
