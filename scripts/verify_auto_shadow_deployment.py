"""Read-only smoke test of real proxy, frozen protocol and failed-gate blocking."""
import json,re
from pathlib import Path
import requests
from app.config import settings
from app.services.release_gate_service import check_offline_gate


def main():
    root=Path(settings.model_path)
    active=root/'active.json';before=active.read_bytes()
    headers={'X-API-Key':settings.admin_api_key}
    result=requests.get('http://frontend/api/v1/predictions/shadow-status',headers=headers,timeout=30)
    result.raise_for_status();status=result.json()
    assert status['purpose']=='diagnostic_only' and status['acceptance_run'] is False
    assert len(status['runs'])==1
    run=status['runs'][0]
    assert run['configuration_matches'] and run['integrity_errors']==0 and run['enabled']
    html=requests.get('http://frontend',timeout=15);html.raise_for_status()
    script=re.search(r'src="([^\"]+[.]js)"',html.text).group(1)
    body=requests.get('http://frontend'+script,timeout=30);body.raise_for_status()
    assert '/predictions/shadow-status' in body.text
    offline=json.loads((root/'totals_time_v9_20261029/summary.json').read_text())
    try:
        check_offline_gate(offline,['gates','residual:T4'],'total')
    except ValueError as exc:
        reason=str(exc)
    else:
        raise AssertionError('Current failing candidate unexpectedly passed release gates')
    assert active.read_bytes()==before
    report=dict(status='PASS',proxy=status,primary=json.loads(before)['version'],
        primary_unchanged=True,failed_offline_gate_blocked=reason,
        frontend_status_bundle_verified=True)
    Path('/tmp/auto_shadow_validation_20261029.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
