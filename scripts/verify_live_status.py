"""Verify protected model status against the published manifest; print no keys."""
import json
import requests
import sys
from app.config import settings
from app.services.model_release_service import read_active_release

base = 'http://frontend' if sys.argv[1:] == ['--via-frontend'] else 'http://backend:8000'
response = requests.get(base+'/api/v1/admin/stats',
    headers={'X-API-Key':settings.admin_api_key},timeout=15)
response.raise_for_status()
reported = response.json()['active_model_version']
published = read_active_release()['version']
assert reported == published, (reported,published)
print(json.dumps(dict(status='PASS',reported_active=reported,published_active=published)))
