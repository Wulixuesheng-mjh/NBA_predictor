"""Inspect public version availability without downloading a full dataset."""
import json
from pathlib import Path
import requests

results = []
for version in (530, 529, 528, 527, 526):
    url = ('https://www.kaggle.com/api/v1/datasets/download/eoinamoore/'
           f'historical-nba-data-and-player-box-scores/Games.csv?datasetVersionNumber={version}')
    row = {'version':version}
    try:
        with requests.get(url, stream=True, timeout=(5,12)) as response:
            row['status'] = response.status_code
            if response.status_code == 200:
                lines = response.iter_lines()
                row['columns'] = next(lines).decode('utf-8-sig')
                row['first_row'] = next(lines).decode('utf-8-sig')
    except requests.RequestException as exc:
        row['error'] = type(exc).__name__
    results.append(row)
    print(json.dumps(row), flush=True)
path = Path('/app/ml/saved_models/data_quality/kaggle_versions_20261028.json')
path.write_text(json.dumps(results, indent=2), encoding='utf-8')
