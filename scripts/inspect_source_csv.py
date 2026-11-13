import csv
import itertools
import json
from pathlib import Path

p = Path('/app/ml/saved_models/data_sources/kaggle/v525')
print([(f.name, f.stat().st_size) for f in p.iterdir()])
for name in ['Games.csv', 'TeamStatistics.csv', 'PlayerStatistics.csv', 'Players.csv']:
    with (p/name).open(encoding='utf-8-sig') as stream:
        rows = list(itertools.islice(csv.DictReader(stream), 2))
    print(name, json.dumps(rows))
