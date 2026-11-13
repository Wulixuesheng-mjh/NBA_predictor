"""Bounded read-only probes; never print API keys or account information."""
import csv
import json
from pathlib import Path
import requests
from app.config import settings


def main():
    output = {'source_recent': [], 'probes': []}
    folder = Path('/app/ml/saved_models/data_sources/kaggle/v525')
    with (folder/'Games.csv').open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['gameDateTimeEst'][:10] >= '2026-10-19':
                output['source_recent'].append({k:row.get(k) for k in (
                    'gameId', 'gameDateTimeEst', 'hometeamName', 'awayteamName', 'homeScore', 'awayScore')})
    urls = [
        ('nba_cdn', 'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0042300405.json', {}),
        ('nba_stats', 'https://stats.nba.com/stats/boxscoretraditionalv3?GameID=0042300405&LeagueID=00&StartPeriod=0&EndPeriod=0&StartRange=0&EndRange=0&RangeType=0',
         {'User-Agent':'Mozilla/5.0', 'Referer':'https://www.nba.com/'}),
        ('bdl_stats', 'https://api.balldontlie.io/v1/stats?game_ids[]=21717972&per_page=100',
         {'Authorization': settings.balldontlie_api_key}),
    ]
    for name, url, headers in urls:
        record = {'source':name, 'url':url}
        try:
            res = requests.get(url, headers=headers, timeout=(5,12))
            record.update(status=res.status_code, content_type=res.headers.get('Content-Type'), bytes=len(res.content))
            if res.status_code == 200:
                data = res.json()
                record['keys'] = list(data)
                if name == 'bdl_stats':
                    record['rows'] = len(data.get('data',[]))
                path = folder.parent.parent / (name+'_probe.json')
                path.write_text(json.dumps(data), encoding='utf-8')
        except requests.RequestException as exc:
            record['error'] = type(exc).__name__
        output['probes'].append(record)
    path = Path('/app/ml/saved_models/data_quality/box_score_probes_20261028.json')
    path.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
