"""Strict first-page NBA official FINAL BOX parsing; no imputed minutes."""
import re
import unicodedata
from datetime import datetime

FIELDS = ('fgm','fga','fg3m','fg3a','ftm','fta','oreb','dreb','reb','ast','pf','stl','tov','blk','plus_minus','pts')


def normalize(value):
    text = ''.join(c for c in unicodedata.normalize('NFKD',value) if not unicodedata.combining(c))
    text = re.sub(r'\s+',' ',text.replace('.','').lower()).strip()
    return 'los angeles '+text[3:] if text.startswith('la ') else text


def seconds(value):
    minutes, seconds = map(int,value.split(':'))
    if not 0 <= seconds < 60:
        raise ValueError('Invalid clock seconds')
    return minutes*60+seconds


def parse_final_box(text):
    if 'OFFICIAL SCORER' not in text or 'FINAL BOX' not in text:
        raise ValueError('Not an official final box')
    date_match = re.search(r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})',text)
    if not date_match:
        raise ValueError('Missing game date')
    day = datetime.strptime(date_match[1],'%B %d, %Y').date()
    teams, current = [], None
    for raw in text.splitlines():
        line = raw.strip()
        header = re.match(r'^(VISITOR|HOME):\s+(.+?)(?:\s+\([\d-]+\))?$',line)
        neutral = re.match(r'^([A-Za-z][A-Za-z .-]+)\s+\(\d+-\d+\)$',line) if not header else None
        if header:
            current = dict(is_home=header[1]=='HOME',name=header[2],players=[],side_label_present=True)
            teams.append(current)
            continue
        if neutral:
            current=dict(is_home=bool(teams),name=neutral[1],players=[],side_label_present=False)
            teams.append(current)
            continue
        if current is None:
            continue
        total = re.match(r'^(\d{3}:\d{2})\s+(.+)$',line)
        if total:
            values = total[2].split()
            if len(values) != len(FIELDS):
                raise ValueError('Unexpected totals columns')
            current['totals'] = dict(zip(FIELDS,map(int,values)))
            current['total_seconds'] = seconds(total[1])
            continue
        team_tov = re.search(r'TOT TO:\s*(\d+)',line)
        if team_tov:
            current['team_turnovers'] = int(team_tov[1])
        if 'totals' in current:
            continue
        if not re.match(r'^\d{1,2}\s',line):
            continue
        clock = re.search(r'\b\d{1,2}:\d{2}\b',line)
        if clock:
            prefix = line[:clock.start()].strip().split(maxsplit=1)
            if len(prefix) != 2:
                raise ValueError('Missing player name')
            name = re.sub(r'\s+(?:F|G|C|F-C|G-F|C-F|F-G)$','',prefix[1])
            values = line[clock.end():].split()
            if len(values) != len(FIELDS):
                raise ValueError('Unexpected player columns: '+name)
            row = dict(zip(FIELDS,map(int,values)))
            row.update(name=name,seconds=seconds(clock[0]))
            current['players'].append(row)
        else:
            dnp = re.match(r'^\d{1,2}\s+(.+?)\s+(?:DNP|DND|NWT)\b',line)
            if dnp:
                current['players'].append(dict.fromkeys(FIELDS,0)|dict(name=dnp[1],seconds=0))
    if len(teams) != 2 or {t['is_home'] for t in teams} != {False,True}:
        raise ValueError('Need exactly one home and visitor table')
    for team in teams:
        if 'totals' not in team or len(team['players']) < 5:
            raise ValueError('Missing full team table')
        if abs(sum(p['seconds'] for p in team['players'])-team['total_seconds']) > 2:
            raise ValueError('Player minutes do not equal official team minutes')
        for field in FIELDS:
            if field == 'plus_minus':
                continue
            if sum(p[field] for p in team['players']) != team['totals'][field]:
                raise ValueError('Player totals mismatch: '+field)
        for row in team['players']:
            if row['seconds'] > team['total_seconds']/5+1:
                raise ValueError('Player minutes exceed game duration')
            if row['pts'] != 2*row['fgm']+row['fg3m']+row['ftm']:
                raise ValueError('Scoring identity mismatch')
        team['totals']['tov'] = team.get('team_turnovers',team['totals']['tov'])
    if teams[0]['total_seconds'] != teams[1]['total_seconds']:
        raise ValueError('Home/away game duration mismatch')
    return dict(game_date=str(day),teams=teams)
