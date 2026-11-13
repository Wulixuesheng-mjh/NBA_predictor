"""Host-network official PDFs, exact first-page parsing and source evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request
import pdfplumber

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from app.collector.gamebook_parser import parse_final_box, normalize


def fetch(game,root):
    path = root/game['filename']
    meta_path = path.with_suffix('.json')
    try:
        if meta_path.exists():
            previous = json.loads(meta_path.read_text(encoding='utf-8'))
            if previous.get('status')=='verified' and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==previous['sha256']:
                return previous
        if not path.exists():
            last = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(game['url'],headers={'User-Agent':'Mozilla/5.0'})
                    with urllib.request.urlopen(req,timeout=20) as response:
                        body = response.read()
                    if not body.startswith(b'%PDF'):
                        raise ValueError('Expected a PDF')
                    temporary = path.with_suffix('.pdf.partial')
                    temporary.write_bytes(body)
                    temporary.replace(path)
                    break
                except Exception as exc:
                    last = exc
                    if getattr(exc,'code',None) in (403,404):
                        raise
                    time.sleep(1+attempt)
            if not path.exists():
                raise last
        observed = datetime.now(timezone.utc).isoformat()
        with pdfplumber.open(path) as pdf:
            text = pdf.pages[0].extract_text()
        parsed = parse_final_box(text)
        if parsed['game_date'] != game['game_date']:
            raise ValueError('Official PDF game date differs from bound game')
        for team in parsed['teams']:
            if team.get('side_label_present') is False:
                if normalize(team['name'])==normalize(game['home_name']):
                    team['is_home']=True
                elif normalize(team['name'])==normalize(game['away_name']):
                    team['is_home']=False
                else:
                    raise ValueError('Unbound neutral-site team')
            side = 'home' if team['is_home'] else 'away'
            if normalize(team['name']) != normalize(game[side+'_name']):
                raise ValueError('Official PDF team differs from bound game')
            if team['totals']['pts'] != game[side+'_score']:
                raise ValueError('Official PDF score differs from bound game')
            team['team_id'] = game[side+'_team_id']
        result = dict(status='verified',game_id=game['game_id'],url=game['url'],filename=game['filename'],
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observed_at=observed,parsed=parsed)
        meta_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        return result
    except Exception as exc:
        return dict(status='rejected',game_id=game['game_id'],url=game['url'],error=f'{type(exc).__name__}: {exc}')


def main(args):
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    games = json.loads(Path(args.plan).read_text(encoding='utf-8'))
    if args.limit:
        games=games[:args.limit]
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fetch,g,root) for g in games]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results)%25==0 or len(results)==len(games):
                summary=dict(processed=len(results),planned=len(games),verified=sum(r['status']=='verified' for r in results))
                print(json.dumps(summary),flush=True)
                (root/'download_progress.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (root/'download_manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',default='tmp/pdfs/dev_gamebooks')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--limit',type=int)
    main(parser.parse_args())
