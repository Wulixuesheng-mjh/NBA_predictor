"""Fetch the NBA's public official scorer report and inspect exact minutes."""
import hashlib
import json
from pathlib import Path
import requests
from PyPDF2 import PdfReader

root = Path('/app/ml/saved_models/data_sources/official_gamebooks')
root.mkdir(parents=True,exist_ok=True)
url = 'https://statsdmz.nba.com/pdfs/20240617/20240617_DALBOS_book.pdf'
response = requests.get(url,timeout=(5,25))
response.raise_for_status()
assert response.content.startswith(b'%PDF')
path = root/'20240617_DALBOS_book.pdf'
path.write_bytes(response.content)
text = PdfReader(str(path)).pages[0].extract_text()
path.with_suffix('.txt').write_text(text,encoding='utf-8')
print(json.dumps(dict(url=url,sha256=hashlib.sha256(response.content).hexdigest(),bytes=len(response.content))))
print(text)
