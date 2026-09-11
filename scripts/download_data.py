"""Fetch the original UCI dataset; retain its documentation and license citation."""
from pathlib import Path
import io
import urllib.request
import zipfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
if __name__ == '__main__':
    dest = ROOT / 'data/raw'
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in ('hour.csv', 'day.csv', 'Readme.txt'):
            (dest / name).write_bytes(archive.read(name))
    print('hour.csv SHA256:', hashlib.sha256((dest / 'hour.csv').read_bytes()).hexdigest())
