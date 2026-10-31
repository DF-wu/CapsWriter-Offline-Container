"""Restore the same public quality corpus; never publish signed asset URLs."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

root = Path(json.loads(Path('/tmp/capswriter-p4-exclusive-state.json').read_text())['scratch'])
repo = Path(__file__).resolve().parents[4]
samples = [s for s in json.loads((repo/'docs/reports/experiments/p4-20261101/samples.json').read_text())
           if s.get('phase') != 'pilot']
url = 'https://datasets-server.huggingface.co/rows?dataset=hf-internal-testing/librispeech_asr_dummy&config=clean&split=validation&offset=0&length=3'
response = json.load(urllib.request.urlopen(url, timeout=60))
for item in response['rows']:
    row = item['row'];sample = next(s for s in samples if s['id'] == 'libri-'+row['id'])
    assert row['text'] == sample['reference']
    path = root / sample['path'].removeprefix('/bench/')
    with urllib.request.urlopen(row['audio'][0]['src'], timeout=60) as source, path.open('wb') as target:
        shutil.copyfileobj(source, target)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sample['audio_sha256']
for sample in samples:
    path = root / sample['path'].removeprefix('/bench/')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sample['audio_sha256'],sample['id']
(root/'samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2)+'\n')
pilot = [dict(s, phase='pilot') for s in samples if s['id'] == 'noise2']
(root/'pilot-samples.json').write_text(json.dumps(pilot, ensure_ascii=False, indent=2)+'\n')
print('Verified the same 3 latency clips and 5 reference quality samples', flush=True)
