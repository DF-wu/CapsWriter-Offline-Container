"""Prepare disposable validation resources without touching host packages/GPU state."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path('/tmp/capswriter-p4-cuda-implementation')
REPO = Path(__file__).resolve().parents[4]
FILES = ['docker/server/entrypoint.sh', 'docker/server/download_models.py',
         'core/server/engines/llama/llama.py', 'fork_server/llama_backend.py',
         *['core/server/engines/' + e + '/inference/__init__.py'
           for e in ['qwen_asr_gguf', 'fun_asr_gguf', 'force_aligner_gguf']]]
BINS = ['core/server/engines/llama/bin',
        'core/server/engines/qwen_asr_gguf/inference/bin',
        'core/server/engines/fun_asr_gguf/inference/bin',
        'core/server/engines/force_aligner_gguf/inference/bin']


def command(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def inspect(name):
    return json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]


def fingerprints(info):
    keys = ['NetworkID', 'Aliases', 'Links', 'IPAMConfig', 'DriverOpts', 'GwPriority']
    sections = {'Config': info['Config'], 'HostConfig': info['HostConfig'],
                'NetworkConfiguration': {n: {k: v.get(k) for k in keys}
                                         for n, v in info['NetworkSettings']['Networks'].items()}}
    return {n: hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for n, v in sections.items()}


def main():
    ROOT.mkdir(exist_ok=True, mode=0o700)
    os.chmod(ROOT, 0o700)
    (ROOT / 'results').mkdir(exist_ok=True)
    services = []
    for name in ['capswriter-server', 'immich-aio']:
        info = inspect(name)
        assert info['State']['Running'], name
        services.append({'name': name, 'id': info['Id'], 'image': info['Image'],
                         'fingerprints': fingerprints(info)})
    state = {'services': services, 'test_container': 'capswriter-p4-cuda-validation',
             'test_image': 'capswriter-p4-cuda:validated-20261101'}
    assert services[0]['id'] == 'd6ab96c622d146f10429fbdc8afd37f7804e474a53a3f372b7bc58f6eb94a37c'
    assert services[1]['id'] == '9b775b9cd582524779a2563cea7a7cf1dcc7502c81d6517970ea9a26525552e9'
    # Retain a private rollback archive; never publish full inspect or environment.
    rollback = Path.home() / '.local/state/capswriter/p4-cuda-20261101'
    rollback.mkdir(parents=True, exist_ok=False, mode=0o700)
    state['rollback_dir'] = str(rollback)
    with (rollback / 'original.tar').open('wb') as out:
        command(['docker', 'exec', services[0]['id'], 'tar', 'cf', '-', '-C', '/',
                 *['app/' + p for p in FILES if p != 'fork_server/llama_backend.py'],
                 *['app/' + p for p in BINS]], stdout=out)
    (ROOT / 'state.json').write_text(json.dumps(state, indent=2))
    (rollback / 'state.json').write_text(json.dumps(state, indent=2))
    info = inspect(services[0]['id'])
    # Exact injected values for the isolated test only. This file stays private.
    (ROOT / 'production.env').write_text('\n'.join(info['Config']['Env']) + '\n')
    os.chmod(ROOT / 'production.env', 0o600)
    for p in FILES:
        target = ROOT / 'overlay' / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / p, target)
    shutil.copy2(Path(__file__).with_name('run_quality.py'), ROOT / 'run_quality.py')
    (ROOT / 'original-shared-bin').mkdir()
    command(['docker', 'cp', services[0]['id'] + ':/app/core/server/engines/llama/bin/.',
             str(ROOT / 'original-shared-bin')])
    shutil.copy2(REPO / 'docker/server/Dockerfile.cuda-overlay', ROOT / 'Dockerfile')
    # Public evaluation controls; URLs with transient signatures stay in memory.
    audio = ROOT / 'audio'; audio.mkdir(exist_ok=True)
    samples = json.loads((REPO / 'docs/reports/experiments/p4-exclusive-20261101/samples.json').read_text())
    rows = json.load(urllib.request.urlopen(samples[0]['dataset_rows_api'], timeout=60))
    for item in rows['rows']:
        row = item['row']; sample = next(s for s in samples if s['id'] == 'libri-' + row['id'])
        assert sample['reference'] == row['text']
        with urllib.request.urlopen(row['audio'][0]['src'], timeout=60) as source:
            (audio / Path(sample['path']).name).write_bytes(source.read())
    for sid in ['noise2', 'codeswitch']:
        sample = next(s for s in samples if s['id'] == sid)
        with urllib.request.urlopen(sample['source'].replace('/blob/', '/resolve/'), timeout=60) as source:
            (audio / (sid + '.wav')).write_bytes(source.read())
    (ROOT / 'samples.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    print(json.dumps({'prepared': True, 'service_ids': [s['id'] for s in services],
                      'rollback_dir': str(rollback)}))


def finalize_audio():
    """Transforms run only in a disposable container, before maintenance."""
    state = json.loads((ROOT / 'state.json').read_text())
    code = '''
import hashlib,json,pathlib,subprocess
import soundfile as sf
r=pathlib.Path('/bench')
subprocess.run(['ffmpeg','-y','-loglevel','error','-i','/bench/audio/codeswitch.wav','-ar','16000','-ac','1','/bench/audio/codeswitch-16k.wav'],check=True)
y,sr=sf.read('/bench/audio/noise2.wav',dtype='float32');assert sr==16000
for n in [2,5,10]:sf.write('/bench/audio/mandarin-'+str(n)+'s.wav',y[:n*16000],16000)
samples=json.loads((r/'samples.json').read_text())
for s in samples:
    path=r/'audio'/pathlib.Path(s['path']).name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==s['audio_sha256'],s['id']
print('All 8 audio hashes verified before maintenance')
'''
    command(['docker', 'run', '--rm', '--label', 'capswriter.task=p4-cuda-20261101',
             '-v', str(ROOT) + ':/bench', '--entrypoint', 'python', state['test_image'], '-c', code])


if __name__ == '__main__':
    if sys.argv[1:] == ['--finalize-audio']:
        finalize_audio()
    else:
        main()
