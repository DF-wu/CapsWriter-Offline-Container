"""Copy only public experiment evidence, not weights/audio/source/signed URLs."""
import json
from pathlib import Path
import shutil
import subprocess

from lifecycle import STATE, assert_config, readiness, services

here = Path(__file__).resolve().parent
state = json.loads(STATE.read_text())
root = Path(state['scratch'])
restored = json.loads((root / 'results/restoration.json').read_text())
assert restored['portainer_parameters_unchanged']
for service in services(state):
    assert readiness(service, assert_config(service))
schedule = json.loads((root / 'results/schedule.json').read_text())
assert len(schedule) == 38 and all(r['exit'] == 0 for r in schedule)
subprocess.run(['docker', 'start', state['container']], check=True, capture_output=True)
shutil.copy2(here / 'capture_runtime.py', root / 'capture_runtime.py')
subprocess.run(['docker', 'exec', state['container'], 'python', '/bench/capture_runtime.py'], check=True)
subprocess.run(['python3', str(here / 'analyze_exclusive.py'), str(root / 'results')], check=True)
summary = json.loads((root / 'results/summary.json').read_text())
assert len(summary) == 13
assert all(v['latency']['n'] == 30 and v['quality']['n'] == 5 for v in summary.values())
for service in services(state):
    assert readiness(service, assert_config(service))
for name in ['results', 'logs']:
    shutil.copytree(root / name, here / name, dirs_exist_ok=True)
shutil.copy2(root / 'samples.json', here / 'samples.json')
# Selected evidence only; full docker inspect can contain credentials.
infos = {}
for key in ['container', 'compiler_container']:
    info = json.loads(subprocess.check_output(['docker', 'inspect', state[key]]))[0]
    assert info['Config']['Labels']['capswriter.experiment'] == state['label']
    infos[key] = {'id': info['Id'], 'image': info['Image'], 'mounts': info['Mounts'],
                  'cpu_quota': info['HostConfig']['NanoCpus'], 'memory_bytes': info['HostConfig']['Memory'],
                  'published_ports': info['HostConfig']['PortBindings']}
    state[key + '_id'] = info['Id']
(here / 'results/test-resources.json').write_text(json.dumps(infos, indent=2) + '\n')
STATE.write_text(json.dumps(state, indent=2) + '\n')
print('Collected public evidence: 390 latency and 65 quality calls; restored services ready.')
