"""Remove only this task's verified disposable Docker resources and scratch."""
import json
from pathlib import Path
import re
import shutil
import subprocess

from prepare import ROOT, inspect, fingerprints

HERE = Path(__file__).resolve().parent


def output(args):
    return subprocess.check_output(args, text=True)


state = json.loads((ROOT / 'state.json').read_text())
accepted = json.loads((ROOT / 'results/deployment.json').read_text())
assert accepted['accepted'] and accepted['services_restored_before_acceptance']
production_ids = {s['id'] for s in state['services']}
for service in state['services']:
    info = inspect(service['id'])
    assert info['State']['Running'] and info['Image'] == service['image']
    assert fingerprints(info) == service['fingerprints']
owned = json.loads((ROOT / 'build-cache-owned.json').read_text())
cache_ids = {row['ID'] for row in owned}
assert len(cache_ids) == 31
before = [json.loads(line) for line in output(['docker', 'buildx', 'du', '--format', 'json']).splitlines() if line.startswith('{')]
unrelated_ids = {row['ID'] for row in before} - cache_ids
removed_containers = []
for container in output(['docker', 'ps', '-aq', '--filter', 'label=capswriter.task=p4-cuda-20261101']).splitlines():
    info = inspect(container)
    assert info['Id'] not in production_ids and info['Config']['Labels']['capswriter.task'] == 'p4-cuda-20261101'
    removed_containers.append(info['Name'])
    subprocess.run(['docker', 'rm', '-f', container], check=True, stdout=subprocess.DEVNULL)
expected_images = {
    'capswriter-p4-cuda:validated-20261101': 'sha256:61358aa0e58210cd6cc15bae08b6abbf8b2da8158977ba6adfeef9094afb4ce0',
    'capswriter-p4-cuda-compiler:temporary': 'sha256:e20636478a35b77e18110f2d7486d23451636d2a2a1b12f5295f6a14776fd7b1',
    'capswriter-p4-base:temporary': state['services'][0]['image'],
}
for tag, digest in expected_images.items():
    info = json.loads(output(['docker', 'image', 'inspect', tag]))[0]
    assert info['Id'] == digest
    subprocess.run(['docker', 'image', 'rm', tag], check=True, stdout=subprocess.DEVNULL)
# Exact anchored IDs; never an unfiltered/global prune. Unrelated records must
# all remain present after reclaiming this task's compiler and overlay cache.
pattern = '^(' + '|'.join(re.escape(value) for value in sorted(cache_ids)) + ')$'
prune = output(['docker', 'buildx', 'prune', '--all', '--force', '--filter', 'id~=' + pattern])
after = [json.loads(line) for line in output(['docker', 'buildx', 'du', '--format', 'json']).splitlines() if line.startswith('{')]
remaining_ids = {row['ID'] for row in after}
assert unrelated_ids <= remaining_ids, 'Unrelated build cache was affected'
assert not cache_ids & remaining_ids, 'Task build cache still present'
assert not output(['docker', 'ps', '-aq', '--filter', 'label=capswriter.task=p4-cuda-20261101']).strip()
for service in state['services']:
    info = inspect(service['id'])
    assert info['State']['Running'] and info['Image'] == service['image']
    assert fingerprints(info) == service['fingerprints']
record = {'task_containers_removed': removed_containers,
          'task_image_tags_removed': list(expected_images), 'task_cache_records_removed': len(cache_ids),
          'unrelated_cache_records_preserved': len(unrelated_ids), 'global_prune_used': False,
          'host_packages_installed': False, 'original_services_and_parameters_unchanged': True,
          'private_rollback_and_cuda_bundle_retained': True, 'scratch_removed': True}
(HERE / 'results/cleanup.json').write_text(json.dumps(record, indent=2) + '\n')
(HERE / 'results/cleanup-cache.txt').write_text(prune)
assert ROOT == Path('/tmp/capswriter-p4-cuda-implementation')
shutil.rmtree(ROOT)
print(json.dumps(record))
