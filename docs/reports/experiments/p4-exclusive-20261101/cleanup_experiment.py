"""Remove only this run's exact disposable resources after evidence export."""
import datetime
import json
from pathlib import Path
import subprocess

from lifecycle import STATE, assert_config, readiness, services

here = Path(__file__).resolve().parent
record = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'commands': []}


def run(args, check=True):
    result = subprocess.run(args, capture_output=True, text=True)
    if args[:2] not in [['docker', 'inspect'], ['ps', '-eo']]:
        record['commands'].append({'argv': args, 'exit': result.returncode,
                                   'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()})
    if check:
        result.check_returncode()
    return result


def main():
    state = json.loads(STATE.read_text())
    root = Path(state['scratch'])
    assert root == Path('/tmp/capswriter-p4-exclusive-i3eh5tn4') and root.resolve() == root
    assert state['label'] == 'p4-exclusive-20261101'
    assert state['created_images'] == ['nvidia/cuda:11.8.0-devel-ubuntu22.04']
    assert len(services(state)) == 2
    summary = json.loads((here / 'results/summary.json').read_text())
    assert len(summary) == 13 and all(v['latency']['n'] == 30 and v['quality']['n'] == 5 for v in summary.values())
    env = json.loads((here / 'results/environment.json').read_text())
    assert len([a for a in env['model_artifacts'] if a.get('matches_prior_experiment')]) == 3
    before = []
    for service in services(state):
        info = assert_config(service)
        assert readiness(service, info)
        before.append({'name': service['production_name'], 'id': info['Id'], 'image': info['Image'],
                       'started_at': info['State']['StartedAt'], 'fingerprints': service['config_fingerprints_before']})
    record['production_before_cleanup'] = before
    resources = []
    for key in ['container', 'compiler_container']:
        info = json.loads(run(['docker', 'inspect', state[key]]).stdout)[0]
        assert info['Id'] == state[key + '_id']
        assert info['Config']['Labels']['capswriter.experiment'] == state['label']
        resources.append({'name': state[key], 'id': info['Id'], 'image': info['Image']})
        if key == 'container':
            mount = next(m for m in info['Mounts'] if m['Destination'] == '/models-original')
            assert mount['Name'] == state['model_volume'] and not mount['RW']
    record['test_resources'] = resources
    image = json.loads(run(['docker', 'image', 'inspect', state['created_images'][0]]).stdout)[0]
    assert image['Id'] == resources[1]['image']
    record['scratch_bytes'] = int(run(['du', '-sb', str(root)]).stdout.split()[0])
    run(['docker', 'exec', state['container'], 'python', '-c',
         "from pathlib import Path;import shutil;p=Path('/bench');"
         "[(c.unlink() if c.is_symlink() or c.is_file() else shutil.rmtree(c)) for c in p.iterdir()]"])
    run(['docker', 'rm', '-f', state['container'], state['compiler_container']])
    run(['docker', 'image', 'rm', state['created_images'][0]])
    root.rmdir()
    STATE.unlink()
    after = []
    for service in services(state):
        info = assert_config(service)
        assert readiness(service, info)
        after.append({'name': service['production_name'], 'id': info['Id'], 'image': info['Image'],
                      'started_at': info['State']['StartedAt'], 'fingerprints': service['config_fingerprints_before']})
        run(['docker', 'image', 'inspect', service['base_image']])
    assert before == after
    record['production_after_cleanup'] = after
    run(['docker', 'volume', 'inspect', state['model_volume']])
    assert not run(['docker', 'ps', '-aq', '--filter', 'label=capswriter.experiment='+state['label']]).stdout.strip()
    assert run(['docker', 'image', 'inspect', state['created_images'][0]], check=False).returncode != 0
    processes = run(['ps', '-eo', 'pid,args']).stdout.splitlines()
    remaining = [p for p in processes if p.strip().split(maxsplit=1)[-1].startswith('nvidia-smi ') and '-lms 200' in p]
    assert not remaining
    record['checks'] = {'scratch_removed': not root.exists(), 'state_removed': not STATE.exists(),
                        'test_containers_removed': True, 'new_devel_image_removed': True,
                        'original_images_and_volume_retained': True, 'model_volume_was_read_only': True,
                        'both_original_containers_ready': True, 'same_ids_and_parameters': True,
                        'cleanup_did_not_restart_production': before == after, 'telemetry_remaining': remaining}
    record['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record['complete'] = True


if __name__ == '__main__':
    try:
        main()
    finally:
        # Retain identity summaries, never full inspect/volume metadata.
        record['commands'] = [c for c in record['commands'] if c['argv'][:3] not in
                              [['docker', 'image', 'inspect'], ['docker', 'volume', 'inspect']]]
        (here / 'results/cleanup.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record.get('checks', {'complete': False}), indent=2))
