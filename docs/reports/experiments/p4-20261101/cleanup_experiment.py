"""One-run cleanup with exact identity guards, preserving production resources.

This records the actual 2026-11-01 cleanup. It is intentionally not a general
Docker cleanup command; it refuses a different scratch/production identity.
"""
import datetime
import json
from pathlib import Path
import subprocess

STATE = Path('/tmp/capswriter-p4-experiment-state.json')
OUTPUT = Path(__file__).resolve().parent / 'results/cleanup.json'
EXPECTED_SCRATCH = Path('/tmp/capswriter-p4-exp-tudqzlgp')
PRODUCTION_ID = 'd6ab96c622d146f10429fbdc8afd37f7804e474a53a3f372b7bc58f6eb94a37c'
record = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'commands': []}


def run(args, check=True):
    result = subprocess.run(args, capture_output=True, text=True)
    record['commands'].append({'argv': args, 'exit': result.returncode,
                               'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()})
    if check:
        result.check_returncode()
    return result


def inspect(name):
    return json.loads(run(['docker', 'inspect', name]).stdout)[0]


def service(info):
    return {'id': info['Id'], 'image': info['Image'], 'status': info['State']['Status'],
            'started_at': info['State']['StartedAt'], 'health': info['State']['Health']['Status']}


def main():
    state = json.loads(STATE.read_text())
    root = Path(state['scratch'])
    assert root == EXPECTED_SCRATCH and root.resolve() == root and root.is_dir()
    assert state['created_images'] == ['nvidia/cuda:11.8.0-devel-ubuntu22.04']
    evidence = OUTPUT.parent
    env = json.loads((evidence / 'environment.json').read_text())
    audit = json.loads((evidence / 'audit.json').read_text())
    assert env['fp32_promotion_reproduction']['backend']['byte_identical']
    assert sum(v.get('latency_n', 0) for v in audit.values()) == 180
    assert sum(v.get('quality_n', 0) for v in audit.values()) == 30
    assert audit['temperature04_smoke']['n'] == 9
    # Never persist a full docker inspect response: it can contain environment
    # credentials. Keep only the selected identity / health / mount evidence.
    before = inspect('capswriter-server')
    record['production_before'] = service(before)
    assert before['Id'] == PRODUCTION_ID and before['State']['Health']['Status'] == 'healthy'
    record['state'] = state
    record['test_resources'] = []
    for name in [state['container'], state['compiler_container']]:
        info = inspect(name)
        assert info['Config']['Labels']['capswriter.experiment'] == 'p4-20261101'
        if name == state['container']:
            assert info['Id'] == state['container_id']
            model_mount = next(m for m in info['Mounts'] if m['Destination'] == '/models-original')
            assert model_mount['Name'] == state['model_volume'] and model_mount['RW'] is False
            record['model_mount_read_only'] = True
        record['test_resources'].append({'name': name, 'id': info['Id'], 'image': info['Image']})
        run(['docker', 'top', name, '-eo', 'pid,args'])
    record['original_model_hashes_rechecked'] = [m for m in env['model_artifacts'] if m['variant'] == 'original-read-only']
    record['scratch_bytes_before'] = int(run(['du', '-sb', str(root)]).stdout.split()[0])
    # Clear root-owned scratch contents from its owning disposable container.
    # pathlib + rmtree unlink symlinks; they never traverse the read-only models.
    run(['docker', 'exec', state['container'], 'python', '-c',
         "from pathlib import Path;import shutil;p=Path('/bench');"
         "[(c.unlink() if c.is_symlink() or c.is_file() else shutil.rmtree(c)) for c in p.iterdir()]"])
    run(['docker', 'rm', '-f', state['container'], state['compiler_container']])
    run(['docker', 'image', 'rm', state['created_images'][0]])
    root.rmdir()
    STATE.unlink()
    after = inspect('capswriter-server')
    record['production_after'] = service(after)
    assert record['production_after'] == record['production_before']
    assert after['Id'] == PRODUCTION_ID
    assert json.loads(run(['docker', 'image', 'inspect', state['base_image']]).stdout)[0]['Id'] == state['base_image']
    assert json.loads(run(['docker', 'volume', 'inspect', state['model_volume']]).stdout)[0]['Name'] == state['model_volume']
    remaining = run(['docker', 'ps', '-aq', '--filter', 'label=capswriter.experiment=p4-20261101']).stdout.strip()
    assert not remaining
    absent_image = run(['docker', 'image', 'inspect', state['created_images'][0]], check=False)
    assert absent_image.returncode != 0
    processes = run(['ps', '-eo', 'pid,args']).stdout.splitlines()
    remaining_telemetry = [line.strip() for line in processes if line.strip().split(maxsplit=1)[-1].startswith('nvidia-smi ')
                           and '-lms 200' in line]
    assert not remaining_telemetry
    run(['nvidia-smi', '--query-gpu=temperature.gpu,memory.used,clocks.current.graphics', '--format=csv'])
    record['checks'] = {'scratch_removed': not root.exists(), 'state_removed': not STATE.exists(),
                        'experiment_containers_remaining': [], 'experiment_telemetry_remaining': remaining_telemetry,
                        'new_devel_image_removed': True, 'base_image_retained': True,
                        'model_volume_retained': True, 'production_identity_and_health_unchanged': True}
    record['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    record['complete'] = True


if __name__ == '__main__':
    try:
        main()
    finally:
        # Remove potentially sensitive full inspect/environment/process output
        # from the command transcript; selected evidence is recorded above.
        record['commands'] = [c for c in record['commands'] if c['argv'][:2] not in
                              (['docker', 'inspect'], ['ps', '-eo'])
                              and c['argv'][:3] not in (['docker', 'image', 'inspect'], ['docker', 'volume', 'inspect'])]
        OUTPUT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record.get('checks', {'complete': False}), indent=2))
