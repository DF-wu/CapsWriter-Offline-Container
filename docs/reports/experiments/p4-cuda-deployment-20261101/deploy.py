"""Bounded same-container deployment with a separate restoration watchdog.

Run prepare.py and build the disposable validation image first. Rollback data
is private under ~/.local/state/capswriter/p4-cuda-20261101, not in git.
"""
import datetime
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

from prepare import ROOT, FILES, BINS, command, fingerprints, inspect

STATE = ROOT / 'state.json'
RESULTS = ROOT / 'results'
RUN_ID = os.environ.get('CAPSWRITER_P4_DEPLOY_RUN', str(time.time_ns()))
MARKERS = ROOT / 'runs' / RUN_ID
AUTH_KEY = ''
NATIVE_OFFSETS = {}


def save(name, value):
    if isinstance(value, dict):
        value = dict(value, run_id=RUN_ID)
    elif isinstance(value, list):
        value = [dict(row, run_id=RUN_ID) for row in value]
    (RESULTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def checked(service):
    value = inspect(service['id'])
    assert value['Id'] == service['id'] and value['Image'] == service['image']
    assert fingerprints(value) == service['fingerprints'], 'Portainer configuration changed'
    return value


def ready(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/ready', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(container, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = inspect(container)
        if not value['State']['Running']:
            raise RuntimeError(f'{container} stopped before readiness')
        if ready(port):
            return
        time.sleep(2)
    raise TimeoutError(f'{container} readiness timeout')


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu,clocks.current.graphics,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True).strip()


def model_hashes(container):
    code = "import pathlib,hashlib,json; p=pathlib.Path('/app/models/Qwen3-ASR/Qwen3-ASR-1.7B'); print(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in p.iterdir() if f.suffix in ['.onnx','.gguf']}))"
    return json.loads(subprocess.check_output(['docker', 'exec', container, 'python', '-c', code]))


def settings_hashes(container):
    code = "import pathlib,hashlib,json; paths=[*pathlib.Path('/app/settings').rglob('*'),pathlib.Path('/app/hot-server.txt')]; print(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths if f.is_file()}))"
    return json.loads(subprocess.check_output(['docker', 'exec', container, 'python', '-c', code]))


def transcribe(port, sample):
    boundary = 'p4-cuda-deployment-20261101'
    fields = {'model': 'whisper-1', 'temperature': '0', 'response_format': 'json'}
    if sample.get('language'):
        fields['language'] = sample['language']
    data = b''
    for key, value in fields.items():
        data += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
    path = ROOT / 'audio' / Path(sample['path']).name
    data += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
    data += path.read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=120)
    started = time.perf_counter()
    headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
    if AUTH_KEY:
        headers['Authorization'] = 'Bearer ' + AUTH_KEY
    connection.request('POST', '/v1/audio/transcriptions', data, headers)
    response = connection.getresponse(); body = response.read(); elapsed = time.perf_counter() - started
    connection.close()
    if response.status != 200:
        raise RuntimeError(f'HTTP transcription failed: {response.status}; {body[:300]!r}')
    value = json.loads(body)
    assert value.get('text', '').strip(), value
    return {'id': sample['id'], 'wall_s': elapsed, 'text': value['text'], 'gpu': gpu()}


def measure(port, phase, *, quality=False):
    samples = json.loads((ROOT / 'samples.json').read_text())
    selected = [s for s in samples if (s.get('phase') != 'latency') == quality]
    state = json.loads(STATE.read_text()); services = state['services']
    rows = []
    for repeat in range(1 if quality else 4):
        for sample in selected:
            assert not checked(services[1])['State']['Running'], 'Immich resumed during measurement'
            row = dict(transcribe(port, sample), repeat=repeat, warmup=not quality and repeat == 0, phase=phase)
            assert not checked(services[1])['State']['Running'], 'Immich resumed during measurement'
            if port == 16017:
                assert not checked(services[0])['State']['Running'], 'Original CapsWriter resumed during validation'
            rows.append(row); save(phase + '.json', rows)
            print(json.dumps({k: row[k] for k in ['id', 'wall_s', 'phase', 'repeat']}, ensure_ascii=False), flush=True)
    return rows


def normalize(text):
    return ''.join(ch.lower() for ch in text if ch.isalnum())


def compare(baseline, candidate):
    result = {}
    for sid in sorted({r['id'] for r in baseline}):
        old = [r for r in baseline if r['id'] == sid and not r['warmup']]
        new = [r for r in candidate if r['id'] == sid and not r['warmup']]
        old_p50 = statistics.median(r['wall_s'] for r in old)
        new_p50 = statistics.median(r['wall_s'] for r in new)
        assert new_p50 < old_p50, ('No measured improvement', sid, old_p50, new_p50)
        result[sid] = {'baseline_p50_s': old_p50, 'candidate_p50_s': new_p50,
                       'speedup': old_p50 / new_p50, 'latency_reduction_percent': (1-new_p50/old_p50)*100,
                       'http_outputs_are_stochastic': True, 'samples_per_variant': len(old)}
    return result


def compare_quality(baseline, candidate):
    old = {r['id']: normalize(r['text']) for r in baseline}
    new = {r['id']: normalize(r['text']) for r in candidate}
    assert old == new, ('Quality transcript changed', old, new)
    return {'matched_normalized_outputs': len(old), 'sample_ids': sorted(old)}


def greedy_quality(backend):
    state = json.loads(STATE.read_text())
    name = state['test_container'] + '-quality'
    image = state['services'][0]['image'] if backend == 'cpu' else state['test_image']
    output = ROOT / 'results' / ('greedy-' + backend + '.json')
    command(['docker', 'run', '-d', '--name', name, '--label', 'capswriter.task=p4-cuda-20261101',
             '--gpus', 'all', '--env-file', str(ROOT / 'test.env'),
             '-e', 'CAPSWRITER_LLAMA_BACKEND=' + backend,
             '-v', 'capswriter_capswriter-server-models:/app/models:ro',
             '-v', str(ROOT) + ':/bench', '-w', '/app', '--entrypoint', 'tail', image, '-f', '/dev/null'],
            stdout=subprocess.DEVNULL)
    try:
        if backend == 'cpu':
            target = '/app/core/server/engines/llama/bin'
            command(['docker', 'exec', '-u', '0', name, 'rm', '-rf', target])
            command(['docker', 'cp', str(ROOT / 'original-shared-bin'), name + ':' + target])
        command(['docker', 'exec', '-e', 'PYTHONPATH=/app', name, 'python', '/bench/run_quality.py',
                 '--backend', backend, '--output', '/bench/results/' + output.name, '--run-id', RUN_ID])
        rows = json.loads(output.read_text())
        assert len(rows) == 5 and all(r['run_id'] == RUN_ID and r['temperature'] == 0 for r in rows)
        return rows
    finally:
        command(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL)


def assert_cuda_logs(container):
    started = inspect(container)['State']['StartedAt']
    logs = subprocess.check_output(['docker', 'logs', '--since', started, container], stderr=subprocess.STDOUT, text=True)
    # The native callback writes INFO records to server_latest.log, while the
    # console only receives warnings. Read only this run's native log suffix.
    offset = NATIVE_OFFSETS.get(container, 0)
    code = ("import pathlib,re; p=pathlib.Path('/app/logs/server_latest.log'); "
            f"offset={offset}; "
            "f=p.open('rb'); f.seek(offset if p.stat().st_size>=offset else 0); "
            "lines=f.read().decode(errors='replace').splitlines(); "
            "print('\\n'.join(s for s in lines if re.search(r'load_tensors:|llama_context:|ggml_cuda|llama_kv_cache:|sched_reserve:',s)))")
    native = subprocess.check_output(['docker', 'exec', container, 'python', '-c', code], text=True)
    logs += '\n' + native
    filtered = '\n'.join(line for line in logs.splitlines() if re.search(r'CUDA|offload|llama|onnx|provider|backend|GPU|Vulkan|n_ctx|n_batch|flash_attn', line, re.I))
    (RESULTS / (container + '-startup.log')).write_text(filtered + '\n')
    assert 'CUDA decoder devices verified:' in logs and 'CUDA0' in logs, 'No verified CUDA decoder'
    assert 'refusing CPU fallback' not in logs
    assert re.search(r'CUDA0\s+model buffer size', native), 'Model buffer is not on CUDA0'
    offload = re.findall(r'offloaded (\d+)/(\d+) layers', logs)
    assert any(int(a) > 0 and a == b for a, b in offload), 'No full GPU layer offload evidence'
    return offload


def rollback_wrapper():
    wrapper = ROOT / 'rollback-entrypoint.sh'
    wrapper.write_text('#!/bin/sh\nset -eu\nrm -rf ' +
                       ' '.join('/app/' + p for p in BINS) +
                       '\ntar --unlink-first -xf /app/p4-cuda-rollback.tar -C /\n'
                       'rm -f /app/p4-cuda-rollback.tar /app/fork_server/llama_backend.py\n'
                       'exec /app/docker/server/entrypoint.sh "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


def restore(reason, *, keep_candidate=False):
    with (ROOT / 'restore.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return restore_locked(reason, keep_candidate=keep_candidate)


def restore_locked(reason, *, keep_candidate=False):
    state = json.loads(STATE.read_text()); prod, immich = state['services']
    subprocess.run(['docker', 'stop', '-t', '10', state['test_container']], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['docker', 'stop', '-t', '10', state['test_container'] + '-quality'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rollback = Path(state['rollback_dir'])
    errors = []
    needs_rollback = (MARKERS / 'patch-started').exists() and not (MARKERS / 'accepted').exists() and not keep_candidate
    try:
        if needs_rollback:
            subprocess.run(['docker', 'stop', '-t', '15', prod['id']], check=True, stdout=subprocess.DEVNULL)
            command(['docker', 'cp', str(rollback / 'original.tar'), prod['id'] + ':/app/p4-cuda-rollback.tar'])
            wrapper = rollback_wrapper()
            command(['docker', 'cp', '-a', str(wrapper), prod['id'] + ':/app/docker/server/entrypoint.sh'])
    except Exception as error:
        errors.append('CapsWriter rollback: ' + str(error))
    # Attempt both service starts even if rollback/config validation failed.
    for service in [prod, immich]:
        try:
            value = inspect(service['id'])
            assert value['Id'] == service['id'] and value['Image'] == service['image']
            if not value['State']['Running']:
                command(['docker', 'start', service['id']], stdout=subprocess.DEVNULL)
        except Exception as error:
            errors.append(service['name'] + ' start: ' + str(error))
    try:
        wait_ready(prod['id'], 6017)
        if needs_rollback:
            command(['docker', 'exec', '-u', '0', prod['id'], 'rm', '-rf', '/opt/capswriter/llama-cuda'])
    except Exception as error:
        errors.append('CapsWriter readiness: ' + str(error))
    deadline = time.monotonic() + 300
    while True:
        try:
            with urllib.request.urlopen('http://127.0.0.1:38080/api/server/ping', timeout=3) as response:
                if json.load(response) == {'res': 'pong'}:
                    break
        except Exception:
            pass
        if time.monotonic() >= deadline:
            errors.append('Immich restore readiness timeout')
            break
        time.sleep(2)
    restored = []
    for service in [prod, immich]:
        try:
            value = inspect(service['id']); actual = fingerprints(value)
            unchanged = actual == service['fingerprints']
            if not value['State']['Running']:
                errors.append(service['name'] + ' is not running at final verification')
            if not unchanged:
                errors.append(service['name'] + ' configuration changed')
            restored.append({'name': service['name'], 'id': value['Id'], 'image': value['Image'],
                             'fingerprints': actual, 'parameters_unchanged': unchanged,
                             'running': value['State']['Running'],
                             'health': value['State'].get('Health', {}).get('Status')})
        except Exception as error:
            errors.append(service['name'] + ' inspect: ' + str(error))
    if not ready(6017):
        errors.append('CapsWriter lost readiness after Immich recovery')
    save('restoration.json', {'reason': reason, 'optimization_accepted': (MARKERS / 'accepted').exists(),
                             'services': restored, 'errors': errors})
    if errors:
        raise RuntimeError('Restoration errors: ' + '; '.join(errors))


def watchdog(pid):
    deadline = time.monotonic() + 1800
    ticks = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    (MARKERS / 'watchdog-ready').touch()
    supervisor = int(os.environ['CAPSWRITER_P4_SUPERVISOR_PID'])
    while not (MARKERS / 'watchdog-cancel').exists():
        try:
            alive = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19] == ticks
        except FileNotFoundError:
            alive = False
        if not alive or not Path(f'/proc/{supervisor}').exists() or time.monotonic() > deadline:
            # The deploy worker owns its session. Fence it and every CLI child
            # before restoring, rather than racing a still-live deployer.
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            while Path(f'/proc/{pid}/stat').exists():
                try:
                    process_state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
                except FileNotFoundError:
                    break
                if process_state == 'Z':
                    break  # supervisor wait() reaps the terminated worker
                time.sleep(.1)
            restore('watchdog: parent disappeared or maintenance deadline expired')
            return
        time.sleep(3)


def deploy():
    global AUTH_KEY
    assert os.getpgrp() == os.getpid(), 'Deploy worker must own its process group'
    MARKERS.mkdir(parents=True, exist_ok=False)
    # Fail before stopping a service if preparation, credentials or rollback
    # data are incomplete. Resolve credentials privately from the exact service.
    state = json.loads(STATE.read_text()); prod, immich = state['services']
    for sample in json.loads((ROOT / 'samples.json').read_text()):
        path = ROOT / 'audio' / Path(sample['path']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sample['audio_sha256']
    assert (Path(state['rollback_dir']) / 'original.tar').is_file()
    credential_code = "from fork_server.http_api.runtime_config import parse_api_key; import os; print(parse_api_key(os.environ), end='')"
    AUTH_KEY = subprocess.check_output(['docker', 'exec', prod['id'], 'python', '-c', credential_code], text=True)
    test_env = ROOT / 'test.env'
    test_env.write_text((ROOT / 'production.env').read_text() +
                        ('CAPSWRITER_HTTP_API_KEY=' + AUTH_KEY + '\n' if AUTH_KEY else ''))
    test_env.chmod(0o600)
    for service in state['services']:
        assert checked(service)['State']['Running']
    models_before = model_hashes(prod['id']); settings_before = settings_hashes(prod['id'])
    NATIVE_OFFSETS[prod['id']] = int(subprocess.check_output(['docker', 'exec', prod['id'], 'python', '-c',
                "from pathlib import Path; print(Path('/app/logs/server_latest.log').stat().st_size)"]))
    save('before.json', {'services': state['services'], 'model_hashes': models_before,
                        'settings_hashes': settings_before, 'gpu': gpu()})
    guard_log = (ROOT / 'watchdog.log').open('a')
    guard = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watchdog', str(os.getpid())],
                             stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True,
                             env=dict(os.environ, CAPSWRITER_P4_DEPLOY_RUN=RUN_ID),
                             pass_fds=(int(os.environ['CAPSWRITER_P4_LOCK_FD']),))
    for sig in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    passed = False
    try:
        deadline = time.monotonic() + 10
        while not (MARKERS / 'watchdog-ready').exists():
            assert guard.poll() is None and time.monotonic() < deadline, 'Watchdog did not start'
            time.sleep(.1)
        command(['docker', 'stop', '-t', '30', immich['id']], stdout=subprocess.DEVNULL)
        baseline = measure(6017, 'baseline-http')
        command(['docker', 'stop', '-t', '30', prod['id']], stdout=subprocess.DEVNULL)
        baseline_quality = greedy_quality('cpu')
        command(['docker', 'run', '-d', '--name', state['test_container'], '--label', 'capswriter.task=p4-cuda-20261101',
                 '--gpus', 'all', '--env-file', str(test_env),
                 '-p', '127.0.0.1:16016:6016', '-p', '127.0.0.1:16017:6017',
                 '-v', 'capswriter_capswriter-server-models:/app/models:ro',
                 '-v', str(ROOT) + ':/bench', state['test_image']], stdout=subprocess.DEVNULL)
        wait_ready(state['test_container'], 16017)
        assert_cuda_logs(state['test_container'])
        candidate = measure(16017, 'cuda-validation-http')
        save('validation-comparison.json', compare(baseline, candidate))
        assert model_hashes(state['test_container']) == models_before
        # Export only tested code, libraries and bundle. No package-manager or
        # compiler state is copied to the production writable layer.
        patch_tar = ROOT / 'tested-patch.tar'
        with patch_tar.open('wb') as output:
            command(['docker', 'exec', state['test_container'], 'tar', 'cf', '-', '-C', '/',
                     *['app/' + p for p in FILES + BINS], 'opt/capswriter/llama-cuda'], stdout=output)
        command(['docker', 'stop', '-t', '30', state['test_container']], stdout=subprocess.DEVNULL)
        candidate_quality = greedy_quality('cuda')
        save('validation-quality-comparison.json', compare_quality(baseline_quality, candidate_quality))
        (MARKERS / 'patch-started').touch()
        with patch_tar.open('rb') as source:
            command(['docker', 'cp', '-a', '-', prod['id'] + ':/'], stdin=source, stdout=subprocess.DEVNULL)
        command(['docker', 'start', prod['id']], stdout=subprocess.DEVNULL)
        wait_ready(prod['id'], 6017)
        assert_cuda_logs(prod['id'])
        deployed = measure(6017, 'deployed-http')
        save('deployed-comparison.json', compare(baseline, deployed))
        measure(6017, 'deployed-quality-http', quality=True)
        assert model_hashes(prod['id']) == models_before
        assert settings_hashes(prod['id']) == settings_before
        for service in state['services']:
            checked(service)
        save('deployment.json', {'accepted': False, 'provisional': True, 'same_container_id': prod['id'],
             'same_image_id': prod['image'], 'models_unchanged': True, 'settings_unchanged': True,
             'portainer_parameters_unchanged': True, 'full_cuda_offload': True, 'gpu': gpu()})
        passed = True
    finally:
        try:
            restore('deployment finished or exception', keep_candidate=passed)
            if passed:
                # Commit only after both original services recovered. A failed
                # coexistence probe still leaves rollback available.
                sample = next(s for s in json.loads((ROOT / 'samples.json').read_text()) if s['id'] == 'mandarin-5s')
                save('coexistence-http.json', transcribe(6017, sample))
                for service in state['services']:
                    assert checked(service)['State']['Running']
                assert ready(6017)
                (MARKERS / 'accepted').touch()
                item = json.loads((RESULTS / 'deployment.json').read_text())
                save('deployment.json', dict(item, accepted=True, provisional=False, services_restored_before_acceptance=True))
                item = json.loads((RESULTS / 'restoration.json').read_text())
                save('restoration.json', dict(item, optimization_accepted=True))
        except BaseException:
            if passed:
                restore('provisional deployment failed restoration/coexistence')
            raise
        finally:
            # On recovery failure the independent watchdog remains armed.
            recovery = json.loads((RESULTS / 'restoration.json').read_text()) if (RESULTS / 'restoration.json').exists() else {}
            if recovery.get('run_id') == RUN_ID and not recovery.get('errors') and all(s['running'] for s in recovery.get('services', [])):
                (MARKERS / 'watchdog-cancel').touch()
                guard.wait(timeout=10)
            guard_log.close()


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--watchdog':
        watchdog(int(sys.argv[2]))
    elif sys.argv[1:] == ['--worker']:
        deploy()
    else:
        with (ROOT / 'deployment.lock').open('a') as exclusive:
            fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                       start_new_session=True,
                       env=dict(os.environ, CAPSWRITER_P4_DEPLOY_RUN=RUN_ID,
                                CAPSWRITER_P4_SUPERVISOR_PID=str(os.getpid()),
                                CAPSWRITER_P4_LOCK_FD=str(exclusive.fileno())),
                       pass_fds=(exclusive.fileno(),))
            sys.exit(worker.wait())
