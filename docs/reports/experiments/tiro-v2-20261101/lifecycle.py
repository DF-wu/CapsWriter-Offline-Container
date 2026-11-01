"""Stop/start the exact existing container; verify Portainer config fingerprints.

The independent watchdog restores service if the benchmark process disappears
or exceeds its deadline. No recreate/compose/deployment operation is used.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

STATE = Path('/tmp/capswriter-tiro-v2-state.json')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def inspect(state):
    value = json.loads(subprocess.check_output(['docker', 'inspect', state['production_id']], text=True))[0]
    assert value['Id'] == state['production_id'] and value['Image'] == state['base_image']
    return value


def fingerprints(info):
    keys = ['NetworkID', 'Aliases', 'Links', 'IPAMConfig', 'DriverOpts', 'GwPriority']
    sections = {'Config': info['Config'], 'HostConfig': info['HostConfig'],
                'NetworkConfiguration': {name: {key: value.get(key) for key in keys}
                                         for name, value in info['NetworkSettings']['Networks'].items()}}
    return {name: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            for name, value in sections.items()}


def assert_config(state):
    value = inspect(state)
    actual = fingerprints(value)
    assert actual == state['config_fingerprints_before'], ('Portainer config changed', actual)
    return value


def services(state):
    return [state, *state.get('additional_services', [])]


def readiness(service, info):
    if not info['State']['Running']:
        return False
    health = info['State'].get('Health', {}).get('Status')
    if health is not None:
        return health == 'healthy'
    probe = service.get('readiness_url')
    if probe:
        try:
            with urllib.request.urlopen(probe, timeout=3) as response:
                return response.status == 200 and json.load(response) == {'res': 'pong'}
        except Exception:
            return False
    return False


def restore(state, reason):
    root = Path(state['scratch'])
    record = {'reason': reason, 'started_utc': utc()}
    # End test GPU work before restoring the production GPU process.
    stopped = subprocess.run(['docker', 'stop', '-t', '15', state['container']], capture_output=True, text=True)
    record['test_stop_exit'] = stopped.returncode
    for service in services(state):
        info = inspect(service)
        if not info['State']['Running']:
            subprocess.run(['docker', 'start', service['production_id']], check=True, capture_output=True)
    deadline = time.monotonic() + 300
    while True:
        infos = [inspect(service) for service in services(state)]
        if all(readiness(service, info) for service, info in zip(services(state), infos)):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('Original containers did not become healthy within 300 seconds')
        time.sleep(3)
    restored=[]
    for service, info in zip(services(state), infos):
        actual=fingerprints(info)
        restored.append({'name':service['production_name'],'id':info['Id'],'image':info['Image'],
                         'started_at':info['State']['StartedAt'],
                         'health':info['State'].get('Health', {}).get('Status'),
                         'running':info['State']['Running'],
                         'readiness_basis':'Docker Healthcheck' if 'Health' in info['State'] else service.get('readiness_url'),
                         'ready':readiness(service, info),
                         'config_fingerprints_after':actual,
                         'parameters_unchanged':actual==service['config_fingerprints_before']})
    info=infos[0];actual=fingerprints(info)
    record.update(completed_utc=utc(), production_id=info['Id'], image=info['Image'],
                  health=info['State']['Health']['Status'], running=info['State']['Running'],
                  started_at=info['State']['StartedAt'], config_fingerprints_after=actual,
                  services=restored,portainer_parameters_unchanged=all(v['parameters_unchanged'] for v in restored))
    (root / 'results/restoration.json').write_text(json.dumps(record, indent=2) + '\n')
    assert record['portainer_parameters_unchanged']
    return record


def start_ticks(pid):
    try:
        # /proc comm can contain spaces; field 22 follows the last ')'.
        return Path('/proc/' + str(pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def watchdog(pid, ticks, deadline):
    state = json.loads(STATE.read_text()); root = Path(state['scratch'])
    cancel = root / 'watchdog.cancel'
    while not cancel.exists():
        gone = start_ticks(pid) != ticks
        expired = time.time() >= deadline
        if gone or expired:
            result = restore(state, 'watchdog: parent disappeared' if gone else 'watchdog: deadline expired')
            (root / 'results/watchdog-restoration.json').write_text(json.dumps(result, indent=2) + '\n')
            return
        time.sleep(3)


class Maintenance:
    def __init__(self, max_seconds=7200):
        self.state = json.loads(STATE.read_text())
        self.root = Path(self.state['scratch'])
        self.max_seconds = max_seconds
        self.guard = None
        self.stopped = False

    def __enter__(self):
        for service in services(self.state):
            info = assert_config(service)
            assert readiness(service, info), service['production_name'] + ' is not ready'
        assert self.state['base_preparation_complete']
        (self.root / 'watchdog.cancel').unlink(missing_ok=True)
        self.guard_log = open(self.root / 'logs/watchdog.log', 'a')
        self.guard = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--watchdog',
                                       str(os.getpid()), start_ticks(os.getpid()), str(time.time()+self.max_seconds)],
                                      stdout=self.guard_log, stderr=subprocess.STDOUT, start_new_session=True)
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            subprocess.run(['docker', 'stop', '-t', '30', *[s['production_id'] for s in services(self.state)]], check=True, capture_output=True)
            self.stopped = True
            assert all(not inspect(service)['State']['Running'] for service in services(self.state))
            (self.root / 'results/maintenance.json').write_text(json.dumps({
                'stopped_utc': utc(), 'production_id': self.state['production_id'],
                'services':[s['production_name'] for s in services(self.state)],
                'watchdog_pid': self.guard.pid, 'config_before_matched': True}, indent=2) + '\n')
            return self
        except BaseException:
            restore(self.state, '__enter__ failed')
            self.cancel_guard()
            raise

    def cancel_guard(self):
        (self.root / 'watchdog.cancel').write_text(utc())
        self.guard.wait(timeout=15)
        self.guard_log.close()

    def __exit__(self, kind, value, traceback):
        # Restoration is required for both success and any benchmark failure.
        result = restore(self.state, 'normal completion' if kind is None else 'benchmark exception: '+kind.__name__)
        self.cancel_guard()
        print(json.dumps({'production_restored': True, 'portainer_parameters_unchanged':
                          result['portainer_parameters_unchanged'], 'health': result['health']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watchdog', nargs=3)
    args = parser.parse_args()
    if args.watchdog:
        pid, ticks, deadline = args.watchdog
        watchdog(int(pid), ticks, float(deadline))
