"""Record the additionally authorized Immich identity without changing it."""
import json
import subprocess

from lifecycle import STATE, assert_config, fingerprints, readiness

state = json.loads(STATE.read_text())
assert_config(state)
info = json.loads(subprocess.check_output(['docker', 'inspect', 'immich-aio']))[0]
service = {'production_name': 'immich-aio', 'production_id': info['Id'],
           'base_image': info['Image'], 'config_fingerprints_before': fingerprints(info),
           'readiness_url': 'http://127.0.0.1:38080/api/server/ping'}
assert readiness(service, info)
if state.get('additional_services'):
    assert state['additional_services'] == [service], 'Do not replace an existing baseline'
state['additional_services'] = [service]
STATE.write_text(json.dumps(state, indent=2) + '\n')
print('Immich identity/configuration recorded; running and ping/pong verified.')
