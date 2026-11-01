"""Recovery failure regression: no Docker/GPU operations are performed."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import deploy


class RestorationTest(unittest.TestCase):
    def test_immich_restart_is_attempted_even_when_capswriter_rollback_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); markers = root / 'run-new'; markers.mkdir()
            (markers / 'patch-started').touch()
            # Stale success/cancel records from a previous run must not apply.
            (root / 'accepted').touch(); (root / 'watchdog-cancel').touch()
            services = [{'name': n, 'id': n, 'image': 'image', 'fingerprints': {}}
                        for n in ['capswriter', 'immich']]
            state = root / 'state.json'
            state.write_text(json.dumps({'services': services, 'test_container': 'test', 'rollback_dir': tmp}))
            commands = []; saved = {}
            def command(args, **kwargs):
                commands.append(args)
                if args[:2] == ['docker', 'cp']:
                    raise OSError('injected rollback copy failure')
            def inspect(container):
                return {'Id': container, 'Image': 'image', 'State': {'Running': False}}
            with (patch.object(deploy, 'ROOT', root), patch.object(deploy, 'STATE', state),
                  patch.object(deploy, 'MARKERS', markers), patch.object(deploy, 'inspect', inspect),
                  patch.object(deploy, 'fingerprints', return_value={}),
                  patch.object(deploy, 'command', command), patch.object(deploy.subprocess, 'run'),
                  patch.object(deploy, 'wait_ready'),
                  patch.object(deploy.urllib.request, 'urlopen', return_value=io.BytesIO(b'{"res":"pong"}')),
                  patch.object(deploy, 'save', side_effect=lambda name,value: saved.update(value))):
                with self.assertRaisesRegex(RuntimeError, 'injected rollback copy failure'):
                    deploy.restore('failure rehearsal')
            self.assertIn(['docker', 'start', 'immich'], commands)
            self.assertFalse(saved['optimization_accepted'])
            self.assertTrue(any('not running at final verification' in error for error in saved['errors']))


if __name__ == '__main__':
    unittest.main()
