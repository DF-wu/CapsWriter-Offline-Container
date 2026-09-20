"""The fork's log directory must work before server bootstrap imports."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class LoggerDirectoryTest(unittest.TestCase):
    def test_server_import_writes_only_to_configured_log_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, CAPSWRITER_LOG_DIR=directory)
            subprocess.run(
                [sys.executable, "-c", "import core.server; core.server.logger.info('test')"],
                env=env, check=True, capture_output=True, text=True,
            )
            self.assertTrue(Path(directory, "server_latest.log").is_file())
