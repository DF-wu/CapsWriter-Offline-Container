# coding: utf-8

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr

from fork_server.http_api.runtime_config import ConfigError
from start_server_docker import run_configured_server


class DummyServer:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True


class StartServerDockerTest(unittest.TestCase):
    def test_run_configured_server_exits_cleanly_on_config_error(self) -> None:
        def apply_env_config() -> None:
            raise ConfigError("bad HTTP env")

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                run_configured_server(apply_env_config, lambda: DummyServer())

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("CapsWriter configuration error: bad HTTP env", stderr.getvalue())

    def test_run_configured_server_starts_after_config_apply(self) -> None:
        calls: list[str] = []
        server = DummyServer()

        def apply_env_config() -> None:
            calls.append("apply")

        run_configured_server(apply_env_config, lambda: server)

        self.assertEqual(calls, ["apply"])
        self.assertTrue(server.started)


if __name__ == "__main__":
    unittest.main()
