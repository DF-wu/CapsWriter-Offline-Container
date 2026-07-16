# coding: utf-8

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

import probe_backend  # noqa: E402


class ProbeBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original_model_type = probe_backend.ServerConfig.model_type
        self.addCleanup(
            setattr,
            probe_backend.ServerConfig,
            "model_type",
            self.original_model_type,
        )

    @staticmethod
    def _factory_module(events: list[str], *, fail_if_called: bool = False) -> ModuleType:
        module = ModuleType("core.server.engines.factory")

        class FakeEngine:
            def cleanup(self) -> None:
                events.append("cleanup")

        class FakeFactory:
            @staticmethod
            def create_asr_engine(model_type: str) -> FakeEngine:
                if fail_if_called:
                    raise AssertionError("factory must not be imported for non-GGUF models")
                events.append(f"factory:{model_type}")
                return FakeEngine()

        module.EngineFactory = FakeFactory
        return module

    def test_environment_is_applied_before_engine_factory(self) -> None:
        events: list[str] = []

        def fake_apply() -> None:
            events.append("apply")
            probe_backend.ServerConfig.model_type = "qwen_asr"

        factory_module = self._factory_module(events)
        with (
            patch("fork_server.env_config.apply", side_effect=fake_apply),
            patch.dict(
                sys.modules,
                {"core.server.engines.factory": factory_module},
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(probe_backend.main(), 0)

        self.assertEqual(events, ["apply", "factory:qwen_asr", "cleanup"])

    def test_invalid_environment_fails_before_engine_import(self) -> None:
        stderr = io.StringIO()
        with (
            patch(
                "fork_server.env_config.apply",
                side_effect=ValueError("invalid backend setting"),
            ),
            redirect_stderr(stderr),
        ):
            self.assertEqual(probe_backend.main(), 1)

        self.assertIn("configured backend probe failed", stderr.getvalue())
        self.assertIn("invalid backend setting", stderr.getvalue())

    def test_non_gguf_skip_uses_validated_model_type(self) -> None:
        events: list[str] = []

        def fake_apply() -> None:
            events.append("apply")
            probe_backend.ServerConfig.model_type = "sensevoice"

        factory_module = self._factory_module(events, fail_if_called=True)
        stdout = io.StringIO()
        with (
            patch("fork_server.env_config.apply", side_effect=fake_apply),
            patch.dict(
                sys.modules,
                {"core.server.engines.factory": factory_module},
            ),
            redirect_stdout(stdout),
        ):
            self.assertEqual(probe_backend.main(), 0)

        self.assertEqual(events, ["apply"])
        self.assertIn("probe skipped: sensevoice", stdout.getvalue())

    def test_probe_timeout_is_finite_positive_and_bounded(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                probe_backend.backend_probe_timeout_seconds(),
                probe_backend.DEFAULT_BACKEND_PROBE_TIMEOUT_SECONDS,
            )
        with patch.dict(
            os.environ,
            {probe_backend.BACKEND_PROBE_TIMEOUT_ENV: "12.5"},
            clear=True,
        ):
            self.assertEqual(probe_backend.backend_probe_timeout_seconds(), 12.5)
        for value in ("nope", "0", "nan", "inf", "1801"):
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ,
                    {probe_backend.BACKEND_PROBE_TIMEOUT_ENV: value},
                    clear=True,
                ),
                self.assertRaises(ValueError),
            ):
                probe_backend.backend_probe_timeout_seconds()

    def test_supervisor_spawns_bounded_worker(self) -> None:
        process = Mock(pid=1234)
        process.wait.return_value = 0
        with (
            patch.dict(
                os.environ,
                {probe_backend.BACKEND_PROBE_TIMEOUT_ENV: "9"},
                clear=True,
            ),
            patch.object(
                probe_backend.subprocess,
                "Popen",
                return_value=process,
            ) as popen,
        ):
            self.assertEqual(probe_backend.supervised_main(), 0)

        command = popen.call_args.args[0]
        self.assertEqual(command[-1], "--worker")
        self.assertEqual(
            popen.call_args.kwargs["start_new_session"],
            os.name != "nt",
        )
        process.wait.assert_called_once_with(timeout=9.0)

    def test_supervisor_reports_worker_timeout(self) -> None:
        stderr = io.StringIO()
        process = Mock(pid=1234)
        process.wait.side_effect = subprocess.TimeoutExpired(["probe"], 4)
        with (
            patch.object(
                probe_backend.subprocess,
                "Popen",
                return_value=process,
            ),
            patch.object(probe_backend, "terminate_probe_process_tree") as terminate,
            patch.dict(
                os.environ,
                {probe_backend.BACKEND_PROBE_TIMEOUT_ENV: "4"},
                clear=True,
            ),
            redirect_stderr(stderr),
        ):
            self.assertEqual(probe_backend.supervised_main(), 1)

        self.assertIn("timed out after 4s", stderr.getvalue())
        terminate.assert_called_once_with(process)

    @unittest.skipIf(os.name == "nt", "POSIX process groups are Linux-only")
    def test_timeout_cleanup_kills_the_entire_probe_process_group(self) -> None:
        process = Mock(pid=4321)
        process.wait.return_value = -signal.SIGKILL
        with patch.object(probe_backend.os, "killpg") as killpg:
            probe_backend.terminate_probe_process_tree(process)

        killpg.assert_called_once_with(4321, signal.SIGKILL)
        process.kill.assert_not_called()
        process.wait.assert_called_once_with(
            timeout=probe_backend.BACKEND_PROBE_KILL_GRACE_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
