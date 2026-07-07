# coding: utf-8

from __future__ import annotations

import io
import subprocess
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from client.cli.scripts import verify


class CliVerifyScriptTest(unittest.TestCase):
    def test_run_passes_configured_step_timeout(self) -> None:
        completed = SimpleNamespace(returncode=0)
        with (
            patch.dict(
                verify.os.environ,
                {verify.CLI_VERIFY_STEP_TIMEOUT_ENV: "2.5"},
            ),
            patch.object(verify.subprocess, "run", return_value=completed) as run,
        ):
            code = verify.run(["cmd"])

        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.kwargs["timeout"], 2.5)

    def test_run_rejects_invalid_step_timeout_before_spawning(self) -> None:
        stderr = io.StringIO()
        with (
            patch.dict(
                verify.os.environ,
                {verify.CLI_VERIFY_STEP_TIMEOUT_ENV: "0"},
            ),
            patch.object(verify.subprocess, "run") as run,
            redirect_stderr(stderr),
        ):
            code = verify.run(["cmd"])

        self.assertEqual(code, 1)
        run.assert_not_called()
        self.assertIn(
            "CAPSWRITER_CLI_VERIFY_STEP_TIMEOUT must be > 0",
            stderr.getvalue(),
        )

    def test_run_timeout_returns_timeout_exit_code(self) -> None:
        stderr = io.StringIO()
        with (
            patch.dict(
                verify.os.environ,
                {verify.CLI_VERIFY_STEP_TIMEOUT_ENV: "1"},
            ),
            patch.object(
                verify.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(["cmd"], timeout=1),
            ),
            redirect_stderr(stderr),
        ):
            code = verify.run(["cmd"])

        self.assertEqual(code, verify.TIMEOUT_EXIT_CODE)
        self.assertIn("Command timed out after 1s: cmd", stderr.getvalue())

    def test_zipapp_stdin_smoke_timeout_preserves_partial_output(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.dict(
                verify.os.environ,
                {verify.CLI_VERIFY_STEP_TIMEOUT_ENV: "1"},
            ),
            patch.object(
                verify.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(
                    ["capswriter-cli.pyz"],
                    timeout=1,
                    output=b"partial\n",
                    stderr=b"warning\n",
                ),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = verify.run_zipapp_stdin_smoke()

        self.assertEqual(code, verify.TIMEOUT_EXIT_CODE)
        self.assertIn("partial\n", stdout.getvalue())
        self.assertIn("warning\n", stderr.getvalue())
        self.assertIn("Command timed out after 1s:", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
