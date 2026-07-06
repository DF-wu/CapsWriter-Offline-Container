# coding: utf-8

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import verify_all  # noqa: E402


class VerifyAllLoggingTest(unittest.TestCase):
    def test_format_command_redacts_sensitive_flags(self) -> None:
        command = verify_all.format_command(
            [
                "python",
                "client/cli/capswriter_cli.py",
                "health",
                "--key",
                "sk-local-secret",
                "--http-key=sk-release-secret",
            ]
        )

        self.assertIn("--key <redacted>", command)
        self.assertIn("--http-key=<redacted>", command)
        self.assertNotIn("sk-local-secret", command)
        self.assertNotIn("sk-release-secret", command)

    def test_format_command_redacts_sensitive_env_assignments(self) -> None:
        command = verify_all.format_command(
            [
                "docker",
                "run",
                "-e",
                "CAPSWRITER_WEB_API_KEY=sk-web-secret",
                "-e",
                "CAPSWRITER_HTTP_API_KEY=sk-http-secret",
            ]
        )

        self.assertIn("CAPSWRITER_WEB_API_KEY=<redacted>", command)
        self.assertIn("CAPSWRITER_HTTP_API_KEY=<redacted>", command)
        self.assertNotIn("sk-web-secret", command)
        self.assertNotIn("sk-http-secret", command)

    def test_run_required_failure_uses_redacted_command(self) -> None:
        completed = SimpleNamespace(returncode=1)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(verify_all.subprocess, "run", return_value=completed),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = verify_all.run_required(["cmd", "--key", "sk-secret"])

        self.assertEqual(code, 1)
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertIn("--key <redacted>", combined)
        self.assertNotIn("sk-secret", combined)

    def test_run_capture_failure_uses_redacted_command(self) -> None:
        completed = SimpleNamespace(returncode=1, stdout="", stderr="")
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.object(verify_all.subprocess, "run", return_value=completed),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code, output = verify_all.run_capture(["cmd", "--key", "sk-secret"])

        self.assertEqual(code, 1)
        self.assertEqual(output, "")
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertIn("--key <redacted>", combined)
        self.assertNotIn("sk-secret", combined)


if __name__ == "__main__":
    unittest.main()
