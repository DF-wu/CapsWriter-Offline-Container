# coding: utf-8

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
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

    def test_run_passes_configured_step_timeout(self) -> None:
        completed = SimpleNamespace(returncode=0)
        with (
            patch.dict(
                verify_all.os.environ,
                {verify_all.VERIFY_STEP_TIMEOUT_ENV: "2.5"},
            ),
            patch.object(verify_all.subprocess, "run", return_value=completed) as run,
            redirect_stdout(io.StringIO()),
        ):
            code = verify_all.run(["cmd"])

        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.kwargs["timeout"], 2.5)

    def test_run_rejects_invalid_step_timeout_before_spawning(self) -> None:
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with (
                    patch.dict(
                        verify_all.os.environ,
                        {verify_all.VERIFY_STEP_TIMEOUT_ENV: value},
                    ),
                    patch.object(verify_all.subprocess, "run") as run,
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(stderr),
                ):
                    code = verify_all.run(["cmd"])

                self.assertEqual(code, 1)
                run.assert_not_called()
                self.assertIn("CAPSWRITER_VERIFY_STEP_TIMEOUT must be > 0", stderr.getvalue())

    def test_run_timeout_uses_redacted_command(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.dict(
                verify_all.os.environ,
                {verify_all.VERIFY_STEP_TIMEOUT_ENV: "1"},
            ),
            patch.object(
                verify_all.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(["cmd"], timeout=1),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code = verify_all.run(["cmd", "--key", "sk-secret"])

        self.assertEqual(code, verify_all.TIMEOUT_EXIT_CODE)
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertIn("Command timed out after 1s: cmd --key <redacted>", combined)
        self.assertNotIn("sk-secret", combined)

    def test_run_capture_timeout_returns_partial_stdout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch.dict(
                verify_all.os.environ,
                {verify_all.VERIFY_STEP_TIMEOUT_ENV: "1"},
            ),
            patch.object(
                verify_all.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(
                    ["cmd"],
                    timeout=1,
                    output=b"partial\n",
                    stderr=b"warning\n",
                ),
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            code, output = verify_all.run_capture(["cmd"])

        self.assertEqual(code, verify_all.TIMEOUT_EXIT_CODE)
        self.assertEqual(output, "partial\n")
        self.assertIn("partial\n", stdout.getvalue())
        self.assertIn("warning\n", stderr.getvalue())
        self.assertIn("Command timed out after 1s: cmd", stderr.getvalue())

    def test_verify_http_uses_key_file_for_live_checks(self) -> None:
        commands: list[list[str]] = []

        def fake_run_required(args: list[str], **_kwargs) -> int:
            commands.append(args)
            return 0

        with (
            patch.object(verify_all, "run_required", side_effect=fake_run_required),
            patch.object(verify_all, "verify_http_transcription", return_value=0) as transcription,
        ):
            code = verify_all.verify_http(
                "http://127.0.0.1:6017",
                "",
                "/run/secrets/capswriter.key",
                "",
                "",
                True,
            )

        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 2)
        for command in commands:
            self.assertIn("--key-file", command)
            self.assertIn("/run/secrets/capswriter.key", command)
            self.assertNotIn("--key", command)
        transcription.assert_called_once_with(
            "http://127.0.0.1:6017",
            "",
            "/run/secrets/capswriter.key",
            "",
            "",
        )

    def test_verify_http_transcription_passes_key_file_to_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "sample.wav"
            audio.write_bytes(b"RIFF")
            with patch.object(verify_all, "run_capture", return_value=(0, "expected text\n")) as run_capture:
                code = verify_all.verify_http_transcription(
                    "http://127.0.0.1:6017",
                    "",
                    "/run/secrets/capswriter.key",
                    str(audio),
                    "expected text",
                )

        self.assertEqual(code, 0)
        command = run_capture.call_args.args[0]
        self.assertIn("--key-file", command)
        self.assertIn("/run/secrets/capswriter.key", command)
        self.assertNotIn("--key", command)
        self.assertIn("--timeout", command)
        self.assertEqual(
            command[command.index("--timeout") + 1],
            "600",
        )


if __name__ == "__main__":
    unittest.main()
