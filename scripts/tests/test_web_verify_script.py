# coding: utf-8

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEB_VERIFY_SCRIPT = ROOT / "client" / "web" / "scripts" / "verify.mjs"
TIMEOUT_EXIT_CODE = 124


class WebVerifyScriptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.node = shutil.which("node")
        if self.node is None:
            self.skipTest("node is required for Web verifier script tests")

    def _make_fake_npm(self, directory: Path) -> Path:
        fake_npm_js = directory / "fake-npm.js"
        fake_npm_js.write_text(
            textwrap.dedent(
                """
                const fs = require("node:fs");
                const args = process.argv.slice(2);
                const command = args.join(" ");
                fs.appendFileSync(process.env.CAPTURE_PATH, `${command}\\n`);
                if (process.env.FAIL_COMMAND === command) {
                  process.exit(Number(process.env.FAIL_STATUS || "1"));
                }
                if (process.env.HANG_COMMAND === command) {
                  setInterval(() => {}, 1000);
                } else {
                  process.exit(0);
                }
                """
            ).strip(),
            encoding="utf-8",
        )

        if os.name == "nt":
            fake_npm = directory / "npm.cmd"
            fake_npm.write_text(
                f'@echo off\r\n"{self.node}" "{fake_npm_js}" %*\r\n',
                encoding="utf-8",
            )
            return fake_npm

        fake_npm = directory / "npm"
        fake_npm.write_text(
            f'#!/bin/sh\nexec "{self.node}" "{fake_npm_js}" "$@"\n',
            encoding="utf-8",
        )
        fake_npm.chmod(0o755)
        return fake_npm

    def _run_verify(self, env: dict[str, str], *, timeout: float = 5) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.node, str(WEB_VERIFY_SCRIPT)],
            cwd=ROOT / "client" / "web",
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )

    def test_verify_runs_test_build_and_clean(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp)
            capture = fake_bin / "calls.txt"
            self._make_fake_npm(fake_bin)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                "CAPTURE_PATH": str(capture),
            }

            result = self._run_verify(env)
            calls = capture.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, ["run test", "run build", "run clean"])

    def test_verify_skips_build_but_still_cleans_after_test_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp)
            capture = fake_bin / "calls.txt"
            self._make_fake_npm(fake_bin)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                "CAPTURE_PATH": str(capture),
                "FAIL_COMMAND": "run test",
                "FAIL_STATUS": "7",
            }

            result = self._run_verify(env)
            calls = capture.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result.returncode, 7)
        self.assertEqual(calls, ["run test", "run clean"])

    def test_verify_rejects_invalid_timeout_before_spawning_npm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp)
            capture = fake_bin / "calls.txt"
            self._make_fake_npm(fake_bin)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                "CAPTURE_PATH": str(capture),
                "CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT": "0",
            }

            result = self._run_verify(env)
            capture_exists = capture.exists()

        self.assertEqual(result.returncode, 1)
        self.assertIn("CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT must be > 0", result.stderr)
        self.assertFalse(capture_exists)

    def test_verify_timeout_returns_timeout_exit_code_and_runs_clean(self) -> None:
        if os.name == "nt":
            self.skipTest("Windows shell timeout behavior is covered by source parity")
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp)
            capture = fake_bin / "calls.txt"
            self._make_fake_npm(fake_bin)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
                "CAPTURE_PATH": str(capture),
                "CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT": "0.5",
                "HANG_COMMAND": "run test",
            }

            result = self._run_verify(env)
            calls = capture.read_text(encoding="utf-8").splitlines()

        self.assertEqual(result.returncode, TIMEOUT_EXIT_CODE)
        self.assertIn("Command timed out after 0.5s: npm run test", result.stderr)
        self.assertEqual(calls, ["run test", "run clean"])


if __name__ == "__main__":
    unittest.main()
