#!/usr/bin/env python3
# coding: utf-8
"""Run isolated CLI checks and always clean generated Python artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
CLI_VERIFY_STEP_TIMEOUT_ENV = "CAPSWRITER_CLI_VERIFY_STEP_TIMEOUT"
DEFAULT_CLI_VERIFY_STEP_TIMEOUT_SECONDS = 600.0
TIMEOUT_EXIT_CODE = 124


def verify_step_timeout_seconds() -> float:
    value = os.environ.get(CLI_VERIFY_STEP_TIMEOUT_ENV, "").strip()
    if not value:
        return DEFAULT_CLI_VERIFY_STEP_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError(f"{CLI_VERIFY_STEP_TIMEOUT_ENV} must be a number") from exc
    if timeout <= 0:
        raise ValueError(f"{CLI_VERIFY_STEP_TIMEOUT_ENV} must be > 0")
    return timeout


def _timeout_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _timeout_error(args: list[str], timeout: float) -> None:
    print(
        f"Command timed out after {timeout:g}s: {' '.join(args)}",
        file=sys.stderr,
    )


def run(args: list[str]) -> int:
    try:
        timeout = verify_step_timeout_seconds()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    try:
        return subprocess.run(
            args,
            cwd=REPO_ROOT,
            check=False,
            timeout=timeout,
        ).returncode
    except subprocess.TimeoutExpired:
        _timeout_error(args, timeout)
        return TIMEOUT_EXIT_CODE


def run_zipapp_stdin_smoke() -> int:
    env = os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="capswriter-cli-tts-") as tmp:
            # Linux CI images do not always include a speech engine. A fake binary is
            # enough because the zipapp is invoked with --dry-run and only needs
            # shutil.which() to select the Linux command path.
            fake_tts = Path(tmp) / "spd-say"
            fake_tts.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_tts.chmod(0o755)
            env["PATH"] = tmp + os.pathsep + env.get("PATH", "")

            proc = subprocess.run(
                [
                    sys.executable,
                    "client/cli/dist/capswriter-cli.pyz",
                    "speak",
                    "--stdin",
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                env=env,
                input="packaged stdin smoke",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=verify_step_timeout_seconds(),
            )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_output_text(error.stdout)
        stderr = _timeout_output_text(error.stderr)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        _timeout_error(
            [
                sys.executable,
                "client/cli/dist/capswriter-cli.pyz",
                "speak",
                "--stdin",
                "--dry-run",
            ],
            error.timeout,
        )
        return TIMEOUT_EXIT_CODE

    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        return proc.returncode
    if "packaged stdin smoke" not in proc.stdout:
        print("zipapp stdin smoke did not read stdin input", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    status = run([sys.executable, "-m", "compileall", "client/cli"])
    if status == 0:
        status = run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "client/cli/tests",
                "-v",
            ]
        )
    if status == 0:
        status = run([sys.executable, "client/cli/scripts/build_zipapp.py"])
    if status == 0:
        status = run([sys.executable, "client/cli/dist/capswriter-cli.pyz", "--help"])
    if status == 0:
        status = run_zipapp_stdin_smoke()
    clean_status = run([sys.executable, "client/cli/scripts/clean.py"])
    return status or clean_status


if __name__ == "__main__":
    raise SystemExit(main())
