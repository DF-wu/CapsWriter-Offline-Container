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


def run(args: list[str]) -> int:
    return subprocess.run(args, cwd=REPO_ROOT, check=False).returncode


def run_zipapp_stdin_smoke() -> int:
    env = os.environ.copy()
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
        )

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
