#!/usr/bin/env python3
# coding: utf-8
"""Run isolated CLI checks and always clean generated Python artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]


def run(args: list[str]) -> int:
    return subprocess.run(args, cwd=REPO_ROOT, check=False).returncode


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
    clean_status = run([sys.executable, "client/cli/scripts/clean.py"])
    return status or clean_status


if __name__ == "__main__":
    raise SystemExit(main())
