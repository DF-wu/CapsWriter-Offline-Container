#!/usr/bin/env python3
"""Run bounded strict-suite, syntax, entrypoint, and cleanup checks."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STEP_TIMEOUT = 300.0


def step_timeout() -> float:
    raw = os.environ.get("CAPSWRITER_TUI_VERIFY_STEP_TIMEOUT", str(DEFAULT_STEP_TIMEOUT))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("CAPSWRITER_TUI_VERIFY_STEP_TIMEOUT must be a number") from exc
    if not math.isfinite(value) or value <= 0 or value > 3600:
        raise ValueError("CAPSWRITER_TUI_VERIFY_STEP_TIMEOUT must be > 0 and <= 3600")
    return value


def run(command: list[str], timeout: float) -> int:
    print(f"+ {' '.join(command)}", flush=True)
    try:
        return subprocess.run(command, cwd=ROOT, check=False, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"timed out after {timeout:g}s", file=sys.stderr)
        return 124


def main() -> int:
    try:
        timeout = step_timeout()
    except ValueError as exc:
        print(f"capswriter-tui verify: {exc}", file=sys.stderr)
        return 2

    steps = [
        [sys.executable, "scripts/verify_tui.py"],
        [sys.executable, "-m", "compileall", "-q", "client/tui"],
        [sys.executable, "-m", "client.tui", "--help"],
    ]
    status = 0
    try:
        for command in steps:
            status = run(command, timeout)
            if status:
                break
    finally:
        cleanup = run([sys.executable, "client/tui/scripts/clean.py"], timeout)
        check = run([sys.executable, "client/tui/scripts/clean.py", "--check"], timeout)
        if status == 0:
            status = cleanup or check
    return status


if __name__ == "__main__":
    raise SystemExit(main())
