#!/usr/bin/env python3
# coding: utf-8
"""Remove Python runtime artifacts created by CLI verification."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    for path in ROOT.rglob("__pycache__"):
        shutil.rmtree(path, ignore_errors=True)
    for path in ROOT.rglob("*.pyc"):
        path.unlink(missing_ok=True)
    shutil.rmtree(ROOT / "dist", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
