#!/usr/bin/env python3
# coding: utf-8
"""Build a dependency-free executable zipapp for capswriter-cli."""

from __future__ import annotations

import shutil
import tempfile
import zipapp
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
TARGET = DIST / "capswriter-cli.pyz"


def main() -> int:
    DIST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="capswriter-cli-zipapp-") as tmp:
        staging = Path(tmp)
        for name in ("__main__.py", "__init__.py", "capswriter_cli.py"):
            shutil.copy2(ROOT / name, staging / name)
        zipapp.create_archive(
            staging,
            target=TARGET,
            interpreter="/usr/bin/env python3",
            compressed=True,
        )
    print(f"Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
