#!/usr/bin/env python3
"""Remove only generated files owned by ``client/tui``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TUI_ROOT = Path(__file__).resolve().parents[1]


def residues() -> list[Path]:
    found: list[Path] = []
    for directory in TUI_ROOT.rglob("__pycache__"):
        if directory.is_dir():
            found.append(directory)
    for pattern in ("*.pyc", "*.pyo"):
        found.extend(path for path in TUI_ROOT.rglob(pattern) if path.is_file())
    for name in (".coverage", "coverage", "htmlcov", ".pytest_cache"):
        path = TUI_ROOT / name
        if path.exists():
            found.append(path)
    return sorted(set(found), key=lambda path: (len(path.parts), str(path)), reverse=True)


def clean() -> None:
    for path in residues():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated residue exists")
    args = parser.parse_args()
    if args.check:
        remaining = residues()
        if remaining:
            for path in remaining:
                print(path.relative_to(TUI_ROOT))
            return 1
        return 0
    clean()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
