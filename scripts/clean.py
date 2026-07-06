#!/usr/bin/env python3
# coding: utf-8
"""Clean generated verification artifacts without touching source files."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "client" / "web"


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def run_web_clean() -> None:
    package_json = WEB_ROOT / "package.json"
    if not package_json.exists() or shutil.which("npm") is None:
        return
    subprocess.run(
        ["npm", "run", "clean"],
        cwd=WEB_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    run_web_clean()

    for path in ROOT.rglob("__pycache__"):
        remove_tree(path)
    for path in ROOT.rglob("*.pyc"):
        remove_file(path)

    for relative in [
        ".pytest_cache",
        ".ruff_cache",
        "htmlcov",
        "coverage",
        "playwright-report",
        "test-results",
        ".drawio-tmp",
        "client/web/dist",
        "client/web/coverage",
        "client/web/.vite",
        "client/web/node_modules/.vite",
        "client/web/playwright-report",
        "client/web/test-results",
        "client/web/.tmp",
    ]:
        remove_tree(ROOT / relative)

    for relative in [
        "client/web/tsconfig.tsbuildinfo",
        "client/web/tsconfig.node.tsbuildinfo",
        "client/web/vite.config.js",
        "client/web/vite.config.d.ts",
    ]:
        remove_file(ROOT / relative)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
