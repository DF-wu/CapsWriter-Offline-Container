#!/usr/bin/env python3
# coding: utf-8
"""Clean generated verification artifacts without touching source files."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "client" / "web"
PRESERVED_DIRS = {
    ROOT / ".git",
    ROOT / "client" / "web" / "node_modules",
    ROOT / "models",
}


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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def should_prune(path: Path, preserved_dirs: set[Path] | None = None) -> bool:
    preserved = preserved_dirs or PRESERVED_DIRS
    resolved = path.resolve()
    return any(_is_relative_to(resolved, item.resolve()) for item in preserved)


def iter_python_cache_artifacts(
    root: Path = ROOT,
    preserved_dirs: set[Path] | None = None,
):
    for current_str, dirs, files in os.walk(root):
        current = Path(current_str)
        next_dirs: list[str] = []
        for name in dirs:
            path = current / name
            if should_prune(path, preserved_dirs):
                continue
            if name == "__pycache__":
                yield path
                continue
            next_dirs.append(name)
        dirs[:] = next_dirs
        for name in files:
            if name.endswith(".pyc"):
                yield current / name


def main() -> int:
    run_web_clean()

    for path in iter_python_cache_artifacts():
        if path.name == "__pycache__":
            remove_tree(path)
        else:
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
        "client/cli/dist",
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
