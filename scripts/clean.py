#!/usr/bin/env python3
# coding: utf-8
"""Clean generated verification artifacts without touching source files."""

from __future__ import annotations

import argparse
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
GENERATED_DIRS = (
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
)
GENERATED_FILES = (
    "client/web/tsconfig.tsbuildinfo",
    "client/web/tsconfig.node.tsbuildinfo",
    "client/web/vite.config.js",
    "client/web/vite.config.d.ts",
)
CLEAN_WEB_TIMEOUT_ENV = "CAPSWRITER_CLEAN_WEB_TIMEOUT"
DEFAULT_CLEAN_WEB_TIMEOUT_SECONDS = 120.0
TIMEOUT_EXIT_CODE = 124


def preserved_dirs_for(root: Path) -> set[Path]:
    return {
        root / ".git",
        root / "client" / "web" / "node_modules",
        root / "models",
    }


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def remove_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def clean_web_timeout_seconds() -> float:
    value = os.environ.get(CLEAN_WEB_TIMEOUT_ENV, "").strip()
    if not value:
        return DEFAULT_CLEAN_WEB_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError(f"{CLEAN_WEB_TIMEOUT_ENV} must be a number") from exc
    if timeout <= 0:
        raise ValueError(f"{CLEAN_WEB_TIMEOUT_ENV} must be > 0")
    return timeout


def run_web_clean() -> int:
    package_json = WEB_ROOT / "package.json"
    if not package_json.exists() or shutil.which("npm") is None:
        return 0
    try:
        timeout = clean_web_timeout_seconds()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    try:
        return subprocess.run(
            ["npm", "run", "clean"],
            cwd=WEB_ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"npm run clean timed out after {timeout:g}s", file=sys.stderr)
        return TIMEOUT_EXIT_CODE


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


def iter_cleanup_residue(
    root: Path = ROOT,
    preserved_dirs: set[Path] | None = None,
):
    yield from iter_python_cache_artifacts(
        root,
        preserved_dirs or preserved_dirs_for(root),
    )
    for relative in GENERATED_DIRS:
        path = root / relative
        if path.exists():
            yield path
    for relative in GENERATED_FILES:
        path = root / relative
        if path.exists():
            yield path


def check_clean(root: Path = ROOT) -> int:
    residue = sorted(
        {path.resolve() for path in iter_cleanup_residue(root)},
        key=lambda path: path.as_posix(),
    )
    if not residue:
        print("Cleanup check passed: no generated verification artifacts found")
        return 0
    print("Cleanup residue found:", file=sys.stderr)
    for path in residue:
        try:
            display = path.relative_to(root.resolve())
        except ValueError:
            display = path
        print(f"  {display}", file=sys.stderr)
    return 1


def clean_generated_artifacts() -> int:
    status = run_web_clean()

    for path in iter_python_cache_artifacts(ROOT, preserved_dirs_for(ROOT)):
        if path.name == "__pycache__":
            remove_tree(path)
        else:
            remove_file(path)

    for relative in GENERATED_DIRS:
        remove_tree(ROOT / relative)

    for relative in GENERATED_FILES:
        remove_file(ROOT / relative)

    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean generated CapsWriter verification artifacts",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check for cleanup residue; do not remove files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        return check_clean()
    return clean_generated_artifacts()


if __name__ == "__main__":
    raise SystemExit(main())
