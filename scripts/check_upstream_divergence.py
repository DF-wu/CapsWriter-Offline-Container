#!/usr/bin/env python3
# coding: utf-8
"""Guard the fork's low-drift contract against upstream-tracked file changes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_REF = "origin/master"
BASE_ENV = "CAPSWRITER_UPSTREAM_BASE"
GIT_TIMEOUT_SECONDS = 15.0
ALLOWED_UPSTREAM_DIVERGENCE = frozenset(
    {
        ".gitignore",
        "LLM/default.py",
        "assets/BUILD_GUIDE.md",
        "readme.md",
        "requirements-server.txt",
        "zip_release.py",
    }
)


def git_output(args: list[str], *, cwd: Path = ROOT) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS:g}s"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def git_success(args: list[str], *, cwd: Path = ROOT) -> bool:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(args)} timed out after {GIT_TIMEOUT_SECONDS:g}s"
        ) from exc
    return completed.returncode == 0


def ref_exists(ref: str, *, cwd: Path = ROOT) -> bool:
    return git_success(
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=cwd,
    )


def path_exists_in_ref(ref: str, path: str, *, cwd: Path = ROOT) -> bool:
    return git_success(["cat-file", "-e", f"{ref}:{path}"], cwd=cwd)


def changed_paths(base_ref: str, *, cwd: Path = ROOT) -> set[str]:
    output = git_output(
        ["diff", "--name-status", "--find-renames", f"{base_ref}..HEAD"],
        cwd=cwd,
    )
    paths: set[str] = set()
    for line in output.splitlines():
        fields = line.split("\t")
        if not fields:
            continue
        status = fields[0]
        if status.startswith(("R", "C")) and len(fields) >= 3:
            paths.add(fields[1])
            paths.add(fields[2])
        elif len(fields) >= 2:
            paths.add(fields[1])
    return paths


def upstream_tracked_changes(base_ref: str, *, cwd: Path = ROOT) -> list[str]:
    upstream_paths = {
        path
        for path in changed_paths(base_ref, cwd=cwd)
        if path_exists_in_ref(base_ref, path, cwd=cwd)
    }
    return sorted(upstream_paths)


def unexpected_changes(paths: list[str], allowed: frozenset[str]) -> list[str]:
    return [path for path in paths if path not in allowed]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check that fork commits only diverge from known upstream files",
    )
    parser.add_argument(
        "--base",
        default=os.environ.get(BASE_ENV, DEFAULT_BASE_REF),
        help=(
            "Upstream base ref to compare against "
            f"(default: {BASE_ENV} or {DEFAULT_BASE_REF})"
        ),
    )
    parser.add_argument(
        "--require-base",
        action="store_true",
        help="Fail instead of skipping when --base is unavailable",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not ref_exists(args.base):
            message = (
                "Upstream divergence guard skipped: "
                f"base ref {args.base!r} is not available"
            )
            if args.require_base:
                print(message, file=sys.stderr)
                return 1
            print(message)
            return 0

        tracked = upstream_tracked_changes(args.base)
    except RuntimeError as error:
        print(f"Upstream divergence guard failed: {error}", file=sys.stderr)
        return 1

    unexpected = unexpected_changes(tracked, ALLOWED_UPSTREAM_DIVERGENCE)
    if unexpected:
        print(
            "Unexpected upstream-tracked file divergence detected:",
            file=sys.stderr,
        )
        for path in unexpected:
            print(f"  {path}", file=sys.stderr)
        print(
            "Either move fork-only changes into fork-owned paths or document the "
            "new divergence in docs/architecture.md and docs/upstream-sync-guide.md.",
            file=sys.stderr,
        )
        return 1

    print(
        "Upstream divergence guard passed: "
        f"{len(tracked)} upstream-tracked file(s) changed"
    )
    for path in tracked:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
