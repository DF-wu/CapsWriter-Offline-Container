#!/usr/bin/env python3
# coding: utf-8
"""Run the repository production-readiness verification gate.

The script keeps dependency installation scoped to project directories and
always runs cleanup before exiting.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "client" / "web"


def run(
    args: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> int:
    print(f"$ {' '.join(args)}", flush=True)
    return subprocess.run(args, cwd=cwd, env=env, check=False).returncode


def run_required(args: list[str], *, cwd: Path = ROOT) -> int:
    code = run(args, cwd=cwd)
    if code != 0:
        print(f"Command failed with exit code {code}: {' '.join(args)}", file=sys.stderr)
    return code


def verify_cli() -> int:
    return run_required([sys.executable, "client/cli/scripts/verify.py"])


def verify_server_compile() -> int:
    return run_required(
        [
            sys.executable,
            "-m",
            "compileall",
            "fork_server",
            "check_http_api.py",
            "start_server_docker.py",
        ]
    )


def ensure_web_deps(*, install: bool) -> int:
    if shutil.which("npm") is None:
        print("npm is required for Web Console verification", file=sys.stderr)
        return 1
    if install:
        return run_required(["npm", "ci", "--no-audit", "--no-fund"], cwd=WEB_ROOT)
    if not (WEB_ROOT / "node_modules").exists():
        print(
            "client/web/node_modules is missing; run without --no-web-install "
            "or run npm ci in client/web",
            file=sys.stderr,
        )
        return 1
    return 0


def verify_web(*, install: bool) -> int:
    code = ensure_web_deps(install=install)
    if code != 0:
        return code
    return run_required(["npm", "run", "verify"], cwd=WEB_ROOT)


def verify_http(base_url: str, api_key: str) -> int:
    if not base_url:
        print("HTTP live check skipped (set --http-base-url or CAPSWRITER_VERIFY_HTTP_BASE)")
        return 0
    args = [
        sys.executable,
        "client/cli/capswriter_cli.py",
        "health",
        "--base-url",
        base_url,
        "--timeout",
        "10",
    ]
    if api_key:
        args.extend(["--key", api_key])
    return run_required(args)


def clean() -> int:
    return run([sys.executable, "scripts/clean.py"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CapsWriter fork verification gates")
    parser.add_argument(
        "--skip-web",
        action="store_true",
        help="Skip Web Console npm tests/build",
    )
    parser.add_argument(
        "--no-web-install",
        action="store_true",
        help="Do not run npm ci when node_modules is missing",
    )
    parser.add_argument(
        "--http-base-url",
        default=os.environ.get("CAPSWRITER_VERIFY_HTTP_BASE", ""),
        help="Optional live HTTP API root for a health check",
    )
    parser.add_argument(
        "--http-key",
        default=os.environ.get("CAPSWRITER_HTTP_API_KEY", ""),
        help="Optional live HTTP API Bearer token",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    status = 0
    try:
        for step in [
            verify_cli,
            verify_server_compile,
            (lambda: 0 if args.skip_web else verify_web(install=not args.no_web_install)),
            (lambda: verify_http(args.http_base_url, args.http_key)),
        ]:
            status = step()
            if status != 0:
                break
        return status
    finally:
        clean_status = clean()
        if status == 0 and clean_status != 0:
            raise SystemExit(clean_status)


if __name__ == "__main__":
    raise SystemExit(main())
