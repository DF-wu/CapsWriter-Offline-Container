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
WEB_VERIFY_IMAGE = "capswriter-web-console:verify"
WEB_VERIFY_CONTAINER = "capswriter-web-console-verify"


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


def run_capture(args: list[str], *, cwd: Path = ROOT) -> tuple[int, str]:
    print(f"$ {' '.join(args)}", flush=True)
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        print(
            f"Command failed with exit code {completed.returncode}: {' '.join(args)}",
            file=sys.stderr,
        )
    return completed.returncode, completed.stdout


def run_cleanup(args: list[str], *, cwd: Path = ROOT) -> int:
    return subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode


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


def verify_server_tests() -> int:
    return run_required(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "fork_server/http_api/tests",
            "-v",
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


def _compact_for_match(value: str) -> str:
    return "".join(value.split()).casefold()


def verify_http_transcription(
    base_url: str,
    api_key: str,
    audio_path: str,
    expected_text: str,
) -> int:
    if not audio_path:
        return 0
    if not base_url:
        print("--http-audio requires --http-base-url", file=sys.stderr)
        return 1
    audio = Path(audio_path)
    if not audio.exists():
        print(f"HTTP transcription sample not found: {audio}", file=sys.stderr)
        return 1
    args = [
        sys.executable,
        "client/cli/capswriter_cli.py",
        "transcribe",
        "--base-url",
        base_url,
        "--timeout",
        "300",
        "--format",
        "text",
    ]
    if api_key:
        args.extend(["--key", api_key])
    args.append(str(audio))
    code, stdout = run_capture(args)
    if code != 0:
        return code

    transcript = stdout.strip()
    if not transcript:
        print("HTTP transcription returned empty text", file=sys.stderr)
        return 1
    if expected_text and _compact_for_match(expected_text) not in _compact_for_match(
        transcript
    ):
        print(
            "HTTP transcription did not contain expected text: "
            f"{expected_text!r}",
            file=sys.stderr,
        )
        return 1
    return 0


def verify_http(
    base_url: str,
    api_key: str,
    audio_path: str,
    expected_text: str,
) -> int:
    if not base_url:
        print("HTTP live check skipped (set --http-base-url or CAPSWRITER_VERIFY_HTTP_BASE)")
        return verify_http_transcription(base_url, api_key, audio_path, expected_text)
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
    code = run_required(args)
    if code != 0:
        return code
    return verify_http_transcription(base_url, api_key, audio_path, expected_text)


def verify_web_docker() -> int:
    if shutil.which("docker") is None:
        print("docker is required for --docker-build-web", file=sys.stderr)
        return 1
    code = run_required(
        [
            "docker",
            "build",
            "-f",
            "client/web/Dockerfile",
            "-t",
            WEB_VERIFY_IMAGE,
            "client/web",
        ]
    )
    if code != 0:
        return code

    run_cleanup(["docker", "rm", "-f", WEB_VERIFY_CONTAINER])
    code = run_required(
        [
            "docker",
            "run",
            "-d",
            "--name",
            WEB_VERIFY_CONTAINER,
            "-e",
            "CAPSWRITER_WEB_API_BASE=http://127.0.0.1:6017",
            "-e",
            "CAPSWRITER_WEB_RESPONSE_FORMAT=text",
            WEB_VERIFY_IMAGE,
        ]
    )
    if code != 0:
        return code
    try:
        return run_required(
            [
                "docker",
                "exec",
                WEB_VERIFY_CONTAINER,
                "sh",
                "-c",
                (
                    "healthy=0; "
                    "for i in $(seq 1 20); do "
                    "if wget -qO- http://127.0.0.1:8080/health | grep -qx ok; then "
                    "healthy=1; break; "
                    "fi; "
                    "sleep 0.5; "
                    "done; "
                    "test \"$healthy\" = 1 && "
                    "wget -qO- http://127.0.0.1:8080/config.js "
                    "| grep 'baseUrl: \"http://127.0.0.1:6017\"' "
                    "&& wget -qO- http://127.0.0.1:8080/config.js "
                    "| grep 'responseFormat: \"text\"'"
                ),
            ]
        )
    finally:
        run_cleanup(["docker", "rm", "-f", WEB_VERIFY_CONTAINER])


def clean_web_docker() -> int:
    if shutil.which("docker") is None:
        return 0
    run_cleanup(["docker", "rm", "-f", WEB_VERIFY_CONTAINER])
    run_cleanup(["docker", "image", "rm", "-f", WEB_VERIFY_IMAGE])
    return 0


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
    parser.add_argument(
        "--http-audio",
        default=os.environ.get("CAPSWRITER_VERIFY_HTTP_AUDIO", ""),
        help="Optional known audio file for a live transcription smoke test",
    )
    parser.add_argument(
        "--http-expect",
        default=os.environ.get("CAPSWRITER_VERIFY_HTTP_EXPECT", ""),
        help="Optional text expected in --http-audio transcription output",
    )
    parser.add_argument(
        "--docker-build-web",
        action="store_true",
        help="Also build the Web Console production Docker image",
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
            verify_server_tests,
            (lambda: 0 if args.skip_web else verify_web(install=not args.no_web_install)),
            (lambda: verify_web_docker() if args.docker_build_web else 0),
            (
                lambda: verify_http(
                    args.http_base_url,
                    args.http_key,
                    args.http_audio,
                    args.http_expect,
                )
            ),
        ]:
            status = step()
            if status != 0:
                break
        return status
    finally:
        clean_status = clean()
        docker_clean_status = clean_web_docker() if args.docker_build_web else 0
        if status == 0 and (clean_status != 0 or docker_clean_status != 0):
            raise SystemExit(clean_status or docker_clean_status)


if __name__ == "__main__":
    raise SystemExit(main())
