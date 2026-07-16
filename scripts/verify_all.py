#!/usr/bin/env python3
# coding: utf-8
"""Run the repository production-readiness verification gate.

The script keeps dependency installation scoped to project directories and
always runs cleanup before exiting.
"""

from __future__ import annotations

import argparse
import math
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "client" / "web"
LIVE_CHECK_TIMEOUT_SECONDS = 10
LIVE_TRANSCRIPTION_TIMEOUT_SECONDS = 600
VERIFY_STEP_TIMEOUT_ENV = "CAPSWRITER_VERIFY_STEP_TIMEOUT"
DEFAULT_VERIFY_STEP_TIMEOUT_SECONDS = 1800.0
TIMEOUT_EXIT_CODE = 124
WEB_VERIFY_RUN_ID = f"{os.getpid()}-{secrets.token_hex(8)}"
WEB_VERIFY_LABEL_KEY = "io.capswriter.verify.run"
WEB_VERIFY_LABEL = f"{WEB_VERIFY_LABEL_KEY}={WEB_VERIFY_RUN_ID}"
WEB_VERIFY_IMAGE = f"capswriter-web-console:verify-{WEB_VERIFY_RUN_ID}"
WEB_VERIFY_CONTAINER = f"capswriter-web-console-verify-{WEB_VERIFY_RUN_ID}"
REDACTED = "<redacted>"
SENSITIVE_FLAGS = {"--key", "--http-key"}
SENSITIVE_ENV_PREFIXES = (
    "CAPSWRITER_HTTP_API_KEY=",
    "CAPSWRITER_WEB_API_KEY=",
)


def redact_command_args(args: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for arg in args:
        if redact_next:
            redacted.append(REDACTED)
            redact_next = False
            continue
        if arg in SENSITIVE_FLAGS:
            redacted.append(arg)
            redact_next = True
            continue
        matched_prefix = next(
            (
                prefix
                for prefix in SENSITIVE_ENV_PREFIXES
                if arg.startswith(prefix)
            ),
            None,
        )
        if matched_prefix is not None:
            redacted.append(f"{matched_prefix}{REDACTED}")
            continue
        flag_with_value = next(
            (
                flag
                for flag in SENSITIVE_FLAGS
                if arg.startswith(f"{flag}=")
            ),
            None,
        )
        if flag_with_value is not None:
            redacted.append(f"{flag_with_value}={REDACTED}")
            continue
        redacted.append(arg)
    return redacted


def format_command(args: list[str]) -> str:
    return " ".join(redact_command_args(args))


def verify_step_timeout_seconds() -> float:
    value = os.environ.get(VERIFY_STEP_TIMEOUT_ENV, "").strip()
    if not value:
        return DEFAULT_VERIFY_STEP_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError(f"{VERIFY_STEP_TIMEOUT_ENV} must be a number") from exc
    if not math.isfinite(timeout):
        raise ValueError(f"{VERIFY_STEP_TIMEOUT_ENV} must be > 0")
    if timeout <= 0:
        raise ValueError(f"{VERIFY_STEP_TIMEOUT_ENV} must be > 0")
    return timeout


def _timeout_error(args: list[str], timeout: float) -> None:
    print(
        f"Command timed out after {timeout:g}s: {format_command(args)}",
        file=sys.stderr,
    )


def _timeout_output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run(
    args: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> int:
    print(f"$ {format_command(args)}", flush=True)
    try:
        timeout = verify_step_timeout_seconds()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=False,
            timeout=timeout,
        ).returncode
    except subprocess.TimeoutExpired:
        _timeout_error(args, timeout)
        return TIMEOUT_EXIT_CODE


def run_required(args: list[str], *, cwd: Path = ROOT) -> int:
    code = run(args, cwd=cwd)
    if code != 0:
        print(
            f"Command failed with exit code {code}: {format_command(args)}",
            file=sys.stderr,
        )
    return code


def run_capture(args: list[str], *, cwd: Path = ROOT) -> tuple[int, str]:
    print(f"$ {format_command(args)}", flush=True)
    try:
        timeout = verify_step_timeout_seconds()
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1, ""
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = _timeout_output_text(error.stdout)
        stderr = _timeout_output_text(error.stderr)
        if stdout:
            print(stdout, end="")
        if stderr:
            print(stderr, end="", file=sys.stderr)
        _timeout_error(args, timeout)
        return TIMEOUT_EXIT_CODE, stdout
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        print(
            f"Command failed with exit code {completed.returncode}: "
            f"{format_command(args)}",
            file=sys.stderr,
        )
    return completed.returncode, completed.stdout


def run_cleanup(args: list[str], *, cwd: Path = ROOT) -> int:
    try:
        timeout = verify_step_timeout_seconds()
    except ValueError:
        return 1
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        ).returncode
    except subprocess.TimeoutExpired:
        return TIMEOUT_EXIT_CODE


def verify_cli() -> int:
    return run_required([sys.executable, "client/cli/scripts/verify.py"])


def verify_upstream_divergence() -> int:
    return run_required([sys.executable, "scripts/check_upstream_divergence.py"])


def verify_docs() -> int:
    return run_required([sys.executable, "scripts/check_docs.py"])


def verify_server_compile() -> int:
    return run_required(
        [
            sys.executable,
            "-m",
            "compileall",
            "fork_server",
            "docker/server",
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


def verify_docker_server_tests() -> int:
    return run_required(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "docker/server/tests",
            "-v",
        ]
    )


def verify_scripts_tests() -> int:
    return run_required(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "scripts/tests",
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


def verify_web_browser_smoke(*, install: bool) -> int:
    code = ensure_web_deps(install=install)
    if code != 0:
        return code
    return run_required(["npm", "run", "browser-smoke"], cwd=WEB_ROOT)


def _compact_for_match(value: str) -> str:
    return "".join(value.split()).casefold()


def verify_http_transcription(
    base_url: str,
    api_key: str,
    api_key_file: str,
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
        str(LIVE_TRANSCRIPTION_TIMEOUT_SECONDS),
        "--format",
        "text",
    ]
    if api_key:
        args.extend(["--key", api_key])
    elif api_key_file:
        args.extend(["--key-file", api_key_file])
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
    api_key_file: str,
    audio_path: str,
    expected_text: str,
    require_ready: bool,
) -> int:
    if not base_url:
        print("HTTP live check skipped (set --http-base-url or CAPSWRITER_VERIFY_HTTP_BASE)")
        return verify_http_transcription(
            base_url,
            api_key,
            api_key_file,
            audio_path,
            expected_text,
        )
    args = [
        sys.executable,
        "client/cli/capswriter_cli.py",
        "health",
        "--base-url",
        base_url,
        "--timeout",
        str(LIVE_CHECK_TIMEOUT_SECONDS),
    ]
    if api_key:
        args.extend(["--key", api_key])
    elif api_key_file:
        args.extend(["--key-file", api_key_file])
    code = run_required(args)
    if code != 0:
        return code
    if require_ready:
        args = [
            sys.executable,
            "client/cli/capswriter_cli.py",
            "ready",
            "--base-url",
            base_url,
            "--timeout",
            str(LIVE_CHECK_TIMEOUT_SECONDS),
        ]
        if api_key:
            args.extend(["--key", api_key])
        elif api_key_file:
            args.extend(["--key-file", api_key_file])
        code = run_required(args)
        if code != 0:
            return code
    return verify_http_transcription(
        base_url,
        api_key,
        api_key_file,
        audio_path,
        expected_text,
    )


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
            "--label",
            WEB_VERIFY_LABEL,
            "-t",
            WEB_VERIFY_IMAGE,
            "client/web",
        ]
    )
    if code != 0:
        return code

    code = run_required(
        [
            "docker",
            "run",
            "-d",
            "--name",
            WEB_VERIFY_CONTAINER,
            "--label",
            WEB_VERIFY_LABEL,
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
                    "| grep 'responseFormat: \"text\"' "
                    "&& headers=\"$(wget -S -O /dev/null http://127.0.0.1:8080/ 2>&1)\" "
                    "&& printf '%s\\n' \"$headers\" "
                    "| grep -qi 'X-Content-Type-Options: nosniff' "
                    "&& printf '%s\\n' \"$headers\" "
                    "| grep -qi 'X-Frame-Options: DENY' "
                    "&& printf '%s\\n' \"$headers\" "
                    "| grep -qi 'Content-Security-Policy:'"
                ),
            ]
        )
    finally:
        _remove_owned_docker_resource("container", WEB_VERIFY_CONTAINER)


def _docker_resource_owned(resource_type: str, name: str) -> bool:
    """Return whether a Docker resource carries this verifier run's label."""
    if resource_type not in {"container", "image"}:
        raise ValueError(f"unsupported Docker resource type: {resource_type}")
    try:
        timeout = verify_step_timeout_seconds()
    except ValueError:
        return False
    try:
        completed = subprocess.run(
            [
                "docker",
                resource_type,
                "inspect",
                "--format",
                f'{{{{ index .Config.Labels "{WEB_VERIFY_LABEL_KEY}" }}}}',
                name,
            ],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0 and completed.stdout.strip() == WEB_VERIFY_RUN_ID


def _remove_owned_docker_resource(resource_type: str, name: str) -> int:
    if not _docker_resource_owned(resource_type, name):
        return 0
    if resource_type == "container":
        return run_cleanup(["docker", "rm", "-f", name])
    return run_cleanup(["docker", "image", "rm", "-f", name])


def clean_web_docker() -> int:
    if shutil.which("docker") is None:
        return 0
    container_status = _remove_owned_docker_resource(
        "container",
        WEB_VERIFY_CONTAINER,
    )
    image_status = _remove_owned_docker_resource("image", WEB_VERIFY_IMAGE)
    return container_status or image_status


def clean() -> int:
    code = run([sys.executable, "scripts/clean.py"])
    if code != 0:
        return code
    return run([sys.executable, "scripts/clean.py", "--check"])


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
        "--http-key-file",
        default=os.environ.get("CAPSWRITER_HTTP_API_KEY_FILE", ""),
        help="Optional UTF-8 file containing the live HTTP API Bearer token",
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
        "--http-require-ready",
        action="store_true",
        default=os.environ.get("CAPSWRITER_VERIFY_HTTP_REQUIRE_READY", "").lower()
        in {"1", "true", "yes", "on"},
        help="Require the live HTTP API /ready endpoint to return success",
    )
    parser.add_argument(
        "--docker-build-web",
        action="store_true",
        help="Also build the Web Console production Docker image",
    )
    parser.add_argument(
        "--web-browser-smoke",
        action="store_true",
        help="Also run a real-browser Web Console smoke test with the mock API",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    status = 0
    cleanup_status = 0
    try:
        for step in [
            verify_upstream_divergence,
            verify_docs,
            verify_cli,
            verify_server_compile,
            verify_server_tests,
            verify_docker_server_tests,
            verify_scripts_tests,
            (lambda: 0 if args.skip_web else verify_web(install=not args.no_web_install)),
            (
                lambda: verify_web_browser_smoke(install=not args.no_web_install)
                if args.web_browser_smoke
                else 0
            ),
            (lambda: verify_web_docker() if args.docker_build_web else 0),
            (
                lambda: verify_http(
                    args.http_base_url,
                    args.http_key,
                    args.http_key_file,
                    args.http_audio,
                    args.http_expect,
                    args.http_require_ready,
                )
            ),
        ]:
            status = step()
            if status != 0:
                break
    finally:
        clean_status = clean()
        docker_clean_status = clean_web_docker() if args.docker_build_web else 0
        cleanup_status = clean_status or docker_clean_status
        if cleanup_status != 0:
            print(
                "Verification cleanup failed "
                f"(repository={clean_status}, docker={docker_clean_status}); "
                "inspect and remove owned verification residue.",
                file=sys.stderr,
            )

    if status == 0 and cleanup_status != 0:
        return cleanup_status
    return status


if __name__ == "__main__":
    raise SystemExit(main())
