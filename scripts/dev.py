#!/usr/bin/env python3
"""Isolated, explicit development commands; run --help for targets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "client" / "web"
ENV_ROOT = ROOT / ".venv-dev"
PROFILES = ("dev", "desktop", "server", "api", "tui", "build", "web")
LOCKS = {
    "desktop": "requirements-desktop-dev.lock",
    "server": "requirements-server-docker.lock",
    "api": "requirements-api-test.lock",
    "tui": "requirements-tui.lock",
    "build": "requirements-windows-build.lock",
}
TIMEOUT = 1800


class DevError(Exception):
    """An actionable prerequisite or command failure."""


def child_env() -> dict[str, str]:
    # Explicit interpreter paths must win over an activated/unrelated environment.
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX"}
        and not key.startswith("UV_")
    }
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def executable(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        hints = {
            "uv": "Install uv from https://docs.astral.sh/uv/getting-started/installation/",
            "npm": "Install Node.js 24 LTS (includes npm)",
        }
        raise DevError(f"{name} is missing from PATH. {hints.get(name, 'Install ' + name)}.")
    return resolved


def run(args: list[str], *, cwd: Path = ROOT, foreground: bool = False) -> None:
    # Only fixed command names are shown: user API keys in inherited env stay private.
    print(f"Running {Path(args[0]).name} ({cwd.relative_to(ROOT) or '.'})", flush=True)
    try:
        result = subprocess.run(
            args, cwd=cwd, env=child_env(), check=False,
            timeout=None if foreground else TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise DevError(f"Command exceeded {TIMEOUT}s; inspect the output and retry.") from exc
    except OSError as exc:
        raise DevError(f"Could not start {Path(args[0]).name}: {exc}") from exc
    if result.returncode:
        raise DevError(f"Command failed with exit code {result.returncode}; see output above.")


def profile_python(profile: str) -> Path:
    return ENV_ROOT / profile / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def validate_platform(profile: str) -> None:
    if profile in {"desktop", "build"} and sys.platform != "win32":
        raise DevError(f"{profile} requires Windows x64. Use setup dev/api/tui/web on Linux.")
    if profile in {"server", "api"} and sys.platform != "linux":
        raise DevError(f"{profile} uses the Linux lock; run this profile in a Linux container or WSL.")
    if profile in {"desktop", "build", "server", "api"} and platform.machine().lower() not in {
        "x86_64", "amd64",
    }:
        raise DevError(f"{profile} requires x86-64 for the reviewed dependency lock.")


def ready_python(profile: str) -> str:
    validate_platform(profile)
    target = profile_python(profile)
    if not target.is_file() or not (ENV_ROOT / profile / ".capswriter-ready").is_file():
        raise DevError(f"Run python scripts/dev.py setup {profile} first.")
    return str(target)


def setup(profile: str) -> None:
    if profile == "web":
        run([executable("npm"), "ci", "--no-audit", "--no-fund"], cwd=WEB_ROOT)
        return
    validate_platform(profile)
    uv = executable("uv")
    target = ENV_ROOT / profile
    # A profile never installs into symlinked environments or a global interpreter.
    if ENV_ROOT.is_symlink() or target.is_symlink():
        raise DevError(".venv-dev and its profile directories must not be symbolic links.")
    marker = target / ".capswriter-ready"
    marker.unlink(missing_ok=True)
    version = "3.10" if profile == "server" else "3.12"
    run([uv, "venv", "--no-config", "--no-project", "--allow-existing", "--python", version, str(target)])
    python = str(profile_python(profile))
    if profile in {"desktop", "server", "build", "tui"}:
        # srt needs build tooling; the strict TUI verifier also requires pip check.
        # uv venv intentionally does not seed either tool by default.
        run([
            uv, "pip", "install", "--no-config", "--python", python,
            "--require-hashes", "--only-binary=:all:", "--no-deps",
            "-r", str(ROOT / "requirements-windows-build-bootstrap.lock"),
        ])
    if profile in LOCKS:
        args = [uv, "pip", "install", "--no-config", "--python", python, "--require-hashes", "--only-binary=:all:"]
        if profile in {"desktop", "server", "build"}:
            args.extend(["--no-binary=srt", "--no-build-isolation"])
        run([*args, "-r", str(ROOT / LOCKS[profile])])
        run([uv, "pip", "check", "--no-config", "--python", python])
    marker.write_text(f"{profile}\nPython {version}\n", encoding="utf-8")
    print(f"Ready: {profile}. No activation is needed.")


def npm(action: str, *, foreground: bool = False) -> None:
    if not (WEB_ROOT / "node_modules").is_dir():
        raise DevError("Run python scripts/dev.py setup web first.")
    run([executable("npm"), "run", action], cwd=WEB_ROOT, foreground=foreground)


def dispatch(args: argparse.Namespace) -> None:
    if args.command == "setup":
        setup(args.profile)
    elif args.command == "check":
        for name in ("uv", "npm"):
            resolved = shutil.which(name)
            print(f"{name}: {'available' if resolved else 'not installed'}")
        for profile in PROFILES[:-1]:
            print(f"{profile}: {'ready' if (ENV_ROOT / profile / '.capswriter-ready').is_file() else 'not set up'}")
        run([sys.executable, "scripts/check_docs.py"])
    elif args.command == "client":
        extra = ["--settings"] if args.settings else []
        run([ready_python("desktop"), "start_client.py", *extra], foreground=True)
    elif args.command == "server":
        run([ready_python("server"), "start_server_docker.py"], foreground=True)
    elif args.command == "tui":
        run([ready_python("tui"), "-m", "client.tui"], foreground=True)
    elif args.command == "web":
        npm("dev", foreground=True)
    elif args.command == "test":
        if args.target == "web":
            npm("verify")
        elif args.target in {"api", "tui"}:
            script = "verify_api_contract.py" if args.target == "api" else "verify_tui.py"
            run([ready_python(args.target), f"scripts/{script}"])
        else:
            python = ready_python("dev")
            for directory in ("scripts/tests", "docker/server/tests", "client/cli/tests"):
                run([python, "-m", "unittest", "discover", "-s", directory, "-v"])
    elif args.command == "build":
        if args.target == "desktop":
            run([ready_python("build"), "-m", "PyInstaller", "--clean", "--noconfirm", "build.spec"])
        elif args.target == "web":
            npm("build")
        else:
            run([ready_python("dev"), "client/cli/scripts/build_zipapp.py"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("setup", help="Create an isolated profile; install locked dependencies")
    install.add_argument("profile", choices=PROFILES, nargs="?", default="dev")
    commands.add_parser("check", help="Show prerequisites/profiles and verify documentation")
    client = commands.add_parser("client", help="Run the Windows hotkey client against your configured server")
    client.add_argument("--settings", action="store_true", help="Open the graphical client settings")
    commands.add_parser("server", help="Run a Linux ASR server in foreground; needs models and free ports")
    commands.add_parser("tui", help="Run the optional terminal client")
    commands.add_parser("web", help="Run the Web frontend at the Vite loopback URL; ASR runs separately")
    tests = commands.add_parser("test", help="Run unit, API contract, TUI or Web checks")
    tests.add_argument("target", choices=("unit", "api", "tui", "web"), nargs="?", default="unit")
    build = commands.add_parser("build", help="Build the Windows package, Web frontend or portable CLI")
    build.add_argument("target", choices=("desktop", "web", "cli"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dispatch(args)
    except DevError as exc:
        print(f"Development setup error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
