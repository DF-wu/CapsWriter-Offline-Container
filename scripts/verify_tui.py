#!/usr/bin/env python3
# coding: utf-8
"""Run the isolated CapsWriter TUI suite without dependency or test skips."""

from __future__ import annotations

import importlib
from importlib import metadata
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import TextIO
import unittest


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements-tui.txt"
LOCK = ROOT / "requirements-tui.lock"
TESTS = ROOT / "client" / "tui" / "tests"
PIN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[A-Za-z0-9_.!+~-]+)"
)
LOCK_PIN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[A-Za-z0-9_.!+~-]+)"
    r"(?:\s*;\s*[^\\]+)?\s*\\?"
)
REQUIRED_IMPORTS = {
    "httpx": ("httpx",),
    "textual": ("textual",),
}
SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX = (3, 12)
DEFAULT_PIP_CHECK_TIMEOUT = 120.0


def _normalize_distribution_name(name: str) -> str:
    return name.casefold().replace("_", "-")


def load_direct_pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    """Load the complete direct TUI runtime dependency contract."""

    pins: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        match = PIN_PATTERN.fullmatch(entry)
        if match is None:
            raise ValueError(f"{path}:{line_number}: expected a strict name==version pin")
        name = _normalize_distribution_name(match.group("name"))
        if name in pins:
            raise ValueError(f"{path}:{line_number}: duplicate dependency pin: {name}")
        pins[name] = match.group("version")
    return pins


def load_locked_versions(path: Path = LOCK) -> dict[str, str]:
    """Read pinned versions from the generated, hash-locked requirements file."""

    versions: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        entry = line.strip()
        if not entry or entry.startswith(("#", "--hash=")):
            continue
        match = LOCK_PIN_PATTERN.fullmatch(entry)
        if match is None:
            continue
        name = _normalize_distribution_name(match.group("name"))
        if name in versions:
            raise ValueError(f"{path}:{line_number}: duplicate locked dependency: {name}")
        versions[name] = match.group("version")
    return versions


def configuration_errors(
    pins: dict[str, str],
    locked_versions: dict[str, str],
) -> list[str]:
    """Return missing, unverified, and direct-pin/lock parity failures."""

    errors: list[str] = []
    required = set(REQUIRED_IMPORTS)
    configured = set(pins)
    for name in sorted(required - configured):
        errors.append(f"missing dependency pin: {name}")
    for name in sorted(configured - required):
        errors.append(f"unverified dependency pin: {name}")
    for name in sorted(configured):
        locked = locked_versions.get(name)
        if locked is None:
            errors.append(f"direct dependency is missing from lock: {name}=={pins[name]}")
        elif locked != pins[name]:
            errors.append(
                f"direct dependency lock mismatch: {name} "
                f"expected {pins[name]}, found {locked}"
            )
    return errors


def dependency_errors(pins: dict[str, str]) -> list[str]:
    """Return installation, version, and import errors for direct runtime pins."""

    errors: list[str] = []
    for distribution in sorted(set(REQUIRED_IMPORTS) & set(pins)):
        expected_version = pins[distribution]
        try:
            installed_version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            errors.append(f"dependency is not installed: {distribution}=={expected_version}")
        else:
            if installed_version != expected_version:
                errors.append(
                    f"dependency version mismatch: {distribution} "
                    f"expected {expected_version}, found {installed_version}"
                )

        for module_name in REQUIRED_IMPORTS[distribution]:
            try:
                importlib.import_module(module_name)
            except Exception as error:  # pragma: no cover - platform import details vary
                errors.append(
                    f"dependency import failed: {module_name} "
                    f"({type(error).__name__}: {error})"
                )
    return errors


def runtime_error(version_info: tuple[int, ...] | None = None) -> str | None:
    """Require a Python version covered by the universal 3.10--3.12 lock."""

    current = tuple(version_info or sys.version_info)[:2]
    if SUPPORTED_PYTHON_MIN <= current <= SUPPORTED_PYTHON_MAX:
        return None
    return (
        "unsupported Python runtime: "
        f"{current[0]}.{current[1]} (expected 3.10 through 3.12)"
    )


def pip_check_timeout() -> float:
    raw = os.environ.get(
        "CAPSWRITER_TUI_PIP_CHECK_TIMEOUT",
        str(DEFAULT_PIP_CHECK_TIMEOUT),
    )
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("CAPSWRITER_TUI_PIP_CHECK_TIMEOUT must be a number") from exc
    if not 0 < value <= 600:
        raise ValueError("CAPSWRITER_TUI_PIP_CHECK_TIMEOUT must be > 0 and <= 600")
    return value


def run_pip_check(timeout: float | None = None) -> list[str]:
    """Use pip's installed-metadata validator for the resolved transitive set."""

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout if timeout is not None else pip_check_timeout(),
        )
    except subprocess.TimeoutExpired:
        return ["pip check timed out"]
    output = completed.stdout.strip()
    if completed.returncode:
        return [f"pip check failed: {output or 'no diagnostic output'}"]
    return []


class StrictTextTestResult(unittest.TextTestResult):
    """A unittest result where a skipped TUI test is a release failure."""

    def wasSuccessful(self) -> bool:  # noqa: N802 - unittest API name
        return super().wasSuccessful() and not self.skipped


class StrictTextTestRunner(unittest.TextTestRunner):
    resultclass = StrictTextTestResult


def load_tui_suite() -> unittest.TestSuite:
    suite = unittest.defaultTestLoader.discover(
        str(TESTS),
        top_level_dir=str(ROOT),
    )
    if suite.countTestCases() == 0:
        raise RuntimeError(f"TUI test discovery found no tests under {TESTS}")
    return suite


def run_tui_tests(*, stream: TextIO = sys.stderr) -> int:
    result = StrictTextTestRunner(stream=stream, verbosity=2).run(load_tui_suite())
    if result.skipped:
        print("Skipped TUI tests are forbidden:", file=stream)
        for test, reason in result.skipped:
            print(f"- {test.id()}: {reason}", file=stream)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    unsupported = runtime_error()
    if unsupported:
        print(f"TUI runtime verification failed: {unsupported}", file=sys.stderr)
        return 1

    try:
        pins = load_direct_pins()
        locked_versions = load_locked_versions()
        errors = configuration_errors(pins, locked_versions)
        errors.extend(dependency_errors(pins))
        errors.extend(run_pip_check())
    except (OSError, ValueError) as error:
        print(f"TUI dependency configuration error: {error}", file=sys.stderr)
        return 1

    if errors:
        print("TUI dependency verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Verified pinned TUI dependencies: "
        + ", ".join(f"{name}=={pins[name]}" for name in sorted(pins)),
        flush=True,
    )
    try:
        return run_tui_tests()
    except RuntimeError as error:
        print(f"TUI test discovery failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
