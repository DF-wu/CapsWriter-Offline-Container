#!/usr/bin/env python3
# coding: utf-8
"""Run the isolated OpenAI HTTP API contract suite without silent skips."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import re
import sys
from typing import Iterable, TextIO
import unittest


ROOT = Path(__file__).resolve().parents[1]
INPUT_REQUIREMENTS = ROOT / "requirements-api-test.txt"
REQUIREMENTS = ROOT / "requirements-api-test.lock"
PIN_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[A-Za-z0-9_.!+~-]+)"
)
REQUIRED_IMPORTS = {
    "colorama": ("colorama",),
    "fastapi": ("fastapi",),
    "httpx": ("httpx",),
    "openai": ("openai",),
    "python-multipart": ("multipart", "python_multipart"),
    "rich": ("rich",),
    "starlette": ("starlette",),
    "uvicorn": ("uvicorn",),
    "websockets": ("websockets",),
}
CONTRACT_TEST_ROOT = ROOT / "fork_server" / "http_api" / "tests"
REQUIRED_CONTRACT_MODULES = (
    "fork_server.http_api.tests.test_api_asgi",
    "fork_server.http_api.tests.test_body_limit",
    "fork_server.http_api.tests.test_openai_sdk_contract",
    "fork_server.http_api.tests.test_uvicorn_prebody",
)


def _normalize_distribution_name(name: str) -> str:
    return name.casefold().replace("_", "-")


def load_pins(path: Path = REQUIREMENTS) -> dict[str, str]:
    """Load exact pins from a direct input or hashed requirements lock."""
    pins: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("--hash="):
            continue
        match = PIN_PATTERN.fullmatch(entry.removesuffix("\\").strip())
        if match is None:
            raise ValueError(f"{path}:{line_number}: expected a strict name==version pin")
        name = _normalize_distribution_name(match.group("name"))
        if name in pins:
            raise ValueError(f"{path}:{line_number}: duplicate dependency pin: {name}")
        pins[name] = match.group("version")
    return pins


def lock_configuration_errors(
    direct_pins: dict[str, str],
    lock_pins: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    for name, version in sorted(direct_pins.items()):
        locked = lock_pins.get(name)
        if locked is None:
            errors.append(f"lock is missing direct dependency: {name}=={version}")
        elif locked != version:
            errors.append(
                f"lock mismatch for {name}: input {version}, lock {locked}"
            )
    return errors


def dependency_errors(pins: dict[str, str]) -> list[str]:
    """Return pin, installation, and import errors for the contract environment."""
    errors: list[str] = []
    required = set(REQUIRED_IMPORTS)
    configured = set(pins)
    for name in sorted(required - configured):
        errors.append(f"missing dependency pin: {name}")
    for distribution in sorted(configured):
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

        for module_name in REQUIRED_IMPORTS.get(distribution, ()):
            try:
                importlib.import_module(module_name)
            except Exception as error:  # pragma: no cover - exact import errors vary
                errors.append(
                    f"dependency import failed: {module_name} "
                    f"({type(error).__name__}: {error})"
                )
    return errors


class StrictTextTestResult(unittest.TextTestResult):
    """A unittest result where a skipped contract is not successful."""

    def wasSuccessful(self) -> bool:  # noqa: N802 - unittest API name
        return super().wasSuccessful() and not self.skipped


class StrictTextTestRunner(unittest.TextTestRunner):
    resultclass = StrictTextTestResult


def load_contract_suite(
    modules: Iterable[str] | None = None,
) -> unittest.TestSuite:
    if modules is None:
        suite = unittest.defaultTestLoader.discover(
            str(CONTRACT_TEST_ROOT),
        )
    else:
        suite = unittest.TestSuite()
        for module_name in modules:
            tests = unittest.defaultTestLoader.loadTestsFromName(module_name)
            if tests.countTestCases() == 0:
                raise RuntimeError(
                    f"contract test module contains no tests: {module_name}"
                )
            suite.addTests(tests)
    if suite.countTestCases() == 0:
        raise RuntimeError(f"API contract discovery found no tests under {CONTRACT_TEST_ROOT}")
    return suite


def run_contract_tests(
    modules: Iterable[str] | None = None,
    *,
    stream: TextIO = sys.stderr,
) -> int:
    result = StrictTextTestRunner(stream=stream, verbosity=2).run(
        load_contract_suite(modules)
    )
    if result.skipped:
        print("Skipped API contract tests are forbidden:", file=stream)
        for test, reason in result.skipped:
            print(f"- {test.id()}: {reason}", file=stream)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    try:
        direct_pins = load_pins(INPUT_REQUIREMENTS)
        pins = load_pins(REQUIREMENTS)
    except (OSError, ValueError) as error:
        print(f"API contract dependency configuration error: {error}", file=sys.stderr)
        return 1

    errors = lock_configuration_errors(direct_pins, pins)
    errors.extend(dependency_errors(pins))
    if errors:
        print("API contract dependency verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "Verified pinned API contract dependencies: "
        + ", ".join(f"{name}=={pins[name]}" for name in sorted(pins)),
        flush=True,
    )
    try:
        return run_contract_tests()
    except RuntimeError as error:
        print(f"API contract test discovery failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
