# coding: utf-8

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTTP_API_REQUIREMENTS = {"fastapi", "starlette", "uvicorn", "python-multipart"}
ASR_AUDIO_REQUIREMENTS = {"sentencepiece", "soundfile"}
HTTP_API_TEST_REQUIREMENTS = {
    "colorama",
    "fastapi",
    "httpx",
    "openai",
    "python-multipart",
    "rich",
    "starlette",
    "uvicorn",
    "websockets",
}
API_TEST_LOCK_REQUIREMENTS = {
    "annotated-doc": "0.0.4",
    "annotated-types": "0.7.0",
    "anyio": "4.14.2",
    "certifi": "2026.6.17",
    "click": "8.4.2",
    "colorama": "0.4.6",
    "distro": "1.9.0",
    "fastapi": "0.139.0",
    "h11": "0.16.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "idna": "3.18",
    "jiter": "0.16.0",
    "markdown-it-py": "4.2.0",
    "mdurl": "0.1.2",
    "openai": "2.45.0",
    "pydantic": "2.13.4",
    "pydantic-core": "2.46.4",
    "pygments": "2.20.0",
    "python-multipart": "0.0.32",
    "rich": "14.3.3",
    "sniffio": "1.3.1",
    "starlette": "1.3.1",
    "tqdm": "4.68.4",
    "typing-extensions": "4.16.0",
    "typing-inspection": "0.4.2",
    "uvicorn": "0.32.1",
    "websockets": "16.0",
}
SUPPORTED_HTTP_STACK = {
    "fastapi": "0.139.0",
    "starlette": "1.3.1",
    "uvicorn": "0.32.1",
    "python-multipart": "0.0.32",
}
SUPPORTED_TUI_RUNTIME = {
    "httpx": "0.28.1",
    "textual": "8.2.8",
}
TUI_LOCK_REQUIREMENTS = {
    "anyio": "4.14.2",
    "certifi": "2026.6.17",
    "exceptiongroup": "1.3.1",
    "h11": "0.16.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "idna": "3.18",
    "linkify-it-py": "2.1.0",
    "markdown-it-py": "4.2.0",
    "mdit-py-plugins": "0.6.1",
    "mdurl": "0.1.2",
    "platformdirs": "4.10.0",
    "pygments": "2.20.0",
    "rich": "15.0.0",
    "textual": "8.2.8",
    "typing-extensions": "4.16.0",
    "uc-micro-py": "2.0.0",
}
DOCKERFILE = ROOT / "docker" / "server" / "Dockerfile"
WINDOWS_BUILD_SPEC = ROOT / "build.spec"
WINDOWS_BUILD_LOCK = ROOT / "requirements-windows-build.lock"
WINDOWS_BUILD_BOOTSTRAP_LOCK = ROOT / "requirements-windows-build-bootstrap.lock"
WINDOWS_BUILD_BOOTSTRAP_REQUIREMENTS = {
    "pip": (
        "26.1.2",
        "382ff9f685ee3bc25864f820aa50505825f10f5458ffff07e30a6d96e5715cab",
    ),
    "setuptools": (
        "83.0.0",
        "29b23c360f22f414dc7336bb39178cc7bcbf6021ed2733cde173f09dba19abb3",
    ),
}
WINDOWS_BUILD_RUNTIME_REQUIREMENTS = {
    "fastapi",
    "onnxruntime-directml",
    "pillow",
    "pydantic",
    "pydantic-core",
    "pyinstaller",
    "python-multipart",
    "sentencepiece",
    "sherpa-onnx",
    "soundfile",
    "starlette",
    "uvicorn",
}


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("-"):
            continue
        entry = entry.removesuffix("\\").strip()
        name = entry.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].strip()
        names.add(name.casefold())
    return names


def requirement_entries(path: Path) -> list[str]:
    entries: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("-"):
            if not current:
                entries.append(stripped)
            else:
                current.append(line.rstrip())
            continue
        if current:
            entries.append("\n".join(current))
        current = [line.rstrip()]
    if current:
        entries.append("\n".join(current))
    return entries


def pinned_version(entry: str) -> tuple[str, str] | None:
    first_line = entry.splitlines()[0]
    if "==" not in first_line:
        return None
    package, version = first_line.split("==", 1)
    name = package.split("[", 1)[0].strip().casefold()
    return name, version.strip().split()[0]


class RequirementsTest(unittest.TestCase):
    def test_windows_build_bootstrap_is_minimal_exact_and_hashed(self) -> None:
        entries = requirement_entries(WINDOWS_BUILD_BOOTSTRAP_LOCK)
        self.assertEqual(len(entries), len(WINDOWS_BUILD_BOOTSTRAP_REQUIREMENTS))

        observed: dict[str, tuple[str, str]] = {}
        for entry in entries:
            first_line, *hash_lines = entry.splitlines()
            self.assertRegex(first_line, r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+~-]+ \\$" )
            self.assertEqual(len(hash_lines), 1)
            match = re.fullmatch(
                r"    --hash=sha256:([0-9a-f]{64})",
                hash_lines[0],
            )
            self.assertIsNotNone(match)
            package, version = first_line.removesuffix(" \\").split("==", 1)
            observed[package.casefold()] = (version, match.group(1))

        self.assertEqual(observed, WINDOWS_BUILD_BOOTSTRAP_REQUIREMENTS)

    def test_server_requirements_include_http_api_runtime(self) -> None:
        for filename in ("requirements-server.txt", "requirements-server-docker.txt"):
            with self.subTest(filename=filename):
                names = requirement_names(ROOT / filename)
                self.assertTrue(
                    HTTP_API_REQUIREMENTS <= names,
                    f"{filename} missing {sorted(HTTP_API_REQUIREMENTS - names)}",
                )
                self.assertTrue(
                    ASR_AUDIO_REQUIREMENTS <= names,
                    f"{filename} missing {sorted(ASR_AUDIO_REQUIREMENTS - names)}",
                )

    def test_supported_http_stack_is_explicitly_pinned_everywhere(self) -> None:
        for filename in (
            "requirements-server.txt",
            "requirements-server-docker.txt",
            "requirements-api-test.txt",
        ):
            with self.subTest(filename=filename):
                versions = dict(
                    item
                    for entry in requirement_entries(ROOT / filename)
                    if (item := pinned_version(entry)) is not None
                )
                self.assertEqual(
                    {name: versions.get(name) for name in SUPPORTED_HTTP_STACK},
                    SUPPORTED_HTTP_STACK,
                )

    def test_api_contract_test_requirements_are_complete_and_pinned(self) -> None:
        path = ROOT / "requirements-api-test.txt"
        self.assertEqual(requirement_names(path), HTTP_API_TEST_REQUIREMENTS)
        for entry in requirement_entries(path):
            with self.subTest(entry=entry):
                self.assertRegex(
                    entry,
                    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?==[A-Za-z0-9_.!+~-]+$",
                )

    def test_api_contract_lock_is_complete_hashed_python312_linux(self) -> None:
        path = ROOT / "requirements-api-test.lock"
        source = path.read_text(encoding="utf-8")
        entries = requirement_entries(path)
        versions = dict(
            item
            for entry in entries
            if (item := pinned_version(entry)) is not None
        )

        self.assertEqual(versions, API_TEST_LOCK_REQUIREMENTS)
        direct_versions = dict(
            item
            for entry in requirement_entries(ROOT / "requirements-api-test.txt")
            if (item := pinned_version(entry)) is not None
        )
        self.assertEqual(
            {name: versions.get(name) for name in direct_versions},
            direct_versions,
        )
        self.assertIn("--python-version 3.12", source)
        self.assertIn("--python-platform x86_64-manylinux_2_17", source)
        self.assertIn("--only-binary :all:", source)
        for entry in entries:
            with self.subTest(entry=entry.splitlines()[0]):
                first_line, *hash_lines = entry.splitlines()
                self.assertRegex(
                    first_line,
                    r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+~-]+ \\$",
                )
                self.assertTrue(hash_lines)
                for index, hash_line in enumerate(hash_lines):
                    suffix = r" \\" if index < len(hash_lines) - 1 else ""
                    self.assertRegex(
                        hash_line,
                        r"^    --hash=sha256:[0-9a-f]{64}" + suffix + r"$",
                    )

    def test_tui_direct_requirements_are_complete_and_pinned(self) -> None:
        path = ROOT / "requirements-tui.txt"
        versions = dict(
            item
            for entry in requirement_entries(path)
            if (item := pinned_version(entry)) is not None
        )

        self.assertEqual(versions, SUPPORTED_TUI_RUNTIME)
        for entry in requirement_entries(path):
            with self.subTest(entry=entry):
                self.assertRegex(
                    entry,
                    r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+~-]+$",
                )

    def test_tui_lock_is_complete_hashed_and_python_310_312_compatible(self) -> None:
        path = ROOT / "requirements-tui.lock"
        source = path.read_text(encoding="utf-8")
        entries = requirement_entries(path)
        versions = dict(
            item
            for entry in entries
            if (item := pinned_version(entry)) is not None
        )

        self.assertEqual(versions, TUI_LOCK_REQUIREMENTS)
        self.assertEqual(
            {name: versions.get(name) for name in SUPPORTED_TUI_RUNTIME},
            SUPPORTED_TUI_RUNTIME,
        )
        self.assertIn("--universal --python-version 3.10", source)
        self.assertIn("exceptiongroup==1.3.1 ; python_full_version < '3.11'", source)
        for entry in entries:
            with self.subTest(entry=entry.splitlines()[0]):
                first_line, *hash_lines = entry.splitlines()
                self.assertRegex(
                    first_line,
                    r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+~-]+"
                    r"(?: ; python_full_version < '3\.11')? \\?$",
                )
                self.assertTrue(hash_lines)
                for index, hash_line in enumerate(hash_lines):
                    suffix = r" \\" if index < len(hash_lines) - 1 else ""
                    self.assertRegex(
                        hash_line,
                        r"^    --hash=sha256:[0-9a-f]{64}" + suffix + r"$",
                    )

    def test_docker_server_lock_is_pinned_and_used_by_image_build(self) -> None:
        lock_path = ROOT / "requirements-server-docker.lock"
        lock_entries = requirement_entries(lock_path)
        lock_names = requirement_names(lock_path)
        declared_names = requirement_names(ROOT / "requirements-server-docker.txt")
        lock_versions = dict(
            item
            for entry in lock_entries
            if (item := pinned_version(entry)) is not None
        )
        declared_versions = dict(
            item
            for entry in requirement_entries(ROOT / "requirements-server-docker.txt")
            if (item := pinned_version(entry)) is not None
        )

        self.assertTrue(declared_names <= lock_names)
        self.assertEqual(
            {name: lock_versions.get(name) for name in declared_versions},
            declared_versions,
        )
        lock_source = lock_path.read_text(encoding="utf-8")
        self.assertIn("--python-version 3.10", lock_source)
        self.assertIn("--python-platform x86_64-manylinux_2_28", lock_source)
        self.assertIn("--only-binary=:all:", lock_source)
        self.assertIn("--no-binary=srt", lock_source)
        for entry in lock_entries:
            with self.subTest(entry=entry.splitlines()[0]):
                first_line, *hash_lines = entry.splitlines()
                self.assertRegex(
                    first_line,
                    r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+~-]+ \\$",
                )
                self.assertTrue(hash_lines)
                for index, hash_line in enumerate(hash_lines):
                    suffix = r" \\" if index < len(hash_lines) - 1 else ""
                    self.assertRegex(
                        hash_line,
                        r"^    --hash=sha256:[0-9a-f]{64}" + suffix + r"$",
                    )

        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("COPY requirements-server-docker.txt requirements-server-docker.lock /app/", dockerfile)
        self.assertIn(
            "python -m pip install --require-hashes --no-build-isolation -r /app/requirements-server-docker.lock",
            dockerfile,
        )

    def test_windows_build_lock_is_complete_hashed_python312_x86_64(self) -> None:
        source = WINDOWS_BUILD_LOCK.read_text(encoding="utf-8")
        entries = requirement_entries(WINDOWS_BUILD_LOCK)
        names = requirement_names(WINDOWS_BUILD_LOCK)
        direct_names = requirement_names(ROOT / "requirements-client.txt") | requirement_names(
            ROOT / "requirements-server.txt"
        )
        versions = dict(
            item
            for entry in entries
            if (item := pinned_version(entry)) is not None
        )
        direct_versions = dict(
            item
            for filename in ("requirements-client.txt", "requirements-server.txt")
            for entry in requirement_entries(ROOT / filename)
            if (item := pinned_version(entry)) is not None
        )

        self.assertGreaterEqual(len(entries), 60)
        self.assertTrue(direct_names <= names)
        self.assertTrue(WINDOWS_BUILD_RUNTIME_REQUIREMENTS <= names)
        self.assertEqual(
            {name: versions.get(name) for name in direct_versions},
            direct_versions,
        )
        self.assertIn("--python-version 3.12", source)
        self.assertIn("--python-platform x86_64-pc-windows-msvc", source)
        self.assertIn("--generate-hashes", source)
        self.assertIn("--only-binary=:all:", source)
        self.assertIn("--no-binary=srt", source)
        self.assertIn("--output-file requirements-windows-build.lock", source)

        for entry in entries:
            with self.subTest(entry=entry.splitlines()[0]):
                first_line, *hash_lines = entry.splitlines()
                self.assertRegex(
                    first_line,
                    r"^[A-Za-z0-9_.-]+==[A-Za-z0-9_.!+~-]+ \\$",
                )
                self.assertTrue(hash_lines)
                for index, hash_line in enumerate(hash_lines):
                    suffix = r" \\" if index < len(hash_lines) - 1 else ""
                    self.assertRegex(
                        hash_line,
                        r"^    --hash=sha256:[0-9a-f]{64}" + suffix + r"$",
                    )

    def test_windows_server_build_includes_fork_http_runtime(self) -> None:
        source = WINDOWS_BUILD_SPEC.read_text(encoding="utf-8")

        self.assertIn("['start_server_universal.py']", source)
        self.assertNotIn("['start_server_docker.py']", source)
        self.assertIn("hiddenimports=server_hiddenimports", source)
        self.assertIn("'soundfile'", source)
        self.assertIn("'README.en.md'", source)
        self.assertIn("filter=lambda name: '.tests' not in name", source)
        for package in (
            "fork_server",
            "fastapi",
            "starlette",
            "uvicorn",
            "pydantic",
            "pydantic_core",
            "multipart",
            "python_multipart",
        ):
            with self.subTest(package=package):
                self.assertIn(repr(package), source)

        server_analysis = source.split("a_1 = Analysis(", 1)[1].split(
            "a_2 = Analysis(", 1
        )[0]
        self.assertNotIn("'pydantic'", server_analysis.split("excludes=", 1)[1])

        client_analysis = source.split("a_2 = Analysis(", 1)[1].split(
            "# 客户端也过滤", 1
        )[0]
        self.assertIn("hiddenimports=hiddenimports", client_analysis)
        self.assertNotIn("server_hiddenimports", client_analysis)

    def test_windows_build_fails_closed_and_copies_portable_payload(self) -> None:
        source = WINDOWS_BUILD_SPEC.read_text(encoding="utf-8")

        self.assertIn("require_importable('sherpa-onnx', 'sherpa_onnx')", source)
        self.assertIn("require_importable('Pillow', 'PIL')", source)
        self.assertIn("require_importable('sentencepiece')", source)
        self.assertIn("sys.version_info[:2] != (3, 12)", source)
        self.assertIn("sys.maxsize <= 2**32", source)
        self.assertIn("raise RuntimeError(f'Cannot collect HTTP API package {package}')", source)
        self.assertIn("required_folders = ('core', 'LLM', 'assets', 'docs')", source)
        self.assertIn("mutable_folders = ('models', 'logs')", source)
        self.assertIn("copytree(", source)
        self.assertIn("dirs_exist_ok=True", source)
        self.assertIn("ignore=portable_copy_ignore", source)
        self.assertIn("is_link_or_junction(candidate)", source)
        self.assertIn("Windows artifact mutable path must be empty", source)
        tree = ast.parse(source, filename=str(WINDOWS_BUILD_SPEC))
        self.assertIsInstance(tree.body[0], ast.Expr)
        module_docstring_node = tree.body[0].value
        self.assertIsInstance(module_docstring_node, ast.Constant)
        non_ascii_runtime_strings = [
            (node.lineno, node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node is not module_docstring_node
            and not node.value.isascii()
        ]
        self.assertEqual([], non_ascii_runtime_strings)
        self.assertNotIn("mklink", source.casefold())
        self.assertNotIn("shell=True", source)
        self.assertNotIn("core_server.py", source)
        self.assertNotIn("core_client.py", source)
        self.assertNotIn("except:\n", source)


if __name__ == "__main__":
    unittest.main()
