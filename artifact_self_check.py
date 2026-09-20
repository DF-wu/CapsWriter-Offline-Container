# coding: utf-8
"""Deterministic, side-effect-light checks for packaged desktop executables.

The release workflow invokes this module through both frozen entrypoints.  It
does not construct a server/client, bind a socket, open an audio device, create
a tray icon, or load a model.  It verifies the portable directory layout and
imports the runtime surfaces that PyInstaller must make available.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable


SELF_CHECK_MARKER = "CAPSWRITER_ARTIFACT_SELF_CHECK="

REQUIRED_DIRECTORIES = ("core", "LLM", "assets", "docs", "models", "logs")
REQUIRED_ROOT_FILES = (
    "config_client.py",
    "config_server.py",
    "hot.txt",
    "hot-server.txt",
    "hot-rule.txt",
    "readme.md",
    "README.en.md",
    "LICENSE",
)
PACKAGED_DIRECTORIES = ("internal",)
PACKAGED_FILES = ("start_server.exe", "start_client.exe")

COMMON_IMPORTS = (
    "PIL.Image",
    "numpy",
    "pypinyin",
    "pystray",
    "rich",
    "watchdog",
    "websockets",
)
SERVER_IMPORTS = COMMON_IMPORTS + (
    "config_server",
    "core.server.app",
    # GGUF engine implementations bind the separately supplied llama.cpp DLLs
    # at import time.  Probe the lazy factory here; model/runtime assets are a
    # release-machine acceptance layer, not part of this Python import smoke.
    "core.server.engines.factory",
    "core.server.engines.paraformer_onnx.asr_engine",
    "core.server.engines.sensevoice_onnx.asr_engine",
    "fork_server.bootstrap",
    "fork_server.http_api.api",
    "fastapi",
    "starlette",
    "uvicorn",
    "pydantic",
    "pydantic_core",
    "multipart",
    "python_multipart",
    "gguf",
    "onnxruntime",
    "sentencepiece",
    "sherpa_onnx",
    "soundfile",
)
CLIENT_IMPORTS = COMMON_IMPORTS + (
    "config_client",
    "core.client",
    "LLM.default",
    "keyboard",
    "pynput",
    "pyclip",
    "sounddevice",
    "openai",
    "ollama",
    "httpx",
    "numba",
    "rapidfuzz",
    "srt",
    "tkhtmlview",
    "typer",
)
ENTRYPOINT_IMPORTS = {
    "server": SERVER_IMPORTS,
    "client": CLIENT_IMPORTS,
}


class ArtifactSelfCheckError(RuntimeError):
    """Raised when a packaged artifact is incomplete or non-portable."""


def artifact_root() -> Path:
    """Return the distribution root for frozen and source invocations."""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _linked_paths(root: Path) -> list[Path]:
    linked: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in tuple(dirnames):
            candidate = parent / name
            if _is_link_or_junction(candidate):
                linked.append(candidate)
                dirnames.remove(name)
        for name in filenames:
            candidate = parent / name
            if _is_link_or_junction(candidate):
                linked.append(candidate)
    return linked


def validate_artifact_layout(root: Path, *, packaged: bool) -> None:
    """Validate required paths and reject links/junctions in the whole tree."""

    root = Path(root)
    if not root.is_dir() or _is_link_or_junction(root):
        raise ArtifactSelfCheckError(f"artifact root is not a real directory: {root}")

    missing: list[str] = []
    for relative in REQUIRED_DIRECTORIES:
        candidate = root / relative
        if not candidate.is_dir() or _is_link_or_junction(candidate):
            missing.append(relative + "/")
    for relative in REQUIRED_ROOT_FILES:
        candidate = root / relative
        if not candidate.is_file() or _is_link_or_junction(candidate):
            missing.append(relative)
    if packaged:
        for relative in PACKAGED_DIRECTORIES:
            candidate = root / relative
            if not candidate.is_dir() or _is_link_or_junction(candidate):
                missing.append(relative + "/")
        for relative in PACKAGED_FILES:
            candidate = root / relative
            if not candidate.is_file() or _is_link_or_junction(candidate):
                missing.append(relative)
    if missing:
        raise ArtifactSelfCheckError(
            "missing required artifact paths: " + ", ".join(sorted(missing))
        )

    linked = _linked_paths(root)
    if linked:
        rendered = ", ".join(
            str(path.relative_to(root)) for path in sorted(linked)
        )
        raise ArtifactSelfCheckError(
            "artifact contains symbolic links or junctions: " + rendered
        )


def import_runtime_surface(module_names: Iterable[str]) -> tuple[str, ...]:
    imported: list[str] = []
    for module_name in module_names:
        importlib.import_module(module_name)
        imported.append(module_name)
    return tuple(imported)


def _render_report(report: dict[str, object]) -> str:
    return SELF_CHECK_MARKER + json.dumps(
        report,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def run_artifact_self_check(
    entrypoint: str,
    *,
    root: Path | None = None,
    packaged: bool | None = None,
) -> int:
    """Run one frozen entrypoint's deterministic layout/import smoke check."""

    if entrypoint not in ENTRYPOINT_IMPORTS:
        raise ValueError(f"unknown artifact entrypoint: {entrypoint}")
    selected_root = artifact_root() if root is None else Path(root)
    is_packaged = bool(getattr(sys, "frozen", False)) if packaged is None else packaged

    try:
        validate_artifact_layout(selected_root, packaged=is_packaged)
        imported = import_runtime_surface(ENTRYPOINT_IMPORTS[entrypoint])
    except Exception as exc:
        print(
            _render_report(
                {
                    "entrypoint": entrypoint,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "error",
                }
            ),
            file=sys.stderr,
        )
        return 1

    print(
        _render_report(
            {
                "entrypoint": entrypoint,
                "frozen": is_packaged,
                "imports": len(imported),
                "python": f"{sys.version_info.major}.{sys.version_info.minor}",
                "status": "ok",
            }
        )
    )
    return 0
