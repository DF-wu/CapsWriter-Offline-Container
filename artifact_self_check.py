# coding: utf-8
"""Deterministic, side-effect-light checks for packaged desktop executables.

The release workflow invokes this module through both frozen entrypoints.  It
does not construct a server/client, bind a socket, open an audio device, create
a tray icon, or load a model.  It verifies the portable directory layout and
imports the runtime surfaces that PyInstaller must make available.

Packaged builds record their variant in ``internal/capswriter-package.json``:
``full`` carries both executables and ``client`` only the desktop client.  The
client variant must not ship server-only runtimes, so their absence is part of
the same contract.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Iterable


SELF_CHECK_MARKER = "CAPSWRITER_ARTIFACT_SELF_CHECK="
PACKAGE_MARKER = Path("internal") / "capswriter-package.json"
PACKAGE_VARIANTS = ("full", "client")

REQUIRED_DIRECTORIES = ("core", "LLM", "assets", "docs", "models", "logs")
REQUIRED_ROOT_FILES = (
    "config_client.py",
    "config_server.py",
    "hot.txt",
    "hot-server.txt",
    "hot-rule.txt",
    "readme.md",
    "README.zh-TW.md",
    "LICENSE",
)
PACKAGED_DIRECTORIES = ("internal",)
PACKAGED_FILES = ("start_server.exe", "start_client.exe")

CLIENT_REQUIRED_DIRECTORIES = ("core", "LLM", "assets", "docs", "logs")
CLIENT_REQUIRED_ROOT_FILES = (
    "config_client.py",
    "hot.txt",
    "hot-rule.txt",
    "readme.md",
    "README.zh-TW.md",
    "LICENSE",
)
CLIENT_PACKAGED_FILES = ("start_client.exe",)
# A client package that still carries these is the full package mislabeled.
CLIENT_FORBIDDEN_PATHS = (
    "start_server.exe",
    "config_server.py",
    "hot-server.txt",
    "models",
    "core/server",
)
SERVER_ONLY_MODULES = (
    "fastapi",
    "fork_server",
    "gguf",
    "llvmlite",
    "numba",
    "onnxruntime",
    "sentencepiece",
    "sherpa_onnx",
    "uvicorn",
)

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
    "numba",
    "onnxruntime",
    "sentencepiece",
    "sherpa_onnx",
    "soundfile",
)
CLIENT_IMPORTS = COMMON_IMPORTS + (
    "config_client",
    "fork_client.autostart",
    "fork_client.console_window",
    "fork_client.settings",
    "fork_client.settings_ui",
    "tkinter.ttk",
    "core.client",
    "LLM.default",
    "keyboard",
    "pynput",
    "pyclip",
    "sounddevice",
    "openai",
    "ollama",
    "httpx",
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


def read_package_variant(root: Path) -> str:
    """Return the variant that build.spec recorded for a packaged artifact."""

    marker = Path(root) / PACKAGE_MARKER
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactSelfCheckError(
            f"cannot read package marker {PACKAGE_MARKER.as_posix()}: {exc}"
        ) from exc
    variant = data.get("variant") if isinstance(data, dict) else None
    if variant not in PACKAGE_VARIANTS:
        raise ArtifactSelfCheckError(f"unknown package variant: {variant!r}")
    return variant


def validate_artifact_layout(
    root: Path,
    *,
    packaged: bool,
    variant: str = "full",
) -> None:
    """Validate required paths and reject links/junctions in the whole tree."""

    if variant not in PACKAGE_VARIANTS:
        raise ValueError(f"unknown package variant: {variant}")
    root = Path(root)
    if not root.is_dir() or _is_link_or_junction(root):
        raise ArtifactSelfCheckError(f"artifact root is not a real directory: {root}")

    client_only = variant == "client"
    directories = CLIENT_REQUIRED_DIRECTORIES if client_only else REQUIRED_DIRECTORIES
    root_files = CLIENT_REQUIRED_ROOT_FILES if client_only else REQUIRED_ROOT_FILES
    packaged_files = CLIENT_PACKAGED_FILES if client_only else PACKAGED_FILES

    missing: list[str] = []
    for relative in directories:
        candidate = root / relative
        if not candidate.is_dir() or _is_link_or_junction(candidate):
            missing.append(relative + "/")
    for relative in root_files:
        candidate = root / relative
        if not candidate.is_file() or _is_link_or_junction(candidate):
            missing.append(relative)
    if packaged:
        for relative in PACKAGED_DIRECTORIES:
            candidate = root / relative
            if not candidate.is_dir() or _is_link_or_junction(candidate):
                missing.append(relative + "/")
        for relative in packaged_files:
            candidate = root / relative
            if not candidate.is_file() or _is_link_or_junction(candidate):
                missing.append(relative)
    if missing:
        raise ArtifactSelfCheckError(
            "missing required artifact paths: " + ", ".join(sorted(missing))
        )

    if client_only:
        unexpected = [
            relative for relative in CLIENT_FORBIDDEN_PATHS
            if (root / relative).exists()
        ]
        if unexpected:
            raise ArtifactSelfCheckError(
                "client package contains server paths: " + ", ".join(unexpected)
            )

    linked = _linked_paths(root)
    if linked:
        rendered = ", ".join(
            str(path.relative_to(root)) for path in sorted(linked)
        )
        raise ArtifactSelfCheckError(
            "artifact contains symbolic links or junctions: " + rendered
        )


def reject_server_only_modules(
    module_names: Iterable[str] = SERVER_ONLY_MODULES,
) -> None:
    """Fail when a client-only package can still import a server runtime."""

    present = [
        module_name for module_name in module_names
        if importlib.util.find_spec(module_name) is not None
    ]
    if present:
        raise ArtifactSelfCheckError(
            "client package bundles server-only modules: " + ", ".join(present)
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
    variant = "full"

    try:
        if is_packaged:
            variant = read_package_variant(selected_root)
        if variant == "client" and entrypoint != "client":
            raise ArtifactSelfCheckError(
                f"{entrypoint} entrypoint is not part of the client package"
            )
        validate_artifact_layout(selected_root, packaged=is_packaged, variant=variant)
        imported = import_runtime_surface(ENTRYPOINT_IMPORTS[entrypoint])
        if variant == "client":
            reject_server_only_modules()
    except Exception as exc:
        print(
            _render_report(
                {
                    "entrypoint": entrypoint,
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": "error",
                    "variant": variant,
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
                "variant": variant,
            }
        )
    )
    return 0
