# coding: utf-8
"""Dependency-light desktop hotkey capability and event normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import platform


@dataclass(frozen=True)
class HotkeyBackendSupport:
    available: bool
    backend: str
    selective_suppression: bool
    detail: str


def detect_hotkey_backend(
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> HotkeyBackendSupport:
    """Return the desktop backend CapsWriter can support without guessing."""

    name = platform_name or platform.system()
    env = os.environ if environ is None else environ

    if name == "Windows":
        return HotkeyBackendSupport(
            available=True,
            backend="win32",
            selective_suppression=True,
            detail="Windows low-level keyboard and mouse hooks",
        )

    if name == "Linux":
        session_type = env.get("XDG_SESSION_TYPE", "").strip().lower()
        wayland_display = env.get("WAYLAND_DISPLAY", "").strip()
        if session_type == "wayland" or (wayland_display and session_type != "x11"):
            return HotkeyBackendSupport(
                available=False,
                backend="wayland",
                selective_suppression=False,
                detail=(
                    "Wayland does not expose a compositor-independent global "
                    "hotkey API to pynput; XWayland capture is not system-wide"
                ),
            )
        if not env.get("DISPLAY", "").strip():
            return HotkeyBackendSupport(
                available=False,
                backend="headless",
                selective_suppression=False,
                detail="X11 hotkeys require DISPLAY in a logged-in desktop session",
            )
        return HotkeyBackendSupport(
            available=True,
            backend="x11",
            selective_suppression=False,
            detail="X11 RECORD/XTest through pynput",
        )

    if name == "Darwin":
        return HotkeyBackendSupport(
            available=True,
            backend="darwin",
            selective_suppression=False,
            detail="macOS pynput event listener",
        )

    return HotkeyBackendSupport(
        available=False,
        backend="unsupported",
        selective_suppression=False,
        detail=f"unsupported desktop platform: {name}",
    )


def pynput_key_name(key: object) -> str:
    """Normalize a pynput Key/KeyCode without importing a platform backend."""

    name = getattr(key, "name", None)
    if name:
        return str(name).casefold()
    char = getattr(key, "char", None)
    if char:
        return str(char).casefold()
    vk = getattr(key, "vk", None)
    if vk is None:
        vk = getattr(getattr(key, "value", None), "vk", None)
    return f"vk_{vk}" if vk is not None else ""


def pynput_button_name(button: object) -> str:
    """Map pynput side-button names across Win32 and X11 backends."""

    name = str(getattr(button, "name", "")).casefold()
    return {
        "x1": "x1",
        "x2": "x2",
        "button8": "x1",
        "button9": "x2",
    }.get(name, name)
