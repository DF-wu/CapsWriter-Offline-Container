# coding: utf-8
"""Dependency-light desktop hotkey capability and event normalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import platform

from fork_client.shortcut_keys import event_key_name


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
        return event_key_name(str(name))
    vk = getattr(key, "vk", None)
    # X11 KeyCodes expose keysyms. Recognize keypad keys before char so the
    # main digit row and numpad retain distinct physical identities.
    if platform.system() == 'Linux' and isinstance(vk, int):
        keypad_navigation = {
            0xFF9E: 'numpad0', 0xFF9C: 'numpad1', 0xFF99: 'numpad2',
            0xFF9B: 'numpad3', 0xFF96: 'numpad4', 0xFF9D: 'numpad5',
            0xFF98: 'numpad6', 0xFF95: 'numpad7', 0xFF97: 'numpad8',
            0xFF9A: 'numpad9', 0xFF9F: 'numpad_decimal',
        }
        if vk in keypad_navigation:
            return keypad_navigation[vk]
        if 0xFFB0 <= vk <= 0xFFB9:
            return f'numpad{vk - 0xFFB0}'
        keypad = ('numpad_multiply', 'numpad_add', 'numpad_separator',
                  'numpad_subtract', 'numpad_decimal', 'numpad_divide')
        if 0xFFAA <= vk <= 0xFFAF:
            return keypad[vk - 0xFFAA]
        if 0xFFBE <= vk <= 0xFFD5:
            return f'f{vk - 0xFFBE + 1}'
    char = getattr(key, "char", None)
    if char:
        return event_key_name(str(char))
    vk = getattr(key, "vk", None)
    if vk is None:
        vk = getattr(getattr(key, "value", None), "vk", None)
    return f"vk_{vk}" if vk is not None else ""


def pynput_key_identity(key: object, name: str) -> tuple:
    """Keep callback press/release identities stable across modifier changes.

    Win32/macOS virtual codes are physical. X11 uses keysyms instead, so the
    printable fallback folds shifted ASCII symbols to the same key, just as
    letter case is folded. Keypad names remain separate from the main key row.
    """
    vk = getattr(key, 'vk', None)
    if vk is None:
        vk = getattr(getattr(key, 'value', None), 'vk', None)
    if platform.system() == 'Linux' and (len(name) == 1 or name == 'plus'):
        shifted = dict(zip('~!@#$%^&*()_+{}|:"<>?', "`1234567890-=[]\\;',./"))
        physical = '=' if name == 'plus' else shifted.get(name, name)
        return ('character', physical)
    return ('vk', vk) if vk is not None else ('name', name)


def pynput_button_name(button: object) -> str:
    """Map pynput side-button names across Win32 and X11 backends."""

    name = str(getattr(button, "name", "")).casefold()
    return {
        "x1": "x1",
        "x2": "x2",
        "button8": "x1",
        "button9": "x2",
    }.get(name, name)


def create_x11_keyboard_listener(keyboard, on_press, on_release):
    """Adapt pynput's raw X11 messages to physical callback identities.

    pynput 1.8.2 has no xorg_event_filter hook. Keep its one private method seam
    here; use public Xlib layout lookup and no private listener state. The base
    listener still translates/delivers events and handles injected callbacks.
    Context is thread-local because Controller callbacks can run on another
    thread while the RECORD thread is dispatching a physical event.
    """
    from threading import local

    context = local()
    special = {key.value.vk: key for key in keyboard.Key}

    def wrap(callback):
        def dispatch(key, injected=False):
            return callback(key, injected, **getattr(context, 'event', {}))
        return dispatch

    class PhysicalKeyboardListener(keyboard.Listener):
        def _handle_message(self, display, event, injected):
            previous = getattr(context, 'event', {})
            try:
                # XKB group bits select the active layout; index zero of that
                # group is unshifted. Always retain event.detail as identity,
                # including when Shift/CapsLock/layout changes before keyup.
                group = (event.state >> 13) & 3
                keysym = (display.keycode_to_keysym(event.detail, group * 2)
                          or display.keycode_to_keysym(event.detail, 0))
                base_key = special.get(keysym)
                if base_key is None:
                    if 0x20 <= keysym <= 0xFF:
                        base_key = keyboard.KeyCode.from_char(chr(keysym), vk=keysym)
                    elif 0x01000100 <= keysym <= 0x0110FFFF:
                        base_key = keyboard.KeyCode.from_char(chr(keysym & 0xFFFFFF), vk=keysym)
                    else:
                        base_key = keyboard.KeyCode.from_vk(keysym)
                base_name = pynput_key_name(base_key)
                context.event = {
                    'physical_identity': ('x11', event.detail),
                    'base_names': frozenset({base_name, f'vk_{keysym}'}) if keysym else frozenset(),
                }
                return super()._handle_message(display, event, injected)
            finally:
                context.event = previous

    return PhysicalKeyboardListener(
        on_press=wrap(on_press), on_release=wrap(on_release), suppress=False,
    )
