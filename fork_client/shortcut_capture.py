"""Focused-window shortcut capture without global hooks or desktop imports."""
from __future__ import annotations

import ctypes
import ctypes.util

from .shortcut_keys import normalize_shortcut


TK_KEYS = {
    "Control_L": "ctrl_l", "Control_R": "ctrl_r",
    "Shift_L": "shift_l", "Shift_R": "shift_r",
    "Alt_L": "alt_l", "Alt_R": "alt_r", "ISO_Level3_Shift": "alt_gr",
    "Meta_L": "cmd_l", "Meta_R": "cmd_r", "Super_L": "cmd_l", "Super_R": "cmd_r",
    "Win_L": "cmd_l", "Win_R": "cmd_r",
    "Return": "enter", "KP_Enter": "enter", "Escape": "esc",
    "BackSpace": "backspace", "Tab": "tab", "ISO_Left_Tab": "tab",
    "space": "space", "Caps_Lock": "caps_lock", "Num_Lock": "num_lock",
    "Scroll_Lock": "scroll_lock", "Print": "print_screen", "Pause": "pause",
    "Prior": "page_up", "Next": "page_down", "Menu": "menu",
    "KP_Add": "numpad_add", "KP_Subtract": "numpad_subtract",
    "KP_Multiply": "numpad_multiply", "KP_Divide": "numpad_divide",
    "KP_Decimal": "numpad_decimal", "KP_Separator": "numpad_separator",
    "KP_Insert": "numpad0", "KP_End": "numpad1", "KP_Down": "numpad2",
    "KP_Next": "numpad3", "KP_Left": "numpad4", "KP_Begin": "numpad5",
    "KP_Right": "numpad6", "KP_Home": "numpad7", "KP_Up": "numpad8",
    "KP_Prior": "numpad9", "KP_Delete": "numpad_decimal",
    **{f"KP_{i}": f"numpad{i}" for i in range(10)},
}

TK_SYMBOLS = {
    "grave": "`", "asciitilde": "~", "plus": "+", "equal": "=",
    "minus": "-", "underscore": "_", "bracketleft": "[", "braceleft": "{",
    "bracketright": "]", "braceright": "}", "backslash": "\\", "bar": "|",
    "semicolon": ";", "colon": ":", "apostrophe": "'", "quotedbl": '"',
    "comma": ",", "less": "<", "period": ".", "greater": ">",
    "slash": "/", "question": "?",
    **dict(zip(("exclam", "at", "numbersign", "dollar", "percent", "asciicircum", "ampersand", "asterisk", "parenleft", "parenright"), "!@#$%^&*()")),
}


def unshifted_character(event) -> str | None:
    """Resolve the focused Tk window's layout without starting input hooks.

    Tk's keycode is a Win32 virtual key or an X11 hardware code. Translate it
    through that window system, rather than assuming US punctuation positions.
    """
    widget = getattr(event, "widget", None)
    if widget is None:
        return None
    backend = widget.tk.call("tk", "windowingsystem")
    if backend == "win32":
        user32 = ctypes.windll.user32
        user32.MapVirtualKeyW.argtypes = (ctypes.c_uint, ctypes.c_uint)
        user32.MapVirtualKeyW.restype = ctypes.c_uint
        codepoint = user32.MapVirtualKeyW(event.keycode, 2) & 0x7FFFFFFF
        return chr(codepoint) if 0 < codepoint <= 0x10FFFF else None
    if backend != "x11":
        return None
    library = ctypes.util.find_library("X11")
    if not library:
        return None
    x11 = ctypes.CDLL(library)
    x11.XOpenDisplay.argtypes = (ctypes.c_char_p,)
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = (ctypes.c_void_p,)
    x11.XCloseDisplay.restype = ctypes.c_int
    x11.XkbKeycodeToKeysym.argtypes = (ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_int, ctypes.c_int)
    x11.XkbKeycodeToKeysym.restype = ctypes.c_ulong
    connection = x11.XOpenDisplay(widget.winfo_screen().encode())
    if not connection:
        return None
    try:
        group = (event.state >> 13) & 3
        keysym = x11.XkbKeycodeToKeysym(connection, event.keycode, group, 0)
        if 0x20 <= keysym <= 0xFF:
            return chr(keysym)
        if 0x01000100 <= keysym <= 0x0110FFFF:
            return chr(keysym & 0xFFFFFF)
        return None
    finally:
        x11.XCloseDisplay(connection)


def tk_key_name(event) -> str:
    """Use symbolic names, never treat X11 hardware keycodes as virtual codes."""
    keysym = event.keysym
    if keysym in TK_KEYS:
        return normalize_shortcut(TK_KEYS[keysym])
    try:
        base = unshifted_character(event)
    except (AttributeError, OSError):
        base = None
    if base and base.isprintable() and not base.isspace():
        return normalize_shortcut("plus" if base == "+" else base)
    name = TK_SYMBOLS.get(keysym, keysym.casefold())
    try:
        return normalize_shortcut(name)
    except ValueError:
        # Non-Latin layouts may supply a printable Unicode character but no
        # recognized keysym. Control characters and unknown keycodes stay invalid.
        char = getattr(event, "char", "")
        if len(char) == 1 and char.isprintable() and not char.isspace():
            return normalize_shortcut("plus" if char == "+" else char)
        raise ValueError("無法辨識此按鍵；請改用手動輸入按鍵名稱。") from None


class ShortcutCapture:
    """Capture one simultaneous chord; auto-repeat never adds another key."""

    def __init__(self):
        self.held: dict[int, str] = {}
        self.chord: str | None = None

    def press(self, event) -> str:
        if event.keycode not in self.held:
            name = tk_key_name(event)
            self.held[event.keycode] = name
        # After the first release the chord is frozen until all keys are up.
        return self.chord or normalize_shortcut("+".join(dict.fromkeys(self.held.values())))

    def release(self, event) -> str | None:
        if event.keycode not in self.held:
            return None
        if self.chord is None:
            self.chord = normalize_shortcut("+".join(dict.fromkeys(self.held.values())))
        del self.held[event.keycode]
        return self.chord if not self.held else None
