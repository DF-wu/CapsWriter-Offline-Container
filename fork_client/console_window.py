"""開機自動啟動時盡早隱藏主控台，避免登入後閃現黑色視窗。"""
from __future__ import annotations

import sys


SW_HIDE = 0
SW_SHOW = 5
GA_ROOT = 3


def _console_window():
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        # HWND 是指標；未設定 restype 會在 64 位元 Windows 截斷。
        kernel32.GetConsoleWindow.restype = wintypes.HWND
        user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
        user32.GetAncestor.restype = wintypes.HWND
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            # Windows Terminal 需操作最外層視窗，與托盤模組一致。
            hwnd = user32.GetAncestor(hwnd, GA_ROOT) or hwnd
        return user32, hwnd
    except (AttributeError, OSError):
        return None, None


def hide_console() -> bool:
    user32, hwnd = _console_window()
    if not hwnd:
        return False
    user32.ShowWindow(hwnd, SW_HIDE)
    return True


def show_console() -> bool:
    user32, hwnd = _console_window()
    if not hwnd:
        return False
    user32.ShowWindow(hwnd, SW_SHOW)
    return True
