# coding: utf-8
"""
全局快捷键子模块

使用 pynput GlobalHotKeys 实现 Windows/X11 全局快捷键监听。
Wayland 不提供可靠的 system-wide capture，启动时会明确拒绝。
"""

from .. import logger
from core.client.global_hotkey.global_hotkey import (
    GlobalHotkeyManager,
    get_global_hotkey_manager,
)

__all__ = [
    'logger',
    'GlobalHotkeyManager',
    'get_global_hotkey_manager',
]
