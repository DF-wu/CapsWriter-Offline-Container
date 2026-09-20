# coding: utf-8
"""
全局快捷键管理器

使用 pynput GlobalHotKeys 实现全局快捷键监听，替代 keyboard 库。

使用示例:
    from core.client.global_hotkey import GlobalHotkeyManager

    manager = GlobalHotkeyManager()
    manager.register('<esc>', lambda: print('ESC pressed'))
    manager.start()
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, Optional

from . import logger
from core.client.shortcut.platform_support import detect_hotkey_backend



class GlobalHotkeyManager:
    """
    全局快捷键管理器

    使用 pynput GlobalHotKeys 实现，支持动态注册/注销快捷键。
    
    对比 keyboard 库的优势：
    - 与 pynput 的其他功能兼容
    - 不需要额外的依赖
    - Windows 与 X11 使用同一注册格式（Wayland 不提供全域监听）
    """

    # 单例实例
    _instance: Optional[GlobalHotkeyManager] = None
    _lock = threading.Lock()

    def __new__(cls) -> GlobalHotkeyManager:
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._hotkeys: Dict[str, Callable] = {}
        self._listener: Optional[Any] = None
        self._running = False
        self._initialized = True
        logger.debug("GlobalHotkeyManager 初始化完成")

    def register(self, key_str: str, callback: Callable) -> None:
        """
        注册全局快捷键

        Args:
            key_str: 快捷键字符串，pynput 格式，如 '<esc>', '<ctrl>+<alt>+h'
            callback: 按下快捷键时的回调函数
        """
        self._hotkeys[key_str] = callback
        logger.debug(f"注册全局快捷键: {key_str}")
        
        # 如果已经在运行，重启监听器以应用新的快捷键
        if self._running:
            self._restart_listener()

    def unregister(self, key_str: str) -> bool:
        """
        注销全局快捷键

        Args:
            key_str: 快捷键字符串

        Returns:
            是否成功注销
        """
        if key_str in self._hotkeys:
            del self._hotkeys[key_str]
            logger.debug(f"注销全局快捷键: {key_str}")
            
            if self._running:
                self._restart_listener()
            return True
        return False

    def start(self) -> None:
        """启动快捷键监听"""
        if self._running:
            logger.debug("GlobalHotkeyManager 已在运行")
            return
        
        if not self._hotkeys:
            logger.warning("没有注册的快捷键，跳过启动")
            return

        support = detect_hotkey_backend()
        if not support.available:
            logger.error(f"全局快捷键不可用 ({support.backend}): {support.detail}")
            return
        
        self._running = True
        if self._start_listener():
            logger.info(f"GlobalHotkeyManager 已启动，注册了 {len(self._hotkeys)} 个快捷键")
        else:
            self._running = False

    def stop(self) -> None:
        """停止快捷键监听"""
        self._running = False
        if self._listener:
            try:
                self._listener.stop()
            except Exception as e:
                logger.warning(f"停止 GlobalHotKeys 监听器时出错: {e}")
            self._listener = None
        logger.info("GlobalHotkeyManager 已停止")

    def _start_listener(self) -> bool:
        """启动监听器"""
        if not self._hotkeys:
            return False
        
        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys(self._hotkeys)
            self._listener.start()
            logger.debug(f"GlobalHotKeys 监听器已启动: {list(self._hotkeys.keys())}")
            return True
        except Exception as e:
            logger.error(f"启动 GlobalHotKeys 监听器失败: {e}")
            self._listener = None
            return False

    def _restart_listener(self) -> None:
        """重启监听器（用于更新快捷键后）"""
        if self._listener:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        
        if self._running and self._hotkeys:
            if not self._start_listener():
                self._running = False


# 全局单例实例
_global_hotkey_manager: Optional[GlobalHotkeyManager] = None


def get_global_hotkey_manager() -> GlobalHotkeyManager:
    """获取全局快捷键管理器单例"""
    global _global_hotkey_manager
    if _global_hotkey_manager is None:
        _global_hotkey_manager = GlobalHotkeyManager()
    return _global_hotkey_manager
