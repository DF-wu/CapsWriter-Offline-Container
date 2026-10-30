# coding: utf-8
"""
快捷键管理器（重构版）

统一管理多个快捷键，处理键盘和鼠标事件，支持：
1. 多快捷键并发处理
2. 防止不同按键互相干扰
3. restore 功能的防自捕获逻辑
4. hold_mode 和 click_mode 支持
"""
from __future__ import annotations
import time
from copy import copy
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fork_client.shortcut_keys import parse_shortcut, key_matches, event_key_name

from . import logger
from core.client.shortcut.key_mapper import *
from core.client.shortcut.key_mapper import KeyMapper
from core.client.shortcut.emulator import ShortcutEmulator
from core.client.shortcut.event_handler import ShortcutEventHandler
from core.client.shortcut.platform_support import (
    HotkeyBackendSupport,
    detect_hotkey_backend,
    pynput_button_name,
    pynput_key_name,
    pynput_key_identity,
    create_x11_keyboard_listener,
)
from core.client.shortcut.task import ShortcutTask

if TYPE_CHECKING:
    from core.client.shortcut.shortcut_config import Shortcut
    from core.client.state import ClientState
    from core.client.app import CapsWriterClient



class ShortcutManager:
    """
    快捷键管理器

    统一管理多个快捷键，使用 pynput 监听键盘和鼠标事件。
    Windows 保留低阶 win32 filter；X11 使用标准 press/release callback。
    """

    def __init__(self, app: CapsWriterClient, shortcuts: List[Shortcut]):
        """
        初始化快捷键管理器

        Args:
            app: 客户端 App 实例
            shortcuts: 快捷键配置列表
        """
        self.app = app
        self.shortcuts = shortcuts

        # 监听器
        self.keyboard_listener: Optional[Any] = None
        self.mouse_listener: Optional[Any] = None

        # 快捷键任务映射（key -> ShortcutTask）
        self.tasks: Dict[str, ShortcutTask] = {}

        # 线程池
        self._pool = ThreadPoolExecutor(max_workers=4)

        self._hotkey_support = detect_hotkey_backend()

        # Unsupported sessions (notably Wayland/headless Linux) must not try to
        # construct pynput controllers, which would fail before a useful
        # diagnostic can be logged.
        if not self._hotkey_support.available:
            self._emulator = None
        elif self._hotkey_support.backend == 'win32':
            # Preserve the upstream Windows initialization behavior.
            self._emulator = ShortcutEmulator()
        else:
            try:
                self._emulator = ShortcutEmulator()
            except Exception as error:
                self._emulator = None
                self._hotkey_support = HotkeyBackendSupport(
                    available=False,
                    backend=self._hotkey_support.backend,
                    selective_suppression=False,
                    detail=f"{self._hotkey_support.detail}; initialization failed: {error}",
                )
                logger.error(self._hotkey_support.detail)

        # 按键恢复状态追踪
        self._restoring_keys = set()

        # 事件处理器
        self._event_handler = ShortcutEventHandler(self.tasks, self._pool, self._emulator)

        # 初始化快捷键任务
        self._init_tasks()

    @property
    def state(self) -> ClientState:
        """快捷访问状态单例"""
        return self.app.state

    def _init_tasks(self) -> None:
        """初始化所有快捷键任务"""
        from config_client import ClientConfig as Config

        for shortcut in self.shortcuts:
            if not shortcut.enabled:
                continue

            task_shortcut = shortcut
            is_chord = len(parse_shortcut(shortcut.key)) > 1
            if shortcut.suppress and is_chord:
                # Earlier keys already reached the application; replay cannot
                # restore their original context safely. Never synthesize a chord.
                logger.warning(
                    f"[{shortcut.key}] chord suppression is disabled; other "
                    "applications receive all keys (no short-tap replay)"
                )
            if shortcut.suppress and (is_chord or not self._hotkey_support.selective_suppression):
                # X11's pynput backend can only grab the whole device. Treat
                # per-key suppression as disabled instead of freezing all
                # desktop input or replaying an event that was never blocked.
                task_shortcut = copy(shortcut)
                task_shortcut.suppress = False

            task = ShortcutTask(self.app, task_shortcut)
            task._manager_ref = lambda: self  # 弱引用，用于回调
            task.pool = self._pool
            task.threshold = shortcut.get_threshold(Config.threshold)
            self.tasks[shortcut.key] = task

    # ========== 监听器创建 ==========

    def create_keyboard_filter(self):
        """创建键盘事件过滤器"""
        def win32_event_filter(msg, data):
            # 只处理 KEYDOWN 和 KEYUP 消息
            if msg not in KEYBOARD_MESSAGES:
                return True

            key_name = KeyMapper.vk_to_name(data.vkCode)

            # 防自捕获检查
            if self._check_emulating(key_name, msg, vk=data.vkCode):
                return True
            if self._check_restoring(key_name, msg):
                return True

            # Ignore externally injected input as well as our own replay.
            if getattr(data, 'flags', 0) & 0x10:  # LLKHF_INJECTED
                return True
            names = {event_key_name(key_name), f'vk_{data.vkCode}'}
            scan_code = getattr(data, 'scanCode', None)
            if scan_code is not None and (len(key_name) == 1 or key_name == 'plus'):
                # Recording uses the unshifted key name. Keep that alias when
                # Shift/AltGr changes the character reported by the Win32 hook.
                base_name = KeyMapper.scan_to_name(scan_code)
                if base_name:
                    names.add(base_name)
            suppress = self._dispatch_keyboard(
                key_name, msg in KEY_DOWN_MESSAGES, ('vk', data.vkCode),
                frozenset(names),
            )
            if suppress and self.keyboard_listener:
                self.keyboard_listener.suppress_event()

            return True

        return win32_event_filter

    def _dispatch_keyboard(self, key_name, pressed, identity=None, names=None):
        """Track physical keys; each binding needs a fresh, complete press cycle.

        Chords never suppress or replay input. A binding is latched on completion,
        ends on the first selected constituent release, and rearms only when all
        matching keys are up. Matching uses distinct physical keys, so one right
        Ctrl cannot satisfy both generic Ctrl and right Ctrl in the same chord.
        """
        if not hasattr(self, '_physical_keys'):
            self._physical_keys = {}
            self._active_bindings = {}
            self._latched_bindings = set()
        identity = identity if identity is not None else ('name', key_name)
        names = names if names is not None else frozenset({event_key_name(key_name)})
        if names.intersection({'=', 'plus'}):
            names = names | frozenset({'=', 'plus'})
        previous = self._physical_keys.get(identity, names)
        repeated = pressed and identity in self._physical_keys
        if pressed:
            if not repeated:
                self._physical_keys[identity] = names
        else:
            self._physical_keys.pop(identity, None)

        suppress = False
        for binding, task in self.tasks.items():
            if getattr(getattr(task, 'shortcut', None), 'type', 'keyboard') == 'mouse':
                continue
            tokens = parse_shortcut(binding)
            if tokens[0] in {'x1', 'x2'}:
                continue
            # Preserve selective Win32 singleton suppression, including repeats.
            if len(tokens) == 1 and key_matches(tokens[0], previous):
                suppress |= getattr(getattr(task, 'shortcut', None), 'suppress', False)
            active = self._active_bindings.get(binding)
            if not pressed and active is not None and identity in active:
                del self._active_bindings[binding]
                self._event_handler.handle_keyup(binding, task)
            if binding in self._latched_bindings:
                if not any(key_matches(token, held) for token in tokens
                           for held in self._physical_keys.values()):
                    self._latched_bindings.discard(binding)
                continue
            if not pressed or repeated:
                continue

            # Bipartite matching prevents aliased/generic modifiers or vk names
            # from counting the same physical key twice.
            owners = {}

            def assign(token, seen):
                for physical, held in self._physical_keys.items():
                    if physical in seen or not key_matches(token, held):
                        continue
                    seen.add(physical)
                    if physical not in owners or assign(owners[physical], seen):
                        owners[physical] = token
                        return True
                return False

            if all(assign(token, set()) for token in tokens):
                self._active_bindings[binding] = set(owners)
                self._latched_bindings.add(binding)
                self._event_handler.handle_keydown(binding, task)
        return suppress

    def create_mouse_filter(self):
        """创建鼠标事件过滤器"""
        def win32_event_filter(msg, data):
            # 只处理 XBUTTON 消息
            if msg not in MOUSE_MESSAGES:
                return True

            # 获取按键标识
            xbutton = (data.mouseData >> 16) & 0xFFFF
            button_name = 'x1' if xbutton == XBUTTON1 else 'x2'

            # 防自捕获检查
            if self._check_emulating(button_name, msg, is_mouse=True):
                return True

            if getattr(data, 'flags', 0) & 0x01:  # LLMHF_INJECTED
                return True

            # 查找匹配的快捷键
            if button_name not in self.tasks:
                return True

            task = self.tasks[button_name]

            # 处理鼠标事件
            if msg == WM_XBUTTONDOWN:
                self._event_handler.handle_keydown(button_name, task)
            elif msg == WM_XBUTTONUP:
                self._handle_mouse_keyup(button_name, task)

            # 阻塞事件
            if task.shortcut.suppress and self.mouse_listener:
                self.mouse_listener.suppress_event()

            return True

        return win32_event_filter

    def create_portable_keyboard_callbacks(self):
        """Create pynput callbacks for the X11/macOS listener APIs."""

        def handle(key, pressed: bool, injected: bool = False,
                   physical_identity=None, base_names=frozenset()):
            key_name = pynput_key_name(key)
            if not key_name:
                return
            vk = getattr(key, 'vk', None)
            if vk is None:
                vk = getattr(getattr(key, 'value', None), 'vk', None)
            message = KEY_DOWN_MESSAGES[0] if pressed else KEY_UP_MESSAGES[0]
            if self._check_emulating(key_name, message, vk=vk):
                return
            if injected:
                return
            if self.is_restoring(key_name):
                if not pressed:
                    self.clear_restoring_flag(key_name)
                return
            identity = physical_identity or pynput_key_identity(key, key_name)
            names = {key_name} | set(base_names)
            if identity[0] == 'character':
                names.add(identity[1])
            if vk is not None:
                names.add(f'vk_{vk}')
            self._dispatch_keyboard(key_name, pressed, identity, frozenset(names))

        return (
            lambda key, injected=False, **context: handle(key, True, injected, **context),
            lambda key, injected=False, **context: handle(key, False, injected, **context),
        )

    def create_portable_mouse_callback(self):
        """Create a pynput side-button callback for X11/macOS."""

        def on_click(x, y, button, pressed, injected=False):
            if injected:
                return
            button_name = pynput_button_name(button)
            if button_name not in {'x1', 'x2'}:
                return
            if self._emulator and self._emulator.is_emulating(button_name):
                if not pressed:
                    self._emulator.clear_emulating_flag(button_name)
                return
            task = self.tasks.get(button_name)
            if task is None:
                return
            if pressed:
                self._event_handler.handle_keydown(button_name, task)
            else:
                self._handle_mouse_keyup(button_name, task)

        return on_click

    def _handle_mouse_keyup(self, button_name: str, task) -> None:
        """处理鼠标按键释放事件"""
        # 单击模式
        if not task.shortcut.hold_mode:
            if task.pressed:
                task.pressed = False
                task.released = True
                task.event.set()
            return

        # 长按模式
        if not task.is_recording:
            return

        duration = time.time() - task.recording_start_time
        logger.debug(f"[{button_name}] 松开按键，持续时间: {duration:.3f}s")

        if duration < task.threshold:
            task.cancel()
            if task.shortcut.suppress:
                logger.debug(f"[{button_name}] 安排异步补发鼠标按键")
                self._pool.submit(self._emulator.emulate_mouse_click, button_name)
        else:
            task.finish()

    # ========== 按键恢复管理 ==========

    def schedule_restore(self, key: str) -> None:
        """
        安排按键恢复（延迟执行，避免在事件处理中阻塞）

        Args:
            key: 要恢复的按键

        注意：标志清除只在按键释放事件中处理（_check_restoring），
        避免在线程中提前清除导致主线程收到重复消息。
        """
        from pynput import keyboard

        self._restoring_keys.add(key)

        def do_restore():
            import time
            time.sleep(0.05)  # 延迟 50ms
            if key == 'caps_lock':
                controller = keyboard.Controller()
                controller.press(keyboard.Key.caps_lock)
                controller.release(keyboard.Key.caps_lock)

        self._pool.submit(do_restore)

    def is_restoring(self, key: str) -> bool:
        """检查是否正在恢复指定按键"""
        return key in self._restoring_keys

    def clear_restoring_flag(self, key: str) -> None:
        """清除恢复标志"""
        self._restoring_keys.discard(key)

    # ========== 防自捕获检查 ==========

    def _check_emulating(self, key_name: str, msg: int, is_mouse: bool = False, vk=None) -> bool:
        """检查是否正在模拟按键"""
        if not self._emulator:
            return False
        replay_names = {
            'shift_l': ('shift_l', 'shift'),
            'ctrl_l': ('ctrl_l', 'left_ctrl', 'ctrl'),
            'alt_l': ('alt_l', 'left_alt', 'alt'),
            '=': ('=', 'plus'),
            'plus': ('plus', '='),
            'alt_r': ('alt_r', 'alt_gr'),
            'alt_gr': ('alt_gr', 'alt_r'),
            'cmd_l': ('cmd_l', 'cmd'),
        }.get(key_name, (key_name,))
        if vk is not None:
            replay_names += (f'vk_{vk}',)
        flagged = [name for name in replay_names if self._emulator.is_emulating(name)]
        if not flagged:
            return False

        # 松开时清除标志
        if is_mouse:
            if msg == WM_XBUTTONUP:
                for name in flagged:
                    self._emulator.clear_emulating_flag(name)
        else:
            if msg in KEY_UP_MESSAGES:
                for name in flagged:
                    self._emulator.clear_emulating_flag(name)

        return True  # 放行

    def _check_restoring(self, key_name: str, msg: int) -> bool:
        """检查是否正在恢复按键"""
        if not self.is_restoring(key_name):
            return False

        if msg in (WM_KEYUP, WM_SYSKEYUP):
            self.clear_restoring_flag(key_name)

        return True  # 放行

    # ========== 公共接口 ==========

    def start(self) -> None:
        """启动所有监听器"""
        if not self._hotkey_support.available:
            logger.error(
                f"全局快捷键不可用 ({self._hotkey_support.backend}): "
                f"{self._hotkey_support.detail}"
            )
            return

        from pynput import keyboard, mouse

        has_keyboard = any(s.type == 'keyboard' for s in self.shortcuts if s.enabled)
        has_mouse = any(s.type == 'mouse' for s in self.shortcuts if s.enabled)

        if (
            not self._hotkey_support.selective_suppression
            and any(s.suppress for s in self.shortcuts if s.enabled)
        ):
            logger.warning(
                f"{self._hotkey_support.backend} 不支持安全的单键阻塞；"
                "已按 suppress=False 监听，其他应用仍会收到按键"
            )

        if has_keyboard:
            if self.keyboard_listener and self.keyboard_listener.is_alive():
                logger.debug("键盘监听器已在运行，跳过启动")
            else:
                if self._hotkey_support.backend == 'win32':
                    self.keyboard_listener = keyboard.Listener(
                        win32_event_filter=self.create_keyboard_filter()
                    )
                else:
                    on_press, on_release = self.create_portable_keyboard_callbacks()
                    if self._hotkey_support.backend == 'x11':
                        self.keyboard_listener = create_x11_keyboard_listener(
                            keyboard, on_press, on_release,
                        )
                    else:
                        self.keyboard_listener = keyboard.Listener(
                            on_press=on_press,
                            on_release=on_release,
                            suppress=False,
                        )
                self.keyboard_listener.start()
                logger.info("键盘监听器已启动")

        if has_mouse:
            if self.mouse_listener and self.mouse_listener.is_alive():
                logger.debug("鼠标监听器已在运行，跳过启动")
            else:
                if self._hotkey_support.backend == 'win32':
                    self.mouse_listener = mouse.Listener(
                        win32_event_filter=self.create_mouse_filter()
                    )
                else:
                    self.mouse_listener = mouse.Listener(
                        on_click=self.create_portable_mouse_callback(),
                        suppress=False,
                    )
                self.mouse_listener.start()
                logger.info("鼠标监听器已启动")

        # 打印所有启用的快捷键
        for shortcut in self.shortcuts:
            if shortcut.enabled:
                mode = "长按" if shortcut.hold_mode else "单击"
                toggle = "可恢复" if shortcut.is_toggle_key() else "普通键"
                task = self.tasks.get(shortcut.key)
                effective_suppress = (
                    task.shortcut.suppress if task is not None else shortcut.suppress
                )
                logger.info(
                    f"  [{shortcut.key}] {mode}模式, "
                    f"阻塞:{effective_suppress}, {toggle}"
                )

    def stop(self) -> None:
        """停止所有监听器和清理资源"""
        if self.keyboard_listener:
            try:
                self.keyboard_listener.stop()
                logger.debug("键盘监听器已停止")
            except Exception:
                pass
            finally:
                self.keyboard_listener = None
                
        if self.mouse_listener:
            try:
                self.mouse_listener.stop()
                logger.debug("鼠标监听器已停止")
            except Exception:
                pass
            finally:
                self.mouse_listener = None

        # 取消所有任务
        for task in self.tasks.values():
            if task.is_recording:
                task.cancel()

        if hasattr(self, '_physical_keys'):
            self._physical_keys.clear()
            self._active_bindings.clear()
            self._latched_bindings.clear()

        # 关闭线程池
        self._pool.shutdown(wait=False)
        logger.debug("快捷键管理器线程池已关闭")
