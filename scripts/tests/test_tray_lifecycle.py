"""托盤的 detached 啟動、退出與重啟不可留下不可操作的程序。"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
TRAY_PATH = ROOT / "core/ui/tray.py"


class FakeIcon:
    def __init__(self, *args, **kwargs):
        self.ready = threading.Event()
        self.stopped = threading.Event()
        self.visible = False

    def run(self, setup):
        setup(self)
        self.ready.set()
        self.stopped.wait(5)

    def stop(self):
        self.stopped.set()


def load_tray():
    tree = ast.parse(TRAY_PATH.read_text(encoding="utf-8"))
    names = {"_TraySystem", "enable_min_to_tray", "stop_tray"}
    module = ast.fix_missing_locations(ast.Module(
        body=[node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names],
        type_ignores=[],
    ))
    namespace = {
        "Optional": Optional, "threading": threading, "time": time,
        "sys": sys, "os": SimpleNamespace(path=SimpleNamespace(basename=lambda value: value)),
        "logger": Mock(), "user32": None, "kernel32": Mock(),
        "_get_console_hwnd": lambda: None, "_enable_close_button": Mock(),
        "_create_icon": lambda _path: object(), "_get_exit_callback": Mock(return_value=None),
        "_set_exit_callback": Mock(), "_check_tray_available": lambda: True,
        "_launch_detached_process": Mock(), "_tray_instance": None, "_lock": threading.Lock(),
        "SW_HIDE": 0, "SW_RESTORE": 9,
    }
    exec(compile(module, str(TRAY_PATH), "exec"), namespace)
    return namespace


class TrayLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.namespace = load_tray()
        pystray = SimpleNamespace(Icon=FakeIcon, MenuItem=lambda *args, **kwargs: object())
        modules = patch.dict(sys.modules, {"pystray": pystray})
        modules.start()
        self.addCleanup(modules.stop)

    def test_detached_process_without_console_still_has_tray_and_exit(self):
        with patch.dict(sys.modules, {"ctypes": SimpleNamespace()}):
            self.namespace["enable_min_to_tray"]("CapsWriter Client")
        tray = self.namespace["_tray_instance"]
        self.assertIsNotNone(tray)
        self.addCleanup(tray.stop)
        self.assertTrue(tray.icon.ready.wait(1))
        self.assertTrue(tray.icon.visible)
        self.assertTrue(tray._tray_thread.daemon)
        self.namespace["stop_tray"]()
        self.assertTrue(tray.should_exit)
        self.assertFalse(tray._tray_thread.is_alive())
        self.assertFalse(tray._monitor_thread.is_alive())
        self.assertIsNone(self.namespace["_tray_instance"])

    def test_console_close_button_is_available_and_console_has_product_title(self):
        self.namespace["_get_console_hwnd"] = lambda: 123
        self.namespace["_TraySystem"]("CapsWriter Client")
        self.namespace["_enable_close_button"].assert_called_once_with(123)
        self.namespace["kernel32"].SetConsoleTitleW.assert_called_once_with("CapsWriter Client")

    def test_restart_uses_absolute_executable_and_one_wait_flag_then_full_exit(self):
        for frozen in (False, True):
            with self.subTest(frozen=frozen), patch.object(sys, "frozen", frozen, create=True), patch.object(
                sys, "argv", ["relative-client.exe", "--wait-for-client-exit"]
            ):
                tray = self.namespace["_TraySystem"]("CapsWriter Client", restart_args=["--wait-for-client-exit"])
                tray.on_exit = Mock()
                launch = self.namespace["_launch_detached_process"]
                launch.reset_mock()
                tray.on_restart(None, None)
                prefix = [sys.executable] if frozen else [sys.executable, "relative-client.exe"]
                launch.assert_called_once_with([*prefix, "--wait-for-client-exit"])
                tray.on_exit.assert_called_once_with(None, None)

    def test_failed_restart_keeps_current_client_alive(self):
        tray = self.namespace["_TraySystem"]("CapsWriter Client", restart_args=["--wait-for-client-exit"])
        tray.on_exit = Mock()
        self.namespace["_launch_detached_process"].side_effect = OSError("launch failed")
        tray.on_restart(None, None)
        tray.on_exit.assert_not_called()

    def test_tray_callback_can_stop_its_own_thread_without_joining_itself(self):
        callback = Mock()
        self.namespace["_get_exit_callback"].return_value = callback
        tray = self.namespace["_TraySystem"]("CapsWriter Client")
        tray._tray_thread = threading.current_thread()
        tray.on_exit(None, None)
        callback.assert_called_once_with()
        self.assertTrue(tray.should_exit)
        self.assertTrue(tray.icon.stopped.is_set())


if __name__ == "__main__":
    unittest.main()
