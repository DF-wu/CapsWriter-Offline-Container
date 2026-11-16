"""開機自動啟動、托盤設定重啟與設定視窗「啟動與系統」頁；不修改真實的 Run 登錄值。"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import start_client
from fork_client import autostart as autostart_module
from fork_client.autostart import (
    AUTOSTART_ARGUMENT, BLOCKED, DISABLED, ENABLED, STALE, UNSUPPORTED, Autostart,
)
from fork_client.instance import (
    ClientInstance, SETTINGS_FROM_CLIENT_ARGUMENT, SETTINGS_RESTART_EXIT_CODE,
)


ROOT = Path(__file__).resolve().parents[2]
COMMAND = [r"C:\Program Files\CapsWriter\start_client.exe", AUTOSTART_ARGUMENT]


class FakeRegistry:
    def __init__(self):
        self.values = {}

    def get(self, key, name):
        return self.values.get((key, name))

    def set(self, key, name, value):
        self.values[(key, name)] = value

    def delete(self, key, name):
        self.values.pop((key, name), None)


class AutostartTest(unittest.TestCase):
    def setUp(self):
        self.registry = FakeRegistry()
        self.autostart = Autostart(self.registry, command=COMMAND)

    def test_enable_writes_quoted_command_and_disable_removes_it(self):
        self.assertEqual(self.autostart.status().state, DISABLED)
        status = self.autostart.set_enabled(True)
        self.assertTrue(status.enabled)
        self.assertEqual(
            self.registry.get(autostart_module.RUN_KEY, autostart_module.VALUE_NAME),
            '"C:\\Program Files\\CapsWriter\\start_client.exe" --autostart',
        )
        self.assertFalse(self.autostart.set_enabled(False).enabled)
        self.assertEqual(self.registry.values, {})

    def test_task_manager_disabled_entry_is_reported_and_cleared_on_enable(self):
        self.autostart.set_enabled(True)
        self.registry.set(autostart_module.APPROVED_KEY, autostart_module.VALUE_NAME, b"\x03" + b"\0" * 11)
        status = self.autostart.status()
        self.assertEqual(status.state, BLOCKED)
        self.assertFalse(status.enabled)
        self.assertIn("工作管理員", status.describe())
        self.assertEqual(self.autostart.set_enabled(True).state, ENABLED)
        self.assertIsNone(self.registry.get(autostart_module.APPROVED_KEY, autostart_module.VALUE_NAME))
        # 0x02 表示工作管理員中為啟用。
        self.registry.set(autostart_module.APPROVED_KEY, autostart_module.VALUE_NAME, b"\x02" + b"\0" * 11)
        self.assertEqual(self.autostart.status().state, ENABLED)

    def test_moved_installation_is_stale_until_reenabled(self):
        self.registry.set(autostart_module.RUN_KEY, autostart_module.VALUE_NAME, r'"D:\Old\start_client.exe" --autostart')
        status = self.autostart.status()
        self.assertEqual(status.state, STALE)
        self.assertIn(r"D:\Old", status.describe())
        self.assertEqual(self.autostart.set_enabled(True).state, ENABLED)

    def test_path_comparison_ignores_windows_case_differences(self):
        self.registry.set(autostart_module.RUN_KEY, autostart_module.VALUE_NAME, self.autostart.expected().upper())
        self.assertEqual(self.autostart.status().state, ENABLED)

    def test_non_windows_reports_unsupported_and_refuses_changes(self):
        with patch.object(autostart_module.sys, "platform", "linux"):
            autostart = Autostart(command=COMMAND)
            self.assertEqual(autostart.status().state, UNSUPPORTED)
            with self.assertRaises(OSError):
                autostart.set_enabled(True)

    def test_launch_command_for_frozen_and_source_runs(self):
        with patch.object(autostart_module.sys, "frozen", True, create=True), \
                patch.object(autostart_module.sys, "executable", r"C:\Apps\CapsWriter\start_client.exe"):
            command = autostart_module.launch_command()
        self.assertEqual(command[1:], [AUTOSTART_ARGUMENT])
        self.assertTrue(command[0].endswith("start_client.exe"))
        with patch.object(autostart_module.sys, "frozen", False, create=True):
            command = autostart_module.launch_command()
        self.assertEqual(Path(command[1]), ROOT / "start_client.py")
        self.assertEqual(command[2], AUTOSTART_ARGUMENT)


@unittest.skipUnless(sys.platform == "win32", "需要 Windows 登錄檔")
class WindowsRegistryRoundTripTest(unittest.TestCase):
    """使用一次性的 HKCU 測試 key，不碰真正的 Run 與 StartupApproved。"""

    def setUp(self):
        import winreg

        self.winreg = winreg
        self.base = rf"Software\CapsWriter-Tests-{uuid.uuid4().hex}"
        self.addCleanup(self.delete_tree, self.base)

    def delete_tree(self, key):
        try:
            with self.winreg.OpenKey(self.winreg.HKEY_CURRENT_USER, key, 0, self.winreg.KEY_ALL_ACCESS) as handle:
                while True:
                    try:
                        child = self.winreg.EnumKey(handle, 0)
                    except OSError:
                        break
                    self.delete_tree(rf"{key}\{child}")
            self.winreg.DeleteKey(self.winreg.HKEY_CURRENT_USER, key)
        except FileNotFoundError:
            pass

    def test_real_registry_round_trip(self):
        autostart = Autostart(run_key=rf"{self.base}\Run", approved_key=rf"{self.base}\Approved", command=COMMAND)
        self.assertEqual(autostart.status().state, DISABLED)
        self.assertTrue(autostart.set_enabled(True).enabled)
        with self.winreg.CreateKeyEx(self.winreg.HKEY_CURRENT_USER, rf"{self.base}\Approved") as handle:
            self.winreg.SetValueEx(handle, autostart.name, 0, self.winreg.REG_BINARY, b"\x03" + b"\0" * 11)
        self.assertEqual(autostart.status().state, BLOCKED)
        self.assertEqual(autostart.set_enabled(True).state, ENABLED)
        self.assertEqual(autostart.set_enabled(False).state, DISABLED)
        # 重複停用不應因值不存在而失敗。
        self.assertEqual(autostart.set_enabled(False).state, DISABLED)


class EntrypointTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.settings = Path(temporary.name) / "settings.json"
        environment = patch.dict(os.environ, {"CAPSWRITER_CLIENT_SETTINGS": str(self.settings)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_autostart_hides_console_then_runs_normal_microphone_client(self):
        with patch.object(start_client, "_hide_console_for_autostart") as hide, \
                patch.object(start_client, "_run_client", return_value=0) as run:
            self.assertEqual(start_client.main([AUTOSTART_ARGUMENT]), 0)
        hide.assert_called_once_with()
        run.assert_called_once_with([])

    def test_autostart_exits_quietly_when_client_already_running(self):
        owner = ClientInstance()
        self.assertTrue(owner.acquire())
        self.addCleanup(owner.release)
        with patch.object(start_client, "_hide_console_for_autostart"), \
                patch.object(start_client, "_run_client") as run, \
                patch("builtins.print"):
            self.assertEqual(start_client.main([AUTOSTART_ARGUMENT]), 0)
        run.assert_not_called()

    def test_console_stays_visible_when_tray_disabled(self):
        hide = Mock()
        console = SimpleNamespace(hide_console=hide)
        with patch("config_client.ClientConfig.enable_tray", False), \
                patch.dict(sys.modules, {"fork_client.console_window": console}):
            start_client._hide_console_for_autostart()
        hide.assert_not_called()
        with patch("config_client.ClientConfig.enable_tray", True), \
                patch.dict(sys.modules, {"fork_client.console_window": console}):
            start_client._hide_console_for_autostart()
        hide.assert_called_once_with()

    def test_settings_from_client_reports_restart_request_by_exit_code(self):
        for requested, expected in ((True, SETTINGS_RESTART_EXIT_CODE), (False, 0)):
            with self.subTest(requested=requested):
                ui = SimpleNamespace(open_settings=Mock(return_value=SimpleNamespace(restart_requested=requested)))
                with patch.dict(sys.modules, {"fork_client.settings_ui": ui, "core.client": None}):
                    self.assertEqual(start_client.main(["--settings", SETTINGS_FROM_CLIENT_ARGUMENT]), expected)
                ui.open_settings.assert_called_once_with(from_client=True)


def load_tray_manager(namespace):
    path = ROOT / "core" / "client" / "manager" / "tray_manager.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TrayManager"]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["TrayManager"]


class TraySettingsRestartTest(unittest.TestCase):
    def setUp(self):
        import threading

        self.launched = []
        self.request_restart = Mock(return_value=True)
        self.toast = Mock()
        # 以假模組隔離托盤、Toast 與 core 套件初始化（colorama／rich 等桌面依賴）。
        self.refresh_menu = Mock()
        tray_module = ModuleType("core.ui.tray")
        tray_module.request_restart = self.request_restart
        tray_module.refresh_menu = self.refresh_menu
        client_ui = ModuleType("core.client.ui")
        client_ui.toast = self.toast
        modules = patch.dict(sys.modules, {
            "core": ModuleType("core"), "core.ui": ModuleType("core.ui"), "core.ui.tray": tray_module,
            "core.client": ModuleType("core.client"), "core.client.ui": client_ui,
        })
        modules.start()
        self.addCleanup(modules.stop)
        self.namespace = {
            "__name__": "core.client.manager.tray_manager", "__package__": "core.client.manager",
            "sys": SimpleNamespace(frozen=True, executable=r"C:\CapsWriter\start_client.exe"),
            "threading": threading, "logger": Mock(),
            "SETTINGS_FROM_CLIENT_ARGUMENT": SETTINGS_FROM_CLIENT_ARGUMENT,
            "SETTINGS_RESTART_EXIT_CODE": SETTINGS_RESTART_EXIT_CODE,
            "_launch_detached_process": self.launch,
        }
        self.manager = load_tray_manager(self.namespace)(SimpleNamespace())

    def launch(self, command):
        process = Mock()
        process.command = command
        process.poll.return_value = None
        process.wait.return_value = 0
        self.launched.append(process)
        return process

    def test_settings_window_is_single_and_marked_as_opened_from_client(self):
        with patch.object(self.manager, "_wait_settings"):
            self.manager._open_settings()
            self.manager._open_settings()
        self.assertEqual(len(self.launched), 1)
        self.assertEqual(self.launched[0].command, [r"C:\CapsWriter\start_client.exe", "--settings", SETTINGS_FROM_CLIENT_ARGUMENT])
        self.launched[0].poll.return_value = 0
        with patch.object(self.manager, "_wait_settings"):
            self.manager._open_settings()
        self.assertEqual(len(self.launched), 2)

    def test_only_restart_exit_code_restarts_client(self):
        self.manager._wait_settings(SimpleNamespace(wait=lambda: 0))
        self.request_restart.assert_not_called()
        # 一般關閉仍要同步托盤勾選，因設定視窗可能改了開機自動啟動。
        self.refresh_menu.assert_called_once_with()
        self.manager._wait_settings(SimpleNamespace(wait=lambda: SETTINGS_RESTART_EXIT_CODE))
        self.request_restart.assert_called_once_with()

    def test_autostart_toggle_inverts_current_status(self):
        registry = FakeRegistry()
        self.namespace["Autostart"] = lambda: Autostart(registry, command=COMMAND)
        self.assertFalse(self.manager._autostart_checked())
        self.manager._toggle_autostart()
        self.assertTrue(self.manager._autostart_checked())
        self.manager._toggle_autostart()
        self.assertFalse(self.manager._autostart_checked())
        self.assertEqual(self.toast.call_count, 2)


class TrayRequestRestartTest(unittest.TestCase):
    def test_request_restart_uses_running_tray_restart_flow(self):
        try:
            from core.ui import tray
        except ImportError as exc:
            self.skipTest(f"desktop UI dependencies unavailable: {exc}")
        instance = SimpleNamespace(should_exit=False, on_restart=Mock(return_value=True))
        with patch.object(tray, "_tray_instance", instance):
            self.assertTrue(tray.request_restart())
        instance.on_restart.assert_called_once_with(None, None)
        # 新程序啟動失敗時，設定監看執行緒必須知道重啟沒有發生。
        instance.on_restart.return_value = False
        with patch.object(tray, "_tray_instance", instance):
            self.assertFalse(tray.request_restart())
        instance.should_exit = True
        with patch.object(tray, "_tray_instance", instance):
            self.assertFalse(tray.request_restart())
        with patch.object(tray, "_tray_instance", None):
            self.assertFalse(tray.request_restart())


class SettingsSystemPageTest(unittest.TestCase):
    def setUp(self):
        try:
            import tkinter as tk
            from fork_client.settings_ui import SettingsWindow
        except ImportError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Display unavailable (run with xvfb-run): {exc}")
        self.addCleanup(self.destroy_root)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "client.json"
        environment = patch.dict(os.environ, {"CAPSWRITER_CLIENT_SETTINGS": str(self.path)})
        environment.start()
        self.addCleanup(environment.stop)
        from config_client import ClientConfig

        self.registry = FakeRegistry()
        self.window = SettingsWindow(self.root, ClientConfig, from_client=True,
                                     autostart=Autostart(self.registry, command=COMMAND))
        self.root.update()

    def destroy_root(self):
        import tkinter as tk
        try:
            exists = self.root.winfo_exists()
        except tk.TclError:
            return
        if exists:
            with patch("fork_client.settings_ui.messagebox.askyesno", return_value=True):
                self.window.close()

    def registered(self):
        return self.registry.get(autostart_module.RUN_KEY, autostart_module.VALUE_NAME)

    def test_autostart_toggle_applies_on_save_and_prompts_when_unsaved(self):
        self.assertFalse(self.window.autostart_var.get())
        self.window.autostart_var.set(True)
        self.assertTrue(self.window.autostart_dirty)
        self.assertIsNone(self.registered())
        with patch("fork_client.settings_ui.messagebox.askyesno", return_value=False) as ask:
            self.window.close()
        ask.assert_called_once()
        self.assertTrue(self.window.save())
        self.assertEqual(self.registered(), Autostart(self.registry, command=COMMAND).expected())
        self.assertFalse(self.window.autostart_dirty)
        self.assertIn("已啟用", self.window.autostart_status.get())

    def test_save_and_restart_requests_restart_only_after_successful_save(self):
        self.window.variables["threshold"].set("not a number")
        self.window.save_and_restart()
        self.assertFalse(self.window.restart_requested)
        self.assertTrue(self.root.winfo_exists())
        self.window.variables["threshold"].set("0.5")
        self.window.save_and_restart()
        self.assertTrue(self.window.restart_requested)

    def test_autostart_still_applies_when_json_settings_are_unreadable(self):
        self.window.read_error = "broken"
        self.window.autostart_var.set(True)
        with patch("fork_client.settings_ui.messagebox.showerror") as error:
            self.assertFalse(self.window.save())
        error.assert_called_once()
        self.assertIsNotNone(self.registered())
        self.assertFalse(self.path.exists())

    def test_registry_failure_keeps_other_changes_unsaved(self):
        self.window.variables["addr"].set("axolotl")
        self.window.autostart_var.set(True)
        with patch.object(self.window.autostart, "set_enabled", side_effect=PermissionError("denied")), \
                patch("fork_client.settings_ui.messagebox.showerror") as error:
            self.assertFalse(self.window.save())
        error.assert_called_once()
        self.assertFalse(self.path.exists())
        self.assertTrue(self.window.autostart_dirty)


class SettingsSystemPageRegistryErrorTest(unittest.TestCase):
    def test_unreadable_autostart_state_disables_option_without_blocking_settings(self):
        try:
            import tkinter as tk
            from fork_client.settings_ui import SettingsWindow
        except ImportError as exc:
            self.skipTest(f"Tk unavailable: {exc}")
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Display unavailable (run with xvfb-run): {exc}")
        self.addCleanup(root.destroy)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "client.json"
        environment = patch.dict(os.environ, {"CAPSWRITER_CLIENT_SETTINGS": str(path)})
        environment.start()
        self.addCleanup(environment.stop)
        from config_client import ClientConfig

        registry = FakeRegistry()
        registry.get = Mock(side_effect=PermissionError("denied"))
        registry.set = Mock()
        registry.delete = Mock()
        window = SettingsWindow(root, ClientConfig, autostart=Autostart(registry, command=COMMAND))
        self.assertIn("無法讀取", window.autostart_status.get())
        self.assertIn("disabled", window.autostart_check.state())
        self.assertFalse(window.autostart_dirty)
        window.variables["addr"].set("axolotl")
        self.assertTrue(window.save())
        registry.set.assert_not_called()
        registry.delete.assert_not_called()
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
