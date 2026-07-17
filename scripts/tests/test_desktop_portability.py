# coding: utf-8

from __future__ import annotations

import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import start_server_universal
from fork_server.http_api.runtime_config import ConfigError


ROOT = Path(__file__).resolve().parents[2]
CLIENT_ROOT = ROOT / "core" / "client"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PLATFORM_SUPPORT = load_module(
    "capswriter_platform_support_test",
    ROOT / "core" / "client" / "shortcut" / "platform_support.py",
)


class FakeLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class FakeListener:
    instances: list["FakeListener"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        type(self).instances.append(self)

    def start(self):
        self.started = True

    def is_alive(self):
        return self.started

    def stop(self):
        self.started = False


class FakeKeyboardListener(FakeListener):
    instances = []


class FakeMouseListener(FakeListener):
    instances = []


class FakeEventHandler:
    def __init__(self, *args):
        self.events = []

    def handle_keydown(self, key_name, task):
        self.events.append(("down", key_name, task))

    def handle_keyup(self, key_name, task):
        self.events.append(("up", key_name, task))


class FakeEmulator:
    def __init__(self):
        self.emulating = set()

    def is_emulating(self, key_name):
        return key_name in self.emulating

    def clear_emulating_flag(self, key_name):
        self.emulating.discard(key_name)


class FakeTask:
    def __init__(self, app, shortcut):
        self.app = app
        self.shortcut = shortcut


class FakeShortcut:
    def __init__(self, key, input_type, suppress=False):
        self.key = key
        self.type = input_type
        self.suppress = suppress
        self.enabled = True
        self.hold_mode = True

    def get_threshold(self, default):
        return default

    def is_toggle_key(self):
        return self.key == "caps_lock"


@contextmanager
def loaded_shortcut_manager():
    core = ModuleType("core")
    core.__path__ = [str(ROOT / "core")]
    client = ModuleType("core.client")
    client.__path__ = [str(ROOT / "core" / "client")]
    shortcut = ModuleType("core.client.shortcut")
    shortcut.__path__ = [str(ROOT / "core" / "client" / "shortcut")]
    shortcut.logger = FakeLogger()

    key_mapper = ModuleType("core.client.shortcut.key_mapper")
    key_mapper.KeyMapper = type("KeyMapper", (), {})
    constants = {
        "KEYBOARD_MESSAGES": (1, 2),
        "KEY_DOWN_MESSAGES": (1,),
        "KEY_UP_MESSAGES": (2,),
        "MOUSE_MESSAGES": (3, 4),
        "WM_KEYUP": 2,
        "WM_SYSKEYUP": 5,
        "WM_XBUTTONDOWN": 3,
        "WM_XBUTTONUP": 4,
        "XBUTTON1": 1,
        "XBUTTON2": 2,
    }
    for name, value in constants.items():
        setattr(key_mapper, name, value)

    emulator = ModuleType("core.client.shortcut.emulator")
    emulator.ShortcutEmulator = FakeEmulator
    event_handler = ModuleType("core.client.shortcut.event_handler")
    event_handler.ShortcutEventHandler = FakeEventHandler
    task = ModuleType("core.client.shortcut.task")
    task.ShortcutTask = FakeTask

    pynput = ModuleType("pynput")
    pynput.keyboard = SimpleNamespace(Listener=FakeKeyboardListener)
    pynput.mouse = SimpleNamespace(Listener=FakeMouseListener)

    config_client = ModuleType("config_client")
    config_client.ClientConfig = SimpleNamespace(threshold=0.3)

    modules = {
        "core": core,
        "core.client": client,
        "core.client.shortcut": shortcut,
        "core.client.shortcut.key_mapper": key_mapper,
        "core.client.shortcut.emulator": emulator,
        "core.client.shortcut.event_handler": event_handler,
        "core.client.shortcut.platform_support": PLATFORM_SUPPORT,
        "core.client.shortcut.task": task,
        "pynput": pynput,
        "config_client": config_client,
    }
    module_name = "core.client.shortcut._portability_manager_test"
    with patch.dict(sys.modules, modules):
        manager = load_module(
            module_name,
            ROOT / "core" / "client" / "shortcut" / "shortcut_manager.py",
        )
        try:
            yield manager
        finally:
            sys.modules.pop(module_name, None)


class DesktopPortabilityTest(unittest.TestCase):
    def test_client_import_surface_has_no_eager_pynput_backend_import(self):
        eager_imports = []
        for path in CLIENT_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if isinstance(node, ast.ImportFrom) and node.module == "pynput":
                    eager_imports.append(str(path.relative_to(ROOT)))
                elif isinstance(node, ast.Import) and any(
                    alias.name == "pynput" for alias in node.names
                ):
                    eager_imports.append(str(path.relative_to(ROOT)))

        self.assertEqual(eager_imports, [])

    def test_key_mapper_does_not_import_win32_backend_on_linux(self):
        core = ModuleType("core")
        core.__path__ = [str(ROOT / "core")]
        client = ModuleType("core.client")
        client.__path__ = [str(ROOT / "core" / "client")]
        shortcut = ModuleType("core.client.shortcut")
        shortcut.__path__ = [str(ROOT / "core" / "client" / "shortcut")]
        shortcut.logger = FakeLogger()
        module_name = "core.client.shortcut._key_mapper_portability_test"

        with patch.dict(
            sys.modules,
            {
                "core": core,
                "core.client": client,
                "core.client.shortcut": shortcut,
            },
        ):
            mapper = load_module(
                module_name,
                ROOT / "core" / "client" / "shortcut" / "key_mapper.py",
            )
            try:
                with patch.object(mapper, "system", return_value="Linux"):
                    self.assertIsNone(mapper._get_key_translator())
                self.assertNotIn("pynput._util.win32", sys.modules)
            finally:
                sys.modules.pop(module_name, None)

    def test_hotkey_backend_policy_distinguishes_windows_x11_and_wayland(self):
        windows = PLATFORM_SUPPORT.detect_hotkey_backend("Windows", {})
        x11 = PLATFORM_SUPPORT.detect_hotkey_backend(
            "Linux", {"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}
        )
        wayland = PLATFORM_SUPPORT.detect_hotkey_backend(
            "Linux",
            {
                "XDG_SESSION_TYPE": "wayland",
                "DISPLAY": ":0",
                "WAYLAND_DISPLAY": "wayland-0",
            },
        )
        headless = PLATFORM_SUPPORT.detect_hotkey_backend("Linux", {})

        self.assertEqual((windows.backend, windows.selective_suppression), ("win32", True))
        self.assertEqual((x11.backend, x11.selective_suppression), ("x11", False))
        self.assertTrue(x11.available)
        self.assertFalse(wayland.available)
        self.assertEqual(wayland.backend, "wayland")
        self.assertFalse(headless.available)
        self.assertEqual(headless.backend, "headless")

    def test_pynput_event_names_are_normalized_without_a_backend_import(self):
        self.assertEqual(
            PLATFORM_SUPPORT.pynput_key_name(SimpleNamespace(name="CAPS_LOCK")),
            "caps_lock",
        )
        self.assertEqual(
            PLATFORM_SUPPORT.pynput_key_name(SimpleNamespace(char="A")),
            "a",
        )
        self.assertEqual(
            PLATFORM_SUPPORT.pynput_button_name(SimpleNamespace(name="button8")),
            "x1",
        )
        self.assertEqual(
            PLATFORM_SUPPORT.pynput_button_name(SimpleNamespace(name="button9")),
            "x2",
        )

    def test_universal_entrypoint_changes_only_http_settings(self):
        class DesktopConfig:
            enable_tray = True
            model_type = "sensevoice"
            addr = "0.0.0.0"

        settings = start_server_universal.configure_http_api(
            {
                "CAPSWRITER_HTTP_API_ENABLE": "true",
                "CAPSWRITER_HTTP_API_BIND": "127.0.0.1",
                "CAPSWRITER_HTTP_API_PORT": "6027",
                "CAPSWRITER_HTTP_API_CORS_ORIGINS": "http://localhost:5173",
                "CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS": "11",
                "CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS": "7200.5",
            },
            DesktopConfig,
        )

        self.assertTrue(settings.enable)
        self.assertEqual(DesktopConfig.http_api_port, 6027)
        self.assertEqual(DesktopConfig.http_api_cors_origins, ["http://localhost:5173"])
        self.assertTrue(DesktopConfig.enable_tray)
        self.assertEqual(DesktopConfig.model_type, "sensevoice")
        self.assertEqual(DesktopConfig.addr, "0.0.0.0")
        self.assertEqual(DesktopConfig.max_websocket_connections, 11)
        self.assertEqual(DesktopConfig.max_websocket_task_seconds, 7200.5)

    def test_universal_entrypoint_rejects_invalid_http_environment(self):
        with self.assertRaises(ConfigError):
            start_server_universal.configure_http_api(
                {"CAPSWRITER_HTTP_API_PORT": "invalid"},
                type("DesktopConfig", (), {}),
            )

        with self.assertRaises(ConfigError):
            start_server_universal.configure_http_api(
                {"CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS": "0"},
                type("DesktopConfig", (), {}),
            )

        with self.assertRaises(ConfigError):
            start_server_universal.configure_http_api(
                {"CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS": "86401"},
                type("DesktopConfig", (), {}),
            )

    def test_universal_entrypoint_applies_explicit_websocket_bind(self):
        class DesktopConfig:
            addr = "0.0.0.0"

        start_server_universal.configure_http_api(
            {"CAPSWRITER_SERVER_ADDR": "127.0.0.1"},
            DesktopConfig,
        )

        self.assertEqual(DesktopConfig.addr, "127.0.0.1")

    def test_universal_entrypoint_rejects_unsafe_websocket_bind(self):
        for value in ("", "http://127.0.0.1", "host name", "../socket"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    start_server_universal.configure_http_api(
                        {"CAPSWRITER_SERVER_ADDR": value},
                        type("DesktopConfig", (), {"addr": "0.0.0.0"}),
                    )

    def test_universal_entrypoint_uses_upstream_server_when_http_is_off(self):
        started = []

        class UpstreamServer:
            def start(self):
                started.append("upstream")

        core = ModuleType("core")
        core.__path__ = []
        server_package = ModuleType("core.server")
        server_package.__path__ = []
        server_app = ModuleType("core.server.app")
        server_app.CapsWriterServer = UpstreamServer
        settings = SimpleNamespace(enable=False)

        with patch.object(
            start_server_universal,
            "configure_http_api",
            return_value=settings,
        ), patch.dict(
            sys.modules,
            {
                "core": core,
                "core.server": server_package,
                "core.server.app": server_app,
            },
        ):
            start_server_universal.main()

        self.assertEqual(started, ["upstream"])

    def test_universal_entrypoint_uses_fork_server_when_http_is_on(self):
        started = []

        class ForkServer:
            def start(self):
                started.append("fork")

        bootstrap = ModuleType("fork_server.bootstrap")
        bootstrap.create_server = ForkServer
        settings = SimpleNamespace(enable=True)

        with patch.object(
            start_server_universal,
            "configure_http_api",
            return_value=settings,
        ), patch.dict(sys.modules, {"fork_server.bootstrap": bootstrap}):
            start_server_universal.main()

        self.assertEqual(started, ["fork"])

    def test_x11_callbacks_dispatch_keyboard_and_side_mouse_events(self):
        with loaded_shortcut_manager() as module:
            manager = module.ShortcutManager.__new__(module.ShortcutManager)
            keyboard_task = object()
            mouse_task = object()
            manager.tasks = {"caps_lock": keyboard_task, "x1": mouse_task}
            manager._emulator = FakeEmulator()
            manager._event_handler = FakeEventHandler()
            manager._restoring_keys = set()
            mouse_events = []
            manager._handle_mouse_keyup = lambda name, task: mouse_events.append((name, task))

            on_press, on_release = manager.create_portable_keyboard_callbacks()
            on_click = manager.create_portable_mouse_callback()
            key = SimpleNamespace(name="caps_lock")
            on_press(key)
            on_release(key)
            on_press(key, injected=True)
            on_click(0, 0, SimpleNamespace(name="button8"), True)
            on_click(0, 0, SimpleNamespace(name="button8"), False)

            self.assertEqual(
                manager._event_handler.events,
                [
                    ("down", "caps_lock", keyboard_task),
                    ("up", "caps_lock", keyboard_task),
                    ("down", "x1", mouse_task),
                ],
            )
            self.assertEqual(mouse_events, [("x1", mouse_task)])

    def test_listener_wiring_preserves_win32_and_adds_x11_callbacks(self):
        with loaded_shortcut_manager() as module:
            for backend in ("win32", "x11"):
                with self.subTest(backend=backend):
                    FakeKeyboardListener.instances.clear()
                    FakeMouseListener.instances.clear()
                    manager = module.ShortcutManager.__new__(module.ShortcutManager)
                    manager._hotkey_support = PLATFORM_SUPPORT.HotkeyBackendSupport(
                        available=True,
                        backend=backend,
                        selective_suppression=backend == "win32",
                        detail="test",
                    )
                    manager.shortcuts = [
                        FakeShortcut("caps_lock", "keyboard"),
                        FakeShortcut("x1", "mouse"),
                    ]
                    manager.tasks = {}
                    manager.keyboard_listener = None
                    manager.mouse_listener = None
                    manager._emulator = FakeEmulator()
                    manager._event_handler = FakeEventHandler()
                    manager.start()

                    keyboard_kwargs = FakeKeyboardListener.instances[-1].kwargs
                    mouse_kwargs = FakeMouseListener.instances[-1].kwargs
                    if backend == "win32":
                        self.assertIn("win32_event_filter", keyboard_kwargs)
                        self.assertIn("win32_event_filter", mouse_kwargs)
                    else:
                        self.assertIn("on_press", keyboard_kwargs)
                        self.assertIn("on_release", keyboard_kwargs)
                        self.assertFalse(keyboard_kwargs["suppress"])
                        self.assertIn("on_click", mouse_kwargs)
                        self.assertFalse(mouse_kwargs["suppress"])

    def test_x11_backend_failure_is_reported_without_constructing_listeners(self):
        class BrokenEmulator:
            def __init__(self):
                raise RuntimeError("cannot connect to display")

        with loaded_shortcut_manager() as module, patch.object(
            module,
            "detect_hotkey_backend",
            return_value=PLATFORM_SUPPORT.HotkeyBackendSupport(
                available=True,
                backend="x11",
                selective_suppression=False,
                detail="X11 test backend",
            ),
        ), patch.object(module, "ShortcutEmulator", BrokenEmulator):
            manager = module.ShortcutManager(object(), [])
            try:
                self.assertFalse(manager._hotkey_support.available)
                self.assertIn("cannot connect to display", manager._hotkey_support.detail)
                self.assertIsNone(manager._emulator)
            finally:
                manager._pool.shutdown(wait=False)

    def test_x11_downgrades_per_key_suppression_without_mutating_config(self):
        with loaded_shortcut_manager() as module:
            shortcut = FakeShortcut("caps_lock", "keyboard", suppress=True)
            manager = module.ShortcutManager.__new__(module.ShortcutManager)
            manager.app = object()
            manager.shortcuts = [shortcut]
            manager.tasks = {}
            manager._pool = object()
            manager._hotkey_support = PLATFORM_SUPPORT.HotkeyBackendSupport(
                available=True,
                backend="x11",
                selective_suppression=False,
                detail="test",
            )

            manager._init_tasks()

            self.assertTrue(shortcut.suppress)
            self.assertFalse(manager.tasks["caps_lock"].shortcut.suppress)


if __name__ == "__main__":
    unittest.main()
