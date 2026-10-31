"""在 Windows 的 detached 子程序測試真實 pystray，不開麥克風或鍵盤 hook。"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
NATIVE_AVAILABLE = os.name == "nt" and all(
    importlib.util.find_spec(name) is not None for name in ("pystray", "PIL")
)


@unittest.skipUnless(NATIVE_AVAILABLE, "需要 Windows、pystray 與 Pillow")
class NativeTrayTest(unittest.TestCase):
    def test_windows_client_version_resource_can_be_compiled(self):
        try:
            from PyInstaller.utils.win32.versioninfo import load_version_info_from_text_file
        except ImportError:
            self.skipTest("需要 PyInstaller 才能驗證 PE version resource")
        resource = load_version_info_from_text_file(str(ROOT / "assets/windows-client-version.txt"))
        self.assertIn("CapsWriter Offline Client".encode("utf-16le"), resource.toRaw())

    def test_detached_tray_can_exit_and_leaves_no_threads(self):
        result = subprocess.run(
            # Windows venv 的轉送 launcher 可能替子程序重建 console；直接測真正的直譯器。
            [sys._base_executable, "-m", "scripts.tests.test_tray_native", "--native-worker"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", timeout=15,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report, {"cycles": 3, "console": False, "stopped": True})


def native_worker():
    core = ModuleType("core")
    core.__path__ = [str(ROOT / "core")]
    ui = ModuleType("core.ui")
    ui.__path__ = [str(ROOT / "core/ui")]
    ui.logger = logging.getLogger("native-tray-test")
    ui.set_ui_logger = lambda _logger: None
    spec = importlib.util.spec_from_file_location("core.ui.tray", ROOT / "core/ui/tray.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"core": core, "core.ui": ui, "core.ui.tray": module}):
        spec.loader.exec_module(module)
        assert not module._get_console_hwnd(), "測試必須在 detached 子程序執行"
        exits = []
        module._set_exit_callback(lambda: exits.append(True))
        for cycle in range(3):
            module.enable_min_to_tray("CapsWriter Client Test")
            tray = module._tray_instance
            deadline = time.monotonic() + 3
            while not tray.icon.visible and time.monotonic() < deadline:
                threading.Event().wait(0.02)
            assert tray.icon.visible, "沒有主控台時托盤未建立"
            tray.on_exit(None, None)
            module.stop_tray()
            assert tray.should_exit
            assert not tray._tray_thread.is_alive(), "托盤執行緒殘留"
            assert not tray._monitor_thread.is_alive(), "監控執行緒殘留"
        assert len(exits) == 3
    print(json.dumps({"cycles": 3, "console": False, "stopped": True}), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--native-worker"]:
        native_worker()
    else:
        unittest.main()
