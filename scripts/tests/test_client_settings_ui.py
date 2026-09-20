"""Run with xvfb-run to exercise the native UI on Linux; no hardware needed."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config_client import ClientConfig
from fork_client.settings import load_overrides


class ClientSettingsUITest(unittest.TestCase):
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
        self.environment = patch.dict(os.environ, {"CAPSWRITER_CLIENT_SETTINGS": str(self.path)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.window = SettingsWindow(self.root, ClientConfig)
        self.root.update()

    def destroy_root(self):
        if self.root.winfo_exists():
            with patch("fork_client.settings_ui.messagebox.askyesno", return_value=True):
                self.window.close()

    def test_save_only_changed_fields_and_reset_to_python_config(self):
        self.window.variables["addr"].set("axolotl")
        self.window.variables["paste"].set(True)
        self.window.save()
        self.assertEqual(load_overrides(self.path), {"addr": "axolotl", "paste": True})
        self.assertFalse(self.window.touched)
        with patch("fork_client.settings_ui.messagebox.askyesno", return_value=True):
            self.window.reset()
        self.assertEqual(load_overrides(self.path), {})
        self.assertEqual(self.window.variables["addr"].get(), ClientConfig.addr)

    def test_invalid_numeric_value_prevents_save_and_selects_field_page(self):
        self.window.variables["threshold"].set("not a number")
        self.window.save()
        self.assertFalse(self.path.exists())
        self.assertIn("錯誤", self.window.errors["threshold"][0].get())
        self.assertEqual(self.window.notebook.select(), str(self.window.pages["錄音與快捷鍵"]))

    def test_edit_shortcut_requires_apply_and_updates_selected_default(self):
        self.window.key.set("f12")
        self.window.save()
        self.assertFalse(self.path.exists())
        self.assertIn("尚未套用", self.window.status.get())
        self.window.update_shortcut()
        self.window.save()
        saved = load_overrides(self.path)["shortcuts"]
        self.assertEqual(len(saved), len(ClientConfig.shortcuts))
        self.assertEqual(saved[0]["key"], "f12")

    def test_device_refresh_populates_choices_without_changing_selection(self):
        selection = {"name": "USB Microphone", "hostapi": "Windows WASAPI"}
        label = "USB Microphone [Windows WASAPI]"
        self.window.pending.put(("devices", ([{"label": label, "value": selection}], "找到 1 個麥克風")))
        self.window.poll()
        self.assertIn(label, self.window.mic.cget("values"))
        self.assertEqual(self.window.variables["input_device"].get(), "")
        self.assertNotIn("input_device", self.window.touched)
        self.window.variables["input_device"].set(label)
        self.window.save()
        self.assertEqual(load_overrides(self.path)["input_device"], selection)


if __name__ == "__main__":
    unittest.main()
