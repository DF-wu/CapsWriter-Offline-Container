"""Run with xvfb-run to exercise the native UI on Linux; no hardware needed."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from config_client import ClientConfig
from fork_client.settings import load_overrides
from fork_client.shortcut_keys import normalize_shortcut


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
        import tkinter as tk
        try:
            exists = self.root.winfo_exists()
        except tk.TclError:
            return
        if exists:
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

    def capture_events(self, keys):
        self.window.record_shortcut()
        self.root.update()
        dialog = self.window.capture_dialog
        dialog.focus_force()
        self.root.update()
        for key in keys:
            dialog.event_generate("<KeyPress>", keysym=key)
            self.root.update()
        for key in keys:
            dialog.event_generate("<KeyRelease>", keysym=key)
            self.root.update()
        self.finish_releases()

    def finish_releases(self):
        self.root.after(30, self.root.quit)
        self.root.mainloop()
        self.root.update()

    def test_record_chord_apply_save_and_reload(self):
        self.capture_events(("F1", "F2", "F3"))
        self.assertIsNone(self.window.capture_dialog)
        self.assertEqual(self.window.key.get(), "f1+f2+f3")
        self.window.save()
        self.assertFalse(self.path.exists())
        self.assertIn("尚未套用", self.window.status.get())
        self.window.update_shortcut()
        self.window.save()
        self.assertEqual(load_overrides(self.path)["shortcuts"][0]["key"], "f1+f2+f3")
        self.window.reload_fields()
        self.assertEqual(self.window.key.get(), "f1+f2+f3")

    def test_record_backtick_combination(self):
        # F13/F14 symbolic capture is covered without a display in
        # test_shortcut_capture: minimal X11 keymaps cannot synthesize them.
        self.capture_events(("grave", "1"))
        self.assertEqual(self.window.key.get(), normalize_shortcut("`+1"))

    def test_cancel_capture_preserves_previous_value_and_draft(self):
        before = self.window.key.get()
        self.window.record_shortcut()
        self.root.update()
        dialog = self.window.capture_dialog
        dialog.focus_force()
        self.root.update()
        dialog.event_generate("<KeyPress>", keysym="F1")
        dialog.event_generate("<KeyPress>", keysym="Escape")
        self.root.update()
        self.assertIsNone(self.window.capture_dialog)
        self.assertTrue(self.root.winfo_exists())
        self.assertEqual(self.window.key.get(), before)
        self.assertFalse(self.window.shortcut_draft_dirty)

    def test_capture_ctrl_s_does_not_save_settings(self):
        self.window.variables["addr"].set("test-host")
        self.capture_events(("Control_L", "s"))
        self.assertEqual(self.window.key.get(), normalize_shortcut("ctrl_l+s"))
        self.assertFalse(self.path.exists())

    def test_capture_focus_loss_discards_partial_keys(self):
        self.window.record_shortcut()
        self.root.update()
        dialog = self.window.capture_dialog
        dialog.focus_force()
        self.root.update()
        dialog.event_generate("<KeyPress>", keysym="Control_L")
        dialog.event_generate("<FocusOut>")
        dialog.event_generate("<KeyPress>", keysym="F1")
        dialog.event_generate("<KeyRelease>", keysym="F1")
        self.finish_releases()
        self.assertEqual(self.window.key.get(), "f1")

    def test_x11_repeat_release_press_does_not_finish_capture(self):
        self.window.record_shortcut()
        self.root.update()
        dialog = self.window.capture_dialog
        dialog.focus_force()
        self.root.update()
        dialog.event_generate("<KeyPress>", keysym="F1", time=10)
        dialog.event_generate("<KeyRelease>", keysym="F1", time=100)
        dialog.event_generate("<KeyPress>", keysym="F1", time=100)
        self.root.update()
        self.assertIs(self.window.capture_dialog, dialog)
        dialog.event_generate("<KeyPress>", keysym="F2", time=110)
        dialog.event_generate("<KeyRelease>", keysym="F1", time=120)
        dialog.event_generate("<KeyRelease>", keysym="F2", time=130)
        self.finish_releases()
        self.assertEqual(self.window.key.get(), "f1+f2")

    def test_record_both_modifier_sides(self):
        self.capture_events(("Control_L", "Control_R"))
        self.assertEqual(self.window.key.get(), "ctrl_l+ctrl_r")

    def test_close_settings_cancels_pending_capture_callback(self):
        self.window.record_shortcut()
        self.root.update()
        dialog = self.window.capture_dialog
        dialog.focus_force()
        self.root.update()
        dialog.event_generate("<KeyPress>", keysym="F1")
        dialog.event_generate("<KeyRelease>", keysym="F1")
        self.window.close()
        self.assertIsNone(self.window.capture_dialog)
        self.assertIsNone(self.window.cancel_capture)


if __name__ == "__main__":
    unittest.main()
