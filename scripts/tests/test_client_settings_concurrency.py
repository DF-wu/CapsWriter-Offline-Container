"""Actual editor save/reset behavior with UI widgets replaced, including processes."""
from __future__ import annotations

import copy
import multiprocessing
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from config_client import ClientConfig
from fork_client import settings
from fork_client.devices import device_label, input_device_choices
from scripts.tests.test_client_settings_runtime import load_class


class Variable:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def editor(path):
    namespace = dict(vars(settings))
    namespace.update(copy=copy, queue=queue, threading=threading,
                     device_label=device_label, input_device_choices=input_device_choices,
                     messagebox=SimpleNamespace(askyesno=lambda *a, **k: True,
                                                showerror=lambda *a, **k: None))
    cls = load_class("fork_client/settings_ui.py", "SettingsWindow", namespace)
    window = cls.__new__(cls)
    window.path = path
    window.config = ClientConfig
    window.overrides = settings.load_overrides(path)
    window.effective = settings.effective_settings(ClientConfig, window.overrides)
    window.variables = {key: Variable("" if value is None else value)
                        for key, value in window.effective.items() if key != "shortcuts"}
    window.shortcuts = copy.deepcopy(window.effective["shortcuts"])
    window.touched = set()
    window.device_choices = {}
    window.errors = {}
    window.status = Variable()
    window.read_error = None
    window.shortcut_draft_dirty = False
    window.root = None
    window.saved = False
    window.render_sources = Mock()
    window.render_shortcuts = Mock()
    return window


def save_in_process(path, ready, proceed, result):
    """Load before another editor saves, then use the real save method."""
    try:
        window = editor(Path(path))
        ready.set()
        if not proceed.wait(10):
            raise RuntimeError("parent did not release editor")
        window.variables["port"].set("7000")
        window.mark_dirty("port")
        window.save()
        result.put(window.saved)
    except Exception as exc:
        result.put(repr(exc))


def contending_write(path, entered, release, finished, operation):
    try:
        if operation == "first":
            original = settings.load_overrides

            def paused_read(path):
                values = original(path)
                entered.set()
                if not release.wait(10):
                    raise RuntimeError("parent did not release lock holder")
                return values

            settings.load_overrides = paused_read
            settings.update_overrides({"addr": "axolotl"}, Path(path))
        else:
            entered.set()
            if operation == "reset":
                settings.save_overrides({}, Path(path))
            else:
                settings.update_overrides({"port": "7000"}, Path(path))
        finished.put(None)
    except Exception as exc:
        finished.put(repr(exc))


class ClientSettingsConcurrencyTest(unittest.TestCase):
    def test_two_open_editors_merge_only_changed_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            first, second = editor(path), editor(path)
            first.variables["addr"].set("axolotl")
            first.mark_dirty("addr")
            first.save()
            second.variables["port"].set("7000")
            second.mark_dirty("port")
            second.save()
            self.assertEqual(settings.load_overrides(path), {"addr": "axolotl", "port": "7000"})
            self.assertEqual(second.variables["addr"].get(), "axolotl")

    def test_stale_editor_does_not_restore_fields_removed_by_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings.save_overrides({"addr": "axolotl", "paste": True}, path)
            resetter, stale = editor(path), editor(path)
            resetter.reset()
            stale.variables["port"].set("7000")
            stale.mark_dirty("port")
            stale.save()
            self.assertEqual(settings.load_overrides(path), {"port": "7000"})

    def test_editor_in_another_process_preserves_intervening_save(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            ready, proceed, result = context.Event(), context.Event(), context.Queue()
            process = context.Process(target=save_in_process, args=(str(path), ready, proceed, result))
            process.start()
            try:
                self.assertTrue(ready.wait(10))
                window = editor(path)
                window.variables["addr"].set("axolotl")
                window.mark_dirty("addr")
                window.save()
                proceed.set()
                self.assertIs(result.get(timeout=10), True)
                process.join(10)
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(settings.load_overrides(path), {"addr": "axolotl", "port": "7000"})
            finally:
                if process.is_alive():
                    process.terminate()
                process.join(10)
                result.close()

    def test_process_writers_and_reset_wait_for_the_read_modify_write_lock(self):
        context = multiprocessing.get_context("spawn")
        for operation, expected in (("merge", {"addr": "axolotl", "port": "7000"}), ("reset", {})):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "settings.json"
                held, attempted, release = context.Event(), context.Event(), context.Event()
                first_result, second_result = context.Queue(), context.Queue()
                first = context.Process(target=contending_write, args=(str(path), held, release, first_result, "first"))
                second = context.Process(target=contending_write, args=(str(path), attempted, release, second_result, operation))
                first.start()
                try:
                    self.assertTrue(held.wait(10))
                    second.start()
                    self.assertTrue(attempted.wait(10))
                    # A second writer must not finish while the first holds a stale read.
                    with self.assertRaises(queue.Empty):
                        second_result.get(timeout=0.3)
                    release.set()
                    self.assertIsNone(first_result.get(timeout=10))
                    self.assertIsNone(second_result.get(timeout=10))
                    self.assertEqual(settings.load_overrides(path), expected)
                finally:
                    release.set()
                    for process in (first, second):
                        if process.pid is not None:
                            process.join(10)
                            if process.is_alive():
                                process.terminate()
                                process.join(10)
                    first_result.close()
                    second_result.close()


if __name__ == "__main__":
    unittest.main()
