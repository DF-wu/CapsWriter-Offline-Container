"""Device selection keeps host API identity across UI, JSON and stream startup."""
from __future__ import annotations

from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fork_client.settings import SettingsError, load_overrides, save_overrides, validate
from fork_client.devices import input_device_choices, resolve_input_device
from scripts.tests.test_client_settings_concurrency import Variable, editor
from scripts.tests.test_client_settings_runtime import load_class


class AudioDevices:
    PortAudioError = RuntimeError

    def __init__(self):
        self.devices = [
            {"index": 0, "name": "USB Microphone", "hostapi": 0, "max_input_channels": 1},
            {"index": 1, "name": "USB Microphone", "hostapi": 1, "max_input_channels": 2},
            {"index": 2, "name": "Speaker", "hostapi": 1, "max_input_channels": 0},
        ]
        self.opened = None

    def query_hostapis(self):
        return [{"name": "MME"}, {"name": "Windows WASAPI"}]

    def query_devices(self, device=None, kind=None):
        if device is None and kind is None:
            return self.devices
        if device is None:
            return self.devices[0]
        if isinstance(device, int):
            return self.devices[device]
        if isinstance(device, str):
            matches = [item for item in self.devices if device.lower() in item["name"].lower()
                       and item["max_input_channels"] > 0]
            if len(matches) == 1:
                return matches[0]
        raise ValueError("input device is missing or ambiguous")

    def InputStream(self, **kwargs):
        # Model the actual audio library's selection, including bare-name ambiguity.
        self.opened = self.query_devices(kwargs["device"], kind="input")
        return SimpleNamespace(start=lambda: None, close=lambda: None)


def stream_manager(selection, sounddevice):
    namespace = {"sd": sounddevice, "Config": SimpleNamespace(input_device=selection),
                 "console": Mock(), "logger": Mock(), "resolve_input_device": resolve_input_device}
    cls = load_class("core/client/audio/stream.py", "AudioStreamManager", namespace)
    return cls(SimpleNamespace(state=SimpleNamespace(stream=None)))


class ClientSettingsDevicesTest(unittest.TestCase):
    def test_refresh_retains_same_name_devices_from_different_host_apis(self):
        with tempfile.TemporaryDirectory() as directory:
            window = editor(Path(directory) / "settings.json")
            window.pending = queue.Queue()
            window.device_names = Variable()
            with patch.dict("sys.modules", {"sounddevice": AudioDevices()}):
                window.refresh_devices()
                kind, (choices, _) = window.pending.get(timeout=5)
            self.assertEqual(kind, "devices")
            self.assertEqual(len(choices), 2)

    def test_ui_selection_reaches_stream_with_host_api_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            window = editor(path)
            window.pending = queue.Queue()
            window.device_names = Variable()
            window.root = Mock()
            window.mic = Mock()
            window.closed = False
            window.poll_after_id = None
            sounddevice = AudioDevices()
            with patch.dict("sys.modules", {"sounddevice": sounddevice}):
                window.refresh_devices()
                result = window.pending.get(timeout=5)
            window.pending.put(result)
            window.poll()
            labels = window.mic.configure.call_args.kwargs["values"]
            self.assertEqual(labels, ("", "USB Microphone [MME]", "USB Microphone [Windows WASAPI]"))
            window.variables["input_device"].set(labels[2])
            window.mark_dirty("input_device")
            window.save()
            selection = load_overrides(path)["input_device"]
            self.assertEqual(selection, {"name": "USB Microphone", "hostapi": "Windows WASAPI"})
            self.assertIsNotNone(stream_manager(selection, sounddevice).start())
            self.assertEqual(sounddevice.opened["index"], 1)

    def test_identical_names_within_one_host_api_can_each_be_selected(self):
        sounddevice = AudioDevices()
        sounddevice.devices[1]["hostapi"] = 0
        choices = input_device_choices(sounddevice)
        self.assertEqual(len(choices), 2)
        self.assertNotEqual(choices[0]["label"], choices[1]["label"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for index, choice in enumerate(choices):
                with self.subTest(index=index):
                    window = editor(path)
                    window.device_choices = {choice["label"]: choice["value"]}
                    window.variables["input_device"].set(choice["label"])
                    window.mark_dirty("input_device")
                    window.save()
                    saved = load_overrides(path)["input_device"]
                    self.assertEqual(saved, {"name": "USB Microphone", "hostapi": "MME", "index": index})
                    self.assertIsNotNone(stream_manager(saved, sounddevice).start())
                    self.assertEqual(sounddevice.opened["index"], index)

    def test_stable_identity_follows_device_when_indices_change(self):
        sounddevice = AudioDevices()
        sounddevice.devices[0], sounddevice.devices[1] = sounddevice.devices[1], sounddevice.devices[0]
        for index, device in enumerate(sounddevice.devices):
            device["index"] = index
        selection = {"name": "USB Microphone", "hostapi": "Windows WASAPI"}
        self.assertIsNotNone(stream_manager(selection, sounddevice).start())
        self.assertEqual(sounddevice.opened["index"], 0)

    def test_duplicate_or_missing_identity_never_silently_uses_another_device(self):
        for selection in (
            {"name": "USB Microphone", "hostapi": "MME"},
            {"name": "USB Microphone", "hostapi": "MME", "index": 2},
            {"name": "USB Microphone", "hostapi": "MME", "index": 99},
        ):
            with self.subTest(selection=selection):
                sounddevice = AudioDevices()
                sounddevice.devices[1]["hostapi"] = 0
                self.assertIsNone(stream_manager(selection, sounddevice).start())
                self.assertIsNone(sounddevice.opened)

    def test_invalid_structured_device_settings_are_rejected(self):
        for value in (
            {}, {"name": "mic"}, {"name": "mic", "hostapi": ""},
            {"name": 1, "hostapi": "MME"}, {"name": "mic", "hostapi": "MME", "index": True},
            {"name": "mic", "hostapi": "MME", "index": -1},
            {"name": "mic", "hostapi": "MME", "extra": 0},
        ):
            with self.subTest(value=value), self.assertRaises(SettingsError):
                validate({"input_device": value})

    def test_structured_selection_round_trips_and_opens_the_selected_host_api(self):
        selection = {"name": "USB Microphone", "hostapi": "Windows WASAPI"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            save_overrides({"input_device": selection}, path)
            saved = load_overrides(path)["input_device"]
            sounddevice = AudioDevices()
            manager = stream_manager(saved, sounddevice)
            self.assertIsNotNone(manager.start())
            self.assertEqual(sounddevice.opened["index"], 1)

    def test_missing_host_api_is_not_replaced_with_same_name_other_device(self):
        manager = stream_manager({"name": "USB Microphone", "hostapi": "missing"}, AudioDevices())
        self.assertIsNone(manager.start())

    def test_ambiguous_legacy_name_is_reported_without_opening_other_hardware(self):
        sounddevice = AudioDevices()
        self.assertIsNone(stream_manager("USB Microphone", sounddevice).start())
        self.assertIsNone(sounddevice.opened)


if __name__ == "__main__":
    unittest.main()
