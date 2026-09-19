"""Settings and startup checks without audio devices, model imports or hooks."""
from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from config_client import ClientConfig
from fork_client.settings import (
    SettingsError, apply_overrides, effective_settings, load_overrides,
    probe_connection, save_overrides, validate, websocket_url,
)
import start_client


class ClientSettingsTest(unittest.TestCase):
    def test_round_trip_only_overrides_and_retains_advanced_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            config = type("Config", (ClientConfig,), {"paste_apps": ["custom.exe"]})
            save_overrides({"addr": "axolotl", "port": 6016, "input_device": "USB 麥克風"}, path)
            self.assertEqual(apply_overrides(config, path), {
                "addr": "axolotl", "port": "6016", "input_device": "USB 麥克風",
            })
            self.assertEqual(config.addr, "axolotl")
            self.assertEqual(config.paste_apps, ["custom.exe"])
            self.assertNotIn("paste", load_overrides(path))
            save_overrides({}, path)
            self.assertEqual(load_overrides(path), {})

    def test_invalid_save_leaves_previous_file_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            save_overrides({"addr": "axolotl"}, path)
            before = path.read_bytes()
            with self.assertRaises(SettingsError):
                save_overrides({"port": 0}, path)
            self.assertEqual(path.read_bytes(), before)
            with patch("fork_client.settings.os.replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    save_overrides({"addr": "new-host"}, path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(set(path.parent.iterdir()), {path, path.with_name("settings.json.lock")})

    def test_rejects_corrupt_and_unknown_version_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for contents in ("{broken", json.dumps({"version": 2, "overrides": {}})):
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(SettingsError):
                    load_overrides(path)
                self.assertEqual(path.read_text(encoding="utf-8"), contents)

    def test_validation_rejects_wrong_types_and_out_of_range_values(self):
        for overrides in (
            {"port": True}, {"port": "9" * 5000}, {"threshold": 10**1000},
            {"threshold": float("nan")}, {"threshold": -0.1}, {"paste": 1},
            {"addr": "[axolotl]"}, {"addr": "[127.0.0.1]"},
            {"addr": "ws://axolotl:6016"}, {"addr": "axolotl/path"},
            {"input_device": " "}, {"language": ""}, {"paste_apps": []},
        ):
            with self.subTest(field=next(iter(overrides))):
                with self.assertRaises(SettingsError):
                    validate(overrides)

    def test_shortcuts_reject_duplicates_and_all_disabled(self):
        shortcut = copy.deepcopy(ClientConfig.shortcuts[0])
        for shortcuts in ([shortcut, shortcut], [{**shortcut, "enabled": False}], []):
            with self.assertRaises(SettingsError):
                validate({"shortcuts": shortcuts})
        values = effective_settings(ClientConfig, {"shortcuts": [shortcut]})
        values["shortcuts"][0]["key"] = "f12"
        self.assertEqual(ClientConfig.shortcuts[0]["key"], "caps_lock")

    def test_ipv6_and_hostname_urls(self):
        self.assertEqual(websocket_url("axolotl", 6016), "ws://axolotl:6016")
        for host in ("::1", "[::1]"):
            self.assertEqual(websocket_url(host, "6016"), "ws://[::1]:6016")

    def test_settings_mode_does_not_import_client_runtime(self):
        ui = SimpleNamespace(run_settings=Mock(return_value=False))
        with patch.dict("sys.modules", {"fork_client.settings_ui": ui, "core.client": None}):
            self.assertEqual(start_client.main(["--settings"]), 0)
        ui.run_settings.assert_called_once_with()

    def test_windows_first_run_cancel_exits_before_loading_hooks(self):
        ui = SimpleNamespace(run_settings=Mock(return_value=False))
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"CAPSWRITER_CLIENT_SETTINGS": str(Path(directory) / "new.json")}):
                with patch.dict("sys.modules", {"fork_client.settings_ui": ui, "core.client": None}):
                    with patch.object(start_client.sys, "platform", "win32"):
                        self.assertEqual(start_client.main([]), 0)
        ui.run_settings.assert_called_once_with()


class ConnectionProbeTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_handshake_sends_no_audio_and_closes_connection(self):
        try:
            import websockets
        except ImportError:
            self.skipTest("websockets is not installed")
        received = []
        closed = asyncio.Event()

        async def handler(socket):
            try:
                async for message in socket:
                    received.append(message)
            finally:
                closed.set()

        async with websockets.serve(handler, "127.0.0.1", 0, subprotocols=["binary"]) as server:
            port = server.sockets[0].getsockname()[1]
            await probe_connection("127.0.0.1", port)
            await asyncio.wait_for(closed.wait(), timeout=2)
        self.assertEqual(received, [])


if __name__ == "__main__":
    unittest.main()
