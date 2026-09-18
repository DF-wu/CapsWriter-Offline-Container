"""Exercise desktop adapters with platform devices replaced at their boundary."""
from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fork_client.settings import websocket_url
from scripts.tests.test_client_recorder_cleanup import (
    FakeArray, FakeState, FakeWebSocketManager, load_recorder_class,
)

ROOT = Path(__file__).resolve().parents[2]


def load_class(relative_path, class_name, namespace):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[class_name]


class ClientSettingsRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_threshold_crossing_keeps_every_audio_block(self):
        recorder_class = load_recorder_class()
        namespace = recorder_class.record_and_send.__globals__
        namespace["Config"].threshold = 0.3
        namespace["Config"].save_audio = False
        blocks = [FakeArray(), FakeArray(), FakeArray()]
        joined = []

        def concatenate(items):
            joined.extend(items)
            return items[0]

        state = FakeState()
        websocket = FakeWebSocketManager()
        recorder = recorder_class(SimpleNamespace(state=state, ws=websocket))
        await state.queue_in.put({"type": "begin", "time": 0})
        for timestamp, data in zip((0.1, 0.2, 0.31), blocks):
            await state.queue_in.put({"type": "data", "time": timestamp, "data": data})
        await state.queue_in.put({"type": "finish", "time": 0.4})
        with patch.object(namespace["np"], "concatenate", side_effect=concatenate):
            await recorder.record_and_send()
        self.assertEqual(joined, blocks)
        self.assertEqual(state.released_chunks, 3)
        self.assertEqual([message.is_final for message in websocket.messages], [False, True])

    async def test_websocket_version_compatibility_and_disconnect_guidance(self):
        for modern in (False, True):
            with self.subTest(modern=modern):
                captured = []
                fail = True

                async def connect(uri, *, proxy="default", **kwargs):
                    captured.append({"uri": uri, "proxy": proxy, **kwargs})
                    if fail:
                        raise ConnectionRefusedError()
                    return SimpleNamespace()

                async def legacy_connect(uri, **kwargs):
                    self.assertNotIn("proxy", kwargs)
                    return await connect(uri, **kwargs)

                console = SimpleNamespace(print=Mock())
                namespace = {
                    "Config": SimpleNamespace(addr="::1", port="6016"),
                    "websocket_url": websocket_url, "inspect": inspect,
                    "websockets": SimpleNamespace(connect=connect if modern else legacy_connect),
                    "console": console, "logger": Mock(),
                    "CLIENT_WEBSOCKET_MAX_MESSAGE_BYTES": 1024,
                    "CLIENT_WEBSOCKET_MAX_QUEUED_MESSAGES": 4,
                }
                manager_class = load_class("core/client/connection/websocket_manager.py", "WebSocketManager", namespace)
                state = SimpleNamespace(is_connected=False, websocket=None)
                manager = manager_class(SimpleNamespace(state=state))
                self.assertFalse(await manager.connect())
                self.assertFalse(await manager.connect())
                self.assertEqual(console.print.call_count, 1)
                self.assertIn("設定", console.print.call_args.args[0])
                fail = False
                self.assertTrue(await manager.connect())
                self.assertFalse(manager._connect_fail_logged)
                self.assertEqual(captured[-1]["uri"], "ws://[::1]:6016")
                self.assertEqual(captured[-1]["open_timeout"], 4)
                self.assertEqual(captured[-1]["proxy"], None if modern else "default")

    async def test_selected_microphone_is_used_and_missing_device_is_recoverable(self):
        sounddevice = SimpleNamespace(
            query_devices=Mock(return_value={"name": "USB microphone", "max_input_channels": 1}),
            InputStream=Mock(return_value=Mock()), PortAudioError=RuntimeError,
        )
        namespace = {"sd": sounddevice, "Config": SimpleNamespace(input_device="USB microphone"), "console": Mock(), "logger": Mock()}
        manager_class = load_class("core/client/audio/stream.py", "AudioStreamManager", namespace)
        state = SimpleNamespace(stream=None)
        manager = manager_class(SimpleNamespace(state=state))
        self.assertIs(manager.start(), state.stream)
        sounddevice.query_devices.assert_called_once_with(device="USB microphone", kind="input")
        self.assertEqual(sounddevice.InputStream.call_args.kwargs["device"], "USB microphone")
        manager.stop()
        sounddevice.query_devices.side_effect = ValueError("device disappeared")
        self.assertIsNone(manager.start())
        self.assertFalse(manager._running)
        self.assertIsNone(state.stream)


if __name__ == "__main__":
    unittest.main()
