# coding: utf-8

from __future__ import annotations

import ast
import asyncio
import math
import os
from pathlib import Path
import threading
import unittest
import uuid
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "core" / "client" / "state.py"
STREAM_PATH = ROOT / "core" / "client" / "audio" / "stream.py"
RECORDER_PATH = ROOT / "core" / "client" / "audio" / "recorder.py"
WEBSOCKET_PATH = ROOT / "core" / "client" / "connection" / "websocket_manager.py"
TRANSCRIBER_PATH = ROOT / "core" / "client" / "transcribe" / "file_transcriber.py"
FILE_RUNNER_PATH = ROOT / "core" / "client" / "manager" / "file_runner.py"
SRT_ADJUSTER_PATH = ROOT / "core" / "client" / "transcribe" / "srt_adjuster.py"


def _class_method(path: Path, class_name: str, method_name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found in {path}")


def _load_ingress_state():
    tree = ast.parse(STATE_PATH.read_text(encoding="utf-8"), filename=str(STATE_PATH))
    constant_names = {
        "CLIENT_AUDIO_QUEUE_MAX_CHUNKS",
        "CLIENT_AUDIO_QUEUE_CONTROL_HEADROOM",
    }
    methods = {
        "reserve_audio_chunk",
        "release_audio_chunk",
        "mark_audio_input_overflow",
        "consume_audio_input_overflow",
        "pending_audio_chunks",
    }
    body: list[ast.stmt] = []
    class_body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constant_names
            for target in node.targets
        ):
            body.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "ClientState":
            class_body.extend(
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name in methods
            )
    body.append(
        ast.ClassDef(
            name="IngressState",
            bases=[],
            keywords=[],
            body=class_body,
            decorator_list=[],
        )
    )
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"threading": threading}
    exec(compile(module, str(STATE_PATH), "exec"), namespace)
    return namespace


def _load_transcribe_method():
    method = _class_method(TRANSCRIBER_PATH, "FileTranscriber", "transcribe")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    class _Logger:
        @staticmethod
        def error(*_args, **_kwargs):
            return None

        @staticmethod
        def debug(*_args, **_kwargs):
            return None

    namespace = {
        "asyncio": asyncio,
        "uuid": uuid,
        "client_file_result_timeout_seconds": lambda: 0.01,
        "logger": _Logger(),
    }
    exec(compile(module, str(TRANSCRIBER_PATH), "exec"), namespace)
    return namespace["transcribe"]


def _load_file_result_timeout_parser():
    tree = ast.parse(TRANSCRIBER_PATH.read_text(encoding="utf-8"))
    keep = {
        "CLIENT_FILE_RESULT_TIMEOUT_ENV",
        "DEFAULT_CLIENT_FILE_RESULT_TIMEOUT_SECONDS",
    }
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in keep
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "client_file_result_timeout_seconds"
        )
    ]
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math, "os": os}
    exec(compile(module, str(TRANSCRIBER_PATH), "exec"), namespace)
    return namespace


class ClientAudioIngressTest(unittest.TestCase):
    def test_callback_reservations_have_a_hard_cap_and_sticky_overflow(self) -> None:
        namespace = _load_ingress_state()
        state = namespace["IngressState"]()
        state._audio_ingress_lock = threading.Lock()
        state._audio_reserved_chunks = 0
        state._audio_input_overflow = False
        limit = namespace["CLIENT_AUDIO_QUEUE_MAX_CHUNKS"]

        self.assertTrue(all(state.reserve_audio_chunk() for _ in range(limit)))
        self.assertFalse(state.reserve_audio_chunk())
        self.assertEqual(state.pending_audio_chunks, limit)
        self.assertTrue(state.consume_audio_input_overflow())
        self.assertFalse(state.consume_audio_input_overflow())

        state.release_audio_chunk()
        self.assertTrue(state.reserve_audio_chunk())
        self.assertEqual(state.pending_audio_chunks, limit)

    def test_portaudio_callback_never_schedules_blocking_queue_put_coroutines(self) -> None:
        callback = _class_method(STREAM_PATH, "AudioStreamManager", "_audio_callback")
        calls = {
            node.func.attr
            for node in ast.walk(callback)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
        }

        self.assertIn("reserve_audio_chunk", calls)
        self.assertIn("call_soon_threadsafe", calls)
        self.assertIn("put_nowait", calls)
        self.assertNotIn("run_coroutine_threadsafe", calls)

    def test_microphone_sends_are_awaited_and_uuid1_is_absent(self) -> None:
        recorder_source = RECORDER_PATH.read_text(encoding="utf-8")
        recorder = _class_method(RECORDER_PATH, "AudioRecorder", "record_and_send")
        unawaited_create_tasks = [
            node
            for node in ast.walk(recorder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "create_task"
        ]
        awaited_sends = [
            node
            for node in ast.walk(recorder)
            if isinstance(node, ast.Await)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "_send_message"
        ]

        self.assertEqual(unawaited_create_tasks, [])
        self.assertGreaterEqual(len(awaited_sends), 3)
        self.assertNotIn("uuid.uuid1", recorder_source)
        self.assertIn("uuid.uuid4", recorder_source)


class ClientWebSocketBackpressureTest(unittest.TestCase):
    def test_client_websocket_buffers_and_send_deadline_are_bounded(self) -> None:
        source = WEBSOCKET_PATH.read_text(encoding="utf-8")

        self.assertIn("max_size=CLIENT_WEBSOCKET_MAX_MESSAGE_BYTES", source)
        self.assertIn("max_queue=CLIENT_WEBSOCKET_MAX_QUEUED_MESSAGES", source)
        self.assertNotIn("max_size=None", source)
        self.assertNotIn("max_queue=None", source)
        send = _class_method(WEBSOCKET_PATH, "WebSocketManager", "send")
        self.assertTrue(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "wait_for"
                for node in ast.walk(send)
            )
        )

    def test_send_timeout_parser_rejects_invalid_values(self) -> None:
        tree = ast.parse(WEBSOCKET_PATH.read_text(encoding="utf-8"))
        keep = {
            "CLIENT_WEBSOCKET_SEND_TIMEOUT_ENV",
            "DEFAULT_CLIENT_WEBSOCKET_SEND_TIMEOUT_SECONDS",
        }
        body = [
            node
            for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in keep
                    for target in node.targets
                )
            )
            or (
                isinstance(node, ast.FunctionDef)
                and node.name == "client_websocket_send_timeout_seconds"
            )
        ]
        module = ast.Module(body=body, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"math": math, "os": os}
        exec(compile(module, str(WEBSOCKET_PATH), "exec"), namespace)
        parser = namespace["client_websocket_send_timeout_seconds"]
        env_name = namespace["CLIENT_WEBSOCKET_SEND_TIMEOUT_ENV"]

        with patch.dict(os.environ, {env_name: "2.5"}):
            self.assertEqual(parser(), 2.5)
        for value in ("0", "nan", "inf", "bad"):
            with self.subTest(value=value), patch.dict(os.environ, {env_name: value}):
                with self.assertRaises(ValueError):
                    parser()

    def test_file_transcription_receives_concurrently_and_cancels_on_send_failure(self) -> None:
        transcribe = _load_transcribe_method()
        events: list[str] = []

        class FakeTranscriber:
            task_id = None

            async def receive(self):
                events.append("receive-started")
                try:
                    await asyncio.Future()
                finally:
                    events.append("receive-cancelled")

            async def send(self):
                events.append("send-started")
                await asyncio.sleep(0)
                return False

        FakeTranscriber.transcribe = transcribe

        result = asyncio.run(FakeTranscriber().transcribe())

        self.assertFalse(result)
        self.assertEqual(
            events,
            ["send-started", "receive-started", "receive-cancelled"],
        )

    def test_file_result_wait_has_a_validated_deadline(self) -> None:
        namespace = _load_file_result_timeout_parser()
        parser = namespace["client_file_result_timeout_seconds"]
        env_name = namespace["CLIENT_FILE_RESULT_TIMEOUT_ENV"]

        with patch.dict(os.environ, {env_name: "2.5"}):
            self.assertEqual(parser(), 2.5)
        for value in ("0", "nan", "inf", "bad"):
            with self.subTest(value=value), patch.dict(os.environ, {env_name: value}):
                with self.assertRaises(ValueError):
                    parser()

    def test_file_result_timeout_cancels_receive_and_closes_connection(self) -> None:
        transcribe = _load_transcribe_method()
        events: list[str] = []

        class FakeWebSocketManager:
            async def close(self):
                events.append("connection-closed")

        class FakeTranscriber:
            task_id = None
            file = Path("silent.wav")
            _ws_manager = FakeWebSocketManager()

            async def receive(self):
                events.append("receive-started")
                try:
                    await asyncio.Future()
                finally:
                    events.append("receive-cancelled")

            async def send(self):
                events.append("send-complete")
                return True

        FakeTranscriber.transcribe = transcribe

        result = asyncio.run(FakeTranscriber().transcribe())

        self.assertFalse(result)
        self.assertEqual(
            events,
            [
                "send-complete",
                "receive-started",
                "receive-cancelled",
                "connection-closed",
            ],
        )

    def test_file_runner_uses_concurrent_transcribe_surface_and_no_uuid1_remains(self) -> None:
        runner_source = FILE_RUNNER_PATH.read_text(encoding="utf-8")
        all_sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (RECORDER_PATH, TRANSCRIBER_PATH, SRT_ADJUSTER_PATH)
        )

        self.assertIn("await transcriber.transcribe()", runner_source)
        self.assertNotIn("await transcriber.send()", runner_source)
        self.assertNotIn("uuid.uuid1", all_sources)


if __name__ == "__main__":
    unittest.main()
