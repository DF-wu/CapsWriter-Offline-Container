# coding: utf-8

from __future__ import annotations

import __future__
import ast
import asyncio
import base64
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
RECORDER_PATH = ROOT / "core" / "client" / "audio" / "recorder.py"


class FakeArray:
    shape = (480, 1)

    def __len__(self) -> int:
        return 480

    def __getitem__(self, _key):
        return self

    def tobytes(self) -> bytes:
        return b"audio"


class FakeNumpy:
    @staticmethod
    def concatenate(items):
        return items[0]

    @staticmethod
    def mean(data, axis):
        del axis
        return data


class TrackingFileManager:
    instances = []
    raise_on_write = False

    def __init__(self) -> None:
        self.file_path = None
        self.file_handle = None
        self.finish_calls = 0
        type(self).instances.append(self)

    def create(self, _channels: int, _time_start: float):
        self.file_path = Path("recording.mp3")
        self.file_handle = object()
        return self.file_path, self.file_handle

    def write(self, _data) -> None:
        if type(self).raise_on_write:
            raise RuntimeError("write failed")

    def finish(self):
        self.finish_calls += 1
        self.file_handle = None
        return self.file_path


class AudioMessage:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)


class FakeState:
    def __init__(self) -> None:
        self.queue_in = asyncio.Queue()
        self.audio_files = {}
        self.released_chunks = 0

    def release_audio_chunk(self) -> None:
        self.released_chunks += 1

    def consume_audio_input_overflow(self) -> bool:
        return False

    def register_audio_file(self, task_id: str, path: Path) -> None:
        self.audio_files[task_id] = path

    def pop_audio_file(self, task_id: str):
        return self.audio_files.pop(task_id, None)


class FakeWebSocketManager:
    def __init__(
        self,
        *,
        block_send: bool = False,
        block_close: bool = False,
    ) -> None:
        self.is_connected = True
        self.block_send = block_send
        self.block_close = block_close
        self.send_started = asyncio.Event()
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_calls = 0
        self.messages = []

    async def send(self, message) -> bool:
        self.messages.append(message)
        self.send_started.set()
        if self.block_send:
            await asyncio.Future()
        return True

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        if self.block_close:
            await self.allow_close.wait()


def load_recorder_class():
    tree = ast.parse(
        RECORDER_PATH.read_text(encoding="utf-8"),
        filename=str(RECORDER_PATH),
    )
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "CLIENT_RECORDER_FAILURE_DRAIN_SECONDS"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.ClassDef)
            and node.name == "AudioRecorder"
        )
    ]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    namespace = {
        "asyncio": asyncio,
        "AudioFileManager": TrackingFileManager,
        "AudioMessage": AudioMessage,
        "base64": base64,
        "Config": SimpleNamespace(
            context="",
            language="auto",
            mic_seg_duration=60,
            mic_seg_overlap=4,
            save_audio=True,
            threshold=0.0,
        ),
        "console": SimpleNamespace(print=lambda *_args, **_kwargs: None),
        "logger": SimpleNamespace(
            debug=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
            info=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
        ),
        "np": FakeNumpy,
        "uuid": uuid,
    }
    exec(
        compile(
            module,
            str(RECORDER_PATH),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        ),
        namespace,
    )
    return namespace["AudioRecorder"]


async def enqueue_recording(state: FakeState) -> None:
    await state.queue_in.put({"type": "begin", "time": 1.0})
    await state.queue_in.put(
        {"type": "data", "time": 2.0, "data": FakeArray()}
    )


class ClientRecorderCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        TrackingFileManager.instances = []
        TrackingFileManager.raise_on_write = False

    def test_cancellation_finalizes_once_and_clears_registration(self) -> None:
        recorder_class = load_recorder_class()

        async def scenario():
            state = FakeState()
            websocket = FakeWebSocketManager(block_send=True)
            recorder = recorder_class(SimpleNamespace(state=state, ws=websocket))
            await enqueue_recording(state)

            task = asyncio.create_task(recorder.record_and_send())
            await websocket.send_started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return state, websocket

        state, websocket = asyncio.run(scenario())
        manager = TrackingFileManager.instances[0]
        self.assertEqual(manager.finish_calls, 1)
        self.assertIsNone(manager.file_handle)
        self.assertEqual(state.audio_files, {})
        self.assertEqual(websocket.close_calls, 1)

    def test_second_cancellation_waits_for_close_and_queue_cleanup(self) -> None:
        recorder_class = load_recorder_class()

        async def scenario():
            state = FakeState()
            websocket = FakeWebSocketManager(
                block_send=True,
                block_close=True,
            )
            recorder = recorder_class(SimpleNamespace(state=state, ws=websocket))
            await enqueue_recording(state)

            task = asyncio.create_task(recorder.record_and_send())
            await websocket.send_started.wait()
            await state.queue_in.put(
                {"type": "data", "time": 3.0, "data": FakeArray()}
            )
            await state.queue_in.put({"type": "finish", "time": 4.0})
            task.cancel()
            await websocket.close_started.wait()
            task.cancel()
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            websocket.allow_close.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(state.queue_in.join(), timeout=0.1)
            return state, websocket

        state, websocket = asyncio.run(scenario())
        manager = TrackingFileManager.instances[0]
        self.assertEqual(manager.finish_calls, 1)
        self.assertIsNone(manager.file_handle)
        self.assertEqual(state.audio_files, {})
        self.assertEqual(state.queue_in.qsize(), 0)
        self.assertEqual(state.released_chunks, 2)
        self.assertEqual(websocket.close_calls, 1)

    def test_cancellation_during_final_only_first_send_closes_once(self) -> None:
        recorder_class = load_recorder_class()

        async def scenario():
            state = FakeState()
            websocket = FakeWebSocketManager(block_send=True)
            recorder = recorder_class(SimpleNamespace(state=state, ws=websocket))
            await state.queue_in.put({"type": "begin", "time": 1.0})
            await state.queue_in.put({"type": "finish", "time": 2.0})

            task = asyncio.create_task(recorder.record_and_send())
            await websocket.send_started.wait()
            self.assertEqual(len(websocket.messages), 1)
            self.assertTrue(websocket.messages[0].is_final)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return state, websocket

        state, websocket = asyncio.run(scenario())
        manager = TrackingFileManager.instances[0]
        self.assertEqual(manager.finish_calls, 1)
        self.assertEqual(state.audio_files, {})
        self.assertEqual(websocket.close_calls, 1)

    def test_write_error_finalizes_once_and_clears_registration(self) -> None:
        recorder_class = load_recorder_class()
        TrackingFileManager.raise_on_write = True

        async def scenario():
            state = FakeState()
            recorder = recorder_class(
                SimpleNamespace(state=state, ws=FakeWebSocketManager())
            )
            await enqueue_recording(state)
            await state.queue_in.put({"type": "finish", "time": 3.0})
            await recorder.record_and_send()
            return state

        state = asyncio.run(scenario())
        manager = TrackingFileManager.instances[0]
        self.assertEqual(manager.finish_calls, 1)
        self.assertIsNone(manager.file_handle)
        self.assertEqual(state.audio_files, {})

    def test_normal_finish_is_not_repeated_by_finally(self) -> None:
        recorder_class = load_recorder_class()

        async def scenario():
            state = FakeState()
            recorder = recorder_class(
                SimpleNamespace(state=state, ws=FakeWebSocketManager())
            )
            await enqueue_recording(state)
            await state.queue_in.put({"type": "finish", "time": 3.0})
            await recorder.record_and_send()
            return state

        state = asyncio.run(scenario())
        manager = TrackingFileManager.instances[0]
        self.assertEqual(manager.finish_calls, 1)
        self.assertIsNone(manager.file_handle)
        self.assertEqual(len(state.audio_files), 1)


if __name__ == "__main__":
    unittest.main()
