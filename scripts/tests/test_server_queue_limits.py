# coding: utf-8

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

SERVER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("rich", "websockets")
)


@unittest.skipUnless(
    SERVER_DEPS_AVAILABLE,
    "server runtime dependencies are not installed",
)
class ServerQueueLimitTest(unittest.TestCase):
    def test_cross_process_input_queue_is_bounded(self) -> None:
        from core.server.queue_limits import (
            INPUT_QUEUE_MAX_TASKS,
            MAX_TASK_AUDIO_BYTES,
            OUTPUT_QUEUE_MAX_RESULTS,
            WEBSOCKET_MAX_MESSAGE_BYTES,
            WEBSOCKET_MAX_QUEUED_MESSAGES,
            WORKER_BUFFER_MAX_TASKS,
        )
        from core.server.state import ServerState

        state = ServerState()
        try:
            self.assertEqual(state.queue_in._maxsize, INPUT_QUEUE_MAX_TASKS)
            self.assertEqual(state.queue_out._maxsize, OUTPUT_QUEUE_MAX_RESULTS)
            self.assertEqual(INPUT_QUEUE_MAX_TASKS, 8)
            self.assertEqual(WORKER_BUFFER_MAX_TASKS, 8)
            self.assertEqual(MAX_TASK_AUDIO_BYTES, 4_096_000)
            self.assertEqual(WEBSOCKET_MAX_MESSAGE_BYTES, 6 * 1024 * 1024)
            self.assertEqual(WEBSOCKET_MAX_QUEUED_MESSAGES, 1)
        finally:
            state.queue_in.close()
            state.queue_in.join_thread()
            state.queue_out.close()
            state.queue_out.join_thread()

    def test_websocket_queue_wait_does_not_block_event_loop(self) -> None:
        from core.protocol import AudioMessage
        from core.server.connection import ws_recv

        class FullThenOpenQueue:
            def __init__(self):
                self.items = []
                self.open = False
                self.attempts = 0

            def put_nowait(self, item):
                self.attempts += 1
                if not self.open:
                    raise queue.Full
                self.items.append(item)

        async def scenario() -> None:
            target_queue = FullThenOpenQueue()
            websocket = SimpleNamespace(
                id="socket",
                close_code=None,
                closed=False,
            )
            message = AudioMessage(
                task_id="task",
                source="file",
                data=base64.b64encode(b"pcm\0").decode("ascii"),
                is_final=True,
                time_start=1.0,
            )
            cache = ws_recv.AudioCache()
            app = SimpleNamespace(state=SimpleNamespace(queue_in=target_queue))

            with patch.object(
                asyncio,
                "to_thread",
                side_effect=AssertionError("WS queue wait must not spawn a thread"),
            ):
                started = time.monotonic()
                submit = asyncio.create_task(
                    ws_recv.message_handler(
                        websocket,
                        json.loads(message.to_json()),
                        cache,
                        app,
                    )
                )
                await asyncio.sleep(0.01)
                elapsed = time.monotonic() - started
                self.assertLess(elapsed, 0.1)
                self.assertFalse(submit.done())
                self.assertGreater(target_queue.attempts, 0)
                target_queue.open = True
                await asyncio.wait_for(submit, timeout=1.0)
            self.assertEqual(len(target_queue.items), 1)

        asyncio.run(scenario())

    def test_websocket_final_cache_is_split_to_maximum_task_size(self) -> None:
        from core.protocol import AudioMessage
        from core.server.connection import ws_recv
        from core.server.queue_limits import MAX_TASK_AUDIO_BYTES

        class RecordingQueue:
            def __init__(self):
                self.items = []

            def put_nowait(self, item):
                self.items.append(item)

        async def scenario() -> None:
            target_queue = RecordingQueue()
            websocket = SimpleNamespace(
                id="socket",
                close_code=None,
                closed=False,
            )
            message = AudioMessage(
                task_id="task",
                source="file",
                data="",
                is_final=True,
                time_start=1.0,
                seg_duration=60.0,
                seg_overlap=4.0,
            )
            cache = ws_recv.AudioCache()
            cache.chunks = b"x" * (MAX_TASK_AUDIO_BYTES + 64_000)
            cache.byte_count = len(cache.chunks)
            app = SimpleNamespace(state=SimpleNamespace(queue_in=target_queue))

            await ws_recv.message_handler(
                websocket,
                json.loads(message.to_json()),
                cache,
                app,
            )

            self.assertEqual(len(target_queue.items), 2)
            self.assertTrue(target_queue.items[-1].is_final)
            self.assertTrue(
                all(len(task.data) <= MAX_TASK_AUDIO_BYTES for task in target_queue.items)
            )

        asyncio.run(scenario())

    def test_websocket_metadata_geometry_and_stream_identity_are_strict(self) -> None:
        from core.constants import AudioFormat
        from core.server.connection import ws_recv

        def message(**overrides):
            payload = {
                "task_id": "task-1",
                "source": "mic",
                "data": base64.b64encode(b"\0" * 4).decode("ascii"),
                "is_final": False,
                "time_start": 1.0,
                "seg_duration": 60.0,
                "seg_overlap": 4.0,
                "context": "",
                "language": "auto",
            }
            payload.update(overrides)
            return payload

        valid, decoded = ws_recv.validate_audio_message(message())
        self.assertEqual(valid.task_id, "task-1")
        self.assertEqual(decoded, b"\0" * 4)

        invalid_cases = (
            (message(source="network"), "source"),
            (message(is_final=1), "is_final"),
            (message(task_id="task\nforged"), "control"),
            (message(context="x" * (ws_recv.MAX_CONTEXT_CHARS + 1)), "context"),
            (message(language="en\rforged"), "language"),
            (message(time_start=float("nan")), "time_start"),
            (message(time_start=10**400), "time_start"),
            (message(seg_duration=60.0, seg_overlap=4.1), "<= 64"),
            (
                message(
                    seg_duration=1 + 1 / AudioFormat.BYTES_PER_SECOND,
                    seg_overlap=0,
                ),
                "sample-aligned",
            ),
            (
                message(
                    seg_duration=1 / AudioFormat.SAMPLE_RATE,
                    seg_overlap=63.999,
                ),
                ">= 1s",
            ),
            (message(data="%%%"), "Base64"),
            (message(data=base64.b64encode(b"abc").decode("ascii")), "sample-aligned"),
        )
        for payload, expected in invalid_cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ws_recv.InvalidAudioMessage, expected):
                    ws_recv.validate_audio_message(payload)

        cache = ws_recv.AudioCache()
        cache.bind_stream("task-1", "mic")
        with self.assertRaisesRegex(ws_recv.InvalidAudioMessage, "changed"):
            cache.bind_stream("task-1", "file")
        with self.assertRaisesRegex(ws_recv.InvalidAudioMessage, "changed"):
            cache.bind_stream("task-2", "mic")

    def test_invalid_websocket_message_closes_with_policy_code(self) -> None:
        from core.server.connection import ws_recv

        class RecordingQueue:
            def __init__(self):
                self.items = []

            def put_nowait(self, item):
                self.items.append(item)

        class FakeWebSocket:
            def __init__(self):
                self.id = "socket"
                self.remote_address = ("127.0.0.1", 12345)
                self.close_code = None
                self.closed = False
                self._messages = iter(
                    [
                        json.dumps(
                            {
                                "task_id": "task",
                                "source": "invalid",
                                "data": "",
                                "is_final": False,
                                "time_start": 1.0,
                            }
                        )
                    ]
                )
                self.close_calls = []

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self._messages)
                except StopIteration:
                    raise StopAsyncIteration

            async def close(self, *, code, reason):
                self.close_calls.append((code, reason))
                self.close_code = code
                self.closed = True

        async def scenario() -> None:
            target_queue = RecordingQueue()
            state = SimpleNamespace(
                queue_in=target_queue,
                sockets={},
                sockets_id=[],
            )
            websocket = FakeWebSocket()
            await ws_recv.ws_recv(websocket, SimpleNamespace(state=state))

            self.assertEqual(
                websocket.close_calls,
                [(1008, "Invalid audio message")],
            )
            self.assertEqual(target_queue.items, [])
            self.assertEqual(state.sockets, {})
            self.assertEqual(state.sockets_id, [])

        asyncio.run(scenario())

    def test_websocket_connection_cap_rejects_immediately_and_releases_on_cancel(self) -> None:
        from core.server.connection import ws_recv

        class EmptyQueue:
            def put_nowait(self, item):
                del item

        class FakeWebSocket:
            def __init__(self, socket_id: str, *, block: bool):
                self.id = socket_id
                self.remote_address = ("127.0.0.1", 12345)
                self.close_code = None
                self.closed = False
                self.close_calls = []
                self.block = block
                self.waiter = asyncio.Event()

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.block:
                    raise StopAsyncIteration
                await self.waiter.wait()
                raise StopAsyncIteration

            async def close(self, *, code, reason):
                self.close_calls.append((code, reason))
                self.close_code = code
                self.closed = True

        async def scenario() -> None:
            limiter = ws_recv.WebSocketConnectionLimiter(1)
            state = SimpleNamespace(
                queue_in=EmptyQueue(),
                sockets={},
                sockets_id=[],
            )
            app = SimpleNamespace(state=state)
            first = FakeWebSocket("first", block=True)
            first_task = asyncio.create_task(
                ws_recv.ws_recv(first, app, admission=limiter)
            )
            for _attempt in range(100):
                if limiter.active == 1 and "first" in state.sockets:
                    break
                await asyncio.sleep(0)
            self.assertEqual(limiter.active, 1)

            excess = FakeWebSocket("excess", block=False)
            await ws_recv.ws_recv(excess, app, admission=limiter)
            self.assertEqual(
                excess.close_calls,
                [(1013, "Server connection limit reached")],
            )
            self.assertNotIn("excess", state.sockets)
            self.assertEqual(limiter.active, 1)

            first_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first_task
            self.assertEqual(limiter.active, 0)
            self.assertEqual(state.sockets, {})
            self.assertEqual(state.sockets_id, [])

            replacement = FakeWebSocket("replacement", block=False)
            await ws_recv.ws_recv(replacement, app, admission=limiter)
            self.assertEqual(replacement.close_calls, [])
            self.assertEqual(limiter.active, 0)

        asyncio.run(scenario())

    def test_websocket_connection_cap_aborts_peer_that_ignores_close(self) -> None:
        from core.server.connection import ws_recv

        class RecordingTransport:
            def __init__(self) -> None:
                self.abort_calls = 0

            def abort(self) -> None:
                self.abort_calls += 1

        class HangingWebSocket:
            def __init__(self) -> None:
                self.transport = RecordingTransport()
                self.close_calls = []

            async def close(self, *, code, reason) -> None:
                self.close_calls.append((code, reason))
                await asyncio.Event().wait()

        async def scenario() -> None:
            limiter = ws_recv.WebSocketConnectionLimiter(1)
            self.assertTrue(limiter.try_acquire())
            websocket = HangingWebSocket()
            with patch.object(
                ws_recv,
                "WEBSOCKET_CLOSE_TIMEOUT_SECONDS",
                0.01,
            ):
                await asyncio.wait_for(
                    ws_recv.ws_recv(
                        websocket,
                        SimpleNamespace(),
                        admission=limiter,
                    ),
                    timeout=0.5,
                )

            self.assertEqual(
                websocket.close_calls,
                [(1013, "Server connection limit reached")],
            )
            self.assertEqual(websocket.transport.abort_calls, 1)
            self.assertEqual(limiter.active, 1)
            limiter.release()

        asyncio.run(scenario())

    def test_websocket_task_audio_ceiling_rejects_and_resets_only_stream(self) -> None:
        from config_server import ServerConfig as Config
        from core.constants import AudioFormat
        from core.server.connection import ws_recv
        from core.server.task_control import WEBSOCKET_TASK_LIMIT_COMMAND

        class RecordingQueue:
            def __init__(self) -> None:
                self.items = []

            def put_nowait(self, item) -> None:
                self.items.append(item)

        def message(data: bytes, *, task_id="shared", is_final=False):
            return {
                "task_id": task_id,
                "source": "file",
                "data": base64.b64encode(data).decode("ascii"),
                "is_final": is_final,
                "time_start": 1.0,
                "seg_duration": 1.0,
                "seg_overlap": 0.0,
                "context": "",
                "language": "auto",
            }

        async def scenario() -> None:
            target_queue = RecordingQueue()
            app = SimpleNamespace(state=SimpleNamespace(queue_in=target_queue))
            first_socket = SimpleNamespace(
                id="socket-a",
                close_code=None,
                closed=False,
            )
            first_cache = ws_recv.AudioCache()
            initial = b"a" * int(0.75 * AudioFormat.BYTES_PER_SECOND)
            overflow = b"b" * int(0.5 * AudioFormat.BYTES_PER_SECOND)

            with patch.object(
                Config,
                "max_websocket_task_seconds",
                1.0,
                create=True,
            ):
                await ws_recv.message_handler(
                    first_socket,
                    message(initial),
                    first_cache,
                    app,
                )
                self.assertEqual(first_cache.byte_count, len(initial))
                self.assertEqual(target_queue.items, [])

                await ws_recv.message_handler(
                    first_socket,
                    message(overflow),
                    first_cache,
                    app,
                )
                self.assertEqual(first_cache.byte_count, 0)
                self.assertEqual(first_cache.task_id, "shared")
                self.assertTrue(first_cache.rejected)
                self.assertEqual(len(target_queue.items), 1)
                control = target_queue.items[0]
                self.assertEqual(control.command, WEBSOCKET_TASK_LIMIT_COMMAND)
                self.assertEqual(
                    (control.socket_id, control.task_id),
                    ("socket-a", "shared"),
                )

                # Continued audio for the rejected identity is discarded and
                # cannot resurrect the worker session. Its final marker only
                # releases the connection-local identity for the next task.
                await ws_recv.message_handler(
                    first_socket,
                    message(b"c" * 4),
                    first_cache,
                    app,
                )
                self.assertEqual(len(target_queue.items), 1)
                self.assertTrue(first_cache.rejected)
                await ws_recv.message_handler(
                    first_socket,
                    message(b"", is_final=True),
                    first_cache,
                    app,
                )
                self.assertEqual(len(target_queue.items), 1)
                self.assertIsNone(first_cache.task_id)
                self.assertFalse(first_cache.rejected)

                await ws_recv.message_handler(
                    first_socket,
                    message(b"\0" * 4, task_id="next", is_final=True),
                    first_cache,
                    app,
                )
                self.assertEqual(len(target_queue.items), 2)
                self.assertEqual(target_queue.items[1].task_id, "next")

                # A colliding task ID on another socket remains independent.
                other_cache = ws_recv.AudioCache()
                other_socket = SimpleNamespace(
                    id="socket-b",
                    close_code=None,
                    closed=False,
                )
                await ws_recv.message_handler(
                    other_socket,
                    message(b"\0" * 4, is_final=True),
                    other_cache,
                    app,
                )
                self.assertEqual(len(target_queue.items), 3)
                self.assertEqual(target_queue.items[2].type, "file")
                self.assertEqual(
                    (target_queue.items[2].socket_id, target_queue.items[2].task_id),
                    ("socket-b", "shared"),
                )

        asyncio.run(scenario())

    def test_websocket_task_audio_ceiling_allows_exact_limit_and_next_task(self) -> None:
        from config_server import ServerConfig as Config
        from core.constants import AudioFormat
        from core.server.connection import ws_recv

        class RecordingQueue:
            def __init__(self) -> None:
                self.items = []

            def put_nowait(self, item) -> None:
                self.items.append(item)

        def final_message(task_id: str, data: bytes):
            return {
                "task_id": task_id,
                "source": "file",
                "data": base64.b64encode(data).decode("ascii"),
                "is_final": True,
                "time_start": 1.0,
                "seg_duration": 1.0,
                "seg_overlap": 0.0,
                "context": "",
                "language": "auto",
            }

        async def scenario() -> None:
            target_queue = RecordingQueue()
            app = SimpleNamespace(state=SimpleNamespace(queue_in=target_queue))
            websocket = SimpleNamespace(
                id="socket",
                close_code=None,
                closed=False,
            )
            cache = ws_recv.AudioCache()
            exact = b"\0" * AudioFormat.BYTES_PER_SECOND
            with patch.object(
                Config,
                "max_websocket_task_seconds",
                1.0,
                create=True,
            ):
                await ws_recv.message_handler(
                    websocket,
                    final_message("first", exact),
                    cache,
                    app,
                )
                await ws_recv.message_handler(
                    websocket,
                    final_message("second", b"\0" * 4),
                    cache,
                    app,
                )

            self.assertEqual([task.task_id for task in target_queue.items], ["first", "second"])
            self.assertTrue(all(task.is_final for task in target_queue.items))
            self.assertEqual(cache.byte_count, 0)
            self.assertIsNone(cache.task_id)

        asyncio.run(scenario())

    def test_websocket_server_uses_bounded_close_timeout(self) -> None:
        from core.server.connection import server_manager
        from core.server.queue_limits import WEBSOCKET_CLOSE_TIMEOUT_SECONDS

        observed = {}

        class ServerContext:
            async def __aenter__(self):
                return SimpleNamespace(close=MagicMock())

            async def __aexit__(self, *_args):
                return False

        def serve(*args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            return ServerContext()

        async def send_results(_app) -> None:
            return None

        async def scenario() -> None:
            app = SimpleNamespace(loop=asyncio.get_running_loop())
            manager = server_manager.SocketManager(app)
            with (
                patch.object(manager, "_check_port", return_value=True),
                patch.object(server_manager.websockets, "serve", side_effect=serve),
                patch.object(server_manager, "ws_send", new=send_results),
            ):
                await manager.start()

        asyncio.run(scenario())
        self.assertEqual(
            observed["kwargs"]["close_timeout"],
            WEBSOCKET_CLOSE_TIMEOUT_SECONDS,
        )

    def test_slow_websocket_does_not_block_http_or_fast_peer_results(self) -> None:
        from core.server.connection import result_dispatcher
        from core.server.schema import Result
        from fork_server.http_api import ws_send_with_http

        class SlowWebSocket:
            def __init__(self) -> None:
                self.id = "slow"
                self.send_started = asyncio.Event()
                self.close_calls = []

            async def send(self, _payload) -> None:
                self.send_started.set()
                await asyncio.Event().wait()

            async def close(self, *, code, reason) -> None:
                self.close_calls.append((code, reason))

        class FastWebSocket:
            def __init__(self) -> None:
                self.id = "fast"
                self.sent = []

            async def send(self, payload) -> None:
                self.sent.append(json.loads(payload))

        async def scenario() -> None:
            slow = SlowWebSocket()
            fast = FastWebSocket()
            queue_out = queue.Queue()
            slow_result = Result(
                task_id="slow-task",
                socket_id="slow",
                type="test",
                text="slow",
            )
            http_result = Result(
                task_id="http-task",
                socket_id="http:http-task",
                type="test",
                text="http",
                is_final=True,
            )
            fast_result = Result(
                task_id="fast-task",
                socket_id="fast",
                type="test",
                text="fast",
                is_final=True,
            )
            for item in (slow_result, http_result, fast_result, None):
                queue_out.put_nowait(item)
            state = SimpleNamespace(
                queue_out=queue_out,
                sockets={"slow": slow, "fast": fast},
                sockets_id=["slow", "fast"],
            )
            resolved = []

            def try_resolve(result) -> bool:
                if result.socket_id == "http:http-task":
                    resolved.append(result)
                    return True
                return False

            with (
                patch.object(
                    ws_send_with_http.task_router,
                    "try_resolve",
                    side_effect=try_resolve,
                ),
                patch.object(
                    result_dispatcher,
                    "WEBSOCKET_SEND_TIMEOUT_SECONDS",
                    0.02,
                ),
                patch.object(
                    result_dispatcher,
                    "WEBSOCKET_CLOSE_TIMEOUT_SECONDS",
                    0.05,
                ),
            ):
                await asyncio.wait_for(
                    ws_send_with_http.ws_send_with_http(
                        SimpleNamespace(state=state)
                    ),
                    timeout=1.0,
                )

            self.assertEqual(resolved, [http_result])
            self.assertEqual(
                [payload["text"] for payload in fast.sent],
                ["fast"],
            )
            self.assertEqual(
                slow.close_calls,
                [(1013, "Slow result consumer")],
            )
            self.assertNotIn("slow", state.sockets)
            self.assertNotIn("slow", state.sockets_id)
            self.assertIn("fast", state.sockets)

        asyncio.run(scenario())

    def test_result_dispatch_preserves_cross_task_finals_and_coalesces_snapshots(self) -> None:
        from core.server.connection.result_dispatcher import (
            WebSocketResultDispatcher,
        )

        class GatedWebSocket:
            def __init__(self) -> None:
                self.sent = []
                self.first_started = asyncio.Event()
                self.release_first = asyncio.Event()

            async def send(self, payload) -> None:
                self.sent.append(payload)
                if len(self.sent) == 1:
                    self.first_started.set()
                    await self.release_first.wait()

        async def scenario() -> None:
            websocket = GatedWebSocket()
            state = SimpleNamespace(sockets={}, sockets_id=[])
            dispatcher = WebSocketResultDispatcher(state, MagicMock())
            dispatcher.submit(
                websocket,
                "active",
                socket_id="socket",
                task_id="active-task",
                is_final=False,
            )
            await asyncio.wait_for(websocket.first_started.wait(), timeout=0.5)
            dispatcher.submit(
                websocket,
                "A-final",
                socket_id="socket",
                task_id="task-A",
                is_final=True,
            )
            dispatcher.submit(
                websocket,
                "B-intermediate",
                socket_id="socket",
                task_id="task-B",
                is_final=False,
            )
            dispatcher.submit(
                websocket,
                "B-final",
                socket_id="socket",
                task_id="task-B",
                is_final=True,
            )

            channel = dispatcher._channels["socket"]
            self.assertEqual(list(channel.pending), ["task-A", "task-B"])
            self.assertTrue(channel.pending["task-A"].is_final)
            self.assertTrue(channel.pending["task-B"].is_final)
            websocket.release_first.set()
            for _attempt in range(100):
                if dispatcher.active_peer_count == 0:
                    break
                await asyncio.sleep(0)
            self.assertEqual(
                websocket.sent,
                ["active", "A-final", "B-final"],
            )
            await dispatcher.aclose()

        asyncio.run(scenario())

    def test_result_dispatch_pending_overflow_is_bounded_and_evicts_peer(self) -> None:
        from core.server.connection.result_dispatcher import (
            WebSocketResultDispatcher,
        )
        from core.server.queue_limits import MAX_PENDING_RESULTS_PER_WEBSOCKET

        class SlowWebSocket:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.close_calls = []

            async def send(self, _payload) -> None:
                self.started.set()
                await asyncio.Event().wait()

            async def close(self, *, code, reason) -> None:
                self.close_calls.append((code, reason))

        async def scenario() -> None:
            websocket = SlowWebSocket()
            state = SimpleNamespace(
                sockets={"socket": websocket},
                sockets_id=["socket"],
            )
            dispatcher = WebSocketResultDispatcher(state, MagicMock())
            dispatcher.submit(
                websocket,
                "active",
                socket_id="socket",
                task_id="active",
                is_final=False,
            )
            await asyncio.wait_for(websocket.started.wait(), timeout=0.5)
            for index in range(MAX_PENDING_RESULTS_PER_WEBSOCKET):
                dispatcher.submit(
                    websocket,
                    f"pending-{index}",
                    socket_id="socket",
                    task_id=f"task-{index}",
                    is_final=True,
                )
            self.assertEqual(
                len(dispatcher._channels["socket"].pending),
                MAX_PENDING_RESULTS_PER_WEBSOCKET,
            )
            dispatcher.submit(
                websocket,
                "overflow",
                socket_id="socket",
                task_id="overflow",
                is_final=True,
            )
            for _attempt in range(100):
                if dispatcher.active_peer_count == 0:
                    break
                await asyncio.sleep(0)

            self.assertEqual(dispatcher.active_peer_count, 0)
            self.assertEqual(
                websocket.close_calls,
                [(1013, "Slow result consumer")],
            )
            self.assertEqual(state.sockets, {})
            self.assertEqual(state.sockets_id, [])
            await dispatcher.aclose()

        asyncio.run(scenario())

    def test_result_dispatch_aborts_peer_that_ignores_bounded_close(self) -> None:
        from core.server.connection import result_dispatcher

        class RecordingTransport:
            def __init__(self) -> None:
                self.abort_calls = 0

            def abort(self) -> None:
                self.abort_calls += 1

        class HangingWebSocket:
            def __init__(self) -> None:
                self.transport = RecordingTransport()

            async def close(self, *, code, reason) -> None:
                del code, reason
                await asyncio.Event().wait()

        async def scenario() -> None:
            websocket = HangingWebSocket()
            state = SimpleNamespace(
                sockets={"socket": websocket},
                sockets_id=["socket"],
            )
            dispatcher = result_dispatcher.WebSocketResultDispatcher(
                state,
                MagicMock(),
            )
            with patch.object(
                result_dispatcher,
                "WEBSOCKET_CLOSE_TIMEOUT_SECONDS",
                0.01,
            ):
                await asyncio.wait_for(
                    dispatcher._evict_and_close("socket", websocket),
                    timeout=0.5,
                )

            self.assertEqual(websocket.transport.abort_calls, 1)
            self.assertEqual(state.sockets, {})
            self.assertEqual(state.sockets_id, [])

        asyncio.run(scenario())

    def test_async_result_reader_exits_after_close(self) -> None:
        from core.server.connection.result_dispatcher import AsyncResultQueueReader

        async def scenario() -> None:
            with patch.object(AsyncResultQueueReader, "_POLL_SECONDS", 0.01):
                reader = AsyncResultQueueReader(queue.Queue())
                self.assertTrue(reader._thread.is_alive())
                await reader.aclose()
                self.assertFalse(reader._thread.is_alive())

        asyncio.run(scenario())

    def test_socket_port_preflight_uses_resolved_ipv6_family(self) -> None:
        import socket

        from config_server import ServerConfig as Config
        from core.server.connection.server_manager import SocketManager

        listener = MagicMock()
        listener.__enter__.return_value = listener
        listener.__exit__.return_value = False
        with (
            patch.object(Config, "addr", "::1"),
            patch.object(Config, "port", "6026"),
            patch(
                "socket.getaddrinfo",
                return_value=[
                    (
                        socket.AF_INET6,
                        socket.SOCK_STREAM,
                        socket.IPPROTO_TCP,
                        "",
                        ("::1", 6026, 0, 0),
                    )
                ],
            ) as resolve,
            patch("socket.socket", return_value=listener) as socket_factory,
        ):
            self.assertTrue(SocketManager(SimpleNamespace())._check_port())

        resolve.assert_called_once_with(
            "::1",
            6026,
            type=socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
        socket_factory.assert_called_once_with(
            socket.AF_INET6,
            socket.SOCK_STREAM,
            socket.IPPROTO_TCP,
        )
        listener.bind.assert_called_once_with(("::1", 6026, 0, 0))


if __name__ == "__main__":
    unittest.main()
