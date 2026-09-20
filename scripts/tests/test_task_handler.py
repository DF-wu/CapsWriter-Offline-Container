# coding: utf-8

from __future__ import annotations

import ast
import queue
import threading
import time
import unittest
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[2]
TASK_HANDLER_PATH = ROOT / "core" / "server" / "worker" / "task_handler.py"


class FakeLogger:
    def __init__(self) -> None:
        self.error_calls = []

    def debug(self, *_args, **_kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        self.error_calls.append((args, kwargs))

    def info(self, *_args, **_kwargs) -> None:
        pass


class FakeGpuBoostManager:
    def __init__(self, state) -> None:
        self.state = state

    def check_idle(self) -> None:
        pass

    def handle_command(self, _task) -> None:
        pass


class FakeState:
    def __init__(self) -> None:
        self.sessions = {}

    def get_session(self, task_id, socket_id="", source=""):
        key = (socket_id, task_id)
        if key not in self.sessions:
            self.sessions[key] = SimpleNamespace(
                result=SimpleNamespace(socket_id=socket_id, type=source)
            )
        return self.sessions[key]

    def cleanup_sessions(self, sockets_id) -> int:
        stale = [
            key
            for key, session in self.sessions.items()
            if session.result.socket_id not in sockets_id
        ]
        for key in stale:
            del self.sessions[key]
        return len(stale)


class ScriptedQueue:
    def __init__(self, items=()) -> None:
        self.items = deque(items)
        self.blocking_timeouts = []
        self.nonblocking_calls = 0

    def _pop(self):
        if not self.items:
            raise queue.Empty
        return self.items.popleft()

    def get(self, timeout=None):
        self.blocking_timeouts.append(timeout)
        return self._pop()

    def get_nowait(self):
        self.nonblocking_calls += 1
        return self._pop()

    def put(self, item) -> None:
        self.items.append(item)


class ContinuousProducerQueue:
    def __init__(self) -> None:
        self.produced = 0
        self.blocking_timeouts = []
        self.nonblocking_calls = 0
        self.shutdown_requested = False

    def _next(self):
        if self.shutdown_requested:
            return None
        task = make_task("continuous", self.produced)
        self.produced += 1
        return task

    def get(self, timeout=None):
        self.blocking_timeouts.append(timeout)
        return self._next()

    def get_nowait(self):
        self.nonblocking_calls += 1
        return self._next()


class ActiveInference:
    def __init__(self) -> None:
        self.values = [0.0, 0.0]
        self.lock = threading.Lock()

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def get_lock(self):
        return self.lock


@dataclass
class WorkerResult:
    task_id: str
    socket_id: str
    type: str
    duration: float = 0.0
    time_start: float = 0.0
    time_submit: float = 0.0
    time_complete: float = 0.0
    text: str = ""
    text_accu: str = ""
    tokens: list[str] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    is_final: bool = False
    error_code: str | None = None
    error_message: str | None = None


def make_task(task_id: str, sequence: int, *, socket_id: str = "socket"):
    return SimpleNamespace(
        task_id=task_id,
        sequence=sequence,
        socket_id=socket_id,
        type="file",
        time_start=10.0 + sequence,
        time_submit=20.0 + sequence,
    )


def load_task_handler_namespace() -> dict:
    source = TASK_HANDLER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TASK_HANDLER_PATH))
    keep_constants = {
        "MAX_QUEUE_DRAIN_PER_CYCLE",
        "IDLE_QUEUE_TIMEOUT_SECONDS",
        "WORKER_TASK_ERROR_CODE",
        "WORKER_TASK_ERROR_MESSAGE",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & keep_constants:
                body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in {
            "TaskBuffer",
            "TaskHandler",
        }:
            body.append(node)

    namespace = {
        "deque": deque,
        "GpuBoostManager": FakeGpuBoostManager,
        "ListProxy": object,
        "logger": FakeLogger(),
        "OrderedDict": OrderedDict,
        "queue": queue,
        "Queue": object,
        "Result": WorkerResult,
        "TaskPipeline": object,
        "time": time,
        "WorkerState": FakeState,
        "WORKER_BUFFER_MAX_TASKS": 8,
        "WEBSOCKET_TASK_LIMIT_COMMAND": "reject_websocket_task_audio_limit",
        "WEBSOCKET_TASK_LIMIT_ERROR_CODE": "websocket_task_audio_limit_exceeded",
        "WEBSOCKET_TASK_LIMIT_ERROR_MESSAGE": (
            "WebSocket task audio exceeds the configured server limit."
        ),
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(TASK_HANDLER_PATH), "exec"), namespace)
    return namespace


def build_handler(
    namespace: dict,
    queue_in,
    *,
    sockets_id=None,
    active_inference=None,
):
    state = FakeState()
    handler = namespace["TaskHandler"](
        queue_in,
        ScriptedQueue(),
        ["socket"] if sockets_id is None else sockets_id,
        state,
        active_inference=active_inference,
    )
    return handler, state


class TaskBufferTest(unittest.TestCase):
    def test_same_task_id_on_two_sockets_has_isolated_sessions_and_fairness(self) -> None:
        namespace = load_task_handler_namespace()
        state = FakeState()
        buffer = namespace["TaskBuffer"](state)
        for task in (
            make_task("shared", 1, socket_id="socket-a"),
            make_task("shared", 2, socket_id="socket-a"),
            make_task("shared", 1, socket_id="socket-b"),
        ):
            buffer.enqueue(task)

        self.assertEqual(
            set(state.sessions),
            {("socket-a", "shared"), ("socket-b", "shared")},
        )
        self.assertIsNot(
            state.sessions[("socket-a", "shared")],
            state.sessions[("socket-b", "shared")],
        )
        self.assertEqual(
            state.sessions[("socket-a", "shared")].result.socket_id,
            "socket-a",
        )
        self.assertEqual(
            state.sessions[("socket-b", "shared")].result.socket_id,
            "socket-b",
        )

        popped = [buffer.pop(), buffer.pop(), buffer.pop()]
        self.assertEqual(
            [(task.socket_id, task.sequence) for task in popped],
            [("socket-a", 1), ("socket-b", 1), ("socket-a", 2)],
        )

    def test_buffer_has_a_hard_task_capacity(self) -> None:
        namespace = load_task_handler_namespace()
        buffer = namespace["TaskBuffer"](FakeState(), max_tasks=2)
        buffer.enqueue(make_task("a", 1))
        buffer.enqueue(make_task("b", 1))

        with self.assertRaises(queue.Full):
            buffer.enqueue(make_task("c", 1))

        self.assertEqual(len(buffer), 2)
        self.assertEqual(buffer.remaining_capacity, 0)
        buffer.pop()
        self.assertEqual(buffer.remaining_capacity, 1)

    def test_oldest_session_round_robin_preserves_session_fifo(self) -> None:
        namespace = load_task_handler_namespace()
        buffer = namespace["TaskBuffer"](FakeState())
        for task in (
            make_task("a", 1),
            make_task("a", 2),
            make_task("b", 1),
            make_task("b", 2),
            make_task("c", 1),
        ):
            buffer.enqueue(task)

        popped = []
        while not buffer.is_empty:
            task = buffer.pop()
            popped.append((task.task_id, task.sequence))

        self.assertEqual(
            popped,
            [("a", 1), ("b", 1), ("c", 1), ("a", 2), ("b", 2)],
        )

    def test_busy_session_cannot_refresh_its_age_and_starve_others(self) -> None:
        namespace = load_task_handler_namespace()
        buffer = namespace["TaskBuffer"](FakeState())
        buffer.enqueue(make_task("busy", 1))
        buffer.enqueue(make_task("busy", 2))
        buffer.enqueue(make_task("waiting", 1))

        self.assertEqual(buffer.pop().task_id, "busy")
        buffer.enqueue(make_task("busy", 3))

        self.assertEqual(buffer.pop().task_id, "waiting")
        self.assertEqual(buffer.pop().sequence, 2)
        self.assertEqual(buffer.pop().sequence, 3)

    def test_cleanup_keeps_round_robin_order_for_live_sessions(self) -> None:
        namespace = load_task_handler_namespace()
        state = FakeState()
        buffer = namespace["TaskBuffer"](state)
        for task in (
            make_task("a", 1),
            make_task("a", 2),
            make_task("b", 1),
            make_task("c", 1),
        ):
            buffer.enqueue(task)

        self.assertEqual(buffer.pop().task_id, "a")
        del state.sessions[("socket", "b")]
        buffer.cleanup_tasks()

        self.assertEqual(buffer.pop().task_id, "c")
        self.assertEqual(buffer.pop().sequence, 2)
        self.assertTrue(buffer.is_empty)


class TaskHandlerDrainTest(unittest.TestCase):
    def test_drain_never_exceeds_worker_buffer_capacity(self) -> None:
        namespace = load_task_handler_namespace()
        queue_in = ContinuousProducerQueue()
        handler, _state = build_handler(namespace, queue_in)
        handler.buffer = namespace["TaskBuffer"](
            handler.state,
            max_tasks=2,
        )

        self.assertTrue(handler.drain_queue())

        self.assertEqual(len(handler.buffer), 2)
        self.assertEqual(queue_in.produced, 2)

    def test_expired_http_task_never_enters_worker_buffer(self) -> None:
        namespace = load_task_handler_namespace()
        task = make_task("expired", 1)
        task.deadline_monotonic = 0.0
        queue_in = ScriptedQueue([task])
        handler, state = build_handler(namespace, queue_in)

        self.assertTrue(handler.drain_queue())

        self.assertTrue(handler.buffer.is_empty)
        self.assertEqual(state.sessions, {})

    def test_dispatch_rechecks_deadline_immediately_before_pipeline(self) -> None:
        namespace = load_task_handler_namespace()
        handler, state = build_handler(namespace, ScriptedQueue())
        task = make_task("expired-buffered", 1)
        task.deadline_monotonic = time.monotonic() + 60
        handler.buffer.enqueue(task)
        buffered = handler.buffer.pop()
        buffered.deadline_monotonic = 0.0
        process = Mock()
        handler.pipeline = SimpleNamespace(process=process, aligner=None)

        handler.dispatch_task(buffered)

        process.assert_not_called()
        self.assertEqual(state.sessions, {})
        self.assertEqual(list(handler.queue_out.items), [])

    def test_continuous_producer_drain_is_bounded(self) -> None:
        namespace = load_task_handler_namespace()
        namespace["MAX_QUEUE_DRAIN_PER_CYCLE"] = 4
        queue_in = ContinuousProducerQueue()
        handler, _state = build_handler(namespace, queue_in)

        self.assertTrue(handler.drain_queue())

        self.assertEqual(queue_in.produced, 4)
        self.assertEqual(
            queue_in.blocking_timeouts,
            [namespace["IDLE_QUEUE_TIMEOUT_SECONDS"]],
        )
        self.assertEqual(queue_in.nonblocking_calls, 3)
        self.assertEqual(
            len(handler.buffer._buffers[("socket", "continuous")]),
            4,
        )

    def test_buffered_work_never_waits_for_an_input_gap(self) -> None:
        namespace = load_task_handler_namespace()
        queue_in = ScriptedQueue()
        handler, _state = build_handler(namespace, queue_in)
        handler.buffer.enqueue(make_task("ready", 1))
        handler.cleanup_engines = Mock()

        self.assertTrue(handler.drain_queue())

        self.assertEqual(queue_in.blocking_timeouts, [])
        self.assertEqual(queue_in.nonblocking_calls, 1)
        handler.cleanup_engines.assert_not_called()

    def test_idle_timeout_runs_engine_cleanup_once(self) -> None:
        namespace = load_task_handler_namespace()
        queue_in = ScriptedQueue()
        handler, _state = build_handler(namespace, queue_in)
        handler.cleanup_engines = Mock()

        self.assertTrue(handler.drain_queue())

        self.assertEqual(
            queue_in.blocking_timeouts,
            [namespace["IDLE_QUEUE_TIMEOUT_SECONDS"]],
        )
        handler.cleanup_engines.assert_called_once_with()

    def test_idle_timeout_reaps_disconnected_non_final_session(self) -> None:
        namespace = load_task_handler_namespace()
        queue_in = ScriptedQueue()
        handler, state = build_handler(
            namespace,
            queue_in,
            sockets_id=["live-socket"],
        )
        state.get_session("stale-task", "gone-socket", "mic")
        state.get_session("live-task", "live-socket", "mic")
        handler.cleanup_engines = Mock()

        self.assertTrue(handler.drain_queue())

        self.assertNotIn(("gone-socket", "stale-task"), state.sessions)
        self.assertIn(("live-socket", "live-task"), state.sessions)
        handler.cleanup_engines.assert_called_once_with()

    def test_sentinel_stays_queued_when_beyond_current_batch(self) -> None:
        namespace = load_task_handler_namespace()
        namespace["MAX_QUEUE_DRAIN_PER_CYCLE"] = 2
        sentinel = None
        queue_in = ScriptedQueue(
            [make_task("a", 1), make_task("b", 1), sentinel]
        )
        handler, _state = build_handler(namespace, queue_in)

        self.assertTrue(handler.drain_queue())
        self.assertEqual(list(queue_in.items), [sentinel])

        self.assertFalse(handler.drain_queue())
        self.assertEqual(list(queue_in.items), [])
        self.assertEqual(
            set(handler.buffer._buffers),
            {("socket", "a"), ("socket", "b")},
        )

    def test_loop_dispatches_even_while_producer_never_goes_idle(self) -> None:
        namespace = load_task_handler_namespace()
        namespace["MAX_QUEUE_DRAIN_PER_CYCLE"] = 3
        queue_in = ContinuousProducerQueue()
        handler, _state = build_handler(namespace, queue_in)
        handled = []

        def handle_audio(task) -> None:
            handled.append(task.sequence)
            queue_in.shutdown_requested = True

        handler.handle_audio_task = handle_audio
        handler.cleanup = Mock()

        handler.loop()

        self.assertEqual(handled, [0])
        self.assertEqual(queue_in.produced, 3)
        handler.cleanup.assert_called_once_with()


class TaskHandlerErrorEnvelopeTest(unittest.TestCase):
    def test_websocket_limit_control_clears_only_composite_session(self) -> None:
        namespace = load_task_handler_namespace()
        handler, state = build_handler(
            namespace,
            ScriptedQueue(),
            sockets_id=["socket-a", "socket-b"],
        )
        for task in (
            make_task("shared", 2, socket_id="socket-a"),
            make_task("other", 1, socket_id="socket-a"),
            make_task("shared", 1, socket_id="socket-b"),
        ):
            handler.buffer.enqueue(task)

        control = make_task("shared", 3, socket_id="socket-a")
        control.type = "cmd"
        control.command = namespace["WEBSOCKET_TASK_LIMIT_COMMAND"]
        handler.handle_command_task(control)

        self.assertNotIn(("socket-a", "shared"), state.sessions)
        self.assertNotIn(("socket-a", "shared"), handler.buffer._buffers)
        self.assertIn(("socket-a", "other"), state.sessions)
        self.assertIn(("socket-a", "other"), handler.buffer._buffers)
        self.assertIn(("socket-b", "shared"), state.sessions)
        self.assertIn(("socket-b", "shared"), handler.buffer._buffers)
        self.assertEqual(len(handler.buffer), 2)

        result = handler.queue_out.items[-1]
        self.assertTrue(result.is_final)
        self.assertEqual(result.task_id, "shared")
        self.assertEqual(result.socket_id, "socket-a")
        self.assertEqual(
            result.error_code,
            namespace["WEBSOCKET_TASK_LIMIT_ERROR_CODE"],
        )
        self.assertEqual(
            result.error_message,
            namespace["WEBSOCKET_TASK_LIMIT_ERROR_MESSAGE"],
        )

    def test_pipeline_error_purges_only_failed_composite_session_segments(self) -> None:
        namespace = load_task_handler_namespace()
        handler, state = build_handler(
            namespace,
            ScriptedQueue(),
            sockets_id=["socket-a", "socket-b"],
        )
        failed = make_task("shared", 1, socket_id="socket-a")
        failed_tail = make_task("shared", 2, socket_id="socket-a")
        other_socket = make_task("shared", 1, socket_id="socket-b")
        handler.buffer.enqueue(failed_tail)
        handler.buffer.enqueue(other_socket)

        class FailingPipeline:
            def process(self, _task):
                raise RuntimeError("private failure")

        handler.pipeline = FailingPipeline()
        handler.handle_audio_task(failed)

        self.assertNotIn(("socket-a", "shared"), handler.buffer._buffers)
        self.assertNotIn(("socket-a", "shared"), state.sessions)
        self.assertIn(("socket-b", "shared"), handler.buffer._buffers)
        self.assertIn(("socket-b", "shared"), state.sessions)
        self.assertEqual(len(handler.buffer), 1)

    def test_native_inference_lease_is_published_and_cleared(self) -> None:
        namespace = load_task_handler_namespace()
        active = ActiveInference()
        handler, _state = build_handler(
            namespace,
            ScriptedQueue(),
            active_inference=active,
        )
        task = make_task("leased-task", 1)
        task.deadline_monotonic = time.monotonic() + 30.0

        class InspectingPipeline:
            def process(self, _task):
                self.observed = tuple(active.values)
                return WorkerResult(
                    task_id=task.task_id,
                    socket_id=task.socket_id,
                    type=task.type,
                    is_final=True,
                )

        pipeline = InspectingPipeline()
        handler.pipeline = pipeline
        handler.handle_audio_task(task)

        self.assertGreater(pipeline.observed[0], 0.0)
        self.assertEqual(pipeline.observed[1], task.deadline_monotonic)
        self.assertEqual(active.values, [0.0, 0.0])

    def test_pipeline_error_emits_safe_final_result_and_cleans_session(self) -> None:
        namespace = load_task_handler_namespace()
        handler, state = build_handler(namespace, ScriptedQueue())
        task = make_task("failed-task", 1)
        state.get_session(task.task_id, task.socket_id, task.type)
        secret = "sensitive model path and token"

        class FailingPipeline:
            def process(self, _task):
                raise RuntimeError(secret)

        handler.pipeline = FailingPipeline()
        handler.handle_audio_task(task)

        self.assertEqual(len(handler.queue_out.items), 1)
        result = handler.queue_out.items[0]
        self.assertTrue(result.is_final)
        self.assertEqual(result.task_id, task.task_id)
        self.assertEqual(result.socket_id, task.socket_id)
        self.assertEqual(
            result.error_code,
            namespace["WORKER_TASK_ERROR_CODE"],
        )
        self.assertEqual(
            result.error_message,
            namespace["WORKER_TASK_ERROR_MESSAGE"],
        )
        self.assertNotIn(secret, repr(result))
        self.assertNotIn((task.socket_id, task.task_id), state.sessions)

        error_args, error_kwargs = namespace["logger"].error_calls[-1]
        self.assertNotIn(secret, " ".join(map(str, error_args)))
        self.assertIs(error_kwargs["exc_info"], True)

    def test_private_http_pipeline_error_suppresses_traceback_detail(self) -> None:
        namespace = load_task_handler_namespace()
        handler, _state = build_handler(namespace, ScriptedQueue())
        task = make_task("private-failure", 1)
        task.log_transcript = False

        class FailingPipeline:
            def process(self, _task):
                raise RuntimeError("private prompt and transcript")

        handler.pipeline = FailingPipeline()
        handler.handle_audio_task(task)

        error_args, error_kwargs = namespace["logger"].error_calls[-1]
        self.assertNotIn("private prompt", " ".join(map(str, error_args)))
        self.assertIs(error_kwargs["exc_info"], False)

    def test_loop_continues_with_next_task_after_pipeline_error(self) -> None:
        namespace = load_task_handler_namespace()
        queue_in = ScriptedQueue(
            [make_task("failed-task", 1), make_task("next-task", 2)]
        )
        handler, state = build_handler(namespace, queue_in)
        processed = []

        class FailingThenSuccessfulPipeline:
            def process(self, task):
                processed.append(task.task_id)
                if task.task_id == "failed-task":
                    raise ValueError("private decoder detail")
                queue_in.put(None)
                return WorkerResult(
                    task_id=task.task_id,
                    socket_id=task.socket_id,
                    type=task.type,
                    text="ok",
                    is_final=True,
                )

        handler.pipeline = FailingThenSuccessfulPipeline()
        handler.loop()

        self.assertEqual(processed, ["failed-task", "next-task"])
        self.assertEqual(len(handler.queue_out.items), 2)
        error_result, success_result = handler.queue_out.items
        self.assertEqual(
            error_result.error_code,
            namespace["WORKER_TASK_ERROR_CODE"],
        )
        self.assertEqual(success_result.text, "ok")
        self.assertIsNone(success_result.error_code)
        self.assertEqual(state.sessions, {})


if __name__ == "__main__":
    unittest.main()
