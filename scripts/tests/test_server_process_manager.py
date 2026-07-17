# coding: utf-8

from __future__ import annotations

import ast
import math
import os
import queue
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PROCESS_MANAGER_PATH = ROOT / "core" / "server" / "worker" / "process_manager.py"


class FakeLogger:
    def __init__(self) -> None:
        self.debug_messages = []
        self.error_messages = []
        self.info_messages = []
        self.warning_messages = []

    def debug(self, message, *args, **kwargs) -> None:
        self.debug_messages.append(str(message))

    def error(self, message, *args, **kwargs) -> None:
        self.error_messages.append(str(message))

    def info(self, message, *args, **kwargs) -> None:
        self.info_messages.append(str(message))

    def warning(self, message, *args, **kwargs) -> None:
        self.warning_messages.append(str(message))


class FakeQueue:
    def __init__(self, *, fail: bool = False, statuses=()) -> None:
        self.fail = fail
        self.items = []
        self.statuses = list(statuses)
        self.get_timeouts = []
        self.put_timeouts = []

    def put(self, item, timeout=None) -> None:
        if self.fail:
            raise RuntimeError("queue closed")
        self.put_timeouts.append(timeout)
        self.items.append(item)

    def get(self, timeout=None):
        self.get_timeouts.append(timeout)
        if not self.statuses:
            raise queue.Empty
        return self.statuses.pop(0)


class FakeProcess:
    pid = 12345
    exitcode = None

    def __init__(self, alive_after_joins) -> None:
        self._alive = True
        self.alive_after_joins = list(alive_after_joins)
        self.join_timeouts = []
        self.kill_called = False
        self.terminate_called = False

    def is_alive(self) -> bool:
        return self._alive

    def join(self, timeout=None) -> None:
        self.join_timeouts.append(timeout)
        if self.alive_after_joins:
            self._alive = self.alive_after_joins.pop(0)

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True

    def start(self) -> None:
        self.start_called = True


class ActiveInference:
    def __init__(self, started: float, deadline: float) -> None:
        self.values = [started, deadline]
        self.lock = threading.Lock()

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def get_lock(self):
        return self.lock


class FakeLoop:
    def __init__(self) -> None:
        self.callbacks = []

    def call_soon_threadsafe(self, callback) -> None:
        self.callbacks.append(callback)


def load_process_manager_namespace() -> dict:
    source = PROCESS_MANAGER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROCESS_MANAGER_PATH))
    keep_names = {
        "SERVER_WORKER_STOP_TIMEOUT_ENV",
        "DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS",
        "SERVER_WORKER_KILL_GRACE_SECONDS",
        "SERVER_WORKER_STALL_TIMEOUT_ENV",
        "DEFAULT_SERVER_WORKER_STALL_TIMEOUT_SECONDS",
        "SERVER_MODEL_LOAD_TIMEOUT_ENV",
        "DEFAULT_SERVER_MODEL_LOAD_TIMEOUT_SECONDS",
        "SERVER_WORKER_WATCHDOG_HTTP_GRACE_SECONDS",
        "SERVER_WORKER_WATCHDOG_POLL_SECONDS",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            if names & keep_names:
                body.append(node)
        elif (
            isinstance(node, ast.FunctionDef)
            and node.name in {
                "_worker_stop_timeout_seconds",
                "_worker_stall_timeout_seconds",
                "_model_load_timeout_seconds",
            }
        ):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "ProcessManager":
            body.append(node)

    logger = FakeLogger()
    namespace = {
        "CapsWriterServer": object,
        "check_model": lambda: None,
        "console": SimpleNamespace(rule=lambda *_args, **_kwargs: None,
                                   line=lambda *_args, **_kwargs: None),
        "logger": logger,
        "Manager": lambda: SimpleNamespace(list=lambda: []),
        "math": math,
        "os": os,
        "Process": object,
        "queue": queue,
        "start_worker": lambda *_args, **_kwargs: None,
        "sys": SimpleNamespace(stdin=SimpleNamespace(fileno=lambda: 0)),
        "threading": threading,
        "time": time,
        "QUEUE_PUT_RETRY_SECONDS": 0.05,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PROCESS_MANAGER_PATH), "exec"), namespace)
    namespace["test_logger"] = logger
    return namespace


def build_manager(namespace: dict, process: FakeProcess, queue_in: FakeQueue):
    manager = namespace["ProcessManager"].__new__(namespace["ProcessManager"])
    manager._process = process
    manager.app = SimpleNamespace(
        state=SimpleNamespace(
            queue_in=queue_in,
            queue_out=FakeQueue(),
            recognizer_watchdog_failed=False,
        )
    )
    manager.is_alive = True
    manager._watchdog_stop = threading.Event()
    manager._watchdog_thread = None
    manager._fail_stop_lock = threading.Lock()
    manager._fail_stop_requested = False
    return manager


class ServerProcessManagerTest(unittest.TestCase):
    def test_model_load_timeout_is_finite_and_positive(self) -> None:
        namespace = load_process_manager_namespace()
        timeout_env = namespace["SERVER_MODEL_LOAD_TIMEOUT_ENV"]

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                namespace["_model_load_timeout_seconds"](),
                namespace["DEFAULT_SERVER_MODEL_LOAD_TIMEOUT_SECONDS"],
            )
        with patch.dict(os.environ, {timeout_env: "42.5"}):
            self.assertEqual(namespace["_model_load_timeout_seconds"](), 42.5)
        for value in ("bad", "0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {timeout_env: value}):
                    with self.assertRaises(ValueError):
                        namespace["_model_load_timeout_seconds"]()

    def test_invalid_model_load_timeout_fails_before_process_or_manager_creation(self) -> None:
        namespace = load_process_manager_namespace()
        calls = []
        namespace["Manager"] = lambda: calls.append("manager")
        namespace["Process"] = lambda *args, **kwargs: calls.append("process")
        app = SimpleNamespace(state=SimpleNamespace())
        manager = namespace["ProcessManager"](app)

        with patch.dict(
            os.environ,
            {namespace["SERVER_MODEL_LOAD_TIMEOUT_ENV"]: "invalid"},
        ):
            with self.assertRaises(ValueError):
                manager.start()

        self.assertEqual(calls, [])
        self.assertFalse(manager.is_alive)
        self.assertIsNone(manager._process)

    def test_model_load_timeout_reaps_live_child_marks_unhealthy_and_raises(self) -> None:
        namespace = load_process_manager_namespace()
        process = FakeProcess([True, False])
        manager = build_manager(namespace, process, FakeQueue())

        with self.assertRaisesRegex(TimeoutError, "model load exceeded"):
            manager._wait_for_models(
                deadline=time.monotonic() - 1.0,
                timeout_seconds=3.0,
            )

        self.assertFalse(manager.is_alive)
        self.assertTrue(manager.app.state.recognizer_watchdog_failed)
        self.assertTrue(process.terminate_called)
        self.assertTrue(process.kill_called)
        self.assertEqual(
            process.join_timeouts,
            [
                namespace["SERVER_WORKER_KILL_GRACE_SECONDS"],
                namespace["SERVER_WORKER_KILL_GRACE_SECONDS"],
            ],
        )

    def test_model_ready_before_deadline_does_not_reap_child(self) -> None:
        namespace = load_process_manager_namespace()
        process = FakeProcess([])
        manager = build_manager(namespace, process, FakeQueue())
        manager.app.state.queue_out = FakeQueue(statuses=[True])

        manager._wait_for_models(
            deadline=time.monotonic() + 1.0,
            timeout_seconds=1.0,
        )

        self.assertTrue(manager.is_alive)
        self.assertFalse(process.terminate_called)
        self.assertFalse(process.kill_called)
        self.assertEqual(len(manager.app.state.queue_out.get_timeouts), 1)
        self.assertLessEqual(manager.app.state.queue_out.get_timeouts[0], 0.1)

    def test_worker_stop_timeout_accepts_default_and_configured_values(self) -> None:
        namespace = load_process_manager_namespace()
        timeout_env = namespace["SERVER_WORKER_STOP_TIMEOUT_ENV"]

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                namespace["_worker_stop_timeout_seconds"](),
                namespace["DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS"],
            )

        with patch.dict(os.environ, {timeout_env: "3.5"}):
            self.assertEqual(namespace["_worker_stop_timeout_seconds"](), 3.5)

    def test_worker_stop_timeout_rejects_invalid_values(self) -> None:
        namespace = load_process_manager_namespace()
        timeout_env = namespace["SERVER_WORKER_STOP_TIMEOUT_ENV"]

        for value in ("bad", "0", "-2", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {timeout_env: value}):
                    with self.assertRaises(ValueError):
                        namespace["_worker_stop_timeout_seconds"]()

    def test_worker_stall_timeout_is_finite_and_positive(self) -> None:
        namespace = load_process_manager_namespace()
        timeout_env = namespace["SERVER_WORKER_STALL_TIMEOUT_ENV"]

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                namespace["_worker_stall_timeout_seconds"](),
                namespace["DEFAULT_SERVER_WORKER_STALL_TIMEOUT_SECONDS"],
            )
        with patch.dict(os.environ, {timeout_env: "42.5"}):
            self.assertEqual(namespace["_worker_stall_timeout_seconds"](), 42.5)
        for value in ("bad", "0", "-1", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {timeout_env: value}):
                    with self.assertRaises(ValueError):
                        namespace["_worker_stall_timeout_seconds"]()

    def test_stop_uses_configured_graceful_timeout(self) -> None:
        namespace = load_process_manager_namespace()
        process = FakeProcess([False])
        queue_in = FakeQueue()
        manager = build_manager(namespace, process, queue_in)

        with patch.dict(os.environ, {namespace["SERVER_WORKER_STOP_TIMEOUT_ENV"]: "1.25"}):
            manager.stop()

        self.assertFalse(manager.is_alive)
        self.assertEqual(queue_in.items, [None])
        self.assertEqual(process.join_timeouts, [1.25])
        self.assertFalse(process.terminate_called)
        self.assertFalse(process.kill_called)

    def test_stop_terminates_and_waits_with_bounded_grace(self) -> None:
        namespace = load_process_manager_namespace()
        process = FakeProcess([True, False])
        queue_in = FakeQueue()
        manager = build_manager(namespace, process, queue_in)

        manager.stop()

        self.assertTrue(process.terminate_called)
        self.assertFalse(process.kill_called)
        self.assertEqual(
            process.join_timeouts,
            [
                namespace["DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS"],
                namespace["SERVER_WORKER_KILL_GRACE_SECONDS"],
            ],
        )

    def test_stop_kills_after_terminate_grace_timeout(self) -> None:
        namespace = load_process_manager_namespace()
        process = FakeProcess([True, True, False])
        queue_in = FakeQueue()
        manager = build_manager(namespace, process, queue_in)

        manager.stop()

        self.assertTrue(process.terminate_called)
        self.assertTrue(process.kill_called)
        self.assertEqual(
            process.join_timeouts,
            [
                namespace["DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS"],
                namespace["SERVER_WORKER_KILL_GRACE_SECONDS"],
                namespace["SERVER_WORKER_KILL_GRACE_SECONDS"],
            ],
        )

    def test_stop_invalid_timeout_falls_back_and_still_stops(self) -> None:
        namespace = load_process_manager_namespace()
        process = FakeProcess([False])
        queue_in = FakeQueue()
        manager = build_manager(namespace, process, queue_in)

        with patch.dict(os.environ, {namespace["SERVER_WORKER_STOP_TIMEOUT_ENV"]: "nan"}):
            manager.stop()

        self.assertEqual(queue_in.items, [None])
        self.assertEqual(
            process.join_timeouts,
            [namespace["DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS"]],
        )
        self.assertTrue(namespace["test_logger"].error_messages)

    def test_stop_queue_failure_still_reaps_process(self) -> None:
        namespace = load_process_manager_namespace()
        process = FakeProcess([False])
        queue_in = FakeQueue(fail=True)
        manager = build_manager(namespace, process, queue_in)

        manager.stop()

        self.assertEqual(
            process.join_timeouts,
            [namespace["DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS"]],
        )
        self.assertTrue(namespace["test_logger"].warning_messages)

    def test_watchdog_reaps_overdue_http_inference_and_schedules_fail_stop(self) -> None:
        namespace = load_process_manager_namespace()
        namespace["SERVER_WORKER_WATCHDOG_POLL_SECONDS"] = 0.001
        process = FakeProcess([False])
        manager = build_manager(namespace, process, FakeQueue())
        loop = FakeLoop()
        stopped = []
        manager.app = SimpleNamespace(
            loop=loop,
            stop=lambda: stopped.append(True),
            state=SimpleNamespace(
                queue_in=FakeQueue(),
                recognizer_active_inference=ActiveInference(
                    time.monotonic(),
                    time.monotonic()
                    - namespace["SERVER_WORKER_WATCHDOG_HTTP_GRACE_SECONDS"]
                    - 1.0,
                ),
                recognizer_watchdog_failed=False,
            ),
        )

        manager._watchdog_loop(stall_timeout=3600.0)

        self.assertTrue(process.terminate_called)
        self.assertTrue(manager.app.state.recognizer_watchdog_failed)
        self.assertEqual(len(loop.callbacks), 1)
        self.assertEqual(stopped, [])
        loop.callbacks[0]()
        self.assertEqual(stopped, [True])

    def test_watchdog_applies_finite_ceiling_to_non_http_inference(self) -> None:
        namespace = load_process_manager_namespace()
        namespace["SERVER_WORKER_WATCHDOG_POLL_SECONDS"] = 0.001
        process = FakeProcess([False])
        manager = build_manager(namespace, process, FakeQueue())
        loop = FakeLoop()
        manager.app = SimpleNamespace(
            loop=loop,
            stop=lambda: None,
            state=SimpleNamespace(
                queue_in=FakeQueue(),
                recognizer_active_inference=ActiveInference(
                    time.monotonic() - 2.0,
                    0.0,
                ),
                recognizer_watchdog_failed=False,
            ),
        )

        manager._watchdog_loop(stall_timeout=1.0)

        self.assertTrue(process.terminate_called)
        self.assertEqual(len(loop.callbacks), 1)

    def test_fail_stop_schedules_server_stop_even_if_child_reap_raises(self) -> None:
        namespace = load_process_manager_namespace()
        manager = build_manager(namespace, FakeProcess([]), FakeQueue())
        loop = FakeLoop()
        manager.app = SimpleNamespace(
            loop=loop,
            stop=lambda: None,
            state=SimpleNamespace(recognizer_watchdog_failed=False),
        )

        def fail_reap():
            raise RuntimeError("simulated process API failure")

        manager._force_stop_process = fail_reap
        with self.assertRaisesRegex(RuntimeError, "process API failure"):
            manager._fail_stop("watchdog test")

        self.assertTrue(manager.app.state.recognizer_watchdog_failed)
        self.assertEqual(len(loop.callbacks), 1)


if __name__ == "__main__":
    unittest.main()
