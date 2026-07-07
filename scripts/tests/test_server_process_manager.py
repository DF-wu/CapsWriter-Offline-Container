# coding: utf-8

from __future__ import annotations

import ast
import math
import os
import queue
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
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.items = []

    def put(self, item) -> None:
        if self.fail:
            raise RuntimeError("queue closed")
        self.items.append(item)


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


def load_process_manager_namespace() -> dict:
    source = PROCESS_MANAGER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROCESS_MANAGER_PATH))
    keep_names = {
        "SERVER_WORKER_STOP_TIMEOUT_ENV",
        "DEFAULT_SERVER_WORKER_STOP_TIMEOUT_SECONDS",
        "SERVER_WORKER_KILL_GRACE_SECONDS",
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
            and node.name == "_worker_stop_timeout_seconds"
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
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(PROCESS_MANAGER_PATH), "exec"), namespace)
    namespace["test_logger"] = logger
    return namespace


def build_manager(namespace: dict, process: FakeProcess, queue_in: FakeQueue):
    manager = namespace["ProcessManager"].__new__(namespace["ProcessManager"])
    manager._process = process
    manager.app = SimpleNamespace(state=SimpleNamespace(queue_in=queue_in))
    manager.is_alive = True
    return manager


class ServerProcessManagerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
