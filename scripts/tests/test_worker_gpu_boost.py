# coding: utf-8
"""Dependency-light worker integration; hardware and model calls are faked."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.tests.test_gpu_boost import load_gpu_boost_namespace
from scripts.tests.test_task_handler import (
    ActiveInference, ScriptedQueue, WorkerResult, build_handler,
    load_task_handler_namespace, make_task,
)


WORKER_PATH = Path(__file__).resolve().parents[2] / "core/server/worker/worker.py"


class WorkerGpuBoostTest(unittest.TestCase):
    def build(self, *, enabled=True):
        config = SimpleNamespace(
            gpu_boost_enabled=enabled, gpu_boost_cmd=["nvidia-smi", "-lmc", "6000"],
            gpu_unboost_cmd=["nvidia-smi", "-rmc"], gpu_unboost_timeout=10.0,
        )
        gpu = load_gpu_boost_namespace(config)
        namespace = load_task_handler_namespace()
        namespace["GpuBoostManager"] = gpu["GpuBoostManager"]
        active = ActiveInference()
        handler, state = build_handler(namespace, ScriptedQueue(), active_inference=active,
                                       sockets_id=["socket", "http:request"])
        state.gpu_boosted = False
        state.gpu_last_active = 0.0
        return gpu, namespace, handler, state, active

    def test_http_file_and_mic_boost_before_inference_and_refresh_after_completion(self):
        for source, socket in (("file", "http:request"), ("file", "socket"), ("mic", "socket")):
            with self.subTest(source=source, socket=socket):
                gpu, namespace, handler, state, active = self.build()
                clock = SimpleNamespace(time=Mock(return_value=1000.0),
                                        monotonic=Mock(return_value=100.0))
                task = make_task("audio", 0, socket_id=socket)
                task.type = source

                def process(_task):
                    self.assertTrue(state.gpu_boosted)
                    self.assertGreater(active.values[0], 0)
                    clock.time.return_value = 2000.0
                    clock.monotonic.return_value = 200.0
                    return WorkerResult(task_id="audio", socket_id=socket, type=source)

                handler.pipeline = SimpleNamespace(process=process)
                with (
                    patch.dict(gpu, {"time": clock}),
                    patch.dict(namespace, {"time": clock}),
                    patch.object(handler.gpu_boost, "_check_admin", return_value=True),
                    patch.object(gpu["subprocess"], "Popen") as popen,
                ):
                    popen.return_value.wait.return_value = 0
                    handler.dispatch_task(task)
                    self.assertEqual(state.gpu_last_active, 200.0)
                    handler.dispatch_task(task)
                self.assertEqual(popen.call_count, 1)
                self.assertEqual(len(handler.queue_out.items), 2)
                self.assertIsNone(handler.queue_out.items[0].error_code)
                self.assertEqual(active.values, [0.0, 0.0])

    def test_inference_failure_still_refreshes_activity_and_clears_watchdog(self):
        gpu, namespace, handler, state, active = self.build()
        state.gpu_boosted = True
        state.gpu_last_active = 50.0
        clock = SimpleNamespace(time=Mock(return_value=1000.0),
                                monotonic=Mock(return_value=100.0))

        def process(_task):
            clock.time.return_value = 2000.0
            clock.monotonic.return_value = 200.0
            gpu["Config"].gpu_boost_enabled = False
            raise RuntimeError("private decoder failure")

        handler.pipeline = SimpleNamespace(process=process)
        with patch.dict(gpu, {"time": clock}), patch.dict(namespace, {"time": clock}):
            handler.handle_audio_task(make_task("failure", 0))
        self.assertEqual(state.gpu_last_active, 200.0)
        self.assertEqual(active.values, [0.0, 0.0])
        self.assertEqual(handler.queue_out.items[0].error_code, "worker_processing_failed")

    def test_command_failure_or_disabled_boost_does_not_fail_recognition(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                gpu, namespace, handler, state, _ = self.build(enabled=enabled)
                task = make_task("audio", 0)
                handler.pipeline = SimpleNamespace(process=lambda _: WorkerResult(
                    task_id="audio", socket_id="socket", type="file", text="recognized"))
                with (
                    patch.object(handler.gpu_boost, "_check_admin", return_value=True),
                    patch.object(gpu["subprocess"], "Popen") as popen,
                ):
                    popen.return_value.wait.return_value = 1
                    handler.dispatch_task(task)
                self.assertFalse(state.gpu_boosted)
                self.assertEqual(handler.queue_out.items[0].text, "recognized")
                self.assertEqual(popen.call_count, int(enabled))

    def test_expired_or_disconnected_task_does_not_boost(self):
        gpu, _, handler, state, _ = self.build()
        disconnected = make_task("disconnected", 0, socket_id="gone")
        expired = make_task("expired", 0)
        expired.deadline_monotonic = 0.0
        with patch.object(gpu["subprocess"], "Popen") as popen:
            handler.dispatch_task(disconnected)
            handler.dispatch_task(expired)
        popen.assert_not_called()
        self.assertFalse(state.gpu_boosted)

    def test_command_wait_oserror_is_nonfatal_and_process_is_cleaned(self):
        gpu, _, handler, state, _ = self.build()
        task = make_task("audio", 0)
        handler.pipeline = SimpleNamespace(process=lambda _: WorkerResult(
            task_id="audio", socket_id="socket", type="file", text="recognized"))
        cleanup = Mock()
        with (
            patch.object(handler.gpu_boost, "_check_admin", return_value=True),
            patch.object(gpu["subprocess"], "Popen") as popen,
            patch.dict(gpu, {"_kill_gpu_command_tree": cleanup}),
        ):
            popen.return_value.wait.side_effect = OSError("wait failed")
            try:
                handler.dispatch_task(task)
            except OSError as exc:
                self.fail(f"GPU command failure escaped recognition: {exc}")
        self.assertFalse(state.gpu_boosted)
        self.assertEqual(handler.queue_out.items[0].text, "recognized")
        cleanup.assert_called_once_with(popen.return_value)

    def test_cancellation_during_boost_drops_task_before_watchdog_and_inference(self):
        for cancellation in ("disconnect", "deadline"):
            with self.subTest(cancellation=cancellation):
                gpu, namespace, handler, state, active = self.build()
                task = make_task("cancelled", 0)
                clock = SimpleNamespace(time=lambda: 100.0, monotonic=Mock(return_value=1.0))
                task.deadline_monotonic = 5.0
                state.get_session(task.task_id, task.socket_id, task.type)
                handler.pipeline = Mock()
                leases_during_command = []

                def command_finished(**_kwargs):
                    leases_during_command.append(tuple(active.values))
                    if cancellation == "disconnect":
                        handler.sockets_id.remove(task.socket_id)
                    else:
                        clock.monotonic.return_value = 10.0
                    return 0

                with (
                    patch.dict(gpu, {"time": clock}),
                    patch.dict(namespace, {"time": clock}),
                    patch.object(handler.gpu_boost, "_check_admin", return_value=True),
                    patch.object(gpu["subprocess"], "Popen") as popen,
                ):
                    popen.return_value.wait.side_effect = command_finished
                    handler.dispatch_task(task)
                self.assertEqual(leases_during_command, [(0.0, 0.0)])
                handler.pipeline.process.assert_not_called()
                self.assertEqual(active.values, [0.0, 0.0])
                self.assertNotIn((task.socket_id, task.task_id), state.sessions)
                self.assertFalse(handler.queue_out.items)

    def test_normal_worker_stop_resets_before_model_cleanup_once(self):
        gpu, _, handler, state, _ = self.build()
        state.gpu_boosted = True
        tree = ast.parse(WORKER_PATH.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "RecognizerWorker")
        namespace = {"Queue": object, "ListProxy": object, "logger": Mock()}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(WORKER_PATH), "exec"), namespace)
        worker = namespace["RecognizerWorker"].__new__(namespace["RecognizerWorker"])
        worker._is_running = True
        worker.handler = handler
        observed = []
        worker.loader = SimpleNamespace(cleanup=lambda: observed.append(state.gpu_boosted))
        with (
            patch.object(handler.gpu_boost, "_check_admin", return_value=True),
            patch.object(gpu["subprocess"], "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 0
            worker.stop()
            worker.stop()
        self.assertEqual(observed, [False])
        self.assertFalse(state.gpu_boosted)
        self.assertEqual(popen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
