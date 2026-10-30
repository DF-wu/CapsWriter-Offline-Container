# coding: utf-8

from __future__ import annotations

import ast
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
GPU_BOOST_PATH = ROOT / "core" / "server" / "worker" / "gpu_boost.py"


def load_gpu_boost_namespace(config: SimpleNamespace) -> dict:
    source = GPU_BOOST_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GPU_BOOST_PATH))
    keep_names = {
        "GPU_BOOST_TIMEOUT_ENV",
        "DEFAULT_GPU_BOOST_TIMEOUT_SECONDS",
        "GPU_COMMAND_CLEANUP_TIMEOUT_SECONDS",
        "GPU_COMMAND_RETRY_COOLDOWN_SECONDS",
    }
    keep_functions = {
        "_gpu_boost_timeout_seconds",
        "_gpu_command_popen_kwargs",
        "_kill_gpu_command_tree",
        "_run_gpu_command",
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
            and node.name in keep_functions
        ):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "GpuBoostManager":
            body.append(node)

    namespace = {
        "Config": config,
        "ctypes": SimpleNamespace(),
        "logger": SimpleNamespace(info=lambda *_args, **_kwargs: None,
                                  warning=lambda *_args, **_kwargs: None),
        "math": __import__("math"),
        "os": os,
        "signal": signal,
        "subprocess": subprocess,
        "time": time,
    }
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(GPU_BOOST_PATH), "exec"), namespace)
    return namespace


class GpuBoostManagerTest(unittest.TestCase):
    def test_source_command_timeout_respects_documented_cleanup_budget(self):
        namespace = load_gpu_boost_namespace(SimpleNamespace())
        for value in ("0.01", "60.1"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "CAPSWRITER_GPU_BOOST_TIMEOUT": value,
            }), self.assertRaises(ValueError):
                namespace["_gpu_boost_timeout_seconds"]()

    def make_manager(self, *, enabled=True, boosted=False):
        namespace = load_gpu_boost_namespace(SimpleNamespace(
            gpu_boost_enabled=enabled,
            gpu_boost_cmd=["nvidia-smi", "-i", "0", "-lmc", "6000"],
            gpu_unboost_cmd=["nvidia-smi", "-i", "0", "-rmc"],
            gpu_unboost_timeout=10.0,
        ))
        state = SimpleNamespace(gpu_boosted=boosted, gpu_last_active=100.0)
        return namespace, state, namespace["GpuBoostManager"](state)

    def test_cpu_backend_never_attempts_clock_changes(self):
        namespace, state, manager = self.make_manager()
        namespace["Config"].gpu_boost_backend_active = False
        with patch.object(subprocess, "Popen") as popen:
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
            manager.ensure_boosted()
            manager.shutdown()
        popen.assert_not_called()
        self.assertFalse(state.gpu_boosted)

    def test_disabled_ignores_warmup_audio_idle_and_shutdown(self):
        namespace, state, manager = self.make_manager(enabled=False)
        with patch.object(subprocess, "Popen") as popen:
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
            popen.assert_not_called()
            manager.handle_command(SimpleNamespace(type="file", command=""))
            manager.check_idle()
            self.assertTrue(callable(getattr(manager, "shutdown", None)))
            manager.shutdown()
        popen.assert_not_called()
        self.assertFalse(state.gpu_boosted)
        self.assertEqual(state.gpu_last_active, 100.0)

    def test_idle_uses_elapsed_monotonic_time_despite_wallclock_jumps(self):
        for partial in (False, True):
            for wallclock, elapsed, reset_expected in ((-1000.0, 11.0, True),
                                                       (10000.0, 5.0, False)):
                with self.subTest(partial=partial, wallclock=wallclock):
                    namespace, state, manager = self.make_manager()
                    clock = SimpleNamespace(time=Mock(return_value=1000.0),
                                            monotonic=Mock(return_value=100.0))
                    with (
                        patch.dict(namespace, {"time": clock}),
                        patch.object(manager, "_check_admin", return_value=True),
                        patch.object(subprocess, "Popen") as popen,
                    ):
                        popen.return_value.wait.return_value = 7 if partial else 0
                        manager.ensure_boosted()
                        clock.time.return_value = wallclock
                        clock.monotonic.return_value = 100.0 + elapsed
                        popen.return_value.wait.return_value = 0
                        manager.check_idle()
                        self.assertEqual(popen.call_count, 2 if reset_expected else 1)
                        if reset_expected:
                            self.assertFalse(state.gpu_boosted)
                            manager.shutdown()
                            self.assertEqual(popen.call_count, 2)
                        else:
                            self.assertEqual(state.gpu_boosted, not partial)

    def test_disabling_boost_preserves_owned_cleanup_for_idle_and_shutdown(self):
        for partial in (False, True):
            for cleanup in ("idle", "shutdown"):
                with self.subTest(partial=partial, cleanup=cleanup):
                    namespace, state, manager = self.make_manager()
                    clock = SimpleNamespace(time=Mock(return_value=100.0),
                                            monotonic=Mock(return_value=100.0))
                    with (
                        patch.dict(namespace, {"time": clock}),
                        patch.object(manager, "_check_admin", return_value=True),
                        patch.object(subprocess, "Popen") as popen,
                    ):
                        popen.return_value.wait.return_value = 7 if partial else 0
                        manager.ensure_boosted()
                        namespace["Config"].gpu_boost_enabled = False
                        manager.ensure_boosted()
                        manager.handle_command(SimpleNamespace(command="gpu_boost"))
                        self.assertEqual(popen.call_count, 1)
                        clock.time.return_value = 111.0
                        clock.monotonic.return_value = 111.0
                        popen.return_value.wait.return_value = 0
                        if cleanup == "idle":
                            manager.check_idle()
                        else:
                            manager.shutdown()
                        self.assertEqual(popen.call_count, 2)
                        self.assertEqual(popen.call_args.args[0],
                                         ["nvidia-smi", "-i", "0", "-rmc"])
                        self.assertFalse(state.gpu_boosted)
                        self.assertEqual(state.gpu_last_active, 0.0)
                        manager.shutdown()
                        manager.check_idle()
                        self.assertEqual(popen.call_count, 2)

    def test_disabled_cleanup_failure_still_retries_with_cooldown(self):
        namespace, state, manager = self.make_manager()
        clock = SimpleNamespace(time=Mock(return_value=100.0),
                                monotonic=Mock(return_value=100.0))
        with (
            patch.dict(namespace, {"time": clock}),
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(subprocess, "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 7
            manager.ensure_boosted()
            namespace["Config"].gpu_boost_enabled = False
            clock.time.return_value = 111.0
            clock.monotonic.return_value = 111.0
            manager.check_idle()
            self.assertEqual(popen.call_count, 2)
            manager.check_idle()
            self.assertEqual(popen.call_count, 2)
            clock.monotonic.return_value = 142.0
            popen.return_value.wait.return_value = 0
            manager.check_idle()
            self.assertEqual(popen.call_count, 3)
            self.assertFalse(state.gpu_boosted)
            manager.shutdown()
            self.assertEqual(popen.call_count, 3)

    def test_linux_attempts_typed_command_without_windows_admin_api(self):
        namespace, state, manager = self.make_manager()
        with (
            patch.dict(namespace, {"os": SimpleNamespace(name="posix", environ={})}),
            patch.object(subprocess, "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 0
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
        self.assertTrue(state.gpu_boosted)
        self.assertEqual(popen.call_args.args[0],
                         ["nvidia-smi", "-i", "0", "-lmc", "6000"])
        self.assertIs(popen.call_args.kwargs["shell"], False)

    def test_windows_admin_check_is_preserved(self):
        for allowed in (False, True):
            with self.subTest(allowed=allowed):
                namespace, state, manager = self.make_manager()
                admin = Mock(return_value=allowed)
                with (
                    patch.dict(namespace, {
                        "os": SimpleNamespace(name="nt", environ={}),
                        "ctypes": SimpleNamespace(windll=SimpleNamespace(
                            shell32=SimpleNamespace(IsUserAnAdmin=admin))),
                    }),
                    patch.object(subprocess, "Popen") as popen,
                ):
                    popen.return_value.wait.return_value = 0
                    manager.handle_command(SimpleNamespace(command="gpu_boost"))
                admin.assert_called_once_with()
                self.assertEqual(state.gpu_boosted, allowed)
                self.assertEqual(popen.call_count, int(allowed))

    def test_denied_windows_admin_check_is_cooled_down(self):
        namespace, state, manager = self.make_manager()
        clock = SimpleNamespace(time=lambda: 200.0, monotonic=Mock(return_value=20.0))
        with (
            patch.dict(namespace, {"time": clock}),
            patch.object(manager, "_check_admin", return_value=False) as admin,
            patch.object(subprocess, "Popen") as popen,
        ):
            for _ in range(20):
                manager.handle_command(SimpleNamespace(command="gpu_boost"))
            self.assertEqual(admin.call_count, 1)
            clock.monotonic.return_value = 120.0
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
            self.assertEqual(admin.call_count, 2)
        popen.assert_not_called()
        self.assertFalse(state.gpu_boosted)

    def test_unrelated_control_command_does_not_boost(self):
        namespace, state, manager = self.make_manager()
        with patch.object(subprocess, "Popen") as popen:
            manager.handle_command(SimpleNamespace(type="cmd", command="other"))
        popen.assert_not_called()
        self.assertFalse(state.gpu_boosted)

    def test_abandoned_warmup_expires_and_can_boost_again(self):
        namespace, state, manager = self.make_manager()
        clock = SimpleNamespace(time=Mock(return_value=200.0),
                                monotonic=Mock(return_value=200.0))
        with (
            patch.dict(namespace, {"time": clock}),
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(subprocess, "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 0
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
            self.assertEqual(state.gpu_last_active, 200.0)
            clock.time.return_value = 205.0
            clock.monotonic.return_value = 205.0
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
            self.assertEqual(state.gpu_last_active, 205.0)
            self.assertEqual(popen.call_count, 1)
            clock.time.return_value = 216.0
            clock.monotonic.return_value = 216.0
            manager.check_idle()
            self.assertFalse(state.gpu_boosted)
            self.assertEqual(popen.call_args.args[0], ["nvidia-smi", "-i", "0", "-rmc"])
            self.assertIs(popen.call_args.kwargs["shell"], False)
            manager.handle_command(SimpleNamespace(type="mic", command=""))
            self.assertTrue(state.gpu_boosted)
            self.assertEqual(popen.call_count, 3)

    def test_failed_boost_retries_after_cooldown_not_per_chunk(self):
        namespace, state, manager = self.make_manager()
        clock = SimpleNamespace(time=lambda: 200.0, monotonic=Mock(return_value=20.0))
        with (
            patch.dict(namespace, {"time": clock}),
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(subprocess, "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 7
            manager.handle_command(SimpleNamespace(type="file", command=""))
            for _ in range(20):
                manager.handle_command(SimpleNamespace(command="gpu_boost"))
            self.assertEqual(popen.call_count, 1)
            self.assertFalse(state.gpu_boosted)
            self.assertEqual(state.gpu_last_active, 100.0)
            clock.monotonic.return_value = 120.0
            popen.return_value.wait.return_value = 0
            manager.handle_command(SimpleNamespace(type="file", command=""))
            self.assertTrue(state.gpu_boosted)
            self.assertEqual(popen.call_count, 2)

    def test_failed_reset_keeps_truthful_state_and_retries_without_hot_loop(self):
        namespace, state, manager = self.make_manager(boosted=True)
        clock = SimpleNamespace(time=lambda: 200.0, monotonic=Mock(return_value=200.0))
        with (
            patch.dict(namespace, {"time": clock}),
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(subprocess, "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 7
            manager.check_idle()
            for _ in range(20):
                manager.check_idle()
            self.assertEqual(popen.call_count, 1)
            self.assertTrue(state.gpu_boosted)
            self.assertEqual(state.gpu_last_active, 100.0)
            clock.monotonic.return_value = 300.0
            popen.return_value.wait.return_value = 0
            manager.check_idle()
            self.assertFalse(state.gpu_boosted)
            self.assertEqual(state.gpu_last_active, 0.0)
            self.assertEqual(popen.call_count, 2)

    def test_shutdown_resets_without_waiting_for_idle_or_failed_reset_cooldown(self):
        namespace, state, manager = self.make_manager(boosted=True)
        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(subprocess, "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 7
            manager.check_idle()
            self.assertTrue(state.gpu_boosted)
            popen.return_value.wait.return_value = 0
            self.assertTrue(callable(getattr(manager, "shutdown", None)))
            manager.shutdown()
            manager.shutdown()
        self.assertFalse(state.gpu_boosted)
        self.assertEqual(popen.call_count, 2)

    def test_missing_executable_does_not_change_state(self):
        namespace, state, manager = self.make_manager()
        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(subprocess, "Popen", side_effect=FileNotFoundError),
        ):
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
        self.assertFalse(state.gpu_boosted)
        self.assertEqual(state.gpu_last_active, 100.0)

    def test_failed_boost_is_not_reported_boosted_but_still_resets_on_shutdown(self):
        for failure in (7, "timeout"):
            with self.subTest(failure=failure):
                namespace, state, manager = self.make_manager()
                with (
                    patch.object(manager, "_check_admin", return_value=True),
                    patch.object(subprocess, "Popen") as popen,
                    patch.dict(namespace, {"_kill_gpu_command_tree": Mock()}),
                ):
                    if failure == "timeout":
                        popen.return_value.wait.side_effect = subprocess.TimeoutExpired("boost", 5)
                    else:
                        popen.return_value.wait.return_value = failure
                    manager.handle_command(SimpleNamespace(command="gpu_boost"))
                    self.assertFalse(state.gpu_boosted)
                    self.assertEqual(state.gpu_last_active, 100.0)
                    popen.return_value.wait.side_effect = None
                    popen.return_value.wait.return_value = 0
                    manager.shutdown()
                self.assertEqual(popen.call_count, 2)
                self.assertEqual(popen.call_args.args[0], ["nvidia-smi", "-i", "0", "-rmc"])
                self.assertFalse(state.gpu_boosted)

    def test_possible_boost_side_effect_expires_and_failed_reset_remains_retryable(self):
        namespace, state, manager = self.make_manager()
        clock = SimpleNamespace(time=Mock(return_value=200.0), monotonic=Mock(return_value=200.0))
        with (
            patch.dict(namespace, {"time": clock}),
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(subprocess, "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 7
            manager.handle_command(SimpleNamespace(command="gpu_boost"))
            clock.time.return_value = 211.0
            clock.monotonic.return_value = 211.0
            manager.check_idle()
            self.assertEqual(popen.call_count, 2)
            for _ in range(20):
                manager.check_idle()
            self.assertEqual(popen.call_count, 2)
            clock.monotonic.return_value = 300.0
            popen.return_value.wait.return_value = 0
            manager.check_idle()
            self.assertEqual(popen.call_count, 3)
            self.assertFalse(state.gpu_boosted)
            manager.shutdown()
            self.assertEqual(popen.call_count, 3)

    def test_boost_command_uses_configured_timeout(self) -> None:
        config = SimpleNamespace(gpu_boost_enabled=True, gpu_boost_cmd="boost-gpu")
        gpu_boost = load_gpu_boost_namespace(config)
        state = SimpleNamespace(gpu_boosted=False, gpu_last_active=42.0)
        manager = gpu_boost["GpuBoostManager"](state)
        task = SimpleNamespace(command="gpu_boost")

        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.dict(os.environ, {gpu_boost["GPU_BOOST_TIMEOUT_ENV"]: "1.25"}),
            patch.object(gpu_boost["subprocess"], "Popen") as popen,
        ):
            process = popen.return_value
            process.wait.return_value = 0
            manager.handle_command(task)

        self.assertTrue(state.gpu_boosted)
        self.assertGreater(state.gpu_last_active, 0)
        self.assertEqual(popen.call_args.args, ("boost-gpu",))
        self.assertTrue(popen.call_args.kwargs["shell"])
        process.wait.assert_called_once_with(timeout=1.25)
        if os.name == "posix":
            self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_unboost_timeout_leaves_boost_state_intact(self) -> None:
        config = SimpleNamespace(
            gpu_boost_enabled=True,
            gpu_unboost_timeout=1,
            gpu_unboost_cmd="unboost-gpu",
        )
        gpu_boost = load_gpu_boost_namespace(config)
        state = SimpleNamespace(gpu_boosted=True, gpu_last_active=time.monotonic() - 10)
        manager = gpu_boost["GpuBoostManager"](state)
        kill_tree = Mock()

        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(gpu_boost["subprocess"], "Popen") as popen,
            patch.dict(
                gpu_boost,
                {"_kill_gpu_command_tree": kill_tree},
            ),
        ):
            process = popen.return_value
            process.wait.side_effect = subprocess.TimeoutExpired(
                "unboost-gpu",
                timeout=2,
            )
            manager.check_idle()

        self.assertTrue(state.gpu_boosted)
        self.assertGreater(state.gpu_last_active, 0)
        self.assertEqual(
            process.wait.call_args.kwargs["timeout"],
            gpu_boost["DEFAULT_GPU_BOOST_TIMEOUT_SECONDS"],
        )
        kill_tree.assert_called_once_with(process)

    def test_invalid_timeout_skips_command_and_state_change(self) -> None:
        config = SimpleNamespace(gpu_boost_enabled=True, gpu_boost_cmd="boost-gpu")
        gpu_boost = load_gpu_boost_namespace(config)
        state = SimpleNamespace(gpu_boosted=False, gpu_last_active=42.0)
        manager = gpu_boost["GpuBoostManager"](state)
        task = SimpleNamespace(command="gpu_boost")

        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.dict(os.environ, {gpu_boost["GPU_BOOST_TIMEOUT_ENV"]: "inf"}),
            patch.object(gpu_boost["subprocess"], "Popen") as popen,
        ):
            manager.handle_command(task)

        self.assertFalse(state.gpu_boosted)
        self.assertEqual(state.gpu_last_active, 42.0)
        popen.assert_not_called()

    def test_nonzero_boost_command_does_not_change_state(self) -> None:
        config = SimpleNamespace(gpu_boost_enabled=True, gpu_boost_cmd="boost-gpu")
        gpu_boost = load_gpu_boost_namespace(config)
        state = SimpleNamespace(gpu_boosted=False, gpu_last_active=42.0)
        manager = gpu_boost["GpuBoostManager"](state)

        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(gpu_boost["subprocess"], "Popen") as popen,
        ):
            popen.return_value.wait.return_value = 7
            manager.handle_command(SimpleNamespace(command="gpu_boost"))

        self.assertFalse(state.gpu_boosted)
        self.assertEqual(state.gpu_last_active, 42.0)

    def test_windows_cleanup_uses_bounded_taskkill_tree(self) -> None:
        gpu_boost = load_gpu_boost_namespace(SimpleNamespace())
        process = Mock(pid=4321)
        process.poll.return_value = None
        fake_os = SimpleNamespace(name="nt")

        with (
            patch.dict(gpu_boost, {"os": fake_os}),
            patch.object(gpu_boost["subprocess"], "run") as run,
        ):
            gpu_boost["_kill_gpu_command_tree"](process)

        self.assertEqual(
            run.call_args.args[0],
            ["taskkill", "/PID", "4321", "/T", "/F"],
        )
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            gpu_boost["GPU_COMMAND_CLEANUP_TIMEOUT_SECONDS"],
        )
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with(
            timeout=gpu_boost["GPU_COMMAND_CLEANUP_TIMEOUT_SECONDS"],
        )

    @unittest.skipUnless(os.name == "posix", "POSIX process-group regression")
    def test_timeout_kills_spawned_descendant_process_group(self) -> None:
        config = SimpleNamespace(gpu_boost_cmd="")
        gpu_boost = load_gpu_boost_namespace(config)

        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory, "child.pid")
            child_code = "import time; time.sleep(30)"
            parent_code = (
                "import pathlib, subprocess, sys, time; "
                f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(30)"
            )
            command = (
                f"{shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"
            )
            child_pid = None
            try:
                with patch.dict(
                    os.environ,
                    {gpu_boost["GPU_BOOST_TIMEOUT_ENV"]: "0.25"},
                ):
                    self.assertFalse(gpu_boost["_run_gpu_command"](command))
                child_pid = int(pid_path.read_text(encoding="utf-8"))
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    try:
                        os.kill(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("timed-out GPU command descendant survived cleanup")
            finally:
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
