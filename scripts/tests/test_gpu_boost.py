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
    def test_boost_command_uses_configured_timeout(self) -> None:
        config = SimpleNamespace(gpu_boost_cmd="boost-gpu")
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
        self.assertEqual(state.gpu_last_active, 0)
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
        state = SimpleNamespace(gpu_boosted=True, gpu_last_active=time.time() - 10)
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
        config = SimpleNamespace(gpu_boost_cmd="boost-gpu")
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
        config = SimpleNamespace(gpu_boost_cmd="boost-gpu")
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
