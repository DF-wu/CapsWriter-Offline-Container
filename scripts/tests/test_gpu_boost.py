# coding: utf-8

from __future__ import annotations

import ast
import os
import subprocess
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
GPU_BOOST_PATH = ROOT / "core" / "server" / "worker" / "gpu_boost.py"


def load_gpu_boost_namespace(config: SimpleNamespace) -> dict:
    source = GPU_BOOST_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(GPU_BOOST_PATH))
    keep_names = {
        "GPU_BOOST_TIMEOUT_ENV",
        "DEFAULT_GPU_BOOST_TIMEOUT_SECONDS",
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
            and node.name in {"_gpu_boost_timeout_seconds", "_run_gpu_command"}
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
            patch.object(gpu_boost["subprocess"], "run") as run,
        ):
            manager.handle_command(task)

        self.assertTrue(state.gpu_boosted)
        self.assertEqual(state.gpu_last_active, 0)
        self.assertEqual(run.call_args.args, ("boost-gpu",))
        self.assertEqual(run.call_args.kwargs["timeout"], 1.25)
        self.assertTrue(run.call_args.kwargs["shell"])

    def test_unboost_timeout_leaves_boost_state_intact(self) -> None:
        config = SimpleNamespace(
            gpu_boost_enabled=True,
            gpu_unboost_timeout=1,
            gpu_unboost_cmd="unboost-gpu",
        )
        gpu_boost = load_gpu_boost_namespace(config)
        state = SimpleNamespace(gpu_boosted=True, gpu_last_active=time.time() - 10)
        manager = gpu_boost["GpuBoostManager"](state)

        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.object(
                gpu_boost["subprocess"],
                "run",
                side_effect=subprocess.TimeoutExpired("unboost-gpu", timeout=2),
            ) as run,
        ):
            manager.check_idle()

        self.assertTrue(state.gpu_boosted)
        self.assertGreater(state.gpu_last_active, 0)
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            gpu_boost["DEFAULT_GPU_BOOST_TIMEOUT_SECONDS"],
        )

    def test_invalid_timeout_skips_command_and_state_change(self) -> None:
        config = SimpleNamespace(gpu_boost_cmd="boost-gpu")
        gpu_boost = load_gpu_boost_namespace(config)
        state = SimpleNamespace(gpu_boosted=False, gpu_last_active=42.0)
        manager = gpu_boost["GpuBoostManager"](state)
        task = SimpleNamespace(command="gpu_boost")

        with (
            patch.object(manager, "_check_admin", return_value=True),
            patch.dict(os.environ, {gpu_boost["GPU_BOOST_TIMEOUT_ENV"]: "inf"}),
            patch.object(gpu_boost["subprocess"], "run") as run,
        ):
            manager.handle_command(task)

        self.assertFalse(state.gpu_boosted)
        self.assertEqual(state.gpu_last_active, 42.0)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
