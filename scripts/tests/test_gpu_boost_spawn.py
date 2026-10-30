"""A spawned worker must receive the parent's resolved GPU controls."""

import ast
import multiprocessing
from pathlib import Path
import unittest
from unittest.mock import patch

from config_server import ServerConfig


def _spawn_probe(output, snapshot):
    """Run the production entrypoint in a fresh interpreter with a fake model worker."""
    source = Path(__file__).resolve().parents[2] / "core/server/worker/__init__.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    entry = next(node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "start_worker")
    before = ServerConfig.gpu_boost_enabled

    class Worker:
        def __init__(self, *_args, **_kwargs):
            self.controls = {
                name: getattr(ServerConfig, name, True) for name in snapshot
            }

        def run(self):
            output.put((before, self.controls))

    namespace = {"Queue": object, "ListProxy": object, "RecognizerWorker": Worker}
    exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
    namespace["start_worker"](None, None, [], None, gpu_boost_config=snapshot)


class GpuBoostSpawnTest(unittest.TestCase):
    def test_spawn_fresh_import_preserves_typed_and_disabled_legacy_controls(self):
        context = multiprocessing.get_context("spawn")
        for enabled, command in (
            (True, ["nvidia-smi", "-i", "1", "-lmc", "6000"]),
            (False, "legacy custom command"),
        ):
            snapshot = {
                "gpu_boost_enabled": enabled, "gpu_boost_backend_active": enabled,
                "gpu_boost_cmd": command, "gpu_unboost_cmd": ["reset", "1"],
                "gpu_unboost_timeout": 2.5,
            }
            with self.subTest(enabled=enabled):
                output = context.Queue()
                process = context.Process(target=_spawn_probe, args=(output, snapshot))
                try:
                    # Deliberately different from config_server's fresh-import default.
                    with patch.object(ServerConfig, "gpu_boost_enabled", True):
                        process.start()
                    before, observed = output.get(timeout=10)
                    process.join(timeout=10)
                    self.assertEqual(process.exitcode, 0)
                    self.assertFalse(before)
                    self.assertEqual(observed, snapshot)
                finally:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=2)
                    output.close()
                    output.join_thread()

    def test_worker_applies_parent_controls_before_constructing_handler(self):
        source = Path(__file__).resolve().parents[2] / "core/server/worker/__init__.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        entry = next(node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == "start_worker")
        observed = {}

        class Worker:
            def __init__(self, *_args, **_kwargs):
                observed.update(
                    enabled=ServerConfig.gpu_boost_enabled,
                    boost=ServerConfig.gpu_boost_cmd,
                    reset=ServerConfig.gpu_unboost_cmd,
                    idle=ServerConfig.gpu_unboost_timeout,
                )

            def run(self):
                observed["ran"] = True

        namespace = {"Queue": object, "ListProxy": object, "RecognizerWorker": Worker}
        exec(compile(ast.Module(body=[entry], type_ignores=[]), str(source), "exec"), namespace)
        with patch.multiple(ServerConfig, gpu_boost_enabled=False,
                            gpu_boost_cmd="old-boost", gpu_unboost_cmd="old-reset",
                            gpu_unboost_timeout=1):
            namespace["start_worker"](None, None, [], None, gpu_boost_config={
                "gpu_boost_enabled": True,
                "gpu_boost_cmd": ["nvidia-smi", "-i", "1", "-lmc", "6000"],
                "gpu_unboost_cmd": ["nvidia-smi", "-i", "1", "-rmc"],
                "gpu_unboost_timeout": 2.5,
            })
        self.assertEqual(observed, {
            "enabled": True,
            "boost": ["nvidia-smi", "-i", "1", "-lmc", "6000"],
            "reset": ["nvidia-smi", "-i", "1", "-rmc"],
            "idle": 2.5, "ran": True,
        })


if __name__ == "__main__":
    unittest.main()
