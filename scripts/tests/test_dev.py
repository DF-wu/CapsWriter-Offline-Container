"""Behavioral checks for the isolated development command entry point."""

from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import dev


class DevelopmentWorkflowTests(unittest.TestCase):
    def test_child_environment_keeps_proxy_but_removes_interpreter_overrides(self):
        env = {"PATH": "/bin", "HTTPS_PROXY": "http://proxy:8080",
               "PYTHONHOME": "/elsewhere", "PYTHONPATH": "/elsewhere",
               "VIRTUAL_ENV": "/elsewhere", "CONDA_PREFIX": "/elsewhere",
               "UV_PROJECT_ENVIRONMENT": "/elsewhere"}
        with patch.dict(os.environ, env, clear=True):
            actual = dev.child_env()
        self.assertEqual(actual["HTTPS_PROXY"], env["HTTPS_PROXY"])
        for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
                     "UV_PROJECT_ENVIRONMENT"):
            self.assertNotIn(name, actual)
        self.assertEqual(actual["PYTHONNOUSERSITE"], "1")

    def test_failed_setup_invalidates_previous_ready_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile = root / "dev"
            profile.mkdir()
            marker = profile / ".capswriter-ready"
            marker.write_text("old")
            with patch.object(dev, "ENV_ROOT", root), \
                 patch.object(dev, "executable", return_value="uv"), \
                 patch.object(dev, "run", side_effect=dev.DevError("failed")):
                with self.assertRaises(dev.DevError):
                    dev.setup("dev")
            self.assertFalse(marker.exists())

    @unittest.skipIf(os.name == "nt", "symlink creation requires Windows privileges")
    def test_setup_refuses_symlinked_profile_without_touching_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outside = root / "outside"
            outside.mkdir()
            marker = outside / ".capswriter-ready"
            marker.write_text("keep")
            envs = root / "envs"
            envs.mkdir()
            (envs / "dev").symlink_to(outside, target_is_directory=True)
            with patch.object(dev, "ENV_ROOT", envs), \
                 patch.object(dev, "executable", return_value="uv"), \
                 patch.object(dev, "run") as run:
                with self.assertRaises(dev.DevError):
                    dev.setup("dev")
            run.assert_not_called()
            self.assertEqual(marker.read_text(), "keep")

    def test_tui_setup_installs_pip_needed_by_strict_verifier(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "tui").mkdir()
            with patch.object(dev, "ENV_ROOT", root), \
                 patch.object(dev, "executable", return_value="uv"), \
                 patch.object(dev, "run") as run:
                dev.setup("tui")
            commands = [call.args[0] for call in run.call_args_list]
            bootstrap = str(dev.ROOT / "requirements-windows-build-bootstrap.lock")
            self.assertTrue(any(bootstrap in command for command in commands))
            self.assertTrue((root / "tui" / ".capswriter-ready").exists())

    def test_test_command_reports_missing_setup_without_running_subprocess(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(dev, "ENV_ROOT", Path(folder)), \
             patch.object(dev, "run") as run, redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(dev.main(["test"]), 1)
        run.assert_not_called()
        self.assertIn("setup dev", stderr.getvalue())

    def test_failed_or_timed_out_command_is_actionable(self):
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess([], 7)):
            with self.assertRaisesRegex(dev.DevError, "exit code 7"):
                dev.run(["tool"])
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("tool", 1)):
            with self.assertRaisesRegex(dev.DevError, "exceeded"):
                dev.run(["tool"])


if __name__ == "__main__":
    unittest.main()
