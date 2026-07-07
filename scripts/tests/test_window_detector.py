# coding: utf-8

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WINDOW_DETECTOR_PATH = ROOT / "core" / "tools" / "window_detector.py"
spec = importlib.util.spec_from_file_location("window_detector", WINDOW_DETECTOR_PATH)
assert spec is not None
window_detector = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(window_detector)


class WindowDetectorTest(unittest.TestCase):
    def test_macos_window_detection_uses_configured_timeout(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="Safari||Example Page\n")
        with (
            patch.dict(os.environ, {window_detector.WINDOW_DETECT_TIMEOUT_ENV: "1.25"}),
            patch.object(window_detector.subprocess, "run", return_value=completed) as run,
        ):
            info = window_detector._get_macos_window_info()

        self.assertEqual(info["process_name"], "Safari")
        self.assertEqual(info["title"], "Example Page")
        self.assertEqual(run.call_args.kwargs["timeout"], 1.25)

    def test_linux_window_detection_uses_default_timeout_and_parses_title(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="0x012 0 0 0 0 Code Editor\n")
        with patch.object(window_detector.subprocess, "run", return_value=completed) as run:
            info = window_detector._get_linux_window_info()

        self.assertEqual(info["title"], "Code Editor")
        self.assertEqual(info["app_name"], "Code")
        self.assertEqual(
            run.call_args.kwargs["timeout"],
            window_detector.DEFAULT_WINDOW_DETECT_TIMEOUT_SECONDS,
        )

    def test_window_detection_timeout_returns_empty_info(self) -> None:
        with patch.object(
            window_detector.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["wmctrl"], timeout=2),
        ):
            self.assertEqual(window_detector._get_linux_window_info(), {})

    def test_invalid_timeout_returns_empty_info_before_spawning(self) -> None:
        with (
            patch.dict(os.environ, {window_detector.WINDOW_DETECT_TIMEOUT_ENV: "nan"}),
            patch.object(window_detector.subprocess, "run") as run,
        ):
            self.assertEqual(window_detector._get_linux_window_info(), {})

        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
