# coding: utf-8

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_tui


ROOT = Path(__file__).resolve().parents[2]


class VerifyTuiTest(unittest.TestCase):
    def test_verifier_consumes_every_direct_tui_pin(self) -> None:
        pins = verify_tui.load_direct_pins(ROOT / "requirements-tui.txt")
        locked = verify_tui.load_locked_versions(ROOT / "requirements-tui.lock")

        self.assertEqual(set(pins), set(verify_tui.REQUIRED_IMPORTS))
        self.assertEqual(verify_tui.configuration_errors(pins, locked), [])
        for distribution, modules in verify_tui.REQUIRED_IMPORTS.items():
            with self.subTest(distribution=distribution):
                self.assertTrue(modules)

    def test_load_direct_pins_rejects_non_exact_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "requirements.txt"
            path.write_text("textual>=8\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strict name==version pin"):
                verify_tui.load_direct_pins(path)

    def test_configuration_errors_report_direct_lock_mismatch(self) -> None:
        errors = verify_tui.configuration_errors(
            {"httpx": "0.28.1", "textual": "8.2.8"},
            {"httpx": "0.28.0"},
        )

        self.assertTrue(any("lock mismatch" in error for error in errors))
        self.assertTrue(any("missing from lock" in error for error in errors))

    def test_dependency_errors_report_version_and_import_failures(self) -> None:
        with (
            patch.object(verify_tui, "REQUIRED_IMPORTS", {"textual": ("textual",)}),
            patch.object(verify_tui.metadata, "version", return_value="8.2.7"),
            patch.object(
                verify_tui.importlib,
                "import_module",
                side_effect=ImportError("broken import"),
            ),
        ):
            errors = verify_tui.dependency_errors({"textual": "8.2.8"})

        self.assertTrue(any("version mismatch" in error for error in errors))
        self.assertTrue(any("import failed" in error for error in errors))

    def test_runtime_range_covers_python_310_and_312_only(self) -> None:
        self.assertIsNone(verify_tui.runtime_error((3, 10, 20)))
        self.assertIsNone(verify_tui.runtime_error((3, 12, 13)))
        self.assertIsNotNone(verify_tui.runtime_error((3, 9, 99)))
        self.assertIsNotNone(verify_tui.runtime_error((3, 13, 0)))

    def test_pip_check_failure_and_timeout_are_release_failures(self) -> None:
        failed = subprocess.CompletedProcess(
            ["python", "-m", "pip", "check"],
            1,
            stdout="textual has incompatible dependency",
        )
        with patch.object(verify_tui.subprocess, "run", return_value=failed):
            self.assertTrue(verify_tui.run_pip_check(1))

        with patch.object(
            verify_tui.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(["pip", "check"], 1),
        ):
            self.assertEqual(verify_tui.run_pip_check(1), ["pip check timed out"])

    def test_strict_runner_is_unsuccessful_when_a_test_is_skipped(self) -> None:
        class SkippedTuiTest(unittest.TestCase):
            @unittest.skip("optional dependency unavailable")
            def runTest(self) -> None:
                self.fail("skip decorator should prevent this")

        result = verify_tui.StrictTextTestRunner(
            stream=io.StringIO(),
            verbosity=0,
        ).run(unittest.TestSuite([SkippedTuiTest()]))

        self.assertEqual(len(result.skipped), 1)
        self.assertFalse(result.wasSuccessful())

    def test_main_stops_before_tests_when_dependency_verification_fails(self) -> None:
        with (
            patch.object(verify_tui, "runtime_error", return_value=None),
            patch.object(verify_tui, "load_direct_pins", return_value={}),
            patch.object(verify_tui, "load_locked_versions", return_value={}),
            patch.object(
                verify_tui,
                "configuration_errors",
                return_value=["missing dependency pin: textual"],
            ),
            patch.object(verify_tui, "dependency_errors", return_value=[]),
            patch.object(verify_tui, "run_pip_check", return_value=[]),
            patch.object(verify_tui, "run_tui_tests") as run_tests,
            patch("sys.stderr", new=io.StringIO()),
        ):
            code = verify_tui.main()

        self.assertEqual(code, 1)
        run_tests.assert_not_called()


if __name__ == "__main__":
    unittest.main()
