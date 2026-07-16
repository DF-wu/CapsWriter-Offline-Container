from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from client.tui.scripts import capture_screenshot, clean, verify


class ScriptTest(unittest.TestCase):
    def test_verify_timeout_is_bounded(self) -> None:
        with mock.patch.dict(os.environ, {"CAPSWRITER_TUI_VERIFY_STEP_TIMEOUT": "0"}):
            with self.assertRaises(ValueError):
                verify.step_timeout()

    def test_local_verifier_delegates_to_no_skip_release_suite(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command: list[str], _timeout: float) -> int:
            commands.append(command)
            return 0

        with mock.patch.object(verify, "run", side_effect=fake_run):
            self.assertEqual(verify.main(), 0)

        self.assertIn([verify.sys.executable, "scripts/verify_tui.py"], commands)
        self.assertNotIn(
            [
                verify.sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "client/tui/tests",
                "-v",
            ],
            commands,
        )
        with mock.patch.dict(os.environ, {"CAPSWRITER_TUI_VERIFY_STEP_TIMEOUT": "3601"}):
            with self.assertRaises(ValueError):
                verify.step_timeout()

    def test_clean_is_scoped_to_supplied_tui_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory, "tui")
            cache = root / "feature" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "module.pyc").write_bytes(b"cache")
            sibling = Path(directory, "outside.pyc")
            sibling.write_bytes(b"keep")
            with mock.patch.object(clean, "TUI_ROOT", root):
                self.assertTrue(clean.residues())
                clean.clean()
                self.assertEqual(clean.residues(), [])
            self.assertEqual(sibling.read_bytes(), b"keep")

    def test_screenshot_metadata_is_accessible_and_keeps_rendered_svg(self) -> None:
        source = (
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<title>Real Textual render</title><rect width="10" height="10"/></svg>'
        )

        result = capture_screenshot.add_accessibility_metadata(
            source,
            "Rendered workbench & controls",
        )

        self.assertIn('role="img"', result)
        self.assertIn('aria-labelledby="tui-shot-title tui-shot-desc"', result)
        self.assertIn('<title id="tui-shot-title">Real Textual render</title>', result)
        self.assertIn(
            '<desc id="tui-shot-desc">Rendered workbench &amp; controls</desc>',
            result,
        )
        self.assertIn('<rect width="10" height="10"/>', result)


if __name__ == "__main__":
    unittest.main()
