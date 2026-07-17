from __future__ import annotations

import html
import os
import re
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

    def test_screenshot_removes_remote_font_sources(self) -> None:
        source = '''
        <style>
        @font-face {
            font-family: "Fira Code";
            src: local("FiraCode-Regular"),
                url("https://cdnjs.cloudflare.com/fonts/FiraCode.woff2") format("woff2"),
                url("https://cdnjs.cloudflare.com/fonts/FiraCode.woff") format("woff");
        }
        </style>
        '''

        result = capture_screenshot.remove_remote_font_sources(source)

        self.assertIn('src: local("FiraCode-Regular");', result)
        self.assertNotIn("cdnjs.cloudflare.com", result)
        self.assertNotIn('url("https://', result)

    def test_screenshot_rejects_unknown_remote_css_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "external CSS URL"):
            capture_screenshot.remove_remote_font_sources(
                '<style>src: url("https://example.test/font.woff2")</style>'
            )


class ScreenshotCaptureTest(unittest.IsolatedAsyncioTestCase):
    async def test_real_capture_is_current_clock_free_and_offline(self) -> None:
        rendered = await capture_screenshot.capture_svg(
            locale="en",
            width=140,
            height=46,
        )
        committed = capture_screenshot.DEFAULT_OUTPUT.read_text(encoding="utf-8")

        self.assertEqual(rendered, committed)
        self.assertIsNone(capture_screenshot.REMOTE_CSS_URL.search(rendered))
        header_text = html.unescape(
            "".join(
                re.findall(
                    r'<text[^>]+clip-path="[^"]+-line-0[^"]*"[^>]*>(.*?)</text>',
                    rendered,
                )
            )
        )
        self.assertIn("CapsWriter", header_text)
        self.assertNotRegex(header_text, r"\b\d{1,2}:\d{2}\b")


if __name__ == "__main__":
    unittest.main()
