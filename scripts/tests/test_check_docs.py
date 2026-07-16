# coding: utf-8

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_docs


class DocumentationCheckerTest(unittest.TestCase):
    def test_heading_anchors_match_duplicate_and_cjk_headings(self) -> None:
        source = "# Hello, World!\n## 中文安裝\n## 中文安裝\n```\n# ignored\n```\n"

        self.assertEqual(
            check_docs.heading_anchors(source),
            {"hello-world", "中文安裝", "中文安裝-1"},
        )

    def test_validates_links_anchors_and_accessible_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            target = docs / "guide.md"
            target.write_text("# Setup\n", encoding="utf-8")
            image = docs / "flow.svg"
            image.write_text(
                "<svg><title>Request flow</title><desc>Client to server.</desc></svg>\n",
                encoding="utf-8",
            )
            source = docs / "index.md"
            source.write_text(
                "# Index\n[guide](guide.md#setup)\n"
                "![Request flow](flow.svg)\n",
                encoding="utf-8",
            )

            self.assertEqual(check_docs.validate_markdown_file(source, root), [])

    def test_referenced_svg_requires_title_and_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "flow.svg"
            image.write_text("<svg/>\n", encoding="utf-8")
            source = root / "index.md"
            source.write_text("# Index\n![Request flow](flow.svg)\n", encoding="utf-8")

            messages = [
                issue.message
                for issue in check_docs.validate_markdown_file(source, root)
            ]

            self.assertTrue(any("no <title>" in message for message in messages))
            self.assertTrue(any("no <desc>" in message for message in messages))

    def test_reports_missing_target_anchor_and_bad_alt_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guide = root / "guide.md"
            guide.write_text("# Existing\n", encoding="utf-8")
            source = root / "index.md"
            source.write_text(
                "# Index\n[missing](missing.md)\n"
                "[bad anchor](guide.md#absent)\n"
                "![](guide.md)\n"
                "![chart.png](guide.md)\n",
                encoding="utf-8",
            )

            messages = [
                issue.message
                for issue in check_docs.validate_markdown_file(source, root)
            ]

            self.assertTrue(any("does not exist" in message for message in messages))
            self.assertTrue(any("anchor does not exist" in message for message in messages))
            self.assertIn("image is missing descriptive alt text", messages)
            self.assertIn("image alt text is only a filename", messages)

    def test_ignores_link_like_content_inside_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "index.md"
            source.write_text(
                "# Regex\n`[]()`\n```html\n<img src=\"missing.png\">\n```\n",
                encoding="utf-8",
            )

            self.assertEqual(check_docs.validate_markdown_file(source, root), [])

    def test_rejects_repository_escape_and_nonportable_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "index.md"
            source.write_text(
                "# Index\n[out](../outside.md)\n[editor](vscode://settings)\n",
                encoding="utf-8",
            )

            messages = [
                issue.message
                for issue in check_docs.validate_markdown_file(source, root)
            ]

            self.assertIn("local target escapes the repository", messages)
            self.assertIn("unsupported link scheme: vscode", messages)

    def test_language_pages_require_counterparts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            english = root / "docs" / "en" / "guide.md"
            english.parent.mkdir(parents=True)
            english.write_text("# Guide\n", encoding="utf-8")

            issues = check_docs.validate_docs([english], root)

            self.assertEqual(len(issues), 1)
            self.assertIn("missing zh-TW counterpart", issues[0].message)

            chinese = root / "docs" / "zh-TW" / "guide.md"
            chinese.parent.mkdir(parents=True)
            chinese.write_text("# 指南\n", encoding="utf-8")

            self.assertEqual(check_docs.validate_docs([english, chinese], root), [])

    def test_tracked_markdown_files_handles_nul_unicode_paths(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="README.en.md\0docs/中文.md\0".encode(),
            stderr=b"",
        )
        with patch.object(subprocess, "run", return_value=completed):
            paths = check_docs.tracked_markdown_files(Path("/repo"))

        self.assertEqual(paths, [Path("/repo/README.en.md"), Path("/repo/docs/中文.md")])


if __name__ == "__main__":
    unittest.main()
