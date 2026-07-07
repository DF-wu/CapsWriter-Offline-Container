# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class DocumentationTest(unittest.TestCase):
    def test_english_readme_uses_locked_web_dependency_install(self) -> None:
        source = (ROOT / "README.en.md").read_text(encoding="utf-8")

        self.assertIn("npm ci --no-audit --no-fund", source)
        self.assertIsNone(re.search(r"(?m)^\s*npm install\s*$", source))

    def test_readmes_point_http_api_users_to_compose_port_mapping(self) -> None:
        for filename in ("readme.md", "README.en.md"):
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                self.assertIn("ports:", source)
                self.assertNotIn("CAPSWRITER_HTTP_API_PORT` 那行的註解", source)
                self.assertNotIn("Expose port `6017`", source)


if __name__ == "__main__":
    unittest.main()
