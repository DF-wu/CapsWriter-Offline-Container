from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENGLISH = ROOT / "docs" / "en" / "maintenance.md"
TRADITIONAL_CHINESE = ROOT / "docs" / "zh-TW" / "maintenance.md"


class MaintenanceDocumentationTests(unittest.TestCase):
    def test_bilingual_counterparts_and_cross_links_exist(self):
        english = ENGLISH.read_text(encoding="utf-8")
        traditional_chinese = TRADITIONAL_CHINESE.read_text(encoding="utf-8")

        self.assertIn("[繁體中文版](../zh-TW/maintenance.md)", english)
        self.assertIn("[English version](../en/maintenance.md)", traditional_chinese)

    def test_both_languages_record_branch_version_and_support_contract(self):
        for path in (ENGLISH, TRADITIONAL_CHINESE):
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                self.assertIn("maintenance/v1", text)
                self.assertIn("archive/v1-legacy", text)
                self.assertIn("2.5-alpha", text)
                self.assertIn("Python 3.10", text)
                self.assertIn("Python 3.12", text)

    def test_readme_links_both_counterparts(self):
        readme = (ROOT / "readme.md").read_text(encoding="utf-8")

        self.assertIn("docs/en/maintenance.md", readme)
        self.assertIn("docs/zh-TW/maintenance.md", readme)


if __name__ == "__main__":
    unittest.main()
