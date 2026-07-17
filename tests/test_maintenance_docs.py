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
        english_readme = (ROOT / "README.en.md").read_text(encoding="utf-8")

        self.assertIn("docs/en/maintenance.md", readme)
        self.assertIn("docs/zh-TW/maintenance.md", readme)
        self.assertIn("README.en.md", readme)
        self.assertIn("readme.md", english_readme)

    def test_v1_docs_separate_server_and_client_deliverables(self):
        for path in (
            ROOT / "readme.md",
            ROOT / "README.en.md",
            ENGLISH,
            TRADITIONAL_CHINESE,
        ):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8").casefold()
                self.assertIn("server", source)
                self.assertIn("client", source)
                self.assertIn("start_client.py", source)
                self.assertIn("source", source)

    def test_v1_compose_never_defaults_to_v2_latest(self):
        paths = (
            ROOT / "docker-compose.yml",
            ROOT / "docker-compose.example.yml",
            ROOT / ".env.example",
            ROOT / "docker" / "server" / ".env.example",
        )
        for path in paths:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(
                    "CAPSWRITER_SERVER_IMAGE=ghcr.io/df-wu/capswriter-offline-server:latest",
                    source,
                )
                self.assertNotIn(
                    "${CAPSWRITER_SERVER_IMAGE:-ghcr.io/df-wu/capswriter-offline-server:latest}",
                    source,
                )

        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("capswriter-offline-v1-local:source", compose)
        self.assertIn("dockerfile: docker/server/Dockerfile", compose)
        for path in (ROOT / "readme.md", ROOT / "README.en.md"):
            self.assertIn("latest", path.read_text(encoding="utf-8").casefold())


if __name__ == "__main__":
    unittest.main()
