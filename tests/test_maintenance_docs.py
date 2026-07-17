from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENGLISH = ROOT / "docs" / "en" / "maintenance.md"
TRADITIONAL_CHINESE = ROOT / "docs" / "zh-TW" / "maintenance.md"
HTTP_API_ENGLISH = ROOT / "docs" / "en" / "http-api.md"
HTTP_API_TRADITIONAL_CHINESE = ROOT / "docs" / "HTTP_API.md"


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
        self.assertIn("docs/HTTP_API.md", readme)
        self.assertIn("docs/en/http-api.md", english_readme)
        self.assertNotIn("docs/CHANGELOG.md", readme + english_readme)
        self.assertTrue((ROOT / "LICENSE").is_file())

    def test_http_api_guides_are_bilingual_and_keep_server_client_roles_separate(self):
        english = HTTP_API_ENGLISH.read_text(encoding="utf-8")
        traditional_chinese = HTTP_API_TRADITIONAL_CHINESE.read_text(encoding="utf-8")

        self.assertIn("[Traditional Chinese](../HTTP_API.md)", english)
        self.assertIn("[English](en/http-api.md)", traditional_chinese)
        for source, caller_label in (
            (english, "external api caller"),
            (traditional_chinese, "外部 api caller"),
        ):
            with self.subTest(caller_label=caller_label):
                folded = source.casefold()
                self.assertIn("server", folded)
                self.assertIn("start_client.py", folded)
                self.assertIn("websocket", folded)
                self.assertIn(caller_label, folded)
                self.assertIn("source", folded)

    def test_http_api_guides_distinguish_native_bind_from_compose_publish(self):
        for path in (HTTP_API_ENGLISH, HTTP_API_TRADITIONAL_CHINESE):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("CAPSWRITER_HTTP_API_BIND", source)
                self.assertIn("CAPSWRITER_HTTP_API_HOST_BIND", source)
                self.assertIn("CAPSWRITER_HTTP_API_KEY", source)
                self.assertIn("127.0.0.1", source)
                self.assertIn("0.0.0.0", source)
                self.assertIn("TLS", source)

        english = HTTP_API_ENGLISH.read_text(encoding="utf-8")
        traditional_chinese = HTTP_API_TRADITIONAL_CHINESE.read_text(encoding="utf-8")
        self.assertIn(
            "| `CAPSWRITER_HTTP_API_BIND` | `127.0.0.1` | "
            "`0.0.0.0` inside the container |",
            english,
        )
        self.assertIn(
            "| `CAPSWRITER_HTTP_API_HOST_BIND` | Not used | "
            "`127.0.0.1` on the host |",
            english,
        )
        self.assertIn(
            "| `CAPSWRITER_HTTP_API_BIND` | `127.0.0.1` | "
            "Container 內 `0.0.0.0` |",
            traditional_chinese,
        )
        self.assertIn(
            "| `CAPSWRITER_HTTP_API_HOST_BIND` | 不使用 | "
            "Host 上 `127.0.0.1` |",
            traditional_chinese,
        )

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

    def test_compose_passes_http_api_with_safe_network_defaults(self):
        compose_paths = (
            ROOT / "docker-compose.yml",
            ROOT / "docker-compose.example.yml",
        )
        required_compose_lines = (
            "CAPSWRITER_HTTP_API_ENABLE: ${CAPSWRITER_HTTP_API_ENABLE:-false}",
            "CAPSWRITER_HTTP_API_BIND: ${CAPSWRITER_HTTP_API_BIND:-0.0.0.0}",
            "CAPSWRITER_HTTP_API_PORT: ${CAPSWRITER_HTTP_API_PORT:-6017}",
            "CAPSWRITER_HTTP_API_KEY: ${CAPSWRITER_HTTP_API_KEY:-}",
            "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB: ${CAPSWRITER_HTTP_API_MAX_UPLOAD_MB:-100}",
            "CAPSWRITER_HTTP_API_TASK_TIMEOUT: ${CAPSWRITER_HTTP_API_TASK_TIMEOUT:-600}",
            '"${CAPSWRITER_HTTP_API_HOST_BIND:-127.0.0.1}:${CAPSWRITER_HTTP_API_PORT:-6017}:${CAPSWRITER_HTTP_API_PORT:-6017}"',
        )
        for path in compose_paths:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                for line in required_compose_lines:
                    self.assertIn(line, source)

        for path in (
            ROOT / ".env.example",
            ROOT / "docker" / "server" / ".env.example",
        ):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("CAPSWRITER_HTTP_API_ENABLE=false", source)
                self.assertIn("CAPSWRITER_HTTP_API_BIND=0.0.0.0", source)
                self.assertIn("CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1", source)

        for path in (ROOT / "readme.md", ROOT / "README.en.md"):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("CAPSWRITER_HTTP_API_BIND=0.0.0.0", source)
                self.assertIn("CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1", source)

    def test_local_image_build_excludes_mutable_hotword_files(self):
        dockerignore = {
            line.strip()
            for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        }
        for path in ("hot-server.txt", "hot.txt", "hot-rule.txt", "hot-rectify.txt"):
            self.assertIn(path, dockerignore)
        self.assertIn("!hot-server.example.txt", dockerignore)

    def test_legacy_image_publish_workflow_is_absent(self):
        self.assertFalse(
            (ROOT / ".github" / "workflows" / "publish-server-image.yml").exists()
        )

    def test_release_notes_do_not_claim_unrecorded_ci_gates(self):
        for path in (
            ROOT / "docs" / "en" / "release-notes.md",
            ROOT / "docs" / "zh-TW" / "release-notes.md",
        ):
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                folded = source.casefold()
                self.assertIn("compose", folded)
                self.assertIn("entrypoint", folded)
                self.assertNotIn("ruff", folded)
                self.assertNotIn("actionlint", folded)
                self.assertNotIn("dependency audit", folded)


if __name__ == "__main__":
    unittest.main()
