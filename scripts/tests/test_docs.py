# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAIRED_RELEASE_DOCS = (
    "README.md",
    "server-and-clients.md",
    "getting-started.md",
    "deployment.md",
    "web-console.md",
    "troubleshooting.md",
    "support-security.md",
    "release-notes.md",
)
RELEASE_DOC_ASSETS = {
    "getting-started.md": "../assets/tui-workbench.svg",
    "deployment.md": "../assets/web-console-architecture.svg",
    "web-console.md": "../assets/web-console-architecture.svg",
    "troubleshooting.md": "../assets/openai-api-lifecycle.svg",
    "support-security.md": "../assets/verification-pipeline.svg",
    "release-notes.md": "../assets/version-tracks.svg",
}


class DocumentationTest(unittest.TestCase):
    def test_root_readmes_present_the_cross_platform_product_truthfully(self) -> None:
        english = (ROOT / "README.en.md").read_text(encoding="utf-8")
        traditional = (ROOT / "readme.md").read_text(encoding="utf-8")

        for filename, source in (
            ("README.en.md", english),
            ("readme.md", traditional),
        ):
            with self.subTest(filename=filename):
                folded = source.casefold()
                self.assertIn("windows", folded)
                self.assertIn("linux", folded)
                self.assertIn("docker", folded)
                self.assertIn("openai", folded)
                self.assertIn("web", folded)
                self.assertIn("cli", folded)
                self.assertIn("tui", folded)

        for stale_claim in (
            "CapsWriter-Offline Linux Server Fork",
            "CapsWriter-Offline Linux Container Fork",
            "If you want the original Windows experience, use the upstream repository",
            "如果你要在 Windows 桌面用語音輸入，用上游",
        ):
            with self.subTest(stale_claim=stale_claim):
                self.assertNotIn(stale_claim, english + traditional)

    def test_paired_release_document_set_and_navigation_are_complete(self) -> None:
        for language, counterpart in (("en", "zh-TW"), ("zh-TW", "en")):
            language_root = ROOT / "docs" / language
            index = (language_root / "README.md").read_text(encoding="utf-8")
            for filename in PAIRED_RELEASE_DOCS:
                with self.subTest(language=language, filename=filename):
                    path = language_root / filename
                    self.assertTrue(path.is_file())
                    source = path.read_text(encoding="utf-8")
                    if filename != "README.md":
                        self.assertIn("README.md", source)
                        self.assertIn(f"../{counterpart}/{filename}", source)
                    self.assertIn(filename, index)

            for filename, asset in RELEASE_DOC_ASSETS.items():
                with self.subTest(language=language, asset=asset):
                    source = (language_root / filename).read_text(encoding="utf-8")
                    self.assertIn(asset, source)

    def test_readmes_make_every_major_user_path_discoverable(self) -> None:
        expected = {
            "README.en.md": (
                "docs/en/README.md",
                "docs/en/getting-started.md",
                "docs/en/deployment.md",
                "docs/en/troubleshooting.md",
                "docs/en/support-security.md",
                "docs/en/release-notes.md",
                "docs/en/desktop-portability.md",
                "docs/en/openai-api.md",
                "docs/en/tui.md",
            ),
            "readme.md": (
                "docs/zh-TW/README.md",
                "docs/zh-TW/getting-started.md",
                "docs/zh-TW/deployment.md",
                "docs/zh-TW/troubleshooting.md",
                "docs/zh-TW/support-security.md",
                "docs/zh-TW/release-notes.md",
                "docs/zh-TW/desktop-portability.md",
                "docs/zh-TW/openai-api.md",
                "docs/zh-TW/tui.md",
            ),
        }
        for filename, links in expected.items():
            source = (ROOT / filename).read_text(encoding="utf-8")
            for link in links:
                with self.subTest(filename=filename, link=link):
                    self.assertIn(link, source)

    def test_entry_docs_separate_server_and_client_roles(self) -> None:
        entry_docs = (
            ROOT / "README.en.md",
            ROOT / "readme.md",
            ROOT / "docs" / "en" / "README.md",
            ROOT / "docs" / "zh-TW" / "README.md",
            ROOT / "docs" / "en" / "server-and-clients.md",
            ROOT / "docs" / "zh-TW" / "server-and-clients.md",
        )

        for path in entry_docs:
            source = path.read_text(encoding="utf-8")
            folded = source.casefold()
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertIn("server", folded)
                self.assertIn("client", folded)
                self.assertIn("6016", source)
                self.assertIn("6017", source)

        for filename in ("README.en.md", "readme.md"):
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn(
                "docs/en/server-and-clients.md"
                if filename == "README.en.md"
                else "docs/zh-TW/server-and-clients.md",
                source,
            )

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

    def test_windows_release_docs_describe_the_real_package_gate(self) -> None:
        build_guide = (ROOT / "assets" / "BUILD_GUIDE.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "requirements-windows-build.lock",
            "--require-hashes",
            "--only-binary=:all:",
            "--no-binary=srt",
            "-m PyInstaller --clean --noconfirm build.spec",
            "windows-package",
            "windows-2022",
            "--artifact-self-check",
            "reparse",
            "models/",
            "logs/",
            "docs/en/desktop-portability.md",
            "docs/zh-TW/desktop-portability.md",
        ):
            with self.subTest(required=required):
                self.assertIn(required, build_guide)

        for stale in (
            "../config.py",
            "link_folders =",
            "core_server.py'",
            "core_client.py'",
            "pydantic', 'torch",
            "Python 版本**: 3.8+",
            "Sherpa-ONNX 版本**: 1.12.20",
        ):
            with self.subTest(stale=stale):
                self.assertNotIn(stale, build_guide)
        self.assertNotIn("6016", build_guide)
        self.assertNotIn("6017", build_guide)

        for language in ("en", "zh-TW"):
            desktop = (ROOT / "docs" / language / "desktop-portability.md").read_text(
                encoding="utf-8"
            )
            support = (ROOT / "docs" / language / "support-security.md").read_text(
                encoding="utf-8"
            )
            combined = desktop + support
            with self.subTest(language=language):
                self.assertIn("requirements-windows-build.lock", desktop)
                self.assertIn("windows-package", desktop)
                self.assertIn("--artifact-self-check", desktop)
                self.assertIn("reparse", combined.casefold())
                self.assertIn("DirectML", combined)
                self.assertIn("known audio", combined.replace("known-audio", "known audio"))

    def test_root_readmes_claim_packaged_windows_evidence_not_source_only(self) -> None:
        for filename in ("README.en.md", "readme.md"):
            source = (ROOT / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn("requirements-windows-build.lock", (
                    ROOT / "docs" / ("en" if filename == "README.en.md" else "zh-TW") /
                    "desktop-portability.md"
                ).read_text(encoding="utf-8"))
                self.assertIn("PyInstaller", source)
                self.assertIn("reparse", source.casefold())
                self.assertIn("ZIP", source)
                self.assertNotIn("PyInstaller source contract", source)


if __name__ == "__main__":
    unittest.main()
