# coding: utf-8

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def dockerignore_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


class DockerignoreTest(unittest.TestCase):
    def test_server_context_excludes_local_models_and_generated_outputs(self) -> None:
        patterns = dockerignore_lines(ROOT / ".dockerignore")

        for pattern in (
            "models/",
            "client/cli/dist/",
            "client/web/node_modules/",
            "client/web/dist/",
            "client/web/coverage/",
            "client/web/.vite/",
            "client/web/playwright-report/",
            "client/web/test-results/",
            "client/web/.tmp/",
            "client/web/*.tsbuildinfo",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, patterns)

    def test_server_context_excludes_local_env_secrets_and_archives(self) -> None:
        patterns = dockerignore_lines(ROOT / ".dockerignore")

        for pattern in (
            ".env",
            ".env.*",
            ".envrc",
            "!.env.example",
            "*.key",
            "*.pem",
            "*.p12",
            "*.pfx",
            "*.crt",
            "*.csr",
            "id_rsa",
            "id_ed25519",
            "*.zip",
            "*.7z",
            "*.tar",
            "*.tar.gz",
            "*.tgz",
            "*.part",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, patterns)

    def test_server_context_no_longer_only_ignores_model_download_cache(self) -> None:
        patterns = dockerignore_lines(ROOT / ".dockerignore")

        self.assertIn("models/", patterns)
        self.assertNotIn("models/.downloads/", patterns)

    def test_web_context_excludes_local_env_and_secret_files(self) -> None:
        patterns = dockerignore_lines(ROOT / "client" / "web" / ".dockerignore")

        for pattern in (
            ".env",
            ".env.*",
            ".envrc",
            "!.env.example",
            "*.key",
            "*.pem",
            "*.p12",
            "*.pfx",
        ):
            with self.subTest(pattern=pattern):
                self.assertIn(pattern, patterns)


if __name__ == "__main__":
    unittest.main()
