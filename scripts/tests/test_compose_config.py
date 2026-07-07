# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

HTTP_ENV_KEYS = {
    "CAPSWRITER_HTTP_API_ENABLE",
    "CAPSWRITER_HTTP_API_BIND",
    "CAPSWRITER_HTTP_API_PORT",
    "CAPSWRITER_HTTP_API_KEY",
    "CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND",
    "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB",
    "CAPSWRITER_HTTP_API_TASK_TIMEOUT",
    "CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS",
    "CAPSWRITER_HTTP_API_CORS_ORIGINS",
    "CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS",
}


def active_yaml_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = re.match(r"\s+([A-Z0-9_]+):", line)
        if match:
            keys.add(match.group(1))
    return keys


class ComposeConfigTest(unittest.TestCase):
    def test_main_compose_passes_http_api_environment(self) -> None:
        keys = active_yaml_keys(ROOT / "docker-compose.yml")
        self.assertTrue(HTTP_ENV_KEYS <= keys)

    def test_example_compose_passes_http_api_environment(self) -> None:
        keys = active_yaml_keys(ROOT / "docker-compose.example.yml")
        self.assertTrue(HTTP_ENV_KEYS <= keys)

    def test_server_image_exposes_websocket_and_http_ports(self) -> None:
        dockerfile = (ROOT / "docker/server/Dockerfile").read_text(encoding="utf-8")
        self.assertRegex(dockerfile, r"(?m)^EXPOSE\s+6016\s+6017$")


if __name__ == "__main__":
    unittest.main()
