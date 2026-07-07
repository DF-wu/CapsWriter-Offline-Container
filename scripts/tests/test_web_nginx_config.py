# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WEB_DOCKERFILE = ROOT / "client" / "web" / "Dockerfile"
WEB_NGINX_CONFIG = ROOT / "client" / "web" / "deploy" / "nginx.conf"
WEB_SECURITY_HEADERS = ROOT / "client" / "web" / "deploy" / "security-headers.conf"
VERIFY_ALL = ROOT / "scripts" / "verify_all.py"


class WebNginxConfigTest(unittest.TestCase):
    def test_security_headers_are_defined_for_static_container(self) -> None:
        source = WEB_SECURITY_HEADERS.read_text(encoding="utf-8")

        self.assertIn('X-Content-Type-Options "nosniff" always', source)
        self.assertIn('X-Frame-Options "DENY" always', source)
        self.assertIn('Referrer-Policy "no-referrer" always', source)
        self.assertIn("Permissions-Policy", source)
        self.assertIn("Content-Security-Policy", source)
        self.assertIn("frame-ancestors 'none'", source)
        self.assertIn("object-src 'none'", source)
        self.assertIn("connect-src 'self' http: https:", source)

    def test_nginx_locations_include_security_headers(self) -> None:
        source = WEB_NGINX_CONFIG.read_text(encoding="utf-8")

        for location in (r"= /health", r"= /config\.js", r"/assets/", r"/"):
            with self.subTest(location=location):
                block = re.search(rf"location {location} \{{(?P<body>.*?)\n    \}}", source, re.S)
                self.assertIsNotNone(block)
                self.assertIn(
                    "include /etc/nginx/snippets/capswriter-security-headers.conf;",
                    block.group("body"),
                )

    def test_web_image_copies_security_header_snippet(self) -> None:
        source = WEB_DOCKERFILE.read_text(encoding="utf-8")

        self.assertIn("deploy/security-headers.conf", source)
        self.assertIn("/etc/nginx/snippets/capswriter-security-headers.conf", source)

    def test_web_docker_smoke_checks_security_headers(self) -> None:
        source = VERIFY_ALL.read_text(encoding="utf-8")

        self.assertIn("X-Content-Type-Options: nosniff", source)
        self.assertIn("X-Frame-Options: DENY", source)
        self.assertIn("Content-Security-Policy:", source)


if __name__ == "__main__":
    unittest.main()
