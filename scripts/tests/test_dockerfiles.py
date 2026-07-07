# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

EXPECTED_BASE_IMAGES = {
    "docker/server/Dockerfile": {
        "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04": (
            "85fb7ac694079fff1061a0140fd5b5a641997880e12112d92589c3bbb1e8b7ca"
        ),
    },
    "client/web/Dockerfile": {
        "node:24-alpine": "a0b9bf06e4e6193cf7a0f58816cc935ff8c2a908f81e6f1a95432d679c54fbfd",
        "nginx:1.27-alpine": "65645c7bb6a0661892a8b03b89d0743208a18dd2f3f17a54ef4b76fb8e2f2a10",
    },
}

EXPECTED_SERVER_BOOTSTRAP_PACKAGES = {
    "packaging": "26.2",
    "pip": "26.1.2",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
}


class DockerfileTest(unittest.TestCase):
    def test_base_images_are_pinned_to_digests(self) -> None:
        for filename, images in EXPECTED_BASE_IMAGES.items():
            source = (ROOT / filename).read_text(encoding="utf-8")
            from_lines = re.findall(r"(?m)^FROM\s+([^\s]+)", source)
            with self.subTest(filename=filename):
                self.assertTrue(from_lines)
                self.assertTrue(all("@sha256:" in line for line in from_lines))

            for image, digest in images.items():
                with self.subTest(filename=filename, image=image):
                    self.assertIn(f"{image}@sha256:{digest}", source)

    def test_server_python_bootstrap_packages_are_pinned(self) -> None:
        source = (ROOT / "docker/server/Dockerfile").read_text(encoding="utf-8")
        self.assertNotIn("pip install --upgrade pip setuptools wheel", source)
        for package, version in EXPECTED_SERVER_BOOTSTRAP_PACKAGES.items():
            with self.subTest(package=package):
                self.assertIn(f"{package}=={version}", source)


if __name__ == "__main__":
    unittest.main()
