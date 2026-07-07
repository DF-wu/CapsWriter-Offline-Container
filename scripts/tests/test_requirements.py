# coding: utf-8

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTTP_API_REQUIREMENTS = {"fastapi", "uvicorn", "python-multipart"}


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("-"):
            continue
        name = entry.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].strip()
        names.add(name.casefold())
    return names


class RequirementsTest(unittest.TestCase):
    def test_server_requirements_include_http_api_runtime(self) -> None:
        for filename in ("requirements-server.txt", "requirements-server-docker.txt"):
            with self.subTest(filename=filename):
                names = requirement_names(ROOT / filename)
                self.assertTrue(
                    HTTP_API_REQUIREMENTS <= names,
                    f"{filename} missing {sorted(HTTP_API_REQUIREMENTS - names)}",
                )


if __name__ == "__main__":
    unittest.main()
