# coding: utf-8

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTTP_API_REQUIREMENTS = {"fastapi", "uvicorn", "python-multipart"}
DOCKERFILE = ROOT / "docker" / "server" / "Dockerfile"


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("-"):
            continue
        name = entry.split(";", 1)[0].split("[", 1)[0].split("=", 1)[0].strip()
        names.add(name.casefold())
    return names


def requirement_entries(path: Path) -> list[str]:
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        entries.append(entry)
    return entries


class RequirementsTest(unittest.TestCase):
    def test_server_requirements_include_http_api_runtime(self) -> None:
        for filename in ("requirements-server.txt", "requirements-server-docker.txt"):
            with self.subTest(filename=filename):
                names = requirement_names(ROOT / filename)
                self.assertTrue(
                    HTTP_API_REQUIREMENTS <= names,
                    f"{filename} missing {sorted(HTTP_API_REQUIREMENTS - names)}",
                )

    def test_docker_server_lock_is_pinned_and_used_by_image_build(self) -> None:
        lock_path = ROOT / "requirements-server-docker.lock"
        lock_entries = requirement_entries(lock_path)
        lock_names = requirement_names(lock_path)
        declared_names = requirement_names(ROOT / "requirements-server-docker.txt")

        self.assertTrue(declared_names <= lock_names)
        for entry in lock_entries:
            with self.subTest(entry=entry):
                self.assertRegex(
                    entry,
                    r"^[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*\])?==[A-Za-z0-9_.!+~-]+$",
                )

        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("COPY requirements-server-docker.txt requirements-server-docker.lock /app/", dockerfile)
        self.assertIn(
            "python -m pip install --no-build-isolation -r /app/requirements-server-docker.lock",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
