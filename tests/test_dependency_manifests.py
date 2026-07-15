from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyManifestTests(unittest.TestCase):
    def test_docker_runtime_contains_optional_http_api_dependencies(self):
        requirements = (ROOT / "requirements-server-docker.txt").read_text(
            encoding="utf-8"
        )

        for package in ("fastapi==", "uvicorn[standard]==", "python-multipart=="):
            with self.subTest(package=package):
                self.assertIn(package, requirements)

    def test_maintenance_dependencies_are_exactly_pinned(self):
        lines = (ROOT / "requirements-maintenance.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        packages = [line for line in lines if line and not line.startswith("#")]

        self.assertTrue(packages)
        self.assertTrue(all("==" in package for package in packages))

    def test_maintenance_server_pins_match_docker_runtime(self):
        docker = (ROOT / "requirements-server-docker.txt").read_text(
            encoding="utf-8"
        )
        maintenance = (ROOT / "requirements-maintenance.txt").read_text(
            encoding="utf-8"
        )

        for version in (
            "rich==14.3.3",
            "websockets==16.0",
            "numpy==1.26.4",
            "fastapi==0.139.0",
            "python-multipart==0.0.31",
        ):
            with self.subTest(version=version):
                self.assertIn(version, docker)
                self.assertIn(version, maintenance)
        self.assertIn("uvicorn[standard]==0.32.1", docker)
        self.assertIn("uvicorn==0.32.1", maintenance)
        self.assertIn("Pillow==12.3.0", docker)


if __name__ == "__main__":
    unittest.main()
