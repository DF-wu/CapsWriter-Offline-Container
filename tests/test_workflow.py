from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "v1-maintenance.yml"


class MaintenanceWorkflowTests(unittest.TestCase):
    def test_ci_uses_pinned_supported_runners_and_actions(self):
        source = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("ubuntu-24.04", source)
        self.assertIn("windows-2022", source)
        self.assertNotIn("-latest", source)
        self.assertIn(
            "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
            source,
        )
        self.assertIn(
            "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
            source,
        )
        self.assertIn("persist-credentials: false", source)
        self.assertIn("timeout-minutes: 15", source)
        self.assertIn('PYTHONDONTWRITEBYTECODE: "1"', source)
        self.assertIn('PYTHONNOUSERSITE: "1"', source)
        self.assertIn("--only-binary=:all:", source)
        self.assertIsNone(re.search(r"uses:\s+[^@\s]+@v\d+", source))

    def test_ci_covers_both_supported_python_versions(self):
        source = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn('- "3.10"', source)
        self.assertIn('- "3.12"', source)
        self.assertIn("python -m unittest discover -s tests", source)


if __name__ == "__main__":
    unittest.main()
