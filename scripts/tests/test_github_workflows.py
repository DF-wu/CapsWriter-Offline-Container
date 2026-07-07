# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


def read_workflow(filename: str) -> str:
    return (WORKFLOWS / filename).read_text(encoding="utf-8")


def workflow_job(source: str, job_name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:|\Z)",
        source,
    )
    if match is None:
        raise AssertionError(f"workflow job not found: {job_name}")
    return match.group("body")


class GitHubWorkflowTest(unittest.TestCase):
    def test_publish_workflows_serialize_per_ref(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            with self.subTest(filename=filename):
                source = read_workflow(filename)

                self.assertIn("concurrency:", source)
                self.assertIn("group: ${{ github.workflow }}-${{ github.ref }}", source)
                self.assertIn("cancel-in-progress: false", source)

    def test_publish_jobs_depend_on_verify_jobs(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            with self.subTest(filename=filename):
                source = read_workflow(filename)
                publish = workflow_job(source, "publish")

                self.assertIn("needs: verify", publish)

    def test_package_write_permission_is_limited_to_publish_jobs(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            with self.subTest(filename=filename):
                source = read_workflow(filename)
                workflow_preamble = source.split("\njobs:", 1)[0]
                verify = workflow_job(source, "verify")
                publish = workflow_job(source, "publish")

                self.assertNotIn("packages: write", workflow_preamble)
                self.assertNotIn("packages: write", verify)
                self.assertIn("permissions:", publish)
                self.assertIn("contents: read", publish)
                self.assertIn("packages: write", publish)

    def test_server_publish_runs_release_gate_before_push(self) -> None:
        source = read_workflow("publish-server-image.yml")
        verify = workflow_job(source, "verify")

        self.assertIn(
            "git fetch --no-tags --depth=1 upstream master:refs/remotes/upstream/master",
            verify,
        )
        self.assertIn("uses: actions/setup-node@v4", verify)
        self.assertIn('node-version: "24"', verify)
        self.assertIn("CAPSWRITER_UPSTREAM_BASE: upstream/master", verify)
        self.assertIn("python scripts/verify_all.py --skip-web", verify)

    def test_web_publish_runs_image_release_gate_before_push(self) -> None:
        source = read_workflow("publish-web-image.yml")
        verify = workflow_job(source, "verify")

        self.assertIn(
            "git fetch --no-tags --depth=1 upstream master:refs/remotes/upstream/master",
            verify,
        )
        self.assertIn("uses: actions/setup-node@v4", verify)
        self.assertIn('node-version: "24"', verify)
        self.assertIn("CAPSWRITER_UPSTREAM_BASE: upstream/master", verify)
        self.assertIn("python scripts/verify_all.py --docker-build-web", verify)


if __name__ == "__main__":
    unittest.main()
