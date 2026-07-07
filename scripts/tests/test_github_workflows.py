# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PINNED_ACTIONS = {
    "actions/checkout": "34e114876b0b11c390a56381ad16ebd13914f8d5",
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/setup-node": "49933ea5288caeca8642d1e84afbd3f7d6820020",
    "docker/setup-buildx-action": "8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
    "docker/login-action": "c94ce9fb468520275223c153574b00df6fe4bcc9",
    "docker/metadata-action": "c299e40c65443455700f0fdfc63efafe5b349051",
    "docker/build-push-action": "10e90e3645eae34f1e60eeb005ba3a3d33f178e8",
}
PINNED_RUNNER = "ubuntu-24.04"


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
    def test_workflows_pin_runner_images(self) -> None:
        for path in sorted(WORKFLOWS.glob("*.yml")):
            source = path.read_text(encoding="utf-8")
            runners = re.findall(r"(?m)^\s+runs-on:\s+([^\s#]+)", source)
            with self.subTest(filename=path.name):
                self.assertTrue(runners)
                self.assertNotIn("ubuntu-latest", runners)
                self.assertTrue(all(runner == PINNED_RUNNER for runner in runners))

    def test_workflows_pin_third_party_actions_to_full_shas(self) -> None:
        for path in sorted(WORKFLOWS.glob("*.yml")):
            with self.subTest(filename=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"uses:\s+[^@\s]+@v\d+(?:\s|$)", source))

        for filename in ("ci.yml", "publish-server-image.yml", "publish-web-image.yml"):
            source = read_workflow(filename)
            for action, sha in PINNED_ACTIONS.items():
                if action.startswith("docker/") and filename == "ci.yml":
                    continue
                with self.subTest(filename=filename, action=action):
                    self.assertIn(f"uses: {action}@{sha}", source)

    def test_checkout_steps_do_not_persist_credentials(self) -> None:
        checkout = f"actions/checkout@{PINNED_ACTIONS['actions/checkout']}"
        for path in sorted(WORKFLOWS.glob("*.yml")):
            source = path.read_text(encoding="utf-8")
            blocks = re.findall(
                rf"(?ms)^      - name: Checkout\n(?P<body>.*?)(?=^      - name:|\Z)",
                source,
            )
            with self.subTest(filename=path.name):
                self.assertTrue(blocks)
            for index, block in enumerate(blocks):
                with self.subTest(filename=path.name, checkout=index):
                    self.assertIn(f"uses: {checkout}", block)
                    self.assertIn("persist-credentials: false", block)

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

    def test_publish_images_include_supply_chain_attestations(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            with self.subTest(filename=filename):
                source = read_workflow(filename)
                publish = workflow_job(source, "publish")

                self.assertIn(
                    f"uses: docker/build-push-action@{PINNED_ACTIONS['docker/build-push-action']}",
                    publish,
                )
                self.assertIn("provenance: true", publish)
                self.assertIn("sbom: true", publish)

    def test_server_publish_runs_release_gate_before_push(self) -> None:
        source = read_workflow("publish-server-image.yml")
        verify = workflow_job(source, "verify")

        self.assertIn(
            "git fetch --no-tags --depth=1 upstream master:refs/remotes/upstream/master",
            verify,
        )
        self.assertIn(f"uses: actions/setup-node@{PINNED_ACTIONS['actions/setup-node']}", verify)
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
        self.assertIn(f"uses: actions/setup-node@{PINNED_ACTIONS['actions/setup-node']}", verify)
        self.assertIn('node-version: "24"', verify)
        self.assertIn("CAPSWRITER_UPSTREAM_BASE: upstream/master", verify)
        self.assertIn("python scripts/verify_all.py --docker-build-web", verify)


if __name__ == "__main__":
    unittest.main()
