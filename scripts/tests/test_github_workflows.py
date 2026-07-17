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
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",
    "docker/setup-buildx-action": "8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
    "docker/login-action": "c94ce9fb468520275223c153574b00df6fe4bcc9",
    "docker/metadata-action": "c299e40c65443455700f0fdfc63efafe5b349051",
    "docker/build-push-action": "10e90e3645eae34f1e60eeb005ba3a3d33f178e8",
}
PINNED_RUNNERS = {"ubuntu-24.04", "windows-2022"}


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
            runners = re.findall(r"(?m)^\s+runs-on:\s+(.+?)\s*$", source)
            with self.subTest(filename=path.name):
                self.assertTrue(runners)
                self.assertNotIn("ubuntu-latest", source)
                self.assertNotIn("windows-latest", source)
                for runner in runners:
                    runner = runner.split("#", 1)[0].strip()
                    if runner == "${{ matrix.os }}":
                        matrix = re.search(
                            r"(?m)^\s+os:\s+\[(?P<values>[^]]+)\]",
                            source,
                        )
                        self.assertIsNotNone(matrix)
                        values = {
                            value.strip().strip("'\"")
                            for value in matrix.group("values").split(",")
                        }
                        self.assertTrue(values)
                        self.assertTrue(values <= PINNED_RUNNERS)
                    else:
                        self.assertIn(runner, PINNED_RUNNERS)

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
                if action == "actions/upload-artifact":
                    continue
                with self.subTest(filename=filename, action=action):
                    self.assertIn(f"uses: {action}@{sha}", source)

        portability = read_workflow("portability.yml")
        for action in (
            "actions/checkout",
            "actions/setup-python",
            "actions/upload-artifact",
        ):
            self.assertIn(f"uses: {action}@{PINNED_ACTIONS[action]}", portability)

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

    def test_ci_runs_api_contract_in_an_isolated_pinned_environment(self) -> None:
        source = read_workflow("ci.yml")
        contract = workflow_job(source, "api-contract")

        self.assertIn(
            f"uses: actions/setup-python@{PINNED_ACTIONS['actions/setup-python']}",
            contract,
        )
        self.assertIn('python-version: "3.12"', contract)
        self.assertIn("timeout-minutes: 15", contract)
        self.assertIn('PYTHONNOUSERSITE: "1"', contract)
        self.assertIn('python -m venv "$RUNNER_TEMP/api-contract"', contract)
        self.assertIn(
            '"$RUNNER_TEMP/api-contract/bin/python" -m pip install',
            contract,
        )
        self.assertIn("--require-hashes", contract)
        self.assertIn("--only-binary=:all:", contract)
        self.assertIn("--requirement requirements-api-test.lock", contract)
        self.assertIn(
            '"$RUNNER_TEMP/api-contract/bin/python" scripts/verify_api_contract.py',
            contract,
        )

    def test_portability_matrix_covers_supported_os_and_python_versions(self) -> None:
        source = read_workflow("portability.yml")
        job = workflow_job(source, "core-cli")

        self.assertIn("runs-on: ${{ matrix.os }}", job)
        self.assertIn("os: [ubuntu-24.04, windows-2022]", job)
        self.assertIn('python-version: ["3.10", "3.12"]', job)
        self.assertIn("fail-fast: false", job)
        self.assertIn("timeout-minutes: 15", job)
        self.assertIn('PYTHONNOUSERSITE: "1"', job)
        self.assertIn("python -m compileall -q", job)
        self.assertIn("compile(Path('build.spec').read_text", job)
        self.assertIn("scripts.tests.test_desktop_portability", job)
        self.assertIn("scripts.tests.test_protocol", job)
        self.assertIn("python client/cli/scripts/verify.py", job)
        self.assertIn("docs/en/desktop-portability.md", job)
        self.assertIn("docs/zh-TW/desktop-portability.md", job)

    def test_portability_builds_relocates_and_smokes_windows_package(self) -> None:
        source = read_workflow("portability.yml")
        job = workflow_job(source, "windows-package")

        self.assertIn("runs-on: windows-2022", job)
        self.assertIn("timeout-minutes: 60", job)
        self.assertIn('python-version: "3.12"', job)
        self.assertIn('PYTHONNOUSERSITE: "1"', job)
        self.assertIn("python -m pip install", job)
        self.assertIn("--no-deps", job)
        self.assertIn(
            "--requirement requirements-windows-build-bootstrap.lock",
            job,
        )
        self.assertIn("--require-hashes", job)
        self.assertIn("--only-binary=:all:", job)
        self.assertIn("--no-binary=srt", job)
        self.assertIn("--no-build-isolation", job)
        self.assertIn("--requirement requirements-windows-build.lock", job)
        self.assertIn("python -m PyInstaller --clean --noconfirm build.spec", job)
        self.assertIn("$env:RUNNER_TEMP", job)
        self.assertIn("Compress-Archive", job)
        self.assertIn("Get-FileHash -LiteralPath $archive -Algorithm SHA256", job)
        self.assertIn("Expand-Archive", job)
        self.assertIn("ReparsePoint", job)
        self.assertIn("@('models', 'logs')", job)
        self.assertIn("--artifact-self-check", job)
        self.assertIn("Invoke-ArtifactSelfCheck 'start_server.exe'", job)
        self.assertIn("Invoke-ArtifactSelfCheck 'start_client.exe'", job)
        self.assertIn(
            f"uses: actions/upload-artifact@{PINNED_ACTIONS['actions/upload-artifact']}",
            job,
        )
        self.assertNotIn("6016", job)
        self.assertNotIn("6017", job)

    def test_ci_runs_four_leg_tui_lock_and_strict_suite(self) -> None:
        source = read_workflow("ci.yml")
        tui = workflow_job(source, "tui")

        self.assertIn(
            "name: ${{ matrix.os }} / TUI Python ${{ matrix.python-version }}",
            tui,
        )
        self.assertIn("runs-on: ${{ matrix.os }}", tui)
        self.assertIn("os: [ubuntu-24.04, windows-2022]", tui)
        self.assertIn("venv_python: ./.venv-tui/bin/python", tui)
        self.assertIn("venv_python: ./.venv-tui/Scripts/python.exe", tui)
        self.assertIn(
            f"uses: actions/setup-python@{PINNED_ACTIONS['actions/setup-python']}",
            tui,
        )
        self.assertIn('python-version: ["3.10", "3.12"]', tui)
        self.assertIn("fail-fast: false", tui)
        self.assertIn("timeout-minutes: 15", tui)
        self.assertIn('PYTHONDONTWRITEBYTECODE: "1"', tui)
        self.assertIn('PYTHONNOUSERSITE: "1"', tui)
        self.assertIn("run: python -m venv .venv-tui", tui)
        self.assertIn("${{ matrix.venv_python }} -m pip install", tui)
        self.assertIn("--require-hashes", tui)
        self.assertIn("--only-binary=:all:", tui)
        self.assertIn("--requirement requirements-tui.lock", tui)
        self.assertIn(
            "run: ${{ matrix.venv_python }} scripts/verify_tui.py",
            tui,
        )
        self.assertNotIn("exclude:", tui)
        self.assertNotRegex(tui, r"(?m)^\s+if:")

    def test_publish_workflows_serialize_runs_per_ref(self) -> None:
        expected_groups = {
            "publish-server-image.yml": "publish-server-image-${{ github.ref }}",
            "publish-web-image.yml": "publish-web-image-${{ github.ref }}",
        }
        for filename, group in expected_groups.items():
            with self.subTest(filename=filename):
                source = read_workflow(filename)

                self.assertIn("concurrency:", source)
                self.assertIn(f"group: {group}", source)
                self.assertNotIn("github.workflow", source)
                self.assertIn("cancel-in-progress: false", source)

    def test_workflow_jobs_have_explicit_deadlines(self) -> None:
        expected = {
            "ci.yml": {"verify": 45, "api-contract": 15, "tui": 15},
            "portability.yml": {"core-cli": 15, "windows-package": 60},
            "publish-server-image.yml": {
                "verify": 45,
                "publish": 120,
                "promote": 10,
            },
            "publish-web-image.yml": {
                "verify": 45,
                "publish": 45,
                "promote": 10,
            },
        }
        for filename, jobs in expected.items():
            source = read_workflow(filename)
            for job_name, minutes in jobs.items():
                with self.subTest(filename=filename, job=job_name):
                    self.assertIn(
                        f"timeout-minutes: {minutes}",
                        workflow_job(source, job_name),
                    )

    def test_publish_jobs_reject_stale_master_commits(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            source = read_workflow(filename)
            for job_name in ("verify", "publish", "promote"):
                with self.subTest(filename=filename, job=job_name):
                    job = workflow_job(source, job_name)
                    self.assertIn("Require current master tip", job)
                    self.assertIn(
                        "git ls-remote --exit-code origin refs/heads/master",
                        job,
                    )
                    self.assertIn('[ "$current_sha" != "$GITHUB_SHA" ]', job)

    def test_publish_uses_guarded_digest_promotion_for_latest(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            source = read_workflow(filename)
            publish = workflow_job(source, "publish")
            promote = workflow_job(source, "promote")
            with self.subTest(filename=filename):
                self.assertIn("type=raw,value=sha-${{ github.sha }}", publish)
                self.assertNotIn("type=raw,value=latest", publish)
                self.assertIn("flavor: latest=false", publish)
                self.assertIn("Resolve lowercase image name", publish)
                self.assertIn("${GITHUB_REPOSITORY_OWNER,,}", publish)
                self.assertIn("images: ${{ steps.image.outputs.name }}", publish)
                self.assertIn("id: build", publish)
                self.assertIn("image: ${{ steps.image.outputs.name }}", publish)
                self.assertIn("digest: ${{ steps.build.outputs.digest }}", publish)
                self.assertNotIn(":latest", publish)
                self.assertIn("needs: publish", promote)
                self.assertIn("IMAGE: ${{ needs.publish.outputs.image }}", promote)
                self.assertIn("DIGEST: ${{ needs.publish.outputs.digest }}", promote)
                self.assertIn("docker buildx imagetools create", promote)
                self.assertIn('^sha256:[0-9a-f]{64}$', promote)
                self.assertIn(
                    '--prefer-index=false --tag "$IMAGE:latest" "$IMAGE@$DIGEST"',
                    promote,
                )
                self.assertIn("docker buildx imagetools inspect", promote)
                self.assertIn('[ "$promoted_digest" != "$DIGEST" ]', promote)
                guard_position = promote.index(
                    "Require current master tip before latest promotion"
                )
                promote_position = promote.index("Promote verified digest to latest")
                self.assertLess(guard_position, promote_position)

    def test_publish_jobs_depend_on_verify_jobs(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            with self.subTest(filename=filename):
                source = read_workflow(filename)
                publish = workflow_job(source, "publish")
                promote = workflow_job(source, "promote")

                self.assertIn("needs: verify", publish)
                self.assertIn("needs: publish", promote)

    def test_publish_workflows_reject_non_master_manual_dispatches(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            with self.subTest(filename=filename):
                source = read_workflow(filename)
                for job_name in ("verify", "publish", "promote"):
                    job = workflow_job(source, job_name)
                    self.assertIn(
                        "if: github.ref == 'refs/heads/master'",
                        job,
                    )

    def test_package_write_permission_is_limited_to_publish_jobs(self) -> None:
        for filename in ("publish-server-image.yml", "publish-web-image.yml"):
            with self.subTest(filename=filename):
                source = read_workflow(filename)
                workflow_preamble = source.split("\njobs:", 1)[0]
                verify = workflow_job(source, "verify")
                publish = workflow_job(source, "publish")
                promote = workflow_job(source, "promote")

                self.assertNotIn("packages: write", workflow_preamble)
                self.assertNotIn("packages: write", verify)
                self.assertIn("permissions:", publish)
                self.assertIn("contents: read", publish)
                self.assertIn("packages: write", publish)
                self.assertIn("permissions:", promote)
                self.assertIn("contents: read", promote)
                self.assertIn("packages: write", promote)

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
        publish = workflow_job(source, "publish")

        self.assertIn(
            "git fetch --no-tags --depth=1 upstream master:refs/remotes/upstream/master",
            verify,
        )
        self.assertIn(f"uses: actions/setup-node@{PINNED_ACTIONS['actions/setup-node']}", verify)
        self.assertIn('node-version: "24"', verify)
        self.assertIn("CAPSWRITER_UPSTREAM_BASE: upstream/master", verify)
        self.assertIn("python scripts/verify_all.py --skip-web", verify)
        self.assertIn('python -m venv "$RUNNER_TEMP/api-contract"', verify)
        self.assertIn("--require-hashes", verify)
        self.assertIn("--only-binary=:all:", verify)
        self.assertIn("--requirement requirements-api-test.lock", verify)
        self.assertIn("scripts/verify_api_contract.py", verify)
        self.assertIn('PYTHONNOUSERSITE: "1"', verify)
        self.assertIn("Verify pushed server image digest", publish)
        self.assertIn("platforms: linux/amd64", publish)
        self.assertIn("candidate=\"$IMAGE@$DIGEST\"", publish)
        self.assertIn('docker pull "$candidate"', publish)
        self.assertIn("--entrypoint python \"$candidate\" -m pip check", publish)
        self.assertIn("sentencepiece", publish)
        self.assertIn("from fork_server.bootstrap import", publish)
        self.assertIn("os.getuid() == 0", publish)
        self.assertIn("--entrypoint sh \"$candidate\" -n", publish)
        self.assertLess(
            publish.index("Build and push immutable image"),
            publish.index("Verify pushed server image digest"),
        )

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
