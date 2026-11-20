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
    "actions/download-artifact": "d3f86a106a0bac45b974a628896c90dbdf5c8093",
    "docker/setup-buildx-action": "8d2750c68a42422c14e847fe6c8ac0403b4cbd6f",
    "docker/login-action": "c94ce9fb468520275223c153574b00df6fe4bcc9",
    "docker/metadata-action": "c299e40c65443455700f0fdfc63efafe5b349051",
    "docker/build-push-action": "10e90e3645eae34f1e60eeb005ba3a3d33f178e8",
}
PINNED_RUNNERS = {"ubuntu-24.04", "windows-2022"}


def read_workflow(filename: str) -> str:
    return (WORKFLOWS / filename).read_text(encoding="utf-8")


def run_blocks(source: str) -> str:
    """Return the bodies of every multi-line ``run:`` script in a workflow."""

    return "\n".join(
        re.findall(r"(?ms)run: [|>]-?\n(.*?)(?=^\s+- name:|\Z)", source)
    )


def unittest_modules(job: str) -> list[str]:
    return re.findall(r"(?m)^\s+(scripts\.tests\.test_\w+)\s*$", job)


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
                if action in ("actions/upload-artifact", "actions/download-artifact"):
                    continue
                with self.subTest(filename=filename, action=action):
                    self.assertIn(f"uses: {action}@{sha}", source)

        package = read_workflow("windows-package.yml")
        for action in (
            "actions/checkout",
            "actions/setup-python",
            "actions/upload-artifact",
        ):
            self.assertIn(f"uses: {action}@{PINNED_ACTIONS[action]}", package)

        release = read_workflow("release.yml")
        for action in (
            "actions/checkout",
            "actions/setup-python",
            "actions/setup-node",
            "actions/download-artifact",
        ):
            self.assertIn(f"uses: {action}@{PINNED_ACTIONS[action]}", release)

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
        self.assertIn("--requirement requirements/api-test.lock", contract)
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

    def test_portability_builds_both_windows_package_variants(self) -> None:
        source = read_workflow("portability.yml")
        job = workflow_job(source, "windows-package")

        self.assertIn("uses: ./.github/workflows/windows-package.yml", job)
        self.assertIn("variant: [full, client]", job)
        self.assertIn("fail-fast: false", job)
        self.assertIn("variant: ${{ matrix.variant }}", job)

    def test_windows_package_builds_relocates_and_smokes_selected_variant(self) -> None:
        source = read_workflow("windows-package.yml")
        trigger = source.split("\njobs:", 1)[0]
        job = workflow_job(source, "package")

        self.assertIn("workflow_call:", trigger)
        self.assertNotIn("push:", trigger)
        self.assertNotIn("pull_request:", trigger)
        self.assertIn("runs-on: windows-2022", job)
        self.assertIn("timeout-minutes: 60", job)
        self.assertIn('python-version: "3.12"', job)
        self.assertIn('PYTHONNOUSERSITE: "1"', job)
        self.assertIn("CAPSWRITER_BUILD_VARIANT: ${{ inputs.variant }}", job)
        self.assertIn("ref: ${{ inputs.ref }}", job)
        # Inputs reach scripts only through the environment, never inline.
        self.assertNotIn("${{ inputs.", run_blocks(job))
        self.assertIn('throw "Unsupported package variant', job)
        self.assertIn("'CapsWriter-Offline-windows-x86_64'", job)
        self.assertIn("'CapsWriter-Offline-Client-windows-x86_64'", job)
        self.assertIn("python -m pip install", job)
        self.assertIn("--no-deps", job)
        self.assertIn(
            "--requirement requirements/windows-build-bootstrap.lock",
            job,
        )
        self.assertIn("--require-hashes", job)
        self.assertIn("--only-binary=:all:", job)
        self.assertIn("--no-binary=srt", job)
        self.assertIn("--no-build-isolation", job)
        self.assertIn("--requirement requirements/windows-build.lock", job)
        self.assertIn("python -m PyInstaller --clean --noconfirm build.spec", job)
        self.assertIn("$env:RUNNER_TEMP", job)
        self.assertIn("Compress-Archive", job)
        self.assertIn("Get-FileHash -LiteralPath $archive -Algorithm SHA256", job)
        self.assertIn("Expand-Archive", job)
        self.assertIn("ReparsePoint", job)
        self.assertIn("$mutable = @('logs')", job)
        self.assertIn("$mutable += @('models')", job)
        self.assertIn("'internal\\capswriter-package.json'", job)
        client_forbidden = job.split("$forbidden = @(", 2)[2]
        for forbidden in ("'start_server.exe'", "'core\\server'", "'internal\\onnxruntime'"):
            with self.subTest(forbidden=forbidden):
                self.assertIn(forbidden, client_forbidden)
        self.assertIn("--artifact-self-check", job)
        self.assertIn('"variant":"', job)
        self.assertIn("Invoke-ArtifactSelfCheck $executable", job)
        self.assertIn("$executables = @('start_server.exe') + $executables", job)
        self.assertIn("Invoke-Packaged 'start_client.exe' @('--status')", job)
        self.assertIn("$status.ExitCode -ne 1", job)
        self.assertIn(
            f"uses: actions/upload-artifact@{PINNED_ACTIONS['actions/upload-artifact']}",
            job,
        )
        self.assertIn("name: ${{ env.CAPSWRITER_ARCHIVE_NAME }}", job)
        self.assertIn("retention-days: ${{ inputs.retention-days }}", job)
        self.assertNotIn("6016", job)
        self.assertNotIn("6017", job)

    def test_release_runs_only_for_fork_tags_on_master(self) -> None:
        source = read_workflow("release.yml")
        trigger = source.split("\njobs:", 1)[0]
        resolve = workflow_job(source, "resolve")

        self.assertIn('      - "fork-v*"', trigger)
        self.assertIn("workflow_dispatch:", trigger)
        self.assertNotIn("branches:", trigger)
        self.assertNotIn("pull_request", trigger)
        self.assertIn("permissions:\n  contents: read", trigger)
        self.assertIn("cancel-in-progress: false", trigger)
        self.assertIn("fetch-depth: 0", resolve)
        self.assertIn(r"^fork-v[0-9]+\.[0-9]+\.[0-9]+", resolve)
        self.assertIn('git merge-base --is-ancestor "$sha" origin/master', resolve)
        self.assertIn("INPUT_TAG: ${{ inputs.tag }}", resolve)
        self.assertNotIn("${{ inputs.", run_blocks(source))
        self.assertNotIn("${{ github.ref_name }}", run_blocks(source))

    def test_release_gates_packages_on_linux_and_windows_tests(self) -> None:
        source = read_workflow("release.yml")
        verify = workflow_job(source, "verify")
        desktop = workflow_job(source, "desktop-tests")
        package = workflow_job(source, "windows-package")

        for name, job in (("verify", verify), ("desktop-tests", desktop)):
            with self.subTest(job=name):
                self.assertIn("needs: resolve", job)
                self.assertIn("ref: ${{ needs.resolve.outputs.sha }}", job)
        self.assertIn("python scripts/verify_all.py --skip-web", verify)
        self.assertIn("CAPSWRITER_UPSTREAM_BASE: upstream/master", verify)
        self.assertIn("--requirement requirements/api-test.lock", verify)
        self.assertIn("scripts/verify_api_contract.py", verify)
        self.assertIn("runs-on: windows-2022", desktop)
        self.assertIn("python client/cli/scripts/verify.py", desktop)

        # The Windows release gate runs exactly the portability unit suites.
        portability = workflow_job(read_workflow("portability.yml"), "core-cli")
        self.assertTrue(unittest_modules(desktop))
        self.assertEqual(unittest_modules(desktop), unittest_modules(portability))

        self.assertIn("needs: [resolve, verify, desktop-tests]", package)
        self.assertIn("uses: ./.github/workflows/windows-package.yml", package)
        self.assertIn("variant: [full, client]", package)
        self.assertIn("ref: ${{ needs.resolve.outputs.sha }}", package)

    def test_release_only_drafts_and_never_edits_published_releases(self) -> None:
        source = read_workflow("release.yml")
        publish = workflow_job(source, "publish-draft")

        self.assertEqual(source.count("contents: write"), 1)
        self.assertIn("contents: write", publish)
        self.assertIn("needs: [resolve, windows-package]", publish)
        self.assertIn("pattern: CapsWriter-Offline-*windows-x86_64", publish)
        self.assertIn("merge-multiple: true", publish)
        self.assertIn('git archive --format=zip --prefix="CapsWriter-Offline-$TAG/"', publish)
        self.assertIn("CapsWriter-Offline-source.zip", publish)
        self.assertIn("CapsWriter-Offline-Client-windows-x86_64.zip", publish)
        self.assertIn("sha256sum -- *.zip > SHA256SUMS", publish)
        self.assertIn("sha256sum -c SHA256SUMS", publish)
        self.assertIn('gh release create "$TAG" --draft --verify-tag', publish)
        self.assertIn('[ "$is_draft" != "true" ]', publish)
        self.assertIn("refusing to modify a published release", publish)
        self.assertNotIn("gh release edit", publish)
        self.assertNotIn("--draft=false", publish)

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
        self.assertIn("--requirement requirements/tui.lock", tui)
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
            "portability.yml": {"core-cli": 15},
            "windows-package.yml": {"package": 60},
            "release.yml": {
                "resolve": 5,
                "verify": 45,
                "desktop-tests": 15,
                "publish-draft": 15,
            },
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
        self.assertIn("--requirement requirements/api-test.lock", verify)
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
