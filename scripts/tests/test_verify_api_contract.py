# coding: utf-8

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import verify_api_contract


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CONTRACT_MODULES = {
    "fork_server.http_api.tests.test_api_asgi",
    "fork_server.http_api.tests.test_body_limit",
    "fork_server.http_api.tests.test_openai_sdk_contract",
    "fork_server.http_api.tests.test_uvicorn_prebody",
}


class VerifyApiContractTest(unittest.TestCase):
    def test_verifier_consumes_every_pinned_dependency_and_required_suite(self) -> None:
        direct_pins = verify_api_contract.load_pins(
            ROOT / "requirements-api-test.txt"
        )
        lock_pins = verify_api_contract.load_pins(
            ROOT / "requirements-api-test.lock"
        )

        self.assertEqual(
            set(direct_pins),
            set(verify_api_contract.REQUIRED_IMPORTS),
        )
        self.assertEqual(
            verify_api_contract.lock_configuration_errors(
                direct_pins,
                lock_pins,
            ),
            [],
        )
        self.assertGreater(set(lock_pins), set(direct_pins))
        self.assertTrue(
            REQUIRED_CONTRACT_MODULES
            <= set(verify_api_contract.REQUIRED_CONTRACT_MODULES)
        )
        discovered = verify_api_contract.load_contract_suite().countTestCases()
        direct_discovery = unittest.defaultTestLoader.discover(
            str(verify_api_contract.CONTRACT_TEST_ROOT),
        ).countTestCases()
        self.assertEqual(discovered, direct_discovery)
        self.assertGreater(discovered, len(REQUIRED_CONTRACT_MODULES))
        for distribution, modules in verify_api_contract.REQUIRED_IMPORTS.items():
            with self.subTest(distribution=distribution):
                self.assertTrue(modules)

    def test_load_pins_rejects_non_exact_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "requirements.txt"
            path.write_text("openai>=2\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "strict name==version pin"):
                verify_api_contract.load_pins(path)

    def test_dependency_errors_report_version_and_import_failures(self) -> None:
        with (
            patch.object(
                verify_api_contract,
                "REQUIRED_IMPORTS",
                {"openai": ("openai",)},
            ),
            patch.object(
                verify_api_contract.metadata,
                "version",
                return_value="2.45.1",
            ),
            patch.object(
                verify_api_contract.importlib,
                "import_module",
                side_effect=ImportError("broken import"),
            ),
        ):
            errors = verify_api_contract.dependency_errors({"openai": "2.45.0"})

        self.assertTrue(any("version mismatch" in error for error in errors))
        self.assertTrue(any("import failed" in error for error in errors))

    def test_strict_runner_is_unsuccessful_when_a_test_is_skipped(self) -> None:
        class SkippedContractTest(unittest.TestCase):
            @unittest.skip("dependency unavailable")
            def runTest(self) -> None:
                self.fail("skip decorator should prevent this")

        result = verify_api_contract.StrictTextTestRunner(
            stream=io.StringIO(),
            verbosity=0,
        ).run(unittest.TestSuite([SkippedContractTest()]))

        self.assertEqual(len(result.skipped), 1)
        self.assertFalse(result.wasSuccessful())

    def test_contract_discovery_rejects_an_empty_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.object(
                verify_api_contract,
                "CONTRACT_TEST_ROOT",
                Path(temporary_directory),
            ):
                with self.assertRaisesRegex(RuntimeError, "found no tests"):
                    verify_api_contract.load_contract_suite()

    def test_main_stops_before_tests_when_dependency_verification_fails(self) -> None:
        with (
            patch.object(verify_api_contract, "load_pins", return_value={}),
            patch.object(
                verify_api_contract,
                "dependency_errors",
                return_value=["missing dependency pin: openai"],
            ),
            patch.object(verify_api_contract, "run_contract_tests") as run_tests,
            patch("sys.stderr", new=io.StringIO()),
        ):
            code = verify_api_contract.main()

        self.assertEqual(code, 1)
        run_tests.assert_not_called()


if __name__ == "__main__":
    unittest.main()
