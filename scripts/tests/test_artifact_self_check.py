# coding: utf-8

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import artifact_self_check
import start_client
import start_server_universal


def create_artifact_layout(
    root: Path,
    *,
    packaged: bool = True,
    variant: str = "full",
) -> None:
    client_only = variant == "client"
    directories = (
        artifact_self_check.CLIENT_REQUIRED_DIRECTORIES
        if client_only else artifact_self_check.REQUIRED_DIRECTORIES
    )
    root_files = (
        artifact_self_check.CLIENT_REQUIRED_ROOT_FILES
        if client_only else artifact_self_check.REQUIRED_ROOT_FILES
    )
    executables = (
        artifact_self_check.CLIENT_PACKAGED_FILES
        if client_only else artifact_self_check.PACKAGED_FILES
    )
    for relative in directories:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in root_files:
        (root / relative).write_text("fixture\n", encoding="utf-8")
    if packaged:
        (root / "internal").mkdir()
        (root / artifact_self_check.PACKAGE_MARKER).write_text(
            json.dumps({"variant": variant}),
            encoding="utf-8",
        )
        for executable in executables:
            (root / executable).write_bytes(b"fixture")


class ArtifactSelfCheckTest(unittest.TestCase):
    def test_layout_accepts_complete_real_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root)

            artifact_self_check.validate_artifact_layout(root, packaged=True)

    def test_layout_rejects_missing_required_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root)
            (root / "hot-server.txt").unlink()

            with self.assertRaisesRegex(
                artifact_self_check.ArtifactSelfCheckError,
                "hot-server.txt",
            ):
                artifact_self_check.validate_artifact_layout(root, packaged=True)

    def test_layout_rejects_nested_link_or_junction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root)
            linked = root / "core" / "linked"
            linked.mkdir()

            with patch.object(
                artifact_self_check,
                "_is_link_or_junction",
                side_effect=lambda path: Path(path) == linked,
            ):
                with self.assertRaisesRegex(
                    artifact_self_check.ArtifactSelfCheckError,
                    "symbolic links or junctions",
                ):
                    artifact_self_check.validate_artifact_layout(root, packaged=True)

    def test_success_report_is_stable_and_imports_selected_surface(self) -> None:
        self.assertIn(
            "core.server.engines.factory",
            artifact_self_check.SERVER_IMPORTS,
        )
        self.assertNotIn(
            "core.server.engines.fun_asr_gguf.asr_engine",
            artifact_self_check.SERVER_IMPORTS,
        )
        self.assertNotIn(
            "core.server.engines.qwen_asr_gguf.asr_engine",
            artifact_self_check.SERVER_IMPORTS,
        )
        self.assertIn("sentencepiece", artifact_self_check.SERVER_IMPORTS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root)
            stdout = io.StringIO()

            with patch.object(
                artifact_self_check,
                "import_runtime_surface",
                return_value=("one", "two"),
            ) as import_surface, redirect_stdout(stdout):
                code = artifact_self_check.run_artifact_self_check(
                    "server",
                    root=root,
                    packaged=True,
                )

            self.assertEqual(code, 0)
            import_surface.assert_called_once_with(artifact_self_check.SERVER_IMPORTS)
            self.assertIn(artifact_self_check.SELF_CHECK_MARKER, stdout.getvalue())
            self.assertIn('"entrypoint":"server"', stdout.getvalue())
            self.assertIn('"status":"ok"', stdout.getvalue())

    def test_failure_report_returns_nonzero_without_importing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with patch.object(
                artifact_self_check,
                "import_runtime_surface",
            ) as import_surface, redirect_stderr(stderr):
                code = artifact_self_check.run_artifact_self_check(
                    "client",
                    root=Path(tmp),
                    packaged=True,
                )

            self.assertEqual(code, 1)
            import_surface.assert_not_called()
            self.assertIn('"entrypoint":"client"', stderr.getvalue())
            self.assertIn('"status":"error"', stderr.getvalue())

    def test_client_layout_accepts_client_only_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root, variant="client")

            self.assertEqual(artifact_self_check.read_package_variant(root), "client")
            artifact_self_check.validate_artifact_layout(
                root,
                packaged=True,
                variant="client",
            )

    def test_client_layout_rejects_server_paths(self) -> None:
        for relative in artifact_self_check.CLIENT_FORBIDDEN_PATHS:
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                create_artifact_layout(root, variant="client")
                target = root / relative
                if relative in ("models", "core/server"):
                    target.mkdir(parents=True)
                else:
                    target.write_text("fixture\n", encoding="utf-8")

                with self.assertRaisesRegex(
                    artifact_self_check.ArtifactSelfCheckError,
                    "client package contains server paths",
                ):
                    artifact_self_check.validate_artifact_layout(
                        root,
                        packaged=True,
                        variant="client",
                    )

    def test_packaged_check_requires_variant_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root)
            (root / artifact_self_check.PACKAGE_MARKER).unlink()
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                code = artifact_self_check.run_artifact_self_check(
                    "client",
                    root=root,
                    packaged=True,
                )

            self.assertEqual(code, 1)
            self.assertIn("package marker", stderr.getvalue())

    def test_client_package_reports_variant_and_rejects_server_modules(self) -> None:
        self.assertNotIn("numba", artifact_self_check.CLIENT_IMPORTS)
        self.assertIn("numba", artifact_self_check.SERVER_IMPORTS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root, variant="client")
            stdout = io.StringIO()
            with patch.object(
                artifact_self_check,
                "import_runtime_surface",
                return_value=("one",),
            ), patch.object(
                artifact_self_check.importlib.util,
                "find_spec",
                return_value=None,
            ), redirect_stdout(stdout):
                code = artifact_self_check.run_artifact_self_check(
                    "client",
                    root=root,
                    packaged=True,
                )

            self.assertEqual(code, 0)
            self.assertIn('"variant":"client"', stdout.getvalue())

            stderr = io.StringIO()
            with patch.object(
                artifact_self_check,
                "import_runtime_surface",
                return_value=("one",),
            ), redirect_stderr(stderr):
                with patch.object(
                    artifact_self_check.importlib.util,
                    "find_spec",
                    side_effect=lambda name: object() if name == "onnxruntime" else None,
                ):
                    code = artifact_self_check.run_artifact_self_check(
                        "client",
                        root=root,
                        packaged=True,
                    )

            self.assertEqual(code, 1)
            self.assertIn("server-only modules: onnxruntime", stderr.getvalue())

    def test_server_entrypoint_is_rejected_in_client_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            create_artifact_layout(root, variant="client")
            stderr = io.StringIO()

            with patch.object(
                artifact_self_check,
                "import_runtime_surface",
            ) as import_surface, redirect_stderr(stderr):
                code = artifact_self_check.run_artifact_self_check(
                    "server",
                    root=root,
                    packaged=True,
                )

            self.assertEqual(code, 1)
            import_surface.assert_not_called()
            self.assertIn("not part of the client package", stderr.getvalue())

    def test_server_flag_uses_self_check_without_configuring_server(self) -> None:
        with patch.object(
            artifact_self_check,
            "run_artifact_self_check",
            return_value=0,
        ) as run_check, patch.object(
            start_server_universal,
            "configure_http_api",
        ) as configure:
            self.assertEqual(
                start_server_universal.main(["--artifact-self-check"]),
                0,
            )

        run_check.assert_called_once_with("server")
        configure.assert_not_called()

    def test_client_flag_uses_self_check_without_constructing_client(self) -> None:
        with patch.object(
            artifact_self_check,
            "run_artifact_self_check",
            return_value=0,
        ) as run_check:
            self.assertEqual(start_client.main(["--artifact-self-check"]), 0)

        run_check.assert_called_once_with("client")


if __name__ == "__main__":
    unittest.main()
