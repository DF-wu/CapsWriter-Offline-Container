# coding: utf-8

from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import clean


class CleanTraversalTest(unittest.TestCase):
    def test_python_cache_iterator_prunes_preserved_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep_cache = root / "src" / "__pycache__"
            keep_cache.mkdir(parents=True)
            keep_pyc = root / "src" / "module.pyc"
            keep_pyc.write_bytes(b"pyc")

            for relative in [
                ".git/hooks/__pycache__",
                "client/web/node_modules/pkg/__pycache__",
                "models/Qwen3-ASR/__pycache__",
            ]:
                cache_dir = root / relative
                cache_dir.mkdir(parents=True)
                (cache_dir / "ignored.pyc").write_bytes(b"pyc")

            artifacts = {
                path.relative_to(root).as_posix()
                for path in clean.iter_python_cache_artifacts(
                    root,
                    {
                        root / ".git",
                        root / "client" / "web" / "node_modules",
                        root / "models",
                    },
                )
            }

        self.assertEqual(artifacts, {"src/__pycache__", "src/module.pyc"})

    def test_should_prune_matches_preserved_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            preserved = {root / "client" / "web" / "node_modules"}

            self.assertTrue(
                clean.should_prune(root / "client/web/node_modules/pkg", preserved)
            )
            self.assertFalse(clean.should_prune(root / "client/web/src", preserved))

    def test_cleanup_residue_reports_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "client/cli/dist").mkdir(parents=True)
            (root / "client/web/dist").mkdir(parents=True)
            (root / "client/web/tsconfig.tsbuildinfo").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            (root / "client/web/tsconfig.tsbuildinfo").write_text("ts", encoding="utf-8")
            (root / "src/__pycache__").mkdir(parents=True)
            (root / "src/module.pyc").write_bytes(b"pyc")

            residue = {
                path.relative_to(root).as_posix()
                for path in clean.iter_cleanup_residue(root)
            }

        self.assertEqual(
            residue,
            {
                "client/cli/dist",
                "client/web/dist",
                "client/web/tsconfig.tsbuildinfo",
                "src/__pycache__",
                "src/module.pyc",
            },
        )

    def test_cleanup_residue_prunes_preserved_python_cache_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in [
                ".git/hooks/__pycache__",
                "client/web/node_modules/pkg/__pycache__",
                "models/Qwen3-ASR/__pycache__",
            ]:
                cache_dir = root / relative
                cache_dir.mkdir(parents=True)
                (cache_dir / "ignored.pyc").write_bytes(b"pyc")

            residue = list(clean.iter_cleanup_residue(root))

        self.assertEqual(residue, [])

    def test_check_clean_returns_failure_with_residue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "client/web/dist").mkdir(parents=True)
            stdout = io.StringIO()
            stderr = io.StringIO()

            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = clean.check_clean(root)

        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("client/web/dist", stderr.getvalue())

    def test_run_web_clean_passes_configured_timeout(self) -> None:
        completed = SimpleNamespace(returncode=0)
        with (
            patch.object(clean, "WEB_ROOT", Path("/tmp/capswriter-web")),
            patch.object(Path, "exists", return_value=True),
            patch.object(clean.shutil, "which", return_value="/usr/bin/npm"),
            patch.dict(clean.os.environ, {clean.CLEAN_WEB_TIMEOUT_ENV: "2.5"}),
            patch.object(clean.subprocess, "run", return_value=completed) as run,
        ):
            code = clean.run_web_clean()

        self.assertEqual(code, 0)
        self.assertEqual(run.call_args.kwargs["timeout"], 2.5)

    def test_run_web_clean_rejects_invalid_timeout_before_spawning(self) -> None:
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value):
                stderr = io.StringIO()
                with (
                    patch.object(clean, "WEB_ROOT", Path("/tmp/capswriter-web")),
                    patch.object(Path, "exists", return_value=True),
                    patch.object(clean.shutil, "which", return_value="/usr/bin/npm"),
                    patch.dict(clean.os.environ, {clean.CLEAN_WEB_TIMEOUT_ENV: value}),
                    patch.object(clean.subprocess, "run") as run,
                    redirect_stderr(stderr),
                ):
                    code = clean.run_web_clean()

                self.assertEqual(code, 1)
                run.assert_not_called()
                self.assertIn("CAPSWRITER_CLEAN_WEB_TIMEOUT must be > 0", stderr.getvalue())

    def test_run_web_clean_timeout_returns_timeout_exit_code(self) -> None:
        stderr = io.StringIO()
        with (
            patch.object(clean, "WEB_ROOT", Path("/tmp/capswriter-web")),
            patch.object(Path, "exists", return_value=True),
            patch.object(clean.shutil, "which", return_value="/usr/bin/npm"),
            patch.dict(clean.os.environ, {clean.CLEAN_WEB_TIMEOUT_ENV: "1"}),
            patch.object(
                clean.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(["npm"], timeout=1),
            ),
            redirect_stderr(stderr),
        ):
            code = clean.run_web_clean()

        self.assertEqual(code, clean.TIMEOUT_EXIT_CODE)
        self.assertIn("npm run clean timed out after 1s", stderr.getvalue())

    def test_clean_generated_artifacts_continues_after_web_clean_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dist = root / "client/web/dist"
            dist.mkdir(parents=True)
            pycache = root / "src/__pycache__"
            pycache.mkdir(parents=True)
            tsbuild = root / "client/web/tsconfig.tsbuildinfo"
            tsbuild.write_text("ts", encoding="utf-8")

            with (
                patch.object(clean, "ROOT", root),
                patch.object(clean, "run_web_clean", return_value=clean.TIMEOUT_EXIT_CODE),
            ):
                code = clean.clean_generated_artifacts()

            self.assertEqual(code, clean.TIMEOUT_EXIT_CODE)
            self.assertFalse(dist.exists())
            self.assertFalse(pycache.exists())
            self.assertFalse(tsbuild.exists())


if __name__ == "__main__":
    unittest.main()
