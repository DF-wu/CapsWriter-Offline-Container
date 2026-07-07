# coding: utf-8

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
