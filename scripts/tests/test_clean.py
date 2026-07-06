# coding: utf-8

from __future__ import annotations

import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
