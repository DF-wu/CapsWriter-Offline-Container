# coding: utf-8

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import zip_release  # noqa: E402


class ZipReleaseTest(unittest.TestCase):
    def test_package_with_7zip_passes_configured_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dist" / "CapsWriter-Offline"
            source.mkdir(parents=True)
            file_list = root / "file_list.txt"
            file_list.write_text("CapsWriter-Offline/app.exe\n", encoding="utf-8")
            output = root / "release" / "CapsWriter-Offline.zip"
            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            info = SimpleNamespace(returncode=0, stdout="10 files\n", stderr="")

            with (
                patch.object(zip_release, "find_7zip", return_value="/usr/bin/7z"),
                patch.dict(os.environ, {zip_release.ZIP_RELEASE_TIMEOUT_ENV: "2.5"}),
                patch.object(zip_release.subprocess, "run", side_effect=[completed, info]) as run,
                redirect_stdout(io.StringIO()),
            ):
                zip_release.package_with_7zip(source, output, file_list)

        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].kwargs["timeout"], 2.5)
        self.assertEqual(run.call_args_list[1].kwargs["timeout"], 2.5)
        self.assertEqual(run.call_args_list[0].kwargs["cwd"], str(source.parent))

    def test_package_rejects_invalid_timeout_before_spawning_7zip(self) -> None:
        with (
            patch.dict(os.environ, {zip_release.ZIP_RELEASE_TIMEOUT_ENV: "0"}),
            patch.object(zip_release, "find_7zip") as find_7zip,
            patch.object(zip_release.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "CAPSWRITER_ZIP_RELEASE_TIMEOUT must be > 0"):
                zip_release.package_with_7zip("dist/app", "release/app.zip", "file_list.txt")

        find_7zip.assert_not_called()
        run.assert_not_called()

    def test_package_rejects_non_finite_timeout_before_spawning_7zip(self) -> None:
        with (
            patch.dict(os.environ, {zip_release.ZIP_RELEASE_TIMEOUT_ENV: "nan"}),
            patch.object(zip_release, "find_7zip") as find_7zip,
            patch.object(zip_release.subprocess, "run") as run,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "CAPSWRITER_ZIP_RELEASE_TIMEOUT must be a finite number",
            ):
                zip_release.package_with_7zip("dist/app", "release/app.zip", "file_list.txt")

        find_7zip.assert_not_called()
        run.assert_not_called()

    def test_main_removes_file_list_after_package_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dist" / "CapsWriter-Offline"
            source.mkdir(parents=True)
            (source / "app.exe").write_text("binary", encoding="utf-8")
            previous_cwd = Path.cwd()
            try:
                os.chdir(root)
                with (
                    patch.object(zip_release, "package_with_7zip", side_effect=RuntimeError("boom")),
                    redirect_stdout(io.StringIO()),
                ):
                    zip_release.main()
            finally:
                os.chdir(previous_cwd)

            self.assertFalse((root / "file_list_0.txt").exists())


if __name__ == "__main__":
    unittest.main()
