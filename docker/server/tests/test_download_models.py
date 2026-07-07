# coding: utf-8

from __future__ import annotations

import io
import os
import sys
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

import download_models  # noqa: E402


class FakeDownloadResponse:
    def __init__(self, *chunks: bytes, fail_after_chunks: int | None = None):
        self._chunks = list(chunks)
        self._index = 0
        self._fail_after_chunks = fail_after_chunks
        self.headers = {"Content-Length": str(sum(len(chunk) for chunk in chunks))}

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        return False

    def read(self, _size: int) -> bytes:
        if self._fail_after_chunks is not None and self._index >= self._fail_after_chunks:
            raise TimeoutError("timed out")
        if self._index >= len(self._chunks):
            return b""
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class DownloadModelsTest(unittest.TestCase):
    def _write_libraries(self, target_dir: Path, names: list[str]) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (target_dir / name).write_bytes(b"")

    def test_resolve_model_type_normalizes_env_value(self) -> None:
        with patch.dict(
            os.environ,
            {"CAPSWRITER_MODEL_TYPE": " Fun_ASR_Nano "},
        ):
            self.assertEqual(download_models._resolve_model_type(), "fun_asr_nano")

    def test_unsupported_model_error_reports_resolved_env_value(self) -> None:
        stderr = io.StringIO()
        with patch.dict(os.environ, {"CAPSWRITER_MODEL_TYPE": "sensevoice"}):
            with redirect_stderr(stderr):
                code = download_models.main()

        self.assertEqual(code, 1)
        message = stderr.getvalue()
        self.assertIn("CAPSWRITER_MODEL_TYPE='sensevoice'", message)
        self.assertIn("qwen_asr", message)
        self.assertIn("fun_asr_nano", message)

    def test_download_timeout_defaults_to_bounded_value(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                download_models._download_timeout_seconds(),
                download_models.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
            )

    def test_download_timeout_rejects_invalid_values(self) -> None:
        for value in ("0", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT": value}):
                    with self.assertRaisesRegex(ValueError, "must be > 0"):
                        download_models._download_timeout_seconds()

    def test_download_streams_with_configured_timeout_and_atomic_replace(self) -> None:
        observed = {}

        def fake_urlopen(url, *, timeout):
            observed["url"] = url
            observed["timeout"] = timeout
            return FakeDownloadResponse(b"abc", b"def")

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "models" / ".downloads" / "asset.zip"
            with (
                patch.object(download_models.urllib.request, "urlopen", side_effect=fake_urlopen),
                patch.dict(os.environ, {"CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT": "7.5"}),
                redirect_stdout(io.StringIO()),
            ):
                download_models._download("https://example.test/asset.zip", destination)

            self.assertEqual(observed, {"url": "https://example.test/asset.zip", "timeout": 7.5})
            self.assertEqual(destination.read_bytes(), b"abcdef")
            self.assertFalse(destination.with_name("asset.zip.part").exists())

    def test_download_removes_partial_file_on_failure(self) -> None:
        def fake_urlopen(_url, *, timeout):
            self.assertEqual(timeout, download_models.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
            return FakeDownloadResponse(b"partial", fail_after_chunks=1)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "asset.zip"
            with (
                patch.object(download_models.urllib.request, "urlopen", side_effect=fake_urlopen),
                patch.dict(os.environ, {}, clear=True),
                redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(TimeoutError, "timed out"),
            ):
                download_models._download("https://example.test/asset.zip", destination)

            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("asset.zip.part").exists())

    def test_extract_rejects_zip_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "model.zip"
            target_dir = root / "target"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../escaped.bin", b"bad")

            with self.assertRaisesRegex(ValueError, "不安全"):
                download_models._extract(archive, target_dir)

            self.assertFalse((root / "escaped.bin").exists())

    def test_extract_llama_rejects_tar_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "llama.tar.gz"
            data = b"bad"
            with tarfile.open(archive, "w:gz") as tar_file:
                info = tarfile.TarInfo("../escaped.so")
                info.size = len(data)
                tar_file.addfile(info, io.BytesIO(data))

            with self.assertRaisesRegex(ValueError, "不安全"):
                download_models._extract_llama_binaries(archive)

            self.assertFalse((root / "escaped.so").exists())

    def test_extract_llama_rejects_link_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "llama.tar.gz"
            with tarfile.open(archive, "w:gz") as tar_file:
                info = tarfile.TarInfo("libllama.so")
                info.type = tarfile.SYMTYPE
                info.linkname = "/etc/passwd"
                tar_file.addfile(info)

            with self.assertRaisesRegex(ValueError, "类型不安全"):
                download_models._extract_llama_binaries(archive)

    def test_llama_download_failure_returns_clean_error(self) -> None:
        fake_asset = download_models.Asset(
            name="unit-test-llama.tar.gz",
            url="https://example.test/llama.tar.gz",
            sha256="",
            target_dir=Path("unused"),
            required_files=[],
        )
        stderr = io.StringIO()
        with (
            patch.object(download_models, "LLAMA_CPP_ASSETS", {"cpu": fake_asset}),
            patch.object(download_models, "_llama_binaries_ready", return_value=False),
            patch.object(download_models, "_download", side_effect=TimeoutError("timed out")),
            patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr),
        ):
            code = download_models._prepare_llama_binaries()

        self.assertEqual(code, 1)
        self.assertIn("下载 llama.cpp 压缩包失败: unit-test-llama.tar.gz: timed out", stderr.getvalue())

    def test_llama_binaries_ready_rejects_unversioned_only_libraries(self) -> None:
        previous_required = [
            "libggml.so",
            "libggml-base.so",
            "libllama.so",
            "libggml-cpu.so",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(target_dir, previous_required)

            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            ):
                self.assertFalse(download_models._llama_binaries_ready())

    def test_llama_binaries_ready_accepts_runtime_linked_cpu_libraries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(target_dir, download_models.LLAMA_REQUIRED_CPU_LIBRARIES)

            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "cpu"}),
            ):
                self.assertTrue(download_models._llama_binaries_ready())

    def test_llama_binaries_ready_requires_vulkan_backend_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target_dir = Path(tmp) / "bin"
            self._write_libraries(target_dir, download_models.LLAMA_REQUIRED_CPU_LIBRARIES)

            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "vulkan"}),
            ):
                self.assertFalse(download_models._llama_binaries_ready())

            self._write_libraries(
                target_dir,
                download_models.LLAMA_REQUIRED_VULKAN_LIBRARIES,
            )
            with (
                patch.object(download_models, "LLAMA_TARGET_DIRS", [target_dir]),
                patch.dict(os.environ, {"CAPSWRITER_LLAMA_BACKEND": "vulkan"}),
            ):
                self.assertTrue(download_models._llama_binaries_ready())


if __name__ == "__main__":
    unittest.main()
