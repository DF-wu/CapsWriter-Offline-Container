# coding: utf-8

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

import download_models  # noqa: E402


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
