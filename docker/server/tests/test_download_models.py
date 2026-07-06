# coding: utf-8

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))

import download_models  # noqa: E402


class DownloadModelsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
