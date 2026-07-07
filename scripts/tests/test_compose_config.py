# coding: utf-8

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

HTTP_ENV_KEYS = {
    "CAPSWRITER_HTTP_API_ENABLE",
    "CAPSWRITER_HTTP_API_BIND",
    "CAPSWRITER_HTTP_API_PORT",
    "CAPSWRITER_HTTP_API_KEY",
    "CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND",
    "CAPSWRITER_HTTP_API_MAX_UPLOAD_MB",
    "CAPSWRITER_HTTP_API_TASK_TIMEOUT",
    "CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS",
    "CAPSWRITER_HTTP_API_CORS_ORIGINS",
    "CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS",
}

MODEL_TUNING_ENV_KEYS = {
    "CAPSWRITER_QWEN_CHUNK_SIZE",
    "CAPSWRITER_QWEN_N_CTX",
    "CAPSWRITER_QWEN_MEMORY_NUM",
    "CAPSWRITER_QWEN_PAD_TO",
    "CAPSWRITER_QWEN_LLAMA_N_BATCH",
    "CAPSWRITER_QWEN_LLAMA_N_UBATCH",
    "CAPSWRITER_QWEN_LLAMA_FLASH_ATTN",
    "CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV",
    "CAPSWRITER_FUNASR_ENABLE_CTC",
    "CAPSWRITER_FUNASR_N_PREDICT",
    "CAPSWRITER_FUNASR_PAD_TO",
    "CAPSWRITER_FUNASR_MAX_HOTWORDS",
    "CAPSWRITER_FUNASR_SIMILAR_THRESHOLD",
}

ENTRYPOINT_DERIVED_OR_UNSUPPORTED_KEYS = {
    "CAPSWRITER_FUNASR_DML_ENABLE",
    "CAPSWRITER_FUNASR_USE_CUDA",
    "CAPSWRITER_FUNASR_VULKAN_ENABLE",
    "CAPSWRITER_QWEN_N_PREDICT",
    "CAPSWRITER_QWEN_USE_CUDA",
    "CAPSWRITER_QWEN_USE_DML",
    "CAPSWRITER_QWEN_VULKAN_ENABLE",
}


def active_yaml_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        match = re.match(r"\s+([A-Z0-9_]+):", line)
        if match:
            keys.add(match.group(1))
    return keys


def active_env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        if "=" in line:
            keys.add(line.split("=", 1)[0].strip())
    return keys


class ComposeConfigTest(unittest.TestCase):
    def test_main_compose_passes_http_api_environment(self) -> None:
        keys = active_yaml_keys(ROOT / "docker-compose.yml")
        self.assertTrue(HTTP_ENV_KEYS <= keys)

    def test_example_compose_passes_http_api_environment(self) -> None:
        keys = active_yaml_keys(ROOT / "docker-compose.example.yml")
        self.assertTrue(HTTP_ENV_KEYS <= keys)

    def test_compose_passes_supported_model_tuning_environment(self) -> None:
        for filename in ("docker-compose.yml", "docker-compose.example.yml"):
            with self.subTest(filename=filename):
                keys = active_yaml_keys(ROOT / filename)
                self.assertTrue(MODEL_TUNING_ENV_KEYS <= keys)

    def test_user_facing_env_templates_avoid_backend_internal_keys(self) -> None:
        for filename in (
            ".env.example",
            "docker-compose.yml",
            "docker-compose.example.yml",
            "docker-compose.fun-asr.yml",
        ):
            with self.subTest(filename=filename):
                path = ROOT / filename
                if path.suffix == ".yml":
                    keys = active_yaml_keys(path)
                else:
                    keys = active_env_keys(path)
                self.assertFalse(ENTRYPOINT_DERIVED_OR_UNSUPPORTED_KEYS & keys)

    def test_server_image_exposes_websocket_and_http_ports(self) -> None:
        dockerfile = (ROOT / "docker/server/Dockerfile").read_text(encoding="utf-8")
        self.assertRegex(dockerfile, r"(?m)^EXPOSE\s+6016\s+6017$")


if __name__ == "__main__":
    unittest.main()
