# coding: utf-8

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

HTTP_ENV_KEYS = {
    "CAPSWRITER_HTTP_API_ENABLE",
    "CAPSWRITER_HTTP_API_BIND",
    "CAPSWRITER_HTTP_API_PORT",
    "CAPSWRITER_HTTP_API_KEY",
    "CAPSWRITER_HTTP_API_KEY_FILE",
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

RESOURCE_ENV_KEYS = {
    "CAPSWRITER_NUM_THREADS",
    "CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS",
    "CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS",
    "CAPSWRITER_MODEL_DOWNLOAD_TIMEOUT",
    "CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT",
    "CAPSWRITER_BACKEND_PROBE_TIMEOUT",
    "CAPSWRITER_ENGINE_FFMPEG_TIMEOUT",
    "CAPSWRITER_GPU_BOOST_TIMEOUT",
    "CAPSWRITER_SERVER_MODEL_LOAD_TIMEOUT",
    "CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT",
    "CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT",
    "CAPSWRITER_REMOVE_MODEL_ARCHIVES",
}

PUBLISH_ENV_KEYS = {
    "CAPSWRITER_SERVER_PUBLISH_HOST",
    "CAPSWRITER_HTTP_API_PUBLISH_HOST",
    "CAPSWRITER_WEB_PUBLISH_HOST",
}

WEB_ENV_KEYS = {
    "CAPSWRITER_WEB_API_BASE",
    "CAPSWRITER_WEB_API_KEY",
    "CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY",
    "CAPSWRITER_WEB_MODEL",
    "CAPSWRITER_WEB_LANGUAGE",
    "CAPSWRITER_WEB_PROMPT",
    "CAPSWRITER_WEB_RESPONSE_FORMAT",
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
    def test_tesla_p4_runtime_is_forwarded_with_opt_in_default(self) -> None:
        for filename in ("docker-compose.yml",):
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                self.assertIn("${CAPSWRITER_TESLA_P4_OPTIMIZATION:-false}", source)
        env = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("CAPSWRITER_TESLA_P4_OPTIMIZATION=false", env)

    def test_main_compose_passes_http_api_environment(self) -> None:
        keys = active_yaml_keys(ROOT / "docker-compose.yml")
        self.assertTrue(HTTP_ENV_KEYS <= keys)

    def test_compose_passes_supported_model_tuning_environment(self) -> None:
        for filename in ("docker-compose.yml",):
            with self.subTest(filename=filename):
                keys = active_yaml_keys(ROOT / filename)
                self.assertTrue(MODEL_TUNING_ENV_KEYS <= keys)

    def test_compose_and_env_template_pass_resource_environment(self) -> None:
        env_keys = active_env_keys(ROOT / ".env.example")
        self.assertTrue(RESOURCE_ENV_KEYS <= env_keys)
        self.assertIn(
            "CAPSWRITER_BACKEND_PROBE_TIMEOUT=300",
            (ROOT / ".env.example").read_text(encoding="utf-8"),
        )
        for filename in ("docker-compose.yml",):
            with self.subTest(filename=filename):
                compose_source = (ROOT / filename).read_text(encoding="utf-8")
                keys = active_yaml_keys(ROOT / filename)
                self.assertTrue(RESOURCE_ENV_KEYS <= keys)
                self.assertIn(
                    "CAPSWRITER_BACKEND_PROBE_TIMEOUT: "
                    "${CAPSWRITER_BACKEND_PROBE_TIMEOUT:-300}",
                    compose_source,
                )

    def test_compose_publishes_ports_on_loopback_by_default(self) -> None:
        env_keys = active_env_keys(ROOT / ".env.example")
        self.assertTrue(PUBLISH_ENV_KEYS <= env_keys)

        expectations = {
            "docker-compose.yml": (
                "${CAPSWRITER_SERVER_PUBLISH_HOST:-127.0.0.1}:${CAPSWRITER_SERVER_PORT:-6016}:${CAPSWRITER_SERVER_PORT:-6016}",
                "${CAPSWRITER_HTTP_API_PUBLISH_HOST:-127.0.0.1}:${CAPSWRITER_HTTP_API_PORT:-6017}:${CAPSWRITER_HTTP_API_PORT:-6017}",
            ),
        }
        for filename, expected_lines in expectations.items():
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                for expected in expected_lines:
                    self.assertIn(expected, source)

    def test_web_compose_requires_explicit_public_api_key_opt_in(self) -> None:
        env_keys = active_env_keys(ROOT / ".env.example")
        self.assertTrue(WEB_ENV_KEYS <= env_keys)

        source = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        for key in WEB_ENV_KEYS:
            with self.subTest(key=key):
                self.assertIn(f"{key}:", source)
        self.assertIn("CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY: ${CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY:-false}", source)

    def test_compose_services_enable_no_new_privileges(self) -> None:
        for filename in (
            "docker-compose.yml",
        ):
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                self.assertIn("security_opt:\n      - no-new-privileges:true", source)

    def test_server_compose_services_drop_linux_capabilities(self) -> None:
        for filename in ("docker-compose.yml",):
            with self.subTest(filename=filename):
                source = (ROOT / filename).read_text(encoding="utf-8")
                self.assertIn("cap_drop:\n      - ALL", source)

    def test_only_one_compose_entrypoint_remains(self) -> None:
        self.assertEqual(sorted(p.name for p in ROOT.glob("*compose*.y*ml")),
                         ["docker-compose.yml"])

    def test_settings_and_storage_are_integrated(self) -> None:
        source = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        for expected in (
            "CAPSWRITER_SETTINGS_ENABLE:",
            "CAPSWRITER_SETTINGS_PATH: /app/settings/server-settings.json",
            "capswriter-server-settings:/app/settings",
            "${CAPSWRITER_MODELS_SOURCE:-capswriter-server-models}:/app/models",
            "source: ./hot-server.txt", "target: /app/hot-server.txt",
            "read_only: true", "create_host_path: false",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, source)

    def test_user_facing_env_templates_avoid_backend_internal_keys(self) -> None:
        for filename in (
            ".env.example",
            "docker-compose.yml",
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

    def test_server_image_installs_intel_amd_vulkan_icd(self) -> None:
        dockerfile = (ROOT / "docker/server/Dockerfile").read_text(encoding="utf-8")
        self.assertIn("mesa-vulkan-drivers", dockerfile)

    def test_entrypoint_full_cpu_fallback_disables_every_gpu_backend(self) -> None:
        entrypoint = ROOT / "docker/server/entrypoint.sh"
        source = entrypoint.read_text(encoding="utf-8")
        functions, separator, _runtime = source.partition('\nconfigure_backend "$@"\n')
        self.assertTrue(separator, "entrypoint function/runtime boundary is missing")
        scenario = functions + """
export CAPSWRITER_LLAMA_BACKEND=vulkan
export CAPSWRITER_QWEN_VULKAN_ENABLE=true
export CAPSWRITER_FUNASR_VULKAN_ENABLE=true
export CAPSWRITER_QWEN_USE_CUDA=true
export CAPSWRITER_FUNASR_USE_CUDA=true
fallback_to_cpu
if gpu_backend_configured; then
  echo GPU_STILL_ENABLED
else
  echo FULL_CPU_ENABLED
fi
env
"""
        result = subprocess.run(
            ["sh"],
            input=scenario,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
            env={"PATH": os.environ.get("PATH", "")},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("FULL_CPU_ENABLED", result.stdout)
        self.assertNotIn("GPU_STILL_ENABLED", result.stdout)
        for expected in (
            "CAPSWRITER_LLAMA_BACKEND=cpu",
            "CAPSWRITER_QWEN_VULKAN_ENABLE=false",
            "CAPSWRITER_FUNASR_VULKAN_ENABLE=false",
            "CAPSWRITER_QWEN_USE_CUDA=false",
            "CAPSWRITER_FUNASR_USE_CUDA=false",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, result.stdout)

    def test_entrypoint_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["sh", "-n", str(ROOT / "docker/server/entrypoint.sh")],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
