# coding: utf-8

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT = ROOT / "docker/server/entrypoint.sh"


class EntrypointTest(unittest.TestCase):
    def test_entrypoint_shell_syntax_is_valid(self) -> None:
        subprocess.run(["sh", "-n", ENTRYPOINT.as_posix()], check=True)

    def test_qwen_cpu_presets_force_cpu_backend(self) -> None:
        script = ENTRYPOINT.read_text(encoding="utf-8")

        self.assertIn('model_type="${CAPSWRITER_MODEL_TYPE:-qwen_asr}"', script)
        self.assertIn(
            '[ "$model_type" = "qwen_asr" ] && [ "$qwen_preset" = "cpu_only" ]',
            script,
        )
        self.assertIn(
            '[ "$model_type" = "qwen_asr" ] && [ "$qwen_preset" = "low_vram_gpu" ]',
            script,
        )
        self.assertIn('export CAPSWRITER_LLAMA_BACKEND="cpu"', script)
        self.assertIn('export CAPSWRITER_QWEN_VULKAN_ENABLE="false"', script)
        self.assertIn('export CAPSWRITER_QWEN_USE_CUDA="false"', script)
        self.assertIn('export CAPSWRITER_QWEN_USE_CUDA="true"', script)

    def test_gpu_bootstrap_failure_retries_and_probes_full_cpu_backend(self) -> None:
        script = ENTRYPOINT.read_text(encoding="utf-8")

        self.assertIn("prepare_and_probe_cpu_fallback()", script)
        self.assertIn(
            "if ! python /app/docker/server/download_models.py; then",
            script,
        )
        self.assertIn("selected GPU backend bootstrap failed", script)
        self.assertIn("prepare_and_probe_cpu_fallback", script)
        self.assertIn("full CPU backend probe failed; refusing to start", script)
        self.assertRegex(
            script,
            r"(?s)prepare_and_probe_cpu_fallback\(\).*?fallback_to_cpu.*?"
            r"download_models\.py.*?probe_backend\.py",
        )


if __name__ == "__main__":
    unittest.main()
