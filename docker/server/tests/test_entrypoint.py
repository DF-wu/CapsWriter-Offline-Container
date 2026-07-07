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

    def test_qwen_cpu_only_preset_forces_cpu_backend(self) -> None:
        script = ENTRYPOINT.read_text(encoding="utf-8")

        self.assertIn('model_type="${CAPSWRITER_MODEL_TYPE:-qwen_asr}"', script)
        self.assertIn(
            '[ "$model_type" = "qwen_asr" ] && [ "$qwen_preset" = "cpu_only" ]',
            script,
        )
        self.assertIn('export CAPSWRITER_LLAMA_BACKEND="cpu"', script)
        self.assertIn('export CAPSWRITER_QWEN_VULKAN_ENABLE="false"', script)
        self.assertIn('export CAPSWRITER_QWEN_USE_CUDA="false"', script)


if __name__ == "__main__":
    unittest.main()
