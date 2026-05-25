"""
Lightweight GPU backend probe.

Tries to instantiate the configured ASR engine via EngineFactory.
If construction fails (e.g. CUDA libraries missing, model files missing,
GPU OOM), exit non-zero so entrypoint.sh can fall back to CPU.

Only used when CAPSWRITER_LLAMA_BACKEND=vulkan and we suspect the GPU
path may not be functional. The main server has the same try-init
behaviour internally, but this probe lets us reset env vars + redownload
binaries BEFORE the long-running supervisor starts.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if ROOT_DIR.as_posix() not in sys.path:
    sys.path.insert(0, ROOT_DIR.as_posix())

from config_server import ServerConfig


def main() -> int:
    model_type = ServerConfig.model_type.lower()

    # Skip probe for non-GGUF engines (sensevoice/paraformer don't load llama.cpp).
    if model_type not in {"qwen_asr", "fun_asr_nano"}:
        print(f"[capswriter] probe skipped: {model_type} does not use llama.cpp")
        return 0

    try:
        from core.server.engines.factory import EngineFactory
        engine = EngineFactory.create_asr_engine(model_type)
    except Exception as error:
        print(f"[capswriter] GPU backend probe failed: {error}", file=sys.stderr)
        return 1

    try:
        engine.cleanup()
    except Exception:
        pass

    print(f"[capswriter] GPU backend probe passed for {model_type}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
