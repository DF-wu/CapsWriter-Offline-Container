"""Run via entrypoint in a disposable NVIDIA container; never loads ASR weights."""
import ctypes
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docker/server"))

from fork_server.llama_backend import tesla_p4_optimization_enabled, validate_cuda_backend
import download_models as dm


def main():
    enabled = tesla_p4_optimization_enabled()
    selected = os.environ["CAPSWRITER_LLAMA_BACKEND"]
    assert selected == ("cuda" if enabled else "vulkan"), selected
    if not enabled:
        assert "cuda" not in dm.LLAMA_CPP_ASSETS
        assert "cuda" not in dm.LLAMA_CPP_OFFICIAL_MANIFESTS
        print("P4 smoke PASS: disabled, original Vulkan selected, CUDA bundle unregistered")
        return

    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="capswriter-p4-smoke-") as tmp:
        try:
            os.chdir(tmp)
            dm.LLAMA_TARGET_DIRS = [Path(tmp) / str(i) / "bin" for i in range(4)]
            dm._register_bundled_cuda()
            assert dm._prepare_llama_binaries() == 0
            assert dm._llama_binaries_ready()
            ggml = ctypes.CDLL(str(dm.LLAMA_TARGET_DIRS[0] / "libggml.so"))
            ggml.ggml_backend_load_all.argtypes = []
            ggml.ggml_backend_load_all.restype = None
            ggml.ggml_backend_load_all()
            validate_cuda_backend(ggml)
            descriptions = [ggml.ggml_backend_dev_description(ggml.ggml_backend_dev_get(i)).decode()
                            for i in range(ggml.ggml_backend_dev_count())]
            print(f"P4 smoke PASS: enabled, verified relocated libraries, devices={descriptions}")
        finally:
            os.chdir(original)


if __name__ == "__main__":
    main()
