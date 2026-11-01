"""Fail visibly when a selected CUDA decoder registered no CUDA device."""
import ctypes
import os


TESLA_P4_FLAG = "CAPSWRITER_TESLA_P4_OPTIMIZATION"


def tesla_p4_optimization_enabled() -> bool:
    """明確 opt-in；未設定時保留原本的 CPU／Vulkan 路徑。"""
    value = os.environ.get(TESLA_P4_FLAG, "false").strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off", ""}:
        return False
    raise ValueError(f"{TESLA_P4_FLAG} must be a boolean")


def require_cuda_device(names: list[str]) -> None:
    if not any(name.startswith("CUDA") for name in names):
        raise RuntimeError(f"CUDA decoder requested but no CUDA device registered: {names}")


def require_tesla_p4_devices(descriptions: list[str]) -> None:
    """避免對其他 NVIDIA 卡套用僅在 P4 驗證的 runtime。"""
    if not descriptions or any("P4" not in description.upper().split() for description in descriptions):
        raise RuntimeError(f"Tesla P4 optimization requires only visible Tesla P4 devices: {descriptions}")


def validate_cuda_backend(ggml) -> None:
    if not tesla_p4_optimization_enabled():
        raise RuntimeError(f"CUDA SM61 decoder requires {TESLA_P4_FLAG}=true")
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
    ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    ggml.ggml_backend_dev_get.restype = ctypes.c_void_p
    ggml.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_dev_name.restype = ctypes.c_char_p
    names = [ggml.ggml_backend_dev_name(ggml.ggml_backend_dev_get(i)).decode()
             for i in range(ggml.ggml_backend_dev_count())]
    require_cuda_device(names)
    ggml.ggml_backend_dev_description.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_dev_description.restype = ctypes.c_char_p
    descriptions = [ggml.ggml_backend_dev_description(ggml.ggml_backend_dev_get(i)).decode()
                    for i, name in enumerate(names) if name.startswith("CUDA")]
    require_tesla_p4_devices(descriptions)
    print(f"[capswriter] CUDA decoder devices verified: {', '.join(names)}", flush=True)


if __name__ == "__main__":
    import sys
    try:
        print("true" if tesla_p4_optimization_enabled() else "false")
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
