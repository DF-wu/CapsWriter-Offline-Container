"""Fail visibly when a selected CUDA decoder registered no CUDA device."""
import ctypes


def require_cuda_device(names: list[str]) -> None:
    if not any(name.startswith("CUDA") for name in names):
        raise RuntimeError(f"CUDA decoder requested but no CUDA device registered: {names}")


def validate_cuda_backend(ggml) -> None:
    ggml.ggml_backend_dev_count.restype = ctypes.c_size_t
    ggml.ggml_backend_dev_get.argtypes = [ctypes.c_size_t]
    ggml.ggml_backend_dev_get.restype = ctypes.c_void_p
    ggml.ggml_backend_dev_name.argtypes = [ctypes.c_void_p]
    ggml.ggml_backend_dev_name.restype = ctypes.c_char_p
    names = [ggml.ggml_backend_dev_name(ggml.ggml_backend_dev_get(i)).decode()
             for i in range(ggml.ggml_backend_dev_count())]
    require_cuda_device(names)
    print(f"[capswriter] CUDA decoder devices verified: {', '.join(names)}", flush=True)
