# coding: utf-8
"""
env_config — 把環境變數套用到上游 config_server 的 class 屬性

設計要點:
1. 上游 config_server.py (v2.5) 是純 class-based config, 無 env 處理。
2. 本檔在 bootstrap 階段執行, 必須在 `import core.server.*` 之前 (因為
   core/server/__init__.py 會在 import 時 setup_logger(level=Config.log_level))。
3. setattr 出來的屬性如果在上游不存在 (例如 http_api_*), 視為純 fork 加值,
   只有 fork_server 自己會讀。
"""

from __future__ import annotations
import os
from typing import Optional, Type

from fork_server.http_api.runtime_config import parse_http_api_env


# ---------- helpers ----------

def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_optional_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_csv(name: str, default: Optional[list[str]] = None) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def _set(cls: Type, attr: str, value) -> None:
    setattr(cls, attr, value)


# ---------- preset resolution ----------

def _resolve_qwen_preset(preset: str) -> dict:
    """
    Qwen preset → 推理後端組合。
    default      = ONNX on GPU (CUDA), llama 走 entrypoint.sh 決定的 backend
    low_vram_gpu = ONNX on GPU (CUDA), llama 強制 CPU
    cpu_only     = ONNX CPU, llama CPU
    """
    preset = (preset or "default").strip().lower()
    if preset == "low_vram_gpu":
        return {"onnx_provider": "CUDA", "llm_use_gpu": False}
    if preset == "cpu_only":
        return {"onnx_provider": "CPU", "llm_use_gpu": False}
    # default
    return {"onnx_provider": "CUDA", "llm_use_gpu": True}


# ---------- main apply ----------

def _absolutize_model_paths() -> None:
    """
    上游 ModelPaths 用相對路徑 (Path() / 'models' / ...), 依賴 cwd=/app。
    某些 inference 模組 (例如 qwen_asr_gguf/inference/llama.py 的 init() 與
    fun_asr_gguf 同形邏輯) 會 os.chdir 到 lib bin/ 目錄, 即便事後 chdir 回去,
    在 worker subprocess 內任何相對路徑都有風險。

    解法: 把 Args classes 中的 model_dir / *_path 改為絕對路徑, 一勞永逸。
    """
    from pathlib import Path
    from config_server import (
        ModelPaths,
        Qwen3ASRGGUFArgs,
        FunASRNanoGGUFArgs,
        ParaformerArgs,
        SenseVoiceArgs,
    )

    base_dir = Path(__file__).resolve().parents[1]  # /app (or repo root in dev)
    abs_model_root = base_dir / "models"

    def _abs(rel: str) -> str:
        return str((base_dir / rel).resolve())

    # Qwen3-ASR
    Qwen3ASRGGUFArgs.model_dir = _abs(Qwen3ASRGGUFArgs.model_dir)

    # Fun-ASR-Nano
    FunASRNanoGGUFArgs.encoder_onnx_path = _abs(FunASRNanoGGUFArgs.encoder_onnx_path)
    FunASRNanoGGUFArgs.ctc_onnx_path = _abs(FunASRNanoGGUFArgs.ctc_onnx_path)
    FunASRNanoGGUFArgs.decoder_gguf_path = _abs(FunASRNanoGGUFArgs.decoder_gguf_path)
    FunASRNanoGGUFArgs.tokens_path = _abs(FunASRNanoGGUFArgs.tokens_path)

    # Paraformer / SenseVoice (best effort, in case they get used)
    try:
        ParaformerArgs.paraformer = _abs(ParaformerArgs.paraformer)
        ParaformerArgs.tokens = _abs(ParaformerArgs.tokens)
    except Exception:
        pass
    try:
        SenseVoiceArgs.encoder_path = _abs(SenseVoiceArgs.encoder_path)
        SenseVoiceArgs.decoder_path = _abs(SenseVoiceArgs.decoder_path)
        SenseVoiceArgs.tokenizer_path = _abs(SenseVoiceArgs.tokenizer_path)
    except Exception:
        pass


def apply() -> None:
    """
    Read env vars and patch upstream config classes in-place.

    Safe to call multiple times (idempotent).
    """
    from config_server import (
        ServerConfig,
        Qwen3ASRGGUFArgs,
        FunASRNanoGGUFArgs,
    )

    # 先做絕對路徑化, 否則 subprocess 內任何 chdir 都會弄壞模型載入
    _absolutize_model_paths()

    # ---- ServerConfig ----
    if (v := _env_str("CAPSWRITER_MODEL_TYPE")):
        _set(ServerConfig, "model_type", v)
    if (v := _env_str("CAPSWRITER_SERVER_ADDR")):
        _set(ServerConfig, "addr", v)
    if (v := _env_str("CAPSWRITER_SERVER_PORT")):
        _set(ServerConfig, "port", v)
    if (v := _env_str("CAPSWRITER_LOG_LEVEL")):
        _set(ServerConfig, "log_level", v.upper())
    _set(ServerConfig, "enable_tray", _env_bool("CAPSWRITER_ENABLE_TRAY", False))

    # ---- hotwords path (Docker 內固定 /app/hot-server.txt) ----
    if (v := _env_str("CAPSWRITER_HOTWORDS_PATH")):
        from pathlib import Path
        _set(ServerConfig, "hotwords_path", Path(v))

    # ---- HTTP API (fork-only attributes) ----
    http_api = parse_http_api_env(os.environ)
    _set(ServerConfig, "http_api_enable",
         http_api.enable)
    _set(ServerConfig, "http_api_bind",
         http_api.bind)
    _set(ServerConfig, "http_api_port",
         http_api.port)
    _set(ServerConfig, "http_api_key",
         http_api.api_key)
    _set(ServerConfig, "http_api_max_upload_mb",
         http_api.max_upload_mb)
    _set(ServerConfig, "http_api_task_timeout",
         http_api.task_timeout)
    _set(ServerConfig, "http_api_cors_origins",
         list(http_api.cors_origins))

    # ---- Qwen preset (decide onnx_provider + llm_use_gpu before specific overrides) ----
    qwen_preset = _env_str("CAPSWRITER_QWEN_PRESET", "default") or "default"
    preset_cfg = _resolve_qwen_preset(qwen_preset)
    _set(Qwen3ASRGGUFArgs, "onnx_provider", preset_cfg["onnx_provider"])
    _set(Qwen3ASRGGUFArgs, "llm_use_gpu", preset_cfg["llm_use_gpu"])

    # ---- Qwen explicit env overrides (advanced) ----
    if _env_str("CAPSWRITER_QWEN_USE_CUDA") is not None:
        _set(Qwen3ASRGGUFArgs, "onnx_provider",
             "CUDA" if _env_bool("CAPSWRITER_QWEN_USE_CUDA", False) else "CPU")
    if _env_str("CAPSWRITER_QWEN_VULKAN_ENABLE") is not None:
        # vulkan implies llama uses GPU. CPU when explicitly disabled.
        _set(Qwen3ASRGGUFArgs, "llm_use_gpu",
             _env_bool("CAPSWRITER_QWEN_VULKAN_ENABLE", True))
    _set(Qwen3ASRGGUFArgs, "chunk_size",
         _env_float("CAPSWRITER_QWEN_CHUNK_SIZE", float(getattr(Qwen3ASRGGUFArgs, "chunk_size", 80.0))))
    _set(Qwen3ASRGGUFArgs, "n_ctx",
         _env_int("CAPSWRITER_QWEN_N_CTX", int(getattr(Qwen3ASRGGUFArgs, "n_ctx", 2048))))
    _set(Qwen3ASRGGUFArgs, "memory_num",
         _env_int("CAPSWRITER_QWEN_MEMORY_NUM", int(getattr(Qwen3ASRGGUFArgs, "memory_num", 1))))
    _set(Qwen3ASRGGUFArgs, "dml_pad_to",
         _env_int("CAPSWRITER_QWEN_PAD_TO", int(getattr(Qwen3ASRGGUFArgs, "dml_pad_to", 30))))

    # Qwen advanced llama overrides (only set if env explicitly provided)
    for env_name, attr in [
        ("CAPSWRITER_QWEN_LLAMA_N_BATCH", "n_batch"),
        ("CAPSWRITER_QWEN_LLAMA_N_UBATCH", "n_ubatch"),
    ]:
        if (v := _env_optional_int(env_name)) is not None:
            _set(Qwen3ASRGGUFArgs, attr, v)
    if _env_str("CAPSWRITER_QWEN_LLAMA_FLASH_ATTN") is not None:
        _set(Qwen3ASRGGUFArgs, "flash_attn",
             _env_bool("CAPSWRITER_QWEN_LLAMA_FLASH_ATTN", False))
    if _env_str("CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV") is not None:
        _set(Qwen3ASRGGUFArgs, "offload_kqv",
             _env_bool("CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV", False))

    # ---- Fun-ASR-Nano ----
    if _env_str("CAPSWRITER_FUNASR_USE_CUDA") is not None:
        _set(FunASRNanoGGUFArgs, "onnx_provider",
             "CUDA" if _env_bool("CAPSWRITER_FUNASR_USE_CUDA", False) else "CPU")
    if _env_str("CAPSWRITER_FUNASR_VULKAN_ENABLE") is not None:
        _set(FunASRNanoGGUFArgs, "llm_use_gpu",
             _env_bool("CAPSWRITER_FUNASR_VULKAN_ENABLE", True))
    if _env_str("CAPSWRITER_FUNASR_ENABLE_CTC") is not None:
        _set(FunASRNanoGGUFArgs, "enable_ctc",
             _env_bool("CAPSWRITER_FUNASR_ENABLE_CTC", True))
    _set(FunASRNanoGGUFArgs, "n_predict",
         _env_int("CAPSWRITER_FUNASR_N_PREDICT",
                  int(getattr(FunASRNanoGGUFArgs, "n_predict", 512))))
    _set(FunASRNanoGGUFArgs, "dml_pad_to",
         _env_int("CAPSWRITER_FUNASR_PAD_TO",
                  int(getattr(FunASRNanoGGUFArgs, "dml_pad_to", 30))))
    _set(FunASRNanoGGUFArgs, "max_hotwords",
         _env_int("CAPSWRITER_FUNASR_MAX_HOTWORDS",
                  int(getattr(FunASRNanoGGUFArgs, "max_hotwords", 20))))
    _set(FunASRNanoGGUFArgs, "similar_threshold",
         _env_float("CAPSWRITER_FUNASR_SIMILAR_THRESHOLD",
                    float(getattr(FunASRNanoGGUFArgs, "similar_threshold", 0.6))))
    if (v := _env_optional_int("CAPSWRITER_NUM_THREADS")) is not None:
        _set(FunASRNanoGGUFArgs, "n_threads", v)
