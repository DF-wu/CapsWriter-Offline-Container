import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

# 版本信息
__version__ = "2.5-alpha"

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


QWEN_PRESET_ALIASES = {
    "default": "default",
    "low_vram_gpu": "default",
    "balanced": "default",
    "quality": "default",
    "defaults": "default",
    "cpu_only": "cpu_only",
    "cpu": "cpu_only",
}

QWEN_PRESET_DEFAULTS: dict[str, dict[str, object]] = {
    "default": {
        # Official recommended profile for this fork on the validated P4/Pascal path.
        # ONNX encoder uses GPU when available; llama / GGUF stays on CPU.
        "use_cuda": True,
        "use_dml": False,
        "vulkan_enable": False,
        "llama_backend": "cpu",
        # Runtime tuning defaults. Explicit env values override these.
        "n_predict": 512,
        "n_threads": None,
        "n_ctx": 2048,
        "chunk_size": 40.0,
        "memory_num": 1,
        "pad_to": 30,
        "llama_n_batch": 4096,
        "llama_n_ubatch": 512,
        "llama_flash_attn": True,
        "llama_offload_kqv": True,
    },
    "cpu_only": {
        # Force the whole Qwen path to CPU.
        "use_cuda": False,
        "use_dml": False,
        "vulkan_enable": False,
        "llama_backend": "cpu",
        # Keep the same tuning defaults unless explicitly overridden.
        "n_predict": 512,
        "n_threads": None,
        "n_ctx": 2048,
        "chunk_size": 40.0,
        "memory_num": 1,
        "pad_to": 30,
        "llama_n_batch": 4096,
        "llama_n_ubatch": 512,
        "llama_flash_attn": True,
        "llama_offload_kqv": True,
    },
}


@dataclass(frozen=True)
class ResolvedQwenRuntimeConfig:
    raw_preset: str
    normalized_preset: str
    inference_hardware: str
    gpu_visible: bool
    nvidia_visible: bool
    use_cuda: bool
    use_dml: bool
    vulkan_enable: bool
    vulkan_force_fp32: bool
    llama_backend: str
    onnx_backend: str
    resolved_profile: str
    n_predict: int
    n_threads: Optional[int]
    n_ctx: int
    chunk_size: float
    memory_num: int
    pad_to: int
    llama_n_batch: int
    llama_n_ubatch: int
    llama_flash_attn: bool
    llama_offload_kqv: bool
    value_sources: dict[str, str]
    notes: tuple[str, ...]


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    return default


def _env_optional_bool(name: str) -> Optional[bool]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip().lower()
    if not value:
        return None
    if value in _BOOL_TRUE:
        return True
    if value in _BOOL_FALSE:
        return False
    return None


def _env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    return int(value)


def _env_present(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and bool(value.strip())


def _env_optional_float(name: str, default: Optional[float]) -> Optional[float]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    return float(value)


def _detect_gpu_visible() -> bool:
    explicit = _env_optional_bool("CAPSWRITER_RUNTIME_GPU_VISIBLE")
    if explicit is not None:
        return explicit
    return os.path.exists("/dev/nvidiactl") or os.path.isdir("/dev/dri")



def _detect_nvidia_visible() -> bool:
    explicit = _env_optional_bool("CAPSWRITER_RUNTIME_NVIDIA_VISIBLE")
    if explicit is not None:
        return explicit
    return os.path.exists("/dev/nvidiactl")



def _normalize_qwen_preset(value: str) -> str:
    normalized = value.strip().lower() if value else "default"
    return QWEN_PRESET_ALIASES.get(normalized, "default")



def _qwen_preset_name() -> str:
    return _normalize_qwen_preset(_env_str("CAPSWRITER_QWEN_PRESET", "default"))



def _normalize_inference_hardware(value: str) -> str:
    normalized = value.strip().lower() if value else "auto"
    if normalized in {"auto", "gpu", "cpu"}:
        return normalized
    return "auto"



def _qwen_preset_defaults(preset: Optional[str] = None) -> dict[str, object]:
    active = preset or _qwen_preset_name()
    defaults = QWEN_PRESET_DEFAULTS.get(active)
    if defaults is None:
        defaults = QWEN_PRESET_DEFAULTS["default"]
    return dict(defaults)



def _qwen_env_override_int(
    resolved: dict[str, object],
    sources: dict[str, str],
    env_name: str,
    key: str,
) -> None:
    if _env_present(env_name):
        resolved[key] = _env_optional_int(env_name, cast(Optional[int], resolved[key])) or cast(int, resolved[key])
        sources[key] = f"env:{env_name}"



def _qwen_env_override_float(
    resolved: dict[str, object],
    sources: dict[str, str],
    env_name: str,
    key: str,
) -> None:
    if _env_present(env_name):
        resolved[key] = _env_optional_float(env_name, cast(Optional[float], resolved[key])) or cast(float, resolved[key])
        sources[key] = f"env:{env_name}"



def _qwen_env_override_bool(
    resolved: dict[str, object],
    sources: dict[str, str],
    env_name: str,
    key: str,
) -> None:
    if _env_present(env_name):
        resolved[key] = _env_bool(env_name, cast(bool, resolved[key]))
        sources[key] = f"env:{env_name}"



def _resolve_qwen_profile_name(onnx_backend: str, llama_backend: str) -> str:
    if onnx_backend == "cpu" and llama_backend == "cpu":
        return "cpu_only"
    if onnx_backend in {"cuda", "dml"} and llama_backend == "cpu":
        return "onnx_gpu_llama_cpu"
    if onnx_backend in {"cuda", "dml"} and llama_backend == "vulkan":
        return "onnx_gpu_llama_vulkan"
    return f"onnx_{onnx_backend}_llama_{llama_backend}"



def resolve_qwen_runtime_config() -> ResolvedQwenRuntimeConfig:
    raw_preset = _env_str("CAPSWRITER_QWEN_PRESET", "default")
    normalized_preset = _normalize_qwen_preset(raw_preset)
    inference_hardware = _normalize_inference_hardware(
        _env_str("CAPSWRITER_INFERENCE_HARDWARE", _env_str("CAPSWRITER_GPU_MODE", "auto"))
    )
    gpu_visible = _detect_gpu_visible()
    nvidia_visible = _detect_nvidia_visible()

    resolved = _qwen_preset_defaults(normalized_preset)
    sources = {key: f"preset:{normalized_preset}" for key in resolved.keys()}
    notes: list[str] = []

    # Advanced / internal backend overrides. These are explicit env overrides that
    # intentionally sit above preset defaults, but can still be constrained by the
    # hardware policy and runtime availability checks below.
    _qwen_env_override_bool(resolved, sources, "CAPSWRITER_QWEN_USE_CUDA", "use_cuda")
    _qwen_env_override_bool(resolved, sources, "CAPSWRITER_QWEN_USE_DML", "use_dml")
    _qwen_env_override_bool(resolved, sources, "CAPSWRITER_QWEN_VULKAN_ENABLE", "vulkan_enable")
    _qwen_env_override_bool(
        resolved,
        sources,
        "CAPSWRITER_QWEN_VULKAN_FORCE_FP32",
        "vulkan_force_fp32",
    )

    # Tunable Qwen settings. Explicit env values override the preset defaults.
    _qwen_env_override_int(resolved, sources, "CAPSWRITER_QWEN_N_PREDICT", "n_predict")
    _qwen_env_override_int(resolved, sources, "CAPSWRITER_NUM_THREADS", "n_threads")
    _qwen_env_override_int(resolved, sources, "CAPSWRITER_QWEN_N_CTX", "n_ctx")
    _qwen_env_override_float(resolved, sources, "CAPSWRITER_QWEN_CHUNK_SIZE", "chunk_size")
    _qwen_env_override_int(resolved, sources, "CAPSWRITER_QWEN_MEMORY_NUM", "memory_num")
    _qwen_env_override_int(resolved, sources, "CAPSWRITER_QWEN_PAD_TO", "pad_to")
    _qwen_env_override_int(
        resolved,
        sources,
        "CAPSWRITER_QWEN_LLAMA_N_BATCH",
        "llama_n_batch",
    )
    _qwen_env_override_int(
        resolved,
        sources,
        "CAPSWRITER_QWEN_LLAMA_N_UBATCH",
        "llama_n_ubatch",
    )
    _qwen_env_override_bool(
        resolved,
        sources,
        "CAPSWRITER_QWEN_LLAMA_FLASH_ATTN",
        "llama_flash_attn",
    )
    _qwen_env_override_bool(
        resolved,
        sources,
        "CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV",
        "llama_offload_kqv",
    )

    # Hardware policy is the final authority. It can still constrain an explicitly
    # overridden backend request because CPU-only startup must remain deterministic.
    if inference_hardware == "cpu":
        resolved["use_cuda"] = False
        sources["use_cuda"] = "hardware:CAPSWRITER_INFERENCE_HARDWARE=cpu"
        resolved["use_dml"] = False
        sources["use_dml"] = "hardware:CAPSWRITER_INFERENCE_HARDWARE=cpu"
        resolved["vulkan_enable"] = False
        sources["vulkan_enable"] = "hardware:CAPSWRITER_INFERENCE_HARDWARE=cpu"
        resolved["llama_backend"] = "cpu"
        sources["llama_backend"] = "hardware:CAPSWRITER_INFERENCE_HARDWARE=cpu"
        notes.append("hardware policy forced CPU-only runtime")
    else:
        # CUDA requires visible NVIDIA runtime.
        if cast(bool, resolved["use_cuda"]) and not nvidia_visible:
            resolved["use_cuda"] = False
            sources["use_cuda"] = "runtime-fallback:no-visible-nvidia-runtime"
            notes.append("CUDA unavailable; ONNX fell back to CPU")

        # DML is not a Linux deployment path, but we preserve the override semantics.
        if cast(bool, resolved["use_dml"]) and not gpu_visible:
            resolved["use_dml"] = False
            sources["use_dml"] = "runtime-fallback:no-visible-gpu-runtime"
            notes.append("DML unavailable; ONNX fell back to CPU")

        # Vulkan requires visible GPU runtime.
        if cast(bool, resolved["vulkan_enable"]) and not gpu_visible:
            resolved["vulkan_enable"] = False
            sources["vulkan_enable"] = "runtime-fallback:no-visible-gpu-runtime"
            notes.append("Vulkan unavailable; llama fell back to CPU")

        # If CUDA is active, prefer it over DML for the resolved backend label.
        if cast(bool, resolved["use_cuda"]):
            resolved["use_dml"] = False
            if sources.get("use_dml", "").startswith("env:"):
                notes.append("explicit DML override ignored because CUDA is active")
            sources["use_dml"] = sources.get("use_dml", f"preset:{normalized_preset}")

    if cast(bool, resolved["vulkan_enable"]):
        resolved["llama_backend"] = "vulkan"
        if not sources.get("llama_backend"):
            sources["llama_backend"] = "derived:vulkan_enable"
    else:
        resolved["llama_backend"] = "cpu"
        sources["llama_backend"] = sources.get("llama_backend", "derived:vulkan_enable")

    onnx_backend = "cpu"
    if cast(bool, resolved["use_cuda"]):
        onnx_backend = "cuda"
    elif cast(bool, resolved["use_dml"]):
        onnx_backend = "dml"

    resolved_profile = _resolve_qwen_profile_name(
        onnx_backend=onnx_backend,
        llama_backend=cast(str, resolved["llama_backend"]),
    )

    return ResolvedQwenRuntimeConfig(
        raw_preset=raw_preset,
        normalized_preset=normalized_preset,
        inference_hardware=inference_hardware,
        gpu_visible=gpu_visible,
        nvidia_visible=nvidia_visible,
        use_cuda=cast(bool, resolved["use_cuda"]),
        use_dml=cast(bool, resolved["use_dml"]),
        vulkan_enable=cast(bool, resolved["vulkan_enable"]),
        vulkan_force_fp32=cast(bool, resolved.get("vulkan_force_fp32", False)),
        llama_backend=cast(str, resolved["llama_backend"]),
        onnx_backend=onnx_backend,
        resolved_profile=resolved_profile,
        n_predict=cast(int, resolved["n_predict"]),
        n_threads=cast(Optional[int], resolved.get("n_threads")),
        n_ctx=cast(int, resolved["n_ctx"]),
        chunk_size=cast(float, resolved["chunk_size"]),
        memory_num=cast(int, resolved["memory_num"]),
        pad_to=cast(int, resolved["pad_to"]),
        llama_n_batch=cast(int, resolved["llama_n_batch"]),
        llama_n_ubatch=cast(int, resolved["llama_n_ubatch"]),
        llama_flash_attn=cast(bool, resolved["llama_flash_attn"]),
        llama_offload_kqv=cast(bool, resolved["llama_offload_kqv"]),
        value_sources=dict(sources),
        notes=tuple(notes),
    )


_QWEN_RESOLVED = resolve_qwen_runtime_config()


def _qwen_preset_int(env_name: str, preset_key: str, fallback: int) -> int:
    if _env_present(env_name):
        return _env_optional_int(env_name, fallback) or fallback
    return int(cast(int, _qwen_preset_defaults().get(preset_key, fallback)))



def _qwen_preset_float(env_name: str, preset_key: str, fallback: float) -> float:
    if _env_present(env_name):
        return _env_optional_float(env_name, fallback) or fallback
    return float(cast(float, _qwen_preset_defaults().get(preset_key, fallback)))



def _qwen_preset_bool(env_name: str, preset_key: str, fallback: bool) -> bool:
    if _env_present(env_name):
        return _env_bool(env_name, fallback)
    return bool(cast(bool, _qwen_preset_defaults().get(preset_key, fallback)))


# 服务端配置
class ServerConfig:
    addr = _env_str("CAPSWRITER_SERVER_ADDR", "0.0.0.0")
    port = _env_str("CAPSWRITER_SERVER_PORT", "6016")

    # 语音模型选择：'fun_asr_nano', 'sensevoice', 'paraformer', 'qwen_asr'
    model_type = _env_str("CAPSWRITER_MODEL_TYPE", "qwen_asr")
    inference_hardware = _QWEN_RESOLVED.inference_hardware

    format_num = True  # 输出时是否将中文数字转为阿拉伯数字
    format_spell = True  # 输出时是否调整中英之间的空格

    enable_tray = _env_bool("CAPSWRITER_ENABLE_TRAY", True)  # 是否启用托盘图标功能

    # 日志配置
    log_dir = _env_str("CAPSWRITER_LOG_DIR", os.path.join(BASE_DIR, "logs"))
    log_level = _env_str(
        "CAPSWRITER_LOG_LEVEL", "INFO"
    )  # 日志级别：'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'


class ModelDownloadLinks:
    """模型下载链接配置"""

    # 统一导向 GitHub Release 模型页面
    models_page = "https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models"


class ModelPaths:
    """模型文件路径配置"""

    # 基础目录
    model_dir = Path() / "models"

    # Paraformer 模型路径
    paraformer_dir = (
        model_dir
        / "Paraformer"
        / "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-onnx"
    )
    paraformer_model = paraformer_dir / "model.onnx"
    paraformer_tokens = paraformer_dir / "tokens.txt"

    # 标点模型路径
    punc_model_dir = (
        model_dir
        / "Punct-CT-Transformer"
        / "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12"
        / "model.onnx"
    )

    # SenseVoice 模型路径，自带标点
    sensevoice_dir = (
        model_dir
        / "SenseVoice-Small"
        / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    )
    sensevoice_model = sensevoice_dir / "model.onnx"
    sensevoice_tokens = sensevoice_dir / "tokens.txt"

    # Fun-ASR-Nano 模型路径，自带标点
    fun_asr_nano_gguf_dir = model_dir / "Fun-ASR-Nano" / "Fun-ASR-Nano-GGUF"
    fun_asr_nano_gguf_encoder_adaptor = (
        fun_asr_nano_gguf_dir / "Fun-ASR-Nano-Encoder-Adaptor.int4.onnx"
    )
    fun_asr_nano_gguf_ctc = fun_asr_nano_gguf_dir / "Fun-ASR-Nano-CTC.int4.onnx"
    fun_asr_nano_gguf_llm_decode = (
        fun_asr_nano_gguf_dir / "Fun-ASR-Nano-Decoder.q5_k.gguf"
    )
    fun_asr_nano_gguf_token = fun_asr_nano_gguf_dir / "tokens.txt"
    fun_asr_nano_gguf_hotwords = Path() / "hot-server.txt"

    # Qwen3-ASR 模型路径，自带标点
    qwen3_asr_gguf_dir = model_dir / "Qwen3-ASR" / "Qwen3-ASR-1.7B"
    qwen3_asr_gguf_encoder_frontend = (
        qwen3_asr_gguf_dir / "qwen3_asr_encoder_frontend.fp16.onnx"
    )
    qwen3_asr_gguf_encoder_backend = (
        qwen3_asr_gguf_dir / "qwen3_asr_encoder_backend.fp16.onnx"
    )
    qwen3_asr_gguf_llm_decode = qwen3_asr_gguf_dir / "qwen3_asr_llm.q4_k.gguf"


class ParaformerArgs:
    """Paraformer 模型参数配置"""

    paraformer = ModelPaths.paraformer_model.as_posix()
    tokens = ModelPaths.paraformer_tokens.as_posix()
    num_threads = _env_optional_int("CAPSWRITER_NUM_THREADS", 4)
    sample_rate = 16000
    feature_dim = 80
    decoding_method = "greedy_search"
    provider = "cpu"
    debug = False


class SenseVoiceArgs:
    """SenseVoice 模型参数配置"""

    model = ModelPaths.sensevoice_model.as_posix()
    tokens = ModelPaths.sensevoice_tokens.as_posix()
    use_itn = True
    language = "zh"
    num_threads = _env_optional_int("CAPSWRITER_NUM_THREADS", 4)
    provider = "cpu"
    debug = False


class FunASRNanoGGUFArgs:
    """Fun-ASR-Nano-GGUF 模型参数配置"""

    # 模型路径
    encoder_onnx_path = ModelPaths.fun_asr_nano_gguf_encoder_adaptor.as_posix()
    ctc_onnx_path = ModelPaths.fun_asr_nano_gguf_ctc.as_posix()
    decoder_gguf_path = ModelPaths.fun_asr_nano_gguf_llm_decode.as_posix()
    tokens_path = ModelPaths.fun_asr_nano_gguf_token.as_posix()
    hotwords_path = ModelPaths.fun_asr_nano_gguf_hotwords.as_posix()

    use_cuda = _env_bool("CAPSWRITER_FUNASR_USE_CUDA", False)

    # 显卡加速
    dml_enable = _env_bool(
        "CAPSWRITER_FUNASR_DML_ENABLE", False
    )  # 是否启用 DirectML 加速 ONNX 模型，实测 AMD 显卡上会慢，因此默认关闭，建议N卡开启
    vulkan_enable = _env_bool(
        "CAPSWRITER_FUNASR_VULKAN_ENABLE", True
    )  # 是否启用 Vulkan 加速 GGUF 模型
    vulkan_force_fp32 = _env_bool(
        "CAPSWRITER_FUNASR_VULKAN_FORCE_FP32", False
    )  # 是否强制 FP32 计算（如果 GPU 是 Intel 集显且出现精度溢出，可设为 True）

    # 模型细节
    enable_ctc = _env_bool(
        "CAPSWRITER_FUNASR_ENABLE_CTC", True
    )  # 是否启用 CTC 热词检索
    n_predict = (
        _env_optional_int("CAPSWRITER_FUNASR_N_PREDICT", 512) or 512
    )  # LLM 最大生成 token 数
    n_threads = _env_optional_int(
        "CAPSWRITER_NUM_THREADS", None
    )  # 线程数，None 表示自动
    similar_threshold = float(
        os.getenv("CAPSWRITER_FUNASR_SIMILAR_THRESHOLD", "0.6")
    )  # 热词相似度阈值
    max_hotwords = (
        _env_optional_int("CAPSWRITER_FUNASR_MAX_HOTWORDS", 20) or 20
    )  # 每次替换的最大热词数
    pad_to = (
        _env_optional_int("CAPSWRITER_FUNASR_PAD_TO", 30) or 30
    )  # GPU/DML 预热填充长度（秒）
    verbose = False


class Qwen3ASRGGUFArgs:
    """Qwen3-ASR-GGUF 模型参数配置。

    解析优先序：
    1. 内建 baseline
    2. preset normalization (`default` / `cpu_only`, plus legacy aliases)
    3. preset defaults
    4. 显式 env 覆盖
    5. `CAPSWRITER_INFERENCE_HARDWARE` 约束
    6. runtime fallback（例如容器内看不到 NVIDIA runtime 时关闭 CUDA）
    """

    # 模型路径
    model_dir = ModelPaths.qwen3_asr_gguf_dir.as_posix()
    encoder_frontend_fn = ModelPaths.qwen3_asr_gguf_encoder_frontend.name
    encoder_backend_fn = ModelPaths.qwen3_asr_gguf_encoder_backend.name
    llm_fn = ModelPaths.qwen3_asr_gguf_llm_decode.name

    # Preset metadata.
    raw_preset = _QWEN_RESOLVED.raw_preset
    preset = _QWEN_RESOLVED.normalized_preset
    inference_hardware = _QWEN_RESOLVED.inference_hardware
    resolved_profile = _QWEN_RESOLVED.resolved_profile
    resolved_onnx_backend = _QWEN_RESOLVED.onnx_backend
    resolved_llama_backend = _QWEN_RESOLVED.llama_backend
    gpu_visible = _QWEN_RESOLVED.gpu_visible
    nvidia_visible = _QWEN_RESOLVED.nvidia_visible
    value_sources = _QWEN_RESOLVED.value_sources
    notes = _QWEN_RESOLVED.notes

    # Backend selection. These fields are the actual values used by engine creation.
    use_cuda = _QWEN_RESOLVED.use_cuda
    use_dml = _QWEN_RESOLVED.use_dml
    vulkan_enable = _QWEN_RESOLVED.vulkan_enable
    vulkan_force_fp32 = _QWEN_RESOLVED.vulkan_force_fp32

    # Runtime tuning values. Explicit env values override the preset defaults.
    n_predict = _QWEN_RESOLVED.n_predict  # LLM 最大生成 token 数
    n_threads = _QWEN_RESOLVED.n_threads  # 线程数，None 表示自动
    n_ctx = _QWEN_RESOLVED.n_ctx  # 上下文窗口大小
    chunk_size = _QWEN_RESOLVED.chunk_size  # 分段长度（秒）
    memory_num = _QWEN_RESOLVED.memory_num  # 参与拼接的历史记忆片段数
    pad_to = _QWEN_RESOLVED.pad_to  # GPU/DML 预热填充长度（秒）
    llama_n_batch = _QWEN_RESOLVED.llama_n_batch  # llama.cpp 逻辑 batch
    llama_n_ubatch = _QWEN_RESOLVED.llama_n_ubatch  # llama.cpp 物理 batch
    llama_flash_attn = _QWEN_RESOLVED.llama_flash_attn  # flash attention 开关
    llama_offload_kqv = _QWEN_RESOLVED.llama_offload_kqv  # KQV offload 开关
    verbose = False

    @classmethod
    def summary_lines(cls) -> list[str]:
        lines = [
            (
                "[qwen-config] "
                f"raw_preset={cls.raw_preset} normalized_preset={cls.preset} "
                f"profile={cls.resolved_profile} inference_hardware={cls.inference_hardware}"
            ),
            (
                "[qwen-config] "
                f"onnx_backend={cls.resolved_onnx_backend} llama_backend={cls.resolved_llama_backend} "
                f"gpu_visible={cls.gpu_visible} nvidia_visible={cls.nvidia_visible}"
            ),
            (
                "[qwen-config] "
                f"n_ctx={cls.n_ctx} chunk_size={cls.chunk_size} memory_num={cls.memory_num} "
                f"pad_to={cls.pad_to} n_predict={cls.n_predict} n_threads={cls.n_threads}"
            ),
            (
                "[qwen-config] "
                f"llama_n_batch={cls.llama_n_batch} llama_n_ubatch={cls.llama_n_ubatch} "
                f"flash_attn={cls.llama_flash_attn} offload_kqv={cls.llama_offload_kqv}"
            ),
        ]
        if cls.notes:
            lines.append(f"[qwen-config] notes={'; '.join(cls.notes)}")
        return lines
