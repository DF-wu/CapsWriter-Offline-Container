"""Validated opt-in NVIDIA memory-clock controls; parsing never touches hardware."""

from dataclasses import dataclass
from collections.abc import Mapping

from fork_server.http_api.runtime_config import (
    ConfigError, parse_bool, parse_float_range, parse_int_range,
)


@dataclass(frozen=True)
class GpuBoostConfig:
    enabled: bool
    device: int
    memory_clock_mhz: int
    idle_seconds: float
    command_timeout: float


def parse_gpu_boost_config(environment: Mapping[str, str]) -> GpuBoostConfig:
    enabled = parse_bool(environment, "CAPSWRITER_GPU_BOOST_ENABLED", False)
    device = parse_int_range(environment, "CAPSWRITER_GPU_BOOST_DEVICE", 0,
                             minimum=0, maximum=1023)
    clock = parse_int_range(environment, "CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ", 0,
                            minimum=0, maximum=100000)
    if enabled and clock == 0:
        raise ConfigError("CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ must be > 0 when GPU boost is enabled")
    idle = parse_float_range(environment, "CAPSWRITER_GPU_BOOST_IDLE_SECONDS", 1.0,
                             minimum=0.1, maximum=3600)
    timeout = parse_float_range(environment, "CAPSWRITER_GPU_BOOST_TIMEOUT", 5.0,
                                minimum=0.1, maximum=60)
    return GpuBoostConfig(enabled, device, clock, idle, timeout)


def apply_gpu_boost_config(target, environment: Mapping[str, str]) -> None:
    config = parse_gpu_boost_config(environment)
    target.gpu_boost_enabled = config.enabled
    target.gpu_unboost_timeout = config.idle_seconds
    # Argument arrays keep settings values out of a shell. Never target all GPUs.
    target.gpu_boost_cmd = ["nvidia-smi", "-i", str(config.device),
                            "-lmc", str(config.memory_clock_mhz)]
    target.gpu_unboost_cmd = ["nvidia-smi", "-i", str(config.device), "-rmc"]


def gpu_boost_shutdown_timeout(environment: Mapping[str, str]) -> float:
    """Allow one in-flight command, its reset, and bounded process cleanup."""
    timeout = parse_float_range(environment, "CAPSWRITER_GPU_BOOST_TIMEOUT", 5.0,
                                minimum=0.1, maximum=60)
    # Windows may need bounded taskkill plus process.wait (two seconds each).
    return 2 * (timeout + 4.0) + 1.0


def _loaded_gpu_backend_active(model_type: str, recognizer) -> bool:
    """只读取各引擎明确拥有的会话，不递归探测任意对象。"""
    if model_type == "sensevoice":
        sessions = (recognizer.engine.encoder.session, recognizer.engine.decoder.session)
        llm_gpu = False
    elif model_type == "qwen_asr":
        encoder = recognizer.engine.encoder
        sessions = (encoder.sess_fe, encoder.sess_be)
        # 当前 GGUF binding 不暴露已加载模型的实际卸载层数；保留加载时的选择。
        llm_gpu = bool(recognizer.config.llm_use_gpu and recognizer.engine.model is not None)
    elif model_type == "fun_asr_nano":
        sessions = (recognizer.models.encoder.sess, recognizer.models.ctc_decoder.sess)
        llm_gpu = bool(recognizer.config.llm_use_gpu and recognizer.models.model is not None)
    elif model_type == "paraformer":
        # Sherpa 的 OfflineRecognizer 不暴露内部 ORT session，使用实例的加载配置。
        return str(recognizer.config.provider).upper() in {"CUDA", "DML", "TRT", "TENSORRT"}
    else:
        return False

    gpu_providers = {"CUDAExecutionProvider", "DmlExecutionProvider", "TensorrtExecutionProvider"}
    return llm_gpu or any(gpu_providers.intersection(session.get_providers()) for session in sessions)


def configure_gpu_boost_backend(target, environment: Mapping[str, str], *, recognizer=None) -> None:
    """启动前按配置判断；worker 加载后用实例更新，在接收 mic hint／音频前完成。"""
    if str(environment.get("CAPSWRITER_INFERENCE_HARDWARE", "")).strip().lower() == "cpu":
        target.gpu_boost_backend_active = False
        return

    model_type = getattr(target, "model_type", "qwen_asr").lower()
    if recognizer is not None:
        target.gpu_boost_backend_active = _loaded_gpu_backend_active(model_type, recognizer)
        return

    from config_server import (
        FunASRNanoGGUFArgs, ParaformerArgs, Qwen3ASRGGUFArgs, SenseVoiceArgs,
    )

    args = {
        "qwen_asr": Qwen3ASRGGUFArgs,
        "fun_asr_nano": FunASRNanoGGUFArgs,
        "sensevoice": SenseVoiceArgs,
        "paraformer": ParaformerArgs,
    }.get(model_type)
    provider = str(getattr(args, "onnx_provider", getattr(args, "provider", "CPU"))).upper()
    target.gpu_boost_backend_active = (
        bool(getattr(args, "llm_use_gpu", False))
        or provider in {"CUDA", "DML", "TRT", "TENSORRT"}
    )
