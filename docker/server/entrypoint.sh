#!/bin/sh
set -eu

configure_full_cpu_backend() {
  export CAPSWRITER_LLAMA_BACKEND="cpu"
  export CAPSWRITER_QWEN_VULKAN_ENABLE="false"
  export CAPSWRITER_FUNASR_VULKAN_ENABLE="false"
  export CAPSWRITER_QWEN_USE_CUDA="false"
  export CAPSWRITER_FUNASR_USE_CUDA="false"
}

configure_backend() {
  # Resolve persisted model/hardware before downloads and backend probing.
  if [ "${CAPSWRITER_SETTINGS_BOOTSTRAPPED:-}" != "1" ]; then
    exec python -m fork_server.settings_bootstrap "$0" "$@"
  fi

  inference_hardware="${CAPSWRITER_INFERENCE_HARDWARE:-${CAPSWRITER_GPU_MODE:-auto}}"
  model_type="${CAPSWRITER_MODEL_TYPE:-qwen_asr}"
  qwen_preset="${CAPSWRITER_QWEN_PRESET:-default}"
  requested_backend="${CAPSWRITER_LLAMA_BACKEND:-auto}"
  case "$requested_backend" in
    auto|cpu|vulkan|cuda) ;;
    *) echo "[capswriter] invalid CAPSWRITER_LLAMA_BACKEND: $requested_backend" >&2; exit 1 ;;
  esac
  llama_backend="cpu"

  gpu_visible="false"
  nvidia_visible="false"
  if [ -e /dev/nvidiactl ] || [ -d /dev/dri ]; then
    gpu_visible="true"
  fi
  if [ -e /dev/nvidiactl ]; then
    nvidia_visible="true"
  fi

  if [ "$inference_hardware" = "cpu" ] || [ "$requested_backend" = "cpu" ]; then
    llama_backend="cpu"
  elif [ "$requested_backend" = "cuda" ]; then
    llama_backend="cuda"
  elif [ "$gpu_visible" = "true" ]; then
    llama_backend="vulkan"
    if [ "$requested_backend" = "auto" ] && [ "$nvidia_visible" = "true" ] &&
       [ -f /opt/capswriter/llama-cuda/runtime.json ]; then
      llama_backend="cuda"
    fi
  else
    llama_backend="cpu"
  fi

  if [ "$llama_backend" = "vulkan" ] || [ "$llama_backend" = "cuda" ]; then
    export CAPSWRITER_LLAMA_BACKEND="$llama_backend"
    # Historical flags mean decoder GPU offload for either native backend.
    export CAPSWRITER_QWEN_VULKAN_ENABLE="true"
    export CAPSWRITER_FUNASR_VULKAN_ENABLE="true"
    if [ "$nvidia_visible" = "true" ]; then
      export CAPSWRITER_QWEN_USE_CUDA="true"
      export CAPSWRITER_FUNASR_USE_CUDA="true"
    else
      export CAPSWRITER_QWEN_USE_CUDA="false"
      export CAPSWRITER_FUNASR_USE_CUDA="false"
    fi
    echo "[capswriter] selected llama backend: $llama_backend"

    if [ "$model_type" = "qwen_asr" ] && [ "$qwen_preset" = "cpu_only" ]; then
      configure_full_cpu_backend
      echo "[capswriter] qwen preset cpu_only enabled: ONNX on CPU, llama on CPU"
    elif [ "$model_type" = "qwen_asr" ] && [ "$qwen_preset" = "low_vram_gpu" ]; then
      export CAPSWRITER_LLAMA_BACKEND="cpu"
      export CAPSWRITER_QWEN_VULKAN_ENABLE="false"
      export CAPSWRITER_QWEN_USE_CUDA="true"
      echo "[capswriter] qwen preset low_vram_gpu enabled: ONNX on GPU, llama on CPU"
    fi
  else
    configure_full_cpu_backend
    if [ "$inference_hardware" != "cpu" ] && [ "$nvidia_visible" = "true" ] &&
       [ "$model_type" = "qwen_asr" ] && [ "$qwen_preset" = "low_vram_gpu" ]; then
      export CAPSWRITER_QWEN_USE_CUDA="true"
      echo "[capswriter] qwen preset low_vram_gpu enabled: ONNX on GPU, llama on CPU"
      return
    fi
    if [ "$inference_hardware" = "gpu" ] || [ "$inference_hardware" = "auto" ]; then
      echo "[capswriter] GPU runtime unavailable, falling back to CPU backend"
    else
      echo "[capswriter] CPU mode forced"
    fi
  fi
}

gpu_backend_configured() {
  [ "${CAPSWRITER_LLAMA_BACKEND:-cpu}" = "cuda" ] ||
    [ "${CAPSWRITER_LLAMA_BACKEND:-cpu}" = "vulkan" ] ||
    [ "${CAPSWRITER_QWEN_USE_CUDA:-false}" = "true" ] ||
    [ "${CAPSWRITER_FUNASR_USE_CUDA:-false}" = "true" ] ||
    [ "${CAPSWRITER_QWEN_VULKAN_ENABLE:-false}" = "true" ] ||
    [ "${CAPSWRITER_FUNASR_VULKAN_ENABLE:-false}" = "true" ]
}

fallback_to_cpu() {
  configure_full_cpu_backend
  echo "[capswriter] selected GPU backend failed, retrying the full CPU backend"
}

prepare_and_probe_cpu_fallback() {
  if [ "${CAPSWRITER_LLAMA_BACKEND:-cpu}" = "cuda" ]; then
    echo "[capswriter] CUDA backend failed; refusing CPU fallback" >&2
    exit 1
  fi
  fallback_to_cpu
  python /app/docker/server/download_models.py
  if ! python /app/docker/server/probe_backend.py; then
    echo "[capswriter] full CPU backend probe failed; refusing to start" >&2
    exit 1
  fi
}

configure_backend "$@"

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

if gpu_backend_configured; then
  if ! python /app/docker/server/download_models.py; then
    echo "[capswriter] selected GPU backend bootstrap failed; trying the full CPU backend" >&2
    prepare_and_probe_cpu_fallback
  elif ! python /app/docker/server/probe_backend.py; then
    prepare_and_probe_cpu_fallback
  fi
else
  python /app/docker/server/download_models.py
fi

exec python /app/start_server_docker.py
