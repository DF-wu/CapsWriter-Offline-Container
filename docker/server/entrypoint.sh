#!/bin/sh
set -eu

print_qwen_summary() {
  python - <<'PY'
from config_server import Qwen3ASRGGUFArgs
for line in Qwen3ASRGGUFArgs.summary_lines():
    print(line)
PY
}

export_resolved_runtime() {
  eval "$(python - <<'PY'
from config_server import Qwen3ASRGGUFArgs
print(f'export CAPSWRITER_LLAMA_BACKEND={Qwen3ASRGGUFArgs.resolved_llama_backend}')
print(f'export CAPSWRITER_QWEN_USE_CUDA={str(Qwen3ASRGGUFArgs.use_cuda).lower()}')
print(f'export CAPSWRITER_QWEN_USE_DML={str(Qwen3ASRGGUFArgs.use_dml).lower()}')
print(f'export CAPSWRITER_QWEN_VULKAN_ENABLE={str(Qwen3ASRGGUFArgs.vulkan_enable).lower()}')
print(f'export CAPSWRITER_QWEN_VULKAN_FORCE_FP32={str(Qwen3ASRGGUFArgs.vulkan_force_fp32).lower()}')
PY
)"

  # Fun-ASR still follows the old hardware-first behavior for now. Keep it stable,
  # but do not let it affect the Qwen resolver semantics.
  inference_hardware="${CAPSWRITER_INFERENCE_HARDWARE:-${CAPSWRITER_GPU_MODE:-auto}}"
  if [ "$inference_hardware" = "cpu" ]; then
    export CAPSWRITER_FUNASR_USE_CUDA="false"
    export CAPSWRITER_FUNASR_VULKAN_ENABLE="false"
  elif [ -e /dev/nvidiactl ] || [ -d /dev/dri ]; then
    export CAPSWRITER_FUNASR_VULKAN_ENABLE="true"
    if [ -e /dev/nvidiactl ]; then
      export CAPSWRITER_FUNASR_USE_CUDA="true"
    else
      export CAPSWRITER_FUNASR_USE_CUDA="false"
    fi
  else
    export CAPSWRITER_FUNASR_USE_CUDA="false"
    export CAPSWRITER_FUNASR_VULKAN_ENABLE="false"
  fi
}

fallback_qwen_onnx_to_cpu() {
  export CAPSWRITER_QWEN_USE_CUDA="false"
  export CAPSWRITER_QWEN_USE_DML="false"
  echo "[capswriter] ONNX GPU probe failed, retrying Qwen ONNX on CPU while keeping resolved llama backend"
}

export_resolved_runtime
print_qwen_summary

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

python /app/docker/server/download_models.py

if [ "${CAPSWRITER_MODEL_TYPE:-qwen_asr}" = "qwen_asr" ]; then
  if ! python /app/docker/server/probe_backend.py; then
    fallback_qwen_onnx_to_cpu
    python /app/docker/server/download_models.py
  fi
elif [ "${CAPSWRITER_LLAMA_BACKEND:-cpu}" = "vulkan" ]; then
  if ! python /app/docker/server/probe_backend.py; then
    export CAPSWRITER_FUNASR_USE_CUDA="false"
    echo "[capswriter] Fun-ASR GPU probe failed, retrying ONNX on CPU while keeping llama backend"
    python /app/docker/server/download_models.py
  fi
fi

exec python /app/start_server.py
