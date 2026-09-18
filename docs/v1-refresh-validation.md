# v1 refresh validation — 2026-09-19

Validated the `feat/v1-upstream-refresh-20260918` source on axolotl, integrating
upstream `84912d5` while retaining the fork's paired llama `b7798` ABI.
No existing service was upgraded. Model/audio sources were mounted read-only;
the inference server ran inside a disposable container with no published ports,
no external network, a 4 CPU limit and a 12 GiB memory limit.

## Regression and configuration checks

- Linux Python 3.12 isolated Docker suite: **391 tests passed** (16 maintenance,
  158 core/client, 153 HTTP API, 64 container bootstrap tests).
- Linux Python 3.10.20 isolated temporary virtual environment: the same
  **391 tests passed**. A Python 3.10 Docker base-image pull stalled and was
  stopped; this second result is native, not a Python 3.10 container result.
- Python 3.10 syntax parsing: **348 modules passed**.
- `uv lock --check --offline`: passed (268 packages resolved).
- Base, NVIDIA, and Intel/AMD Compose configurations: passed `config --quiet`.
- `bash -n docker/server/entrypoint.sh`: passed.
- Migrated benchmark entrypoint: `python 04-Benchmark.py --help` passed in the
  native-runtime test container.

The headless suite is reproducible without model downloads:

```bash
docker build -f docker/server/Dockerfile.test -t capswriter-v1-tests:local .
docker run --rm --init --network none \
  -e CAPSWRITER_LOG_DIR=/tmp/capswriter-logs capswriter-v1-tests:local
```

Use `--build-arg PYTHON_VERSION=3.10` for the older Python image. The test
Dockerfile has its own ignore file because release/build source is needed by
the regression suite, while local settings, secrets and model weights remain
excluded. `--init` reaps process descendants used by timeout-cleanup tests.

## Real CPU inference

Runtime dependencies came from `requirements-server-docker.lock`. The disposable
runtime used Python 3.12, FFmpeg, Qwen3-ASR-1.7B fp16 frontend/backend ONNX models,
the q4_k GGUF decoder, and existing llama `b7798` CPU libraries. Canonical model
filenames and all engine library directories were prepared in container scratch
storage with symlinks to read-only assets.

Input: the existing 5.592-second `zh.wav` fixture under
`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/test_wavs`.

| Check | Observed result |
| --- | --- |
| `/health` | `status=ok`, `model=qwen_asr`, application version `2.6` |
| `/ready` | HTTP 200; router bound, worker alive, FFmpeg available |
| WebSocket microphone message | Final result returned in 53.44 s |
| Authenticated HTTP `model=whisper-1` upload | HTTP 200 in 50.59 s |
| Both transcription texts | `開放時間：早上九點至下午五點。` |

These timings describe a constrained smoke test, not a performance benchmark.
ForcedAligner assets were absent, so precise word alignment was not qualified;
the service used its fallback timestamps. The initial scratch attempts exposed
missing native-library aliases in the test setup; preparing both unversioned
and versioned `.so` files resolved that setup issue. The downloader now also
installs verified libraries into the new shared `engines/llama/bin` directory,
covered by an extraction regression test.

## Qualification limits

The CUDA production Dockerfile was not rebuilt and a cold online model download
was not exercised. This validation does not qualify GPU inference, English
known-audio accuracy, all supported models, Windows desktop hotkeys/microphone/
clipboard/tray, or Windows packaged artifacts. Those remain explicit release
checks. No production token, recording, model, volume or service was modified.
