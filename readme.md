# CapsWriter-Offline Linux Server Fork

> Offline speech recognition that **runs** on Linux servers and **speaks** OpenAI Whisper's API.

[![License](https://img.shields.io/badge/license-MIT-blue)](#)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![OpenAI-Compatible](https://img.shields.io/badge/OpenAI%20Whisper-compatible-10A37F?logo=openai&logoColor=white)](docs/HTTP_API.md)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20%2B%20CPU%20fallback-76B900?logo=nvidia&logoColor=white)](#-configuration-that-matters-first)
[![Upstream](https://img.shields.io/badge/upstream-HaujetZhao%2FCapsWriter--Offline-181717?logo=github)](https://github.com/HaujetZhao/CapsWriter-Offline)

This repository is a focused fork of [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline). It keeps the upstream recognition stack, but redesigns the **server deployment path** for Linux hosts, containers, GPU-backed machines, and predictable long-running operation.

> Use the upstream project for the original Windows desktop workflow. Use this fork when you want to run CapsWriter as a Linux-friendly server.

---

## 📑 Table of Contents

- [🚀 Quick start](#-quick-start)
- [🤖 OpenAI-Compatible ASR API](#-openai-compatible-asr-api)
- [🎯 Supported deployment scope](#-supported-deployment-scope)
- [⚙️ Configuration that matters first](#%EF%B8%8F-configuration-that-matters-first)
- [🧠 Operational model](#-operational-model)
- [📦 Example files](#-example-files)
- [▶️ Common start modes](#%EF%B8%8F-common-start-modes)
- [💾 Persistence](#-persistence)
- [✅ What success looks like](#-what-success-looks-like)
- [📚 Docs and repository map](#-docs-and-repository-map)
- [🔗 Relationship to upstream](#-relationship-to-upstream)
- [🤝 Contributing](#-contributing)
- [🙏 Acknowledgements](#-acknowledgements)

---

## Why this fork exists

CapsWriter-Offline already solves offline speech input well on Windows. What it did not provide was a clean path for people who want to run the recognition server on Linux, package it in Docker, and operate it as a stable service.

This fork exists for that deployment target. The goal is straightforward: make the server easier to bootstrap, run, restart, inspect, and recover without changing the core recognition story of the upstream project.

## What you get here

| Capability | Detail |
| --- | --- |
| 🐳 **Docker-first** | Linux-oriented Docker image and Compose entry point |
| 📥 **Auto bootstrap** | Models download automatically at container startup |
| 🎮 **GPU-aware** | GPU-first runtime selection with graceful CPU fallback |
| 🖥️ **Headless-safe** | Server defaults tuned for container deployment (no tray, no UI) |
| 🤖 **OpenAI-compatible** | Optional `POST /v1/audio/transcriptions` endpoint — drop-in for any OpenAI SDK |
| 🧪 **Easy onboarding** | Root-level example files (`.env.example`, `docker-compose.example.yml`) |

---

## 🚀 Quick start

### Prerequisites

- Linux
- Docker Engine
- Docker Compose plugin
- NVIDIA driver and NVIDIA Container Toolkit if you want GPU acceleration

### 1. Prepare local files

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
```

### 2. Start the server

```bash
docker compose up -d capswriter-server
```

### 3. Verify health

```bash
docker compose ps
docker compose logs -f capswriter-server
```

The default WebSocket endpoint is:

```text
ws://127.0.0.1:6016
```

### 4. Stop it

```bash
docker compose down
```

---

## 🤖 OpenAI-Compatible ASR API

Any OpenAI Whisper client (Python / Node / curl / your favourite app) can talk to this server with **zero code changes** — just point `base_url` at the local service. The endpoint is opt-in and runs **alongside** the WebSocket server, sharing the same recognition subprocess.

**Three steps to enable:**

```bash
# 1. Turn it on in .env (or in docker compose environment)
echo "CAPSWRITER_HTTP_API_ENABLE=true" >> .env

# 2. Expose port 6017 (uncomment the line in docker-compose.yml under `ports:`)
#    Or restart with both ports bound:
docker compose up -d --force-recreate capswriter-server

# 3. Point the OpenAI SDK at it
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:6017/v1', api_key='dummy')
with open('sample.mp3', 'rb') as f:
    print(client.audio.transcriptions.create(model='whisper-1', file=f).text)
"
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/audio/transcriptions` | Whisper-compatible transcription (`json` / `text` / `srt` / `vtt` / `verbose_json`) |
| `GET  /v1/models` | OpenAI SDK introspection |
| `GET  /health` | Liveness probe |

> 📖 **Full reference, security guidance, OpenAI SDK examples, design notes & troubleshooting →  [`docs/HTTP_API.md`](docs/HTTP_API.md)**

---

## 🎯 Supported deployment scope

This repository is intentionally narrow in scope.

- **Primary target:** Linux + Docker + server deployment
- **Validated priority path:** NVIDIA Pascal / Tesla P4 class hardware
- **Default model path:** `qwen_asr`
- **Alternate supported path:** `fun_asr_nano` via environment configuration

This repo is **not** a Linux desktop port of the full project. It is a server-focused deployment fork.

---

## ⚙️ Configuration that matters first

These are the environment variables most users need first:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CAPSWRITER_SERVER_IMAGE` | `ghcr.io/df-wu/capswriter-offline-server:latest` | Docker image to run |
| `CAPSWRITER_MODEL_TYPE` | `qwen_asr` | Selects the server model |
| `CAPSWRITER_QWEN_PRESET` | `default` | Qwen runtime preset |
| `CAPSWRITER_INFERENCE_HARDWARE` | `auto` | `auto`, `gpu`, or `cpu` |
| `CAPSWRITER_GPU_DEVICE_COUNT` | `all` | GPU request at the Compose layer |
| `CAPSWRITER_SERVER_PORT` | `6016` | WebSocket port |
| `CAPSWRITER_LOG_LEVEL` | `INFO` | Server log verbosity |
| `CAPSWRITER_NUM_THREADS` | `4` | CPU thread hint for CPU-bound stages |
| `CAPSWRITER_HTTP_API_ENABLE` | `false` | Opt-in OpenAI-compatible REST endpoint (see [docs/HTTP_API.md](docs/HTTP_API.md)) |
| `CAPSWRITER_HTTP_API_PORT` | `6017` | HTTP API port, if enabled |

See [`.env.example`](.env.example) for the full deployment-oriented template.

---

## 🧠 Operational model

At startup, the container follows a fixed boot path:

1. [`docker-compose.yml`](docker-compose.yml) defines the service, port, environment, and volumes.
2. [`docker/server/entrypoint.sh`](docker/server/entrypoint.sh) selects the hardware path.
3. [`docker/server/download_models.py`](docker/server/download_models.py) downloads missing model assets and Linux `llama.cpp` libraries.
4. [`docker/server/probe_backend.py`](docker/server/probe_backend.py) verifies the selected GPU backend when applicable.
5. [`start_server.py`](start_server.py) and [`core_server.py`](core_server.py) bring up the WebSocket service (and HTTP API if enabled).
6. [`util/server/service.py`](util/server/service.py) runs recognition in a separate subprocess so model inference does not block the main server loop.

The practical outcome is simple: prefer GPU when available, fall back to CPU when necessary, and keep the service start path predictable.

---

## 📦 Example files

This fork includes root-level examples so onboarding does not depend on nested Docker folders:

- [`.env.example`](.env.example)
- [`docker-compose.example.yml`](docker-compose.example.yml)
- [`hot-server.example.txt`](hot-server.example.txt)

If you want a local variant without touching the default compose file:

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
cp docker-compose.example.yml docker-compose.local.yml
docker compose -f docker-compose.local.yml up -d capswriter-server
```

---

## ▶️ Common start modes

### Default `qwen_asr`

```bash
docker compose up -d capswriter-server
```

### Switch to `fun_asr_nano`

Use the bundled compose override (no `.env` editing needed):

```bash
docker compose -f docker-compose.yml -f docker-compose.fun-asr.yml up -d
```

Or, set it inline for a one-off:

```bash
CAPSWRITER_MODEL_TYPE=fun_asr_nano \
docker compose up -d --force-recreate capswriter-server
```

**When to pick which model:**

| Model | Best for | Trade-off |
| --- | --- | --- |
| `qwen_asr` (default) | Long-form transcription, highest accuracy | Slower per request |
| `fun_asr_nano` | HTTP API / real-time / airi / interactive dictation | Lower accuracy on long/complex sentences |

### Force CPU-only startup

```bash
CAPSWRITER_GPU_DEVICE_COUNT=0 \
CAPSWRITER_INFERENCE_HARDWARE=cpu \
docker compose up -d --force-recreate capswriter-server
```

### Enable OpenAI-compatible API

```bash
CAPSWRITER_HTTP_API_ENABLE=true \
docker compose up -d --force-recreate capswriter-server
# Remember to also uncomment the second port mapping in docker-compose.yml.
```

---

## 💾 Persistence

The default Compose setup mounts:

- `./models:/app/models`
- `./hot-server.txt:/app/hot-server.txt`
- `capswriter-server-logs:/app/logs`

In practice:

- `models/` stores model assets, download cache, and prepared runtime libraries
- `hot-server.txt` provides the server-side hotword file, mainly relevant for `fun_asr_nano`
- `capswriter-server-logs` keeps logs persistent without requiring a host bind mount

---

## ✅ What success looks like

The deployment is in a good state when all three are true:

1. `docker compose ps` shows the service as `healthy`
2. `docker compose logs -f capswriter-server` shows model loading and server startup messages
3. Your client or test tool can connect to `ws://127.0.0.1:${CAPSWRITER_SERVER_PORT}` (and, if enabled, `http://127.0.0.1:${CAPSWRITER_HTTP_API_PORT}/health` returns `{"status":"ok"}`)

---

## 📚 Docs and repository map

Start with these files if you want to understand or extend the server path:

- [`readme.md`](readme.md), project front page
- [`docs/docker-server.md`](docs/docker-server.md), deeper deployment notes
- [`docs/HTTP_API.md`](docs/HTTP_API.md), OpenAI-compatible HTTP API reference
- [`docker-compose.yml`](docker-compose.yml), default deployment entry point
- [`config_server.py`](config_server.py), runtime configuration surface
- [`core_server.py`](core_server.py), server bootstrap
- [`docker/server/Dockerfile`](docker/server/Dockerfile), image definition
- [`util/server/service.py`](util/server/service.py), recognition subprocess management

---

## 🔗 Relationship to upstream

- Upstream project: [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
- Upstream focus: offline speech input on Windows
- This fork: Linux- and Docker-oriented server deployment

This fork extends the upstream deployment story. It does not replace the upstream project.

---

## 🤝 Contributing

Issues and pull requests that improve the Linux server path are welcome. If you are changing runtime behavior, Docker packaging, or deployment defaults, keep the server-first scope intact and prefer changes that preserve predictable startup and fallback behavior.

---

## 🙏 Acknowledgements

- [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [FastAPI](https://fastapi.tiangolo.com/) & [uvicorn](https://www.uvicorn.org/) (HTTP API layer)
