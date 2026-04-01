# CapsWriter-Offline Linux Server Fork

Production-ready Linux and Docker packaging for the CapsWriter server.

This repository is a focused fork of [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline). It keeps the upstream recognition stack, but redesigns the **server deployment path** for Linux hosts, containers, GPU-backed machines, and predictable long-running operation.

**Start here:** [Quick start](#quick-start) · [Deployment notes](docs/docker-server.md) · [Upstream project](https://github.com/HaujetZhao/CapsWriter-Offline)

> Use the upstream project for the original Windows desktop workflow. Use this fork when you want to run CapsWriter as a Linux-friendly server.

## Why this fork exists

CapsWriter-Offline already solves offline speech input well on Windows. What it did not provide was a clean path for people who want to run the recognition server on Linux, package it in Docker, and operate it as a stable service.

This fork exists for that deployment target. The goal is straightforward: make the server easier to bootstrap, run, restart, inspect, and recover without changing the core recognition story of the upstream project.

## What you get here

- A Linux-oriented Docker image and Compose entry point
- Automatic model bootstrap at container startup
- GPU-first runtime selection with CPU fallback
- Headless-safe server defaults for container deployment
- Root-level example files for faster onboarding

## Quick start

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

## Supported deployment scope

This repository is intentionally narrow in scope.

- **Primary target:** Linux + Docker + server deployment
- **Validated priority path:** NVIDIA Pascal / Tesla P4 class hardware
- **Default model path:** `qwen_asr`
- **Alternate supported path:** `fun_asr_nano` via environment configuration

This repo is **not** a Linux desktop port of the full project. It is a server-focused deployment fork.

## Configuration that matters first

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

See [`.env.example`](.env.example) for the full deployment-oriented template.

## Operational model

At startup, the container follows a fixed boot path:

1. [`docker-compose.yml`](docker-compose.yml) defines the service, port, environment, and volumes.
2. [`docker/server/entrypoint.sh`](docker/server/entrypoint.sh) selects the hardware path.
3. [`docker/server/download_models.py`](docker/server/download_models.py) downloads missing model assets and Linux `llama.cpp` libraries.
4. [`docker/server/probe_backend.py`](docker/server/probe_backend.py) verifies the selected GPU backend when applicable.
5. [`start_server.py`](start_server.py) and [`core_server.py`](core_server.py) bring up the WebSocket service.
6. [`util/server/service.py`](util/server/service.py) runs recognition in a separate subprocess so model inference does not block the main server loop.

The practical outcome is simple: prefer GPU when available, fall back to CPU when necessary, and keep the service start path predictable.

## Example files

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

## Common start modes

### Default `qwen_asr`

```bash
docker compose up -d capswriter-server
```

### Switch to `fun_asr_nano`

```bash
CAPSWRITER_MODEL_TYPE=fun_asr_nano \
docker compose up -d --force-recreate capswriter-server
```

### Force CPU-only startup

```bash
CAPSWRITER_GPU_DEVICE_COUNT=0 \
CAPSWRITER_INFERENCE_HARDWARE=cpu \
docker compose up -d --force-recreate capswriter-server
```

## Persistence

The default Compose setup mounts:

- `./models:/app/models`
- `./hot-server.txt:/app/hot-server.txt`
- `capswriter-server-logs:/app/logs`

In practice:

- `models/` stores model assets, download cache, and prepared runtime libraries
- `hot-server.txt` provides the server-side hotword file, mainly relevant for `fun_asr_nano`
- `capswriter-server-logs` keeps logs persistent without requiring a host bind mount

## What success looks like

The deployment is in a good state when all three are true:

1. `docker compose ps` shows the service as `healthy`
2. `docker compose logs -f capswriter-server` shows model loading and server startup messages
3. Your client or test tool can connect to `ws://127.0.0.1:${CAPSWRITER_SERVER_PORT}`

## Docs and repository map

Start with these files if you want to understand or extend the server path:

- [`readme.md`](readme.md), project front page
- [`docs/docker-server.md`](docs/docker-server.md), deeper deployment notes
- [`docker-compose.yml`](docker-compose.yml), default deployment entry point
- [`config_server.py`](config_server.py), runtime configuration surface
- [`core_server.py`](core_server.py), server bootstrap
- [`docker/server/Dockerfile`](docker/server/Dockerfile), image definition
- [`util/server/service.py`](util/server/service.py), recognition subprocess management

## Relationship to upstream

- Upstream project: [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
- Upstream focus: offline speech input on Windows
- This fork: Linux- and Docker-oriented server deployment

This fork extends the upstream deployment story. It does not replace the upstream project.

## Contributing

Issues and pull requests that improve the Linux server path are welcome. If you are changing runtime behavior, Docker packaging, or deployment defaults, keep the server-first scope intact and prefer changes that preserve predictable startup and fallback behavior.

## Acknowledgements

- [HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
- [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx)
- [FunASR](https://github.com/alibaba-damo-academy/FunASR)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
