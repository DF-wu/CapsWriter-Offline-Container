# CapsWriter-Offline — Windows + Linux Fork

> Offline speech recognition for Windows desktops, Linux desktops and servers,
> with Docker deployment, an optional OpenAI-compatible API, and Web, CLI, and
> TUI clients.
>
> English · [繁體中文](readme.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux-334155)](docs/en/desktop-portability.md)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![OpenAI compatible](https://img.shields.io/badge/OpenAI%20audio-compatible-10A37F)](docs/en/openai-api.md)

This fork builds on the upstream CapsWriter v2 model/inference algorithms and
Windows desktop experience while carrying a documented compatibility/security
patch set, including narrow engine I/O and privacy-logging contacts. It extends
the product into a tested cross-platform delivery surface with a bounded Linux
X11 desktop path, a Linux container server, an opt-in HTTP API, browser and
terminal clients, reproducible dependency locks, and release/security gates.

It is no longer accurate to describe this repository as a Linux-only server
overlay. Windows remains a first-class desktop and packaging target. Linux is
supported through several distinct paths whose limits are documented honestly:
X11 desktop hotkeys have reduced suppression behavior, Wayland global hotkeys
are not supported, and headless Linux uses the server, Web, CLI, or TUI paths.

Model inference is local once the required assets are present. A first-time
container bootstrap may download model and runtime assets; no cloud inference
service is required.

## What is included

| Surface | Intended use | Current evidence and boundary |
|---|---|---|
| Windows desktop | Tray, shortcuts, recording, file transcription, optional HTTP API | Windows 2022/Python 3.12 CI hash-installs, builds, relocates, extracts, inspects, and runs import self-checks through both packaged EXEs; tray/audio/model/hardware behavior remains manual release evidence |
| Linux X11 desktop | Desktop shortcuts and the upstream client on an X11 session | Portable callback contract is tested; listening works, but selective single-key suppression is deliberately disabled |
| Linux `amd64` server/container | Long-running local or shared ASR service | Docker/Compose, health/readiness, model bootstrap, GPU preference, and CPU fallback paths are gated; ARM64 is not release-gated |
| OpenAI-compatible API | SDK, curl, Web, CLI, and TUI transcription | Opt-in `whisper-1` file-transcription subset with explicit unsupported-capability errors |
| Web Console | Browser recording, upload, STT formats, downloads, local browser TTS | React/Vite tests, production build, browser smoke, and static image smoke |
| No-GUI CLI | Scripts, SSH, batch transcription, local OS TTS | Standard-library zipapp; Linux/Windows Python portability matrix |
| Textual TUI | Keyboard-first diagnostics, file transcription, optional microphone | Hash-locked Python 3.10–3.12 runtime and no-skip Pilot suite |

See the full [support and security matrix](docs/en/support-security.md) before
making a production or desktop-support claim.

## Choose your path

| Goal | Start here |
|---|---|
| Use or package the Windows desktop app | [Desktop portability](docs/en/desktop-portability.md#windows-package-and-http-api) |
| Run the desktop client on Linux X11 | [Linux X11 hotkeys and limits](docs/en/desktop-portability.md#linux-x11-hotkeys) |
| Start a Linux server with Docker | [Getting started](docs/en/getting-started.md#path-c-linux-container-server) |
| Deploy server + browser console | [Deployment guide](docs/en/deployment.md) |
| Use an OpenAI SDK or curl | [OpenAI-compatible API](docs/en/openai-api.md) |
| Automate from a shell or SSH session | [No-GUI CLI guide](docs/en/cli-client.md) |
| Use a keyboard-first terminal UI | [TUI guide](docs/en/tui.md) |
| Diagnose a failure | [Troubleshooting](docs/en/troubleshooting.md) |
| Upgrade or review the current release | [Release notes](docs/en/release-notes.md) |

The [English documentation home](docs/en/README.md) links every supported user,
operator, and maintainer path.

## Fastest server start

Prerequisites: a `linux/amd64` host, Docker Engine, the Compose plugin, and enough disk space
for the selected model and image. GPU support is optional.

```bash
git clone https://github.com/DF-wu/CapsWriter-Offline-Container.git
cd CapsWriter-Offline-Container
cp .env.example .env
cp hot-server.example.txt hot-server.txt
docker compose up -d capswriter-server
docker compose ps
docker compose logs -f capswriter-server
```

The base Compose file makes no vendor-device reservation, so this command also
works on CPU-only Docker hosts. To expose NVIDIA GPUs, add the explicit override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d capswriter-server
```

For an Intel/AMD Linux iGPU, obtain the host GIDs with
`stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*`, set
`CAPSWRITER_DRI_RENDER_GID` and `CAPSWRITER_DRI_VIDEO_GID` in `.env`, and add
`docker-compose.igpu.yml`. Models persist in a Docker named volume by default;
add `docker-compose.models-bind.yml` only when you need to manage host
`./models` directly, after assigning the directory to the container user. See
the [deployment guide](docs/en/deployment.md#linux-container-profile) for the
exact commands and locking requirements.

The default Compose deployment publishes the WebSocket server on host loopback
port `6016`. The OpenAI-compatible HTTP API is off by default.

To opt in, set a strong local token in `.env`:

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
```

Then uncomment the HTTP mapping under `ports:` in
[`docker-compose.yml`](docker-compose.yml), recreate the service, and verify
both liveness and readiness:

```bash
docker compose up -d --force-recreate capswriter-server
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

Binding the enabled API beyond loopback requires authentication unless the
explicit insecure test-network escape hatch is enabled. Read
[deployment](docs/en/deployment.md) and
[support/security](docs/en/support-security.md) before exposing it to a LAN.

## Client entry points

### Web Console

Development mode uses the locked Node dependency tree:

```bash
cd client/web
npm ci --no-audit --no-fund
npm run dev
```

For a static container deployment, use:

```bash
docker compose -f docker-compose.web.yml up -d --build capswriter-web
```

See the [English deployment guide](docs/en/deployment.md#web-console-profile)
for CORS, microphone security contexts, runtime configuration, and
production-image verification.

### No-GUI CLI

```bash
python client/cli/capswriter_cli.py ready \
  --base-url http://127.0.0.1:6017 \
  --key-file /path/to/capswriter-http.key
python client/cli/capswriter_cli.py transcribe meeting.wav --format text
python client/cli/scripts/build_zipapp.py
```

See [the CLI guide](docs/en/cli-client.md) for batch output, portable filenames,
timeouts, zipapp packaging, and local TTS.

### Textual TUI

```bash
python3.12 -m venv .venv-tui
.venv-tui/bin/python -m pip install \
  --require-hashes --only-binary=:all: \
  --requirement requirements-tui.lock
.venv-tui/bin/python -m client.tui --base-url http://127.0.0.1:6017
```

The [TUI guide](docs/en/tui.md) includes Windows commands, Traditional Chinese
UI, keyboard controls, the file-only fallback, and strict verification.

## Desktop path

Windows packaging analyzes the client plus
[`start_server_universal.py`](start_server_universal.py), preserving the
upstream desktop configuration when the HTTP API is disabled and adding only
validated `CAPSWRITER_HTTP_API_*` settings when enabled. Linux desktop hotkeys
are supported only on X11 and never use an unsafe whole-keyboard grab.

Follow [desktop portability](docs/en/desktop-portability.md) for exact Windows
build commands, X11 requirements, Wayland/headless limitations, and the release
evidence that still must be collected on real hardware.

## Security defaults

- The HTTP API is disabled by default; Compose publishes service ports on
  `127.0.0.1` by default.
- An enabled non-loopback API requires a Bearer key or key file unless the
  explicit insecure-bind override is set.
- Transcript and prompt logging is off by default.
- Web runtime configuration refuses to publish a default API key unless a
  separate public-key opt-in is enabled; entering the key in the UI is safer.
- Container privileges are reduced with `no-new-privileges` and dropped Linux
  capabilities.
- Docker/TUI Python dependencies and Web dependencies have reproducible locks;
  publish workflows attach provenance and SBOM attestations.

Security behavior, private-data boundaries, supported/unsupported platforms,
and reporting guidance are collected in
[support and security](docs/en/support-security.md).

## Verification and release policy

Portable contracts run on pinned Ubuntu and Windows runners with Python 3.10
and 3.12. A separate Python 3.12 Windows job installs the fully hashed production
lock, builds both PyInstaller executables, ZIP-round-trips the artifact outside
the checkout, rejects reparse points, import-smokes both EXEs, and uploads the
exact tested ZIP. The API contract and hash-locked TUI have isolated no-skip
jobs. The root gate covers server, Docker, CLI, Web, documentation, workflow
source guards, and cleanup. Model/audio/tray/display/hardware evidence remains
an explicit release-candidate responsibility rather than being inferred from
import smoke or unit tests.

![Fork release flow showing active v2 integration gates and isolated v1 maintenance](docs/assets/version-tracks.svg)

Text equivalent: released upstream changes merge into active fork v2 and pass
Linux, Windows, API, TUI, Web, and security gates before a v2 release. Legacy
v1 remains isolated; only critical or security fixes are manually backported
and pass separate legacy gates.

Run the local gate:

```bash
python scripts/verify_all.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_docs.py
python scripts/clean.py --check
```

See [verification](docs/verification.md), the
[v1/v2 policy](docs/en/versioning.md), and the
[current release notes](docs/en/release-notes.md).

## Documentation

| Document | Purpose |
|---|---|
| [Documentation home](docs/en/README.md) | Complete task and audience index |
| [Getting started](docs/en/getting-started.md) | Select and validate the right Windows/Linux path |
| [Deployment](docs/en/deployment.md) | Container, desktop-source, Web, network, upgrade, and rollback operations |
| [Troubleshooting](docs/en/troubleshooting.md) | Diagnostic ladder for desktop, container, API, and clients |
| [Support and security](docs/en/support-security.md) | Support matrix, limits, secrets, privacy, supply chain, reporting |
| [Release notes](docs/en/release-notes.md) | Current fork v2 snapshot, changes, migration, known limits |
| [Desktop portability](docs/en/desktop-portability.md) | Windows package contract; Linux X11, Wayland, and headless truth |
| [OpenAI-compatible API](docs/en/openai-api.md) | HTTP/SDK contract and resource controls |
| [TUI](docs/en/tui.md) | Textual workbench installation and operation |
| [Architecture](docs/architecture.md) | Sidecar integration and upstream-drift strategy |

## Upstream, lifecycle, and license

This fork is based on
[HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)
and continues to consume its model/inference algorithms and desktop product
work. Narrow changes inside upstream-tracked engine files are limited to bounded
I/O and privacy-aware logging; the fork adds delivery, portability, API, client,
and operational surfaces without renaming upstream releases or claiming
ownership of upstream model work.

Fork v2 is the active development line. Legacy fork v1 is isolated for critical
and security maintenance only. See the [version policy](docs/en/versioning.md)
before merging or backporting across generations.

License: [MIT](LICENSE).
