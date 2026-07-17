# Desktop portability and native hotkeys

> [Documentation home](README.md) · [繁體中文](../zh-TW/desktop-portability.md) · English

CapsWriter v2 keeps the upstream Windows desktop behavior while adding a
bounded Linux desktop path. These are separate support claims: the Linux
server/container and the portable CLI work without a graphical session, but
native desktop hotkeys require a supported display server.

## Supported paths

| Path | Support level | Important boundary |
|---|---|---|
| Windows desktop executable | Supported; upstream tray and shortcut behavior preserved | Windows CI builds, relocates, extracts, and import-smokes the portable artifact; hardware behavior still needs a real release test |
| Windows or Linux CLI | Supported on Python 3.10 and 3.12 | No global hotkey, tray, microphone capture, or automatic text injection |
| Linux X11 desktop hotkeys | Supported with limitations | Listening works; selective suppression does not |
| Linux Wayland desktop hotkeys | Not supported | XWayland cannot provide reliable system-wide capture |
| Linux headless server | Supported through the container/source server paths | No desktop hotkeys or tray |

The portability CI matrix runs dependency-light compilation, packaging source
guards, desktop backend policy tests, and the complete standard-library CLI
verifier on `ubuntu-24.04` and `windows-2022`, using Python 3.10 and 3.12. A
separate `windows-package` job on `windows-2022`/Python 3.12 installs the fully
hashed Windows lock, builds both executables, relocates and ZIP-round-trips the
distribution outside the checkout, rejects reparse points, executes bounded
import self-checks through both EXEs, and uploads the exact tested ZIP. Neither
job claims to test audio hardware, a real display server, a Windows tray,
model loading, known-audio transcription, or a hardware accelerator.

## Windows package and HTTP API

[`build.spec`](../../build.spec) still emits `start_server.exe` and
`start_client.exe`. The server executable is analyzed from
[`start_server_universal.py`](../../start_server_universal.py), not the Docker
entrypoint. The universal entrypoint:

- retains `config_server.py` values such as `enable_tray`, the selected model,
  the WebSocket address, and accelerator settings;
- keeps the OpenAI-compatible HTTP API disabled by default;
- validates and applies `CAPSWRITER_HTTP_API_*` plus the explicit optional
  `CAPSWRITER_SERVER_ADDR` WebSocket bind override;
- includes the fork server and dynamic FastAPI/Starlette/Uvicorn/Pydantic
  modules in the server analysis without adding them to the desktop client
  analysis; and
- preserves the upstream `freeze_support()` and executable names.

Build from a disposable Windows Python 3.12 x86-64 environment. Production
builds install only the committed hash lock; the `srt` source distribution is
the sole exception to the wheel-only policy:

```powershell
py -3.12 -m venv "$env:TEMP\capswriter-build"
& "$env:TEMP\capswriter-build\Scripts\python.exe" -m pip install `
  --require-hashes `
  --only-binary=:all: `
  --no-deps `
  --requirement requirements-windows-build-bootstrap.lock
& "$env:TEMP\capswriter-build\Scripts\python.exe" -m pip install `
  --require-hashes `
  --only-binary=:all: `
  --no-binary=srt `
  --no-build-isolation `
  --requirement requirements-windows-build.lock
& "$env:TEMP\capswriter-build\Scripts\python.exe" -m PyInstaller --clean --noconfirm build.spec
```

The small bootstrap lock pins `pip` and `setuptools` before the `srt` source
distribution is built with isolation disabled; the runner's preinstalled
toolchain is not part of the release contract.

The resulting `dist/CapsWriter-Offline/` contains real copies of `core/`, `LLM/`,
`assets/`, and `docs/`, plus real empty `models/` and `logs/` directories. It
contains no source-tree junctions. Local models, logs, caches, secrets,
archives, and non-Windows shared libraries are filtered from the payload.
Missing required files or required collection dependencies fail the build.

After archiving, extract under a different directory outside the checkout and
run:

```powershell
.\start_server.exe --artifact-self-check
.\start_client.exe --artifact-self-check
```

Both commands must exit zero with a `CAPSWRITER_ARTIFACT_SELF_CHECK=` report
whose status is `ok`. They validate layout and server/client imports without
binding sockets, opening devices, creating hooks/trays, launching FFmpeg, or
loading models. See the bilingual
[production build guide](../../assets/BUILD_GUIDE.md) for the exact artifact
contract, CI sequence, and lock-regeneration command.

## Prepare a downloaded Windows package

The release ZIP is the exact program artifact tested by CI. It deliberately
keeps `models/` empty and does not include GGUF runtime DLLs or FFmpeg. Do not
interpret the EXE import self-check as model-load evidence.

For the default `model_type = 'qwen_asr'` local Server, provision these two
separate upstream assets before starting `start_server.exe`:

| Purpose | Exact asset | SHA-256 | Destination |
|---|---|---|---|
| Default Qwen model | [`Qwen3-ASR-1.7B-q5_k.zip`](https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q5_k.zip) | `f40040fe62a5ef0c09f8699fdbcb30f18bb8ae2bcd515ed4954e1f62b8b0e88f` | Extract under `models/Qwen3-ASR/`; the result must be `models/Qwen3-ASR/Qwen3-ASR-1.7B/` |
| Windows x86-64 GGUF runtime | [`llama-b7798-bin-win-vulkan-x64.zip`](https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-win-vulkan-x64.zip) | `d478b7070dd12a5c64478a398352e1f880d488c4c346a8f00e7051935ef6f8e8` | Copy its DLL files into the shared `core/server/engines/llama/bin/` directory |

Run the following from the extracted `CapsWriter-Offline` directory in
PowerShell. The commands stop before extraction if either download hash differs:

```powershell
$modelZip = Join-Path $env:TEMP 'Qwen3-ASR-1.7B-q5_k.zip'
$runtimeZip = Join-Path $env:TEMP 'llama-b7798-bin-win-vulkan-x64.zip'
$runtimeStage = Join-Path $env:TEMP 'capswriter-llama-b7798'

Invoke-WebRequest `
  'https://github.com/HaujetZhao/CapsWriter-Offline/releases/download/models/Qwen3-ASR-1.7B-q5_k.zip' `
  -OutFile $modelZip
if ((Get-FileHash $modelZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne `
    'f40040fe62a5ef0c09f8699fdbcb30f18bb8ae2bcd515ed4954e1f62b8b0e88f') {
  throw 'Qwen model SHA-256 mismatch'
}
New-Item -ItemType Directory -Force '.\models\Qwen3-ASR' | Out-Null
Expand-Archive $modelZip -DestinationPath '.\models\Qwen3-ASR' -Force

Invoke-WebRequest `
  'https://github.com/ggml-org/llama.cpp/releases/download/b7798/llama-b7798-bin-win-vulkan-x64.zip' `
  -OutFile $runtimeZip
if ((Get-FileHash $runtimeZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne `
    'd478b7070dd12a5c64478a398352e1f880d488c4c346a8f00e7051935ef6f8e8') {
  throw 'llama.cpp runtime SHA-256 mismatch'
}
Remove-Item $runtimeStage -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $runtimeZip -DestinationPath $runtimeStage

$runtimeTarget = '.\core\server\engines\llama\bin'
New-Item -ItemType Directory -Force $runtimeTarget | Out-Null
Copy-Item -Path (Join-Path $runtimeStage '*.dll') -Destination $runtimeTarget -Force
```

Before startup, verify that at least these files exist:

```text
models/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder_frontend.onnx
models/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_encoder_backend.onnx
models/Qwen3-ASR/Qwen3-ASR-1.7B/qwen3_asr_llm.gguf
core/server/engines/llama/bin/ggml.dll
core/server/engines/llama/bin/ggml-base.dll
core/server/engines/llama/bin/llama.dll
```

Other model choices have different archives, hashes, paths, and hardware
profiles; do not mix them with the default instructions. The
[upstream model release](https://github.com/HaujetZhao/CapsWriter-Offline/releases/tag/models)
describes those choices. They remain unqualified by this release candidate
until tested on the target Windows host.

Desktop file/media transcription also requires `ffmpeg.exe` on `PATH` or in the
package root; `ffprobe.exe` enables duration/progress reporting. Install both
from a trusted [FFmpeg distribution](https://ffmpeg.org/download.html). The
microphone/WebSocket path does not require FFmpeg. A Client that connects to a
remote Server does not need a local model or GGUF runtime.

The package starts with the same desktop defaults when no HTTP environment
variables are present:

```powershell
& .\dist\CapsWriter-Offline\start_server.exe
```

To opt in to the loopback HTTP endpoint without changing the tray setting:

```powershell
$env:CAPSWRITER_HTTP_API_ENABLE = "true"
$env:CAPSWRITER_SERVER_ADDR = "127.0.0.1"
$env:CAPSWRITER_HTTP_API_BIND = "127.0.0.1"
& .\dist\CapsWriter-Offline\start_server.exe
```

Without `CAPSWRITER_SERVER_ADDR`, the WebSocket listener deliberately keeps the
upstream `0.0.0.0` default. Set it whenever the desktop server should be local
to one machine.

Use [`start_server_docker.py`](../../start_server_docker.py) for container and
environment-driven server deployments. That entrypoint intentionally applies
headless defaults and is not the Windows desktop packaging entrypoint.

## Desktop WebSocket safety limits

The source client and packaged EXE keep microphone ingress bounded to 128 audio
chunks plus eight control-message slots. If the callback outruns network
sending, CapsWriter cancels that recording and closes the WebSocket instead of
retaining unbounded private audio or sending the final marker ahead of data.
Client messages are limited to 16 MiB with four queued receive messages, and
all audio sends remain ordered.

Two optional process environment variables control the failure deadlines:

- `CAPSWRITER_CLIENT_WEBSOCKET_SEND_TIMEOUT` defaults to 30 seconds for each
  WebSocket send; and
- `CAPSWRITER_CLIENT_FILE_RESULT_TIMEOUT` defaults to 600 seconds for the final
  result after a file upload finishes.

Both values must be finite numbers greater than zero. A deadline closes the
connection as protocol-level cancellation. Increase the file-result deadline
only for a deliberately slow model; do not disable it.

## Linux X11 hotkeys

The desktop shortcut manager uses `pynput` callbacks on X11. It requires all
of the following:

- a logged-in X11 session with `DISPLAY` exported to the CapsWriter process;
- the Python client dependencies, including `pynput` and its X11 dependency;
- access to the same user's X server; running the client as root is neither
  required nor recommended; and
- an X server with the RECORD and XTest extensions used by `pynput`.

The `pynput` backend is imported lazily only when simulated input is actually
used. Pure Wayland and headless sessions can therefore reach the documented
detect-and-decline path without crashing during the client import phase.

Keyboard press/release events and the commonly mapped side buttons 8/9
(`x1`/`x2`) are supported. Side-button numbering is hardware/driver dependent,
so it must be verified on the target workstation.

When paste mode is disabled, both plain recognition output and LLM streaming
output use the same `pynput` X11/XTest controller on Linux. They do not use the
`keyboard` package's root-only `/dev/input` backend. Windows retains its
existing `keyboard.write` behavior. Text injection remains application-,
layout-, and IME-dependent, so validate it with the target editor and language.

X11 has an important safety limit: `pynput` can suppress the entire keyboard
or pointer by grabbing the device, but it cannot safely suppress only the
configured CapsWriter key. CapsWriter therefore never enables the X11-wide
grab. Any X11 shortcut configured with `suppress=True` is handled internally
as `suppress=False`, and the application logs a warning. Other applications
still receive the key. Prefer a low-impact key such as `f12` and set
`suppress=False`; mouse back/forward buttons may still navigate the focused
application while recording starts.

## Wayland and headless limitations

Wayland intentionally prevents ordinary clients from observing arbitrary
system-wide input. There is no stable compositor-independent `pynput` API that
matches the Windows global-hook behavior. An XWayland `DISPLAY` may expose only
events from X11 applications and must not be treated as global coverage.
CapsWriter detects a Wayland session and declines to start its native global
hotkey listeners rather than silently providing partial capture.

The same compositor policy can restrict simulated paste/typing and clipboard
automation. Native hotkey detection alone would not prove that the full
desktop client output path works on a given Wayland compositor.

For Wayland or a headless host, use the
[portable CLI](cli-client.md), the browser console, or the HTTP API instead
of native global shortcuts. If system-wide desktop shortcuts are required,
log in to an X11 session and validate the selected key on that workstation.

## Release evidence

The Windows package job supplies clean hash-install, PyInstaller,
archive-relocation, no-reparse-point, and executable import-smoke evidence.
Before calling its ZIP releasable, supplement that automation on the target
Windows hardware:

1. verify the uploaded ZIP digest and retain its workflow run/dependency lock;
2. start `start_server.exe` without HTTP variables and confirm the tray and
   WebSocket workflow retain their upstream behavior;
3. enable the loopback HTTP API and verify readiness and one transcription;
4. start `start_client.exe` and validate the configured keyboard/mouse hooks;
5. validate microphone and file workflows, FFmpeg, the selected model/runtime
   assets, known audio, and every advertised CPU/DirectML/GPU profile; and
6. confirm shutdown removes both tray icons and child processes.

For Linux X11, record the distribution, desktop environment, X server,
keyboard shortcut, mouse mapping if used, and whether text injection was also
tested. A passing headless CI job is not evidence for any of those desktop
properties.
