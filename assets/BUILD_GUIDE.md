# Windows production build guide / Windows 正式打包指南

This guide describes the supported Windows x86-64 release build. For desktop
runtime behavior and support boundaries, see the paired
[English desktop guide](../docs/en/desktop-portability.md) and
[繁體中文桌面指南](../docs/zh-TW/desktop-portability.md).

本指南說明受支援的 Windows x86-64 正式打包流程。桌面執行行為與支援邊界請搭配
[English desktop guide](../docs/en/desktop-portability.md) 與
[繁體中文桌面指南](../docs/zh-TW/desktop-portability.md)閱讀。

## Release contract / 發布契約

- Build on 64-bit Windows with CPython 3.12.
- Bootstrap `pip` and the `setuptools` build backend from
  [`requirements-windows-build-bootstrap.lock`](../requirements-windows-build-bootstrap.lock)
  with exact wheel hashes and no dependencies.
- Install the complete client/server build graph from
  [`requirements-windows-build.lock`](../requirements-windows-build.lock) with
  hash checking. Every resolved package is pinned and hashed.
- Build both `start_server.exe` and `start_client.exe` with
  [`build.spec`](../build.spec).
- Distribute a real directory tree. The artifact must not contain symbolic
  links, junctions, or other reparse points into the checkout.
- `core/`, `LLM/`, `assets/`, and `docs/` are immutable copies. `models/` and
  `logs/` are real, initially empty directories.
- A missing required file, Sherpa/Pillow dependency, or dynamic HTTP API package
  fails the build. The spec does not silently publish a partial artifact.

換言之：正式 artifact 必須由 Python 3.12 x64 Windows 建立、只從完整 hash lock
安裝、同時產生兩個 executable，且搬離 source checkout 後仍能自我檢查。打包不再
建立 junction；缺少必要 dependency 或 payload 會直接失敗。

## What is and is not copied / 會複製與不會複製的內容

Required root files:

```text
config_client.py
config_server.py
hot.txt
hot-server.txt
hot-rule.txt
readme.md
README.en.md
LICENSE
```

Required product trees are copied with caches, logs, local secret material,
archives, model blobs, links, and non-Windows shared libraries filtered out.
In particular, the repository's local `models/` contents are never copied.
Obsolete `core_server.py` and `core_client.py` files are not part of the
artifact contract.

Windows llama.cpp runtime DLLs that a selected GGUF profile requires are model
runtime assets, not Python dependencies. A clean CI checkout does not download
models or exercise those DLLs. Add model/runtime assets from a trusted release
after extraction, record their source and checksum, and validate the chosen
profile on the release machine.

必要 product tree 會排除 cache、log、本機 secret、壓縮檔、model blob、link 與
非 Windows shared library；尤其不會把開發機上龐大的 `models/` 帶進 artifact。
GGUF profile 所需的 Windows llama.cpp DLL 屬 model runtime asset，必須另外以可信
來源與 checksum 管理，不能把 Python import smoke 誤當成 model-load 證據。

## Clean build / 乾淨打包

Run these commands from the repository root in PowerShell. Use a disposable
environment so globally installed packages cannot fill gaps in the lock.

```powershell
$venv = Join-Path $env:TEMP 'capswriter-windows-build'
Remove-Item -LiteralPath $venv -Recurse -Force -ErrorAction SilentlyContinue
py -3.12 -m venv $venv

& "$venv\Scripts\python.exe" -m pip install `
  --require-hashes `
  --only-binary=:all: `
  --no-deps `
  --requirement requirements-windows-build-bootstrap.lock

& "$venv\Scripts\python.exe" -m pip install `
  --require-hashes `
  --only-binary=:all: `
  --no-binary=srt `
  --no-build-isolation `
  --requirement requirements-windows-build.lock

& "$venv\Scripts\python.exe" -m PyInstaller --clean --noconfirm build.spec
```

The bootstrap lock makes the build frontend and backend explicit before the
main install disables build isolation. `srt` is the one explicit
source-distribution exception because its release has no wheel. All other
packages must be binary distributions. Do not replace either locked install
with unconstrained `pip install -r requirements-client.txt` or
`requirements-server.txt` in a production build.

Bootstrap lock 會先明確固定 build frontend 與 backend，再讓主安裝停用 build
isolation。正式 build 中，`srt` 是唯一明確允許的 source distribution；其他
dependency 必須是 binary distribution。不得用未鎖定的兩份 `.txt` requirements
取代任一 hash lock。

## Expected layout / 預期目錄

```text
dist/CapsWriter-Offline/
├── start_server.exe
├── start_client.exe
├── internal/
├── core/
├── LLM/
├── assets/
├── docs/
├── models/                     # real and empty at release-build time
├── logs/                       # real and empty at release-build time
├── config_client.py
├── config_server.py
├── hot.txt
├── hot-server.txt
├── hot-rule.txt
├── readme.md
├── README.en.md
└── LICENSE
```

## Relocation smoke / 搬移後 smoke

Archive the directory, extract it under a different directory outside the
checkout, and run both executable self-checks from the extracted root:

```powershell
Set-Location "$env:TEMP\capswriter-package-extracted\CapsWriter-Offline"
.\start_server.exe --artifact-self-check
.\start_client.exe --artifact-self-check
```

Each command must exit `0` and print a line beginning with
`CAPSWRITER_ARTIFACT_SELF_CHECK=` whose JSON report has `"status":"ok"` and
`"frozen":true`. The self-check verifies required files/directories, rejects
links/junctions, and imports the server or client runtime surface. It does not:

- bind a WebSocket or HTTP socket;
- create tray icons or global keyboard/mouse hooks;
- open a microphone or launch FFmpeg;
- load a model or run known-audio transcription; or
- validate DirectML, CUDA, Vulkan, or llama.cpp execution.

兩個 self-check 都不會啟動正常 server/client lifecycle，因此可在 CI 安全執行；
它們只證明 archive 搬移後的 layout、embedded Python runtime 與 import surface。

## CI evidence / CI 證據

The `windows-package` job in
[`.github/workflows/portability.yml`](../.github/workflows/portability.yml) runs
on pinned `windows-2022` with Python 3.12. It performs the hash-only install,
PyInstaller build, relocation, ZIP extraction, reparse-point/empty-directory
inspection, bounded EXE self-checks, and upload of the exact tested ZIP.

This is automated production-package evidence, not hardware acceptance. Before
publishing a desktop release, retain manual results for:

1. server and client tray creation and clean removal;
2. configured keyboard and mouse hooks;
3. microphone capture and file transcription, including FFmpeg integration;
4. selected model/runtime asset load and known-audio transcription;
5. CPU plus every advertised DirectML/GPU backend; and
6. clean shutdown of worker and child processes.

`windows-package` 是正式 package gate，但不是硬體驗收。Tray、hook、麥克風、
FFmpeg、model、known audio、DirectML/GPU 與 child-process shutdown 仍須在目標
Windows release machine 留存人工證據。

## Lock maintenance / Lock 維護

Regenerate the lock only as an intentional dependency update:

```bash
UV_CACHE_DIR=/tmp/capswriter-windows-lock-cache uv pip compile \
  requirements-client.txt requirements-server.txt \
  --python-version 3.12 \
  --python-platform x86_64-pc-windows-msvc \
  --generate-hashes \
  --only-binary=:all: \
  --no-binary=srt \
  --no-emit-index-url \
  --output-file requirements-windows-build.lock
```

Review the dependency/version diff, run the lock/source-contract tests, and let
the Windows package job prove that pip can install the lock and PyInstaller can
build it. Remove the temporary cache afterward.

## Troubleshooting / 疑難排解

- Bootstrap or dependency hash mismatch/missing binary: do not disable hash/binary enforcement;
  regenerate and review the lock only if the dependency update is intentional.
- Missing required payload: restore the required tracked file/tree; the spec is
  designed to fail instead of skipping it.
- Self-check import failure: inspect the EXE's captured stderr and PyInstaller
  warnings, then correct collection/hidden-import rules.
- Reparse point found: discard the artifact. A portable ZIP must contain real
  files and directories only.
- Model load failure after self-check succeeds: validate the separately supplied
  models, llama.cpp DLL set, profile configuration, and hardware backend. The
  self-check intentionally does not cover this layer.

Last reviewed / 最後檢視：2026-07-16. Target: CPython 3.12 x86-64 Windows,
with dependency and PyInstaller versions defined by the committed lock.
