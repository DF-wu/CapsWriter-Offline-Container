# Development workflow / 開發流程

Windows is the first supported desktop platform. The server runs separately on
Linux (for example axolotl); the Web frontend and CLI use its HTTP API. The
Windows hotkey client uses its WebSocket endpoint. Starting the frontend alone
does not start an ASR server.

Windows 為桌面端第一支援平台；Linux Server、Windows 快捷鍵 Client 與 Web
前端是獨立程序。快捷鍵連 WebSocket，Web／CLI 連 HTTP API。請分別設定連線位址。

## Prerequisites / 必要工具

- Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
  Python 3.12 is the default; `setup server` uses Python 3.10 to match the Linux
  server lock. uv can download a missing interpreter. Existing Python 3.10–3.12
  can run the wrapper; no global Python package installation is required.
- For the Web frontend install Node.js 24 LTS, including npm.
- Windows desktop/build profiles require Windows x64. API/server lock profiles
  require Linux x86-64; use WSL or an isolated Linux container from Windows.
- A native Linux server also needs the system libraries in
  [the server Dockerfile](../docker/server/Dockerfile) and separately provisioned
  models. Setup never downloads models or changes a running service.

Run commands from the repository root. PowerShell and Linux use the same arguments.
If Python is not on PATH, replace `python` with `uv run --no-project --python 3.12`.
This explicit `--no-project` keeps uv from resolving unrelated project groups.

請在專案根目錄執行；PowerShell 與 Linux 的參數相同。未安裝 Python 時可用
`uv run --no-project --python 3.12 scripts/dev.py ...`。套件只裝到專案的
`.venv-dev/<profile>`，不需啟用環境、不更動全域 Python。設定不下載模型，也不部署服務。

## Windows client / Windows 快捷鍵端

```powershell
python scripts/dev.py setup desktop
python scripts/dev.py client --settings
python scripts/dev.py client
```

Set axolotl's WebSocket address, microphone and shortcut in the settings window.
This profile installs only the desktop runtime, without ASR engines or PyInstaller.
It uses `requirements-desktop-dev.lock`, constrained to the reviewed Windows
release versions. Windows microphone, shortcut and foreground text insertion
still require a real Windows session for acceptance testing.

設定視窗填入 axolotl 的 WebSocket 位址，選擇麥克風與快捷鍵。此環境只裝桌面執行
依賴；快捷鍵、實際錄音、輸入目前視窗需在 Windows 實機驗收。

## Tests / 測試

```sh
python scripts/dev.py setup dev
python scripts/dev.py test
python scripts/dev.py setup api
python scripts/dev.py test api
python scripts/dev.py setup tui
python scripts/dev.py test tui
python scripts/dev.py check
```

`dev` has no third-party packages. `test` runs script, Docker configuration and
dependency-free CLI tests; dependency-dependent tests may skip there. `test api`
and `test tui` use their existing dedicated verification gates and hash-locked
dependencies to exercise those contracts. TUI setup also installs the reviewed
pip/setuptools bootstrap so its strict `pip check` gate can run. These checks do not call a production
server or require a microphone/model. `check` reports installed tools/profiles and
checks documentation; it is not a live service health check.

`dev` 不含第三方套件；API／TUI 分別使用既有鎖定依賴與驗證入口，避免缺依賴而
誤認所有功能已測完。上述測試不連正式服務、不需要模型或麥克風。

## Web frontend / Web 前端

```sh
python scripts/dev.py setup web
python scripts/dev.py web
# In another terminal / 另一個終端機：
python scripts/dev.py test web
python scripts/dev.py build web
```

Open the loopback URL printed by Vite and configure the separate HTTP API address
in the frontend. `setup web` uses `npm ci` with `client/web/package-lock.json`.
The development server stays in the foreground; Ctrl+C stops it.

## Linux server and optional TUI / Linux Server 與 TUI

```sh
python scripts/dev.py setup server
# Provision models and review config_server.py/environment first.
# 請先備妥模型並檢查設定、綁定位址與連接埠。
python scripts/dev.py server

python scripts/dev.py setup tui
python scripts/dev.py tui
```

`server` runs `start_server_docker.py` in the foreground and honors the existing
`CAPSWRITER_*` environment configuration. It does not read a Compose `.env` file
automatically: export the intended variables in this terminal or use the
documented Docker configuration. Use separate ports, model/cache locations and
containers when testing on a machine with an existing deployment. For GPU system
libraries and server image builds, use the existing
[deployment guide](zh-TW/deployment.md). These commands do not upgrade
or restart an existing axolotl service.

## Builds / 建置

```powershell
# On Windows x64 / Windows x64：
python scripts/dev.py setup build
python scripts/dev.py build desktop
```

The build profile installs the existing hash-locked bootstrap and Windows build
requirements, then runs the same `build.spec` as Windows CI. Output is
`dist/CapsWriter-Offline`. Build from a checkout without private models/logs in
the output directory; the specification rejects dirty mutable payloads.

```sh
python scripts/dev.py setup dev
python scripts/dev.py build cli
```

CLI output is in `client/cli/dist`; Web output is in `client/web/dist`.

## Dependency updates and cleanup / 更新依賴與清理

`pyproject.toml` has empty default dependencies and opt-in `desktop`, `server`,
`api`, `tui`, `dev`, `build` groups for source exploration. There is deliberately
no second universal `uv.lock`: the reproducible setup commands use the reviewed
`requirements-*.lock` files. Plain `uv sync --group ...` resolves a fresh local
environment and is not a substitute for release or acceptance verification.
Changing a group does not silently alter a locked profile. Review and regenerate
the corresponding lock when changing dependencies.

To regenerate the focused desktop lock after reviewing Windows release pins:

```sh
uv pip compile --group desktop --constraint requirements-windows-build.lock --python-version 3.12 --python-platform x86_64-pc-windows-msvc --generate-hashes --only-binary=:all: --no-binary=srt --no-annotate --output-file requirements-desktop-dev.lock
```

The wrapper removes inherited Python/uv environment-selection overrides so an
activated environment cannot redirect installation. It honors normal network
proxy variables. Setup and verification commands have a 30-minute timeout;
interactive clients and servers run until Ctrl+C.

`python scripts/clean.py` removes the repository's known generated build/test
outputs. It intentionally retains installed dependencies and user data. When
finished with these development profiles, close their processes and remove only
`.venv-dev` (PowerShell: `Remove-Item -Recurse -Force .venv-dev`; Linux:
`rm -rf -- .venv-dev`). Keep models, recordings, settings and unrelated Docker
resources. Remove only containers/images/volumes created by your own test run.

清理先使用 `python scripts/clean.py`，需連開發環境一起移除時再刪 `.venv-dev`。
請保留模型、錄音、設定與非本次建立的 Docker 資源。
