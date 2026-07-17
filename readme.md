# CapsWriter-Offline — Windows + Linux 跨平台 Fork

> Windows 桌面、Linux 桌面與 server 的離線語音辨識，另提供 Docker 部署、
> 選用 OpenAI 相容 API，以及 Web、CLI、TUI clients。
>
> 繁體中文 · [English](README.en.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux-334155)](docs/zh-TW/desktop-portability.md)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![OpenAI compatible](https://img.shields.io/badge/OpenAI%20audio-compatible-10A37F)](docs/zh-TW/openai-api.md)

本 fork 以上游 CapsWriter v2 的模型／推論算法與 Windows 桌面體驗為基線，並保留
一組有文件記錄的相容性／安全 patch（含少數 engine I/O 與 privacy logging
接觸點），再把產品擴充為有自動化驗證的跨平台交付面：加入有明確邊界的 Linux
X11 桌面路徑、Linux container server、選用 HTTP API、browser／terminal clients、
可重現的 dependency lock，以及 release／security gates。

因此，現在把此 repository 稱為「僅限 Linux server 的貼皮」已不正確。Windows
仍是一等桌面與打包目標；Linux 則依使用方式有不同支援邊界：X11 global hotkey
不提供單鍵選擇性阻擋，Wayland global hotkey 不支援，而 headless Linux 應使用
server、Web、CLI 或 TUI 路徑。

模型資產備妥後，推論會留在本機；container 第一次 bootstrap 可能需要下載模型
與 runtime asset，但不需要雲端推論服務。

## 內含功能

| 介面 | 適用情境 | 現有證據與邊界 |
|---|---|---|
| Windows 桌面 | Tray、快捷鍵、錄音、檔案轉錄、選用 HTTP API | Windows 2022／Python 3.12 CI 會 hash-install、build、搬移、解壓、檢查並透過兩個 packaged EXE 執行 import self-check；tray／audio／model／hardware 行為仍是 manual release evidence |
| Linux X11 桌面 | X11 session 內的桌面快捷鍵與上游 client | 已測 portable callback contract；可監聽，但刻意停用單鍵 selective suppression |
| Linux `amd64` server/container | 長時間執行的本機或共享 ASR service | Docker／Compose、health／readiness、model bootstrap、GPU preference 與 CPU fallback 都有 gate；ARM64 沒有 release gate |
| OpenAI 相容 API | SDK、curl、Web、CLI、TUI 轉錄 | 選用 `whisper-1` 檔案轉錄子集；不支援能力會明確回錯 |
| Web Console | Browser 錄音、upload、STT 格式、下載、本機 browser TTS | React/Vite tests、production build、browser smoke、static image smoke |
| 無 GUI CLI | Script、SSH、batch 轉錄、本機 OS TTS | Standard-library zipapp；Linux／Windows Python portability matrix |
| Textual TUI | 鍵盤優先診斷、檔案轉錄、選用麥克風 | Hash-locked Python 3.10–3.12 runtime 與禁止 skip 的 Pilot suite |

正式部署或宣稱桌面支援前，請先讀完整的
[支援與安全矩陣](docs/zh-TW/support-security.md)。

## 選擇使用路徑

| 目標 | 從這裡開始 |
|---|---|
| 使用或打包 Windows 桌面程式 | [桌面可攜性](docs/zh-TW/desktop-portability.md#windows-打包與-http-api) |
| 在 Linux X11 執行桌面 client | [Linux X11 快捷鍵與限制](docs/zh-TW/desktop-portability.md#linux-x11-快捷鍵) |
| 用 Docker 啟動 Linux server | [開始使用](docs/zh-TW/getting-started.md#路徑-c-linux-container-server) |
| 部署 server + browser console | [部署指南](docs/zh-TW/deployment.md) |
| 使用 OpenAI SDK 或 curl | [OpenAI 相容 API](docs/zh-TW/openai-api.md) |
| 透過 shell／SSH 自動化 | [無 GUI CLI 指南](docs/zh-TW/cli-client.md) |
| 使用鍵盤優先 terminal UI | [TUI 指南](docs/zh-TW/tui.md) |
| 排查故障 | [疑難排解](docs/zh-TW/troubleshooting.md) |
| 升級或檢視目前 release | [Release notes](docs/zh-TW/release-notes.md) |

[繁體中文文件首頁](docs/zh-TW/README.md)會串起所有 user、operator 與 maintainer
路徑。

## 最快的 server 啟動方式

先決條件：`linux/amd64` host、Docker Engine、Compose plugin，以及足以容納所選 model 與
image 的磁碟空間。GPU 為選用。

```bash
git clone https://github.com/DF-wu/CapsWriter-Offline-Container.git
cd CapsWriter-Offline-Container
cp .env.example .env
cp hot-server.example.txt hot-server.txt
docker compose up -d capswriter-server
docker compose ps
docker compose logs -f capswriter-server
```

Base Compose 不會要求任何 vendor device，因此 CPU-only Docker host 也能直接
使用上述 command。若要把 NVIDIA GPU 暴露給 container，請明確加入 override：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d capswriter-server
```

Linux Intel／AMD iGPU 請先用
`stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*` 取得 host GID、填入 `.env` 的
`CAPSWRITER_DRI_RENDER_GID`／`CAPSWRITER_DRI_VIDEO_GID`，再加入
`docker-compose.igpu.yml`。Model 預設保留在 Docker named volume；只有需要直接
管理 host `./models` 時才加入 `docker-compose.models-bind.yml`，並先設定正確
container-user ownership。完整 command 與 lock 要求見
[部署指南](docs/zh-TW/deployment.md#linux-container-profile)。

Compose 預設只在 host loopback 的 `6016` port 發布 WebSocket server；OpenAI
相容 HTTP API 預設關閉。

若要啟用，請在 `.env` 設定強健的本機 token：

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
```

接著取消 [`docker-compose.yml`](docker-compose.yml) 內 `ports:` 下方 HTTP
mapping 的註解，重建 service，並分別檢查 liveness 與 readiness：

```bash
docker compose up -d --force-recreate capswriter-server
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

只要把啟用的 API 綁到 loopback 以外，就必須認證，除非明確打開僅供隔離測試
網路使用的 insecure escape hatch。發布到 LAN 前，請先讀
[部署](docs/zh-TW/deployment.md)與
[支援／安全](docs/zh-TW/support-security.md)。

## Client 入口

### Web Console

開發模式使用 locked Node dependency tree：

```bash
cd client/web
npm ci --no-audit --no-fund
npm run dev
```

Static container 部署：

```bash
docker compose -f docker-compose.web.yml up -d --build capswriter-web
```

[Web Console 指南](docs/web-console.md)說明 CORS、麥克風 secure context、runtime
configuration 與 production image 驗證。

### 無 GUI CLI

```bash
python client/cli/capswriter_cli.py ready \
  --base-url http://127.0.0.1:6017 \
  --key-file /path/to/capswriter-http.key
python client/cli/capswriter_cli.py transcribe meeting.wav --format text
python client/cli/scripts/build_zipapp.py
```

[CLI 指南](docs/zh-TW/cli-client.md)包含 batch output、可攜檔名、timeout、zipapp
package 與本機 TTS。

### Textual TUI

```bash
python3.12 -m venv .venv-tui
.venv-tui/bin/python -m pip install \
  --require-hashes --only-binary=:all: \
  --requirement requirements-tui.lock
.venv-tui/bin/python -m client.tui --base-url http://127.0.0.1:6017
```

[TUI 指南](docs/zh-TW/tui.md)包含 Windows command、繁體中文 UI、快捷鍵、
file-only fallback 與 strict verification。

## 桌面路徑

Windows packaging 會分析 client 與
[`start_server_universal.py`](start_server_universal.py)：HTTP API 關閉時保留上游
desktop configuration；啟用時只加入經驗證的 `CAPSWRITER_HTTP_API_*` 設定。
Linux 桌面 global hotkey 只支援 X11，且絕不使用不安全的 whole-keyboard grab。

完整 Windows build command、X11 requirements、Wayland／headless 限制，以及
仍需由真實 hardware 補齊的 release evidence，請見
[桌面可攜性](docs/zh-TW/desktop-portability.md)。

## 安全預設

- HTTP API 預設關閉；Compose 預設在 `127.0.0.1` 發布 service port。
- 啟用且非 loopback 的 API 必須使用 Bearer key 或 key file，除非明確設定
  insecure-bind override。
- Transcript 與 prompt logging 預設關閉。
- Web runtime config 除非另設 public-key opt-in，否則拒絕發布預設 API key；
  在 UI 內輸入 key 較安全。
- Container 使用 `no-new-privileges` 並 drop Linux capabilities。
- Docker／TUI Python 與 Web dependency 都有 reproducible lock；publish workflow
  會產出 provenance 與 SBOM attestation。

完整 security behavior、private-data boundary、支援／不支援平台與 reporting
說明集中在[支援與安全](docs/zh-TW/support-security.md)。

## 驗證與 release policy

Portable contract 會在 pinned Ubuntu、Windows runner 上，以 Python 3.10 與 3.12
執行。另有獨立 Python 3.12 Windows job，會安裝完整 hash production lock、build
兩個 PyInstaller executable，把 artifact 搬離 checkout 後經 ZIP 壓縮／解壓、拒絕
reparse point、透過兩個 EXE 執行 import smoke，再上傳實際測過的 ZIP。API contract
與 hash-locked TUI 有各自的 isolated no-skip job；root gate 涵蓋 server、Docker、
CLI、Web、文件、workflow source guard 與 cleanup。Model、audio、tray、display、
hardware evidence 仍是 release candidate 的明確責任，不會從 import smoke 或 unit
test 推論。

![Fork release 流程：active v2 經跨平台與安全閘門，legacy v1 維持隔離](docs/assets/version-tracks.svg)

文字等價說明：上游正式變更會合併進 active fork v2，通過 Linux、Windows、
API、TUI、Web 與 security gates 後才發布 v2。Legacy v1 保持隔離，只有重大或
安全修正會人工 backport，並通過獨立 legacy gates。

本機 gate：

```bash
python scripts/verify_all.py
PYTHONDONTWRITEBYTECODE=1 python scripts/check_docs.py
python scripts/clean.py --check
```

另請參閱[驗證](docs/verification.md)、[v1／v2 policy](docs/zh-TW/versioning.md)與
[目前 release notes](docs/zh-TW/release-notes.md)。

## 文件

| 文件 | 用途 |
|---|---|
| [文件首頁](docs/zh-TW/README.md) | 完整 task／audience index |
| [開始使用](docs/zh-TW/getting-started.md) | 選擇並驗證正確的 Windows／Linux 路徑 |
| [部署](docs/zh-TW/deployment.md) | Container、desktop source、Web、network、升級、rollback 操作 |
| [疑難排解](docs/zh-TW/troubleshooting.md) | Desktop、container、API、client 的診斷順序 |
| [支援與安全](docs/zh-TW/support-security.md) | Support matrix、限制、secret、隱私、supply chain、reporting |
| [Release notes](docs/zh-TW/release-notes.md) | 目前 fork v2 snapshot、變更、migration、已知限制 |
| [桌面可攜性](docs/zh-TW/desktop-portability.md) | Windows package contract；Linux X11、Wayland、headless 真實邊界 |
| [OpenAI 相容 API](docs/zh-TW/openai-api.md) | HTTP／SDK contract 與 resource controls |
| [TUI](docs/zh-TW/tui.md) | Textual workbench 安裝與操作 |
| [架構](docs/architecture.md) | Sidecar integration 與 upstream drift strategy |

## 上游、生命週期與授權

本 fork 基於
[HaujetZhao/CapsWriter-Offline](https://github.com/HaujetZhao/CapsWriter-Offline)，
並持續使用其模型／推論算法與 desktop product 工作。Upstream-tracked engine
檔案內的少數修改只處理 bounded I/O 與 privacy-aware logging；本 fork 加入
delivery、portability、API、client 與 operational surfaces，不會重新命名上游
release，也不會把上游 model work 宣稱為 fork 所有。

Fork v2 是 active development line；legacy fork v1 只接受重大與安全維護。
跨世代 merge 或 backport 前，請先讀[版本政策](docs/zh-TW/versioning.md)。

License：[MIT](LICENSE)。
