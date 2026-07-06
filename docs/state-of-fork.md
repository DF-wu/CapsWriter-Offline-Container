# Fork 現況（State of Fork）

> **更新時間**：2026-07-07
> **基底**：`origin/master` @ `7d7fac3`（upstream v2.6）
> **工作分支**：`feature/universal-asr-client`

---

## 1. 一句話定位

本 fork 在不改動 ASR 引擎核心的前提下，將 CapsWriter-Offline 做成可部署、可測、可發布的離線語音服務：

| 面向 | 交付 |
|---|---|
| Server | Docker 化、env 驅動設定、GPU/CPU fallback、OpenAI Whisper-compatible HTTP API |
| Client | 標準函式庫 no-GUI CLI、瀏覽器 Web Console、保留上游桌面 GUI |
| Web | React/Vite STT/TTS 工作台、Nginx static image、runtime `/config.js` |
| Verification | repo-level gate、CLI/server/Web unit tests、Docker smoke、可選 live STT |
| Release | server/web GHCR publish workflows |
| Docs | 架構、HTTP API、CLI、Web、Docker、驗證與上游同步文件，含 SVG 架構圖 |

ASR/標點/對齊引擎仍完全來自 upstream `core/server/engines/*`。

---

## 2. 分支設計現況

| | |
|---|---|
| 修改的 upstream 檔案 | **2**：`.gitignore`、`readme.md` |
| Fork 新增主要目錄 | `fork_server/`、`docker/`、`client/cli/`、`client/web/`、`docs/`、`.github/workflows/` |
| Hook 策略 | `ForkedCapsWriterServer` 子類化 + `server_manager.ws_send` 單點 monkey-patch |
| 唯一高漂移點 | [`fork_server/http_api/ws_send_with_http.py`](../fork_server/http_api/ws_send_with_http.py) 內嵌 upstream `ws_send` loop，merge upstream 後需比對 |
| 預設相容性 | 不設 `CAPSWRITER_HTTP_API_ENABLE=true` 時，server 啟動路徑保持接近 upstream WebSocket 模式 |

詳細架構見 [architecture.md](architecture.md)。上游同步 SOP 見 [upstream-sync-guide.md](upstream-sync-guide.md)。

---

## 3. 目前可用功能

### 3.1 Server / HTTP API

- [`start_server_docker.py`](../start_server_docker.py) 在 import upstream server 前套用 [`fork_server/env_config.py`](../fork_server/env_config.py)。
- [`fork_server/http_api/api.py`](../fork_server/http_api/api.py) 提供：
  - `GET /health`
  - `GET /ready`
  - `GET /v1/models`
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`（明確 501）
- HTTP 上傳以 chunk 讀取並套用 `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB`，成功回應帶 `X-CapsWriter-Task-ID`。
- `/ready` 回報 `task_router` 與 `ffmpeg` readiness，不暴露 secrets。
- `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` 對 HTTP 轉錄請求做 request-slot backpressure。
- HTTP API 啟用且 bind 到非 loopback 時會要求 Bearer key；無 KEY 只允許 loopback 或明確 insecure opt-out。
- HTTP error 使用 OpenAI-style `{"error": ...}` JSON envelope，保留原 HTTP status。
- `language` 會正規化 OpenAI-style alias（如 `zh`/`en`）並傳入 recognizer；`prompt` 會正規化後作為 upstream `Task.context`。
- `response_format` 支援 `json`、`text`、`verbose_json`、`srt`、`vtt`。

### 3.2 Docker / 模型

- [`docker/server/Dockerfile`](../docker/server/Dockerfile) 建 server image。
- [`docker/server/download_models.py`](../docker/server/download_models.py) 依 `CAPSWRITER_MODEL_TYPE` 自動下載模型。
- [`docker/server/entrypoint.sh`](../docker/server/entrypoint.sh) 做 GPU/CPU backend 選擇與 fallback。
- [`docker/server/healthcheck.py`](../docker/server/healthcheck.py) 一律檢查 WebSocket port；若啟用 HTTP API，還要求 `/ready` 回 `status="ok"`。
- [`docker-compose.yml`](../docker-compose.yml) 啟動 server；[`docker-compose.fun-asr.yml`](../docker-compose.fun-asr.yml) 切低延遲 Fun-ASR。

### 3.3 No-GUI CLI

- [`client/cli/capswriter_cli.py`](../client/cli/capswriter_cli.py) 無第三方 Python dependency。
- 支援 `health`、`ready`、`models`、`transcribe`、`speak`。
- HTTP error 會解析 OpenAI-style `error.message` 與舊版 `detail`。
- Linux TTS：`spd-say` / `espeak-ng` / `espeak`；Windows TTS：PowerShell `System.Speech`。
- 測試使用 in-process mock HTTP server，不需要模型。

### 3.4 Web Console

- [`client/web`](../client/web/) 是 React/Vite app。
- 支援錄音、上傳、播放、STT、五種輸出格式、HTTP readiness diagnostics、歷史紀錄、下載、browser Web Speech TTS。
- API client 會解析 OpenAI-style `error.message` 與舊版 `detail`。
- `npm run browser-smoke` 以 `agent-browser` 驗證真實瀏覽器 health/readiness、upload、transcribe workflow。
- [`client/web/Dockerfile`](../client/web/Dockerfile) 產出 Nginx static image。
- [`docker-compose.web.yml`](../docker-compose.web.yml) 提供 local build 部署。
- runtime config 由 container 啟動時寫入 `/config.js`。

### 3.5 CI / Release

- [`scripts/verify_all.py`](../scripts/verify_all.py) 是 repo-level gate。
- [`scripts/clean.py`](../scripts/clean.py) 清 Python/Web/Docker 驗證輸出。
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 跑 root gate。
- [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml) 發 server image。
- [`.github/workflows/publish-web-image.yml`](../.github/workflows/publish-web-image.yml) 發 Web Console image。

---

## 4. 已跑驗證

本分支已跑過以下 gate，且執行後確認無 build/cache/Docker 殘留：

| Gate | 結果 |
|---|---|
| `python -m unittest discover -s client/cli/tests -v` | 通過：CLI 10 tests，含 `/ready` ok/degraded diagnostic command 與 OpenAI-style error parsing |
| `python -m unittest discover -s docker/server/tests -v` | 通過：Docker server 11 tests，含 HTTP `/ready` healthcheck、healthcheck env parsing 與 model downloader env diagnostics |
| `python -m unittest discover -s scripts/tests -v` | 通過：Verifier 4 tests，含 live HTTP API key log redaction |
| `python scripts/verify_all.py --web-browser-smoke --docker-build-web --http-base-url http://127.0.0.1:6017` | 通過：CLI 10 tests、server compile、HTTP 42 tests、Docker server 11 tests、Verifier 4 tests、Web 18 tests/build、browser health/readiness/upload/transcribe smoke、Web Docker smoke、live `/health` |

`--http-require-ready` 已加入 root verifier；目前 `127.0.0.1:6017` 上的 live process 仍是舊版 `v2.5`，需重啟到本分支後 `/ready` 才會從 404 變成可驗證 endpoint。因目前 shell 沒有 live server 的 API key，模型音檔 smoke 會在 `/v1/audio/transcriptions` 收到 401；release evidence 需提供 `--http-key`。

Release candidate 若要宣稱「模型轉錄品質已驗證」，需另外提供已知內容音檔並跑：

```bash
python scripts/verify_all.py \
  --http-base-url http://127.0.0.1:6017 \
  --http-audio /path/to/known-speech.wav \
  --http-expect "expected transcript text"
```

---

## 5. 已知限制

| 限制 | 狀態 / 處理 |
|---|---|
| CI 不下載模型 | 預期設計；CI 驗證協議、格式、build、Docker smoke。模型品質由 `--http-audio` release gate 補足 |
| Browser TTS 可用性依賴瀏覽器 / OS voice | Web Console 文件已標明；不走雲端 TTS |
| `ws_send_with_http.py` 需人工追 upstream | 已對齊 `origin/master @ 7d7fac3`；upstream merge checklist 已列入 |
| 公開 image 需 merge 到 master 後由 workflow 發布 | feature branch 只提供 workflow 與 local Docker smoke |

---

## 6. 檔案地圖

| 路徑 | 用途 |
|---|---|
| [`fork_server/`](../fork_server/) | Server sidecar、HTTP API、env config、upstream hook |
| [`docker/`](../docker/) | Server image、entrypoint、模型下載、backend probe |
| [`client/cli/`](../client/cli/) | no-GUI Python CLI 與 tests |
| [`client/web/`](../client/web/) | React/Vite Web Console、mock API、Docker/Nginx deployment |
| [`docs/assets/`](assets/) | 架構與驗證 SVG |
| [`scripts/`](../scripts/) | root verify/clean automation |
| [`.github/workflows/`](../.github/workflows/) | CI 與 GHCR publish workflows |
| [`docker-compose*.yml`](../docker-compose.yml) | server、Fun-ASR、Web deployment entry points |

---

## 7. 回滾與追溯

本分支每個功能切片都有 commit：

```bash
git log --oneline origin/master..feature/universal-asr-client
```

回滾單一切片優先使用 `git revert <commit>`。不要用 `git reset --hard` 清工作樹，除非已明確決定丟棄整個分支狀態。
