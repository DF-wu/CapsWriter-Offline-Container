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
| 修改的 upstream-tracked 檔案 | **5**：`.gitignore`、`readme.md`、`requirements-server.txt`、`LLM/default.py`、`assets/BUILD_GUIDE.md` |
| Fork 新增主要目錄 | `fork_server/`、`docker/`、`client/cli/`、`client/web/`、`docs/`、`.github/workflows/` |
| Hook 策略 | `ForkedCapsWriterServer` 子類化 + `server_manager.ws_send` 單點 monkey-patch |
| 唯一高漂移點 | [`fork_server/http_api/ws_send_with_http.py`](../fork_server/http_api/ws_send_with_http.py) 內嵌 upstream `ws_send` loop；HTTP unit test 會做 AST source guard，merge upstream 後若失敗需 re-port |
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
- HTTP API 預設不把 prompt/context 或轉錄文字寫入 server log/console；需明確設定 `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS=true` 才輸出全文。
- Server、模型調校與 HTTP API env 在啟動時 fail fast；錯誤 boolean、port、Qwen preset、數值範圍或 CORS origin 不會靜默退回預設值。
- `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` 對 HTTP 轉錄請求做 request-slot backpressure。
- HTTP API 啟用且 bind 到非 loopback 時會要求 Bearer key；server/client 皆支援 UTF-8 key file，無 KEY 只允許 loopback 或明確 insecure opt-out。
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
- `speak` 支援直接文字、UTF-8 檔案與 stdin，可串接 `transcribe --format text | speak --stdin`。
- HTTP error 會解析 OpenAI-style `error.message`、舊版 `detail`、bounded non-JSON body previews，並把 expected JSON endpoint 的 invalid JSON response 轉成帶 endpoint/status 的診斷。
- Linux TTS：`spd-say` / `espeak-ng` / `espeak`；Windows TTS：PowerShell `System.Speech`。
- 測試使用 in-process mock HTTP server，不需要模型。

### 3.4 Web Console

- [`client/web`](../client/web/) 是 React/Vite app。
- 支援錄音、上傳、播放、STT、五種輸出格式、HTTP readiness diagnostics、歷史紀錄、下載、browser Web Speech TTS。
- API client 會解析 OpenAI-style `error.message`、舊版 `detail`、bounded non-JSON HTTP error previews，並把 invalid JSON response 轉成帶 endpoint/status 的診斷。
- `npm run browser-smoke` 以 `agent-browser` 驗證真實瀏覽器 health/readiness、upload、transcribe workflow。
- [`client/web/Dockerfile`](../client/web/Dockerfile) 產出 Nginx static image。
- [`docker-compose.web.yml`](../docker-compose.web.yml) 提供 local build 部署。
- runtime config 由 container 啟動時寫入 `/config.js`，deploy-time 字串會 escape 成有效 JS string literal。

### 3.5 CI / Release

- [`scripts/verify_all.py`](../scripts/verify_all.py) 是 repo-level gate。
- [`scripts/clean.py`](../scripts/clean.py) 清 Python/Web/Docker 驗證輸出，並提供 `--check` residue gate。
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 跑 root gate。
- [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml) 發 server image。
- [`.github/workflows/publish-web-image.yml`](../.github/workflows/publish-web-image.yml) 發 Web Console image。

---

## 4. 已跑驗證

本分支已跑過以下 gate，且執行後確認無 build/cache/Docker 殘留：

| Gate | 結果 |
|---|---|
| `python -m unittest discover -s client/cli/tests -v` | 通過：CLI 31 tests，含 `/ready` ok/degraded diagnostic command、key-file auth/empty-file rejection、positive timeout validation、valid JSON output files、OpenAI-style error parsing、non-JSON/invalid JSON diagnostics、streamed multipart upload、multipart filename escaping 與 `speak --stdin` pipeline input |
| `python -m unittest discover -s docker/server/tests -v` | 通過：Docker server 17 tests，含 HTTP `/ready` healthcheck、healthcheck env parsing、model downloader env diagnostics、llama.cpp runtime library readiness 與 entrypoint Qwen CPU preset guard |
| `python -m unittest discover -s scripts/tests -v` | 通過：Verifier/diagnostic 35 tests，含 upstream divergence guard、live HTTP API key log redaction/key-file pass-through/empty-file rejection、diagnostic streamed multipart/configurable timeout/real HTTP body delivery/POST 401 handling、Docker Compose HTTP/model tuning env guard、HTTP API dependency guard、role template secret/default guard、`/health` 401 API-key guidance、cleanup traversal/residue gate 與 HTTP decode timeout source guard |
| `python scripts/verify_all.py --web-browser-smoke --docker-build-web --http-base-url http://127.0.0.1:6017` | 通過：CLI 24 tests、server compile、HTTP 51 tests、Docker server 12 tests、Verifier/diagnostic 23 tests、Web 36 tests/build、browser health/readiness/upload/transcribe smoke、Web Docker smoke、live `/health` |
| `python scripts/verify_all.py --skip-web --http-base-url http://127.0.0.1:16017 --http-key ... --http-require-ready --http-audio benchmarks/audio/arctic_a0001.wav --http-expect "Author of"` | 通過：CLI 24 tests、server compile、HTTP 51 tests、Docker server 15 tests、Verifier/diagnostic 23 tests、current-branch live `/health` v2.6、`/ready` ok、Qwen ASR model-backed STT (`Author of the Danger Trail, Philip Steels, etc.`) |
| `python scripts/verify_all.py --skip-web` | 通過：CLI 24 tests、server compile、HTTP 57 tests（含 fail-fast server/model env validation）、Docker server 17 tests（含 entrypoint Qwen CPU preset guard）、Verifier/diagnostic 28 tests（含 Docker Compose HTTP/model tuning env guard）、cleanup |
| `python scripts/verify_all.py` | 通過：upstream divergence guard、CLI 31 tests、server compile、HTTP 62 tests（含 server key-file auth 與 ws_send drift guard）、Docker server 17 tests、Verifier/diagnostic 35 tests、Web 40 tests/build（含 malformed history/runtime-config filtering）、cleanup + residue check |

`127.0.0.1:6017` 仍是既有外部服務；本分支驗證使用隔離容器掛載目前 checkout，對外映射 `127.0.0.1:16017`，並以 temporary API key 執行 `/ready` 與已知音檔 STT gate。共享 verifier log 會將 `--http-key` 顯示為 `<redacted>`。

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
| `ws_send_with_http.py` 需人工追 upstream | 已對齊 `origin/master @ 7d7fac3`；HTTP unit test 會偵測未 re-port 的 upstream loop drift，失敗時仍需人工搬回上游修改 |
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
