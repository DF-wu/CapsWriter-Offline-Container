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
| 修改的 upstream-tracked 檔案 | **23**：`.gitignore`、`readme.md`、`requirements-server.txt`、`LLM/default.py`、`assets/BUILD_GUIDE.md`、`zip_release.py`、`core/client/audio/file_manager.py`、`core/client/hotword/hotword_standalone.py`、`core/client/hotword/hotword_standalone.ipynb`、`core/client/manager/tray_manager.py`、`core/client/transcribe/media_tool.py`、`core/client/transcribe/file_transcriber.py`、`core/server/engines/qwen_asr_gguf/export/gguf/utility.py`、`core/server/engines/force_aligner_gguf/export/gguf/utility.py`、`core/server/engines/fun_asr_gguf/export/gguf/utility.py`、`core/server/engines/qwen_asr_gguf/inference/audio.py`、`core/server/engines/force_aligner_gguf/inference/audio.py`、`core/server/engines/sensevoice_onnx/inference/audio.py`、`core/server/engines/fun_asr_gguf/inference/audio.py`、`core/server/worker/gpu_boost.py`、`core/server/worker/process_manager.py`、`core/tools/window_detector.py`、`core/ui/tray.py` |
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
- HTTP 音訊解碼以 server task timeout 限制 ffmpeg，且 timeout 後的 kill/wait cleanup 也有上限。
- `/ready` 回報 `task_router` 與 `ffmpeg` readiness，不暴露 secrets。
- HTTP API 預設不把 prompt/context 或轉錄文字寫入 server log/console；需明確設定 `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS=true` 才輸出全文。
- Server、模型調校與 HTTP API env 在啟動時 fail fast；錯誤 boolean、port、Qwen preset、數值範圍、non-finite timeout 或 CORS origin 不會靜默退回預設值。
- `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` 對 HTTP 轉錄請求做 request-slot backpressure。
- HTTP timeout/client cancel 會清除 pending future 與合成 socket id，並以 bounded tombstone 吸收晚到 recognizer result，避免被誤派到 WebSocket 路徑。
- HTTP API 啟用且 bind 到非 loopback 時會要求 Bearer key；server/client 皆支援 UTF-8 key file，無 KEY 只允許 loopback 或明確 insecure opt-out。
- HTTP error 使用 OpenAI-style `{"error": ...}` JSON envelope，保留原 HTTP status。
- `language` 會正規化 OpenAI-style alias（如 `zh`/`en`）、限制長度與安全字元後傳入 recognizer；`prompt` 會正規化後作為 upstream `Task.context`。
- `response_format` 支援 `json`、`text`、`verbose_json`、`srt`、`vtt`。

### 3.2 Docker / 模型

- [`docker/server/Dockerfile`](../docker/server/Dockerfile) 建 server image。
- [`docker/server/download_models.py`](../docker/server/download_models.py) 依 `CAPSWRITER_MODEL_TYPE` 自動下載模型，archive 下載有 idle timeout，先寫 `.part` 再 atomic replace，且解壓前拒絕 traversal / link / special-file archive members。
- [`docker/server/entrypoint.sh`](../docker/server/entrypoint.sh) 做 GPU/CPU backend 選擇與 fallback。
- [`docker/server/healthcheck.py`](../docker/server/healthcheck.py) 一律檢查 WebSocket port；若啟用 HTTP API，還要求 `/ready` 回 `status="ok"`。
- [`docker-compose.yml`](../docker-compose.yml) 啟動 server；[`docker-compose.fun-asr.yml`](../docker-compose.fun-asr.yml) 切低延遲 Fun-ASR。

### 3.3 No-GUI CLI

- [`client/cli/capswriter_cli.py`](../client/cli/capswriter_cli.py) 無第三方 Python dependency。
- 支援 `health`、`ready`、`models`、`transcribe`、`speak`。
- CLI 與診斷工具預設 timeout 對齊 server `CAPSWRITER_HTTP_API_TASK_TIMEOUT=600`，長音訊可用 `--timeout` 覆寫；CLI 另以 `--max-response-mb` 限制 HTTP response body 記憶體上限。
- `speak` 支援直接文字、UTF-8 檔案與 stdin，可串接 `transcribe --format text | speak --stdin`，local TTS 子程序預設以 `--tts-timeout 120` 設上限。
- `transcribe --output` / `--output-dir` 以 same-directory temp file + atomic replace 寫入 transcript；`--output-dir` batch 每個成功項目會立即落盤，replace 失敗時保留舊檔並清掉暫存檔。
- `transcribe --output-dir` 會 sanitize generated filename stems（Windows invalid chars / reserved device names / hidden-dot forms / overly long stems），並在送出 HTTP 前拒絕重複生成的輸出路徑，避免 batch 檔案被靜默覆寫。
- `--base-url` 僅接受 absolute `http://` / `https://` root（可帶 path prefix 與尾端 `/v1`）；URL credentials、query、fragment 與非 HTTP scheme 會在送 request 前被拒絕。
- HTTP error 會解析 OpenAI-style `error.message`、舊版 `detail`、bounded non-JSON body previews，並把 expected JSON endpoint 的 invalid JSON response 轉成帶 endpoint/status 的診斷。
- Linux TTS：`spd-say` / `espeak-ng` / `espeak`；Windows TTS：PowerShell `System.Speech`。
- 測試使用 in-process mock HTTP server，不需要模型。

### 3.4 Web Console

- [`client/web`](../client/web/) 是 React/Vite app。
- 支援錄音、鍵盤可操作的檔案選擇、拖放上傳、根據 `/ready.config.max_upload_mb` 做上傳大小預檢、播放、STT、五種輸出格式、可 abort 的 HTTP readiness diagnostics、歷史紀錄、下載、browser Web Speech TTS；轉錄中會鎖定音訊替換，取消/換檔/卸載後會忽略 stale result，卸載後也會忽略晚到的 diagnostic、clipboard、voice loading、speech callback 與 media stream，且 dev StrictMode 不會誤判已卸載而丟棄 diagnostics。
- API client 會以 16 MiB 上限讀取 response body，解析 OpenAI-style `error.message`、舊版 `detail`、bounded non-JSON HTTP error previews，並把 invalid JSON response 轉成帶 endpoint/status 的診斷；health/readiness/model diagnostics 有 10 秒 timeout，transcription request 有 server-aligned 600 秒 timeout，兩者都保留 caller abort。
- API root 會在 health/readiness/model/transcription request 前驗證，只允許 absolute HTTP(S) root，並拒絕 URL credentials、query、fragment 與非 HTTP scheme。
- 下載檔名會 sanitize path separator、控制字元、OS 保留字元與 Windows reserved device name；歷史資料即使被手動污染也不會直接成為原始 download filename。
- Web settings controls 與 persisted settings/history 會做型別、格式與字串長度邊界檢查；手動污染或過大的 localStorage record 不會直接餵回 UI。
- `npm run browser-smoke` 以 `agent-browser` 驗證真實瀏覽器 health/readiness、upload、transcribe workflow；`agent-browser` subprocess、HTTP startup probe 與 mock API/Vite child cleanup 都有 bounded timeout。
- [`client/web/Dockerfile`](../client/web/Dockerfile) 產出 Nginx static image，並套用 CSP、frame、MIME sniffing、referrer 與 permissions-policy 等基本瀏覽器安全 header。
- [`docker-compose.web.yml`](../docker-compose.web.yml) 提供 local build 部署。
- runtime config 由 container 啟動時寫入 `/config.js`，deploy-time 字串會 escape 成有效 JS string literal。

### 3.5 CI / Release

- [`scripts/verify_all.py`](../scripts/verify_all.py) 是 repo-level gate。
- [`scripts/clean.py`](../scripts/clean.py) 清 Python/Web/Docker 驗證輸出，bounded `npm run clean` 後再跑 Python fallback，並提供 `--check` residue gate。
- [`zip_release.py`](../zip_release.py) 的 7-Zip release packaging subprocess 有 bounded timeout，失敗時也會清掉暫存 `file_list_*.txt`。
- GUI recorder 的 [`core/client/audio/file_manager.py`](../core/client/audio/file_manager.py) MP3 `ffmpeg` finalize 以 `CAPSWRITER_CLIENT_AUDIO_FINISH_TIMEOUT` 限制，timeout 後會嘗試 bounded kill cleanup。
- [`core/client/hotword/hotword_standalone.py`](../core/client/hotword/hotword_standalone.py) 的 local Ollama chat helper 以 `CAPSWRITER_OLLAMA_CHAT_TIMEOUT` 限制 request，避免 demo/client 流程被未回應的本機 LLM endpoint 卡住。
- GUI file transcription 的 [`core/client/transcribe/media_tool.py`](../core/client/transcribe/media_tool.py) `ffprobe` duration probe 與 [`core/client/transcribe/file_transcriber.py`](../core/client/transcribe/file_transcriber.py) `ffmpeg` streaming subprocess 以 `CAPSWRITER_CLIENT_MEDIA_TIMEOUT` 限制，timeout 後會嘗試 bounded kill cleanup。
- Qwen / force-aligner / Fun-ASR GGUF export utilities 的 remote safetensor `GET`/`HEAD` requests 以 `CAPSWRITER_GGUF_EXPORT_HTTP_TIMEOUT` 限制，避免轉檔工具被未回應的 Hugging Face endpoint 卡住。
- Qwen / force-aligner / SenseVoice / Fun-ASR direct file-decode helpers 的 `ffmpeg` subprocess 以 `CAPSWRITER_ENGINE_FFMPEG_TIMEOUT` 限制，timeout 後會嘗試 bounded kill cleanup，stderr diagnostic 也會截斷。
- [`core/server/worker/gpu_boost.py`](../core/server/worker/gpu_boost.py) 的 GPU boost/unboost shell command 以 `CAPSWRITER_GPU_BOOST_TIMEOUT` 限制，避免自訂管理命令卡住 worker loop。
- [`core/server/worker/process_manager.py`](../core/server/worker/process_manager.py) 的 recognizer worker shutdown 以 `CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT` 限制 graceful join，之後會 bounded terminate wait 並在可用時 kill。
- Standalone hotword `.py` 與 notebook demo 都以 `CAPSWRITER_OLLAMA_CHAT_TIMEOUT` 限制 local Ollama request。
- GUI tray restart 與 hotword-file default opener subprocesses 使用 detached stdio/session launch，避免使用者動作或重啟 child 繼承 console/verification handles。
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 跑 root gate。
- [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml) 先跑 `python scripts/verify_all.py --skip-web`，通過後發 server image。
- [`.github/workflows/publish-web-image.yml`](../.github/workflows/publish-web-image.yml) 先跑 `python scripts/verify_all.py --docker-build-web`，含 Web Docker smoke，通過後發 Web Console image。

---

## 4. 已跑驗證

本分支已跑過以下 gate，且執行後確認無 build/cache/Docker 殘留：

| Gate | 結果 |
|---|---|
| `python -m unittest discover -s client/cli/tests -v` | 通過：CLI 58 tests，含 `/ready` ok/degraded diagnostic command、key-file auth/empty-file rejection、base URL validation before requests、server-aligned default timeout/positive timeout validation、bounded HTTP response body reads / configurable `--max-response-mb`、valid JSON output files、atomic transcript output writes/temp cleanup、per-item durable batch output、portable sanitized `--output-dir` target generation/duplicate rejection、OpenAI-style error parsing、non-JSON/invalid JSON diagnostics、streamed multipart upload、multipart filename escaping、`speak --stdin` pipeline input、bounded local TTS timeout handling 與 standalone CLI verifier timeout handling |
| `python -m unittest discover -s docker/server/tests -v` | 通過：Docker server 25 tests，含 HTTP `/ready` healthcheck、healthcheck env parsing、model downloader bounded atomic download/timeout diagnostics、安全 archive extraction traversal/link guard、clean download-failure reporting、llama.cpp runtime library readiness 與 entrypoint Qwen CPU preset guard |
| `python -m unittest discover -s scripts/tests -v` | 通過：Verifier/diagnostic 116 tests，含 upstream divergence guard bounded git subprocess handling、root verifier per-command timeout/redacted timeout reporting、bounded Web clean/verify subprocess handling、bounded browser-smoke `agent-browser` subprocess/HTTP probe/child cleanup handling、Web Nginx security-header guard、GHCR publish workflow release-gate guard、release ZIP packaging timeout/temp-file cleanup guards、desktop window detector bounded `osascript`/`wmctrl` subprocess guards、standalone hotword Ollama request timeout/notebook source guard、GUI tray detached process launch guard、GUI recorder MP3 finalize timeout/cleanup guard、GUI file transcription media subprocess timeout/cleanup guard、GGUF export HTTP request timeout guard、direct engine ffmpeg decode timeout/cleanup/error-preview guard、server GPU boost command timeout guard、server worker shutdown timeout/cleanup guard、live HTTP API key log redaction/key-file pass-through/empty-file rejection、diagnostic host/port validation、diagnostic streamed multipart/bounded response body reads/server-aligned default timeout/configurable timeout/real HTTP body delivery/POST 401 handling、Docker Compose HTTP/model/resource env guard、HTTP API dependency guard、role template secret/default guard、`/health` 401 API-key guidance、cleanup traversal/residue gate 與 HTTP decode timeout source guard |
| `cd client/web && npm run browser-smoke` | 通過：bounded `agent-browser` subprocess、bounded HTTP startup probe、bounded mock API/Vite child cleanup、dev StrictMode diagnostics、real-browser health/readiness/upload/transcribe workflow |
| `python scripts/verify_all.py --web-browser-smoke --docker-build-web --http-base-url http://127.0.0.1:6017` | 通過：CLI 24 tests、server compile、HTTP 51 tests、Docker server 12 tests、Verifier/diagnostic 23 tests、Web 36 tests/build、browser health/readiness/upload/transcribe smoke、Web Docker smoke、live `/health` |
| `python scripts/verify_all.py --skip-web --http-base-url http://127.0.0.1:16017 --http-key ... --http-require-ready --http-audio benchmarks/audio/arctic_a0001.wav --http-expect "Author of"` | 通過：CLI 24 tests、server compile、HTTP 51 tests、Docker server 15 tests、Verifier/diagnostic 23 tests、current-branch live `/health` v2.6、`/ready` ok、Qwen ASR model-backed STT (`Author of the Danger Trail, Philip Steels, etc.`) |
| `python scripts/verify_all.py --skip-web` | 通過：CLI 24 tests、server compile、HTTP 57 tests（含 fail-fast server/model env validation）、Docker server 17 tests（含 entrypoint Qwen CPU preset guard）、Verifier/diagnostic 28 tests（含 Docker Compose HTTP/model tuning env guard）、cleanup |
| `python scripts/verify_all.py` | 通過：upstream divergence guard（含 bounded git subprocess handling）、CLI 58 tests + packaged stdin smoke（含 bounded HTTP response body reads、atomic transcript output writes/temp cleanup、per-item durable batch output、standalone verifier timeout handling 與 portable `--output-dir` filename sanitization/collision guard）、server compile、HTTP 73 tests（含 server key-file auth、translations endpoint auth consistency、bounded finite ffmpeg timeout validation/kill cleanup、bounded language hints、late canceled HTTP result absorption 與 ws_send drift guard）、Docker server 25 tests（含 bounded atomic model archive downloads、安全 archive extraction traversal/link guard 與 clean download-failure reporting）、Verifier/diagnostic 116 tests（含 root verifier per-command timeout/redacted timeout reporting、bounded diagnostic response body reads、bounded Web clean/verify subprocess handling、bounded browser-smoke subprocess/HTTP probe/cleanup handling、Web Nginx security-header guard、GHCR publish workflow release-gate guard、release ZIP packaging timeout/temp-file cleanup guards、desktop window detector subprocess guards、standalone hotword Ollama request timeout/notebook source guard、GUI tray detached process launch guard、GUI recorder MP3 finalize timeout guard、GUI file transcription media subprocess timeout guard、GGUF export HTTP request timeout guard、direct engine ffmpeg decode timeout guard、server GPU boost command timeout guard、server worker shutdown timeout guard 與 Docker Compose resource env pass-through guard）、Web 77 tests/build（含 bounded API response body reads、bounded transcription request timeout/abort handling、API root validation before requests、abortable/StrictMode-safe diagnostics、keyboard-accessible upload、readiness upload-size preflight、drag/drop highlight stability、transcription-time audio replacement lock、stale result suppression after cancel/unmount、late diagnostic/clipboard suppression、recording startup cleanup success/failure、TTS voice lifecycle/callback cleanup、download filename sanitization、bounded settings controls/storage、malformed history/runtime-config filtering）、cleanup + residue check |

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
