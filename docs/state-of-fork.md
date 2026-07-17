# Fork 現況（State of Fork）

> **更新時間**：2026-07-17
> **基底**：`origin/master` @ `7d7fac3`（upstream v2.6）
> **工作分支**：`feature/universal-asr-client`

---

## 1. 一句話定位

本 fork 以 upstream ASR 模型與推論算法為基線，僅在已記錄的 engine I/O／logging
接觸點套用安全邊界，並將 CapsWriter-Offline 做成可部署、可測、可發布的離線語音服務：

| 面向 | 交付 |
|---|---|
| Server | Docker 化、env 驅動設定、GPU/CPU fallback、OpenAI Whisper-compatible HTTP API |
| Client | 標準函式庫 no-GUI CLI、瀏覽器 Web Console、Textual TUI、Windows／Linux X11 桌面 GUI |
| Web | React/Vite STT/TTS 工作台、Nginx static image、runtime `/config.js` |
| Verification | repo-level gate、CLI/server/Web unit tests、Docker smoke、Windows PyInstaller build／relocation／雙 EXE import smoke、可選 live STT |
| Release | server/web GHCR publish workflows、Windows tested-ZIP artifact upload |
| Docs | 架構、HTTP API、CLI、Web、Docker、驗證與上游同步文件，含 SVG 架構圖 |

ASR／標點／對齊模型與推論算法仍由 upstream `core/server/engines/*` 提供；該目錄
並非完全 untouched：目前有 10 個 upstream-tracked engine 檔承載 bounded network／
`ffmpeg` I/O 或 task-local privacy logging guard，不改模型權重、數學或輸出語意。

---

## 2. 分支設計現況

| | |
|---|---|
| 修改的 upstream-tracked 檔案 | **59**；精確集合由 `scripts/check_upstream_divergence.py` 驗證，分組如下 |
| Fork 新增主要目錄 | `fork_server/`、`docker/`、`client/cli/`、`client/web/`、`client/tui/`、`docs/`、`.github/workflows/` |
| Hook 策略 | Sidecar 子類化／單點 monkey-patch + 已分組的 protocol、worker、engine safety touchpoints |
| 唯一高漂移點 | [`fork_server/http_api/ws_send_with_http.py`](../fork_server/http_api/ws_send_with_http.py) 內嵌 upstream `ws_send` loop；HTTP unit test 會做 AST source guard，merge upstream 後若失敗需 re-port |
| 預設相容性 | 不設 `CAPSWRITER_HTTP_API_ENABLE=true` 時，server 啟動路徑保持接近 upstream WebSocket 模式 |

詳細架構見 [architecture.md](architecture.md)。上游同步 SOP 見 [upstream-sync-guide.md](upstream-sync-guide.md)。

| Divergence 群組 | 數量 | 原因摘要 |
|---|---:|---|
| Repository、build、release | 7 | Fork metadata/README、universal Windows server packaging、dependency/ignore/release guards |
| LLM role 安全預設 | 3 | 空 key、disabled network role 與相符使用文件 |
| Desktop portability、bounded audio 與 lifecycle | 24 | Windows/X11 backend contract、headless／pure Wayland lazy input imports、platform-aware text injection、portable shortcut callback、bounded audio/WebSocket、media/desktop subprocess timeout 與 cleanup |
| Protocol 與安全錯誤傳遞 | 3 | Optional worker error fields 與 HTTP task privacy/deadline metadata |
| WebSocket ingress 與 transport controls | 2 | Bind preflight、bounded frame/prefetch、worker-queue backpressure 與 active-stream policy |
| Worker/service resource controls | 8 | Privacy、fair scheduling、deadline、安全錯誤、bounded result dispatch/shutdown/GPU command、inference watchdog |
| Engine export I/O | 3 | Remote safetensor request timeout |
| Engine audio decode I/O | 4 | Bounded `ffmpeg` timeout、kill cleanup 與 stderr preview |
| Engine privacy logging | 3 | Prompt／context／token／audio-derived detected-hotword redaction；推論語意不變 |
| Upstream 文件正確性／a11y | 2 | Text-merger 文件對齊與 image alt text |
| **合計** | **59** | 完整路徑與 merge handling 見[架構](architecture.md)與[上游同步指南](upstream-sync-guide.md) |

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
- HTTP 音訊解碼以 server task timeout 限制 ffmpeg，且 timeout、client cancellation 或 repeated cancellation 都會先完成 bounded kill／reap cleanup。
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

- [`docker/server/Dockerfile`](../docker/server/Dockerfile) 建 `linux/amd64` server image；目前沒有 ARM64 dependency／native runtime／image gate。
- [`docker/server/download_models.py`](../docker/server/download_models.py) 依 `CAPSWRITER_MODEL_TYPE` 自動下載模型，archive 下載有 idle timeout，先寫 `.part` 再 atomic replace。模型 ZIP 會拒絕 traversal、link、加密或 special-file members；llama.cpp tar 只接受留在 archive 內、無循環且最終指向普通檔案的 link chain，並把驗證過的 alias materialize 成普通檔案，不在磁碟建立 archive link。
- llama.cpp CPU／Vulkan readiness 會對照由官方 b7798 archive 實際輸出固定而成的 pinned manifest；不信任 ready marker 自述的 library hashes，marker 也必須與該官方 manifest 完全一致。
- [`.dockerignore`](../.dockerignore) 排除本機 `models/`、Web/CLI generated outputs、`.env*`、package-registry／cloud／SSH credential files、key/cert files 與 local archives，避免 server image build context 帶入本機模型或 secrets。
- [`docker/server/entrypoint.sh`](../docker/server/entrypoint.sh) 做 GPU/CPU backend 選擇與 fallback。
- [`docker/server/probe_backend.py`](../docker/server/probe_backend.py) 以 bounded supervisor 執行 native backend probe；POSIX timeout 會 kill 隔離的 process group，並 bounded wait／reap probe child。
- [`docker/server/healthcheck.py`](../docker/server/healthcheck.py) 一律檢查 WebSocket port；若啟用 HTTP API，還要求 `/ready` 回 `status="ok"`。
- [`docker-compose.yml`](../docker-compose.yml) 啟動 CPU-safe server 並以 named volume 保存模型；[`docker-compose.gpu.yml`](../docker-compose.gpu.yml) 明確要求 NVIDIA runtime、[`docker-compose.igpu.yml`](../docker-compose.igpu.yml) 掛載 Intel／AMD `/dev/dri`、[`docker-compose.models-bind.yml`](../docker-compose.models-bind.yml) 才把 operator-managed `./models` bind-mount 進 container，另可用 [`docker-compose.fun-asr.yml`](../docker-compose.fun-asr.yml) 切低延遲 Fun-ASR。

### 3.3 No-GUI CLI

- [`client/cli/capswriter_cli.py`](../client/cli/capswriter_cli.py) 無第三方 Python dependency。
- 支援 `health`、`ready`、`models`、`transcribe`、`speak`。
- CLI 與診斷工具預設 timeout 對齊 server `CAPSWRITER_HTTP_API_TASK_TIMEOUT=600`，長音訊可用 `--timeout` 覆寫；CLI 另以 `--max-response-mb` 限制 HTTP response body 記憶體上限。
- `speak` 支援直接文字、UTF-8 檔案與 stdin，可串接 `transcribe --format text | speak --stdin`，local TTS 子程序預設以 `--tts-timeout 120` 設上限。
- `transcribe --output` / `--output-dir` 以 same-directory temp file + atomic replace 寫入 transcript；`--output-dir` batch 每個成功項目會立即落盤，replace 失敗時保留舊檔並清掉暫存檔。
- `transcribe --output-dir` 會 sanitize generated filename stems（Windows invalid chars / reserved device names / hidden-dot forms / overly long stems），並在送出 HTTP 前拒絕重複生成的輸出路徑，避免 batch 檔案被靜默覆寫。
- `--base-url` 僅接受 absolute `http://` / `https://` root（可帶 path prefix 與尾端 `/v1`）；URL credentials、query、fragment 與非 HTTP scheme 會在送 request 前被拒絕。
- 所有 request 都不讀取 ambient proxy、也不跟隨 redirect，避免把 Bearer key 或私密音訊轉送到 proxy／redirect target。
- HTTP error 會解析 OpenAI-style `error.message`、舊版 `detail`、bounded non-JSON body previews，將 control character 壓成安全單行、遮蔽回應中反射的當次 key，並把 expected JSON endpoint 的 invalid JSON response 轉成帶 endpoint/status 的診斷。
- Linux TTS：`spd-say` / `espeak-ng` / `espeak`；Windows TTS：PowerShell `System.Speech`。
- 測試使用 in-process mock HTTP server，不需要模型。

### 3.4 Web Console

- [`client/web`](../client/web/) 是 React/Vite app。
- 支援錄音、鍵盤可操作的檔案選擇、拖放上傳、根據 `/ready.config.max_upload_mb` 做上傳大小預檢、播放、STT、五種輸出格式、可 abort 的 HTTP readiness diagnostics、歷史紀錄、下載、browser Web Speech TTS；錄音使用 `idle → starting → recording → stopping` lifecycle，permission pending／shutdown 期間不會重入、換檔或轉錄。轉錄中也會鎖定音訊替換；取消/換檔/卸載後會忽略 stale result，卸載後也會忽略晚到的 diagnostic、clipboard、voice loading、speech callback 與 media stream，且 dev StrictMode 不會誤判已卸載而丟棄 diagnostics。
- API client 會以 16 MiB 上限讀取 response body，解析 OpenAI-style `error.message`、舊版 `detail`、bounded non-JSON HTTP error previews，將 control character 壓成安全單行並遮蔽反射的當次 key，再把 invalid JSON response 轉成帶 endpoint/status 的診斷；health/readiness/model diagnostics 有 10 秒 timeout，transcription request 有 server-aligned 600 秒 timeout，兩者都保留 caller abort，且 deadline 會持續涵蓋完整 response-body consumption（不只等到 headers），所有 fetch 也以 `redirect: "error"` 禁止重送 key 或私密音訊。
- API root 會在 health/readiness/model/transcription request 前驗證，只允許 absolute HTTP(S) root，並拒絕 URL credentials、query、fragment 與非 HTTP scheme。
- 下載檔名會 sanitize path separator、控制字元、OS 保留字元與 Windows reserved device name；歷史資料即使被手動污染也不會直接成為原始 download filename。
- Web settings controls 與 persisted settings/history 會做型別、格式與字串長度邊界檢查；手動污染或過大的 localStorage record 不會直接餵回 UI。最近 20 筆 transcript／raw history 仍是 browser localStorage 內的明文敏感資料；清除全部歷史前會要求不可復原確認。Static container 只有在明確設定 `CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true` 時才會把預設 API key 寫入公開 `/config.js`。
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
- GUI file transcription 的 [`core/client/transcribe/media_tool.py`](../core/client/transcribe/media_tool.py) `ffprobe` duration probe 與 [`core/client/transcribe/file_transcriber.py`](../core/client/transcribe/file_transcriber.py) `ffmpeg` streaming subprocess 以 `CAPSWRITER_CLIENT_MEDIA_TIMEOUT` 限制；即使 repeated cancellation，仍會先完成 bounded kill／reap cleanup，再傳遞取消。
- GUI mic recorder 的 cancellation cleanup 在 repeated `Task.cancel()` 下仍會完成 WebSocket close、queue release、audio exactly-once finalize 與 stale registration clear；即使 first in-flight send 只有 final marker，取消也會關閉連線作為 protocol-level cancellation。
- Qwen / force-aligner / Fun-ASR GGUF export utilities 的 remote safetensor `GET`/`HEAD` requests 以 `CAPSWRITER_GGUF_EXPORT_HTTP_TIMEOUT` 限制，避免轉檔工具被未回應的 Hugging Face endpoint 卡住。
- Qwen / force-aligner / SenseVoice / Fun-ASR direct file-decode helpers 的 `ffmpeg` subprocess 以 `CAPSWRITER_ENGINE_FFMPEG_TIMEOUT` 限制，timeout 後會嘗試 bounded kill cleanup，stderr diagnostic 也會截斷。
- [`core/server/worker/gpu_boost.py`](../core/server/worker/gpu_boost.py) 的 GPU boost/unboost shell command 以 `CAPSWRITER_GPU_BOOST_TIMEOUT` 限制；timeout 會清掉整個 POSIX process group／Windows process tree 並回收 shell，nonzero exit 也不會誤標為成功。
- [`core/server/worker/process_manager.py`](../core/server/worker/process_manager.py) 以 `CAPSWRITER_SERVER_MODEL_LOAD_TIMEOUT` 限制 startup model load，超時先回收 child 再同步讓啟動失敗；worker shutdown 以 `CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT` 限制 graceful join，之後會 bounded terminate wait 並在可用時 kill；parent inference watchdog 另以 HTTP deadline 與 `CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT` 偵測 native hang，回收 child 後讓整個 server fail-stop。
- HTTP/WebSocket listener supervisor 在任一 service 結束或 server stop 後會 bounded 清理兩者；repeated cancellation 也不會取消獨立 cleanup task 或留下 listener。
- Standalone hotword `.py` 與 notebook demo 都以 `CAPSWRITER_OLLAMA_CHAT_TIMEOUT` 限制 local Ollama request。
- GUI tray restart 與 hotword-file default opener subprocesses 使用 detached stdio/session launch，避免使用者動作或重啟 child 繼承 console/verification handles。
- [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) 以 pinned `ubuntu-24.04` runner 跑 root gate，且 checkout 不保留 git credentials。
- [`.github/workflows/publish-server-image.yml`](../.github/workflows/publish-server-image.yml) 先跑 `python scripts/verify_all.py --skip-web`，通過後才 push 完整 commit SHA 的 immutable server tag 與 provenance/SBOM；獨立 10 分鐘 promotion job 會重查目前 `master` tip，再以已驗證 digest 更新並回讀 `latest`。
- [`.github/workflows/publish-web-image.yml`](../.github/workflows/publish-web-image.yml) 對 Web Docker smoke 使用同一 release gate／immutable-tag／guarded-digest-promotion 流程。兩條 workflow 都以固定 per-ref group 序列化、設定 job deadline、使用 pinned runner/actions、non-persistent checkout credentials，且只有 registry mutation jobs 取得 `packages: write`。

---

## 4. 已跑驗證

本分支已跑過以下 gate，且執行後確認無 build/cache/Docker 殘留：

| Gate | 結果 |
|---|---|
| `python scripts/verify_all.py --web-browser-smoke --docker-build-web` | 通過：59-file upstream divergence、46-file docs check、dependency-light repo suites、Web unit/build、真實瀏覽器 upload/transcribe、production Nginx image／security-header smoke，以及 cleanup residue check |
| `python -m unittest discover -s client/cli/tests -v` | 通過：CLI **63/63**；含 single wall-clock request deadline／slow-drip rejection、direct/no-proxy transport、redirect rejection、跨 preview boundary reflected-key redaction、streamed multipart／Windows early-response abort recovery、bounded response、atomic/batch output 與 Linux／Windows TTS contract |
| `python -m unittest discover -s docker/server/tests -v` | 通過：Docker helper **63/63**；含 secure model bootstrap、warm read-only fast path、archive bounds、healthcheck、entrypoint，以及完整 env 必須先套用到 backend probe 的 regression |
| pinned Python 3.12：`python -m unittest discover -s fork_server/http_api/tests -v` | 通過：完整 HTTP API **153/153**，zero skips |
| pinned Python 3.12：`python scripts/verify_api_contract.py` | 通過：嚴格完整 HTTP API **153/153**；會驗證 exact installed pins、imports、lock parity、全目錄非零 discovery，且任何 skip 都使 gate 失敗 |
| `python -m unittest discover -s scripts/tests -v` | 通過：Verifier／diagnostic **301 tests**，其中 **16** 個為預期的 dependency-only skips；同 16 個 `scripts.tests.test_server_queue_limits` 已在 pinned environment **16/16**、zero skips，覆蓋 bounded queue、WebSocket ingress／audio ceiling 與 result dispatch isolation |
| `cd client/web && npm run verify && npm run browser-smoke` | 通過：Web **103/103**、TypeScript／Vite production build、真實瀏覽器 health/readiness/upload/transcribe；含 recorder overlap／delayed-stop lifecycle 與 transcript accessible-name regressions，root gate另驗證 isolated Nginx image 與安全 headers |
| `scripts/verify_tui.py`（hash-locked isolated Python 3.10 與 3.12） | 兩個 interpreter 各通過 TUI **63/63**，zero skips；含 single wall-clock request deadline／slow-drip rejection，並驗證 direct-pin/lock parity、installed versions、imports 與 `pip check` |
| `python -m unittest scripts.tests.test_compose_config scripts.tests.test_docs scripts.tests.test_check_docs -v` + `python scripts/check_docs.py` | Compose **15/15**、docs unit **15/15**、Markdown checker **46 files**；real `docker compose config --quiet` 另通過 base、NVIDIA、iGPU、model bind、Fun-ASR、Web 與 combined overrides |
| isolated real Qwen CPU bootstrap／warm／probe | 真實 Qwen 1.7B model files 以 read-only mount 使用；llama.cpp CPU archive 完整下載、hash 驗證、atomic 三目錄 promotion，第二次執行命中不建立 lock 的 warm fast path，修正後 backend probe 成功 construct／cleanup engine；未發布 host port、未改 live model tree |

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
