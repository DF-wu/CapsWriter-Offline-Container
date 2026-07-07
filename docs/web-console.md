# Web Console

CapsWriter Web Console 是獨立於上游桌面 client 的瀏覽器工作台，位於 [`client/web`](../client/web)。它透過本 fork 的 OpenAI-compatible HTTP API 進行 STT，並使用瀏覽器 Web Speech API 做 TTS 播放。Linux、Windows 與其他桌面系統只要有現代瀏覽器，就能使用同一套語音轉錄與播放介面。

![CapsWriter Web Console data flow](assets/web-console-architecture.svg)

## 功能範圍

| 類別 | 功能 |
|---|---|
| STT | 麥克風錄音、鍵盤可操作的音訊檔選擇、拖放上傳、readiness 上傳大小預檢、音訊預覽播放、OpenAI Whisper-compatible 轉錄 |
| 格式 | `json`、`text`、`verbose_json`、`srt`、`vtt` |
| TTS | 瀏覽器語音播放、voice 選擇、速度與音高調整、暫停/繼續/停止 |
| 工作流 | API health/readiness diagnostics、model list、Bearer token、language hint、prompt 欄位 |
| 留存 | localStorage 歷史、複製、下載匯出 |
| 隔離 | 前端依賴只在 `client/web/node_modules`，build/cache 可用 `npm run clean` 清理 |

## Server 設定

Web Console 從瀏覽器直接呼叫 HTTP API，因此 server 必須啟用 HTTP API，並允許前端 dev origin。

`.env`：

```bash
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=sk-local-dev
CAPSWRITER_HTTP_API_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Docker Compose 需要同時開 port：

```yaml
ports:
  - "127.0.0.1:6016:6016"
  - "127.0.0.1:6017:6017"
```

Compose 的 host publish 預設是 `127.0.0.1`。若要讓 LAN 上其他裝置連到 WebSocket、HTTP API 或 Web Console，才把對應的 `CAPSWRITER_*_PUBLISH_HOST` 改成 `0.0.0.0`，並為 HTTP API 設定 token。

啟動：

```bash
docker compose up -d --force-recreate capswriter-server
python check_http_api.py --host 127.0.0.1 --port 6017 --key sk-local-dev
```

## Frontend 開發

所有命令都在 `client/web` 內執行，不需要全域安裝套件。

```bash
cd client/web
npm install
npm run dev
```

開啟 Vite 顯示的本機 URL，預設是 `http://127.0.0.1:5173`。介面中的 `API root` 填 `http://127.0.0.1:6017`，`API key` 填 `.env` 的 `CAPSWRITER_HTTP_API_KEY`。

若只要在沒有模型的隔離環境測試前端流程，可另開一個 shell：

```bash
cd client/web
npm run mock-api
```

Mock API 會在 `http://127.0.0.1:6017` 提供 `/health`、`/ready`、`/v1/models`、`/v1/audio/transcriptions`。它只回傳固定文字，用來測 UI、CORS、readiness diagnostics、multipart upload、格式解析與匯出；真實 STT 驗證仍需使用 CapsWriter server。

Real-browser smoke test:

```bash
cd client/web
npm run browser-smoke
```

This starts a temporary mock API and Vite server on free local ports, opens the app with `agent-browser`, sets the API root, checks health/readiness diagnostics, uploads a generated WAV, runs transcription, and asserts the transcript textarea. It does not need a model server.
Each `agent-browser` subprocess is bounded by `CAPSWRITER_WEB_BROWSER_AGENT_TIMEOUT_MS`, default `30000` ms. Browser-smoke HTTP startup probes are bounded by `CAPSWRITER_WEB_BROWSER_HTTP_PROBE_TIMEOUT_MS`, default `2000` ms per attempt. Temporary mock API/Vite child processes are first asked to stop, then force-stopped if they do not exit within `CAPSWRITER_WEB_BROWSER_CHILD_SHUTDOWN_TIMEOUT_MS`, default `5000` ms.

## Production build

```bash
cd client/web
npm run build
npm run preview
```

build 產物在 `client/web/dist`，不提交到 Git。若要完成一次乾淨驗證：

```bash
npm run verify
```

`verify` 會依序執行單元測試、production build 與清理腳本；即使 build 失敗也會嘗試清理已產生的暫存輸出。每個內部 `npm run test/build/clean` subprocess 由 `CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT` 限制，預設 `600` 秒；在特別慢的 release 機器可調大：

```bash
CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT=1200 npm run verify
```

## Production Docker

Web Console can be served as a static Nginx container. Runtime configuration is written to `/config.js` when the container starts, so the same image can point to different CapsWriter HTTP API hosts.
Runtime values are escaped before writing `config.js`, so quotes, backslashes, newlines, and carriage returns in deploy-time strings do not break the JavaScript file. The container also sets baseline browser security headers, including `Content-Security-Policy`, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and `Permissions-Policy`. The Compose entry point enables `no-new-privileges` for the static web service.

Build and run only the web service:

```bash
docker compose -f docker-compose.web.yml up -d --build capswriter-web
```

Published image:

```bash
docker run -d --name capswriter-web --restart unless-stopped \
  -p 127.0.0.1:8080:8080 \
  -e CAPSWRITER_WEB_API_BASE=http://localhost:6017 \
  ghcr.io/df-wu/capswriter-offline-web:latest
```

Run it beside the server:

```bash
CAPSWRITER_HTTP_API_ENABLE=true \
CAPSWRITER_HTTP_API_BIND=0.0.0.0 \
CAPSWRITER_HTTP_API_KEY=sk-local-dev \
CAPSWRITER_HTTP_API_CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080 \
CAPSWRITER_WEB_API_KEY=sk-local-dev \
docker compose -f docker-compose.yml -f docker-compose.web.yml up -d --build
```

Open `http://localhost:8080`. The browser calls `CAPSWRITER_WEB_API_BASE`, which defaults to `http://localhost:6017`.

Runtime variables:

| Variable | Default | Description |
|---|---|---|
| `CAPSWRITER_WEB_PUBLISH_HOST` | `127.0.0.1` | Docker Compose host interface for the static web service; use `0.0.0.0` only for deliberate LAN sharing |
| `CAPSWRITER_WEB_PORT` | `8080` | Host port for the static web service |
| `CAPSWRITER_WEB_API_BASE` | `http://localhost:6017` | Default API root shown in the UI; must be absolute `http://` or `https://` |
| `CAPSWRITER_WEB_API_KEY` | _(empty)_ | Optional default token; written to public `/config.js`, so only use for trusted private deployments |
| `CAPSWRITER_WEB_MODEL` | `whisper-1` | OpenAI-compatible model field |
| `CAPSWRITER_WEB_LANGUAGE` | _(empty)_ | Optional language hint; server accepts aliases such as `zh`, `en`, `ja`, `ko`, `yue` |
| `CAPSWRITER_WEB_PROMPT` | _(empty)_ | Optional recognizer context, normalized and capped server-side |
| `CAPSWRITER_WEB_RESPONSE_FORMAT` | `verbose_json` | Default response format |

Health check:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/config.js
```

Container image build can also be included in the root gate:

```bash
python scripts/verify_all.py --docker-build-web
```

Maintainers publish the static image through [`.github/workflows/publish-web-image.yml`](../.github/workflows/publish-web-image.yml). The workflow first runs `python scripts/verify_all.py --docker-build-web`, including the production Nginx image smoke, then builds from [`client/web/Dockerfile`](../client/web/Dockerfile), pushes `ghcr.io/<owner>/capswriter-offline-web:{latest,sha-*}`, and publishes provenance/SBOM attestations.

## 清理策略

```bash
cd client/web
npm run clean
```

清理項目：

| 路徑 | 說明 |
|---|---|
| `dist` | production build 輸出 |
| `coverage` | 測試覆蓋率輸出 |
| `.vite` / `node_modules/.vite` | Vite cache |
| `playwright-report` / `test-results` | 瀏覽器測試輸出 |
| `.tmp` | 臨時檔 |

`node_modules` 是隔離依賴目錄，預設不由 `clean` 移除，避免每次驗證都重新下載。若要完全還原前端環境，可刪除 `client/web/node_modules` 後重新 `npm install`。

## 驗證清單

| 項目 | 命令或動作 | 預期 |
|---|---|---|
| 依賴安全 | `npm install` | `found 0 vulnerabilities` |
| API root validation | `npm run test -- capswriter.test.ts` | Rejects non-HTTP schemes, URL credentials, query strings, and fragments before fetch |
| 單元測試 | `npm run test` | API parsing、OpenAI-style / legacy / bounded non-JSON error parsing、bounded response body reads、invalid JSON diagnostics、bounded/abortable health/readiness/model diagnostics、bounded/abortable transcription request handling、partial readiness display when model listing needs auth、StrictMode-safe diagnostics mounted guard、keyboard-accessible audio upload、readiness upload-size preflight、drag/drop highlight stability、transcription-time audio replacement lock、stale result suppression after cancel/unmount、stale/late diagnostic result suppression、recording cleanup including delayed `getUserMedia` success/failure、download object URL cleanup、download filename sanitization（含 Windows reserved device name）、TTS voice handler/lifecycle cleanup and late callback guards、clipboard copy denial and late result cleanup、blocked localStorage handling、bounded settings controls、bounded/malformed settings/history/runtime-config recovery 與 App render 測試通過 |
| Web verifier timeout | `python -m unittest discover -s scripts/tests -v` | fake npm 測試覆蓋 hung internal npm step 會回 `124`，且仍會嘗試執行 clean |
| Browser smoke | `npm run browser-smoke` | 真實瀏覽器完成 health/readiness、upload、transcribe workflow；`agent-browser` 與臨時 child cleanup 都有 timeout |
| Production build | `npm run build` | Vite 輸出 `dist` |
| 清理 | `npm run clean` | build/cache/test artifacts 被移除 |
| Server 語法 | `python -m compileall fork_server check_http_api.py start_server_docker.py` | 無 syntax error |
| 真實 STT | 上傳 wav/mp3 後按「轉錄」 | 回傳文字與格式化輸出 |
| 真實 TTS | 在 TTS 面板按「播放」 | 瀏覽器播放語音 |

## 架構邊界

- Web Console 不修改上游 `start_client.py` 或桌面 client 行為。
- STT 只透過 `POST /v1/audio/transcriptions` 呼叫 server。
- API root 必須是 absolute `http://` 或 `https://` URL；可帶部署 path prefix 與尾端 `/v1`，但不能帶 username/password、query、fragment 或非 HTTP scheme。無效值會在 health/readiness/model/transcription request 前被拒絕。
- HTTP error 會優先顯示 OpenAI-style `error.message`，並相容舊版 `detail` payload；非 JSON 錯誤 body 會壓成單行且限制長度，避免 proxy HTML 直接塞滿 UI。API response body 會先以 16 MiB 上限讀取，再進入 JSON parsing 或 transcript rendering，避免 misrouted proxy / broken server 導致瀏覽器無界緩衝。Health/readiness/model list 診斷有 10 秒前端 timeout；轉錄 request 有 600 秒前端 timeout，與 server 預設 task timeout 對齊。若只有模型列表失敗（例如缺 API key），已取得的 health/readiness diagnostics 仍會保留在畫面上。若 `/ready` 或 JSON 格式轉錄回應不是合法 JSON，錯誤訊息會包含 HTTP status 與 endpoint，方便定位 proxy 或舊版 server 問題。
- 若 `/ready` 提供有效的 `config.max_upload_mb`，Web Console 會在檔案選擇與送出轉錄前做大小預檢，避免使用者先建立大型預覽或送出必然被 server 以 `413` 拒絕的請求；未取得或不合法的 readiness limit 不會阻斷既有流程，server 仍是最終大小限制執行者。
- 轉錄進行中會鎖定音訊替換入口；使用者需按「取消」或等待完成後再換檔。取消或更換音訊會讓舊請求結果失效，避免 server/proxy 延遲回應覆蓋目前 UI。
- 下載檔名會移除 path separator、控制字元與常見 OS 保留字元，並避開 Windows reserved device name，避免瀏覽器或作業系統把歷史資料中的異常檔名當作路徑或非法名稱處理。
- TTS 目前是 browser-local Web Speech API；不把音訊傳到雲端。
- localStorage 只保存非敏感使用者設定與最近 20 筆轉錄歷史；settings controls 會套用同一組字串長度上限。手動輸入的 API key 只留在目前頁面記憶體中。讀回時會檢查型別、格式與字串長度，過大或 malformed record 會被忽略。若瀏覽器封鎖 storage 或 quota 用完，保存會 best-effort 失敗但不阻斷目前操作。
- 若透過 `CAPSWRITER_WEB_API_KEY` 注入預設 token，該值會出現在公開的 `/config.js`。
- 若 server 有設定 API key，前端只使用 Bearer token header，不使用 cookie。

localStorage 字串邊界：

| 欄位 | 上限 |
|---|---:|
| Runtime/default API root | 2048 chars |
| Runtime/default API key | 4096 chars |
| Model | 128 chars |
| Language hint | 32 chars |
| Prompt | 16384 chars |
| History id | 128 chars |
| History source name | 512 chars |
| History text | 200000 chars |
| History raw payload | 500000 JSON/string chars |

## 常見問題

**瀏覽器顯示 CORS error**

確認 `CAPSWRITER_HTTP_API_CORS_ORIGINS` 包含目前頁面的 origin，例如 `http://127.0.0.1:5173`。`localhost` 與 `127.0.0.1` 是不同 origin，需要同時列出。

**麥克風無法啟動**

確認瀏覽器權限、系統麥克風權限，以及頁面使用 `http://localhost` / `http://127.0.0.1` / HTTPS。一般 HTTP 遠端頁面不能使用 `getUserMedia`。

**TTS 沒有聲音**

Web Speech API 依賴作業系統與瀏覽器提供 voice。先確認系統音量、瀏覽器 voice 清單，以及瀏覽器是否允許頁面播放音訊。

**上傳後 500 並提到 ffmpeg**

server 需要 ffmpeg 解碼非 raw PCM 音訊。Docker image 已包含相關路徑；裸機模式需自行安裝 ffmpeg。
