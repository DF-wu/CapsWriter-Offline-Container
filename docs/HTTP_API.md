# OpenAI-Compatible ASR HTTP API

CapsWriter-Offline 在 WebSocket 服務之外，可選擇性提供與 [OpenAI Whisper Audio API](https://platform.openai.com/docs/api-reference/audio) 同形的 HTTP 端點。任何使用 OpenAI SDK 的程式只要把 `base_url` 指向本服務，即可零修改改用本地離線識別。

> **核心承諾**：完整離線、與既有 WebSocket 識別共用同一個模型行程、可丟給任何 OpenAI 相容客戶端使用。

---

## 1. 啟用

預設**關閉**。透過環境變數啟用：

| 變數 | 預設 | 說明 |
|---|---|---|
| `CAPSWRITER_HTTP_API_ENABLE` | `false` | `true` 啟用 HTTP API |
| `CAPSWRITER_HTTP_API_BIND` | `127.0.0.1` | 監聽位址；對外請改 `0.0.0.0` 並一定要設 `KEY` |
| `CAPSWRITER_HTTP_API_PORT` | `6017` | 監聽 port（與 WebSocket port 不同） |
| `CAPSWRITER_HTTP_API_KEY` | _(空)_ | Bearer token；空字串視為不啟用認證 |
| `CAPSWRITER_HTTP_API_KEY_FILE` | _(空)_ | 伺服器端 Bearer token 檔案；適合 Docker secrets / service manager，明確 `KEY` 優先 |
| `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND` | `false` | 允許非 loopback bind 在無 KEY 下啟動；只適合受信任測試網路 |
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `100` | 單次上傳上限（MB） |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `600` | 單次轉錄超時（秒）；ffmpeg 解碼與等待識別共用 |
| `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` | `2` | 同時允許進入上傳/解碼/等待識別的 HTTP 轉錄請求數 |
| `CAPSWRITER_HTTP_API_CORS_ORIGINS` | _(空)_ | 逗號分隔的瀏覽器 origin allowlist；空字串表示不加 CORS middleware |
| `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS` | `false` | 預設不把 prompt/context 或轉錄文字寫入 server log/console；設 `true` 才保留全文診斷輸出 |

HTTP API env 會在 server 啟動時驗證；明確設定的無效值會讓啟動失敗，而不是靜默退回預設值。
當 `CAPSWRITER_HTTP_API_ENABLE=true` 且 `BIND` 不是 loopback（例如 `0.0.0.0`、`::`、LAN IP 或 hostname）時，必須設定 `CAPSWRITER_HTTP_API_KEY` 或 `CAPSWRITER_HTTP_API_KEY_FILE`；若確定只在受信任測試網路使用，才可設 `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true` 明確覆寫。

| 變數 | 驗證規則 |
|---|---|
| `CAPSWRITER_HTTP_API_ENABLE` | `true/false`、`yes/no`、`on/off`、`1/0` |
| `CAPSWRITER_HTTP_API_PORT` | `1..65535` |
| `CAPSWRITER_HTTP_API_KEY_FILE` | UTF-8 檔案必須可讀且內容去除空白後不可為空 |
| `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND` | `true/false`、`yes/no`、`on/off`、`1/0` |
| `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS` | `true/false`、`yes/no`、`on/off`、`1/0` |
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `>= 1` |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `>= 1` 秒 |
| `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` | `>= 1` |
| `CAPSWRITER_HTTP_API_CORS_ORIGINS` | `http://` 或 `https://` origin，不含 path/query；`*` 允許但只建議本機測試 |

### 1.1 Docker

[`docker-compose.yml`](../docker-compose.yml) 已有預留註解段。最小啟用：

```yaml
environment:
  CAPSWRITER_HTTP_API_ENABLE: "true"
  CAPSWRITER_HTTP_API_BIND: 0.0.0.0
  CAPSWRITER_HTTP_API_PORT: "6017"
  CAPSWRITER_HTTP_API_KEY: "sk-your-token"   # 對外時必填
  # 或用 secret file:
  # CAPSWRITER_HTTP_API_KEY_FILE: "/run/secrets/capswriter-http.key"
  CAPSWRITER_HTTP_API_CORS_ORIGINS: "http://127.0.0.1:5173"
ports:
  - "6017:6017"
```

```bash
docker compose up -d --force-recreate capswriter-server
docker compose logs -f capswriter-server | grep "HTTP API 監聽"
```

### 1.2 裸機

```bash
export CAPSWRITER_HTTP_API_ENABLE=true
python start_server_docker.py
```

啟動成功後 log 出現：

```
HTTP API 監聽 127.0.0.1:6017 (auth=off)
```

---

## 2. 端點

### 2.1 `POST /v1/audio/transcriptions`

OpenAI Whisper 規格的 multipart 端點。

**Form fields**：

| 欄位 | 必填 | 預設 | 說明 |
|---|---|---|---|
| `file` | ✅ | — | 音訊檔；ffmpeg 能解的格式都行（mp3/wav/m4a/flac/ogg/webm/...） |
| `model` | — | `whisper-1` | OpenAI 相容占位，**實際模型由 `CAPSWRITER_MODEL_TYPE` 決定** |
| `language` | — | `auto` | 語言提示；接受 `chinese`/`english` 等統一名稱，也接受常見 OpenAI-style alias 如 `zh`、`en`、`ja`、`ko` |
| `prompt` | — | _(無)_ | 轉為 upstream `Task.context` 傳給 recognizer；會正規化換行並截斷到前 2048 字元 |
| `response_format` | — | `json` | 五選一：`json`/`text`/`verbose_json`/`srt`/`vtt` |
| `temperature` | — | `0.0` | OpenAI 相容占位 |

上傳檔案會以 1 MiB chunk 讀取並即時套用 `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB`，超限會立刻回 `413`，避免把整個大檔讀入記憶體。

**Headers**：

```
Authorization: Bearer <CAPSWRITER_HTTP_API_KEY>   # 僅在伺服器有設 KEY 時需要
```

**回應**：

| `response_format` | Content-Type | 範例 |
|---|---|---|
| `json` | `application/json` | `{"text": "你好世界"}` |
| `text` | `text/plain; charset=utf-8` | `你好世界` |
| `srt` | `application/x-subrip; charset=utf-8` | 標準 SRT |
| `vtt` | `text/vtt; charset=utf-8` | 標準 WebVTT |
| `verbose_json` | `application/json` | 含 `duration` / `segments` / `words` |

成功回應會帶 `X-CapsWriter-Task-ID` header，方便把客戶端請求對回 server log。

`verbose_json` 範例：

```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 3.21,
  "text": "你好世界",
  "segments": [
    {"id": 0, "seek": 0, "start": 0.0, "end": 3.21, "text": "你好世界"}
  ],
  "words": [
    {"word": "你", "start": 0.10, "end": 0.32},
    {"word": "好", "start": 0.32, "end": 0.55}
  ]
}
```

**錯誤回應**：

非 `/ready` 端點的 HTTP error 會使用 OpenAI-style JSON envelope，方便 SDK / proxy / Web client 統一處理：

```json
{
  "error": {
    "message": "Missing or invalid Authorization header",
    "type": "authentication_error",
    "param": null,
    "code": null
  }
}
```

| Status | 觸發條件 |
|---|---|
| `400` | 空檔案、檔案無法解碼、音訊過短（< 0.05s） |
| `401` | 已設 `KEY` 但 `Authorization` header 缺失或 token 錯誤 |
| `413` | 上傳超過 `MAX_UPLOAD_MB` |
| `422` | multipart/form 欄位缺失或型別錯誤 |
| `500` | 找不到 `ffmpeg`、識別子進程異常 |
| `501` | `/v1/audio/translations` 明確未實作 |
| `504` | 任務超過 `TASK_TIMEOUT` |

### 2.2 `POST /v1/audio/translations`

明確回傳 `501 Not Implemented`。CapsWriter 本地模型不做語種翻譯。

### 2.3 `GET /health`

```json
{"status": "ok", "model": "qwen_asr", "version": "2.5"}
```

### 2.4 `GET /ready`

部署用 readiness/diagnostic endpoint。它不需要 API key，也不暴露 token；用來確認 HTTP sidecar 已綁定 `task_router` 且系統可找到 `ffmpeg`。

Ready 時回 `200`：

```json
{
  "status": "ok",
  "model": "qwen_asr",
  "version": "2.5",
  "checks": {
    "task_router_bound": true,
    "ffmpeg_available": true
  },
  "config": {
    "auth_enabled": true,
    "max_upload_mb": 100,
    "task_timeout": 600.0,
    "max_concurrent_requests": 2,
    "cors_enabled": true,
    "cors_origins_count": 2
  }
}
```

若必要檢查失敗，回 `503` 且 `status="degraded"`。

### 2.5 `GET /v1/models`

OpenAI SDK 某些初始化路徑會打這個。回應一筆「當前 model」紀錄：

```json
{"object": "list", "data": [{
  "id": "qwen_asr", "object": "model",
  "owned_by": "capswriter-offline", "created": 0
}]}
```

---

## 3. 整合範例

### 3.1 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6017/v1",
    api_key="dummy",
)

with open("meeting.mp3", "rb") as f:
    r = client.audio.transcriptions.create(
        model="whisper-1",
        file=f,
        response_format="verbose_json",
    )

print(r.text)
for seg in r.segments:
    print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")
```

### 3.2 curl

```bash
# 最小：拿 text
curl -X POST http://localhost:6017/v1/audio/transcriptions \
  -F file=@meeting.mp3 \
  -F response_format=text

# verbose JSON + 認證
curl -X POST https://your-host:6017/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-your-token" \
  -F file=@meeting.mp3 \
  -F response_format=verbose_json
```

模型已載入後，可用診斷工具同時檢查 health、models 與三種轉錄格式。若有已知內容的測試音訊，加入 `--expect` 讓檢查失敗於錯誤轉錄：

```bash
python check_http_api.py \
  --host 127.0.0.1 \
  --port 6017 \
  --audio /path/to/known-speech.wav \
  --expect "expected transcript text" \
  --timeout 300
```

If auth is enabled, the diagnostic client can pass `--key-file /run/secrets/capswriter-http.key` or set `CAPSWRITER_HTTP_API_KEY_FILE` so the client-side token is read from a UTF-8 file instead of being placed directly in the shell command.

### 3.3 Node / TypeScript

```ts
import OpenAI from "openai";
import fs from "fs";

const client = new OpenAI({
  baseURL: "http://localhost:6017/v1",
  apiKey: "dummy",
});

const r = await client.audio.transcriptions.create({
  model: "whisper-1",
  file: fs.createReadStream("meeting.mp3"),
  response_format: "srt",
});
console.log(r);
```

---

## 4. 架構

```
┌────────────┐  HTTP POST    ┌────────────────────────────────┐
│ OpenAI SDK │ ────────────► │ FastAPI ([fork_server/http_api│
└────────────┘               │   /api.py](../fork_server/http_│
                             │   api/api.py))                  │
                             │  ├─ ffmpeg → PCM                │
                             │  ├─ task_router.register        │
                             │  └─ split & enqueue Task        │
┌────────────┐  WebSocket    │           │                     │
│  WS client │ ────────────► │  ws_recv  ▼                     │
└────────────┘               │     ┌──────────┐                │
                             │     │ queue_in │                │
                             │     └────┬─────┘                │
                             │          ▼                      │
                             │  ┌─────────────────┐            │
                             │  │ recognizer 子   │            │
                             │  │ 进程（單一）    │            │
                             │  └────┬────────────┘            │
                             │       ▼                         │
                             │     ┌──────────┐                │
                             │     │queue_out │                │
                             │     └────┬─────┘                │
                             │          ▼                      │
                             │  [fork_server/http_api/         │
                             │   ws_send_with_http.py]         │
                             │   ├─ try_resolve(HTTP future)   │
                             │   └─ WebSocket broadcast        │
                             └─────────────────────────────────┘
```

### 4.1 共用 recognizer

整個服務只有**一個** recognizer 子進程。WebSocket 與 HTTP 任務都丟同一個 `state.queue_in`。模型只載入一次。

### 4.2 結果回流

[`fork_server/http_api/ws_send_with_http.py`](../fork_server/http_api/ws_send_with_http.py) 從 `state.queue_out` 拉結果時：

1. 先讓 `task_router.try_resolve(result)` 攔截 HTTP 任務（中間/最終結果都會被吸收）。
2. 若不是 HTTP 任務，走原本的 WebSocket 派發路徑（與上游 `core/server/connection/ws_send.py` 邏輯相同）。

### 4.3 合成 socket_id

HTTP 任務使用合成 `socket_id="http:<task_id>"` 並加入 `state.sockets_id`（跨進程 `Manager().list()`）。recognizer 子進程的 TaskHandler 檢查 `task.socket_id not in sockets_id` 來判定上游是否還在；合成的 socket_id 滿足這個檢查，讓 HTTP 任務不會被丟棄。

HTTP request 成功、timeout、server error 或客戶端取消時，都會清理 pending future 與合成 socket id，避免中斷請求留下不可回收的路由狀態。

### 4.4 並發

- **識別嚴格串行**：recognizer 一次處理一個 Task，多個 HTTP 請求 = 自然 FIFO backpressure。
- **HTTP 請求有上限**：`CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` 限制同時進入上傳、解碼與等待識別的 request 數，避免大量請求同時佔用記憶體與 ffmpeg subprocess。
- **解碼/格式化並行**：ffmpeg 解碼與 OpenAI 格式化在 asyncio loop 內可重疊。
- **想高並發/低延遲 SLA**：在 HTTP 前加佇列或 LB 並橫向擴充 server 實例；不要在單一進程內多開 recognizer。

---

## 5. 推薦的模型搭配

HTTP API 偏 batch/互動式場景，推薦切到 `fun_asr_nano` 降延遲：

```bash
CAPSWRITER_HTTP_API_ENABLE=true \
docker compose -f docker-compose.yml -f docker-compose.fun-asr.yml \
  up -d --force-recreate
```

| 模型 | 平均單次延遲（5s 音訊） | 適用 |
|---|---|---|
| `qwen_asr`（預設） | ~2-4s | 高精度長段轉錄、字幕 |
| `fun_asr_nano` | ~0.5-1.5s | 互動聊天、whisper SDK 即時呼叫 |

切換時客戶端**不需要重啟**，只需重啟 server。

---

## 6. 安全建議

| 場景 | 建議 |
|---|---|
| 僅本機用 | 預設即可（`BIND=127.0.0.1`、無 KEY） |
| 同 LAN 共享 | 設 `KEY`、`BIND=0.0.0.0`、防火牆限制來源 IP |
| 對外公網 | 設 `KEY`、reverse proxy 加 TLS、限制 IP、設低 `MAX_UPLOAD_MB` 與 `TASK_TIMEOUT` |

KEY 為純 Bearer token，比對時使用 constant-time compare；header 必須剛好是 `Bearer <token>` 兩段，`Bearer` scheme 大小寫不敏感。建議使用 ≥ 32 字元隨機字串：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

HTTP API 預設只記錄 task id、時延、音訊大小、格式、語言與文字長度，不記錄 prompt/context 或轉錄內容。若需要在單人本機 debug 時看全文，才設定 `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS=true`；共享主機、LAN 或公網部署請維持預設 `false`。

---

## 7. 已知限制

| 限制 | 原因 | 影響 |
|---|---|---|
| `model` / `temperature` 為占位 | 由 `CAPSWRITER_MODEL_TYPE` 決定模型；無 sampling | OpenAI SDK 寫什麼都不影響 |
| 語言提示依 backend 支援度生效 | Qwen / FunASR / SenseVoice / Paraformer 支援的語言集合不同 | 不支援的語言會落回該 backend 的預設行為 |
| 無 streaming（SSE） | OpenAI Whisper API 本身亦無 streaming | 需 streaming 請用 WebSocket |
| `/v1/audio/translations` 永不實作 | 本地模型不做翻譯 | 客戶端如用 `translate()`，請改 `transcribe()` |

---

## 8. 故障排除

| Symptom | 原因 / 解法 |
|---|---|
| 啟動時 `CAPSWRITER_HTTP_API_KEY or CAPSWRITER_HTTP_API_KEY_FILE is required...BIND is not loopback` | HTTP API 啟用且 bind 到非 loopback；設定 `CAPSWRITER_HTTP_API_KEY` 或 `CAPSWRITER_HTTP_API_KEY_FILE`，或只在受信任測試網路設 `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true` |
| 啟動時 `CAPSWRITER_HTTP_API_KEY_FILE could not be read` | key file 路徑不存在或 container/service 沒有讀取權限；確認 secret mount 與檔案權限 |
| 啟動時 `HTTP API 已啟用但系統找不到 ffmpeg` | Docker image 應已內建；裸機請 `apt install ffmpeg` |
| `500 Server misconfigured: ffmpeg not found` | 同上 |
| `400 Audio decode failed` | 不是音訊檔、編碼損壞，或 ffmpeg 解碼超過 `TASK_TIMEOUT`；錯誤內容會截斷避免巨大 response/log，本機可用 `ffmpeg -i <file>` 看完整細節 |
| `503 /ready degraded` | `ffmpeg` 不在 PATH 或 HTTP router 尚未綁定；看 `checks` 欄位 |
| `504 Recognition timeout` | 已完成解碼但 recognizer 等待超過 `TASK_TIMEOUT`；對長音訊調高 timeout，對 CPU 部署考慮 `fun_asr_nano` 或啟用 GPU |
| `413 File too large` | 調高 `MAX_UPLOAD_MB` 或客戶端側分片 |
| `401 Missing or invalid Authorization` | header 必須是 `Bearer <token>` 格式（scheme 大小寫不敏感）；token 對應 `CAPSWRITER_HTTP_API_KEY` |

---

## 9. 程式檔案地圖

| 檔案 | 職責 |
|---|---|
| [`fork_server/http_api/api.py`](../fork_server/http_api/api.py) | FastAPI app + 端點實作 |
| [`fork_server/http_api/auth.py`](../fork_server/http_api/auth.py) | Bearer token parsing and constant-time comparison |
| [`fork_server/http_api/errors.py`](../fork_server/http_api/errors.py) | OpenAI-style JSON error payload handlers |
| [`fork_server/http_api/privacy.py`](../fork_server/http_api/privacy.py) | prompt/context 與轉錄文字的 redacted/opt-in logging helper |
| [`fork_server/http_api/readiness.py`](../fork_server/http_api/readiness.py) | `/ready` payload/status helper |
| [`fork_server/http_api/task_router.py`](../fork_server/http_api/task_router.py) | task_id ↔ asyncio.Future 路由；合成 socket_id 管理 |
| [`fork_server/http_api/transcription_tasks.py`](../fork_server/http_api/transcription_tasks.py) | HTTP audio segmentation, language alias normalization, prompt/context normalization |
| [`fork_server/http_api/audio_decoder.py`](../fork_server/http_api/audio_decoder.py) | FFmpeg subprocess → 16k/f32/mono PCM |
| [`fork_server/http_api/openai_formatter.py`](../fork_server/http_api/openai_formatter.py) | 5 種 `response_format` 輸出 |
| [`fork_server/http_api/ws_send_with_http.py`](../fork_server/http_api/ws_send_with_http.py) | 結果分派；HTTP 攔截點 `try_resolve` |
| [`fork_server/http_api/serve.py`](../fork_server/http_api/serve.py) | uvicorn cotask 啟動 |
| [`fork_server/env_config.py`](../fork_server/env_config.py) | `CAPSWRITER_HTTP_API_*` 等環境變數綁定 |
| [`fork_server/bootstrap.py`](../fork_server/bootstrap.py) | `ForkedCapsWriterServer.start()` 並行 ws_send + http_serve |

---

## 10. 相容性承諾

- 端點與回應格式對齊 OpenAI Whisper API 規格；行為偏差皆列於 §7
- 環境變數命名前綴 `CAPSWRITER_HTTP_API_` 為穩定承諾，不更名
- 預設 disable 是穩定承諾；未啟用 HTTP API 時 server 行為與上游完全一致
