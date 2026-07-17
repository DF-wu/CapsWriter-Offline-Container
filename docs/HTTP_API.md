# v1 Server：OpenAI 相容 ASR HTTP API

> [English](en/http-api.md) · 繁體中文 · [專案 README](../readme.md)

CapsWriter-Offline v1 除了 WebSocket 服務，也能選擇性啟用一個與
[OpenAI Whisper Audio API](https://platform.openai.com/docs/api-reference/audio)
同形的檔案轉錄端點。這個端點屬於 **ASR Server**、在 Server process 內執行，
並與 WebSocket 共用 model 與 recognizer queue；預設不啟用。

它只實作本文件明列的 OpenAI 相容 subset，不是完整 OpenAI Audio API，也不是
另一個 desktop Client。

---

## 1. 先分清楚 Server 與 Client

| 角色 | Protocol | 責任 |
|---|---|---|
| **v1 ASR Server** | WebSocket `6016`；選用 HTTP `6017` | 載入 FFmpeg 與 model、接收 audio、執行 inference、回傳逐字稿 |
| **Legacy Windows Desktop Client**（`start_client.py`） | 只使用 WebSocket `6016` | 負責 mic、tray、hotkey、clipboard 與 text injection；不會 host 或呼叫這個 HTTP API |
| **外部 API caller** | HTTP `6017` | 另一個 curl、OpenAI SDK 或相容應用程式，把 audio file 上傳給 Server |

HTTP 與 WebSocket 共用一個 Server recognizer process。Model memory 不會載入兩次，
但工作會進入同一條 serial queue；大量 HTTP request 可能增加 Desktop Client 的延遲，
反之亦然。

v1 Release 只保留 `start_client.py` 的 compatibility Client **source**；它走
WebSocket，不是 HTTP API wrapper。目前的 v1 source-only Release 不附 Windows
executable。

## 2. 為什麼存在這個 API

- 多數 ASR 應用都已實作 OpenAI SDK 對接，提供相容端點就能直接接管它們的流量。
- 不重新發明 protocol；不為 HTTP 額外載入模型。
- 既有 WebSocket 服務不受影響，opt-in 即可。

---

## 3. 啟用與網路邊界

服務預設**關閉** HTTP API。Python Server 的 native default 與 Compose template
刻意不同：

| 設定 | Native Server default | v1 Compose template | 用途 |
|---|---|---|---|
| `CAPSWRITER_HTTP_API_ENABLE` | `false` | `false` | 設為 `true` 才啟動 HTTP Server |
| `CAPSWRITER_HTTP_API_BIND` | `127.0.0.1` | Container 內 `0.0.0.0` | Python HTTP Server 的 listen address |
| `CAPSWRITER_HTTP_API_HOST_BIND` | 不使用 | Host 上 `127.0.0.1` | 僅供 Compose 控制 container port 發布位址 |
| `CAPSWRITER_HTTP_API_PORT` | `6017` | `6017` | HTTP listen／publish port |
| `CAPSWRITER_HTTP_API_KEY` | 空字串 | 空字串 | Bearer token；空字串會停用 authentication |
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `100` | `100` | 單次 upload 上限（MiB） |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `600` | `600` | 單次 recognition timeout（秒） |

Native default 只監聽 loopback。Docker process 必須在 **container 內** bind
`0.0.0.0`，Docker 才能 forward；Compose 再用
`CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1` 把 host publish 限制在 loopback。這兩個
address 保護不同的網路邊界，不可混為一談。

Compose 即使在 API disabled 時仍宣告 `6017` port mapping；只有
`CAPSWRITER_HTTP_API_ENABLE=true` 才會真正啟動 HTTP listener。

### 3.1 Docker Compose：只供本機使用

[`docker-compose.yml`](../docker-compose.yml) 已傳入 opt-in 設定，並預設只在 host
loopback 發布 HTTP port。最小啟用方式是在 `.env` 設定：

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=sk-your-token
```

Container 內必須 bind `0.0.0.0` 才能接受 Docker forwarding；host 端仍由
`CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1` 限制在 loopback。只有在前方已有可信任、
具 authentication 與 TLS 的 reverse proxy 時，才應放寬 host bind。

```bash
docker compose up -d --force-recreate capswriter-server
docker compose logs -f capswriter-server | grep "HTTP API 监听"
```

### 3.2 Native Server：只供本機使用

```bash
export CAPSWRITER_HTTP_API_ENABLE=true
export CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
python core_server.py
```

啟動後 log 會出現：

```
HTTP API 监听 127.0.0.1:6017 (auth=on)
```

若沒有設定 `CAPSWRITER_HTTP_API_KEY`，才會顯示 `auth=off`；無 key 只適合受 loopback
限制的本機使用。

`CAPSWRITER_HTTP_API_HOST_BIND` 只供 Compose 使用，native Server 不會讀取。

### 3.3 LAN 或 remote access

不要把未加密、未驗證的 raw HTTP listener 直接暴露到 LAN 或 Internet。任何
non-loopback deployment 都必須同時做到：

1. 設定非空、足夠長且隨機的 `CAPSWRITER_HTTP_API_KEY`。
2. 由可信任的 reverse proxy terminate TLS，只透過 HTTPS 傳送 Bearer token。
3. Reverse proxy 與 Server 同 host 時，保留
   `CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1`。若 proxy 必須從另一個 interface
   存取，盡量把 `CAPSWRITER_HTTP_API_HOST_BIND` 設為該 private interface 的精確
   IP，並加上 firewall allowlist，不要無條件使用 `0.0.0.0`。
4. Compose 內保留 `CAPSWRITER_HTTP_API_BIND=0.0.0.0`；native Server 則只 bind
   reverse proxy 實際需要的 interface。
5. 在 reverse proxy 另外設定 body size、timeout 與 rate limit。

內建 Bearer token 只提供 authentication，不提供 encryption；Server 本身不包含
TLS 或 rate limiter。`GET /health`、`GET /v1/models` 與固定回傳 `501` 的
translations endpoint 不受 transcription token 保護，需要時應由 proxy 限制。

---

## 4. API 端點

### 4.1 `POST /v1/audio/transcriptions`

OpenAI Whisper 規格的多模態 multipart 端點。

**Form fields：**

| 欄位 | 必填 | 預設 | 說明 |
|---|---|---|---|
| `file` | ✅ | — | 音訊檔案；任何 ffmpeg 能解的格式都可（mp3/wav/m4a/flac/ogg/webm/...） |
| `model` | — | `whisper-1` | OpenAI 相容占位，**實際模型由 `CAPSWRITER_MODEL_TYPE` 決定**，本欄位被忽略 |
| `language` | — | _(無)_ | 不影響識別，僅在 `verbose_json` 回填 |
| `prompt` | — | _(無)_ | 目前僅 log，**未注入** recognizer context（見 §7 已知限制） |
| `response_format` | — | `json` | 五選一：`json` / `text` / `verbose_json` / `srt` / `vtt` |
| `temperature` | — | `0.0` | OpenAI 相容占位，被忽略 |

**Headers：**

```
Authorization: Bearer <CAPSWRITER_HTTP_API_KEY>   # 僅在伺服器有設 KEY 時需要
```

**回應格式：**

| `response_format` | 回應 Content-Type | 範例 |
|---|---|---|
| `json` | `application/json` | `{"text": "你好世界"}` |
| `text` | `text/plain; charset=utf-8` | `你好世界` |
| `srt` | `application/x-subrip; charset=utf-8` | 標準 SRT |
| `vtt` | `text/vtt; charset=utf-8` | 標準 WebVTT |
| `verbose_json` | `application/json` | 含 `duration` / `segments` / `words` 的完整結構 |

**`verbose_json` 範例：**

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
    {"word": "好", "start": 0.32, "end": 0.55},
    {"word": "世", "start": 1.20, "end": 1.45},
    {"word": "界", "start": 1.45, "end": 1.80}
  ]
}
```

`verbose_json` 一律包含 `segments` 與 `words` 欄位，但內容取決於 configured model
是否提供對齊的 token／timestamp；資訊不足時 array 可能為空。SRT／VTT 若無法取得
對齊資料，會退化為涵蓋完整 duration 的單一 segment。不要假設其切段結果與 OpenAI
Whisper 完全相同。

**錯誤回應：**

| Status | 觸發條件 |
|---|---|
| `400` | 空檔案、檔案無法解碼（格式損壞、ffmpeg 解碼失敗）、音訊過短（< 0.05s） |
| `401` | 已設 `KEY` 但 `Authorization` header 缺失或 token 錯誤 |
| `413` | 上傳超過 `MAX_UPLOAD_MB` |
| `415` | Transcription request 不是 `multipart/form-data` |
| `422` | Multipart field 或 value 未通過 request validation |
| `500` | 伺服器找不到 `ffmpeg`；或識別子進程異常 |
| `504` | 任務超過 `TASK_TIMEOUT` |

### 4.2 `POST /v1/audio/translations`

明確回傳 `501 Not Implemented`。CapsWriter 的本地模型不做語種翻譯。

### 4.3 `GET /health`

```json
{"status": "ok", "model": "qwen_asr", "version": "2.5-alpha"}
```

### 4.4 `GET /v1/models`

OpenAI SDK 在某些初始化路徑會呼叫此端點。回應一筆「本服務當前 model」紀錄：

```json
{"object": "list", "data": [{
  "id": "qwen_asr",
  "object": "model",
  "owned_by": "capswriter-offline",
  "created": 0
}]}
```

---

## 5. 整合範例

### 5.1 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:6017/v1",
    api_key="dummy",     # 若伺服器沒設 KEY 則任意字串都可
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

### 5.2 curl

```bash
# 最小：拿 plain text
curl -X POST http://localhost:6017/v1/audio/transcriptions \
  -F file=@meeting.mp3 \
  -F response_format=text

# 完整：verbose JSON + 認證
curl -X POST https://your-host:6017/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-your-token" \
  -F file=@meeting.mp3 \
  -F response_format=verbose_json
```

### 5.3 Node / TypeScript

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

其他語言只要使用 [OpenAI 官方/社群 SDK](https://platform.openai.com/docs/libraries) 並覆寫 base URL 即可。

---

## 6. 架構與並發模型

```
┌────────────┐  HTTP POST   ┌─────────────────────────┐
│ OpenAI SDK │ ───────────► │ FastAPI app             │
└────────────┘              │  ├─ ffmpeg → PCM        │
                            │  ├─ TaskRouter.register │
                            │  └─ split & enqueue     │
┌────────────┐  WebSocket   │           │             │
│  WS client │ ───────────► │  ws_recv  ▼             │
└────────────┘              │     ┌──────────┐        │
                            │     │ queue_in │ (multiprocessing.Queue)
                            │     └────┬─────┘        │
                            │          ▼              │
                            │  ┌────────────────┐     │
                            │  │ recognizer 子   │     │
                            │  │ 进程 (单例)     │     │
                            │  └────┬───────────┘     │
                            │       ▼                 │
                            │     ┌──────────┐        │
                            │     │queue_out │        │
                            │     └────┬─────┘        │
                            │          ▼              │
                            │     ws_send             │
                            │   ├─ try_resolve ───┐   │
                            │   │   (HTTP future) │   │
                            │   └─ WebSocket send │   │
                            └─────────────────────┴───┘
```

### 6.1 共用 recognizer

整個服務只有**一個** recognizer 子進程，由 [`util/server/service.py`](../util/server/service.py) 啟動。WebSocket 與 HTTP 任務都丟同一個 `multiprocessing.Queue` (`Cosmic.queue_in`)。模型只載入一次，記憶體不重複。

### 6.2 結果回流

`recognizer` 把 `Result` 丟回 `Cosmic.queue_out`。[`util/server/server_ws_send.py`](../util/server/server_ws_send.py) 從 queue 拉結果時：

1. 先讓 `task_router.try_resolve(result)` 攔截 HTTP 任務（中間/最終結果都會被吸收）。
2. 若不是 HTTP 任務，走原本的 WebSocket 派發路徑。

HTTP 與 WebSocket 的結果 routing 彼此分開，不會把逐字稿送錯 caller；但兩者共用
serial recognizer queue，仍可能互相增加等待時間。

### 6.3 合成 socket_id

HTTP 任務使用合成 socket_id `http:<task_id>` 並加入 `Cosmic.sockets_id`（跨進程 `Manager().list()`）。這是因為 recognizer 子進程在處理任務前會檢查 `task.socket_id not in sockets_id` 來判定上游是否還在；合成的 socket_id 滿足這個檢查、讓 HTTP 任務不會被丟棄。任務完成或取消時，TaskRouter 會把這個 socket_id 移除。

### 6.4 並發

- **識別本身嚴格串行：** 識別子進程一次處理一個 Task。多個 HTTP 請求 = 多組 Task 進入同一個 queue，自然形成 FIFO backpressure。
- **解碼/格式化可並行：** ffmpeg 解碼（subprocess）與 OpenAI 格式化（純 Python）在 asyncio loop 內可重疊。
- **建議：** 若需高並發或低延遲 SLA，請在 HTTP 前加佇列或 LB 並橫向擴充 server 實例；不要嘗試在單一進程內多開 recognizer，會浪費 VRAM。

---

## 7. 已知限制

| 限制 | 原因 | 影響 |
|---|---|---|
| `prompt` 僅 log，未注入 recognizer context | 不同識別後端（FunASR / SenseVoice / Qwen）對 prompt 支援方式差異大；統一處理會破壞穩定性 | 想做 hot-word/上下文引導請改用 WebSocket 客戶端模式，或在客戶端側做 post-processing |
| `model` / `temperature` 為占位 | 本服務由 `CAPSWRITER_MODEL_TYPE` 決定模型；無 sampling 概念 | OpenAI SDK 寫什麼都不影響行為 |
| 不做語言自動偵測 | 本服務的中文模型輸出語言固定 | `language` 欄位只回填，不影響識別 |
| 無 streaming（SSE）回應 | OpenAI Whisper API 本身亦無 streaming；需要 streaming 請改用 WebSocket | 對長音訊請設較大的 `TASK_TIMEOUT` |
| `/v1/audio/translations` 永不實作 | 本地模型不做語種翻譯 | 客戶端如使用 `translate()`，請改 `transcribe()` |
| HTTP timeout 中止的任務不會立刻釋放 recognizer 資源 | recognizer 子進程要走完佇列才會跳過 cancelled 任務；與 WebSocket 客戶端中斷行為一致 | 短時間記憶體佔用略有殘留，不影響正確性 |

---

## 8. 推薦的模型搭配

HTTP API 是 batch / 互動式場景，**推薦切到 `fun_asr_nano` 以降低延遲**：

```bash
# 一行切到 fun_asr_nano + 開 HTTP API
CAPSWRITER_HTTP_API_ENABLE=true \
docker compose -f docker-compose.yml -f docker-compose.fun-asr.yml \
  up -d --force-recreate
```

| 模型 | 平均單次延遲 (5s 音訊) | 適用 |
|---|---|---|
| `qwen_asr` (預設) | 較高 (~2-3s) | 高精度長段轉錄、字幕生成 |
| `fun_asr_nano` | 較低 (~0.5-1.5s) | airi / 互動聊天 / OpenAI SDK 即時呼叫 |

切換時客戶端**不需要重啟**；只需重啟 server。

---

## 9. 安全建議

| 場景 | 建議 |
|---|---|
| 僅本機 Native Server | 保留 `BIND=127.0.0.1`；仍建議設定 `KEY` |
| 僅本機 Compose | 保留 container `BIND=0.0.0.0` 與 host `CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1`；仍建議設定 `KEY` |
| 同 LAN 共享 | 將 `CAPSWRITER_HTTP_API_HOST_BIND` 設為需要的 private host IP、設定 `KEY`、以 reverse proxy 加 TLS，並用 firewall 限制來源 IP |
| Remote／Internet | 優先保留 `CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1`，只讓同 host reverse proxy 對外提供 TLS；同時設定 `KEY`、IP／rate／body-size／timeout limits |

`KEY` 為純 Bearer token 比對，建議使用 ≥ 32 字元的隨機字串：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 10. 故障排除

### 10.1 啟動時看到 `HTTP API 已启用但系统找不到 ffmpeg`

- 裸機：`apt install ffmpeg` 或 `brew install ffmpeg`
- Docker：請確認 image 是 0.x.x 之後的版本（已內建 ffmpeg）。自建 image 請檢查 [`docker/server/Dockerfile`](../docker/server/Dockerfile) 是否包含 `ffmpeg`

### 10.2 `500 Server misconfigured: ffmpeg not found`

同上。ffmpeg 沒安裝；伺服器側問題。

### 10.3 `400 Audio decode failed`

ffmpeg 拒絕解碼。可能原因：

- 上傳的不是音訊檔
- 容器/編碼損壞
- 試試在本機用 `ffmpeg -i <file>` 看完整錯誤

### 10.4 `504 Recognition timeout`

任務超過 `CAPSWRITER_HTTP_API_TASK_TIMEOUT` 還沒做完。

- 對長音訊：將 timeout 設更高（例如 1800 秒）
- 對 CPU-only 部署：考慮切換到 `fun_asr_nano` 或啟用 GPU
- 若整體吞吐才是瓶頸：橫向擴充

### 10.5 `413 File too large`

上傳超過 `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB`。把該上限調高、或客戶端側先分片。

### 10.6 `401 Missing or invalid Authorization header`

```bash
# 正確
curl -H "Authorization: Bearer sk-token" ...

# 常見錯誤：
# - 沒加 "Bearer " prefix
# - token 與伺服器設定不一致
```

---

## 11. 程式檔案地圖

| 檔案 | 職責 |
|---|---|
| [`util/server/http_api.py`](../util/server/http_api.py) | FastAPI app、端點實作、上傳處理、分段提交 |
| [`util/server/task_router.py`](../util/server/task_router.py) | task_id ↔ asyncio.Future 路由；合成 socket_id 管理 |
| [`util/server/audio_decoder.py`](../util/server/audio_decoder.py) | FFmpeg subprocess 解碼到 16k/f32/mono PCM |
| [`util/server/openai_formatter.py`](../util/server/openai_formatter.py) | 5 種 `response_format` 輸出格式化 |
| [`util/server/server_ws_send.py`](../util/server/server_ws_send.py) | 結果分派；HTTP 攔截點 `try_resolve` |
| [`config_server.py`](../config_server.py) | `CAPSWRITER_HTTP_API_*` 環境變數綁定 |
| [`core_server.py`](../core_server.py) | 與 WebSocket 共用 event loop 啟動 |

---

## 12. 變更與相容性

- 端點與回應格式對齊 OpenAI Whisper API subset；行為偏差皆已列於 §7。
- 環境變數命名前綴 `CAPSWRITER_HTTP_API_` 為穩定承諾；未來不會更名。
- 預設 disable 是穩定承諾；未啟用 HTTP API 時 server 行為與舊版完全一致。
