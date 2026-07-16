# OpenAI 相容語音轉錄 API

> 繁體中文 · [English](../en/openai-api.md)

CapsWriter 可選擇性提供本機 `POST /v1/audio/transcriptions` 端點。它實作
官方 OpenAI Python SDK 使用的 `whisper-1` 檔案轉錄子集；實際離線識別器仍由
`CAPSWRITER_MODEL_TYPE` 選擇。

這是一層明確界定的相容介面，並不宣稱本機已具備 OpenAI 現行 Audio API 的
每一項能力。支援欄位會通過官方 SDK；不支援的 streaming、說話者分離、
log probability 與翻譯功能會回明確錯誤，不會靜默忽略。

![有界的 OpenAI 相容請求生命週期：先做 pre-body 認證，再經 admission、解碼、共用 ASR 推論、格式化與清理](../assets/openai-api-lifecycle.svg)

文字等價說明：server 會先驗證認證與宣告的 body 大小，才讀取上傳內容。
只有設定數量的 active request 可以進入，等待佇列也有上限。multipart 會在
raw-body 上限內 spool；解碼後 PCM 有時長上限；同一個 deadline 從排隊一路
涵蓋到 cleanup。佇列滿時回 `429`；認證失敗或宣告超限時回 `401`／`413`，
而且不讀取 body。

## 相容範圍速覽

| 能力 | 本機契約 |
|---|---|
| 端點 | `POST /v1/audio/transcriptions` |
| wire 上的 model 名稱 | 只接受 `whisper-1` |
| 實際 ASR engine | `CAPSWRITER_MODEL_TYPE`（`qwen_asr`、`fun_asr_nano` 與上游支援的 engine） |
| 回應格式 | `json`、`text`、`verbose_json`、`srt`、`vtt` |
| timestamp 粒度 | `verbose_json` 可用 `segment`、`word` 或兩者 |
| SDK 證據 | 官方 OpenAI Python SDK `2.45.0` 契約測試 |
| Streaming | 不支援；`stream=true` 回 `400` |
| 說話者分離 | 不支援；diarization 欄位回 `400` |
| Log probability | 不支援；`include[]=logprobs` 回 `400` |
| 翻譯 | `/v1/audio/translations` 回 `501` |

相容目標依據官方 [create transcription
reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create/)
與 [speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text/)。
CapsWriter 只公開離線 pipeline 能誠實表達的能力。

## 啟用端點

HTTP API 預設關閉。本機 source checkout 可這樣啟動：

```bash
export CAPSWRITER_HTTP_API_ENABLE=true
export CAPSWRITER_HTTP_API_BIND=127.0.0.1
python start_server_docker.py
```

Windows PowerShell：

```powershell
$env:CAPSWRITER_HTTP_API_ENABLE = "true"
$env:CAPSWRITER_HTTP_API_BIND = "127.0.0.1"
python .\start_server_universal.py
```

Docker 請把 [`.env.example`](../../.env.example) 複製成 `.env`，設定 key，並只在
host loopback 發布 HTTP port：

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
```

取消 [`docker-compose.yml`](../../docker-compose.yml) 內 optional HTTP mapping 的
註解，再重建 service：

```bash
docker compose up -d --force-recreate capswriter-server
curl http://127.0.0.1:6017/ready
```

只要啟用的 API 綁定非 loopback 位址，就必須設定
`CAPSWRITER_HTTP_API_KEY` 或 `CAPSWRITER_HTTP_API_KEY_FILE`。明確逃生門
`CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true` 只能用在受信任、隔離的測試網路。

## 官方 Python SDK

把 SDK 指向本機 `/v1` base URL。這裡的 API key 是本機 server token，不會送往
OpenAI。

```python
from openai import OpenAI

client = OpenAI(
    api_key="replace-with-your-local-token",
    base_url="http://127.0.0.1:6017/v1",
)

with open("meeting.wav", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        response_format="verbose_json",
        timestamp_granularities=["segment", "word"],
        language="zh-TW",
        prompt="CapsWriter、Qwen3-ASR、繁體中文",
        temperature=0,
    )

print(transcript.text)
for segment in transcript.segments or []:
    print(segment.start, segment.end, segment.text)
```

最小 curl request：

```bash
curl http://127.0.0.1:6017/v1/audio/transcriptions \
  -H "Authorization: Bearer $CAPSWRITER_HTTP_API_KEY" \
  -F "file=@meeting.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

## Request 欄位

| 欄位 | 預設 | 行為 |
|---|---|---|
| `file` | 必填 | 一個 multipart file；可使用已安裝 ffmpeg build 能解碼的格式。 |
| `model` | 必填 | 必須完全等於 `whisper-1`；這是相容 ID，不是內部 engine 名稱。和官方 API 一樣，省略時會拒絕請求。 |
| `language` | `auto` | `en`、`zh-TW`、`ja` 等 ISO-style alias 與可讀語言名稱會正規化成 ASR hint。 |
| `prompt` | 空 | 正規化並限制為 2,048 字元，再作為 recognizer context。 |
| `response_format` | `json` | `json`、`text`、`verbose_json`、`srt`、`vtt` 之一。 |
| `temperature` | `0` | 驗證為 0 到 1 的有限數字，並寫入 verbose segment metadata；不會重新調校所選本機 ASR engine。 |
| `timestamp_granularities[]` | verbose JSON 預設 `segment` | 可重複傳 `segment`／`word`；只可搭配 `verbose_json`。也接受不含中括號的拼法。 |
| `stream` | false | false 可接受；true 會回明確的不支援錯誤。 |

未知欄位、重複 scalar、未支援 model 與錯誤數值一律回 OpenAI-style `400`。
如此可避免新版 SDK 欄位看似成功，實際上本機 engine 根本沒有套用。

### Timestamp 的意義

CapsWriter engine 提供的是 backend token／字元 alignment，不是跨語言一致的
語言學單字 tokenizer。因此 `words[]` 是 SDK-compatible alignment 近似；依所選
engine 不同，每一項可能是字元、subword 或 token。時間會正規化為有限、非負且
全域單調的值。請勿把這份近似當成說話者分離或法律用途的單字邊界資料。

## 回應

預設 JSON：

```json
{"text":"你好，世界"}
```

`verbose_json` 包含 Whisper-compatible 核心欄位：

```json
{
  "task": "transcribe",
  "language": "chinese",
  "duration": 1.0,
  "text": "你好，世界",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 1.0,
      "text": "你好，世界",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": 0.0,
      "compression_ratio": 0.0,
      "no_speech_prob": 0.0
    }
  ]
}
```

無法取得的 confidence 欄位使用中性數值 placeholder，並不是捏造的 model
measurement。成功回應會帶 `X-CapsWriter-Task-ID`，方便對照 server log。

`verbose_json.language` 是 request metadata，不是偵測語言的 metadata。若有傳
language hint，這個欄位會回正規化後的 hint（例如 `en` 變成 `english`）；未傳時
則回本機 sentinel `"auto"`。目前共用 `Result` 契約沒有 backend-independent 且
可靠的 detected language，因此不會捏造偵測結果。

## 資源與安全控制

| 環境變數 | 預設 | 強制行為 |
|---|---:|---|
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `100` | file bytes 上限；完整 multipart body 另有含固定 overhead 的獨立上限。 |
| `CAPSWRITER_HTTP_API_MAX_AUDIO_SECONDS` | `3600` | 解碼後 16 kHz mono PCM 時長上限；ffmpeg 轉換期間便限制 output。 |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `600` | admission wait、upload parse、decode、submit、inference、format、cleanup 共用一個 deadline。 |
| `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` | `2` | active transcription request 數，包含 upload 與 decode。 |
| `CAPSWRITER_HTTP_API_MAX_PENDING_REQUESTS` | `4` | 等待 admission 的 caller 上限；`0` 表示不等待，超出回 `429`。 |
| `CAPSWRITER_HTTP_API_CORS_ORIGINS` | 空 | 強制執行的 browser origin allowlist；帶 Origin 的 transcription POST 若不在清單內，會在讀取 body 前被拒絕。空值拒絕所有 browser origin；`*` 會刻意允許任何網站提交工作，未啟用驗證時並不安全。 |
| `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS` | `false` | 除非明確啟用，prompt 與 transcript text 不寫入 HTTP、worker、engine 或 console log；WebSocket desktop task 保留原有輸出政策。 |
| `CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS` | `8` | 可 admitted 的 WebSocket client 上限；啟動時只接受 `1..1024`。超額 client 收到 `1013`，若不回應 close handshake，會在一秒後 abort。 |
| `CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS` | `3600` | 單一 WebSocket task 的累計音訊上限；啟動時只接受 `1..86400` 秒。Overflow 只 reset 該 composite stream，並回傳 final `websocket_task_audio_limit_exceeded` result。 |
| `CAPSWRITER_SERVER_MODEL_LOAD_TIMEOUT` | `600` | recognizer child 載入模型的有限 startup deadline；超時會 terminate／kill／reap child 並讓啟動失敗，交由 supervisor 重啟。 |
| `CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT` | `900` | 單次同步 recognizer call 的 hard ceiling（秒），desktop／WebSocket work 也包含在內；必須為有限正數。 |

跨 process 的音訊 staging 使用固定、不可設定的安全上限：
`multiprocessing.Queue` 最多 8 個 task，child 內的公平 buffer 另最多 8 個。
每個 task 最長 64 秒，格式為 float32／16 kHz／mono PCM，恰為
4,096,000 bytes。HTTP producer 會在 thread 內以短 timeout 重試 bounded queue
put，並逐次檢查 request deadline、cancellation 與 synthetic-socket liveness。
WebSocket producer 以 non-blocking queue put 在 event loop 上 cooperative retry；每個
admitted connection 的 incoming frame 上限為 6 MiB，只預取一個 message，而且單一
message 解碼後不得超過 4,096,000 bytes。預設最多 admitted 8 個 client。
單一 task 預設最多累計 3,600 秒音訊；超限 control message 只清除該
`(socket_id, task_id)` worker session，connection、其他 task 與其他 socket 上碰撞的
task ID 都保持可用。

Result queue 最多 8 筆。每個 peer 同時只有一個 active send，另保留最多 8 個依 task
排序的 snapshot；同 task 的 intermediate snapshot 會 coalesce，不同 task 的 final
result 仍會保留。Send 超過五秒或 peer 超出 pending 上限時只會隔離該 peer。

預設同時兩個 active HTTP request 時，保留的 upload 加 decoded PCM 額度恰為
`2 × (104,857,600 + 230,400,000) = 670,515,200` bytes；queue 與 worker buffer
payload 最多再加 `16 × 4,096,000 = 65,536,000` bytes。若把 worker 正在處理的
一個 segment，以及兩個 active HTTP producer 各自 blocked 的一個 segment 也算入，
保守的 HTTP logical audio-payload ceiling 恰為 `748,339,200` bytes
（713.671875 MiB）。此數字不含 model memory、Python／process allocator overhead、
ffmpeg working memory與 multiprocessing serialization 的瞬間 copy；WebSocket 的
frame／cache memory 是 per-connection 有界，但不列入這個 HTTP-only 總數。

Multipart parser 只接受一個 file 與最多十二個文字欄位。成功、validation 失敗、
timeout 與 cancellation 都會關閉 upload spool。若外層 request cancellation 中斷
ffmpeg，server 會先 kill 並 reap child process，再離開 request。

Upload 完成後，handler 會讓 inference result 與 ASGI disconnect signal 競速。
Client 斷線時會立即移除 router entry 與 synthetic socket，尚未開始的 segment
因而被丟棄；disconnect watcher 也一定會回收。同一個 event-loop turn 已完成的
result 優先。

Native inference 本身是同步呼叫，無法在 worker 內安全中斷；parent 會監看原子化
active-inference lease。HTTP call 若超過全程 deadline 兩秒仍未結束，或任何 call
超過 `CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT`，就會 bounded terminate／kill child
並讓整個 server fail-stop。Server 刻意不 hot-restart shared worker，因為那會默默
破壞或遺失 worker 內的 desktop／WebSocket session state；請交由 service supervisor
重新啟動。

## Error 契約

錯誤一律使用 `{"error":{"message","type","param","code"}}`，讓 OpenAI SDK
依 status code 做一致 mapping。

| Status | 意義 |
|---:|---|
| `400` | Multipart 契約錯誤、不支援能力、音訊無法解碼／空白／過短 |
| `401` | Bearer token 缺失或錯誤 |
| `403` | Browser `Origin` 不在強制 allowlist |
| `413` | 宣告／raw body／file upload 上限或解碼時長上限超出 |
| `429` | Active work 加 bounded pending queue 已滿；請查看 `Retry-After` |
| `500` | Decoder 不可用或通用內部 recognition failure |
| `501` | Translation endpoint 刻意未提供 |
| `503` | Recognizer 不可用或正進入 fail-stop；待 supervisor 重啟後再試 |
| `504` | 全程 transcription deadline 到期 |

預設不會把內部 exception string、traceback、prompt、transcript 或 path 放進 HTTP
body 或 log。只有在可信診斷環境明確設定
`CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS=true`，才會同時啟用詳細 HTTP exception log。

## 維運與驗證

- `GET /health` 證明 HTTP app 正在執行。
- `GET /ready` 檢查 router binding、recognizer child liveness 與 ffmpeg，並回報
  不含 secret 的 limit；dependency degraded 或 recognizer 已死亡時回 `503`。
- `GET /v1/models` 回 `whisper-1` 相容 ID；啟用 key 時也需要 auth。

本機可跑 dependency-light API suite：

```bash
python -m unittest discover -s fork_server/http_api/tests -v
```

真實 multipart 與官方 SDK tests 以
[`requirements-api-test.txt`](../../requirements-api-test.txt) 作為精簡 direct input，
執行時使用完整解析的 Python 3.12/Linux
[`requirements-api-test.lock`](../../requirements-api-test.lock)：

```bash
python -m pip install --require-hashes --only-binary=:all: \
  -r requirements-api-test.lock
python scripts/verify_api_contract.py
```

請放在一次性 virtual environment／container，不要塞進 server runtime environment。
Strict verifier 會先檢查 direct-pin／lock parity、installed version 與 import，再
discover 完整 HTTP API test tree；zero tests、任何 failure 或任何 skip 都會讓 job
失敗。CI 與 publication gate 會強制每個 transitive version 與 wheel hash；direct
input 有變更時必須明確重產 lock。

Source、container 與 contract-test requirement 都固定使用 FastAPI `0.139.0`、
Starlette `1.3.1` 與 python-multipart `0.0.32`。重建 Windows executable 或 source
install 時，請維持這組配對版本；cancellation-safe multipart spool cleanup 會針對
這份明確 parser contract 進行測試。

較舊的部署細節仍可參閱 legacy [HTTP API reference](../HTTP_API.md)。
[v1/v2 雙軌維護政策](versioning.md) 則說明哪一軌持續開發契約，哪一軌只收
security backport。
