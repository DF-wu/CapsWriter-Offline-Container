# OpenAI-compatible transcription API

> [Documentation home](README.md) · [繁體中文](../zh-TW/openai-api.md) · English

CapsWriter exposes an optional local `POST /v1/audio/transcriptions` endpoint.
It implements the file-transcription subset used by the official OpenAI Python
SDK for `whisper-1`, while the actual offline recognizer remains selected by
`CAPSWRITER_MODEL_TYPE`.

This is a compatibility surface, not a claim that every current OpenAI audio
capability exists locally. Supported fields work through the official SDK;
unsupported streaming, diarization, log-probability, and translation features
return explicit errors instead of being silently ignored.

![Bounded OpenAI-compatible request lifecycle from pre-body authentication through admission, decoding, shared ASR inference, formatting, and cleanup](../assets/openai-api-lifecycle.svg)

Text equivalent: authentication and declared body size are checked before the
server reads the upload. At most the configured active requests enter, with a
bounded pending queue. Multipart data is spooled under a raw-body cap, decoded
PCM is duration-capped, and one deadline covers queueing through cleanup. A
full queue returns `429`; failed auth and declared oversize return `401` or
`413` without reading the body.

## Compatibility at a glance

| Capability | Local contract |
|---|---|
| Endpoint | `POST /v1/audio/transcriptions` |
| Model name on the wire | `whisper-1` only |
| Actual ASR engine | `CAPSWRITER_MODEL_TYPE` (`qwen_asr`, `fun_asr_nano`, and supported upstream engines) |
| Response formats | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| Timestamp granularities | `segment`, `word`, or both with `verbose_json` |
| SDK evidence | Official OpenAI Python SDK `2.45.0` contract test |
| Streaming | Not supported; `stream=true` returns `400` |
| Speaker diarization | Not supported; diarization fields return `400` |
| Log probabilities | Not supported; `include[]=logprobs` returns `400` |
| Translation | `/v1/audio/translations` returns `501` |

The compatibility target follows the official [create transcription
reference](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create/)
and [speech-to-text guide](https://developers.openai.com/api/docs/guides/speech-to-text/).
CapsWriter deliberately exposes only capabilities its offline pipeline can
represent honestly.

## Enable the endpoint

The HTTP API is off by default. For a local source checkout:

```bash
export CAPSWRITER_HTTP_API_ENABLE=true
export CAPSWRITER_HTTP_API_BIND=127.0.0.1
python start_server_docker.py
```

PowerShell on Windows:

```powershell
$env:CAPSWRITER_HTTP_API_ENABLE = "true"
$env:CAPSWRITER_HTTP_API_BIND = "127.0.0.1"
python .\start_server_universal.py
```

For Docker, copy [`.env.example`](../../.env.example) to `.env`, set a key,
and publish the HTTP port on the host loopback interface:

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
```

Uncomment the optional HTTP mapping in
[`docker-compose.yml`](../../docker-compose.yml), then recreate the service:

```bash
docker compose up -d --force-recreate capswriter-server
curl http://127.0.0.1:6017/ready
```

Binding an enabled API to a non-loopback address requires
`CAPSWRITER_HTTP_API_KEY` or `CAPSWRITER_HTTP_API_KEY_FILE`. The explicit
`CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true` escape hatch is for a trusted,
isolated test network only.

## Official Python SDK

Point the SDK at the local `/v1` base URL. The API key is a local server token;
it is not sent to OpenAI.

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
        language="en",
        prompt="CapsWriter, Qwen3-ASR",
        temperature=0,
    )

print(transcript.text)
for segment in transcript.segments or []:
    print(segment.start, segment.end, segment.text)
```

Minimal curl request:

```bash
curl http://127.0.0.1:6017/v1/audio/transcriptions \
  -H "Authorization: Bearer $CAPSWRITER_HTTP_API_KEY" \
  -F "file=@meeting.wav" \
  -F "model=whisper-1" \
  -F "response_format=json"
```

## Request fields

| Field | Default | Behavior |
|---|---|---|
| `file` | required | One multipart file. Formats supported by the installed ffmpeg build are accepted. |
| `model` | required | Must be exactly `whisper-1`; it is the compatibility ID, not the internal engine name. Like the official API, omission is rejected. |
| `language` | `auto` | ISO-style aliases such as `en`, `zh-TW`, `ja`, and readable language names are normalized into an ASR hint. |
| `prompt` | empty | Normalized and bounded to 2,048 characters before being passed as recognizer context. |
| `response_format` | `json` | One of `json`, `text`, `verbose_json`, `srt`, or `vtt`. |
| `temperature` | `0` | Validated as a finite number from 0 through 1 and reported in verbose segment metadata; it does not retune the selected local ASR engine. |
| `timestamp_granularities[]` | `segment` for verbose JSON | Repeated `segment` and/or `word` values; valid only with `verbose_json`. The unbracketed spelling is also accepted. |
| `stream` | false | False is accepted. True returns an explicit unsupported-capability error. |

Unknown fields, duplicate scalar fields, unsupported models, and invalid numeric
values return `400` in an OpenAI-style error envelope. This prevents a new SDK
feature from appearing to work when the local engine did not apply it.

### Timestamp meaning

CapsWriter engines expose backend token or character alignment rather than a
universal linguistic word tokenizer. Therefore `words[]` is an SDK-compatible
alignment approximation: entries may be characters, subwords, or tokens
depending on the selected engine. Timestamps are normalized to finite,
non-negative, globally monotonic values. Do not use this approximation as a
speaker diarization or legal word-boundary source.

## Responses

Default JSON:

```json
{"text":"hello world"}
```

`verbose_json` includes the official Whisper-compatible core fields:

```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 1.0,
  "text": "hello world",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 1.0,
      "text": "hello world",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": 0.0,
      "compression_ratio": 0.0,
      "no_speech_prob": 0.0
    }
  ]
}
```

Unavailable confidence values use neutral numeric placeholders; they are not
fabricated model measurements. Successful responses include
`X-CapsWriter-Task-ID` for correlation with server logs.

`verbose_json.language` is request metadata, not detected-language metadata.
When a language hint is supplied it contains that normalized hint (for example,
`en` becomes `english`). Without a hint it is the local sentinel `"auto"`,
because the shared `Result` contract currently exposes no reliable
backend-independent detected language.

## Resource and security controls

| Environment variable | Default | Enforcement |
|---|---:|---|
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `100` | Maximum file bytes; the complete multipart body is independently capped with bounded overhead. |
| `CAPSWRITER_HTTP_API_MAX_AUDIO_SECONDS` | `3600` | Maximum decoded 16 kHz mono PCM duration. ffmpeg output is capped during conversion. |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `600` | One deadline for admission wait, upload parsing, decode, submission, inference, formatting, and cleanup. |
| `CAPSWRITER_HTTP_API_MAX_CONCURRENT_REQUESTS` | `2` | Active transcription requests, including upload and decode work. |
| `CAPSWRITER_HTTP_API_MAX_PENDING_REQUESTS` | `4` | Maximum callers waiting for admission. `0` disables waiting; overflow returns `429`. |
| `CAPSWRITER_HTTP_API_CORS_ORIGINS` | empty | Enforced browser-origin allowlist. An Origin-bearing transcription POST not on the list is rejected before its body is read; empty rejects all browser origins. `*` deliberately allows any website to submit work and is unsafe without authentication. |
| `CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS` | `false` | Prompts and transcript text stay out of HTTP, worker, engine, and console logs unless explicitly enabled. WebSocket desktop tasks keep their existing output policy. |
| `CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS` | `8` | Maximum admitted WebSocket clients; startup accepts only `1..1024`. Excess clients receive `1013`, and an ignored close handshake is aborted after one second. |
| `CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS` | `3600` | Maximum cumulative audio in one WebSocket task; startup accepts `1..86400` seconds. Overflow resets only that composite stream and returns a final `websocket_task_audio_limit_exceeded` result. |
| `CAPSWRITER_SERVER_MODEL_LOAD_TIMEOUT` | `600` | Finite startup deadline for the recognizer child to load models. Timeout terminates/kills/reaps the child and fails startup for supervisor restart. |
| `CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT` | `900` | Hard ceiling in seconds for one synchronous recognizer call, including desktop/WebSocket work. Must be finite and positive. |

Cross-process audio staging has fixed, non-configurable safety bounds: the
`multiprocessing.Queue` holds at most 8 tasks and the child-local fair buffer
holds at most 8 more. Every task is at most 64 seconds of float32 16 kHz mono
PCM, exactly 4,096,000 bytes. HTTP producers retry bounded queue puts in a
thread while checking the request deadline, cancellation, and synthetic-socket
liveness. WebSocket producers retry non-blocking queue puts cooperatively on
the event loop; incoming frames are limited to 6 MiB with only one queued
message per admitted connection, and decoded audio in one message cannot
exceed 4,096,000 bytes. At most 8 clients are admitted by default, and one
task may contain at most 3,600 cumulative audio seconds. A task-limit control
message clears only its `(socket_id, task_id)` worker session; the connection,
other tasks, and colliding task IDs on other sockets remain live.

The result queue holds at most 8 items. Delivery has one active send and at
most 8 ordered per-task snapshots per peer; same-task intermediate snapshots
coalesce, while cross-task final results are preserved. A send taking more
than five seconds or a peer overflowing that bound evicts only that peer.

With the default two active HTTP requests, the retained upload plus decoded
PCM allowance is exactly
`2 × (104,857,600 + 230,400,000) = 670,515,200` bytes. Queue and worker-buffer
payload add at most `16 × 4,096,000 = 65,536,000` bytes. Including one active
worker segment and one blocked producer segment for each active HTTP request,
the conservative logical HTTP audio-payload ceiling is therefore exactly
`748,339,200` bytes (713.671875 MiB). This accounting excludes model memory,
Python/process allocator overhead, ffmpeg working memory, and transient
multiprocessing serialization copies. WebSocket connection-local frame/cache
memory is bounded per connection but is outside that HTTP-only total.

Multipart parsing accepts one file and at most twelve text fields. Temporary
upload spools close on success, validation failure, timeout, and cancellation.
If an outer request cancellation interrupts ffmpeg, the child process is killed
and reaped before the request exits.

After upload, the handler races the inference result against the ASGI disconnect
signal. A disconnected caller immediately loses its router entry and synthetic
socket, so queued segments are discarded and the disconnect watcher is always
reaped. A result completed in the same event-loop turn wins the race.

Native inference itself is synchronous and cannot be safely interrupted in the
worker. The parent therefore watches an atomic active-inference lease. An HTTP
call still active two seconds beyond its end-to-end deadline, or any call beyond
`CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT`, triggers bounded terminate/kill of the
child and a whole-server fail-stop. The server deliberately does not hot-restart
the shared worker because doing so would silently corrupt or abandon in-worker
desktop/WebSocket session state; use the service supervisor to restart it.

## Error contract

Errors use `{"error":{"message","type","param","code"}}` so the OpenAI SDK
maps status codes consistently.

| Status | Meaning |
|---:|---|
| `400` | Invalid multipart contract, unsupported capability, undecodable/empty/too-short audio |
| `401` | Missing or invalid Bearer token |
| `403` | Browser `Origin` is absent from the enforced allowlist |
| `413` | Declared/raw/file upload limit or decoded-duration limit exceeded |
| `429` | Active work plus the bounded pending queue is full; inspect `Retry-After` |
| `500` | Decoder unavailable or generic internal recognition failure |
| `501` | Translation endpoint is intentionally unavailable |
| `503` | Recognizer is unavailable or entering fail-stop; retry after supervisor restart |
| `504` | End-to-end transcription deadline expired |

Internal exception strings, tracebacks, prompts, transcripts, and paths are
neither returned nor logged by default. The explicit
`CAPSWRITER_HTTP_API_LOG_TRANSCRIPTS=true` privacy opt-in also enables detailed
HTTP exception logging for trusted diagnostics.

## Operations and verification

- `GET /health` proves the HTTP app is running.
- `GET /ready` checks router binding, recognizer-child liveness, and ffmpeg,
  and reports non-secret limit settings. A degraded dependency or dead
  recognizer returns `503`.
- `GET /v1/models` returns the `whisper-1` compatibility ID and requires auth
  when the API key is enabled.

Run the dependency-light API suite locally:

```bash
python -m unittest discover -s fork_server/http_api/tests -v
```

The real multipart and official-SDK tests use
[`requirements-api-test.txt`](../../requirements-api-test.txt) as their concise
direct input and the fully resolved Python 3.12/Linux
[`requirements-api-test.lock`](../../requirements-api-test.lock) for execution:

```bash
python -m pip install --require-hashes --only-binary=:all: \
  -r requirements-api-test.lock
python scripts/verify_api_contract.py
```

Use a disposable virtual environment/container, not the server runtime
environment. The strict verifier checks direct-pin/lock parity, installed
versions and imports, then discovers the complete HTTP API test tree; zero
tests, any failure, or any skip fails the job. CI and the publication gate
enforce every transitive version and wheel hash; regenerate the lock explicitly
when changing the direct input.

The supported HTTP parser stack is pinned consistently as FastAPI `0.139.0`,
Starlette `1.3.1`, and python-multipart `0.0.32` in the source, container, and
contract-test requirement sets. Keep those paired versions when rebuilding a
Windows executable or source install;
the cancellation-safe multipart spool cleanup is covered against that exact
parser contract.

For older deployment detail, see the legacy [HTTP API reference](../HTTP_API.md).
The paired [v1/v2 maintenance policy](versioning.md) explains which contract is
actively developed and which receives security-only backports.
