# v1 Server: OpenAI-Compatible ASR HTTP API

> English · [Traditional Chinese](../HTTP_API.md) ·
> [Project README](../../README.en.md)

CapsWriter-Offline v1 can expose an optional HTTP transcription endpoint next
to its WebSocket service. The endpoint runs in the **ASR Server** process,
shares that server's model and recognizer queue, and is disabled by default.

This is a documented compatibility subset of the OpenAI Whisper Audio API. It
is not the complete OpenAI Audio API, and it is not a second desktop Client.

## 1. Server and Client boundary

| Role | Protocol | Responsibility |
|---|---|---|
| **v1 ASR Server** | WebSocket `6016`; optional HTTP `6017` | Loads FFmpeg and the selected model, accepts audio, runs inference, and returns transcripts |
| **Legacy Windows desktop Client** (`start_client.py`) | WebSocket `6016` only | Owns microphone capture, tray, hotkeys, clipboard, and text injection; it does not host or call this HTTP API |
| **External API caller** | HTTP `6017` | A separate curl, OpenAI SDK, or compatible application that uploads an audio file to the Server |

The HTTP and WebSocket paths use one Server recognizer process. Model memory is
not duplicated, but recognition jobs share one serial queue. Heavy HTTP use can
therefore increase latency for the desktop Client, and vice versa.

In v1 Releases, `start_client.py` is compatibility-preserved Client **source**;
it connects over WebSocket and is not an HTTP API wrapper. The current v1
source-only release does not attach a Windows executable.

## 2. Enable the API safely

The Python Server and the Compose template intentionally have different bind
defaults:

| Setting | Native Server default | v1 Compose template | Purpose |
|---|---|---|---|
| `CAPSWRITER_HTTP_API_ENABLE` | `false` | `false` | Enables the HTTP server when set to `true` |
| `CAPSWRITER_HTTP_API_BIND` | `127.0.0.1` | `0.0.0.0` inside the container | Address on which the Python HTTP server listens |
| `CAPSWRITER_HTTP_API_HOST_BIND` | Not used | `127.0.0.1` on the host | Compose-only address used to publish the container port |
| `CAPSWRITER_HTTP_API_PORT` | `6017` | `6017` | HTTP listen and published port |
| `CAPSWRITER_HTTP_API_KEY` | Empty | Empty | Bearer token; an empty value disables authentication |
| `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` | `100` | `100` | Maximum uploaded file size in MiB |
| `CAPSWRITER_HTTP_API_TASK_TIMEOUT` | `600` | `600` | Recognition timeout in seconds |

Non-loopback listeners, including Compose's `0.0.0.0`, require an API key by
default. `CAPSWRITER_HTTP_API_KEY_FILE` supports a mounted secret. The explicit
`CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND` override is only for deliberately
isolated environments.

The native default is loopback-only. In Docker, the process must listen on
`0.0.0.0` **inside the container** so Docker can forward traffic; Compose then
publishes that port only on host loopback with
`CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1`. The two addresses protect different
network boundaries and must not be treated as interchangeable.

Compose declares the `6017` port mapping even while the feature is disabled.
No HTTP listener starts until `CAPSWRITER_HTTP_API_ENABLE=true`.

### 2.1 Docker Compose: local access

Copy `.env.example` to `.env`, then set:

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
CAPSWRITER_HTTP_API_MAX_UPLOAD_MB=100
CAPSWRITER_HTTP_API_TASK_TIMEOUT=600
```

Recreate the Server after changing `.env`:

```bash
docker compose up -d --force-recreate capswriter-server
docker compose logs -f capswriter-server
```

The API base URL is then `http://127.0.0.1:6017/v1` on the Docker host.

Generate a suitable token with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2.2 Native Server: local access

The native Server already defaults to `127.0.0.1:6017`:

```bash
export CAPSWRITER_HTTP_API_ENABLE=true
export CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
python start_server_universal.py
```

`CAPSWRITER_HTTP_API_HOST_BIND` has no effect outside Compose.

### 2.3 LAN or remote access

Do not expose the raw unauthenticated HTTP listener to a LAN or the Internet.
For every non-loopback deployment:

1. Set a non-empty, long random `CAPSWRITER_HTTP_API_KEY`.
2. Put the API behind a trusted reverse proxy that terminates TLS, and send the
   Bearer token only over HTTPS.
3. Keep `CAPSWRITER_HTTP_API_HOST_BIND=127.0.0.1` when the proxy runs on the same
   host. If the proxy must reach another host interface, set
   `CAPSWRITER_HTTP_API_HOST_BIND` to that exact private interface address where
   possible, then restrict it with a firewall allowlist.
4. Keep `CAPSWRITER_HTTP_API_BIND=0.0.0.0` inside Compose. For a native Server,
   change this setting only to the interface required by the reverse proxy.
5. Configure proxy-side request-size, timeout, and rate limits in addition to
   the application limits.

The built-in Bearer token is authentication, not encryption. This Server does
not provide TLS or a rate limiter itself. `GET /health`, `GET /v1/models`, and
the `501` translations response are not protected by the transcription token;
restrict them at the proxy when metadata disclosure matters.

## 3. Supported API subset

| Endpoint | Status | Contract |
|---|---|---|
| `POST /v1/audio/transcriptions` | Supported | Multipart file transcription with five response formats |
| `POST /v1/audio/translations` | Not implemented | Always returns `501`; v1 transcribes but does not translate |
| `GET /health` | Supported | Server status, configured model, and v1 source version |
| `GET /v1/models` | Supported | One compatibility record for the Server's configured model |

Only file transcription is compatible. Chat completions, embeddings,
real-time/SSE streaming, translation, model selection per request, and other
OpenAI endpoints are outside this v1 contract.

### 3.1 `POST /v1/audio/transcriptions`

Send `multipart/form-data` with these fields:

| Field | Required | Default | v1 behavior |
|---|---|---|---|
| `file` | Yes | None | Audio in a format that the Server's FFmpeg can decode |
| `model` | Yes | None | Must be `whisper-1`; the actual engine uses `CAPSWRITER_MODEL_TYPE` |
| `language` | No | Auto | Supported language hint passed to recognition; backend capabilities vary |
| `prompt` | No | None | Passed into recognizer context; its contents are not logged by default |
| `response_format` | No | `json` | `json`, `text`, `verbose_json`, `srt`, or `vtt` |
| `temperature` | No | `0.0` | Validated as a finite number from 0 to 1; does not override engine sampling |

`stream=true`, diarization, logprobs, and unknown fields are rejected. Existing
callers must explicitly send `model=whisper-1` after this refresh.

When `CAPSWRITER_HTTP_API_KEY` is non-empty, include:

```text
Authorization: Bearer <CAPSWRITER_HTTP_API_KEY>
```

Response formats:

| `response_format` | Content type | Result |
|---|---|---|
| `json` | `application/json` | `{"text":"..."}` |
| `text` | `text/plain` | Transcript only |
| `verbose_json` | `application/json` | Text, duration, segments, and word/token timestamps when available |
| `srt` | `application/x-subrip` | SRT subtitles |
| `vtt` | `text/vtt` | WebVTT subtitles |

`verbose_json` timing arrays depend on aligned token data from the configured
model and can be empty. Do not assume Whisper-identical segmentation.

### 3.2 Limits and errors

| Status | Meaning |
|---|---|
| `400` | Empty, undecodable, corrupt, or shorter-than-0.05-second audio |
| `401` | Missing, malformed, or incorrect Bearer token when a key is configured |
| `413` | Uploaded file exceeds `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` |
| `415` | Transcription request is not `multipart/form-data` |
| `422` | Multipart field or value fails request validation |
| `429` | Concurrent and pending request capacity is exhausted |
| `503` | The recognizer is not ready or has stopped |
| `500` | FFmpeg is unavailable or recognition fails |
| `504` | Recognition exceeds `CAPSWRITER_HTTP_API_TASK_TIMEOUT` |

Authentication and media-type checks happen before multipart field parsing.
The application reads the uploaded file with its configured size bound, but a
remote deployment should also reject oversized bodies at its reverse proxy.

## 4. Call the API

### 4.1 curl

```bash
curl http://127.0.0.1:6017/v1/audio/transcriptions \
  -H "Authorization: Bearer replace-with-a-long-random-token" \
  -F "file=@meeting.mp3" \
  -F "model=whisper-1" \
  -F "response_format=text"
```

Omit the `Authorization` header only when the Server key is empty and the
listener is restricted to a trusted local boundary.

### 4.2 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:6017/v1",
    api_key="replace-with-a-long-random-token",
)

with open("meeting.mp3", "rb") as audio:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio,
        response_format="verbose_json",
    )

print(transcript.text)
```

The OpenAI SDK requires a non-empty `api_key` argument. If Server
authentication is disabled, the SDK value may be any non-empty placeholder;
that does not add security to the Server.

### 4.3 OpenAI Node SDK

```ts
import fs from "node:fs";
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:6017/v1",
  apiKey: "replace-with-a-long-random-token",
});

const transcript = await client.audio.transcriptions.create({
  model: "whisper-1",
  file: fs.createReadStream("meeting.mp3"),
  response_format: "text",
});

console.log(transcript);
```

## 5. Operational behavior

- FFmpeg decodes uploads to the Server's internal PCM format.
- Long audio is split into overlapping tasks and submitted to the same queue as
  WebSocket work.
- One recognizer process handles inference serially. Multiple HTTP requests are
  queued; they are not processed by a separate worker pool.
- Decode and response formatting can overlap, but this does not make model
  inference concurrent.
- Scale with multiple isolated Server instances when a workload needs greater
  throughput. Each instance loads its own model and needs its own resource
  budget.

## 6. Known limitations

| Limitation | Practical effect |
|---|---|
| Transcription subset only | `/v1/audio/translations` returns `501`; other OpenAI APIs do not exist |
| Only `model=whisper-1` is accepted | The configured Server engine is used; temperature does not override engine sampling |
| Language hints depend on the backend | Unsupported languages are not made available by passing a hint |
| Prompt handling depends on the backend | Context is forwarded, but its effect varies across engines |
| No HTTP streaming | Use the WebSocket Client protocol when incremental recognition is required |
| Shared, serial recognizer queue | HTTP and WebSocket jobs can delay each other |
| Timeout cancellation is not instantaneous | A submitted recognizer task may retain resources until the queue reaches it |
| No built-in TLS or rate limiting | A reverse proxy and firewall are required for non-loopback access |

CI covers the API contract, authentication, upload bounds, formats, and routing.
It does not certify model quality, real audio, GPU/CPU performance, or a remote
production deployment.

## 7. Troubleshooting

### `500 Server misconfigured: ffmpeg not found`

Install FFmpeg for a native Server. The repository's Server Dockerfile includes
FFmpeg; if a custom image fails, inspect that image rather than the Client.

### `400 Audio decode failed`

Confirm that the upload is a real, non-corrupt audio file and that the Server's
FFmpeg build supports its codec. Running `ffmpeg -i <file>` on the Server host
usually reveals the decoder error.

### `504 Recognition timeout`

Increase `CAPSWRITER_HTTP_API_TASK_TIMEOUT` for long files or CPU-only hosts.
If queued work is the bottleneck, use separate Server instances instead of
assuming the one recognizer can process concurrent inference.

### `401 Missing or invalid Authorization header`

Confirm the `Bearer ` prefix, token value, and the Server container's effective
environment. Recreate the Compose service after changing `.env`.

## 8. Implementation map

| File | Server responsibility |
|---|---|
| [`fork_server/http_api/api.py`](../../fork_server/http_api/api.py) | FastAPI routes, authentication, upload handling, and task submission |
| [`fork_server/http_api/limits.py`](../../fork_server/http_api/limits.py) | Bounded upload reads |
| [`fork_server/http_api/audio_decoder.py`](../../fork_server/http_api/audio_decoder.py) | FFmpeg decoding |
| [`fork_server/http_api/task_router.py`](../../fork_server/http_api/task_router.py) | HTTP future and recognizer-result routing |
| [`fork_server/http_api/openai_formatter.py`](../../fork_server/http_api/openai_formatter.py) | JSON, text, SRT, and VTT responses |
| [`fork_server/env_config.py`](../../fork_server/env_config.py) | Native `CAPSWRITER_HTTP_API_*` configuration |
| [`docker-compose.yml`](../../docker-compose.yml) | Container bind and host publish boundary |
