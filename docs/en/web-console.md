# Web Console client

> [Documentation home](README.md) · [繁體中文](../zh-TW/web-console.md) · [Server and client roles](server-and-clients.md)

The Web Console is a **client**, not an ASR server. Its static files are served
on port `8080`; the browser then sends audio to a CapsWriter server's opt-in
HTTP API on port `6017`. The browser does not load a recognition model.

![CapsWriter Web Console data flow from browser input through the HTTP API to the local ASR worker](../assets/web-console-architecture.svg)

## Responsibilities

| Web Console owns | ASR server owns |
|---|---|
| Microphone/file selection, playback, UI state, history, downloads | Model download/load, FFmpeg decode, inference, hotwords, readiness |
| Browser-local Web Speech TTS | Speech-to-text result generation |
| API root/key entered in page memory | Authentication validation, request limits, queue/deadline |

Browser TTS is local browser/OS functionality. It is not a server TTS endpoint.

## 1. Prepare the server

Enable HTTP and configure an explicit browser origin:

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
```

Uncomment the HTTP mapping under `ports:` in `docker-compose.yml`, recreate the
server, and verify readiness:

```bash
docker compose up -d --force-recreate capswriter-server
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

## 2. Run the production Web client

Build the local static image and publish it only on loopback:

```bash
docker compose -f docker-compose.web.yml up -d --build capswriter-web
curl http://127.0.0.1:8080/health
```

Open `http://127.0.0.1:8080`, confirm the API root is
`http://127.0.0.1:6017`, and enter the server token in the masked key field.

`CAPSWRITER_WEB_API_KEY` is written to public `/config.js`. Prefer leaving it
empty. A deployment that deliberately publishes a default key must also set
`CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true` and accept that every browser able to
load the page can read it.

## Development

```bash
cd client/web
npm ci --no-audit --no-fund
npm run dev
```

For UI-only work without a model server, run `npm run mock-api` in another
terminal. It returns fixed data and is not evidence of real transcription.

Verification:

```bash
npm run verify
npm run browser-smoke
```

## Browser security and privacy

- Microphone capture requires localhost or HTTPS plus browser/OS permission.
- `localhost` and `127.0.0.1` are different CORS origins; allow the exact one
  used by the page.
- Manually entered API keys remain in page memory and are not saved in history.
- Settings and up to 20 plaintext transcript/raw history records use
  `localStorage`; clear site data on shared machines.
- Downloads sanitize path separators, control characters, and reserved names.
- Diagnostic and transcription fetches reject redirects so private audio and
  Bearer keys are not resent to another origin.

## Common failures

| Symptom | Check |
|---|---|
| Web page loads but diagnostics fail | Server HTTP `6017`, API root, key, and `/ready` |
| CORS error | Exact page origin in `CAPSWRITER_HTTP_API_CORS_ORIGINS` |
| Microphone unavailable | Localhost/HTTPS, browser permission, OS permission |
| TTS silent | Browser/OS voice availability and autoplay/audio policy |
| `413` upload rejection | Server `max_upload_mb` and selected file size |

For network exposure, runtime variables, upgrades, and rollback, continue with
[Deployment](deployment.md#web-console-profile) and
[Support and security](support-security.md).

