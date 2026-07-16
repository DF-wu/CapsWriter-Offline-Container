# Troubleshooting

> [Documentation home](README.md) · [繁體中文](../zh-TW/troubleshooting.md) · [Deployment](deployment.md)

Diagnose CapsWriter from the server outward. A client error is often only the
last visible symptom of an earlier readiness, authentication, decode, or model
failure.

![Bounded OpenAI-compatible request lifecycle from pre-body checks through admission, decoding, shared ASR inference, formatting, and cleanup](../assets/openai-api-lifecycle.svg)

Text equivalent: authentication and declared size are checked before the body
is consumed; bounded admission precedes multipart spooling and capped ffmpeg
decode; the request then enters the shared ASR worker, response formatter, and
cleanup path. Queue overflow, invalid input, timeout, and cancellation leave
through explicit error/cleanup paths rather than silently continuing.

## Diagnostic ladder

Run these checks in order and stop at the first failure:

1. **Process/container:** is the expected process running, and is it the current
   source/image/configuration?
2. **Transport:** can the client reach the configured host and port?
3. **Health:** does the HTTP app answer `/health`?
4. **Readiness:** does `/ready` report the router, recognizer child, dependencies,
   and ffmpeg ready?
5. **Models/auth:** does `/v1/models` succeed with the same Bearer token?
6. **Known audio:** can a small known file transcribe as `text`?
7. **Requested feature:** only then add verbose JSON, subtitles, prompts,
   browser microphone, TUI recording, or batch processing.

The no-GUI CLI provides the first five HTTP checks without browser state:

```bash
export CAPSWRITER_API_BASE=http://127.0.0.1:6017
export CAPSWRITER_HTTP_API_KEY_FILE=/path/to/capswriter-http.key
python client/cli/capswriter_cli.py health
python client/cli/capswriter_cli.py ready
python client/cli/capswriter_cli.py models
```

## Desktop problems

| Symptom | Likely boundary | Action |
|---|---|---|
| Windows source works, packaged executable fails | Hidden import/native library/package layout | Rebuild from a clean environment, keep build logs, verify both packaged executables, and compare exact dependency/PyInstaller versions |
| Tray or shortcut does not respond on Windows | Desktop permission/session/device-specific runtime | Confirm normal interactive session, configured shortcut, audio permission/device, and that another app has not reserved the key |
| Linux shortcuts are unavailable | Not an X11 session or listener initialization failed | Check `XDG_SESSION_TYPE=x11`, `DISPLAY`, X11 dependencies, and startup diagnostics |
| Linux `suppress=True` has no effect | Expected X11 safety policy | Do not enable a whole-keyboard grab; choose a shortcut that does not need suppression |
| Wayland/headless reports unsupported hotkeys | Intentional support boundary | Use X11 for desktop hotkeys, or use CLI/TUI/Web/file workflows without global input capture |
| File transcription works but microphone does not | Device/permission/native dependency | Verify the OS input device and client audio stack independently; do not treat server readiness as microphone proof |

See [desktop portability](desktop-portability.md) for the exact claims.

## Container and model problems

### Container remains `starting`

First bootstrap may download several assets during the long health-check start
period. Inspect, do not repeatedly recreate:

```bash
docker compose ps
docker compose logs --tail=300 capswriter-server
```

Look for a bounded download error, archive validation error, missing hotword
mount, backend probe failure, model load failure, or recognizer child exit.

When containers share model storage, one may be waiting on
`/app/models/.capswriter-bootstrap.lock` for at most
`CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT` (1800 seconds by default). A timeout
or a symlink/hard-link/ownership/mode rejection means another writer is stuck
or the storage does not satisfy the secure-lock contract. Stop redundant
writers, verify coherent POSIX `flock(2)` across containers, and correct bind
ownership. Do not remove a lock still held by a live process. A fully warm
normal start does not create or write the lock.

### GPU is not used

```bash
docker compose config
docker compose logs --tail=300 capswriter-server
```

Confirm the host driver/toolkit, requested GPU count, visible devices, and
selected model backend. After a configured GPU probe fails,
`CAPSWRITER_INFERENCE_HARDWARE=auto` disables every CUDA/Vulkan path, prepares
the CPU runtime, and requires the second CPU engine probe to pass. The
container refuses startup if that probe also fails. Prove functional CPU
transcription before tuning GPU flags.

`probe_backend.py` constructs a complete ASR engine. Never run it with
`docker compose exec` inside an active server: that can load a second model and
exhaust RAM/VRAM. If logs are insufficient and a deep probe is necessary, use
a maintenance window and keep the service stopped while the one-shot container
runs (run the final `start` even if the probe fails):

```bash
docker compose stop capswriter-server
docker compose run --rm --no-deps --entrypoint python capswriter-server \
  docker/server/probe_backend.py
docker compose start capswriter-server
```

For an Intel/AMD iGPU, inspect the explicit override and actual host GIDs:

```bash
stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*
docker compose -f docker-compose.yml -f docker-compose.igpu.yml config
docker compose -f docker-compose.yml -f docker-compose.igpu.yml exec capswriter-server id
```

Put the numeric node GIDs in `CAPSWRITER_DRI_RENDER_GID` and
`CAPSWRITER_DRI_VIDEO_GID`, then recreate. The image contains Mesa's Vulkan ICD,
but a missing `/dev/dri`, unavailable host driver, or wrong supplementary group
still makes the GPU probe fail.

For CPU-only hosts, set:

```dotenv
CAPSWRITER_INFERENCE_HARDWARE=cpu
```

Use only `docker-compose.yml`; do not include `docker-compose.gpu.yml` or
`docker-compose.igpu.yml`. Device exposure/reservation is a Docker admission
decision and cannot fall back inside the container if the host runtime or
device node is unavailable.

### Readiness says ffmpeg or model is unavailable

- Confirm ffmpeg is installed in a source deployment or present in the image.
- Base Compose uses the `capswriter-server-models` named volume. If the explicit
  `docker-compose.models-bind.yml` override is present, confirm `./models` is
  writable by the image's `appuser`, including lock/staging/marker creation.
- If multiple containers share model storage, confirm the filesystem provides
  coherent POSIX advisory locks and inspect
  `CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT` failures before retrying.
- Confirm `CAPSWRITER_MODEL_TYPE` and tuning values are valid; startup rejects
  malformed values instead of silently choosing a different model.
- Inspect recognizer-child logs rather than retrying only `/health`.

## HTTP status and contract problems

| Status/symptom | Meaning | Next check |
|---|---|---|
| Connection refused | Nothing reachable at client host/port | Publish/bind addresses, container state, firewall, browser-visible hostname |
| `400` | Invalid multipart field, model/format/value, unsupported option, or undecodable/too-short audio | Start with one small file, `model=whisper-1`, `response_format=text`, no optional fields |
| `401` | Bearer token missing or invalid | Use the same trimmed token/key file; do not embed credentials in the URL |
| `413` | Declared/raw/file upload or decoded-duration limit exceeded | Read `/ready` limits; reduce/split audio or deliberately raise bounded server limits |
| `429` | Active work plus bounded pending queue is full | Honor `Retry-After`, reduce concurrency, or tune measured capacity |
| `500` mentioning decoder/internal recognition | ffmpeg unavailable/failure or recognizer error | Server logs, ffmpeg path, model child, bounded error preview |
| `501` on translations | Intentional unsupported endpoint | Use transcription; local translation is not implemented |
| `504` | End-to-end task deadline expired | Queue depth, decode/inference time, hardware, then bounded timeout configuration |
| Invalid JSON from a JSON endpoint | Wrong upstream/proxy/path or old incompatible server | Inspect status/content type and confirm API root; do not parse proxy HTML as a transcript |
| Redirect response | Client intentionally does not follow redirects with credentials | Configure the final trusted API URL explicitly |

The HTTP API rejects unsupported capabilities rather than silently ignoring
them. Confirm the documented [API compatibility surface](openai-api.md) before
filing a server bug.

## Web Console problems

| Symptom | Action |
|---|---|
| CORS error | Add the exact browser origin to `CAPSWRITER_HTTP_API_CORS_ORIGINS`; `localhost` and `127.0.0.1` are different origins |
| Web health works but API diagnostics fail | `/config.js` API root is resolved by the browser; set a browser-reachable URL and verify API publish/firewall |
| Browser microphone denied | Use loopback or HTTPS, check browser/OS permission, and stop other exclusive audio users |
| Transcription rejected before upload | `/ready.config.max_upload_mb` preflight found a file that will exceed server limits |
| TTS has no voice/sound | Browser Web Speech depends on locally installed browser/OS voices and user-gesture/audio policy |
| Default API key does not appear | Expected unless both a key and `CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true` are set; entering it in UI is safer |
| Old result appears after cancel/change | Capture version and reproduction; current client guards stale results, so this is a regression candidate |

Use `npm run browser-smoke` only after unit/build checks and when
`agent-browser` is installed. It uses a mock API; a passing smoke does not prove
a real model or microphone.

## CLI problems

- Use `--base-url` with an absolute HTTP(S) root, optionally ending in `/v1`.
  Credentials, query strings, fragments, and other schemes are rejected.
- Prefer `--key-file` or `CAPSWRITER_HTTP_API_KEY_FILE` for persistent use.
- `--timeout` and `--max-response-mb` are bounded; increase them only after
  measuring a legitimate long request.
- Batch output refuses portable-name collisions before requests. Rename source
  files or choose separate output directories instead of bypassing the guard.
- Local `speak` requires PowerShell System.Speech on Windows or a supported
  local Linux speech command. It does not call the CapsWriter server.

## TUI problems

| Symptom | Action |
|---|---|
| Verifier reports dependency mismatch | Recreate the venv and install `requirements-tui.lock` with `--require-hashes --only-binary=:all:` |
| **FILE ONLY** | Core operation is healthy; optional `sounddevice`/PortAudio/device stack is unavailable |
| F5 health works but model is degraded | Read each diagnostic line; do not transcribe until readiness/model state is understood |
| Audio path rejected | Use an existing file visible to the TUI process; container paths and host paths are different |
| Save rejected | Destination must differ from source audio and be outside the private recording directory |
| Terminal layout cramped | Use at least 100×30 cells when possible; narrower terminals switch to stacked layout |

See the [TUI guide](tui.md) for keys, bounds, privacy, and screenshot provenance.

## Safe evidence collection

Record:

- source commit or immutable image digest;
- OS/session (`Windows`, `X11`, `Wayland`, headless), Python version, and client;
- sanitized `.env`/Compose effective values relevant to the failure;
- `/health` and `/ready` responses with secrets removed;
- exact command, status code, response format, and bounded log excerpt;
- model type, CPU/GPU selection, and whether a small known file succeeds.

Do **not** post API keys, key-file contents, private transcripts/prompts,
recordings, full browser localStorage, signed URLs, or unrestricted environment
dumps. Preserve originals privately if a maintainer needs a minimal redacted
reproducer.

If the failure is a suspected vulnerability, follow the private-reporting
guidance in [support and security](support-security.md) rather than opening a
public exploit report.
