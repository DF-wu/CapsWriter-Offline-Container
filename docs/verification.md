# Verification And Cleanup

This fork now has a repository-level verification gate for production readiness work. It is intentionally composed from smaller isolated checks so server, no-GUI CLI, and Web Console behavior can be tested without installing global dependencies or leaving generated files behind.

![CapsWriter verification pipeline](assets/verification-pipeline.svg)

## One-command local gate

```bash
python scripts/verify_all.py
```

The command runs:

| Step | Command | Coverage |
|---|---|---|
| Upstream divergence | `python scripts/check_upstream_divergence.py` | Confirms fork commits only modify the documented upstream-tracked files when the configured base ref is available; skips cleanly in shallow checkouts without that ref; bounds each git subprocess and reports git failures without a traceback |
| CLI | `python client/cli/scripts/verify.py` | CLI syntax, health/readiness/models calls, key-file auth including empty-file rejection, server-aligned default timeout and positive timeout validation, bounded HTTP response body reads with configurable `--max-response-mb`, base URL validation before requests, streamed multipart upload including filename escaping and early error responses, mock HTTP transcription, valid JSON output files, atomic output writes with temp cleanup and per-item durable batch output, portable sanitized `--output-dir` target generation and duplicate/case-collision rejection before requests, HTTP and invalid JSON diagnostics, output files, Linux/Windows TTS command selection, direct/file/stdin TTS input handling, bounded local TTS timeout handling, bounded standalone verifier subprocesses, zipapp packaging and packaged stdin smoke |
| Server | `python -m compileall fork_server docker/server check_http_api.py start_server_docker.py` + HTTP and Docker server unit tests | HTTP sidecar, strict Bearer header parsing, server/client key-file auth support, translations endpoint auth consistency, bounded prompt and language hints, late canceled/timeout HTTP result absorption, HTTP-aware `ws_send` upstream drift guard, Docker entrypoint/healthcheck syntax, readiness-gated container healthcheck, privacy-preserving transcript logging, SRT/VTT timestamp formatting, bounded ffmpeg decode diagnostics, timeout validation, and post-timeout kill cleanup, model downloader diagnostics including bounded atomic archive downloads, safe archive extraction traversal/link guards, and runtime-linked llama libraries, dependency-light request limit, runtime config tests, and fail-fast server/model env validation |
| Verifier/diagnostic | `python -m unittest discover -s scripts/tests -v` | Root gate helper behavior, including upstream divergence guard parsing/filtering, bounded git subprocess failures, bounded/redacted root verifier subprocess failures, redaction of live HTTP API keys from shared logs, key-file pass-through, empty-file rejection, cleanup residue detection, bounded Web clean subprocess handling, bounded Web verifier subprocess handling, bounded browser-smoke subprocess/cleanup source guards, release ZIP packaging timeout/temp-file cleanup guards, desktop window detector subprocess timeout guards, standalone hotword Ollama request timeout guards, GUI recorder MP3 finalize timeout/cleanup guards, GUI file transcription media subprocess timeout/cleanup guards, GGUF export HTTP request timeout guards, direct engine ffmpeg decode timeout/cleanup/error-preview guards, server GPU boost command timeout guards, server worker shutdown timeout/cleanup guards, plus diagnostic host/port validation, streamed multipart upload escaping, bounded diagnostic response body reads, real HTTP body delivery, POST 401 handling, server-aligned default and configurable live transcription timeout, Docker Compose HTTP/model/resource env guards, HTTP API dependency guards, role-template secret/default guards, `/health` 401 API-key guidance, cleanup traversal pruning, and source guard for configured HTTP ffmpeg decode timeout |
| Web | `npm ci --no-audit --no-fund` then `npm run verify` in `client/web` | React/Vite tests, bounded internal Web verifier subprocesses, bounded Web API response body reads, bounded Web API error/invalid JSON/diagnostic request timeout and caller-abort handling, API root validation before requests, partial readiness diagnostics when model listing fails, StrictMode-safe mounted guards for dev/browser-smoke diagnostics, keyboard-accessible audio upload, stable drag/drop highlighting, transcription-time audio replacement locking, stale transcription/diagnostic result suppression after cancel/unmount, recording resource cleanup including delayed `getUserMedia` success/failure, deferred export URL cleanup, sanitized download filenames including Windows reserved device names, TTS voice handler/lifecycle cleanup and late callback guards, clipboard failure handling and late result cleanup, best-effort localStorage persistence, bounded settings controls and malformed settings/history/runtime-config recovery, runtime `/config.js` escaping, TypeScript, production build, web clean script |
| Optional Web browser smoke | temporary mock API + Vite + `agent-browser` | Real browser can check server health, upload audio, and transcribe through the UI; `agent-browser` subprocesses and temporary child-service shutdown are bounded |
| Optional Web image | `docker build` + temporary `docker run` smoke check | Production Nginx/static image can build and serve `/health` + runtime `/config.js` |
| Optional live HTTP | `client/cli/capswriter_cli.py health` + optional readiness and known-audio transcription | Real server health, deployment readiness, and model-backed STT when configured |
| Cleanup | `python scripts/clean.py` then `python scripts/clean.py --check` | Removes build/cache/pycache artifacts and fails if known generated residue remains |

The web dependency install is scoped to `client/web/node_modules`. Nothing is installed globally.

## Options

Skip web verification:

```bash
python scripts/verify_all.py --skip-web
```

Use existing `client/web/node_modules` and fail if it is missing:

```bash
python scripts/verify_all.py --no-web-install
```

Each subprocess launched by the root gate is bounded by `CAPSWRITER_VERIFY_STEP_TIMEOUT`, which defaults to `1800` seconds. Set a larger positive value on unusually slow release machines:

```bash
CAPSWRITER_VERIFY_STEP_TIMEOUT=3600 python scripts/verify_all.py
```

Within `client/web`, `npm run verify` also bounds each internal `npm run test/build/clean` step with `CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT`, which defaults to `600` seconds:

```bash
CAPSWRITER_WEB_VERIFY_STEP_TIMEOUT=1200 npm run verify
```

Legacy PyInstaller ZIP packaging through `zip_release.py` bounds each 7-Zip create/list subprocess with `CAPSWRITER_ZIP_RELEASE_TIMEOUT`, default `900` seconds, and removes generated `file_list_*.txt` files even if packaging fails.

Desktop foreground-window detection bounds macOS `osascript` and Linux `wmctrl` subprocesses with `CAPSWRITER_WINDOW_DETECT_TIMEOUT`, default `2` seconds. Detection falls back to empty window info on timeout or invalid local configuration, preserving the existing fail-safe output behavior.

Standalone hotword Ollama chat calls bound the local `http://localhost:11434/api/chat` request with `CAPSWRITER_OLLAMA_CHAT_TIMEOUT`, default `30` seconds. Invalid local timeout configuration is reported through the helper's existing fail-safe error path before opening the request.

GUI recorder MP3 finalization bounds the local `ffmpeg` encoder wait with `CAPSWRITER_CLIENT_AUDIO_FINISH_TIMEOUT`, default `30` seconds. Timeout attempts a bounded kill cleanup after recorder stdin is closed.

GUI file transcription bounds local `ffprobe` duration probing and `ffmpeg` audio streaming subprocesses with `CAPSWRITER_CLIENT_MEDIA_TIMEOUT`, default `120` seconds. Timeout attempts a bounded kill cleanup before the client returns from the send/probe path.

GGUF export utilities for remote Hugging Face safetensor metadata/range reads bound `requests.get` and `requests.head` with `CAPSWRITER_GGUF_EXPORT_HTTP_TIMEOUT`, default `60` seconds. Invalid timeout configuration is rejected before opening the request.

Direct engine file-decode helpers for Qwen, force-aligner, SenseVoice, and Fun-ASR bound `ffmpeg` subprocesses with `CAPSWRITER_ENGINE_FFMPEG_TIMEOUT`, default `120` seconds. Timeout attempts a bounded kill cleanup and ffmpeg failure diagnostics are capped before being surfaced.

Server GPU boost/unboost shell commands are bounded by `CAPSWRITER_GPU_BOOST_TIMEOUT`, default `5` seconds. Timeout or invalid local timeout configuration leaves the current boost state unchanged instead of blocking the worker loop.

Server recognizer worker shutdown waits up to `CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT`, default `2` seconds, for the worker to consume the graceful sentinel. If it remains alive, shutdown escalates through bounded terminate and final kill waits.

Add a live server health check:

```bash
python scripts/verify_all.py --http-base-url http://127.0.0.1:6017
```

Require the live server readiness endpoint as well:

```bash
python scripts/verify_all.py \
  --http-base-url http://127.0.0.1:6017 \
  --http-require-ready
```

Add a model-backed transcription smoke check with a known audio sample:

```bash
python scripts/verify_all.py \
  --http-base-url http://127.0.0.1:6017 \
  --http-audio /path/to/known-speech.wav \
  --http-expect "expected transcript text"
```

If `--http-expect` is omitted, the gate only requires a non-empty transcript. Keep release-candidate sample audio outside Git unless it is small, redistributable, and intentionally part of the repository.

Build the Web Console production image as part of the gate:

```bash
python scripts/verify_all.py --docker-build-web
```

This uses the temporary image tag `capswriter-web-console:verify`, starts a temporary container named `capswriter-web-console-verify`, checks `/health` and `/config.js`, then removes both during cleanup.

Run the browser-level Web Console smoke:

```bash
python scripts/verify_all.py --web-browser-smoke
```

This requires `npx agent-browser` to be available. It starts temporary local services on free ports and removes browser/test artifacts through the normal cleanup path.
Each `agent-browser` command is bounded by `CAPSWRITER_WEB_BROWSER_AGENT_TIMEOUT_MS`, default `30000` ms. Temporary mock API/Vite child processes are first asked to stop, then force-stopped if they do not exit within `CAPSWRITER_WEB_BROWSER_CHILD_SHUTDOWN_TIMEOUT_MS`, default `5000` ms.

With auth:

```bash
python scripts/verify_all.py \
  --http-base-url http://127.0.0.1:6017 \
  --http-key sk-local-dev
```

Secret-file alternative:

```bash
python scripts/verify_all.py \
  --http-base-url http://127.0.0.1:6017 \
  --http-key-file /run/secrets/capswriter-http.key
```

Environment alternative:

```bash
CAPSWRITER_VERIFY_HTTP_BASE=http://127.0.0.1:6017 \
CAPSWRITER_VERIFY_HTTP_REQUIRE_READY=true \
CAPSWRITER_VERIFY_HTTP_AUDIO=/path/to/known-speech.wav \
CAPSWRITER_VERIFY_HTTP_EXPECT="expected transcript text" \
CAPSWRITER_HTTP_API_KEY_FILE=/run/secrets/capswriter-http.key \
python scripts/verify_all.py
```

## Cleanup

```bash
python scripts/clean.py
python scripts/clean.py --check
```

Cleanup removes:

| Path pattern | Reason |
|---|---|
| `__pycache__`, `*.pyc` | Python verification output |
| `client/cli/dist` | Packaged no-GUI CLI zipapp output |
| `client/web/dist` | Vite production build output |
| `client/web/.vite`, `client/web/node_modules/.vite` | Vite cache |
| `coverage`, `htmlcov`, `playwright-report`, `test-results` | Test/report output |
| `.drawio-tmp` | Diagram sidecars generated during local authoring |
| TypeScript emitted config artifacts | Guard against accidental `tsc -b` output |

`client/web/node_modules` and `models` are not removed by default. They are isolated dependency/model directories and are ignored by Git; removing or walking them on every verification would force unnecessary downloads or slow cleanup on large model installs. Delete them manually if a completely fresh dependency or model install is required.

If `client/web/package.json` and `npm` are available, cleanup first runs `npm run clean` in `client/web`. That subprocess is bounded by `CAPSWRITER_CLEAN_WEB_TIMEOUT`, which defaults to `120` seconds; the Python cleanup fallback still removes known generated paths afterward.

`scripts/verify_all.py` runs cleanup in a `finally` block, then runs
`scripts/clean.py --check`. A successful root gate therefore proves the known
verification residue was removed, not just that cleanup was attempted.

## CI

[`ci.yml`](../.github/workflows/ci.yml) runs on push, pull request, and manual dispatch:

```text
checkout -> fetch HaujetZhao upstream base -> setup Python 3.12 -> setup Node 24
  -> CAPSWRITER_UPSTREAM_BASE=upstream/master python scripts/verify_all.py
```

The explicit upstream fetch makes the low-drift guard fail CI when fork commits
touch a new upstream-tracked file without documenting that divergence.

The publish workflows remain separate. CI verifies source, tests, and local builds; [`publish-server-image.yml`](../.github/workflows/publish-server-image.yml) builds the server image and [`publish-web-image.yml`](../.github/workflows/publish-web-image.yml) builds the static Web Console image when maintainers choose to publish.

## Evidence expected before release

For a release candidate, keep these artifacts or logs:

| Requirement | Evidence |
|---|---|
| Upstream merged | Git merge commit in branch history |
| Server syntax, Docker server helpers, and HTTP sidecar valid | `python scripts/verify_all.py` logs |
| Shared gate logs do not expose API keys | `scripts/tests` output from inside the root gate |
| Web Console build valid | `npm run verify` logs from inside the root gate |
| Web Console browser workflow valid | `--web-browser-smoke` gate output |
| CLI valid | `client/cli/scripts/verify.py` logs from inside the root gate |
| Real HTTP server reachable and ready | `--http-base-url --http-require-ready` gate output or `check_http_api.py` output |
| Model-backed STT sample works | `--http-audio` + `--http-expect` gate output or `check_http_api.py --audio ... --expect ... --timeout ...` |
| No generated trash committed | `git status --short` plus `python scripts/clean.py --check` |

Default CI does not download models or commit binary audio fixtures. Use `--http-audio` for release-candidate evidence when a real server and known sample are available.
