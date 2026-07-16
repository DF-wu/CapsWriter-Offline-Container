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
| Upstream divergence | `python scripts/check_upstream_divergence.py` | Compares the configured upstream base directly with the working tree, so committed, staged, and unstaged tracked edits must stay within the documented 59 paths; skips cleanly in shallow checkouts without that ref; bounds each git subprocess and reports git failures without a traceback |
| CLI | `python client/cli/scripts/verify.py` | CLI syntax, health/readiness/models calls, key-file auth including empty-file rejection, one server-aligned wall-clock deadline across connection/upload/body reads (including slow-drip rejection), positive timeout validation, bounded HTTP response body reads with configurable `--max-response-mb`, base URL validation before requests, streamed multipart upload including filename escaping and early error responses, mock HTTP transcription, valid JSON output files, atomic output writes with temp cleanup and per-item durable batch output, portable sanitized `--output-dir` target generation and duplicate/case-collision rejection before requests, HTTP and invalid JSON diagnostics, output files, Linux/Windows TTS command selection, direct/file/stdin TTS input handling, bounded local TTS timeout handling, bounded standalone verifier subprocesses, zipapp packaging and packaged stdin smoke |
| Server | `python -m compileall fork_server docker/server check_http_api.py start_server_docker.py` + HTTP and Docker server unit tests | HTTP sidecar, strict Bearer header parsing, server/client key-file auth support, translations endpoint auth consistency, bounded prompt and language hints, late canceled/timeout HTTP result absorption, cancellation-resilient ffmpeg reap and listener cotask teardown, HTTP-aware `ws_send` upstream drift guard, Docker entrypoint/healthcheck syntax, readiness-gated container healthcheck, privacy-preserving transcript logging, SRT/VTT timestamp formatting, bounded ffmpeg decode diagnostics, finite timeout validation, and post-timeout kill cleanup, model downloader diagnostics including bounded atomic archive downloads, safe archive extraction traversal/link guards, schema-2 SHA-256 model/runtime readiness and same-size corruption rejection, runtime-linked llama libraries, bounded configured-backend probes, mandatory GPU-bootstrap/probe-to-CPU reconfiguration and second probe, dependency-light request limit, runtime config tests, and fail-fast server/model env validation |
| Verifier/diagnostic | `python -m unittest discover -s scripts/tests -v` | Root gate helper behavior, including upstream divergence guard parsing/filtering, bounded git subprocess failures, bounded/redacted root verifier subprocess failures, redaction of live HTTP API keys from shared logs, key-file pass-through, empty-file rejection, cleanup residue detection, Windows 3.12 x86-64 hash-lock completeness, fail-closed portable PyInstaller payload and both-entrypoint self-check contracts, Windows package workflow relocation/reparse/upload guards, bounded Web clean subprocess handling, bounded Web verifier subprocess handling, bounded browser-smoke subprocess/HTTP probe/cleanup source guards, Web Nginx security-header source guards, top-level Web dependency install and Compose port-mapping docs, GHCR publish workflow release-gate/permission/attestation/action-pin/runner-pin/checkout-credential source guards, Dockerfile base-image digest, Python bootstrap package pin, Web Docker lockfile install, Docker runtime dependency version/hash lock, and build-context ignore source guards, release ZIP packaging timeout/temp-file cleanup guards, desktop window detector subprocess timeout guards, standalone hotword Ollama request timeout and notebook source guards, GUI tray detached process launch guards, GUI recorder MP3 finalize timeout/cleanup guards, GUI file transcription media subprocess timeout/cleanup guards, GGUF export HTTP request timeout guards, direct engine ffmpeg decode timeout/cleanup/error-preview guards, server GPU boost command timeout guards, server worker shutdown timeout/cleanup guards, plus diagnostic host/port validation, streamed multipart upload escaping, bounded diagnostic response body reads, real HTTP body delivery, POST 401 handling, server-aligned default and configurable live transcription timeout, Docker Compose HTTP/model/resource env, Web public API-key opt-in, loopback publish, no-new-privileges, and server capability-drop guards, HTTP API dependency guards, role-template secret/default guards, `/health` 401 API-key guidance, cleanup traversal pruning, and source guard for configured HTTP ffmpeg decode timeout |
| Web | `npm ci --no-audit --no-fund` then `npm run verify` in `client/web` | React/Vite tests, bounded internal Web verifier subprocesses, bounded Web API response body reads, bounded Web API error/invalid JSON/diagnostic request timeout and caller-abort handling, bounded transcription request timeout and caller-abort handling, API root validation before requests, partial readiness diagnostics when model listing fails, StrictMode-safe mounted guards for dev/browser-smoke diagnostics, keyboard-accessible audio upload, readiness upload-size preflight before preview/transcription, stable drag/drop highlighting, transcription-time audio replacement locking, stale transcription/diagnostic result suppression after cancel/unmount, recording resource cleanup including delayed `getUserMedia` success/failure, deferred export URL cleanup, sanitized download filenames including Windows reserved device names, TTS voice handler/lifecycle cleanup and late callback guards, clipboard failure handling and late result cleanup, best-effort localStorage persistence, bounded settings controls and malformed settings/history/runtime-config recovery, runtime `/config.js` escaping and public API-key opt-in validation, TypeScript, production build, web clean script |
| Optional Web browser smoke | temporary mock API + Vite + `agent-browser` | Real browser can check server health, upload audio, and transcribe through the UI; `agent-browser` subprocesses and temporary child-service shutdown are bounded |
| Optional Web image | `docker build` + temporary `docker run` smoke check | Production Nginx/static image can build and serve `/health` + runtime `/config.js`; smoke also checks baseline security headers |
| Optional live HTTP | `client/cli/capswriter_cli.py health` + optional readiness and known-audio transcription | Real server health, deployment readiness, and model-backed STT when configured |
| Cleanup | `python scripts/clean.py` then `python scripts/clean.py --check` | Removes build/cache/pycache artifacts and fails if known generated residue remains |

The web dependency install is scoped to `client/web/node_modules`. Nothing is installed globally.

The dependency-bearing TUI gate is intentionally separate from the
dependency-light repository gate. A release candidate must run both supported
interpreter endpoints, not one representative version. On POSIX, install the
reviewed lock into two disposable environments and run the strict consumer in
each:

```bash
for version in 3.10 3.12; do
  venv=".venv-tui-${version}"
  "python${version}" -m venv "${venv}"
  "${venv}/bin/python" -m pip install \
    --require-hashes --only-binary=:all: \
    --requirement requirements-tui.lock
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    "${venv}/bin/python" scripts/verify_tui.py
done
```

This verifier checks direct-pin/lock parity, installed versions, imports, and
`pip check`, then runs the complete TUI unit/Pilot suite. Missing dependencies,
zero discovered tests, and skipped tests are failures rather than optional
coverage. See the [English](en/tui.md) or [繁體中文](zh-TW/tui.md) guide for the
runtime workflow and Windows commands.

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

The standalone hotword notebook mirrors the same timeout helper and has a source guard to catch stale unbounded `requests.post` examples.

GUI recorder MP3 finalization bounds the local `ffmpeg` encoder wait with `CAPSWRITER_CLIENT_AUDIO_FINISH_TIMEOUT`, default `30` seconds. Timeout attempts a bounded kill cleanup after recorder stdin is closed.

GUI file transcription bounds local `ffprobe` duration probing and `ffmpeg` audio streaming subprocesses with `CAPSWRITER_CLIENT_MEDIA_TIMEOUT`, default `120` seconds. Timeout attempts a bounded kill cleanup before the client returns from the send/probe path.

GGUF export utilities for remote Hugging Face safetensor metadata/range reads bound `requests.get` and `requests.head` with `CAPSWRITER_GGUF_EXPORT_HTTP_TIMEOUT`, default `60` seconds. Invalid timeout configuration is rejected before opening the request.

GUI tray restart and hotword-file default-opener subprocesses are launched with detached stdio/session settings. This keeps user-triggered helper processes from inheriting console, verifier, or parent-process handles.

Direct engine file-decode helpers for Qwen, force-aligner, SenseVoice, and Fun-ASR bound `ffmpeg` subprocesses with `CAPSWRITER_ENGINE_FFMPEG_TIMEOUT`, default `120` seconds. Timeout attempts a bounded kill cleanup and ffmpeg failure diagnostics are capped before being surfaced.

Server GPU boost/unboost shell commands are bounded by `CAPSWRITER_GPU_BOOST_TIMEOUT`, default `5` seconds. Timeout kills the complete POSIX process group or bounded Windows process tree and reaps the shell; invalid timeout configuration and nonzero exit leave the current boost state unchanged instead of blocking or falsely advancing the worker state.

Docker model readiness uses schema-2 `.capswriter-model-ready.json` and
`.capswriter-llama-ready.json` metadata. The warm path re-hashes required model
artifacts and every installed llama `.so`, then compares size/SHA-256 manifests,
selected backend, archive identity, and archive SHA-256 exactly. The Docker helper
suite covers missing/invalid markers, same-size content corruption, backend/runtime
mismatch, transactional repair, and the read-only fully warm fast path.

Configured backend construction runs in a supervised child with
`CAPSWRITER_BACKEND_PROBE_TIMEOUT` (default `300` seconds, valid range `> 0`
through `1800`). A selected GPU runtime bootstrap failure or configured GPU
probe failure must disable every CUDA/Vulkan path, rerun model/runtime preparation
for CPU, and pass a second CPU probe. Timeout or failure of that second probe is
a startup failure, not a successful fallback.

Recognizer model loading is bounded by `CAPSWRITER_SERVER_MODEL_LOAD_TIMEOUT`, default `600` seconds. Invalid explicit values fail before a Manager or child is created; timeout marks readiness unhealthy, terminates/kills/reaps the live child, and synchronously fails startup so the service supervisor can restart it.

Server recognizer worker shutdown waits up to `CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT`, default `2` seconds, for the worker to consume the graceful sentinel. If it remains alive, shutdown escalates through bounded terminate and final kill waits.

The parent also watches every synchronous inference lease. An HTTP inference still active two seconds beyond its request deadline, or any inference active beyond `CAPSWRITER_SERVER_WORKER_STALL_TIMEOUT` (default `900` seconds), is treated as a wedged native call: the child is reaped with bounded terminate/kill and the whole server fail-stops for supervisor restart.

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

This uses a run-unique, ownership-labelled temporary image and container, checks `/health`, `/config.js`, and baseline security headers, then removes only those owned resources during cleanup.

Run the browser-level Web Console smoke:

```bash
python scripts/verify_all.py --web-browser-smoke
```

This requires `npx agent-browser` to be available. It starts temporary local services on free ports and removes browser/test artifacts through the normal cleanup path.
Each `agent-browser` command is bounded by `CAPSWRITER_WEB_BROWSER_AGENT_TIMEOUT_MS`, default `30000` ms. HTTP startup probes inside the smoke script are bounded by `CAPSWRITER_WEB_BROWSER_HTTP_PROBE_TIMEOUT_MS`, default `2000` ms per attempt. Temporary mock API/Vite child processes are first asked to stop, then force-stopped if they do not exit within `CAPSWRITER_WEB_BROWSER_CHILD_SHUTDOWN_TIMEOUT_MS`, default `5000` ms.

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

CI and publish jobs pin the hosted runner label to `ubuntu-24.04` instead of
the moving `ubuntu-latest` alias, and checkout steps disable persisted git
credentials because the workflows do not push through the checked-out remote.
The explicit upstream fetch makes the low-drift guard fail CI when the checked-out
fork tree touches a new upstream-tracked file without documenting that divergence;
the same command also catches staged or unstaged tracked drift during local review.

The same workflow has an unconditional `tui` matrix job for Python 3.10 and
3.12. Each matrix leg creates a `$RUNNER_TEMP` virtual environment, installs
`requirements-tui.lock` with `--require-hashes --only-binary=:all:`, and runs
`scripts/verify_tui.py` with user-site packages and bytecode writes disabled.
The matrix does not infer TUI support from the dependency-light root job.

[`portability.yml`](../.github/workflows/portability.yml) has two independent
gates. `core-cli` runs the dependency-light desktop/package/CLI contracts on
Ubuntu 24.04 and Windows 2022 with Python 3.10 and 3.12. `windows-package` runs
on Windows 2022 with Python 3.12, installs
`requirements-windows-build.lock` with hashes and the wheel-only policy (apart
from the documented `srt` exception), builds both executables, copies the
distribution outside the checkout, ZIP-round-trips it, rejects reparse points
and nonempty mutable directories, runs bounded `--artifact-self-check` through
both extracted EXEs, and uploads the exact tested ZIP. This proves packaging,
relocation, layout, and importability; it does not start a normal server/client,
load models, or exercise tray, hooks, audio, FFmpeg, known-audio recognition,
DirectML, or GPU execution.

The publish workflows keep their own release gates before pushing GHCR images. [`publish-server-image.yml`](../.github/workflows/publish-server-image.yml) first runs `CAPSWRITER_UPSTREAM_BASE=upstream/master python scripts/verify_all.py --skip-web`, then builds and pushes the immutable `linux/amd64` server candidate. Before promotion, it pulls that exact digest and verifies `pip check`, runtime imports, non-root execution, and entrypoint syntax. [`publish-web-image.yml`](../.github/workflows/publish-web-image.yml) first runs `CAPSWRITER_UPSTREAM_BASE=upstream/master python scripts/verify_all.py --docker-build-web`, including the production Nginx image smoke, then builds and pushes the static Web Console image. Each workflow uses a fixed, non-cancelling per-ref concurrency group, pushes only the full `sha-<commit>` tag from the long build job, and moves `latest` in a separate 10-minute promotion job only after re-reading the current `master` tip. Promotion uses the exact build digest and verifies the resulting registry digest; a historical rerun can therefore publish at most its immutable SHA tag. Only the immutable-publish and promotion jobs receive `packages: write`, all third-party workflow actions are pinned to full commit SHAs, and `docker/build-push-action` publishes provenance and SBOM attestations with each image.

Both verification and publication jobs also require `refs/heads/master`, so a
manual dispatch from a feature branch cannot overwrite the production
`latest` tag.

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
| TUI dependency lock and Textual/Pilot suite valid | isolated `scripts/verify_tui.py` logs on Python 3.10 and 3.12, with zero skips |
| Portable Windows production package valid | `windows-package` hash-install/build/relocation/reparse inspection and both-EXE self-check logs, uploaded ZIP, and its digest |
| Docker asset integrity and fallback contract valid | Docker helper tests for schema-2 model/runtime manifests, same-size corruption rejection, bounded backend probes, and mandatory CPU rebootstrap/second probe |
| Real HTTP server reachable and ready | `--http-base-url --http-require-ready` gate output or `check_http_api.py` output |
| Model-backed STT sample works | `--http-audio` + `--http-expect` gate output or `check_http_api.py --audio ... --expect ... --timeout ...` |
| No generated trash committed | `git status --short` plus `python scripts/clean.py --check` |

Default CI does not download models or commit binary audio fixtures. Use `--http-audio` for release-candidate evidence when a real server and known sample are available.
