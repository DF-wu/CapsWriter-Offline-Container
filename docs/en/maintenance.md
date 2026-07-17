# Legacy v1 Maintenance, Versioning, and Support

[繁體中文版](../zh-TW/maintenance.md)

## Status and branch model

This repository keeps a separate compatibility line for deployments that cannot move to the current v2 line yet.

- Development branch: `maintenance/v1`
- Pull request base: `archive/v1-legacy`
- Snapshot at the start of maintenance: `b46ca74`
- Safety tag: `fork-pre-reset-20260525-1411`

The name “v1” identifies the fork's legacy release track. The snapshot itself contains upstream 2.5-alpha-era code plus the fork's Linux server, container, and HTTP API additions, so the internal `__version__` remains `2.5-alpha`. Do not retarget v1 maintenance pull requests to `master`, and do not interpret that internal string as the branch name.

## Maintenance scope

Accepted changes are deliberately narrow:

- security and privacy fixes;
- crash, shutdown, resource-leak, and protocol correctness fixes;
- dependency compatibility required to keep an already-supported path operational;
- regression tests, CI maintenance, and documentation corrections.

New product surfaces, model-family migrations, broad refactors, and features developed for v2 should remain on v2. A backport should be small enough to review independently and should preserve the established WebSocket wire format and Windows desktop behavior.

No end-of-life date or response-time SLA is promised. Fixes are provided on a best-effort basis while the line remains maintainable.

## Support matrix

| Path | Maintenance status | Automated coverage | Important limits |
| --- | --- | --- | --- |
| Windows desktop client | Compatibility-preserved | Windows syntax and dependency-light protocol tests on Python 3.10 and 3.12 | Global hotkeys, tray, microphone, clipboard, and PyInstaller output require a real Windows validation host before release. |
| Linux Docker server | Primary legacy server path | Ubuntu syntax, protocol/server unit tests, Compose validation, and entrypoint shell validation; image runtime uses Ubuntu 22.04 / Python 3.10 | Model download, GPU providers, and model-backed inference are not exercised by the lightweight PR gate. |
| Linux bare-metal server | Best effort | Same Python 3.10/3.12 dependency-light server tests | Operators own FFmpeg, model files, native libraries, and service supervision. |
| macOS client/server | Not a release-qualified path | Syntax may compile, but there is no macOS CI job | Permissions and input/audio integration are not maintained here. |
| OpenAI-compatible HTTP API | Optional legacy server feature | Upload-limit and response-format helper tests | It is transcription-only; translation is not implemented, and model/token timestamps are approximations rather than a full OpenAI service implementation. |

Passing CI means the portable protocol and server control paths passed. It does not certify audio hardware, desktop integration, a GPU backend, model accuracy, or a distributable executable.

“Windows desktop client” means only the upstream-era `start_client.py` source
workflow retained in this tree. This line does not include the v2 Web Console,
no-GUI CLI, Textual TUI, or universal Windows package. Release notes must list
server/API/container deliverables separately from Windows client source/binary
status. Never advertise a Windows download unless the release attaches an
artifact that passed real-Windows qualification.

The current v1 release path is source-only. Compose builds a local v1 image from
the checkout; the public `ghcr.io/df-wu/capswriter-offline-server:latest` tag is
v2 and must never be presented as a v1 image.

## Ingress and security baseline

The maintenance line retains the existing protocol for valid clients and applies these defensive limits:

- WebSocket JSON messages default to an 8 MiB frame limit (`CAPSWRITER_WS_MAX_MESSAGE_MB`).
- One decoded audio chunk may be at most 4 MiB. The official 60-second float32/16 kHz/mono chunk is 3,840,000 bytes and remains valid.
- `seg_duration` must be greater than zero and no more than 300 seconds; `seg_overlap` must be between 0 and 30 seconds. Their byte geometry must resolve to nonzero float32 sample boundaries.
- A task ID is limited to 128 characters and context to 8,192 characters.
- A connection must finalize its active task before changing task ID or source.
- Recognition state is namespaced by connection, so identical client task IDs cannot merge transcripts across WebSocket sessions.
- HTTP uploads are read in bounded chunks and stop once `CAPSWRITER_HTTP_API_MAX_UPLOAD_MB` is exceeded.
- Bearer tokens use constant-time comparison. Bind the HTTP API to loopback unless authentication and a trusted reverse proxy/TLS boundary are configured.
- The transcription middleware checks Bearer authentication and rejects non-multipart request bodies before Starlette invokes its form parser.

These are application safeguards, not an internet-facing security perimeter. Do not expose the unauthenticated WebSocket service directly to an untrusted network.

## Verification

The PR gate runs the following portable checks on Ubuntu and Windows with Python 3.10 and Python 3.12:

```bash
python -m pip install -r requirements-maintenance.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q config_client.py config_server.py core_client.py core_server.py start_client.py start_server.py util docker/server
```

The Linux/Python 3.10 lane additionally validates `docker/server/entrypoint.sh` and the Compose configuration. Release qualification should also include a disposable container build, a model-backed Mandarin and English transcription smoke test, and a real Windows desktop smoke test. Record the model, backend, driver/runtime, sample audio provenance, and result with the release.

## Backport and release procedure

1. Create the change from `maintenance/v1` and compare it only with `archive/v1-legacy`.
2. Keep each backport focused and document any behavior change or new limit.
3. Run the portable gate in an isolated environment. Do not use production models, credentials, or writable production volumes.
4. For runtime-affecting changes, perform the release qualification described above.
5. Open the pull request with base `archive/v1-legacy`; never merge the legacy line into v2 as a whole.
6. Tag legacy releases as `fork-v1.<minor>.<patch>` (or add `-rc.<n>` for a
   pre-release), include the exact source commit, and state separately whether
   the release ships server source, a container image, Windows client source,
   or a qualified Windows binary. Change the internal application version only
   as part of an intentional release decision.

## Known residual risks

- Application middleware authenticates before multipart parsing, but it does not impose a raw-body cap on chunked transfer encoding or multipart overhead. A reverse proxy should enforce request-size and authentication policy when the HTTP API is exposed beyond loopback.
- A small compressed upload can expand into much larger decoded PCM in FFmpeg; upload size alone is not an audio-duration or decoded-memory limit.
- Recognition uses a single worker and multiprocessing queues without a durable job store or strict global pending-job quota.
- Native model libraries, GPU providers, desktop hooks, and packaging have a larger platform-specific dependency surface than the lightweight CI gate covers.
- The legacy dependency set will accumulate upstream end-of-support risk. A dependency jump that requires broad application changes belongs on v2.

Report a vulnerability privately when possible. Do not include API keys, transcripts, audio, model artifacts, or full production logs in a public issue.
