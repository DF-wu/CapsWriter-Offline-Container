# fork-v1.0.0-rc.1 release notes

> [Maintenance policy](maintenance.md) · [繁體中文](../zh-TW/release-notes.md) · [Project README](../../README.en.md)

Release-candidate date: **2026-07-18**. This is a source-only GitHub
pre-release for the isolated legacy maintenance line. It is not a v2 release
and does not claim completed real-device/model qualification.

## Deliverables by role

### Server

- Legacy Linux bare-metal and Docker server source.
- WebSocket service on port `6016`.
- Optional, transcription-only OpenAI-compatible HTTP API on port `6017`.
- Model bootstrap, GPU preference/CPU fallback, health checks, and persistent
  model/hotword/log paths.
- Defensive WebSocket, decoded-audio, task/context, multipart, authentication,
  cancellation, routing, and error/logging bounds.

The release does **not** publish a v1 container image. Compose builds
`capswriter-offline-v1-local:source` from this tagged source. The public
`capswriter-offline-server:latest` image belongs to v2.

### Client

- Compatibility-preserved upstream-era Windows desktop source:
  `start_client.py`, tray, hotkeys, microphone, clipboard, and text injection.
- The desktop client connects to the v1 server over WebSocket `6016`.

The release does **not** attach a Windows executable. It also does not contain
the v2 Web Console, no-GUI CLI, Textual TUI, or universal Windows package.

### External API callers

Compatible OpenAI SDK/curl callers may repoint their base URL to the documented
`whisper-1` file-transcription subset. This endpoint is a server interface, not
a bundled v1 client application. Translation and the complete OpenAI Audio API
are not implemented.

## Maintenance changes

- Bound WebSocket frames, decoded chunks, segmentation geometry, task IDs, and
  context without changing valid-client wire format.
- Isolate recognition state by connection so identical task IDs cannot merge
  or misroute transcripts.
- Authenticate and media-type-check rejected HTTP requests before multipart
  parsing; bound uploads and clean cancellation state.
- Redact reflected secrets and private exception detail from errors/logs.
- Correct subtitle timestamp carry behavior.
- Update security-sensitive runtime dependencies and focused regressions.
- Remove the unsafe v1 documentation/default reference to the v2 `latest`
  image; v1 now builds its own local source image.

## Automated evidence

- Ubuntu 24.04 and Windows 2022.
- Python 3.10 and 3.12.
- Every matrix leg runs the full maintenance test suite and compile checks.
- Compose and entrypoint validation run in the Ubuntu 24.04 / Python 3.10
  validation job.
- Duplicate push/PR matrices passed at the maintenance source baseline.

## Still required before a stable v1 release

- Disposable v1 image build and cold model bootstrap.
- Model-backed Mandarin and English known-audio transcription.
- Target CPU/GPU backend and driver/runtime evidence.
- Real Windows desktop launch/exit, tray, hotkey, microphone, clipboard, FFmpeg,
  model, and child-cleanup validation.
- A separately reviewed immutable v1 image workflow before advertising any v1
  container image.

Use [fork v2](https://github.com/DF-wu/CapsWriter-Offline-Container) for active
development and the Web/CLI/TUI/universal package surfaces.
