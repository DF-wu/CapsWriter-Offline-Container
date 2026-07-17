# Support and security

> [Documentation home](README.md) · [繁體中文](../zh-TW/support-security.md) · [Release notes](release-notes.md)

This page defines what “supported” means for fork v2 and documents the default
security/privacy posture. It distinguishes automated contract evidence from
manual release evidence.

![CapsWriter verification pipeline covering upstream divergence, portable desktop and CLI contracts, server/API/Docker checks, Web and TUI gates, and cleanup](../assets/verification-pipeline.svg)

Text equivalent: independent checks validate upstream drift, portable Python
contracts, a hash-installed Windows PyInstaller build plus ZIP relocation and
EXE import smoke, server/API/Docker behavior, Web build/browser paths, the
hash-locked TUI, documentation, and generated-artifact cleanup. Real model,
audio, GPU, display, browser-device, tray/hook, and known-audio checks remain
explicit release evidence outside automated CI.

## Support vocabulary

- **Supported:** documented entrypoint and automated portable contract exist;
  release operators collect any listed hardware/artifact evidence.
- **Supported with limitations:** intended use is available, but an important
  behavior differs by platform and is documented below.
- **Not supported:** the fork intentionally refuses or does not implement the
  path; falling back or failing clearly is expected.
- **Not release-gated:** code may happen to run, but this project makes no
  support claim without a pinned automated/manual gate.

## Platform and surface matrix

| Surface | Status | Automated evidence | Required manual/release evidence |
|---|---|---|---|
| Windows desktop package | Supported | Windows 2022/Python 3.12 hash-only dependency install, PyInstaller build, relocation/ZIP extraction, reparse-point inspection, and both-EXE import smoke; four-leg source matrix | Tray, shortcuts, microphone, FFmpeg, model/runtime assets, known audio, optional API, CPU, DirectML, and GPU profiles, plus clean child shutdown on real Windows |
| Linux X11 desktop shortcuts | Supported with limitations | Backend detection, callback mapping, listener wiring, failure downgrade tests | Real X11 session, keyboard/mouse device, chosen shortcut and text-injection workflow |
| Linux Wayland global shortcuts | Not supported | Backend reports unavailable and avoids listener construction | Use X11 or non-global-hotkey Web/CLI/TUI/file workflows |
| Linux headless desktop shortcuts | Not supported | Headless backend reports unavailable | Use container/source server plus non-desktop clients |
| Linux container server (`linux/amd64`) | Supported | Dockerfile/Compose/source guards, unit tests, health/readiness/image smoke gates | Selected model download/load, known audio, intended CPU/GPU host, persistence and restart |
| Linux ARM64 container | Not release-gated | No ARM64 dependency lock, native runtime bundle, image build, or model-load gate | Use a supported `linux/amd64` host; do not infer support from portable Python-only checks |
| CPU inference fallback | Supported | Backend selection/fallback tests and configuration guards | Known-audio latency/quality on target CPU |
| GPU acceleration | Supported where selected backend/driver works | Probe/fallback and configuration tests | Actual driver/device visibility, model load, memory use, known audio, fallback observation |
| OpenAI-compatible transcription subset | Supported when enabled | Dedicated exact-pin API contract job with no skips | Live authenticated deployment and known audio for production release |
| Web Console | Supported on modern Windows/Linux browsers | Unit/type/build, browser mock smoke, static-image smoke | Target browser, microphone permission/secure context when used, real API/model |
| No-GUI CLI | Supported on Windows/Linux Python 3.10 and 3.12 | Four-leg OS/Python portability matrix and packaged zipapp smoke | Target local TTS command/voice when `speak` is required |
| Textual TUI core file workflow | Supported on Windows/Linux Python 3.10–3.12 | Four-leg Ubuntu 24.04/Windows 2022 × Python 3.10/3.12 hash-locked no-skip Pilot matrix | Target terminal rendering; Windows terminal run for a Windows release |
| TUI optional microphone | Supported when native stack/device works | Bounded recorder unit/Pilot contracts | `sounddevice`, PortAudio, permissions, real device and cleanup on target OS |
| macOS product | Not release-gated | Only isolated portable helpers may have tests | No project-level desktop/server/client support claim |

“Windows + Linux fork” means the documented surfaces above, not identical
behavior on every display server, browser, audio device, or accelerator.

## Security defaults

### Network exposure

- The HTTP API is disabled by default.
- Compose publishes WebSocket, optional HTTP, and Web ports on host loopback by
  default.
- The upstream source/desktop WebSocket default is `0.0.0.0`; set
  `CAPSWRITER_SERVER_ADDR=127.0.0.1` with the universal entrypoint for a
  loopback-only desktop server.
- Enabling an API bind outside loopback requires a Bearer key/key file unless
  `CAPSWRITER_HTTP_API_ALLOW_INSECURE_BIND=true` is deliberately set for an
  isolated trusted test network.
- CORS uses an explicit allowlist. It is not a replacement for authentication.
- Clients reject URL credentials, non-HTTP schemes, query strings, and
  fragments in API roots; credential-bearing redirects are not followed.

For remote access, use a maintained TLS reverse proxy or private overlay
network, keep authentication enabled, publish only required ports, and preserve
body/time/concurrency limits.

### Secrets

- Prefer `CAPSWRITER_HTTP_API_KEY_FILE` or a secret mount/service-manager secret
  for long-lived server and CLI deployments.
- TUI keys are masked and kept in memory; there is no command-line key option.
- Web keys entered in the UI remain in page memory. A key injected into
  `/config.js` is public to everyone who can load the page, so the container
  requires the separate `CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true` opt-in.
- `.env`, key files, certificates, and local archives are excluded from the
  Docker build context. Do not commit them.
- Web OpenAI-style, legacy `detail`, non-JSON, malformed-response, and network
  errors are made control-safe and bounded, then stripped of the request's
  in-memory key before React state or rendering. Shared verification logs also
  redact configured keys where they could otherwise be reflected.

Rotate a key after suspected disclosure, then update server and clients during
a controlled restart. Do not paste an active key into an issue or transcript.

### Audio, transcripts, and logs

- Inference is local after assets are present, but first-time model/runtime
  bootstrap may contact configured download sources.
- HTTP prompt/transcript logging is off by default. Enabling it changes the
  privacy boundary and requires a retention/access policy.
- TUI microphone WAV files use a private temporary directory and are removed on
  successful upload, cancellation, replacement, and normal exit; failed upload
  retention is only for an in-session retry.
- CLI/TUI output is written to user-selected paths with atomic replacement
  guards. Source audio is not overwritten by the TUI.
- Web settings and up to 20 plaintext transcript/raw history records are stored
  in browser localStorage; those history records can be sensitive. Manually
  entered API keys are not persisted there. Clear site data on shared machines;
  browser storage is not an encrypted records system.
- Server logs live in a named volume by default. Operators own log access,
  rotation, backup, and deletion.

Do not use private recordings/transcripts as public bug fixtures without
explicit consent and redaction.

### Resource and parser boundaries

The API bounds raw/file upload size, decoded audio duration, active and pending
requests, response formats, multipart fields, ffmpeg output, and one end-to-end
deadline. Clients independently bound timeouts and response bodies. Unsupported
fields/capabilities are rejected rather than silently ignored.

These controls reduce accidental exhaustion and ambiguous behavior; they do not
turn an unauthenticated public endpoint into a safe internet service.

### Container and supply chain

- The server container drops Linux capabilities and enables
  `no-new-privileges`.
- Docker, TUI, and Windows production-build Python dependencies use fully
  resolved SHA-256 hash locks. The server image lock and bundled llama runtime
  target `linux/amd64`; the Windows lock targets CPython 3.12 x86-64 Windows
  and permits only the documented `srt` source exception. Web uses
  `package-lock.json` and `npm ci`. No ARM64 server image is claimed.
- CI runners and third-party actions are pinned; checkout does not retain write
  credentials.
- Server/Web publish workflows gate builds and request provenance/SBOM
  attestations. Before `latest` promotion, the server workflow pulls and
  smoke-tests the exact pushed digest for dependency consistency, runtime
  imports, non-root execution, and entrypoint syntax. Consumers should still
  pin immutable image digests and verify the expected registry/attestations.
- Model assets are runtime data, not proof supplied by a source-code unit test.
  Record their source/version/checksum in controlled deployments.

## Vulnerability reporting

Use the fork repository's private vulnerability-reporting channel when it is
available. Include the affected source commit/image digest, surface, minimal
reproduction, impact, and a proposed disclosure timeline. Do not include live
keys, private audio, or unrelated environment dumps.

If no private channel is enabled, contact the fork maintainers through the
repository's maintainer/contact surface and request a private channel. A public
issue may state that a security report exists, but should not contain exploit
details before a fix and disclosure plan are agreed.

Upstream engine/product vulnerabilities that are not introduced by the fork
should also be coordinated with the upstream project. Fork-specific API,
container, portability, Web, CLI, TUI, or release-pipeline issues belong here.

## Normal support requests

Before filing:

1. Follow the [troubleshooting diagnostic ladder](troubleshooting.md#diagnostic-ladder).
2. Reproduce on a clean supported path when possible.
3. Include the source commit/image digest, platform/session, Python/client,
   model/hardware selection, sanitized readiness response, and bounded logs.
4. State whether a small known file works.
5. Remove keys, recordings, transcripts, prompts, and personal paths unless a
   minimal redacted sample is essential.

Hardware or model-quality reports need target hardware/model/audio evidence;
CI output alone is insufficient.

## Release evidence

Before a supported release claim, retain:

- portable Ubuntu/Windows matrix logs;
- isolated API contract and four-leg Ubuntu/Windows TUI no-skip logs;
- root verification, Web build/browser/image smoke, docs, and cleanup logs;
- the exact uploaded Windows ZIP, its digest, and the hash-install/build/
  relocation/both-EXE self-check workflow results when shipping Windows
  binaries;
- real X11 results when claiming Linux desktop shortcuts;
- live readiness and known-audio transcription on each advertised CPU/GPU/model
  profile;
- immutable source/image references and dependency/model provenance;
- rollback instructions and known limitations.

See [verification](../verification.md) for commands and
[release notes](release-notes.md) for the current snapshot.

## Lifecycle

Fork v2 is active. Legacy fork v1 is isolated and accepts only critical,
security, compatibility, and model-asset maintenance. Never merge v2 wholesale
into v1; manually port a small reviewed fix and test it on the target generation.
See [v1/v2 maintenance policy](versioning.md).
