# Release notes and changelog

> [Documentation home](README.md) · [繁體中文](../zh-TW/release-notes.md) · [Getting started](getting-started.md)

## fork-v2.0.0-rc.1 — cross-platform release candidate

Release-candidate date: **2026-07-18**. `fork-v2.0.0-rc.1` is a GitHub
pre-release, not the final `fork-v2.0.0` support claim. Automated source, API,
TUI, Web, image, and Windows package gates are complete; the real-device/model
qualification listed below remains required before a stable release.

![Fork maintenance flow: released upstream changes enter active v2 while only critical or security fixes are manually backported to isolated v1](../assets/version-tracks.svg)

Text equivalent: upstream releases merge into active fork v2, which passes
Linux, Windows, API, TUI, Web, and security gates before release. Legacy fork
v1 never merges v2; a critical/security fix may be manually ported and must pass
separate legacy container, API, and model-asset checks.

## Release theme

This snapshot turns the former Linux-container-focused fork documentation into
an evidence-based Windows + Linux product surface while retaining the upstream
recognition engines:

- Windows desktop and package entrypoints remain supported and gain an optional
  validated HTTP API path.
- Linux desktop gains bounded X11 hotkey support with explicit Wayland/headless
  refusal and no unsafe selective-suppression claim.
- Linux container/server operation remains the primary headless deployment.
- Web, standard-library CLI, and hash-locked Textual TUI clients expose the
  local transcription service without replacing the desktop application.

## Added

### Desktop portability

- Universal server entrypoint that preserves upstream desktop defaults when
  the HTTP API is disabled and applies only validated HTTP environment settings
  when enabled.
- PyInstaller specification integration for the universal server and required
  server hidden imports in the Windows distribution.
- Platform-aware shortcut backend policy: native Windows behavior, bounded X11
  callbacks, and actionable Wayland/headless unavailability.
- Pinned Ubuntu 24.04 / Windows 2022, Python 3.10 / 3.12 portability matrix for
  portable desktop/package contracts and the no-GUI CLI.

### Server and API

- Linux Docker/Compose server path with model bootstrap, GPU preference, CPU
  fallback, bounded backend probing, persistent model/hotword/log locations,
  and readiness-aware health checking.
- Opt-in OpenAI-compatible `whisper-1` file transcription surface with health,
  readiness, models, five response formats, explicit capability errors,
  bounded multipart/decode/admission/deadline handling, and OpenAI-style error
  envelopes.
- Dedicated exact-pin API contract environment that fails on missing
  dependencies, empty discovery, failures, or skipped contract tests.

### Clients

- React/Vite Web Console with recording/upload, readiness-aware preflight,
  cancellable transcription, five formats, downloads, local browser TTS,
  runtime configuration, browser smoke, and static Nginx image.
- Standard-library no-GUI CLI for health/readiness/models, single/batch
  transcription, atomic output, portable filenames, local OS TTS, and zipapp
  packaging.
- Bilingual Textual TUI for diagnostics, file transcription, optional bounded
  microphone capture, cancellation, atomic save, and memory-only keys.
- Fully resolved SHA-256 TUI dependency lock consumed on Python 3.10 and 3.12,
  with a strict no-skip Pilot/unit verifier.

### Release, security, and documentation

- Root verification/cleanup orchestration, documentation link/accessibility
  checker, upstream-divergence guard, pinned workflow runners/actions, and
  guarded server/Web image publishing with provenance/SBOM requests.
- Safer defaults and bounded subprocess/network cleanup across Docker model
  downloads, ffmpeg helpers, GUI recording/file transcription, worker shutdown,
  GPU boost helpers, hotword Ollama calls, and GGUF metadata reads.
- Paired English/Traditional Chinese getting-started, deployment,
  troubleshooting, support/security, versioning, portability, API, TUI, and
  release documentation with accessible SVG diagrams and a real Textual capture.

## Changed behavior and compatibility notes

| Change | Operator/user impact |
|---|---|
| Root identity is Windows + Linux | Windows users stay in this fork's documented desktop/package path; Linux support is split into X11 desktop and headless/client profiles |
| HTTP API remains opt-in | Existing WebSocket/desktop behavior is not replaced merely by upgrading |
| Non-loopback enabled API requires auth | Set a key/key file, or use the explicit insecure override only on an isolated test network |
| API validates unknown/unsupported fields | Newer SDK fields cannot appear to work when the local engine ignored them |
| `/v1/audio/translations` is explicit `501` | Use transcription; no local translation claim is made |
| X11 forces shortcut suppression off | Listening remains available without risking a whole-keyboard/pointer grab |
| Wayland/headless desktop hotkeys fail clearly | Use X11 or Web/CLI/TUI/file paths |
| Web default key publication requires two settings | A configured key is not written to public `/config.js` accidentally |
| TUI core install uses a hash lock | Recreate the venv instead of mixing arbitrary global/user packages |
| Logs omit prompts/transcripts by default | Enable full transcript logging only with an explicit privacy/retention decision |

## Security fixes and hardening

- Request authentication and declared body size can be rejected before upload
  consumption; raw/file/decoded limits and bounded admission constrain work.
- Cancellation/timeouts close client streams, decoder processes, pending API
  routes, and TUI-owned temporary audio through bounded cleanup paths.
- Key-file support avoids persistent command-line tokens; UI keys are masked or
  held in memory according to each client contract.
- Container publishes loopback by default, drops capabilities, enables
  `no-new-privileges`, and excludes local secrets/models/archives from build
  context.
- Client/server errors bound untrusted response/log previews and redact a
  reflected configured secret.
- Dependency locks, full-SHA actions, read-only workflow permissions, release
  gates, and attestations improve supply-chain traceability.

See [support and security](support-security.md) for the complete boundary and
reporting path.

## Migration from the earlier Linux-container-only fork presentation

1. Keep the existing deployment stopped but recoverable. Back up `.env`,
   `hot-server.txt`, Compose overrides, key files, model/cache paths, and logs
   according to local policy.
2. Use a fresh v2 checkout or immutable image. Do not copy an old source tree
   over the new one.
3. Diff the current `.env.example`. The HTTP API is still disabled by default;
   enable it deliberately, set authentication, and uncomment the current
   Compose HTTP `ports:` mapping.
4. Confirm model and hardware selections. Use explicit CPU settings on hosts
   that should not request a GPU.
5. Validate `/health`, `/ready`, `/v1/models`, then one small known transcription
   in every required format.
6. Move one client at a time: desktop, CLI, TUI, SDK, then Web. Configure exact
   CORS origins for the browser.
7. Retain the previous source/image/configuration until the rollback window
   closes.

Users who want Windows desktop behavior should follow this fork's
[desktop portability guide](desktop-portability.md), not be redirected away
from the repository.

## Migration from fork v1

Fork v1 and v2 are separate product generations with divergent architecture and
Git history. Treat the migration as a parallel deployment:

1. back up v1 configuration and model assets;
2. deploy v2 on different ports;
3. run readiness/model/known-audio checks;
4. point one client at v2;
5. migrate gradually and keep v1 stopped but recoverable through rollback.

Never merge or bulk cherry-pick v2 into `maintenance/v1`. See the
[maintenance policy](versioning.md).

## Known limitations

- CI does not download every production model or prove recognition quality.
- Windows CI hash-installs, builds, relocates, ZIP-round-trips, inspects, and
  import-smokes both packaged EXEs. Each shipped artifact still needs real
  tray, shortcut, audio/FFmpeg, model/known-audio, hardware, and exit tests.
- Linux global shortcuts require X11; Wayland/headless desktop hotkeys are not
  supported. X11 cannot selectively suppress one key safely.
- GPU usability, memory, performance, and fallback depend on the target
  driver/device/model and require hardware evidence.
- Browser microphone requires loopback or HTTPS and user permission. Browser TTS
  depends on locally available browser/OS voices.
- TUI microphone support depends on optional platform-native
  `sounddevice`/PortAudio and a real input device; file mode remains available.
- The local API intentionally implements a bounded transcription subset, not
  streaming, diarization, translation, or every current OpenAI Audio feature.

## Qualification still required before stable fork-v2.0.0

- Green portable Ubuntu/Windows matrix and isolated API/TUI jobs.
- Root verification, documentation, cleanup, Web browser/image smoke, and
  supply-chain workflow source guards.
- Exact Windows package-job ZIP/digest plus real desktop/hardware checks when
  publishing binaries.
- Real X11 check when advertising Linux desktop shortcuts.
- Live readiness and known-audio results for advertised model/CPU/GPU profiles.
- Immutable image/source references, model/dependency provenance, SBOM and
  attestations where applicable.
- Upgrade/rollback rehearsal and final known-limit review.

## Changelog sources

- This page is the fork v2 delivery/portability/API/client changelog.
- [`docs/CHANGELOG.md`](../CHANGELOG.md) records upstream product and recognition
  history that this fork inherits.
- [`docs/state-of-fork.md`](../state-of-fork.md) is the detailed implementation
  and verification inventory for reviewers.
- Git tags and merge history remain the authoritative record of what was
  actually released; an unreleased heading is not a release tag.

Start a fresh install with [getting started](getting-started.md), or plan an
upgrade with [deployment](deployment.md#upgrade-and-rollback).
