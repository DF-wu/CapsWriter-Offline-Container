# Release notes and changelog

> [Documentation home](README.md) · [繁體中文](../zh-TW/release-notes.md) · [Getting started](getting-started.md)

## fork-v2.0.0 — stable release

Release date: **2026-09-30**. This is the first stable fork v2 release and
supersedes `fork-v2.0.0-rc.1`. The tag points at the `master` commit that passed
CI, portability/Windows-package, server-image and Web-image workflows; the GitHub
Release attaches that commit's Windows ZIP, `SHA256SUMS` and image references.

### Highlights

- **Upstream synced through `84912d5`**: upstream v2.6 (GPU pre-boost, short-phrase
  punctuation, delayed Enter, tray Diary/Restart, ITN improvements) plus the later
  split-token text-loss fix, two-pass subtitle line breaking and sentence end times.
  The fork keeps Python 3.10–3.12 and the pinned llama.cpp b7798 ABI instead of
  upstream's Python 3.14/b10621 environment.
- **Daily settings UI**: a first-run/tray settings window for the Windows client
  (connection, microphone, shortcuts, output) and an opt-in, authenticated server
  settings page in the Web Console. See [daily settings](../settings.md).
- **Unified development commands**: `scripts/dev.py` sets up, runs, tests and builds
  in isolated environments; see [development](../development.md).
- **Subtitle fix**: for engines without native timestamps (such as Qwen3-ASR), HTTP
  `srt`/`vtt`/`verbose_json` segments now use the final formatted text, so English
  keeps its spaces and Chinese keeps punctuation and number formatting.
- **Reliability**: startup cancellation and tray shutdown wait for completion, the
  server terminates gracefully, redirected Windows console encoding errors no longer
  interrupt recognition, and concurrent settings edits no longer clobber each other.
- **Documentation**: rewritten READMEs, a new [illustrated usage guide](usage.md) and
  real Web/CLI/TUI captures.

### Changes to note

- Fork dependency locks moved to `requirements/` (for example
  `requirements-tui.lock` → `requirements/tui.lock`). Update custom install scripts;
  upstream's `requirements-client.txt` and `requirements-server.txt` stay at the root.
- Dated validation/review records moved to `docs/reports/`.
- Settings window and Web Console hints no longer suggest a specific host name.

### Evidence for this release

- GitHub Actions: CI, the portability matrix (Ubuntu 24.04/Windows 2022 × Python
  3.10/3.12), the Windows package job (hash-locked build, relocation, ZIP round trip,
  reparse-point rejection, `--artifact-self-check` for both EXEs) and server/Web image
  publication.
- A full local `verify_all` run: upstream divergence guard, documentation, CLI 63,
  HTTP API 178, Docker bootstrap 64, scripts 398 and Web 118 tests.
- Real model: Linux x86-64, Qwen3-ASR 1.7B on CPU (`cpu_only` preset) from the source
  runtime. `/health` and `/ready` returned ok; a known 5.6-second Chinese clip
  transcribed over HTTP as 開放時間：早上九點至下午五點。 and an English clip was
  correct in `text`, `srt`, `vtt` and `verbose_json`. The Web Console, CLI and TUI were
  driven against the same server for the screenshots. Earlier in-container WebSocket,
  HTTP and settings-lifecycle checks are in the
  [validation record](../reports/validation-20260919.md).

### Not covered by this release's evidence

These were not accepted on real hardware for this release; please confirm them in
your environment and report results:

- Windows physical microphone, global hotkeys, tray, foreground text insertion and
  model inference on Windows.
- NVIDIA/Vulkan/DirectML GPU inference and performance.
- Linux X11 desktop hotkeys.
- Upgrade/rollback rehearsal from rc.1 or fork v1.

## fork-v2.0.0-rc.1 — cross-platform release candidate

Release-candidate date: **2026-07-18**. `fork-v2.0.0-rc.1` is intended for
GitHub pre-release distribution, not the final `fork-v2.0.0` support claim. The
exact tagged `master` commit must pass CI, portability/Windows-package,
server-image, and Web-image workflows; earlier branch or baseline runs cannot
substitute. The GitHub pre-release records those exact run, artifact, checksum,
image-tag, and digest references. The real-device/model qualification listed
below remains required before a stable release.

This candidate's original maintenance policy limited v1 to focused backports.
The current policy additionally permits the one-time v1 upstream refresh
described above. Both tracks still require their own checks before release.

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

The approved one-time v1 refresh imports upstream on its own branch and ports
v1 integration; it does not merge the v2 product into `maintenance/v1`. See the
[maintenance policy](versioning.md).

## Known limitations

- CI does not download every production model or prove recognition quality.
- Windows CI hash-installs, builds, relocates, ZIP-round-trips, inspects, and
  import-smokes both packaged EXEs. That exact ZIP keeps `models/` empty and
  excludes GGUF runtime DLLs and FFmpeg; follow the checksummed prerequisite
  procedure in the desktop guide. Each shipped artifact still needs real tray,
  shortcut, audio/FFmpeg, model/known-audio, hardware, and exit tests.
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

## Pre-stable qualification listed at rc.1

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
