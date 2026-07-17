# CapsWriter fork v2 documentation

> English · [繁體中文](../zh-TW/README.md) · [Project README](../../README.en.md)

CapsWriter has two roles: an **ASR server** performs local model inference, and
one or more **clients** capture/select audio and present transcripts. Start
with [Server and client roles](server-and-clients.md) if that distinction is
not already clear.

## New user: follow these three steps

1. **Understand the components:** [Server and client roles](server-and-clients.md).
2. **Choose where the server runs:** [Getting started](getting-started.md).
3. **Choose a client:** desktop, Web, CLI, TUI, or SDK from the table below.

## Server documentation

| I need to… | Read |
|---|---|
| Start Docker, Windows native, or a source server | [Deployment](deployment.md) |
| Enable HTTP `6017` for Web/CLI/TUI/SDK | [OpenAI-compatible API](openai-api.md) |
| Understand WebSocket `6016`, HTTP `6017`, and readiness | [Server and client roles](server-and-clients.md#server-interfaces) |
| Configure exposure, authentication, privacy, or hardware claims | [Support and security](support-security.md) |
| Fix model, container, readiness, or API failures | [Troubleshooting](troubleshooting.md) |

The server owns models, server-side FFmpeg decoding, inference, hotwords,
queues, and readiness. It does not provide desktop hotkeys, browser UI,
terminal UI, or client-local TTS. The desktop client may use its own FFmpeg
process for local media preparation, but never runs ASR inference.

## Client documentation

| Client | Server connection | Guide | Client-owned behavior |
|---|---|---|---|
| Windows/Linux X11 desktop | WebSocket `6016` | [Desktop portability](desktop-portability.md) | Tray, hotkeys, microphone, files, text injection |
| Web Console | HTTP `6017` | [Web Console guide](web-console.md) | Browser recording, history, downloads, browser/OS TTS |
| No-GUI CLI | HTTP `6017` | [CLI reference](cli-client.md) | File/batch automation, atomic output, optional OS TTS |
| Textual TUI | HTTP `6017` | [TUI guide](tui.md) | Keyboard workflow, files, optional native microphone, save |
| OpenAI SDK/curl | HTTP `6017/v1` | [API guide](openai-api.md) | Integration-specific file upload and result handling |

No client above contains an ASR model. Web/CLI/TUI/SDK require the opt-in HTTP
API; the desktop client uses WebSocket and normally does not.

## Choose by deployment

| Goal | Server | Client | Start here |
|---|---|---|---|
| Personal Windows dictation | Windows packaged/native server | Windows desktop client | [Windows desktop path](getting-started.md#path-a-windows-desktop) |
| Linux desktop dictation | Native source server in X11 | Linux X11 desktop client | [Linux X11 path](getting-started.md#path-b-linux-x11-desktop) |
| NAS/headless/shared ASR | Linux `amd64` container | Web, CLI, TUI, or SDK | [Container path](getting-started.md#path-c-linux-container-server) |
| Existing OpenAI integration | Any server with HTTP enabled | SDK/curl | [API compatibility](openai-api.md#compatibility-at-a-glance) |

## Operations and release documentation

| Document | Audience and purpose |
|---|---|
| [Release notes](release-notes.md) | Users reviewing changes, migration, limits, and release evidence |
| [Troubleshooting](troubleshooting.md) | Users/operators following a server-first diagnostic ladder |
| [Verification and cleanup](../verification.md) | Contributors and release operators |
| [v1/v2 maintenance policy](versioning.md) | Maintainers and users planning upgrades |
| [Architecture](../architecture.md) | Contributors changing server, sidecar, or routing integration |
| [Current fork state](../state-of-fork.md) | Reviewers checking implementation and evidence inventory |
| [Upstream synchronization](../upstream-sync-guide.md) | Maintainers merging released upstream changes |
| [Upstream product changelog](../CHANGELOG.md) | Recognition-engine and upstream desktop history |

## Reading order by audience

- Desktop user: **Server/client roles → Getting started → Desktop portability → Troubleshooting**.
- Server operator: **Server/client roles → Deployment → Support/security → Troubleshooting**.
- API integrator: **Server/client roles → API guide → chosen client guide**.
- Release reviewer: **Release notes → Support/security → Verification → Current fork state**.
- Maintainer: **Architecture → Versioning → Upstream synchronization → Verification**.

Automated evidence has explicit limits. A compile, unit test, or package import
self-check does not prove a real microphone, display server, model, GPU, tray,
or global shortcut on the target machine.
