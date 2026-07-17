# CapsWriter fork v2 documentation

> English · [繁體中文](../zh-TW/README.md) · [Project README](../../README.en.md)

This is the documentation home for the active Windows + Linux fork. Choose a
task below instead of assuming that one deployment path represents the whole
product.

## Start by task

| I want to… | Read |
|---|---|
| Decide between Windows desktop, Linux desktop, container, Web, CLI, and TUI | [Getting started](getting-started.md) |
| Operate a server or browser deployment | [Deployment](deployment.md) |
| Fix a desktop, container, API, or client failure | [Troubleshooting](troubleshooting.md) |
| Understand supported, limited, and unsupported claims | [Support and security](support-security.md) |
| Review changes and migrate to the current v2 snapshot | [Release notes](release-notes.md) |

## Product surfaces

| Surface | Guide | Important boundary |
|---|---|---|
| Windows desktop and package | [Desktop portability](desktop-portability.md#windows-package-and-http-api) | CI validates source/package contracts; a real Windows release still needs build, audio, tray, and shortcut evidence |
| Linux desktop | [Desktop portability](desktop-portability.md#linux-x11-hotkeys) | X11 is supported with no selective suppression; Wayland global hotkeys are unsupported |
| Linux `amd64` server and Docker | [Deployment](deployment.md#linux-container-profile) | Containers are the primary headless deployment path; ARM64 is not release-gated |
| OpenAI-compatible HTTP API | [API guide](openai-api.md) | Opt-in file-transcription subset, not the entire upstream OpenAI Audio API |
| Web Console | [Deployment](deployment.md#web-console-profile) | Browser microphone needs a secure context; browser TTS depends on local browser/OS voices |
| No-GUI CLI | [CLI reference](cli-client.md) | Standard-library client; no desktop global hotkey or tray |
| Textual TUI | [TUI guide](tui.md) | Core file mode is hash locked; microphone is an optional native stack |

## Operating and release documentation

| Document | Audience |
|---|---|
| [Verification and cleanup](../verification.md) | Contributors and release operators |
| [v1/v2 maintenance policy](versioning.md) | Maintainers and users planning upgrades |
| [Architecture](../architecture.md) | Contributors changing fork integration points |
| [Current fork state](../state-of-fork.md) | Reviewers who need implementation and evidence inventory |
| [Upstream synchronization](../upstream-sync-guide.md) | Maintainers merging released upstream changes |
| [Upstream product changelog](../CHANGELOG.md) | Users reviewing recognition-engine and upstream desktop history |

## Suggested reading paths

- New desktop user: **Getting started → Desktop portability → Troubleshooting**.
- New server operator: **Getting started → Deployment → Support/security**.
- API/client integrator: **Getting started → API guide → CLI, Web, or TUI guide**.
- Release reviewer: **Release notes → Support/security → Verification → Current fork state**.
- Maintainer: **Architecture → Versioning → Upstream synchronization → Verification**.

When a page describes automated evidence, read it as the boundary of that
evidence. A compile or mock contract test does not prove a real microphone,
display server, GPU, model download, or packaged desktop artifact.
