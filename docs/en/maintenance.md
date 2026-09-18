# v1 upstream integration, versioning, and support

[繁體中文版](../zh-TW/maintenance.md)

## Status and branch model

v1 is a separate release track for the Windows desktop client and Linux/Docker
server. It accepts full upstream updates, including model changes and necessary
refactors, with daily usability and stability taking priority. This supersedes
the earlier critical-fixes-only policy.

- Maintained line: `maintenance/v1`
- Standing comparison/release PR base: `archive/v1-legacy`
- Original maintenance snapshot: `b46ca74`
- Historical safety tag: `fork-pre-reset-20260525-1411`
- Current upstream integration: `84912d5`

The application version `2.6` in Python modules describes upstream application
compatibility, not the fork release track. A `2.x` internal version does not turn
this branch into fork v2. Do not merge the whole v1 line into `master`.

## Scope and compatibility

Upstream recording, hotkey, transcription, model, dependency, and architecture
improvements belong on v1 when they can be integrated safely. Preserve the
Windows source entrypoint, documented WebSocket behavior, the fork HTTP API,
and Linux/container operation. Document changed defaults and config/model
migration; add regression coverage for changed behavior.

The v2 Web Console, no-GUI CLI, Textual TUI, and universal Windows package
remain separate product surfaces. Other desktop platforms retain extension
points but are not the current client qualification target.

Native Python 3.10–3.12 compatibility is retained, with Python 3.12 the setup
baseline. The pinned Docker runtime remains Python 3.10. Upstream's Python 3.14
and uv migration is adapted to platform dependencies; it does not establish
Python 3.14 release qualification here. Native llama bindings stay paired with
`b7798`; adoption of upstream `b10621` requires a coordinated binding/binary
migration and inference validation. See [migration notes](../v1-upstream-refresh.md).

## Support and release evidence

| Path | Target | Qualification boundary |
| --- | --- | --- |
| Windows desktop client | First client platform; `start_client.py` | Portable syntax/protocol checks do not validate global hotkeys, tray, microphone, clipboard, text injection, or a PyInstaller executable. |
| Linux Docker server | Primary server deployment | A passing unit gate does not prove a successful cold model download, native model load, CPU/GPU inference, or container bootstrap. |
| Native server | `start_server_universal.py` for the fork runtime | FFmpeg, models, native libraries, and supervision must be checked on the target host. |
| HTTP API | Optional transcription subset | Verify authentication, limits, cancellation, and live model-backed requests; this is not the complete OpenAI API. |
| Other desktop platforms | Future expansion | No macOS/Linux desktop qualification is claimed. |

The current v1 release path is source-only. Compose builds
`capswriter-offline-v1-local:source` from the checkout. The public
`ghcr.io/df-wu/capswriter-offline-server:latest` image is v2 and must not be used
as a v1 release. Never advertise a Windows executable without a separately
attached, qualified artifact.

No end-of-life date or response-time SLA is promised. Support is best effort.

## Runtime boundaries

The fork preserves bounded protocol validation, connection-scoped recognition
state, cancellation cleanup, and HTTP authentication/upload controls while
updating upstream internals. Consult [HTTP API](../HTTP_API.md) and the current
runtime settings for precise limits; limits on compressed uploads are not a
substitute for decoded-audio limits. Keep unauthenticated WebSocket service on
a trusted network. Native model libraries, platform hooks, and accelerator
providers require separate validation from portable Python tests.

## Verification

Run the current CI gate in an isolated environment:

```bash
python -m pip install -r requirements-maintenance.txt
python scripts/verify_v1.py
python -m compileall -q config_client.py config_server.py start_client.py start_server.py start_server_universal.py start_server_docker.py core fork_server docker/server
bash -n docker/server/entrypoint.sh
docker compose --env-file .env.example config --quiet
```

The workflow file is the authority for its current OS/Python matrix. Record
actual test results against the exact source revision; historical CI evidence
does not certify this refresh. Release qualification should include disposable
container build/bootstrap, Mandarin and English known-audio transcription,
CPU/GPU backend details, and a real Windows desktop smoke test. Identify the
model, native runtime/driver, audio provenance, and observed result.

Tests on axolotl must use isolated containers, ports, networks, and disposable
storage. Do not upgrade existing services or attach writable production data.
Clean only resources created for the test; retain the validation report.

## Integration and release procedure

1. Create an isolated working branch from the maintained v1 line.
2. Record the upstream revision and review intentional integration exceptions.
3. Migrate configuration values and verify model layout before startup; do not
   overwrite updated config modules with old copies.
4. Run portable and relevant runtime qualification, documenting anything not
   exercised. Do not claim Windows or real-model verification without evidence.
5. Open the v1 PR in the fork repository with the appropriate v1 base, never in
   the upstream repository or against v2 `master`.
6. Use `fork-v1.<minor>.<patch>` tags, with optional `-rc.<n>`, and identify the
   exact source commit. List server source, images, Windows source, and Windows
   binaries separately. A PR is not automatically a release or deployment.

Report vulnerabilities privately where possible. Remove keys, transcripts,
audio, model artifacts, and private production logs from public reports.
