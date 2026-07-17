# CapsWriter TUI v2

> [繁體中文](../zh-TW/tui.md) · English

CapsWriter TUI is the keyboard-first client for the fork v2 local
OpenAI-compatible transcription API. It runs on Windows and Linux, checks the
server before work begins, transcribes an existing audio file, optionally
records a microphone into a private temporary WAV file, and saves the returned
text or subtitle atomically.

It is a source-checkout client, not a replacement for the upstream Windows
desktop input workflow. Audio goes only to the API root you configure; for a
fully offline setup, point it at your own CapsWriter server.

![The real CapsWriter Textual application showing server connection controls, diagnostics, speech input, and transcript panels](../assets/tui-workbench.svg)

This is a real SVG capture of `CapsWriterTui`, mounted headlessly with Textual
Pilot at 140×46 terminal cells and exported with Textual's SVG renderer. It is
not a drawn mockup. The capture uses file-only mode so its initial state is
deterministic and does not open a microphone. Its live clock is disabled and
external CSS font URLs are removed; the strict suite requires a fresh render to
match the committed SVG exactly.

Text equivalent: a dark, high-contrast terminal workbench places the API root,
masked in-memory key, and health/readiness/model diagnostics at the top. A file
and optional microphone workspace sits beside the transcript/status panel.
Persistent footer hints expose the primary keyboard commands.

## Support boundary

| Item | Supported behavior |
|---|---|
| Operating systems | Windows and Linux in a modern terminal |
| Python | Lock supports 3.10 through 3.12; CI runs the complete strict suite in four Ubuntu 24.04/Windows 2022 × Python 3.10/3.12 legs |
| API | CapsWriter fork v2 `health`, `ready`, `models`, and `audio/transcriptions` routes |
| Input | Existing local file; optional 16 kHz mono microphone capture |
| Formats | `text`, `json`, `verbose_json`, `srt`, and `vtt` |
| UI languages | English and Traditional Chinese (`zh-Hant`) |
| Secrets | Optional Bearer token, masked and held in memory only |

The server still decides which audio containers its installed ffmpeg can
decode. A successful TUI installation does not prove that a model is loaded or
that a particular audio codec is available; the F5 diagnostics make those
states visible.

## Install from the hash lock

Run from the repository root. Use a dedicated virtual environment so no user
or global package can satisfy the release gate accidentally.

Linux or macOS-style shell on a supported Linux host:

```bash
python3.12 -m venv .venv-tui
.venv-tui/bin/python -m pip install \
  --disable-pip-version-check \
  --require-hashes \
  --only-binary=:all: \
  --requirement requirements-tui.lock
.venv-tui/bin/python -m client.tui --help
```

PowerShell on Windows:

```powershell
py -3.12 -m venv .venv-tui
& .\.venv-tui\Scripts\python.exe -m pip install `
  --disable-pip-version-check `
  --require-hashes `
  --only-binary=:all: `
  --requirement requirements-tui.lock
& .\.venv-tui\Scripts\python.exe -m client.tui --help
```

Python 3.10 consumes the same lock and installs the marked `exceptiongroup`
backport; Python 3.12 omits it. [`requirements-tui.txt`](../../requirements-tui.txt)
contains only the reviewed direct pins, while
[`requirements-tui.lock`](../../requirements-tui.lock) resolves every core
transitive dependency and SHA-256 hash. The core lock intentionally preserves
file-only operation without a native audio stack.

### Optional microphone

Microphone recording is a progressive enhancement. It requires a compatible
`sounddevice` package and a working PortAudio input device in the same virtual
environment. Those native, platform-specific pieces are not part of the core
hash lock. If they are absent or fail to load, the TUI displays **FILE ONLY**
and leaves file transcription fully available.

For a managed deployment, pin and audit `sounddevice` separately for the target
OS and provision PortAudio through that OS's package mechanism. The TUI never
downloads a microphone driver at runtime.

## Enable and connect to the server

The HTTP API is disabled by default. Configure the local server as described in
the [OpenAI-compatible API guide](openai-api.md). A minimal loopback deployment
uses:

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token
```

Start the server, then confirm readiness before launching the client:

```bash
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

Launch on Linux:

```bash
CAPSWRITER_HTTP_API_KEY=replace-with-a-long-random-token \
  .venv-tui/bin/python -m client.tui \
  --base-url http://127.0.0.1:6017 \
  --lang en
```

Launch from PowerShell:

```powershell
$env:CAPSWRITER_HTTP_API_KEY = "replace-with-a-long-random-token"
& .\.venv-tui\Scripts\python.exe -m client.tui `
  --base-url http://127.0.0.1:6017 `
  --lang en
```

`--base-url` accepts either the API root or the same URL ending in `/v1`; the
client normalizes the latter. It accepts only absolute HTTP(S) URLs without
embedded credentials, query strings, or fragments. There is deliberately no
`--api-key` command-line option, which keeps the token out of shell history and
process arguments. You may also paste the key into the masked field after the
app starts.

## Operating workflow

![Bilingual CapsWriter TUI workflow from hash-locked installation through diagnostics, transcription, atomic save, retry, and cancellation cleanup](../assets/tui-workflow.svg)

Text equivalent: install the reviewed hash lock, set the API root and
memory-only key, and press F5. Failed health, readiness, model, authentication,
or limit checks loop back through repair. When ready, choose a file or optional
microphone, press Ctrl+T for a bounded cancellable request, review the response,
and press Ctrl+S for an atomic save. Cancel or exit closes active work and
cleans TUI-owned temporary recordings.

1. Press **F5**. `Health`, `Readiness`, and `Models` remain separately visible,
   so a live HTTP process is not mistaken for a loaded recognizer.
2. Paste an existing audio path, or start optional recording with **F8**. The
   model field should remain `whisper-1`; that is the wire compatibility ID,
   while the server selects the real offline engine.
3. Optionally add a language hint and a prompt of at most 2,048 characters.
   Choose the desired response format.
4. Press **Ctrl+T**. The UI disables conflicting controls while the request is
   active. Press **Esc** to cancel safely.
5. Review the returned content. For successful file input, the TUI suggests a
   portable sibling filename with the matching `.txt`, `.json`, `.srt`, or
   `.vtt` extension.
6. Press **Ctrl+S**. Saving writes UTF-8 through a same-directory temporary file,
   flushes it, and atomically replaces the destination. The client refuses to
   overwrite the source audio or save into its private recording directory.

## Keyboard map

| Key | Action |
|---|---|
| `F5` | Refresh health, readiness, and model diagnostics |
| `F8` | Start recording; while recording, stop and select the WAV |
| `F9` | Cancel microphone recording and delete its temporary WAV |
| `Ctrl+T` | Start file or recorded-audio transcription |
| `Esc` | Cancel the active request or active recording |
| `Ctrl+S` | Save the current transcript atomically |
| `Ctrl+L` | Switch between English and Traditional Chinese |
| `Ctrl+O` | Focus the audio-path field |
| `Ctrl+Q` | Quit and clean TUI-owned temporary audio |

At widths below 100 cells the workspace switches to a narrow stacked layout.
An 80×24 terminal works, but 100×30 or larger is easier to scan. All primary
actions remain keyboard reachable, focus has a visible cyan border, and status
text uses words such as `OK`, `DEGRADED`, `WORKING`, and `ERROR` instead of
relying on color alone.

## Bounds, privacy, and cancellation

- Diagnostics default to a 10-second timeout. Transcription defaults to 600
  seconds. Both are finite and may not exceed 900 seconds.
- Response bodies default to 16 MiB and may not exceed 64 MiB. The client
  rejects an oversized declared body and also stops a streamed body that grows
  past the limit.
- Microphone capture defaults to 300 seconds and a 2 MiB callback buffer; CLI
  bounds cap them at 1,800 seconds and 64 MiB.
- HTTP redirects and ambient proxy environment variables are not followed.
  Use an explicit trusted HTTPS API root when traffic leaves loopback.
- The API key is not written by the TUI. Peer errors are redacted if they echo
  the in-memory key unexpectedly.
- Recorder directories are best-effort mode `0700`; WAV files are best-effort
  `0600`. Successful recorded-audio upload, cancel, replacement, and normal
  exit remove TUI-owned audio. A failed upload retains the WAV only for an
  in-session retry, then exit cleanup removes it.

Useful launch bounds:

```bash
.venv-tui/bin/python -m client.tui \
  --diagnostic-timeout 10 \
  --transcription-timeout 600 \
  --max-response-mb 16 \
  --max-recording-seconds 300 \
  --recording-buffer-mb 2
```

## Verification and screenshot provenance

The strict verifier must run inside an environment installed from the lock:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  .venv-tui/bin/python scripts/verify_tui.py
```

It validates the supported interpreter range, direct pin and lock parity,
installed versions, imports, and `pip check`; then it discovers the complete
TUI unit/Pilot suite. Zero discovered tests, dependency mismatch, import
failure, test failure, timeout, or even one skipped test fails the command. CI
runs this consumer independently in four Ubuntu 24.04/Windows 2022 × Python
3.10/3.12 legs.

Regenerate the real application screenshot after an intentional UI change:

```bash
.venv-tui/bin/python client/tui/scripts/capture_screenshot.py \
  --output docs/assets/tui-workbench.svg \
  --lang en \
  --width 140 \
  --height 46
```

Clean only TUI-owned Python residue:

```bash
.venv-tui/bin/python client/tui/scripts/clean.py
.venv-tui/bin/python client/tui/scripts/clean.py --check
```

To update dependencies, change a direct exact pin first, regenerate the
universal lock from Python 3.10 using the command recorded at its top, and run
both interpreter gates on Ubuntu and Windows (all four CI legs). Never hand-edit
a version or hash in the lock.

## Troubleshooting

| Symptom | Check |
|---|---|
| Connection refused | Confirm the API is enabled, the port is published, and the TUI API root names the host as seen from the client |
| `401` | Set the same server token in `CAPSWRITER_HTTP_API_KEY` or the masked key field; do not put it in the URL |
| Health works, readiness is degraded | Inspect the separate readiness detail for model child liveness, router binding, dependency state, or ffmpeg |
| Models fail but health works | Verify authentication and `/v1/models`; do not begin transcription until diagnostics are coherent |
| Request times out | Check server queue/inference logs, then raise the bounded timeout only when the audio and hardware justify it |
| Response too large | Prefer `text`, `srt`, or `vtt`, or raise `--max-response-mb` within the 64 MiB cap |
| **FILE ONLY** | Install and verify the optional native microphone stack, or continue with an existing audio file |
| Save is rejected | Choose a permanent path that differs from the source audio and is outside the private recorder directory |
