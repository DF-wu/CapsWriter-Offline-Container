# 2026-09-19 upstream/settings validation

Validated on axolotl in isolated test containers, with existing model/audio
assets mounted read-only. No existing service was upgraded or restarted.
The v2 implementation commits are `58b3467`, `0489547`, and `607d890` on top
of upstream merge `a1cdd45` (`84912d5`). This is development/PR evidence,
not a release or a Windows hardware qualification.

## Automated gates

| Gate | Result |
|---|---|
| Python script regressions, Python 3.12 API environment + Xvfb | 335 tests exercised without skips; two obsolete documentation assertions were corrected and the 11-test documentation suite then passed |
| Strict API contract | 175 passed, no skips |
| Strict TUI | 72 passed, no skips |
| Docker bootstrap/downloader suite | 63 passed before final shared-library fix; all 41 downloader tests passed after adding the regression |
| CLI | 63 passed; zipapp build, help and stdin smoke passed |
| Desktop settings/UI/runtime with Xvfb | 16 passed, no skips or pending Tk callback errors |
| Web | 112 passed; TypeScript/Vite production build passed |
| Documentation and upstream divergence | Links checked; 63 reviewed upstream-tracked paths |
| Final focused docs, packaging, upstream and downloader regressions | 81 passed |
| Final settings/desktop/Compose regressions | 41 passed |
| Development setup | Isolated dev, API and TUI profiles created; CLI build and dev check succeeded |

Representative commands (inside the applicable isolated environment):

```sh
python scripts/dev.py setup dev
python scripts/dev.py setup api
python scripts/dev.py setup tui
python scripts/dev.py test
python scripts/verify_api_contract.py
python scripts/verify_tui.py
xvfb-run -a python -m unittest discover -s scripts/tests -v
python client/cli/scripts/verify.py
npm test --prefix client/web
npm run build --prefix client/web
python scripts/check_docs.py
python scripts/check_upstream_divergence.py --require-base
```

The minimal dev profile intentionally has no third-party dependencies; its
dependency/display-dependent skips were exercised with the API/Xvfb environment.
Test counts above overlap and must not be added into a unique total.

## Real model and settings lifecycle

Qwen3-ASR-1.7B used the existing fp16 encoder and q4_k decoder on CPU. Input
was a known 5.592-second Chinese WAV. After refreshing the final source:

- `/health` and `/ready` returned 200 with recognizer, router and FFmpeg ready.
- WebSocket binary-subprotocol microphone task returned final text
  `開放時間：早上 9 點至下午 5 點。` in approximately 8.63 seconds.
- HTTP multipart JSON transcription returned 200 and
  `開放時間：早上九點至下午五點。` in approximately 7.79 seconds.
- Real settings GET/PATCH without credentials returned 401; invalid values
  returned 422; stale revisions returned 409.
- Upload limit saved as 23 MB while running value remained 100 MB. After
  restarting through `fork_server.settings_bootstrap`, readiness and settings
  both reported 23 MB, source `saved`, with no pending restart.
- Explicit `cpu_only` environment preset overrode a saved `default` preset.
- Cross-origin PATCH preflight returned 200 with the configured origin;
  unauthenticated PATCH remained 401 and included the CORS response headers.

The actual downloader extraction path verified local b7798 files against its
official SHA-256 manifest and atomically installed all four target directories.
`/proc` mappings confirmed the recognizer loaded the regular file
`core/server/engines/llama/bin/libllama.so`, SHA-256
`bbe4fd0f99a5062bac8e585192047f82399988e9e24d993a19fdd113454828ec`.
The network download itself was stopped because it was slow; cached library
bytes were still checked against the official manifest.

## Rendered Web settings

Chromium checks used a local mock settings API at 1440×1000 and 390×844.
Checked authenticated revision saves, environment locks, bounds validation,
running versus next-start values, conflict recovery, and horizontal overflow.
The only console error was the deliberately simulated 409 response. The actual
HTTP authentication, CORS and restart behavior were checked separately above.

## Limits and cleanup

Subsequent GitHub checks exposed two issues fixed before PR handoff: the TUI
fault-injection test retained a 10 ms deadline for its healthy follow-up recording,
and redirected Windows cp1252 output could raise during Chinese status messages.
The former now scopes its shortened deadlines to the fault; the latter preserves
the chosen encoding with `backslashreplace` before colorama wraps standard output.
Two subprocess regressions cover strict cp1252 plain/Rich output and absent GUI
streams. The documented upstream divergence inventory is consequently 64 paths.

The ASGI request fixture now keeps ordinary clients connected until response
completion and triggers deliberate disconnects with an explicit event. This
removes a Windows scheduling race that incorrectly turned successful responses
into 499 errors; all 175 API tests passed after the correction. Screenshot capture
also waits for the footer's asynchronous binding widgets to finish layout before
exporting, with a bounded timeout and a delayed-footer regression. Golden SVG
assertions remain unchanged.

GitHub Windows 2022 production packaging and both executable self-checks passed
for the redirected-console fix. These checks do not exercise physical devices
or foreground desktop interaction.

No real Windows microphone, global keyboard hook, foreground text insertion,
Windows package execution, GPU inference or precision-alignment test was run
locally. ForcedAligner assets were absent: text inference succeeded through the
existing fallback, but precise timestamps were not qualified. Production CUDA
image build and download-from-empty model bootstrap are not established by
these CPU runtime-container tests.

Temporary dev/ASR containers were removed after use. Test images and temporary
v1 worktree are removed after both PR branches are pushed; model files, audio,
settings and preexisting services are retained. `scripts/clean.py` removes only
the repository's generated build/test outputs; source commits and this record
remain available for review.
