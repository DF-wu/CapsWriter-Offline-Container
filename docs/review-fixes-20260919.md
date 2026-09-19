# Review fixes — 2026-09-19

This follow-up addresses the behavior findings on PR #4 (v2) and PR #5 (v1).
The shared fixes are applied to both branches; only v2 contains the graphical
client settings editor. Existing services on axolotl were not upgraded.

| Review area | Result and regression coverage |
|---|---|
| Microphone identity (v2) | Persist name + host API, resolve the current PortAudio index at startup, and distinguish identical same-API names by an explicit index. Tests cover UI choices, persistence, stream creation, reordering, ambiguity and missing devices. Identical hardware with identical metadata can swap indices invisibly; the UI asks users to reselect after device changes. Legacy name strings remain readable. |
| Concurrent client editors (v2) | Merge only touched fields under a persistent cross-process file lock. Save reloads effective values; reset clears all overrides under the same lock. Real spawned-process tests cover stale editors, lock contention and reset without resurrecting untouched values. Last save wins for the same field. |
| File mode and auto-enter | Preserve all requested file paths so missing paths report errors instead of switching to microphone mode or disappearing from a mixed list. Foreground executable matching is case-insensitive for auto-enter and paste. |
| Subtitle splitting and timing | Chinese commas split without whitespace; English commas retain their whitespace rule. Punctuation does not inflate unit counts, and Han/kana count as characters. End times cannot precede the final aligned token; the last line adds only that token's tail. Complete multi-token and generated SRT output regressions cover both defects. |
| Chinese numbers | Narrow the rate shorthand and verb exceptions; restore omitted-leading-one numbers and positive/negative leading-point decimals with units. Preserve genuine click counters, rate terms and existing decimal output format. |
| Qwen options and CPU limits | Carry all four advertised llama tuning overrides through the real schema/factory/context classes. Apply the shared thread limit to both Qwen generation and batching, including native startup. Tests inspect actual b7798 ctypes context arguments while isolating asset loading; defaults and ABI remain unchanged. |
| Shutdown and unattended errors | SIGTERM requests cleanup immediately and only once. Interactive SIGINT retains confirmation. Closed output cannot prevent cleanup. Actual subprocess tests verify worker reaping and listener release; startup failure tests cover disabled prompts, non-TTY stdin and EOF. |
| Legacy packaging | Retire the unsupported junction-based Win7/Python 3.8 client-only spec. v1's build guide now states its source-only Python 3.10–3.12 contract; v2 directs users to its qualified build.spec workflow. |
| Integration checks | Restore v2's log-directory environment override, already present in v1. Use an independent unittest loader for nested API discovery on older supported Python versions. |

## Validation

- v1: **428 tests passed** in an isolated Python 3.11.16 container (16 maintenance,
  195 scripts, 153 API, 64 container/bootstrap). Python 3.12.13 host venv had passed
  426 before the two additional closed-output regressions; the final lifecycle
  cases were also verified separately.
- v2: **386 script tests**, **382 passed and 4 Node-dependent skips**, in an
  isolated Python 3.11.16 container with Xvfb. All four real Tk tests passed.
  The Node-dependent script tests also run in the Python 3.12.13 host venv with
  host Node available; its four skips are the Tk display cases exercised above.
- v2 strict API contract: **175 passed**; container/bootstrap: **64 passed**.
- The dependency-free development profile retains its intentional dependency
  and display skips. It also passes the script suite.
- Documentation links, whitespace and the documented **75-path** v2 upstream
  divergence inventory are checked. Test counts overlap and must not be summed
  as a unique total.

The test containers use read-only source mounts, no network during tests,
4 CPU / 4 GiB limits, and an init process to reap test descendants. A Python
3.12 base-image download stalled; the container uses a cached Python 3.11.16
base with a fresh virtual environment instead. No application from that base
image is started. Hosted PR checks supply the Windows/Python 3.10–3.12 and v2
portable-package gates for the pushed revision.

This round does not rerun native model/GPU inference or physical Windows
microphone, hotkey, foreground typing or hardware identity checks. The earlier
real-model evidence remains in each branch's validation record; it is distinct
from these regression and configuration tests.

## Second review follow-up

- Numeric normalization preserves ordering, selecting and counting phrases such
  as `我点三个菜` and `请点三个人回答问题`. Action counters are distinguished
  from physical units and `个百分点`; signed leading-point decimals retain
  their existing output. An ambiguous phrase such as `点五杯水` is treated as
  ordering five cups; `零点五杯水` explicitly expresses half a cup.
- Halfwidth kana and supplementary Han participate in character-based subtitle
  splitting and final-token duration estimates.
- Server startup observes shutdown between setup stages and around child
  creation. Client tray shutdown requests processor exit before closing its
  transport and completes cleanup on the owning event loop.
- Retired client-only packaging instructions have been removed. v1's archive
  command rejects binary releases; v2 ignores stale client-only output.
- Web connection edits require confirmation before discarding a draft or
  switching during a write. A write retains its original endpoint and shows its
  result after a switch; uncertain outcomes instruct the user to return to the
  original connection before verifying. Loaded revisions cannot be submitted
  to a different endpoint.
- The upstream divergence inventory includes microphone-runner exit ownership
  and is synchronized at 76 paths across the guard and maintenance documents.
