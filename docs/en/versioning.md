# v1 and v2 maintenance policy

> [Documentation home](README.md) · [繁體中文](../zh-TW/versioning.md) · English

This fork maintains two product generations. The names **fork v1** and
**fork v2** describe this fork's generations; they are not the upstream
`v1.0` and `v2.x` tags.

Upstream changes normally enter v2. A separately approved, one-time **full
upstream refresh through `84912d5`** is also being prepared for v1, including
the `util/` to `core/` migration. This does not merge the two product tracks:
v1 retains its own server, container, and API contracts and release channel.

## Tracks and authoritative refs

| Track | Authoritative branch | Upstream lineage | Change policy |
|---|---|---|---|
| fork v1 | `maintenance/v1` | Still the legacy `v2.5-alpha` plus `3419171` baseline until the separate refresh PR merges | One-time full upstream refresh approved; preserve v1 server/container/API behavior, then resume focused maintenance |
| fork v2 | `master`; current work on `feat/v2-upstream-settings-20260918` | Feature-branch merge `a1cdd45` includes upstream `84912d5`; this is not yet a `master` release | Active cross-platform product development; use short-lived branches and merge-based upstream sync |
| v1 audit snapshot | `archive/v1-legacy` and tag `fork-pre-reset-20260525-1411` | Last pre-reset v1 tree, `b46ca74` | Immutable recovery/audit point; do not develop directly |

The v1 and v2 Git histories diverged before upstream's large `util/` to
`core/` refactor. The approved v1 refresh imports upstream changes on a
separate branch and ports the v1 integration to `core/`; it is not a bulk
import of the v2 product. Until that PR merges, `maintenance/v1` remains on
its legacy architecture. Routine backports target the architecture actually
present on the receiving branch and include generation-specific checks.

## Support matrix

| Capability | fork v1 | fork v2 |
|---|---|---|
| Existing Windows desktop workflow | Preserved legacy behavior | Preserved upstream behavior |
| Linux server container | Maintenance support | Primary deployment path |
| Windows native server | Legacy/manual build | Supported universal entrypoint and packaging gate |
| Scriptable CLI | Legacy scripts only | Supported on Windows and Linux |
| Interactive TUI | Not backported | Supported on Windows and Linux |
| Browser console | Not backported | Supported on modern Windows/Linux browsers |
| OpenAI-style transcription API | Preserve the existing v1 contract through the refresh | Tested `whisper-1` transcription contract with explicit capability errors |
| New features | Upstream changes included in the approved one-time refresh; otherwise focused maintenance | Yes |

“Supported” means that the documented entrypoint has an automated gate for its
portable logic. Hardware-, terminal-, and model-backed release evidence is
listed separately; one Linux container test is never used as proof of Windows
runtime behavior.

## Refresh and backport rules for v1

The approved full refresh through `84912d5` may migrate architecture and
dependencies where needed to integrate upstream. It must retain the v1
server/container/API behavior, pass separate v1 checks, and land through its
own PR. It does not authorize a production deployment or release.

Outside this one-time refresh, a v1 change must meet every rule below:

1. It fixes a critical/security issue, restores a model asset, or preserves a
   documented external contract.
2. It is implemented against the current v1 architecture (`util/` before the
   refresh, `core/` after it). Do not copy a v2 product module wholesale.
3. It includes a focused regression test or an isolated executable smoke test.
4. It does not change v1 defaults unless the old default is unsafe.
5. It is released under a v1-only Git tag. An image may be published only by
   a separate, explicitly reviewed v1 workflow/tag; no such image automation
   is configured in the current v2 tree.

Ongoing product feature work and UI redesigns belong to v2. Future broad v1
upstream refreshes require a new explicit scope decision.

## Version and image names

- v1 release tags: `fork-v1.<minor>.<patch>`
- v2 release tags: `fork-v2.<minor>.<patch>`
- release candidates append the SemVer pre-release suffix `-rc.<n>` and are
  marked as GitHub pre-releases; a final tag never reuses an RC tag
- current automated v2 image tags: immutable `sha-<full-git-sha>` plus a
  guarded `latest` promotion for the current `master` tip
- the current workflows publish no moving `v1` or `v2` channel tags
- v1 image publication is not automated in this tree and must never reuse the
  v2 `latest` tag

Upstream tags remain upstream identifiers and are never recreated as fork
release tags.

## Moving from v1 to v2

Treat migration as a parallel deployment, not an in-place Git merge:

1. Back up `.env`, hotword files, role files, and any local model cache.
2. Start v2 on different WebSocket and HTTP ports.
3. Run `/health`, `/ready`, model listing, and a known Chinese and English audio
   transcription.
4. Point one CLI/TUI client at v2 and verify the required response formats.
5. Move clients gradually; keep v1 stopped but recoverable until the rollback
   window closes.

Configuration names and capability differences are documented in the paired
release notes. Never copy a v1 Python source tree over v2.

## Upstream synchronization

Only commits merged into upstream `master` are candidates for the regular v2
sync. Large unmerged pull requests are reviewed as design input, not treated
as releases. The current v2 refresh keeps Python 3.10–3.12 and the pinned
llama.cpp b7798 ABI compatibility; upstream's Python 3.14 and b10621 runtime
assumptions are not adopted automatically. See
[the upstream synchronization guide](../upstream-sync-guide.md)
for the divergence guard and merge procedure.
