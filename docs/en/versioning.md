# v1 and v2 maintenance policy

> [Documentation home](README.md) · [繁體中文](../zh-TW/versioning.md) · English

This fork maintains two product generations. The names **fork v1** and
**fork v2** describe this fork's generations; they are not the upstream
`v1.0` and `v2.x` tags.

![CapsWriter fork maintenance flow: upstream releases merge into v2, while only critical fixes are manually backported to the isolated v1 branch](../assets/version-tracks.svg)

Text equivalent: released upstream commits flow into v2, where Linux,
Windows, API, TUI, Web, and security gates must pass. Legacy v1 never merges
v2. A critical or security fix may be manually backported to v1 and must pass
its separate container, API, and model-asset checks before a v1-only release.

## Tracks and authoritative refs

| Track | Authoritative branch | Upstream lineage | Change policy |
|---|---|---|---|
| fork v1 | `maintenance/v1` | Logical snapshot of upstream `v2.5-alpha` plus the hotword optimization at `3419171` | Critical security, compatibility, and model-asset fixes only |
| fork v2 | `master`; PR #2 product merge at `afc8c58` | Includes upstream master `7d7fac3` (v2.6 plus all later merged commits as of 2026-07-16) | Active cross-platform product development; use short-lived branches and merge-based upstream sync |
| v1 audit snapshot | `archive/v1-legacy` and tag `fork-pre-reset-20260525-1411` | Last pre-reset v1 tree, `b46ca74` | Immutable recovery/audit point; do not develop directly |

The v1 and v2 Git histories diverged before upstream's large `util/` to
`core/` refactor. A merge or bulk cherry-pick between them is unsupported.
Backports are small, reviewed ports with tests written against the target
generation.

## Support matrix

| Capability | fork v1 | fork v2 |
|---|---|---|
| Existing Windows desktop workflow | Preserved legacy behavior | Preserved upstream behavior |
| Linux server container | Maintenance support | Primary deployment path |
| Windows native server | Legacy/manual build | Supported universal entrypoint and packaging gate |
| Scriptable CLI | Legacy scripts only | Supported on Windows and Linux |
| Interactive TUI | Not backported | Supported on Windows and Linux |
| Browser console | Not backported | Supported on modern Windows/Linux browsers |
| OpenAI-style transcription API | Legacy compatibility; security fixes only | Tested `whisper-1` transcription contract with explicit capability errors |
| New features | No | Yes |

“Supported” means that the documented entrypoint has an automated gate for its
portable logic. Hardware-, terminal-, and model-backed release evidence is
listed separately; one Linux container test is never used as proof of Windows
runtime behavior.

## Backport rules for v1

A v1 change must meet every rule below:

1. It fixes a critical/security issue, restores a model asset, or preserves a
   documented external contract.
2. It is implemented against the v1 `util/` architecture. Do not copy a v2
   module wholesale.
3. It includes a focused regression test or an isolated executable smoke test.
4. It does not change v1 defaults unless the old default is unsafe.
5. It is released under a v1-only Git tag. An image may be published only by
   a separate, explicitly reviewed v1 workflow/tag; no such image automation
   is configured in the current v2 tree.

Feature work, UI redesigns, dependency migrations, and broad upstream merges
belong to v2.

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
as releases. See [the upstream synchronization guide](../upstream-sync-guide.md)
for the divergence guard and merge procedure.
