# Releasing fork v2

> [Documentation home](README.md) · [繁體中文](../zh-TW/releasing.md) · English

A release is cut by pushing a `fork-vX.Y.Z` tag on a reviewed `master` commit.
[`release.yml`](../../.github/workflows/release.yml) then runs the release
gates, builds the Windows packages from that exact commit and assembles a
**draft** GitHub Release. Publishing stays a manual decision.

## Release matrix

| Asset | Platform | Who it is for |
|---|---|---|
| `CapsWriter-Offline-Client-windows-x86_64.zip` | Windows 10/11 x64 | Desktop client that connects to a remote or container server |
| `CapsWriter-Offline-windows-x86_64.zip` | Windows 10/11 x64 | Server and client on one offline machine |
| `CapsWriter-Offline-source.zip` | Any | `git archive` of the tagged commit |
| `SHA256SUMS` | Any | Checksums of the three ZIPs |
| Container images (not release files) | linux/amd64 | `ghcr.io/df-wu/capswriter-offline-server` and `-web`, published by the image workflows on every `master` push; Linux servers deploy these with [Compose](../compose.md) |

There is deliberately no Windows server-only ZIP (the full package covers it) and
no Linux tarball (the image is the Linux deliverable).

## What the workflow checks

| Job | Runner | Gate |
|---|---|---|
| `resolve` | Ubuntu 24.04 | Tag matches `fork-vX.Y.Z[-suffix]` and its commit is on `master` |
| `verify` | Ubuntu 24.04 | `scripts/verify_all.py --skip-web` (source, docs, upstream divergence, server and script suites) and the functional HTTP API contract in an isolated hash-locked environment |
| `desktop-tests` | Windows 2022 | The same desktop/packaging unit suites as `portability.yml`, plus the dependency-free CLI verifier |
| `windows-package` | Windows 2022 | [`windows-package.yml`](../../.github/workflows/windows-package.yml) for `full` and `client`: hash-locked build, relocation, layout and reparse-point checks, client-only exclusions, `--artifact-self-check` of every EXE, and a `start_client.exe --status` smoke |
| `publish-draft` | Ubuntu 24.04 | Downloads the exact tested ZIPs, adds the source ZIP, writes and re-verifies `SHA256SUMS`, looks up the `sha-<commit>` image digests and creates or refreshes the draft |

Only `publish-draft` has `contents: write`. It never edits a published release:
if the tag's release is already public, the job fails instead. Rerunning it for
an existing draft replaces the assets and only the generated section between
`<!-- release-assets:start -->` and `<!-- release-assets:end -->`; anything you
wrote outside those markers is kept. The draft state is re-checked right before
every write. To also rule out publishing during a running refresh, enable
**Settings → General → Releases → Immutable releases**: GitHub then rejects any
asset change after publication.

## Steps

1. Merge the release PR and wait for `ci`, `portability` and both image
   workflows on `master` to pass.
2. Tag the merge commit and push the tag:

   ```bash
   git tag -a fork-v2.3.0 -m "fork-v2.3.0" <master-commit>
   git push origin fork-v2.3.0
   ```

3. Wait for the `release` workflow. To rebuild an existing tag's draft (for
   example after a flaky runner), run it manually with the `tag` input.
4. Open the draft release, write the highlights and upgrade notes outside the
   generated markers, attach `VALIDATION.md`, and check the image digests in the
   draft notes. If an image is listed as "not published yet", wait for its
   workflow and rerun step 3; your notes are kept.
   Upgrade notes can rely on the [`user/` folder](user-config.md): users extract
   the new ZIP over the old folder only after customizations live in `user/`.
   For the first migration from a release without `user/`, instruct users to extract
   into a separate new folder, then run `start_client.exe --import-config <old folder>`
   from there using the intact old folder. Never overwrite the old folder before import.
5. Publish the draft.

Tags on `maintenance/v1` do not run this workflow: the tagged v1 tree has no
`release.yml`, and v1 stays a source-only line.

## Local dry run

The packaging part can be reproduced on Windows with the commands in the
[build guide](../../assets/BUILD_GUIDE.md). Set
`CAPSWRITER_BUILD_VARIANT=client` for the client-only package.
