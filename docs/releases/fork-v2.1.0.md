# fork-v2.1.0 — draft release body and checklist

**Status: unreleased preparation; not a published tag or release.** Latest
published: `fork-v2.0.0` (2026-09-30). No new CI, image, hardware or latency
acceptance is claimed here. Replace every `PENDING` with actual evidence before
publication; retain explicit limitations for anything not qualified.

## Draft release body

This candidate adds opt-in NVIDIA memory-clock pre-boost controls for the Docker
server and explicit environment configuration for the native universal entrypoint.
The upstream baseline remains **v2.6 / `84912d5`**, Python 3.10–3.12 and the pinned
llama.cpp b7798 ABI.

- Fork runtime/package version becomes `2.1.0` (`fork_server.__version__` and
  `pyproject.toml`). `/health` and `/ready` add `fork_version: "2.1.0"`; their
  existing upstream `version: "2.6"` remains unchanged. Runtime metadata does not
  establish that a release tag/image has been published.
- Validated boolean/numeric controls select one visible GPU index, an explicit
  supported memory clock and an idle window. Docker saved settings expose these
  four controls, with environment priority and restart required; the expert
  command timeout remains environment-only. Native startup preserves existing
  Python defaults unless explicit boost env is supplied.
- Boost is ensured before HTTP, file and microphone inference, retains the early
  mic hint, refreshes activity after inference, and attempts reset at idle/normal
  shutdown. Command failures are nonfatal with bounded timeout/retry; enabled
  boost-specific shutdown allowance is `2 * (command_timeout + 4) + 1` seconds,
  including Windows taskkill/reap (129 seconds at the 60-second command maximum);
  a larger configured worker-stop timeout is preserved.
- Forced CPU and full CPU fallback skip clock commands; `low_vram_gpu` remains
  eligible while CUDA is active. Loaded ONNX session providers are rechecked
  before worker readiness; GGUF offload and Sherpa internals retain the loaded
  configuration limitations described in the recovery guide.
  Resolved controls are transferred across worker
  spawn, with fresh-interpreter regression coverage for enabled typed arguments
  and disabled legacy commands.
- The default image remains `appuser` with boost disabled. The optional
  `docker-compose.nvidia-boost.yml`, after the NVIDIA GPU override, explicitly
  selects root, `SYS_ADMIN`, `DAC_OVERRIDE`, `CHOWN`, `FOWNER` and a 150-second stop
  grace period. It alone does not enable boost: explicit `true` and a supported
  positive MHz value are also required. No automatic privilege elevation occurs.

Read the [NVIDIA boost configuration and recovery guide](../nvidia-boost.md)
before opting in. NVIDIA requires root for `-lmc`/`-rmc`; Hopper deferred clocks
are outside this feature. Not every NVIDIA GPU/driver supports runtime locking.
SIGKILL, host power loss and watchdog forced kills cannot guarantee reset;
manual host `nvidia-smi -i <index> -rmc` may be needed after confirming GPU identity
and other workloads. Clock changes affect other applications on the selected
physical GPU; multiple instances do not coordinate ownership/reset. Root-written
model/log/runtime data may need targeted ownership repair to the image's actual
`appuser` UID/GID before non-root rollback. No latency improvement has been
measured for this candidate.

The NFS model UID mismatch is tracked separately: the `myServices` recipe
now includes a named volume, default-disabled typed boost variables and a separate
permissions override in [draft PR #18](https://github.com/DF-wu/myServices/pull/18),
but has not reached Portainer's tracked branch or been
deployed; the live stack is untouched.
Do not remove old appdata models until Portainer has pulled the correct recipe and the
running mount/readiness are confirmed; retain the parent and `hot-server.txt`.

## Exact release references — fill after verification

| Required record | Value |
|---|---|
| Canonical source / image-workflow / release repository | `DF-wu/CapsWriter-Offline-Container` (old `DF-wu/CapsWriter-Offline` name redirects here) |
| Separate deployment recipe repository | `myServices`; tracked branch/revision PENDING |
| Release date / final or RC tag | PENDING; no tag created by preparation |
| Reviewed implementation / PR | [Draft PR #10](https://github.com/DF-wu/CapsWriter-Offline-Container/pull/10); reviewed implementation `312a8866135707855138034846f8e620e9c88675`; final head is recorded in the PR |
| Full 40-character merge SHA / intended tag target | PENDING |
| CI run URL / exact SHA / result | PENDING |
| Portability matrix + Windows package run URLs / exact SHA | PENDING |
| Server image publish/verify/promote run URL / exact SHA | PENDING |
| Server immutable `sha-<full-merge-sha>` tag / `sha256:<digest>` | PENDING |
| Web image publish/verify/promote run URL / exact SHA | PENDING |
| Web immutable `sha-<full-merge-sha>` tag / `sha256:<digest>` | PENDING |
| `latest` digest readback / observation time (server and Web) | PENDING; moving tag is not a release pin |
| SBOM / provenance references for both digests | PENDING |
| Windows ZIP artifact URL / SHA256SUMS / relocation self-check | PENDING |
| Local validation commands / results | [Candidate validation record](validation-fork-v2.1.0.md); exact head SHA recorded in the draft PR |
| Hardware qualification record / GPU-driver-OS / known audio | PENDING; no hardware acceptance claimed |
| Separate `myServices` recipe candidate | [Draft PR #18](https://github.com/DF-wu/myServices/pull/18), `feat/capswriter-v2.1-recipe`, `9184841f5adef68fef32b855c62af9aa275b07e8`; Portainer tracked-branch merge and live rollout remain PENDING |

## Publication and acceptance checklist

- [ ] Review the final diff; confirm upstream `2.6` / `84912d5` is retained and
  runtime/package metadata plus additive health/readiness fields agree on `2.1.0`.
- [ ] Attach exact-SHA root verification, strict pinned API contract, Web
  tests/build, settings/env/native-default and worker lifecycle regression results.
- [ ] Attach Compose validation for default, NVIDIA and NVIDIA+boost layers;
  verify default non-root/disabled behavior and explicit permissions-only opt-in.
- [ ] Record actual GPU identity/index mapping, driver/runtime, supported clock
  value and permission checks on the intended hardware; keep boost off otherwise.
- [ ] On an authorized isolated hardware run, exercise HTTP, desktop-file and mic
  paths, post-inference idle reset, normal shutdown, unsupported/denied commands
  and timeout/retry behavior. Record evidence; mocks are not hardware validation.
- [ ] Record forced-kill recovery limitations and the manual host-reset procedure.
  Any performance claim needs reproducible before/after data, workload and method.
- [ ] Confirm publication authorization before pushing/merging `master`: that
  action triggers automatic image publication and guarded `latest` promotion,
  independently of creating a GitHub Release. Source and image workflows share
  the canonical CapsWriter repository; the `myServices` recipe repository is
  separate. Confirm which revision each workflow actually builds.
- [ ] After actual publication, fill every image run/tag/digest/provenance field
  from registry/workflow evidence and verify it matches the full merge SHA.
- [ ] Separately commit/push the `myServices` recipe to Portainer's tracked branch
  under its own rollout approval; record remote revision, pulled recipe, running
  named-volume mount and readiness before considering old appdata-model cleanup.
- [ ] Keep `hot-server.txt`, its parent, backups and rollback image/configuration.
  Candidate preparation changes neither the actual stack nor old model data.
- [ ] Inspect model/log/runtime ownership after root-enabled use; repair affected
  paths to the image's actual `appuser` UID/GID as needed and verify non-root
  readiness before completing rollback without the permissions override.
- [ ] Fill release date/tag/artifact/checksum references and the hardware coverage
  statement; do not carry forward fork-v2.0.0 results as v2.1.0 evidence.

Paired summaries: [English](../en/release-notes.md) ·
[繁體中文](../zh-TW/release-notes.md).
