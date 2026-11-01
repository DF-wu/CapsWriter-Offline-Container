# Tesla P4 CUDA decoder implementation plan

**Goal:** Deploy the measured b7798 CUDA SM61 decoder while retaining the original FP16 encoder and Q5_K_M weights.
**Architecture:** Build CUDA 11.8 / SM61 from a pinned, checksum-verified source archive. Register the bundled runtime with the existing verified downloader and atomic runtime promotion. Prefer CUDA on NVIDIA when the bundle exists, retain CPU/Vulkan and presets, and reject CUDA device enumeration failure before engine construction succeeds.
**Spec:** docs/reports/p4-exclusive-container-experiments-20261101.md; issue #18.

## Constraints

Preserve production container IDs, image IDs, Config/HostConfig/network fingerprints, model hashes, and Portainer injected values. Never put CUDA stubs in runtime search paths. Keep original encoder/decoder. No host driver, thermal or power changes. Stop Immich only for bounded GPU validation and restore it. No global pruning.

## Task 1: Verified runtime integration

- [x] Add failing behavioral tests: bundled archive extraction and corruption rejection, CUDA required library, NVIDIA auto preference, CPU/low-VRAM preset and explicit Vulkan, missing CUDA device rejection.
- [x] Run `rtk proxy python3 -m unittest discover -s docker/server/tests -q` and verify the new behaviors fail.
- [x] Add source-pinned builder/package generation, register local asset in `docker/server/download_models.py`, integrate stage in Dockerfile, select CUDA in entrypoint and validate native device registration in shared loader.
- [x] Rerun server tests and relevant env/compose regression tests.

## Task 2: GPU validation and application

- [x] Build artifact in isolated CUDA 11.8 compiler; inspect SM61 cubins and dependencies. Build an overlay validation image from the exact deployed base and modified files.
- [x] Capture sanitized service fingerprints and original runtime rollback files. Stop original CapsWriter/Immich under a restoration guard; run isolated startup and real ASR on original read-only weights, record outputs and latency.
- [x] Copy the tested bundle/code to the original container and restart the same ID; verify CUDA offload, health, HTTP transcription, unchanged settings/model hashes. Roll back on failure; always restore Immich.
- [x] Remove task scratch/test/compiler containers, temporary image tags and exact task-owned cache records, preserving the rollback artifact and verified CUDA bundle privately.

Quality validation adjustment: the existing HTTP worker still uses its default temperature 0.4, so HTTP output equality is not a valid greedy gate. Use identical temperature=0 direct-engine controls in disposable containers; keep HTTP measurements for actual service latency and smoke checks. Commit acceptance only after both original services recover and a coexistence transcription passes.

## Task 3: Review and report

- [x] Independent code review; resolve concrete findings and run affected checks.
- [x] Commit/push branch and record deployment limitations (same-ID writable-layer hotfix disappears on recreate; source Dockerfile is durable integration).
- [x] Publish a sanitized, attractive HTML report with measured results, modifications and caveats using filehost; update issue #18 and #14.
