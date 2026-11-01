# P4 CUDA runtime deployment, 2026-11-01

Issue: [#18](https://github.com/DF-wu/CapsWriter-Offline-Container/issues/18). Branch: `perf/p4-cuda-sm61-runtime`.

Public report: https://aionc.dfder.tw/raw/BWZMEN6EYF5QyQq/index.html (share 108, HTTP 200 and byte-for-byte verification passed).

This is a machine-specific deployment record, not a generic installer. The scripts intentionally check the original container IDs, images, fingerprints and original audio hashes. Private state, environment files, downloaded audio and build products were outside git and were removed after collection. Recovery data remains under `~/.local/state/capswriter/p4-cuda-20261101/` with directory mode 0700.

## Result

Original FP16 ONNX encoder and Q5_K_M GGUF weights, now using b7798 CUDA 11.8 SM61 decoder. Production retains its exact container ID, image ID and Portainer-injected Config/HostConfig/network configuration. Immich retains its original ID/image/configuration and was restored before acceptance. HTTP readiness, Docker health, one coexistence transcription, source/runtime integrity and model hashes were verified again after cleanup.

Accepted run: `1793508011791913579`. **52 inference calls:** 36 HTTP latency calls (27 measured + 9 warmups), 10 direct-engine greedy quality calls, 5 full-audio production HTTP smoke calls, and one post-restoration coexistence call. Five normalized greedy outputs match. HTTP output is stochastic because the existing pipeline calls `decode_stream` with its default temperature 0.4 even when the API accepts temperature 0. This behavior was documented, not modified.

`results/attempt-1/` contains 17 earlier HTTP calls; the run stopped before production patching because the checker read console output instead of the native log. `results/attempt-2/` contains 29 earlier HTTP calls; it stopped before patching because strict text equality exposed the HTTP temperature behavior. Both runs restored original services with matching fingerprints. Neither is mixed into the accepted comparison.

## Build and validation

All compilation and package installs took place inside disposable Docker containers/build stages. No host CUDA/compiler/packages, drivers, clocks, power or fan settings were changed. The original model volume was read-only in disposable tests.

The source is b7798 commit `c301172f660a1fe0b42023da990bf7385d69adb4`, tar SHA-256 `4420071ba535356cb62f8cd65edb924dee321942d45b7e5e402df1f2906e1031`. CUDA 11.8 compiler image is pinned by digest. `build_cuda.sh` sets SM61 and `CMAKE_BUILD_RPATH_USE_ORIGIN=ON`; `package_cuda.py` writes archive and per-library hashes. The normal downloader verifies these hashes and atomically promotes all four library directories.

The initial artifact revealed absolute build-tree RUNPATH. An isolated compiler container reconfigured the same build with origin-relative RPATH, relinked eight targets and repackaged it. `results/relocation.txt` records before/after; `results/rpath.txt` verifies `$ORIGIN` without CUDA stubs. The corrected flag is in the committed fresh-build recipe. `results/cubins.txt` verifies SM61 cubins. A clean runtime container without `/src/build` loaded the relocated DSOs with the real injected driver. A separate rollback rehearsal restored 126 original files and left no CUDA DSOs.

Historical sequence (requires this machine's original IDs and fresh private scratch/backup paths; do not rerun against the deployed state):

```sh
rtk proxy python3 docs/reports/experiments/p4-cuda-deployment-20261101/prepare.py
rtk proxy docker buildx build -f docker/server/Dockerfile.cuda --output type=local,dest=/tmp/capswriter-p4-cuda-implementation/bundle .
rtk proxy docker tag sha256:b413de423317afd354ddeaa17898fabbd4ae03bcb8040f261dc029ec451f5ea5 capswriter-p4-base:temporary
rtk proxy docker buildx build --build-arg BASE_IMAGE=capswriter-p4-base:temporary --load -t capswriter-p4-cuda:validated-20261101 /tmp/capswriter-p4-cuda-implementation
rtk proxy python3 docs/reports/experiments/p4-cuda-deployment-20261101/prepare.py --finalize-audio
rtk proxy python3 docs/reports/experiments/p4-cuda-deployment-20261101/deploy.py
```

`deploy.py` starts an isolated worker, per-run markers and an independent watchdog. The inherited exclusive lock prevents overlapping attempts. A deadline fences the worker process group before serialized restoration. Each original service is restarted independently even if another restoration step fails. Acceptance follows both-service recovery and a coexistence request. A selected CUDA backend refuses CPU fallback; general CPU/Vulkan modes and low-VRAM presets remain supported.

The tested patch contains only the changed Python/shell code, runtime DSOs and immutable bundle. It excludes compilers, package-manager state and EGL packages from the validation image. The full source Dockerfile also installs EGL for Vulkan fallback compatibility. The full production Dockerfile was integrated but not rebuilt end-to-end here; the exact deployed base plus a disposable overlay image was built and tested. No GHCR latest tag was replaced.

## Tests, artifacts and cleanup

106 relevant tests: 73 Docker server, 15 compose, 10 environment, 7 upstream divergence, 1 restoration failure regression. Shell syntax and upstream divergence guard passed. `results/verification.json` records exact commands. Console logs have trailing whitespace normalized for git; measurement JSON values are unchanged. `results/final-state.json` verifies original model hashes and exact deployed source files after cleanup. `results/browser-qa.json` records desktop/mobile report rendering and duration controls.

`results/cleanup.json` and `cleanup-cache.txt`: two remaining task containers, three temporary image tags, 31 exact task-owned cache records and scratch removed; all 70 unrelated cache records preserved. Only the duplicated new CUDA archive cache was removed from the model volume. Original model files and unrelated services/resources were retained. No global prune was used.

Generate the shareable page from collected accepted evidence:

```sh
rtk proxy python3 docs/reports/experiments/p4-cuda-deployment-20261101/make_report.py
```

## Persistence and rollback

The running service uses a same-ID writable-layer update. It survives ordinary restart. Recreating from the old image loses the change. Build/publish the updated source image before a future Portainer recreate; the branch has not been merged into master or published over GHCR latest.

Private recovery directory retains `original.tar`, the state/fingerprints, `rollback-entrypoint.sh`, and the verified `cuda-bundle/`. The rehearsed wrapper clears all four library directories before extracting original files, so stale CUDA DSOs cannot remain mixed with the original runtime. Recovery uses the same production container ID and does not recreate its Config/HostConfig/network parameters.

The CUDA decoder adds approximately 1.7 GiB of observed GPU allocation. Immich was restored and a coexistence request passed; Immich peak GPU workload and client release-to-screen latency were not stress-tested. Timing uses only three measured requests per duration/configuration, in sequential blocks with observed clock/thermal variation. It is a local improvement check, not a statistical or broad quality benchmark.
