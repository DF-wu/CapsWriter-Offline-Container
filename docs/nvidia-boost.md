# NVIDIA memory-clock pre-boost (fork-v2.1.0 preparation)

**Unreleased candidate.** The latest published release remains `fork-v2.0.0`.
This guide describes the v2.1.0 candidate contract, not completed hardware
qualification or measured latency improvements. See the
[release draft and verification checklist](releases/fork-v2.1.0.md).

Pre-boost asks `nvidia-smi` to lock one selected GPU's memory clock before ASR
work, then reset it after inactivity. It is separate from enabling GPU inference;
ordinary NVIDIA inference does not require the boost permissions override.

## Hardware and permissions

- Confirm the selected device supports runtime memory-clock locking and choose
  a supported MHz value. A valid number in the settings range is not proof that
  the GPU/driver accepts it. There is no universal frequency or all-NVIDIA support
  claim; locking can affect power, heat and other workloads on the same GPU.
- NVIDIA documents `-lmc` and `-rmc` as requiring root. Hopper requires deferred
  memory clocks instead; this runtime feature does not implement deferred clocks.
  See the [official NVIDIA System Management Interface documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#device-modification-options).
- The runtime accepts one **numeric index**, as visible to `nvidia-smi` inside
  the server's execution environment. Verify its GPU identity there; container
  visibility and host enumeration can differ, and indexes may change on reboot.
  UUIDs, lists, shell fragments and an all-GPU selector are not accepted.
  Commands target that physical GPU and can affect its other applications;
  separate server instances do not coordinate clock ownership or resets.

For an operator's later inspection (these queries do not change clocks):

```bash
nvidia-smi -L
nvidia-smi -i <visible-index> -q -d SUPPORTED_CLOCKS
```

An unavailable/unsupported clock query is not permission to guess a frequency.
Leave boost disabled until hardware support and the correct device are confirmed.

## Configuration

| Environment variable | Default | Accepted value |
|---|---|---|
| `CAPSWRITER_GPU_BOOST_ENABLED` | `false` | Boolean; explicit `true` enables boost |
| `CAPSWRITER_GPU_BOOST_DEVICE` | `0` | Integer `0..1023`, one visible GPU index |
| `CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ` | `0` | Integer `0..100000`; `0` means unset; enabling requires an explicit, hardware-supported value `>0` |
| `CAPSWRITER_GPU_BOOST_IDLE_SECONDS` | `1` | Finite seconds `0.1..3600` |
| `CAPSWRITER_GPU_BOOST_TIMEOUT` | `5` | Expert command timeout, finite seconds `0.1..60` |

Safe initial configuration:

```dotenv
CAPSWRITER_GPU_BOOST_ENABLED=false
CAPSWRITER_GPU_BOOST_DEVICE=0
CAPSWRITER_GPU_BOOST_MEMORY_CLOCK_MHZ=0
CAPSWRITER_GPU_BOOST_IDLE_SECONDS=1
CAPSWRITER_GPU_BOOST_TIMEOUT=5
```

Before opting in, replace the clock's `0` with the verified supported value and
confirm the device index, then explicitly set `CAPSWRITER_GPU_BOOST_ENABLED=true`.
Malformed/range-invalid configuration, or enabling with an unset clock, is rejected;
runtime command failures are handled separately as nonfatal failures.

The opt-in, authenticated Docker server settings page exposes typed controls for
enabled, device index, memory clock and idle seconds. The expert command timeout
remains environment-only. Explicit environment values override saved settings;
even an explicit `false` or `0` takes priority. Saving settings requires a server
restart before they affect the worker. Changed container environment requires
recreation with the same deployment files. Neither saving nor restarting grants
GPU permissions.

For native `start_server_universal.py`, explicit boost environment controls opt
into this validated configuration even without HTTP. Without explicit boost env,
existing `config_server.py` Python defaults/customizations remain authoritative;
native saved settings do not expose the Docker boost fields. Do not assume the
legacy Python shell-command configuration has the typed env path's restrictions.

## Opt-in Docker permissions

The default image still runs as `appuser`, with boost disabled. No setting
automatically elevates privileges. The candidate adds
`docker-compose.nvidia-boost.yml`, applied **after** `docker-compose.gpu.yml`:

```bash
docker compose -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.nvidia-boost.yml up -d capswriter-server
```

This is an opt-in deployment example, not a step performed during release
preparation. The permissions override uses `user: "0:0"`, adds `SYS_ADMIN`,
`DAC_OVERRIDE`, `CHOWN` and `FOWNER`, and sets `stop_grace_period: 150s`.
It broadens container privileges; it alone does **not** enable boost or select a
frequency. Both explicit enablement and a supported positive MHz value are needed.
Host driver/container runtime support remains a prerequisite, not a guarantee.
Keep any other required settings/HTTP overrides in the deployment as well.

## Worker lifecycle and recovery

- The worker ensures boost before HTTP, desktop-file and microphone inference.
  Forced CPU, `cpu_only`, and entrypoint full-CPU fallback skip clock commands.
  After loading, the worker checks actual ONNX session providers for SenseVoice,
  Qwen and Fun-ASR: CPU-only sessions skip boost when no GPU decoder is selected;
  either GPU session preserves eligibility, including `low_vram_gpu`.
  The current GGUF binding does not expose actual offloaded layers, and Sherpa
  does not expose its internal sessions. Those components use their loaded
  instance configuration; silent native fallback cannot be certified here.
  The existing early microphone hint remains, and activity is refreshed after
  inference so long tasks do not immediately expire the idle window.
- Idle expiry and normal worker shutdown attempt to reset the selected device.
  Commands have bounded timeouts and bounded retry; a failed boost does not stop
  transcription or count as a successful clock change.
- Resolved controls, including backend eligibility, are passed explicitly to the
  child worker. Fresh-import/spawn workers retain enablement, disablement and
  typed or legacy command configuration rather than reverting to Python defaults.
- When boost is enabled, the graceful worker-stop budget is at least
  `2 * (command_timeout + 4) + 1` seconds for an in-flight command, reset and
  Windows taskkill/reap cleanup. This boost allowance reaches 129 seconds at the
  60-second command maximum; a larger `CAPSWRITER_SERVER_WORKER_STOP_TIMEOUT`
  still wins. The opt-in Compose grace period is 150 seconds; if choosing a larger
  worker-stop timeout, allow corresponding supervisor/container shutdown time.
- SIGKILL, host power loss, driver failure and watchdog forced kills cannot
  guarantee reset. If needed, an administrator must first confirm the corresponding
  **host GPU index** and that resetting will not disrupt another workload, then run
  on the host: `sudo nvidia-smi -i <index> -rmc`. A container index must not be blindly
  reused on the host. This requests default memory clocks, not a whole-GPU reset.

To disable, set enabled to `false`, restart/recreate as appropriate, and verify
clock state. Before returning to the default `appuser`, inspect ownership and
write access for model, log and runtime paths: a first boot or new archives/markers
written as root can leave root-owned volume content. If needed, repair ownership
only on the affected paths to the image's actual `appuser` UID/GID, then validate
non-root startup/readiness before completing rollback. Removing the permissions
override alone does not repair ownership; disabling the setting alone does not
remove already-granted container privileges.

## Separate deployment issue: NFS models

The recorded model-bootstrap failure is a UID mismatch (server UID 1000 versus
NFS lock owner UID 3000), not a GPU boost failure. The local `myServices` recipe
has been updated for a named model volume, default-disabled typed boost variables
and a separate permissions override. It has not reached Portainer's tracked
branch or been deployed; the actual stack has not been changed by preparation.

Source and image workflows share the canonical
`DF-wu/CapsWriter-Offline-Container` repository (the old
`DF-wu/CapsWriter-Offline` name redirects there). Publishing `master` triggers
automatic image publication and guarded `latest` promotion. The `myServices`
deployment recipe is in a separate repository: it must separately be
committed/pushed to the branch Portainer tracks, then its exact remote revision
must be confirmed.

Only **after** Portainer has pulled the correct recipe, the running model mount
is confirmed to be the intended named volume, and readiness/model loading is
confirmed may the old appdata `models` directory be considered for removal after
backup/rollback review. Keep its parent appdata directory and `hot-server.txt`;
do not remove the parent just to clean up models. This guide authorizes no live
deployment or deletion.
