# Deployment guide

> [Documentation home](README.md) · [繁體中文](../zh-TW/deployment.md) · [Troubleshooting](troubleshooting.md)

This guide separates desktop-local, Linux container, Web Console, and remote
client profiles. Choose the smallest network surface that meets the use case.

![Web Console deployment architecture from browser through runtime configuration and the bounded CapsWriter HTTP API](../assets/web-console-architecture.svg)

Text equivalent: a browser loads the static Web Console and its runtime
configuration, then calls the CapsWriter HTTP API directly for diagnostics and
transcription. The Web container does not proxy audio to the server. CORS,
authentication, publish addresses, and HTTPS therefore belong to the browser →
API boundary.

## Deployment profiles

| Profile | Server process | Client | Network recommendation |
|---|---|---|---|
| Windows desktop-local | `start_server_universal.py` or packaged server | Packaged/upstream desktop client | Keep WebSocket and optional HTTP API on loopback unless another local user needs access |
| Linux X11 desktop-local | `start_server_universal.py` | Source desktop client | X11 session only; keep service loopback-local |
| Linux container (`linux/amd64`) | `docker-compose.yml` | Desktop, CLI, TUI, SDK | Default loopback publishing; use a key and trusted network/TLS when remote |
| Server + Web Console | Server Compose + `docker-compose.web.yml` | Modern browser | Browser must reach API directly; configure exact CORS origins and secure-context microphone access |
| Headless automation | Source/container server | CLI or SDK | Prefer key files, bounded timeouts, readiness checks, and a service supervisor |

## Linux container profile

The published server image, locked native Python wheels, CUDA base, and
downloaded llama runtime target `linux/amd64`. ARM64 is not release-gated and
is not currently a supported server-image architecture.

Create deployment-local files:

```bash
cp .env.example .env
cp hot-server.example.txt hot-server.txt
```

Use the configured image, or build the current checkout explicitly:

```bash
docker build -f docker/server/Dockerfile -t capswriter-server:local .
CAPSWRITER_SERVER_IMAGE=capswriter-server:local \
  docker compose up -d capswriter-server
```

Without `CAPSWRITER_SERVER_IMAGE`, Compose uses the image reference declared in
`docker-compose.yml`. Pin an immutable image digest for controlled production
rollouts rather than relying on a moving channel tag.

Core settings:

```dotenv
CAPSWRITER_MODEL_TYPE=qwen_asr
CAPSWRITER_INFERENCE_HARDWARE=auto
CAPSWRITER_BACKEND_PROBE_TIMEOUT=300
CAPSWRITER_SERVER_PUBLISH_HOST=127.0.0.1
CAPSWRITER_SERVER_PORT=6016
CAPSWRITER_SERVER_MAX_WEBSOCKET_CONNECTIONS=8
CAPSWRITER_SERVER_MAX_WEBSOCKET_TASK_SECONDS=3600
```

The base Compose file has no GPU reservation and therefore starts on CPU-only
Docker hosts without requiring an NVIDIA runtime. For an intentional CPU
deployment, set `CAPSWRITER_INFERENCE_HARDWARE=cpu` and use only the base file.
To expose NVIDIA devices, set `CAPSWRITER_GPU_DEVICE_COUNT` to `all` or a
positive integer and add `-f docker-compose.gpu.yml`:

```bash
CAPSWRITER_GPU_DEVICE_COUNT=all \
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  up -d --force-recreate capswriter-server
```

With the GPU override, `auto` prefers a usable backend. If either GPU runtime
bootstrap or configured engine construction fails, the entrypoint must disable
every CUDA and Vulkan path, prepare the CPU runtime again, and pass a second CPU
engine probe before startup. A failed or timed-out CPU probe refuses startup;
there is no unverified fallback. Each supervised probe uses
`CAPSWRITER_BACKEND_PROBE_TIMEOUT` (default `300` seconds, valid range `> 0`
through `1800`). This cannot recover from a host-level Docker GPU reservation
failure, which happens before the entrypoint and is why the override is explicit.

Linux Intel/AMD iGPUs use the separate `/dev/dri` override. Record the host's
numeric render/card group IDs in `.env`, then recreate the service:

```bash
stat -c '%n %g' /dev/dri/renderD* /dev/dri/card*
# CAPSWRITER_DRI_RENDER_GID=<renderD node GID>
# CAPSWRITER_DRI_VIDEO_GID=<card node GID>
docker compose -f docker-compose.yml -f docker-compose.igpu.yml \
  up -d --force-recreate capswriter-server
```

The image includes the Mesa Vulkan ICD. The host kernel driver, device nodes,
and correct group access are still required; the fallback GIDs `109` and `44`
are not portable across distributions.

### Model persistence and bootstrap locking

Base Compose stores `/app/models` in the `capswriter-server-models` named
volume. Docker preserves it across container recreation and initializes its
ownership for the image's non-root `appuser`; `docker compose down -v`
deliberately removes it.

For operator-managed host files, use the explicit bind override after making
the directory writable by the image user:

```bash
image="$(docker compose config --images | sed -n '1p')"
uid="$(docker run --rm --entrypoint id "$image" -u appuser)"
gid="$(docker run --rm --entrypoint id "$image" -g appuser)"
mkdir -p models
sudo chown -R "$uid:$gid" models
docker compose -f docker-compose.yml -f docker-compose.models-bind.yml \
  up -d capswriter-server
```

During bootstrap, `appuser` must be able to create the lock, download,
staging, and readiness-marker files. Containers sharing model storage serialize
mutation through `/app/models/.capswriter-bootstrap.lock`; the bounded wait is
controlled by `CAPSWRITER_MODEL_BOOTSTRAP_LOCK_TIMEOUT` (default 1800 seconds,
valid range `> 0` through `86400`). The downloader rechecks readiness after
acquiring the lock. A fully warm runtime with archive cleanup disabled performs
no write and does not create the lock, so read-only warm starts remain valid.
Use only volume drivers/filesystems with coherent POSIX advisory locking; do
not assume NFS/SMB semantics without validation.

Warm readiness is content-based, not an existence check. Schema-2
`.capswriter-model-ready.json` markers bind the selected archive identity and
SHA-256 to a size/SHA-256 manifest of every required model artifact. Schema-2
`.capswriter-llama-ready.json` markers similarly bind the selected CPU/Vulkan
runtime archive and hash every installed `.so` in each inference directory.
Startup re-hashes the files and requires an exact marker match, so same-size
corruption, a stale backend marker, a missing runtime marker, or a changed
library invalidates readiness and enters the locked repair/bootstrap path.

### Enable the HTTP API

The API remains off until explicitly enabled:

```dotenv
CAPSWRITER_HTTP_API_ENABLE=true
CAPSWRITER_HTTP_API_BIND=0.0.0.0
CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1
CAPSWRITER_HTTP_API_PORT=6017
CAPSWRITER_HTTP_API_KEY_FILE=/run/secrets/capswriter-http.key
CAPSWRITER_HTTP_API_CORS_ORIGINS=
```

The key-file path must exist inside the container; add a read-only secret mount
in a local Compose override. Explicit `CAPSWRITER_HTTP_API_KEY` takes precedence
if both forms are configured.

Uncomment the HTTP mapping under `ports:` in `docker-compose.yml`, then recreate
the service. Keep `CAPSWRITER_HTTP_API_PUBLISH_HOST=127.0.0.1` unless a remote
client genuinely needs direct access.

An enabled API bound inside the container to `0.0.0.0` is still host-loopback
only when the published host address remains `127.0.0.1`. These are separate
network boundaries.

### Readiness and operations

```bash
docker compose ps
docker compose logs --tail=200 capswriter-server
curl http://127.0.0.1:6017/health
curl http://127.0.0.1:6017/ready
```

- `/health` proves that the HTTP process responds.
- `/ready` checks router/model-worker/ffmpeg readiness and reports configured
  operational limits. Route production traffic only when it is ready.
- Compose health also checks the WebSocket server and, when HTTP is enabled,
  requires HTTP readiness.

First startup may remain in the health-check start period while models and
runtime libraries download. Inspect logs before treating that delay as a crash.

## Windows and source-server profile

Use [`start_server_universal.py`](../../start_server_universal.py) for desktop
source and Windows package paths. It preserves normal upstream server behavior
when the HTTP API is disabled; only validated `CAPSWRITER_HTTP_API_*` values are
applied when enabled.

Loopback example in PowerShell:

```powershell
$env:CAPSWRITER_HTTP_API_ENABLE = "true"
$env:CAPSWRITER_SERVER_ADDR = "127.0.0.1"
$env:CAPSWRITER_HTTP_API_BIND = "127.0.0.1"
$env:CAPSWRITER_HTTP_API_PORT = "6017"
$env:CAPSWRITER_HTTP_API_KEY = "replace-with-a-long-random-token"
python .\start_server_universal.py
```

The upstream desktop WebSocket default is `0.0.0.0`; the explicit
`CAPSWRITER_SERVER_ADDR` line is what makes this whole example loopback-only.
Omitting it preserves upstream behavior for compatibility.

For a packaged release, validate the same behavior through the packaged server
executable; do not assume a source run proves hidden imports or bundled native
libraries. See [desktop portability](desktop-portability.md).

## Web Console profile

Build and start the static application:

```bash
CAPSWRITER_WEB_API_BASE=http://127.0.0.1:6017 \
  docker compose -f docker-compose.web.yml up -d --build capswriter-web
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/config.js
```

The default Web publish address is loopback. For a remote browser:

1. publish the Web service on a deliberate interface or reverse proxy;
2. set `CAPSWRITER_WEB_API_BASE` to an API URL reachable from the browser, not
   merely from the Web container;
3. add the exact Web origin to `CAPSWRITER_HTTP_API_CORS_ORIGINS`;
4. use HTTPS for non-loopback microphone access;
5. authenticate the API.

Do not inject a shared secret through `CAPSWRITER_WEB_API_KEY` unless every
person who can load `/config.js` is allowed to read it. The container refuses
that configuration unless `CAPSWRITER_WEB_ALLOW_PUBLIC_API_KEY=true`; entering a
token in the browser UI keeps it in page memory instead.

See the detailed [Web Console reference](../web-console.md).

## Persistence and backup

| State | Location | Backup guidance |
|---|---|---|
| Models and archive cache | `capswriter-server-models` named volume; optional `./models` bind override | Back up only if download cost/availability warrants it; keep provenance and checksums with copied assets |
| Downloaded llama runtime | Container writable layer | Recreated by verified bootstrap; do not treat it as persistent backup |
| Server hotwords | Read-only `./hot-server.txt` bind | Back up as configuration; review before moving between model types |
| Server logs | `capswriter-server-logs` named volume | Retain according to privacy/operations policy; transcript logging is off by default |
| `.env` and Compose overrides | Deployment host only | Back up securely; never commit keys or local paths |
| Web history/settings | Browser localStorage | The latest transcript/raw history is plaintext and potentially sensitive; per-browser, best effort, and not a server backup or encrypted-record system |
| TUI/CLI transcript output | User-selected filesystem path | Back up like any other user document |

Compose bind-mounts only the existing `./hot-server.txt` file read-only by
default and refuses to auto-create a missing source path; models and logs use
named volumes. Deleting a container does not remove either named volume or the
host hotword file, while `docker compose down -v` removes named volumes. With
`docker-compose.models-bind.yml`, `./models` becomes operator-managed host
state.

## Remote access and TLS

CapsWriter does not terminate public TLS by itself. For traffic beyond a trusted
host, place a maintained reverse proxy or private overlay network in front of
the API, preserve request-size/time limits, and forward the Authorization
header. Do not weaken authentication merely because TLS exists.

Keep WebSocket and HTTP endpoints private unless they are both required. Use an
explicit CORS allowlist; CORS is a browser control, not an authentication
mechanism for non-browser clients.

## Upgrade and rollback

1. Read [release notes](release-notes.md) and [version policy](versioning.md).
2. Back up `.env`, hotwords, Compose overrides, and any irreplaceable model
   assets or logs.
3. Pin the candidate source commit or image digest.
4. Run the repository/image gate and a known-audio readiness/transcription test.
5. Recreate one instance and verify `/health`, `/ready`, `/v1/models`, and the
   required response formats.
6. Move clients gradually. Keep the previous image digest/source checkout and
   configuration available through the rollback window.

Rollback by restoring the previous immutable source/image plus its compatible
configuration. Do not `git reset --hard` a dirty deployment checkout, and do not
merge the isolated v1 and v2 product generations.

## Production checklist

- [ ] Correct platform/profile selected; unsupported Wayland hotkeys are not assumed.
- [ ] Immutable source or image reference recorded.
- [ ] API off when unused; otherwise authenticated and minimally published.
- [ ] `/health` and `/ready` both pass after restart.
- [ ] Startup logs show a passed configured backend probe, or a passed mandatory CPU fallback probe.
- [ ] Known audio succeeds with the required model/format.
- [ ] CORS contains only deliberate browser origins.
- [ ] No default Web key is exposed unintentionally.
- [ ] Models, hotwords, configuration, logs, and output have explicit retention policy.
- [ ] Rollback reference and operator steps are recorded.

For symptoms and status codes, continue to [troubleshooting](troubleshooting.md).
