# Verification And Cleanup

This fork now has a repository-level verification gate for production readiness work. It is intentionally composed from smaller isolated checks so server, no-GUI CLI, and Web Console behavior can be tested without installing global dependencies or leaving generated files behind.

![CapsWriter verification pipeline](assets/verification-pipeline.svg)

## One-command local gate

```bash
python scripts/verify_all.py
```

The command runs:

| Step | Command | Coverage |
|---|---|---|
| CLI | `python client/cli/scripts/verify.py` | CLI syntax, multipart upload, mock HTTP transcription, output files, Linux/Windows TTS command selection |
| Server | `python -m compileall fork_server check_http_api.py start_server_docker.py` + HTTP unit tests | HTTP sidecar, Docker entrypoint, diagnostic script syntax, dependency-light request limit tests |
| Web | `npm ci --no-audit --no-fund` then `npm run verify` in `client/web` | React/Vite tests, TypeScript, production build, web clean script |
| Optional Web image | `docker build` + temporary `docker run` smoke check | Production Nginx/static image can build and serve `/health` + runtime `/config.js` |
| Optional live HTTP | `client/cli/capswriter_cli.py health` + optional known-audio transcription | Real server health and model-backed STT when configured |
| Cleanup | `python scripts/clean.py` | Removes build/cache/pycache artifacts |

The web dependency install is scoped to `client/web/node_modules`. Nothing is installed globally.

## Options

Skip web verification:

```bash
python scripts/verify_all.py --skip-web
```

Use existing `client/web/node_modules` and fail if it is missing:

```bash
python scripts/verify_all.py --no-web-install
```

Add a live server health check:

```bash
python scripts/verify_all.py --http-base-url http://127.0.0.1:6017
```

Add a model-backed transcription smoke check with a known audio sample:

```bash
python scripts/verify_all.py \
  --http-base-url http://127.0.0.1:6017 \
  --http-audio /path/to/known-speech.wav \
  --http-expect "expected transcript text"
```

If `--http-expect` is omitted, the gate only requires a non-empty transcript. Keep release-candidate sample audio outside Git unless it is small, redistributable, and intentionally part of the repository.

Build the Web Console production image as part of the gate:

```bash
python scripts/verify_all.py --docker-build-web
```

This uses the temporary image tag `capswriter-web-console:verify`, starts a temporary container named `capswriter-web-console-verify`, checks `/health` and `/config.js`, then removes both during cleanup.

With auth:

```bash
python scripts/verify_all.py \
  --http-base-url http://127.0.0.1:6017 \
  --http-key sk-local-dev
```

Environment alternative:

```bash
CAPSWRITER_VERIFY_HTTP_BASE=http://127.0.0.1:6017 \
CAPSWRITER_VERIFY_HTTP_AUDIO=/path/to/known-speech.wav \
CAPSWRITER_VERIFY_HTTP_EXPECT="expected transcript text" \
CAPSWRITER_HTTP_API_KEY=sk-local-dev \
python scripts/verify_all.py
```

## Cleanup

```bash
python scripts/clean.py
```

Cleanup removes:

| Path pattern | Reason |
|---|---|
| `__pycache__`, `*.pyc` | Python verification output |
| `client/web/dist` | Vite production build output |
| `client/web/.vite`, `client/web/node_modules/.vite` | Vite cache |
| `coverage`, `htmlcov`, `playwright-report`, `test-results` | Test/report output |
| `.drawio-tmp` | Diagram sidecars generated during local authoring |
| TypeScript emitted config artifacts | Guard against accidental `tsc -b` output |

`client/web/node_modules` is not removed by default. It is an isolated dependency directory and is ignored by Git; removing it on every verification would force unnecessary downloads. Delete it manually if a completely fresh dependency install is required.

## CI

[`ci.yml`](../.github/workflows/ci.yml) runs on push, pull request, and manual dispatch:

```text
checkout -> setup Python 3.12 -> setup Node 24 -> python scripts/verify_all.py
```

The publish workflow remains separate. CI verifies source, tests, and local builds; [`publish-server-image.yml`](../.github/workflows/publish-server-image.yml) builds and publishes the server image when maintainers choose to publish.

## Evidence expected before release

For a release candidate, keep these artifacts or logs:

| Requirement | Evidence |
|---|---|
| Upstream merged | Git merge commit in branch history |
| Server syntax and HTTP sidecar valid | `python scripts/verify_all.py` logs |
| Web Console build valid | `npm run verify` logs from inside the root gate |
| CLI valid | `client/cli/scripts/verify.py` logs from inside the root gate |
| Real HTTP server reachable | `--http-base-url` gate output or `check_http_api.py` output |
| Model-backed STT sample works | `--http-audio` + `--http-expect` gate output or `check_http_api.py --audio ... --expect ...` |
| No generated trash committed | `git status --short` plus cleanup scan |

Default CI does not download models or commit binary audio fixtures. Use `--http-audio` for release-candidate evidence when a real server and known sample are available.
