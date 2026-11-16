# Research archive, evidence index and offline reproduction

The [main English report](../rx570-qwen3-asr-research-20261115.md), [Traditional Chinese reading edition](../../zh-TW/rx570-research-20261115.md), [self-contained HTML](report.html) and [print PDF](report.pdf) archive the November 10–15 Qwen3-ASR quantization/RX570 investigation. This directory is documentation and retrospective analysis, not a runtime integration or deployment patch.

## Reading and evidence coverage

| Research question / result family | Prose | Audited public data and exact selectors |
|---|---|---|
| Nov10 package identity, binary common-F16 tensor comparison | [Source/history appendix §2](source-history.md#2-nov10-decoder-identity-and-common-source-result) | [historical.json](data/historical.json): `model_validation`, `deployed_payload_validation`, `source_files`; original model validation and per-tensor source hash |
| Nov10 primary raw/normalized/script CER/WER, legitimate bootstrap CIs | [Appendix §3](source-history.md#3-nov10-measurement-contract-and-results) | `historical.json`: `pairs`; each aggregate identifies phase/language/scoring/source, exact paired n/errors/reference units; `observations` has raw line, edit counts and output-text hash |
| Nov10 short/long latency, excluded thermal observations, missing schedule | Appendix §3 | `historical.json`: `latency`, `paired_latency`, `audit`, `observations.latency_exclusions`; source hashes of original raw/telemetry/analysis |
| Nov10 auxiliary auto/context/noise/sampling/stress and failures | Appendix §3 | `historical.json`: `condition_effects`, `sampling_variation`, `anomalies`, `failures`, all 874 allowlisted sample records |
| Community gfx803, architecture, alternatives and source-only predictions | [Appendix §§4–6](source-history.md#4-architecture-required-stages-optional-implementations-and-backend-alternatives) | [checked sources](data/checked-sources.json), [scenario/context numbers](data/source-context.json), exact historical source snapshot hash tables; predictions explicitly hypothetical |
| Nov14 direct adapter smoke, actual placement and CPU control | [Live appendix](live-history.md#study-1-adapter-smoke-not-full-application-e2e) | [live.json](data/live.json): `smoke`, source/line selectors and per-call observations |
| Nov14 application CLI/HTTP/WS, startup, strict GPU events, numerical arrays | [Live appendix](live-history.md#study-2-real-application-acquisition-and-correction-chronology) | `live.json.e2e`: per-mode client/worker records, `numeric`, `profile_execution`, exact source hashes/event indices |
| Nov14 explicit compiler intervals, failed attempts, acquisition/resources | Live appendix | `live.json.e2e.compile_log_intervals`, `runtime_acquisition`, `session_probes`, resource extremum selectors and all-guard summary |
| Installed host/source/runtime fingerprints and decoder placement | Main §§4,6.4 | [artifacts.json](data/artifacts.json), [sources.json](data/sources.json); decoder buffer extract omits raw callbacks/addresses; profile event counts in [placement.json](data/placement.json) |
| Installed complete 22-request sequence per mode, repeat medians, RTF, errors | Main §§5–6.2 | [requests.json](data/requests.json), [http.csv](data/http.csv); every datum has private-source filename/line and full precision |
| Installed timing components and complete stable decoder IDs | Main §§6.3,6.5 | [tasks.json](data/tasks.json), [loads.json](data/loads.json), exact task sequence and instrument-line selectors |
| Installed numerical gates versus output agreement | Main §§6.4–6.5 | [numerical.json](data/numerical.json), [diagnostic.json](data/diagnostic.json), [baseline-parity.json](data/baseline-parity.json); summary JSON pointers preserved |
| Installed sampled temperature/VRAM/RAM/RSS/busy and transport | Main §6.6 | [resources.json](data/resources.json) gives extremum line, sample n and units; [transport.json](data/transport.json) retains WS/error/shutdown values only |
| Audio hashes, transforms and public reference/download provenance | Main §4.3; Live appendix | [fixtures.json](data/fixtures.json); acquired Nov10 audio hashes also in historical observations; raw full Nov10 manifest fingerprint retained |

The figures use the same compact data as the tables. `http-repeat` derives ordinary repeat medians and displays all actual repeats. `http-sequence` plots all HTTP observations in sequence. `startup-components` reads load/task rows without adding nested intervals. `numerical-gate` plots every saved comparison and the original conjunctive threshold. `pipeline` is explicitly schematic. `q5-q6-quality` plots retained historical aggregate estimates/intervals. All SVGs include titles/descriptions; Markdown uses PNG with meaningful alt text. PNGs are 2240 pixels wide; SVGs remain vector. Bars start at zero; the expanded cosine scale is labeled.

## Public extract versus private raw evidence

The staging packet was a **private compilation input**, not a publication-approved archive. The packet fingerprint was independently checked as `f53295845c868d0bc1c38247d8ca3afaf7d6733f16c44c56711b2c2170ea6b37`. This repository contains explicit allowlisted extracts only. It excludes full profiles, NPY embeddings, models, audio, full environment/configuration dumps, credentials/auth receipts, personal conversations and unaudited operational logs. Relative labels replace private filesystem roots:

| Label | Private input mapping (descriptive, no host paths) |
|---|---|
| `NOV10` | Original final Q5/Q6 experiment root; the compact packet omitted some raw files, which were found and read in the original archive |
| `STAGED-NOV10` / `QUANT-ID` | Packet's deep investigation / package-identification roots |
| `RX-SOURCE` | Original source-only RX570 assessment and pinned public-source snapshots |
| `LIVE-SMOKE` | Original November 14 adapter validation root |
| `LIVE-E2E` | Original November 14 complete-application validation root |
| `INSTALLED` | Packet's November 15 installed runtime root; complete baseline and app-only MIGraphX continuation kept distinct from partial/failed attempts |
| `SOURCE-FETCH` | November 15 primary-source reader extracts; content hashes identify the fetched representation, not necessarily the original HTTP bytes |

The private mapping is held by the archive owner. Public hashes fingerprint those inputs; **they do not make raw evidence publicly downloadable**. A reader can fully recompute public repeat medians/RTFs, order, conjunctive gates, text/token pairing, historical latency quantiles and primary error totals. Public numerical metrics do not substitute for original embedding arrays. The original Live verifier was re-executed read-only and checked all 196 bundle file hashes and NPY metrics. The installed GPU metrics were independently array-verified in the parent investigation, and this archive rechecks retained summary values, strict profiles and task pairing. CPU/hybrid byte-identical-array receipts are retained privately. Public readers cannot independently recompute those array norms from this compact archive.

The original failed baseline aggregate status, installed AF_UNIX failure and Live numerical assertion are preserved privately and described in the chronology. They are not overwritten by successful resume receipts. No new inference was performed for this archive.

## Offline checks and regeneration

From repository root, using an existing Python 3 interpreter:

```sh
python docs/reports/rx570-20261115/validate.py
python scripts/check_docs.py
git diff --check
sha256sum --check docs/reports/rx570-20261115/SHA256SUMS
```

`validate.py` uses only the standard library and makes no network calls. It checks 66 HTTP/69 task observations, six repeat groups per mode, fixture durations/hash linkage, RTF, text/token agreement, 0/22 numerical passes, CPU/hybrid parity, actual profile-event counts, all 23 historical latency aggregates, primary historical error totals, displayed installed medians/startup rows, figure presence and a targeted privacy scan. It does **not** claim fresh ASR or GPU validation, regenerate retained bootstrap intervals, or recover absent raw arrays.

To regenerate static figures, an existing `rsvg-convert` (librsvg) executable is required; there are no Python plotting dependencies:

```sh
python docs/reports/rx570-20261115/figures.py
```

The script builds deterministic vector primitives, explicit axes/labels and raster versions from compact data. Figure geometry and units are documented in its source. It does not invent uncertainty or synthesize measured points.

The HTML renderer uses existing Python-Markdown, Fontconfig and DejaVu fonts. Optional PDF rendering uses an **already installed** Playwright Chromium; it never downloads a browser or starts a server:

```sh
python docs/reports/rx570-20261115/render.py
python docs/reports/rx570-20261115/render.py --pdf
```

The HTML embeds SVGs and DejaVu fonts and loads no network resources. System fallback renders Chinese reference glyphs; the checked PDF embeds all used font subsets, including CJK. [render.json](data/render.json) records renderer/browser versions and font hashes. The renderer refreshes [SHA256SUMS](SHA256SUMS) for the report publication files, including paired navigation pages. PDF metadata dates vary on regeneration, so PDF byte identity is not an analytical reproducibility claim. Markdown/JSON/CSV/SVG are the primary review artifacts. Font redistribution notices accompany the reading edition in [FONT-LICENSES.txt](FONT-LICENSES.txt).

Private-source owners may recheck source hashes without exposing local paths in the public record:

```sh
python docs/reports/rx570-20261115/validate.py \
  --packet "$PRIVATE_PACKET_ROOT" --nov10 "$NOV10_ROOT" \
  --live-smoke "$LIVE_SMOKE_ROOT" --live-e2e "$LIVE_E2E_ROOT"
python docs/reports/rx570-20261115/extract.py "$PRIVATE_PACKET_ROOT"
python docs/reports/rx570-20261115/extract_historical.py \
  "$NOV10_ROOT" "$TEMPORARY_HISTORICAL_JSON"
```

These shell variables are caller-supplied paths, not system options or secrets. `extract.py` writes only the installed compact `data` files; it reads the private packet and never copies full logs. `extract_historical.py` writes its explicitly named output, independently recalculating edit counts and latency quantiles from raw/telemetry; retained bootstrap intervals are copied and attributed. Compare parsed JSON rather than whitespace formatting. Live compact observations were built by an independent read-only audit; their exact source selectors permit private spot checks, and the optional validator fingerprints all 94 Live sources. Re-executing the complete old hardware harness is neither required nor authorized by these offline commands.

## Audio attribution and redistribution limits

No recordings are committed. The English Live/installed fixtures derive from LibriSpeech via Hugging Face's public dummy dataset at fixed revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`; attribution and exact recording IDs are in `fixtures.json`. The public Mandarin fixture is AISHELL sample `BAC009S0724W0121` from Wenet's fixed test-resource commit, with a separately retained reference annotation. The noninteger and repeated-Mandarin variants inherit their parent recording's rights. Public availability and a repository's software license do not alone establish recording redistribution permission. **Authoritative license text for these exact Live/installed recording copies was not retained in the inspected roots; redistribution remains unverified.** Use the pinned provenance to obtain recordings subject to original terms rather than treating this repository as an audio license grant.

The November 10 corpus used FLEURS Mandarin/English and LibriSpeech test-other via fixed dataset revisions; its retained provenance records CC-BY-4.0. Preserve author/dataset attribution and consult original terms before redistribution. `historical.json.fixtures` retains all 240 recording IDs, dataset repositories/revisions/configs/splits/rows, transformed and original audio hashes, recorded licenses and reference-text hashes; it omits recordings and bulk transcript text. That larger corpus was independently recorded human speech; the installed synthetic repetition is a different test. No Taiwan-accent set was verified. Dataset access does not establish that a model's training excluded those recordings.

## Review boundaries

The change is confined to documentation, static assets and offline analysis/rendering scripts. No source runtime, service configuration, model, global driver, deployment or host state is changed. Publication requires the requested parent evidence/privacy review; this archive does not merge or deploy itself. Unknowns remain explicit: numerical-mismatch cause, isolated installed compiler time, mitigation efficacy, broad accent/accuracy and long-term/microphone stability.
