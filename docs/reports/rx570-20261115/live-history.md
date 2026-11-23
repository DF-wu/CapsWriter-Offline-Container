# Nov 14 Live smoke and application audit

Prepared 2026-11-15 from immutable retained observations. This is an integration note for the repository report, not a fresh hardware experiment. All paths below use source labels. [Live observations](data/live.json) contains full-precision compact records and a SHA-256 inventory; source references are root + relative file + JSON Pointer or 1-based JSONL/log line. `retention=original-only` explicitly identifies files present only in the immutable original rather than the staged parent packet. No models, arrays, audio, private host paths, service addresses, credentials, or raw log dumps are incorporated in these audit artifacts.

## Executive interpretation

The Nov 14 work establishes one-clip hybrid adapter feasibility, followed by strict two-graph community MIGraphX execution and diagnostic full-application E2E on five small public fixtures. It does not establish a production-ready full GPU configuration. Exact hybrid/MIGraphX text and decoder-token equality coexists with failed embedding qualification on every fixture. Those are separate outcomes. The application changed shape and incurred explicit provider compile intervals even after returning to a previously used shape. A single Live hardware session, serial nonrandomized requests, small dependent fixture set, background extraction/control overlap and instrumentation prevent generalized performance or accuracy inference.

The Live system was Ryzen 9 3900X, 64 GiB, RX570 8 GiB, Linux 7.1.8-arch1-3, Mesa/vulkan-radeon 26.1.7. Do not combine its observations with the Nov 15 installed Arch LTS/Mesa results as if they were the same environment.

## Evidence roots and audit method

- `LIVE-SMOKE`: source packet `rx570-asr-smoke-20261114`; adapter feasibility execution approximately 11:39–11:50 UTC on 2026-11-14.
- `LIVE-E2E`: source packet `rx570-asr-e2e-20261114`; acquisition, strict GPU probes and application execution approximately 12:00–12:38 UTC on the same date.
- The complete parent packet archive SHA-256 is parent-supplied `f53295845c868d0bc1c38247d8ca3afaf7d6733f16c44c56711b2c2170ea6b37`. This audit hashes every source it actually uses independently in [Live observations](data/live.json).

The original offline verifiers were re-executed on 2026-11-15 in memory with only their `summary.json` write removed, preserving the originals. Both exited 0. Smoke checked exact five-call tokens/text, source/model/library hashes, individual placement, clean statuses and kernel interval. E2E checked 196 bundle file hashes, exact fixture/weight hashes, source AST equivalence outside encoder construction, real transport events, decoder-token parity, individual placement, actual two-graph provider events, numerical metrics independently recomputed from retained FP16 NPY arrays using only the Python standard library, guards and shutdown/cleanup. This is offline verification of retained records, not new inference. No runtime installation, model load or hardware manipulation occurred in this audit.

## Study 1: adapter smoke, not full application E2E

`LIVE-SMOKE:scripts/smoke.py` constructs parent-package path/logger stubs to avoid application startup, then imports the unchanged `QwenASREngine` adapter, mel/encoder/decoder implementations, and shared llama binding from clean CapsWriter source commit `193882e8b11787bb01eceafbea588d2a4ef3be90`. The source snapshot manifest hashes copied files. The waveform is decoded with soundfile before timing. Each run creates a stream and accepts the waveform before `perf_counter()` starts around synchronous `adapter.decode_stream(...)`. Thus adapter wall time excludes file read/decode and pre-acceptance, includes adapter inference, and is neither HTTP file E2E nor microphone latency.

Two first attempts failed before model inference because the snapshot omitted `core.server.privacy`, then the isolated environment lacked `srt`. Both raw failure directories remain. Snapshot/dependency corrections did not change numerical inference code. An initially prepared 10.435-second Narsil recording was not transcribed; all measured calls used the pinned LibriSpeech reference below. The earlier initialization failures had touched the Vulkan backend, so “first clip” must not be described as an untouched driver-cache cold start.

Configuration: original FP16 split encoder on ORT 1.23.2 CPU, Q5_K_M decoder via llama.cpp b7798; Python 3.12.16, NumPy 2.2.6, gguf 0.17.1; explicit 8 decoder/batch threads, unchanged ORT threading, seed 1729 reset each call, temperature 0, English, empty context, no aligner, context 2048, requested batch 4096 runtime-clamped to 2048, ubatch 512, flash attention and KQV offload enabled. Engine construction retains the original two-second zero-audio encoder warmup. Models were in tmpfs; construction is not a cold disk, reboot or empty compiler-cache metric.

The smoke has four Vulkan calls (first + three immediate repeats) and only one CPU call. All five emitted the same 22 decoder IDs and transcript “Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.” Reference normalization expands `Mr.` to `Mister`, lowercases and removes punctuation; word edits are 0/17. Every embedding is finite with shape [76, 2048], prompt positions 98, no aborted/retried/limit-hit generation.

### Exact adapter timings

| Path / index | Adapter wall s | Encoder s | Decoder s | Native prefill call s | Native generation call s |
|---|---:|---:|---:|---:|---:|
| vulkan / 0 | 0.805146194 | 0.188424721 | 0.614406829 | 0.260421753 | 0.351609707 |
| vulkan / 1 | 0.632866545 | 0.217750384 | 0.412946597 | 0.002304554 | 0.408281565 |
| vulkan / 2 | 0.643334408 | 0.224497326 | 0.416681719 | 0.002332449 | 0.411721230 |
| vulkan / 3 | 0.632400328 | 0.217748423 | 0.412476792 | 0.002230644 | 0.407757521 |
| cpu / 0 | 1.697982057 | 0.207863372 | 1.487580190 | 0.806682825 | 0.671203613 |

All original full precision is in `smoke.<mode>.records`, attributed to each JSONL line. Vulkan engine construction is 2.083173201000136 s and process-to-loaded is 2.3591585929998473 s; CPU construction is 2.021949844000119 s. The Vulkan three-repeat median is 0.6328665450000699 s, RTF = 0.6328665450000699 / 5.855 = 0.10808993082836377. Construction + first clip = 2.888319395000508 s (a sum of two measured boundaries, excluding imports and small unmeasured intervening overhead). One CPU request gives no warm CPU distribution or general speedup inference. Native timings are synchronous host API call boundaries; asynchronous GPU work may be charged to a later synchronizing call. They are not individual synchronized kernel timings.

### Physical decoder placement and the b7798 caveat

Both FP16 encoder sessions list CPUExecutionProvider only. On Vulkan the actual physical RX570 RADV POLARIS10 is selected, per-layer records assign every decoder layer 0–28 to Vulkan0, and model/KV/compute buffers are on Vulkan0. Vulkan model buffer is 1193.93 MiB, host model buffer 204.02 MiB and compute buffer 304.75 MiB. Device-reported native FP16/BF16/int-dot absence did not prevent this Q5_K_M decoder from executing. It is not evidence of accelerated encoder execution.

The CPU control assigns every individual layer and all buffers to CPU and samples 0% GPU busy. Yet b7798 also emits a generic “offloaded 29/29 layers to GPU” summary for that CPU control, because the binding leaves requested layer count -1 while disabling device selection. The summary alone is not placement evidence; use per-layer devices + buffer placement + physical telemetry. [Live observations](data/live.json) retains every smoke layer assignment with exact log line and source hash.

### Smoke resource observations

| Metric (sampled) | Vulkan (n=6) | CPU (n=5) |
|---|---:|---:|
| temperature_C | 62.000000 | 55.000000 |
| vram_GiB | 2.623829 | 0.935665 |
| gpu_busy_percent | 100.000000 | 0.000000 |
| available_memory_GiB | 51.006542 | 49.705704 |

Approximately one-second sampling, VRAM includes desktop, extrema are not true peaks. Vulkan baseline VRAM is about 0.935 GiB. Guards stop only the task child group for >=85°C, available RAM <8 GiB, kernel GPU fault/reset/timeout, or 600 s. No stop condition or interval GPU fault/reset/timeout was found. No sustained thermal, power, energy, clock-throttling or RAM certification result follows.

The smoke queried the community image manifest, not layers. Compressed total 16,941,808,679 bytes (~15.78 GiB), digest `sha256:60ac8963d7d43899d242973e1a346233f350816971419f423883a46609b116ca`. Both inspected tags resolved to that digest. Expanded storage was not measured. Full image extraction was outside that bounded smoke; no MIGraphX provider initialization or encoder execution happened in Study 1.

## Study 2: real application, acquisition and correction chronology

Unlike the smoke, the full application uses no parent-package stubs. The original CLI runs in a child process, streams multipart file upload over a task loopback HTTP listener, then the real API admission/parser/FFmpeg decode, task routing/multiprocessing.Queue, RecognizerWorker/ModelLoader/TaskHandler/TaskPipeline, encoder/prompt/decoder, merge/formatting/result routing, HTTP response, and original CLI JSON parsing execute. CLI default compatibility model label `whisper-1` does not select the loaded worker; readiness and engine metadata confirm `qwen_asr`. The configured Q5_K_M package is unchanged.

The supplementary WebSocket harness sends real AudioMessage JSON/Base64 float32 PCM in nominal one-second chunks, immediately without real-time pacing, followed by final marker, and checks RecognitionMessage. File waveform reading and opening the socket precede its timer. This is real protocol/worker E2E, but not 5.855 s of live capture, hotkeys, desktop microphone, tray, text injection, LLM postprocessing, or validated aligner latency. Optional forced-aligner weights were missing and the original uniform-character timestamp fallback ran; returned character tokens/timestamps are not generated decoder token IDs or measured word alignment.

Chronology of retained failures and corrections:

1. The first hybrid application startup exited before ready because `markdown` was absent. Isolated GUI-import dependencies were installed (markdown, tkhtmlview, pyperclip, requests and their dependencies; retained freeze records exact final versions). This is not the Nov 15 AF_UNIX/TMPDIR failure.
2. The preliminary Mandarin sample was paired with text inferred from a generic example description rather than an independently established annotation. It was excluded from final quality results and preserved as unscored evidence. Replacing it with pinned Wenet/AISHELL audio failed initially with soundfile “Format not recognised”, then a RIFF assertion. The final successful acquisition has exact WAV hash and independent transcript source. Do not silently count the preliminary sample as scored Mandarin success.
3. An attempted large base-layer stream (8,672,333,352 compressed bytes) truncated with tar “unexpected end of data”. A bounded resume-capable retry was later superseded by the smaller successful runtime path; partial large-layer artifacts were not executed. No whole-image pull or container runtime install occurred.
4. Community ORT 1.29.0 initially could not load `libonnxruntime_providers_migraphx.so` because `libmigraphx_c.so.3` was missing. Python wrapper fallback then exposed a conflicting explicit CPU-EP/disabled-fallback error. A provider name in the CPU wheel was never treated as an AMD runtime.
5. Digest-verified selected image layers plus missing official AMD package contents supplied actual MIGraphX, community HIP/HSA/rocBLAS/MIOpen, shim and compiler/runtime dependencies. Packages were extracted into task tmpfs, never globally installed; patched community files were preserved while adding missing stock dependencies. A private bubblewrap mount namespace maps task runtime to the canonical ROCm layout; host ROCm directory remained absent.
6. With dependencies resolved, frontend graph compilation failed `same_layout: gpu::convolution: Layouts do not match`. Setting extra MIGraphX FP16 conversion off alone did not fix frontend; an independently initialized backend did run (synthetic hidden states in that probe, not the full real-audio encoder).
7. Exact pinned MIGraphX GPU target source exposed a convolution-layout option. `MIGRAPHX_GPU_OPTIONS={convolution_layout:channels_first}`, `migraphx_fp16_enable=0`, optimization ORT_ENABLE_ALL and strict fallback-off settings enabled both unchanged FP16 ONNX graphs. No graph or model weight conversion/rewrite was applied. FP16 IO/storage does not imply every instruction executes natively in FP16.
8. Original numeric gate `relative_l2 < .02 AND cosine > .999` asserted and stopped the initial full-GPU app phase on reference L2 0.03768892247462869, cosine .9992985801590629. Comparison code was changed to retain gate-failure booleans and allow diagnostic application transcript tests. The gate was not relaxed or declared passed. Final application transport success is additional evidence, not a replacement numerical qualification.

The only numerical runtime source edit is an explicit provider/strict-session construction branch. Offline AST verification confirms all non-constructor encoder methods are identical to original. Observation wrappers add host timings, seeded sampling and embedding saves; no model weights, embedding shape semantics, mel implementation, prompt logic or decoding implementation changed. Those wrappers can affect timing.

### Runtime acquisition exact compact measurements

| Component layer | Compressed bytes | Selected expanded bytes | Extraction elapsed s |
|---|---:|---:|---:|
| `sha256:c146440a37b2e2f229073b93ed51b35196293558b1fec3d4d6957ed38881484a` | 261399299 | 1608220387 | 22.657343468 |
| `sha256:e7f78f7c9b6a356bd5acdfaf1f693f0c4f489af0d33c560bef0b4d0af3a97a1d` | 27816163 | 27929318 | 2.928791695 |
| `sha256:c18c552fe5ff2cd0f1ccb41c5b17b015b1c15de535ddf33a1a202655c3fb790f` | 12447 | 42584 | 0.971343917 |

Missing official AMD package compressed bytes total 199,408,112 (~190.17 MiB): amdrocm-base10.1, llvm10.1, profiler-base10.1, runtime-dev10.1, runtime10.1, sysdeps10.1. Each retained package record includes expected SHA-256, checked before extraction. Useful expanded runtime was reported about 2.6 GiB; entire target directory about 7.5 GiB including models, partial unused extraction, environments/caches/evidence. These rounded `du`/inventory observations are not model memory requirements. No huge builder stack was needed for this demonstrated path.

Runtime source revision Schaka/rocm-gfx803 `8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78`, image digest above. ROCm header version 10.1.0; patched HIP 7.16.26385 / 7da43026e4; MIGraphX 2.18.0 / 9567291-dirty; rocBLAS 5.7.0 / 39d8d60365-dirty. ORT wheel `onnxruntime_migraphx-1.29.0+gfx803-cp312-cp312-linux_x86_64.whl`, SHA-256 `b34f7ae304029a17823619b7331d0458c19dac8260ba3cb273ba8258784dd73d`. This is community gfx803 enablement, not official AMD GPU support certification.

Strict flags: requested MIGraphX device 0 only; exhaustive tune 0; extra FP16 pass 0; `session.disable_cpu_ep_fallback=1`; `InferenceSession(..., enable_fallback=False)` and post-creation `disable_fallback()`; channels_first option; HSA override gfx version 8.0.3, pre-Vega enablement, visible HIP device 0; gfx803 SGEMM shim preload. Actual E2E profiles have 70 frontend + 9 backend MIGraphX provider fused-node events, zero CPU provider events (including constructor warmup). `get_providers()` still returns MIGraphX and CPU. Registered/listed CPU does not mean executed CPU nodes; actual node events plus disabled fallback establish the strict placement claim. These are fused ORT nodes, not GPU instruction tracing.

## Fixtures and licensing

All final audio is mono 16 kHz. Fixture metadata, transcripts, source URLs, sample counts and hashes are in `e2e.fixtures`; fixture SHA values also bind each HTTP result. LibriSpeech reference/short/long are pinned to Hugging Face dummy dataset revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`, parquet SHA-256 `4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603`. IDs: reference 1272-128104-0000; short 1272-128104-0014; long 1272-128104-0004. Wenet/AISHELL sample and independent annotation are pinned to commit `be6ad97fdb512589df113042f7ff3a29d392fb38`, sample `aishell-BAC009S0724W0121`.

| Fixture | Samples | Duration s | SHA-256 |
|---|---:|---:|---|
| reference | 93680 | 5.855 | `4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41` |
| short | 35920 | 2.245 | `5e7e3b96d975c74c952209a4d3a0731ec57019689685e1cf5f3c73a76f6f8a36` |
| long | 470400 | 29.4 | `07244790e9a8300bfcbf12c28ac5230792e75238d03b2ac167a72bf3943c5404` |
| mandarin | 68496 | 4.281 | `2f9fc9c912bb71c85fb286cb88b599c81efb8f727c727a5ea8f6d1c89c55ac13` |
| noninteger | 95653 | 5.9783125 | `143fa201adc2c27fe2c4605a29d5386ff521aa034fc613f2bd4870f3cc2162d7` |

Noninteger is the complete reference plus exactly 1,973 zero samples, written as PCM16 FLAC. It is dependent on reference and cannot count as an independent speech trial. Short and long are from the same speaker/chapter as reference. No source annotation is supplied to the recognizer as context.

**License gap:** authoritative license text for the actual recordings was not retained in either Live evidence root inspected here. Public URL/provenance is not a redistribution license. This audit does not infer that Wenet's source-code license automatically covers embedded AISHELL audio or that the dummy dataset changes upstream recording terms. For the public archive use hashes and exact download provenance, omit recordings, and explicitly leave audio redistribution unverified unless a separate authoritative-source check resolves it. The derivative silence fixture inherits the recording's unresolved redistribution status. No current license webpage was fetched by this read-only local audit.

## Complete final HTTP application observations

All seconds below cover complete CLI process invocation: process creation/import, file read/upload, server admission/FFmpeg decode/queue/inference, response and CLI exit. Files are on tmpfs. These are not adapter times, encoder-only times or microphone timings. Original application decode temperature remains adapter default 0.4: the parsed HTTP temperature field is not propagated by the unchanged worker. Seed 1729 is reset per task; eight decoder threads, original context 2048/batch/ubatch/flash-attention/KQV and empty context are retained. The preceding smoke instead used temperature 0.

Final order for hybrid and MIGraphX: reference first, reference repeat 1, reference repeat 2, short, long, verified Mandarin, noninteger, then WebSocket reference. CPU has one HTTP request for each fixture, no same-shape repeats and no WS. Historical `phase=warm` only means process already exists, not that the shape is warm.

| Request | CPU CLI s | Hybrid CLI s | MIGraphX + Vulkan CLI s |
|---|---:|---:|---:|
| reference | 1.812279947 | 0.781071840 | 17.520089144 |
| reference repeat 1 | not measured | 0.749752336 | 0.675352948 |
| reference repeat 2 | not measured | 0.731020069 | 0.670904294 |
| short | 0.918203159 | 0.414108557 | 17.020726935 |
| long | 7.524728678 | 2.526945670 | 19.227700989 |
| mandarin | 1.089001257 | 0.499074358 | 16.918802816 |
| noninteger | 1.821321908 | 0.736071002 | 17.160271009 |

### Startup and phase boundaries

| Path | Engine construction s | Start to ready s | Ready + first HTTP s | Same-shape repeat median s (n) |
|---|---:|---:|---:|---:|
| cpu | 2.004868200 | 2.718318731 | 4.530598678 | not measured (0) |
| hybrid | 2.156564121 | 2.823988282 | 3.605060122 | 0.740386202 (2) |
| migraphx | 19.685777671 | 20.386495020 | 37.906584164 | 0.673128621 (2) |

Engine construction includes two-second zero-waveform warmup. Process-ready includes application/server startup. Ready + first HTTP is an arithmetic sum of two measured intervals, not a continuously measured application stopwatch; report that formula. Host timers (`perf_counter`) wrap encode/frontend/backend/prompt/decode. `queue_s=time.time()-task.time_submit` uses epoch wall clock rather than monotonic and is subject to clock corrections. Encoder original `reported_s` uses `time.time()`; probe outer encode timer uses perf_counter. Do not conflate them. Phase totals overlap hierarchically; pipeline already includes encoder and decoder, and encoder includes frontend/backend/mel overhead. Do not stack overlapping totals. Native `t_prefill/t_generate` values are seconds even where schema comments suggest milliseconds.

Instrumentation saves embedding NPY arrays after the wrapped encoder timer stops but before the enclosing task pipeline returns; this save overhead is in pipeline/HTTP timing, not in recorded `encoder_s`. ORT profiling also remains on for GPU runs. Protocol/file preparation and phase overhead differ, so subtracting CLI and WS times does not isolate transport cost.

### Explicit compile events

Unlike installed-Nov15 evidence, this Live task has provider `Model Compile: Begin/Complete` logs. The exact eight local log intervals below are timestamp differences between those events, not an isolated external compiler profiler or individual kernel timings.

| Source log lines (1-based) | Logged compile interval s | Interpretation |
|---|---:|---|
| 20–22 | 0.677694494 | frontend during constructor |
| 25–26 | 16.697768479 | backend warmup during constructor |
| 43–44 | 16.519116912 | first HTTP reference |
| 80–81 | 16.439251124 | short shape |
| 89–90 | 16.966776593 | long shape |
| 98–99 | 16.273749530 | Mandarin shape |
| 107–108 | 16.312417584 | noninteger shape |
| 118–119 | 16.462257223 | returned reference shape for WS |

The two immediate reference repeats are fast, but changing shape costs about 16–19 s end-to-end in this Live application. Returning after noninteger incurs another backend compile interval and WS 17.20418686300036 s (hybrid WS 0.6387244190000274 s). Do not label all `phase=warm` requests cache hits. No fixed-60-second padding, buckets, explicit persistent graph cache, dynamic-shape compiler redesign, or alternative-backend mitigation was measured.

Standalone encoder probes have two passes per fixture on CPU ORT1.23.2, CPU community ORT1.29.0 and strict MIGraphX ORT1.29.0. The second pass is just one measured same-shape warm encode per fixture. Their warm ranges (CPU about .114–.802 s; GPU about .052–.256 s) cannot substitute for the HTTP study or independent repeat distributions. Exact first/repeat values and both outer/reported timers are preserved in `e2e.encoder-*` records.

## Numerical qualification and transcription/error analysis

Let c be FP16 CPU embedding converted elementwise to float64, g analogous GPU embedding, d=g-c, N number of elements. Retained comparison computes max absolute = max |d|, RMSE=sqrt(sum d²/N), relative L2=||d||₂/||c||₂, cosine=(c·g)/(||c||₂||g||₂), with whole arrays flattened by norm/dot. The denominator is the original ORT1.23.2 CPU reference norm. This is not mean relative error, percentage of mismatched elements, or WER. Finite arrays and shape/dtype match were checked; every same-GPU repeat array is bitwise equal. Independent stdlib verifier recomputed metric tolerances `rel_tol=1e-9, abs_tol=1e-10` from NPY bytes.

Metrics in this reading table are rounded for legibility; [Live observations](data/live.json) retains the full-precision values and exact source selectors. Rounding here does not change the numerical gate calculation.

| Fixture | max | RMSE | relative L2 % | cosine | CPU1.29 vs1.23 L2 % | GPU vs community CPU L2 % | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| reference | 0.003838 | 0.0006735 | 3.7689 | 0.9992986 | 0.1004 | 3.7730 | FAIL |
| short | 0.009384 | 0.0008740 | 4.6669 | 0.9989106 | 0.1165 | 4.7192 | FAIL |
| long | 0.028595 | 0.0009943 | 5.7955 | 0.9983275 | 0.2779 | 5.8255 | FAIL |
| mandarin | 0.010742 | 0.0007919 | 3.6889 | 0.9993270 | 0.0934 | 3.6701 | FAIL |
| noninteger | 0.020172 | 0.0008889 | 4.5997 | 0.9989425 | 0.1795 | 4.4944 | FAIL |

Every fixture fails the conjunction, even reference and Mandarin whose cosine individually clears .999. The CPU-version control changes L2 by only .093–.278%, while GPU versus same-build community CPU remains 3.67–5.83%. This rules out attributing most discrepancy solely to ORT version, but does not isolate a proven cause in arithmetic, kernels, layout, fusion or precision. The record supports no claim that disabling the FP16 pass restored numerical equivalence.

The five paired final HTTP fixture outputs and generated decoder IDs are exact between hybrid and MIGraphX; repeat references and WS also match. CPU and hybrid have the same normalized words, but long uses `Mr.` on CPU versus `Mister` on Vulkan and has 89 versus 86 generated IDs. English normalization lowercases, expands `Mr.` to `Mister`, strips punctuation, retains apostrophe-internal words; Mandarin scores individual Han characters after punctuation removal. WER/CER = Levenshtein edit distance / number of reference units, not text equality. Reference and noninteger are 0/17 words, short 0/5 words, verified Mandarin 0/12 characters. Long is 4/68 = 5.88235294117647% WER on every mode. Name spellings Linnell/Linell and Birket/Birkit plus “at em”/“Adam” account for the four edits; punctuation and Mr./Mister are deliberately normalized. Avoid calling all fixture outputs error-free.

This report's fixture reference is independently supplied annotation, not perfect ground truth for arbitrary spoken language. Hand-selected small samples, one speaker/chapter for English, one short Mandarin utterance, repeated same waveform and synthetic trailing silence cannot establish broad multilingual accuracy, long-form stability or acceptable numerical risk. Do not make p-values, invented confidence intervals, population latency claims or general recognition-quality conclusions.

## Resource observations, transport errors and shutdown

| Path | Guard sample n | Max sampled °C | Max total VRAM GiB | Min available RAM GiB | Max GPU busy % |
|---|---:|---:|---:|---:|---:|
| cpu | 32 | 55 | 0.935665131 | 51.358608246 | 0 |
| hybrid | 21 | 63 | 2.624134064 | 52.360069275 | 100 |
| migraphx | 246 | 63 | 3.787498474 | 45.526405334 | 100 |

Across all 697 guard samples (including separate probes) maximum temperature is 63°C, minimum available RAM 45.48982238769531 GiB, minimum free Live overlay 8.97164535522461 GiB. Approximate 0.5-second guard cycle includes a kernel query; these are samples, not true hardware peaks. VRAM includes the active desktop and concurrent task activity; not per-process allocated VRAM or fair isolated memory benchmarks. Initial CPU/hybrid application runs overlapped streamed extraction; MIGraphX application startup overlapped the short community-CPU control. Guards enforced >=85°C thermal cutoff, 12 GiB RAM reserve, 4 GiB Live overlay reserve, GPU-fault/reset/timeout interval regex and 1200-second task workload deadline. None triggered, and complete task interval matched no GPU fault/reset/timeout. No power readings, joules, sustained stress, randomized isolation, memory test or multi-hour thermal validation were performed.

Final hybrid and MIGraphX reject malformed WebSocket messages with close 1008; all three reject invalid audio with HTTP400 “Audio could not be decoded”. No permanent public listener is involved. SIGTERM-to-exit seconds: hybrid .6140580709998176; CPU .5640282360000128; MIGraphX .5640714679993835. Original worker exit=0 and both task listeners closed; no task inference/extraction process remained before cleanup. Readiness and error events are compacted with precise source lines, omitting raw configuration or addresses.

Models, isolated Python environments, task runtime, caches, huge arrays/profiles and partial downloads were removed from target tmpfs after evidence export and independent checks. A task-created uv executable link was also removed. Original Live kernel, existing SSD mount and render/KFD devices remained; host ROCm path stayed absent. This is a verified historical cleanup observation, not an instruction to rerun cleanup or touch current hosts. Reproduction convenience script was assembled from executed component steps and syntax-checked, not itself executed as an extra hardware trial. Do not describe that recipe as independently hardware-validated.

## Shared exact model/decoder provenance

Both Live studies use CapsWriter source `193882e8b11787bb01eceafbea588d2a4ef3be90` and original release Qwen3-ASR-1.7B-q5_k.zip, SHA-256 `f40040fe62a5ef0c09f8699fdbcb30f18bb8ae2bcd515ed4954e1f62b8b0e88f`, 1,951,570,656 bytes. Weights:

- decoder `qwen3_asr_llm.gguf`: `89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5`;
- frontend `qwen3_asr_encoder_frontend.onnx`: `fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd`;
- backend `qwen3_asr_encoder_backend.onnx`: `4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3`.

llama.cpp b7798 source `c301172f660a1fe0b42023da990bf7385d69adb4`, official Ubuntu Vulkan x64 archive SHA-256 `5a4ee2db7e6f9d5c0a04741f79437b433e63862d72a962466b9f28f09f7ffff9`, selected Vulkan library `0826ad3bfa43c30209cf155edb226dab776d98befce72a7df89f4055b61a541d`. Model is qwen3vl with 311 tensors: 113 f32, 169 q5_K, 29 q6_K. Q5_K_M mixed tensor storage is compatible with the exact deployed adapter, not an alternate 709-tensor transcribe.cpp model. The historical public release URLs are retained in source reports and source manifests; source-based alternatives must stay separate from these observations.

## Integration gaps and constraints

1. Retained Live roots do not establish authoritative audio redistribution terms. Keep audio out of public git; publish hashes+download attribution and explicit unresolved terms unless separately checked.
2. Some exact package freezes/model hashes and original successful-path shell helper are original-only in this compact staged packet. `data/live.json.sources` marks all such gaps. Existing original data is immutable; public compact report can carry extracted audited values and source hashes, not pretend the staged packet retains every original.
3. No compiler trace finer than provider log begin/complete, no per-GPU-instruction profiler, no quantitative mechanism for numeric drift, no accuracy corpus, no microphone latency, no energy measurement, no randomized trials, no confidence intervals, no cache/bucket/fixed60 experiment, no kernel/driver support validation across systems.
4. Profile event counts differ by study (Live E2E 70/9; installed counts belong elsewhere). Never copy installed 257/24 into Live.
5. Do not rename successful diagnostic E2E to numerical qualification. Keep all gate failures and failed attempts discoverable.
6. Source report `LIVE-E2E:REPORT.md` rounds output tables; this compact audit uses raw client/instrument records. Whole-array numeric recomputation differs from saved NumPy cosine at floating-point rounding only; preserve original full-precision numbers when generating plots.
7. This audit does not rerun acquisition or fetch linked webpages. Its cited external provenance URLs reflect historical retained records, not new access-date verification. Parent's primary-source review should supply checked access dates/references for broader gfx803/backend claims.

## Source index

Every numeric table above derives from the records in [Live observations](data/live.json), where each source file has a complete SHA-256. Table-cell selectors are the corresponding field names within the retained JSONL line or JSON object. Resource records additionally identify the exact extremum sample and formula. Figure builders should read these values, round only for labels, and retain units and n. Use source identifiers rather than local private directory names in the public report.
