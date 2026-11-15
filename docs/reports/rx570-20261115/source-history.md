# Source and historical-methods audit for the Qwen3-ASR/RX570 archive

Audit date: **2026-11-15**. This note is an English research appendix prepared from immutable retained evidence and checked primary sources. No models, GPU runtimes, service processes, or acquisition scripts were run. Source-root labels below deliberately replace private filesystem locations. The archive maintainer holds their local mapping. Numeric observations are available in the companion [historical observations](data/historical.json); `extract_historical.py` regenerates that allowlisted extract from a supplied private `NOV10` root using the Python standard library.

## Evidence roots and precedence

| Label | Meaning | Authority |
| --- | --- | --- |
| NOV10 | Original closed Nov10 Q5/Q6 investigation | Definitive retained `results/raw.jsonl`, final `results/summary.json`, `results/audit.json`, model validation and corpus manifest |
| STAGED-NOV10 | Sanitized staging packet, `qwen3-asr-q5-q6-deep-20261110` | Final report and `collected/TECHNICAL_APPENDIX.md`; some top-level checkpoint files predate final analysis |
| QUANT-ID | Staged `qwen3-asr-quants-20261110` | Read-only deployed-package identification and discovery history |
| RX-SOURCE | Original Nov10 source-only `rx570-assessment` | Pinned Schaka code/document snapshots, official ORT snapshots, model config, source manifests and explicit hypothetical scenarios |
| LIVE-SMOKE | Staged `rx570-asr-smoke-20261114` | Nov14 original-adapter smoke evidence and retained source snapshot |
| LIVE-E2E | Staged `rx570-asr-e2e-20261114` | Nov14 real-application E2E report, compiler/source diagnostics and numeric-array comparison |
| INSTALLED | Staged `evergarden-capswriter-runtime-20261115` | Installed three-runtime results; separate report/data chapter owns these measurements |
| SOURCE-FETCH | This audit's Nov15 primary-source fetch results | Extracted page content and URLs; fetch containers are local, not automatically suitable for publication |

**Do not use the top-level `INDEPENDENT-COUNTS.json` as the final denominator.** It records 803 nonwarmup calls (372 Q5, 431 Q6). The final raw contains 881 JSONL records: 7 metadata records and 874 samples, including 14 warmups and **860 measured samples** (372 Q5, 488 Q6). The staged `collected/independent-check.json` independently records the final 860. The final source SHA256 is retained in the provenance tables below. This audit independently re-counted it and recomputed primary raw/normalized edit totals and all 23 retained wall-time aggregate cells from individual records and telemetry.

The Nov10 source/provenance note deliberately predates successful downloads and direct tensor comparisons. Its uncertainty about the F16 candidate being locally validated was appropriate at that stage; the later binary validation supersedes that narrower uncertainty. Its warning that publisher metadata alone cannot identify a historical quantizer/checkpoint remains valid.

## 1. Chronology, failed attempts and what each stage answered

1. **Nov10 package identification:** read-only hashes identified the deployed split package as original FP16 ONNX frontend/backend plus a decoder named Q5. It made no inference request. Earlier Nov1 P4 timings and five quality examples were historical context, not a fresh result. `QUANT-ID/README.md` explicitly treats its accompanying search JSON as discovery, not proof of unfetched WER/model claims.
2. **Nov10 primary-source investigation:** publisher and official metadata distinguished decoder-only `qwen3vl` GGUFs from complete `qwen3_asr` runtime models, identified a same-publisher floating source, and proposed licensed human-speech corpora. The first source-only RX570 assessment used pinned community source and historical P4 observations. Its RX570 envelopes were assumptions, never measured distributions.
3. **Nov10 closed P4 experiment:** the final harness preserved the production adapter and original encoder, tested deployed Q5 against a compatible Q6 and directly requantized a common floating source for tensor comparisons. Earlier setup/provider/storage interruptions and two pilot attempts were excluded. The first pilot lacked materialized image libraries; the second used intrusive token-decode instrumentation. The final harness removed that wrapper. No provider error was reclassified as an ASR-quality failure.
4. **Nov10 interruption/recovery:** guarded acquisition began 18:38:52 UTC; the last persisted call ended 19:00:46 UTC. Recovery inferred a host boot at about 19:14:14 UTC; cause unknown. Both controller and detached watchdog were gone. Original server restart was 19:17:51 UTC, approximately 39 minutes after maintenance began, exceeding the intended 30-minute restoration window. Acquisition was closed rather than rerun. The final analysis explicitly preserves this restoration-bound failure. It is not evidence that a model caused the reboot.
5. **Nov14 Live smoke:** established actual original-adapter Vulkan decoder execution on RX570, with all 29 decoder layers offloaded, followed by separate real-application E2E work. Read timings in its own method/host context; do not pool engine smoke, application CLI, encoder-only and installed measurements.
6. **Nov14 Live E2E:** extracted a community gfx803 runtime locally; an early large-layer stream truncated and was superseded by smaller verified runtime components. Missing libraries initially prevented MIGraphX provider loading. After dependencies were complete, frontend compilation failed at convolution shape/layout validation. Disabling additional FP16 conversion alone did not solve it. Pinned MIGraphX GPU-target source exposed the `convolution_layout` option; `channels_first` allowed both unchanged original graphs to compile/run. Explicit Live logs record `Model Compile`; that supports compilation wording for those events. Numerical gates failed and remained failed while diagnostic app E2E tested transcript/token behavior separately.
7. **Nov15 installed comparison:** original dynamic-shape CPU, hybrid and MIGraphX pairings were measured separately. An AF_UNIX path-length failure prevented the initial application phase; an app-only continuation used short namespace-private temporary paths. A virtualenv was not mounted in the offline test namespace; identical standard-library tests ran with system Python. These are harness failures/adjustments, not discarded model results. No buckets, fixed60 transformation or compiled-cache mitigation ran. Installed phase stalls have no independent compiler timer: describe them as shape-dependent stalls consistent with compilation, rather than assigning the entire interval to measured compiler time.

Evidence pointers: `STAGED-NOV10/collected/TECHNICAL_APPENDIX.md` §§1,9–10; `NOV10/results/interruption.json`; `LIVE-E2E/REPORT.md` runtime acquisition/diagnostics sections; `INSTALLED/REPORT.md` attempts/limitations sections. Full service inspections, credentials, restoration receipts and raw operational logs remain private and are not needed in the public report.

## 2. Nov10 decoder identity and common-source result

The deployed file was **Q5_K_M**, binary `file_type=17`, `qwen3vl`, 311 tensors, embedding width 2048. It mixed **169 Q5_K + 29 Q6_K + 113 F32 tensors**. Q6 `file_type=18` had **198 Q6_K + 113 F32 tensors**. Thus “Q5 pipeline” means decoder quantization, not uniform encoder-and-decoder five-bit computation.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| Deployed Q5_K_M | 1471800896 | `89e198470f41d487ab5b6b8a36a35400b3b1534451c4499236983050fe9193e5` |
| Publisher Q6_K | 1673003488 | `534b653ee0074264c80533df9637e9eb556c04457f30cd1ba08fe48dcb608263` |
| Same-publisher F16 source | 4069675488 | `5b97b1d6b7a8b882314916d364dd0b0650e1683491a974f0c3719cfe8cf8901f` |
| Direct controlled Q5_K_M | 1471801824 | `e2fc4365bc0115c257314815021d78626b09803121ecd2162bb57cab23700c18` |
| Direct controlled Q6_K | 1673003488 | `90299d206261c2cc37c8c438581c58d1c1931507492b052af50f1257e5702b08` |
| Original frontend ONNX | 24063978 | `fc288ba9a0ca732251b33628c803cd5188d3dd77a2cd7336f0f0fd37969ed4cd` |
| Original backend ONNX | 611276553 | `4b0b1e34e8393f29222006aae0719283b0ff28b61f289b6ff8377b3ddbda27e3` |

Publisher revision: [mradermacher/Qwen3-ASR-1.7B-GGUF@664b960e358348943aefcdc26b3cf09eff1fb0ac](https://huggingface.co/mradermacher/Qwen3-ASR-1.7B-GGUF/tree/664b960e358348943aefcdc26b3cf09eff1fb0ac), Apache-2.0. Both direct quantizations used llama.cpp b7798, commit `c301172f660a1fe0b42023da990bf7385d69adb4`; quantizer SHA256 `a906d00fee23305c16fd0fb975c0f850422e2b9bcca0c6bbad636276afd553f2`.

**All 311 deployed Q5 stored tensor payloads matched direct Q5, and all 311 publisher Q6 payloads matched direct Q6 from the same validated F16.** Whole-file hashes differed because metadata differed. Layout, vocabulary tokens, merges, token types and EOS matched. Deployed chat-template/BOS/padding metadata differed; the adapter manually constructs the prompt. This is strong numerical common-source evidence, without reconstructing the original publisher's historical build command or precise upstream checkpoint revision. No Q5-to-Q6 “upquantization” occurred. **Zero inference calls on the controlled derived files completed**, so tensor equivalence must not be called a completed controlled-file recognition experiment.

The [official pinned Qwen config](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/config.json) declares BF16. The publisher's verified F16 decoder is consequently not a proven original official FP16 checkpoint. Official text-model dimensions are 28 layers, hidden width 2048, FFN 6144, 16 query and 8 KV heads, head dimension 128, vocabulary 151936; audio encoder output width is 2048. The audio/text model includes separate components; scaling total model-name parameter counts is not a valid kernel-speed model.

Evidence: `NOV10/results/model-validation.json`, `results/deployed-payload-validation.json`, `sources/tensor-hashes.json`, `sources/*-metadata.json`; the first two are retained with basename-only model paths in [historical observations](data/historical.json).

## 3. Nov10 measurement contract and results

Hardware/runtime: Tesla P4 SM61, 7680 MiB, driver 580.196.07, CUDA 11.8, ORT GPU 1.18.0, Python 3.10.12, NumPy 1.26.4, llama.cpp b7798. VMware exposed 48 Xeon E5-2680 v4 vCPUs; disposable container quota was 16 CPUs and 16 GiB, original decoder threads/batch threads 16/16. CPU cores were shared without exclusive affinity or a captured per-call CPU-load gate. Deployed image digest was `sha256:137c2147e3a8c8c183d690409a8347c2c052779e9b07e2a1e5aea0aa567eb9bd`. Every final process logged 29/29 CUDA decoder offload and CUDA-first encoder sessions. Effective context/batch/ubatch: 2048/2048/512; requested batch 4096 was clamped. Flash attention and KQV offload were enabled.

Primary settings: unchanged `decode_stream`, new stream per call, original read-only FP16 encoders, explicit language, no supplied context, temperature 0, seed 1729; two warmups per process. The engine wall interval excludes model load, file reading, telemetry/cooling and HTTP/network/UI operations. Warmup and process initialization are distinct. Internal phase medians need not sum to the wall median.

The fixed manifest contained 240 independently recorded human utterances and 38.78 minutes: FLEURS Mandarin 120/1361.32 s, FLEURS English 60/596.24 s, LibriSpeech test-other 60/369.23 s. Deterministic SHA256 ordering and reference deduplication preceded inference. FLEURS converted revision: `168de341b3db6859a9bac1c50a2ef5e3b47647e0`; LibriSpeech converted revision: `2b9f39377850ffce6bf6358257ae9f84b2349497`. Recorded source license: CC-BY-4.0. Public test accessibility does not prove training exclusion. No Taiwan-accent corpus was verified; `cmn_hans_cn` must not be relabeled Taiwanese Mandarin. LibriSpeech test-other is harder read speech, without retained per-record natural-noise/accent labels. Lexical digit/name/technical/Latin indicators are overlapping reference heuristics, not acoustic strata or proof of natural code-switching.

Q5 primary acquired/valid: 180/178; Q6: 239/236. The valid paired intersection contained 89 Mandarin and 87 English recordings (44 FLEURS and 43 LibriSpeech). Seven native failures were excluded as infrastructure failures, including plausible partial text: Q5 `libri-other-009`, `libri-other-023`, `noise10db-fleurs-zh-004`; Q6 `long-79-0`, `fleurs-zh-058`, `fleurs-en-034`, `fleurs-zh-097`. Four failed IDs affect the paired primary set; Q6 zh-097 was outside acquired Q5 coverage. A temporary-token lifetime problem was suspected but not established. No failed call was retried; undetected corruption remains a limitation.

Micro CER/WER = sum(edit distance)/sum(reference units), not mean per-utterance percent. Raw Mandarin uses Unicode characters; raw English splits whitespace. Normalization applies NFKC/casefold; Mandarin retains alphanumerics, English removes apostrophes and maps other punctuation to spaces. OpenCC t2s is a separate additional Chinese score. Numeric words/digits and abbreviations remain distinct. Chinese and English units must not be pooled into one accuracy percentage.

| Metric | Paired n | Q5 errors/units | Q6 errors/units | Q6−Q5 pp | Retained paired-bootstrap 95% interval, pp |
| --- | ---: | ---: | ---: | ---: | --- |
| Mandarin normalized CER | 89 | 170/3352 | 167/3352 | -0.089499 | [-0.200461, +0.000000] |
| English normalized WER | 87 | 69/1725 | 71/1725 | +0.115942 | [-0.110869, +0.361239] |
| Mandarin raw CER | 89 | 360/3781 | 360/3781 | +0.000000 | [-0.130787, +0.129806] |
| English raw WER | 87 | 811/1716 | 816/1716 | +0.291375 | [-0.117926, +0.807859] |

For exact unrounded CIs use companion `pairs`, copied directly from the retained summary. The intervals use 10,000 paired utterance resamples, seed 1729. They are legitimate retained Nov10 analysis, unlike confidence intervals one might invent from the small serial Nov15 repeated calls. All displayed quality intervals contain or reach zero; no overall recognition advantage was established. Only seven primary utterances changed normalized error count: four Q6 wins and three losses. Ties in error count need not imply identical text. Script-aware Chinese totals equal the script-sensitive normalized totals here. Raw English WER is disproportionately affected by uppercase unpunctuated references.

Short latency used 15 independent source recordings, five per 2/5/10 s bin, with unaligned prefix crops solely for latency. Calls were randomized within alternating process blocks; the final Q5 block is missing. Repeated calls are not independent speech samples. Timed validity excluded native/call/context errors, temperature ≥87°C, thermal slowdown, graphics clock <700 MHz, or missing telemetry over each call interval ±1 s at 1 Hz. Cooling was outside timing.

| Audio | Q5 valid n | Q5 p50/p95, s | Q6 valid n | Q6 p50/p95, s | Matched n | Mean Q6−Q5, ms [retained clip-bootstrap 95% interval] |
| --- | ---: | --- | ---: | --- | ---: | --- |
| 2 s | 30 | 0.099419/0.201666 | 40 | 0.122869/0.218329 | 30 | 13.981768 [1.144920,30.816773] |
| 5 s | 30 | 0.400908/0.535523 | 39 | 0.436775/0.600791 | 29 | 41.985961 [27.175403,57.753784] |
| 10 s | 30 | 0.757695/0.847033 | 40 | 0.829966/0.972435 | 30 | 92.191229 [73.245908,116.400478] |

Tables here use rounded values for reading; [historical observations](data/historical.json) is authoritative for exact numeric transcription. Matched analysis averages repetitions within each of five clips, then bootstraps the clip means 10,000 times, seed 1729. P95 uses empirical `higher` quantiles. Q6 5/10 s median penalties were about 8.95%/9.54%. The 2 s bin had 18/30 Q5 and 16/40 Q6 empty outputs; its timing difference partly reflects generation behavior and establishes no 2 s accuracy result.

Long 30/60/79 s probes concatenated whole human utterances and silence, three sequences/bin repeated twice. Native-valid all-completed Q5 medians were 3.150/5.272/7.981 s; Q6 2.541/6.867/9.434 s. Thermal-valid counts were only Q5 1/1/1 and Q6 2/2/0. These cannot establish a clean long-audio ranking. Twenty-nine long calls and one short call failed latency gates (30/246 attempted short+long calls); overlapping reasons were temperature 29, low clock 28, thermal slowdown 28, one call/context error and one missing-telemetry event. Largest prompt 1047; largest prompt+reported generation 1245, within context 2048. All test inputs were ≤79 s, below the adapter's 80 s truncation threshold.

Observed process-interval device memory peaks: 2985 MiB Q5, 3147 MiB Q6. Maximum temperature 89°C; thermal slowdown active in 280/431 Q5 and 362/623 Q6 telemetry samples; median graphics clocks 620/645 MHz. These process-interval telemetry statistics are not per-request power/energy integrals or instantaneous allocation peaks. Recorded compute-process count was zero or one, but no continuous cgroup identity map exists. Pre-block unknown-owner gates passed.

Auxiliary findings remain separate from primary quality:

- Auto-language matched baseline error count on all acquired within-decoder subsets: Q5 9+9 and Q6 12+12 Mandarin/English cases.
- Generic context hints increased Mandarin normalized CER by 1.818182 pp Q5 (n9) and 1.377410 pp Q6 (n12); English by 0.478469 pp Q5 (n9) and 0 Q6 (n12). This tests generic hints, not precise hotword/name prompting.
- Artificial 10 dB Gaussian noise increased observed normalized Mandarin CER by 2.118644 pp Q5 (n8) and 1.652893 pp Q6 (n12); English WER by 2.666667 pp Q5 (n9) and 3.585657 pp Q6 (n12). These small auxiliary intervals and denominators are retained in `condition_effects`; no multiple-comparison correction supports broad claims.
- Temperature 0.4 with seeds 1729/1730/1731 yielded 27 Q5 and 35 Q6 acquired calls over 12 base utterances each; one Q5 base had multiple normalized outputs, none Q6. Unequal acquisition prevents a quantization-stability claim. Sampling bootstrap clusters by base utterance, not repetition.
- Synthetic silence/noise stress yielded empty outputs for all acquired seven calls (Q5 3/4 scheduled, Q6 4/4). This does not prove general hallucination absence. There were zero recorded aborts, repetition-heuristic hits, generation caps, capacity overflow or input truncations in acquired final data. A missing/unpersisted call is never counted successful.

## 4. Architecture: required stages, optional implementations and backend alternatives

The current model requires audio preprocessing, an audio encoder, and autoregressive text decoding; **the split into two ONNX sessions plus llama.cpp is an implementation choice**. It enables independent placement: CPU encoder + CPU decoder, CPU encoder + Vulkan decoder, or community MIGraphX encoder + Vulkan decoder. Installing a ROCm image alone does not change this application's explicit encoder provider selection. A `whisper-1` compatibility request label likewise does not imply Whisper inference when the configured engine is Qwen ASR.

Retained `LIVE-SMOKE/source-snapshot/core/server/engines/qwen_asr_gguf/inference/encoder.py`, SHA256 `8e37be8fda85e5fdb3f2fb89befe28971c884f9f02ac9841b29e6501bf78b1f8`, lines 120–260, demonstrates:

- Provider list begins CPU; explicit branches add TensorRT, DirectML or CUDA when available. MIGraphX is absent in the original constructor. The CPU registered-provider name is neither evidence of CPU node execution nor proof of successful GPU partitioning.
- Frontend accepts `(1,128,100)` mel chunks, executed in a per-chunk loop. Concatenated hidden states are sliced to the exact source output-length formula, approximately 13 embeddings/audio second. These are audio prompt positions, not recognized text tokens.
- Exactly, `T=S//160` for S audio samples; `N=13*(T//100)+ceil((T%100)/8)`. This is algebraically equivalent to `get_feat_extract_output_lengths` at lines 109–118. The backend sees N valid positions, and observed installed N values are 76/29/382/56/350/78 for reference/short/long/Mandarin/synthetic-Mandarin/noninteger. Prompt lengths are N+22 in these recorded cases, not a universal prompt overhead for every language/context. Nominal 60-second source arithmetic gives N=780; no such padded inference was executed.
- Backend receives variable sequence length N and an `(1,1,N,N)` attention mask, then outputs width-2048 embeddings. Existing non-DirectML warmup covers two seconds, not every future length.
- Fixed padding is conditional on active DirectML, with zero-padded hidden states, masked padded keys and sliced output. Passing `dml_pad_to=60` while DirectML is inactive does not make MIGraphX fixed60. A MIGraphX fixed-shape implementation would require code, semantic and numerical validation.
- `session.run()` returns NumPy arrays, so the original interface includes host materialization between stages. Device-resident I/O binding or a full runtime could change copying, ownership and synchronization contracts; no measured copy-elimination result exists here.

The older `RX-SOURCE/evidence/local-audit.md` inspected source revision `2aed68d7f793670dee0594e610d88e950c814b9f`, origin HaujetZhao/CapsWriter-Offline. It is historical source evidence only, **not** authorization to write that upstream. The local encoder snapshot and its manifest provide the exact retained implementation bytes; do not invent a public commit for a modified/exported snapshot.

**CUDA versus AMD:** native NVIDIA CUDA libraries are not a direct RX570 backend. The alternatives investigated here are Vulkan for llama.cpp; community HIP/ROCm/MIGraphX for AMD; CPU; and a separately validated full-model runtime. A ROCm PyTorch program may expose the Python name `torch.cuda`; that API compatibility is not evidence that NVIDIA CUDA binaries execute on RX570. No CUDA translation layer was evaluated.

| Candidate | Primary source / retained basis | Necessary validation / present status |
| --- | --- | --- |
| Existing split CPU encoder + Vulkan decoder | Original source and Nov14/Nov15 actual placement evidence | Measured useful variable-duration choice; retain explicit final transcript and backend checks |
| Split MIGraphX encoder + Vulkan decoder | [ORT MIGraphX provider docs](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html), pinned community stack and actual graph profiles | GPU execution achieved; numerical gate failed; shape stalls remain |
| HIP llama.cpp decoder | [Pinned Schaka patch/recipe](https://github.com/Schaka/rocm-gfx803/tree/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/llama-cpp-gfx803) | Wrapper ABI, gfx803 build, packed kernel branch, placement, latency and transcript qualification untested in this ASR pairing |
| Complete transcribe.cpp Qwen runtime | [Pinned model contract](https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/docs/models/qwen3-asr-1.7b.md), [pinned CMake](https://github.com/handy-computer/transcribe.cpp/blob/5700f127e38dd53c4735ea23ef743b4b538eacd4/CMakeLists.txt) | Vulkan/CUDA source options exist; no RX570 test here. Replaces preprocessing/encoder/decoder, so effects cannot be attributed solely to quantization. Whole-model format is not decoder substitution |
| Official Qwen transformers / vLLM | [Pinned official card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/7278e1e70fe206f11671096ffdd38061171dd6e5/README.md) | Genuine Qwen ASR paths exist; community gfx803 text-model vLLM tests do not validate this audio model, speech preprocessing, model registration, kernels or service contract. Explicit FP16/correctness work required |
| Re-export / standalone MIGraphX / other ONNX provider | Existing ONNX graph contract and provider docs | New export precision/layout/opset, dynamic shapes, I/O and adapter behavior require qualification; no measured alternate encoder produced here |
| Whisper or CTranslate2/faster-whisper | Prior repository alternatives audit | Different speech model and quality contract; no current Qwen3-ASR drop-in converter/runtime validated in this study. Do not claim equivalent accuracy or RX570 speed |

The retained earlier binary audit distinguishes a complete `qwen3_asr` model (709 tensors, `enc.*`, `dec.*`) from the decoder-only `qwen3vl` 311-tensor file expected by the adapter. Its complete-runtime model/public author WER and cross-device timing reports are independent author results; they do not validate the present split adapter or this RX570. The earlier pinned transcribe.cpp runtime is `5700f127e38dd53c4735ea23ef743b4b538eacd4` and model revision `0e0a596e3959c3f3f2ab7d443a5b6f0f4a4bba6f`; historical 709-tensor observation is retained in the canonical repository report `p4-q6-alternative-runtime-evidence-20261101.md` §4. This audit fetched its pinned model contract/CMake, but did not reread a model header or build it.

## 5. Community gfx803 evidence and why it did not predict ASR latency

Pinned community source is [Schaka/rocm-gfx803@8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78](https://github.com/Schaka/rocm-gfx803/tree/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78), retained commit dated 2026-10-06. It reports RX470 image validation and maintains local gfx803 patches. This is **community support**, not ordinary official RX570 support. Mutable image tags never substitute for the actually used digest. Later actual work pinned final image digest `sha256:60ac8963d7d43899d242973e1a346233f350816971419f423883a46609b116ca` and ORT wheel SHA256 `b34f7ae304029a17823619b7331d0458c19dac8260ba3cb273ba8258784dd73d`.

The [pinned llama.cpp investigation](https://github.com/Schaka/rocm-gfx803/blob/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/llama-cpp-gfx803/POSSIBLE_LLAMA_CPP_IMPROVEMENTS.md) compares identical Qwen3.5-9B Q5_K_M, fully offloaded on RX470 8 GiB, llama.cpp `34af94cd9ab277632e27caeec2d41de2fd091b31`, patched ROCm 7.14, pp128/tg64:

| Backend | Prefill positions/s | Generated text tokens/s |
| --- | ---: | ---: |
| Stock HIP | 39.55 | 18.21 |
| Packed-dp4a HIP | 58.38 | 17.79 |
| Vulkan | 163.45 | 24.94 |

These are repository-author results, not ours, and not Qwen3-ASR. The packed patch improves quantized prefill without improving the different decoder kernel path. gfx803 lacks native integer dot instructions; unpacking/instruction work and kernel tiling can dominate. A whole-card TFLOPS/bandwidth or inverse-model-size multiplier does not produce a defensible ASR forecast. The graph experiment's broken segmented fast path also means its tested workaround does not establish that all future HIP graph implementations are irrelevant.

Other community vLLM results use different text models, dtypes, revisions and concurrency; they cannot be pooled into an ASR rate. The pinned README explicitly excludes usable BF16 for gfx803 and explains its FP16 fallback/rejection behavior. The official model's BF16 default therefore cannot simply be inherited. Pinned MIOpen wrong-answer and rocBLAS/Tensile notes establish that “compiles” is insufficient evidence of correct output in a patched stack; they do **not** establish the cause of this study's embedding mismatch. Different ORT version, kernels, accumulation, precision or fusion remain hypotheses until isolated.

Official ORT [ROCm EP documentation](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html), accessed Nov15, says that provider was removed since 1.23 and directs applications to MIGraphX. [MIGraphX documentation](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) documents explicit provider selection, FP16/BF16 conversion, tuning and compiled-model save/load/cache controls. Generic package availability does not certify gfx803. A registered CPU provider can coexist with zero CPU node events; actual strict no-fallback configuration and both-graph execution profiles carry the placement claim.

## 6. Shape sensitivity and the untested fixed60 proposal

The phase model is `T_call = T_mel + T_frontend + T_backend + T_prompt/copies + T_prefill + T_generation + T_application_overhead`, with boundaries taken from actual instrumentation. Approximately 13 audio embeddings/second enter decoder prefill, plus text prompt/context positions; generated text-token count depends on content/language/stopping. Text tokens/s is not audio seconds/s. Noninteger durations change the exact output-length calculation; an audio-duration label alone does not identify graph shapes.

Strictly static shapes are not a mathematical requirement of Qwen ASR. They are a possible runtime optimization/specialization policy. The observed stack can execute multiple shapes, but new and returned shapes stall. Live records explicitly include compiler events. Installed records localize most of the delay to backend calls and show shape-dependent behavior without isolating compiler-only time. Returning to a previously used length does not prove a compiled-shape cache survived or was selected.

Fixed60 would pad backend hidden states to approximately 780 positions, apply the correct attention mask, and slice valid output. It might amortize shape specialization if an actual implementation retained a compatible compiled graph; it also introduces padded computation, allocator/workspace pressure and possibly numerical/masking differences. A rough dense-attention N² term grows by 36× for 60 versus 10 seconds and 900× versus 2 seconds; these are **algebraic mask/attention-size ratios, not measured total runtime or memory multipliers**. Other backend operations scale differently, and optimized masked computation may differ. No estimate is a tested speedup.

Potential alternatives are a small tested bucket set, an explicit bounded compiled-graph cache, profile-guided prewarming, direct MIGraphX dynamic shapes, I/O binding/device-resident intermediate buffers, kernel/precision isolation, or retaining the CPU encoder. Every option requires its own semantic/numerical gate and first/new/returned-shape timing. A cache path in official docs proves an option exists, not that the measured build reused every shape correctly. No fixed60, bucket, cache alternative, input-duration cap change, or new full-runtime backend was executed in the installed study. The failed numerical gate barred optional bucket experimentation. A proposal's motivation and its validation status must remain separate.

### Pre-measurement scenario history, retained as assumptions

The Nov10 source-only assessment chose hypothetical Q5 effective rates of 150–500 prefill positions/s and 25–80 generated text tokens/s for patched HIP, or 300–1000 and 35–100 for Vulkan. These were deliberately assumed rates, not measured ASR rates or credible bounds. Hypothetical other-stage budgets were 0.10–0.50, 0.15–0.85, 0.25–1.50 and 0.65–4.50 s at 2/5/10/30 s audio. The workload used prompt/output counts 46/3, 85/11, 150/24 and 410/72; the 30 s output count was itself assumed. Q6 sensitivity multiplied decoder time only by an assumed 0.8–1.3. It did not assert that Q6 was faster on Polaris.

| Audio s | Assumed backend | Hypothetical Q5 total s | Hypothetical Q6 sensitivity total s |
| --- | --- | --- | --- |
| 2 | Patched HIP | 0.230–0.927 | 0.204–1.055 |
| 2 | Vulkan | 0.176–0.739 | 0.161–0.811 |
| 5 | Patched HIP | 0.458–1.857 | 0.396–2.159 |
| 5 | Vulkan | 0.345–1.448 | 0.306–1.627 |
| 10 | Patched HIP | 0.850–3.460 | 0.730–4.048 |
| 10 | Vulkan | 0.640–2.686 | 0.562–3.041 |
| 30 | Patched HIP | 2.370–10.113 | 2.026–11.797 |
| 30 | Vulkan | 1.780–7.924 | 1.554–8.951 |

These totals excluded capture, upload, queueing, cold setup and compilation. They assumed a successful warm GPU encoder and cannot describe the original CPU-encoder path. They were useful for exposing workload sensitivity before hardware tests, and are superseded by actual host-specific observations for recommendation. A later synthesis substituted the fresh P4 median 32 generated tokens for the historical 24 at 10 s: under the same hypothetical HIP assumptions this changes 0.85–3.46 s to 0.95–3.78 s, without calibrating an RX570 forecast.

The historical Nov1 P4 nested-Mandarin engine medians behind the initial workload were Q5 0.133461918/0.302596692/0.583110983 s versus Q6 0.144044989/0.356510041/0.655918234 s at 2/5/10 s; Q5 output counts 3/11/24, Q6 3/12/24. Those were ten repeats of nested clips and five small quality examples, not the Nov10 independent 240-utterance plan. Memory figures 2981–2983 versus 3131–3141 MiB were call endpoints, not true peaks. Historical P4 phases or native DP4A behavior cannot validate gfx803 timing. All original unrounded scenario/context numbers and source hashes are in the allowlisted [source/context observations](data/source-context.json).

## 7. Source gaps and publication boundaries

- No isolated installed compiler timer, cached restart study, randomized independent installed trial design, broad accent/dictation accuracy set, fixed60 implementation, Q6 RX570 experiment or alternate full-runtime RX570 measurement exists in retained evidence.
- No retained result identifies the exact cause of numerical divergence; another model's community correctness patch is context, not diagnosis.
- No retained evidence proves the historical published Q6 build command/checkpoint beyond the later common-source tensor payload match. No controlled-derived-file inference completed.
- Exact installed current-host results belong to their own source-root and timing contract; historical P4 thermal behavior and Nov1 nested clips cannot be transferred to it.
- Fixed60 appears in the requested research scope; no tested artifact for that proposal was found. The discussion above is explicitly prospective design analysis rather than a recovered experiment.
- Source audio should remain uncommitted here. Retain original IDs, pinned download provenance, license/attribution and content hashes. Original FLEURS/LibriSpeech recordings and transformed clips inherit source obligations; publicity alone is not a license. The Nov14 Mandarin fixture license must be independently recorded from its actual source, not inherited from Nov10 corpora.
- Compact numeric extracts remove operational paths, addresses, process IDs, credentials and full transcripts. This does not authorize wholesale copying staged reports/logs/scripts. The archive keeps large raw profiles, arrays, audio and model binaries locally; SHA manifests document integrity and retained location through root labels.

## 8. Verification performed by this audit

The stdlib extractor reads original evidence without importing its code or writing its root. It independently reconstructs four primary raw/normalized micro edit-count totals; 860/874/881 measured/sample/all-record denominators; every retained latency-cell count, median and empirical higher p95; and the original per-call telemetry exclusion policy. It retains sample-level allowlisted timing/token/error metrics and original JSONL line references. The archived CIs are copied unchanged from `NOV10/results/summary.json`, with their original resampling unit/seed/method documented; the extractor does not claim to recompute them.

Current web checks used `smart-search fetch URL --format json --output FILE` on exact primary URLs; all nine fetches returned nonempty content and `ok=true`. Initial official/pinned-source fetches used the working fetch provider. Later Tavily fetch attempts encountered HTTP525; the CLI used its same-capability Firecrawl fallback successfully. Provider status was inspected; no provider configuration, credential or shared environment changed. Parsed content was read before claims were retained. Unread search suggestions are not citations. Current access date is Nov15; original source manifests preserve Nov10/Nov11 retrieval dates independently.

The following provenance tables are generated from the actual retained bytes and current extracted content, not hand-transcribed hashes.

### Nov10 exact retained source hashes

| Root label | Relative source file | Bytes | SHA256 |
| --- | --- | ---: | --- |
| NOV10 | `results/raw.jsonl` | 2458424 | `1dcc41240e0b99ccdd54af808f8508e76004c69d3a897cb62ad9e127b96eca02` |
| NOV10 | `results/telemetry.jsonl` | 483567 | `03e5424b1567bd56e5b8aa58cc201027c83561584b491d2917f9b5dd0207d693` |
| NOV10 | `results/summary.json` | 108553 | `f60b687b4651de2e94b24f3dc19092cb396586f662d001d1f00ace9f611d204c` |
| NOV10 | `results/audit.json` | 8161 | `7ec4f0890d56228930711301da4f1758b99799515cf7852012f6473b15ce1207` |
| NOV10 | `results/latency-items.csv` | 15097 | `ee12ef821b265ec311b74659a5043b1400c77e45aa0c21c098ea7c23e45fc042` |
| NOV10 | `results/paired-quality-items.csv` | 68863 | `90b23a1b06502cf1ee727aa040e264e814d326eea349d5684e83a402079b2a5e` |
| NOV10 | `results/environment.json` | 32819 | `693e4434b51ccb0407e635b46d3d9045b2eeae3f13232021b19ee146bf0f66d9` |
| NOV10 | `results/model-validation.json` | 1723 | `03004d49f7419f52d6d47adad6ea2b7a04e5494e3248c763fd24c9291d004974` |
| NOV10 | `results/deployed-payload-validation.json` | 329 | `48bbff092277f8f0eaa5f52ef1aa6737f4a8e2596920f0c697e8f38874ccab79` |
| NOV10 | `manifest.json` | 348543 | `4f1626131529f976afcea8e6701f90eae942dbdae0eaa448a321a230640b4185` |
| NOV10 | `probes.json` | 191105 | `02957c5d23a2113096c4589a5efb02582f390918da2a2964467c7e114afd8809` |
| NOV10 | `schedule.json` | 2029855 | `0a70bda66961de19e53d92c24f5e15a16b2dc230e281cd0b553a1c8f411a0bee` |
| NOV10 | `sources/dataset-downloads.json` | 906 | `9c5040dafa9b67949db0e9ebd9a6ad5b6d5eb3cbdc178ffc1fccadc4042f30bf` |
| NOV10 | `sources/tensor-hashes.json` | 91315 | `b58e860f0a2ac04240c87c6c4d4afef3f146ff7e2de083e9ca984f938060b5da` |
| NOV10 | `analyze.py` | 13394 | `1f82af83df07bdc305d97a5b4ba53ac031168ad3c9d486beb8d0e230ed0b16ea` |
| NOV10 | `finalize_evidence.py` | 6602 | `8c3fc286222b028bf42605cff5db0b2e33b78d0080455e994fe54109be67fa7b` |
| NOV10 | `VERSIONS.txt` | 239 | `7fb04cfdde7f8dd86ec1019781aa540eca635521648435ff7a580b6863beaab3` |

### Pinned historical primary-source snapshots

Original pinned-source manifest: `RX-SOURCE/evidence/fetch-manifest.json`, retrieval 2026-11-10; additional manifest retrieval 2026-11-11. These hashes identify saved original bytes, whereas current fetch content below is reader-extracted Markdown.

| Relative snapshot | Source URL | Bytes | SHA256 |
| --- | --- | ---: | --- |
| `RX-SOURCE/evidence/schaka/README.md` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/README.md) | 19821 | `60c9f13bbc497cc2a0e643625923f0a1eaaabb261ddfa45953c49ae690a5740e` |
| `RX-SOURCE/evidence/schaka/llama-cpp-gfx803/POSSIBLE_LLAMA_CPP_IMPROVEMENTS.md` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/llama-cpp-gfx803/POSSIBLE_LLAMA_CPP_IMPROVEMENTS.md) | 9615 | `14e39bdf1b4b29b67b59e5fb2d3c461d77e8fd0df155484b73e2a261630b3216` |
| `RX-SOURCE/evidence/schaka/llama-cpp-gfx803/README.md` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/llama-cpp-gfx803/README.md) | 908 | `f22db37777545f84273aba693b5ebddbc11e53bfb3f8055929577e173bd0c91b` |
| `RX-SOURCE/evidence/schaka/llama-cpp-gfx803/gfx803-packed-dp4a.patch` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/llama-cpp-gfx803/gfx803-packed-dp4a.patch) | 1638 | `8bfa762c7f4fb27d394a54b9f9893db25ba1ee55af0a3477be0486ea507443b8` |
| `RX-SOURCE/evidence/schaka/vllm/NOTES.md` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/vllm/NOTES.md) | 53335 | `436a325c0bdf152c750645f2520e0de75994afc75d40494d0d9f1cc411842eb1` |
| `RX-SOURCE/evidence/schaka/vllm/BUILD.md` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/vllm/BUILD.md) | 9018 | `a3face136f6fbefa6570d5343b419708c3e63e3296754d5cfa3e4143bffccad4` |
| `RX-SOURCE/evidence/schaka/scripts/build/ort.sh` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/scripts/build/ort.sh) | 2365 | `4e028c8ae402899915b1609a8506c0d94dd5847a1ba03038e26b9b41a8c1cb0b` |
| `RX-SOURCE/evidence/schaka/scripts/build/migraphx.sh` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/scripts/build/migraphx.sh) | 3085 | `38454eaa0588b4a5d019c00666540834eef35f54e46887bc59d4e5406c805841` |
| `RX-SOURCE/evidence/schaka/docker/ort.Dockerfile` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/docker/ort.Dockerfile) | 3176 | `dc5183b2b26ede3b343f7ddc42248023fb52a9dfa72b7cd2513a1f1227292834` |
| `RX-SOURCE/evidence/schaka/docker-bake.hcl` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/docker-bake.hcl) | 24720 | `1ef3a59e41aa5b82a7f464137bea5e965b934d360ed382d6e733c925d64bb4a3` |
| `RX-SOURCE/evidence/schaka/patches/migraphx/gfx-default-rocblas-hipblaslt-off-build-failure.patch` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/migraphx/gfx-default-rocblas-hipblaslt-off-build-failure.patch) | 4219 | `cd4e5107724dcbf78ba91487045d0d4985d3231fb64e588615ff940d1f53345b` |
| `RX-SOURCE/evidence/schaka/patches/rocblas/TENSILE_GFX803_FP16_TODO.md` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/rocblas/TENSILE_GFX803_FP16_TODO.md) | 30155 | `f8c6469e93510803aac200b4b77a41fe5b727a567d549ed28419c5a1d5427e5b` |
| `RX-SOURCE/evidence/schaka/patches/miopen/winograd-fused-conv-miscompute.patch` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/patches/miopen/winograd-fused-conv-miscompute.patch) | 9334 | `639ba4c0957fbca8d9a61cd55fa66ca7989aea9dd14de8c01efcccf51457c65e` |
| `RX-SOURCE/evidence/ort-migraphx.html` | [primary source](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) | 59357 | `9f69e766e6b99469849a9539bf87adb6f4b9212d07340ff5de9cc7475b90d02f` |
| `RX-SOURCE/evidence/ort-rocm.html` | [primary source](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) | 57531 | `faf891edc1358441c15572ee9b11f88e9386471523ec7ddec2519f7bfd12bcaf` |
| `RX-SOURCE/evidence/qwen3-asr-model-info.json` | [primary source](https://huggingface.co/api/models/Qwen/Qwen3-ASR-1.7B) | 5234 | `fdb5878e5ddb528e013b4d5e3abbe1594fd4a2876db052137ff17c1d66afa188` |
| `RX-SOURCE/evidence/qwen3-asr-config.json` | [primary source](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/resolve/7278e1e70fe206f11671096ffdd38061171dd6e5/config.json) | 6194 | `2e74a751548b8ad7d7526d29365ad8144c345d8b412b1152d25dc6698452712f` |
| `RX-SOURCE/evidence/schaka-commit.json` | [primary source](https://api.github.com/repos/Schaka/rocm-gfx803/commits/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78) | 11570 | `f5075422d4cd2f80ded813cff84a7d4ede50a86cd0e5f6ddddac8125c496d0e2` |
| `RX-SOURCE/evidence/schaka-tree.json` | [primary source](https://api.github.com/repos/Schaka/rocm-gfx803/git/trees/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78?recursive=1) | 3850314 | `c6e92a5c0b17d74947e3043296d139febcc96891987d1aea00294486466c08ad` |

### Nov15 primary-source access verification

| Extract label | Exact primary URL | Reader content bytes | Reader content SHA256 |
| --- | --- | ---: | --- |
| `mradermacher-card` | [primary source](https://huggingface.co/mradermacher/Qwen3-ASR-1.7B-GGUF/raw/664b960e358348943aefcdc26b3cf09eff1fb0ac/README.md) | 3994 | `2c53b858a5dd93a2a98b64cdda52502e705c05b1f5dfd96294aac6c066f79f9e` |
| `ort-migraphx` | [primary source](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html) | 14645 | `acf41d681e81e8a7d634394390a0840f30bd135a17dd0615c26534332d02e0fc` |
| `ort-rocm` | [primary source](https://onnxruntime.ai/docs/execution-providers/ROCm-ExecutionProvider.html) | 10458 | `83ed753f66efc0ebb6abdbae34ffd9b221a6518f42a3a3d32820493597ff8294` |
| `qwen-card` | [primary source](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/7278e1e70fe206f11671096ffdd38061171dd6e5/README.md) | 57456 | `5058416891bc47a2051557765997e8c42f8eb78a0e33c3e775bd17d4b0ba4d50` |
| `qwen-config` | [primary source](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/resolve/7278e1e70fe206f11671096ffdd38061171dd6e5/config.json) | 4771 | `51fc92ca6b2255d371b6d84fa0b08793db9f748b3411c902464a8a2aaa07bb8b` |
| `schaka-llama` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/llama-cpp-gfx803/POSSIBLE_LLAMA_CPP_IMPROVEMENTS.md) | 9619 | `f221a29ec82a424d02cad26193af6bc82202c0f60017d15674ebeec81a8115bb` |
| `schaka-readme` | [primary source](https://raw.githubusercontent.com/Schaka/rocm-gfx803/8fdcf7b1305d0cfc6fc8c1fc7b1bfb5f73494e78/README.md) | 19539 | `cf8fe73891dee3ecad9f834bc6e81f2d861c253180e1fe770240c01daaf3ad57` |
| `transcribe-cmake` | [primary source](https://raw.githubusercontent.com/handy-computer/transcribe.cpp/5700f127e38dd53c4735ea23ef743b4b538eacd4/CMakeLists.txt) | 24101 | `5f384f0e9967eae341d9c15484880d46acab580b70af933a903aaa5698fb7821` |
| `transcribe-qwen` | [primary source](https://raw.githubusercontent.com/handy-computer/transcribe.cpp/5700f127e38dd53c4735ea23ef743b4b538eacd4/docs/models/qwen3-asr-1.7b.md) | 7714 | `15b17e9e0d4866dd7b124578dbc4db0f583f96a24b7b2d00828bba0ef1b483d3` |

### Public artifact integrity

The final [SHA256SUMS](SHA256SUMS) fingerprints the publication files after compact extraction, metadata normalization and rendering. The [historical observations](data/historical.json), [extractor](extract_historical.py), [checked source manifest](data/checked-sources.json) and [source/context observations](data/source-context.json) are the public artifacts. Their private raw input hashes remain unchanged in the source tables above. Formatting and explicitly audited fixture metadata additions to a public extract do not alter the original raw evidence.
