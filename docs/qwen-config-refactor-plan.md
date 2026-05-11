# Qwen ASR Config / Runtime Refactor Plan

## Status

- Scope: `qwen_asr` only
- Project context: this repository is a server-focused fork built on top of upstream CapsWriter-Offline
- Decision status: **approved for implementation**
- Approval summary:
  - Official Qwen preset surface may be reduced to two presets: `default` and `cpu_only`
  - `default` may be redefined as the recommended P4/Pascal path: **ONNX on GPU + llama on CPU**
  - `CAPSWRITER_QWEN_USE_CUDA` and `CAPSWRITER_QWEN_VULKAN_ENABLE` may remain supported, but should be treated as advanced / internal overrides rather than the primary user-facing config surface

---

## 1. Goals

This refactor has four primary goals:

1. Make `qwen_asr` runtime behavior **consistent with configuration values**.
2. Make preset selection the **primary backend profile selector**.
3. Make configuration resolution **explicit, deterministic, documented, and reviewable**.
4. Reduce user confusion by exposing a **small official config surface** while still allowing advanced overrides when necessary.

The target user experience is:

- choose one official Qwen preset
- optionally tune a small number of documented Qwen parameters
- understand exactly which values came from preset defaults, which came from explicit overrides, and which were changed by runtime fallback

---

## 2. Confirmed product decisions

The following design decisions are approved and should be treated as requirements during implementation.

### 2.1 Official Qwen preset surface

Only two official presets should remain in the public Qwen configuration surface:

- `default`
- `cpu_only`

### 2.2 Official meaning of `default`

`default` should represent the best currently validated configuration for this fork's Qwen server path on the current target hardware class:

- **ONNX encoder on GPU**
- **llama / GGUF decoder on CPU**

This is the current recommended cost/performance profile for Tesla P4 / Pascal based on the repository benchmark work completed before this plan.

### 2.3 Advanced overrides remain supported

The following variables may remain supported, but they should no longer be presented as the primary configuration surface for normal users:

- `CAPSWRITER_QWEN_USE_CUDA`
- `CAPSWRITER_QWEN_VULKAN_ENABLE`
- `CAPSWRITER_QWEN_USE_DML`
- `CAPSWRITER_QWEN_VULKAN_FORCE_FP32`

These should be documented as **advanced/internal overrides**.

---

## 3. Current problems to fix

The current repository behavior has several inconsistencies.

### 3.1 Preset normalization is not unified

At the time of writing:

- `config_server.py` recognizes `default` and `low_vram_gpu` as the active preset values it uses for defaults
- the shell entrypoint still uses older naming assumptions such as `balanced`
- real-world user config may contain values like `defaults`
- docs, compose defaults, and runtime mutation are not all using the same normalization rules

Result: the same preset string can be interpreted differently depending on where it is read.

### 3.2 Runtime backend selection is split across shell and Python

Current behavior is distributed across:

- `.env`
- `docker-compose.yml`
- `docker/server/entrypoint.sh`
- `config_server.py`
- `docker/server/probe_backend.py`

Result: the final runtime path is not derived from a single authoritative resolver.

### 3.3 Backend profile selection is not fully encoded in preset defaults

Current Qwen preset defaults mainly cover tuning values such as:

- `n_ctx`
- `chunk_size`
- `memory_num`
- `pad_to`
- `n_predict`
- `llama_n_batch`
- `llama_n_ubatch`
- `llama_flash_attn`
- `llama_offload_kqv`

But actual backend routing decisions are still influenced elsewhere by runtime shell mutations such as:

- `CAPSWRITER_QWEN_USE_CUDA`
- `CAPSWRITER_QWEN_VULKAN_ENABLE`
- `CAPSWRITER_LLAMA_BACKEND`

Result: preset names feel like full profiles, but only actually control part of the behavior.

### 3.4 Public docs still expose too many low-level knobs too early

The current docs and examples surface too many expert-level variables before clearly defining:

- which knobs are official
- which knobs are advanced
- which knobs override preset behavior
- what the exact precedence rules are

### 3.5 Current default runtime path is not aligned with the best validated Qwen profile

The benchmark work on this fork indicates that the most cost-effective Qwen configuration on the current Pascal/P4 target is:

- ONNX GPU + llama CPU

But current runtime behavior can still resolve to:

- ONNX GPU + llama GPU/Vulkan

Result: the practical default behavior is not aligned with the best validated recommendation.

### 3.6 Config precedence is not explicitly documented

There is currently no single authoritative document that precisely explains:

- preset normalization
- preset defaults
- explicit env overrides
- hardware policy constraints
- runtime fallback behavior

Result: users cannot reliably predict final runtime behavior from config files alone.

---

## 4. Target configuration model

The refactor should move `qwen_asr` to the following model.

### 4.1 Official preset names

Public / documented preset names:

- `default`
- `cpu_only`

### 4.2 Preset aliases for backward compatibility

Legacy and compatibility values should be normalized into the official values.

Recommended alias mapping:

- `low_vram_gpu` -> `default`
- `balanced` -> `default`
- `quality` -> `default`
- `defaults` -> `default`
- `cpu` -> `cpu_only` (optional compatibility alias if helpful)

Only `default` and `cpu_only` should be shown as official values in current user-facing docs.

### 4.3 Meaning of each official preset

#### `default`

Intended resolved behavior:

- ONNX encoder: GPU when available and allowed
- llama / GGUF decoder: CPU
- if GPU is unavailable or disallowed, fallback to CPU-only resolved runtime

#### `cpu_only`

Intended resolved behavior:

- ONNX encoder: CPU
- llama / GGUF decoder: CPU
- runtime should not attempt to prefer Vulkan or ONNX CUDA for Qwen

---

## 5. Target precedence rules

This precedence model should become the authoritative behavior for Qwen config resolution.

### 5.1 Resolution order

1. Start from hardcoded baseline defaults.
2. Normalize the preset name.
3. Apply preset defaults.
4. Apply explicit Qwen env overrides.
5. Apply hardware-policy constraints from `CAPSWRITER_INFERENCE_HARDWARE`.
6. Apply runtime fallback if requested hardware is unavailable.
7. Emit a fully resolved runtime config.

### 5.2 Meaning of each layer

#### Layer 1: hardcoded baseline

This is the repository's last-resort default when nothing else is specified.

#### Layer 2: preset normalization

Normalize aliases such as `low_vram_gpu`, `balanced`, `quality`, and `defaults` into official values.

#### Layer 3: preset defaults

Apply the full preset profile, including both:

- backend profile defaults
- tuning defaults

This is the layer that turns `default` into a complete semantic choice rather than just a handful of tuning fields.

#### Layer 4: explicit env overrides

If the user explicitly sets a supported env key, that value should override the preset default.

Important rule:

- empty string should be treated as **unset**, not as an active override

#### Layer 5: hardware policy constraints

`CAPSWRITER_INFERENCE_HARDWARE` should act as a policy constraint.

- `cpu` forces the final runtime onto CPU
- `auto` allows GPU when available and appropriate for the resolved profile
- `gpu` requests GPU first, but does not require a hard startup failure if runtime support is absent

#### Layer 6: runtime fallback

If the resolved config requests GPU-backed behavior but the required runtime is not available, the system should apply a deterministic fallback and log exactly what changed.

---

## 6. Target field taxonomy

The final docs and examples should group fields into three clear layers.

### 6.1 Official top-level surface

These are the first-line settings most users should see.

- `CAPSWRITER_MODEL_TYPE`
- `CAPSWRITER_QWEN_PRESET`
- `CAPSWRITER_INFERENCE_HARDWARE`
- `CAPSWRITER_GPU_DEVICE_COUNT`
- `CAPSWRITER_NUM_THREADS`
- `CAPSWRITER_SERVER_PORT`
- `CAPSWRITER_LOG_LEVEL`
- `CAPSWRITER_REMOVE_MODEL_ARCHIVES`

### 6.2 Qwen tuning overrides

These are legitimate, documented performance / behavior tuning keys which should override preset defaults if explicitly set.

- `CAPSWRITER_QWEN_CHUNK_SIZE`
- `CAPSWRITER_QWEN_N_CTX`
- `CAPSWRITER_QWEN_MEMORY_NUM`
- `CAPSWRITER_QWEN_PAD_TO`
- `CAPSWRITER_QWEN_N_PREDICT`
- `CAPSWRITER_QWEN_LLAMA_N_BATCH`
- `CAPSWRITER_QWEN_LLAMA_N_UBATCH`
- `CAPSWRITER_QWEN_LLAMA_FLASH_ATTN`
- `CAPSWRITER_QWEN_LLAMA_OFFLOAD_KQV`

### 6.3 Advanced / internal backend overrides

These should remain supported only for advanced control and debugging.

- `CAPSWRITER_QWEN_USE_CUDA`
- `CAPSWRITER_QWEN_VULKAN_ENABLE`
- `CAPSWRITER_QWEN_USE_DML`
- `CAPSWRITER_QWEN_VULKAN_FORCE_FP32`

These should be clearly documented as non-primary overrides.

---

## 7. Required code changes

The refactor should touch the following files.

### 7.1 `config_server.py`

This should become the authoritative Qwen config resolution layer, or delegate to a new Python resolver module that it owns.

Required work:

- unify Qwen preset normalization
- define official preset aliases
- define full preset defaults, including backend profile defaults
- make explicit env override behavior deterministic
- make empty-string env values behave as unset
- expose a resolved Qwen config object that can be reused by runtime, probes, and docs-oriented inspection helpers
- attach precise inline comments for each setting

### 7.2 `docker/server/entrypoint.sh`

This file should stop owning a second, partially divergent copy of Qwen preset logic.

Required work:

- reduce shell-side decision logic
- keep only runtime capability detection and process bootstrap concerns
- delegate Qwen backend resolution to Python
- export only the resolved values needed by the runtime
- print clear logs for final resolved path

### 7.3 `docker/server/probe_backend.py`

Required work:

- verify the resolved backend profile rather than only checking one narrow env-driven path
- align probe expectations with the official preset semantics
- clearly distinguish verification of ONNX CUDA from llama backend expectations

### 7.4 `docker-compose.yml`

Required work:

- align comments and defaults with the new official preset model
- stop implying legacy preset names are part of the current primary surface
- clearly separate official settings from expert-level overrides

### 7.5 `.env.example`

Required work:

- rewrite as a precise deployment template for the new configuration model
- add exact inline documentation for every exposed field
- document allowed values, meaning, default, and precedence behavior
- remove misleading or contradictory examples
- ensure all examples use official preset names only

### 7.6 `docs/docker-server.md`

Required work:

- add a dedicated Qwen configuration resolution section
- document official presets and alias behavior
- document precedence order explicitly
- document fallback behavior explicitly
- document which settings are primary vs advanced

### 7.7 `readme.md`

Required work:

- keep the front page concise
- align wording with the new official preset semantics
- ensure `default` is described consistently with the validated recommendation

### 7.8 `04-Benchmark.py` (recommended)

Recommended, though not strictly required in the first pass:

- allow explicit preset-oriented benchmarking
- print normalized preset and resolved backend profile in benchmark output

---

## 8. Recommended implementation architecture

### 8.1 Single-source-of-truth resolver

The most important architecture rule is:

> Qwen runtime resolution should be defined in exactly one authoritative Python implementation.

Possible implementation options:

- extend `config_server.py` to be the authoritative resolver
- or add a dedicated helper such as `util/server/qwen_runtime_profile.py`

Either option is acceptable, but only one should own the logic.

### 8.2 Shell should not own policy

`entrypoint.sh` should not continue to maintain an independent Qwen preset policy.

Shell should only:

- detect runtime capability (`gpu_visible`, `nvidia_visible`, etc.)
- invoke the resolver
- export or consume resolved values
- continue bootstrap / model preparation / probe / server startup

### 8.3 Resolved config should be observable

The system should expose a clearly inspectable resolved config representation.

At minimum, startup logs should show:

- raw preset input
- normalized preset
- resolved ONNX backend
- resolved llama backend
- any fallback that occurred
- source of important values where practical

---

## 9. Documentation requirements for each config key

Every documented Qwen-related config key should eventually have precise notes covering:

1. Purpose
2. Allowed values or type
3. Default behavior
4. Whether it belongs to preset defaults
5. Whether explicit env values override the preset
6. Whether runtime hardware policy can still constrain it

Recommended comment format inside `.env.example`:

```env
# CAPSWRITER_QWEN_PRESET
# Purpose: Selects the official Qwen runtime profile.
# Allowed: default | cpu_only
# Default: default
# Resolution: aliases normalize first, then preset defaults apply, then explicit overrides apply.
# Override priority: preset -> explicit Qwen env overrides -> hardware-policy constraint -> runtime fallback.
CAPSWRITER_QWEN_PRESET=default
```

This style should be applied consistently across the full example template.

---

## 10. Verification plan

Implementation should not be considered complete until the following verification layers pass.

### 10.1 Resolution / normalization checks

Expected normalization examples:

- `default` -> `default`
- `low_vram_gpu` -> `default`
- `balanced` -> `default`
- `quality` -> `default`
- `defaults` -> `default`
- `cpu_only` -> `cpu_only`

### 10.2 Backend resolution checks

Expected examples:

#### Case A

- preset: `default`
- hardware policy: `auto`
- GPU visible: yes

Expected resolved behavior:

- ONNX GPU
- llama CPU

#### Case B

- preset: `default`
- hardware policy: `cpu`

Expected resolved behavior:

- ONNX CPU
- llama CPU

#### Case C

- preset: `cpu_only`
- hardware policy: `auto`
- GPU visible: yes

Expected resolved behavior:

- ONNX CPU
- llama CPU

#### Case D

- preset: `default`
- hardware policy: `auto`
- GPU visible: no

Expected resolved behavior:

- ONNX CPU
- llama CPU
- explicit fallback log present

### 10.3 Override precedence checks

Example:

- preset default gives `n_ctx=2048`
- explicit env sets `CAPSWRITER_QWEN_N_CTX=1536`

Expected resolved result:

- final `n_ctx=1536`

### 10.4 Startup log checks

The final startup path should log enough information to validate behavior quickly. Example desired output shape:

```text
resolved_qwen_preset=default
resolved_qwen_profile=onnx_gpu_llama_cpu
resolved_qwen_onnx_backend=cuda
resolved_qwen_llama_backend=cpu
resolved_qwen_n_ctx=1536 source=env
resolved_qwen_chunk_size=30 source=env
resolved_qwen_llama_flash_attn=true source=preset
```

### 10.5 Benchmark regression checks

After refactor, benchmark should confirm that:

- `default` resolves to ONNX GPU + llama CPU on compatible hardware
- `cpu_only` resolves to pure CPU
- the validated performance relationship remains intact

---

## 11. Review checklist

Before merging the implementation, review should confirm:

- docs and runtime use the same official preset names
- `defaults` is no longer presented as an official value
- shell does not maintain a second, divergent preset policy
- explicit env overrides are applied deterministically
- empty-string env values do not accidentally count as real overrides
- startup logs explain the final resolved path
- `.env.example`, `docker-compose.yml`, `docs/docker-server.md`, and `readme.md` all describe the same precedence model

---

## 12. Rollout sequence

Recommended execution order:

### Phase 1: resolver design and implementation

- implement preset normalization
- implement resolved config model
- implement precedence logic

### Phase 2: runtime wiring

- integrate resolver with `config_server.py`
- slim down `entrypoint.sh`
- align `probe_backend.py`

### Phase 3: configuration surface cleanup

- update `docker-compose.yml`
- rewrite `.env.example`
- align Qwen-related comments and defaults

### Phase 4: documentation alignment

- update `docs/docker-server.md`
- update `readme.md`

### Phase 5: review and verification

- resolution checks
- startup verification
- benchmark regression
- docs consistency review

---

## 13. Non-goals for this refactor

To keep the work focused, the following are out of scope unless separately approved:

- redesigning `fun_asr_nano`
- large-scale upstream parity work unrelated to `qwen_asr`
- implementing true llama layer-based partial offload (`n_gpu_layers`) in this same change set
- reworking every benchmark utility beyond what is needed to validate the new Qwen profile behavior

---

## 14. Expected end state

When this plan is fully implemented, the repository should provide:

- a Qwen server configuration model with **two official presets only**
- a default Qwen path aligned with the validated best recommendation for current target hardware
- a single authoritative configuration resolver
- deterministic and inspectable precedence behavior
- precise inline documentation for every user-facing Qwen config key
- docs, compose, and runtime behavior that all agree with one another

This will make the fork easier to operate, easier to reason about, and safer to modify in future iterations.