# KVarN Implementation Backlog for llama.cpp

Source paper: KVarN (arXiv:2606.03458v1)
Research docs: `.ai/research/kvarn/`
Related llama.cpp docs: `.ai/docs/07-kv-cache.md`, `.ai/docs/05-quantization.md`,
`.ai/docs/06-inference-engine.md`, `.ai/docs/01-ggml-tensor-library.md`,
`.ai/docs/02-ggml-backend.md`, `.ai/docs/18-c-api.md`

## Spike results

Spike 01 (`spikes/01_kvarn_core/`) confirmed the core algorithm is viable:
- VarN converges (Imb 312 -> 12 after 8 iterations, -> 7 after 16)
- KVarN beats Hadamard+Q2 on 3/3 test matrices (1.2x-8.7x on top-5% E_M/E_T)
- KVarN beats plain Q2 RTN on heavy-tailed data (4.9x), equal on Gaussian
- Hadamard rotation alone HURTS at 2-bit; VarN is essential
- FP16 scales: 2.375 bpe (vs 2.25 with FP8); acceptable for initial impl
- Dequant formula confirmed: `K_dq = (K_q - zpt) * s1 * s2`
- Default VarN iterations should be 12 (not 8) for safety margin
- See `.ai/self-study/2026-06-25_01_kvarn_core.md` for full analysis

## Context

KVarN is a calibration-free 2-bit KV-cache quantizer combining:
1. Channel-dimension Hadamard rotation (already partially in llama.cpp)
2. Dual-axis variance normalization (Sinkhorn-style, 8 iterations)
3. Round-to-nearest 2-bit quantization with dual-scale + zeropoint storage

Current llama.cpp KV-cache quantization (per `.ai/docs/07-kv-cache.md`):
- Supports `type_k`/`type_v` as any `ggml_type` (`include/llama.h:365-366`)
- Already applies Walsh-Hadamard rotation when K/V is quantized
  (`src/llama-kv-cache.cpp:332-351`, applied via `ggml_mul_mat_aux` in
  `src/llama-graph.cpp:2327-2332`)
- K-shift graph dequantizes -> rotates -> RoPE -> re-quantizes
  (`src/llama-kv-cache.cpp:1905-1951`)
- No variance normalization, no dual-scale, no three-region layout, no 2-bit
  type suitable for KVarN's storage format

## What needs to be built

The implementation is decomposed into independent items ordered by dependency.
Each item has a falsifiable exit criterion and cites the specific files and
paper sections it addresses.

---

### 01 - KVarN quantization type (`GGML_TYPE_Q2_KVARN`)

- **Status**: done
- **Depends on**: nothing
- **Pipeline**: implementation
- **Exit criterion**: A new `ggml_type` enum value `GGML_TYPE_Q2_KVARN` exists
  in `ggml/include/ggml.h` with:
  - Block struct `block_q2_kvarn` in `ggml/src/ggml-common.h` containing:
    - `qs[32]` (uint8, 2-bit packed: 128 elements in 32 bytes)
    - `d` (ggml_half, FP16 zeropoint, 1 per channel per group of 128)
    - `s1` (ggml_half, FP8-as-F16 primary scale, 1 per channel per group)
    - `s2` (ggml_half, FP8-as-F16 secondary scale, 1 per token per group)
  - `ggml_type_traits` entry with `blck_size=128`, `type_size=sizeof(block_q2_kvarn)`
  - `quantize_q2_kvarn` and `dequantize_q2_kvarn` functions in `ggml-quants.c`
  - Block size = 128 elements (matching KVarN group size G=128)
  - Effective bits/element: 2.25 (2.0 data + 0.125 zeropoint + 0.0625 s1 + 0.0625 s2)
  - Verify: `ggml_quantize_chunk` dispatches to `quantize_q2_kvarn` and
    `ggml_type_traits.to_float` dispatches to `dequantize_q2_kvarn`.
- **Paper ref**: `.ai/research/kvarn/05-kvarn-pipeline.md` (storage format)
- **llama.cpp ref**: `ggml/include/ggml.h:389-433` (enum),
  `ggml/src/ggml-common.h:280-450` (block structs),
  `ggml/src/ggml-quants.c` (kernels), `ggml/src/ggml.c:7706-7780` (dispatch)

### 02 - Dequantization kernel for Q2_KVARN

- **Status**: done
- **Depends on**: 01
- **Pipeline**: implementation
- **Exit criterion**: `dequantize_row_q2_kvarn` exists in `ggml-quants.c` and:
  - Reads `block_q2_kvarn`, unpacks 2-bit values, applies
    `K_dq = (K_q + d) * s1 * s2` (dual-scale fused)
  - SIMD implementation (AVX2/AVX512/NEON) for CPU
  - Registered in `ggml_type_traits[GGML_TYPE_Q2_KVARN].to_float`
  - Verify: dequantize a known-quantized block, compare to expected FP32 output
    within 1e-3 tolerance.
- **Paper ref**: `.ai/research/kvarn/07-dequantization.md`
- **llama.cpp ref**: `ggml/src/ggml-quants.c` (existing dequant kernels),
  `ggml/include/ggml.h:2817-2825` (type_traits)

### 03 - Quantization kernel for Q2_KVARN (RTN + dual-scale)

- **Status**: done
- **Depends on**: 01
- **Pipeline**: implementation
- **Exit criterion**: `quantize_q2_kvarn` exists in `ggml-quants.c` and:
  - Takes FP32 input, computes per-channel zeropoint (FP16), per-channel
    scale s1 (absorbs VarN column scale), per-token scale s2 (VarN row scale)
  - Round-to-nearest with asymmetric quantization (zeropoint offset)
  - Stores packed 2-bit values + scales + zeropoint in `block_q2_kvarn`
  - Registered in `ggml_type_traits[GGML_TYPE_Q2_KVARN].from_float_ref`
  - `ggml_quantize_chunk` dispatches to it
  - Verify: quantize then dequantize a test vector, check MSE is finite
    and that the per-token norm error `E_M/ E_T < 0.5` for the median token.
- **Paper ref**: `.ai/research/kvarn/05-kvarn-pipeline.md` (RTN step),
  `.ai/research/kvarn/04-variance-normalization.md` (s1/s2 derivation)
- **llama.cpp ref**: `ggml/src/ggml-quants.c` (existing quant kernels),
  `ggml/src/ggml.c:7706-7780` (dispatch)

### 04 - Variance normalization function (VarN)

- **Status**: done
- **Depends on**: nothing
- **Pipeline**: implementation
- **Exit criterion**: A function `kvarn_variance_normalize` exists in a new
  `src/llama-kvarn.h` / `src/llama-kvarn.cpp` (or in ggml) that:
  - Takes a tile `T` of shape `[R, C]` (e.g. `[128, 128]`), iteration count K=12,
    clamp limits `c_min`, `c_max`
  - Implements Algorithm 1 (Appendix H of the paper): log-domain dual-scale
    Sinkhorn-style balancing with `Imb()` metric and best-state tracking
  - Returns the normalized tile and two scale vectors `S_c` (1xC) and
    `S_r` (Rx1)
  - CPU implementation using SIMD or at minimum efficient scalar
  - Verify: given a tile with known row/column variance imbalance, the
    output has row variances and column variances both within 5% of each
    other after 8 iterations.
- **Paper ref**: `.ai/research/kvarn/04-variance-normalization.md`
  (Algorithm 1 verbatim)
- **llama.cpp ref**: new file; the log-domain approach mirrors concepts from
  SINQ (not in llama.cpp; this is novel code)

### 05 - Integrate VarN into KV-cache quantization path

- **Status**: done
- **Depends on**: 03, 04
- **Pipeline**: implementation
- **Exit criterion**: When `type_k == GGML_TYPE_Q2_KVARN` (or `type_v`), the
  KV-cache write path applies:
  1. Hadamard rotation (already done via `attn_rot_k`/`attn_rot_v`,
     `src/llama-kv-cache.cpp:332-351`)
  2. Variance normalization on the tile before RTN quantization
  3. RTN quantization producing `block_q2_kvarn` with absorbed VarN scales
  - The VarN step runs online when a full group of G=128 tokens is available
  - Verify: a context with `type_k=GGML_TYPE_Q2_KVARN` produces quantized
    K tensors whose dequantized values have lower per-token magnitude error
    than plain `GGML_TYPE_Q4_0` (measured via the E_M/E_T metric from
    `02-error-decomposition.md`).
- **Paper ref**: `.ai/research/kvarn/05-kvarn-pipeline.md` (full pipeline)
- **llama.cpp ref**: `src/llama-kv-cache.cpp:332-351` (rotation trigger),
  `src/llama-kv-cache.cpp:1905-1951` (K-shift graph with dequant/requant),
  `src/llama-graph.cpp:2327-2332` (rotation in graph builder)

### 06 - Three-region cache layout (sink/body/recent)

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation
- **Exit criterion**: `llama_kv_cache` supports a three-region layout when
  KVarN quantization is active:
  - First S=128 tokens stored as FP16 (sink region, unquantized)
  - Middle tokens stored as `GGML_TYPE_Q2_KVARN` in groups of G=128 (body)
  - Last R=128 tokens stored as FP16 (recent region, unquantized)
  - `llama_kv_cells` or a wrapper tracks which region each cell belongs to
  - Attention reads sink/recent at FP16, body at Q2_KVARN
  - Verify: with `type_k=GGML_TYPE_Q2_KVARN` and a 500-token context, the
    first 128 and last 128 tokens are FP16, the middle 244 tokens are
    quantized (2 groups of 128 minus the overlap), and attention output
    matches FP16 baseline within KL-divergence < 0.1.
- **Paper ref**: `.ai/research/kvarn/06-three-region-layout.md`
- **llama.cpp ref**: `src/llama-kv-cache.h:217-285` (cache internals),
  `src/llama-kv-cells.h:32` (cell metadata), `src/llama-kv-cache.cpp:245-246`
  (tensor allocation)

### 07 - K-shift graph support for Q2_KVARN

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation
- **Exit criterion**: `build_rope_shift` in `src/llama-kv-cache.cpp:1826-1876`
  handles `GGML_TYPE_Q2_KVARN`:
  - Dequantizes Q2_KVARN to F32 (using dual-scale formula)
  - Rotates backward (Hadamard, already implemented)
  - Applies RoPE shift
  - Rotates forward
  - Re-quantizes to Q2_KVARN (re-running VarN on the shifted tile)
  - Verify: calling `llama_memory_seq_add` on a Q2_KVARN cache produces
    correct shifted positions without corruption.
- **Paper ref**: indirect (K-shift is a llama.cpp mechanism; KVarN must
  preserve it)
- **llama.cpp ref**: `src/llama-kv-cache.cpp:1826-1876` (build_rope_shift),
  `src/llama-kv-cache.cpp:1905-1951` (build_graph_shift)

### 08 - Context params for KVarN configuration

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation
- **Exit criterion**: `llama_context_params` (`include/llama.h:347-395`)
  includes optional KVarN parameters:
  - `kvarn_group_size` (default 128)
  - `kvarn_sink_tokens` (default 128)
  - `kvarn_recent_tokens` (default 128)
  - `kvarn_varn_iterations` (default 8)
  - These are only read when `type_k` or `type_v` is `GGML_TYPE_Q2_KVARN`
  - `llama_context_default_params` sets sensible defaults
  - Verify: setting `type_k=GGML_TYPE_Q2_KVARN` in params and calling
    `llama_init_from_model` succeeds without error, and the cache uses
    the specified group/sink/recent sizes.
- **Paper ref**: `.ai/research/kvarn/06-three-region-layout.md` (defaults),
  `.ai/research/kvarn/09-experimental-setup.md` (G=128, S=128, R=128)
- **llama.cpp ref**: `include/llama.h:347-395` (context_params struct),
  `src/llama-context.cpp:49-243` (cparams init),
  `src/llama-context.cpp:3472-3473` (default type_k/type_v)

### 09 - GPU backend kernel for Q2_KVARN dequantization

- **Status**: done
- **Depends on**: 02
- **Pipeline**: implementation
- **Exit criterion**: At least one GPU backend (CUDA preferred) implements
  a fused dequantization kernel for `GGML_TYPE_Q2_KVARN`:
  - Custom CUDA kernel or Triton-style kernel that reads `block_q2_kvarn`
    and produces FP16/F32 output with the dual-scale formula applied
  - Integrated into the attention kernel (flash attention or mul_mat path)
    so dequant happens inline during Q*K^T without a separate dequant pass
  - `supports_op` returns true for `GGML_OP_MUL_MAT` when input type is
    `GGML_TYPE_Q2_KVARN`
  - Verify: `llama-bench` with `type_k=GGML_TYPE_Q2_KVARN` on a CUDA GPU
    shows <2% dequant overhead vs `GGML_TYPE_Q8_0`.
- **Paper ref**: `.ai/research/kvarn/07-dequantization.md` (kernel fusion,
  <1.4% overhead target)
- **llama.cpp ref**: `ggml/src/ggml-cuda/` (CUDA kernels),
  `ggml/src/ggml-cuda/fattn.cu` (flash attention),
  `ggml/src/ggml-cuda/mmq.cu` (mul_mat quantized),
  `.ai/docs/02-ggml-backend.md` (backend supports_op interface)

### 10 - Pseudo-decode evaluation harness

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation
- **Exit criterion**: A test or example exists at `tests/test-kvarn-pseudo-decode.cpp`
  (or `examples/kvarn-eval/`) that:
  - Takes a model file and a prompt
  - Processes the prompt in blocks of b=128 tokens
  - After each block, quantizes the KV-cache
  - Subsequent blocks compute with the quantized cache
  - Measures per-layer attention output reconstruction error vs FP16 baseline
  - Reports the error as a function of context length
  - Verify: running on Qwen3-4B with a 1000-token wikitext prompt produces
    a monotonically increasing error curve, and KVarN's curve is lower
    than KIVI (Q4_0) at every context length.
- **Paper ref**: `.ai/research/kvarn/08-pseudo-decode-eval.md`
- **llama.cpp ref**: `src/llama-context.cpp:1680` (decode loop),
  `src/llama-kv-cache.cpp:760-824` (prepare/init_batch)

### 11 - Error decomposition measurement tool

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation
- **Exit criterion**: A function or test exists that, given a quantized and
  full-precision K matrix, computes:
  - Per-token `E_M` (magnitude error), `E_D` (directional error), `E_T` (total)
  - The ratio `E_M / E_T` for the top-k% worst tokens
  - Outputs a histogram or summary table
  - Verify: on a KVarN-quantized cache, `E_M/E_T` for the top 5% errors is
    lower than for Q4_0-quantized cache (proving magnitude errors are
    suppressed).
- **Paper ref**: `.ai/research/kvarn/02-error-decomposition.md` (Eq 3)
- **llama.cpp ref**: new utility; uses `ggml_tensor` data access

### 12 - CLI flag for KVarN quantization

- **Status**: done
- **Depends on**: 08
- **Pipeline**: implementation
- **Exit criterion**: `llama-cli` and `llama-server` accept a flag or config
  to set `type_k=GGML_TYPE_Q2_KVARN` and `type_v=GGML_TYPE_Q2_KVARN`:
  - `--cache-type-k q2_kvarn` / `--cache-type-v q2_kvarn` in the arg parser
  - The `common_arg` system (`common/arg.cpp`) parses `q2_kvarn` to
    `GGML_TYPE_Q2_KVARN`
  - `llama-bench` supports `--cache-type-k q2_kvarn` for benchmarking
  - Verify: `llama-cli -m model.gguf --cache-type-k q2_kvarn -p "Hello"`
    runs without error and produces coherent output.
- **Paper ref**: N/A (integration)
- **llama.cpp ref**: `common/arg.cpp` (arg parser),
  `.ai/docs/14-cli-tools.md` (CLI architecture),
  `tools/main/main.cpp` (llama-cli),
  `tools/server/server-context.cpp` (server)

### 13 - State save/load for Q2_KVARN cache

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation
- **Exit criterion**: `llama_state_get_data` / `llama_state_set_data`
  (`src/llama-kv-cache.cpp:2000-2062`) correctly serialize and deserialize
  Q2_KVARN-quantized K/V tensors:
  - The raw quantized bytes are written/read as-is (same as other quant types)
  - The three-region layout metadata is serialized (which tokens are
    sink/body/recent)
  - Verify: save state with Q2_KVARN cache, load into a fresh context,
    and decode produces identical output to the pre-save context.
- **Paper ref**: N/A (persistence)
- **llama.cpp ref**: `src/llama-kv-cache.cpp:2000-2062` (state save/load),
  `include/llama.h:778-907` (state API)

### 14 - Integration test: KVarN vs FP16 on reasoning benchmark

- **Status**: done
- **Depends on**: 06, 09, 12
- **Pipeline**: testing
- **Exit criterion**: An integration test runs a small reasoning task
  (e.g. a few MATH-500 problems or a simple arithmetic chain) comparing:
  - FP16 baseline
  - Q4_0 K-cache (current best low-bit)
  - Q2_KVARN K-cache (KVarN)
  - Reports accuracy and token count for each
  - Verify: Q2_KVARN accuracy is within 5% of FP16 on at least one model,
    and >= Q4_0 accuracy. (Full parity with paper results requires the GPU
    kernel; this test can run on CPU first.)
- **Paper ref**: `.ai/research/kvarn/10-results-summary.md` (target metrics)
- **llama.cpp ref**: new test file; uses `llama_decode` and output comparison

---

## Dependency graph

```
01 (Q2_KVARN type) ──┬──> 02 (dequant kernel)
                     ├──> 03 (quant kernel) ──┐
                     │                        ├──> 05 (VarN integration) ──┬──> 06 (three-region layout)
04 (VarN function) ─┘                        │                             ├──> 07 (K-shift support)
                                              │                             ├──> 08 (context params)
                                              │                             ├──> 10 (pseudo-decode eval)
                                              │                             ├──> 11 (error decomposition)
                                              │                             ├──> 13 (state save/load)
                                              │                             └──> 14 (integration test)
02 ─────────────────────────────────────────────┴──> 09 (GPU kernel)
06 + 09 + 08 ──────────────────────────────────────────> 12 (CLI flag)
06 + 09 + 12 ───────────────────────────────────────────> 14 (integration test)
```

## Priority order

1. **01** (type definition) - foundation, no deps
2. **04** (VarN function) - independent, can parallelize with 01
3. **02** (dequant kernel) - needs 01
4. **03** (quant kernel) - needs 01
5. **05** (VarN integration) - needs 03 + 04, the core feature
6. **08** (context params) - needs 05, enables user-facing config
7. **07** (K-shift) - needs 05, correctness for long contexts
8. **06** (three-region layout) - needs 05, full paper fidelity
9. **10** (pseudo-decode eval) - needs 05, validation tooling
10. **11** (error decomposition) - needs 05, measurement tooling
11. **13** (state save/load) - needs 05, persistence
12. **09** (GPU kernel) - needs 02, performance
13. **12** (CLI flag) - needs 06 + 09 + 08, user-facing
14. **14** (integration test) - needs 06 + 09 + 12, end-to-end validation

## Notes

- Items 01 and 04 can be developed in parallel immediately
- Items 02 and 03 can be developed in parallel after 01
- The existing Hadamard rotation (`attn_rot_k`/`attn_rot_v`) in
  `src/llama-kv-cache.cpp:332-351` already covers KVarN's step 1
  (channel-dimension rotation). The rotation is applied via
  `ggml_mul_mat_aux` in `src/llama-graph.cpp:2327-2332`.
- The main novelty is the VarN step (Algorithm 1) and the dual-scale
  storage format. The Hadamard part is not new to llama.cpp.
- FP8 scales: the paper uses FP8 (E4M3). llama.cpp does not have native FP8
  support, so scales should be stored as FP16 initially (increasing effective
  bits/element from 2.25 to ~2.31). FP8 can be added later as optimization.
- The three-region layout (item 06) is an optimization for accuracy; the
  core KVarN method works without it (uniform 2-bit throughout). Item 06
  can be deferred if initial results are acceptable.
- CPU-only implementation (items 01-08, 10-11, 13-14) is sufficient for
  correctness validation. GPU kernel (item 09) and CLI flag (item 12) are
  for production use.