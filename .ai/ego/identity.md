# Identity: llama.cpp KVarN Implementation

## Who we are

We are implementing KVarN (Variance-Normalized KV-Cache Quantization) into
llama.cpp. KVarN is a calibration-free 2-bit KV-cache quantizer from a paper
by Huawei (arXiv:2606.03458v1) that combines Hadamard rotation, dual-axis
variance normalization, and 2-bit RTN quantization.

## How we work

- **Language**: C (ggml/) and C++ (src/). No Python in the core library.
- **Build**: CMake. No Makefile (deprecated). Configure with
  `cmake -B build`, build with `cmake --build build`.
- **Tests**: C/C++ test binaries in `tests/`, registered via `tests/CMakeLists.txt`
  using `llama_build_and_test()`. Run with `ctest` or directly.
- **Code style**: No comments unless genuinely non-obvious. No emdash, no
  unicode arrows. Use ASCII: `-`, `->`, `x`, `...`. Follow AGENTS.md strictly.
- **Public API**: C ABI in `include/llama.h`. Opaque structs, PIMPL pattern.
  No C++ exceptions cross the C boundary.
- **Error handling**: GGML_ABORT/GGML_ASSERT for invariants in ggml/. C++
  exceptions acceptable in src/ (caught at C API boundary).
- **Quantization**: New types go in `ggml/include/ggml.h` enum, block structs in
  `ggml/src/ggml-common.h`, kernels in `ggml/src/ggml-quants.c`, type_traits
  in `ggml/src/ggml.c`. GPU kernels in `ggml/src/ggml-cuda/`.

## What exists

- `.ai/docs/` - 19 architecture docs covering all llama.cpp features
- `.ai/docs/07-kv-cache.md` - current KV cache with Hadamard rotation
- `.ai/docs/05-quantization.md` - quant type system (how to add a new type)
- `.ai/research/kvarn/` - 12 paper documentation files
- `.ai/undone/KVARN_UNDONE.md` - 14-item implementation backlog
- `spikes/01_kvarn_core/` - validated core algorithm (spike MET)
- The existing Hadamard rotation in KV cache (`attn_rot_k`/`attn_rot_v`)
  already covers KVarN's step 1. The novel parts are VarN (Algorithm 1) and
  the dual-scale 2-bit storage format.

## Key decisions from spike

- Default VarN iterations: 12 (not 8, for safety margin on imbalanced data)
- FP16 scales initially (2.375 bpe), FP8 later as optimization (2.25 bpe)
- Dequant formula: `K_dq = (K_q + z) * s1 * s2`. CRITICAL: `z` must be in
  QUANTIZED units (`z = min / half_range`), not original units. The quantizer
  currently stores `z = min`, which is wrong - see self-study 02.
- Compare against 2-bit baselines, not 4-bit (fair bit-budget comparison)

## Hard-won lessons

- **K and V are not symmetric, even when the code path looks identical.** K's
  quant error sits inside `Q.K^T` behind a softmax (forgiving). V's error is the
  output and is then Hadamard-un-rotated (unforgiving). "It works for K" is NOT
  evidence the quant kernel is correct. (self-study 02)
- **Orthogonal rotation amplifies structured error.** Hadamard spreads random
  error evenly but concentrates a constant (DC) vector into one coordinate with
  a sqrt(n) gain. Any systematic bias in the quantizer is dangerous precisely
  because we rotate K and V.
- **Round-trip RMSE tests can pass while the kernel is broken.** The per-block
  `s2` norm correction masks per-element bias in aggregate metrics. Test the
  DC component and the rotated/un-rotated round-trip per element, not just norm.
- **Validate storage-format conventions end-to-end, including units.** The
  zeropoint sign was reasoned about; its units were not. `(q+z)*s1` only inverts
  the quantizer when `z = min/half_range`.