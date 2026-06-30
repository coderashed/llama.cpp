# DP4A / INT8 acceleration backlog for Q2_KVARN (gfx906 / MI50)

Intuition source: `.ai/intuitions/2026-06-26_q2_kvarn_int8_dp4a.md` (read this
first - it contains the exact algebra, the /QI8_1 subtlety, and the falsifiable
predictions this backlog is built on).
Proven template kernel: `vec_dot_fattn_vec_KQ_q4_1` (`ggml/src/ggml-cuda/fattn-common.cuh:180-207`).
Target kernel to change: `vec_dot_fattn_vec_KQ_q2_kvarn` (`ggml/src/ggml-cuda/fattn-common.cuh:664-703`).

## Difficulty: LOW-to-MODERATE

Blast radius is one device function. No new ggml type, no block-format change,
no graph change, no public API change. The hard conceptual work (proving the K
dot is exactly a DP4A, and the field semantics) is already done in the intuition
doc. The remaining risk is mechanical correctness (2-bit unpack + the /QI8_1
double-count) and access to gfx906 hardware to validate.

What makes it NOT trivial:
- It only matters / can be validated on a DP4A-capable card (MI50 = gfx906, or
  any NVIDIA with __dp4a). A wrong /QI8_1 or ds.y assumption produces subtly
  wrong logits, not a crash - so it needs a numerical-equivalence test, not
  just "it runs".

## Preconditions (verified 2026-06-26, do NOT re-litigate)

- gfx906 has native DP4A; `ggml_cuda_dp4a` is defined and AMD-aware
  (`ggml/src/ggml-cuda/common.cuh:694`; arch guard `vendors/hip.h:182`).
- Q is ALREADY quantized to Q8_1 for `type_K == Q2_KVARN`: the FA vec kernel
  sets `Q_q8_1 = type_K != F16 && != BF16` (`fattn-vec.cuh:97`), so `Q_q8`
  (int8) and `Q_ds` (float2 = {d_Q, sum Q}) are already available to the K
  vec_dot. No new quantization plumbing is required.
- The FA vec path is the one actually used for single-token decode on AMD
  (`fattn.cu:28`, `Q->ne[1] <= 32/ncols2` selects vec; gfx906 has no
  turing/AMD-mma path). So optimizing this kernel hits the MI50 decode hot path.
- `block_q8_1.ds = {d, sum}` with `ds.y = Sum of ORIGINAL Q floats`
  (`ggml/src/ggml-cuda/quantize.cu:42-51`). The zeropoint correction uses ds.y.

## Scope boundary

- K side only. The V path (`dequantize_V_q2_kvarn`, `fattn-common.cuh:622`) is a
  float softmax-weighted reduction and is intentionally OUT of scope - see the
  intuition doc, section "Why V gets (almost) nothing". Do not INT8 the V path.
- Flash-attention path only. A non-FA mmvq kernel is a separate stretch item
  (04) and only matters if KVarN is run without FA.

---

### 01 - Numerical-equivalence test harness for the K vec_dot

- **Status**: done
- **Depends on**: nothing
- **Pipeline**: implementation
- **Why first**: the failure mode of this work is "subtly wrong logits", which
  only a per-element comparison catches. Build the oracle before the kernel.
- **Exit criterion**: a test (extend `tests/test-q2-kvarn-gpu.cpp`) that, for
  randomized K blocks and a randomized Q vector, computes the attention logit
  `Sum_i Q_i * (k_i + d) * s1 * s2` two ways - (a) the current float
  `vec_dot_fattn_vec_KQ_q2_kvarn`, (b) a direct fp64 reference - and asserts
  agreement within fp16-quantization tolerance. This pins the contract the DP4A
  kernel must meet. Test is skipped (not failed) when no DP4A-capable device is
  present.

### 02 - DP4A K vec_dot kernel

- **Status**: done
- **Depends on**: 01
- **Pipeline**: implementation
- **Exit criterion**: `vec_dot_fattn_vec_KQ_q2_kvarn`
  (`ggml/src/ggml-cuda/fattn-common.cuh:664`) computes the per-block logit as
  `s1*s2 * ( Q_ds.x * sumi + d * Q_ds.y / QI8_1 )` where
  `sumi = ggml_cuda_dp4a(k_int8_packed, q_int8, 0)`, the 2-bit codes are
  expanded to int8 lanes
  (`(b&3) | ((b>>2)&3)<<8 | ((b>>4)&3)<<16 | ((b>>6)&3)<<24`), and the result
  passes the item-01 equivalence test within tolerance on a DP4A device. No
  change to `block_q2_kvarn` or any dequant path. The float kernel may be kept
  behind a compile/runtime guard for non-DP4A archs as a fallback.
- **Gotcha**: `QK8_1=32`, `QI8_1=8` (`ggml-common.h:259,121`); a 128-element
  KVarN block spans 4 Q8_1 sub-blocks with constant `d,s1,s2`, so the zeropoint
  term must be divided by `QI8_1` exactly as q4_1 does (line 206) to avoid an
  8x double-count.

### 03 - MI50 (gfx906) benchmark validation

- **Status**: undone
- **Depends on**: 02
- **Pipeline**: spike
- **Exit criterion**: on an MI50, `llama-bench` (or equivalent) decode
  throughput is measured with `--cache-type-k q2_kvarn` before and after item
  02, holding everything else fixed, across at least two context lengths and two
  head_dims. Records the measured delta and a verdict against the intuition
  doc's prediction (10-30 percent of attention time, NOT 2x). A self-study
  write-back in `.ai/self-study/` confirms or falsifies prediction #2 and
  updates the intuition doc's confidence line.

### 04 - (STRETCH) mmvq kernel for the non-FA quantized-matmul path

- **Status**: undone
- **Depends on**: 02
- **Pipeline**: implementation
- **Exit criterion**: a `vec_dot_q2_kvarn_q8_1` mmvq kernel exists and is
  registered so `ggml_mul_mat` with a `Q2_KVARN` K cache runs without flash
  attention on DP4A hardware, matching the FA-path logits within tolerance.
  Only pursue if a no-FA use case appears - with FA on, item 02 covers the hot
  path. There are currently no `q2_kvarn` entries under
  `ggml/src/ggml-cuda/mmq*`/`mmvq*` (verified 2026-06-26).

---

## Ordering / frontier

```
01 (test harness) --> 02 (dp4a kernel) --> 03 (MI50 benchmark)
                                       \--> 04 (mmvq, stretch, optional)
```

Frontier at creation: item 01 (no dependencies).
