# Intuition: Q2_KVARN can use DP4A on the K side (gfx906/MI50), not the V side

Date: 2026-06-26
Hardware in question: AMD MI50 = gfx906 (Vega 20), GCN5.
Status: intuition + exact algebra, NOT benchmarked. Predictions are falsifiable.

## The one-sentence intuition

The K-cache dot product in flash-attention is an integer dot in disguise and
maps exactly onto gfx906's `v_dot4_i32_i8` (DP4A); the V-cache accumulation is
a float reduction and does not. So on MI50 the realistic INT8 win for KVarN is
K-only, and it is an optimization the current code leaves entirely unused.

## What the hardware actually offers

- gfx906 has DP4A (`v_dot4_i32_i8`): one int8.int8 -> int32 4-wide dot per op.
  Exposed in ggml as `ggml_cuda_dp4a`; gfx906 is in the native path
  (`ggml/src/ggml-cuda/vendors/hip.h:182`, guards `__gfx900__ || __gfx906__`).
- gfx906 has NO matrix cores. MFMA/WMMA start at gfx908 (CDNA). So "leverage
  the INT8 instruction set" means DP4A, full stop. Do not expect tensor-core
  style gains.
- Why q4_0/q4_1 feel fast here: their hot kernels are built on DP4A. Proof on
  the K side is `vec_dot_fattn_vec_KQ_q4_1` (`fattn-common.cuh:201`):
  `sumi = ggml_cuda_dp4a(v, u, 0)`.

## Where KVarN sits today (precise)

- `vec_dot_fattn_vec_KQ_q2_kvarn` (`fattn-common.cuh:664`) does FLOAT math: it
  unpacks each 2-bit code to float and multiplies by `(float)q[b]` even though
  the query `Q_q8` is already Q8_1 int8 and `Q_ds` is already available. The
  integer dot is sitting right there, unused.
- There are NO mmq/mmvq kernels for `q2_kvarn` (verified: grep of
  `ggml/src/ggml-cuda/mmq*`, `mmvq*` returns nothing). KVarN only runs through
  the flash-attention path. The non-FA quantized matmul DP4A path does not
  exist for it.

## Why the K dot is exactly a DP4A (the algebra, not a hand-wave)

KVarN dequant is affine: `K_dq_i = (k_i + d) * s1 * s2`, where `k_i` is the
2-bit code (0..3), `d` is the stored zeropoint (after our fix, `d = min*inv_d`),
and `s1*s2` is a per-128-block scalar.

The attention logit for one key is `Sum_i Q_i * K_dq_i`. Expand:

    Sum_i Q_i (k_i + d) s1 s2
  = s1 s2 [ Sum_i Q_i k_i  +  d Sum_i Q_i ]

Now use the Q8_1 quantization of Q. From `quantize.cu:42-51`:
  - `Q_i ~= q_i * d_Q` with `q_i` int8, `d_Q = ds.x`
  - `ds.y = Sum_i Q_i` (sum of the ORIGINAL query floats, not the quantized ints)

Substitute:
  - `Sum_i Q_i k_i ~= d_Q * Sum_i q_i k_i = ds.x * dp4a(q_int8, k_int8)`
  - `d * Sum_i Q_i = d * ds.y`

So the whole per-block contribution is:

    s1*s2 * ( ds.x * sumi  +  d * ds.y )      with sumi = dp4a(q_int8, k_int8)

This is structurally identical to the proven q4_1 kernel
(`fattn-common.cuh:206`: `K_dm.x*Q_ds.x*sumi + K_dm.y*Q_ds.y/QI8_1`), with the
mapping q4_1 `d4 -> s1*s2`, q4_1 `m4 -> d*s1*s2`.

### The one subtlety that WILL bite if ignored: the /QI8_1 factor

`QK8_1 = 32`, `QI8_1 = 8` (`ggml-common.h:259,121`). A Q8_1 block is 32
elements; a KVarN block is 128, so one KVarN block spans 4 Q8_1 sub-blocks. The
loop adds the `d * ds.y` correction once per int (per `QI8_1` iterations) while
`d`, `s1`, `s2` are constant across the block. q4_1 handles the analogous
double-count by dividing the min term by `QI8_1` (line 206). The KVarN kernel
must do the same: `... + s1*s2 * d * ds.y / QI8_1`. Forget this and the
zeropoint correction is counted 8x -> wrong logits, likely subtle quality loss
rather than an obvious crash.

## The 2-bit packing tax (why this is not as good as q4_0)

- q4_1: one loaded int = 4 nibbles; one shift+mask -> dp4a. Very cheap unpack.
- KVarN: one byte = 4 codes (2-bit). To feed dp4a you must expand a byte into
  4 int8 lanes: `(b&3) | ((b>>2)&3)<<8 | ((b>>4)&3)<<16 | ((b>>6)&3)<<24`,
  ~3-4 ALU ops per dp4a. dp4a still covers 4 elements per call, so 32 dp4a +
  ~32 unpacks per 128-block.

Net: the unpack overhead is ~2-3x q4_1's, but it still replaces the current
inner loop of 4 int->float converts + 4 float FMAs per 4 elements with
~4 int ops + 1 dp4a. I expect the K-dot inner loop to drop roughly 2x in op
count, and dp4a issues at ~1/clk on gfx906.

## Falsifiable predictions

1. A DP4A rewrite of `vec_dot_fattn_vec_KQ_q2_kvarn` produces logits within
   fp16 rounding of the current float kernel (the algebra above is exact up to
   Q8_1 rounding, which the float path also incurs). If outputs diverge
   materially, the /QI8_1 or the ds.y semantics were gotten wrong.
2. End-to-end decode speedup on MI50 with K=q2_kvarn will be modest, my guess
   10-30 percent of attention time, NOT 2x, because:
   - V accumulation stays float (see below) and is a comparable share of FA,
   - softmax + the K load/dequant share the kernel,
   - the 2-bit unpack eats into the dp4a win.
   If someone measures >1.5x end-to-end from this alone, my mental model of the
   FA cost breakdown is wrong.
3. The win grows with head_dim and context length (more K dots per token), and
   shrinks at small head_dim where unpack overhead dominates.

## Why V gets (almost) nothing

V is consumed as `O = Sum_t p_t * V_t` where `p_t` are FLOAT post-softmax
probabilities (`fattn-vec.cuh:346-355`, `dequantize_V_q2_kvarn` at
`fattn-common.cuh:622`). There is no int8.int8 contraction: one operand is a
float distribution. You could imagine requantizing `p` to int8 per tile, but
the dynamic range of softmax weights is awful for 8-bit and the accumulation is
the numerically sensitive part (this is the same cache whose Hadamard
un-rotation we just saw amplify small structured errors into gibberish). My
strong intuition: do not try to INT8 the V path. KIVI-family methods keep V
dequant in float for exactly this reason.

## If we wanted more INT8 than FA gives

The bigger structural lever is an mmvq/mmq `q2_kvarn` kernel for the non-FA
quantized-matmul path (none exists today). That path is where q4_0 gets its
reputation on these cards. But it only matters if KVarN is ever run without
flash-attention; with FA on, the FA vec kernel above is the hot path and the
K-dot DP4A rewrite is the highest-value, lowest-risk change.

## Confidence

- "K dot is exactly a DP4A, modeled on q4_1": high. It is algebra plus a cited
  matching kernel.
- "/QI8_1 correction needed": high, by direct analogy to line 206.
- "10-30 percent end-to-end, not 2x": medium. This is a guess about the FA cost
  breakdown on gfx906 and should be benchmarked before anyone invests.
- "V path not worth INT8": high on the math, medium on "never" - a clever tile
  requant might claw back a little, but the risk/reward is bad.
