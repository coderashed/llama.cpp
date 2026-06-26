# Self-Study 02: Why Q2_KVARN V-cache produces gibberish (K is fine) - DIAGNOSED

## Question

`--cache-type-k q2_kvarn` works. Adding `--cache-type-v q2_kvarn` produces
gibberish. The K and V paths share the same block format, the same quantizer,
the same dequant formula, and the same (involutive) Hadamard rotation. So why
does one cache work and the other collapse?

## Evidence

### The zeropoint is stored in the wrong units

`quantize_q2_kvarn_block` (ggml/src/ggml-quants.c:96-122):

```
half_range = range / 3
inv_d      = 1 / half_range
q          = round((x - min) * inv_d)   // quantizer subtracts min/half_range
y->d  = min                              // <-- stores min
y->s1 = half_range
```

Every dequant site computes `(q + d) * s1 * s2`. For that to invert the
quantizer, `d` must equal `min / half_range` (= `min * inv_d`), NOT `min`.
As written:

- correct reconstruction:  `q*s1 + min`
- code actually computes:   `(q + min)*s1 = q*s1 + min*half_range`

A systematic per-block constant offset of `min*(half_range - 1)` is baked into
every dequantized element. Confirmed with a standalone round-trip
(scratchpad/rt.c): setting `d = min*inv_d` makes the no-s2 dequant EXACTLY
equal the correct `q*s1 + min`.

Same wrong zeropoint on GPU (ggml/src/ggml-cuda/cpy-utils.cuh:228,
`y->d = min_val`), which ALSO hard-codes `s2 = 1.0` (line 230) - so CUDA does
not even get the norm band-aid.

### s2 hides per-element RMSE but not the DC component

`compute_s2_scale` rescales each block so its L2 norm matches the original.
Round-trip rel-RMSE stays ~0.25 with or without the bug - the norm correction
masks it. What it cannot remove is the DC (mean) component of the error:

```
DC error, buggy formula   : +7.5% of block RMS
DC error, correct formula : -2.4% of block RMS (just quant noise)
```

### Why K tolerates the offset and V does not

- K is consumed as Q.K^T. A constant offset per key only shifts attention
  logits, which softmax largely absorbs, and K is never un-rotated. Tolerable.
- V is the OUTPUT: `O = (P.V).H^T`. The per-block offset is `beta * ones`.
  The Hadamard un-rotation maps `ones -> sqrt(n) * e_0`, so the offset does not
  average out - it concentrates and is amplified ~x8 into a single output
  channel per head (channels 0 and 64). In the test that spike is ~60% of the
  signal RMS. Those channels feed `wo` and blow up. Gibberish.

The rotation machinery itself is correct: the Sylvester Hadamard
(ggml_gen_hadamard, src/llama-kv-cache.cpp:22) is a symmetric involution
(R.R = I), and build_attn applies `self_v_rot` to BOTH v_cur and the attention
output (src/llama-graph.cpp:2331-2333, 2362-2364). The upstream zeropoint bug
is what turns the V un-rotation from "mild quality loss" into catastrophe.

## What we learned

1. **A round-trip RMSE test would NOT have caught this.** s2 makes the buggy
   and correct formulas look equally good on aggregate norm/RMSE. The bug only
   shows up in the DC component of the per-element error, and only becomes fatal
   after the V-side Hadamard un-rotation concentrates that DC term. Our existing
   test-q2-kvarn-dequant / -error tests almost certainly pass DESPITE the bug.

2. **K and V are not symmetric even when the code path looks symmetric.** K's
   error lands in a dot-product behind a softmax (forgiving); V's error lands
   directly in the output and then gets rotated (unforgiving). An error budget
   that is fine for K can be lethal for V. "It works for K" is not evidence the
   quant kernel is correct.

3. **Orthogonal rotation amplifies structured (low-rank/DC) error.** Hadamard
   spreads random error evenly but concentrates a constant vector into one
   coordinate with a sqrt(n) gain. Any systematic bias in the quantizer is
   dangerous specifically because we rotate.

4. **The zeropoint convention was never validated end-to-end.** The format
   comment (ggml-common.h:189 "FP16 zeropoint (not negated)") shows we reasoned
   about sign but not about units. `(q+z)*s1` requires z in quantized units
   (z = min/half_range), and the quantizer stored z in original units (min).

## Therefore we will

1. **Fix the stored zeropoint in both quantizers**, not the read paths:
   - ggml-quants.c:120  -> `y->d = GGML_FP32_TO_FP16(min * inv_d)`
   - ggml-quants.c:83   -> compute_s2_scale dq_val uses `(qval + min*inv_d)*half_range`
     (then s2 ~= 1, as intended)
   - cpy-utils.cuh:228  -> `y->d = __float2half(min_val * inv_s1)`
     (and compute a real s2 instead of 1.0)
   All read paths (dequantize_row_q2_kvarn, CUDA dequantize_block_q2_kvarn,
   vec_dot_fattn_vec_KQ_q2_kvarn, dequantize_V_q2_kvarn) already use
   `(q+d)*s1*s2` and need no change once d is stored correctly.

2. **Add a V-specific regression test**, not just a round-trip: quantize ->
   rotate -> dequant -> un-rotate and assert the recovered tensor matches the
   original within 2-bit tolerance PER ELEMENT (not just in norm). A DC-offset
   assertion (mean error near zero) would have caught this directly.

3. **Treat the GPU s2 = 1.0 shortcut as a separate defect.** Even with the
   zeropoint fixed, dropping the per-block norm correction on CUDA leaves the
   GPU path measurably worse than CPU. Compute s2 in the device kernel.

4. **When evaluating any future quant kernel, separate the error into DC vs AC**
   and test the rotated/un-rotated round-trip, because that is the regime KVarN
   actually runs in.
