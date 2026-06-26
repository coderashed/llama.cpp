---
title: Q2_KVARN Dequantization -- Safety / Specification Contract
---

# Safety Contract for [dequantize_row_q2_kvarn] Item 02

## 1. Bit Mask Correctness (CRITICAL — FIXED)

| Field | Value | Status |
|-------|-------|--------|
| Mask | `& 0x03` | Correct: extracts both bits of each 2-bit field |
| Shift | `>> (b * 2)` | Correct (0, 2, 4, 6) |

**Bug (fixed)**: `& 0x01` caused values 2 and 3 to be decoded as 0 and 1. Changed to `& 0x03`.

## 2. Precondition Contract

| Condition | Enforcement | Failure mode |
|-----------|-------------|-------------|
| `k % 128 == 0` | `assert(k % qk == 0)` | Assertion failure (debug) / UB (release) |
| `x` non-NULL | Implicit (no check) | Segfault on dereference |
| `y` non-NULL | Implicit (no check) | Segfault on write |
| `x` has `k/128` valid blocks | Caller responsibility | Out-of-bounds read |
| `y` has `k` valid float slots | Caller responsibility | Out-of-bounds write |
| No aliasing between x and y | `GGML_RESTRICT` annotation | UB if aliased (compiler assumes no alias) |

## 3. Postcondition Contract

| Property | Guarantee |
|----------|-----------|
| Output range | `y[i]` in `[d * s1 * s2, (3 + d) * s1 * s2]` for each block |
| No NaN/Inf | Guaranteed from valid FP16 inputs (FP16->FP32 never produces NaN from valid FP16) |
| Deterministic | Same input always produces same output (no rounding, no randomness) |
| Read-only input | `x` is `const` -- never modified |
| Thread safety | Reentrant: no globals, no malloc, no locks |

## 4. Numerical Tolerance

### Roundtrip accuracy (quantize -> dequantize)

```
For any input x[i] in FP32:
  qval = clamp(round((x[i] - d) / s1), 0, 3)
  y[i] = (qval + d) * s1 * s2

Error bound:
  |y[i] - x[i]| <= s1 * s2 / 2   (half-step quantization error)
  
  With s2 = 1.0 (reference quant):
    |y[i] - x[i]| <= s1 / 2 = range / 6

  For typical range ~4.0:
    max error ~ 0.67
```

### Test 1 (bit_mask) tolerance: 1e-3 per element

Test 1 uses a **known-quantized block** (not roundtrip):

```
Given a pre-computed block_q2_kvarn with known values:
  qs[j] = 0xE4 (0b_11_10_01_00) for all 32 bytes
  d     = 0.0f
  s1    = 1.0f
  s2    = 1.0f

Expected output:
  y[4*j+0] = 0.0, y[4*j+1] = 1.0, y[4*j+2] = 2.0, y[4*j+3] = 3.0

Assert: |y_actual[i] - y_expected[i]| < 1e-3 for all i in [0, 128)
```

This tests the dequant kernel in isolation, independent of quantizer accuracy.

### Test 3 (dual_scale) test vector

```
block_q2_kvarn test_vector = {
  .qs = { 0xE4, 0xE4, ..., 0xE4 },  // 32 bytes, each = 0b_11_10_01_00
  .d  = fp16(1.0f),
  .s1 = fp16(2.0f),
  .s2 = fp16(3.0f),
};

// q0=0, q1=1, q2=2, q3=3 repeated 32 times
// Expected: y[4*j+0]=6.0, y[4*j+1]=12.0, y[4*j+2]=18.0, y[4*j+3]=24.0
```

## 5. Invariants Summary

| # | Invariant | Enforced by | Failure mode |
|---|-----------|-------------|-------------|
| 1 | `& 0x03` mask extracts 2 bits | Code review + Test 1 (bit_mask) | Wrong dequant values |
| 2 | `k % 128 == 0` | `assert` | Debug crash / release UB |
| 3 | `x` and `y` not aliased | `GGML_RESTRICT` | Compiler UB if aliased |
| 4 | Output matches formula | Test 1 (bit_mask) + Test 3 (dual_scale) | Wrong dequant |
| 5 | No NaN/Inf from valid input | FP16->FP32 guarantees | N/A |
| 6 | Thread-safe / reentrant | No globals | Data race |
| 7 | `to_float` registered non-NULL | Static init (Item 01) | Backend crash |
