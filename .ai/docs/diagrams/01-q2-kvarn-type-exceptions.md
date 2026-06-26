---
title: Q2_KVARN Type Definition -- Safety / Specification Contract
---

# Safety Contract for [GGML_TYPE_Q2_KVARN] Item 01

## 1. Memory Alignment Invariants

| Field | Offset | Type | Alignment | Overlap risk? |
|-------|--------|------|-----------|---------------|
| `qs[32]` | 0 | `uint8_t[32]` | 1-byte | None |
| `d` | 32 | `ggml_half` | 2-byte | Aligned (32 is 2*16) |
| `s1` | 34 | `ggml_half` | 2-byte | Aligned (34 is 2*17) |
| `s2` | 36 | `ggml_half` | 2-byte | Aligned (36 is 2*18) |

- **Total sizeof**: 38 bytes
- **No padding required**: all fields are naturally aligned
- **static_assert** `GGML_ASSERT(sizeof(block_q2_kvarn) == 38)` must be added
- **No trailing padding**: blck_size (128) and sizeof (38) are coprime; memcpy of block arrays is safe because C guarantees array elements are packed with no inter-element padding
- **ABI compatible** across C/C++/CUDA/Metal: all fields are standard-layout types; the struct is a standard-layout POD

## 2. Thread Safety

| Function | Thread-safe? | Rationale |
|----------|-------------|-----------|
| `quantize_q2_kvarn(src, dst, nrow, ...)` | **Yes** | Stateless; reads `src`, writes `dst`. Overlapping dst regions across threads would be a caller error. |
| `quantize_row_q2_kvarn_ref(x, y, k)` | **Yes** | Stateless; no mutable globals. |
| `dequantize_row_q2_kvarn(x, y, k)` | **Yes** | Stateless; no mutable globals. |
| `ggml_get_type_traits(type)` | **Yes** | Reads from static const array. |
| `type_traits[GGML_TYPE_Q2_KVARN]` | **Yes** | Entire array is `static const` initialized at compile time. |

- **No mutable global state** is introduced by this item.
- **No locks** needed.
- **Reentrant**: All functions are safe to call from signal handlers (no malloc/free, no locks).

## 3. Numerical Bounds

### Dequant Output Range

```
val = (qval + d) * s1 * s2

where:
  qval in [0, 3]     (uint, 2-bit unpacked)
  d    is the FP16 zeropoint (no negation), typically d = min_value
  s1   = (max_value - min_value) / 3.0f
  s2   in [0.74, 22026]   (~exp([-0.3, 10.0]) from VarN)

Upper bound (worst case):
  qval = 3, d = 0, s1 = large, s2 = 22026
  -> val = (3 + 0) * s1 * s2 = 3 * s1 * 22026
  -> If s1 = 1.0, max output = 66078 (fits in FP32, no overflow)

Lower bound (worst case):
  qval = 0, d = -max, s1 = positive, s2 = 0.74
  -> val = (0 + d_neg) * s1 * 0.74
  -> If d = -3.4e38 (FP16 min), val could be large negative
  -> However, s1 * s2 damps: FP16 max abs = 65504, s2 >= 0.74
```

**Guarantee**: `dequantize_row_q2_kvarn` never produces NaN or infinity from valid `block_q2_kvarn` input. The FP16-to-FP32 conversion clamps subnormals; the multiply chain is linear and preserves sign.

### Quant Input Clamping

```
qval = clamp(round((x[j] - d) / s1), 0, 3)

d  = min(x)        or computed zeropoint (FP16 rounded)
s1 = range / 3.0   (where range >= 0)

Invariant:
  If s1 == 0 (all x[j] equal), set qval[j] = 0 for all j.
  (Division by zero is avoided: if range == 0, skip quantization.)
```

## 4. Dispatch Safety

### Type Traits Registration

| Check | Enforcement |
|-------|-------------|
| `blck_size = 128` | Used in `ggml_quantize_chunk` assertion `start % blck_size == 0` |
| `type_size = sizeof(block_q2_kvarn)` | Used in `ggml_row_size()` calculations |
| `to_float` not NULL | GPU backend may crash if NULL -- guaranteed non-NULL for Q2_KVARN |
| `from_float_ref` not NULL | Required for quantization path -- guaranteed non-NULL |

### Quantize Chunk Guarantee

```
result = quantize_q2_kvarn(src, dst, nrows, n_per_row, ...)
assert(result == nrows * ggml_row_size(GGML_TYPE_Q2_KVARN, n_per_row))
```

- **No out-of-bounds writes**: `dst` must be at least `result` bytes; the assert validates.
- **No uninitialized reads**: `src` must have `nrows * n_per_row` valid floats; the function reads only that.
- **Idempotent**: Repeated `quantize -> dequantize -> requantize` on the same data produces identical blocks (deterministic).

### Enum Position Safety

```
[GGML_TYPE_Q2_KVARN = 42] added after Q1_0 = 41
GGML_TYPE_COUNT updated from 42 to 43
```

- **Backward compatible**: New types are appended at the end per the comment `// NOTE: always add types at the end of the enum to keep backward compatibility` (ggml.h:388)
- **Array bounds**: `type_traits[GGML_TYPE_COUNT]` has 43 elements; index 42 is valid
- **ggml_quantize_chunk switch**: Default case `assert(false)` triggers for unhandled types. Q2_KVARN case is handled.

## 5. Invariants Summary

| # | Invariant | Enforced by | Failure mode |
|---|-----------|-------------|-------------|
| 1 | `sizeof(block_q2_kvarn) == 38` | `static_assert` in ggml-common.h | Compile error |
| 2 | `+d` is stored FP16, sign matches dequant formula | Documentation + code review | Wrong dequant output |
| 3 | `s2` defaults to 1.0 in reference quant | `quantize_row_q2_kvarn_ref` | (None; identity) |
| 4 | `qval` clamped to [0, 3] | `clamp(round(...), 0, 3)` | Out-of-range quant |
| 5 | `blck_size == 128` must match `sizeof(qs)*4` | Design invariant | Wrong dequant stride |
| 6 | `type_traits[Q2_KVARN].to_float` never NULL | Static initialization | Backend crash |
| 7 | `type_traits[Q2_KVARN].from_float_ref` never NULL | Static initialization | Quant path crash |
| 8 | All functions are reentrant | No globals, no malloc | Data race |
