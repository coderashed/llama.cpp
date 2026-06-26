---
title: Q2_KVARN Quantization -- Safety Contract & Exceptions
---

```mermaid
graph TD
    subgraph "Preconditions (caller must ensure)"
        P1["k % 128 == 0<br/>(elements multiple of block size)"]
        P2["src != NULL, dst != NULL<br/>(non-null pointers)"]
        P3["n_per_row % 128 == 0<br/>(row alignment)"]
        P4["dst buffer size >=<br/>nrow * ggml_row_size(Q2_KVARN, n_per_row)"]
    end

    subgraph "Numerical Bounds (implementation guarantees)"
        N1["qs values: 0..3<br/>(clamped after rounding)"]
        N2["d = min(x_block)<br/>FP16 finite"]
        N3["s1 = (max-min)/3<br/>FP16 finite, >= 0"]
        N4["s2 = norm_orig / norm_dq<br/>FP16 finite, >= 1e-10"]
        N5["If range == 0 (all equal):<br/>s1 = 0, qs = 0, d = value"]
    end

    subgraph "Postconditions (verification)"
        V1["ggml_quantize_chunk returns<br/>nrow * 38 bytes"]
        V2["dequantize roundtrip:<br/>RMSE < 1.0 (2-bit bound)"]
        V3["E_M/E_T median ratio < 0.5<br/>(per-token magnitude error)"]
        V4["E_T finite for all tokens<br/>(no NaN/inf in dequant)"]
    end

    subgraph "Edge Cases"
        E1["All-zero input: d=0, s1=0, qs=0, s2=1"]
        E2["Constant input (all same):<br/>d=value, s1=0, qs=0, s2=1"]
        E3["Single NaN in input:<br/>min=NaN, max=NaN -> s1=NaN<br/>(caller must sanitize)"]
        E4["Infinity input:<br/>min=-inf, max=+inf -> s1=inf<br/>(caller must sanitize)"]
    end

    P1 --> N1
    P2 --> N1
    P3 --> N1
    P4 --> N1
    N1 --> V1
    N2 --> V2
    N3 --> V2
    N4 --> V2
    N5 --> V2
    V2 --> V3
    V2 --> V4
```

### Safety Contract Summary

| Condition | Assertion | Location |
|-----------|-----------|----------|
| Block alignment | `assert(k % qk == 0)` | `quantize_row_q2_kvarn_ref:77` |
| qs range | Clamped to `[0, 3]` | `quantize_row_q2_kvarn_ref:109-110` |
| s1 non-negative | `half_range = (max-min)/3 >= 0` | `quantize_row_q2_kvarn_ref:92` |
| s2 identity | `norm_orig / norm_dq` (or `1.0f` if `norm_dq < 1e-10`) | `compute_s2_scale:90` |
| Zero range | `inv_d = 1.0f` when `half_range == 0` | `quantize_row_q2_kvarn_ref:112` |
| RMSE bound | `< 1.0` for 2-bit | `test-q2-kvarn-type.cpp:115` |
| E_M/E_T median | `< 0.5` | Test 6 (NEW) |

### E_M/E_T Threshold Rationale

Per `.ai/research/kvarn/02-error-decomposition.md`:
- E_M/E_T > 0.5 means magnitude error dominates total error for that token
- The median token should have E_M/E_T < 0.5, meaning directional error is the larger component
- This is the expected behavior for a well-calibrated quantizer: most tokens have small magnitude error, and the dominant error is angular noise from rounding
- If the median E_M/E_T >= 0.5, the quantizer is systematically mis-scaling tokens (s1/s2 are wrong)
