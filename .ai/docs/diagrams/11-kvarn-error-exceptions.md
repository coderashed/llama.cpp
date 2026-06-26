---
title: Error Decomposition Measurement -- Numerical Stability & Edge Cases
---

```mermaid
graph TD
    subgraph "Preconditions (caller must ensure)"
        P1["orig != NULL, dequant != NULL<br/>(non-null pointers)"]
        P2["n_tokens > 0, dim > 0<br/>(valid dimensions)"]
        P3["orig and dequant are FP32<br/>(not NaN, not Inf)"]
        P4["orig and dequant have same shape<br/>[n_tokens x dim] each"]
    end

    subgraph "Numerical Safeguards (implementation guarantees)"
        N1["dot product uses double precision<br/>to avoid catastrophic cancellation"]
        N2["norm product denominator:<br/>norm_K * norm_Kdq + 1e-30<br/>prevents division by zero"]
        N3["cos_theta clamped to [-1.0, 1.0]<br/>after division (FP rounding)"]
        N4["E_T denominator in ratio:<br/>E_M / (E_T + 1e-30)<br/>prevents division by zero"]
        N5["All intermediate values<br/>computed in double,<br/>cast to float at output"]
    end

    subgraph "Postconditions (verification)"
        V1["All ratios in [0.0, 1.0]<br/>(E_M <= E_T by construction)"]
        V2["E_T == E_M + E_D<br/>(decomposition identity holds)"]
        V3["E_M >= 0, E_D >= 0, E_T >= 0<br/>(all non-negative)"]
        V4["No NaN or Inf in any output<br/>(finite check on all arrays)"]
        V5["Histogram bin counts sum to n_tokens<br/>(no tokens lost)"]
    end

    subgraph "Edge Cases"
        E1["Zero vector (all zeros):<br/>norm_K = 0, norm_Kdq = 0<br/>cos_theta = 0/1e-30 = 0<br/>E_M = 0, E_D = 0, E_T = 0<br/>ratio = 0/1e-30 = 0"]
        E2["Perfect reconstruction:<br/>orig == dequant<br/>norm_K == norm_Kdq, cos_theta = 1<br/>E_M = 0, E_D = 0, E_T = 0<br/>ratio = 0"]
        E3["Anti-parallel vectors:<br/>cos_theta = -1<br/>E_D = 4 * norm_K * norm_Kdq<br/>E_M = (norm_K - norm_Kdq)^2<br/>ratio may be near 0 or 1"]
        E4["Single token (n_tokens = 1):<br/>top-k% = that single token<br/>histogram has 1 entry<br/>median = that token's ratio"]
        E5["All tokens identical:<br/>all ratios equal<br/>top-k% mean == overall mean<br/>histogram: single bin full"]
        E6["Extreme outlier token:<br/>norm_K >> norm_Kdq<br/>E_M dominates, ratio -> 1.0<br/>E_D negligible by comparison"]
        E7["Dequant all zeros (broken quant):<br/>norm_Kdq = 0<br/>cos_theta = 0/1e-30 = 0<br/>E_M = norm_K^2, E_D = 0<br/>E_T = norm_K^2, ratio = 1.0"]
    end

    P1 --> N1
    P2 --> N1
    P3 --> N1
    P4 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> V1
    N5 --> V2
    N5 --> V3
    N5 --> V4
    N5 --> V5
```

### Safety Contract Summary

| Condition | Assertion | Location |
|-----------|-----------|----------|
| Pointer validity | `assert(orig != NULL && dequant != NULL)` | `compute_error_decomposition` |
| Dimension validity | `assert(n_tokens > 0 && dim > 0)` | `compute_error_decomposition` |
| Division by zero (cos) | Denominator `norm_K * norm_Kdq + 1e-30` | `compute_error_decomposition` |
| cos_theta range | Clamped to `[-1.0, 1.0]` | `compute_error_decomposition` |
| Division by zero (ratio) | Denominator `E_T + 1e-30` | `compute_error_decomposition` |
| FP precision | All accumulators `double` | `compute_error_decomposition` |
| Output finiteness | `assert(isfinite(ratios[t]))` | After decomposition |
| Histogram integrity | `assert(sum(counts) == n_tokens)` | `error_histogram` |

### Numerical Stability Rationale

1. **Double-precision accumulation**: The dot product `sum(orig * dequant)` can lose precision in single-precision when `dim` is large (e.g., 128). Using `double` for all accumulators keeps relative error below 1e-12.

2. **1e-30 epsilon**: Chosen to be safely below any valid FP32 norm product (minimum non-zero FP32 is ~1.4e-45, but typical K matrix values give norms > 1e-10). The epsilon is large enough to prevent overflow in division but small enough to not affect valid ratios.

3. **cos_theta clamping**: FP rounding can produce values like `1.0000001` after division. Clamping to `[-1.0, 1.0]` prevents `E_D` from becoming negative (which would violate the decomposition identity).

4. **E_T + 1e-30 in ratio**: When `orig == dequant` (perfect reconstruction), `E_T = 0` and the ratio is undefined. The epsilon maps this to `ratio = 0`, which is correct (no magnitude error).

### Threshold Rationale

Per `.ai/research/kvarn/02-error-decomposition.md`:

- **Top 5% E_M/E_T < 0.6 for KVarN**: KVarN's dual-scale mechanism explicitly corrects magnitude for outlier tokens. The top 5% worst tokens should have E_M/E_T significantly below the Q4_0 baseline.
- **KVarN top-5% < Q4_0 top-5%**: The primary exit criterion. KVarN must demonstrably suppress magnitude errors better than plain Q4_0 on the same data.
- **Expected values** (from spike results, `.ai/self-study/2026-06-25_01_kvarn_core.md`):
  - Q4_0 top-5% E_M/E_T: ~0.75-0.85
  - KVarN top-5% E_M/E_T: ~0.45-0.55
  - KVarN improvement: 1.2x-8.7x depending on data distribution
