# VarN Safety / Specification Contract

## Memory Ownership

| Requirement | Specification |
|---|---|
| Caller owns all buffers | `T`, `S_c`, `S_r` are caller-allocated `float*` buffers. Function never allocates or frees. |
| `T` size | Must be `R * C` floats. Read-write; modified in-place. |
| `S_c` size | Must be `C` floats. Write-only output. |
| `S_r` size | Must be `R` floats. Write-only output. |
| Temporary storage | All temporaries (column/row log-scales, best-state copies) are allocated on the stack or via `std::vector` internally. |

## Thread Safety

| Aspect | Guarantee |
|---|---|
| Reentrancy | Fully reentrant. No global or static mutable state. |
| Thread safety | Safe to call concurrently on disjoint buffer sets. |
| Side effects | None. Only modifies the caller-provided buffers. |

## Numerical Stability

| Condition | Behavior |
|---|---|
| `log(0)` | Never occurs because abs-mean is computed first, then clamped via `c_min`. With `c_min = -5.0`, the minimum abs-mean is `exp(-5.0) ~ 0.0067 > 0`. |
| `exp(clamp(c_max))` | `exp(5.0) ~ 148.4`, well within `float` range (`FLT_MAX ~ 3.4e38`). No overflow. |
| Subnormal values | Possible if input tile contains extreme outliers; clamped away by subsequent iterations because the log-mean drives values toward unit variance. |
| NaN propagation | Any NaN in input `T` will propagate through `abs` -> `log` -> `exp`. Function does **not** detect or sanitize NaN. |
| Inf propagation | `±Inf` in input `T` will produce `abs(Inf) = Inf`, `log(Inf) = Inf`, which is clamped by `c_max`. Output `S_c` / `S_r` may contain `Inf` if `exp(clamp(Inf,..))` = `exp(c_max)`. |

## Precondition Contract

| Condition | Assertion | Behavior |
|---|---|---|
| `R < 0` | Unchecked | Undefined behavior (indexing with negative stride). |
| `C < 0` | Unchecked | Undefined behavior. |
| `R == 0` | Implicit | Loop over rows is skipped. `S_r` remains zero-length (no writes). `T` unchanged. |
| `C == 0` | Implicit | Loop over columns is skipped. `S_c` remains zero-length (no writes). `T` unchanged. |
| `K <= 0` | Implicit | No iterations. Best state = initial state (no-op). `T`, `S_c`, `S_r` remain at initial values. |
| `c_min > c_max` | Unchecked | `std::clamp` behavior is undefined per [alg.clamp] (iterations may produce NaN). |
| `T == nullptr` | Unchecked | Undefined behavior (null dereference). |
| `S_c == nullptr` | Unchecked | Undefined behavior (null dereference). |
| `S_r == nullptr` | Unchecked | Undefined behavior (null dereference). |
| `T` non-finite | Unchecked | NaN/Inf propagates; output is mathematically meaningless but will not crash within the function. |
| `T` contains `NaN` | Unchecked | `variance(NaN)` = NaN; Imb = NaN; best-state comparison `NaN < best_Imb` is `false` (NaN comparisons always false), so first iteration never saves. On subsequent iterations the same issue persists. Output will be all-NaN if input is all-NaN. |

## Recommended Caller-Side Guards

```cpp
// Suggested guard pattern (not in function, at call sites)
if (R <= 0 || C <= 0 || K <= 0) return;                 // no-op
GGML_ASSERT(T != nullptr && S_c != nullptr && S_r != nullptr);
GGML_ASSERT(c_min < c_max);
```

## Algorithmic Complexity

| Aspect | Cost |
|---|---|
| Time | `O(K * R * C)` per call. For G=128, K=12: ~393k float ops. |
| Space | `O(R + C)` for log-scale vectors + `O(R * C)` for best-tile copy. |
| Flops | ~`2 * K * R * C` abs-adds + `K * C` log + `K * R` log + `K * C` exp + `K * R` exp + `O(K * (R + C))` variance. |

## Spike-Verified Convergence

Confirmed by `spikes/01_kvarn_core/kvarn_core.cpp`:

| K (iterations) | Imb() | Notes |
|---|---|---|
| 0 | 312.3 | Initial imbalance |
| 4 | 20.0 | 93% reduction |
| 8 | 12.0 | 96% reduction. **Exit criterion: < 1.05 ratio across row/col variances** |
| 12 | 8.7 | Default K = 12 (safety margin) |
| 16 | 6.9 | Diminishing returns |
