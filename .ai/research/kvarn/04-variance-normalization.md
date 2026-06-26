# Variance Normalization (Algorithm 1)

**Paper section**: 3.3, Appendix H (Algorithm 1)

## Purpose

Variance normalization equalizes the per-row (token) and per-column (channel)
variance of the K/V tile before quantization. This ensures that no single token
has a wildly different scale that would produce an outlier error after rounding.
The normalization is a dual-axis, iterative (Sinkhorn-Knopp-style) balancing in
the log domain.

## Algorithm 1 (verbatim from Appendix H)

```
Input:  Batched tiles T in R^{N x R x C}, Iterations K, Limits c_min, c_max
Output: Balanced tiles T_bal, Scales S_c in R^{N x 1 x C}, S_r in R^{N x R x 1}

function Imb(X):
    v_c <- Var_col(X)   // variance across dim 2 (columns)
    v_r <- Var_row(X)   // variance across dim 1 (rows)
    return max(v_c) / max(min(v_c), 1e-8) + max(v_r) / max(min(v_r), 1e-8)

 1: L_c <- 0_{N x 1 x C};  L_r <- 0_{N x R x 1}    // init log-scales
 2: C <- (T / exp(L_c)) / exp(L_r)                 // current normalized tile
 3: I_best <- Imb(C)
 4: S_c* <- exp(L_c);  S_r* <- exp(L_r)            // track best scales
 5: for k = 1 to K:
 6:     v_col <- clamp(Var_row(C), c_min, c_max)
 7:     L_c <- clamp(L_c + 0.5 * log(v_col), -0.3, 10.0)
 8:     C <- (T / exp(L_c)) / exp(L_r)
 9:     v_row <- clamp(Var_col(C), c_min, c_max)
10:     L_r <- clamp(L_r + 0.5 * log(v_row), -0.3, 10.0)
11:     C <- (T / exp(L_c)) / exp(L_r)
12:     I_curr <- Imb(C)
13:     for n = 1 to N:                           // batched conditional update
14:         if I_curr[n] <= I_best[n]:
15:             I_best[n] <- I_curr[n]
16:             S_c*[n] <- exp(L_c[n])
17:             S_r*[n] <- exp(L_r[n])
18: return (T / S_c*) / S_r*,  S_c*,  S_r*
```

## Parameters

| Parameter | Value used | Meaning |
|-----------|-----------|---------|
| K (iterations) | 8 | Number of alternating row/col normalization passes |
| c_min | (not specified explicitly; likely ~1e-8 or small) | Lower clamp for variance to avoid div-by-zero |
| c_max | (not specified explicitly) | Upper clamp for variance |
| Log-scale clamp | [-0.3, 10.0] | Prevents scales from exploding or vanishing; exp(10) ~ 22026, exp(-0.3) ~ 0.74 |
| Tile shape | N x R x C | N=batch, R=tokens (e.g. 128), C=channels (head_dim, e.g. 128) |

## How it works

1. **Imbalance metric (Imb)**: Measures how non-uniform the variances are across
   rows and columns. `max/min` ratio per axis; lower = more balanced.

2. **Alternating normalization**: In each iteration, compute the variance along
   one axis, take the log, add half of it to the log-scale for that axis, and
   re-normalize. The `0.5 * log(v)` update is the Sinkhorn step that drives the
   variance toward 1.0.

3. **Best-state tracking**: Because the iterations may oscillate, the algorithm
   tracks the best (lowest imbalance) scales seen across all iterations per tile.

4. **Output**: The balanced tile `(T / S_c) / S_r` plus the two scale vectors
   `S_c` (per-column, shape 1xC) and `S_r` (per-row, shape Rx1).

## Integration with KVarN

- `S_c` is absorbed into the standard RTN scale (`s1 = S_c * rt_scale`), stored
  as one FP8 value per channel per group.
- `S_r` becomes the **second scale** `s2`, stored as one FP8 value per token per
  group.
- The zeropoint `z` is the standard RTN zeropoint (FP16, per channel per group).

See [05-kvarn-pipeline.md](05-kvarn-pipeline.md) for the full storage format.

## Cost

[Sec 4.2] For Qwen3-4B: 8 iterations of variance normalization across all
attention layers takes **1.9 ms** per 128-token block, compared to 1050 ms for
generating those 128 tokens. This is a **0.18%** overhead. For larger models
the relative cost decreases (normalization is O(R*C) per tile, generation is
O(model_params)).