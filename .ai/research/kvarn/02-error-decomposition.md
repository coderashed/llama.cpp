# Error Decomposition: Magnitude vs. Direction

**Paper section**: 3.1, Eq. 2-3, Fig 1a

## The decomposition

Given a full-precision key vector `K` and its dequantized approximation `K_dq`,
the squared L2 error decomposes into magnitude and directional components:

```
||K - K_dq||^2  =  E_M + E_D
```

where:

```
E_M = (||K|| - ||K_dq||)^2                    // magnitude error
E_D = 2 * ||K|| * ||K_dq|| * (1 - cos(theta))  // directional error
```

and `theta` is the angle between `K` and `K_dq`.

**Derivation** [Eq 2-3]:

```
||K - K_dq||^2 = ||K||^2 - 2*||K||*||K_dq||*cos(theta) + ||K_dq||^2

Add and subtract 2*||K||*||K_dq||:

= (||K||^2 - 2*||K||*||K_dq|| + ||K_dq||^2)    [magnitude squared]
+ (2*||K||*||K_dq|| - 2*||K||*||K_dq||*cos(theta))  [directional]

= (||K|| - ||K_dq||)^2 + 2*||K||*||K_dq||*(1 - cos(theta))
```

## Why this matters

The decomposition separates two failure modes:
- **Magnitude error (E_M)**: the quantized token has the wrong norm. In attention,
  this scales the contribution of that token by an incorrect multiplicative factor.
  Across decode steps, incorrect multipliers compound exponentially.
- **Directional error (E_D)**: the quantized token points in a slightly wrong
  direction. This adds angular noise to attention but does not compound in the
  same multiplicative way.

## Empirical finding [Fig 1a]

Using the ratio `E_M / E_T` (fraction of total error due to magnitude):

- Among the **top k% largest errors** (outliers), the vast majority of the error
  is magnitude-based, not directional.
- For the bulk (non-outlier) errors, magnitude and direction contribute more
  equally.

**Conclusion**: To fix outliers, one must fix token magnitude (scale). Directional
errors are a secondary concern.

## Implementation relevance

A quantizer targeting reasoning quality should:
1. Measure per-token magnitude error, not just MSE.
2. Ensure that the dequantized norm `||K_dq||` closely matches `||K||` for every
   token, especially the worst-case ones.
3. KVarN achieves this by storing an explicit second scale `s2` (per row/token
   axis) derived from variance normalization, so the dequantized magnitude is
   directly corrected rather than left to the rounding lottery.

## Measurement recipe

To replicate the decomposition for a given quantizer:
1. Collect all `(K, K_dq)` pairs across a validation set.
2. For each pair, compute `E_M`, `E_D`, `E_T = E_M + E_D`.
3. Sort tokens by `E_T` (descending).
4. For the top-k% percentile, compute `mean(E_M / E_T)`.
5. If this ratio is high (>0.5), magnitude is the dominant outlier driver.