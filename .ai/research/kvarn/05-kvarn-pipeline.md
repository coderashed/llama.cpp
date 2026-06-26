# KVarN Pipeline & Storage Format

**Paper sections**: 2.1, 2.3, 3.3, Appendix D

## Full pipeline

```
K, V (fp16, shape [n_tokens, n_heads, head_dim])
  |
  |  [1] Channel-dim Hadamard rotation (per head, block-diagonal)
  |      -> K_rot, V_rot
  |      (online, O(N log N) per head)
  |
  |  [2] After every G tokens (block), apply VarN (Algorithm 1)
  |      -> K_norm = K_rot / (S_c * S_r),  V_norm = V_rot / (S_c * S_r)
  |      -> produces S_c (1xC), S_r (Rx1) per tile
  |      (~8 iterations, 0.18% overhead)
  |
  |  [3] RTN quantize K_norm per channel (K) or per token (V)
  |      -> K_q (2-bit), zeropoint z (FP16 per channel per group)
  |      -> standard scale s1 = S_c * rt_scale (FP8 per channel per group)
  |
  |  [4] Store:
  |      K_q:     2-bit, G=128 elements per group, per channel
  |      z:       FP16, one per channel per group  -> G elements share 1 z
  |      s1:      FP8,  one per channel per group   (absorbs S_c)
  |      s2:      FP8,  one per token per group      (S_r)
  |
  v
  Dequant: K_dq = (K_q + z) * s1 * s2
```

## Bits-per-element accounting

[App D, "Effective memory overhead"]

For a group of G=128 elements:

| Tensor | Type | Count per group | Bits per element contribution |
|--------|------|----------------|------------------------------|
| K_q (quantized values) | 2-bit | 128 | 2.0 |
| z (zeropoint) | FP16 (16-bit) | 1 (per channel) | 16/128 = 0.125 |
| s1 (scale, absorbs S_c) | FP8 (8-bit) | 1 (per channel) | 8/128 = 0.0625 |
| s2 (second scale, S_r) | FP8 (8-bit) | 1 (per token) | 8/128 = 0.0625 |
| **Total** | | | **2.25 bits/element** |

The paper reports **2.3 bits/element** effective (rounding up, accounting for
the three-region layout where some tokens stay in FP16).

## Comparison to baselines

| Method | K/V bits | Effective bits/elem | Uniform precision? |
|--------|---------|---------------------|--------------------|
| KIVI | 2/2 | 2.3 | Yes |
| QuaRot | 2/2 | 2.3 | Yes |
| KVarN | 2/2 | 2.3 (2.25 raw) | Yes |
| KVQuant | 2/2 | 2.4 | No (1% channels in FP16) |
| Kitty | 2/2 | 2.4 | No (top 12% channels at 4-bit) |
| PolarQuant | 4/2 | 3.3 | Yes |
| TurboQuant | 3/3 | 4.6 | Yes |

## K vs V asymmetry

[Sec 2.1] Following KIVI:
- **K is quantized per channel** (along the token dimension). K channels have
  consistent statistics across tokens, so per-channel scaling works well.
- **V is quantized per token** (along the channel dimension). V tokens have
  consistent statistics across channels, so per-token scaling works well.

KVarN applies the same Hadamard + VarN + RTN pipeline to both, but the RTN
grouping axis differs:
- K: group along tokens, scale along channels. `s1` is per-channel, `s2` is
  per-token.
- V: group along channels, scale along tokens. `s1` is per-token, `s2` is
  per-channel.

## Dequantization formula

```
K_dq = (K_q + z) * s1 * s2
```

Where:
- `K_q` is the 2-bit quantized value (unsigned integer)
- `z` is the FP16 zeropoint
- `s1` is the FP8 primary scale (absorbs VarN column scale + RTN scale)
- `s2` is the FP8 secondary scale (VarN row scale)

At dequant time, `s2` is fused into the dequant kernel so no extra HBM
round-trip is needed (see [07-dequantization.md](07-dequantization.md)).