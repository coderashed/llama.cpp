# Dequantization

**Paper sections**: 4.2 (Runtime Overhead), Appendix I, Fig 11

## Formula

```
K_dq = (K_q + z) * s1 * s2
```

- `K_q`: 2-bit quantized integer (unsigned)
- `z`: FP16 zeropoint, one per channel per group of G=128
- `s1`: FP8 primary scale (absorbs VarN column scale + RTN scale)
- `s2`: FP8 secondary scale (VarN row scale), one per token per group

The multiplication by `s2` is the only addition over the standard KIVI single-scale
dequant (`K_dq = (K_q + z) * s`).

## Kernel fusion

[App I] The second scale `s2` is **fused into the dequant kernel** so that no
extra HBM round-trip is incurred. The dequant reads `K_q`, `z`, `s1`, `s2` and
writes `K_dq` in a single kernel pass. The reference implementation uses Triton
on GPU.

## Overhead measurement

[App I, Fig 11] Measured dequant time per call for a full attention layer
(16 heads, head_dim=128, group_size=128) across context lengths 4k-32k:

| Context | KIVI (ms) | KVarN (ms) | Overhead |
|---------|-----------|------------|----------|
| 4k | baseline | ~+1.4% | measurable |
| 16k | baseline | ~+0% | within noise |
| 32k | baseline | ~+0% | within noise |

**Maximum overhead: 1.4%** at short contexts, diminishing to measurement noise
at longer contexts. The relative cost of the extra multiply decreases as more
data is processed per kernel launch.

## Quantization-time overhead

[Sec 4.2, Fig 6] For Qwen3-4B generating 128 tokens:
- Token generation (fp16, vLLM): 1050 ms
- VarN normalization (8 iterations, all layers): 1.9 ms
- **Overhead: 0.18%** of generation time

For larger models, the relative overhead decreases further (generation cost
scales with model size, VarN cost scales with cache size).

## Comparison to other methods

[Sec 4.2] Prior methods with codebook lookups (TurboQuant) or mixed-precision
handling (KVQuant, Kitty) have **larger** dequantization overhead than KVarN's
single extra multiply. KVarN's dual-scale approach is the cheapest dequant path
among all 2-bit methods except plain KIVI.

## Implementation notes

1. Store `s2` interleaved with `s1` or adjacent in memory to maximize cache hits
   during dequant.
2. The FP8 scales can be E4M3 format (7-bit mantissa) — the paper does not
   specify, but FP8 in vLLM typically uses E4M3.
3. For CPU/backends without FP8 support, s2 can be stored as FP16 at the cost
   of 0.0625 extra bits/element (total 2.3125 vs 2.25).
4. The dequant kernel should handle the K and V asymmetrically: K groups along
   tokens (s1 per-channel, s2 per-token), V groups along channels (s1 per-token,
   s2 per-channel).