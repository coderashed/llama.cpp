# Problem & Motivation

**Paper sections**: 1 (Introduction), 3.2 (Error Accumulation), 2.4 (Key Ideas)

## The memory bottleneck in test-time scaling

Test-time scaling (long chain-of-thought decoding) pushes the KV-cache to be
the dominant memory consumer. KV-cache quantization (reducing K/V from 16-bit
to 2-4 bit) is the primary mitigation, but existing methods are designed and
evaluated for the prefill scenario (quantize a fixed long context in parallel).

## Error accumulation under autoregressive decoding

[Sec 3.2, Fig 4] During autoregressive decoding, the KV-cache is quantized
incrementally as tokens are generated. This creates a compounding error chain:

1. Transformer block `B_l` computes attention with a **quantized** KV-cache.
2. The attention output is slightly wrong, which makes the K/V matrices that
   block `B_l` outputs to `B_{l+1}` slightly wrong (even before quantization).
3. `B_{l+1}` quantizes its already-corrupted K/V, adding more error.
4. This propagates across layers and eventually across timesteps.

The longer the generation, the worse the accumulation. Standard prefill-style
evaluation (quantize once, then run full-precision attention) does not capture
this effect at all.

## Pseudo-decode evaluation

[Sec 3.2] The authors propose a "pseudo-decode" protocol to measure accumulation:
- Divide the prefill sequence into blocks of size `b`.
- After every `b` tokens pass through the model, quantize the KV-cache.
- All subsequent tokens compute latents with the **quantized** cache.
- This mimics real decoding without actually generating tokens.

See [08-pseudo-decode-eval.md](08-pseudo-decode-eval.md) for full detail.

## Outliers dominate end-to-end quality

[Sec 2.4, Fig 3] The central empirical claim:

- The **top 5%** largest quantization errors cause the majority of end-to-end
  quality degradation (measured by KL-divergence of output logits).
- Paradoxically, these top-5% errors contribute a **minority** of total MSE
  (Fig 9, Appendix C). MSE is therefore a misleading proxy for end-to-end quality.
- Fixing the worst 5% of errors improves KL-divergence more than fixing the
  remaining 95%.

**Implication for implementation**: A KV quantizer must prioritize suppressing
worst-case (tail) errors over minimizing average MSE. This is the opposite of
what standard RTN quantization optimizes for.

## Token magnitude is the outlier driver

[Sec 3.1, Fig 1a] Among the top-k% largest errors, the fraction attributable to
magnitude (scale) error vs. directional error is overwhelmingly magnitude-driven
(see [02-error-decomposition.md](02-error-decomposition.md) for the decomposition).

Standard quantization methods (KIVI, Hadamard-only) fail to preserve per-token
norms. When the token scale is wrong by a multiplicative factor, the error
compounds exponentially across decode steps (repeated multiplication by the
incorrect scale). KVarN's dual-scale variance normalization directly fixes this
by storing an explicit per-token (or per-row) high-precision scale.