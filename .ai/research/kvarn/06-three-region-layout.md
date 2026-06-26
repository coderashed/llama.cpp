# Three-Region Cache Layout

**Paper section**: Appendix D ("Quantization")

## Layout

The KV-cache is divided into three regions along the token dimension:

```
  [ S sink tokens | G-group quantized body ... | R recent tokens ]
  |     FP16       |     2-bit quantized        |     FP16        |
  |  (unquantized)  |  (KVarN: Hadamard+VarN+RTN) |  (unquantized)  |
```

| Region | Symbol | Precision | Purpose |
|--------|--------|-----------|---------|
| Sink | S | FP16 | Preserve attention-sink behavior (first few tokens get high attention regardless of content) |
| Body | (quantized) | 2-bit + scales | The bulk of the cache, quantized in groups of G |
| Recent | R | FP16 | Most recently generated tokens that haven't filled a complete group yet |

## Default parameters

[App D]

| Parameter | Default | IFEval override |
|-----------|---------|-----------------|
| G (group size) | 128 | 128 |
| S (sink tokens) | 128 | 32 |
| R (recent tokens) | 128 | 128 |

The paper uses these "classical" values throughout. S=32 for IFEval because
instruction-following tasks are shorter and need fewer sink tokens.

## Group formation

- Tokens in the body region are processed in chunks of G=128.
- Each chunk becomes one quantization tile of shape `(head_dim x G)`, e.g.
  `128 x 128` for Llama-3.1-8B.
- VarN (Algorithm 1) operates on this tile.
- The tile is then RTN-quantized, producing:
  - One 2-bit value per element
  - One FP16 zeropoint per channel per group
  - One FP8 scale s1 per channel per group
  - One FP8 scale s2 per token per group

## When does a token move from "recent" to "body"?

The recent region `R` holds tokens that have been generated but not yet
quantized. Once R tokens fill a complete group of G, that group is quantized
and moved to the body region. In practice, this happens every G generated tokens.

During prefill (batch prompt processing), all prompt tokens can be quantized at
once (they are available in a single forward pass). During decoding, groups
form incrementally.

## Mixed-precision methods (contrast)

Methods like KVQuant, Kitty, and PolarQuant use mixed precision **within** the
body region (some channels at higher bit-width). KVarN uses uniform 2-bit
throughout the body — all elements in K (or V) share the same precision. This
simplifies the kernel (no branching on precision) and the storage format.

## Implementation relevance

1. The cache needs three bookkeeping regions, not a single contiguous buffer.
2. Token positions must track which region they belong to (sink/body/recent).
3. When a token transitions from recent to body, it triggers a quantization
   operation on the full group of G tokens.
4. The sink and recent regions are stored as FP16, so attention over those
   tokens is computed at full precision (no dequant needed).