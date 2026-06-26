# Pseudo-Decode Evaluation

**Paper section**: 3.2, Fig 4

## Motivation

Standard KV-cache quantization evaluation uses a **prefill** setting: quantize
the entire KV-cache at once, then run attention with the quantized cache. This
does not model the error accumulation that happens during autoregressive
decoding, where:
1. Block `B_l` computes attention with a quantized cache -> output is wrong.
2. `B_l`'s wrong output feeds `B_{l+1}`, whose K/V are already corrupted.
3. `B_{l+1}` quantizes its corrupted K/V, adding more error.
4. Errors compound across layers and across timesteps.

## Protocol

The "pseudo-decode" protocol:

```
Given a prefill sequence of N tokens:
1. Process the first b tokens through the full model (full precision KV-cache).
2. Quantize the KV-cache (all layers, all heads).
3. Process the next b tokens with the QUANTIZED KV-cache.
4. Quantize the newly produced KV entries (add to cache).
5. Repeat steps 3-4 until all N tokens are processed.
6. Measure attention output reconstruction error vs full-precision baseline.
```

This mimics real decoding without actually sampling tokens — the input tokens
are fixed (from a prefill sequence), but the KV-cache is incrementally quantized
just as it would be during generation.

## Block size

The block size `b` controls the granularity of quantization. The paper uses
`b = 128` (matching the group size G). Every 128 tokens, the cache is quantized.

## What it measures

[Fig 5] The pseudo-decode setting measures the **accumulated** reconstruction
error of attention outputs across all layers, as a function of context length.
Key findings:
- KIVI's error grows with context length (accumulation).
- KVarN's error grows much more slowly.
- The gap between KVarN and KIVI **widens** as context gets longer (Fig 5b/c),
  confirming that KVarN specifically suppresses accumulation.

## Static vs. accumulated comparison

| Setting | What it models | Error behavior |
|---------|---------------|----------------|
| Static (prefill) | Quantize once, use full-precision attention after | No accumulation; underestimates real-world error |
| Accumulated (pseudo-decode) | Incremental quantization mimicking decode | Accumulation across blocks; matches real decoding |

## Why NiaH is too easy

[App G] The standard Needle-in-a-Haystack benchmark under static (prefill)
quantization is too easy — it does not show error accumulation. Under the
accumulated pseudo-decode setting, the differences between methods become
much more visible. The authors recommend line-retrieval as a more informative
benchmark (see [10-results-summary.md](10-results-summary.md)).

## Implementation recipe

To implement pseudo-decode evaluation:
1. Load model, process prompt tokens normally.
2. Hook into the KV-cache write path.
3. After every `b` tokens, call the quantize function on the full cache.
4. For subsequent tokens, the attention reads from the quantized cache.
5. Record attention output norms at each layer for error measurement.
6. Compare to a full-precision baseline run.