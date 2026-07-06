# Q2_KVARN: 2-bit variance-normalized KV-cache quantization

`q2_kvarn` is a 2-bit KV-cache quantization type. It stores each K and/or V
cache entry in ~2.375 bits per element instead of 16, cutting KV-cache memory by
about 6.7x versus `f16` while keeping the attention math working through the
flash-attention path.

It is selected like any other KV-cache type, via `--cache-type-k` / `-ctk` and
`--cache-type-v` / `-ctv`.

> Status: this is a memory-reduction feature. It is not a speedup (see
> Performance below), and the output-quality impact of quantizing the V cache
> has not yet been validated. Treat it as experimental.

## Quick start

```
# K in 2-bit kvarn, V in 4-bit q4_0 (flash attention is required)
llama-cli -m model.gguf -fa on -ctk q2_kvarn -ctv q4_0 -c 16384 -ngl 999

# Both K and V in 2-bit kvarn (maximum memory savings)
llama-cli -m model.gguf -fa on -ctk q2_kvarn -ctv q2_kvarn -c 16384 -ngl 999
```

`-fa on` is mandatory: `q2_kvarn` only has flash-attention kernels.

## Memory savings

Measured KV-cache allocation reported by `llama_kv_cache: size = ...` for
llama-2-7b (32 layers, n_ctx = 8192):

| -ctk / -ctv        | K size   | V size   | total    |
|--------------------|----------|----------|----------|
| f16 / f16          | 2048 MiB | 2048 MiB | 4096 MiB |
| q4_0 / q4_0        |  576 MiB |  576 MiB | 1152 MiB |
| q2_kvarn / q4_0    |  304 MiB |  576 MiB |  880 MiB |
| q2_kvarn / q2_kvarn|  304 MiB |  304 MiB |  608 MiB |

Each `q2_kvarn` side is 304 MiB here, exactly 2.375 bits/element: a 128-element
block stores 32 bytes of 2-bit codes plus three fp16 scales (6 bytes) = 38 bytes
= 2.375 bits/element. That is 6.74x smaller than `f16` and 1.89x smaller than
`q4_0`. The reduction is identical for K and V; `q2_kvarn` is applied
symmetrically to both.

## Requirements

- Flash attention enabled: `-fa on`.
- Head dimension of 64, 128, or 256 (these are the compiled kernel
  instances). The Hadamard rotation also requires head_dim % 64 == 0.
- A CUDA or HIP (ROCm) GPU backend. Tested on AMD gfx906 (MI50) and gfx1100
  (RX 7900 XTX), and on NVIDIA via the shared CUDA path.

## Supported K/V combinations

`q2_kvarn` for K can be paired with several V types; `f16` K can be paired with
`q2_kvarn` V:

| -ctk      | -ctv      |
|-----------|-----------|
| q2_kvarn  | q8_0      |
| q2_kvarn  | q4_0      |
| q2_kvarn  | f16       |
| q2_kvarn  | q2_kvarn  |
| f16       | q2_kvarn  |
| q3_kvarn  | q3_kvarn  |
| q4_kvarn  | q4_kvarn  |

Other combinations fall back to the generic dispatch and may abort; stick to the
pairs above.

## 3-bit and 4-bit siblings (q3_kvarn, q4_kvarn)

`q3_kvarn` (~3.4 bits/element) and `q4_kvarn` (~4.4 bits/element) are sibling
types with the same block layout (128-element blocks, d/s1/s2 scales) at higher
code widths. Like `q2_kvarn` they are per-token by default, read directly by
the flash-attention kernels at full speed, and require `-fa on`. Same-type K/V
pairing only (see the table above).

KL divergence vs an f16 cache (Qwen3.6-35B-A3B UD-Q6_K, wikitext-2, 16 chunks):

| -ctk / -ctv           | mean KLD | 99.9% KLD | PPL ratio |
|-----------------------|----------|-----------|-----------|
| q2_kvarn / q2_kvarn   | 0.0647   | 1.21      |           |
| q3_kvarn / q3_kvarn   | 0.0202   | 0.48      | 1.010     |

`q3_kvarn / q3_kvarn` is the recommended configuration: under 1% PPL cost and
~4.7x smaller than f16, with prefill and decode at f16-cache speed (measured
on 2x MI50: identical prefill throughput to f16 and q2_kvarn caches).

## How it works (brief)

Per 128-element block the cache write does:

1. A Hadamard (fast Walsh-Hadamard) rotation of K and V, which spreads outliers
   across channels. Q is rotated to match K so the rotation cancels inside the
   Q*K dot; the V rotation is undone on the attention output.
2. Asymmetric min/max 2-bit quantization: scale `s1 = (max-min)/3`, zeropoint
   `d`, codes in 0..3.
3. A per-block norm correction `s2 = ||x|| / ||dequant(x)||` so the block's L2
   norm (variance/energy) is preserved after quantization.

Dequantization is `(code + d) * s1 * s2`.

On read, K is consumed as an integer dot product (DP4A where available) and V is
dequantized to float and accumulated with the softmax weights.

The attention rotation can be disabled for debugging with the environment
variable `LLAMA_ATTN_ROT_DISABLE=1`.

## Performance

`q2_kvarn` trades compute for memory; it is not expected to speed up decoding.
Single-stream `tg` benchmarks on 2x MI50 (gfx906) and 2x RX 7900 XTX (gfx1100)
across two models and three context depths showed end-to-end changes within
roughly +/-3% (slightly positive on gfx906 at long context, slightly negative on
gfx1100). Decode on these models is bound by reading the model weights, so
shrinking the KV cache does not move throughput much. The value of this type is
fitting longer contexts or larger batches into the same VRAM, not raw speed.

## Limitations and caveats

- Quality of V quantization is unverified. The V cache is the numerically
  sensitive side (it is accumulated after softmax). Quantizing V (`-ctv
  q2_kvarn`) may degrade output more than quantizing K; validate on your
  workload before relying on it.
- Simplified relative to the source method. This implementation applies a
  Hadamard rotation plus per-block min/max and a per-block norm correction to
  both K and V. The paper's full scheme - per-channel K quantization plus VarN
  dual-axis variance normalization (SINQ log-domain std-dev scaling) - is
  available as an experimental, env-gated path; see
  [kvarn-faithful.md](kvarn-faithful.md).
- Flash-attention only. There is no non-FA (mmvq/mmq) path for `q2_kvarn`.
