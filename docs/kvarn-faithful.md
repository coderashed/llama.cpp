# Faithful KVarN: per-channel K + VarN (experimental)

The default `q2_kvarn` cache type (see [kvarn.md](kvarn.md)) quantizes K
per-token. The paper the type is based on (arXiv:2606.03458) instead quantizes
K **per-channel** and adds **VarN**, a dual-axis variance normalization (SINQ
log-domain std-dev scaling). This repository contains a faithful implementation
of that scheme for the K cache. It is experimental, off by default, and enabled
entirely through environment variables on top of `-ctk q2_kvarn`.

Use it when output quality at 2-bit K matters more than prefill speed.

## Quality

40-chunk wikitext-2 perplexity (llama-2-7b Q4_0, `-c 512 -fa on -ctk q2_kvarn
-ctv q4_0`):

| config                            | PPL  |
|-----------------------------------|------|
| f16 KV reference                  | 5.61 |
| per-token q2_kvarn (default)      | 6.68 |
| faithful (per-channel K + VarN)   | 6.09 - 6.11 |

KL-divergence vs f16 (16 chunks, only the KV type varies):

| metric            | per-token | faithful |
|-------------------|-----------|----------|
| mean KLD          | 0.1730    | 0.0731   |
| max KLD           | 18.87     | 3.80     |
| same top-1 token  | 81.8%     | 87.8%    |

## Environment variables

All kvarn cache types (`q2_kvarn`, `q3_kvarn`, `q4_kvarn`) default to the
per-token path with no environment variables; the gates below opt any of them
into the faithful path.

All gates parse their value: **unset or `0` means off, any nonzero value means
on** (same convention as `LLAMA_GRAPH_REUSE_DISABLE`).

| variable                       | effect |
|--------------------------------|--------|
| `LLAMA_KVARN_PERCHANNEL_READ`  | Master gate. Allocates the per-channel region tensors (`k_body`, `k_recent`, VarN scales), writes complete 128-token groups per-channel on the fly, and reads K back by reconstructing it from the per-channel store instead of the per-token cache. Also disables graph reuse (the region offsets are baked at graph-build time). |
| `LLAMA_KVARN_VARN`             | Runs VarN when a group is quantized and applies the stored row/column scales on read. Only meaningful together with `PERCHANNEL_READ`; on its own it does nothing. Without it, per-channel K quality is poor - always set both. |
| `LLAMA_KVARN_FUSED_FA`         | Uses the fused per-channel attention op (`GGML_OP_KVARN_FA`) that reads the 2-bit per-channel blocks directly inside the kernel, instead of reconstructing K to F16 first. Requires both gates above; engages for group-aligned prefill only and falls back to reconstruction otherwise. The current kernel is a correctness prototype and is slower than reconstruction at long-prompt prefill - leave it unset unless you are working on the kernel. |
| `LLAMA_KVARN_DEBUG`            | Traces the per-channel write path (head position, batch size, contiguity) at layer 0. |

When every variable is unset (or `0`), behavior and memory use are identical
to the default per-token path for all three types.

## Usage

Perplexity (the standard 40-chunk gate used for the numbers above):

```
LLAMA_KVARN_PERCHANNEL_READ=1 LLAMA_KVARN_VARN=1 \
  llama-perplexity -m model.gguf -f wiki.test.raw \
  -ctk q2_kvarn -ctv q4_0 -fa on -ngl 999 -c 512 -b 512 -ub 512 --chunks 40
```

Server:

```
LLAMA_KVARN_PERCHANNEL_READ=1 LLAMA_KVARN_VARN=1 \
  llama-server -m model.gguf -ctk q2_kvarn -ctv q4_0 -fa on -ngl 999 --parallel 1
```

## Faithful V and bit widths (3-bit / 4-bit)

With both gates set, `-ctv q2_kvarn` additionally routes V through the same
three-region scheme (VarN-normalized group tiles, per-token quantization,
reconstruction on read). The 3-bit and 4-bit siblings `q3_kvarn` and `q4_kvarn`
follow the same rule as `q2_kvarn`: per-token by default (dedicated
flash-attention kernels, full speed - see [kvarn.md](kvarn.md)), faithful when
the gates are set. The fused attention op stays 2-bit only; faithful 3/4-bit K
always uses reconstruction.

KL divergence vs f16 (Qwen3.6-35B-A3B UD-Q6_K, wikitext-2, 16 chunks), faithful
gates on unless marked per-token:

| `-ctk` / `-ctv`       | mean KLD | 99.9% KLD | PPL ratio |
|-----------------------|----------|-----------|-----------|
| q2_kvarn / q2_kvarn   | 0.0575   | 1.04      | 1.047     |
| q2_kvarn / q4_kvarn   | 0.0259   | 0.76      | 1.026     |
| q3_kvarn / q3_kvarn   | 0.0188   | 0.28      | 1.010     |
| q3/q3 PER-TOKEN       | 0.0202   | 0.48      | 1.010     |
| q4_kvarn / q4_kvarn   | 0.0092   | 0.23      | 1.006     |

At 3-bit the faithful machinery buys almost nothing over per-token (0.0188 vs
0.0202 mean, identical PPL ratio) while costing roughly 2x prefill and 4x
decode; the per-token default is the recommended configuration. The faithful
path remains the quality reference and the only 2-bit-K option that beats
per-token materially.

## Performance and limitations

- **Prefill is roughly 2x slower** than the per-token path on long prompts.
  Reconstruction materializes the full F16 K (and casts V to F16) every
  forward; the result cannot be cached without giving back the 2-bit memory
  win. A tiled fused kernel that removes this cost is in progress.
- **Graph reuse is disabled** while the path is active, which adds per-step
  graph-build overhead to decode.
- **Single stream only**: the read path asserts one sequence stream, so the
  server must run with `--parallel 1`.
- **V options**: plain `-ctv q4_0` (no faithful machinery) or a kvarn type
  for the faithful V path; at 4 bits the two measure the same, so faithful V
  earns its cost only at 3 bits and below.
- **head_dim up to 256** is supported on the GPU (CUDA/HIP); this covers
  current large models.
- Large contexts with the gates on can hit a graph node-budget abort; the
  gates-off path is unaffected.
- Cache operations beyond plain generation (defrag, sequence copy/remove,
  K-shift, state save/load of the per-channel regions, speculative-decode
  rollback) are not yet supported on the per-channel layout.
