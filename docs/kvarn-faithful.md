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

All gates parse their value: **unset or `0` means off, any nonzero value means
on** (same convention as `LLAMA_GRAPH_REUSE_DISABLE`).

| variable                       | effect |
|--------------------------------|--------|
| `LLAMA_KVARN_PERCHANNEL_READ`  | Master gate. Allocates the per-channel region tensors (`k_body`, `k_recent`, VarN scales), writes complete 128-token groups per-channel on the fly, and reads K back by reconstructing it from the per-channel store instead of the per-token cache. Also disables graph reuse (the region offsets are baked at graph-build time). |
| `LLAMA_KVARN_VARN`             | Runs VarN when a group is quantized and applies the stored row/column scales on read. Only meaningful together with `PERCHANNEL_READ`; on its own it does nothing. Without it, per-channel K quality is poor - always set both. |
| `LLAMA_KVARN_FUSED_FA`         | Uses the fused per-channel attention op (`GGML_OP_KVARN_FA`) that reads the 2-bit per-channel blocks directly inside the kernel, instead of reconstructing K to F16 first. Requires both gates above; engages for group-aligned prefill only and falls back to reconstruction otherwise. The current kernel is a correctness prototype and is slower than reconstruction at long-prompt prefill - leave it unset unless you are working on the kernel. |
| `LLAMA_KVARN_DEBUG`            | Traces the per-channel write path (head position, batch size, contiguity) at layer 0. |

When every variable is unset (or `0`), behavior and memory use are identical to
the default per-token `q2_kvarn` path.

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

## Performance and limitations

- **Prefill is roughly 2x slower** than the per-token path on long prompts.
  Reconstruction materializes the full F16 K (and casts V to F16) every
  forward; the result cannot be cached without giving back the 2-bit memory
  win. A tiled fused kernel that removes this cost is in progress.
- **Graph reuse is disabled** while the path is active, which adds per-step
  graph-build overhead to decode.
- **Single stream only**: the read path asserts one sequence stream, so the
  server must run with `--parallel 1`.
- **K-side only**: V stays whatever `-ctv` selects (`q4_0` recommended).
- **head_dim up to 256** is supported on the GPU (CUDA/HIP); this covers
  current large models.
- Large contexts with the gates on can hit a graph node-budget abort; the
  gates-off path is unaffected.
- Cache operations beyond plain generation (defrag, sequence copy/remove,
  K-shift, state save/load of the per-channel regions, speculative-decode
  rollback) are not yet supported on the per-channel layout.
