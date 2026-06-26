# Limitations

**Paper section**: Appendix E

## 1. Not applicable to state-space models (SSMs)

KVarN quantizes the KV-cache, which is specific to transformer attention
architectures. State-space models (e.g. Mamba, S4) do not have a KV-cache —
they maintain a recurrent state instead. KVarN cannot be applied to such
architectures.

In the llama.cpp context: items 08 (Memory Management / Recurrent) and the
recurrent memory backends (`llama_memory_recurrent`) are not compatible with
KVarN. It applies only to attention-based KV caches.

## 2. Multi-Head Latent Attention (MLA) uncertainty

Some recent models (e.g. DeepSeek-V2/V3) use Multi-Head Latent Attention, which
compresses the KV-cache at training time via a low-rank projection. The paper
notes that "it is unclear how such attention mechanisms affect quantization
quality" — KVarN has not been tested on MLA-based models.

In the llama.cpp context: the DSA KV-cache variant (`llama_kv_cache_dsa`)
handles DeepSeek V3.2-style MLA. Applying KVarN on top of the already-compressed
MLA latent would require investigation.

## 3. 2-bit KV-cache not supported in serving frameworks

At the time of writing, the paper notes that "currently such frameworks do not
support 2-bit KV-Caches" — the vLLM implementation is custom. Standard serving
frameworks (vLLM upstream, TGI, etc.) need kernel support for 2-bit KV-cache
quantization/dequantization.

In the llama.cpp context: this is the main implementation barrier. llama.cpp's
KV-cache quantization currently supports Q4_0, Q8_0, and Q8_1 for K-cache (see
[07-kv-cache.md](../docs/07-kv-cache.md)). Adding 2-bit support (Q2_K or a
custom KVarN format) would require:
- New ggml type or custom storage format for the dual-scale 2-bit layout
- Dequantization kernels for the `(K_q + z) * s1 * s2` formula
- Hadamard rotation integrated into the attention graph (online, per-head)
- VarN normalization as an online operation during cache write
- Three-region layout bookkeeping in `llama_kv_cache`

## 4. VarN requires a full group before normalizing

Variance normalization operates on a tile of G=128 tokens. During autoregressive
decoding, tokens arrive one at a time. The normalization can only be applied once
a full group has been accumulated. This means the "recent" region (R=128 tokens
in FP16) must be at least one group in size. For very short sequences (< G tokens
total), KVarN degenerates to plain FP16 (no quantization happens).

## 5. Overhead scales with head count and layer count

The VarN cost is per-attention-layer (every layer's K and V tiles are
normalized independently). For models with many layers and heads, the 0.18%
relative overhead may increase if the generation is very fast (e.g. small model,
short sequences). The paper notes that "for larger models the relative overhead
decreases."

## 6. FP8 scale precision assumption

The storage format assumes FP8 (E4M3) scales. On hardware without native FP8
support, scales must be stored as FP16, increasing the effective bits/element
from 2.25 to ~2.31. This is still competitive but reduces the compression ratio
advantage slightly.