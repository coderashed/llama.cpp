---
title: KVarN Pipeline Integration -- File Ownership & Data Flow
---

```mermaid
graph TD
    subgraph "src/ (llama.cpp application layer)"
        KV_CACHE_H["llama-kv-cache.h<br/>llama_kv_cache class<br/>cpy_k() / cpy_v() API"]
        KV_CACHE_CPP["llama-kv-cache.cpp<br/>- cpy_k() implementation<br/>- VarNTileProcessor (NEW)<br/>- Hadamard rotation setup<br/>- build_rope_shift()"]
        KVARN_H["llama-kvarn.h<br/>kvarn_variance_normalize()<br/>declaration"]
        KVARN_CPP["llama-kvarn.cpp<br/>VarN algorithm implementation"]
        GRAPH_CPP["llama-graph.cpp<br/>build_attn()<br/>- applies Hadamard rotation<br/>- calls cpy_k()"]
    end

    subgraph "ggml/ (low-level tensor library)"
        GGML_COMMON_H["ggml-common.h<br/>block_q2_kvarn struct"]
        GGML_QUANTS_C["ggml-quants.c<br/>- quantize_row_q2_kvarn_ref()<br/>- quantize_row_q2_kvarn_varn() (NEW)<br/>- dequantize_row_q2_kvarn()"]
        GGML_C["ggml.c<br/>- type_traits registration<br/>- ggml_quantize_chunk() dispatch"]
    end

    subgraph "Data Flow: New K write with VarN"
        DIRECTION1["Direction: F32 K_cur -> Hadamard -> VarN -> Quantized -> Cache"]
    end

    GRAPH_CPP -- "applies Hadamard rotation" --> KV_CACHE_CPP
    KV_CACHE_CPP -- "calls per tile" --> KVARN_H
    KVARN_H --> KVARN_CPP
    KVARN_CPP -- "returns S_c, S_r" --> KV_CACHE_CPP
    KV_CACHE_CPP -- "calls with S_c, S_r" --> GGML_QUANTS_C
    GGML_QUANTS_C -- "writes" --> GGML_COMMON_H
    KV_CACHE_CPP -- "ggml_set_rows" --> GGML_C

    subgraph "Data Flow: K-shift (re-quantize)"
        DIRECTION2["Direction: Quantized -> Dequant -> Rot back -> RoPE -> Rot fwd -> Quantized"]
        KV_CACHE_CPP -- "dequant via" --> GGML_QUANTS_C
        KV_CACHE_CPP -- "re-quant via" --> GGML_QUANTS_C
        note for KV_CACHE_CPP "K-shift re-quantize uses plain<br/>quantize_row_q2_kvarn_ref(),<br/>NOT the VarN variant"
    end

    subgraph "Registration Chain"
        GGML_C -- "type_traits[Q2_KVARN]" --> GGML_QUANTS_C
        GGML_C -- "ggml_quantize_chunk dispatch" --> GGML_QUANTS_C
    end

    style KV_CACHE_CPP fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style KVARN_CPP fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style GGML_QUANTS_C fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style KVARN_H fill:#bbdefb,stroke:#1565c0
    style GGML_COMMON_H fill:#bbdefb,stroke:#1565c0
```

### Data Flow Summary

```
New K write (decode step):
  k_cur (F32) [n_embd_head, n_head, n_tokens]
    |
    | [in graph] ggml_mul_mat_aux with Hadamard matrix (build_attn)
    v
  k_rot (F32) [n_embd_head, n_head, n_tokens]
    |
    | [CPU, in cpy_k() when type_k == Q2_KVARN]
    | VarNTileProcessor:
    |   for each head h:
    |     for each tile of G=128 tokens:
    |       1. Extract T[G][head_dim]
    |       2. kvarn_variance_normalize(T, G, head_dim, ...) -> S_c, S_r
    |       3. quantize_row_q2_kvarn_varn(T, block, G*head_dim, S_c, S_r)
    v
  k_quant (block_q2_kvarn) [n_embd_gqa, n_tokens]
    |
    | [in graph] ggml_set_rows(k_cache, k_quant, k_idxs)
    v
  k_cache (block_q2_kvarn) [n_embd_gqa, kv_size, n_stream]

K-shift (context shift):
  k_cache (block_q2_kvarn)
    |
    | ggml_cast -> F32
    | ggml_mul_mat_aux (Hadamard^-1)
    | ggml_rope_ext (RoPE shift)
    | ggml_mul_mat_aux (Hadamard)
    | ggml_cpy -> re-quantize (plain quantize_row_q2_kvarn_ref)
    v
  k_cache (block_q2_kvarn, shifted)
```

### New Function Signatures

| Function | File | Purpose |
|----------|------|---------|
| `quantize_row_q2_kvarn_varn(x, y, k, S_c, S_r)` | `ggml/src/ggml-quants.c` | Quantize with external S_c/S_r; absorbs into s1/s2 |
| `VarNTileProcessor::process_tile(data, n_tokens, head_dim)` | `src/llama-kv-cache.cpp` | Orchestrates tile extraction, VarN call, and quantize |
