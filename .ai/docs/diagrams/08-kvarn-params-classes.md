---
title: KVarN Context Params -- Struct Fields, Defaults & Validation
---

```mermaid
classDiagram

    class llama_context_params {
        <<POD struct (include/llama.h:336)>>
        +uint32_t n_ctx
        +uint32_t n_batch
        +uint32_t n_ubatch
        +uint32_t n_seq_max
        +enum ggml_type type_k
        +enum ggml_type type_v
        +uint32_t kvarn_group_size       <<NEW>>  // G = 128 tokens per tile
        +uint32_t kvarn_sink_tokens      <<NEW>>  // S = 128 sink tokens
        +uint32_t kvarn_recent_tokens    <<NEW>>  // R = 128 recent tokens
        +uint32_t kvarn_varn_iterations  <<NEW>>  // K = 8 VarN iterations
        // ... existing fields follow
    }

    class llama_context_default_params {
        <<function (src/llama-context.cpp:3447)>>
        +returns llama_context_params with:
        +type_k = GGML_TYPE_F16
        +type_v = GGML_TYPE_F16
        +kvarn_group_size = 128
        +kvarn_sink_tokens = 128
        +kvarn_recent_tokens = 128
        +kvarn_varn_iterations = 8
    }

    class llama_cparams {
        <<internal struct (src/llama-cparams.h:10)>>
        +uint32_t n_ctx
        +uint32_t n_batch
        +uint32_t n_ubatch
        +uint32_t n_seq_max
        +enum ggml_type type_k
        +enum ggml_type type_v
        +uint32_t kvarn_group_size       <<NEW>>
        +uint32_t kvarn_sink_tokens      <<NEW>>
        +uint32_t kvarn_recent_tokens    <<NEW>>
        +uint32_t kvarn_varn_iterations  <<NEW>>
        // ... existing fields follow
    }

    class llama_kv_cache {
        <<class (src/llama-kv-cache.h:20)>>
        +ggml_type type_k
        +ggml_type type_v
        +uint32_t kvarn_group_size       <<NEW>>
        +uint32_t kvarn_sink_tokens      <<NEW>>
        +uint32_t kvarn_recent_tokens    <<NEW>>
        +uint32_t kvarn_varn_iterations  <<NEW>>
        +bool attn_rot_k
        +bool attn_rot_v
        +void cpy_k(ctx, k_cur, k_idxs, il, sinfo)
        +void cpy_v(ctx, v_cur, v_idxs, il, sinfo)
    }

    class VarNTileProcessor {
        <<internal (src/llama-kv-cache.cpp)>>
        +int G                           // kvarn_group_size
        +int head_dim
        +int K                           // kvarn_varn_iterations
        +float* tile_buf
        +float* S_c_buf
        +float* S_r_buf
        +void process_tile(data, n_tokens, head_dim)
        +void process_partial(data, n_remaining)
    }

    class ThreeRegionLayout {
        <<concept (.ai/research/kvarn/06-three-region-layout.md)>>
        +int sink_count                  // kvarn_sink_tokens
        +int recent_count                // kvarn_recent_tokens
        +int middle_count                // ctx - sink - recent
        +bool is_sink(pos)               // pos < sink_count
        +bool is_recent(pos, ctx_pos)    // ctx_pos - pos < recent_count
        +bool is_middle(pos, ctx_pos)    // otherwise
    }

    llama_context_params --> llama_context_default_params : defaults defined in
    llama_context_params --> llama_cparams : copied during init (llama-context.cpp)
    llama_cparams --> llama_kv_cache : passed at construction
    llama_kv_cache --> VarNTileProcessor : G and K used by
    llama_kv_cache --> ThreeRegionLayout : S and R used by

    note for llama_context_params "New fields appended at end of struct\n(ABI-compatible: old callers zero-initialize)\nOnly read when type_k or type_v == GGML_TYPE_Q2_KVARN"
    note for llama_cparams "Internal copy; validated during context_init()\nbefore cache construction"
    note for VarNTileProcessor "G = group_size (tile height)\nK = varn_iterations\nS_c/S_r buffers sized by G and head_dim"
    note for ThreeRegionLayout "Three-region KV cache layout:\n  [0..S-1] = sink (always preserved)\n  [S..ctx-S-R] = middle (evictable)\n  [ctx-R..ctx-1] = recent (always preserved)"
```

### Default Values

| Field | Default | Rationale |
|-------|---------|-----------|
| `kvarn_group_size` | 128 | Tile size for VarN normalization; matches block size of Q2_KVARN |
| `kvarn_sink_tokens` | 128 | Sink region size; preserves initial tokens from eviction |
| `kvarn_recent_tokens` | 128 | Recent region size; preserves last tokens from eviction |
| `kvarn_varn_iterations` | 8 | VarN convergence iterations; 8 is sufficient for ~N(0,1) |

### Field Semantics

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `kvarn_group_size` | `uint32_t` | [1, 1024], must divide `n_embd_head_k` | Number of tokens per VarN tile |
| `kvarn_sink_tokens` | `uint32_t` | [0, n_ctx) | Number of tokens in the sink (always-preserved) region |
| `kvarn_recent_tokens` | `uint32_t` | [0, n_ctx) | Number of tokens in the recent (always-preserved) region |
| `kvarn_varn_iterations` | `uint32_t` | [1, 100] | Number of variance normalization iterations |
