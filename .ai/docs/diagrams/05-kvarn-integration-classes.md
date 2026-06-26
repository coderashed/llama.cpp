---
title: KVarN Pipeline Integration -- Class / Struct Diagram
---

```mermaid
classDiagram

    class llama_kv_cache {
        <<class (src/llama-kv-cache.h:20)>>
        +ggml_type type_k
        +ggml_type type_v
        +bool attn_rot_k
        +bool attn_rot_v
        +ggml_tensor* cpy_k(ctx, k_cur, k_idxs, il, sinfo)
        +ggml_tensor* cpy_v(ctx, v_cur, v_idxs, il, sinfo)
        +slot_info_vec_t prepare(ubatches)
        +void apply_ubatch(sinfo, ubatch)
    }

    class llama_kv_cache_context {
        <<class (src/llama-kv-cache.h:320)>>
        +ggml_tensor* cpy_k(ctx, k_cur, k_idxs, il)
        +ggml_tensor* cpy_v(ctx, v_cur, v_idxs, il)
        +void set_input_k_rot(dst)
        +void set_input_v_rot(dst)
    }

    class kvarn_variance_normalize {
        <<function (src/llama-kvarn.h:15)>>
        +float* T           // [R][C] tile, modified in-place
        +int R              // rows (G=128)
        +int C              // columns (head_dim)
        +int K              // iterations (default 12)
        +float c_min        // clamp lower bound (-5.0)
        +float c_max        // clamp upper bound (+5.0)
        +float* S_c         // [C] column scale output
        +float* S_r         // [R] row scale output
    }

    class quantize_row_q2_kvarn_varn {
        <<NEW function (ggml/src/ggml-quants.c)>>
        +const float* x          // FP32 normalized tile, [k] elements
        +block_q2_kvarn* y       // quantized output
        +int64_t k               // total elements (must be multiple of 128)
        +const float* S_c        // [k/128][C] pre-computed column scales (reserved)
        +const float* S_r        // [k/128][R] pre-computed row scales (reserved)
        // Per block:
        //   Delegates to quantize_q2_kvarn_block() for min/max/pack
        //   s2 = L2 ratio (same as ref) — S_c/S_r absorption happens at caller
    }

    class block_q2_kvarn {
        <<struct (ggml/src/ggml-common.h:185)>>
        +uint8_t  qs[32]    // 128 x 2-bit packed
        +ggml_half d         // FP16 zeropoint (min)
        +ggml_half s1        // FP16 primary scale = (max-min)/3 * S_c
        +ggml_half s2        // FP16 secondary scale = S_r (VarN row scale)
    }

    class VarNTileProcessor {
        <<NEW internal (src/llama-kv-cache.cpp)>>
        +int G                           // tile size (128 tokens)
        +int head_dim                    // columns per head
        +float* tile_buf                 // [G * head_dim] scratch buffer
        +float* S_c_buf                  // [head_dim] column scale scratch
        +float* S_r_buf                  // [G] row scale scratch
        +void process_tile(float* data, int n_tokens, int head_dim)
        +void process_partial(float* data, int n_remaining)
    }

    class HadamardRotation {
        <<existing (src/llama-kv-cache.cpp:60)>>
        +ggml_tensor* ggml_mul_mat_aux(ctx, cur, rot)
        +void set_input_k_rot(dst)
        +void set_input_v_rot(dst)
    }

    class ggml_set_rows {
        <<existing graph op>>
        +quantizes on write when src/dst types differ
        +handles type conversion internally
    }

    llama_kv_cache --> HadamardRotation : uses attn_rot_k/v
    llama_kv_cache --> ggml_set_rows : cpy_k/cpy_v delegate to
    llama_kv_cache --> VarNTileProcessor : NEW: pre-processes F32 data before set_rows
    VarNTileProcessor --> kvarn_variance_normalize : calls per tile
    VarNTileProcessor --> quantize_row_q2_kvarn_varn : calls per tile with S_c, S_r
    quantize_row_q2_kvarn_varn --> block_q2_kvarn : writes with absorbed scales
    llama_kv_cache_context --> llama_kv_cache : delegates cpy_k/cpy_v

    note for VarNTileProcessor "New internal helper in llama-kv-cache.cpp\nRuns on CPU after Hadamard rotation,\nbefore ggml_set_rows quantization\n\nTile layout:\n  T[G=128][head_dim] = tokens[0..127] x channels[0..head_dim-1]\n  S_c[head_dim] = column scales\n  S_r[G] = row scales (per token)"
    note for quantize_row_q2_kvarn_varn "New function variant that accepts\npre-computed S_c and S_r vectors.\nDelegates to quantize_q2_kvarn_block() for core quantization.\nS_c/S_r are reserved for future absorption.\n\nSignature:\n  void quantize_row_q2_kvarn_varn(\n    const float* x,\n    block_q2_kvarn* y,\n    int64_t k,\n    const float* S_c,\n    const float* S_r)"
    note for block_q2_kvarn "Current implementation:\n  s1 = (max-min)/3 (no S_c absorption)\n  s2 = L2 ratio (no S_r absorption)\nDequant: val = (qval + d) * s1 * s2"
```
