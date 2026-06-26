# 07 — K-shift graph structure for Q2_KVARN

## Pipeline: dequant → rotate back → RoPE → rotate forward → requant

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    llama_kv_cache::build_graph_shift                         │
│                     src/llama-kv-cache.cpp:1947                              │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  For each layer:                                                     │   │
│  │                                                                      │   │
│  │  ┌─ is_kvarn? ──── YES ──────────────────────────────────────────┐   │   │
│  │  │                                                               │   │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │   │   │
│  │  │  │ Region Sink  (F16)  [0 .. S)                            │  │   │   │
│  │  │  │   view_3d(k_sink, n_rot, n_head_kv, S*n_stream)         │  │   │   │
│  │  │  │   build_rope_shift(k_sink_view)  ──► F16 path (no deq)  │  │   │   │
│  │  │  └─────────────────────────────────────────────────────────┘  │   │   │
│  │  │                                                               │   │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │   │   │
│  │  │  │ Region Body  (Q2_KVARN)  [S .. S+B)                     │  │   │   │
│  │  │  │   view_3d(k_body, n_rot, n_head_kv, B*n_stream)         │  │   │   │
│  │  │  │   build_rope_shift(k_body_view)  ──► quantized path:    │  │   │   │
│  │  │  │     ggml_cast → F32                                     │  │   │   │
│  │  │  │     ggml_mul_mat_aux (rot^T = rot, Hadamard back)      │  │   │   │
│  │  │  │     ggml_rope_ext (apply shift)                         │  │   │   │
│  │  │  │     ggml_mul_mat_aux (rot, Hadamard fwd)                │  │   │   │
│  │  │  │     ggml_cpy → k_body (re-quantize to Q2_KVARN)        │  │   │   │
│  │  │  └─────────────────────────────────────────────────────────┘  │   │   │
│  │  │                                                               │   │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │   │   │
│  │  │  │ Region Recent (F16)  [S+B .. kv_size)                   │  │   │   │
│  │  │  │   view_3d(k_recent, n_rot, n_head_kv, R*n_stream)       │  │   │   │
│  │  │  │   build_rope_shift(k_recent_view)  ──► F16 path         │  │   │   │
│  │  │  └─────────────────────────────────────────────────────────┘  │   │   │
│  │  │                                                               │   │   │
│  │  └─── NO ────────────────────────────────────────────────────┐   │   │   │
│  │                                                               │   │   │   │
│  │  ┌─────────────────────────────────────────────────────────┐  │   │   │   │
│  │  │ Single tensor (any type)  [0 .. kv_size)                │  │   │   │   │
│  │  │   view_3d(layer.k, n_rot, n_head_kv, kv_size*n_stream) │  │   │   │   │
│  │  │   build_rope_shift(k_view)  ──► quantized or F16 path   │  │   │   │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │   │   │
│  │                                                               │   │   │   │
│  └───────────────────────────────────────────────────────────────┘   │   │   │
│                                                                      │   │   │
└──────────────────────────────────────────────────────────────────────┘   │   │
                                                                           │   │
┌─────────────────────────────────────────────────────────────────────────┘   │
│                          llama_kv_cache::build_rope_shift                    │
│                           src/llama-kv-cache.cpp:1868                        │
│                                                                              │
│  ┌─ ggml_is_quantized(cur->type)? ── YES ──────────────────────────────┐     │
│  │                                                                     │     │
│  │  tmp = ggml_cast(ctx, cur, GGML_TYPE_F32)     // dequantize        │     │
│  │  tmp = ggml_mul_mat_aux(ctx, tmp, rot)        // rotate back       │     │
│  │  tmp = ggml_rope_ext(ctx, tmp, ...)           // RoPE shift        │     │
│  │  tmp = ggml_mul_mat_aux(ctx, tmp, rot)        // rotate forward    │     │
│  │  tmp = ggml_cpy(ctx, tmp, cur)                // requantize        │     │
│  │                                                                     │     │
│  └─── NO ──────────────────────────────────────────────────────────┐   │     │
│                                                                     │   │     │
│  tmp = ggml_rope_ext_inplace(ctx, cur, ...)    // in-place RoPE     │   │     │
│                                                                     │   │     │
└─────────────────────────────────────────────────────────────────────┘   │     │
                                                                           │     │
┌───────────────────────────────────────────────────────────────────────────┘     │
│                          ggml_mul_mat_aux (helper)                               │
│                           src/llama-kv-cache.cpp:60                               │
│                                                                                  │
│  res = ggml_reshape_2d(ctx, cur, n, nelements/n)                                │
│  res = ggml_mul_mat(ctx, rot, res)            // H * x                          │
│  ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD)                         │
│  res = ggml_reshape_4d(ctx, res, ...)                                            │
│                                                                                  │
│  Note: H^2 = I, so rotate-back and rotate-forward use the same matrix.           │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Key types

| Symbol | Type | Role |
|--------|------|------|
| `k_sink` | `GGML_TYPE_F16` | First S cells, never quantized |
| `k_body` | `GGML_TYPE_Q2_KVARN` | Middle B cells, VarN-quantized |
| `k_recent` | `GGML_TYPE_F16` | Last R cells, never quantized |
| `k_shift` | `GGML_TYPE_I32` | Per-cell shift values (input) |
| `k_rot` | `GGML_TYPE_F32` | Hadamard matrix (input, H^2=I) |
| `rope_factors` | `GGML_TYPE_F32` | YaRN frequency scaling factors |
| `tmp` (quant path) | `GGML_TYPE_F32` | Dequantized working buffer |
| `rot` | `GGML_TYPE_F32` | Hadamard matrix, n×n |

## Partition constants

```
S = kv_size / 4          // sink region size
R = kv_size / 4          // recent region size
B = kv_size - S - R      // body region size
```
