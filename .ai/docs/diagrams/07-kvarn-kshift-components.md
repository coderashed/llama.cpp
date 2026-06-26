# 07 — K-shift components: file ownership and graph node dependencies

## File ownership

| File | Lines | Owns | Responsibility |
|------|-------|------|----------------|
| `src/llama-kv-cache.cpp` | 1868-1917 | `build_rope_shift` | Dequant → rotate → RoPE → rotate → requant pipeline |
| `src/llama-kv-cache.cpp` | 1947-1993 | `build_graph_shift` | Iterates layers, creates 3-region views, calls `build_rope_shift` |
| `src/llama-kv-cache.cpp` | 60-74 | `ggml_mul_mat_aux` | Hadamard multiply helper (reshape → mul_mat → hint → reshape) |
| `src/llama-kv-cache.cpp` | 826-905 | `update` | Detects `has_shift`, triggers graph build + compute |
| `src/llama-kv-cache.cpp` | 1920-1945 | `llm_graph_input_k_shift` | Input class: owns `k_shift` and `k_rot` tensors |
| `src/llama-kv-cache.cpp` | 579-627 | `seq_add` | Sets `has_shift`, accumulates per-cell deltas |
| `src/llama-kv-cache.cpp` | 629-666 | `seq_div` | Sets `has_shift`, integer-divide positions |
| `src/llama-kv-cache.h` | 223-241 | `kv_layer` struct | Holds `k_sink`, `k_body`, `k_recent` pointers |
| `src/llama-kv-cache.cpp` | 245-303 | Constructor | Allocates 3-region tensors when `is_kvarn` |
| `ggml/src/ggml.c` | 7714+ | `ggml_quantize_chunk` | Re-quantization kernel (F32 → Q2_KVARN) |
| `ggml/src/ggml.c` | — | `ggml_cast` | Dequantization kernel (Q2_KVARN → F32) |

## Graph node types and producers

| ggml op | Producer | Consumes | Produces |
|---------|----------|----------|----------|
| `ggml_view_3d` | `build_graph_shift` | `layer.k` / `k_sink` / `k_body` / `k_recent` | View tensor (no data copy) |
| `ggml_cast` | `build_rope_shift` | Quantized view | F32 tensor |
| `ggml_reshape_2d` | `ggml_mul_mat_aux` | F32 tensor | Reshaped 2D view |
| `ggml_mul_mat` | `ggml_mul_mat_aux` | `rot` + reshaped tensor | Rotated tensor |
| `ggml_reshape_4d` | `ggml_mul_mat_aux` | Rotated 2D tensor | Restored 4D shape |
| `ggml_rope_ext` | `build_rope_shift` | F32 tensor + `k_shift` + factors | RoPE-shifted F32 |
| `ggml_cpy` | `build_rope_shift` | F32 tensor | Quantized output (re-quant) |
| `ggml_rope_ext_inplace` | `build_rope_shift` | F16 tensor + `k_shift` | In-place shifted F16 |

## Input tensor ownership

```
llm_graph_input_k_shift
  │
  ├── k_shift  (GGML_TYPE_I32, [kv_size * n_stream])
  │     │  set_input: llama_kv_cache::set_input_k_shift()
  │     │  Filled with per-cell shift values from cells.shift[]
  │     │
  └── k_rot   (GGML_TYPE_F32, [n_rot, n_rot])
           │  set_input: llama_kv_cache::set_input_k_rot()
           │  Filled with Hadamard matrix from attn_rot_hadamard[n_rot]
           │  Only present when attn_rot_k == true
```

## Graph expansion order (per layer)

```
For each layer il:
  1. k_view = ggml_view_3d(layer.k, n_rot, n_head_kv, size*n_stream,
                            row_size(n_embd_head_k), row_size(n_embd_k_gqa),
                            row_size(n_embd_nope))
  2. cur = build_rope_shift(cparams, ctx, k_view, k_shift, k_rot,
                             rope_factors, freq_base, freq_scale, il)
  3. ggml_build_forward_expand(gf, cur)
```

For Q2_KVARN, step 1 is repeated three times (sink/body/recent views),
each calling `build_rope_shift` independently.  The three `ggml_cpy`
outputs write back to the three separate tensors.

## Backend scheduling

The graph is submitted to `ggml_backend_graph_compute` in `update()`.
The backend scheduler resolves data dependencies and dispatches to
the appropriate device (CPU or GPU per layer device).
