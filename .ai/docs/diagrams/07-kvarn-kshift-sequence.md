# 07 — K-shift sequence: dequant Q2_KVARN → rotate back → RoPE → rotate forward → requant

## Shift sequence for the body region (Q2_KVARN)

```
Time ──────────────────────────────────────────────────────────────────────────►

llama_memory_seq_add()          llama_kv_cache::update()      build_graph_shift()
        │                               │                           │
        │  seq_add(seq, p0, p1, delta)   │                           │
        │  ─────────────────────────────►│                           │
        │    sets has_shift = true       │                           │
        │    updates cell positions      │                           │
        │    updates shift[i] += delta   │                           │
        │                               │                           │
        │  (return to caller)           │                           │
        │◄──────────────────────────────│                           │
        │                               │                           │
        │  ... next decode step ...     │                           │
        │                               │                           │
        │  init_update(lctx, optimize)   │                           │
        │  ────────────────────────────►│                           │
        │                               │  get_has_shift() == true  │
        │                               │  do_shift = true          │
        │                               │                           │
        │                               │  build_graph_shift(res)   │
        │                               │  ───────────────────────► │
        │                               │                           │
        │                               │  ┌─ For each layer: ──┐   │
        │                               │  │                     │   │
        │                               │  │  is_kvarn?          │   │
        │                               │  │    │                │   │
        │                               │  │    ├─ YES ──► 3 views │   │
        │                               │  │    │  k_sink_view     │   │
        │                               │  │    │  k_body_view     │   │
        │                               │  │    │  k_recent_view   │   │
        │                               │  │    │                  │   │
        │                               │  │    └─ NO ──► 1 view  │   │
        │                               │  │       layer.k_view   │   │
        │                               │  │                     │   │
        │                               │  │  For each view:     │   │
        │                               │  │  build_rope_shift() │   │
        │                               │  └─────────────────────┘   │
        │                               │                           │
        │                               │  return gf                 │
        │                               │◄──────────────────────────│
        │                               │                           │
        │                               │  ggml_backend_graph_compute│
        │                               │  ──────────────────────────► (backend)
        │                               │                           │
        │                               │  cells.reset_shift()      │
        │                               │  has_shift = false        │
        │                               │                           │
        │◄──────────────────────────────│                           │
        │                               │                           │
```

## Per-tensor shift pipeline (body region, Q2_KVARN)

```
k_body (Q2_KVARN, n_rot × n_head_kv × B*n_stream)
  │
  │  view_3d offset = n_embd_nope (skip non-RoPE dims)
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ build_rope_shift(k_body_view)                                       │
│                                                                     │
│  Step 1:  ggml_cast(k_body_view, GGML_TYPE_F32)                     │
│  ────────► tmp (F32, same shape)                                    │
│             Dequantizes Q2_KVARN → F32 using dual-scale formula:    │
│               x = s1 * round(x_q / s2)                              │
│               where s1 = d, s2 = d2 (per-block dual scales)         │
│                                                                     │
│  Step 2:  ggml_mul_mat_aux(tmp, rot)                                │
│  ────────► tmp = rot × tmp  (Hadamard rotate back)                 │
│             rot is the Walsh-Hadamard matrix (H^2 = I)              │
│             This undoes the pre-quantization rotation                │
│                                                                     │
│  Step 3:  ggml_rope_ext(tmp, shift, factors, ...)                   │
│  ────────► tmp = RoPE(tmp, pos + shift_delta)                       │
│             Applies rotary position embedding shift                 │
│             shift_delta from k_shift tensor (I32 per cell)         │
│                                                                     │
│  Step 4:  ggml_mul_mat_aux(tmp, rot)                                │
│  ────────► tmp = rot × tmp  (Hadamard rotate forward)              │
│             Re-applies Hadamard rotation for quantization           │
│                                                                     │
│  Step 5:  ggml_cpy(tmp, k_body_view)                                │
│  ────────► k_body_view = quantize(tmp)                              │
│             Re-quantizes F32 → Q2_KVARN                             │
│             Re-runs VarN on the shifted tile                        │
│             Writes back to the original cache buffer                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Per-tensor shift pipeline (sink/recent regions, F16)

```
k_sink (F16) or k_recent (F16)
  │
  │  view_3d offset = n_embd_nope
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ build_rope_shift(k_sink_view)                                      │
│                                                                     │
│  ggml_is_quantized(GGML_TYPE_F16) → false                           │
│                                                                     │
│  Step 1:  ggml_rope_ext_inplace(k_sink_view, shift, ...)            │
│  ────────► k_sink_view = RoPE(k_sink_view, pos + shift_delta)      │
│             In-place F16 RoPE shift, no dequant/requant needed     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Graph node dependency DAG (body region)

```
k_body (Q2_KVARN) ──► ggml_cast ──► tmp (F32)
                                        │
k_rot (Hadamard) ──────────────────────► ggml_mul_mat_aux (rotate back)
                                        │
k_shift (I32) ──┐                      │
rope_factors ────┤                      │
freq_base ───────┤                      │
freq_scale ──────┤                      │
rope_type ───────┤                      │
n_ctx_orig ──────┤                      │
yarn_* ──────────┤                      │
                 ▼                      ▼
          ggml_rope_ext ◄───────────────┘
                 │
k_rot ───────────► ggml_mul_mat_aux (rotate forward)
                 │
                 ▼
          ggml_cpy ──► k_body (Q2_KVARN, re-quantized)
```

## Invariant: H^2 = I

The Walsh-Hadamard matrix satisfies H^2 = I (up to scaling factor 1/n).
Therefore the same `k_rot` tensor is used for both rotate-back and
rotate-forward steps.  The `ggml_mul_mat_set_hint` hint allows backend
kernels to optimize the Hadamard product.
