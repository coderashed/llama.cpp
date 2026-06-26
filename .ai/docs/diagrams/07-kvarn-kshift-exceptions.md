# 07 — K-shift exceptions: safety contract and three-region awareness

## Safety contract

| Condition | Guard | Consequence |
|-----------|-------|-------------|
| `!other` (no cell sharing) | `GGML_ASSERT` in `build_graph_shift:1949` | Abort if cache shares cells with another cache |
| `get_has_shift()` false | `update:868` | `GGML_ABORT("K-shift not supported")` |
| `n_pos_per_embd() > 1` | `seq_add:586` | `GGML_ASSERT` — multi-position embeddings cannot shift |
| `k_rot` dimension mismatch | `set_input_k_rot:1826` | `GGML_ASSERT(attn_rot_hadamard.count(n_rot))` |
| `k_shift` buffer not host | `set_input_k_shift` | `GGML_ASSERT(ggml_backend_buffer_is_host(...))` |
| `ggml_cast` on non-quantized | `build_rope_shift:1895` | `ggml_is_quantized` check prevents this path |
| `ggml_cpy` type mismatch | Backend | Backend asserts src/dst have same `ne[]` dimensions |
| `ggml_mul_mat` shape mismatch | Backend | Backend asserts `rot->ne[0] == cur->ne[0]` |
| `ggml_rope_ext` shift size | Backend | Backend asserts `shift->ne[0] == n_rot` dimension |

## Three-region awareness: what changes for Q2_KVARN

### `build_graph_shift` (line 1947)

**Current code** (single tensor):
```cpp
ggml_tensor * k = ggml_view_3d(ctx, layer.k, ...);
ggml_tensor * cur = build_rope_shift(..., k, ...);
ggml_build_forward_expand(gf, cur);
```

**Required for Q2_KVARN** (three regions):
```cpp
if (is_kvarn) {
    // Region 1: sink (F16) — in-place RoPE, no dequant
    ggml_tensor * k_sink_view = ggml_view_3d(ctx, layer.k_sink, n_rot, n_head_kv, S*n_stream, ...);
    ggml_tensor * cur_sink = build_rope_shift(..., k_sink_view, ...);
    ggml_build_forward_expand(gf, cur_sink);

    // Region 2: body (Q2_KVARN) — dequant → rotate → RoPE → rotate → requant
    ggml_tensor * k_body_view = ggml_view_3d(ctx, layer.k_body, n_rot, n_head_kv, B*n_stream, ...);
    ggml_tensor * cur_body = build_rope_shift(..., k_body_view, ...);
    ggml_build_forward_expand(gf, cur_body);

    // Region 3: recent (F16) — in-place RoPE, no dequant
    ggml_tensor * k_recent_view = ggml_view_3d(ctx, layer.k_recent, n_rot, n_head_kv, R*n_stream, ...);
    ggml_tensor * cur_recent = build_rope_shift(..., k_recent_view, ...);
    ggml_build_forward_expand(gf, cur_recent);
} else {
    // Original single-tensor path
    ggml_tensor * k = ggml_view_3d(ctx, layer.k, ...);
    ggml_tensor * cur = build_rope_shift(..., k, ...);
    ggml_build_forward_expand(gf, cur);
}
```

### `build_rope_shift` (line 1868)

**No changes needed** — the function already handles both quantized and
non-quantized types via `ggml_is_quantized(cur->type)`:

- F16 sink/recent views → `ggml_rope_ext_inplace` (fast path, no dequant)
- Q2_KVARN body view → `ggml_cast` → rotate → RoPE → rotate → `ggml_cpy`

### `get_k` (line 1281)

**Must be updated** to return the correct region tensor based on cell index.
Currently returns `layers[ikv].k` (a single tensor).  For Q2_KVARN, the
three tensors are separate; `get_k` must select among `k_sink`, `k_body`,
`k_recent` based on the cell position.

### `cpy_k` (line 1333)

**Must be updated** to write to the correct region tensor.  Currently writes
to `layers[ikv].k`.  For Q2_KVARN, must dispatch to `k_sink`, `k_body`,
or `k_recent` based on the target cell index.

## Three-region boundary invariants

```
Cell index:   0 ... S-1 | S ... S+B-1 | S+B ... kv_size-1
              ──────────┼─────────────┼──────────────────
Region:       sink      │ body        │ recent
Type:         F16       │ Q2_KVARN    │ F16
Shift path:   in-place  │ deq→rot→...│ in-place
```

- S = kv_size / 4, R = kv_size / 4, B = kv_size - S - R
- Sink and recent are always F16 (no quantization error for anchor tokens)
- Body is Q2_KVARN (VarN-quantized, benefits from Hadamard rotation)
- The three tensors are independent allocations; no single `layer.k` exists
- `layer.k` is `nullptr` when `is_kvarn == true` (line 278-284)

## Failure modes specific to Q2_KVARN K-shift

| Condition | Symptom | Guard |
|-----------|---------|-------|
| `k_sink` is nullptr | Segfault in `build_graph_shift` | `GGML_ASSERT(layer.k_sink)` before view |
| `k_body` is nullptr | Segfault in `build_graph_shift` | `GGML_ASSERT(layer.k_body)` before view |
| `k_recent` is nullptr | Segfault in `build_graph_shift` | `GGML_ASSERT(layer.k_recent)` before view |
| Cell index crosses region boundary | Wrong tensor written | `get_k`/`cpy_k` must check cell index |
| `ggml_cast` not implemented for Q2_KVARN | Backend error | Must be implemented in ggml (item 01) |
| `ggml_quantize_chunk` not implemented for Q2_KVARN | Backend error | Must be implemented in ggml (item 03) |
| Hadamard dimension not power of 2 | `ggml_gen_hadamard` assert | `n_rot` must be power of 2 |
| Stream count > 1 with three-region | View stride mismatch | `k_stream` views already handle this (line 291) |

## Exit criterion verification

The exit criterion states: `build_rope_shift` handles `GGML_TYPE_Q2_KVARN`
with the dequant → rotate back → RoPE → rotate forward → requant pipeline.

Verification steps:
1. `ggml_is_quantized(GGML_TYPE_Q2_KVARN)` returns `true` → enters quantized path
2. `ggml_cast(..., GGML_TYPE_F32)` dequantizes using dual-scale formula
3. `ggml_mul_mat_aux(..., rot)` applies Hadamard (H^2 = I, same for back/fwd)
4. `ggml_rope_ext(...)` applies position shift
5. `ggml_mul_mat_aux(..., rot)` re-applies Hadamard
6. `ggml_cpy(..., k_body)` re-quantizes F32 → Q2_KVARN via `ggml_quantize_chunk`

The three-region awareness is in `build_graph_shift`, not in
`build_rope_shift` itself.  `build_rope_shift` is type-agnostic and
works correctly for any quantized type including Q2_KVARN.
