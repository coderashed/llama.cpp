---
title: KVarN Pipeline Integration -- Safety Contract & Exceptions
---

```mermaid
graph TD
    subgraph "Preconditions (caller must ensure)"
        P1["type_k == GGML_TYPE_Q2_KVARN<br/>enables VarN path"]
        P2["head_dim % 64 == 0<br/>(Hadamard rotation requirement)"]
        P3["head_dim == n_embd_head_k<br/>(tile columns = head dimension)"]
        P4["n_tokens >= 1<br/>(at least one token to write)"]
        P5["k_cur is F32 after rotation<br/>(not yet quantized)"]
        P6["G = 128 divides head_dim<br/>(tile alignment for quantizer)"]
    end

    subgraph "Tile Alignment Contract"
        T1["Full tile: n_tokens >= G<br/>Process G tokens at a time"]
        T2["Partial tile: n_tokens < G<br/>Fallback to plain RTN (no VarN)"]
        T3["Tile stride: G tokens per head<br/>T[G][head_dim] contiguous in memory"]
        T4["S_c size: head_dim floats<br/>One per column (channel)"]
        T5["S_r size: G floats<br/>One per token (row)"]
    end

    subgraph "Thread Safety"
        S1["VarNTileProcessor is per-cache-instance<br/>NOT shared between streams"]
        S2["Each stream has its own k_cur buffer<br/>No concurrent writes to same tile"]
        S3["kvarn_variance_normalize is reentrant<br/>Safe on disjoint buffer sets"]
        S4["quantize_row_q2_kvarn_varn is pure<br/>No global state, no allocations"]
    end

    subgraph "Partial Group Handling"
        H1["n_tokens < G (e.g. last ubatch)"]
        H2["Extract remaining tokens as tile T[n_rem][head_dim]"]
        H3["Skip VarN (not enough rows for meaningful normalization)"]
        H4["Call plain quantize_row_q2_kvarn_ref instead"]
        H5["S_c = 1.0, S_r = 1.0 (identity)"]
    end

    subgraph "K-Shift Re-quantize (no VarN)"
        K1["build_rope_shift re-quantizes via ggml_cpy"]
        K2["Uses plain quantize_row_q2_kvarn_ref<br/>NOT the VarN variant"]
        K3["VarN scales are already absorbed in stored blocks<br/>Re-quantize preserves s1/s2 from original write"]
        K4["Rationale: K-shift is a maintenance operation<br/>on already-quantized data; re-applying VarN<br/>would change the stored scales"]
    end

    subgraph "Numerical Bounds"
        N1["VarN output T is ~N(0,1) per element<br/>(unit variance target)"]
        N2["RTN on normalized tile:<br/>min ~ -3.0, max ~ +3.0 (3-sigma)"]
        N3["s1 = (max-min)/3 * S_c ~ 2.0 * S_c<br/>Well within FP16 range"]
        N4["s2 = S_r (VarN row scale)<br/>Typical range: [0.5, 2.0]"]
        N5["Dequant: (qval + d) * s1 * s2<br/>Restores original magnitude"]
    end

    subgraph "Error Conditions"
        E1["head_dim % 128 != 0<br/>Tile alignment fails -> assert"]
        E2["n_tokens == 0<br/>No-op (skip processing)"]
        E3["NaN in k_cur after rotation<br/>Propagates through VarN -> NaN output"]
        E4["Inf in k_cur after rotation<br/>Clamped by VarN c_min/c_max bounds"]
    end

    P1 --> T1
    P2 --> T1
    P3 --> T1
    P4 --> T2
    P5 --> T1
    P6 --> T1
    T1 --> S1
    T2 --> H1
    H1 --> H2
    H2 --> H3
    H3 --> H4
    H4 --> H5
    K1 --> K2
    K2 --> K3
    K3 --> K4
    T1 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
```

### Safety Contract Summary

| Condition | Assertion | Location |
|-----------|-----------|----------|
| Type check | `GGML_ASSERT(type_k == GGML_TYPE_Q2_KVARN)` | `VarNTileProcessor` entry |
| Head dim alignment | `GGML_ASSERT(head_dim % 128 == 0)` | Tile extraction |
| Tile size | `GGML_ASSERT(G == 128)` | Compile-time constant |
| Buffer ownership | Caller owns k_cur data; tile_buf is stack/scratch | `VarNTileProcessor` |
| Partial group | `if (n_remaining < G) { fallback_to_plain_rtn(); }` | `process_partial()` |
| K-shift re-quantize | Uses `quantize_row_q2_kvarn_ref` (no VarN) | `build_rope_shift` |
| Thread safety | No global state; per-instance scratch buffers | All functions |
| NaN propagation | Not detected; caller should sanitize if needed | All functions |

### Key Design Decisions

1. **VarN runs on CPU, not in graph**: The VarN normalization is a CPU-side pre-processing step that runs inside `cpy_k()` before `ggml_set_rows`. This avoids adding new graph ops and keeps the GPU backend path unchanged.

2. **New quantizer variant**: `quantize_row_q2_kvarn_varn` accepts external `S_c` and `S_r` vectors. It does NOT compute its own L2 ratio s2 — instead it uses `S_r` directly. The primary scale `s1` is computed as `(max-min)/3 * S_c[block]`.

3. **K-shift does NOT re-apply VarN**: When the cache is shifted (context shift), the re-quantization uses the plain `quantize_row_q2_kvarn_ref`. The VarN scales are already absorbed into the stored `s1`/`s2` fields, so re-quantization preserves them. Re-applying VarN during shift would be incorrect because the tile structure changes.

4. **Partial groups skip VarN**: When fewer than G=128 tokens are available (e.g., last ubatch in a decode step), VarN is skipped and plain RTN quantization is used. This is acceptable because:
   - Partial groups are rare (only at the end of a batch)
   - The magnitude error from a single partial group is negligible
   - VarN requires a full tile for meaningful row/column statistics

5. **Scratch buffer allocation**: `VarNTileProcessor` allocates its tile buffer (`G * head_dim` floats) once during cache construction, not per-call. This avoids repeated allocation overhead.
