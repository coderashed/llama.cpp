---
title: Q2_KVARN GPU Kernel -- Kernel Launch Sequence
---

```mermaid
sequenceDiagram
    participant Sched as ggml_backend_sched
    participant CUDA as ggml_backend_cuda_context
    participant Supp as supports_op
    participant Dispatch as ggml_cuda_mul_mat_q
    participant Switch as ggml_cuda_mul_mat_q_switch_type
    participant Kernel as mul_mat_q_case<Q2_KVARN>
    participant Load as load_tiles_q2_kvarn (shared mem)
    participant Dot as vec_dot_q2_kvarn_q8_1 (mma/dp4a)
    participant WB as mmq_write_back

    Note over Sched,WB: === MUL_MAT PATH (mmq.cu) ===

    Sched->>CUDA: graph_compute(split_graph)
    CUDA->>Supp: supports_op(dev, MUL_MAT, src0=Q2_KVARN)
    Supp-->>CUDA: true (new case added)
    CUDA->>Dispatch: ggml_cuda_mul_mat_q(ctx, src0, src1, dst)
    Note over Dispatch: src0 = block_q2_kvarn weights<br/>src1 = F32 activations<br/>dst = F32 output
    Dispatch->>Switch: ggml_cuda_mul_mat_q_switch_type(args)
    Switch->>Kernel: mul_mat_q_case<GGML_TYPE_Q2_KVARN>(ctx, args, stream)

    Note over Kernel: Grid: [n_cols / mmq_x, n_rows / mmq_y, 1]<br/>Block: [warp_size, nwarps, 1]

    Kernel->>Load: load_tiles_q2_kvarn(x, x_tile, kbx0, i_max, stride)
    Note over Load: Reads block_q2_kvarn from global memory<br/>Unpacks 2-bit values, applies dual-scale formula<br/>Stores dequantized tile in shared memory<br/>s2 multiply applied inline during load
    Load-->>Kernel: x_tile in shared memory

    Kernel->>Dot: vec_dot_q2_kvarn_q8_1(x_tile, y_tile, sum, k00)
    Note over Dot: Dot product of dequantized K tile<br/>against Q in q8_1 format<br/>Accumulates partial sums per warp
    Dot-->>Kernel: partial sums in registers

    Kernel->>WB: mmq_write_back(sum, get_rows_to_sorted, dst, stride, i_max, j_max)
    Note over WB: Writes final F32 result to global memory
    WB-->>Kernel: done

    Kernel-->>Switch: kernel complete
    Switch-->>Dispatch: return
    Dispatch-->>CUDA: return
    CUDA-->>Sched: GGML_STATUS_SUCCESS

    Note over Sched,WB: === FLASH ATTENTION PATH (fattn-vec.cuh) ===

    Sched->>CUDA: graph_compute(split_graph with FLASH_ATTN_EXT)
    CUDA->>Dispatch: ggml_cuda_flash_attn_ext_vec(ctx, dst)
    Note over Dispatch: Q = F32, K = block_q2_kvarn, V = block_q2_kvarn/F32

    Dispatch->>Kernel: flash_attn_ext_vec<D, ncols, Q2_KVARN, type_V>(...)

    Note over Kernel: Grid: [ncols, nheads, nseq]<br/>Block: [128, 1, 1]

    Note over Kernel: --- STEP 1: Convert Q to q8_1 in shared memory ---
    Note over Kernel: Q_reg[ncols][D/2/nthreads_KQ] in registers<br/>Q_i32[ncols][D/sizeof(int)] in registers<br/>Q_ds[ncols][D/QK8_1] in registers

    Note over Kernel: --- STEP 2: KQ loop over KV tokens ---
    loop per KV token block (FATTN_KQ_STRIDE = 256)
        Note over Kernel: Load K block from global memory
        Kernel->>Kernel: vec_dot_fattn_vec_KQ_q2_kvarn(K_c, Q_v, Q_q8, Q_ds)
        Note over Kernel: Inline dequant of block_q2_kvarn<br/>Dual-scale formula applied per element<br/>Dot product accumulated in float sum
        Note over Kernel: Store KQ result in shared memory KQ[ncols][D/2]
        Note over Kernel: Online softmax: update KQ_max, KQ_sum
    end

    Note over Kernel: --- STEP 3: V loop ---
    loop per KV token block
        Note over Kernel: Load V block from global memory
        Kernel->>Kernel: dequantize_V_q2_kvarn(V_c, VKQ, ...)
        Note over Kernel: Dequant V inline, accumulate into VKQ
    end

    Note over Kernel: --- STEP 4: Write output ---
    Kernel->>WB: Write dst[ncols][D] and dst_meta[ncols][2]
    WB-->>Kernel: done

    Kernel-->>Dispatch: kernel complete
    Dispatch-->>CUDA: return
    CUDA-->>Sched: GGML_STATUS_SUCCESS
```

### Launch Configuration Summary

| Path | Kernel | Grid | Block | Shared Mem |
|------|--------|------|-------|------------|
| MUL_MAT (mmq) | `mul_mat_q_case<Q2_KVARN>` | `[n_cols/mmq_x, n_rows/mmq_y, 1]` | `[warp_size, nwarps, 1]` | x_tile + y_tile |
| Flash Attn (fattn-vec) | `flash_attn_ext_vec<D, ncols, Q2_KVARN, V>` | `[ncols, nheads, nseq]` | `[128, 1, 1]` | KQ + VKQ + temp |

### Key Design Points

1. **No separate dequant kernel launch**: Dequant is fused into the load_tiles step (mmq path) or the vec_dot_KQ step (fattn path). This avoids an extra HBM round-trip.

2. **s2 multiply is free**: The extra `* s2` multiply is a single FMAD instruction per element, which is hidden by memory latency. This is why the paper reports <1.4% overhead.

3. **Shared memory tile layout**: For mmq path, the dequantized tile stores values in the same format as Q2_K tiles, with s2 already applied. The vec_dot functions are identical to Q2_K's -- only the load_tiles function differs.

4. **Flash attention path**: The `vec_dot_fattn_vec_KQ_q2_kvarn` function is added to `get_vec_dot_KQ` dispatch. It follows the same pattern as existing `vec_dot_fattn_vec_KQ_q4_0` etc.
