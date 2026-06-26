---
title: Q2_KVARN GPU Kernel -- Safety Contract, Memory Coalescing & Warp Alignment
---

```mermaid
graph TD
    subgraph "Preconditions (caller must ensure)"
        P1["src0->type == GGML_TYPE_Q2_KVARN<br/>Weight tensor is Q2_KVARN quantized"]
        P2["src1->type == GGML_TYPE_F32<br/>Activations are F32"]
        P3["dst->type == GGML_TYPE_F32<br/>Output is F32"]
        P4["block_q2_kvarn struct is 38 bytes<br/>No padding between fields"]
        P5["GPU compute capability >= 5.0<br/>(Maxwell+ for FP16 support)"]
    end

    subgraph "Memory Coalescing Contract"
        M1["block_q2_kvarn layout in global memory:<br/>  qs[32] at offset 0  (32 bytes)<br/>  d     at offset 32 (2 bytes, FP16)<br/>  s1    at offset 34 (2 bytes, FP16)<br/>  s2    at offset 36 (2 bytes, FP16)<br/>  Total: 38 bytes, padded to 40?"]
        M2["load_tiles reads 38 bytes per block<br/>Warp of 32 threads reads 32 blocks = 1216 bytes<br/>Coalesced if blocks are contiguous in memory"]
        M3["qs[32] access pattern:<br/>  Thread t reads qs[t % 32] from each block<br/>  Adjacent threads read adjacent bytes -> coalesced"]
        M4["d, s1, s2 access:<br/>  All threads read same offset within block<br/>  Broadcast from same cache line -> no penalty"]
        M5["vec_dot_fattn_vec_KQ reads K sequentially:<br/>  K stride = nb11 = sizeof(block_q2_kvarn)<br/>  Threads with consecutive tid read consecutive blocks<br/>  Fully coalesced 128-byte transactions"]
    end

    subgraph "Warp Alignment Contract"
        W1["nthreads_KQ must divide WARP_SIZE (32)<br/>  For D=128: nthreads_KQ = 32 -> 1 warp per row<br/>  For D=64:  nthreads_KQ = 16 -> 2 warps per row"]
        W2["nthreads_V must divide WARP_SIZE<br/>  For D=128: nthreads_V = 32 -> 1 warp per row"]
        W3["static_assert(WARP_SIZE % nthreads_KQ == 0)<br/>  Already in fattn-vec.cuh:90"]
        W4["static_assert(WARP_SIZE % nthreads_V == 0)<br/>  Already in fattn-vec.cuh:91"]
        W5["MMQ path: nwarps = 8, warp_size = 32<br/>  Block size = 256 threads<br/>  x_tile stride aligned to warp boundaries"]
    end

    subgraph "Numerical Contract"
        N1["Dequant formula: v = (q + d) * s1 * s2<br/>  q in {0,1,2,3} (2-bit unsigned)<br/>  d in FP16 range [-65504, +65504]<br/>  s1, s2 in FP16 range"]
        N2["FP16->FP32 conversion is exact<br/>  No rounding in dequant path"]
        N3["s2 multiply adds 1 FMAD per element<br/>  vs Q2_K dequant: (q + d) * s<br/>  Overhead: 1 FMA / 128 elements per block"]
        N4["No NaN/Inf from valid FP16 inputs<br/>  FP16->FP32 never produces NaN from valid FP16"]
        N5["Deterministic: same input -> same output<br/>  No atomics, no rounding in dequant"]
    end

    subgraph "Error Conditions"
        E1["src0->type != Q2_KVARN<br/>  Caught by switch_type() -> GGML_ABORT"]
        E2["src1->type != F32<br/>  Caught by GGML_ASSERT in ggml_cuda_mul_mat_q"]
        E3["dst->type != F32<br/>  Caught by GGML_ASSERT in ggml_cuda_mul_mat_q"]
        E4["Misaligned block_q2_kvarn pointer<br/>  block_q2_kvarn has alignment 2 (FP16 fields)<br/>  CUDA global memory loads require 1-byte alignment minimum<br/>  No explicit alignment check needed"]
        E5["Out-of-bounds K access in flash attention<br/>  KV_max_ptr bounds the loop<br/>  No out-of-bounds if KV_max is correct"]
        E6["supports_op returns false for unknown types<br/>  Default case returns false<br/>  Q2_KVARN must be explicitly added"]
    end

    subgraph "Thread Safety"
        T1["All device functions are reentrant<br/>  No global state, no device mallocs"]
        T2["Shared memory is per-block<br/>  No cross-block data races"]
        T3["Global memory writes are disjoint<br/>  Each block writes to unique dst region"]
        T4["No atomics used in dequant path<br/>  (Atomics may be used in softmax reduction)"]
    end

    P1 --> M1
    P2 --> M1
    P3 --> M1
    M1 --> M2
    M2 --> M3
    M2 --> M4
    M5 --> W1
    W1 --> W3
    W2 --> W4
    M3 --> N1
    N1 --> N3
    N3 --> N4
    N4 --> N5
    E1 --> T1
    E2 --> T1
    E3 --> T1
```

### Safety Contract Summary

| Condition | Assertion / Enforcement | Location | Failure Mode |
|-----------|------------------------|----------|-------------|
| src0 type | `GGML_ASSERT(src0->type == GGML_TYPE_Q2_KVARN)` | `mmq.cu` switch dispatch | `GGML_ABORT` |
| src1 type | `GGML_ASSERT(src1->type == GGML_TYPE_F32)` | `ggml_cuda_mul_mat_q` | `GGML_ABORT` |
| dst type | `GGML_ASSERT(dst->type == GGML_TYPE_F32)` | `ggml_cuda_mul_mat_q` | `GGML_ABORT` |
| Block size | `static_assert(sizeof(block_q2_kvarn) == 38)` | `ggml-common.h:191` | Compile error |
| Warp alignment | `static_assert(WARP_SIZE % nthreads_KQ == 0)` | `fattn-vec.cuh:90` | Compile error |
| Warp alignment | `static_assert(WARP_SIZE % nthreads_V == 0)` | `fattn-vec.cuh:91` | Compile error |
| supports_op | `case GGML_TYPE_Q2_KVARN: return true` | `ggml-cuda.cu:5156-5186` | Falls to CPU fallback |
| Memory coalescing | Contiguous block layout | `kvarn.cuh` load_tiles | Performance degradation |
| No NaN/Inf | FP16->FP32 is exact | `kvarn.cuh` dequant | N/A (guaranteed) |
| Thread safety | No globals, no atomics in dequant | All device functions | Data race (impossible) |

### Overhead Budget

| Component | Instructions | vs Q2_K baseline |
|-----------|-------------|-------------------|
| Unpack 2-bit | 2 shifts + 2 AND | Same |
| Add d | 1 FADD | Same |
| Mul by s1 | 1 FMUL | Same |
| **Mul by s2** | **1 FMUL** | **+1 FMA per element** |
| Total per element | 6 ops | +1 FMA (16.7% more ops) |
| Hides behind memory latency? | Yes | FMA is free when memory-bound |

The extra `* s2` multiply is a single FMA instruction per element. Since the dequant kernel is memory-bound (reading 38 bytes to produce 512 bytes of output), the extra arithmetic is hidden by memory latency. This is why the paper reports <1.4% overhead at short contexts and ~0% at long contexts.

### Key Design Decisions

1. **No separate dequant kernel**: Dequant is fused into `load_tiles` (mmq path) and `vec_dot_KQ` (fattn path). This avoids an extra HBM write of the dequantized tensor.

2. **s2 absorbed in load_tiles**: The `load_tiles_q2_kvarn` function applies `* s2` during the tile load into shared memory. The downstream `vec_dot` functions are identical to Q2_K's -- they operate on already-dequantized values.

3. **Flash attention inline dequant**: The `vec_dot_fattn_vec_KQ_q2_kvarn` function dequantizes K on-the-fly from registers. No shared memory intermediate for K values.

4. **Reuse Q2_K constants**: `VDR_Q2_K_Q8_1_MMQ`, `QR2_K`, `QI2_K` are reused from Q2_K since the block structure is identical (same packing, same 128-element block size).

5. **No FP8 on GPU**: The paper mentions FP8 for s1/s2, but the GPU implementation stores them as FP16 to avoid FP8 conversion overhead on pre-Hopper GPUs. This adds 0.0625 bits/element (total 2.3125 vs 2.25), which is negligible.
