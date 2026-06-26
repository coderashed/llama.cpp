---
title: Q2_KVARN GPU Kernel -- CUDA Kernel Structure & Type Traits
---

```mermaid
classDiagram

    class block_q2_kvarn {
        <<struct (ggml-common.h)>>
        +uint8_t  qs[32]     // 128 x 2-bit packed
        +ggml_half d          // FP16 zeropoint
        +ggml_half s1         // FP16 primary scale (column)
        +ggml_half s2         // FP16 secondary scale (row)
    }

    class dequantize_q2_kvarn {
        <<__device__ __forceinline__ (kvarn.cuh)>>
        +void dequantize_q2_kvarn(
        +    const void * vx,
        +    const int64_t ib,
        +    const int iqs,
        +    float2 & v)
        //
        // Reads block_q2_kvarn[ib]
        // Unpacks 2-bit values at iqs, iqs+1
        // v.x = (q0 + d) * s1 * s2
        // v.y = (q1 + d) * s1 * s2
    }

    class vec_dot_fattn_vec_KQ_q2_kvarn {
        <<__device__ (fattn-common.cuh)>>
        +float vec_dot_fattn_vec_KQ_q2_kvarn(
        +    const char * K_c,
        +    const void * Q_v,
        +    const int * Q_q8,
        +    const void * Q_ds)
        //
        // Inline dequant of K during KQ dot product
        // No separate dequant pass to HBM
    }

    class load_tiles_q2_kvarn {
        <<__device__ __forceinline__ (mmq.cuh)>>
        +void load_tiles_q2_kvarn(
        +    const char * x,
        +    int * x_tile,
        +    int kbx0, int i_max, int stride)
        //
        // Loads block_q2_kvarn into shared memory tiles
        // Dequant scales inline during tile load
    }

    class vec_dot_q2_kvarn_q8_1 {
        <<__device__ __forceinline__ (mmq.cuh)>>
        +void vec_dot_q2_kvarn_q8_1_mma(...)
        +void vec_dot_q2_kvarn_q8_1_dp4a(...)
        //
        // Dot product with dequantized K tile
        // against Q in q8_1 format
    }

    class ggml_cuda_type_traits {
        <<template specialization (common.cuh)>>
        +static constexpr int qk = QK_K
        +static constexpr int qr = QR2_K
        +static constexpr int qi = QI2_K
    }

    class mmq_type_traits {
        <<template specialization (mmq.cuh)>>
        +static constexpr int vdr
        +static constexpr load_tiles_mmq_t load_tiles
        +static constexpr vec_dot_mmq_t vec_dot_mma
        +static constexpr vec_dot_mmq_t vec_dot_dp4a
    }

    class get_vec_dot_KQ {
        <<constexpr dispatch (fattn-common.cuh)>>
        +vec_dot_KQ_t operator()()
        // if constexpr (type_K == GGML_TYPE_Q2_KVARN)
        //   return vec_dot_fattn_vec_KQ_q2_kvarn
    }

    class get_dequantize_V {
        <<constexpr dispatch (fattn-common.cuh)>>
        +dequantize_V_t operator()()
        // if constexpr (type_V == GGML_TYPE_Q2_KVARN)
        //   return dequantize_V_q2_kvarn
    }

    class ggml_cuda_mul_mat_q_switch_type {
        <<host dispatch (mmq.cu)>>
        +void ggml_cuda_mul_mat_q_switch_type(...)
        // case GGML_TYPE_Q2_KVARN:
        //   mul_mat_q_case<GGML_TYPE_Q2_KVARN>(...)
    }

    class ggml_backend_cuda_device_supports_op {
        <<host query (ggml-cuda.cu)>>
        +bool supports_op(dev, op)
        // case GGML_OP_MUL_MAT:
        //   switch (a->type):
        //     case GGML_TYPE_Q2_KVARN: return true
    }

    block_q2_kvarn --> dequantize_q2_kvarn : "read by"
    dequantize_q2_kvarn --> vec_dot_fattn_vec_KQ_q2_kvarn : "called by"
    dequantize_q2_kvarn --> load_tiles_q2_kvarn : "called by"
    load_tiles_q2_kvarn --> vec_dot_q2_kvarn_q8_1 : "feeds"
    vec_dot_fattn_vec_KQ_q2_kvarn --> get_vec_dot_KQ : "returned by"
    get_vec_dot_KQ --> flash_attn_ext_vec : "used in KQ step"
    get_dequantize_V --> flash_attn_ext_vec : "used in V step"
    mmq_type_traits --> load_tiles_q2_kvarn : ".load_tiles ="
    mmq_type_traits --> vec_dot_q2_kvarn_q8_1 : ".vec_dot_mma ="
    mmq_type_traits --> vec_dot_q2_kvarn_q8_1 : ".vec_dot_dp4a ="
    ggml_cuda_type_traits --> ggml_cuda_mul_mat_q_switch_type : "used by dispatch"
    ggml_cuda_mul_mat_q_switch_type --> mmq_type_traits : "instantiates"
    ggml_backend_cuda_device_supports_op --> ggml_cuda_mul_mat_q_switch_type : "guards dispatch"

    note for dequantize_q2_kvarn "Signature matches existing dequantize.cuh pattern:\n  void dequantize_q2_kvarn(vx, ib, iqs, float2 &v)\n\nUnpack:\n  q0 = (qs[byte] >> (bit*2)) & 0x03\n  q1 = (qs[byte] >> (bit*2+2)) & 0x03\n\nFormula:\n  v.x = (q0 + d) * s1 * s2\n  v.y = (q1 + d) * s1 * s2\n\ns2 multiply is the only addition over Q2_K dequant"
    note for vec_dot_fattn_vec_KQ_q2_kvarn "Inlined into flash_attn_ext_vec kernel.\nNo separate dequant kernel launch.\nK is dequantized on-the-fly from registers/shared memory.\n\nThread mapping:\n  nthreads_KQ = D/4 (for D=128: 32 threads)\n  Each thread processes 4 elements per K iteration"
    note for mmq_type_traits "New specialization:\n  mmq_type_traits<mmq_x, mmq_y, need_check, GGML_TYPE_Q2_KVARN>\n\n  vdr = VDR_Q2_K_Q8_1_MMQ (reuse Q2_K value)\n  load_tiles = load_tiles_q2_kvarn\n  vec_dot_mma = vec_dot_q2_kvarn_q8_1_mma\n  vec_dot_dp4a = vec_dot_q2_kvarn_q8_1_dp4a"
    note for ggml_backend_cuda_device_supports_op "Add case GGML_TYPE_Q2_KVARN to the\nMUL_MAT / MUL_MAT_ID switch at\nggml-cuda.cu:5156-5186\n\nAlso add to GET_ROWS switch if needed\nfor cache read path"
```

### Dequant Data Flow: block_q2_kvarn -> FP32 Register

```
block_q2_kvarn (38 bytes, 128 elements)
  |
  |-- qs[32]  --[unpack 2-bit per byte]--> 4 x uint8_t in {0,1,2,3}
  |-- d       --[FP16->FP32]-------------> float zeropoint
  |-- s1      --[FP16->FP32]-------------> float primary scale
  |-- s2      --[FP16->FP32]-------------> float secondary scale
  |
  |  for each of 128 elements:
  |    y[i] = (qval + d) * s1 * s2
  |
  v
float2 v (2 x FP32, 8 bytes)  -- per thread per call
```

### Byte-to-4-values Unpack Detail

```
qs[j] (1 byte, 8 bits)
  bits: [7:6] [5:4] [3:2] [1:0]
         q3    q2    q1    q0

q0 = (qs[j] >> 0) & 0x03   // bits 1:0
q1 = (qs[j] >> 2) & 0x03   // bits 3:2
q2 = (qs[j] >> 4) & 0x03   // bits 5:4
q3 = (qs[j] >> 6) & 0x03   // bits 7:6

Each q in {0, 1, 2, 3}  (2-bit unsigned)
```
