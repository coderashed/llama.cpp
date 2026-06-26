---
title: Q2_KVARN Type Definition -- Class / Struct Diagram
---

```mermaid
classDiagram

    class ggml_type {
        <<enum (ggml/include/ggml.h)>>
        +GGML_TYPE_F32 = 0
        +GGML_TYPE_Q4_0 = 2
        +...
        +GGML_TYPE_NVFP4 = 40
        +GGML_TYPE_Q1_0 = 41
        +GGML_TYPE_Q2_KVARN = 42      <<NEW>>
        +GGML_TYPE_COUNT = 43         <<UPDATED>>
    }

    class block_q2_kvarn {
        <<struct (ggml/src/ggml-common.h)>>
        +uint8_t  qs[32]    // 128 x 2-bit packed            offset=0    size=32
        +ggml_half d         // FP16 zeropoint (not negated)  offset=32   size=2
        +ggml_half s1        // FP16 primary scale             offset=34   size=2
        +ggml_half s2        // FP16 secondary scale (row)     offset=36   size=2
        // total sizeof = 38 bytes
        // blck_size = 128, bpw = 38*8/128 = 2.375 (<FP16 scales)
        // target bpw = 2.25 (with FP8 for s1/s2, future optimization)
    }

    class ggml_type_traits {
        <<struct (ggml/include/ggml.h)>>
        +const char*        type_name
        +int64_t            blck_size
        +int64_t            blck_size_interleave
        +size_t             type_size
        +bool               is_quantized
        +ggml_to_float_t    to_float
        +ggml_from_float_t  from_float_ref
    }

    class type_traits_entry {
        <<static const (ggml/src/ggml.c)>>
        +type_name      = "q2_kvarn"
        +blck_size      = 128
        +type_size      = sizeof(block_q2_kvarn)   // 38 bytes
        +is_quantized   = true
        +to_float       = dequantize_row_q2_kvarn
        +from_float_ref = quantize_row_q2_kvarn_ref
    }

    class QuantizeFunctions {
        <<function group (ggml/src/ggml-quants.c)>>
        +size_t quantize_q2_kvarn(src, dst, nrow, n_per_row, quant_weights)
        +void   quantize_row_q2_kvarn_ref(x, y, k)
        +size_t quantize_row_q2_kvarn(x, y, k)        // alias for ref
    }

    class DequantizeFunctions {
        <<function group (ggml/src/ggml-quants.c)>>
        +void dequantize_row_q2_kvarn(x, y, k)
        // Scalar: for each block, unpack 2-bit, apply:
        //   val[j] = (qval + d) * s1 * s2   [FP32 output]
    }

    ggml_type *--> type_traits_entry : "type_traits[GGML_TYPE_Q2_KVARN]"
    type_traits_entry --> QuantizeFunctions : from_float_ref
    type_traits_entry --> DequantizeFunctions : to_float
    type_traits_entry ..> block_q2_kvarn : type_size = sizeof

    note for block_q2_kvarn "qs[32]: 2-bit packed layout\n  byte 0: elem[0..3] (bits 0-1 elem0, 2-3 elem1, ...)\n  byte 31: elem[124..127]\n\nd:    FP16 zeropoint (stored as-is, sign = conventional -zpt)\ns1:   FP16 scale (channel/group, absorbs VarN S_c)s2:   FP16 scale (token/group, VarN S_r, initially 1.0)"
```

### Data Flow

```
FP32 input [128 elements]
       |
       | quantize_row_q2_kvarn_ref()
       v
block_q2_kvarn {
    d  = quant_min                     // FP16, 1 per channel per group
    s1 = (quant_max - quant_min) / 3   // FP16, range scale
    s2 = 1.0f                          // FP16, identity (set by VarN later)
    qs = pack_2bit( clamp( round((x[i] - d) / s1), 0, 3 ) )
}
       |
       | dequantize_row_q2_kvarn()
       v
FP32 output [128 elements]
       val[i] = (qval[i] + d) * s1 * s2
```
