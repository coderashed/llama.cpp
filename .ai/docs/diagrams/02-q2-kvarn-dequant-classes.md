---
title: Q2_KVARN Dequantization -- Function Signature & Data Flow
---

```mermaid
classDiagram

    class block_q2_kvarn {
        <<struct (ggml/src/ggml-common.h)>>
        +uint8_t  qs[32]    // 128 x 2-bit packed
        +ggml_half d         // FP16 zeropoint
        +ggml_half s1        // FP16 primary scale (column)
        +ggml_half s2        // FP16 secondary scale (row)
    }

    class dequantize_row_q2_kvarn {
        <<function (ggml/src/ggml-quants.c)>>
        +void dequantize_row_q2_kvarn(
        +    const block_q2_kvarn * GGML_RESTRICT x,
        +    float * GGML_RESTRICT y,
        +    int64_t k)
        //
        // Pre:  k % 128 == 0
        //       x points to k/128 valid blocks
        //       y points to k valid floats
        // Post: y[0..k-1] = dequantized values
        //       x is unmodified
    }

    class ggml_type_traits {
        <<struct (ggml/include/ggml.h)>>
        +ggml_to_float_t to_float
    }

    class float_array {
        <<output>>
        +float y[k]
        // y[i] = (qval + d) * s1 * s2
    }

    block_q2_kvarn --> dequantize_row_q2_kvarn : "read by"
    dequantize_row_q2_kvarn --> float_array : "writes"
    ggml_type_traits --> dequantize_row_q2_kvarn : ".to_float ="

    note for dequantize_row_q2_kvarn "Signature matches ggml_to_float_t typedef:\n  void (*)(const void *, float *, int64_t)\n\nk = total elements (not blocks)\nassert(k %% 128 == 0)\nnb = k / 128 blocks"
```

### Data Flow: block_q2_kvarn -> float

```
block_q2_kvarn (38 bytes)
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
float y[128]  (512 bytes)
```

### Byte-to-4-values unpack detail

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
