---
title: Q2_KVARN Quantization -- Function Signatures & Data Flow
---

```mermaid
classDiagram

    class compute_s2_scale {
        <<static float (ggml/src/ggml-quants.c:76)>>
        +const float* x          // FP32 input, [qk] elements
        +const uint8_t* qs       // packed 2-bit quantized values
        +int qk                  // block size (128)
        +float min               // FP32 zeropoint
        +float half_range        // FP32 primary scale
        // Returns: norm_orig / norm_dq, or 1.0f if norm_dq near zero
        // Per-token L2 ratio preserving magnitude: dequant = (qval+d)*s1*s2
    }

    class quantize_row_q2_kvarn_ref {
        <<void (ggml/src/ggml-quants.c:93)>>
        +const float* x          // FP32 input, [k] elements
        +block_q2_kvarn* y       // quantized output, [k/QK2_KVARN] blocks
        +int64_t k               // total elements (must be multiple of 128)
        // Per block:
        //   min = min(x[i*128 .. i*128+127])
        //   max = max(x[i*128 .. i*128+127])
        //   d   = min                          (FP16 zeropoint)
        //   s1  = (max-min)/3                  (FP16 scale)
        //   s2  = compute_s2_scale()           (L2 ratio, not identity)
        //   qs  = pack_2bit(round((x-d)/s1))   (clamped 0..3)
    }

    class quantize_q2_kvarn {
        <<size_t (ggml/src/ggml-quants.c:2123)>>
        +const float* src         // FP32 input, [nrow * n_per_row]
        +void* dst                // quantized output buffer
        +int64_t nrow             // number of rows
        +int64_t n_per_row        // elements per row (must be multiple of 128)
        +const float* quant_weights // optional importance weights (NULL = flat)
        // Returns: total bytes written = nrow * ggml_row_size(Q2_KVARN, n_per_row)
        // Dispatches to quantize_row_q2_kvarn_ref per row (or bulk if no weights)
    }

    class block_q2_kvarn {
        <<struct (ggml/src/ggml-common.h)>>
        +uint8_t  qs[32]    // 128 x 2-bit packed
        +ggml_half d         // FP16 zeropoint (min)
        +ggml_half s1        // FP16 primary scale ((max-min)/3)
        +ggml_half s2        // FP16 secondary scale (1.0, set by VarN)
    }

    class EM_ET_Metric {
        <<verification function>>
        +float compute_EM(float* orig, float* dequant, int n_tokens, int dim)
        +float compute_ET(float* orig, float* dequant, int n_tokens, int dim)
        +float median_ratio(float* orig, float* dequant, int n_tokens, int dim)
        // E_M = (||K|| - ||K_dq||)^2 per token
        // E_D = 2*||K||*||K_dq||*(1 - cos(theta)) per token
        // E_T = sqrt(E_M + E_D) per token
        // Returns: median(E_M / E_T) across tokens
    }

    class ggml_type_traits_registration {
        <<static const (ggml/src/ggml.c)>>
        +type_name      = "q2_kvarn"
        +blck_size      = 128
        +type_size      = 38
        +is_quantized   = true
        +to_float       = dequantize_row_q2_kvarn
        +from_float_ref = quantize_row_q2_kvarn_ref
    }

    quantize_q2_kvarn --> quantize_row_q2_kvarn_ref : delegates per row
    quantize_row_q2_kvarn_ref --> block_q2_kvarn : writes
    block_q2_kvarn --> dequantize_row_q2_kvarn : read by
    dequantize_row_q2_kvarn --> EM_ET_Metric : feeds verification
    ggml_type_traits_registration --> quantize_row_q2_kvarn_ref : from_float_ref
```

### Data Flow

```
FP32 src [nrow x n_per_row]
    |
    | quantize_q2_kvarn()
    |   |-- quantize_row_q2_kvarn_ref() per block
    v
block_q2_kvarn[]  (38 bytes per 128 elements)
    |
    | dequantize_row_q2_kvarn()
    v
FP32 dst_f32 [nrow x n_per_row]
    |
    | EM_ET_Metric::median_ratio()
    v
E_M/E_T ratio  (must be < 0.5 for median token)
```
