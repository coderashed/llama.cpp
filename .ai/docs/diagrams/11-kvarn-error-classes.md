---
title: Error Decomposition Measurement -- Function Signatures & Data Flow
---

```mermaid
classDiagram

    class compute_em_et {
        <<static void (tests/test-q2-kvarn-quant.cpp:26)>>
        +const float* orig          // FP32 original K matrix, [n_tokens x dim]
        +const float* dequant       // FP32 dequantized K matrix, [n_tokens x dim]
        +int n_tokens               // number of tokens (rows)
        +int dim                    // head dimension (columns)
        +float* ratios_out          // output array, [n_tokens] E_M/E_T per token
        // For each token:
        //   norm_K  = sqrt(sum(orig^2))
        //   norm_Kdq = sqrt(sum(dequant^2))
        //   dot     = sum(orig * dequant)
        //   cos_theta = dot / (norm_K * norm_Kdq + 1e-30)
        //   E_M = (norm_K - norm_Kdq)^2
        //   E_D = 2 * norm_K * norm_Kdq * (1 - cos_theta)
        //   E_T = E_M + E_D
        //   ratios_out[t] = E_M / (E_T + 1e-30)
    }

    class compute_error_decomposition {
        <<utility function (NEW)>>
        +const float* orig          // FP32 original K matrix
        +const float* dequant       // FP32 dequantized K matrix
        +int n_tokens               // number of tokens
        +int dim                    // head dimension
        +float* em_out              // output: E_M per token, [n_tokens]
        +float* ed_out              // output: E_D per token, [n_tokens]
        +float* et_out              // output: E_T per token, [n_tokens]
        // Returns: void
        // Full decomposition: writes all three components per token
    }

    class topk_em_et_ratio {
        <<utility function (NEW)>>
        +const float* ratios        // E_M/E_T ratios, [n_tokens]
        +int n_tokens               // number of tokens
        +float percentile           // e.g. 0.05 for top 5%
        // Returns: mean(E_M/E_T) over the top `percentile` fraction
        // Sorts ratios descending, takes top ceil(n_tokens * percentile) tokens
        // Computes mean of their E_M/E_T values
    }

    class error_histogram {
        <<utility function (NEW)>>
        +const float* ratios        // E_M/E_T ratios, [n_tokens]
        +int n_tokens               // number of tokens
        +int n_bins                 // number of histogram bins (e.g. 10)
        +float* bin_edges_out       // output: bin boundaries, [n_bins + 1]
        +int* bin_counts_out        // output: count per bin, [n_bins]
        // Returns: void
        // Evenly spaced bins from 0.0 to 1.0
        // Counts tokens whose E_M/E_T falls in each bin
    }

    class compare_quantizers {
        <<verification function (NEW)>>
        +const float* orig          // FP32 original K matrix
        +const float* dequant_a     // dequantized by quantizer A (e.g. Q4_0)
        +const float* dequant_b     // dequantized by quantizer B (e.g. Q2_KVARN)
        +int n_tokens               // number of tokens
        +int dim                    // head dimension
        +float percentile           // e.g. 0.05 for top 5%
        // Returns: struct { float ratio_a, float ratio_b, bool kvarn_wins }
        // Computes top-k% E_M/E_T for both quantizers
        // Asserts ratio_b < ratio_a (KVarN suppresses magnitude errors better)
    }

    class error_summary {
        <<output struct (NEW)>>
        +float median_ratio         // median(E_M/E_T) across all tokens
        +float mean_ratio           // mean(E_M/E_T) across all tokens
        +float top_5pct_ratio      // mean(E_M/E_T) for top 5% worst tokens
        +float top_1pct_ratio      // mean(E_M/E_T) for top 1% worst tokens
        +float worst_ratio          // max(E_M/E_T) across all tokens
        +float mean_em             // mean(E_M) across all tokens
        +float mean_ed             // mean(E_D) across all tokens
        +float mean_et             // mean(E_T) across all tokens
        +int* histogram_counts     // histogram bin counts, [n_bins]
        +float* histogram_edges    // histogram bin edges, [n_bins + 1]
    }

    compute_error_decomposition --> compute_em_et : extends with per-component output
    topk_em_et_ratio --> compute_em_et : consumes ratios
    error_histogram --> compute_em_et : consumes ratios
    compare_quantizers --> compute_error_decomposition : uses per quantizer
    compare_quantizers --> topk_em_et_ratio : uses for comparison
    error_summary --> error_histogram : contains histogram
```

### Data Flow

```
FP32 orig [n_tokens x dim]    FP32 dequant [n_tokens x dim]
    |                                |
    +----------+---------------------+
               |
               v
    compute_error_decomposition()
               |
               +---> em_out[]  (magnitude error per token)
               +---> ed_out[]  (directional error per token)
               +---> et_out[]  (total error per token)
               |
               v
         ratios[t] = em_out[t] / et_out[t]
               |
               +---> topk_em_et_ratio(ratios, 0.05)  -> top 5% mean
               +---> topk_em_et_ratio(ratios, 0.01)  -> top 1% mean
               +---> error_histogram(ratios, 10)     -> 10-bin histogram
               +---> median(ratios)                   -> median ratio
               |
               v
         error_summary struct
               |
               v
         compare_quantizers(orig, dequant_Q4_0, dequant_KVarN)
               -> asserts KVarN top-5% E_M/E_T < Q4_0 top-5% E_M/E_T
```
