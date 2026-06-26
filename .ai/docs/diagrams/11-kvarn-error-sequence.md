---
title: Error Decomposition Measurement -- Computation Sequence
---

```mermaid
sequenceDiagram
    participant T as Test Harness<br/>(test-q2-kvarn-error.cpp)
    participant Q as Quantizer<br/>(ggml_quantize_chunk)
    participant DQ as Dequantizer<br/>(to_float)
    participant ED as compute_error_decomposition
    participant TK as topk_em_et_ratio
    participant EH as error_histogram
    participant CQ as compare_quantizers

    Note over T: Phase 1: Generate K matrix
    T->>T: Create synthetic K matrix<br/>(random normal + outlier channels)

    Note over T: Phase 2: Quantize with both methods
    T->>Q: ggml_quantize_chunk(Q2_KVARN, src, dst_kvarn, ...)
    Q-->>T: quantized KVarN buffer
    T->>Q: ggml_quantize_chunk(GGML_TYPE_Q4_0, src, dst_q40, ...)
    Q-->>T: quantized Q4_0 buffer

    Note over T: Phase 3: Dequantize both
    T->>DQ: to_float(dst_kvarn, dequant_kvarn, nelem)
    DQ-->>T: dequant_kvarn (FP32)
    T->>DQ: to_float(dst_q40, dequant_q40, nelem)
    DQ-->>T: dequant_q40 (FP32)

    Note over T: Phase 4: Compute per-token decomposition
    T->>ED: compute_error_decomposition(src, dequant_kvarn, n_tokens, dim, em, ed, et)
    Note over ED: For each token t:
    Note over ED:   norm_K  = sqrt(sum(src[t*..]^2))
    Note over ED:   norm_Kdq = sqrt(sum(dequant[t*..]^2))
    Note over ED:   dot     = sum(src[t*..] * dequant[t*..])
    Note over ED:   cos_theta = dot / (norm_K * norm_Kdq + 1e-30)
    Note over ED:   cos_theta = clamp(cos_theta, -1.0, 1.0)
    Note over ED:   em[t] = (norm_K - norm_Kdq)^2
    Note over ED:   ed[t] = 2 * norm_K * norm_Kdq * (1 - cos_theta)
    Note over ED:   et[t] = em[t] + ed[t]
    ED-->>T: em[], ed[], et[]

    Note over T: Phase 5: Compute ratios
    T->>T: ratios[t] = em[t] / (et[t] + 1e-30)

    Note over T: Phase 6: Summary statistics
    T->>TK: topk_em_et_ratio(ratios, n_tokens, 0.05)
    Note over TK: Sort ratios descending
    Note over TK: Take top ceil(n_tokens * 0.05) tokens
    Note over TK: Return mean of their ratios
    TK-->>T: top_5pct_ratio

    T->>TK: topk_em_et_ratio(ratios, n_tokens, 0.01)
    TK-->>T: top_1pct_ratio

    T->>EH: error_histogram(ratios, n_tokens, 10, edges, counts)
    Note over EH: 10 bins from 0.0 to 1.0
    Note over EH: Count tokens per bin
    EH-->>T: histogram

    Note over T: Phase 7: Cross-quantizer comparison
    T->>CQ: compare_quantizers(src, dequant_q40, dequant_kvarn, n_tokens, dim, 0.05)
    Note over CQ: Compute top-5% E_M/E_T for Q4_0
    Note over CQ: Compute top-5% E_M/E_T for KVarN
    Note over CQ: Assert KVarN ratio < Q4_0 ratio
    CQ-->>T: comparison result

    Note over T: Phase 8: Output & assert
    Note over T: Print summary table:
    Note over T:   | Metric          | Q4_0  | KVarN |
    Note over T:   | median E_M/E_T  | 0.42  | 0.31  |
    Note over T:   | top 5% E_M/E_T  | 0.78  | 0.52  |
    Note over T:   | top 1% E_M/E_T  | 0.91  | 0.63  |
    Note over T: Assert top_5pct_kvarn < top_5pct_q40
```

### Computation Pipeline

```
K matrix (FP32)  --[quantize]-->  quantized buffer  --[dequant]-->  K_dq (FP32)
     |                                                                   |
     +----------------------------+--------------------------------------+
                                  |
                                  v
                    compute_error_decomposition()
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
                  E_M[]         E_D[]         E_T[]
                    |             |             |
                    +------+------+             |
                           |                    |
                           v                    |
                    ratios[t] = E_M/E_T --------+
                           |
              +------------+------------+
              |            |            |
              v            v            v
         top-5% mean   top-1% mean   histogram
              |            |            |
              +------+-----+            |
                     |                  |
                     v                  v
              error_summary struct --> printed table
```
