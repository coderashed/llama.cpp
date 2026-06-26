---
title: Q2_KVARN Quantization -- File Ownership & Registration
---

```mermaid
graph TD
    subgraph "Implementation Files"
        GGML_COMMON_H["ggml-common.h<br/>block_q2_kvarn struct"]
        GGML_QUANTS_C["ggml-quants.c<br/>quantize_row_q2_kvarn_ref()<br/>dequantize_row_q2_kvarn()<br/>quantize_q2_kvarn()"]
        GGML_C["ggml.c<br/>type_traits[GGML_TYPE_Q2_KVARN] entry<br/>ggml_quantize_chunk dispatch"]
    end

    subgraph "Test Files"
        TEST_CPP["tests/test-q2-kvarn-type.cpp<br/>Test 4: roundtrip RMSE<br/>Test 6 (NEW): E_M/E_T verification"]
    end

    subgraph "Research / Design"
        ERR_DECOMP["research/kvarn/02-error-decomposition.md<br/>E_M/E_T definition"]
        PIPELINE["research/kvarn/05-kvarn-pipeline.md<br/>RTN step in pipeline"]
        DIAGRAMS["docs/diagrams/03-q2-kvarn-quant-*.md<br/>Design diagrams"]
    end

    GGML_QUANTS_C --> GGML_COMMON_H : uses block_q2_kvarn
    GGML_C --> GGML_QUANTS_C : registers from_float_ref
    TEST_CPP --> GGML_C : calls ggml_quantize_chunk
    TEST_CPP --> ERR_DECOMP : implements E_M/E_T metric
    DIAGRAMS --> ERR_DECOMP : references metric
    DIAGRAMS --> PIPELINE : references pipeline

    style TEST_CPP fill:#d4f,stroke:#333,stroke-width:2px
    style GGML_QUANTS_C fill:#adf,stroke:#333,stroke-width:2px
    style GGML_C fill:#adf,stroke:#333,stroke-width:2px
```

### Registration Chain

```
ggml_quantize_chunk()
    -> switch(GGML_TYPE_Q2_KVARN)
        -> quantize_q2_kvarn()          [ggml-quants.c:2123]
            -> quantize_row_q2_kvarn_ref()  [ggml-quants.c:74]

ggml_type_traits[GGML_TYPE_Q2_KVARN]
    .from_float_ref = quantize_row_q2_kvarn_ref
    .to_float       = dequantize_row_q2_kvarn
    .type_name      = "q2_kvarn"
    .blck_size      = 128
    .type_size      = 38
    .is_quantized   = true
```
