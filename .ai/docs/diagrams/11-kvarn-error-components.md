---
title: Error Decomposition Measurement -- File Ownership & Dependencies
---

```mermaid
graph TD
    subgraph "New Utility File"
        ERROR_UTIL["tests/test-q2-kvarn-error.cpp (NEW)<br/>compute_error_decomposition()<br/>topk_em_et_ratio()<br/>error_histogram()<br/>compare_quantizers()<br/>main() test harness"]
    end

    subgraph "Existing Test Files"
        TEST_QUANT["tests/test-q2-kvarn-quant.cpp<br/>compute_em_et()<br/>test_em_et_median_ratio()<br/>test_mse_finite()<br/>test_worst_token_em_et()"]
    end

    subgraph "Implementation Files"
        GGML_QUANTS_C["ggml/src/ggml-quants.c<br/>quantize_row_q2_kvarn_ref()<br/>dequantize_row_q2_kvarn()"]
        GGML_C["ggml/src/ggml.c<br/>ggml_quantize_chunk()<br/>type_traits[]"]
        GGML_COMMON_H["ggml/src/ggml-common.h<br/>block_q2_kvarn struct"]
    end

    subgraph "Research / Design"
        ERR_DECOMP["research/kvarn/02-error-decomposition.md<br/>E_M/E_D/E_T definitions"]
        DIAGRAMS["docs/diagrams/11-kvarn-error-*.md<br/>Design diagrams"]
    end

    subgraph "CMake Registration"
        CMAKE["tests/CMakeLists.txt<br/>llama_build_and_test(test-q2-kvarn-error)"]
    end

    ERROR_UTIL --> GGML_C : calls ggml_quantize_chunk()
    ERROR_UTIL --> ERR_DECOMP : implements E_M/E_D/E_T formulas
    ERROR_UTIL --> TEST_QUANT : reuses compute_em_et() pattern
    ERROR_UTIL --> CMAKE : registered as test target
    DIAGRAMS --> ERR_DECOMP : references metric definitions
    DIAGRAMS --> ERROR_UTIL : documents function signatures

    style ERROR_UTIL fill:#d4f,stroke:#333,stroke-width:2px
    style TEST_QUANT fill:#d4f,stroke:#333,stroke-width:2px
    style GGML_QUANTS_C fill:#adf,stroke:#333,stroke-width:2px
    style GGML_C fill:#adf,stroke:#333,stroke-width:2px
    style CMAKE fill:#fda,stroke:#333,stroke-width:2px
```

### Dependency Graph

```
test-q2-kvarn-error.cpp
    |
    |-- ggml.h              (ggml_quantize_chunk, ggml_type_traits)
    |-- ggml-cpu.h          (ggml_cpu_init)
    |-- math.h              (sqrt, isfinite)
    |-- algorithm           (std::sort, std::nth_element)
    |-- vector              (std::vector)
    |-- numeric             (std::accumulate)
    |-- cstdio              (printf for summary table)
    |
    v
ggml_quantize_chunk(GGML_TYPE_Q2_KVARN, ...)
ggml_quantize_chunk(GGML_TYPE_Q4_0, ...)
    |
    v
ggml_get_type_traits(type)->to_float(...)
```

### CMake Registration (tests/CMakeLists.txt)

```cmake
# Add after existing kvarn test entries (~line 267):
llama_build_and_test(test-q2-kvarn-error.cpp)
```

### File Responsibilities

| File | Provides | Responsibility |
|------|----------|----------------|
| `test-q2-kvarn-error.cpp` | `compute_error_decomposition()` | Per-token E_M, E_D, E_T computation |
| | `topk_em_et_ratio()` | Top-k% mean ratio calculation |
| | `error_histogram()` | Histogram binning of ratios |
| | `compare_quantizers()` | Cross-quantizer comparison |
| | `main()` | Test harness with summary table output |
| `test-q2-kvarn-quant.cpp` | `compute_em_et()` | Existing E_M/E_T (ratios only, no per-component) |
| `CMakeLists.txt` | Test registration | `llama_build_and_test(test-q2-kvarn-error)` |
