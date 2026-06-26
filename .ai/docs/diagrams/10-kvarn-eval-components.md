---
title: KVarN Pseudo-Decode Evaluation -- File Ownership & Dependencies
---

```mermaid
graph TD
    subgraph "Test Layer (tests/)"
        TEST_CPP["test-kvarn-pseudo-decode.cpp (NEW)<br/>PseudoDecodeHarness<br/>BlockProcessor, QuantizeStep<br/>ErrorMeasurer, BaselineRunner<br/>ErrorReporter, AttentionOutputHook<br/>ModelLoader"]
        TEST_CMAKE["CMakeLists.txt<br/>llama_build_and_test() registration<br/>LABEL 'model'<br/>FIXTURES_REQUIRED test-download-model"]
    end

    subgraph "Application Layer (src/)"
        KV_CACHE_H["llama-kv-cache.h<br/>llama_kv_cache class<br/>cpy_k() / cpy_v() API<br/>quantize interface"]
        KV_CACHE_CPP["llama-kv-cache.cpp<br/>- cpy_k() implementation<br/>- VarNTileProcessor<br/>- quantize_row_q2_kvarn_varn() calls<br/>- build_rope_shift()"]
        KVARN_H["llama-kvarn.h<br/>kvarn_variance_normalize()<br/>declaration"]
        KVARN_CPP["llama-kvarn.cpp<br/>VarN algorithm implementation"]
        CONTEXT_H["llama-context.h<br/>llama_context API<br/>llama_decode(), llama_batch"]
        CONTEXT_CPP["llama-context.cpp<br/>decode loop (line 1680)<br/>attention computation"]
    end

    subgraph "Low-Level Layer (ggml/)"
        GGML_COMMON_H["ggml-common.h<br/>block_q2_kvarn struct"]
        GGML_QUANTS_C["ggml-quants.c<br/>- quantize_row_q2_kvarn_ref()<br/>- quantize_row_q2_kvarn_varn()<br/>- dequantize_row_q2_kvarn()"]
        GGML_C["ggml.c<br/>- type_traits registration<br/>- ggml_quantize_chunk() dispatch"]
    end

    subgraph "Research / Design"
        RESEARCH_MD[".ai/research/kvarn/<br/>08-pseudo-decode-eval.md<br/>Evaluation methodology"]
        UNDONE_MD[".ai/undone/KVARN_UNDONE.md<br/>Item 10 exit criterion"]
    end

    subgraph "Build & Test Infrastructure"
        CMAKE_TESTS["tests/CMakeLists.txt<br/>llama_build_and_test()"]
        CI_YML[".github/workflows/build.yml<br/>CI test execution"]
        MODEL_FIXTURE["test-download-model<br/>tinyllamas/stories15M-q4_0.gguf<br/>or Qwen3-4B for verification"]
    end

    subgraph "Output Artifacts"
        CSV_OUT["error_curve.csv<br/>block_idx, context_length,<br/>layer_0_error, ..., layer_N_error"]
        JSON_OUT["error_curve.json<br/>structured error data<br/>per layer, per head, per block"]
        STDOUT["stdout summary<br/>monotonic check<br/>KVarN vs KIVI comparison"]
    end

    TEST_CPP -- "includes" --> KVARN_H
    TEST_CPP -- "includes" --> CONTEXT_H
    TEST_CPP -- "includes" --> KV_CACHE_H
    TEST_CPP -- "links" --> KVARN_CPP
    TEST_CPP -- "links" --> CONTEXT_CPP
    TEST_CPP -- "links" --> KV_CACHE_CPP
    TEST_CPP -- "links" --> GGML_QUANTS_C
    TEST_CPP -- "writes" --> CSV_OUT
    TEST_CPP -- "writes" --> JSON_OUT
    TEST_CPP -- "prints" --> STDOUT

    KV_CACHE_CPP -- "calls" --> KVARN_H
    KVARN_H --> KVARN_CPP
    KV_CACHE_CPP -- "calls" --> GGML_QUANTS_C
    GGML_QUANTS_C -- "writes" --> GGML_COMMON_H
    KV_CACHE_CPP -- "ggml_set_rows" --> GGML_C

    TEST_CMAKE -- "registered in" --> CMAKE_TESTS
    CMAKE_TESTS -- "depends on" --> MODEL_FIXTURE
    CMAKE_TESTS -- "executed by" --> CI_YML

    RESEARCH_MD -- "informs design of" --> TEST_CPP
    UNDONE_MD -- "defines exit criterion for" --> TEST_CPP

    style TEST_CPP fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style KV_CACHE_CPP fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style KVARN_CPP fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style GGML_QUANTS_C fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style RESEARCH_MD fill:#fff3e0,stroke:#e65100,stroke-width:1px
    style UNDONE_MD fill:#fff3e0,stroke:#e65100,stroke-width:1px
```

### File Dependency Table

| File | Role | Depends On |
|------|------|------------|
| `tests/test-kvarn-pseudo-decode.cpp` | Main test harness (NEW) | `llama.h`, `llama-kvarn.h`, `llama-context.h`, `llama-kv-cache.h` |
| `tests/CMakeLists.txt` | Test registration | `llama_build_and_test()` function |
| `src/llama-kvarn.h` | VarN function declarations | None (standalone header) |
| `src/llama-kvarn.cpp` | VarN algorithm | `llama-kvarn.h` |
| `src/llama-kv-cache.cpp` | KV-cache quantize path | `llama-kvarn.h`, `ggml-quants.h` |
| `ggml/src/ggml-quants.c` | Quantize/dequantize primitives | `ggml-common.h` |
| `ggml/src/ggml-common.h` | `block_q2_kvarn` struct | None |

### Build Registration (CMakeLists.txt addition)

```cmake
# In tests/CMakeLists.txt, alongside other kvarn tests:
llama_build_and_test(test-kvarn-pseudo-decode.cpp LABEL "model"
    ARGS -m "${MODEL_DEST}" -p "The meaning of life is" -b 128 -t q2_kvarn)
set_tests_properties(test-kvarn-pseudo-decode PROPERTIES
    FIXTURES_REQUIRED test-download-model)
```

### CLI Interface

```
Usage: test-kvarn-pseudo-decode [options]

Options:
  -m <path>         Model file (GGUF format)
  -p <text>         Prompt text (default: wikitext sample)
  -b <int>          Block size in tokens (default: 128)
  -t <type>         Quantization type: q2_kvarn | q4_0 (default: q2_kvarn)
  --kivi            Also run KIVI Q4_0 comparison
  --csv <path>      Write error curve CSV
  --json <path>     Write error curve JSON
  -v                Verbose output (per-head errors)
  -h                Show help

Exit code:
  0 = all checks passed
  1 = error curve not monotonically increasing
  2 = KVarN curve not below KIVI at all points
```
