---
title: KVarN CLI Flag -- Validation & Error Handling
---

```mermaid
graph TD
    subgraph "CLI Arg Parsing Errors"
        A1["--cache-type-k q2_kvarn<br/>but q2_kvarn NOT in kv_cache_types?"]
        A2["--cache-type-k q2_kvarn<br/>but ggml_type_name(Q2_KVARN)<br/>returns unexpected string?"]
        A3["--cache-type-k invalid_type<br/>e.g. --cache-type-k q8_1"]
        A4["--cache-type-k<br/>without value (missing arg)"]
        A5["--cache-type-k q2_kvarn<br/>in llama-bench but<br/>ggml_type_from_name()<br/>doesn't have q2_kvarn case"]
    end

    subgraph "common/arg.cpp Error Handling"
        H1["kv_cache_type_from_str()<br/>throws runtime_error<br/>'Unsupported cache type: ...'"]
        H2["common_params_parse_ex()<br/>catches exception at<br/>arg.cpp:600-601<br/>-> print_usage() + exit(1)"]
        H3["Missing value:<br/>common_arg handler expects<br/>string arg; parser detects<br/>missing value and errors"]
    end

    subgraph "llama-bench Error Handling"
        B1["ggml_type_from_name()<br/>returns GGML_TYPE_COUNT<br/>on unknown string"]
        B2["parse_cmd_params()<br/>sets invalid_param = true<br/>-> print_usage() + exit(1)"]
    end

    subgraph "Context Init Validation (src/llama-context.cpp)"
        V1["type_k == GGML_TYPE_Q2_KVARN?"]
        V2["kvarn_group_size == 0?"]
        V3["kvarn_group_size does not<br/>divide n_embd_head_k?"]
        V4["sink + recent >= n_ctx?"]
        V5["varn_iterations == 0?"]
        V6["varn_iterations > 100?"]
        V7["group_size > 1024?"]
    end

    subgraph "Context Init Error Actions"
        E1["LOG_ERROR + return nullptr<br/>(llama_init_from_model fails)"]
        E2["LOG_WARN + clamp to default"]
        E3["LOG_WARN + clamp to 1024"]
        E4["LOG_WARN + clamp to 100"]
        E5["LOG_WARN + clamp recent<br/>to n_ctx - sink - 1"]
    end

    A1 -->|"ggml_type_name mismatch"| H1
    A2 -->|"ggml_type_name mismatch"| H1
    A3 -->|"no match in kv_cache_types"| H1
    A4 -->|"argc/argv underflow"| H3
    A5 -->|"returns GGML_TYPE_COUNT"| B1

    H1 --> H2
    B1 --> B2

    V1 -->|"yes"| V2
    V1 -->|"no"| E2
    V2 -->|"== 0"| E1
    V2 -->|"> 0"| V3
    V3 -->|"fails"| E1
    V3 -->|"passes"| V4
    V4 -->|">= n_ctx"| E5
    V4 -->|"< n_ctx"| V5
    V5 -->|"== 0"| E2
    V5 -->|">= 1"| V6
    V6 -->|"> 100"| E4
    V6 -->|"<= 100"| V7
    V7 -->|"> 1024"| E3
    V7 -->|"<= 1024"| E2

    style H1 fill:#ffebee,stroke:#c62828
    style H2 fill:#ffebee,stroke:#c62828
    style B1 fill:#fff3e0,stroke:#e65100
    style B2 fill:#fff3e0,stroke:#e65100
    style E1 fill:#ffebee,stroke:#c62828
    style E2 fill:#fff9c4,stroke:#f9a825
```

### Error Handling Table

| # | Scenario | Where | Error | User Sees |
|---|---|---|---|---|
| 1 | `--cache-type-k q2_kvarn` but `GGML_TYPE_Q2_KVARN` not in `kv_cache_types` | `common/arg.cpp:318` | `runtime_error("Unsupported cache type: q2_kvarn")` | `error: Unsupported cache type: q2_kvarn` + usage |
| 2 | `--cache-type-k q8_1` (nonexistent type) | `common/arg.cpp:318` | `runtime_error("Unsupported cache type: q8_1")` | `error: Unsupported cache type: q8_1` + usage |
| 3 | `--cache-type-k` without value | `common/arg.cpp` parser | Missing argument detection | `error: ...` + usage |
| 4 | `--cache-type-k q2_kvarn` in llama-bench, no case in `ggml_type_from_name()` | `llama-bench.cpp:504` | Returns `GGML_TYPE_COUNT` | `invalid_param = true` + usage |
| 5 | `type_k=Q2_KVARN` but `kvarn_group_size == 0` | `src/llama-context.cpp` | `LOG_ERROR` + `return nullptr` | `llama_init_from_model() failed` |
| 6 | `kvarn_group_size` does not divide `n_embd_head_k` | `src/llama-context.cpp` | `LOG_ERROR` + `return nullptr` | `llama_init_from_model() failed` |
| 7 | `sink + recent >= n_ctx` | `src/llama-context.cpp` | `LOG_WARN` + clamp recent | Warning log message |
| 8 | `varn_iterations == 0` | `src/llama-context.cpp` | `LOG_WARN` + default to 8 | Warning log message |
| 9 | `varn_iterations > 100` | `src/llama-context.cpp` | `LOG_WARN` + clamp to 100 | Warning log message |
| 10 | `group_size > 1024` | `src/llama-context.cpp` | `LOG_WARN` + clamp to 1024 | Warning log message |

### Validation Flow (llama_init_from_model)

```
type_k == Q2_KVARN?
  |-- NO:  zero KVarN cparams, skip validation
  |-- YES:
       |-- kvarn_group_size == 0?  -> ERROR (return nullptr)
       |-- group_size divides head_dim?  -> ERROR (return nullptr)
       |-- sink + recent >= n_ctx?  -> WARN (clamp recent)
       |-- varn_iterations == 0?  -> WARN (default to 8)
       |-- varn_iterations > 100?  -> WARN (clamp to 100)
       |-- group_size > 1024?  -> WARN (clamp to 1024)
       |-- PASS: copy KVarN params to cparams, proceed
```

### Key Design Decisions

1. **Fail-fast for CLI args**: `kv_cache_type_from_str()` throws immediately on unknown type. The exception propagates to `common_params_parse_ex()` which prints usage and exits. No silent fallback.

2. **Fail-soft for context params**: Validation in `llama_init_from_model()` uses warnings + clamping for out-of-range values, only hard-errors on structural violations (zero group_size, non-divisible group_size). This matches the existing pattern in `08-kvarn-params-exceptions.md`.

3. **llama-bench is separate**: The bench tool has its own parser. If the `q2_kvarn` case is forgotten there, the type string silently returns `GGML_TYPE_COUNT` and the tool exits with usage. This is a compile-time-visible gap (no test coverage for bench's parser).
