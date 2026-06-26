---
title: KVarN CLI Flag -- File Ownership & Registration Points
---

```mermaid
graph TD
    subgraph "Arg Parser (common/)"
        ARG_H["common/arg.h<br/>- common_arg class<br/>- common_params struct<br/>- common_params_context"]
        ARG_CPP["common/arg.cpp<br/>- kv_cache_types vector (line 300)<br/>  <b>+GGML_TYPE_Q2_KVARN (NEW)</b><br/>- kv_cache_type_from_str() (line 312)<br/>- --cache-type-k def (line 2134)<br/>- --cache-type-v def (line 2147)<br/>- --spec-draft-type-k def (line 3589)<br/>- --spec-draft-type-v def (line 3602)"]
        COMMON_H["common/common.h<br/>- common_params::cache_type_k (line 579)<br/>- common_params::cache_type_v (line 580)<br/>- common_params_speculative::cache_type_k (line 332)<br/>- common_params_speculative::cache_type_v (line 333)"]
        COMMON_CPP["common/common.cpp<br/>- common_init_from_params()<br/>  copies cache_type_k -> cparams.type_k (line 1601)"]
    end

    subgraph "llama-cli (tools/cli/)"
        CLI_MAIN["tools/cli/main.cpp<br/>- entry point -> llama_cli()"]
        CLI_CPP["tools/cli/cli.cpp<br/>- llama_cli() at line 367<br/>- common_params_parse() at line 374<br/>- ctx_server.load_model(params)"]
    end

    subgraph "llama-server (tools/server/)"
        SVR_CTX["tools/server/server-context.cpp<br/>- propagates cache_type_k/v to draft<br/>  model params (lines 1092-1237)"]
    end

    subgraph "llama-bench (tools/llama-bench/)"
        BENCH_CPP["tools/llama-bench/llama-bench.cpp<br/>- ggml_type_from_name() (line 478)<br/>  <b>+q2_kvarn case (NEW)</b><br/>- parse_cmd_params() (line 507)<br/>  parses -ctk/-ctv (lines 613-649)"]
    end

    subgraph "Inference Engine (src/)"
        LLAMA_H["include/llama.h<br/>- llama_context_params struct<br/>- type_k, type_v fields"]
        CPARAMS_H["src/llama-cparams.h<br/>- llama_cparams struct<br/>- type_k, type_v"]
        CONTEXT_CPP["src/llama-context.cpp<br/>- context_init() copies params<br/>- validates KVarN params"]
        KV_CACHE_H["src/llama-kv-cache.h<br/>- llama_kv_cache class"]
        KV_CACHE_CPP["src/llama-kv-cache.cpp<br/>- cache construction<br/>- VarNTileProcessor init"]
    end

    ARG_CPP --> ARG_H : implements
    ARG_CPP --> COMMON_H : writes to
    COMMON_CPP --> COMMON_H : reads from
    CLI_CPP --> ARG_CPP : calls common_params_parse()
    CLI_CPP --> COMMON_CPP : calls common_init_from_params()
    CLI_CPP --> SVR_CTX : ctx_server.load_model()
    SVR_CTX --> COMMON_H : reads cache_type_k/v
    SVR_CTX --> LLAMA_H : passes to llama_init_from_model()
    BENCH_CPP --> BENCH_CPP : self-contained parser
    COMMON_CPP --> LLAMA_H : passes to llama_init_from_model()
    LLAMA_H --> CPARAMS_H : copied during init
    CPARAMS_H --> KV_CACHE_H : passed to cache ctor
    KV_CACHE_H --> KV_CACHE_CPP : implemented in

    style ARG_CPP fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style BENCH_CPP fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style COMMON_H fill:#bbdefb,stroke:#1565c0
    style COMMON_CPP fill:#e3f2fd,stroke:#1565c0
```

### File Ownership

| File | Owns | Change Required |
|---|---|---|
| `common/arg.cpp` | `kv_cache_types` vector, `kv_cache_type_from_str()`, `--cache-type-k/v` defs | Add `GGML_TYPE_Q2_KVARN` to `kv_cache_types` |
| `common/common.h` | `common_params::cache_type_k/v` fields | None (field already exists) |
| `common/common.cpp` | `common_init_from_params()` copy logic | None (generic `type_k` copy) |
| `tools/cli/cli.cpp` | `llama_cli()` entry, `common_params_parse()` call | None (uses shared parser) |
| `tools/server/server-context.cpp` | Server context, draft model propagation | None (reads `cache_type_k/v` generically) |
| `tools/llama-bench/llama-bench.cpp` | Custom `ggml_type_from_name()`, `parse_cmd_params()` | Add `"q2_kvarn"` case to `ggml_type_from_name()` |
| `include/llama.h` | `llama_context_params` struct | None (type_k/v already exist) |
| `src/llama-context.cpp` | Validation + context init | None (already handles Q2_KVARN) |

### Registration Points Summary

| # | File | Line | What to add |
|---|---|---|---|
| 1 | `common/arg.cpp` | 300-310 | `GGML_TYPE_Q2_KVARN` to `kv_cache_types` vector |
| 2 | `tools/llama-bench/llama-bench.cpp` | 478-504 | `if (s == "q2_kvarn") return GGML_TYPE_Q2_KVARN;` |
