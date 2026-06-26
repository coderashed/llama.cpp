---
title: KVarN Context Params -- File Ownership & API Surface
---

```mermaid
graph TD
    subgraph "Public C API (include/)"
        LLAMA_H["include/llama.h<br/>- llama_context_params struct<br/>- kvarn_group_size (NEW)<br/>- kvarn_sink_tokens (NEW)<br/>- kvarn_recent_tokens (NEW)<br/>- kvarn_varn_iterations (NEW)<br/>- llama_context_default_params()"]
    end

    subgraph "Internal Context (src/)"
        CPARAMS_H["src/llama-cparams.h<br/>- llama_cparams struct<br/>- kvarn_group_size (NEW)<br/>- kvarn_sink_tokens (NEW)<br/>- kvarn_recent_tokens (NEW)<br/>- kvarn_varn_iterations (NEW)"]
        CONTEXT_CPP["src/llama-context.cpp<br/>- llama_context_default_params()<br/>  sets KVarN defaults<br/>- context_init()<br/>  copies params -> cparams<br/>- llama_init_from_model()<br/>  validates KVarN params"]
        KV_CACHE_H["src/llama-kv-cache.h<br/>- llama_kv_cache class<br/>- kvarn_group_size (NEW)<br/>- kvarn_sink_tokens (NEW)<br/>- kvarn_recent_tokens (NEW)<br/>- kvarn_varn_iterations (NEW)"]
        KV_CACHE_CPP["src/llama-kv-cache.cpp<br/>- llama_kv_cache ctor<br/>  receives KVarN params<br/>- VarNTileProcessor init<br/>  uses G and K<br/>- three-region layout<br/>  uses S and R"]
        KVARN_H["src/llama-kvarn.h<br/>- kvarn_variance_normalize()<br/>  receives K (iterations)"]
    end

    subgraph "Research Reference"
        RESEARCH["research/kvarn/<br/>06-three-region-layout.md<br/>09-experimental-setup.md"]
    end

    subgraph "API Surface"
        API1["llama_context_default_params()<br/>returns defaults"]
        API2["llama_init_from_model(model, params)<br/>validates + constructs"]
        API3["llama_context_params<br/>POD struct, ABI-stable"]
    end

    LLAMA_H --> CPARAMS_H : "params copied to cparams"
    CPARAMS_H --> KV_CACHE_H : "cparams passed to cache ctor"
    KV_CACHE_H --> KV_CACHE_CPP : "implemented in"
    CONTEXT_CPP --> LLAMA_H : "reads/writes struct"
    CONTEXT_CPP --> CPARAMS_H : "writes cparams"
    CONTEXT_CPP --> KV_CACHE_H : "creates cache"
    KV_CACHE_CPP --> KVARN_H : "uses K (iterations)"
    RESEARCH --> LLAMA_H : "informs defaults"

    style LLAMA_H fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style CONTEXT_CPP fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style CPARAMS_H fill:#bbdefb,stroke:#1565c0
    style KV_CACHE_H fill:#bbdefb,stroke:#1565c0
    style KV_CACHE_CPP fill:#e3f2fd,stroke:#1565c0
```

### File Ownership

| File | Owns | Responsibility |
|------|------|----------------|
| `include/llama.h` | `llama_context_params` fields | Public ABI; new fields appended at end |
| `src/llama-cparams.h` | `llama_cparams` fields | Internal copy; zeroed when type != Q2_KVARN |
| `src/llama-context.cpp` | Defaults + validation + copy | `llama_context_default_params()`, `context_init()`, `llama_init_from_model()` |
| `src/llama-kv-cache.h` | `llama_kv_cache` fields | Cache-level storage of KVarN config |
| `src/llama-kv-cache.cpp` | Cache construction | VarNTileProcessor init, three-region layout |
| `src/llama-kvarn.h` | `kvarn_variance_normalize()` | Receives K (iterations) as parameter |

### New API Surface

| Symbol | Type | File | Notes |
|--------|------|------|-------|
| `llama_context_params::kvarn_group_size` | `uint32_t` | `include/llama.h` | Appended at end of struct |
| `llama_context_params::kvarn_sink_tokens` | `uint32_t` | `include/llama.h` | Appended at end of struct |
| `llama_context_params::kvarn_recent_tokens` | `uint32_t` | `include/llama.h` | Appended at end of struct |
| `llama_context_params::kvarn_varn_iterations` | `uint32_t` | `include/llama.h` | Appended at end of struct |
| `llama_cparams::kvarn_group_size` | `uint32_t` | `src/llama-cparams.h` | Internal only |
| `llama_cparams::kvarn_sink_tokens` | `uint32_t` | `src/llama-cparams.h` | Internal only |
| `llama_cparams::kvarn_recent_tokens` | `uint32_t` | `src/llama-cparams.h` | Internal only |
| `llama_cparams::kvarn_varn_iterations` | `uint32_t` | `src/llama-cparams.h` | Internal only |

### ABI Compatibility

New fields are appended at the end of `llama_context_params`, after `ctx_other`. Old callers that zero-initialize (or use `llama_context_default_params()`) will get the correct defaults. No existing field offsets change.
