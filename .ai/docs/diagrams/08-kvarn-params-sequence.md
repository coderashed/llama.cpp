---
title: KVarN Context Params -- Init Flow: params -> cparams -> cache creation
---

```mermaid
sequenceDiagram
    participant U as User Code
    participant D as llama_context_default_params
    participant V as llama_init_from_model
    participant CI as llama_context ctor
    participant CP as cparams init block
    participant KC as llama_kv_cache ctor
    participant VP as VarNTileProcessor init

    U->>D: llama_context_default_params()
    D-->>U: params { type_k=F16, type_v=F16,<br/>kvarn_group_size=128, kvarn_sink_tokens=128,<br/>kvarn_recent_tokens=128, kvarn_varn_iterations=8 }

    U->>U: params.type_k = GGML_TYPE_Q2_KVARN
    U->>U: params.kvarn_group_size = 64  (optional override)

    U->>V: llama_init_from_model(model, params)

    Note over V: --- VALIDATION PHASE ---

    alt type_k == GGML_TYPE_Q2_KVARN
        V->>V: validate kvarn_group_size > 0
        V->>V: validate kvarn_group_size divides n_embd_head_k
        V->>V: validate kvarn_sink_tokens + kvarn_recent_tokens < n_ctx
        V->>V: validate kvarn_varn_iterations >= 1
        Note over V: Log warnings for out-of-range values<br/>Return nullptr on hard errors
    end

    Note over V: --- CONSTRUCTION PHASE ---

    V->>CI: new llama_context(*model, params)

    CI->>CP: context_init(model, params)

    Note over CP: --- CPARNS COPY ---

    CP->>CP: cparams.type_k = params.type_k
    CP->>CP: cparams.type_v = params.type_v

    alt type_k == GGML_TYPE_Q2_KVARN || type_v == GGML_TYPE_Q2_KVARN
        CP->>CP: cparams.kvarn_group_size = params.kvarn_group_size
        CP->>CP: cparams.kvarn_sink_tokens = params.kvarn_sink_tokens
        CP->>CP: cparams.kvarn_recent_tokens = params.kvarn_recent_tokens
        CP->>CP: cparams.kvarn_varn_iterations = params.kvarn_varn_iterations
    else
        Note over CP: KVarN params ignored (set to 0)<br/>when type_k/type_v != Q2_KVARN
        CP->>CP: cparams.kvarn_group_size = 0
        CP->>CP: cparams.kvarn_sink_tokens = 0
        CP->>CP: cparams.kvarn_recent_tokens = 0
        CP->>CP: cparams.kvarn_varn_iterations = 0
    end

    Note over CP: --- CACHE CONSTRUCTION ---

    CP->>KC: llama_kv_cache(type_k, type_v, kvarn_group_size,<br/>kvarn_sink_tokens, kvarn_recent_tokens,<br/>kvarn_varn_iterations, ...)

    alt type_k == GGML_TYPE_Q2_KVARN
        KC->>VP: VarNTileProcessor(G=kvarn_group_size,<br/>K=kvarn_varn_iterations,<br/>head_dim=n_embd_head_k)
        Note over VP: Allocates tile_buf[G * head_dim]<br/>Allocates S_c_buf[head_dim]<br/>Allocates S_r_buf[G]
        VP-->>KC: ready
    end

    Note over KC: Three-region layout configured:<br/>sink = kvarn_sink_tokens<br/>recent = kvarn_recent_tokens<br/>middle = n_ctx - sink - recent

    KC-->>CP: cache ready
    CP-->>CI: context initialized
    CI-->>V: ctx pointer
    V-->>U: llama_context* (non-null on success)
```

### Init Flow Summary

```
User Code                    llama.cpp Internals
    |                              |
    |-- llama_context_default_params()
    |     returns defaults          |
    |-- set type_k = Q2_KVARN      |
    |-- set kvarn_group_size = 64  |
    |-- llama_init_from_model()    |
    |     |                        |
    |     |-- validate KVarN params|
    |     |-- new llama_context()  |
    |     |     |-- context_init() |
    |     |     |     |-- copy to cparams
    |     |     |     |-- llama_kv_cache ctor
    |     |     |     |     |-- VarNTileProcessor
    |     |     |     |     |-- three-region layout
    |     |     |     |-- return
    |     |     |-- return
    |     |-- return ctx
    v                              v
```

### Key Invariants

1. KVarN params are **only copied to cparams** when `type_k == Q2_KVARN` or `type_v == Q2_KVARN`; otherwise they are zeroed.
2. Validation happens **before** construction; invalid params return `nullptr`.
3. `VarNTileProcessor` is only instantiated when `type_k == Q2_KVARN`.
4. The three-region layout is configured at cache construction time and is immutable for the lifetime of the cache.
