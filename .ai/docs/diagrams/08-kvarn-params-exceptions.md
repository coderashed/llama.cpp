---
title: KVarN Context Params -- Validation Rules, Defaults & Edge Cases
---

```mermaid
graph TD
    subgraph "Validation Rules (llama_init_from_model)"
        V1["type_k == GGML_TYPE_Q2_KVARN<br/>OR type_v == GGML_TYPE_Q2_KVARN?"]
        V2["kvarn_group_size > 0?"]
        V3["kvarn_group_size divides<br/>n_embd_head_k for all layers?"]
        V4["kvarn_sink_tokens + kvarn_recent_tokens<br/>< n_ctx?"]
        V5["kvarn_varn_iterations >= 1?"]
        V6["kvarn_group_size <= 1024?"]
        V7["kvarn_varn_iterations <= 100?"]
    end

    subgraph "Defaults (llama_context_default_params)"
        D1["kvarn_group_size = 128"]
        D2["kvarn_sink_tokens = 128"]
        D3["kvarn_recent_tokens = 128"]
        D4["kvarn_varn_iterations = 8"]
    end

    subgraph "Conditional Copy (context_init)"
        C1["type_k == Q2_KVARN<br/>OR type_v == Q2_KVARN?"]
        C2["Copy KVarN params<br/>to cparams"]
        C3["Zero KVarN params<br/>in cparams"]
    end

    subgraph "Edge Cases"
        E1["kvarn_group_size = 1<br/>(each token is its own tile)"]
        E2["kvarn_sink_tokens = 0<br/>(no sink region)"]
        E3["kvarn_recent_tokens = 0<br/>(no recent region)"]
        E4["kvarn_sink_tokens + kvarn_recent_tokens<br/>>= n_ctx<br/>(no middle region)"]
        E5["kvarn_varn_iterations = 1<br/>(single iteration, fast but less accurate)"]
        E6["type_k != Q2_KVARN but<br/>type_v == Q2_KVARN<br/>(V-only KVarN)"]
    end

    subgraph "Error Handling"
        ERR1["group_size == 0<br/>-> LOG_ERROR + return nullptr"]
        ERR2["group_size does not divide head_dim<br/>-> LOG_ERROR + return nullptr"]
        ERR3["sink + recent >= n_ctx<br/>-> LOG_WARN + clamp recent"]
        ERR4["varn_iterations == 0<br/>-> LOG_WARN + default to 8"]
        ERR5["group_size > 1024<br/>-> LOG_WARN + clamp to 1024"]
        ERR6["varn_iterations > 100<br/>-> LOG_WARN + clamp to 100"]
    end

    V1 -->|"yes"| V2
    V1 -->|"no"| C3
    V2 -->|"pass"| V3
    V2 -->|"fail"| ERR1
    V3 -->|"pass"| V4
    V3 -->|"fail"| ERR2
    V4 -->|"pass"| V5
    V4 -->|"fail"| ERR3
    V5 -->|"pass"| V6
    V5 -->|"fail"| ERR4
    V6 -->|"pass"| V7
    V6 -->|"fail"| ERR5
    V7 -->|"pass"| C2
    V7 -->|"fail"| ERR6

    D1 --> V2
    D2 --> V4
    D3 --> V4
    D4 --> V5

    C2 --> E1
    C2 --> E2
    C2 --> E3
    C2 --> E4
    C2 --> E5
    C2 --> E6
```

### Validation Table

| # | Rule | Severity | Action | Rationale |
|---|------|----------|--------|-----------|
| 1 | `kvarn_group_size == 0` | **Error** | `return nullptr` | Group size must be positive for tile processing |
| 2 | `kvarn_group_size > n_embd_head_k` | **Error** | `return nullptr` | Tile cannot be larger than head dimension |
| 3 | `n_embd_head_k % kvarn_group_size != 0` | **Error** | `return nullptr` | Group size must divide head dim for tile alignment |
| 4 | `kvarn_group_size > 1024` | Warning | Clamp to 1024 | Practical upper bound; larger tiles waste memory |
| 5 | `kvarn_sink_tokens + kvarn_recent_tokens >= n_ctx` | Warning | Clamp `recent_tokens = n_ctx - sink_tokens - 1` | Must leave at least 1 slot for middle region |
| 6 | `kvarn_varn_iterations == 0` | Warning | Default to 8 | Zero iterations = no normalization |
| 7 | `kvarn_varn_iterations > 100` | Warning | Clamp to 100 | Diminishing returns beyond 100 iterations |
| 8 | `type_k != Q2_KVARN && type_v != Q2_KVARN` | Info | Zero all KVarN cparams | Params are irrelevant; silently ignore |

### Default Value Rationale

| Default | Source | Rationale |
|---------|--------|-----------|
| `G = 128` | Research paper §9 | Matches Q2_KVARN block size; good statistical sample for VarN |
| `S = 128` | Research paper §6 | Preserves ~1 sentence of context from eviction |
| `R = 128` | Research paper §6 | Preserves last ~1 sentence from eviction |
| `K = 8` | Research paper §9 | 8 iterations sufficient for convergence to ~N(0,1) |

### Three-Region Layout Invariants

```
|<- sink (S) ->|<- middle (ctx - S - R) ->|<- recent (R) ->|
+--------------+---------------------------+-----------------+
0              S              ctx-R                      ctx-1
```

- **Sink region** `[0, S)`: Always preserved; never evicted.
- **Recent region** `[ctx-R, ctx)`: Always preserved; never evicted.
- **Middle region** `[S, ctx-R)`: Subject to eviction when cache is full.
- Invariant: `S + R < n_ctx` (enforced by validation rule #5).

### Conditional Copy Logic

```
if (type_k == GGML_TYPE_Q2_KVARN || type_v == GGML_TYPE_Q2_KVARN) {
    cparams.kvarn_group_size      = params.kvarn_group_size;
    cparams.kvarn_sink_tokens     = params.kvarn_sink_tokens;
    cparams.kvarn_recent_tokens   = params.kvarn_recent_tokens;
    cparams.kvarn_varn_iterations = params.kvarn_varn_iterations;
} else {
    cparams.kvarn_group_size      = 0;
    cparams.kvarn_sink_tokens     = 0;
    cparams.kvarn_recent_tokens   = 0;
    cparams.kvarn_varn_iterations = 0;
}
```

This ensures that non-KVarN caches have zeroed KVarN fields, making it safe to check `cparams.kvarn_group_size > 0` as a gate for KVarN-specific code paths.
