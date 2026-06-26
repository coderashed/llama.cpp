---
title: KVarN CLI Flag -- Arg Parsing Flow, Type String -> Enum Mapping
---

```mermaid
classDiagram
    class kv_cache_types {
        <<static const vector~ggml_type~ (common/arg.cpp:300)>>
        +GGML_TYPE_F32
        +GGML_TYPE_F16
        +GGML_TYPE_BF16
        +GGML_TYPE_Q8_0
        +GGML_TYPE_Q4_0
        +GGML_TYPE_Q4_1
        +GGML_TYPE_IQ4_NL
        +GGML_TYPE_Q5_0
        +GGML_TYPE_Q5_1
        +GGML_TYPE_Q2_KVARN      <<NEW>>
    }

    class kv_cache_type_from_str {
        <<static function (common/arg.cpp:312)>>
        +string s
        +returns ggml_type
        +iterates kv_cache_types, matches ggml_type_name(type) == s
        +throws runtime_error on mismatch
    }

    class get_all_kv_cache_types {
        <<static function (common/arg.cpp:321)>>
        +returns string
        +joins ggml_type_name(type) for all kv_cache_types
    }

    class common_arg__cache_type_k {
        <<common_arg (common/arg.cpp:2134)>>
        +args: {"-ctk", "--cache-type-k"}
        +arg_type: "TYPE"
        +help: "KV cache data type for K\\nallowed values: %s\\n(default: %s)"
        +handler: [](params, value) { params.cache_type_k = kv_cache_type_from_str(value); }
        +env: LLAMA_ARG_CACHE_TYPE_K
    }

    class common_arg__cache_type_v {
        <<common_arg (common/arg.cpp:2147)>>
        +args: {"-ctv", "--cache-type-v"}
        +arg_type: "TYPE"
        +help: "KV cache data type for V\\nallowed values: %s\\n(default: %s)"
        +handler: [](params, value) { params.cache_type_v = kv_cache_type_from_str(value); }
        +env: LLAMA_ARG_CACHE_TYPE_V
    }

    class common_arg__spec_draft_type_k {
        <<common_arg (common/arg.cpp:3589)>>
        +args: {"--spec-draft-type-k", "-ctkd", "--cache-type-k-draft"}
        +handler: [](params, value) { params.speculative.draft.cache_type_k = kv_cache_type_from_str(value); }
    }

    class common_arg__spec_draft_type_v {
        <<common_arg (common/arg.cpp:3602)>>
        +args: {"--spec-draft-type-v", "-ctvd", "--cache-type-v-draft"}
        +handler: [](params, value) { params.speculative.draft.cache_type_v = kv_cache_type_from_str(value); }
    }

    class common_params {
        <<struct (common/common.h:440)>>
        +ggml_type cache_type_k    // line 579
        +ggml_type cache_type_v    // line 580
    }

    class common_params_speculative {
        <<struct (common/common.h:295)>>
        +ggml_type cache_type_k    // line 332
        +ggml_type cache_type_v    // line 333
    }

    class llama_bench__ggml_type_from_name {
        <<static function (llama-bench.cpp:478)>>
        +string s
        +returns ggml_type
        +manual if-else chain: "f16"->F16, "bf16"->BF16, "q8_0"->Q8_0, ...
        +"q2_kvarn"->GGML_TYPE_Q2_KVARN  <<NEW>>
        +returns GGML_TYPE_COUNT on unknown
    }

    class llama_bench__parse_cmd_params {
        <<function (llama-bench.cpp:507)>>
        +parses -ctk/--cache-type-k at line 613
        +parses -ctv/--cache-type-v at line 633
        +uses ggml_type_from_name() for each value
        +supports comma-separated multi-value
    }

    kv_cache_type_from_str --> kv_cache_types : iterates
    get_all_kv_cache_types --> kv_cache_types : iterates
    common_arg__cache_type_k --> kv_cache_type_from_str : delegates
    common_arg__cache_type_v --> kv_cache_type_from_str : delegates
    common_arg__spec_draft_type_k --> kv_cache_type_from_str : delegates
    common_arg__spec_draft_type_v --> kv_cache_type_from_str : delegates
    common_arg__cache_type_k --> common_params : writes cache_type_k
    common_arg__cache_type_v --> common_params : writes cache_type_v
    common_arg__spec_draft_type_k --> common_params_speculative : writes cache_type_k
    common_arg__spec_draft_type_v --> common_params_speculative : writes cache_type_v
    llama_bench__parse_cmd_params --> llama_bench__ggml_type_from_name : delegates

    note for kv_cache_types "Add GGML_TYPE_Q2_KVARN to this vector.\nggml_type_name(Q2_KVARN) == 'q2_kvarn'\nmaps automatically via kv_cache_type_from_str()"
    note for llama_bench__ggml_type_from_name "llama-bench has its OWN parser.\nMust add 'q2_kvarn' case manually."
```

### String -> Enum Mapping

| Input string | Enum value | Parser |
|---|---|---|
| `"q2_kvarn"` | `GGML_TYPE_Q2_KVARN` | `kv_cache_type_from_str()` (common/arg.cpp) via `ggml_type_name()` |
| `"q2_kvarn"` | `GGML_TYPE_Q2_KVARN` | `ggml_type_from_name()` (llama-bench.cpp) via manual if-else |

### Registration Points

| # | File | Line | Change |
|---|---|---|---|
| 1 | `common/arg.cpp` | 300-310 | Add `GGML_TYPE_Q2_KVARN` to `kv_cache_types` vector |
| 2 | `tools/llama-bench/llama-bench.cpp` | 478-504 | Add `"q2_kvarn"` case to `ggml_type_from_name()` |
