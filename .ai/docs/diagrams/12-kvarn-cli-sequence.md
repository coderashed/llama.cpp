---
title: KVarN CLI Flag -- CLI Init -> Arg Parse -> Context Create
---

```mermaid
sequenceDiagram
    participant User as User Shell
    participant CLI as llama-cli main()
    participant Parser as common_params_parse()
    participant ArgDef as common_arg defs
    participant Params as common_params
    participant Init as common_init_from_params()
    participant Ctx as llama_context
    participant Cache as llama_kv_cache

    User->>CLI: llama-cli -m model.gguf --cache-type-k q2_kvarn -p "Hello"
    CLI->>Parser: common_params_parse(argc, argv, LLAMA_EXAMPLE_CLI)

    Note over Parser: --- STEP 1: Build parser context ---
    Parser->>ArgDef: common_params_parser_init()
    ArgDef-->>Parser: ctx with all common_arg defs

    Note over Parser: --- STEP 2: Parse --cache-type-k ---
    Parser->>ArgDef: match "--cache-type-k"
    ArgDef->>ArgDef: handler: kv_cache_type_from_str("q2_kvarn")
    ArgDef->>ArgDef: iterate kv_cache_types, match ggml_type_name(type) == "q2_kvarn"
    ArgDef-->>Parser: GGML_TYPE_Q2_KVARN
    Parser->>Params: params.cache_type_k = GGML_TYPE_Q2_KVARN

    Note over Parser: --- STEP 3: Remaining args ---
    Parser->>Params: params.prompt = "Hello"
    Parser-->>CLI: params populated

    CLI->>Init: common_init_from_params(params)

    Note over Init: --- STEP 4: Model + context init ---
    Init->>Init: llama_model_load_from_file(...)
    Init->>Ctx: llama_init_from_model(model, ctx_params)

    Note over Ctx: --- STEP 5: KVarN validation ---
    Ctx->>Ctx: type_k == GGML_TYPE_Q2_KVARN?
    Ctx->>Ctx: validate kvarn_group_size > 0
    Ctx->>Ctx: validate kvarn_group_size divides n_embd_head_k
    Ctx->>Ctx: validate sink + recent < n_ctx

    Note over Ctx: --- STEP 6: Cache construction ---
    Ctx->>Cache: llama_kv_cache(type_k=Q2_KVARN, ...)
    Cache->>Cache: VarNTileProcessor init
    Cache->>Cache: three-region layout
    Cache-->>Ctx: cache ready
    Ctx-->>Init: ctx pointer
    Init-->>CLI: context ready

    CLI->>CLI: llama_decode() with prompt "Hello"
    CLI-->>User: streaming output
```

### Flow Summary

```
User: --cache-type-k q2_kvarn
  |
  v
common_params_parse()
  |-- common_params_parser_init()  -- builds option definitions
  |-- match "--cache-type-k"
  |     |-- kv_cache_type_from_str("q2_kvarn")
  |     |     |-- iterate kv_cache_types[]
  |     |     |-- ggml_type_name(GGML_TYPE_Q2_KVARN) == "q2_kvarn"  -> match
  |     |     |-- return GGML_TYPE_Q2_KVARN
  |     |-- params.cache_type_k = GGML_TYPE_Q2_KVARN
  |
  v
common_init_from_params(params)
  |-- llama_init_from_model(model, ctx_params)
  |     |-- validate KVarN params
  |     |-- llama_kv_cache(type_k=Q2_KVARN)
  |     |     |-- VarNTileProcessor init
  |     |     |-- three-region layout
  |     |-- return ctx
  |
  v
llama_decode() -- inference with KVarN cache
```
