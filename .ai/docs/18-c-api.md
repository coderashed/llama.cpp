# 18. C API (libllama) -- Stable C ABI for llama.cpp

## Purpose

The C API (`include/llama.h`) provides a stable C ABI (Application Binary Interface) for llama.cpp, enabling language bindings and external programs to load models, tokenize text, perform inference, sample tokens, and manage state -- all without depending on C++ ABI details. Every other feature in the project is accessed through this API. It is the single public contract between llama.cpp internals and all consumers (server, CLI tools, Python bindings, etc.).

## The `LLAMA_API` Macro

All public functions are decorated with `LLAMA_API`. Under shared-library builds (`LLAMA_SHARED`), it expands to `__declspec(dllexport)` on Windows or `__attribute__((visibility("default")))` on GCC/Clang. In static builds it is a no-op.

- `include/llama.h:15-27`

## Opaque Struct Pattern

Five key types are exposed as opaque, forward-declared structs that callers can only pass by pointer:

| Opaque handle | Declared | Fully defined in |
|---|---|---|
| `struct llama_vocab` | `include/llama.h:61` | `src/llama-vocab.h:72` |
| `struct llama_model` | `include/llama.h:62` | `src/llama-model.h:522` |
| `struct llama_context` | `include/llama.h:63` | `src/llama-context.h:42` |
| `struct llama_sampler` | `include/llama.h:64` | `include/llama.h:1275` (header-only) |
| `struct llama_adapter_lora` | `include/llama.h:442` | `src/llama-adapter.h:63` |

Additionally `llama_memory_t` (`include/llama.h:66`) is a pointer to `llama_memory_i`, an abstract interface struct.

Callers never see the struct body; they construct and destroy objects through factory/free functions. This decouples the ABI from C++ layout, vtable, and exception machinery.

## PIMPL Mapping (Pointer-to-Implementation)

Each opaque handle delegates to an internal C++ class. The C API functions are thin wrappers that translate calls into C++ method invocations:

### `llama_model`
- C API `llama_model_load_from_file()` at `src/llama.cpp:426` calls `llama_model_load_from_file_impl()` which creates a `llama_model_loader` and constructs `llama_model` via `llama_model_create()`.
- `llama_model` internally uses `struct llama_model::impl` (private PIMPL at `src/llama-model.cpp:973`) to hide the ggml context list, buffer management, and device assignment.
- Query functions like `llama_model_n_embd()` at `src/llama-model.cpp:2299` directly access `model->hparams.n_embd`.
- Lifecycle: `delete model` at `src/llama-model.cpp:2291`.

### `llama_context`
- C API `llama_init_from_model()` at `src/llama-context.cpp:3490` calls `new llama_context(*model, params)`.
- `struct llama_context` at `src/llama-context.h:42` is a concrete C++ struct with full member visibility. It holds model reference, cparams, memory, schedulers, logits buffers, sampling state, and timing data.
- `llama_decode()` at `src/llama-context.cpp:4054` calls `ctx->decode(batch)`.
- Lifecycle: `delete ctx` at `src/llama-context.cpp:3581`.

### `llama_vocab`
- Owned inline by `llama_model` (`src/llama-model.h:529`).
- Internally uses `struct llama_vocab::impl` (PIMPL at `src/llama-vocab.cpp:1765`) for the token table, BPE merges, and per-type configuration.
- C API functions like `llama_vocab_get_text()` at `src/llama-vocab.cpp:4109` call `vocab->token_get_text(token)` which delegates to `pimpl->id_to_token.at(id).text`.

### `llama_sampler`
- `struct llama_sampler` is defined in `include/llama.h:1275` as a public struct with an interface pointer `iface` and opaque context `ctx`. This is NOT fully opaque -- callers can implement custom samplers by populating a `llama_sampler_i` vtable.
- The sampler chain (`llama_sampler_chain`) internal struct is at `src/llama-sampler.h:12` and stores `std::vector<info>` of sub-samplers plus pre-allocated buffers.
- `llama_sampler_chain_init()` at `src/llama-sampler.cpp:792` creates a `llama_sampler` wrapping `new llama_sampler_chain`.

### `llama_adapter_lora`
- Defined at `src/llama-adapter.h:63`. Contains weight maps, ggml contexts, backend buffers, and optional aLoRA invocation tokens.
- C API `llama_adapter_lora_init()` at `src/llama-adapter.cpp` constructs the adapter via `llama_model_loader`.
- Lifecycle tracked via `llama_model::loras` set (`src/llama-model.h:598`) for automatic cleanup.

## Major API Categories

### 1. Backend lifecycle (`include/llama.h:451-461`)
- `llama_backend_init()` -- initialize ggml backend, load plugin backends (`src/llama.cpp:89`)
- `llama_backend_free()` -- clean up (`src/llama.cpp:116`)
- `llama_numa_init()` -- optional NUMA binding (`src/llama.cpp:104`)

### 2. Model loading / unloading (`include/llama.h:475-512`)
- `llama_model_load_from_file()` / `llama_model_load_from_splits()` / `llama_model_load_from_file_ptr()` / `llama_model_init_from_user()`
- `llama_model_free()` / deprecated `llama_free_model()`
- `llama_model_save_to_file()`

### 3. Context creation / destruction (`include/llama.h:514-524`)
- `llama_init_from_model()` -- create inference context; validates params, returns `nullptr` on error (`src/llama-context.cpp:3490`)
- `llama_free()` -- delete context (`src/llama-context.cpp:3581`)

### 4. Model property queries (`include/llama.h:540-638`)
- Dimensions: `llama_model_n_embd()`, `llama_model_n_layer()`, `llama_model_n_head()`, etc.
- Capabilities: `llama_model_has_encoder()`, `llama_model_is_recurrent()`, `llama_model_is_hybrid()`, etc.
- Metadata: `llama_model_meta_val_str()`, `llama_model_chat_template()`, `llama_model_desc()`

### 5. Vocabulary queries (`include/llama.h:558-616, 1064-1120`)
- Type: `llama_vocab_type()`
- Special tokens: `llama_vocab_bos()`, `llama_vocab_eos()`, etc.
- Token properties: `llama_vocab_get_text()`, `llama_vocab_get_score()`, `llama_vocab_is_eog()`

### 6. Tokenization / detokenization (`include/llama.h:1121-1170`)
- `llama_tokenize()` -- text -> tokens (`src/llama-vocab.cpp:4303`)
- `llama_token_to_piece()` -- single token -> text (`src/llama-vocab.cpp:4314`)
- `llama_detokenize()` -- tokens -> text (`src/llama-vocab.cpp:4324`)

### 7. Batch management (`include/llama.h:912-936`)
- `llama_batch_get_one()` -- simple single-sequence batch helper
- `llama_batch_init()` / `llama_batch_free()` -- heap-allocated batch

### 8. Encoding / Decoding (`include/llama.h:938-962`)
- `llama_encode()` -- encoder pass, returns 0 on success, <0 on error (`src/llama-context.cpp:4043`)
- `llama_decode()` -- decoder pass, returns 0=success, 1=no KV slot, 2=aborted, etc. (`src/llama-context.cpp:4054`)

### 9. Output retrieval (`include/llama.h:998-1059`)
- `llama_get_logits()` / `llama_get_logits_ith()` -- logit access
- `llama_get_embeddings()` / `llama_get_embeddings_ith()` / `llama_get_embeddings_seq()` -- embedding access
- Backend sampling outputs: `llama_get_sampled_token_ith()`, etc.

### 10. Sampling API (`include/llama.h:1197-1495`)
- Sampler chain: `llama_sampler_chain_init()`, `llama_sampler_chain_add()`, etc.
- Individual samplers: `llama_sampler_init_greedy()`, `llama_sampler_init_dist()`, `llama_sampler_init_top_k()`, `llama_sampler_init_grammar()`, `llama_sampler_init_penalties()`, `llama_sampler_init_dry()`, etc.
- `llama_sampler_sample()` -- convenience: apply chain + accept (`src/llama-sampler.cpp:806`)

### 11. Adapters (`include/llama.h:648-701`)
- `llama_adapter_lora_init()` / `llama_adapter_lora_free()`
- `llama_set_adapters_lora()` -- activate on context
- `llama_set_adapter_cvec()` -- control vectors

### 12. Memory management (`include/llama.h:704-776`)
- `llama_memory_clear()`, `llama_memory_seq_rm()`, `llama_memory_seq_cp()`, etc.

### 13. State save/load (`include/llama.h:778-907`)
- Full context: `llama_state_get_data()`, `llama_state_set_data()`
- Per-sequence: `llama_state_seq_get_data()`, `llama_state_seq_set_data()`
- File I/O: `llama_state_save_file()`, `llama_state_load_file()`

### 14. Chat templates (`include/llama.h:1172-1195`)
- `llama_chat_apply_template()` -- applies built-in templates (`src/llama.cpp:470`)
- `llama_chat_builtin_templates()` -- list available (`src/llama-chat.cpp:950`)

### 15. Performance / diagnostics (`include/llama.h:1520-1550`)
- `llama_perf_context()` / `llama_perf_sampler()` -- timing data
- `llama_print_system_info()` -- backend features (`src/llama.cpp:559`)
- `llama_log_get()` / `llama_log_set()` -- logging callbacks

### 16. Thread management (`include/llama.h:462-468`)
- `llama_attach_threadpool()` / `llama_detach_threadpool()`

### 17. Training (`include/llama.h:1552-1583`)
- `llama_opt_init()`, `llama_opt_epoch()` -- experimental training support

### 18. Model quantization (`include/llama.h:637-643`)
- `llama_model_quantize()` -- quantize a model file

### 19. Model splitting (`include/llama.h:1497-1508`)
- `llama_split_path()` / `llama_split_prefix()` -- build/parse split GGUF paths

## ABI Stability Conventions

The library provides the following ABI guarantees:

1. **C linkage only** (`extern "C"` at `include/llama.h:51`). All public symbols are `extern "C"`, preventing C++ name mangling.

2. **Fixed-size parameter structs** (`include/llama.h:291-395`). `llama_model_params`, `llama_context_params`, etc. are Plain Old Data (POD) structs with explicit fields. New fields are appended at the end, so old callers continue to work if they zero-initialize (or use the `_default_params()` functions). The structs are annotated with `// Keep the booleans together to avoid misalignment during copy-by-value` (`include/llama.h:318,374`).

3. **Deprecation path**. The `DEPRECATED()` macro (`include/llama.h:29-35`) marks old functions with a compiler hint and a hint string directing callers to the replacement. Old functions continue to exist as thin wrappers (e.g., `llama_load_model_from_file` -> `llama_model_load_from_file` at `src/llama.cpp:420`). Deprecated functions are explicitly labeled in the header: `DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(...)`.

4. **No C++ exceptions cross the boundary**. Exceptions are caught internally (e.g., `src/llama.cpp:335` catches `std::exception` and logs an error); the public API never throws.

5. **Opaque pointer returns**. All handles are returned as pointers to forward-declared structs. The caller never dereferences them.

6. **No STL types in public signatures**. Arguments use only C types: `int32_t`, `float`, `char*`, `size_t`, `FILE*`, etc.

## Error Reporting Convention

The API uses return-value-based error reporting (no exceptions, no `errno`):

- **Model loading functions** return `nullptr` on failure after logging via `LLAMA_LOG_ERROR` (e.g., `src/llama.cpp:439`).
- **`llama_encode()`**: returns 0 on success, `<0` on error (`include/llama.h:941-942`).
- **`llama_decode()`**: returns 0 on success, `1` = KV slot not found, `2` = aborted, `-1` = invalid input, `< -1` = fatal error (`include/llama.h:954-958`).
- **Tokenization** (`llama_tokenize`): returns `>=0` for actual token count, negative for failure; `INT32_MIN` for overflow (`include/llama.h:1129-1131`).
- **String metadata** (`llama_model_meta_val_str`): returns string length on success, `-1` on failure (`include/llama.h:592-593`).
- **State save/load**: `llama_state_seq_set_data()` returns `>0` = ok, `0` = failed (`include/llama.h:852-854`).
- **Bool-returning functions**: `llama_memory_seq_rm()` returns `false` on partial failure (`include/llama.h:714`).
- **`llama_model_quantize()`**: returns 0 on success (`include/llama.h:638`).

## Versioning and Session Formats

Versioning is done via session and state file magic numbers:

- `LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN` at `include/llama.h:45-46`
- `LLAMA_SESSION_VERSION = 9` at `include/llama.h:46`
- `LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ` at `include/llama.h:48`
- `LLAMA_STATE_SEQ_VERSION = 2` at `include/llama.h:49`

## Touch Points (Every Other Feature Is Accessed Through This API)

The C API is the sole entry point for all functionality:

- **ggml-backend**: accessed indirectly through model and context creation. Context stores `ggml_backend_sched_ptr` (`src/llama-context.h:340`), `ggml_backend_ptr` vector (`src/llama-context.h:345`).
- **Tokenization / vocab**: via `llama_tokenize()` and `llama_vocab_*()` functions.
- **Quantization**: via `llama_model_quantize()` (`include/llama.h:639`).
- **LoRA adapters**: via `llama_adapter_lora_*()` family.
- **Grammar sampling**: via `llama_sampler_init_grammar()`.
- **Chat templates**: via `llama_chat_apply_template()`.
- **Embeddings**: via `llama_get_embeddings*()`.
- **KV cache**: managed internally; callers manipulate through `llama_memory_seq_*()` and `llama_state_*()` functions.
- **Performance measurement**: via `llama_perf_context()` / `llama_perf_sampler()`.
- **Training**: via `llama_opt_*()`.

## Failure Modes

1. **Null pointer crashes**: The API performs minimal null-checks on function entry. For example, `llama_init_from_model()` checks `if (!model)` at `src/llama-context.cpp:3493`, but many accessor functions (e.g., `llama_n_ctx()` at `src/llama-context.cpp:3585`) will segfault if passed `nullptr`.
2. **Resource leaks**: Callers must pair `_init`/`_free` calls (e.g., `llama_model_load_from_file` / `llama_model_free`, `llama_sampler_chain_init` / `llama_sampler_free`).
3. **Invalid param combos**: `llama_init_from_model()` validates several preconditions (e.g., `n_batch` and `n_ubatch` cannot both be zero, `src/llama-context.cpp:3498`) and returns `nullptr`.
4. **Sampler misuse**: `llama_perf_sampler()` asserts at runtime that the sampler is a chain (`llama_sampler.cpp:3856`). Calling `llama_sampler_free()` on a chained sub-sampler (without removing it first) double-frees.
5. **Thread safety**: The API is NOT fully thread-safe. Backend init (`llama_backend_init`) must be called once before any other function. Context-level operations (`llama_decode`, sampling) are not reentrant. Tokenization is documented as thread-safe (`include/llama.h:1124`).
