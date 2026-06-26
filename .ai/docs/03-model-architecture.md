# Model Architecture Registry and Model Loading

## 1. Purpose

llama.cpp supports **132 named model architectures** (as of `llm_arch` enum, plus `LLM_ARCH_UNKNOWN`) covering decoder-only transformers (LLaMA, Falcon, Qwen, Gemma, Phi), encoder-only (BERT, BERT-derived), encoder-decoder (T5), state-space / recurrent (Mamba, RWKV), hybrid (Jamba, Granite Hybrid, Qwen3Next), diffusion (LLaDA, Dream, RND1), multimodal (Qwen2VL, Hunyuan VL, CogVLM), embedding-only (Gemma Embedding, Pangu Embed), and vision / audio (WavTokenizer, PaddleOCR). Each architecture gets a dedicated C++ model class that implements four pure-virtual methods: `load_arch_hparams`, `load_arch_tensors`, `build_arch_graph`, and optionally overrides `load_hparams` base behavior.

## 2. Entry Points

| Step | File:Line | Function |
|------|-----------|----------|
| Public API entry | `include/llama.h:489` | `llama_model_load_from_file()` |
| Public API entry | `include/llama.h:475` | `llama_model_init_from_user()` (GGUF metadata + custom data loader) |
| Public API entry | `include/llama.h:500` | `llama_model_load_from_splits()` |
| Public API entry | `include/llama.h:494` | `llama_model_load_from_file_ptr()` |
| Internal dispatch | `src/llama.cpp:279` | `llama_model_load()` - orchestrates loader + model creation |
| Architecture detection | `src/llama-model-loader.cpp:512` | `llama_model_loader()` constructor reads `general.architecture` from GGUF |
| String-to-arch lookup | `src/llama-arch.cpp:850` | `llm_arch_from_string()` - linear scan over `LLM_ARCH_NAMES` map |
| Arch-to-C++-class factory | `src/llama-model.cpp:38` | `llama_model_mapping()` - giant switch returning per-arch model instance |
| Model instantiation | `src/llama-model.cpp:306` | `llama_model_create(llm_arch, params)` |
| Model instantiation | `src/llama-model.cpp:320` | `llama_model_create(llama_model_loader &, params)` - reads arch from loader |
| Model save | `src/llama-model-saver.cpp:37` | `llama_model_saver` constructor |
| Quantizer entry | `src/llama-quant.cpp:886` | `llama_model_quantize_internal()` creates model via `llama_model_create(ml, ...)` |

## 3. Data Flow

### 3.1 GGUF File to Architecture Detection

1. `llama_model_loader` constructor (`src/llama-model-loader.cpp:512`) opens the GGUF file via `gguf_init_from_file()`.
2. At line 553 it reads `general.architecture` key-value pair via `get_key(LLM_KV_GENERAL_ARCHITECTURE, arch_name, false)`.
3. At line 554 it calls `llm_arch_from_string(arch_name)` which maps the GGUF string to the `llm_arch` enum (`src/llama-arch.cpp:850`).
4. If the string is unrecognized, `llm_arch_from_string` returns `LLM_ARCH_UNKNOWN` (line 857).
5. The `LLM_KV` helper is set to the detected architecture: `llm_kv = LLM_KV(llm_arch_from_string(arch_name))` (line 554), which controls how all subsequent KV lookups format their GGUF key names (using `%s.*` patterns with the arch name substituted).

### 3.2 Architecture Detection to Model Instantiation

6. `llama_model_load()` (`src/llama.cpp:279`) calls `llama_model_create(ml, params)` (line 286).
7. In `llama_model_create()` (`src/llama-model.cpp:320`), `ml.get_arch()` retrieves the detected arch.
8. If `arch == LLM_ARCH_UNKNOWN`, a `std::runtime_error` is thrown (line 323).
9. Otherwise, `llama_model_mapping(arch, params)` (`src/llama-model.cpp:38`) dispatches to the per-architecture C++ class (e.g., `new llama_model_llama(params)`).
10. If no switch case matches, the default at line 301 throws: `"unsupported model architecture: '<name>'"`.

### 3.3 Model Instantiation to Hyperparameter Extraction

11. `model->load_hparams(ml)` is called (`src/llama.cpp:309`).
12. `llama_model_base::load_hparams()` (`src/llama-model.cpp:1023`):
    - Copies all GGUF KV metadata into `gguf_kv` unordered_map (lines 1027-1035).
    - Reads `n_ctx_train`, `n_embd`, `n_layer_all`, `n_expert`, `n_head_arr`, `n_head_kv_arr`, `n_ff_arr`, etc. from GGUF via `ml.get_key()` / `ml.get_key_or_arr()`.
    - Reads RoPE parameters, SWA pattern, pooling type, SSM params, expert config, etc.
    - Default-initializes per-layer arrays (`n_head_arr`, `is_swa_impl`, etc.).
13. The per-architecture `load_arch_hparams(ml)` is called (pure virtual) to read architecture-specific keys such as `ssm_d_conv`, `time_mix_extra_dim`, `expert_gating_func`, `n_embd_head_k_mla`, etc.

### 3.4 Hyperparameter Extraction to Weight Loading

14. `model->load_vocab(ml)` reads tokenizer data (in `llama_model_base::load_vocab`).
15. `model->load_stats(ml)` reads metadata stats.
16. `model->load_tensors(ml)` (`src/llama.cpp:330`) delegates to `llama_model_base::load_tensors()` then calls `load_arch_tensors(ml)`.
17. Per-architecture `load_arch_tensors(ml)`:
    - Uses the `LLAMA_LOAD_LOCALS` macro (`src/llama-model.h:713`) to unpack `n_layer`, `n_head`, `n_embd`, etc. from `hparams`.
    - Uses `LLM_TN(arch)` helper (`src/llama-arch.h:631`) to build tensor names from `llm_tensor` enum (e.g., `tn(LLM_TENSOR_ATTN_Q, "weight", i)` to `"blk.%d.attn_q.weight"`).
    - Calls `create_tensor(ml, tn(...), {dims}, flags)` (`src/llama-model-loader.cpp:1047`) which allocates `ggml_tensor` metadata.
18. `llama_model_loader::create_tensor()` uses `llm_tensor_info_for(tensor)` (`src/llama-arch.cpp:860`) to look up the `llm_tensor_info` (layer classification + expected `ggml_op`) and assigns the tensor to the correct buffer type (CPU, GPU, etc.).

### 3.5 Weight Loading to ggml_tensor Population

19. After all tensors are created, `ml.init_mappings()` (`src/llama-model-loader.cpp:1335`) sets up mmap regions.
20. `ml.load_all_data()` (`src/llama-model-loader.cpp:1408`) copies or mmaps tensor data from file to buffer.
21. With mmap: tensor `data` pointers point directly into the mapped file region (line 1391).
22. Without mmap: data is read via `file->read_raw()` into pre-allocated buffers (line 1400).
23. If `check_tensors` is set, each tensor is validated via `ggml_validate_row_data()` (line 1403).

## 4. Key Types and Protocols

### 4.1 `llm_arch` Enum

Defined in `src/llama-arch.h:13-147`. Contains 131 named architectures plus `LLM_ARCH_UNKNOWN`. The enum values correspond to string names in `LLM_ARCH_NAMES` map (`src/llama-arch.cpp:8-142`). Helper functions:

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `llm_arch_name()` | `src/llama-arch.cpp:842` | Enum to string |
| `llm_arch_from_string()` | `src/llama-arch.cpp:850` | String to enum |
| `llm_arch_is_recurrent()` | `src/llama-arch.cpp:864` | Mamba, RWKV variants |
| `llm_arch_is_hybrid()` | `src/llama-arch.cpp:878` | Jamba, Granite Hybrid, etc. |
| `llm_arch_is_diffusion()` | `src/llama-arch.cpp:898` | Dream, LLaDA, RND1 |
| `llm_arch_supports_sm_tensor()` | `src/llama-arch.cpp:920` | Allows tensor-parallel split |
| `llm_arch_supports_rs_rollback()` | `src/llama-arch.cpp:910` | Qwen3.5 recurrent state rollback |

### 4.2 `llm_kv` Enum

Defined in `src/llama-arch.h:149-363`. ~170 enumerated GGUF key identifiers. Each maps to a GGUF key name template in `LLM_KV_NAMES` (`src/llama-arch.cpp:144-357`). Keys use `%s` as placeholder for the architecture name (e.g., `LLM_KV_ATTENTION_HEAD_COUNT` to `"%s.attention.head_count"`). The `LLM_KV` class (`src/llama-arch.h:589-596`) formats the final key string via `LLM_KV::operator()`.

### 4.3 `llm_tensor` Enum

Defined in `src/llama-arch.h:365-580`. ~215 enumerated tensor types (token embeddings, attention Q/K/V/O, FFN gate/down/up, SSM, RWKV time-mix, MLA projections, etc.). Each maps to a name template in `LLM_TENSOR_NAMES` (`src/llama-arch.cpp:359-572`). Naming convention: global tensors use flat names (`"token_embd"`, `"output"`), per-layer tensors use `"blk.%d.<name>"` patterns, encoder/decoder layers use `"enc.blk.%d."` / `"dec.blk.%d."` prefixes.

The `LLM_TN` helper class (`src/llama-arch.h:631-643`) generates tensor name strings at load time:

```cpp
const auto tn = LLM_TN(LLM_ARCH_LLAMA);
std::string name = tn(LLM_TENSOR_ATTN_Q, "weight", 3);  // "blk.3.attn_q.weight"
```

### 4.4 `llm_tensor_info` and `LLM_TENSOR_INFOS`

`src/llama-arch.h:646-649` and `src/llama-arch.cpp:584-801`. Maps each `llm_tensor` to a `(llm_tensor_layer, ggml_op)` pair. This is used by `create_tensor()` in the model loader to:
- Determine the buffer layer type: `LLM_TENSOR_LAYER_INPUT`, `LLM_TENSOR_LAYER_REPEATING`, or `LLM_TENSOR_LAYER_OUTPUT`.
- Probe backend support for the expected `ggml_op` to decide buffer placement.
- Raise errors if the declared op does not match what the backend supports.

### 4.5 `llama_model` Struct

Defined in `src/llama-model.h:522-662`. The central model representation:
- `arch` (`llm_arch`): the detected architecture.
- `hparams` (`llama_hparams`): all extracted hyperparameters.
- `vocab` (`llama_vocab`): tokenizer data.
- `layers` (`vector<llama_layer>`): per-block tensor pointers.
- `tok_embd`, `output_norm`, `output`: global tensor pointers.
- Pure virtual interface: `load_stats()`, `load_hparams()`, `load_vocab()`, `load_tensors()`, `load_arch_hparams()`, `load_arch_tensors()`, `build_arch_graph()`.
- `build_graph()`: public entry to produce a `ggml_cgraph` for inference (`src/llama-model.h:645`).

### 4.6 `llama_model_base`

Defined in `src/llama-model.h:668-707`. The CRTP-like base that implements the common loading pipeline. Key methods:
- `create_tensor(ml, tn, ne, flags)` at line 684 - delegates to `llama_model_loader::create_tensor()`.
- `create_tensor_gate_up_exps()` (line 690): tries merged `ffn_gate_up_exps` first, falls back to separate `ffn_gate` + `ffn_up`.
- `create_tensor_qkv()` (line 694): tries merged `attn_qkv` first, falls back to separate Q, K, V.

### 4.7 `llama_model_loader`

Defined in `src/llama-model-loader.h:31-207`. Manages the loading pipeline:
- `weights_map`: `map<string, llama_tensor_weight>` sorted by layer index for nice ordering (`weight_name_comparer`, line 53).
- `create_tensor()` (line 181): allocates tensor metadata, tracking layer assignment and backend buffer type.
- `get_key<T>()` (line 155): reads GGUF KV values with optional override support.
- `get_arr()` / `get_key_or_arr()`: for per-layer arrays (e.g., per-layer `n_head`).
- `init_mappings()` (line 189): sets up mmap regions.
- `load_all_data()` (line 197): bulk tensor data loading with async GPU upload support.
- `done_getting_tensors()` (line 187): validates all tensors were consumed.

### 4.8 `llama_hparams`

Defined in `src/llama-hparams.h:39-398`. The complete set of model hyperparameters:
- Architecture-independent: `n_ctx_train`, `n_embd`, `n_layer_all`, `n_expert`, `n_head_arr`, `n_ff_arr`, RoPE params, SWA config, SSM params, pooling type, etc.
- Architecture-specific: `ssm_d_state` (Mamba), `time_mix_extra_dim` (RWKV), `wkv_head_size` (RWKV), `n_embd_head_k_mla` (DeepSeek MLA), `indexer_n_head` (DSA), `n_deepstack_layers` (Qwen3VL), etc.
- Per-layer arrays: `n_head_arr[LLAMA_MAX_LAYERS]`, `n_head_kv_arr[]`, `n_ff_arr[]`, `is_swa_impl[]`, `is_recr_impl[]`, `swiglu_clamp_exp[]`.
- Helper methods: `n_head(il)`, `n_ff(il)`, `n_embd_head_k(il)`, `n_embd_inp()`, `n_embd_out()`, `n_layer()`, `is_mla()`, `has_kv(il)`, `is_swa(il)`, `is_recr(il)`.

### 4.9 `llama_cparams`

Defined in `src/llama-cparams.h:10-59`. Context (runtime) parameters: `n_ctx`, `n_batch`, `n_ubatch`, `n_seq_max`, `rope_freq_base/scale`, flags for flash attention, fused delta net, embeddings, etc.

### 4.10 `llama_layer`

Defined in `src/llama-model.h:223-507`. Per-block tensor storage with ~160 `ggml_tensor*` fields covering attention weights, normalization weights, FFN weights, MoE expert weights, SSM weights, RWKV time-mix weights, BitNet scales, Laurel/Altup projections, KDA convolutions, etc. Default-initialized to `nullptr`.

## 5. Touch Points

### 5.1 Inference Engine

`llama_model::build_graph()` (`src/llama-model.h:645`) produces the compute graph. Each architecture implements `build_arch_graph()` in its own model class (e.g., `llama_model_llama::build_arch_graph()` in `models/llama.cpp`). The graph builder indexes per-layer tensors from `model->layers[i]` and constructs `ggml_*` ops referencing the `ggml_tensor` objects populated during loading.

### 5.2 Quantizer

`llama_model_quantize_internal()` (`src/llama-quant.cpp:886`) creates a model via `llama_model_create(ml, ...)` to inspect all tensors, then iterates `model->tensors_by_name` to quantize each weight.

### 5.3 Model Saver

`llama_model_saver` (`src/llama-model-saver.cpp`):
- Constructor checks `llama_model_saver_supports_arch(arch)` (line 15) - 12 architectures are explicitly unsupported (Plamo3, Gemma3, Gemma3n, Cohere2/2Moe, Olmo2, BitNet, T5, Exaone-MoE, AFMoE, Apertus, Mimo2, Step35, Mellum).
- `add_kv_from_model()` (line 145) writes `hparams` back as GGUF KV pairs.
- `add_tensors_from_model()` (line 385) writes all tensor data back.
- `save()` (line 414) calls `gguf_write_to_file()`.

### 5.4 Backend Buffer Assignment

`llama_model_loader::create_tensor()` (`src/llama-model-loader.cpp:1047`) uses `llm_tensor_info_for()` to determine expected `ggml_op` and probes each backend for op support. If a tensor cannot be placed on the preferred backend, it falls back to another buffer type (tracked for debug via `n_tensors_moved`). The `LLM_TENSOR_INFOS` map (`src/llama-arch.cpp:584-801`) is the single source of truth for these assignments.

## 6. Failure Modes

### 6.1 Unsupported Architecture

If `general.architecture` in GGUF contains an unrecognized string:
- `llm_arch_from_string()` returns `LLM_ARCH_UNKNOWN` (`src/llama-arch.cpp:857`).
- `llama_model_create(ml, params)` (`src/llama-model.cpp:322-323`) throws: `"unknown model architecture: '<name>'"`.
- Caught by `llama_model_load()` at `src/llama.cpp:335`.

### 6.2 Known Architecture Without C++ Class

If `llm_arch_from_string()` succeeds but `llama_model_mapping()` has no corresponding `new` expression:
- The `switch` falls through to the default case (`src/llama-model.cpp:300-301`), throwing: `"unsupported model architecture: '<name>'"`.

### 6.3 Missing Required GGUF Key

`ml.get_key(kid, result, true)` (default `required=true`) throws at `src/llama-model-loader.cpp:409`: `"key not found in model: %s"`. Or in `get_key_or_arr()`, line 444.

### 6.4 Corrupted Tensor Data

In `llama_tensor_weight` constructor (`src/llama-model-loader.h:46-47`):
```
if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size())
    throw std::runtime_error("tensor '%s' data is not within the file bounds, model is corrupted or incomplete");
```

### 6.5 Tensor Shape Mismatch

`check_tensor_dims()` (`src/llama-model-loader.cpp:864`) throws if expected dimensions do not match: `"tensor '%s' has wrong shape; expected %s, got %s"`.

### 6.6 Wrong Number of Tensors

`done_getting_tensors()` (`src/llama-model-loader.cpp:1317-1327`):
- If `n_created > n_tensors`: `"too many tensors created; expected %d, got %d"`.
- If `n_created < n_tensors` (and not partial load): `"wrong number of tensors; expected %d, got %d"`.

### 6.7 Invalid Tensor Data (Validation)

With `check_tensors=true`, `load_data_for()` (`src/llama-model-loader.cpp:1403-1404`) calls `ggml_validate_row_data()` and throws `"'%s' has invalid data"` on failure.

### 6.8 Split File Mismatch

- Wrong split count (`src/llama-model-loader.cpp:606`): `"invalid split count, given: %zu splits, but expected %d"`.
- Wrong split order (`src/llama-model-loader.cpp:596`): `"illegal split file idx: %d ... model must be loaded with the first split"`.
- Wrong split index in file (`src/llama-model-loader.cpp:634`): `"invalid split file idx: %d (file: %s), expected %d"`.

### 6.9 Duplicate Tensor Names

Detected during initial weight map population (`src/llama-model-loader.cpp:579`): `"invalid model: tensor '%s' is duplicated"`.

### 6.10 Tensor Not Found (Required)

`require_weight()` (`src/llama-model-loader.cpp:843`) throws: `"tensor '%s' not found"`.

### 6.11 CLIP as Main Model

`src/llama.cpp:313-314`: if `arch == LLM_ARCH_CLIP`, throws `"CLIP cannot be used as main model, use it with --mmproj instead"`.

## 7. Evidence Quality Notes

- All cited line numbers are accurate as of the current codebase state.
- The `llama_model_mapping()` switch in `src/llama-model.cpp` continues through line ~302 (there are additional entries), but the pattern (switch to return `new llama_model_<arch>(params)`) is consistent and complete.
- `llama_model_loader::create_tensor()` implementation at line 1047 is the main tensor allocation path; there is also an overload at line 1289 for tensor views, but detailed step-by-step tracing of the buffer-type-selection logic (which involves backend probing) would require deeper read of lines 1047-1288 not shown here.
- The `load_hparams` base implementation (`src/llama-model.cpp:1023-1122+`) continues past line 1122; subsequent lines handle RoPE scaling type, YaRN params, SSM params, pooling type, SWA pattern, etc. The pattern is consistent: `ml.get_key(LLM_KV_*, field, optional)`.
- Per-architecture model implementations reside in `src/llama-model.cpp` (inline) and `models/*.cpp` files. The `llama_model_mapping()` switch dispatch is the definitive reference for the architecture-to-class mapping.
- 13 architectures are explicitly blocked from re-serialization in `llama_model_saver_supports_arch()` (`src/llama-model-saver.cpp:15-35`). The remaining ~119 should round-trip successfully.

## 8. Citation Count

This document contains **59 unique file:line citations** across 12 source files:
- `include/llama.h`: 4
- `src/llama-arch.h`: 6
- `src/llama-arch.cpp`: 13
- `src/llama-model.h`: 5
- `src/llama-model.cpp`: 6
- `src/llama-model-loader.h`: 2
- `src/llama-model-loader.cpp`: 13
- `src/llama-model-saver.cpp`: 2
- `src/llama-hparams.h`: 1
- `src/llama-cparams.h`: 1
- `src/llama.cpp`: 5
- `src/llama-quant.cpp`: 1
