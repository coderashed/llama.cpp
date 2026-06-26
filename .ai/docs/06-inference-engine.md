# Core LLM Inference Engine

## Purpose

The inference engine transforms batches of token IDs (or precomputed embeddings) into logits and/or embeddings via a neural network forward pass through the loaded model. It orchestrates token embedding lookup, positional encoding, multi-head self-attention (with KV-cache lookup/update), feed-forward computation (dense or MoE), output projection, and pooling, across one or more compute backends (CPU, CUDA, Metal, Vulkan, etc.).

It is the central entry point for all generation and embedding extraction workflows. Sampling, KV-cache management, LoRA adapters, control vectors, and backend sampling are all layered on top of the inference engine.

---

## Context Lifecycle: `llama_init_from_model`

### Creation

A `llama_context` is created via `llama_init_from_model` (`src/llama-context.cpp:3490`). The deprecated alias `llama_new_context_with_model` (line 3575) simply forwards.

The constructor (`src/llama-context.cpp:33`) performs:

1. **Parameter validation** (`src/llama-context.cpp:3493-3562`): checks for null model, zero batch/ctx, unsupported arch/backend combinations, flash-attention constraints, MTP presence.

2. **Compute parameters (`llama_cparams`) initialization** (`src/llama-context.cpp:49-243`): resolves `n_ctx` (padded to multiple of 256), `n_batch`, `n_ubatch`, `n_seq_max`, RoPE/YaRN parameters, pooling type, attention type, flash-attention toggle, thread counts, and pipeline-parallel mode.

3. **Backend initialization** (`src/llama-context.cpp:268-306`): GPU backends from `model.devices`, ACCEL backends (e.g. BLAS), then CPU backend.

4. **Memory module creation** (`src/llama-context.cpp:324-333`): calls `model.create_memory(params_mem, cparams)` which returns a `llama_memory_i` instance (KV cache, recurrent state, or hybrid). The KV cache is the standard implementation (`llama_kv_cache`, `src/llama-kv-cache.h:20`).

5. **Scheduler reservation** (`src/llama-context.cpp:399`): via `sched_reserve()` (line 439), which creates a `ggml_backend_sched`, allocates worst-case graph results, resolves auto-FA/fused-GDN, and reserves compute buffers via `graph_reserve()`.

6. **Backend sampler setup** (`src/llama-context.cpp:115-129`): if `ctx_params.samplers` are provided, attaches per-sequence sampler chains.

### State held by `llama_context`

Per `llama_context` (`src/llama-context.h:42`):

| Field | Type | Purpose |
|-------|------|---------|
| `model` | `const llama_model &` | Reference to the loaded model (weights, hparams, architecture) |
| `cparams` | `llama_cparams` | Resolved compute parameters (`src/llama-cparams.h:10`) |
| `memory` | `llama_memory_ptr` | KV cache / recurrent state (`src/llama-memory.h:73`) |
| `sched` | `ggml_backend_sched_ptr` | Backend scheduler for graph execution |
| `logits` | `buffer_view<float>` | Output logits buffer `[n_outputs][n_vocab]` |
| `embd` | `buffer_view<float>` | Output embeddings buffer `[n_outputs][n_embd]` |
| `embd_seq` | `map<llama_seq_id, vector<float>>` | Pooled sequence embeddings |
| `balloc` | `unique_ptr<llama_batch_allocr>` | Reusable batch allocator |
| `output_ids` | `vector<int32_t>` | Maps batch token positions to logit/embd buffer indices |
| `buf_output` | `ggml_backend_buffer_ptr` | Host-side output buffer |
| `sampling` | `sampling_info` | Backend sampling state (samplers, sampled tokens, probs) |
| `cvec` | `llama_adapter_cvec_ptr` | Control vector |
| `loras` | `llama_adapter_loras_ptr` | LoRA adapters |
| `cross` | `llama_cross` | Encoder-decoder cross-attention state |
| `gf_res_prev` | `llm_graph_result_ptr` | Previous graph result (for reuse) |
| `gf_res_reserve` | `llm_graph_result_ptr` | Reserve graph result (for buffer sizing) |
| Thread pools | `ggml_threadpool_t` | `threadpool` (single-token) and `threadpool_batch` (multi-token) |
| `set_n_threads_fns` | `vector<fn>` | Per-backend thread-setter functions |

### Destruction

`llama_free` (`src/llama-context.cpp:3581`) calls `delete ctx`. The destructor (`src/llama-context.cpp:419`) logs compute buffer sizes and frees the optimization context if present.

---

## Graph Building: `llama_model::build_graph`

Graph construction is triggered in `llama_context::process_ubatch` (`src/llama-context.cpp:1337`) when the previous graph cannot be reused (`can_reuse` returns false at line 1318).

### Entry

The public method `llama_model::build_graph` (`src/llama-model.cpp:2234`) receives an `llm_graph_params` struct that packages:

- `arch` / `hparams` / `cparams` -- architecture and parameters
- `ubatch` -- the micro-batch to process
- `gtype` -- graph type: DEFAULT, ENCODER, DECODER, or DECODER_MTP
- `sched` / `backend_cpu` -- scheduler and CPU backend handles
- `cvec` / `loras` / `mctx` / `cross` -- optional modifiers
- `samplers` -- per-sequence backend samplers
- `n_outputs` / `cb` / `res` -- output settings

### Architecture-specific graph

Each model architecture provides `build_arch_graph` (e.g., `src/models/llama.cpp:94` for LLaMA). The topology for a standard decoder-only transformer (LLaMA family):

1. **`build_inp_embd(tok_embd)`** (`src/llama-graph.cpp:1833`): Creates I32 input tensor `inp_tokens` and F32 input tensor `inp_embd`. Uses `ggml_get_rows` for token->embedding lookup. Supports LoRA on the embedding, padding for deepstack, and embedding scaling.

2. **`build_inp_pos()`** (`src/llama-graph.cpp:1922`): Creates I32 position tensor.

3. **`build_attn_inp_kv()`** (`src/llama-graph.h:997`): Creates KV-cache index tensors (`k_idxs`, `v_idxs`) and attention mask (`kq_mask`). Mask dimensions: `[n_kv, n_tokens/n_stream, 1, n_stream]`.

4. **Per-layer loop** (`src/models/llama.cpp:126-228`): For each of `n_layer` layers:
   - **Attention block**: RMS norm -> QKV projection (`build_qkv`, `src/llama-graph.h:880`) -> RoPE (`ggml_rope_ext`) -> optional KQ norm -> attention (`build_attn`) -> output projection + residual add
   - **FFN block**: RMS norm -> `build_ffn` (dense SiLU-gated, `src/llama-graph.h:888`) or `build_moe_ffn` (MoE, line 905) -> residual add
   - **Control vector**: `build_cvec` applied per layer

5. **Final norm + output**: RMS norm -> `build_lora_mm(model.output)` -> logits tensor

6. **Post-architecture layers** (`src/llama-model.cpp:2237-2249`):
   - `build_pooling` -- pooled embeddings (mean/CLS/LAST/RANK)
   - `build_sampling` -- backend sampling (on-device sampling)
   - `build_dense_out` -- optional sentence-transformers dense layers
   - `res->set_outputs(params)` -- registers output tensors with the result

### Graph result (`llm_graph_result`)

Defined at `src/llama-graph.h:703`. Holds:
- `gf` -- the `ggml_cgraph` compute graph
- `ctx_compute` -- ggml compute context
- `buf_compute_meta` -- metadata buffer
- `inputs` -- vector of `llm_graph_input_i` subclasses (embd, pos, attn, etc.)
- Output tensors: `t_inp_tokens`, `t_inp_embd`, `t_logits`, `t_embd`, `t_embd_pooled`, `t_h_nextn`
- Per-layer input tensors: `t_layer_inp[il]`
- Sampling tensors: `t_sampled_logits`, `t_candidates`, `t_sampled`, `t_sampled_probs`
- `params` -- snapshot of the `llm_graph_params` used, for reuse comparison

### Graph reuse

`llm_graph_result::can_reuse` (at `src/llama-graph.h:731`) checks if new `llm_graph_params` would produce a graph with identical topology. If so, only input tensor data is updated, avoiding rebuild/alloc overhead. Fields compared include ubatch shape, n_outputs, samplers, embeddings flags, arch, gtype, cvec, loras, and cross.

### Graph execution

`llama_context::graph_compute` (`src/llama-context.cpp:2421`) sets thread counts, calls `ggml_backend_sched_graph_compute_async`, and returns the status.

---

## The Decode Loop: `llama_decode`

### Public entry

`llama_decode` (`src/llama-context.cpp:4054`) calls `ctx->decode(batch)` and logs any error for non-zero returns.

### `llama_context::decode` (`src/llama-context.cpp:1680`)

The decode method implements the full inference pipeline:

1. **Precondition checks** (lines 1681-1693): at least one of `token`/`embd` must be non-NULL; `n_tokens > 0`.

2. **Backend sampling validation** (lines 1708-1729): if backend samplers are active, at most one output per sequence is required.

3. **Batch initialization** (`balloc->init`, line 1731): the `llama_batch_allocr` (`src/llama-batch.h:72`) sanitizes tokens/seq_ids, auto-generates missing position data from memory, and auto-fills seq_id=0 when none provided. Returns false for invalid input.

4. **Scheduler reservation** (`sched_reserve()`, line 1761): allocates compute buffers if not yet done.

5. **Memory update** (`memory_update(false)`, line 1766): handles any pending KV-cache shifts/copies.

6. **Memory `init_batch` loop** (lines 1770-1811):
   - Calls `memory->init_batch(*balloc, cparams.n_ubatch, output_all)` which splits user batch into micro-batches and finds KV slots.
   - Status `FAILED_PREPARE`: may retry after `memory_update(true)` (cache optimization/defrag). If still fails, returns 1 (no KV slot).
   - Status `FAILED_COMPUTE`: returns -2.

7. **Output buffer reservation** (`output_reserve`, line 1814): grows `logits`, `embd`, `embd_nextn`, and sampling buffers if needed.

8. **Micro-batch loop** (lines 1822-2000):
   - Gets next ubatch from `mctx->get_ubatch()`.
   - Counts outputs in this ubatch.
   - Calls `process_ubatch(ubatch, gtype, mctx)` which:
     - Builds/rebuilds graph (or reuses)
     - Sets input tensor data
     - Computes graph
   - Extracts logits: `ggml_backend_tensor_get_async(logits_out, n_outputs*n_vocab*sizeof(float))`
   - Extracts embeddings (per-token or pooled per-sequence)
   - Extracts `h_nextn` (for MTP speculation)
   - Extracts layer input embeddings (if enabled)
   - Copies backend sampling results (sampled tokens, probs, logits, candidates)
   - Advances to next ubatch: `mctx->next()`

9. **Output reordering** (lines 2006-2049): the `output_ids` map is built to match user-provided batch ordering. Selection-sort swaps are recorded in `output_swaps` and applied lazily on access.

10. **Return**: 0 on success.

---

## Batching: `llama_batch` and `llama_batch_get_one`

### `llama_batch` (`include/llama.h:240`)

The public input batch struct:

| Field | Type | Description |
|-------|------|-------------|
| `n_tokens` | `int32_t` | Number of tokens in this batch |
| `token` | `llama_token *` | Token IDs (used when `embd` is NULL) |
| `embd` | `float *` | Precomputed embeddings (used when `token` is NULL) |
| `pos` | `llama_pos *` | Positions of each token (NULL = auto-tracked) |
| `n_seq_id` | `int32_t *` | Number of sequence IDs per token (NULL = 1 per token) |
| `seq_id` | `llama_seq_id **` | Sequence IDs per token (NULL = all seq_id=0) |
| `logits` | `int8_t *` | Per-token flag: non-zero means output logits for this token (NULL = last-token-only, or all for embeddings) |

### `llama_batch_get_one` (`src/llama-batch.cpp:863`)

Helper for single-sequence decode. Sets tokens and n_tokens only; all ancillary fields are NULL, triggering auto-fill in `llama_batch_allocr::init`:
- Positions auto-tracked from memory state
- seq_id assumed 0
- logits flag: NULL means output only the last token

### `llama_batch_init` / `llama_batch_free` (`src/llama-batch.cpp:877`)

Allocates/frees heap-backed batches with explicit per-token sequence ID support.

### `llama_ubatch` (`src/llama-batch.h:15`)

Internal micro-batch representation. Fields include `n_tokens`, `n_seq_tokens`, `n_seqs`, `token`, `embd`, `pos`, `seq_id`, `output` (output flags). The `equal_seqs` flag indicates whether all sequences have the same number of tokens (enabling more efficient graph topology).

### `llama_batch_allocr` (`src/llama-batch.h:72`)

Splits a user batch into micro-batches:
- `split_simple` -- arbitrary sequence lengths
- `split_equal` -- equal-length sequence sets
- `split_seq` -- one sequence-set per ubatch
- `ubatch_reserve` -- creates a well-defined ubatch for scheduling

---

## Compute Graph Topology for a Forward Pass

The following describes the ggml compute graph for a standard LLaMA-style decoder forward pass. The actual structure is built in `src/models/llama.cpp:99`.

```
Input tokens (I32 [n_batch])
  |
  v
ggml_get_rows (tok_embd: F32 [n_embd, n_vocab], inp_tokens)
  --> inpL: F32 [n_embd, n_batch]                              [src/llama-graph.cpp:1858]
  |
  v
Position input (I32 [n_batch])                                  [src/llama-graph.cpp:1927]
KV cache indices (I64 [n_batch])                                [src/llama-graph.h:326-327]
Attention mask (F32/F16 [n_kv, n_tokens, 1, n_stream])         [src/llama-graph.cpp:36]

For each layer il in [0, n_layer):
  |
  attn_norm = ggml_rms_norm(inpL)                              [src/llama-graph.h:870]
  Qcur, Kcur, Vcur = ggml_mul_mat(wq/wk/wv, attn_norm)         [src/llama-graph.h:880]
  Qcur = ggml_rope_ext(Qcur, inp_pos, rope_factors)            [src/models/llama.cpp:146]
  Kcur = ggml_rope_ext(Kcur, inp_pos, rope_factors)            [src/models/llama.cpp:152]
  |
  [KV cache: cpy_k(Kcur, k_idxs) -> store in cache cells]
  [KV cache: get_k(...) -> k_from_cache]
  [KV cache: get_v(...) -> v_from_cache]
  |
  attn_out = build_attn(inp_attn, wo, Qcur, Kcur, Vcur)        [src/llama-graph.h:999]
    internal: ggml_flash_attn_ext or ggml_mul_mat-based MHA    [src/llama-graph.cpp:2066]
    -> ggml_mul_mat(wo, attn_out)                               [src/llama-graph.h:1011]
  |
  inpSA = inpL
  ffn_inp = ggml_add(attn_out, inpSA)                          [src/models/llama.cpp:178]
  |
  ffn_norm = ggml_rms_norm(ffn_inp)                            [src/models/llama.cpp:184]
  |
  if dense FFN:
    ffn_out = build_ffn(ffn_norm, up, gate, down)              [src/llama-graph.h:888]
      internal: up = ggml_mul_mat(w_up, cur)
                gate = ggml_mul_mat(w_gate, cur)
                act = ggml_silu(gate)                           [LLM_FFN_SILU]
                cur = ggml_mul(up, act)
                out = ggml_mul_mat(w_down, cur)
  if MoE FFN:
    ffn_out = build_moe_ffn(ffn_norm, gate_inp, exp_weights)   [src/llama-graph.h:905]
      internal: router_logits = ggml_mul_mat(gate_inp, cur)
                selected_experts = ggml_softmax(topk router)
                for each expert: expert_ffn(selected tokens)
                cur = sum(scaled expert outputs)
  |
  cur = ggml_add(ffn_out, ffn_inp)                              [src/models/llama.cpp:220]
  cur = build_cvec(cur, il)                                     [src/llama-graph.h:853]
  inpL = cur                                                    [src/models/llama.cpp:227]

Final:
  cur = ggml_rms_norm(inpL)                                     [src/models/llama.cpp:231]
  t_embd = cur                                                  [src/models/llama.cpp:236]
  t_logits = ggml_mul_mat(output, cur)                          [src/models/llama.cpp:240]
  |
  [optional: build_pooling, build_sampling, build_dense_out]
  |
  ggml_build_forward_expand(gf, cur)                            [src/models/llama.cpp:246]
```

### Major ggml operations

| Operation | Purpose |
|-----------|---------|
| `GGML_OP_GET_ROWS` | Token embedding lookup |
| `GGML_OP_MUL_MAT` | Matrix multiply (QKV projections, FFN, output) |
| `GGML_OP_RMS_NORM` | RMS layer normalization |
| `GGML_OP_ROPE` | Rotary Position Embedding |
| `GGML_OP_FLASH_ATTN_EXT` | Flash Attention (fused QK^T V) |
| `GGML_OP_MUL_MAT` (in attention) | Scaled dot-product attention (non-flash path) |
| `GGML_OP_SOFT_MAX` (in attention) | Attention softmax |
| `GGML_OP_SILU` | SiLU activation in FFN |
| `GGML_OP_ADD` | Residual connections |
| `GGML_OP_SCALE` | Embedding/attention scaling |
| `GGML_OP_VIEW` / `GGML_OP_PERMUTE` | Tensor reshaping for attention |
| `GGML_OP_CPY` | KV cache copy operations |
| `GGML_OP_SET_GROUPS` | KV cache set_rows for k/v storage |
| `GGML_OP_SOFT_MAX` (MoE) | Expert gating softmax |
| `GGML_OP_MUL` | Element-wise gating in FFN |

---

## Touch Points with Other Features

### Sampling

Sampling reads logits produced by inference:
- `llama_get_logits` / `llama_get_logits_ith` (`include/llama.h:1004-1010`) return pointers into the `buf_output` buffer populated by the decode loop at `src/llama-context.cpp:1890-1901`.
- Backend sampling (on-device) is integrated into the graph: `build_sampling` at `src/llama-graph.h:1142` adds sampling ops to the compute graph. Results are copied back at `src/llama-context.cpp:1986-1996`.

### KV Cache

The KV cache (`llama_kv_cache`, `src/llama-kv-cache.h:20`) is the primary memory implementation. It:
- Manages cell allocation per sequence (`find_slot`, `src/llama-kv-cache.cpp:907`)
- Is updated during graph execution via `cpy_k`/`cpy_v` operations
- Is read during attention via `get_k`/`get_v`
- Triggers cache shifts via `init_update` when context runs full

### LoRA Adapters

Applied during graph building:
- `build_lora_mm` (`src/llama-graph.h:858`) wraps `ggml_mul_mat` with LoRA weight addition
- `build_lora_mm_id` (line 864) for per-expert MoE LoRA
- Token embedding LoRA at `src/llama-graph.cpp:1861-1876`
- Set via `llama_set_adapters_lora` (`include/llama.h:683`)

### Control Vectors

Applied via `build_cvec` (`src/llama-graph.h:853`) at each layer output. Set via `llama_set_adapter_cvec` (`include/llama.h:695`).

### Encoder-Decoder (Cross-Attention)

Cross-attention state is held in `llama_cross` (`src/llama-graph.h:63`). The encoder stores embeddings via `llama_encode`, and the decoder reads them via `build_attn_cross`.

### Backend Sampling

When the context is created with `ctx_params.samplers`, backend sampling nodes are added to the graph (`build_sampling`). Results are available via `llama_get_sampled_token_ith` and related functions (`include/llama.h:1039-1058`).

### Training / Optimization

The inference engine also supports gradient-based training via `opt_epoch` (`src/llama-context.h:199`), reusing the same graph building machinery but extending it with backward ops.

---

## Failure Modes

`llama_decode` returns (`include/llama.h:954-961`):

| Return | Meaning | Trigger |
|--------|---------|---------|
| `0` | Success | |
| `1` | No KV slot found | KV cache full; try reducing batch or increasing context |
| `2` | Aborted | `abort_callback` returned true during compute |
| `-1` | Invalid input | `n_tokens == 0`, invalid token IDs, invalid seq_ids, duplicate outputs with backend sampling |
| `-2` | Fatal error (prepare) | Memory module `FAILED_PREPARE` after optimization; memory `FAILED_COMPUTE`; output buffer alloc failure |
| `-3` | Fatal error (compute) | `ggml_backend_sched_graph_compute_async` returned `GGML_STATUS_FAILED` |

Internal error propagation:
- `process_ubatch` returns `nullptr` on failure, setting `ret` to `GGML_STATUS_ABORTED`, `GGML_STATUS_ALLOC_FAILED`, or `GGML_STATUS_FAILED` (`src/llama-context.cpp:1341-1369`).
- `graph_compute` returns the status from `ggml_backend_sched_graph_compute_async` (`src/llama-context.cpp:2440`).
- Memory module returns `LLAMA_MEMORY_STATUS_FAILED_PREPARE` from `init_batch` when no KV slot can be found (`src/llama-kv-cache.cpp:745`).
- Input validation in `llama_batch_allocr::init` returns false for invalid tokens/seq_ids (`src/llama-batch.cpp:44-67`).
