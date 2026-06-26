# 17. Embeddings

## Purpose

Extract dense vector representations (embeddings) of text from transformer hidden states. These vectors encode semantic meaning and are used for downstream tasks: semantic search, clustering, retrieval-augmented generation (RAG), classification, and reranking.

Without this feature the inference engine only produces logits (probability distribution over the vocabulary). Embeddings expose the model's internal representation of input tokens and sequences as fixed-size float vectors.

## Enabling embeddings

Embedding extraction is enabled per-context via the `llama_context_params.embeddings` flag (`include/llama.h:375`). This is exposed through `llama_set_embeddings(ctx, bool)` (`include/llama.h:976`, implemented at `src/llama-context.cpp:1132`).

When enabled the flag `cparams.embeddings` is set to `true`, which causes the decode path to:
1. Reserve an `embd` output buffer alongside the `logits` buffer (`src/llama-context.cpp:2074`)
2. Copy the `result_embd_pooled` tensor from GPU/backend to host after each decode (`src/llama-context.cpp:1465`, `src/llama-context.cpp:1905`)
3. Mark all tokens for output when `batch.logits` is NULL (`include/llama.h:234-237`, `src/llama-context.cpp:1702`)

## Embedding extraction API

### Per-token embeddings (pooling NONE)

```c
float * llama_get_embeddings(struct llama_context * ctx);
float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);
```

When `pooling_type == LLAMA_POOLING_TYPE_NONE` every token produces its own embedding vector. The `embd` buffer stores all token embeddings contiguously (`src/llama-context.h:290-292`). Access via `llama_get_embeddings_ith()` uses `output_ids` to map batch token position -> buffer row (`src/llama-context.cpp:887-905`).

### Sequence-pooled embeddings (MEAN, CLS, LAST)

```c
float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);
```

When pooling is active the pooled result per unique sequence is stored in the `embd_seq` map (`src/llama-context.h:322-324`). The map key is `llama_seq_id`, value is `std::vector<float>`. Returned via `llama_get_embeddings_seq()` at `src/llama-context.cpp:908-915`.

### Rerank scores (RANK)

When `pooling_type == LLAMA_POOLING_TYPE_RANK` the pooled output contains classification scores (`n_cls_out` floats per sequence) rather than an embedding vector (`src/llama-context.cpp:1497-1511`). The scores are stored in `embd_seq` and returned by `llama_get_embeddings_seq()` (`include/llama.h:1029-1030`).

### Nextn / MTP embeddings

Staging API in `src/llama-ext.h` for speculative decoding / MTP draft models: `llama_get_embeddings_nextn`, `llama_get_embeddings_nextn_ith`, `llama_get_embeddings_layer_inp`. These extract pre-norm hidden states or per-layer input embeddings needed by multi-token-prediction draft models (`src/llama-context.cpp:918-956`).

## Pooling strategies

Defined in `include/llama.h:171-178`:

| Enum | Value | Behavior |
|------|-------|----------|
| `LLAMA_POOLING_TYPE_UNSPECIFIED` | -1 | Not yet resolved; falls through to model default or NONE |
| `LLAMA_POOLING_TYPE_NONE` | 0 | Per-token embeddings; no pooling applied |
| `LLAMA_POOLING_TYPE_MEAN` | 1 | Mean-pool all tokens in sequence |
| `LLAMA_POOLING_TYPE_CLS` | 2 | Take the first token's embedding |
| `LLAMA_POOLING_TYPE_LAST` | 3 | Take the last token's embedding |
| `LLAMA_POOLING_TYPE_RANK` | 4 | Classification head applied; used for reranking models |

Pooling type is resolved at context creation time (`src/llama-context.cpp:183-187`):
- User-specified in `llama_context_params.pooling_type` overrides the model default
- Model default comes from GGUF metadata key `"%s.pooling_type"` (`src/llama-arch.cpp:203`) loaded at `src/llama-model.cpp:1050`
- If both are UNSPECIFIED, falls to `LLAMA_POOLING_TYPE_NONE`
- A warning is logged if user-specified value differs from model default (`src/llama-context.cpp:3551-3554`)

## How pooling works at the graph level

Pooling is a graph transformation applied by `llm_graph_context::build_pooling()` in `src/llama-graph.cpp:2956-3054`. The function:

1. Guards on `cparams.embeddings` (returns early if false)
2. Finds the `result_norm`/`result_embd` tensor from the model's forward pass
3. Applies the pooling operation:
   - **NONE**: Identity -- raw `result_norm` tensor used directly
   - **MEAN**: Matrix-multiply with `inp_mean` (a normalized weight per token = 1/n_tokens_per_seq) (`src/llama-graph.cpp:2987-2991`, `src/llama-graph.cpp:249-292`)
   - **CLS / LAST**: Gather rows via `inp_cls` index tensor -- CLS gathers position 0, LAST gathers the final token position (`src/llama-graph.cpp:2992-2997`, `src/llama-graph.cpp:295-319`)
   - **RANK**: Mean or CLS pooling followed by classification head: `dense` -> `tanh`/`gelu` -> optional `cls_norm` -> optional `cls_out` -> optional softmax (`src/llama-graph.cpp:2998-3043`)
4. Labels the result tensor `"result_embd_pooled"` and adds it to the compute graph

The `inp_mean` and `inp_cls` input tensors are populated per-ubatch in `set_input()` methods (`llm_graph_input_mean::set_input`, `llm_graph_input_cls::set_input`) at `src/llama-graph.cpp:249-319`.

After graph execution the pooled tensor is copied to host via `ggml_backend_tensor_get_async` in `src/llama-context.cpp:1464-1516` (encoder path) and `src/llama-context.cpp:1904-1961` (decoder path).

### Note on `llama_pool`

No function named `llama_pool` exists in the codebase. The pooling graph operation is performed by `build_pooling()` and there is no separate public API function for applying pooling post-hoc. This name appears only in the research backlog as a potential future feature.

## Embeddings vs Logits

| Aspect | Logits | Embeddings |
|--------|--------|------------|
| Shape | `[n_outputs, n_vocab]` | `[n_outputs, n_embd_out]` or `[n_seqs, n_embd_out]` |
| Semantic meaning | Probability distribution over tokens | Dense vector representation of input |
| Storage | `logits` buffer (`src/llama-context.h:288`) | `embd` / `embd_seq` buffers (`src/llama-context.h:290-292, 322-324`) |
| Extraction API | `llama_get_logits_ith()` | `llama_get_embeddings_ith()` / `llama_get_embeddings_seq()` |
| Pooling applied | Never | Yes (graph-level transformation) |
| Post-processing | Softmax / sampling | Normalization (`common_embd_normalize`) |
| Backend output tensor | `t_logits` | `t_embd` (from `result_embd_pooled`) |

Both logits and embeddings are copied from backend to host simultaneously during `llama_decode()` (`src/llama-context.cpp:1460-1462` vs `1464-1516`).

## Normalization helper

`common_embd_normalize()` in `common/common.cpp:1734-1768` supports:

| Norm value | Meaning |
|-----------|---------|
| -1 | None (identity) |
| 0 | Max-absolute-int16 (scale to fit int16 range) |
| 1 | Taxicab / L1 |
| 2 | Euclidean / L2 (default) |
| >2 | P-norm |

`common_embd_similarity_cos()` computes cosine similarity between two embedding vectors (`common/common.cpp:1770`).

## Harness and examples

### `examples/embedding/` (llama-embedding)

CLI tool for batch embedding extraction (`examples/embedding/embedding.cpp`). Supports:
- Multiple input prompts separated by `--embd-separator`
- All pooling types including RANK with classification pair splitting
- Output formats: plain, array, JSON (OpenAI-compatible), JSON+ with cosine similarity matrix, raw
- Normalization control via `--embd-normalize`
- The `LLAMA_EXAMPLE_EMBEDDING` flag sets `params.embedding = true` (`embedding.cpp:108`)

Usage example: `./llama-embedding -m model.gguf --pooling mean -p "Hello World!"`

### `examples/retrieval/` (llama-retrieval)

Interactive RAG demo (`examples/retrieval/retrieval.cpp`):
- Chunks files by configurable separator and chunk size
- Embeds chunks with `llama_get_embeddings_seq()`
- Interactive loop: prompt query -> embed -> find top-k by cosine similarity

### No `tools/embedding/` directory

The research backlog references `tools/embedding/` but this directory does not exist. The primary harness is `examples/embedding/`.

## Server touch points

The server (`tools/server/`) has full embedding support:

- **Routes**: `POST /embedding` (legacy), `POST /embeddings`, `POST /v1/embeddings` (OpenAI-compatible) (`tools/server/server.cpp:219-221`)
- **Task type**: `SERVER_TASK_TYPE_EMBEDDING` (`tools/server/server-task.h:18`)
- **Extraction path**: `server_slot::send_embedding()` at `tools/server/server-context.cpp:2173-2216`
- **Per-token (NONE)**: calls `llama_get_embeddings_ith()` for each token with `batch.logits[i] != 0`
- **Pooled**: calls `llama_get_embeddings_seq()` and breaks after first hit per seq
- **Rerank**: `server_slot::send_rerank()` at `tools/server/server-context.cpp:2218-2247`
- **Pooling restriction**: `can_split()` returns false when pooling is not LAST or memory is absent (`tools/server/server-context.cpp:377-382`)
- **Capability gating**: server must be started with `--embeddings`; the `need_embd()` method gates whether embedding extraction is enabled per-task (`tools/server/server-task.h:183-191`)
- **OAI compliance**: `/v1/embeddings` rejects `pooling_type == NONE` (`tools/server/server-context.cpp:5198-5202`)
- **Normalization**: pooled embeddings are normalized via `common_embd_normalize`; per-token (NONE) embeddings are returned raw (`tools/server/server-context.cpp:2204-2208`)
- **Tests**: `tools/server/tests/unit/test_embedding.py` and `tools/server/tests/unit/test_vision_api.py`

## Touch points summary

| Component | Files | Role |
|-----------|-------|------|
| Public API | `include/llama.h:171-178, 375, 556, 564, 974-976, 1012-1032` | Pooling type enum, params, accessors |
| Context | `src/llama-context.cpp:67-69, 183-187, 1132-1139, 1464-1516, 1882-1961, 2071-2097` | Enable flag, buffer allocation, tensor copy |
| Context state | `src/llama-context.h:290-292, 322-324` | `embd` buffer, `embd_seq` map |
| Graph pooling | `src/llama-graph.cpp:249-319, 2956-3054` | `build_pooling()`, mean/cls input builders |
| Model loading | `src/llama-model.cpp:1050, 2237-2238` | Read `pooling_type` from GGUF, append pooling layer |
| Server | `tools/server/server-context.cpp:2173-2247, 5192-5288` | `/embeddings`, `/v1/embeddings` endpoints |
| Example tools | `examples/embedding/embedding.cpp`, `examples/retrieval/retrieval.cpp` | CLI harness, interactive RAG demo |
| Common utilities | `common/common.h:1005-1007`, `common/common.cpp:1734-1783` | Normalization, cosine similarity |

## Failure modes

1. **Encoder-decoder models**: explicitly unsupported (`examples/embedding/embedding.cpp:154-157`)
2. **Pooling NONE with OAI endpoint**: `/v1/embeddings` rejects it (`tools/server/server-context.cpp:5198-5202`)
3. **Pooling NONE in retrieval example**: blocks with an error (`examples/retrieval/retrieval.cpp:171-174`)
4. **Empty input**: `/v1/embeddings` checks `tokenized_prompts` is non-empty (`tools/server/server-context.cpp:5232-5235`)
5. **Null embeddings**: `llama_get_embeddings_seq()` returns nullptr if seq_id not found; `llama_get_embeddings_ith()` returns nullptr or aborts depending on NDEBUG (`src/llama-context.cpp:891-905, 908-915`)
6. **Batch-size limits**: embedding models require `n_batch >= n_ctx`; the example enforces this or warns (`examples/embedding/embedding.cpp:123-126`)
7. **No embedding support at server level**: returns 501 if server started without `--embeddings` (`tools/server/server-context.cpp:5195`)
8. **Missing SEP/EOS token**: warning logged if last token is not SEP or EOS (`examples/embedding/embedding.cpp:225-229`)
