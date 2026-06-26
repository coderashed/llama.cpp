# Speculative Decoding

## Purpose

Speculative decoding accelerates autoregressive token generation by having a cheaper **draft model** (or heuristic) predict several tokens ahead, then having the **target model** verify/accepted them all in a single forward pass. Because batched evaluation is far more efficient than sequential token-by-token decoding, this yields significant speedups when draft tokens are frequently correct.

The core intuition (described in `docs/speculative.md:5`): computing `n` tokens in a batch costs much less than computing `n` sequentially.

## Architecture

The speculative decoding system lives in **`common/speculative.{h,cpp}`**, not in the core `src/llama` library. It is a pure-common-layer abstraction that orchestrates multiple **implementations** via a polymorphic base class.

### Key types

- **`struct common_speculative`** (`speculative.h:6`, `speculative.cpp:1772`) — opaque owning struct holding a vector of `common_speculative_impl` unique_ptrs plus per-sequence draft params and a back-pointer to which impl produced the last draft.
- **`struct common_speculative_impl`** (`speculative.cpp:129`) — abstract base class with virtual methods: `begin()`, `process()`, `draft()`, `accept()`, `need_embd()`, `need_embd_nextn()`. Each subclass implements one draft strategy.
- **`struct common_speculative_draft_params`** (`speculative.h:30`) — per-sequence drafting parameters passed between the server and the speculative layer. Contains `drafting` flag, `n_max` cap, `n_past`, `id_last`, `prompt` pointer, and `result` output pointer.
- **`enum common_speculative_type`** (`common.h:160`) — enumerates the 9 types (NONE + 8 implementations).
- **`struct common_params_speculative`** (`common.h:361`) — configuration struct holding per-type sub-configs and the type list.

> **Note**: There are **no** types named `llama_speculative`, `llama_draft_model`, or `llama_verify` anywhere in the codebase. These names do not appear in any `.h` or `.cpp` file. The speculative abstraction lives entirely in the `common/` layer.

## Draft Model Workflow (`common/speculative.cpp`)

### Initialization

`common_speculative_init()` (`speculative.cpp:1919`) takes a `common_params_speculative` and an `n_seq` count. It:
1. Determines which implementations are enabled from the `types` bitmask (`speculative.cpp:1923-1935`)
2. Builds a prioritized list of configs (priority order: `ngram-simple` > `ngram-map-k` > `ngram-map-k4v` > `ngram-mod` > `ngram-cache` > `draft-simple` > `draft-eagle3` > `draft-mtp`, see `speculative.cpp:1942-1967`)
3. Instantiates one `common_speculative_impl` subclass per enabled config (`speculative.cpp:1972-2034`)

### Draft generation: `common_speculative_draft()`

The draft orchestration (`speculative.cpp:2121`):
1. Iterates through implementations in priority order
2. For each impl, calls `impl->draft(dparams)` (`speculative.cpp:2147`)
3. If an impl produces a non-empty draft for a sequence, that sequence's `drafting` flag is cleared (`speculative.cpp:2159-2160`)
4. Falls through to the next impl for sequences that still have `drafting == true` (`speculative.cpp:2182-2188`)
5. After all impls, any still-drafting sequences have their flag forcibly cleared (`speculative.cpp:2193-2199`)

This **chaining** mechanism allows mixing implementations: e.g., `ngram-mod` then `ngram-map-k4v` then `draft-simple`, where the first to produce a draft wins.

### Implementation strategies

| Type | File | Mechanism | Requires ctx_dft | Extracts h_nextn |
|---|---|---|---|---|
| `draft-simple` | `speculative.cpp:175` | Standalone draft model, top-k=10 sampling, p_min early stop | Yes | No |
| `draft-eagle3` | `speculative.cpp:419` | Reads target hidden states → encoder → decoder; deferred boundary KV pattern | Yes | Yes (layer_inp) |
| `draft-mtp` | `speculative.cpp:896` | Multi-Token Prediction heads; 3 modes (plain, chain_heads, mem_shared) | Yes | Yes (nextn) |
| `ngram-simple` | `speculative.cpp:1343` | Simple n-gram lookup in history | No | No |
| `ngram-map-k` | `speculative.cpp:1392` | Hash-map key→value lookup, tracks acceptance stats | No | No |
| `ngram-map-k4v` | `speculative.cpp:1392` | Same as map-k but with 4 value slots per key (higher quality) | No | No |
| `ngram-mod` | `speculative.cpp:1450` | Rolling LCG hash per n-gram, ~16MB shared pool; adaptive reset on low accept | No | No |
| `ngram-cache` | `speculative.cpp:1629` | 3-level cache (context + dynamic + static); file-loadable lookups | No | No |

### Compatibility check for draft model

`common_speculative_are_compatible()` (`speculative.cpp:56`) verifies:
- Vocab type must match (difference < `SPEC_VOCAB_MAX_SIZE_DIFFERENCE` = 128, `speculative.cpp:21`)
- BOS/EOS add flags and token IDs must match
- Token content must match from token ID 5 onwards (`SPEC_VOCAB_CHECK_START_TOKEN_ID`)
- Same check logic appears in `examples/speculative/speculative.cpp:107-157`

## Verification — how the target model accepts/rejects draft tokens

There are two verification paths:

### A. Simple greedy verification (used by `speculative-simple` example)

`common_sampler_sample_and_accept_n()` (`sampling.cpp:624`) iterates through draft positions:
1. Sample the target model's token at position `i` (`sampling.cpp:632`)
2. If it matches `draft[i]`, accept and continue (`sampling.cpp:638`)
3. If it doesn't match, stop and return accepted prefix (`sampling.cpp:638-639`)
4. Always returns at least 1 token (the target-sampled token)

For stochastic verification (`temp > 0`), the `examples/speculative/speculative.cpp` uses rejection sampling on probability ratios (`p_tgt / p_dft`, line 302) with residual distribution correction (`speculative.cpp:316-346`).

### B. Server-integrated verification

In `server-context.cpp:3805-3928`:
1. Save sampler state clone (`line 3819`)
2. Call `common_sampler_sample_and_accept_n()` with `spec_i_batch` indices and `spec_draft` tokens (`line 3822`)
3. Compute `n_rollback = draft.size() + 1 - accepted.size()` (`line 3827`)
4. If partial acceptance and context only supports full rollback: restore checkpoint, truncate draft, recycle (`lines 3834-3862`)
5. Call `common_speculative_accept()` for stats and impl-specific state update (`line 3870`)
6. Process all accepted tokens, send to client, clear KV beyond `n_past` (`lines 3904-3922`)

## Sampling integration

`common_sampler_sample_and_accept_n()` (`sampling.h:83`, `sampling.cpp:624`) is the key bridge between sampling and speculative decoding:
- Takes a sampler, context, batch indices, and draft tokens
- Reuses the normal `common_sampler_sample()` at each position
- Returns the vector of accepted tokens (length >= 1)
- Used by both `speculative-simple` (`speculative-simple.cpp:249`) and server (`server-context.cpp:3822`)

## Inference engine touch points

1. **`llama_decode(ctx_tgt, batch)`** — the core decode that evaluates target + draft tokens in one batch (server-context.cpp:3599)
2. **`llama_decode(ctx_dft, batch)`** — decode on draft model, called inside each impl's `draft()` and `process()` methods
3. **`llama_encode(ctx_dft, batch)`** — EAGLE3 encoder call (`speculative.cpp:622`)
4. **`llama_get_embeddings_nextn(ctx)` / `llama_get_embeddings_nextn_ith(ctx, i)`** — extract the target model's hidden states for MTP/EAGLE3 (`speculative.cpp:12`, staging API in `src/llama-ext.h`)
5. **`llama_set_embeddings_nextn(ctx, true, masked)`** — enable pre-norm embedding extraction (`speculative.cpp:511, 986-987`)
6. **`llama_set_embeddings_layer_inp(ctx, layer_id, true)`** — EAGLE3 enables extraction of specific target layer inputs (`speculative.cpp:506`)
7. **`llama_memory_seq_rm`, `llama_memory_seq_keep`, `llama_memory_seq_cp`** — KV cache management for draft/target context after partial acceptance

## Performance characteristics and tuning parameters

From `docs/speculative.md:342-365` and `speculative.cpp:2260-2303`, each implementation prints:
- `#calls(b,g,a)`: begin/draft/accept call counts
- `#gen drafts` / `#acc drafts`: how many drafts were generated vs accepted
- `#gen tokens` / `#acc tokens`: total tokens generated vs accepted
- `dur(b,g,a)`: timing breakdown in ms
- Mean acceptance length and per-position acceptance rates

Key tuning knobs (`common.h:316-372`):
| Parameter | Default | Effect |
|---|---|---|
| `draft.n_max` | 3 | Max draft tokens; higher = more potential speedup but more waste on rejections |
| `draft.n_min` | 0 | Min draft tokens; discard drafts shorter than this |
| `draft.p_min` | 0.0 | Confidence threshold: stop drafting early if probability drops below this (saves compute) |
| `draft.p_split` | 0.1 | Tree-splitting probability for multi-branch speculation (`speculative.cpp:61`) |
| `ngram_mod.n_match` | 24 | Lookup length for rolling hash |
| `ngram_mod.n_max` / `n_min` | 64 / 48 | Draft length bounds for ngram-mod |
| `ngram_simple.size_n` / `size_m` | 12 / 48 | N-gram lookup and draft sizes |
| `ngram_map_k.size_n` / `size_m` | 12 / 48 | Same for map variants |

## Harnesses

### `examples/speculative/speculative.cpp`
Full-featured demonstration (`examples/speculative/README.md:3`):
- Tree-based multi-sequence speculative decoding (`n_seq_dft` branches, line 58)
- Stochastic verification with rejection sampling (`speculative.cpp:256-380`) or greedy (`speculative.cpp:381-406`)
- Branch splitting based on `p_draft_split` probability (`speculative.cpp:518-556`)
- Manual KV cache management: `llama_memory_seq_keep`, `llama_memory_seq_cp`, `llama_memory_seq_rm`
- Prints draft statistics: n_draft, n_predict, n_drafted, n_accept, acceptance % (`speculative.cpp:633-637`)

### `examples/speculative-simple/speculative-simple.cpp`
Simpler demonstration (`examples/speculative-simple/README.md:3`):
- Greedy speculative decoding using the `common_speculative` API
- Single sequence, no tree speculation
- Calls `common_speculative_init()`, `common_speculative_begin()`, `common_speculative_get_draft_params()`, `common_speculative_draft()`
- Uses `common_sampler_sample_and_accept_n()` for verification (`speculative-simple.cpp:249`)
- Checkpoint-based partial acceptance support (`speculative-simple.cpp:258-281`)
- Draft model expects `--model-draft` flag (`speculative-simple.cpp:27-82`)

### Server (`tools/server/server-context.cpp`)
Main production integration:
- Initializes `common_speculative` during slot setup (`server-context.cpp:1335`)
- Calls `common_speculative_process()` after every `llama_decode()` (`server-context.cpp:3656`) — this is what feeds target h_nextn into EAGLE3/MTP
- Calls `common_speculative_begin()` when prompt processing completes (`server-context.cpp:3745`)
- Calls `common_speculative_draft()` in `update_slots()` generation loop (`server-context.cpp:2978`)
- Calls `common_speculative_accept()` after verification (`server-context.cpp:3870`)
- Calls `common_speculative_get_state()`/`common_speculative_set_state()` during checkpoint save/restore (`server-context.cpp:2351, 3308`)
- Reports speculative metrics in `/v1/chat/completions` response (`server-context.cpp:519-523, 642`)

## Failure modes

1. **Incompatible vocabularies** — draft and target tokenizers must match closely (`speculative.cpp:56-121`); mismatch causes runtime error at init
2. **Context doesn't support partial sequence removal** (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`) — speculative decoding falls back to checkpoints, which is slower but functional
3. **Context doesn't support sequence removal at all** (`COMMON_CONTEXT_SEQ_RM_TYPE_NO`) — speculative decoding is disabled (`server-context.cpp:1319-1320`)
4. **Low acceptance rate** — ngram-mod automatically resets its hash pool when acceptance drops below 25% for 5 consecutive calls (`speculative.cpp:1607-1616`)
5. **MTP/EAGLE3 extraction not enabled** — if `need_embd_nextn()` returns true but embeddings are not extracted, `llama_get_embeddings_nextn` returns nullptr and the process aborts (`speculative.cpp:596, 631`)
6. **Batch overflow** — the batch must be large enough to hold all sampled + draft tokens (`server-context.cpp:477`)
7. **Sub-batch incompatible with speculative indices** — `post_decode()` throws if `spec_i_batch` indices fall outside the current sub-batch (`server-context.cpp:3697-3703`)

## Key design decisions

- **Priority chaining**: Multiple speculative types can be composed; the first to produce a draft wins (`speculative.cpp:2144-2190`)
- **EAGLE3 deferred boundary**: The last position of each `process()` batch is deferred to the next call because it needs the next token (`speculative.cpp:392-413`). Per-seq carry-over state (`pending_g_last`, `pending_pos_last`) bridges ubatches.
- **MTP head sharing**: Gemma4 uses `is_mem_shared` where MTP shares the target KV cache; step35 models use `chain_heads` with separate KV per head (`speculative.cpp:912-914`)
- **Decoupled from core library**: The entire speculative decoding system lives in `common/`, not `src/`. The core library exposes only the embedding extraction hooks needed by EAGLE3/MTP.
