# Memory Management (Recurrent / State-Space Models)

## Purpose

Traditional transformer-based LLMs use a KV cache that stores key-value pairs for each token position, growing linearly with sequence length. Recurrent and state-space models (Mamba, RWKV, Griffin, etc.) do not attend over a history of positions; instead, they maintain a fixed-size per-sequence state (convolutional state + SSM state) that is updated on each forward pass. This state is compact (typically `n_embd_r` + `n_embd_s` per layer, per sequence) and is overwritten in place.

The memory management subsystem provides a unified interface (`llama_memory_i`, `src/llama-memory.h:73`) that abstracts over both the traditional KV cache and recurrent/SSM state, allowing the inference engine to handle all model architectures with the same driver loop.

---

## The `llama_memory_i` Interface

Defined in `src/llama-memory.h:73`. It is the base abstraction for all memory types (KV cache, recurrent state, hybrid, ISWA, DSA).

### Virtual Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `init_batch` | `(balloc, n_ubatch, embd_all) -> context_ptr` | Split batch into ubatches, verify they fit in cache |
| `init_full` | `() -> context_ptr` | Simulate full cache for worst-case compute buffer allocation |
| `init_update` | `(lctx, optimize) -> context_ptr` | Prepare pending memory updates (shifts, copies) |
| `get_can_shift` | `() -> bool` | Whether cache supports position shifting |
| `clear` | `(data)` | Clear metadata (and optionally data buffers) |
| `seq_rm` | `(seq_id, p0, p1) -> bool` | Remove sequence state within position range |
| `seq_cp` | `(src, dst, p0, p1)` | Copy sequence state range |
| `seq_keep` | `(seq_id)` | Remove all sequences except the one specified |
| `seq_add` | `(seq_id, p0, p1, shift)` | Shift positions in range (cache shifting) |
| `seq_div` | `(seq_id, p0, p1, d)` | Divide positions in range |
| `seq_pos_min/max` | `(seq_id) -> pos` | Return min/max position for a sequence |
| `memory_breakdown` | `() -> map<buft, size>` | Per-buffer-type memory sizes |
| `state_write/read` | `(io, seq_id, flags)` | Serialize/deserialize cache state |

### The `llama_memory_context_i` Sub-Interface

Defined in `src/llama-memory.h:51`. A transient object returned by `init_batch()` / `init_full()` / `init_update()` that holds the ubatch split state and provides:

- `next()` — advance to the next ubatch; returns `false` when done
- `apply()` — commit the current ubatch's memory state to the memory object
- `get_ubatch()` — return the current ubatch
- `get_status()` — return `LLAMA_MEMORY_STATUS_SUCCESS`, `NO_UPDATE`, `FAILED_PREPARE`, or `FAILED_COMPUTE`

### Lifecycle

1. **Creation** — `llama_model::create_memory()` (`src/llama-model.cpp:2003`) is called during `llama_context` construction (`src/llama-context.cpp:333`). It selects the implementation based on architecture.

2. **Update phase** — Before each decode, `llama_context::memory_update()` (`src/llama-context.cpp:762`) calls `memory->init_update(this, optimize)` to handle pending cache shifts/copies, then calls `memory->init_full()` to reserve worst-case compute buffers.

3. **Batch decode** — The inference loop calls `memory->init_batch(balloc, n_ubatch, output_all)` (`src/llama-context.cpp:1771`) to split the input batch into ubatches. It then iterates: `get_ubatch()` -> `process_ubatch()` -> `mctx->apply()` -> `mctx->next()` (`src/llama-context.cpp:1842-1871`).

---

## Hybrid Memory (Cell + Recurrent)

Defined in `src/llama-memory-hybrid.h:19`. Used for architectures like Jamba, Falcon H1, Plamo2, Granite Hybrid, LFM2, Nemotron-H, Qwen3Next, Kimi Linear, Qwen3.5 (`src/llama-arch.cpp:878`).

Composes two sub-memories:
- `mem_attn` — a `llama_kv_cache` (or `llama_kv_cache_iswa`) for attention layers
- `mem_recr` — a `llama_memory_recurrent` for recurrent layers

Layer filtering is done via `filter_attn` / `filter_recr` callbacks passed at construction (`src/llama-memory-hybrid.cpp:34-65`). Default filters use `hparams.is_recr(il)` to classify layers.

All `llama_memory_i` operations (seq_rm, seq_cp, seq_keep, seq_add, seq_div, clear, state_write/read) are delegated to both sub-memories. The `init_batch` path (`src/llama-memory-hybrid.cpp:67-123`) first calls `mem_recr->prepare(ubatches)`, then `mem_attn->prepare(ubatches)`, and creates a `llama_memory_hybrid_context` that holds both sub-contexts.

The `llama_memory_hybrid_context` (`src/llama-memory-hybrid.h:93`) wraps `ctx_attn` and `ctx_recr`. In `next()` and `apply()`, both sub-contexts are stepped together. Status is combined via `llama_memory_status_combine()` (`src/llama-memory.cpp:3`).

`seq_pos_min` and `seq_pos_max` are computed as the outer bounds across both caches (`src/llama-memory-hybrid.cpp:173-179`).

---

## ISWA Hybrid

Defined in `src/llama-memory-hybrid-iswa.h:19`. Used for hybrid architectures that also have sliding-window attention (SWA), selected when `hparams.swa_type != LLAMA_SWA_TYPE_NONE` (`src/llama-model.cpp:2087`).

Identical in structure to `llama_memory_hybrid`, except `mem_attn` is a `llama_kv_cache_iswa` (`src/llama-kv-cache-iswa.h:14`) which internally maintains two sub-caches:
- `kv_base` — full attention KV cache
- `kv_swa` — sliding-window attention KV cache

The `llama_memory_hybrid_iswa_context` constructor (`src/llama-memory-hybrid-iswa.cpp:234-244`) creates:
- `ctx_attn` — a `llama_kv_cache_iswa_context` with separate `sinfos_base` and `sinfos_swa` slot info vectors
- `ctx_recr` — a `llama_memory_recurrent_context`

### Cache Shifting

The `seq_add` operation shifts positions in both sub-caches:
- For the recurrent side (`src/llama-memory-recurrent.cpp:304-332`): since recurrent state does not depend on position history, only the `pos` field of the tail cell is updated. The operation is trivially O(1) per sequence.
- For the ISWA attention side (`src/llama-kv-cache-iswa.cpp:111-113`): delegates to both `kv_base->seq_add()` and `kv_swa->seq_add()`.

`get_can_shift()` returns `mem_attn->get_can_shift()` (`src/llama-memory-hybrid-iswa.cpp:139-140`) — the recurrent side always supports trivial shifting (`src/llama-memory-recurrent.cpp:698-701`).

---

## Recurrent-Only Memory Backends

Defined in `src/llama-memory-recurrent.h:17`, implemented in `src/llama-memory-recurrent.cpp`.

Used for architectures: Mamba, Mamba2, RWKV6, RWKV7, aRWKV7 (`src/llama-arch.cpp:864`).

### Storage

Per layer, two tensors are allocated:
- `r_l[il]` — recurrent R-state (convolutional state for Mamba, token-shift for RWKV), shape `[n_embd_r, mem_size * (1 + n_rs_seq)]`
- `s_l[il]` — recurrent S-state (SSM state for Mamba, WKV state for RWKV), shape `[n_embd_s, mem_size * (1 + n_rs_seq)]`

The extra `(1 + n_rs_seq)` dimension enables per-sequence rollback snapshots (`n_rs_seq` snapshots + 1 current state), bounded by `hparams.n_rs_seq` (`src/llama-memory-recurrent.cpp:99`).

### Cell Management

Each sequence maps to exactly one "tail" cell via `cells[seq_id].tail` (`src/llama-memory-recurrent.h:88-107`). Cells track:
- `pos` — last position
- `src` — source cell index (for in-place state reuse)
- `src0` — initial source before reordering
- `tail` — -1 or index of the cell holding this seq's state
- `seq_id` — set of sequences sharing this cell

`find_slot()` (`src/llama-memory-recurrent.cpp:487-696`) is the slot allocation algorithm. It:
1. Verifies all seq_ids are within bounds
2. Finds empty cells for sequences without existing state
3. Reorders cells into a contiguous range `[min, max]`
4. Finds a "zero state" cell (`rs_z`) that has no source references
5. Sets `src0` / `src` on each cell for graph-level state copying

### Graph Integration

`build_rs()` (`src/llama-graph.cpp:2758`) is the key graph-building helper:
1. Clears the zero-state cell to reset stale state: `ggml_scale_inplace(state_zero, 0)`
2. Copies states from source rows to contiguous output rows via `ggml_get_rows(states, state_copy_main)` — this produces the `n_seqs` output states for the current ubatch
3. Copies extra states (between `n_seqs` and `n_rs`) to their destination positions

The `llm_graph_input_rs` (`src/llama-graph.h:246`) creates a single `s_copy` tensor `I32 [n_rs]` that is split into:
- `s_copy_main` — first `n_seqs` rows (states to output)
- `s_copy_extra` — remaining rows (states to copy in place)

Each recurrent model (Mamba: `src/models/mamba-base.cpp:38,124`, RWKV: `src/models/rwkv6.cpp:104`, etc.) calls `build_rs_inp()` once per graph, then calls `build_rs()` for each of its state tensors.

`apply()` (`src/llama-memory-recurrent.cpp:1194-1208`) simply calls `mem->find_slot(ubatches[i_next])` — it re-executes the slot search to commit the ubatch's state.

### State Rollback

The `n_rs_seq` parameter controls per-sequence rollback depth. When set, tensors are widened to `(1 + n_rs_seq)` groups. Rollback indices are stored in `rs_idx[seq_id]` and reset on `seq_rm`. This is used for speculative decoding or rejection sampling where a sequence's state must be restored to a previous point.

---

## How Memory Integrates with the Inference Engine

The inference engine in `llama_context::decode()` (`src/llama-context.cpp:1731`) follows:

1. **Pre-decode update**: `memory_update(false)` at line 1766 — calls `init_update(this, false)`. For recurrent memory this is a no-op returning `NO_UPDATE` (`src/llama-memory-recurrent.cpp:456-461`). For hybrid/attention caches, this handles pending rope-shift and stream copy operations.

2. **init_batch attempt**: line 1771 — calls `memory->init_batch(*balloc, cparams.n_ubatch, output_all)`. On `FAILED_PREPARE`, may retry with `memory_update(true)` to optimize cache usage.

3. **Ubatch loop**: lines 1822-1871 — for each ubatch:
   - `process_ubatch()` builds and executes the compute graph
   - The graph uses `mctx.get()` (a `llama_memory_context_i*`) passed to `llm_graph_params` (`src/llama-graph.h:603`)
   - Graph builders cast `mctx` to the appropriate context type (e.g., `llama_memory_recurrent_context`) for state access
   - `mctx->apply()` commits state changes

4. **Graph result reuse**: `llm_graph_result::can_reuse()` (`src/llama-graph.h:631`) checks ubatch shape, sequences, outputs, and memory context — if compatible, the same graph is reused with updated inputs.

---

## Touch Points

### Model Architecture

- `src/llama-arch.cpp:864-896` — `llm_arch_is_recurrent()` and `llm_arch_is_hybrid()` classify architectures
- `src/llama-model.cpp:2003-2200` — `create_memory()` factory selects the memory implementation per architecture
- `src/llama-model.cpp:1096` — `is_recr_impl` initialized to 1 for recurrent archs
- `src/models/mamba-base.cpp` — Mamba recurrent layer graph (uses `build_rs`)
- `src/models/rwkv6.cpp` — RWKVv6 graph (uses `build_rs`)
- `src/models/qwen35.cpp` — Qwen3.5 hybrid graph (uses both `build_attn_inp_kv` and `build_rs`)

### Inference Engine

- `src/llama-context.cpp:324-333` — `memory.reset(model.create_memory(...))`
- `src/llama-context.cpp:762-815` — `memory_update()` calls `init_update` + `init_full`
- `src/llama-context.cpp:1771` — `init_batch()` call in decode loop
- `src/llama-context.cpp:1843` — `process_ubatch()` with memory context `mctx.get()`
- `src/llama-graph.h:603` — `llm_graph_params::mctx` carries the memory context into graph building
- `src/llama-graph.cpp:2758-2835` — `build_rs()` graph construction for recurrent state

### Memory Files

- `src/llama-memory.h` — Interface definitions
- `src/llama-memory.cpp` — Helper functions (`combine`, `is_fail`)
- `src/llama-memory-recurrent.h/.cpp` — Recurrent-only memory
- `src/llama-memory-hybrid.h/.cpp` — Hybrid (attention + recurrent) memory
- `src/llama-memory-hybrid-iswa.h/.cpp` — Hybrid with ISWA attention
- `src/llama-kv-cache.h/.cpp` — Standard KV cache (attention-only)
- `src/llama-kv-cache-iswa.h/.cpp` — ISWA KV cache (base + SWA)
- `src/llama-kv-cache-dsa.h/.cpp` — DeepSeek dual-stream attention cache

---

## Failure Modes

| Failure | Location | Cause |
|---------|----------|-------|
| `FAILED_PREPARE` from `init_batch` | `src/llama-memory-recurrent.cpp:449` | Seq_id exceeds `n_seq_max`, or cells exhausted, or ubatch incompatible |
| `FAILED_PREPARE` from `init_batch` (hybrid) | `src/llama-memory-hybrid.cpp:121-122` | Either recurrent or attention prepare fails |
| Non-consecutive positions | `src/llama-memory-recurrent.cpp:644` | Sequence position backtracks or skips; warns but continues |
| `FAILED_COMPUTE` | `src/llama-memory.cpp:52-55` | Only returned by attention cache shift/update; recurrent memory never returns this |
| State write with mixed rollback indices | `src/llama-memory-recurrent.cpp:766` | Cannot write shared state when seqs have different rs_idx |
| State read with mismatched layer count/type | `src/llama-memory-recurrent.cpp:1050-1084` | Model architecture changed between save and load |
| MTP hybrid fallback | `src/llama-model.cpp:2049-2052` | MTP context on Qwen3.5 hybrid uses plain KV cache instead of hybrid memory |
