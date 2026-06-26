# KV Cache Management

## Purpose

The KV cache avoids recomputing key (K) and value (V) activations for every
previously-seen token at each autoregressive decode step.  Without it, attention
over an N-token context would be O(N\^2) per step; with caching it becomes
O(N) per step (one forward pass for the new token, then attend to N cached
keys/values).

All claims below cite the implementing file and line.  Where no evidence exists
(e.g. speculative performance numbers) this is noted explicitly.

---

## Cell-based cache: `llama_kv_cell` / `llama_kv_cells`

### `llama_kv_cell_ext` (`src/llama-kv-cells.h:13`)

Extra per-cell metadata:

| Field | Type | Purpose |
|-------|------|---------|
| `x` | `llama_pos` | 2D spatial position (M-RoPE coordinate 1) |
| `y` | `llama_pos` | 2D spatial position (M-RoPE coordinate 2) |

`is_2d_gt(ox, oy)` returns `true` when `(y > oy) || (y == oy && x > ox)`,
used for causal masking of 2D positions (`src/llama-kv-cells.h:19`).

### `llama_kv_cells` (`src/llama-kv-cells.h:32`)

A container managing an array of cells (one per cache slot).  Each cell tracks:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `pos[i]` | `llama_pos` | `-1` | Token position; `-1` means empty |
| `ext[i]` | `llama_kv_cell_ext` | `{0,0}` | 2D spatial position for M-RoPE |
| `shift[i]` | `llama_pos` | `0` | Accumulated position shift (for context-shift) |
| `seq[i]` | `bitset<LLAMA_MAX_SEQ>` | all `false` | Which sequence(s) occupy this cell |

A single cell can be shared by **multiple sequences** (e.g. after a `seq_cp` that
stays within the same stream).  When the last sequence leaves a cell, the cell
is recycled (`src/llama-kv-cells.h:247-255`).

#### Lifecycle

1. **Empty**  `pos[i] == -1`
2. **Occupied**  `pos[i] >= 0`, one or more `seq[i]` bits set
3. **Recycled**  after `rm()` or `seq_rm()` when the last sequence departs;
   `pos` reset to `-1`, bitset cleared (`src/llama-kv-cells.h:222-234`)

Helper `used` set (`std::set<uint32_t>`, private `src/llama-kv-cells.h:462`)
tracks indices with `pos[i] != -1`; `used_min()` / `used_max_p1()` give
the occupied range for graph sizing.

#### Sequence-position index (`seq_pos`)

Private member `seq_pos[LLAMA_MAX_SEQ]` is a
`std::map<llama_pos, int>` per sequence (`src/llama-kv-cells.h:499`).
It stores a count of how many cells hold a given position for that sequence.
A multimap is needed because positions can repeat (cache reuse via rm+add,
vision models with repeating positions).  `seq_pos_min(s)` and
`seq_pos_max(s)` (`src/llama-kv-cells.h:334-360`) use `begin()->first` /
`rbegin()->first` to answer the min/max position.  The SWA system uses
these to determine which cells are masked.

#### Shift accounting

`shift[i]` accumulates all `pos_add()` and `pos_div()` deltas since the
last `reset_shift()` call.  The boolean `has_shift` (`src/llama-kv-cells.h:459`)
signals that at least one cell has a non-zero shift.  The `update()` method
of `llama_kv_cache` consumes this flag to trigger a K-shift graph
computation.

---

## `llama_kv_cache` — the cache engine

### Class hierarchy

```
llama_memory_i                    (interface, src/llama-memory.h:73)
  |
  +-- llama_kv_cache              (basic KV cache, src/llama-kv-cache.h:20)
  +-- llama_kv_cache_dsa          (DSA: two caches, src/llama-kv-cache-dsa.h:15)
  +-- llama_kv_cache_iswa         (ISWA: two caches, src/llama-kv-cache-iswa.h:14)
```

### Construction (`src/llama-kv-cache.cpp:80-377`)

Constructor parameters:

- `type_k`, `type_v` — quantization types for K and V tensors
- `v_trans` — if `true`, V cache is transposed (non-FA path)
- `offload` — whether to place KV tensors on GPU
- `unified` — if `true`, single stream; else one stream per sequence
- `kv_size` — number of cells (context length)
- `n_seq_max` — max concurrent sequences
- `n_swa`, `swa_type` — sliding-window attention config
- `mem_other` — pointer to a "source" cache for cell sharing
- `filter` — callback to exclude layers from caching
- `reuse` — callback to alias one layer's cache to another's
- `share` — callback to share tensors with a source cache

Each layer gets a K tensor `[n_embd_k_gqa, kv_size, n_stream]` and a V tensor
`[n_embd_v_gqa, kv_size, n_stream]` (or transposed layout for V when
`v_trans`).  Tensors are allocated on the appropriate backend buffer
(CPU or GPU per layer device).

Internal structure (`src/llama-kv-cache.h:217-285`):

| Member | Type | Purpose |
|--------|------|---------|
| `v_cells` | `llama_kv_cells_vec &` | Per-stream cell metadata |
| `v_heads` | `vector<uint32_t>` | Ring-buffer search heads per stream |
| `seq_to_stream` | `vector<uint32_t>` | Maps `seq_id` -> stream index |
| `sc_info` | `stream_copy_info` | Pending cross-stream buffer copies |
| `layers` | `vector<kv_layer>` | K/V tensor views per layer |
| `attn_rot_k/v` | `bool` | Whether to apply Hadamard rotation |
| `n_swa` / `swa_type` | — | SWA configuration |

### Streams

When `unified == false`, each sequence gets its own stream (one set of K/V
tensors per stream).  This enables parallel processing of multiple sequences
without interference.  `seq_to_stream` maps `seq_id` to `stream_id`
(`src/llama-kv-cache.cpp:162-169`).  Cross-stream `seq_cp` enqueues a
`stream_copy_info` that the next `update()` call will execute as a
`ggml_backend_tensor_copy` (`src/llama-kv-cache.cpp:517-519`).

### Sequence management

#### `seq_rm` (`src/llama-kv-cache.cpp:392-458`)

Removes all cells belonging to `seq_id` whose positions are in `[p0, p1)`.
When `seq_id == -1`, matches any sequence.  If a cell becomes empty after
removing the sequence, it is recycled.  Updates `v_heads` if a lower-index
slot became free.

#### `seq_cp` (`src/llama-kv-cache.cpp:460-550`)

Copy all cells of `seq_id_src` in `[p0, p1)` to `seq_id_dst`.

- **Same stream**: only metadata is updated (seq bitset); no data copy needed.
- **Cross-stream**: enqueues a `stream_copy_info`; actual buffer copy happens
  in `update()`.  Currently only supported for full buffer copies
  (`p0 <= 0 && p1 >= size`).

#### `seq_keep` (`src/llama-kv-cache.cpp:552-577`)

Keeps only cells belonging to `seq_id`; all other cells are recycled.

#### `seq_add` (`src/llama-kv-cache.cpp:579-627`)

Adds `shift` to all positions of `seq_id` in `[p0, p1)`.
Cells whose position becomes negative after shifting are recycled.
Sets `has_shift = true`, consumed later by `update()`.

#### `seq_div` (`src/llama-kv-cache.cpp:629-666`)

Integer-divide positions of `seq_id` in `[p0, p1)` by `d`.
Also sets `has_shift`.

#### `seq_pos_min` / `seq_pos_max` (`src/llama-kv-cache.cpp:668-692`)

Delegates to `v_cells[stream].seq_pos_min/max(seq_id)`.

### Public API wrappers (`include/llama.h:709-775`)

| API function | Delegates to |
|---|---|
| `llama_memory_clear` | `mem->clear(data)` |
| `llama_memory_seq_rm` | `mem->seq_rm(seq_id, p0, p1)` |
| `llama_memory_seq_cp` | `mem->seq_cp(src, dst, p0, p1)` |
| `llama_memory_seq_keep` | `mem->seq_keep(seq_id)` |
| `llama_memory_seq_add` | `mem->seq_add(seq_id, p0, p1, delta)` |
| `llama_memory_seq_div` | `mem->seq_div(seq_id, p0, p1, d)` |
| `llama_memory_seq_pos_min` | `mem->seq_pos_min(seq_id)` |
| `llama_memory_seq_pos_max` | `mem->seq_pos_max(seq_id)` |
| `llama_memory_can_shift` | `mem->get_can_shift()` |

All wrappers in `src/llama-context.cpp:3827-3922`.

---

## Slot finding (`find_slot`)

`find_slot` (`src/llama-kv-cache.cpp:907-1104`) searches for a slot big enough
to hold `ubatch.n_tokens`.  The algorithm is a ring-buffer scan starting from
`v_heads[stream]`.  A cell can be reused if:

1. It is empty (`cells.is_empty(idx)`) **or**
2. It is occupied by a single sequence and the cell's position is SWA-masked
   relative to that sequence's current max position (`src/llama-kv-cache.cpp:1051-1069`).

For non-contiguous slots, each token is tested individually; for contiguous
slots (`cont == true`), the entire run must be consecutive empty/reusable cells.

The "KV cache full" case returns an empty `slot_info`, which propagates to
`init_batch` returning `LLAMA_MEMORY_STATUS_FAILED_PREPARE`
(`src/llama-kv-cache.cpp:745`).

### `apply_ubatch` (`src/llama-kv-cache.cpp:1106-1182`)

Emplaces the ubatch tokens into the cells found by `find_slot`.  Steps:

1. For each token, if the target cell is occupied, records the max position
   being overwritten (`seq_pos_max_rm`) and calls `cells.rm(idx)`.
2. Sets cell position (and 2D ext if applicable).
3. Adds all sequence IDs for this token to the cell.
4. **Post-condition invariant**: all positions `[pos_min, pos_max]` for each
   sequence must be present.  So any positions <= `seq_pos_max_rm` are purged
   via `seq_rm` (`src/llama-kv-cache.cpp:1156-1174`).

### `prepare` (`src/llama-kv-cache.cpp:760-824`)

Calls `find_slot` + `apply_ubatch` for each ubatch in sequence, snapshotting
cell state before each placement.  If any placement fails, the state is rolled
back (restored from snapshots in reverse order).  This ensures the cache is
never left in a half-placed state.

---

## DSA (Dynamic Stale Attention)

**Used by**: `LLM_ARCH_DEEPSEEK32` (`src/llama-model.cpp:2026-2041`).

DSA manages **two** `llama_kv_cache` instances internally:

| Internal cache | Purpose |
|---|---|
| `kv_mla` (`llama_kv_cache`) | Full MLA (Multi-head Latent Attention) key/value cache |
| `kv_lid` (`llama_kv_cache`) | Lightning Indexer key cache (single-head MQA, `n_head_kv = 1`) |

The indexer cache uses a separate `llama_hparams` instance with `n_head_kv`
overridden to `1` and `n_embd_head_k_full` set to `indexer_head_size`
(`src/llama-kv-cache-dsa.cpp:42-44`).  This lets it reuse all cache
infrastructure with different tensor dimensions.

All sequence ops (`seq_rm`, `seq_cp`, etc.) are forwarded to both internal
caches (`src/llama-kv-cache-dsa.cpp:60-87`).  `can_shift` requires both
caches to support shifting **and** have equal sizes
(`src/llama-kv-cache-dsa.cpp:157-161`).

The context (`llama_kv_cache_dsa_context`) wraps two `llama_kv_cache_context`
instances (`ctx_mla`, `ctx_lid`) and iterates them in lockstep
(`src/llama-kv-cache-dsa.cpp:217-228`).

---

## ISWA (Internal Sliding Window Attention)

**Used by**: Any model with `swa_type != NONE` (e.g. Cohere2, Gemma2, Gemma4,
Mellum, Qwen3.5 hybrid variant).

ISWA also manages **two** `llama_kv_cache` instances:

| Internal cache | Purpose |
|---|---|
| `kv_base` (`llama_kv_cache`) | Non-SWA layers, full context |
| `kv_swa` (`llama_kv_cache`) | SWA layers, windowed context |

Layer filtering is chained:

- `filter_base`: passes layers where `!hparams.is_swa(il)` AND passes the
  caller's outer filter (`src/llama-kv-cache-iswa.cpp:32-38`).
- `filter_swa`: passes layers where `hparams.is_swa(il)` AND passes the
  caller's outer filter (`src/llama-kv-cache-iswa.cpp:40-46`).

The SWA cache size is: `pad(min(kv_size, n_swa * seq_count + n_ubatch), 256)`
(`src/llama-kv-cache-iswa.cpp:52`).  Can be overridden to full size via
`swa_full` parameter.  The `256` padding is a performance optimization per
issue #17037.

All sequence ops are forwarded to both caches.  `seq_pos_min/max` delegates
only to `kv_swa` (SWA cache is a subset of base, so its range is tighter;
`src/llama-kv-cache-iswa.cpp:122-128`).

For state save/restore, the `PARTIAL_ONLY` flag skips `kv_base`
(`src/llama-kv-cache-iswa.cpp:238-252`).

---

## Cache clearing and shifting

### `clear(bool data)` (`src/llama-kv-cache.cpp:379-390`)

Resets all cell metadata and optionally zeroes the backend buffer data.
DSA/ISWA forward to both internal caches.

### `update` (`src/llama-kv-cache.cpp:826-905`)

Called by `llama_kv_cache_context::apply()` when `ubatches` is empty
(`src/llama-kv-cache.cpp:2539-2543`).  Two phases:

1. **Stream copy**: if `sc_info` is non-empty, issues
   `ggml_backend_tensor_copy` for each K/V layer from source stream to
   destination stream (`src/llama-kv-cache.cpp:843-863`).

2. **K-shift**: if `do_shift` is true (i.e. `get_has_shift()`), builds and
   executes a compute graph that applies RoPE rotation to each cache K tensor.
   The graph is built by `build_graph_shift` (`src/llama-kv-cache.cpp:1905-1951`).
   After execution, `cells.reset_shift()` clears the per-cell shift accumulators.

### K-shift graph (`build_graph_shift`)

For each cached layer:
1. View the K tensor as `[n_rot, n_head_kv, kv_size * n_stream]` with
   offset `n_embd_nope` (non-RoPE dimensions are skipped).
2. Call `build_rope_shift` (`src/llama-kv-cache.cpp:1826-1876`) which:
   - If quantized: dequantize -> rotate back (Hadamard) -> apply RoPE ->
     rotate forward -> quantize back
   - If not quantized: apply RoPE in-place on the first `n_rot` dimensions

The shift tensor `k_shift` (`I32[kv_size*n_stream]`) carries the per-cell
shift values.  The rotation matrix `k_rot` (Hadamard) is only used when
K-cache quantization is active (`attn_rot_k`).

---

## Graph integration (attention computation)

### Input classes

| Class | File:line | Used for |
|---|---|---|
| `llm_graph_input_attn_kv` | `src/llama-graph.h:305` | Standard KV-cache attention |
| `llm_graph_input_attn_k` | `src/llama-graph.h:347` | V-less KV-cache (MLA, #19067) |
| `llm_graph_input_attn_k_dsa` | `src/llama-graph.h:378` | DSA (MLA + indexer) |
| `llm_graph_input_attn_kv_iswa` | `src/llama-graph.h:416` | ISWA (base + SWA) |

### Cache lookup during attention

During `set_input()` for each input class:

| Step | Sets what | Calls |
|---|---|---|
| 1 | K indices | `mctx->set_input_k_idxs(k_idxs, ubatch)` |
| 2 | V indices | `mctx->set_input_v_idxs(v_idxs, ubatch)` |
| 3 | KQ mask | `mctx->set_input_kq_mask(kq_mask, ubatch, causal)` |
| 4 | K rotation (Hadamard) | `mctx->set_input_k_rot(k_rot)` |
| 5 | V rotation (Hadamard) | `mctx->set_input_v_rot(v_rot)` |

Model-specific graph builders then:

- `mctx->get_k(ctx, il)` creates a 4D view of the cache's K tensor
- `mctx->get_v(ctx, il)` creates a 4D view of the cache's V tensor
- `mctx->cpy_k(ctx, k_cur, k_idxs, il)` stores computed K into the cache via
  `ggml_set_rows`
- `mctx->cpy_v(ctx, v_cur, v_idxs, il)` stores computed V into the cache

### Cache update after attention (`apply`)

The `apply()` method of `llama_kv_cache_context` (`src/llama-kv-cache.cpp:2536-2550`):
- If `ubatches` is empty: calls `kv->update(lctx, do_shift, sc_info)` (handles
  K-shift and stream copies).
- If `ubatches` is present: calls `kv->apply_ubatch(sinfo, ubatch)` to
  emplace tokens into the cells and updates `n_kv`.

---

## K-cache quantization

Type `type_k` / `type_v` is passed at construction (`src/llama-kv-cache.cpp:82-83`).
K-cache quantization uses Walsh-Hadamard rotation to reduce quantization error:

- `attn_rot_k` is enabled when ALL of:
  - Not disabled via env `LLAMA_ATTN_ROT_DISABLE`
  - `n_embd_head_k_all > 0` (all layers share same head size)
  - `type_k` is a quantized type
  - `n_embd_head_k % 64 == 0`
  (`src/llama-kv-cache.cpp:332-337`)

- For DeepSeek V3.2 DSA, `attn_rot_k` is also enabled when
  `n_embd_head_k_full == indexer_head_size` (`src/llama-kv-cache.cpp:339-342`).

- `attn_rot_v` uses same logic but always uses 64x64 rotation matrices
  (smaller is better per #21038, `src/llama-kv-cache.cpp:1437-1439` comment).

The Hadamard matrix is generated by `ggml_gen_hadamard`
(`src/llama-kv-cache.cpp:22-58`) using a recursive Sylvester construction
(standard Walsh-Hadamard).  It's stored in `attn_rot_hadamard` map keyed by
matrix dimension (`src/llama-kv-cache.cpp:357-372`).

During `build_rope_shift`, quantized K tensors are:
1. Dequantized to F32
2. Rotated backward (multiply by Hadamard\^-1, which == Hadamard since H\^2 = I)
3. RoPE shift applied
4. Rotated forward
5. Re-quantized back to original type

---

## Touch points

### Inference engine

The KV cache is created in `llama_model::create_memory` (`src/llama-model.cpp`)
and stored as `llama_memory_i` pointer (accessed via `llama_get_memory`).
During `llama_decode`, the flow is:

1. `memory->init_batch(balloc, n_ubatch, embd_all)` -> splits batch into
   ubatches and prepares slot info
2. Iterate ubatches: for each layer, build attention graph using `mctx->get_k()`,
   `mctx->cpy_k()`, etc.
3. `mctx->apply()` -> writes K/V to cache cells
4. `mctx->next()` -> advance to next ubatch
5. After all tokens: `memory->init_update(lctx, optimize)` -> applies K-shift
   and stream copies if pending

### Sampling

Sampling does **not** directly interact with the KV cache.  It reads the
output logits (last token's hidden state).  However, sampling decisions
(which sequences to keep/remove) feed back into `seq_rm`, `seq_keep`, and
`seq_cp` before the next decode step.

---

## Failure modes

| Condition | Symptom | Source |
|---|---|---|
| `KV cache full` | `init_batch` returns `LLAMA_MEMORY_STATUS_FAILED_PREPARE` | `src/llama-kv-cache.cpp:745` |
| `K-shift unsupported` | `GGML_ABORT` in `update()` | `src/llama-kv-cache.cpp:868` |
| `Step35 architecture` | `get_can_shift()` returns `false` | `src/llama-kv-cache.cpp:1185-1187` |
| `n_pos_per_embd() > 1` | `seq_add/div` unsupported assertion | `src/llama-kv-cache.cpp:586,636` |
| `Cross-stream seq_cp partial` | `GGML_ASSERT(is_full)` | `src/llama-kv-cache.cpp:515` |
| `State restore mismatch` | throws `std::runtime_error`, clears cache | `src/llama-kv-cache.cpp:2035-2062` |
| `Memory allocation failure` | throws `std::runtime_error` at init | `src/llama-kv-cache.cpp:239,300` |

---

## Key invariants

1. All positions in `[pos_min, pos_max]` for each sequence are guaranteed to
   be present in the cache (`src/llama-kv-cache.cpp:1156-1174`, ref #13746).
2. A cell can be occupied by multiple sequences only when they share the same
   stream (same K/V data).  Cross-stream copies are always full-buffer.
3. The `has_shift` flag is cleared only by `update()` via `reset_shift()`;
   `apply_ubatch` does not clear it.
4. DSA and ISWA maintain the invariant that both internal caches have equal
   `kv_size` before allowing shift (`src/llama-kv-cache-dsa.cpp:160`,
   `src/llama-kv-cache-iswa.cpp:235`).
