# GGML Tensor Library

## Purpose

GGML is a C/C++ tensor library that implements three core facilities described at the top of its public header:

> "a set of tensor operations / automatic differentiation / basic optimization algorithms" (`ggml/include/ggml.h:15-17`)

It provides a **minimalistic approach for various machine learning tasks** including "linear regression, support vector machines, neural networks" (`ggml/include/ggml.h:18-24`). The user defines a function using available tensor operations; this definition is stored internally as a **computation graph** where each tensor operation becomes a node. The graph can then be evaluated (forward pass) and/or differentiated (backward pass) to compute gradients with respect to input variables. Optionally, optimization algorithms can be applied (`ggml/include/ggml.h:26-30`).

The library is written in plain C11 with a thin C++ wrapper (`ggml/src/ggml.cpp:1-26`) that installs a `std::terminate` handler to print backtraces on uncaught exceptions (`ggml/src/ggml.cpp:21-24`). GGML is the **foundational layer** for the entire llama.cpp project; every model evaluation, KV-cache operation, and training step passes through it.

## Key Types and Protocols

### `ggml_type` — Data type descriptor

An enum spanning 42 entries from `GGML_TYPE_F32` (0) through `GGML_TYPE_COUNT` (42). It covers FP32, FP16, BF16, FP64, 8/16/32/64-bit integers, and approximately 30 quantized formats (Q4_0, Q5_K, IQ2_XXS, etc.) (`ggml/include/ggml.h:389-433`). Each type has associated traits via `ggml_type_traits` which carries `type_name`, `blck_size`, `type_size`, `is_quantized`, and conversion function pointers (`ggml/include/ggml.h:2817-2825`). Quantized types are implemented in `ggml/src/ggml-quants.c` (5591 lines of SIMD-heavy quantize/dequantize routines).

### `ggml_context` — Memory pool and object arena

```c
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    int    n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

(`ggml/src/ggml.c:956-966`)

All tensors, graphs, and internal bookkeeping objects are allocated from a flat memory buffer provided at init time (`ggml/include/ggml.h:661`). If the user passes `mem_buffer = NULL`, GGML calls `ggml_aligned_malloc` internally and owns the buffer (`ggml/src/ggml.c:1596`). Objects are tracked via an intrusive linked list of `ggml_object` headers (`ggml/src/ggml.c:939-948`) with three types: `GGML_OBJECT_TYPE_TENSOR`, `GGML_OBJECT_TYPE_GRAPH`, and `GGML_OBJECT_TYPE_WORK_BUFFER` (`ggml/include/ggml.h:628-632`). The `no_alloc` flag delays tensor data allocation (`ggml/include/ggml.h:663`), enabling zero-copy loading from model files.

Initialization is done via `ggml_init()`, which records a one-shot global time-system init on first call (`ggml/src/ggml.c:1571-1582`). The context can be reset (`ggml_reset`, `ggml/src/ggml.c:1613`) or freed (`ggml_free`, `ggml/src/ggml.c:1623`).

### `ggml_tensor` — The fundamental data object

```c
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t ne[GGML_MAX_DIMS];   // number of elements per dimension
    size_t  nb[GGML_MAX_DIMS];   // stride in bytes per dimension
    enum ggml_op op;             // operation that produced this tensor
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t flags;
    struct ggml_tensor * src[GGML_MAX_SRC];  // source tensors (for graph edges)
    struct ggml_tensor * view_src;           // source tensor for views
    size_t               view_offs;
    void * data;
    char name[GGML_MAX_NAME];
    void * extra;  // backend-specific (e.g. CUDA pointer)
};
```

(`ggml/include/ggml.h:667-699`)

Key invariants:
- Up to 4 dimensions (`GGML_MAX_DIMS = 4`, `ggml/include/ggml.h:222`).
- Row-major storage with explicit strides (`nb`), enabling non-contiguous views for transpose/permute (`ggml/include/ggml.h:125-129`).
- `src[0..GGML_MAX_SRC-1]` point to predecessor tensors, forming the computation graph edges (`ggml/include/ggml.h:686`).
- `view_src`/`view_offs` implement zero-copy views (`ggml/include/ggml.h:689-690`).
- Flags: `GGML_TENSOR_FLAG_INPUT`, `OUTPUT`, `PARAM`, `LOSS`, `COMPUTE` (`ggml/include/ggml.h:644-650`).
- `op` is set to the operation enum (e.g. `GGML_OP_MUL_MAT`) for intermediate tensors; leaf tensors have `GGML_OP_NONE` (`ggml/include/ggml.h:679`).

### `ggml_cgraph` — Computation graph

```c
struct ggml_cgraph {
    int size;       // capacity
    int n_nodes;
    int n_leafs;
    struct ggml_tensor ** nodes;
    struct ggml_tensor ** grads;
    struct ggml_tensor ** grad_accs;
    struct ggml_tensor ** leafs;
    int32_t * use_counts;
    struct ggml_hash_set visited_hash_set;
    enum ggml_cgraph_eval_order order;
    uint64_t uid;
};
```

(`ggml/src/ggml-impl.h:329-347`)

A directed acyclic graph where `nodes[]` holds the topological ordering of tensor operations and `leafs[]` holds constant/input tensors. The `ggml_hash_set` (linear-probing hash table, `ggml/src/ggml-impl.h:226-230`) provides O(1) deduplication during graph construction. An `uid` field optionally identifies equivalent graphs (`ggml/src/ggml-impl.h:346`). Graphs can be sliced with `ggml_graph_view()` to get a read-only subgraph (`ggml/src/ggml-impl.h:352`).

### `ggml_init_params` — Context configuration

```c
struct ggml_init_params {
    size_t mem_size;    // bytes
    void * mem_buffer;  // NULL => internal allocation
    bool   no_alloc;
};
```

(`ggml/include/ggml.h:659-664`)

### `ggml_cplan` — Compute plan

```c
struct ggml_cplan {
    size_t    work_size;
    uint8_t * work_data;
    int n_threads;
    struct ggml_threadpool * threadpool;
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;
    bool use_ref;
};
```

(`ggml/include/ggml-cpu.h:12-25`)

Returned by `ggml_graph_plan()` (`ggml/include/ggml-cpu.h:66-69`) and passed to `ggml_graph_compute()` (`ggml/include/ggml-cpu.h:70`).

## Data Flow

The canonical lifecycle, as shown in the header's worked example (`ggml/include/ggml.h:32-74`):

```
ggml_init(params)                  # 1. Create context
       |
       v
ggml_new_tensor_1d(ctx, type, ne)  # 2. Create data tensors
ggml_set_param(ctx, x)             # 3. Mark trainable params
       |
       v
ggml_mul(ctx, x, x)                # 4. Build expression tree
ggml_add(ctx, ggml_mul(...), ...)  #    (each call adds a graph node)
       |
       v
ggml_new_graph(ctx)                # 5. Create graph container
ggml_build_forward_expand(gf, f)   # 6. Populate graph from expression
       |
       v
ggml_set_f32(x, 2.0f)             # 7. Set input values
ggml_set_f32(a, 3.0f)
ggml_set_f32(b, 4.0f)
       |
       v
ggml_graph_compute_with_ctx(...)   # 8. Execute forward pass
       |
       v
ggml_get_f32_1d(f, 0)             # 9. Read results
```

Implementation details for each step:

1. **Context creation**: `ggml_init()` allocates the context struct and the memory pool (`ggml/src/ggml.c:1571-1611`). If `mem_size == 0`, it defaults to `GGML_MEM_ALIGN` (16 or 8 bytes, `ggml/include/ggml.h:235-244`).

2. **Tensor creation**: `ggml_new_tensor_impl()` allocates a `ggml_object` header + `ggml_tensor` struct (and optionally data) from the context's pool (`ggml/src/ggml.c:1726-1802`). Strides are computed as `nb[0] = type_size`, `nb[i] = nb[i-1] * ne[i-1]` (`ggml/src/ggml.c:1793-1797`). The helper wrappers `ggml_new_tensor_1d/2d/3d/4d` delegate here (`ggml/src/ggml.c:1812-1829`).

3. **Parameter marking**: `ggml_set_param()` asserts `op == GGML_OP_NONE` and sets `GGML_TENSOR_FLAG_PARAM` (`ggml/src/ggml.c:7653-7656`). Similarly `ggml_set_input`, `ggml_set_output`, `ggml_set_loss` (`ggml/src/ggml.c:7645-7662`).

4. **Graph construction**: Each `ggml_op` function (e.g., `ggml_mul` at `ggml/include/ggml.h:955`) calls `ggml_new_tensor_impl()` to create a result tensor with `op`, `src[0]`, `src[1]` set, then returns it. No computation happens yet (`ggml/include/ggml.h:55-56`).

5. **Graph assembly**: `ggml_new_graph()` allocates a `ggml_cgraph` from the context (`ggml/src/ggml.c:7178-7180`). `ggml_build_forward_expand()` recursively walks the tensor's `src[]` chain and inserts each tensor into the graph's `nodes[]` and `leafs[]` arrays via a hash set for deduplication (`ggml/src/ggml.c:6997-6998`).

6. **Graph computation**: `ggml_graph_compute()` (`ggml/src/ggml-cpu/ggml-cpu.c:3308-3381`) iterates `nodes[]` in topological order, dispatching each to a threadpool. It validates the compute plan (`GGML_ASSERT(cplan)`, `GGML_ASSERT(cplan->n_threads > 0)` at `ggml/src/ggml-cpu/ggml-cpu.c:3311-3313`). `ggml_graph_compute_with_ctx()` is a convenience wrapper that calls `ggml_graph_plan()` then `ggml_graph_compute()` (`ggml/include/ggml-cpu.h:72-74`).

7. **Backward pass**: `ggml_build_backward_expand()` (`ggml/src/ggml.c:7001-7069`) builds gradient nodes for all parameter/loss tensors, using the same node-creation primitives.

## Entry Points

| Operation | Declaration | Implementation |
|---|---|---|
| Context init/free | `ggml/include/ggml.h:801-803` | `ggml/src/ggml.c:1571,1623` |
| Tensor creation | `ggml/include/ggml.h:814-848` | `ggml/src/ggml.c:1726-1829` |
| Data access | `ggml/include/ggml.h:862-863` | `ggml/src/ggml.c` |
| Flags (param/input/output) | `ggml/include/ggml.h:871-874` | `ggml/src/ggml.c:7645-7662` |
| Arithmetic ops (add, mul, etc.) | `ggml/include/ggml.h:889-969` | `ggml/src/ggml.c` |
| Matrix multiply | `ggml/include/ggml.h:1418-1421` | `ggml/src/ggml.c:3243` |
| Activation functions | `ggml/include/ggml.h:1159-1198` | `ggml/src/ggml.c` |
| Normalization (rms_norm, etc.) | `ggml/include/ggml.h:1361-1405` | `ggml/src/ggml.c` |
| Graph creation | `ggml/include/ggml.h:2730-2731` | `ggml/src/ggml.c:7133,7178` |
| Forward build | `ggml/include/ggml.h:2720-2722` | `ggml/src/ggml.c:6997` |
| Backward build | `ggml/include/ggml.h:2724-2727` | `ggml/src/ggml.c:7001` |
| Graph plan | `ggml/include/ggml-cpu.h:66-69` | `ggml/src/ggml-cpu/ggml-cpu.c` |
| Graph compute | `ggml/include/ggml-cpu.h:70` | `ggml/src/ggml-cpu/ggml-cpu.c:3308` |
| Quantization | `ggml/include/ggml.h:2787-2794` | `ggml/src/ggml-quants.c` |
| Type traits | `ggml/include/ggml.h:2827` | `ggml/src/ggml.c` |

## Touch Points (What Calls GGML)

GGML is the **foundation of the entire llama.cpp project**. It has zero dependencies on other parts of the codebase; everything else depends on it.

- **`llama.h`** — The public llama API includes `ggml.h`, `ggml-cpu.h`, `ggml-backend.h`, `ggml-opt.h`, and `gguf.h` at its head (`include/llama.h:4-8`). Every model evaluation, KV-cache operation, and training step uses GGML tensors and graphs internally.
- **`ggml-backend`** (`ggml/src/ggml-backend.c`) — Backend abstraction layer built on top of `ggml_tensor` and `ggml_cgraph`.
- **`ggml-cpu`** (`ggml/src/ggml-cpu/`) — CPU-specific implementations of every `ggml_op`. Includes AVX, NEON, SVE, AMX, and CUDA/Metal backends.
- **`ggml-quants.c`** — Quantized type support, included from `ggml.c` (`ggml/src/ggml.c:11`).
- **`ggml.cpp`** — C++ wrapper that hooks `std::set_terminate` (`ggml/src/ggml.cpp:21-24`).
- **Training code** — The optimization step ops (`GGML_OP_OPT_STEP_ADAMW`, `GGML_OP_OPT_STEP_SGD` at `ggml/include/ggml.h:582-583`) and cross-entropy loss ops (`ggml/include/ggml.h:580-581`) are defined as first-class GGML graph nodes.
- **Model loading** — GGUF reader (`gguf.h`/`gguf.c`) works alongside GGML to deserialize tensors into `ggml_tensor` objects.

## Failure Modes

GGML has a multi-layered error handling strategy:

### Fatal errors via `GGML_ABORT` / `GGML_ASSERT`

The macro `GGML_ABORT(...)` expands to `ggml_abort(__FILE__, __LINE__, __VA_ARGS__)` (`ggml/include/ggml.h:287`). The function prints file/line context, calls an optional user-registered callback (`ggml_abort_callback_t`, `ggml/include/ggml.h:349`), and calls `abort()` (`ggml/src/ggml.c:252-272`).

`GGML_ASSERT(x)` expands to `if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)` (`ggml/include/ggml.h:288`). Assertion failures are **always fatal** and cannot be recovered from.

Typical assertion failures include:
- Invalid tensor types or dimensions (`ggml/src/ggml.c:1734-1735`)
- Shape mismatches in binary ops (e.g., `ggml_can_repeat`, `ggml/src/ggml.c:2023`)
- Invalid permutation axes (`ggml/src/ggml.c:3783-3793`)
- Backward pass without params/loss (`ggml/src/ggml.c:7023-7024`)

### Return codes via `enum ggml_status`

```c
enum ggml_status {
    GGML_STATUS_ALLOC_FAILED = -2,
    GGML_STATUS_FAILED = -1,
    GGML_STATUS_SUCCESS = 0,
    GGML_STATUS_ABORTED = 1,
};
```

(`ggml/include/ggml.h:358-363`)

`ggml_graph_compute()` returns `enum ggml_status` (`ggml/include/ggml-cpu.h:70`). The abort callback mechanism (`ggml_abort_callback`, `ggml/include/ggml.h:706`) allows cooperative cancellation: when the callback returns true, the threadpool sets `ec = GGML_STATUS_ABORTED` and computation stops.

### Memory exhaustion

The context arena uses a **soft-fail + optional ABORT** pattern:
- On overflow: logs a warning and returns `NULL` (`ggml/src/ggml.c:1694-1696`)
- In debug builds (`!NDEBUG`): also calls `GGML_ABORT` (`ggml/src/ggml.c:1698`)
- `ggml_malloc`/`ggml_calloc` wrappers abort on failure (`ggml/src/ggml.c:413,427`)

### Unreachable code

`GGML_UNREACHABLE()` (`ggml/include/ggml.h:270-277`) aborts in debug builds or invokes compiler intrinsics (`__builtin_unreachable`/`__assume(0)`) in release builds.

### Backward-pass unsupported ops

The backward builder asserts that every op in the forward graph has a backward implementation; unsupported ops trigger `GGML_ABORT` with the op name (`ggml/src/ggml.c:6887`).

### Summary

| Mechanism | Scope | Recoverable? |
|---|---|---|
| `GGML_ASSERT` | Precondition checks | No — `abort()` |
| `GGML_ABORT` | Fatal errors | No — `abort()` |
| `enum ggml_status` | Graph computation | Yes — caller checks return value |
| `ggml_abort_callback` | Cooperative cancellation | Yes — polling during compute |
| Logged warning + `NULL` | Arena OOM (release) | Yes — caller checks for NULL |
| `ggml_validate_row_data` | Data validation | Yes — returns `bool` (`ggml/include/ggml.h:797`) |
