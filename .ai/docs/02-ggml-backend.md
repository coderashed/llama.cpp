# GGML Backend System

## Purpose

The GGML Backend System provides a uniform hardware abstraction layer so that the same compute graph can be executed on CPU, CUDA, Metal, Vulkan, SYCL, RPC, and other accelerators without changing the caller's code. Instead of calling device-specific routines directly, the caller interacts with abstract **backends**, **devices**, **buffers**, and **buffer types**. Each hardware platform implements a small set of interface vtable structs (`ggml_backend_i`, `ggml_backend_device_i`, `ggml_backend_reg_i`, `ggml_backend_buffer_i`, `ggml_backend_buffer_type_i`) to expose its capabilities. The scheduler (`ggml_backend_sched`) automatically assigns graph nodes to the most suitable backend based on operator support, buffer location, and priority ordering.

## Architecture Overview

```
llama.cpp / src/llama.cpp
  |
  v  llama_backend_init() -> ggml_backend_load_all()
  v  llama_prepare_model_devices() -> selects devices
  v  load_tensors() -> allocates model weights on chosen devices
  v  inference -> ggml_backend_sched_graph_compute()
                    |
                    v  ggml_backend_sched_split_graph()  (assigns ops to backends)
                    v  ggml_backend_sched_alloc_splits()  (allocates tensors per split)
                    v  ggml_backend_sched_compute_splits() (copies inputs, dispatches each split)
  |
  v  per-split: ggml_backend_graph_compute_async(backend, &split->graph)
                  |
                  v  backend->iface.graph_compute(backend, cgraph)
                  v  e.g. ggml_backend_cpu_graph_compute() -> ggml_graph_compute()
                  v  e.g. CUDA/Metal/Vulkan kernel launches
```

## Key Types and Protocols

### `ggml_backend_reg` (Registry)

Each hardware platform exposes a singleton `ggml_backend_reg` that owns a list of devices and provides a name and optional extension functions via `get_proc_address`. Defined in `ggml-backend-impl.h:226-230`:

```c
struct ggml_backend_reg {
    int api_version;                // must be GGML_BACKEND_API_VERSION
    struct ggml_backend_reg_i iface;
    void * context;
};
```

The interface (`ggml-backend-impl.h:214-224`):
- `get_name` -- human-readable backend name (e.g. "CUDA", "Metal")
- `get_device_count` / `get_device` -- enumerate available devices
- `get_proc_address` -- optional; lookup extension functions by name

Registration examples:
- CPU: `ggml_backend_cpu_reg()` at `ggml-cpu/ggml-cpu.cpp:690`
- CUDA: `ggml_backend_cuda_reg()` at `ggml-cuda/ggml-cuda.cu:5650`
- Metal: `ggml_backend_metal_reg()` at `ggml-metal/ggml-metal.cpp:908`
- Vulkan: `ggml_backend_vk_reg()` at `ggml-vulkan/ggml-vulkan.cpp:17805`
- RPC: `ggml_backend_rpc_reg()` at `ggml-rpc/ggml-rpc.cpp:1902`

### `ggml_backend_dev` (Device)

A physical or logical compute device. Defined in `ggml-backend-impl.h:204-208`:

```c
struct ggml_backend_device {
    struct ggml_backend_device_i iface;
    ggml_backend_reg_t reg;
    void * context;
};
```

The interface (`ggml-backend-impl.h:160-202`) includes:
- `get_name`, `get_description`, `get_memory`, `get_type` -- basic properties
- `get_props` -- fills a `ggml_backend_dev_props` struct (name, desc, memory, type, device_id, caps)
- `init_backend` -- create a `ggml_backend_t` (stream) from this device
- `get_buffer_type` / `get_host_buffer_type` -- preferred buffer types
- `buffer_from_host_ptr` -- wrap a host pointer in a backend buffer
- `supports_op` / `supports_buft` / `offload_op` -- capability queries
- `event_new` / `event_free` / `event_synchronize` -- optional event sync

Device types are enumerated in `ggml-backend.h:134-145`:
- `GGML_BACKEND_DEVICE_TYPE_CPU`
- `GGML_BACKEND_DEVICE_TYPE_GPU`
- `GGML_BACKEND_DEVICE_TYPE_IGPU`
- `GGML_BACKEND_DEVICE_TYPE_ACCEL`
- `GGML_BACKEND_DEVICE_TYPE_META`

Capabilities are described in `ggml_backend_dev_caps` (`ggml-backend.h:148-157`): `async`, `host_buffer`, `buffer_from_host_ptr`, `events`.

### `ggml_backend` (Stream)

A backend "stream" -- an execution context for a specific device. Defined in `ggml-backend-impl.h:142-147`:

```c
struct ggml_backend {
    ggml_guid_t guid;
    struct ggml_backend_i iface;
    ggml_backend_dev_t device;
    void * context;
};
```

The stream interface (`ggml-backend-impl.h:105-140`) includes:
- `get_name`, `free`
- `set_tensor_async` / `get_tensor_async` / `cpy_tensor_async` -- async tensor ops (optional; falls back to sync)
- `synchronize` -- flush pending operations
- `graph_plan_create` / `graph_plan_free` / `graph_plan_update` / `graph_plan_compute` -- optional graph plans (not used currently per comment on line 120)
- `graph_compute` -- execute a compute graph (always async if supported)
- `event_record` / `event_wait` -- cross-stream sync
- `graph_optimize` -- optional graph-level optimization

### `ggml_backend_buffer_type` and `ggml_backend_buffer`

Buffer types describe allocation strategies; buffers are allocated instances. Defined in `ggml-backend-impl.h:17-70`.

Buffer type interface (`ggml-backend-impl.h:17-29`):
- `get_name`, `alloc_buffer`, `get_alignment`, `get_max_size` (optional), `get_alloc_size` (optional), `is_host` (optional)

Buffer interface (`ggml-backend-impl.h:41-62`):
- `free_buffer`, `get_base`, `init_tensor` (optional), `memset_tensor`, `set_tensor`, `get_tensor`, `cpy_tensor` (optional), `clear`, `reset` (optional)

Buffer usage flags (`ggml-backend.h:49-53`):
- `GGML_BACKEND_BUFFER_USAGE_ANY` -- default
- `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` -- model weights (hints scheduler to prefer same backend)
- `GGML_BACKEND_BUFFER_USAGE_COMPUTE` -- intermediate results

### `ggml_backend_sched` (Scheduler)

The scheduler orchestrates multi-backend execution. Defined as a private struct at `ggml-backend.cpp:774-828`:

```c
struct ggml_backend_sched {
    bool is_reset, is_alloc;
    int n_backends;
    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];    // priority-ordered
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;
    // ... hash maps, split list, copy/event tracking, callbacks
};
```

Properties:
- Backends are priority-ordered by array index (lower index = higher priority)
- The last backend MUST be CPU (`ggml-backend.cpp:1736`)
- Supports pipeline parallelism via `n_copies` and `events` (`ggml-backend.cpp:804-808`)
- SPLIT_MAX_INPUTS = 30, MAX_BACKENDS = 16, MAX_COPIES = 4

### Split Structure

```c
struct ggml_backend_sched_split {  // ggml-backend.cpp:764-772
    int backend_id;
    int i_start, i_end;           // node range in the original graph
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    struct ggml_cgraph graph;     // graph view for this split
};
```

### Meta Backend

The meta backend (`ggml-backend-meta.cpp`) wraps multiple devices for tensor parallelism. Created via `ggml_backend_meta_device()` (`ggml-backend.h:402-403`). Key types:
- `ggml_backend_meta_device_context` (`ggml-backend-meta.cpp:54-83`) -- holds a vector of `simple_devs`, split state function, and user data
- `ggml_backend_meta_split_state` (`ggml-backend.h:376-394`) -- describes how a tensor is split across devices (axis, segments, repetitions)
- Split axes: `GGML_BACKEND_SPLIT_AXIS_0..3` (tensor dimension), `MIRRORED` (all values on all devices), `PARTIAL` (partial sum per device)

## Entry Points

### Backend Initialization and Loading

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `llama_backend_init()` | `src/llama.cpp:89` | Called by user code; loads all backends if none registered |
| `ggml_backend_load_all()` | `ggml-backend-reg.cpp:555` | Loads known backends from files matching `[lib]ggml-{name}-*.{so,dll}` |
| `ggml_backend_load_all_from_path()` | `ggml-backend-reg.cpp:559` | Same but with a custom search directory |
| `ggml_backend_load()` | `ggml-backend-reg.cpp:386` | Load a single backend .so/.dll by path |
| `ggml_backend_unload()` | `ggml-backend-reg.cpp:390` | Unload a dynamically loaded backend |

### Registration

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `ggml_backend_register()` | `ggml-backend-reg.cpp:291` | Register a backend + its devices |
| `ggml_backend_device_register()` | `ggml-backend-reg.cpp:295` | Register a standalone device |
| `ggml_backend_reg_count()` | `ggml-backend-reg.cpp:309` | Number of registered backends |
| `ggml_backend_dev_count()` | `ggml-backend-reg.cpp:329` | Number of registered devices globally |

### Compile-Time Registration

The static registry constructor at `ggml-backend-reg.cpp:115-167` calls `register_backend()` for each backend whose `GGML_USE_*` macro is defined, in a specific order. The CPU backend is registered last (`ggml-backend-reg.cpp:164-166`).

### Dynamic Loading

Dynamic backends use `GGML_BACKEND_DL_IMPL` to expose `ggml_backend_init()` and `ggml_backend_score()` (`ggml-backend-impl.h:240-271`). The `load_backend` method (`ggml-backend-reg.cpp:213-257`) checks the API version and score, then loads the shared library.

### Device Enumeration

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `ggml_backend_dev_get(i)` | `ggml-backend-reg.cpp:333` | Get device by global index |
| `ggml_backend_dev_by_name()` | `ggml-backend-reg.cpp:338` | Find device by name (case-insensitive) |
| `ggml_backend_dev_by_type()` | `ggml-backend-reg.cpp:348` | Find first device of a given type |
| `ggml_backend_dev_init()` | `ggml-backend.cpp:599` | Create a backend (stream) from a device |
| `ggml_backend_init_best()` | `ggml-backend-reg.cpp:375` | Init GPU if available, else iGPU, else CPU |

### Buffer Allocation

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `ggml_backend_buft_alloc_buffer()` | `ggml-backend.cpp:38` | Allocate buffer from a buffer type |
| `ggml_backend_alloc_buffer()` | `ggml-backend.cpp:242` | Allocate default buffer type for a backend |
| `ggml_backend_cpu_buffer_type()` | `ggml-backend.cpp:2328` | Get the global CPU buffer type |
| `ggml_backend_cpu_buffer_from_ptr()` | `ggml-backend.cpp:2368` | Wrap existing host memory in a CPU buffer |

### Scheduler

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `ggml_backend_sched_new()` | `ggml-backend.cpp:1727` | Create scheduler with ordered backends + optional bufts |
| `ggml_backend_sched_reserve()` | `ggml-backend.cpp:1847` | Reserve space using a measure graph |
| `ggml_backend_sched_alloc_graph()` | `ggml-backend.cpp:1864` | Allocate tensors for a concrete graph |
| `ggml_backend_sched_graph_compute()` | `ggml-backend.cpp:1883` | Synchronous: alloc + compute + sync |
| `ggml_backend_sched_graph_compute_async()` | `ggml-backend.cpp:1889` | Async: alloc + compute (no sync) |
| `ggml_backend_sched_split_graph()` | `ggml-backend.cpp:1014` | Assign backends and split graph |
| `ggml_backend_sched_reset()` | `ggml-backend.cpp:1821` | Reset allocations between graphs |
| `ggml_backend_sched_set_tensor_backend()` | `ggml-backend.cpp:1960` | Manually assign a tensor to a backend |

## Data Flow: Graph Dispatch

### Step 1: Backend Assignment (`ggml_backend_sched_split_graph`, line 1014)

Five passes assign each graph node to a backend:

1. **Pass 1** (line 1036-1069): Assign nodes with pre-allocated buffers (weights, views, graph inputs). Uses `ggml_backend_sched_backend_id_from_cur()` (line 878) which checks, in order:
   - Tensor's own buffer backend
   - View source buffer backend
   - Graph input flag -> assign to CPU (last backend)
   - Weight (USAGE_WEIGHTS) buffer backend
   - Optionally, offload to higher-priority backend if `op_offload` is set

2. **Pass 2** (line 1072-1149): Expand GPU backends up and down through adjacent nodes, then expand all backends. CPU-range nodes are not expanded past (treated as boundaries).

3. **Pass 3** (line 1152-1211): Upgrade nodes to higher-priority backends with compatible buffer types. For unassigned nodes, pick the backend supporting the most inputs.

4. **Pass 4** (line 1213-1243): Backfill remaining unassigned sources from their destination backend or view source. Every node MUST be assigned by the end of this pass (assert at line 1242).

5. **Pass 5** (line 1245-1487): Split the graph into contiguous ranges (`ggml_backend_sched_split`) where all nodes run on the same backend. Cross-backend tensor references become split inputs that will be copied. The copy tensors are created and tracked via `tensor_id_copy` (line 832).

After splitting, `ggml_backend_graph_optimize` is called per-split (line 1417) for backend-specific optimizations. The backend IDs and leaf IDs are stored for the allocator.

### Step 2: Allocation (`ggml_backend_sched_alloc_splits`, line 1489)

- Compares current backend IDs with previous ones
- If unchanged and the gallocr can reuse the previous allocation, skips reallocation
- Otherwise, reserves and allocates via `ggml_gallocr_reserve_n` + `ggml_gallocr_alloc_graph`

### Step 3: Compute (`ggml_backend_sched_compute_splits`, line 1541)

For each split:
1. Copy input tensors from their home backend to the split's backend (line 1554-1674)
   - For MoE `MUL_MAT_ID` ops with USAGE_WEIGHTS inputs, only copies the used experts (lines 1576-1660)
   - Tries async copy first (`cpy_tensor_async`), falls back to sync copy
2. Call `ggml_backend_graph_compute_async()` on the split's backend (line 1678)
3. Record an event on the split backend to synchronize the next copy iteration (line 1718-1720)

### Step 4: Synchronize (`ggml_backend_sched_graph_compute`, line 1883)

The synchronous wrapper calls compute_async then synchronizes all backends.

### Pipeline Parallelism

When `parallel=true` in `ggml_backend_sched_new`, the scheduler creates `n_copies` (up to 4) copies of each split input tensor and cycles through them (`cur_copy`, `next_copy`). Events track when each copy is safe to reuse.

## Touch Points

### llama.cpp Initialization

- `llama_backend_init()` at `src/llama.cpp:89-102`:
  - Calls `ggml_backend_load_all()` if no backends are registered
- `llama_backend_free()` at `src/llama.cpp:116-118`:
  - Calls `ggml_quantize_free()` (no backend-specific teardown)

### Model Loading (`src/llama.cpp`)

- `llama_prepare_model_devices()` at `src/llama.cpp:125-276`:
  - Enumerates all `ggml_backend_dev_t` via `ggml_backend_dev_get()`
  - Categorizes into GPUs, iGPUs, RPC servers, CPU
  - Deduplicates GPUs by PCI device ID (lines 203-223)
  - For `LLAMA_SPLIT_MODE_TENSOR`: creates a `ggml_backend_meta_device` that wraps all non-CPU devices
  - For `LLAMA_SPLIT_MODE_NONE` with `main_gpu >= 0`: uses only the specified device
  - Populates `model->devices` with `{bool is_meta, ggml_backend_dev_t dev}` pairs
  - Logs device info (name, description, device_id, free memory)

- `llama_model_base::load_tensors()` at `src/llama-model.cpp:1209`:
  - Builds buffer type lists (`buft_list_t`) per device via `make_cpu_buft_list()` and `make_gpu_buft_list()`
  - Calculates layer-to-device splits based on free memory or user-provided `tensor_split`
  - Creates `ggml_backend_buffer_type_t` lists for each layer
  - Allocates model tensors on the assigned devices

### Inference

- The context creates a `ggml_backend_sched` via `ggml_backend_sched_new()` with the selected device backends
- The compute graph is built using `ggml_*` ops
- `ggml_backend_sched_graph_compute()` dispatches the graph to the scheduler
- The abort callback is set on backends that support it (CPU via `ggml_backend_cpu_set_abort_callback`, `ggml-cpu/ggml-cpu.cpp:272`; CUDA via `ggml_backend_cuda_reg_get_proc_address`)

### Sampling

- Samplers can use `ggml_backend_buffer_type_t` for GPU-accelerated sampling (`include/llama.h:1255`)

## Failure Modes

### Error Reporting

The backend system uses `enum ggml_status` (`ggml/include/ggml.h:359-362`):

| Status | Meaning |
|--------|---------|
| `GGML_STATUS_SUCCESS` (0) | Operation completed normally |
| `GGML_STATUS_FAILED` (-1) | Generic failure |
| `GGML_STATUS_ALLOC_FAILED` (-2) | Memory allocation failed |
| `GGML_STATUS_ABORTED` (1) | User callback requested abort |

### Error Propagation in Key Functions

- **Buffer allocation failure**: `ggml_backend_buft_alloc_buffer()` returns NULL; callers like `ggml_backend_cpu_buffer_type_alloc_buffer` (`ggml-backend.cpp:2305-2314`) log an error and return NULL.
- **Graph allocation failure**: `ggml_backend_sched_alloc_graph()` returns `false` (boolean). `ggml_backend_sched_alloc_splits()` at line 1532-1534 logs an error and returns false.
- **Graph compute failure**: `ggml_backend_sched_compute_splits()` returns the `enum ggml_status` from `ggml_backend_graph_compute_async()`. If it is not `GGML_STATUS_SUCCESS`, the scheduler returns it immediately (lines 1679-1681, 1701-1703). The synchronous wrapper `ggml_backend_sched_graph_compute()` and `ggml_backend_graph_compute()` return the error code after synchronizing.
- **Dynamic loading failure**: `ggml_backend_registry::load_backend()` reports errors via `GGML_LOG_ERROR` and returns NULL. Possible causes:
  - Cannot load `.so`/`.dll` (line 217)
  - Backend reports score 0 (unsupported on this system, line 223)
  - `ggml_backend_init` symbol not found (line 232)
  - Init returns NULL (line 243)
  - API version mismatch (line 245)
- **Backend not found**: `ggml_backend_init_by_name()` returns NULL; `ggml_backend_dev_by_name()` returns NULL (`ggml-backend-reg.cpp:338-346`). Model loading checks `ggml_backend_reg_count()` at `src/llama.cpp:367` and returns an error if zero.
- **No CPU backend**: `llama_prepare_model_devices` asserts CPU backend exists at `src/llama.cpp:106-107`. `ggml_backend_sched_new()` asserts last backend is CPU at `ggml-backend.cpp:1736`.
- **Pre-allocated tensor on wrong backend**: `ggml_backend_sched_backend_id_from_cur()` aborts at `ggml-backend.cpp:898` if a pre-allocated tensor's buffer doesn't support the operation.
- **Allocation errors in backends**: CPU backend returns `GGML_STATUS_ALLOC_FAILED` from `ggml_backend_cpu_graph_compute()` at `ggml-cpu/ggml-cpu.cpp:180` if it can't allocate work buffer.
- **Assertion failures**: The codebase uses `GGML_ASSERT()` extensively for invariant violations (e.g., `ggml-backend.cpp:1242` asserts every node is assigned).
- **Abort callback**: Backends that support it (CPU: `ggml-cpu/ggml-cpu.cpp:272`; CUDA: `ggml-cuda/ggml-cuda.cu:5630-5631` get_proc_address) check the callback during compute and return `GGML_STATUS_ABORTED` if the user requests cancellation.

### Logging

All backends use the `GGML_LOG_*` macros (`GGML_LOG_ERROR`, `GGML_LOG_INFO`, `GGML_LOG_DEBUG`) for diagnostics. These are controlled by compile-time `NDEBUG` and runtime environment variables.

## Backend Registration Sequence (Static Builds)

The `ggml_backend_registry` constructor at `ggml-backend-reg.cpp:115-167` registers backends in this order:

1. CUDA (`GGML_USE_CUDA`, line 117)
2. Metal (`GGML_USE_METAL`, line 120)
3. SYCL (`GGML_USE_SYCL`, line 123)
4. Vulkan (`GGML_USE_VULKAN`, line 126; runtime-disable via `GGML_DISABLE_VULKAN`)
5. WebGPU (`GGML_USE_WEBGPU`, line 134)
6. ZDNN (`GGML_USE_ZDNN`, line 137)
7. VirtGPU (`GGML_USE_VIRTGPU_FRONTEND`, line 140)
8. OpenCL (`GGML_USE_OPENCL`, line 144)
9. ZEN DNN (`GGML_USE_ZENDNN`, line 147)
10. Hexagon (`GGML_USE_HEXAGON`, line 150)
11. CANN (`GGML_USE_CANN`, line 153)
12. BLAS (`GGML_USE_BLAS`, line 156)
13. RPC (`GGML_USE_RPC`, line 159)
14. OpenVINO (`GGML_USE_OPENVINO`, line 162)
15. CPU (`GGML_USE_CPU`, line 165) -- always last

Each `register_backend()` call also registers all devices from that backend (`ggml-backend-reg.cpp:195-197`).

## Extension Mechanism

Backends can provide custom functions via `get_proc_address` (`ggml-backend-impl.h:222-223`). Standard extension function pointer types are defined in `ggml-backend.h:206-223`:

- `ggml_backend_comm_init_t` -- initialize inter-backend communication (tensor parallelism)
- `ggml_backend_comm_free_t` -- free communication context
- `ggml_backend_comm_allreduce_tensor_t` -- all-reduce across backends
- `ggml_backend_split_buffer_type_t` -- create split buffer type (deprecated)
- `ggml_backend_set_n_threads_t` -- set thread count
- `ggml_backend_set_abort_callback_t` -- set abort callback
- `ggml_backend_get_features_t` -- get feature flags
- `ggml_backend_dev_get_extra_bufts_t` -- get extra buffer types from device

Example: CUDA exposes comm functions for multi-GPU all-reduce (`ggml-cuda/ggml-cuda.cu:5616-5618`).

---

*Documentation generated from the llama.cpp codebase. For the most current information, refer to the source files.*
