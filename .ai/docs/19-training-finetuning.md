# Training & Finetuning (llama.cpp)

## Purpose

llama.cpp provides gradient-based model optimization through two distinct subsystems: the **GGML optimizer framework** (`ggml_opt`) for tensor-level training, and the **llama-level training loop** (`llama_opt`) which wraps the GGML optimizer for language model finetuning. The importance matrix (`imatrix`) subsystem enables improved quantization quality by capturing activation statistics during inference.

These features solve:
- **Finetuning** trained FP32 models on new data via cross-entropy loss
- **Quantization quality** improvement via importance-aware rounding during quantization
- **Memory optimization** via the `fit-params` heuristic that auto-tunes model/context params to available device memory

---

## 1. The Optimizer Framework: `ggml_opt`

### Overview

The `ggml_opt` API (`ggml/include/ggml-opt.h`) provides a C-level high-level framework for training GGML compute graphs. It manages datasets, forward/backward passes, gradient accumulation, and optimizer steps (AdamW or SGD).

**Maintainer:** Johannes Gäßler (`ggml/include/ggml-opt.h:5`)

### Key Types

| Type | Purpose |
|------|---------|
| `ggml_opt_dataset_t` | Opaque handle to a training dataset (data + optional labels) |
| `ggml_opt_context_t` | Opaque handle to the optimization context (graphs, gradients, momenta, optimizers) |
| `ggml_opt_result_t` | Accumulates loss/prediction/accuracy across eval calls |

(`ggml/include/ggml-opt.h:18-24`)

### Loss Functions

Four built-in loss types (`ggml/include/ggml-opt.h:30-35`):

```c
enum ggml_opt_loss_type {
    GGML_OPT_LOSS_TYPE_MEAN,
    GGML_OPT_LOSS_TYPE_SUM,
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
};
```

Custom losses can be defined via MEAN or SUM which simply reduce outputs to a scalar.

### Dataset

Datasets are initialized with data/label types, per-datapoint sizes, total count, and shard size (`ggml/include/ggml-opt.h:39-45`). Internally the dataset stores tensors on the CPU backend (`ggml/src/ggml-opt.cpp:86-130`). Data copying supports both device-tensor and host-memory access (`ggml_opt_dataset_get_batch` vs `ggml_opt_dataset_get_batch_host`). Shuffling uses the opt_ctx RNG and operates at shard granularity (`ggml/src/ggml-opt.cpp:150-161`).

### Optimizers

Two optimizers are supported (`ggml/include/ggml-opt.h:77-82`):

```c
enum ggml_opt_optimizer_type {
    GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    GGML_OPT_OPTIMIZER_TYPE_SGD,
};
```

**AdamW** (`ggml/src/ggml-cpu/ops.cpp:11279-11343`):
- Uses momentum buffers `m` and `v` per parameter
- Bias-corrected: `beta1h = 1/(1-beta1^iter)`, `beta2h = 1/(1-beta2^iter)`
- Update: `m = beta1*m + (1-beta1)*grad`; `v = beta2*v + (1-beta2)*grad^2`
- Weight decay decoupled from gradient: `w = w*(1-alpha*wd) - alpha*mh/(sqrt(vh)+eps)`
- References: [AdamW paper](https://arxiv.org/pdf/1711.05101v3.pdf)

**SGD** (`ggml/src/ggml-cpu/ops.cpp:11364-11406`):
- No momentum
- Update: `w = w*(1-alpha*wd) - alpha*grad`

Optimizer parameters are obtained via a user-supplied callback `ggml_opt_get_optimizer_params` (`ggml/include/ggml-opt.h:99-101`). Defaults: AdamW lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0; SGD lr=1e-3, wd=0.0 (`ggml/src/ggml-opt.cpp:223-237`).

### Graph Building (`ggml_opt_build`)

The function `ggml_opt_build` (`ggml/src/ggml-opt.cpp:322-547`) constructs up to three computation graphs:

1. **Forward only** (`GGML_OPT_BUILD_TYPE_FORWARD`): loss computation only
2. **Grad** (`GGML_OPT_BUILD_TYPE_GRAD`): forward + backward pass (gradient computation)
3. **Opt** (`GGML_OPT_BUILD_TYPE_OPT`): forward + backward + optimizer step

The build process:
- Tags input/output tensors (`ggml_set_input`, `ggml_set_output`)
- Counts parameter tensors (`GGML_TENSOR_FLAG_PARAM`)
- Allocates gradient accumulators and AdamW momenta (`m`, `v`) in a static context
- Constructs the loss tensor (mean, sum, cross-entropy, or MSE)
- For grad: duplicates forward graph and calls `ggml_build_backward_expand`
- For opt: duplicates grad graph, appends `ggml_opt_step_adamw`/`ggml_opt_step_sgd` ops for each parameter

### Gradient Accumulation

Controlled by `opt_period` (logical batch / physical batch). When `opt_period > 1`, gradients accumulate across multiple micro-batches before an optimizer step (`ggml/include/ggml-opt.h:123`). The state machine cycles `FORWARD -> GRAD -> ... -> GRAD -> OPT`.

### High-Level API: `ggml_opt_fit`

The function `ggml_opt_fit` (`ggml/src/ggml-opt.cpp:999-1079`) is a convenience wrapper that:
1. Creates an opt context with `ggml_opt_init`
2. Optionally shuffles the dataset
3. Runs `nepoch` epochs, each calling `ggml_opt_epoch`
4. Splits data into training/validation sets per `val_split`
5. Reports progress via optional callback

The lower-level `ggml_opt_epoch` (`ggml/src/ggml-opt.cpp:881-924`) iterates over batches in a single pass, doing backward passes for the training split and forward-only passes for the validation split.

---

## 2. Llama-Level Finetuning (`examples/training/`)

### Overview

The `examples/training/finetune.cpp` example implements language model finetuning using the `llama_opt` API. It is explicitly a **work in progress** (`examples/training/README.md:4`): "Finetuning is technically functional (for FP32 models and limited hardware setups) but the code is very much WIP."

### Workflow

1. **Load model** (FP32 required -- mmap is disabled to get writable weights, `finetune.cpp:29-33`)
2. **Force f32 cache types** due to `OUT_PROD` lacking f16 support (`finetune.cpp:34-41`)
3. **Tokenize** input text and create a dataset via `common_opt_dataset_init` (`common/common.cpp:1915-1930`)
4. **Initialize optimizer** via `llama_opt_init` (`llama.h:1574`, implementation at `src/llama-context.cpp:3245-3284`)
5. **Run epochs**: for each epoch, call `llama_opt_epoch` which iterates over the dataset in context-sized chunks (`src/llama-context.cpp:3395-3441`)
6. **Save** the finetuned model via `llama_model_save_to_file` (`finetune.cpp:96`)

### `llama_opt_init` Internals (`src/llama-context.cpp:3245-3284`)

- Asserts no existing opt_ctx
- Optionally overrides `n_ctx_train`
- Creates a GGML opt context with `GGML_OPT_LOSS_TYPE_CROSS_ENTROPY`
- Tags eligible tensors as trainable parameters via `llama_set_param`:
  - Embeddings (type_embd, pos_embd)
  - Layer norms (tok_norm, output_norm, output_norm_enc)
  - Output projection (output, output_b)
  - Classification heads (cls, cls_b, cls_out, cls_out_b, cls_norm)
  - All layer tensors (attention + FFN weights, norms)
  - **Excluded** (FIXME): token_embd.weight, rope_freqs.weight (`src/llama-context.cpp:3236-3241`)

The `llama_set_param` function only tags f32 tensors that pass the `param_filter` callback (`src/llama-context.cpp:3229-3243`).

### `llama_opt_epoch` Internals (`src/llama-context.cpp:3395-3441`)

- Splits dataset into training/validation portions
- For each datapoint, calls `opt_epoch_iter` which:
  - Builds the model compute graph with `model.build_graph(gparams)`
  - Calls `ggml_opt_prepare_alloc` and `ggml_opt_alloc` (backward=True for training, False for eval)
  - Sets sparse one-hot labels for cross-entropy
  - Calls `ggml_opt_eval`

### `llama_opt_params` (`llama.h:1562-1572`)

```c
struct llama_opt_params {
    uint32_t n_ctx_train;
    llama_opt_param_filter param_filter;
    void * param_filter_ud;
    ggml_opt_get_optimizer_params get_opt_pars;
    void * get_opt_pars_ud;
    enum ggml_opt_optimizer_type optimizer_type;
};
```

The `param_filter` callback determines which tensors are trainable. `llama_opt_param_filter_all` always returns `true` (`src/llama-context.cpp:4103`).

### Learning Rate Scheduling (`common/common.cpp:1932-1983`)

The `lr_opt` struct computes a half-life decay schedule:
```cpp
float lr_opt::get_lr(float epoch) const {
    return epoch >= decay_epochs ? lr_min :
           lr0 * pow(0.5f, epoch * scale_epoch);
}
```

---

## 3. Importance Matrix (`imatrix`)

### What It Is

An importance matrix stores per-element **squared activation sums** for each tensor in a model, collected during inference on calibration data. These values indicate which weights are "important" for model output quality -- weights with higher squared activations are more sensitive to quantization error.

**Purpose:** During quantization, the importance matrix guides rounding decisions to preserve important weights with higher precision (`tools/imatrix/README.md:3`).

### How It Is Computed (`tools/imatrix/imatrix.cpp`)

The `IMatrixCollector` class hooks into the computation graph via a callback (`collect_imatrix`):

1. **Hooks into every `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID` operation** (`imatrix.cpp:236-243`)
2. Filters: skips batches <16 tokens, skips non-f32 src1, skips non-`blk.*` tensors (unless `--process-output`)
3. Accumulates **squared activations** of the input (src1) for each weight tensor (`imatrix.cpp:321`)
4. Tracks counts per tensor/chunk for normalization

The data is stored per-tensor in `std::unordered_map<std::string, Stats>` where `Stats = {vector<float> values, vector<int64_t> counts}`.

### File Format

Two formats:
- **Legacy .dat** binary format (`save_imatrix_legacy`, `imatrix.cpp:407-499`)
- **GGUF format** (default, `imatrix.cpp:500+`)

GGUF metadata keys (`common/imatrix-loader.h:8-10`):
- `imatrix.datasets` -- dataset filenames
- `imatrix.chunk_count` -- number of chunks processed
- `imatrix.chunk_size` -- tokens per chunk

### Statistics (`imatrix.cpp:125-198`)

The tool can compute per-tensor statistics from imatrix data:
- `Sum(Act^2)` -- total importance score
- `Min/Max/Mean/Stddev` of squared activations
- `% Active` -- proportion of elements above 1e-5 threshold
- `Entropy` -- Shannon entropy of activation distribution
- `ZD Score` -- z-score distribution ([Layer-Wise Quantization](https://arxiv.org/abs/2406.17415))
- `CosSim` -- cosine similarity vs previous layer's squared activations

### Usage in Quantization (`src/llama-quant.cpp`)

The quantizer accepts `llama_model_quantize_params::imatrix` (`llama.h:420`), a pointer to `llama_model_imatrix_data` entries (`llama.h:402-406`):

```c
struct llama_model_imatrix_data {
    const char * name;
    const float * data;
    size_t size;
};
```

During quantization (`src/llama-quant.cpp:1180-1197`):
- Each tensor's imatrix entry is looked up by name (with remapping for split tensors)
- The imatrix pointer is passed to `ggml_quantize_chunk` for importance-aware quantization
- Some quantization types *require* an imatrix (IQ3_XXS, Q4_0/Q5_0 early layers) (`src/llama-quant.cpp:519-609`)

### CLI Tool

`tools/imatrix/imatrix.cpp` provides the `llama-imatrix` binary. Run on a model + calibration text:
```sh
./llama-imatrix -m model.gguf -f calibration.txt -ngl 99
./llama-quantize --imatrix imatrix.gguf model.gguf model-q4_k_m.gguf Q4_K_M
```

---

## 4. `tools/fit-params/` Tuning Harness

### Purpose

The `fit-params` subsystem auto-tunes `llama_model_params` and `llama_context_params` so that a model fits within available device memory (`tools/fit-params/README.md:1-6`). It is not about training; it's a **memory fitting** heuristic.

### How It Works (`common/fit.cpp`)

The core function `common_fit_params` (`common/fit.h:19-27`, impl at `common/fit.cpp:789+`):

1. **Loads the model** with `no_alloc=true` and `use_mmap=false` to estimate memory without allocating (`fit.cpp:55-60`)
2. **Gets memory breakdown** via `llama_get_memory_breakdown` (`fit.cpp:76-98`)
3. **Compares projected usage vs free device memory** (with configurable margin, default 1 GiB per device)
4. **Iteratively reduces** (if needed):
   - Context size (`n_ctx`) down to `n_ctx_min`
   - GPU layers (`n_gpu_layers`)
   - Offloads specific tensor groups to CPU via `tensor_buft_overrides`

### CLI Tool `llama-fit-params`

`tools/fit-params/fit-params.cpp` (`llama_fit_params`) prints the CLI arguments that achieve the fitted memory use:
```sh
./llama-fit-params --model model.gguf
# Outputs: -c 4096 -ngl 48 -ot blk.14.ffn_up=CPU,...
```

The output can be piped to any llama.cpp binary:
```sh
./llama-fit-params --model model.gguf | xargs ./llama-server --model model.gguf
```

### Integration

- `llama-bench` uses `fit_params_target` and `fit_params_min_ctx` to adjust settings per benchmark scenario (`tools/llama-bench/llama-bench.cpp:1267-1268`)
- Server auto-fits on startup (`tools/server/server-context.cpp:1082+`)
- The `common_params` struct has `fit_params=true` by default for all tools (`common/common.h:467`)

---

## 5. Touch Points

### GGML Tensor Library

- **Optimizer ops**: `GGML_OP_OPT_STEP_ADAMW` and `GGML_OP_OPT_STEP_SGD` are native ggml ops with CPU kernel implementations (`ggml/src/ggml-cpu/ops.cpp:11279-11418`)
- **Backward pass**: `ggml_build_backward_expand` enables automatic differentiation through arbitrary compute graphs (`ggml/src/ggml-opt.cpp:490`)
- **Loss ops**: `ggml_cross_entropy_loss`, `ggml_sum`, `ggml_sub`, `ggml_sqr`, `ggml_scale`
- **Tensor flags**: `GGML_TENSOR_FLAG_PARAM`, `GGML_TENSOR_FLAG_LOSS`, `GGML_TENSOR_FLAG_INPUT`, `GGML_TENSOR_FLAG_OUTPUT`

### Quantization

- `ggml_quantize_chunk` accepts an optional `imatrix` pointer per row for importance-aware quantization (`src/llama-quant.cpp:709-738`)
- Some quantization types *require* an imatrix (`src/llama-quant.cpp:1043-1053`)
- The `llama_model_imatrix_data` struct is the bridge between imatrix collection and quantization (`llama.h:402-406`)

---

## 6. Failure Modes

| Failure Mode | Cause | Evidence |
|-------------|-------|----------|
| F32-only training | `llama_set_param` skips non-f32 tensors | `src/llama-context.cpp:3230` |
| No mmap during training | Writable weight pointers required | `finetune.cpp:29-33` |
| F16 cache not supported | `OUT_PROD` lacks f16 kernel | `finetune.cpp:34-41` |
| token_embd, rope_freqs excluded | Explicit FIXMEs | `src/llama-context.cpp:3236-3241` |
| High memory usage | Requires ~24 GB for 1B model | `examples/training/README.md:5` |
| Tensor name mismatches during quantization | Split tensors vs imatrix naming | `src/llama-quant.cpp:1183-1197` |
| Incomplete imatrix (MoE) | Unused experts have zero counts | `imatrix.cpp:441-448` |
| Non-finite values in imatrix | NaNs/Infs abort quantization | `src/llama-quant.cpp:913-916` |
| fit-params is not thread-safe | Modifies global logger state | `common/fit.h:16` |

---

## File Index

| File | Role |
|------|------|
| `ggml/include/ggml-opt.h` | Public API: dataset, context, result, loss, optimizers, fit/epoch |
| `ggml/include/ggml.h` | `ggml_opt_step_adamw`, `ggml_opt_step_sgd` declarations |
| `ggml/src/ggml-opt.cpp` | Implementation: dataset, build, eval, epoch, fit |
| `ggml/src/ggml-cpu/ops.cpp` | CPU kernels for AdamW and SGD steps |
| `include/llama.h` | `llama_opt_params`, `llama_opt_init`, `llama_opt_epoch`, `llama_model_imatrix_data` |
| `src/llama-context.cpp` | llama-level training loop: `opt_init`, `opt_epoch_iter`, `opt_epoch` |
| `examples/training/finetune.cpp` | CLI finetuning example |
| `examples/training/README.md` | Finetuning usage and caveats |
| `tools/imatrix/imatrix.cpp` | Importance matrix collector and CLI |
| `tools/imatrix/README.md` | imatrix usage and statistics |
| `common/imatrix-loader.h` | imatrix data structures and loading |
| `common/imatrix-loader.cpp` | GGUF and legacy imatrix loader |
| `src/llama-quant.cpp` | Quantization with imatrix support |
| `common/fit.h` | `common_fit_params` declaration |
| `common/fit.cpp` | Memory fitting implementation |
| `tools/fit-params/fit-params.cpp` | `llama-fit-params` CLI |
| `tools/fit-params/README.md` | fit-params usage |
| `common/common.cpp` | `common_opt_dataset_init`, `common_opt_lr_pars`, `lr_opt` |
