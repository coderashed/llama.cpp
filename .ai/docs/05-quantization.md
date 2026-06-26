# Quantization

## Purpose

Quantization reduces model weight precision from 32-bit or 16-bit floats to
lower-bit representations (2--8 bits per weight), shrinking model file size and
memory footprint at inference time in exchange for a controlled accuracy loss.

The quantizer reads an existing model (GGUF file), dequantizes each tensor to
f32, then re-quantizes to the target type and writes a new GGUF file. The
runtime loader memory-maps the quantized file and dequantizes weights on the
fly during inference using `ggml_type_traits.to_float` callbacks.

## The Quant Type System

### `ggml_type` enum (lower-level, per-tensor)

Defined at `ggml/include/ggml.h:389-433`. Each value maps to a specific block
storage format. Quantized types all share a block size (either `QK_K = 256`
or `QK4_NL = 32`, defined at `ggml/src/ggml-common.h:89` and
`ggml/src/ggml-common.h:437`):

- Legacy: `GGML_TYPE_Q4_0` (2), `GGML_TYPE_Q4_1` (3), `GGML_TYPE_Q5_0` (6),
  `GGML_TYPE_Q5_1` (7), `GGML_TYPE_Q8_0` (8)
- K-quants: `GGML_TYPE_Q2_K` (10), `GGML_TYPE_Q3_K` (11), `GGML_TYPE_Q4_K`
  (12), `GGML_TYPE_Q5_K` (13), `GGML_TYPE_Q6_K` (14), `GGML_TYPE_Q8_K` (15)
- IQ-family: `GGML_TYPE_IQ2_XXS` (16), `GGML_TYPE_IQ2_XS` (17),
  `GGML_TYPE_IQ3_XXS` (18), `GGML_TYPE_IQ1_S` (19), `GGML_TYPE_IQ4_NL` (20),
  `GGML_TYPE_IQ3_S` (21), `GGML_TYPE_IQ2_S` (22), `GGML_TYPE_IQ4_XS` (23),
  `GGML_TYPE_IQ1_M` (29)
- Ternary: `GGML_TYPE_TQ1_0` (34), `GGML_TYPE_TQ2_0` (35)
- MX/NV FP4: `GGML_TYPE_MXFP4` (39), `GGML_TYPE_NVFP4` (40)
- `GGML_TYPE_Q1_0` (41)
- Non-quantized: `GGML_TYPE_F32` (0), `GGML_TYPE_F16` (1), `GGML_TYPE_BF16`
  (30), `GGML_TYPE_I8` (24) through `GGML_TYPE_F64` (28)

### `llama_ftype` enum (model-level, file metadata)

Defined at `include/llama.h:116-160`. Values like `LLAMA_FTYPE_MOSTLY_Q4_K_M`
(15) describe the overall model file type. These map to `ggml_type` via
`llama_ftype_get_default_type()` at `src/llama-quant.cpp:792-834`. Each
`LLAMA_FTYPE_MOSTLY_*` corresponds to one or more `GGML_TYPE_*` values.
For K-quant mixtures, multiple variants use the same base
`GGML_TYPE_*`:

| `llama_ftype` | `ggml_type` (base) | Effective bpw |
|---|---|---|
| `LLAMA_FTYPE_MOSTLY_Q2_K` (10) | `GGML_TYPE_Q2_K` | ~2.6 |
| `LLAMA_FTYPE_MOSTLY_Q2_K_S` (21) | `GGML_TYPE_Q2_K` | ~2.6 |
| `LLAMA_FTYPE_MOSTLY_Q3_K_S` (11) | `GGML_TYPE_Q3_K` | ~3.4 |
| `LLAMA_FTYPE_MOSTLY_Q3_K_M` (12) | `GGML_TYPE_Q3_K` | ~3.4 |
| `LLAMA_FTYPE_MOSTLY_Q3_K_L` (13) | `GGML_TYPE_Q3_K` | ~3.4 |
| `LLAMA_FTYPE_MOSTLY_Q4_K_S` (14) | `GGML_TYPE_Q4_K` | ~4.5 |
| `LLAMA_FTYPE_MOSTLY_Q4_K_M` (15) | `GGML_TYPE_Q4_K` | ~4.5 |
| `LLAMA_FTYPE_MOSTLY_Q5_K_S` (16) | `GGML_TYPE_Q5_K` | ~5.5 |
| `LLAMA_FTYPE_MOSTLY_Q5_K_M` (17) | `GGML_TYPE_Q5_K` | ~5.5 |
| `LLAMA_FTYPE_MOSTLY_Q6_K` (18) | `GGML_TYPE_Q6_K` | ~6.6 |
| `LLAMA_FTYPE_MOSTLY_IQ2_XXS` (19) | `GGML_TYPE_IQ2_XXS` | 2.0625 |
| `LLAMA_FTYPE_MOSTLY_IQ2_XS` (20) | `GGML_TYPE_IQ2_XS` | 2.3125 |
| `LLAMA_FTYPE_MOSTLY_IQ3_XXS` (23) | `GGML_TYPE_IQ3_XXS` | 3.0625 |
| `LLAMA_FTYPE_MOSTLY_IQ1_S` (24) | `GGML_TYPE_IQ1_S` | 1.5625 |
| `LLAMA_FTYPE_MOSTLY_IQ4_NL` (25) | `GGML_TYPE_IQ4_NL` | 4.5 |
| `LLAMA_FTYPE_MOSTLY_IQ3_S` (26) | `GGML_TYPE_IQ3_S` | 3.4375 |
| `LLAMA_FTYPE_MOSTLY_IQ3_M` (27) | `GGML_TYPE_IQ3_S` | ~3.66 |
| `LLAMA_FTYPE_MOSTLY_IQ2_S` (28) | `GGML_TYPE_IQ2_XS` | 2.5 |
| `LLAMA_FTYPE_MOSTLY_IQ2_M` (29) | `GGML_TYPE_IQ2_S` | ~2.7 |
| `LLAMA_FTYPE_MOSTLY_IQ4_XS` (30) | `GGML_TYPE_IQ4_XS` | 4.25 |
| `LLAMA_FTYPE_MOSTLY_IQ1_M` (31) | `GGML_TYPE_IQ1_M` | 1.75 |
| `LLAMA_FTYPE_MOSTLY_TQ1_0` (36) | `GGML_TYPE_TQ1_0` | 1.69 |
| `LLAMA_FTYPE_MOSTLY_TQ2_0` (37) | `GGML_TYPE_TQ2_0` | 2.06 |
| `LLAMA_FTYPE_MOSTLY_MXFP4_MOE` (38) | `GGML_TYPE_MXFP4` | 4 |
| `LLAMA_FTYPE_MOSTLY_NVFP4` (39) | `GGML_TYPE_NVFP4` | 4 |
| `LLAMA_FTYPE_MOSTLY_Q1_0` (40) | `GGML_TYPE_Q1_0` | 1 |

(Mapping source: `src/llama-quant.cpp:792-834`, names and bpw from
`src/llama-model-loader.cpp:30-74`)

### `ggml_ftype` enum (older ggml-level)

Defined at `ggml/include/ggml.h:448-476`. A subset that mirrors
`llama_ftype` but without the fine-grained K-quant variants (no S/M/L
distinction). Maps to `ggml_type` via `ggml_ftype_to_ggml_type()` at
`ggml/src/ggml.c:1400-1436`.

### Block Structures

Each quantized type has a corresponding C struct in
`ggml/src/ggml-common.h:280-450`. Key examples:

| Struct | Defined at | Fields |
|---|---|---|
| `block_q2_K` | `ggml-common.h:288-298` | `d`, `dmin` (half), `scales[16]`, `qs[64]` |
| `block_q3_K` | `ggml-common.h:305-311` | `d` (half), `hmask[32]`, `qs[64]`, `scales[12]` |
| `block_q4_K` | `ggml-common.h:317-328` | `d`, `dmin` (half), `scales[K_SCALE_SIZE]`, `qs[128]` |
| `block_q5_K` | `ggml-common.h:334-346` | `d`, `dmin` (half), `scales[...]`, `qh[32]`, `qs[128]` |
| `block_q6_K` | `ggml-common.h:352-358` | `d` (half), `ql[128]`, `qh[64]`, `scales[16]` |
| `block_iq2_xxs` | `ggml-common.h:371-375` | `d` (half), `qs[32]` (uint16) |
| `block_iq3_xxs` | `ggml-common.h:397-401` | `d` (half), `qs[96]` |
| `block_iq3_s` | `ggml-common.h:405-412` | `d` (half), `qs[64]`, `qh[8]`, `signs[32]`, `scales[4]` |
| `block_iq1_s` | `ggml-common.h:415-420` | `d` (half), `qs[32]`, `qh[8]` (uint16) |
| `block_iq1_m` | `ggml-common.h:423-428` | `qs[32]`, `qh[16]`, `scales[8]` (no d) |
| `block_iq4_nl` | `ggml-common.h:438-442` | `d` (half), `qs[16]` |
| `block_iq4_xs` | `ggml-common.h:444-450` | `d` (half), `scales_h` (uint16), `scales_l[4]`, `qs[128]` |

All K-quant blocks use super-block size `QK_K = 256`
(`ggml-common.h:89`). IQ4_NL uses `QK4_NL = 32` (`ggml-common.h:437`).

## Quantization Pipeline (`src/llama-quant.cpp`)

### Entry point: `llama_model_quantize`

Public API at `include/llama.h:639-642`, implementation at
`src/llama-quant.cpp:1309-1321`. Returns 0 on success, 1 on exception.
Delegates immediately to `llama_model_quantize_impl`.

### Main function: `llama_model_quantize_impl`

`src/llama-quant.cpp:857-1282`. Synchronous, single-threaded dispatch
(workers are spawned per-tensor). Steps:

1. **Resolve ftype** (line 858). Gets `default_type` via
   `llama_ftype_get_default_type()` (line 866).
2. **Load model** via `llama_model_loader` (line 881-883). Mmap unless macOS.
3. **Load hparams and stats** (line 893-894).
4. **Create quantize state** (`quantize_state_impl`, line 896).
5. **Load importance matrix** if provided (lines 903-921). Validates no NaN/Inf.
6. **Initialize output GGUF** context (line 924). Copies KV pairs, sets
   quantization version and file type (lines 934-936).
7. **Build tensor list** (lines 964-991). Optionally prune layers, remap
   names, sort by split.
8. **Preliminary loop** (lines 1021-1056). For each tensor:
   - `tensor_allows_quantization()` (line 1031) -- name-based exclusions
   - `llama_tensor_get_type()` (line 1034) -- selects target type
   - `tensor_requires_imatrix()` (line 1039) -- checks if imatrix needed
   - Validates imatrix availability (lines 1041-1055)
9. **Main loop** (lines 1115-1263). For each tensor:
   - Load data via `ml.load_data_for()` (line 1134)
   - If no change, copy directly (line 1174-1175)
   - If quantizing: dequantize to f32 via `llama_tensor_dequantize_impl()`
     (line 1218), then quantize via `llama_tensor_quantize_impl()` (line
     1247, uses `ggml_quantize_chunk`)
   - Write to output file (line 1260)

### Tensor type selection: `llama_tensor_get_type`

`src/llama-quant.cpp:661-703`. Outer wrapper that:
1. Skips non-quantizable tensors (return original type).
2. Checks user overrides for token_embedding/output types.
3. Calls `llama_tensor_get_type_impl()` (line 695) for per-category logic.
4. Calls `tensor_type_fallback()` (line 699) for shape incompatibility.

### Per-category type selection: `llama_tensor_get_type_impl`

`src/llama-quant.cpp:411-658`. A large switch-like function that examines
category (attention V, Q, K, FFN down/gate/up, output) and adjusts the target
type based on `ftype`, `n_gqa`, `n_expert`, layer index, and imatrix
availability. This is where K-quant mixture variants (S/M/L) are implemented:
different tensor categories get different effective bit widths even though they
share the same base `ggml_type`.

### Fallback: `tensor_type_fallback`

`src/llama-quant.cpp:362-408`. When tensor shape (`ncols`) is not divisible by
the target type's block size, falls back to a compatible type (e.g.,
`GGML_TYPE_IQ2_XXS` -> `GGML_TYPE_IQ4_NL`, `GGML_TYPE_Q4_K` ->
`GGML_TYPE_Q5_0`). If the fallback also fails, uses `GGML_TYPE_F16`.

### Dequantization: `llama_tensor_dequantize_impl`

`src/llama-quant.cpp:212-282`. Used during quantization to convert loaded
tensors back to f32 before re-quantizing. Dispatches to
`ggml_type_traits.to_float` for quantized types, or to
`ggml_fp16_to_fp32_row`/`ggml_bf16_to_fp32_row` for float types. Multi-threaded
when `nthread >= 2`.

### Quantization: `llama_tensor_quantize_impl`

`src/llama-quant.cpp:709-761`. Multi-threaded wrapper around
`ggml_quantize_chunk`. Each thread grabs a chunk via atomic counter, calls
`ggml_quantize_chunk()`, then validates via `ggml_validate_row_data()`.

### `ggml_quantize_chunk`

`ggml/src/ggml.c:7706-7780`. Big switch dispatching to type-specific
`quantize_*` functions (e.g., `quantize_q4_K`, `quantize_iq2_xxs`,
`quantize_iq4_nl`). These functions reside in `ggml/src/ggml-quants.c`.

### `ggml_quantize_init`

`ggml/src/ggml.c:7666-7682`. Lazy-once initialization. Only needed for IQ
types that use pre-computed lookup tables/grids:
`GGML_TYPE_IQ2_XXS`, `GGML_TYPE_IQ2_XS`, `GGML_TYPE_IQ2_S`,
`GGML_TYPE_IQ1_S`, `GGML_TYPE_IQ1_M` (all call `iq2xs_init_impl`),
`GGML_TYPE_IQ3_XXS` (calls `iq3xs_init_impl(256)`),
`GGML_TYPE_IQ3_S` (calls `iq3xs_init_impl(512)`).

### `ggml_quantize_requires_imatrix`

`ggml/src/ggml.c:7698-7704`. Returns `true` for `GGML_TYPE_IQ2_XXS`,
`GGML_TYPE_IQ2_XS`, and `GGML_TYPE_IQ1_S`. (Note: `IQ1_M` is
commented out at line 7703, and `tensor_requires_imatrix` at
`src/llama-quant.cpp:767-786` also includes `IQ3_XXS`, `IQ2_S`, and
`Q2_K` for `LLAMA_FTYPE_MOSTLY_Q2_K_S`.)

## K-Quant Formats

K-quants (Q2_K through Q6_K) divide a super-block of 256 values (`QK_K`)
into sub-blocks (typically 16 elements each for Q2_K/Q6_K, 32 for Q4_K/Q5_K).
Each super-block stores:

- A super-block scale factor `d` (and optional `dmin` for asymmetric types)
- Sub-block scales, themselves quantized (4-bit for Q2_K, 6-bit for Q4_K/Q5_K,
  8-bit for Q6_K)
- Quantized weights (2--6 bits per weight)

Block structs at `ggml/src/ggml-common.h:288-358`.

The S/M/L variants (`LLAMA_FTYPE_MOSTLY_Q3_K_S`/`M`/`L`, etc.) all use the
same underlying `GGML_TYPE_Q3_K` but differ in which per-tensor overrides
`llama_tensor_get_type_impl` applies -- for example, attention V tensors get
Q5_K instead of Q4_K in Q3_K_L mode (`src/llama-quant.cpp:530`), and early
and late FFN down layers get Q5_K/Q6_K in Q4_K_M and Q5_K_M modes
(`src/llama-quant.cpp:590-601`).

## IQ-Family Formats

IQ (Importance-aware Quantization) types use importance matrices to guide
non-uniform quantization. They rely on pre-computed codebooks (lookup tables)
initialized by `ggml_quantize_init()`.

Block structs at `ggml/src/ggml-common.h:371-450`.

| Type | bpw | Block size | Notes |
|---|---|---|---|
| IQ1_S | 1.5625 | 256 | 2-bit index into 3-element codebook |
| IQ1_M | 1.75 | 256 | No explicit `d`, scales encoded inline |
| IQ2_XXS | 2.0625 | 256 | 16-bit codes for 32 weights |
| IQ2_XS | 2.3125 | 256 | Sub-block scales |
| IQ2_S | 2.5625 | 256 | Full sub-block structure |
| IQ3_XXS | 3.0625 | 256 | True 3-bit (almost) |
| IQ3_S | 3.4375 | 256 | Includes sign bits |
| IQ4_NL | 4.5 | 32 | Non-linear, small block size |
| IQ4_XS | 4.25 | 256 | Cross-block scale sharing |

## How `llama-model-loader` Applies Quantization at Load Time

The model loader (`src/llama-model-loader.cpp`) does **not** dequantize during
loading. It memory-maps or reads the raw quantized bytes via
`load_data_for()` (`src/llama-model-loader.cpp:1385-1406`). The tensor's
`type` field stores the quantized `ggml_type`. At inference time, the ggml
backend calls `ggml_type_traits.to_float` (function pointer at
`ggml/include/ggml.h:2823`) to dequantize blocks on the fly during matrix
multiplication.

Key: `ggml_type_traits` struct at `ggml/include/ggml.h:2817-2825` contains
`to_float` and `from_float_ref`. For each quantized type, these are populated
in `ggml-quants.c` (e.g., `dequantize_row_q4_K`).

## Public API

### `llama_model_quantize`

`include/llama.h:639-642`, defined at `src/llama-quant.cpp:1309-1321`.
Parameters struct `llama_model_quantize_params` at
`include/llama.h:409-424`. Defaults at `src/llama-quant.cpp:1288-1307`.

```c
uint32_t llama_model_quantize(
    const char * fname_inp,
    const char * fname_out,
    const llama_model_quantize_params * params);
```

Key fields: `nthread`, `ftype`, `output_tensor_type`, `token_embedding_type`,
`allow_requantize`, `quantize_output_tensor`, `only_copy`, `pure`,
`keep_split`, `dry_run`, `imatrix`, `kv_overrides`, `tt_overrides`,
`prune_layers`.

### `llama_model_quantize_default_params`

`include/llama.h:449`, defined at `src/llama-quant.cpp:1288-1307`. Default
ftype is `LLAMA_FTYPE_MOSTLY_Q8_0`.

### `llama_ftype_get_default_type`

`src/llama-ext.h:20`, defined at `src/llama-quant.cpp:792-834`. Maps
`llama_ftype` to `ggml_type`.

### `llama_ftype_name` (internal)

`static llama_model_ftype_name()` at `src/llama-model-loader.cpp:30-74`.
Not exposed as public API. The `llama_model_loader::ftype_name()` method
(`src/llama-model-loader.cpp:1692-1693`) calls it. No `llama_ftype_name`
public function exists in the current codebase.

### Extension API (`src/llama-ext.h`)

`llama_quant_init()` (line 24), `llama_quant_free()` (line 28),
`llama_quant_model_from_metadata()` (line 45),
`llama_quant_tensor_allows_quantization()` (line 48),
`llama_quant_compute_types()` (line 55). Used by the type-selection test
(`tests/test-quant-type-selection.cpp`).

### Quantization CLI

`tools/quantize/quantize.cpp:391-654`. Parses arguments, calls
`llama_model_quantize()`.

## Touch Points: Which Other Features Call Into Quantization

1. **Model conversion** (`tools/convert-*.py`): Converts source model formats
   to GGUF; the quantizer takes the resulting GGUF file as input.
2. **Inference runtime** (`src/llama.cpp`, `src/llama-model.cpp`): Loads
   quantized models via `llama_model_loader`, which reads the quantized
   tensor data as-is. No dequant happens until compute.
3. **KV cache** (`src/llama-kv-cache.cpp:1854`): For K-cache quantization,
   dequantizes to f32 -> applies RoPE -> quantizes back. Uses
   `ggml_type_traits.to_float` and `ggml_quantize_chunk`.
4. **GGML backend** (`ggml/src/ggml-quants.c`): Implements all
   quantize/dequantize kernels for each type. Populates `ggml_type_traits`.
5. **ggml_quantize_init/free**: Memory management for IQ lookup tables at
   `ggml/src/ggml.c:7666-7696`. Called automatically by `ggml_quantize_chunk`.
6. **GGML Vulkan backend** (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:13785`):
   Calls `ggml_quantize_chunk` for on-device quantization.

## Failure Modes

1. **Missing imatrix**: Low-bit IQ types (IQ2_XXS, IQ2_XS, IQ1_S) require an
   importance matrix. `tensor_requires_imatrix()` at
   `src/llama-quant.cpp:767-786` checks this, and the main pipeline throws
   `"this quantization requires an imatrix!"` at line 1053 or
   `"Missing importance matrix"` at line 1208.
2. **Shape mismatch**: `tensor_type_fallback()` at
   `src/llama-quant.cpp:362-408` warns and falls back when `ncols` is not
   divisible by block size. If no valid fallback exists for a type, throws
   `"no tensor type fallback is defined"` (line 391). If final fallback
   F16 also fails, aborts (line 403).
3. **Invalid imatrix values**: Non-finite values in the importance matrix cause
   `"imatrix contains non-finite value"` at line 915.
4. **Requantization disabled**: If `allow_requantize` is false and a tensor is
   already quantized, throws `"requantizing from type ... is disabled"` at
   line 1216.
5. **Invalid ftype**: If `llama_ftype_get_default_type()` returns
   `GGML_TYPE_COUNT`, throws `"invalid output file type"` at line 868.
6. **Dequantization unsupported**: If a quantized type has no `to_float`
   callback, throws `"type ... unsupported for integer quantization"` at
   line 224.
7. **Validation failure**: `ggml_validate_row_data()` in
   `llama_tensor_quantize_impl()` throws `"quantized data validation failed"`
   at lines 714 and 757 if quantized data contains NaN/Inf.
