# Model Conversion

## Purpose

llama.cpp converts models from HuggingFace (transformers / safetensors / PyTorch), legacy GGML formats, and PEFT LoRA adapters into the GGUF binary format for efficient inference. The conversion pipeline reads architecture metadata from `config.json`, iterates over weight files, maps HF tensor names to GGUF tensor names, applies architecture-specific transformations (permute, fuse, split, dequantize), optionally quantizes, and writes a GGUF file.

## Entry Points

### CLI entry points (script registrations)

Three console scripts are registered in `pyproject.toml:29-31`:

| Script | Entry function | Purpose |
|---|---|---|
| `llama-convert-hf-to-gguf` | `convert_hf_to_gguf:main` | HuggingFace / safetensors / PyTorch -> GGUF |
| `llama-convert-lora-to-gguf` | `convert_lora_to_gguf:main` | PEFT LoRA adapter -> GGUF adapter file |
| `llama-convert-llama-ggml-to-gguf` | `convert_llama_ggml_to_gguf:main` | Legacy GGML/GGMF/GGJT -> GGUF |

### Converter base class

`conversion/base.py:75` defines `ModelBase`, the abstract root of all converters. It holds shared state (hparams, tensor map, GGUF writer) and defines the pipeline methods `prepare_tensors()`, `prepare_metadata()`, and `write()`.

Two intermediate mixin classes:

- `TextModel` (`conversion/base.py:1100`) — adds `hf_arch` detection, `block_count`, `tensor_map` initialization, `set_gguf_parameters()` with common LoRA/RoPE/expert KV entries, and helper vocab methods.
- `MmprojModel` — for multimodal projector export (not shown separately but referenced in `conversion/__init__.py:252` and registered via `ModelBase.register`).

### Per-architecture modules

Each file under `conversion/` registers one or more HF architecture string -> converter class mappings. The full list of 80+ modules is in `conversion/__init__.py TEXT_MODEL_MAP` (line 19) and `MMPROJ_MODEL_MAP` (line 252). Each module is loaded lazily by `get_model_class()` (`conversion/__init__.py:340`).

### Legacy GGML converter

`convert_llama_ggml_to_gguf.py:414` (`main()`) parses the legacy binary format (magic bytes `lmgg`/`fmgg`/`tjgg`), instantiates `GGMLModel`, then wraps it in `GGMLToGGUF` which calls `GGUFWriter` directly. This path is best-effort per the warning at line 418.

## Data Flow

### HF model download -> weight loading -> tensor mapping -> GGUFWriter output

**High-level flow** (from `convert_hf_to_gguf.py:171` `main()`):

```
args parsed (line 172)
  -> if --remote: snapshot_download from HF Hub (line 191)
  -> ModelBase.load_hparams(dir_model) reads config.json (base.py:1042)
  -> get_model_architecture(hparams) looks up "architectures" in config (base.py:???)
  -> get_model_class(arch) returns registered converter class (__init__.py:340)
  -> model_instance = converter_class(dir_model, ftype, fname_out, ...) (line 272)
     |  -> __init__ calls self.index_tensors() (base.py:149)
     |  -> creates GGUFWriter (base.py:178)
  -> model_instance.write() or write_vocab() (line 288/292)
```

**Weight indexing** (`base.py:198` `index_tensors()`):

1. If `remote_hf_model_id` is set, list tensors via `gguf.utility.SafetensorRemote.get_list_tensors_hf_model` (line 205).
2. Otherwise, enumerate `.safetensors` or `.pytorch_model.bin` shards in `dir_model` (lines 215-218).
3. For each shard, iterate keys, apply `filter_tensors()` (line 269), store `(name, data_gen)` pairs in `self.model_tensors`.
4. If an index file (`model.safetensors.index.json` / `pytorch_model.bin.index.json`) exists, validate tensor coverage (lines 274-285).

**Weight loading strategies**:

- **Local safetensors**: `gguf.utility.SafetensorsLocal` context manager (line 247); lazy via `LazyTorchTensor.from_local_tensor` (line 259), eager via `mmap_bytes()` then `torch.from_numpy().view()` (line 262).
- **Local PyTorch**: `torch.load(..., mmap=True, weights_only=True)` (line 249); lazy via `LazyTorchTensor.from_eager` (line 266), eager returns tensor directly (line 268).
- **Remote safetensors**: `SafetensorRemote.get_list_tensors_hf_model` (line 205), lazy via `LazyTorchTensor.from_remote_tensor` (line 207).

**Dequantization**: `dequant_model()` (base.py:305) handles quant_method values: `bitnet`, `fp8`, `gptq`, `compressed-tensors`, `modelopt`. Replaces packed/quantized weight tensors with dequantized float tensors.

**Tensor mapping and output** (`base.py:785` `prepare_tensors()`):

1. NVFP4 repacking if detected (line 830).
2. Dequantization (line 855).
3. Iterate `generate_extra_tensors()` then `get_tensors()` (line 863).
4. For each tensor, `modify_tensors()` maps HF name -> GGUF name and applies arch-specific transforms (base.py:615).
5. `tensor_force_quant()` decides quantization type (base.py:642).
6. `gguf.quants.quantize()` does the actual quantization (line 965), with fallback to F16 on error (line 966-969).
7. `gguf_writer.add_tensor()` stores the numpy buffer (line 979).

**Write pipeline** (`base.py:1022` `write()`):

```
prepare_tensors()   -> populates gguf_writer.tensors + kv_data
prepare_metadata()  -> adds metadata KV pairs via gguf_writer
gguf_writer.write_header_to_file()   -> magic + version + tensor count + kv count (gguf_writer.py:214)
gguf_writer.write_kv_data_to_file()  -> key-value metadata (gguf_writer.py:237)
gguf_writer.write_tensors_to_file()  -> tensor info + tensor data (gguf_writer.py:438)
gguf_writer.close()
```

## Key Types and Protocols

### `ModelBase` (`conversion/base.py:75`)

Central class with these key attributes:

- `model_arch: gguf.MODEL_ARCH` (line 100) — subclass must set
- `block_count: int` (line 103) — initialized in `TextModel.__init__`
- `tensor_map: gguf.TensorNameMap` (line 104) — maps HF names to GGUF names
- `gguf_writer: gguf.GGUFWriter` (line 91) — output writer
- `model_tensors: dict[str, Callable[[], Tensor]]` (line 90) — lazy tensor generators
- `hparams: dict[str, Any]` (line 89) — loaded from `config.json`

Key protocols overridden by subclasses:

| Method | Line | Purpose |
|---|---|---|
| `filter_tensors()` | 569 | Filter/reject tensors before indexing |
| `modify_tensors()` | 615 | Map HF->GGUF names, permute, fuse, split experts |
| `set_gguf_parameters()` | 612 | Write architecture-specific KV metadata |
| `set_vocab()` | (TextModel:1162) | Write tokenizer data |
| `generate_extra_tensors()` | 650 | Compute derived tensors (rope_freqs, etc.) |
| `tensor_force_quant()` | 642 | Override quantization type per-tensor |

Registration decorator (`base.py:1075`):

```python
@ModelBase.register("LlamaForCausalLM", "MistralForCausalLM", ...)
class LlamaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
```

### `GGUFWriter` (`gguf-py/gguf/gguf_writer.py:65`)

Writes binary GGUF files. Key state:

- `fout: list[BufferedWriter]` — file handles (multiple for shards)
- `tensors: list[dict[str, TensorInfo]]` — tensor metadata per shard
- `kv_data: list[dict[str, GGUFValue]]` — KV metadata per shard
- `arch: str` — architecture name for KV key formatting

Key methods:

| Method | Line | Purpose |
|---|---|---|
| `add_tensor()` | 375 | Register a tensor (name + numpy array) |
| `add_tensor_info()` | 330 | Low-level tensor metadata registration |
| `add_key_value()` | 277 | Register a KV pair |
| `write_header_to_file()` | 214 | Write GGUF magic + version + counts |
| `write_kv_data_to_file()` | 237 | Write KV metadata |
| `write_tensors_to_file()` | 438 | Write tensor metadata + weights |
| `add_architecture()` | 499 | Write `general.architecture` |

Supports **sharding** (split files) via `split_max_tensors` / `split_max_size` (line 86-101).

### Per-architecture converter modules

The pattern across all converter modules:

1. Import `ModelBase`, `TextModel`, `gguf` from `conversion.base`.
2. Define a class subclassing `TextModel` (or `MmprojModel` for vision).
3. Set `model_arch = gguf.MODEL_ARCH.<ARCH>`.
4. Decorate with `@ModelBase.register("HFArchName1", "HFArchName2", ...)`.
5. Override `set_gguf_parameters()`, `modify_tensors()`, `set_vocab()`, and optionally `filter_tensors()` / `generate_extra_tensors()`.

**Example: `conversion/llama.py`**

- `LlamaModel` (line 31) registered for `LLaMAForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM`, `MixtralForCausalLM`, etc.
- `undo_permute = True` — applies Q/K RoPE permutation (`modify_tensors`: line 244-248).
- Expert stacking in `modify_tensors()` (line 251-278): collects individual expert tensors into per-layer dicts, then `torch.stack()` into 3D tensors.
- `generate_extra_tensors()` computes llama3-style rope frequencies (line 282-310).
- Subclasses: `ArceeModel` (line 348), `Llama4Model` (line 360), `SmolLM3Model` (line 406), `ApertusModel` (line 411), `LlamaEmbedNemotronModel` (line 401).

**Example: `conversion/gemma.py`**

- `GemmaModel` (line 17) for `GemmaForCausalLM` -> `GEMMA`.
- Norm shift: `data_torch + 1` applied in `modify_tensors()` (line 64) per the Gemma RMSNorm convention.
- `Gemma2Model` (line 71) adds `attn_logit_softcapping`, `final_logit_softcapping`, `sliding_window` (lines 92-98).
- `Gemma3Model` (line 121) adds multi-window attention parameters.
- `Gemma3VisionModel` (line 251) handles multimodal projector export.
- `Gemma4Model` (line 618) adds per-layer head counts, `sliding_window_pattern`, expert FFN lengths, etc.
- `Gemma4AssistantModel` (line 789) handles speculative decoding assistant models.
- `Gemma4UnifiedVisionAudioModel` (line 887) handles vision+audio encoder export.

**Example: `conversion/qwen.py`**

- `QwenModel` (line 14) for `QWenLMHeadModel` -> `QWEN`, with Qwen-specific BPE vocab.
- `Qwen2Model` (line 52) for `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, etc. -> `QWEN2`.
- `Qwen2MoeModel` (line 72) handles MoE expert tensor reshaping (`modify_tensors`: lines 86-100).
- `_Qwen35MtpMixin` (referenced in `convert_hf_to_gguf.py:262`) adds MTP (multi-token prediction) head splitting.

### LoRA Conversion

`convert_lora_to_gguf.py` converts PEFT LoRA adapters. The key approach:

1. Load adapter weights from `adapter_model.safetensors` or `.bin` (lines 363-370).
2. Load `adapter_config.json` for hyperparams (line 373).
3. Load base model config via `ModelBase.load_hparams()` or `AutoConfig.from_pretrained()` (lines 377-396).
4. Create `LoraModel` — a subclass of the detected model converter (line 407) that overrides:
   - `set_type()` — writes `GGUFType.ADAPTER` + adapter type string (lines 422-424).
   - `set_gguf_parameters()` — writes `lora_alpha` and optional `alora_invocation_tokens` (lines 426-451).
   - `generate_extra_tensors()` — returns empty (line 455), adapters don't need rope_freqs, etc.
   - `get_tensors()` — pairs `lora_A` / `lora_B` tensors into `LoraTorchTensor` objects (lines 457-499).
   - `modify_tensors()` — splits `LoraTorchTensor` back into `.lora_a` / `.lora_b` weight tensors (lines 501-526).
5. `LoraTorchTensor` (line 40) is a proxy that stores `(A, B)` factorized weights and supports reshape/transpose/split operations during tensor mapping.

### Legacy GGML-to-GGUF

`convert_llama_ggml_to_gguf.py` handles the older `GGML`/`GGMF`/`GGJT` binary formats (magic checks at line 146-166). The `GGMLToGGUF` class (line 203) manually maps tensors using `gguf.get_tensor_name_map()` (line 226), then writes via `GGUFWriter`.

## Touch Points

### GGUF format (doc item 00)

The GGUF binary format is defined by `GGUFWriter` in `gguf-py/gguf/gguf_writer.py`:
- Header: magic (`GGUF`), version, tensor count, KV count (line 230-233).
- KV data: typed key-value pairs (line 242-249).
- Tensor info: name, n_dims, shape, dtype, offset (line 259-271).
- Tensor data: raw weight bytes (line 466-474).
- Alignment: configurable via `data_alignment` (default `GGUF_DEFAULT_ALIGNMENT` = 32, line 94).
- Sharding: split files with names like `<stem>-00001-of-00003.gguf` (line 38).

### GGML types

- `gguf.GGMLQuantizationType` enum used in `gguf_writer.py:23` and throughout `base.py` for tensor quantization.
- `GGMLQuantizationType` values encode the quantization scheme (F32, F16, BF16, Q8_0, Q4_0, etc.).
- `gguf.GGML_QUANT_SIZES` dict maps type -> `(blksize, tysize)` used in `convert_llama_ggml_to_gguf.py:117`.
- Quantization is applied by `gguf.quants.quantize()` at `base.py:965`.

### C++ model saver for round-tripping

`src/llama-model-saver.cpp` writes in-memory `llama_model` structs back to GGUF:

- `llama_model_saver` class (`llama-model-saver.h:12`) wraps a `gguf_context`.
- `add_kv_from_model()` (line 145) serializes all `llama_hparams`, `llama_vocab`, and metadata from the loaded model.
- `add_tensors_from_model()` (line 385) iterates `model->layers` and writes every `ggml_tensor*`.
- `save()` calls `gguf_write_to_file()` (line 415).
- Some architectures are **excluded** from round-trip: `llama_model_saver_supports_arch()` (`llama-model-saver.cpp:15`) returns false for `PLAMO3`, `GEMMA3`, `GEMMA3N`, `COHERE2`, `BITNET`, `T5`, `EXAONE_MOE`, `AFMOE`, `APERTUS`, `MIMO2`, `STEP35`, `MELLUM`.

## Failure Modes

### Unsupported architectures

- `convert_hf_to_gguf.py:243-245` catches `NotImplementedError` from `get_model_class()`, logs "Model {arch} is not supported", and exits with code 1.
- `conversion/__init__.py:343-344` raises `NotImplementedError(f"Architecture {name!r} not supported!")` if the HF arch name is not in `TEXT_MODEL_MAP` or `MMPROJ_MODEL_MAP`.
- Same pattern in `convert_lora_to_gguf.py:403-405`.
- `--print-supported-models` flag (`convert_hf_to_gguf.py:109-110`) lists all registered architectures.

### Missing keys / tensor errors

- Tensor name mapping failure: `base.py:609` raises `ValueError(f"Can not map tensor {name!r}")` if `tensor_map.get_name()` returns None.
- Missing shard files: `base.py:280-281` raises `ValueError` with list of missing files and missing tensors when weight map and actual files don't match.
- Missing hparam key: `base.py:196` raises `KeyError(f"could not find any of: {keys}")` from `find_hparam()`.
- Unknown GGML format magic: `convert_llama_ggml_to_gguf.py:166` raises `ValueError(f"Unexpected file magic ...")`.
- `convert_lora_to_gguf.py:479` exits with error when a tensor is neither `.lora_A` nor `.lora_B`.

### Conversion errors

- Quantization fallback: `base.py:966-969` catches `gguf.QuantError`, logs warning, falls back to F16.
- `tensor_force_quant()` mismatch: `base.py:962` raises `ValueError(f"Unknown file type: {self.ftype.name}")`.
- GGUF already: `convert_llama_ggml_to_gguf.py:148` rejects files already in GGUF format.
- Quantized legacy files: `convert_llama_ggml_to_gguf.py:168-178` raises `ValueError` for incompatible quantization formats in old GGML files.
- Mistral format requires `mistral-common`: `convert_hf_to_gguf.py:230-231` raises `ImportError` with install instructions.
- LoRA lm_head issue: `convert_lora_to_gguf.py:507-508` raises `ValueError` when adapter targets lm_head but base model ties embeddings.
- Temp file + split conflict: `convert_hf_to_gguf.py:215-217` rejects `--use-temp-file` with `--split-max-*`.
- MTP mutual exclusivity: `convert_hf_to_gguf.py:257-259` rejects `--mtp` + `--no-mtp` together.
- Missing target model dir for EAGLE-3: `conversion/llama.py:58-61` raises `ValueError`.

## Citation Summary

| File | Lines Cited |
|---|---|
| `pyproject.toml` | 29-31 |
| `convert_hf_to_gguf.py` | 48-168, 171-294 |
| `convert_lora_to_gguf.py` | 279-323, 337-546 |
| `convert_llama_ggml_to_gguf.py` | 27-45, 134-200, 203-358, 388-450 |
| `conversion/__init__.py` | 19-308, 315-358 |
| `conversion/base.py` | 75-107, 116-182, 198-287, 305-566, 568-651, 785-979, 981-1028, 1042-1097, 1100-1303 |
| `conversion/llama.py` | 17-444 |
| `conversion/gemma.py` | 16-947 |
| `conversion/qwen.py` | 13-100 |
| `gguf-py/gguf/gguf_writer.py` | 41-103, 113-162, 185-271, 277-324, 326-400, 406-503 |
| `src/llama-model-saver.cpp` | 15-35, 37-420 |
| `src/llama-model-saver.h` | 10-44 |

**Total file:line citations: ~190**

## Evidence Quality Concerns

- The `get_model_architecture()` function referenced in `convert_hf_to_gguf.py:22` and called at line 239 is not defined in `conversion/base.py` or `conversion/__init__.py` exports — it may be imported from elsewhere or generated dynamically. Its exact location was not verified.
- `conversion/base.py` is very long (~2000 lines); the `prepare_tensors` method (line 785) and `set_gguf_parameters` (TextModel, line 1193) are the primary extension points, but architecture-specific overrides in individual modules may skip or override significant portions.
- `MmprojModel` and `LazyTorchTensor` are referenced but their full implementations were not read; they are documented based on their usage patterns.
