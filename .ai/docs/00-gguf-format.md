# GGUF File Format

## Purpose

GGUF (GGML Universal Format) is a binary serialization format for neural network
model weights and metadata, designed to replace the earlier GGML and GGJT formats.
It provides a self-describing container with typed key-value metadata followed by
contiguous aligned tensor data.

The format's header comment describes the full layout at `ggml/include/gguf.h:1-31`.
The module maintainer is Johannes Gassler (`ggml/include/gguf.h:32`).

## Entry Points

### C/C++ Reading

| Function | File | Role |
|---|---|---|
| `gguf_init_from_file` | `ggml/src/gguf.cpp:979` | Opens a GGUF file by path; wraps `gguf_init_from_file_ptr` |
| `gguf_init_from_file_ptr` | `ggml/src/gguf.cpp:928` | Reads from an open `FILE*` |
| `gguf_init_from_buffer` | `ggml/src/gguf.cpp:966` | Reads from an in-memory buffer |
| `gguf_init_from_callback` | `ggml/src/gguf.cpp:896` | Reads via a user-supplied callback (`gguf_reader_callback_t`, `ggml/include/gguf.h:80`) |
| `gguf_init_from_reader` | `ggml/src/gguf.cpp:451` | Internal: all init paths converge here; validates magic, version, KV pairs, tensor info |

### C/C++ Writing

| Function | File | Role |
|---|---|---|
| `gguf_write_to_file` | `ggml/src/gguf.cpp:1660` | Writes an entire `gguf_context` to a file path |
| `gguf_write_to_file_ptr` | `ggml/src/gguf.cpp:1647` | Writes to an open `FILE*` |
| `gguf_write_to_buf` | `ggml/src/gguf.cpp:1642` | Serializes to a `std::vector<int8_t>` buffer |
| `gguf_get_meta_size` / `gguf_get_meta_data` | `ggml/src/gguf.cpp:1677,1684` | Serialize only the metadata section (no tensor data) |

The writer template `gguf_write_out` at `ggml/src/gguf.cpp:1603-1640` is the
shared serialization function used by all three write entry points.

### Python (gguf-py)

| Class / Function | File | Role |
|---|---|---|
| `GGUFWriter.__init__` | `gguf-py/gguf/gguf_writer.py:86` | Creates a writer for a target path/arch |
| `GGUFWriter.write_header_to_file` | `gguf-py/gguf/gguf_writer.py:214` | Writes magic, version, tensor count, KV count |
| `GGUFWriter.write_kv_data_to_file` | `gguf-py/gguf/gguf_writer.py:237` | Writes all key-value metadata pairs |
| `GGUFWriter.write_ti_data_to_file` | `gguf-py/gguf/gguf_writer.py:254` | Writes tensor info (name, shape, dtype, offset) |
| `GGUFWriter.write_tensors_to_file` | `gguf-py/gguf/gguf_writer.py:438` | Writes the actual tensor data blob |
| `GGUFWriter.add_tensor` | `gguf-py/gguf/gguf_writer.py:375` | Registers a numpy tensor for writing |
| `GGUFReader.__init__` | `gguf-py/gguf/gguf_reader.py:132` | Memory-maps a GGUF file and parses all fields + tensors |
| `convert_hf_to_gguf.py` | `convert_hf_to_gguf.py:1` | CLI entry point for HuggingFace -> GGUF conversion |

## Binary Layout

The on-disk structure is documented at `ggml/include/gguf.h:1-31`:

```
[0x00] Magic        "GGUF" (4 bytes)
[0x04] Version      uint32_t  (currently 3, see GGUF_VERSION at ggml/include/gguf.h:42)
[0x08] n_tensors    int64_t
[0x10] n_kv         int64_t
--- KV pairs ---
  for each of n_kv:
    key              string  (uint64_t length + UTF-8 bytes)
    value_type       int32_t (GGUF_TYPE_* enum)
    if value_type == GGUF_TYPE_ARRAY:
      array_type     int32_t (GGUF_TYPE_* for element type)
      array_length   uint64_t
    value            binary representation of the value
--- Tensor info ---
  for each of n_tensors:
    name             string
    n_dims           uint32_t
    dims[]           int64_t[n_dims]
    tensor_type      int32_t (ggml_type enum)
    data_offset      uint64_t (offset from start of data blob)
--- Tensor data blob (aligned) ---
  binary blob containing all tensor data
  padded to ctx->alignment (default 32, see GGUF_DEFAULT_ALIGNMENT at ggml/include/gguf.h:46)
```

Strings are serialized as `uint64_t length` followed by the raw bytes without
null terminator (`ggml/include/gguf.h:26`). All enums are stored as `int32_t`
(`ggml/include/gguf.h:27`). Bools are `int8_t` (`ggml/include/gguf.h:28`).

The special KV key `"general.alignment"` (`GGUF_KEY_GENERAL_ALIGNMENT`,
`ggml/include/gguf.h:44`) controls alignment; if absent,
`GGUF_DEFAULT_ALIGNMENT = 32` is used.

### Endianness

At load time, the reader detects endianness mismatches by checking whether the
version field has the high 16 bits zeroed out (indicating byte-swapped data):

- `ggml/src/gguf.cpp:497-499`: version & 0x0000FFFF == 0 triggers a "host vs.
  model endianness mismatch" error.
- `gguf-py/gguf/gguf_reader.py:143-148`: The Python reader checks `temp_version[0] & 65535 == 0`,
  and if so, switches `byte_order` to `'S'` (swapped).

## Key-Value Metadata System

### C++ Type System

The `gguf_type` enum (`ggml/include/gguf.h:53-68`):

| Constant | Value | C++ Storage |
|---|---|---|
| `GGUF_TYPE_UINT8`   | 0 | `uint8_t` |
| `GGUF_TYPE_INT8`    | 1 | `int8_t` |
| `GGUF_TYPE_UINT16`  | 2 | `uint16_t` |
| `GGUF_TYPE_INT16`   | 3 | `int16_t` |
| `GGUF_TYPE_UINT32`  | 4 | `uint32_t` |
| `GGUF_TYPE_INT32`   | 5 | `int32_t` |
| `GGUF_TYPE_FLOAT32` | 6 | `float` |
| `GGUF_TYPE_BOOL`    | 7 | `bool` (stored as `int8_t`) |
| `GGUF_TYPE_STRING`  | 8 | `std::string` |
| `GGUF_TYPE_ARRAY`   | 9 | `std::vector<T>` |
| `GGUF_TYPE_UINT64`  | 10 | `uint64_t` |
| `GGUF_TYPE_INT64`   | 11 | `int64_t` |
| `GGUF_TYPE_FLOAT64` | 12 | `double` |

Size lookups are in `GGUF_TYPE_SIZE` (`ggml/src/gguf.cpp:92-106`). Type names
are mapped in `GGUF_TYPE_NAME` (`ggml/src/gguf.cpp:109-124`).

The C++ template `type_to_gguf_type` (`ggml/src/gguf.cpp:29-90`) maps C++ types
to their GGUF enum equivalents.

### Python Type System

`GGUFValueType` at `gguf-py/gguf/constants.py:4481-4494` mirrors the C enum.
`GGMLQuantizationType` at `gguf-py/gguf/constants.py:4383-4417` mirrors `ggml_type`
for tensor data. `GGUFEndian` at `gguf-py/gguf/constants.py:4476-4478` specifies
little (`0`) or big (`1`) endian.

### KV Storage (C++)

Each KV pair is stored in a `gguf_kv` struct (`ggml/src/gguf.cpp:131-210`):

- `key`: `std::string`
- `is_array`: `bool`
- `type`: `enum gguf_type` (the element type, or the array element type if `is_array`)
- `data`: `std::vector<int8_t>` (raw bytes for scalar/array values)
- `data_string`: `std::vector<std::string>` (for string values)

The `gguf_context` struct (`ggml/src/gguf.cpp:217-228`) holds:

- `version`: `uint32_t`
- `kv`: `std::vector<gguf_kv>`
- `info`: `std::vector<gguf_tensor_info>`
- `alignment`: `size_t`
- `offset`: `size_t` (file offset of tensor data blob start)
- `size`: `size_t` (total tensor data blob size)
- `data`: `void*` (pointer to loaded tensor data)

### `gguf_init_params`

Defined at `ggml/include/gguf.h:72-77`:

```c
struct gguf_init_params {
    bool no_alloc;
    struct ggml_context ** ctx;
};
```

- `no_alloc = true`: only create tensor metadata, do not load the binary blob
  (used by `llama-model-loader.cpp:543`).
- `no_alloc = false`: load tensor data into a new `ggml_context` and point each
  tensor's `data` member to its slice of the blob
  (`ggml/src/gguf.cpp:837-858`).
- `ctx`: if non-NULL, a `ggml_context` is created and tensors are allocated
  within it (`ggml/src/gguf.cpp:826-831`).

### Reading KV Pairs

Reading is done by `gguf_init_from_reader` (`ggml/src/gguf.cpp:451`):

1. Magic ("GGUF", 4 bytes) validated at lines 456-477.
2. Version read (uint32_t) and range-checked at lines 484-510; v1 is rejected
   (line 503), versions > `GGUF_VERSION` (3) are rejected (line 506).
3. `n_tensors` and `n_kv` as `int64_t` with overflow checks at lines 515-535.
4. For each of `n_kv` KV pairs (lines 544-607):
   - Read key string (line 552)
   - Check for duplicate keys (lines 560-564)
   - Read type enum (line 570)
   - If type is `GGUF_TYPE_ARRAY`, read element type and count (lines 571-575)
   - Dispatch to `gguf_read_emplace_helper<T>` template (lines 580-599)
5. Read alignment from kv metadata (lines 609-616), validate it is a power of 2.
6. For each of `n_tensors` tensor info records (lines 620-742):
   - Tensor name, duplicate check, length check (lines 625-653)
   - Number of dimensions (uint32_t, max `GGML_MAX_DIMS`), dimension sizes
     (int64_t), overflow checks (lines 657-691)
   - Tensor `ggml_type` (lines 697-733), validate it is in range, validate
     row-size divisible by block size, compute strides (`nb[0..3]`)
   - Tensor data offset (uint64_t, line 739)
7. Seek to aligned data start (line 752)
8. Optionally load tensor data blob and create `ggml_tensor` objects
   (lines 785-891).

### Writing KV Pairs

The function `gguf_write_out` (`ggml/src/gguf.cpp:1603-1640`):

1. Write 4-byte magic (lines 1609-1612)
2. Write version, n_tensors, n_kv (lines 1613-1615)
3. For each KV pair, call `gw.write(ctx->kv[i])` which dispatches through
   `gguf_writer_base::write(const gguf_kv&)` at `ggml/src/gguf.cpp:1459-1498`
4. Write tensor metadata via `write_tensor_meta` (lines 1623-1625,
   implementation at lines 1500-1511)
5. Pad to alignment (line 1628)
6. If not `only_meta`, write tensor data (lines 1637-1639)

Two writer backends exist:
- `gguf_writer_buf` (`ggml/src/gguf.cpp:1522-1557`): writes to `std::vector<int8_t>`
- `gguf_writer_file` (`ggml/src/gguf.cpp:1560-1601`): writes to `FILE*`

## Tensor Storage

### On Disk

Each tensor is stored as a contiguous block of bytes. The `data_offset` field in
each tensor info record specifies the offset from the start of the data blob.
Tensors are laid out sequentially with padding between them to maintain alignment:

```
llama-model-loader.cpp:573-584   shows tensor data offset calculation
ggml/src/gguf.cpp:766-781        validates sequential offset == expected
ggml/src/gguf.cpp:773            GGML_PAD(nbytes, alignment) between tensors
```

### In Memory

When `params.ctx != nullptr` and `params.no_alloc == false`, the reader:
1. Allocates a `ggml_context` with total size = overhead + `ctx->size`
   (`ggml/src/gguf.cpp:802-817`)
2. Creates a single 1D `GGML_TYPE_I8` tensor for the entire blob
   (`ggml/src/gguf.cpp:838`)
3. Reads the blob from the file (`ggml/src/gguf.cpp:847`)
4. Creates individual tensors, setting each tensor's `data` pointer to the
   appropriate offset within the blob (`ggml/src/gguf.cpp:877-878`)

When `no_alloc == true`, tensors are created without data backing
(`ggml/src/gguf.cpp:792-800`). The loading code (model loader) then maps the
file data later via mmap or direct I/O.

### In Python

The `GGUFReader` uses `numpy.memmap` (`gguf-py/gguf/gguf_reader.py:133`). Tensor
data is accessed as numpy arrays via `self._get(data_offs, item_type, item_count)`
(`gguf-py/gguf/gguf_reader.py:367`). Quantized types with non-float storage are
mapped as `np.uint8` arrays with special shapes via `quant_shape_to_byte_shape`
(`gguf-py/gguf/gguf_reader.py:358-359`).

## How gguf-py and ggml/src/gguf.cpp Collaborate

The two implementations are independent; there is no code sharing between the
C++ (ggml) and Python (gguf-py) implementations. They produce/consume the same
binary format:

- **ggml/src/gguf.cpp**: Reference C++ implementation; used by model loading,
  model saving, quantization, and all internal tools. Static dispatch via
  templates. No Python dependency.
- **gguf-py/**: Python implementation used by `convert_hf_to_gguf.py` and all
  conversion scripts. Uses `struct.pack`/`numpy.tofile` for binary output and
  `numpy.memmap` for reading.

The conversion pipeline (`conversion/base.py` and per-model modules) uses
`GGUFWriter` to produce `.gguf` files:

1. `gguf_writer = GGUFWriter(path=None, arch=...)` (`conversion/base.py:178`)
2. `gguf_writer.add_tensor(name, tensor)` for each weight
   (`conversion/base.py:979`)
3. `gguf_writer.write_header_to_file(path=...)` (`conversion/base.py:1025`)
4. `gguf_writer.write_kv_data_to_file()` (`conversion/base.py:1026`)
5. `gguf_writer.write_tensors_to_file(progress=True)` (`conversion/base.py:1027`)

## Key Types and Protocols

### `gguf_context` (opaque handle)

Internal definition at `ggml/src/gguf.cpp:217-228`. Public API via `ggml/include/gguf.h:70`
as an incomplete type. Accessor functions:
- `gguf_get_n_kv`, `gguf_find_key`, `gguf_get_key` at `ggml/include/gguf.h:98-100`
- `gguf_get_val_*` family at `ggml/include/gguf.h:106-118`
- `gguf_get_n_tensors`, `gguf_find_tensor`, `gguf_get_tensor_*` at
  `ggml/include/gguf.h:128-133`
- `gguf_set_val_*` family at `ggml/include/gguf.h:139-150`
- `gguf_add_tensor` at `ggml/include/gguf.h:162`

### `gguf_writer_base` (abstract writer interface)

Defined at `ggml/src/gguf.cpp:1415-1519`. Provides virtual methods:
- `write(int8_t)` / `write(vector<int8_t>)` -- byte-level output
- `write_tensor_data(...)` -- copies contiguous tensor bytes to output
- Template `write<T>()` for any serializable type (line 1426)
- `write(const gguf_kv&)` for KV serialization (lines 1459-1498)
- `write_tensor_meta(const gguf_tensor_info&)` for tensor metadata
  (lines 1500-1511)
- `pad(alignment)` to write zero bytes for alignment (lines 1513-1518)

### `GGUFWriter` (Python writer class)

`gguf-py/gguf/gguf_writer.py:65`. Supports:
- Sharding (`split_max_tensors`, `split_max_size`) via separate file lists
  (`self.fout`, `self.tensors`) at lines 363-373
- Temp file mode (`use_temp_file`, lines 386-399) for large model conversion
- Endianness conversion (byteswap if needed, lines 383-385, 415-417)
- Simple scalar packing (`_simple_value_packing` dict at lines 72-84)
- Structured `add_*` methods for architecture-specific metadata

### `GGUFReader` (Python reader class)

`gguf-py/gguf/gguf_reader.py:111`. Key design:
- Memory-maps the file with `numpy.memmap` (line 133)
- Parses fields recursively via `_get_field_parts` (lines 221-257)
- Builds `ReaderField` NamedTuple (lines 39-98) for each KV pair
- Builds `ReaderTensor` NamedTuple (lines 100-108) for each tensor
- Supports r/r+/c modes for read-only or modification access (line 132)
- Detects byte-swapped files and fixes up via numpy dtype byte-order views
  (lines 142-148)

### C++ KV Override System

The `GGUFMeta::GKV<T>` template (`src/llama-model-loader.cpp:155-267`) provides
type-safe access to GGUF metadata with optional runtime overrides:
- `GKV<T>::set(ctx, key, target, override)` at line 260
- `GKV<T>::get_kv(ctx, kid)` at line 160 (throws on type mismatch)
- Type specializations at lines 115-126 map C++ types to `gguf_type`
- `ArrayInfo` struct and reader at lines 136-153

## Touch Points

### Model Loader (`src/llama-model-loader.cpp`)

- `llama_model_loader::llama_model_loader()` at line 539-665: Opens the main GGUF
  file via `gguf_init_from_file` (line 547), reads architecture from
  `LLM_KV_GENERAL_ARCHITECTURE` (line 553), loads split files if `n_split > 1`
  (lines 590-664).
- Each tensor is mapped through `weights_map` using its GGUF name
  (lines 576-584, 642-651).
- Metadata is read via the `GGUFMeta::GKV<T>` template system (lines 103-267).
- Loading from a `FILE*` pointer (not path) is supported at lines 666-677.

### Model Saver (`src/llama-model-saver.cpp`)

- `llama_model_saver::llama_model_saver()` initializes an empty `gguf_context`
  via `gguf_init_empty()` (line 38).
- `add_kv()` methods (lines 51-128) wrap `gguf_set_val_*` and `gguf_set_arr_*`.
- `add_tensor()` (lines 131-143) wraps `gguf_add_tensor`.
- `add_kv_from_model()` (lines 145-383) copies all hyperparameters, vocab, and
  config into KV pairs.
- `add_tensors_from_model()` (lines 385-412) iterates model layers.
- `save(path)` calls `gguf_write_to_file` (line 415); `save(FILE*)` calls
  `gguf_write_to_file_ptr` (line 419).
- Not all architectures are supported; `llama_model_saver_supports_arch`
  (lines 15-35) excludes PLAMO3, GEMMA3, GEMMA3N, COHERE2, COHERE2MOE, OLMO2,
  BITNET, T5, EXAONE_MOE, AFMOE, APERTUS, MIMO2, STEP35, MELLUM.

### Quantizer (`src/llama-quant.cpp`)

- References `GGUF_DEFAULT_ALIGNMENT` at `src/llama-quant.cpp:923` for
  alignment computation during quantization.

### Converters (`conversion/` directory)

- 80+ per-architecture converter modules in `conversion/` (e.g., `llama.py`,
  `qwen.py`, `gemma.py`).
- `conversion/base.py` provides the `ModelBase` class that uses `GGUFWriter`
  for output. All converters converge on the same three write calls
  (`write_header_to_file`, `write_kv_data_to_file`, `write_tensors_to_file`)
  at `conversion/base.py:1025-1027`.
- `convert_hf_to_gguf.py` is the CLI frontend (line 1).
- `convert_llama_ggml_to_gguf.py` handles legacy GGML -> GGUF migration.
- `convert_lora_to_gguf.py` handles LoRA adapter -> GGUF.

### llama.h Public API

- `llama_model_create_from_gguf_metadata` / similar at
  `include/llama.h:472`: creates a model from GGUF metadata plus a callback
  for tensor data loading.
- Functions at `include/llama.h:585-589` and `654-658` provide access to
  GGUF metadata scalar values from loaded models/adapters.

### Metadata Key Namespace

Standard KV keys are defined in:
- C++ side: `GGUF_KEY_GENERAL_ALIGNMENT` at `ggml/include/gguf.h:44`
- Python side: `Keys` class at `gguf-py/gguf/constants.py:20-382`
  - `Keys.General.*`: architecture, alignment, quantization version, licensing,
    authorship, source URLs
  - `Keys.LLM.*`: model hyperparameters (context length, embedding size, block
    count, expert config, etc.)
  - `Keys.Attention.*`: attention-specific params (head count, RoPE, layer norm eps)
  - `Keys.Tokenizer.*`: tokenizer model, vocab, special token IDs, chat template
  - `Keys.Split.*`: shard/split metadata
  - `Keys.Clip*`, `Keys.ClipVision*`, `Keys.ClipAudio*`: multimodal config
  - `Keys.SSM.*`, `Keys.WKV.*`, `Keys.Rope.*`: architecture-specific params

### Sharding / Split Files

GGUF supports multi-file shards. The Python writer creates shard filenames via
`SHARD_NAME_FORMAT = "{:s}-{:05d}-of-{:05d}.gguf"` (`gguf-py/gguf/gguf_writer.py:38`).
Shard metadata keys (`Keys.Split.*` at `gguf-py/gguf/constants.py:216-219`):
- `split.no`: shard index (uint16)
- `split.count`: total shard count (uint16)
- `split.tensors.count`: total tensors across all shards (int32)

The loader reads shard metadata and validates shard ordering
(`llama-model-loader.cpp:586-662`).

## Failure Modes

### C++ Implementation

Errors are reported via two mechanisms:

1. **`GGML_LOG_ERROR`** with `return nullptr`: Recoverable errors in the reader
   (the caller receives `nullptr` and can handle it).
2. **`GGML_ABORT`**: Unrecoverable errors (abort the process). Used for
   programming errors like duplicate tensor names when adding
   (`ggml/src/gguf.cpp:1371`), invalid types during KV copy
   (`ggml/src/gguf.cpp:1332,1361`), and tensor not found
   (`ggml/src/gguf.cpp:1384,1409`).

Specific error conditions:

| Condition | Location | Error Message |
|---|---|---|
| Failed to read magic | `ggml/src/gguf.cpp:462` | "failed to read magic" |
| Invalid magic bytes | `ggml/src/gguf.cpp:473` | "invalid magic characters: '...', expected 'GGUF'" |
| Version == 0 | `ggml/src/gguf.cpp:486` | "bad GGUF version: 0" |
| Endianness mismatch detected | `ggml/src/gguf.cpp:498` | "this GGUF file version ... is extremely large, is there a mismatch between the host and model endianness?" |
| Version == 1 (deprecated) | `ggml/src/gguf.cpp:503` | "GGUFv1 is no longer supported" |
| Version too new | `ggml/src/gguf.cpp:507` | "this GGUF file is version X but this software only supports up to version Y" |
| n_tensors out of range | `ggml/src/gguf.cpp:518` | "number of tensors is X but must be in [0, Y]" |
| n_kv out of range | `ggml/src/gguf.cpp:529` | "number of key value pairs is X but must be in [0, Y]" |
| Duplicate KV key | `ggml/src/gguf.cpp:562` | "duplicate key '...' for tensors ... and ..." |
| Invalid GGUF type | `ggml/src/gguf.cpp:596` | "key '...' has invalid GGUF type N" |
| Alignment not power of 2 | `ggml/src/gguf.cpp:613` | "alignment N is not a power of 2" |
| Tensor name too long | `ggml/src/gguf.cpp:636` | "tensor name ... is too long: N >= M" |
| Duplicate tensor name | `ggml/src/gguf.cpp:645` | "duplicate tensor name '...' for tensors ... and ..." |
| Invalid n_dims | `ggml/src/gguf.cpp:660` | "tensor '...' has invalid number of dimensions" |
| Negative dimension | `ggml/src/gguf.cpp:673` | "tensor '...' dimension N has invalid number of elements" |
| Dimension overflow (INT64_MAX) | `ggml/src/gguf.cpp:685` | "total number of elements ... is >= 9223372036854775807" |
| Invalid ggml_type | `ggml/src/gguf.cpp:702` | "tensor '...' has invalid ggml type N" |
| Row size not multiple of block size | `ggml/src/gguf.cpp:713` | "tensor '...' of type N has ... elements per row, not a multiple of block size" |
| Tensor size overflow (SIZE_MAX) | `ggml/src/gguf.cpp:722` | "tensor '...' with shape ... has a size in bytes > N" |
| Data section seek fail | `ggml/src/gguf.cpp:753` | "failed to seek to beginning of data section" |
| Offset mismatch | `ggml/src/gguf.cpp:767-768` | "tensor '...' has offset X, expected Y" |
| Accumulated size overflow | `ggml/src/gguf.cpp:776` | "tensor '...' size overflow" |
| Memory size overflow | `ggml/src/gguf.cpp:794,803,811` | "memory size overflow while allocating ggml context" |
| ggml_init failed | `ggml/src/gguf.cpp:828` | "failed to initialize ggml context for storing tensors" |
| Failed to read tensor data | `ggml/src/gguf.cpp:850` | "failed to read tensor data binary blob" |
| Failed to create tensors | `ggml/src/gguf.cpp:883` | "failed to create tensors" |
| std::length_error (key/value) | `ggml/src/gguf.cpp:433,553,628` | "encountered length_error while reading ..." |
| std::bad_alloc (key/value) | `ggml/src/gguf.cpp:436,556,631` | "encountered bad_alloc error while reading ..." |
| fputc/fwrite failure (writer) | `ggml/src/gguf.cpp:1572,1580` | std::runtime_error with details |
| Duplicate tensor on add | `ggml/src/gguf.cpp:1371` | GGML_ABORT "duplicate tensor name: ..." |
| Tensor not found on set | `ggml/src/gguf.cpp:1384,1409` | GGML_ABORT "tensor not found: ..." |
| File open failure (reader) | `ggml/src/gguf.cpp:983` | "failed to open GGUF file '...' (...)" |
| File open failure (writer) | `ggml/src/gguf.cpp:1664` | "failed to open file '...' for writing GGUF data" |

### Python Implementation

Errors are reported via exceptions:

| Condition | Location | Exception |
|---|---|---|
| Invalid magic | `gguf-py/gguf/gguf_reader.py:138` | `ValueError('GGUF magic invalid')` |
| Unsupported version | `gguf-py/gguf/gguf_reader.py:150` | `ValueError('Sorry, file appears to be version ...')` |
| Bad alignment type | `gguf-py/gguf/gguf_reader.py:176` | `ValueError('Bad type for general.alignment field')` |
| Alignment not power of 2 | `gguf-py/gguf/gguf_reader.py:180` | `ValueError('Invalid alignment: must be a non-zero power of two')` |
| Duplicate field | `gguf-py/gguf/gguf_reader.py:209` | `KeyError('Duplicate ... already in list at offset ...')` |
| Duplicate tensor name | `gguf-py/gguf/gguf_reader.py:326` | `ValueError('Found duplicated tensor with name ...')` |
| Unknown field type | `gguf-py/gguf/gguf_reader.py:257` | `ValueError('Unknown/unhandled field type ...')` |
| Duplicate tensor name (writer) | `gguf-py/gguf/gguf_writer.py:338` | `ValueError('Duplicated tensor name ...')` |
| Invalid alignment (writer) | `gguf-py/gguf/gguf_writer.py:507` | `ValueError('Invalid alignment: must be a non-zero power of two')` |
| Wrong writer state | `gguf-py/gguf/gguf_writer.py:221,239,256,335,408` | `ValueError('Expected output file to be ...')` |
| Duplicate KV key (writer) | `gguf-py/gguf/gguf_writer.py:278` | `logger.warning` only (non-fatal) |

### Loader Error Propagation

The model loader (`llama-model-loader.cpp`) wraps GGUF errors as
`std::runtime_error` with `format()` messages, which are caught higher in the
call stack by `llama.cpp`'s error handling. Examples:

- `llama-model-loader.cpp:550`: "failed to load model from ..."
- `llama-model-loader.cpp:580`: "invalid model: tensor '...' is duplicated"
- `llama-model-loader.cpp:596`: "illegal split file idx: ..."
- `llama-model-loader.cpp:623`: "failed to load GGUF split from ..."
- `llama-model-loader.cpp:630`: "missing key ... in GGUF split ..."
- `llama-model-loader.cpp:634`: "invalid split file idx: ... expected ..."
- `llama-model-loader.cpp:646`: "invalid model: tensor '...' is duplicated"
- `llama-model-loader.cpp:660`: "corrupted model: N tensors expected but M found"
