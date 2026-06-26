# Multimodal / Vision Support

## Purpose

LLM decoders natively process only text tokens. Multimodal support enables the model to "see" images (and hear audio) by encoding pixel data into embedding vectors that can be fed alongside text tokens into the same transformer decoder. This allows a single model to answer questions like "What is in this picture?" or to accept a diagram and generate code from it.

## Architecture Overview

llama.cpp implements multimodal support as a **separate sub-project** (`tools/mtmd/`) rather than integrating vision into the core `libllama`. The user supplies **two GGUF files**:

1. **Text model** -- the standard language model.
2. **mmproj (multimodal projector) file** -- a CLIP-style vision encoder + projection layers.

This two-file design keeps vision encoder development independent from the core inference engine. The mmproj is loaded by `libmtmd`, which provides a clean C API for tokenizing and encoding multimodal inputs, then feeding the resulting embeddings into `llama_decode`.

Relevant reading: `tools/mtmd/README.md:3-15`, `tools/mtmd/README-dev.md:8-9`.

## CLIP-Based Vision Encoder

The vision encoder lives in `tools/mtmd/clip.cpp` and `tools/mtmd/clip-model.h`. The core struct is `clip_ctx` (`tools/mtmd/clip.cpp:146`), which holds:

- A `clip_model` containing all vision tensors (patch embeddings, transformer layers, projection weights).
- A `ggml_backend_sched` for GPU/CPU compute.
- An optional audio encoder (`ctx_a`) alongside the visual encoder (`ctx_v`).

The encoder follows a Vision Transformer (ViT) design. The `clip_graph` base class (`tools/mtmd/clip-graph.h:18-136`) provides the building blocks:

- `build_inp_raw()` -- converts raw RGB pixel data into a ggml tensor.
- `build_inp()` -- applies the 2D convolution (patch embedding) to produce patch tokens.
- `build_vit()` -- runs the standard ViT encoder stack: layer norm, multi-head self-attention (via `build_attn`), FFN (via `build_ffn`), residual connections.
- `build_patch_merge_permute()` -- pixel-shuffle / patch merger for Kimi-VL style models.

Each supported model architecture subclasses `clip_graph` and overrides `build()`. The graph factory dispatch happens inside `clip.cpp` during `clip_init()`:

```cpp
// tools/mtmd/clip.cpp:3134
struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params);
```

This reads the GGUF metadata, determines the projector type, loads tensors, calls model-specific `init_ctx()` to instantiate the right `clip_graph` subclass, and runs an optional warmup pass. The model-specific graph builders are registered by name in `tools/mtmd/models/models.h` and their source files live under `tools/mtmd/models/`.

### Image Preprocessing

Image preprocessing is model-dependent and handled by subclasses of `mtmd_image_preprocessor` in `tools/mtmd/mtmd-image.h:30`. Four strategies exist:

- **Fixed-size** (`mtmd_image_preprocessor_fixed_size`): resize/pad the input to a fixed square (e.g., 336x336 for LLaVA 1.5).
- **Dynamic-size** (`mtmd_image_preprocessor_dyn_size`): resize to multiples of `patch_size * n_merge` preserving aspect ratio (Qwen-VL, Pixtral, Kimi-VL).
- **Longest-edge** (`mtmd_image_preprocessor_longest_edge`): resize so the longest edge equals `image_longest_edge`.
- **LLaVA-UHD** (`mtmd_image_preprocessor_llava_uhd`): slice large images into tiles (overview + grid). Used by MiniCPM-V, LLaVA 1.6, Idefics3, InternVL, etc.

The preprocessor selection happens in `mtmd_context::init_vision()` (`tools/mtmd/mtmd.cpp:389-658`) based on the projector type.

## Multimodal Projections

After the ViT encoder produces vision features, a **projection head** maps them into the text model's embedding space. The `clip_model` stores projection tensors as `mm_*` weight matrices (`tools/mtmd/clip-model.h:384-553`).

The projector type is read from the GGUF key `clip.projector_type` (`tools/mtmd/clip-impl.h:32`). The enum `projector_type` (`tools/mtmd/clip-impl.h:322-375`) lists over 40 known types, including:

| Type | Description | Source example |
|------|-------------|----------------|
| `PROJECTOR_TYPE_MLP` | Simple 2-layer MLP | LLaVA (`tools/mtmd/models/llava.cpp`) |
| `PROJECTOR_TYPE_QWEN2VL` | Qwen2VL spatial merger + 2D RoPE | (`tools/mtmd/models/qwen2vl.cpp`) |
| `PROJECTOR_TYPE_QWEN3VL` | Qwen3VL deepstack merger | (`tools/mtmd/models/qwen3vl.cpp`) |
| `PROJECTOR_TYPE_INTERNVL` | InternVL tile-based projector | (`tools/mtmd/models/internvl.cpp`) |
| `PROJECTOR_TYPE_MINICPMV` | Resampler (perceiver-style) | (`tools/mtmd/models/minicpmv.cpp`) |
| `PROJECTOR_TYPE_GEMMA3` | Siglip-based projection | (`tools/mtmd/models/siglip.cpp`) |
| `PROJECTOR_TYPE_PIXTRAL` | Pixtral learnable merger | (`tools/mtmd/models/pixtral.cpp`) |
| `PROJECTOR_TYPE_LLAMA4` | Llama 4 tile projection | (`tools/mtmd/models/llama4.cpp`) |
| `PROJECTOR_TYPE_HUNYUANVL` | HunyuanVL perceiver + M-RoPE | (`tools/mtmd/models/hunyuanvl.cpp`) |
| `PROJECTOR_TYPE_COGVLM` | CogVLM projector with BOI/EOI | (`tools/mtmd/models/cogvlm.cpp`) |

The dimensionality of the projection output must match the text model's `n_embd_inp`. This is validated at init time in the `mtmd_context` constructor (`tools/mtmd/mtmd.cpp:375-380`):

```cpp
if (n_embd_text > 0 && n_embd_text != n_embd_clip) {
    throw std::runtime_error("mismatch between text model ... and mmproj");
}
```

The actual encoding happens via `clip_image_batch_encode()` (`tools/mtmd/clip.h:85`), which calls the model's graph builder and runs inference on the CLIP backend.

## Architecture-Specific Support

All 34 model sources are in `tools/mtmd/models/`. Each implements a `clip_graph_*` subclass with a `build()` override that constructs the model-specific computation graph. Key examples:

- **LLaVA** (`tools/mtmd/models/llava.cpp:5-373`): The reference ViT + MLP projector. Used by multiple models including Granite and GLM-Edge.
- **Qwen2VL** (`tools/mtmd/models/qwen2vl.cpp:1-205`): Uses 2D patch embeddings with temporal frame merge (2-frame video support at line 17-29). M-RoPE position encoding with 4-dimensional rope sections.
- **Qwen3VL** (`tools/mtmd/models/qwen3vl.cpp:1-186`): Extends Qwen2VL with "deepstack" feature layers -- features from multiple transformer layers are stacked and merged via learned `deepstack_fc1`/`deepstack_fc2` projections (`clip-model.h:217-222`).
- **InternVL** (`tools/mtmd/models/internvl.cpp`): Tile-based dynamic resolution with a signed position embedding and an InternVL-specific projection.
- **MiniCPM-V** (`tools/mtmd/models/minicpmv.cpp`): Uses a resampler (perceiver-style) with learned queries and cross-attention to compress visual features.
- **Gemma 3** (`tools/mtmd/models/siglip.cpp`): SigLIP vision encoder with `mm_input_proj_w` and `mm_soft_emb_norm_w` for the soft embedding normalization.
- **Pixtral** (`tools/mtmd/models/pixtral.cpp`): Variable-resolution with pixel-shuffle merging and `[IMG_BREAK]` tokens.

## How Multimodal Inputs Are Batched with Text

The pipeline is orchestrated by `libmtmd` (`tools/mtmd/mtmd.h`, `tools/mtmd/mtmd.cpp`):

1. **Tokenization** (`mtmd_tokenize` at `tools/mtmd/mtmd.cpp:1428`): The user provides a text prompt with `<__media__>` markers and a list of bitmaps (images/audio). The tokenizer:
   - Splits the text at markers (`split_text` at line 1381).
   - For Qwen-VL-style models, merges consecutive frames (`n_merge_frames` at line 924).
   - Calls the image/audio preprocessor to produce `mtmd_input_chunk` objects.
   - Injects model-specific special tokens (e.g., `<|vision_start|>`, `<img>`, `[IMG]`) around media chunks.

2. **Encoding** (`mtmd_encode_chunk` at line 1512): Each media chunk is encoded through the CLIP model, producing float embedding vectors via `clip_image_batch_encode()`.

3. **Evaluation** (`mtmd_helper_eval_chunk_single` at `tools/mtmd/mtmd-helper.cpp:338-407`):
   - Text chunks are tokenized and passed to `llama_decode()` directly.
   - Image/audio chunks are first encoded via `mtmd_encode_chunk()`, then the resulting embeddings are placed into a `llama_batch` and passed to `llama_decode()`.
   - Non-causal masks are used for models that need them (e.g., Gemma 3).

4. **Batching**: The `mtmd_batch` API (`tools/mtmd/mtmd.cpp:1535-1580`) allows multiple images with the same dimensions to be encoded together in a single CLIP forward pass, provided the CLIP model supports batching.

### M-RoPE Support

For models like Qwen2VL/Qwen3VL that use M-RoPE (multi-dimensional rotary position embedding), the `mtmd_image_tokens` struct carries `nx`/`ny` spatial dimensions and a `pos` field (`MTMD_POS_TYPE_MROPE`). The `mtmd_image_tokens_get_decoder_pos()` function (`mtmd.h:259`) computes per-token `(t, x, y)` positions that are fed to `llama_decode()`.

## The `clip_*` API

The `clip_*` API (`tools/mtmd/clip.h:66-98`) is the lower-level interface for the CLIP vision encoder:

| Function | Purpose | File:Line |
|----------|---------|-----------|
| `clip_init()` | Load mmproj, init CLIP context | `clip.h:66`, `clip.cpp:3134` |
| `clip_free()` | Free CLIP context | `clip.h:68` |
| `clip_image_encode()` | Encode single image | `clip.h:84` |
| `clip_image_batch_encode()` | Encode batched images | `clip.h:85` |
| `clip_n_mmproj_embd()` | Get embedding dimension | `clip.h:81` |
| `clip_n_output_tokens()` | Get number of output tokens | `clip.h:73` |
| `clip_n_output_tokens_x/y()` | Get spatial dimensions (2D grid) | `clip.h:77-78` |
| `clip_has_vision_encoder()` | Check if vision available | `clip.h:91` |
| `clip_has_audio_encoder()` | Check if audio available | `clip.h:92` |
| `clip_support_batch()` | Check if batched encoding supported | `clip.h:94` |
| `clip_get_cap()` | Check capabilities of mmproj file | `clip.h:100-103` |
| `clip_get_mem_usage()` | Get memory usage per backend | `clip.h:98` |

**Note**: There is **no `llama_mm_*` API** in the codebase. The research backlog symbol is outdated. The public multimodal API is the `mtmd_*` family in `mtmd.h:78-427`.

## The `mtmd_*` API

The `mtmd_*` API (`tools/mtmd/mtmd.h`) is the higher-level, user-facing interface:

| Function | Purpose | File:Line |
|----------|---------|-----------|
| `mtmd_init_from_file()` | Load mmproj and bind to text model | `mtmd.h:123` |
| `mtmd_free()` | Free mtmd context | `mtmd.h:127` |
| `mtmd_tokenize()` | Split text + bitmaps into chunks | `mtmd.h:277` |
| `mtmd_encode_chunk()` | Encode a media chunk | `mtmd.h:289` |
| `mtmd_get_output_embd()` | Get encoded embeddings | `mtmd.h:295` |
| `mtmd_batch_add_chunk()` | Add chunk to batch | `mtmd.h:309` |
| `mtmd_batch_encode()` | Encode all batched chunks | `mtmd.h:313` |
| `mtmd_batch_get_output_embd()` | Get per-chunk embeddings from batch | `mtmd.h:314` |
| `mtmd_support_vision()` | Check vision support | `mtmd.h:137` |
| `mtmd_support_audio()` | Check audio support | `mtmd.h:140` |
| `mtmd_get_cap_from_file()` | Check mmproj capabilities (server) | `mtmd.h:327` |

**Helper API** (`tools/mtmd/mtmd-helper.h`):

| Function | Purpose |
|----------|---------|
| `mtmd_helper_eval_chunks()` | Tokenize + encode + decode all chunks in sequence |
| `mtmd_helper_eval_chunk_single()` | Process a single chunk through encode + decode |
| `mtmd_helper_decode_image_chunk()` | Feed pre-encoded embeddings to llama_decode with proper batching and non-causal mask support |
| `mtmd_helper_bitmap_init_from_file()` | Load image/audio file into a bitmap |
| Video loading helpers | Frame-based video input via ffmpeg subprocess |

## Touch Points

### Inference Engine

- `tools/mtmd/clip.cpp` -- CLIP model loading, graph construction, and inference dispatch.
- `tools/mtmd/mtmd.cpp` -- Multimodal context, tokenization, chunk management, encoding orchestration.
- `tools/mtmd/mtmd-helper.cpp` -- Bridge between mtmd encoding and `llama_decode()`. Manages `llama_batch` creation for text tokens and image embeddings.

### Tokenization

- `tools/mtmd/mtmd.cpp:1402-1425` -- `mtmd_tokenize_text_internal()` wraps `llama_tokenize()`.
- `tools/mtmd/mtmd.cpp:1012-1040` -- `add_text()` merges consecutive text chunks and injects BOS/EOS tokens.
- `tools/mtmd/mtmd.cpp:766-779` -- `lookup_token()` scans the vocabulary to find special token IDs (e.g., `<|vision_start|>`, `<img>`).

### llama-server Integration

- `tools/server/server-context.cpp:1254-1268` -- Loads mmproj and initializes `mtmd_context`.
- `tools/server/server-context.cpp:1528` -- Passes `mtmd_support_vision()` to the server's input validation.
- `tools/server/server-common.cpp:986-1016` -- Validates image/audio/video input presence against mmproj capabilities.

### Quantization

- `src/llama-quant.cpp:345` -- Some multimodal tensors are excluded from quantization.

### Graph / Backend

- `src/llama-graph.cpp:1903` -- Raw embeddings (multimodal inputs) bypass scaling.
- `src/llama.cpp:314` -- `CLIP cannot be used as main model, use it with --mmproj instead`.

## Failure Modes

1. **Mismatched embedding dimensions** -- If `n_embd` of the mmproj does not match the text model's `n_embd_inp`, init fails with a clear error (`tools/mtmd/mtmd.cpp:375-380`).
2. **Missing mmproj** -- Running a vision-capable model without `--mmproj` causes the server/CLI to reject image inputs (`tools/server/server-common.cpp:988`).
3. **Wrong mmproj for model** -- Using a mmproj from a different model family silently produces garbage outputs. The error message at `mtmd.cpp:377` hints at this.
4. **Image too large** -- Dynamic-resolution models may OOM on very large images. The `image_max_tokens` parameter limits this.
5. **VRAM exhaustion** -- The mmproj runs on GPU if available (`clip.cpp:191-194`). Large tile-based models (LLaVA-UHD with many tiles) can exceed GPU memory.
6. **Clip model not loading** -- If the mmproj path is wrong or the file is corrupted, `clip_init` returns null (`clip.cpp:3169-3176`).
7. **Vision/audio modality mismatch** -- Attempting to encode an image when only audio support exists (or vice versa), handled at `mtmd.cpp:1051` and `mtmd.cpp:1252`.
8. **Placeholder-only prompts** -- If all bitmaps are placeholders (e.g., for token counting), `mtmd_encode_chunk` will return an error (`mtmd.cpp:1453-1456`).
9. **Temporal merge limitations** -- Qwen-VL style frame merging is limited to 2 frames (`mtmd.cpp:926`). Merging more causes an assertion failure.
