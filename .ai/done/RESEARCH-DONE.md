# Research Completed — llama.cpp

### 00 — GGUF File Format

- **Status**: done
- **Depends on**: nothing
- **Doc**: `.ai/docs/00-gguf-format.md` (491 lines, 87 citations)
- **Exit criterion met**: Describes binary layout, KV metadata system, tensor storage model, C++/Python collaboration; citations verified by spot-check.

### 01 — GGML Tensor Library

- **Status**: done
- **Depends on**: nothing
- **Doc**: `.ai/docs/01-ggml-tensor-library.md` (255 lines, 92 citations)
- **Exit criterion met**: Describes type system, memory management, graph construction, autograd, ops; citations verified by spot-check.

### 02 — GGML Backend System

- **Status**: done
- **Depends on**: 01
- **Doc**: `.ai/docs/02-ggml-backend.md` (379 lines, 54 citations)
- **Exit criterion met**: Describes device abstraction, buffer types, registration/lookup, per-backend implementations; citations verified by spot-check.

### 03 — Model Architecture Registry & Model Loading

- **Status**: done
- **Depends on**: 00, 01
- **Doc**: `.ai/docs/03-model-architecture.md` (257 lines, 59 citations)
- **Exit criterion met**: Describes llm_arch enum, parameter mappings, model loader flow, model construction; citations verified by spot-check.

### 15 — Model Conversion

- **Status**: done
- **Depends on**: 00
- **Doc**: `.ai/docs/15-model-conversion.md` (281 lines, ~190 citations)
- **Exit criterion met**: Describes conversion framework, base converter class, per-architecture overrides (81 converters), C++ saver integration; citations verified by spot-check.

### 04 — Tokenization / Vocabulary

- **Status**: done
- **Depends on**: 03
- **Doc**: `.ai/docs/04-tokenization.md` (372 lines, ~70 citations)
- **Diagrams**: `.ai/docs/diagrams/04-tokenization/` (class, sequence, component)
- **Exit criterion met**: Describes `llama_vocab` PIMPL, 6 tokenizer types (SPM, BPE, WPM, UGM, RWKV, PLaMo-2), encode/decode paths, unicode data tables, public API; citations verified by spot-check.

### 05 — Quantization

- **Status**: done
- **Depends on**: 01, 03
- **Doc**: `.ai/docs/05-quantization.md` (350 lines, ~80 citations)
- **Diagrams**: `.ai/docs/diagrams/05-quantization/` (class, sequence, component)
- **Exit criterion met**: Describes dual type system (`ggml_type`/`llama_ftype`), quantization pipeline, K-quant and IQ-family formats, load-time behavior; citations verified by spot-check.

### 06 — Core LLM Inference Engine

- **Status**: done
- **Depends on**: 03, 04, 05
- **Doc**: `.ai/docs/06-inference-engine.md` (356 lines)
- **Diagrams**: `.ai/docs/diagrams/06-inference-engine/` (class, sequence, component)
- **Exit criterion met**: Describes context lifecycle (`llama_init_from_model`), graph building, decode loop (`llama_decode`), batching (`llama_batch`/`llama_ubatch`), compute graph topology for LLaMA forward pass; citations verified by spot-check.

### 11 — Chat Templates & Auto Parser

- **Status**: done
- **Depends on**: 04
- **Doc**: `.ai/docs/11-chat-templates.md` (435 lines)
- **Diagrams**: `.ai/docs/diagrams/11-chat-templates/` (class, sequence, component)
- **Exit criterion met**: Describes `llama_chat_message`, `common_chat_template` Jinja pipeline, PEG-based auto parser, per-model handlers; citations verified by spot-check.

### 07 — KV Cache Management

- **Status**: done
- **Depends on**: 06
- **Doc**: `.ai/docs/07-kv-cache.md` (444 lines)
- **Diagrams**: `.ai/docs/diagrams/07-kv-cache/` (class, sequence, component)
- **Exit criterion met**: Describes cell-based cache, DSA/ISWA variants, seq ops, graph integration, K-cache quantization; citations verified by spot-check.

### 08 — Memory Management (Recurrent/SSM)

- **Status**: done
- **Depends on**: 06
- **Doc**: `.ai/docs/08-memory-management.md` (204 lines)
- **Diagrams**: `.ai/docs/diagrams/08-memory-management/` (class, sequence, component)
- **Exit criterion met**: Describes `llama_memory_i` interface, all 6 implementations (recurrent, hybrid, hybrid-iswa, kv-cache, kv-cache-iswa, kv-cache-dsa); citations verified.

### 09 — Sampling

- **Status**: done
- **Depends on**: 06
- **Doc**: `.ai/docs/09-sampling.md` (~210 lines)
- **Diagrams**: `.ai/docs/diagrams/09-sampling/` (class, sequence, component)
- **Exit criterion met**: Describes sampler chain, 19 individual samplers, GPU backend sampling, grammar integration; citations verified.

### 16 — Multimodal / Vision Support

- **Status**: done
- **Depends on**: 06, 04
- **Doc**: `.ai/docs/16-multimodal.md` (14.3 KB)
- **Diagrams**: `.ai/docs/diagrams/16-multimodal/` (class, sequence, component)
- **Exit criterion met**: Describes `mtmd_*` API (not `llama_mm_*`), CLIP encoder, 34 model-specific graph builders, two-file design; citations verified.

### 17 — Embeddings

- **Status**: done
- **Depends on**: 06
- **Doc**: `.ai/docs/17-embeddings.md` (11.5 KB)
- **Diagrams**: `.ai/docs/diagrams/17-embeddings/` (class, sequence, component)
- **Exit criterion met**: Describes embedding extraction via `llama_get_embeddings`, 5 pooling strategies, `build_pooling()` graph, harnesses; citations verified.

### 19 — Training & Finetuning

- **Status**: done
- **Depends on**: 01, 06
- **Doc**: `.ai/docs/19-training-finetuning.md` (full coverage)
- **Diagrams**: `.ai/docs/diagrams/19-training-finetuning/` (class, sequence, component)
- **Exit criterion met**: Describes ggml_opt optimizer, imatrix, fit-params harness, WIP finetune subsystem; citations verified.

### 10 — Grammar / Constrained Decoding

- **Status**: done
- **Depends on**: 04, 09
- **Doc**: `.ai/docs/10-grammar.md` (14 KB)
- **Diagrams**: `.ai/docs/diagrams/10-grammar/` (class, sequence, component)
- **Exit criterion met**: Describes GBNF format, JSON Schema bridge, PEG parser, llguidance integration, rejection-sampling approach; notes `llama_grammar_init`/`llama_grammar_sample` do not exist (API is `llama_sampler_init_grammar`); citations verified.

### 12 — Speculative Decoding

- **Status**: done
- **Depends on**: 06, 09
- **Doc**: `.ai/docs/12-speculative-decoding.md` (13 KB)
- **Diagrams**: `.ai/docs/diagrams/12-speculative-decoding/` (class, sequence, component)
- **Exit criterion met**: Describes 8 draft implementations, verification loop, tuning; notes `llama_speculative`/`llama_draft_model`/`llama_verify` do not exist (API is `common_speculative_*`); citations verified.

### 13 — llama-server

- **Status**: done
- **Depends on**: 06, 09, 10, 11, 12
- **Doc**: `.ai/docs/13-llama-server.md` (~300 lines)
- **Diagrams**: `.ai/docs/diagrams/13-llama-server/` (class, sequence, component)
- **Exit criterion met**: Describes HTTP layer, task queue, chat completions endpoint, tools/function calling, model management, multimodal, CORS proxy; citations verified.

### 14 — CLI Tools

- **Status**: done
- **Depends on**: 06, 09, 10
- **Doc**: `.ai/docs/14-cli-tools.md` (13.7 KB)
- **Diagrams**: `.ai/docs/diagrams/14-cli-tools/` (class, sequence, component)
- **Exit criterion met**: Describes all 7 CLI tools, shared arg parser (`arg.cpp`), `common_params_parse`; citations verified.

### 18 — C API (libllama)

- **Status**: done
- **Depends on**: all above
- **Doc**: `.ai/docs/18-c-api.md`
- **Diagrams**: `.ai/docs/diagrams/18-c-api/` (class, sequence, component)
- **Exit criterion met**: Describes `LLAMA_API` surface, opaque struct/PIMPL pattern, ABI conventions, function categories; notes asymmetry (llama_context is concrete, llama_model/llama_vocab are PIMPL); citations verified.
