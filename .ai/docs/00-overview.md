# llama.cpp — Architecture Overview

This directory documents the major subsystems of llama.cpp.
Each file covers one feature with code-cited descriptions and Mermaid UML diagrams.

## Done

| # | Feature | Doc | Diagrams |
|---|---------|-----|----------|
| 00 | GGUF File Format | [00-gguf-format.md](00-gguf-format.md) | [diagrams/00-gguf-format/](diagrams/00-gguf-format/) |
| 01 | GGML Tensor Library | [01-ggml-tensor-library.md](01-ggml-tensor-library.md) | [diagrams/01-ggml-tensor-library/](diagrams/01-ggml-tensor-library/) |
| 02 | GGML Backend System | [02-ggml-backend.md](02-ggml-backend.md) | [diagrams/02-ggml-backend/](diagrams/02-ggml-backend/) |
| 03 | Model Architecture Registry & Loading | [03-model-architecture.md](03-model-architecture.md) | [diagrams/03-model-architecture/](diagrams/03-model-architecture/) |
| 04 | Tokenization / Vocabulary | [04-tokenization.md](04-tokenization.md) | [diagrams/04-tokenization/](diagrams/04-tokenization/) |
| 05 | Quantization | [05-quantization.md](05-quantization.md) | [diagrams/05-quantization/](diagrams/05-quantization/) |
| 06 | Core LLM Inference Engine | [06-inference-engine.md](06-inference-engine.md) | [diagrams/06-inference-engine/](diagrams/06-inference-engine/) |
| 07 | KV Cache Management | [07-kv-cache.md](07-kv-cache.md) | [diagrams/07-kv-cache/](diagrams/07-kv-cache/) |
| 08 | Memory Management (Recurrent/SSM) | [08-memory-management.md](08-memory-management.md) | [diagrams/08-memory-management/](diagrams/08-memory-management/) |
| 09 | Sampling | [09-sampling.md](09-sampling.md) | [diagrams/09-sampling/](diagrams/09-sampling/) |
| 10 | Grammar / Constrained Decoding | [10-grammar.md](10-grammar.md) | [diagrams/10-grammar/](diagrams/10-grammar/) |
| 11 | Chat Templates & Auto Parser | [11-chat-templates.md](11-chat-templates.md) | [diagrams/11-chat-templates/](diagrams/11-chat-templates/) |
| 12 | Speculative Decoding | [12-speculative-decoding.md](12-speculative-decoding.md) | [diagrams/12-speculative-decoding/](diagrams/12-speculative-decoding/) |
| 13 | llama-server | [13-llama-server.md](13-llama-server.md) | [diagrams/13-llama-server/](diagrams/13-llama-server/) |
| 14 | CLI Tools | [14-cli-tools.md](14-cli-tools.md) | [diagrams/14-cli-tools/](diagrams/14-cli-tools/) |
| 15 | Model Conversion | [15-model-conversion.md](15-model-conversion.md) | [diagrams/15-model-conversion/](diagrams/15-model-conversion/) |
| 16 | Multimodal / Vision Support | [16-multimodal.md](16-multimodal.md) | [diagrams/16-multimodal/](diagrams/16-multimodal/) |
| 17 | Embeddings | [17-embeddings.md](17-embeddings.md) | [diagrams/17-embeddings/](diagrams/17-embeddings/) |
| 18 | C API (libllama) | [18-c-api.md](18-c-api.md) | [diagrams/18-c-api/](diagrams/18-c-api/) |
| 19 | Training & Finetuning | [19-training-finetuning.md](19-training-finetuning.md) | [diagrams/19-training-finetuning/](diagrams/19-training-finetuning/) |

## Pending

| # | Feature | Depends on |
|---|---------|-----------|

