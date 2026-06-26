---
title: Three-Region Cache Layout -- File Ownership & Integration
---

```mermaid
graph TD
    subgraph "src/ (llama.cpp application layer)"
        KV_CELLS_H["llama-kv-cells.h<br/>- kvarn_region enum (NEW)<br/>- llama_kv_cell_ext: region field (MODIFIED)<br/>- llama_kv_cells: set_region/get_region/count_region (NEW)"]
        KV_CACHE_H["llama-kv-cache.h<br/>- kv_layer: k_sink/v_sink/k_body/v_body/k_recent/v_recent (MODIFIED)<br/>- RegionClassifier (NEW)<br/>- GroupQuantizer (NEW)"]
        KV_CACHE_CPP["llama-kv-cache.cpp<br/>- cpy_k/cpy_v: region-aware dispatch (MODIFIED)<br/>- RegionClassifier::classify() (NEW)<br/>- GroupQuantizer::quantize_group() (NEW)<br/>- rebuild_region_metadata() (NEW)<br/>- tensor allocation: 3 tensors per layer (MODIFIED)<br/>- build_rope_shift: FP16 sink/recent skip dequant (MODIFIED)"]
        GRAPH_CPP["llama-graph.cpp<br/>- build_attn: get_k/get_v return region-aware views (MODIFIED)<br/>- attention over mixed-precision regions"]
    end

    subgraph "ggml/ (low-level tensor library)"
        GGML_COMMON_H["ggml-common.h<br/>block_q2_kvarn struct<br/>(unchanged from Item 05)"]
        GGML_QUANTS_C["ggml-quants.c<br/>quantize/dequant routines<br/>(unchanged from Item 05)"]
        GGML_C["ggml.c<br/>type_traits, set_rows<br/>(unchanged from Item 05)"]
    end

    subgraph "Tensor Layout per Layer"
        LAYOUT["kv_layer layout:<br/><br/>k_sink:  [n_embd_k_gqa, S, n_stream]  type=F16<br/>k_body:  [n_embd_k_gqa, kv_size-S-R, n_stream]  type=Q2_KVARN<br/>k_recent:[n_embd_k_gqa, R, n_stream]  type=F16<br/><br/>v_sink:  [n_embd_v_gqa, S, n_stream]  type=F16<br/>v_body:  [n_embd_v_gqa, kv_size-S-R, n_stream]  type=Q2_KVARN<br/>v_recent:[n_embd_v_gqa, R, n_stream]  type=F16"]
    end

    subgraph "Region Transition Flow"
        FLOW1["Token arrives -> classify(pos) -> region"]
        FLOW2["SINK: write FP16 to k_sink"]
        FLOW3["BODY: Hadamard -> VarN -> quantize -> write to k_body"]
        FLOW4["RECENT: write FP16 to k_recent"]
        FLOW5["RECENT group full (G tokens):<br/>quantize group -> move to k_body<br/>update cell region: RECENT -> BODY"]
    end

    KV_CELLS_H --> KV_CACHE_H
    KV_CACHE_H --> KV_CACHE_CPP
    KV_CACHE_CPP --> GRAPH_CPP
    KV_CACHE_CPP --> GGML_QUANTS_C
    KV_CACHE_CPP --> GGML_C
    KV_CACHE_CPP --> LAYOUT
    LAYOUT --> FLOW1
    FLOW1 --> FLOW2
    FLOW1 --> FLOW3
    FLOW1 --> FLOW4
    FLOW4 --> FLOW5

    style KV_CACHE_CPP fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style KV_CELLS_H fill:#bbdefb,stroke:#1565c0
    style KV_CACHE_H fill:#bbdefb,stroke:#1565c0
    style LAYOUT fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style FLOW5 fill:#fff3e0,stroke:#e65100,stroke-width:2px

    note for KV_CACHE_CPP "Key integration points:<br/>1. cpy_k() dispatches to sink/body/recent tensor<br/>2. get_k() returns a view that spans all three regions<br/>   (or returns separate views for attention)<br/>3. build_rope_shift skips dequant/re-quant for sink/recent<br/>4. find_slot() must account for region boundaries"
    note for LAYOUT "Total cells = S + (kv_size - S - R) + R = kv_size<br/>Body region shrinks/grows as recent groups are quantized<br/>S and R are compile-time constants (default 128)"
```
