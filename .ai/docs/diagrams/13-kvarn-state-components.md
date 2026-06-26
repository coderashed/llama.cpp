---
title: KVarN State Save/Load -- File Ownership & API Surface
---

```mermaid
graph TD
    subgraph "Public API (include/llama.h)"
        API_GET_SIZE["llama_state_get_size(ctx)\nReturns total bytes needed"]
        API_GET_DATA["llama_state_get_data(ctx, dst, size)\nSerializes full cache state"]
        API_SET_DATA["llama_state_set_data(ctx, src, size)\nDeserializes full cache state"]
        API_SEQ_GET["llama_state_seq_get_data(ctx, dst, size, seq_id)\nSerializes single sequence"]
        API_SEQ_SET["llama_state_seq_set_data(ctx, src, size, dest_seq_id)\nDeserializes single sequence"]
        API_FILE_SAVE["llama_state_save_file(ctx, path, tokens, n)\nSave to file"]
        API_FILE_LOAD["llama_state_load_file(ctx, path, tokens, cap, n_out)\nLoad from file"]
    end

    subgraph "Implementation (src/llama-kv-cache.cpp)"
        STATE_WRITE["state_write(io, seq_id, flags)\nEntry point: iterates streams,\nbuilds cell_ranges_t"]
        STATE_READ["state_read(io, seq_id, flags)\nEntry point: validates n_stream,\niterates streams"]
        STATE_WRITE_META["state_write_meta(io, cr, seq_id)\nPer-cell: pos, n_seq_id, ext, seq_ids"]
        STATE_READ_META["state_read_meta(io, strm, cell_count, sinfo, dest_seq_id)\nReconstructs cell metadata + slot_info"]
        STATE_WRITE_DATA["state_write_data(io, cr)\nPer-layer tensor data"]
        STATE_READ_DATA["state_read_data(io, strm, cell_count, sinfo)\nPer-layer tensor data restore"]
    end

    subgraph "New: Q2_KVARN dispatch (llama-kv-cache.cpp)"
        KVARN_WRITE["write_kvarn_regions(io, cr, layer)\nWrites k_sink, k_body, k_recent,\nv_sink, v_body, v_recent"]
        KVARN_READ["read_kvarn_regions(io, cell_count, sinfo, layer)\nReads three regions,\nvalidates types/sizes"]
    end

    subgraph "Context glue (src/llama-context.cpp)"
        CTX_GET_SIZE["llama_state_get_size()\nCalls memory->state_write(dry-run io)\nto compute size"]
        CTX_GET_DATA["llama_state_get_data()\nCalls memory->state_write(real io)"]
        CTX_SET_DATA["llama_state_set_data()\nCalls memory->state_read(real io)"]
    end

    subgraph "I/O abstraction (src/llama-io.h)"
        IO_WRITE["llama_io_write_i\nwrite() / write_tensor()"]
        IO_READ["llama_io_read_i\nread() / read_tensor()"]
    end

    subgraph "Cell metadata (src/llama-kv-cells.h)"
        CELLS["llama_kv_cells\npos_get/set, seq_has/add,\next_get/set, is_empty"]
    end

    API_GET_SIZE --> CTX_GET_SIZE
    API_GET_DATA --> CTX_GET_DATA
    API_SET_DATA --> CTX_SET_DATA
    API_SEQ_GET --> CTX_GET_DATA
    API_SEQ_SET --> CTX_SET_DATA
    API_FILE_SAVE --> CTX_GET_DATA
    API_FILE_LOAD --> CTX_SET_DATA

    CTX_GET_SIZE --> STATE_WRITE
    CTX_GET_DATA --> STATE_WRITE
    CTX_SET_DATA --> STATE_READ

    STATE_WRITE --> STATE_WRITE_META
    STATE_WRITE --> STATE_WRITE_DATA
    STATE_READ --> STATE_READ_META
    STATE_READ --> STATE_READ_DATA

    STATE_WRITE_DATA -.->|"if is_kvarn"| KVARN_WRITE
    STATE_READ_DATA -.->|"if is_kvarn"| KVARN_READ

    STATE_WRITE_META --> CELLS
    STATE_READ_META --> CELLS
    STATE_WRITE_DATA --> IO_WRITE
    STATE_READ_DATA --> IO_READ
    KVARN_WRITE --> IO_WRITE
    KVARN_READ --> IO_READ

    style KVARN_WRITE fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style KVARN_READ fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style STATE_WRITE_DATA fill:#fff3e0,stroke:#e65100
    style STATE_READ_DATA fill:#fff3e0,stroke:#e65100

    note for KVARN_WRITE "NEW: Dispatched when type_k == GGML_TYPE_Q2_KVARN\nWrites 6 region tensors per layer (3 for K, 3 for V)\nEach region: type_i + row_size + raw bytes\nSink/Recent are F16, Body is Q2_KVARN"
    note for KVARN_READ "NEW: Dispatched when type_k == GGML_TYPE_Q2_KVARN\nReads 6 region tensors per layer\nValidates type and row_size match runtime values\nUses io.read_tensor() for raw byte copy"
    note for STATE_WRITE_DATA "EXISTING: writes k_stream[cr.strm] and v_stream[cr.strm]\nFor Q2_KVARN, k_stream is a 2D view of k_sink only\n-> MUST be replaced with three-region write"
    note for STATE_READ_DATA "EXISTING: reads into k_stream[cr.strm] and v_stream[cr.strm]\nFor Q2_KVARN, this only populates the sink region\n-> MUST be replaced with three-region read"
```
