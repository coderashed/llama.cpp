---
title: KVarN State Save/Load -- Safety Contract, Versioning & Compatibility
---

```mermaid
graph TD
    subgraph "Format Versioning"
        V1["format_version == 1<br/>Legacy single-tensor layout<br/>k_stream[cr.strm] written as-is"]
        V2["format_version == 2<br/>Q2_KVARN three-region layout<br/>6 region tensors per layer"]
        CHECK["state_read checks format_version<br/>from header field"]
        V1 -->|"type_k == Q2_KVARN"| INCOMPAT["Throw: 'cannot restore Q2_KVARN state<br/>from legacy format'"]
        V1 -->|"type_k != Q2_KVARN"| OK1["Proceed with legacy read path"]
        V2 -->|"type_k == Q2_KVARN"| OK2["Proceed with three-region read path"]
        V2 -->|"type_k != Q2_KVARN"| INCOMPAT2["Throw: 'format_version=2 requires<br/>Q2_KVARN cache type'"]
    end

    subgraph "Preconditions (caller must ensure)"
        P1["dst/src buffer is large enough<br/>(use llama_state_get_size to query)"]
        P2["Cache is not shared (other == null)"]
        P3["seq_id is valid (-1 for all, or valid seq_id)"]
        P4["For seq_id != -1: dest_seq_id exists<br/>or will be created"]
    end

    subgraph "Validation Checks (state_read)"
        C1["n_stream_cur == n_stream"]
        C2["cell_count <= cells.size()"]
        C3["n_layer == layers.size()"]
        C4["v_trans flag matches runtime"]
        C5["Per-layer K type matches runtime"]
        C6["Per-layer K row_size matches runtime"]
        C7["Per-layer V type matches runtime"]
        C8["Per-layer V row_size matches runtime"]
        C9["For Q2_KVARN: region types match<br/>sink=F16, body=Q2_KVARN, recent=F16"]
        C10["For Q2_KVARN: region sizes match<br/>S, B, R from header"]
        C11["seq_id in range [0, n_seq_max)"]
    end

    subgraph "Error Recovery"
        E1["state_read_meta returns false"]
        E2["state_read_data returns false"]
        E3["Any validation check fails"]
        E4["seq_id == -1: clear(true) entire cache"]
        E5["seq_id != -1: seq_rm(seq_id) single sequence"]
        E6["throw std::runtime_error('failed to restore kv cache')"]
    end

    subgraph "Compatibility Matrix"
        COMP1["Save v1, Load v1: OK (existing)"]
        COMP2["Save v2, Load v2: OK (new code)"]
        COMP3["Save v1, Load v2 (Q2_KVARN): ERROR"]
        COMP4["Save v2, Load v1: N/A (old code can't read v2)"]
        COMP5["Save v2 (Q2_KVARN), Load v2 (non-kvarn): ERROR<br/>type mismatch on region tensors"]
        COMP6["Save v2 (non-kvarn), Load v2 (Q2_KVARN): ERROR<br/>header says v2 but no three-region data"]
    end

    subgraph "Size Computation (llama_state_get_size)"
        S1["Dry-run: state_write with counting IO"]
        S2["For Q2_KVARN: sum of all 6 region tensors<br/>+ header + cell metadata"]
        S3["Sink region: S * n_layer * ggml_row_size(F16, n_embd_gqa)"]
        S4["Body region: B * n_layer * ggml_row_size(Q2_KVARN, n_embd_gqa)"]
        S5["Recent region: R * n_layer * ggml_row_size(F16, n_embd_gqa)"]
        S6["V transposed: n_embd_v_gqa * S/B/R * el_size"]
    end

    subgraph "Thread Safety"
        T1["state_write is const: read-only, safe"]
        T2["state_read mutates cache: must hold exclusive lock"]
        T3["No concurrent state_read + decode"]
        T4["state_write during decode: safe (read-only snapshot)"]
    end

    CHECK --> V1
    CHECK --> V2
    V1 --> INCOMPAT
    V1 --> OK1
    V2 --> OK2
    V2 --> INCOMPAT2

    C1 --> E3
    C2 --> E3
    C3 --> E3
    C4 --> E3
    C5 --> E3
    C6 --> E3
    C7 --> E3
    C8 --> E3
    C9 --> E3
    C10 --> E3
    C11 --> E3

    E1 --> E4
    E1 --> E5
    E2 --> E4
    E2 --> E5
    E3 --> E4
    E3 --> E5
    E4 --> E6
    E5 --> E6

    note for V1 "Existing format: no explicit version field<br/>Implicit version 1 by convention.<br/>New code writes format_version=2 in header."
    note for V2 "New header field 'format_version'<br/>written at start of state_write_data<br/>Read and validated at start of state_read_data"
    note for COMP3 "Cannot restore Q2_KVARN state from legacy save<br/>because legacy format only has k_stream (sink view)<br/>Body and recent regions are lost"
    note for S2 "Size computation must account for all three regions<br/>Existing llama_state_get_size uses dry-run state_write<br/>which will automatically count the new region tensors"
```
