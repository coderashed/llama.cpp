---
title: Q2_KVARN Type Definition -- Component / Package Diagram
---

```mermaid
graph TB
    subgraph "ggml/include/ggml.h"
        ENUM["ggml_type enum<br/><b>+GGML_TYPE_Q2_KVARN = 42</b><br/>GGML_TYPE_COUNT = 43"]
        TRAITS_DECL["ggml_type_traits struct<br/>(to_float, from_float_ref,<br/>blck_size, type_size fields)"]
        CHUNK_DECL["ggml_quantize_chunk()<br/>declaration"]
    end

    subgraph "ggml/src/ggml-common.h"
        STRUCT["block_q2_kvarn<br/>qs[32]   : uint8_t<br/>d        : ggml_half<br/>s1       : ggml_half<br/>s2       : ggml_half<br/>sizeof=38, blck=128"]
    end

    subgraph "ggml/src/ggml.c"
        TRAITS["type_traits[] static init<br/>[GGML_TYPE_Q2_KVARN] = {<br/>  .type_name = 'q2_kvarn',<br/>  .blck_size = 128,<br/>  .type_size = sizeof(block_q2_kvarn),<br/>  .is_quantized = true,<br/>  .to_float = dequantize_row_q2_kvarn,<br/>  .from_float_ref = quantize_row_q2_kvarn_ref<br/>}"]
        DISPATCH["ggml_quantize_chunk()<br/>  case GGML_TYPE_Q2_KVARN:<br/>    quantize_q2_kvarn(src, dst, ...)<br/>    break;"]
    end

    subgraph "ggml/src/ggml-quants.c"
        QUANT["quantize_q2_kvarn()<br/>  -> quantize_row_q2_kvarn_ref()<br/>  Scalar RTN: d=min, s1=range/3, s2=1.0<br/>  2-bit pack -> block_q2_kvarn"]
        DEQUANT["dequantize_row_q2_kvarn()<br/>  Scalar: unpack 2-bit<br/>  val = (qval + d) * s1 * s2"]
    end

    subgraph "ggml/include/ggml.h type_traits client"
        GETTER["ggml_get_type_traits()<br/>  return &type_traits[type]<br/>  (used by backends for to_float)"]
    end

    %% Dependency arrows
    ENUM --->|"enum value"| TRAITS
    ENUM --->|"case label"| DISPATCH
    
    STRUCT --->|"sizeof reference"| TRAITS
    STRUCT --->|"read/write"| QUANT
    STRUCT --->|"read"| DEQUANT

    TRAITS_DECL --->|"implements"| TRAITS
    CHUNK_DECL --->|"implements"| DISPATCH
    
    TRAITS -.->|".to_float calls"| DEQUANT
    TRAITS -.->|".from_float_ref calls"| QUANT
    
    DISPATCH -.->|"dispatches to"| QUANT
    GETTER -.->|"reads"| TRAITS

    %% Style
    classDef new fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class ENUM,STRUCT,TRAITS,DISPATCH,QUANT,DEQUANT new
```

### File Ownership Summary

| File | What it contributes |
|------|---------------------|
| `ggml/include/ggml.h` | Enum value, struct declaration, chunk declaration |
| `ggml/src/ggml-common.h` | `block_q2_kvarn` struct definition |
| `ggml/src/ggml.c` | type_traits entry, quantize_chunk dispatch case |
| `ggml/src/ggml-quants.c` | `quantize_q2_kvarn`, `quantize_row_q2_kvarn_ref`, `dequantize_row_q2_kvarn` implementations |

### External Interfaces

| Interface | Direction | Consumers |
|-----------|-----------|-----------|
| `ggml_quantize_chunk(type=Q2_KVARN, ...)` | Input | llama.cpp KV-cache layer, model conversion |
| `ggml_type_traits[Q2_KVARN].to_float()` | Input | CPU backend (ggml_compute_forward), GPU backend |
| `ggml_type_traits[Q2_KVARN].from_float_ref()` | Input | CPU backend, quantization path |
| `block_q2_kvarn` memory layout | Data | All backends that read/write Q2_KVARN tensors |
