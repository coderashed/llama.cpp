---
title: Q2_KVARN Dequantization -- File Ownership & Registration
---

```mermaid
graph TB
    subgraph "ggml/src/ggml-quants.c"
        DQFUNC["dequantize_row_q2_kvarn()<br/>Scalar implementation<br/>  assert(k % 128 == 0)<br/>  nb = k / 128<br/>  for each block:<br/>    d=FP16_TO_FP32(x[i].d)<br/>    s1=FP16_TO_FP32(x[i].s1)<br/>    s2=FP16_TO_FP32(x[i].s2)<br/>    for each byte j in qs[32]:<br/>      q0 = (byte>>0) & 0x03<br/>      q1 = (byte>>2) & 0x03<br/>      q2 = (byte>>4) & 0x03<br/>      q3 = (byte>>6) & 0x03<br/>      y[4*j+0] = (q0+d)*s1*s2<br/>      y[4*j+1] = (q1+d)*s1*s2<br/>      y[4*j+2] = (q2+d)*s1*s2<br/>      y[4*j+3] = (q3+d)*s1*s2"]
    end

    subgraph "ggml/src/ggml.c"
        TRAITS["type_traits[GGML_TYPE_Q2_KVARN] = {<br/>  .to_float = dequantize_row_q2_kvarn,<br/>  ...<br/>}"]
    end

    subgraph "ggml/src/ggml-common.h"
        STRUCT["block_q2_kvarn {<br/>  uint8_t qs[32];<br/>  ggml_half d;<br/>  ggml_half s1;<br/>  ggml_half s2;<br/>}"]
    end

    subgraph "tests/test-q2-kvarn-dequant.cpp"
        TEST1["Test 1: bit_mask (0x03 mask)"]
        TEST2["Test 2: roundtrip (1e-3 per-element)"]
        TEST3["Test 3: dual_scale (s1*s2 fused)"]
    end

    subgraph "ggml/include/ggml.h"
        TYPEDEF["ggml_to_float_t typedef<br/>  void (*)(const void*, float*, int64_t)"]
    end

    STRUCT -->|"read by"| DQFUNC
    TYPEDEF -->|"signature match"| DQFUNC
    DQFUNC -->|"registered as"| TRAITS
    DQFUNC -->|"tested by"| TEST1
    DQFUNC -->|"tested by"| TEST2
    DQFUNC -->|"tested by"| TEST3

    classDef bug fill:#ffcccc,stroke:#cc0000,stroke-width:2px
    classDef new fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    class DQFUNC new
    class TEST6 new
```

### File Ownership

| File | Role | Changes needed |
|------|------|----------------|
| `ggml/src/ggml-quants.c` | Dequant implementation | Fix `& 0x01` -> `& 0x03` at line 503 |
| `ggml/src/ggml.c` | Type traits registration | Already done (Item 01) |
| `ggml/src/ggml-common.h` | Block struct definition | Already done (Item 01) |
| `tests/test-q2-kvarn-dequant.cpp` | Dequant tolerance tests | Tests 1-4: bit_mask, roundtrip, dual_scale, block_size |

### Registration Chain

```
ggml_type_traits[GGML_TYPE_Q2_KVARN].to_float
    |
    v
dequantize_row_q2_kvarn(x, y, k)
    |
    | called by:
    v
ggml_compute_forward_dup (CPU backend)
ggml_compute_forward_get_rows (CPU backend)
ggml_compute_forward_cpy (CPU backend)
Any backend that calls to_float for type conversion
```

### No SIMD (by design)

The codebase convention is **scalar dequant + SIMD dot product**. No existing type has a SIMD dequant path. SIMD for Q2_KVARN would go in:
- `ggml-cpu/arch/x86/quants.c` (AVX2/AVX512)
- `ggml-cpu/arch/arm/quants.c` (NEON)

But this is deferred -- the scalar implementation satisfies the functional requirement.
