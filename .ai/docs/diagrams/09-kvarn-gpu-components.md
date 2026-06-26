---
title: Q2_KVARN GPU Kernel -- File Ownership & CUDA Build Integration
---

```mermaid
graph TD
    subgraph "New Files"
        KVARN_CUH["kvarn.cuh (NEW)<br/>- dequantize_q2_kvarn() device function<br/>- vec_dot_fattn_vec_KQ_q2_kvarn() device function<br/>- dequantize_V_q2_kvarn() device function<br/>- load_tiles_q2_kvarn() device function<br/>- vec_dot_q2_kvarn_q8_1_mma() device function<br/>- vec_dot_q2_kvarn_q8_1_dp4a() device function"]
    end

    subgraph "Modified Files"
        FATTN_COMMON_CUH["fattn-common.cuh<br/>- get_vec_dot_KQ(): add Q2_KVARN case<br/>- get_dequantize_V(): add Q2_KVARN case"]
        MMQ_CUH["mmq.cuh<br/>- mmq_get_q8_1_ds_layout(): add Q2_KVARN case<br/>- mmq_type_traits specialization for Q2_KVARN<br/>- load_tiles_q2_kvarn() (or in kvarn.cuh)"]
        MMQ_CU["mmq.cu<br/>- ggml_cuda_mul_mat_q_switch_type(): add Q2_KVARN case"]
        COMMON_CUH["common.cuh<br/>- ggml_cuda_type_traits specialization for Q2_KVARN"]
        GGML_CUDA_CU["ggml-cuda.cu<br/>- supports_op(): add Q2_KVARN to MUL_MAT switch<br/>- supports_op(): add Q2_KVARN to GET_ROWS switch"]
        CMakeLists_txt["CMakeLists.txt<br/>- kvarn.cuh added to GGML_HEADERS_CUDA (auto-glob)"]
    end

    subgraph "Existing Files (unchanged)"
        FATTN_VEC_CUH["fattn-vec.cuh<br/>- flash_attn_ext_vec kernel template<br/>- Uses get_vec_dot_KQ / get_dequantize_V dispatch"]
        FATTN_CU["fattn.cu<br/>- Host dispatch for flash attention"]
        DECONVERT_CUH["dequantize.cuh<br/>- Existing dequant functions (Q4_0, Q8_0, etc.)"]
        VECDOTQ_CUH["vecdotq.cuh<br/>- Existing vec_dot helpers (get_int_b1, etc.)"]
    end

    subgraph "Build Integration (CMakeLists.txt:102-130)"
        GLOB_HEADERS["file(GLOB GGML_HEADERS_CUDA *.cuh)"]
        GLOB_SOURCES["file(GLOB GGML_SOURCES_CUDA *.cu)"]
        GLOB_TEMPLATES["file(GLOB template-instances/mmq*.cu)"]
        ADD_LIB["ggml_add_backend_library(ggml-cuda<br/>  ${GGML_HEADERS_CUDA}<br/>  ${GGML_SOURCES_CUDA})"]
    end

    subgraph "Template Instances (template-instances/)"
        MMQ_INSTANCES["mmq-instance-q2_kvarn.cu (NEW)<br/>#include 'mmq.cuh'<br/>template void mul_mat_q_case<GGML_TYPE_Q2_KVARN>(...)"]
    end

    KVARN_CUH --> FATTN_COMMON_CUH : "included by"
    KVARN_CUH --> MMQ_CUH : "included by"
    FATTN_COMMON_CUH --> FATTN_VEC_CUH : "included by"
    FATTN_VEC_CUH --> FATTN_CU : "included by"
    MMQ_CUH --> MMQ_CU : "included by"
    MMQ_CU --> GGML_CUDA_CU : "linked"
    COMMON_CUH --> FATTN_COMMON_CUH : "included by"
    COMMON_CUH --> MMQ_CUH : "included by"
    GLOB_HEADERS --> KVARN_CUH : "auto-discovers"
    GLOB_SOURCES --> MMQ_CU : "auto-discovers"
    GLOB_TEMPLATES --> MMQ_INSTANCES : "auto-discovers"
    GLOB_HEADERS --> ADD_LIB
    GLOB_SOURCES --> ADD_LIB
    GLOB_TEMPLATES --> ADD_LIB

    style KVARN_CUH fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style MMQ_INSTANCES fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    style FATTN_COMMON_CUH fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style MMQ_CUH fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style MMQ_CU fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style COMMON_CUH fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style GGML_CUDA_CU fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style CMakeLists_txt fill:#fff9c4,stroke:#f57f17,stroke-width:2px
```

### File Change Summary

| File | Change | Type |
|------|--------|------|
| `ggml/src/ggml-cuda/kvarn.cuh` | New file: dequant, vec_dot, load_tiles, dequantize_V for Q2_KVARN | **NEW** |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-q2_kvarn.cu` | New file: explicit template instantiation for `mul_mat_q_case<Q2_KVARN>` | **NEW** |
| `ggml/src/ggml-cuda/fattn-common.cuh` | Add Q2_KVARN cases to `get_vec_dot_KQ()` and `get_dequantize_V()` | **MODIFY** |
| `ggml/src/ggml-cuda/mmq.cuh` | Add Q2_KVARN to `mmq_get_q8_1_ds_layout()`, add `mmq_type_traits` specialization | **MODIFY** |
| `ggml/src/ggml-cuda/mmq.cu` | Add `case GGML_TYPE_Q2_KVARN` to `ggml_cuda_mul_mat_q_switch_type()` | **MODIFY** |
| `ggml/src/ggml-cuda/common.cuh` | Add `ggml_cuda_type_traits<GGML_TYPE_Q2_KVARN>` specialization | **MODIFY** |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Add `case GGML_TYPE_Q2_KVARN` to `supports_op()` MUL_MAT and GET_ROWS switches | **MODIFY** |
| `ggml/src/ggml-cuda/CMakeLists.txt` | No change needed: `file(GLOB *.cuh)` auto-discovers `kvarn.cuh`; `file(GLOB template-instances/mmq*.cu)` auto-discovers instance file | **NONE** |

### Build Integration Notes

- **No CMake changes required**: The existing `file(GLOB GGML_HEADERS_CUDA "*.cuh")` at line 102 and `file(GLOB SRCS "template-instances/mmq*.cu")` at line 110 will automatically pick up the new files.
- **Template instantiation pattern**: Following the existing pattern in `template-instances/mmq-q2_k.cu` (or similar), the new `mmq-instance-q2_kvarn.cu` provides the explicit template instantiation that the linker needs.
- **Flash attention template instances**: If flash attention with Q2_KVARN is needed, a new `template-instances/fattn-vec-instance-q2_kvarn-q2_kvarn.cu` may be required, following the pattern at lines 120-124 of CMakeLists.txt.
