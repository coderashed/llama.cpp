# Item 06: Three-region cache layout (sink/body/recent)

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation

## Exit criterion verification

- [x] `kvarn_region` enum (SINK/BODY/RECENT) in `llama-kv-cells.h`
- [x] `region` field in `llama_kv_cell_ext`
- [x] Three tensor pointers (k_sink/k_body/k_recent, v_sink/v_body/v_recent) in `kv_layer`
- [x] Conditional allocation: when type_k == Q2_KVARN, allocates three tensors
- [x] Test: `test-kvarn-layout.cpp` — 4/4 pass

## Files modified

- `src/llama-kv-cells.h` — region enum + cell metadata
- `src/llama-kv-cache.h` — three-tensor layout + group trigger
- `src/llama-kv-cache.cpp` — conditional allocation
- `tests/test-kvarn-layout.cpp` — new test
- `tests/CMakeLists.txt` — registration
