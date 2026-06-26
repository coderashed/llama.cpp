# Item 13: State save/load for Q2_KVARN cache

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation

## Exit criterion verification

- [x] `llama_kv_cache_state_size_three_region` — computes header + 6 tensor sizes
- [x] `llama_kv_cache_state_write_three_region` — serializes header + all 6 tensors
- [x] `llama_kv_cache_state_read_three_region` — validates header, restores all 6 tensors
- [x] Test: `test-kvarn-state.cpp` — 4/4 pass

## Files modified

- `src/llama-kv-cache.h` — declarations
- `src/llama-kv-cache.cpp` — implementations
- `tests/test-kvarn-state.cpp` — new test
- `tests/CMakeLists.txt` — registration
