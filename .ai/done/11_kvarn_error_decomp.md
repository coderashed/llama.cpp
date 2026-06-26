# Item 11: Error decomposition measurement tool

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation

## Exit criterion verification

- [x] `ggml_kvarn_compute_errors` — per-token E_M, E_D, E_T
- [x] `ggml_kvarn_topk_ratio` — top-k% mean E_M/E_T ratio
- [x] `ggml_kvarn_error_histogram` — histogram bins
- [x] Cross-quantizer comparison (KVarN vs Q4_0)
- [x] Test: `test-q2-kvarn-error.cpp` — 4/4 pass

## Files modified

- `tests/test-q2-kvarn-error.cpp` — new test
- `tests/CMakeLists.txt` — registration
