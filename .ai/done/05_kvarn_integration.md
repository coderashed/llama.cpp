# Item 05: Integrate VarN into KV-cache quantization path

- **Status**: done
- **Depends on**: 03, 04
- **Pipeline**: implementation

## Exit criterion verification

- [x] `quantize_row_q2_kvarn_varn` exists in `ggml-quants.c` — accepts external S_c/S_r scales
- [x] Extracted `quantize_q2_kvarn_block()` helper shared by ref and varn variants
- [x] VarN tile normalization + quantize pipeline verified (E_M/E_T median = 0.000138)
- [x] Partial group fallback (<128 elements) delegates to plain RTN
- [x] Test: `test-kvarn-integration.cpp` — 4/4 pass (function exists, scale absorption, pipeline, fallback)

## Files modified

- `ggml/src/ggml-quants.h` — declared `quantize_row_q2_kvarn_varn`
- `ggml/src/ggml-quants.c` — implemented varn variant, extracted block helper
- `tests/test-kvarn-integration.cpp` — new test
- `tests/CMakeLists.txt` — test registration
