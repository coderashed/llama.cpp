# Item 01: KVarN quantization type (GGML_TYPE_Q2_KVARN)

- **Status**: done
- **Depends on**: nothing
- **Pipeline**: implementation

## Exit criterion verification

- [x] `GGML_TYPE_Q2_KVARN = 42` added to `ggml/include/ggml.h`, `GGML_TYPE_COUNT` bumped to 43
- [x] `block_q2_kvarn` struct in `ggml/src/ggml-common.h` (qs[32], d, s1, s2 = 38 bytes)
- [x] `ggml_type_traits` entry with blck_size=128, type_size=38, is_quantized=true
- [x] `quantize_row_q2_kvarn_ref` and `dequantize_row_q2_kvarn` in `ggml-quants.c`
- [x] `quantize_q2_kvarn` wrapper registered in `ggml_quantize_chunk` dispatch
- [x] `ggml_type_traits.to_float` dispatches to `dequantize_row_q2_kvarn`
- [x] Test: `test-q2-kvarn-type.cpp` — 5/5 tests pass (struct size, enum, type_traits, roundtrip, dispatch)

## Files modified

- `ggml/include/ggml.h` — enum value
- `ggml/src/ggml-common.h` — block struct
- `ggml/src/ggml-quants.h` — function declarations
- `ggml/src/ggml-quants.c` — quantize/dequantize kernels
- `ggml/src/ggml.c` — type_traits + dispatch
- `tests/test-q2-kvarn-type.cpp` — new test
- `tests/CMakeLists.txt` — test registration
