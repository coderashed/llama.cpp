# Item 02: Dequantization kernel for Q2_KVARN

- **Status**: done
- **Depends on**: 01
- **Pipeline**: implementation

## Exit criterion verification

- [x] `dequantize_row_q2_kvarn` exists in `ggml-quants.c`
- [x] Reads `block_q2_kvarn`, unpacks 2-bit values, applies `K_dq = (K_q + d) * s1 * s2`
- [x] Bug fixed: `& 0x01` → `& 0x03` on line 503
- [x] Registered in `ggml_type_traits[GGML_TYPE_Q2_KVARN].to_float`
- [x] Test: `test-q2-kvarn-dequant.cpp` — 3/3 functional tests pass (bit mask, roundtrip, dual-scale)

## Files modified

- `ggml/src/ggml-quants.c` — fixed bit mask bug
- `tests/test-q2-kvarn-dequant.cpp` — new test
- `tests/CMakeLists.txt` — test registration
