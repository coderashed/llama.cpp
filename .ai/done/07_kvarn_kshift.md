# Item 07: K-shift graph support for Q2_KVARN

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation

## Exit criterion verification

- [x] `build_rope_shift` handles Q2_KVARN via `ggml_is_quantized` (already generic)
- [x] `from_float` registered in CPU type_traits for F32→Q2_KVARN cpy
- [x] `quantize_row_q2_kvarn` wrapper in `ggml-cpu/quants.c`
- [x] Three-region shift: sink/recent F16 in-place, body Q2_KVARN dequant→rotate→requant
- [x] Test: `test-kvarn-kshift.cpp` — 4/4 pass

## Files modified

- `ggml/src/ggml-cpu/quants.c` — quantize wrapper
- `ggml/src/ggml-cpu/quants.h` — declaration
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU type_traits
- `tests/test-kvarn-kshift.cpp` — new test
- `tests/CMakeLists.txt` — registration
