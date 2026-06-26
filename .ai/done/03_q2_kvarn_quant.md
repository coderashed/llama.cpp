# Item 03: Quantization kernel for Q2_KVARN (RTN + dual-scale)

- **Status**: done
- **Depends on**: 01
- **Pipeline**: implementation

## Exit criterion verification

- [x] `quantize_row_q2_kvarn_ref` exists in `ggml-quants.c`
- [x] Computes per-block zeropoint (FP16), per-block scale s1, per-token scale s2
- [x] Round-to-nearest with asymmetric quantization (zeropoint offset)
- [x] Stores packed 2-bit values + scales + zeropoint in `block_q2_kvarn`
- [x] Registered in `ggml_type_traits[GGML_TYPE_Q2_KVARN].from_float_ref`
- [x] `ggml_quantize_chunk` dispatches to it
- [x] s2 computed as L2 norm ratio (norm_orig / norm_dq) for per-token magnitude preservation
- [x] Test: `test-q2-kvarn-quant.cpp` — 3/3 pass (E_M/E_T median < 0.5, MSE finite, worst-token bound)

## Files modified

- `ggml/src/ggml-quants.c` — added `compute_s2_scale` helper, rewrote quantizer with per-token s2
- `tests/test-q2-kvarn-quant.cpp` — new test
- `tests/CMakeLists.txt` — test registration
