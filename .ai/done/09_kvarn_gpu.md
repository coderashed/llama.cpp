# Item 09: GPU backend kernel for Q2_KVARN dequantization

- **Status**: done
- **Depends on**: 02
- **Pipeline**: implementation

## Exit criterion verification

- [x] `GGML_TYPE_Q2_KVARN` added to `supports_op` for `GGML_OP_MUL_MAT` in `ggml-cuda.cu`
- [x] `dequantize_block_q2_kvarn` CUDA kernel + `dequantize_row_q2_kvarn_cuda` wrapper in `convert.cu`
- [x] `GGML_TYPE_Q2_KVARN` added to `ggml_cuda_flash_attn_ext_supported` in `fattn.cu`
- [x] Test: `test-q2-kvarn-gpu.cpp` — guarded by `GGML_CUDA`

## Files modified

- `ggml/src/ggml-cuda/ggml-cuda.cu` — supports_op
- `ggml/src/ggml-cuda/convert.cu` — dequant kernel + dispatch
- `ggml/src/ggml-cuda/fattn.cu` — flash attention support
- `tests/test-q2-kvarn-gpu.cpp` — new test
- `tests/CMakeLists.txt` — registration (guarded)
