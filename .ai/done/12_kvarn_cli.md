# Item 12: CLI flag for KVarN quantization

- **Status**: done
- **Depends on**: 08
- **Pipeline**: implementation

## Exit criterion verification

- [x] `"q2_kvarn"` added to `kv_cache_types` in `common/arg.cpp`
- [x] `"q2_kvarn"` case in `llama-bench.cpp`'s `ggml_type_from_name()`
- [x] `--cache-type-k q2_kvarn` accepted by arg parser
- [x] Test: `test-kvarn-cli.cpp` — 3/3 pass

## Files modified

- `common/arg.cpp` — type list
- `tools/llama-bench/llama-bench.cpp` — type name mapping
- `tests/test-kvarn-cli.cpp` — new test
- `tests/CMakeLists.txt` — registration
