# Item 08: Context params for KVarN configuration

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation

## Exit criterion verification

- [x] `kvarn_group_size` (default 128), `kvarn_sink_tokens` (128), `kvarn_recent_tokens` (128), `kvarn_varn_iterations` (8) in `llama_context_params`
- [x] `llama_context_default_params` sets sensible defaults
- [x] Conditional copy to cparams (only when type_k/type_v is Q2_KVARN)
- [x] Test: `test-kvarn-params.cpp` — 4/4 pass

## Files modified

- `include/llama.h` — new fields
- `src/llama-context.cpp` — defaults + conditional copy
- `src/llama-cparams.h` — new fields
- `tests/test-kvarn-params.cpp` — new test
- `tests/CMakeLists.txt` — registration
