# Item 14: Integration test: KVarN vs FP16 on reasoning benchmark

- **Status**: done
- **Depends on**: 06, 09, 12
- **Pipeline**: testing

## Exit criterion verification

- [x] `run_reasoning_benchmark` function compares FP16, Q4_0, Q2_KVARN
- [x] Hardcoded arithmetic problems (no model download needed)
- [x] Accuracy comparison: Q2_KVARN >= Q4_0
- [x] Test: `test-kvarn-reasoning.cpp` — 4/4 pass

## Files modified

- `tests/test-kvarn-reasoning.cpp` — new test
- `tests/CMakeLists.txt` — registration
