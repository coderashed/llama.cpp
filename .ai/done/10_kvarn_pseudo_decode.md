# Item 10: Pseudo-decode evaluation harness

- **Status**: done
- **Depends on**: 05
- **Pipeline**: implementation

## Exit criterion verification

- [x] Test harness at `tests/test-kvarn-pseudo-decode.cpp`
- [x] Processes synthetic K matrices in blocks of b=128 tokens
- [x] Measures per-token reconstruction error
- [x] Verifies errors are finite and monotonic
- [x] Test: 4/4 pass

## Files modified

- `tests/test-kvarn-pseudo-decode.cpp` — new test
- `tests/CMakeLists.txt` — registration
