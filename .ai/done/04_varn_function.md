# Item 04: Variance normalization function (VarN)

- **Status**: done
- **Depends on**: nothing
- **Pipeline**: implementation

## Exit criterion verification

- [x] `kvarn_variance_normalize` exists in `src/llama-kvarn.h` / `src/llama-kvarn.cpp`
- [x] Takes tile T[R][C], iteration count K, clamp limits c_min/c_max
- [x] Implements Algorithm 1: log-domain dual-scale Sinkhorn balancing with Imb() metric and best-state tracking
- [x] Returns normalized tile and scale vectors S_c (1xC) and S_r (Rx1)
- [x] CPU implementation using efficient scalar with std::vector RAII
- [x] Test: `test-kvarn-varn.cpp` — 7/7 tests pass (basic call, convergence, scales written, edge cases)

## Files created/modified

- `src/llama-kvarn.h` — new header
- `src/llama-kvarn.cpp` — new implementation
- `src/CMakeLists.txt` — registration
- `tests/test-kvarn-varn.cpp` — new test
- `tests/CMakeLists.txt` — test registration
- `.ai/docs/diagrams/04-varn-classes.mmd` — updated
- `.ai/docs/diagrams/04-varn-sequence.mmd` — updated
