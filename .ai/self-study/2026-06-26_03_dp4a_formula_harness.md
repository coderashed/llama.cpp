# Sprint DP4A-01: Numerical-equivalence test harness for K vec_dot -- DONE

## Question

Can we write a test that validates the K-dot formula
`Sum_i Q[i] * (k2bit[i] + d) * s1 * s2` against an fp64 oracle, pinning the
contract that the DP4A kernel (item 02) must satisfy?

## Evidence

### Test 4 passes on gfx906 MI50 hardware

```
test-q2-kvarn-gpu:
  Test 1 (supports_op): PASSED
  Test 2 (dequant_kernel): PASSED
  Test 3 (flash_attn): PASSED
  Test 4 (vec_dot_k_formula): PASSED

All tests passed
```

Device: `AMD Radeon Graphics, gfx906:sramecc+:xnack- (0x906)`

### Test implementation

`tests/test-q2-kvarn-gpu.cpp:test_vec_dot_k_formula()` (test 4):
- 20 random runs, srand(12345)
- Random 2-bit K block (32 bytes / 128 elements)
- Scales stored via fp16 round-trip to match GPU kernel read precision
- fp64 oracle: `ref64 += Q[i] * (k2bit[i] + d) * s1 * s2`
- Float emulation: same formula, same loop structure as the GPU kernel
- Assertion: relative error < 1e-4 (well within float precision)
- All 20 runs pass with no near-misses

## What we learned

1. **The formula is correct and float precision is adequate.** The float emulation
   and fp64 oracle agree to relative error < 1e-4 across all random inputs. Float
   accumulation of 128 elements introduces rounding but not enough to threaten
   the oracle. The DP4A kernel must beat this tolerance to be considered correct.

2. **The test file guard was GGML_CUDA only, not GGML_HIP.** The GPU test binary
   was never built on this ROCm system. Fixed in tests/CMakeLists.txt:
   `if (GGML_CUDA OR GGML_HIP)`.

3. **Missing supports_op entry for CPY Q2_KVARN -> F32.** Item 09 added the
   dequant kernel in convert.cu and the CPY F32->Q2_KVARN path, but omitted the
   reverse direction (Q2_KVARN->F32) from `ggml_backend_cuda_device_supports_op`.
   Fixed in ggml-cuda.cu:5274. Tests 1-3 now all pass.

4. **The skip condition for DP4A tests.** The item says "skipped when no DP4A
   device". The CPU formula test (test 4) runs regardless of device. When item 02
   adds the DP4A kernel test, THAT test should be skipped if no DP4A device. For
   now test 4 always runs -- it validates the oracle, not the DP4A path.

## Therefore we will

1. **Item 02 (DP4A K vec_dot kernel) is unblocked.** The oracle is in place;
   implement the DP4A rewrite and test it against test_vec_dot_k_formula's oracle.

2. **When item 02 is implemented**, extend test_vec_dot_k_formula or add
   test_vec_dot_k_dp4a() to call the DP4A path and compare against the same
   fp64 oracle at the same 1e-4 tolerance. Add a DP4A device check and skip
   gracefully if the device lacks DP4A support.

3. **Note for item 02**: The /QI8_1 = /8 correction (intuition doc section
   "The one subtlety") was NOT exercised by this test because test 4 uses float
   Q values, not Q8_1-quantized Q. Item 02's test must also validate the
   zeropoint term `d * Q_ds.y / QI8_1` with actual Q8_1 blocks.
