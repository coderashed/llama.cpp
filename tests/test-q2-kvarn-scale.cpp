// RED phase test for KVARN_FAITHFUL/02b - reconstruction-aware scalar scale.
// CPU-only. No GPU. No CUDA headers.
//
// Contract: .ai/design/02b_scale_design.md
//
// Property 1 (FAILS today, PASSES after GREEN):
//   On a 128-element block with 4 large outliers the in-tree quantizer MSE
//   must be strictly lower than the baseline min/max MSE by more than 5%.
//   TODAY: in-tree IS min/max, so mse_real == mse_base -> assert FAILS.
//   AFTER: grid search picks f<1 for the outlier block -> PASSES.
//
// Property 2 (PASSES today and after GREEN):
//   On a smooth no-outlier 128-element block the in-tree MSE must not exceed
//   baseline MSE by more than 5%. After GREEN, the grid search always includes
//   f=1.0 (the baseline interval), so mse_real <= mse_base.
//
// Property 3 (sanity, PASSES both):
//   All-equal block reconstructs the constant, no NaN/Inf anywhere.
//   Exercises the range==0 degenerate branch.

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <random>

// Block size for q2_kvarn: 128 elements per block (QK2_KVARN).
// Do not include ggml-common.h here; use the public row-size API instead.
static const int QK = 128;

// ---------------------------------------------------------------------------
// Inline baseline: the current (pre-02b) min/max 2-bit quantizer + dequant.
// Mirrors quantize_q2_kvarn_block + compute_s2_scale + dequantize_row_q2_kvarn
// from ggml/src/ggml-quants.c exactly, including fp16 round-trips for d/s1/s2.
// This makes mse_baseline == mse_real today (same algorithm), so
// assert(mse_real < mse_base * 0.95) is false -> Property 1 FAILS.
// ---------------------------------------------------------------------------
static float baseline_mse(const float * x) {
    // 1. Scan for min, max
    float mn = x[0], mx = x[0];
    for (int j = 1; j < QK; j++) {
        if (x[j] < mn) mn = x[j];
        if (x[j] > mx) mx = x[j];
    }
    const float range     = mx - mn;
    const float half_range = range / 3.0f;                  // s1 in today's code
    const float inv_d      = (half_range > 0.0f) ? 1.0f / half_range : 1.0f;

    // 2. Encode 2-bit codes (same as quantize_q2_kvarn_block encode loop)
    uint8_t qs[QK / 4];
    for (int j = 0; j < QK / 4; j++) {
        uint8_t byte = 0;
        for (int b = 0; b < 4; b++) {
            float val = (x[j*4 + b] - mn) * inv_d;
            if (val < 0.0f) val = 0.0f;
            if (val > 3.0f) val = 3.0f;
            byte |= ((uint8_t)(val + 0.5f)) << (b * 2);
        }
        qs[j] = byte;
    }

    // 3. Store d and s1 as fp16 then read back (matching the real quantizer's
    //    fp16 rounding), so that baseline and real dequant values are identical
    //    today and the MSEs are bit-for-bit equal.
    const ggml_fp16_t d_fp16  = ggml_fp32_to_fp16(mn * inv_d);
    const ggml_fp16_t s1_fp16 = ggml_fp32_to_fp16(half_range);
    const float d_val  = ggml_fp16_to_fp32(d_fp16);
    const float s1_val = ggml_fp16_to_fp32(s1_fp16);

    // 4. Compute s2 (mirrors compute_s2_scale in ggml-quants.c)
    double sum_sq_orig = 0.0, sum_sq_dq = 0.0;
    for (int j = 0; j < QK / 4; j++) {
        const uint8_t byte = qs[j];
        for (int b = 0; b < 4; b++) {
            const float qv  = (float)((byte >> (b * 2)) & 0x03);
            const float dqv = (qv + d_val) * s1_val;
            sum_sq_dq   += (double)dqv * (double)dqv;
            sum_sq_orig += (double)x[j*4 + b] * (double)x[j*4 + b];
        }
    }
    const float norm_orig = sqrtf((float)sum_sq_orig);
    const float norm_dq   = sqrtf((float)sum_sq_dq);
    const float s2 = (norm_dq > 1e-10f) ? norm_orig / norm_dq : 1.0f;
    const float s2_val = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s2));   // fp16 round-trip

    // 5. Dequantize (mirrors dequantize_row_q2_kvarn: y = (qval + d) * s1 * s2)
    const float combined = s1_val * s2_val;
    double mse = 0.0;
    for (int j = 0; j < QK / 4; j++) {
        const uint8_t byte = qs[j];
        for (int b = 0; b < 4; b++) {
            const float qv  = (float)((byte >> (b * 2)) & 0x03);
            const float dqv = (qv + d_val) * combined;
            const double diff = (double)x[j*4 + b] - (double)dqv;
            mse += diff * diff;
        }
    }
    return (float)(mse / QK);
}

// ---------------------------------------------------------------------------
// Real path: quantize via ggml_quantize_chunk (dispatches to
// quantize_row_q2_kvarn -> quantize_row_q2_kvarn_ref) and dequantize via
// type_traits->to_float (dispatches to dequantize_row_q2_kvarn).
// ---------------------------------------------------------------------------
static float real_mse(const float * x, float * dq_out) {
    const size_t row_size = ggml_row_size(GGML_TYPE_Q2_KVARN, QK);
    std::vector<uint8_t> qbuf(row_size + 64);

    ggml_quantize_chunk(GGML_TYPE_Q2_KVARN, x, qbuf.data(), 0, 1, QK, NULL);

    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    tt->to_float(qbuf.data(), dq_out, QK);

    double mse = 0.0;
    for (int i = 0; i < QK; i++) {
        const double diff = (double)x[i] - (double)dq_out[i];
        mse += diff * diff;
    }
    return (float)(mse / QK);
}

// ---------------------------------------------------------------------------
// Property 1 (FAILS today):
//   Outlier block: 124 samples from N(0,1) + 4 large outliers.
//   Assert: mse_real < mse_base * 0.95
//   TODAY: in-tree IS min/max -> mse_real == mse_base -> assert FAILS.
//   AFTER GREEN: grid search picks f<1 -> mse_real < mse_base * 0.95 -> PASSES.
// ---------------------------------------------------------------------------
static void test_outlier_block_strict_improvement(void) {
    float x[QK];

    // 124 samples from N(0,1) with fixed seed
    std::mt19937 rng(0x1337u);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 124; i++) {
        x[i] = dist(rng);
    }
    // 4 large outliers that dominate the min/max range
    x[124] = +12.0f;
    x[125] = -15.0f;
    x[126] = +20.0f;
    x[127] = -18.0f;

    const float mse_base = baseline_mse(x);

    float dq[QK];
    const float mse_real = real_mse(x, dq);

    printf("  Property 1: mse_base=%.6f  mse_real=%.6f  threshold(0.95x)=%.6f\n",
           mse_base, mse_real, mse_base * 0.95f);
    printf("  (mse_real < threshold must hold; TODAY this FAILS because in-tree IS min/max)\n");

    // This assert FAILS today: mse_real == mse_base, so mse_real < mse_base*0.95 is false.
    // After GREEN (grid search), a smaller f is chosen for this outlier block and PASSES.
    assert(mse_real < mse_base * 0.95f);
}

// ---------------------------------------------------------------------------
// Property 2 (PASSES today and after GREEN):
//   Smooth no-outlier block: 128 samples from N(0,1), no injected outliers.
//   Assert: mse_real <= mse_base * 1.05  (not more than 5% worse).
//   After GREEN: grid search always includes f=1.0 so mse_real <= mse_base.
// ---------------------------------------------------------------------------
static void test_smooth_block_no_regression(void) {
    float x[QK];

    std::mt19937 rng(0xABCD1234u);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < QK; i++) {
        x[i] = dist(rng);
    }

    const float mse_base = baseline_mse(x);

    float dq[QK];
    const float mse_real = real_mse(x, dq);

    printf("  Property 2: mse_base=%.6f  mse_real=%.6f\n", mse_base, mse_real);

    // Not more than 5% worse than baseline.
    // TODAY: mse_real == mse_base (same algorithm) -> PASSES.
    // AFTER GREEN: grid search includes f=1.0, so mse_real <= mse_base -> PASSES.
    assert(mse_real <= mse_base * 1.05f);
}

// ---------------------------------------------------------------------------
// Property 3 (sanity, PASSES both):
//   All-equal block: exercises the range==0 degenerate branch.
//   Every dequantized value must be finite and equal to the input constant.
// ---------------------------------------------------------------------------
static void test_constant_block_finite(void) {
    const float CVAL = 0.7f;
    float x[QK];
    for (int i = 0; i < QK; i++) x[i] = CVAL;

    float dq[QK];
    real_mse(x, dq);   // result not checked numerically here, only finiteness

    for (int i = 0; i < QK; i++) {
        assert(isfinite(dq[i]));
        // s2 corrects the magnitude: reconstruction must equal CVAL within 1%
        assert(fabsf(dq[i] - CVAL) < fabsf(CVAL) * 0.01f + 1e-4f);
    }
    printf("  Property 3: constant block OK, all finite, dq[0]=%.6f (expected %.6f)\n",
           dq[0], CVAL);
}

int main(void) {
    ggml_cpu_init();

    printf("test-q2-kvarn-scale:\n");

    // Property 1: FAILS today (RED), PASSES after GREEN
    printf("  Test 1 (outlier_block_strict_improvement): ");
    fflush(stdout);
    test_outlier_block_strict_improvement();
    printf("PASSED\n");

    // Property 2: PASSES today and after GREEN
    printf("  Test 2 (smooth_block_no_regression): ");
    fflush(stdout);
    test_smooth_block_no_regression();
    printf("PASSED\n");

    // Property 3: sanity, PASSES both
    printf("  Test 3 (constant_block_finite): ");
    fflush(stdout);
    test_constant_block_finite();
    printf("PASSED\n");

    printf("\n3/3 tests passed\n");
    return 0;
}
