// Failing tests for Item 03 - Q2_KVARN Quantization (RTN + dual-scale)
// These tests verify the E_M/E_T error decomposition metric:
//   1. E_M/E_T median ratio < 0.5 (will fail — s2=1.0 identity, VarN missing)
//   2. MSE is finite (NaN/Inf check)
//   3. Per-token worst E_M/E_T < 0.8 (will fail — same reason)

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

// E_M/E_T per token (from .ai/research/kvarn/02-error-decomposition.md)
//   E_M = (||K|| - ||K_dq||)^2                  (magnitude error)
//   E_D = 2*||K||*||K_dq||*(1 - cos(theta))     (directional error)
//   E_T = E_M + E_D                              (total squared error)
//   ratio = E_M / E_T
static void compute_em_et(
    const float * orig, const float * dequant,
    int n_tokens, int dim,
    float * ratios_out) {

    for (int t = 0; t < n_tokens; t++) {
        const float * row_orig = orig    + t * dim;
        const float * row_dq   = dequant + t * dim;

        double norm_K = 0.0, norm_Kdq = 0.0, dot = 0.0;
        for (int j = 0; j < dim; j++) {
            norm_K  += (double)row_orig[j] * (double)row_orig[j];
            norm_Kdq += (double)row_dq[j]   * (double)row_dq[j];
            dot     += (double)row_orig[j] * (double)row_dq[j];
        }
        norm_K  = sqrt(norm_K);
        norm_Kdq = sqrt(norm_Kdq);

        const double E_M = (norm_K - norm_Kdq) * (norm_K - norm_Kdq);

        double cos_theta = dot / (norm_K * norm_Kdq + 1e-30);
        if (cos_theta > 1.0)  cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        const double E_D = 2.0 * norm_K * norm_Kdq * (1.0 - cos_theta);

        const double E_T = E_M + E_D;
        ratios_out[t] = (float)(E_M / (E_T + 1e-30));
    }
}

static float median(float * arr, int n) {
    std::vector<float> v(arr, arr + n);
    std::sort(v.begin(), v.end());
    if (n % 2 == 0) {
        return (v[n/2 - 1] + v[n/2]) / 2.0f;
    }
    return v[n/2];
}

// Test 1: E_M/E_T median ratio < 0.5
// This will FAIL because s2=1.0 (identity) — VarN not yet implemented
static void test_em_et_median_ratio(void) {
    const int n_tokens = 128;
    const int dim      = 128;
    const int64_t nelem = n_tokens * dim;

    // Realistic K-matrix data: random normal + outlier channels
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> src(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        src[i] = dist(rng);
    }
    // Every 16th channel has 5x magnitude (outlier pattern)
    for (int t = 0; t < n_tokens; t++) {
        for (int j = 0; j < dim; j += 16) {
            src[t * dim + j] *= 5.0f;
        }
    }

    size_t row_size = ggml_row_size(GGML_TYPE_Q2_KVARN, dim);
    size_t dst_size = row_size * n_tokens;
    std::vector<uint8_t> dst(dst_size + 64);

    size_t written = ggml_quantize_chunk(
        GGML_TYPE_Q2_KVARN,
        src.data(),
        dst.data(),
        0,
        n_tokens,
        dim,
        NULL);
    assert(written == dst_size);

    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    std::vector<float> dst_f32(nelem);
    tt->to_float(dst.data(), dst_f32.data(), nelem);

    std::vector<float> ratios(n_tokens);
    compute_em_et(src.data(), dst_f32.data(), n_tokens, dim, ratios.data());

    float med = median(ratios.data(), n_tokens);
    printf("  E_M/E_T median ratio: %f\n", med);
    assert(med < 0.5f);
}

// Test 2: MSE is finite (no NaN, no Inf)
static void test_mse_finite(void) {
    const int64_t n_per_row = 128;
    const int64_t nelem = n_per_row;

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 2.0f);
    std::vector<float> src(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        src[i] = dist(rng);
    }

    size_t dst_size = ggml_row_size(GGML_TYPE_Q2_KVARN, n_per_row);
    std::vector<uint8_t> dst(dst_size + 64);

    ggml_quantize_chunk(GGML_TYPE_Q2_KVARN, src.data(), dst.data(), 0, 1, n_per_row, NULL);

    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    std::vector<float> dst_f32(nelem);
    tt->to_float(dst.data(), dst_f32.data(), nelem);

    double sum_sq = 0.0;
    for (int64_t i = 0; i < nelem; i++) {
        double diff = (double)src[i] - (double)dst_f32[i];
        sum_sq += diff * diff;
    }
    float mse = (float)(sum_sq / (double)nelem);

    printf("  MSE: %f\n", mse);
    assert(isfinite(mse));
}

// Test 3: Per-token worst E_M/E_T < 0.8 (upper bound)
// This will FAIL because s2=1.0 (identity) — VarN not yet implemented
static void test_worst_token_em_et(void) {
    const int n_tokens = 32;
    const int dim      = 128;
    const int64_t nelem = n_tokens * dim;

    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> src(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        src[i] = dist(rng);
    }
    for (int t = 0; t < n_tokens; t++) {
        for (int j = 0; j < dim; j += 16) {
            src[t * dim + j] *= 5.0f;
        }
    }

    size_t row_size = ggml_row_size(GGML_TYPE_Q2_KVARN, dim);
    size_t dst_size = row_size * n_tokens;
    std::vector<uint8_t> dst(dst_size + 64);

    ggml_quantize_chunk(GGML_TYPE_Q2_KVARN, src.data(), dst.data(), 0, n_tokens, dim, NULL);

    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    std::vector<float> dst_f32(nelem);
    tt->to_float(dst.data(), dst_f32.data(), nelem);

    std::vector<float> ratios(n_tokens);
    compute_em_et(src.data(), dst_f32.data(), n_tokens, dim, ratios.data());

    float worst = 0.0f;
    for (int t = 0; t < n_tokens; t++) {
        if (ratios[t] > worst) worst = ratios[t];
    }
    printf("  Worst-token E_M/E_T: %f\n", worst);
    assert(worst < 0.8f);
}

int main(void) {
    ggml_cpu_init();

    printf("test-q2-kvarn-quant:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (em_et_median_ratio): ");
    test_em_et_median_ratio();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (mse_finite): ");
    test_mse_finite();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (worst_token_em_et): ");
    test_worst_token_em_et();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
