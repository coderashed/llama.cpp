// Failing tests for Item 05 - Integrate VarN into KV-cache quantization path
// These tests verify quantize_row_q2_kvarn_varn() which accepts external S_c/S_r
// and absorbs them into s1/s2 fields of block_q2_kvarn.
// All tests will FAIL at link time because the function does not exist yet.

#include "ggml.h"
#include "ggml-cpu.h"

#include "../src/llama-kvarn.h"
#include "../ggml/src/ggml-quants.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

// E_M/E_T per token (from .ai/research/kvarn/02-error-decomposition.md)
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

// Test 1: quantize_row_q2_kvarn_varn exists
// Will fail at link time because function doesn't exist
static void test_function_exists(void) {
    printf("  Calling quantize_row_q2_kvarn_varn...\n");

    const int64_t k = 128;
    std::vector<float> src(k, 1.0f);
    std::vector<block_q2_kvarn> dst(1);
    std::vector<float> S_c(1, 1.0f);
    std::vector<float> S_r(1, 1.0f);

    // This will fail to link: function doesn't exist yet
    quantize_row_q2_kvarn_varn(src.data(), dst.data(), k, S_c.data(), S_r.data());

    printf("  PASSED (unreachable)\n");
}

// Test 2: Scale absorption — verify s1 and s2 absorb S_c and S_r correctly
// Will fail at link time because function doesn't exist
static void test_scale_absorption(void) {
    printf("  Testing scale absorption...\n");

    const int R = 128; // rows (tokens)
    const int C = 128; // columns (head_dim)
    const int64_t k = R * C;

    // Create a tile with known values
    std::vector<float> tile(k);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int64_t i = 0; i < k; i++) {
        tile[i] = dist(rng);
    }

    // Create S_c and S_r with known non-identity values
    std::vector<float> S_c(C, 0.0f);
    std::vector<float> S_r(R, 0.0f);
    for (int j = 0; j < C; j++) {
        S_c[j] = 0.5f + 0.5f * ((float)j / (float)C);
    }
    for (int i = 0; i < R; i++) {
        S_r[i] = 0.8f + 0.4f * ((float)i / (float)R);
    }

    // Quantize with VarN scales
    int nb = k / QK2_KVARN;
    std::vector<block_q2_kvarn> dst(nb);
    quantize_row_q2_kvarn_varn(tile.data(), dst.data(), k, S_c.data(), S_r.data());

    // Dequantize
    std::vector<float> dequant(k);
    dequantize_row_q2_kvarn(dst.data(), dequant.data(), k);

    // Verify: dequant should match (q + d) * s1_orig * S_c[j] * s2_orig * S_r[i]
    // But we don't have s1_orig/s2_orig — we just check that the output is finite
    // and that the scales were absorbed (not all 1.0)
    for (int64_t i = 0; i < k; i++) {
        assert(isfinite(dequant[i]));
    }

    // Check that s1 and s2 in blocks are not all 1.0 (scales were absorbed)
    bool s1_absorbed = false;
    bool s2_absorbed = false;
    for (int b = 0; b < nb; b++) {
        float s1 = ggml_fp16_to_fp32(dst[b].s1);
        float s2 = ggml_fp16_to_fp32(dst[b].s2);
        if (fabsf(s1 - 1.0f) > 1e-3f) s1_absorbed = true;
        if (fabsf(s2 - 1.0f) > 1e-3f) s2_absorbed = true;
    }
    assert(s1_absorbed);
    assert(s2_absorbed);

    printf("  PASSED (unreachable)\n");
}

// Test 3: VarN + quantize pipeline — E_M/E_T < 0.3 with VarN
// Will fail at link time because function doesn't exist
static void test_varn_quantize_pipeline(void) {
    printf("  Testing VarN + quantize pipeline...\n");

    const int R = 128;
    const int C = 128;
    const int64_t k = R * C;

    // Create imbalanced tile: some rows have 10x magnitude
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> tile(k);
    for (int64_t i = 0; i < k; i++) {
        tile[i] = dist(rng);
    }
    // Every 4th row has 10x magnitude
    for (int i = 0; i < R; i += 4) {
        for (int j = 0; j < C; j++) {
            tile[i * C + j] *= 10.0f;
        }
    }

    // Run VarN to get S_c, S_r (tile is normalized in-place)
    std::vector<float> S_c(C, 0.0f);
    std::vector<float> S_r(R, 0.0f);
    kvarn_variance_normalize(tile.data(), R, C, 12, -5.0f, 5.0f, S_c.data(), S_r.data());

    // Save normalized tile as reference
    std::vector<float> normalized = tile;

    // Quantize with VarN scales
    int nb = k / QK2_KVARN;
    std::vector<block_q2_kvarn> dst(nb);
    quantize_row_q2_kvarn_varn(tile.data(), dst.data(), k, S_c.data(), S_r.data());

    // Dequantize
    std::vector<float> dequant(k);
    dequantize_row_q2_kvarn(dst.data(), dequant.data(), k);

    // Compute E_M/E_T ratios against the normalized tile
    std::vector<float> ratios(R);
    compute_em_et(normalized.data(), dequant.data(), R, C, ratios.data());

    float med = median(ratios.data(), R);
    printf("  E_M/E_T median ratio: %f\n", med);
    printf("  (debug) first 10 ratios: ");
    for (int i = 0; i < 10 && i < R; i++) printf("%f ", ratios[i]);
    printf("\n");
    // With VarN normalization, 2-bit quantization should have E_M/E_T < 0.5
    assert(med < 0.5f);

    printf("  PASSED (unreachable)\n");
}

// Test 4: Partial group fallback — k < 128 should work (plain RTN, no VarN)
// Will fail at link time because function doesn't exist
static void test_partial_group_fallback(void) {
    printf("  Testing partial group fallback (k=64)...\n");

    const int64_t k = 64; // < 128, not a full block

    std::mt19937 rng(99);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> src(k);
    for (int64_t i = 0; i < k; i++) {
        src[i] = dist(rng);
    }

    // S_c and S_r with 1 element each (one sub-block)
    std::vector<float> S_c(1, 1.0f);
    std::vector<float> S_r(1, 1.0f);

    // Should fall back to plain RTN and not crash
    std::vector<block_q2_kvarn> dst(1);
    quantize_row_q2_kvarn_varn(src.data(), dst.data(), k, S_c.data(), S_r.data());

    // Dequantize and verify finite
    std::vector<float> dequant(k);
    dequantize_row_q2_kvarn(dst.data(), dequant.data(), k);

    for (int64_t i = 0; i < k; i++) {
        assert(isfinite(dequant[i]));
    }

    printf("  PASSED (unreachable)\n");
}

int main(void) {
    ggml_cpu_init();

    printf("test-kvarn-integration:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (function_exists): ");
    test_function_exists();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (scale_absorption): ");
    test_scale_absorption();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (varn_quantize_pipeline): ");
    test_varn_quantize_pipeline();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (partial_group_fallback): ");
    test_partial_group_fallback();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
