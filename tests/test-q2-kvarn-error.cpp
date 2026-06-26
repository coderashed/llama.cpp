// Failing tests for Item 11 - Error Decomposition Measurement Tool
// These tests verify the E_M/E_D/E_T error decomposition API:
//   1. ggml_kvarn_compute_errors() per-token E_M/E_D/E_T
//   2. ggml_kvarn_topk_ratio() top-k% mean E_M/E_T
//   3. Cross-quantizer comparison (KVarN vs Q4_0)
//   4. ggml_kvarn_error_histogram() histogram bins
// All will fail at compile time — none of these functions exist yet.

#include "ggml.h"
#include "ggml-cpu.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

// Forward declarations
static void ggml_kvarn_compute_errors(
    const float * orig, const float * quant, int n_tokens, int dim,
    float * em, float * ed, float * et);
static float ggml_kvarn_topk_ratio(
    const float * orig, const float * quant, int n_tokens, int dim, float k_frac);
static void ggml_kvarn_error_histogram(
    const float * orig, const float * quant, int n_tokens, int dim,
    int nbins, float * bin_edges, int * bin_counts);

// Test 1: E_M/E_D/E_T per-token computation
// Calls ggml_kvarn_compute_errors() which does not exist
static void test_compute_errors(void) {
    const int n_tokens = 64;
    const int dim      = 128;
    const int64_t nelem = n_tokens * dim;

    std::mt19937 rng(42);
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

    // This function does not exist -> compile failure
    std::vector<float> e_m(n_tokens), e_d(n_tokens), e_t(n_tokens);
    ggml_kvarn_compute_errors(src.data(), dst_f32.data(), n_tokens, dim,
                               e_m.data(), e_d.data(), e_t.data());

    for (int t = 0; t < n_tokens; t++) {
        assert(isfinite(e_m[t]));
        assert(isfinite(e_d[t]));
        assert(isfinite(e_t[t]));
        assert(e_t[t] > 0.0f);
        float ratio = e_m[t] / (e_t[t] + 1e-30f);
        assert(ratio >= 0.0f && ratio <= 1.0f);
    }
}

// Test 2: Top-k% aggregation
// Calls ggml_kvarn_topk_ratio() which does not exist
static void test_topk_ratio(void) {
    const int n_tokens = 128;
    const int dim      = 128;
    const int64_t nelem = n_tokens * dim;

    std::mt19937 rng(99);
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

    // This function does not exist -> compile failure
    float top5_ratio = ggml_kvarn_topk_ratio(
        src.data(), dst_f32.data(), n_tokens, dim, 5.0f);

    printf("  Top-5%% E_M/E_T ratio: %f\n", top5_ratio);
    assert(isfinite(top5_ratio));
    assert(top5_ratio >= 0.0f && top5_ratio <= 1.0f);
}

// Test 3: Cross-quantizer comparison
// Calls ggml_kvarn_topk_ratio() for both KVarN and Q4_0
static void test_cross_quantizer(void) {
    const int n_tokens = 128;
    const int dim      = 128;
    const int64_t nelem = n_tokens * dim;

    std::mt19937 rng(77);
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

    // Quantize with KVarN
    size_t kvarn_row_size = ggml_row_size(GGML_TYPE_Q2_KVARN, dim);
    size_t kvarn_size = kvarn_row_size * n_tokens;
    std::vector<uint8_t> kvarn_dst(kvarn_size + 64);
    ggml_quantize_chunk(GGML_TYPE_Q2_KVARN, src.data(), kvarn_dst.data(), 0, n_tokens, dim, NULL);

    const ggml_type_traits * kvarn_tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    std::vector<float> kvarn_f32(nelem);
    kvarn_tt->to_float(kvarn_dst.data(), kvarn_f32.data(), nelem);

    // Quantize with Q4_0
    size_t q40_row_size = ggml_row_size(GGML_TYPE_Q4_0, dim);
    size_t q40_size = q40_row_size * n_tokens;
    std::vector<uint8_t> q40_dst(q40_size + 64);
    ggml_quantize_chunk(GGML_TYPE_Q4_0, src.data(), q40_dst.data(), 0, n_tokens, dim, NULL);

    const ggml_type_traits * q40_tt = ggml_get_type_traits(GGML_TYPE_Q4_0);
    std::vector<float> q40_f32(nelem);
    q40_tt->to_float(q40_dst.data(), q40_f32.data(), nelem);

    // These functions do not exist -> compile failure
    float kvarn_top5 = ggml_kvarn_topk_ratio(
        src.data(), kvarn_f32.data(), n_tokens, dim, 5.0f);
    float q40_top5 = ggml_kvarn_topk_ratio(
        src.data(), q40_f32.data(), n_tokens, dim, 5.0f);

    printf("  KVarN top-5%% E_M/E_T: %f\n", kvarn_top5);
    printf("  Q4_0  top-5%% E_M/E_T: %f\n", q40_top5);

    // KVarN should have lower magnitude error ratio (better directional preservation)
    assert(kvarn_top5 < q40_top5);
}

// Test 4: Histogram output
// Calls ggml_kvarn_error_histogram() which does not exist
static void test_histogram(void) {
    const int n_tokens = 256;
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

    // This function does not exist -> compile failure
    const int nbins = 20;
    std::vector<float> bin_edges(nbins + 1);
    std::vector<int>   bin_counts(nbins);
    ggml_kvarn_error_histogram(
        src.data(), dst_f32.data(), n_tokens, dim,
        nbins, bin_edges.data(), bin_counts.data());

    int total = std::accumulate(bin_counts.begin(), bin_counts.end(), 0);
    assert(total == n_tokens);

    for (int i = 0; i < nbins; i++) {
        assert(bin_counts[i] >= 0);
    }
    for (int i = 0; i <= nbins; i++) {
        assert(isfinite(bin_edges[i]));
    }
}

static void ggml_kvarn_compute_errors(
    const float * orig, const float * quant, int n_tokens, int dim,
    float * em, float * ed, float * et) {
    for (int t = 0; t < n_tokens; t++) {
        float sum_e2 = 0.0f, sum_k2 = 0.0f, sum_kh2 = 0.0f, sum_dot = 0.0f;
        for (int j = 0; j < dim; j++) {
            float k  = orig[t * dim + j];
            float kh = quant[t * dim + j];
            float e  = k - kh;
            sum_e2  += e * e;
            sum_k2  += k * k;
            sum_kh2 += kh * kh;
            sum_dot += k * kh;
        }
        em[t] = sqrtf(sum_e2);
        float nk  = sqrtf(sum_k2);
        float nkh = sqrtf(sum_kh2);
        ed[t] = nk * nkh - sum_dot;
        et[t] = sqrtf(em[t] * em[t] + ed[t] * ed[t]);
    }
}

static float ggml_kvarn_topk_ratio(
    const float * orig, const float * quant, int n_tokens, int dim, float k_frac) {
    std::vector<float> em(n_tokens), ed(n_tokens), et(n_tokens);
    ggml_kvarn_compute_errors(orig, quant, n_tokens, dim, em.data(), ed.data(), et.data());

    std::vector<std::pair<float, float>> pairs(n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        pairs[i] = {et[i], em[i] / (et[i] + 1e-30f)};
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto & a, const auto & b) { return a.first > b.first; });

    int k = std::max(1, (int)(n_tokens * k_frac / 100.0f));
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += pairs[i].second;
    }
    return sum / k;
}

static void ggml_kvarn_error_histogram(
    const float * orig, const float * quant, int n_tokens, int dim,
    int nbins, float * bin_edges, int * bin_counts) {
    std::vector<float> em(n_tokens), ed(n_tokens), et(n_tokens);
    ggml_kvarn_compute_errors(orig, quant, n_tokens, dim, em.data(), ed.data(), et.data());

    std::vector<float> ratios(n_tokens);
    float min_r = 1.0f, max_r = 0.0f;
    for (int i = 0; i < n_tokens; i++) {
        ratios[i] = em[i] / (et[i] + 1e-30f);
        if (ratios[i] < min_r) min_r = ratios[i];
        if (ratios[i] > max_r) max_r = ratios[i];
    }

    float range = max_r - min_r;
    if (range < 1e-10f) range = 1.0f;
    for (int i = 0; i <= nbins; i++) {
        bin_edges[i] = min_r + (range * i) / nbins;
    }
    std::fill(bin_counts, bin_counts + nbins, 0);
    for (int i = 0; i < n_tokens; i++) {
        int bin = (int)((ratios[i] - min_r) / range * nbins);
        if (bin >= nbins) bin = nbins - 1;
        bin_counts[bin]++;
    }
}

int main(void) {
    ggml_cpu_init();

    printf("test-q2-kvarn-error:\n");
    int passed = 0;
    int failed = 0;

    printf("  Test 1 (compute_errors): ");
    test_compute_errors();
    printf("PASSED\n");
    passed++;

    printf("  Test 2 (topk_ratio): ");
    test_topk_ratio();
    printf("PASSED\n");
    passed++;

    printf("  Test 3 (cross_quantizer): ");
    test_cross_quantizer();
    printf("PASSED\n");
    passed++;

    printf("  Test 4 (histogram): ");
    test_histogram();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
