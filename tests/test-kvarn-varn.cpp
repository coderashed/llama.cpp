#include "../src/llama-kvarn.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static float var(const float* data, int n) {
    double mean = 0.0, m2 = 0.0;
    for (int i = 0; i < n; i++) {
        mean += (double)data[i];
    }
    mean /= (double)n;
    for (int i = 0; i < n; i++) {
        double d = (double)data[i] - mean;
        m2 += d * d;
    }
    return (float)(m2 / (double)n);
}

static float maxf(float a, float b) {
    return a > b ? a : b;
}

static float imb(const float* T, int R, int C) {
    float min_col_var = INFINITY, max_col_var = -INFINITY;
    float min_row_var = INFINITY, max_row_var = -INFINITY;

    // Column variances
    for (int j = 0; j < C; j++) {
        std::vector<float> col(R);
        for (int i = 0; i < R; i++) {
            col[i] = T[i * C + j];
        }
        float v = var(col.data(), R);
        if (v < min_col_var) min_col_var = v;
        if (v > max_col_var) max_col_var = v;
    }

    // Row variances
    for (int i = 0; i < R; i++) {
        float v = var(&T[i * C], C);
        if (v < min_row_var) min_row_var = v;
        if (v > max_row_var) max_row_var = v;
    }

    float eps = 1e-8f;
    return (max_col_var / maxf(min_col_var, eps)) * (max_row_var / maxf(min_row_var, eps));
}

int main() {
    const int R = 128;
    const int C = 128;

    // Test 1: Function call does not crash on a uniform tile
    {
        printf("Test 1: Basic function call...\n");
        std::vector<float> T(R * C, 1.0f);
        std::vector<float> S_c(C, 0.0f);
        std::vector<float> S_r(R, 0.0f);
        kvarn_variance_normalize(T.data(), R, C, 12, -5.0f, 5.0f, S_c.data(), S_r.data());
        printf("  PASSED\n");
    }

    // Test 2: Convergence on imbalanced tile
    {
        printf("Test 2: Convergence...\n");
        std::vector<float> T(R * C, 1.0f);
        // Row 0 has 10x larger magnitude => higher variance
        for (int j = 0; j < C; j++) {
            T[0 * C + j] = 10.0f;
        }

        float initial_imb = imb(T.data(), R, C);
        printf("  Initial Imb: %.3f\n", initial_imb);

        std::vector<float> S_c(C, 0.0f);
        std::vector<float> S_r(R, 0.0f);
        kvarn_variance_normalize(T.data(), R, C, 12, -5.0f, 5.0f, S_c.data(), S_r.data());

        float final_imb = imb(T.data(), R, C);
        printf("  Final Imb:   %.3f\n", final_imb);
        assert(final_imb < 2.0f);
        printf("  PASSED\n");
    }

    // Test 3: Scale vectors S_c and S_r are written (not all 1.0)
    {
        printf("Test 3: Scale vectors written...\n");
        std::vector<float> T(R * C, 1.0f);
        // Create imbalance so that normalization is meaningful
        for (int j = 0; j < C; j++) {
            T[0 * C + j] = 10.0f;
        }

        std::vector<float> S_c(C, 0.0f);
        std::vector<float> S_r(R, 0.0f);
        kvarn_variance_normalize(T.data(), R, C, 12, -5.0f, 5.0f, S_c.data(), S_r.data());

        bool sc_written = false;
        for (int j = 0; j < C; j++) {
            if (fabsf(S_c[j] - 1.0f) > 1e-6f) {
                sc_written = true;
                break;
            }
        }
        assert(sc_written);

        bool sr_written = false;
        for (int i = 0; i < R; i++) {
            if (fabsf(S_r[i] - 1.0f) > 1e-6f) {
                sr_written = true;
                break;
            }
        }
        assert(sr_written);

        printf("  PASSED\n");
    }

    // Test 4: Edge case R=1 does not crash
    {
        printf("Test 4: R=1 single row...\n");
        std::vector<float> T(1 * C, 1.0f);
        std::vector<float> S_c(C, 0.0f);
        std::vector<float> S_r(1, 0.0f);
        kvarn_variance_normalize(T.data(), 1, C, 12, -5.0f, 5.0f, S_c.data(), S_r.data());
        printf("  PASSED\n");
    }

    // Test 5: Edge case C=1 does not crash
    {
        printf("Test 5: C=1 single column...\n");
        std::vector<float> T(R * 1, 1.0f);
        std::vector<float> S_c(1, 0.0f);
        std::vector<float> S_r(R, 0.0f);
        kvarn_variance_normalize(T.data(), R, 1, 12, -5.0f, 5.0f, S_c.data(), S_r.data());
        printf("  PASSED\n");
    }

    // Test 6: Edge case K=0 does not crash (no-op)
    {
        printf("Test 6: K=0 no iterations...\n");
        std::vector<float> T(R * C, 1.0f);
        std::vector<float> S_c(C, 0.0f);
        std::vector<float> S_r(R, 0.0f);
        kvarn_variance_normalize(T.data(), R, C, 0, -5.0f, 5.0f, S_c.data(), S_r.data());
        // T should be unchanged when K=0
        for (int i = 0; i < R; i++) {
            for (int j = 0; j < C; j++) {
                assert(T[i * C + j] == 1.0f);
            }
        }
        printf("  PASSED\n");
    }

    // Test 7: Edge case R=1, C=1 does not crash
    {
        printf("Test 7: R=1 C=1 single element...\n");
        std::vector<float> T(1, 42.0f);
        std::vector<float> S_c(1, 0.0f);
        std::vector<float> S_r(1, 0.0f);
        kvarn_variance_normalize(T.data(), 1, 1, 12, -5.0f, 5.0f, S_c.data(), S_r.data());
        printf("  PASSED\n");
    }

    printf("\nAll tests passed!\n");
    return 0;
}
