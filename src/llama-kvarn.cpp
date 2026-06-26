#include "llama-kvarn.h"
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define KVARN_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define KVARN_RESTRICT __restrict
#else
#define KVARN_RESTRICT
#endif

static void kvarn_normalize_columns(float* KVARN_RESTRICT T, float* KVARN_RESTRICT S_c, int R, int C, float c_min, float c_max) {
    for (int j = 0; j < C; j++) {
        double sum_abs = 0.0;
        for (int i = 0; i < R; i++) {
            sum_abs += (double)std::abs(T[i * C + j]);
        }
        float mean = (float)(sum_abs / (double)R);
        mean = std::log(mean);
        mean = std::clamp(mean, c_min, c_max);
        float scale = std::exp(mean);

        for (int i = 0; i < R; i++) {
            T[i * C + j] -= mean;
        }
        S_c[j] *= scale;
    }
}

static void kvarn_normalize_rows(float* KVARN_RESTRICT T, float* KVARN_RESTRICT S_r, int R, int C, float c_min, float c_max) {
    for (int i = 0; i < R; i++) {
        double sum_abs = 0.0;
        for (int j = 0; j < C; j++) {
            sum_abs += (double)std::abs(T[i * C + j]);
        }
        float mean = (float)(sum_abs / (double)C);
        mean = std::log(mean);
        mean = std::clamp(mean, c_min, c_max);
        float scale = std::exp(mean);

        for (int j = 0; j < C; j++) {
            T[i * C + j] -= mean;
        }
        S_r[i] *= scale;
    }
}

static float kvarn_imb_metric(const float* KVARN_RESTRICT tile, int R, int C) {
    float min_col_var = FLT_MAX, max_col_var = -FLT_MAX;
    float min_row_var = FLT_MAX, max_row_var = -FLT_MAX;

    for (int j = 0; j < C; j++) {
        double mean = 0.0, m2 = 0.0;
        for (int i = 0; i < R; i++) mean += tile[i * C + j];
        mean /= (double)R;
        for (int i = 0; i < R; i++) { double d = tile[i * C + j] - mean; m2 += d * d; }
        float v = (float)(m2 / (double)R);
        if (v < min_col_var) min_col_var = v;
        if (v > max_col_var) max_col_var = v;
    }

    for (int i = 0; i < R; i++) {
        double mean = 0.0, m2 = 0.0;
        for (int j = 0; j < C; j++) mean += tile[i * C + j];
        mean /= (double)C;
        for (int j = 0; j < C; j++) { double d = tile[i * C + j] - mean; m2 += d * d; }
        float v = (float)(m2 / (double)C);
        if (v < min_row_var) min_row_var = v;
        if (v > max_row_var) max_row_var = v;
    }

    float eps = 1e-8f;
    return (max_col_var / std::max(min_col_var, eps)) *
           (max_row_var / std::max(min_row_var, eps));
}

void kvarn_variance_normalize(
    float* KVARN_RESTRICT T, int R, int C, int K,
    float c_min, float c_max,
    float* KVARN_RESTRICT S_c, float* KVARN_RESTRICT S_r)
{
    if (R <= 0 || C <= 0 || K <= 0) {
        for (int i = 0; i < R; i++) S_r[i] = 1.0f;
        for (int j = 0; j < C; j++) S_c[j] = 1.0f;
        return;
    }

    for (int j = 0; j < C; j++) S_c[j] = 1.0f;
    for (int i = 0; i < R; i++) S_r[i] = 1.0f;

    std::vector<float> best_T(R * C);
    std::vector<float> best_S_c(C);
    std::vector<float> best_S_r(R);
    float best_Imb = FLT_MAX;

    for (int iter = 0; iter < K; iter++) {
        kvarn_normalize_columns(T, S_c, R, C, c_min, c_max);
        kvarn_normalize_rows(T, S_r, R, C, c_min, c_max);

        float curr_Imb = kvarn_imb_metric(T, R, C);
        if (curr_Imb < best_Imb) {
            best_Imb = curr_Imb;
            std::memcpy(best_T.data(), T, (size_t)R * C * sizeof(float));
            std::memcpy(best_S_c.data(), S_c, (size_t)C * sizeof(float));
            std::memcpy(best_S_r.data(), S_r, (size_t)R * sizeof(float));
        }
    }

    std::memcpy(T, best_T.data(), (size_t)R * C * sizeof(float));
    std::memcpy(S_c, best_S_c.data(), (size_t)C * sizeof(float));
    std::memcpy(S_r, best_S_r.data(), (size_t)R * sizeof(float));
}
