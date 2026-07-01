#include "llama-kvarn.h"
#include "ggml.h"
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

// Population variance below this is treated as zero.
static const double KVARN_VAR_EPS = 1e-12;

// Column pass: for each column j, compute population variance.
// If variance <= VAR_EPS (uniform column), fall back to (col_mean - tile_mean)^2.
// If that is also <= VAR_EPS (globally uniform tile), skip (no-op).
// Otherwise delta = 0.5*log(v); clamp accumulated log-scale; divide column by exp(eff).
// Invariant: orig[r,c] == T[r,c] * exp(L_c[c]) * exp(L_r[r]) is preserved per step.
static void kvarn_normalize_columns(
    float* KVARN_RESTRICT T,
    float* KVARN_RESTRICT L_c,
    int R, int C,
    float c_min, float c_max)
{
    // Tile mean for fallback when a column is uniform but differs from the tile average.
    double tile_mean = 0.0;
    for (int k = 0; k < R * C; k++) {
        tile_mean += (double)T[k];
    }
    tile_mean /= (double)(R * C);

    for (int j = 0; j < C; j++) {
        double mean = 0.0;
        for (int i = 0; i < R; i++) {
            mean += (double)T[i * C + j];
        }
        mean /= (double)R;

        double m2 = 0.0;
        for (int i = 0; i < R; i++) {
            double d = (double)T[i * C + j] - mean;
            m2 += d * d;
        }
        double v = m2 / (double)R;

        if (v <= KVARN_VAR_EPS) {
            // Column is uniform; fall back to squared deviation of mean from tile mean.
            double dev = mean - tile_mean;
            v = dev * dev;
            if (v <= KVARN_VAR_EPS) {
                // Globally uniform or single-element: pure no-op.
                continue;
            }
        }

        double delta  = 0.5 * std::log(v);
        float  L_new  = std::clamp((float)((double)L_c[j] + delta), c_min, c_max);
        float  eff    = L_new - L_c[j];
        float  f      = std::exp(eff);

        for (int i = 0; i < R; i++) {
            T[i * C + j] /= f;
        }
        L_c[j] = L_new;
    }
}

// Row pass: symmetric to column pass with rows and L_r.
static void kvarn_normalize_rows(
    float* KVARN_RESTRICT T,
    float* KVARN_RESTRICT L_r,
    int R, int C,
    float c_min, float c_max)
{
    // Tile mean for fallback when a row is uniform but differs from the tile average.
    double tile_mean = 0.0;
    for (int k = 0; k < R * C; k++) {
        tile_mean += (double)T[k];
    }
    tile_mean /= (double)(R * C);

    for (int i = 0; i < R; i++) {
        double mean = 0.0;
        for (int j = 0; j < C; j++) {
            mean += (double)T[i * C + j];
        }
        mean /= (double)C;

        double m2 = 0.0;
        for (int j = 0; j < C; j++) {
            double d = (double)T[i * C + j] - mean;
            m2 += d * d;
        }
        double v = m2 / (double)C;

        if (v <= KVARN_VAR_EPS) {
            // Row is uniform; fall back to squared deviation of mean from tile mean.
            double dev = mean - tile_mean;
            v = dev * dev;
            if (v <= KVARN_VAR_EPS) {
                // Globally uniform or single-element: pure no-op.
                continue;
            }
        }

        double delta  = 0.5 * std::log(v);
        float  L_new  = std::clamp((float)((double)L_r[i] + delta), c_min, c_max);
        float  eff    = L_new - L_r[i];
        float  f      = std::exp(eff);

        for (int j = 0; j < C; j++) {
            T[i * C + j] /= f;
        }
        L_r[i] = L_new;
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

void kvarn_varn_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth; (void) userdata;
    if (ith != 0) {
        return; // single-task: VarN's best-Imb snapshot is not tile-parallel
    }
    const struct ggml_tensor * src = dst->src[0];
    // src is [n_tok, head_dim, n_head]. A 2D input (n_head=1) means whole-tile VarN
    // over all channels; a 3D input runs VarN per head (paper: 128 head-dim x 128 tok).
    const int n_tok    = (int) src->ne[0];  // VarN columns C (tokens)
    const int head_dim = (int) src->ne[1];  // VarN rows R (channels per head)
    const int n_head   = (int) src->ne[2];
    const int n_ch     = head_dim * n_head;
    const size_t nt    = (size_t) n_ch * n_tok;

    const float * Tin = (const float *) src->data;
    float * out = (float *) dst->data;   // packed [T_norm(nt) ++ S_r(n_ch) ++ S_c(n_head*n_tok)]

    std::memcpy(out, Tin, nt * sizeof(float));
    float * S_r = out + nt;              // [n_ch] per-channel row scales
    float * S_c = S_r + n_ch;            // [n_head*n_tok] per-head per-token column scales
    for (int h = 0; h < n_head; h++) {
        float * Th = out + (size_t) h * head_dim * n_tok; // [head_dim, n_tok] sub-tile, in place
        kvarn_variance_normalize(Th, head_dim, n_tok, 12, -5.0f, 5.0f,
                                 S_c + (size_t) h * n_tok,
                                 S_r + (size_t) h * head_dim);
    }
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

    // Log-domain accumulated scales; S_c[j] = exp(L_c[j]), S_r[i] = exp(L_r[i]).
    std::vector<float> L_c(C, 0.0f);
    std::vector<float> L_r(R, 0.0f);

    // Best-state snapshot: T, L_c, L_r must be snapshotted together to preserve invariant.
    std::vector<float> best_T(R * C);
    std::vector<float> best_L_c(C, 0.0f);
    std::vector<float> best_L_r(R, 0.0f);
    float best_Imb = FLT_MAX;

    for (int iter = 0; iter < K; iter++) {
        kvarn_normalize_columns(T, L_c.data(), R, C, c_min, c_max);
        kvarn_normalize_rows(T, L_r.data(), R, C, c_min, c_max);

        float curr_Imb = kvarn_imb_metric(T, R, C);
        if (curr_Imb < best_Imb) {
            best_Imb = curr_Imb;
            std::memcpy(best_T.data(),   T,          (size_t)R * C * sizeof(float));
            std::memcpy(best_L_c.data(), L_c.data(), (size_t)C     * sizeof(float));
            std::memcpy(best_L_r.data(), L_r.data(), (size_t)R     * sizeof(float));
        }
    }

    // Restore best snapshot and emit multiplicative scales.
    std::memcpy(T, best_T.data(), (size_t)R * C * sizeof(float));

    for (int j = 0; j < C; j++) {
        S_c[j] = std::exp(best_L_c[j]);
    }
    for (int i = 0; i < R; i++) {
        S_r[i] = std::exp(best_L_r[i]);
    }
}
