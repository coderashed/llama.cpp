// Item 04 premise test (KVARN_FAITHFUL/04): does VarN + scale-fold reduce
// per-channel K quantization error vs no-VarN? This is the falsifiable claim the
// gate needs before wiring VarN into the graph (VarN runs on CPU per the design).
//
// Fold (design 7.1, orientation R=channels/C=tokens):
//   VarN(T[C_ch, C_tok]) -> T_norm, S_c (per-token), S_r (per-channel), with
//   K_orig[ch,t] = T_norm[ch,t] * S_c[t] * S_r[ch].
//   Quantize T_norm per channel (RTN scale s_rtn per channel), fold S_r into the
//   block scale: block.s = s_rtn * S_r[ch]; keep S_c as a per-token side vector.
//   Dequant: (code+z)*block.s * S_c[t] = (code+z)*s_rtn*S_r[ch]*S_c[t] ~= K_orig.

#include "ggml.h"
#include "ggml-quants.h"
#include "../src/llama-kvarn.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <random>

static const int N_CH  = 128; // channels (rows, R)
static const int N_TOK = 128; // tokens   (cols, C)

static double mse(const std::vector<float> & a, const std::vector<float> & b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); i++) { double d = (double)a[i] - (double)b[i]; s += d*d; }
    return s / (double)a.size();
}

// Build a channel-major [N_CH x N_TOK] tile with per-TOKEN magnitude imbalance
// (some tokens ~10x larger) -- the magnitude-outlier regime VarN's S_c targets,
// and exactly what the per-channel-only block (no s2) fails to correct.
static std::vector<float> make_imbalanced_tile(void) {
    std::mt19937 rng(0x5177A11U);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> T(N_CH * N_TOK);
    for (int c = 0; c < N_CH; c++) {
        for (int t = 0; t < N_TOK; t++) {
            const float tok_scale = (t % 16 == 0) ? 10.0f : 1.0f; // heavy tokens
            T[c*N_TOK + t] = nd(rng) * tok_scale;
        }
    }
    return T;
}

int main(void) {
    printf("test-q2-kvarn-k-varn:\n");

    const std::vector<float> K = make_imbalanced_tile();

    // (a) per-channel, NO VarN.
    std::vector<block_q2_kvarn_k> ba(N_CH);
    quantize_row_q2_kvarn_k_ref(K.data(), ba.data(), N_CH, N_TOK);
    std::vector<float> ra(N_CH * N_TOK);
    dequantize_row_q2_kvarn_k(ba.data(), ra.data(), N_CH, N_TOK);
    const double mse_novarn = mse(K, ra);

    // (b) per-channel + VarN fold.
    std::vector<float> Tn = K; // VarN normalizes in place
    std::vector<float> S_c(N_TOK), S_r(N_CH);
    kvarn_variance_normalize(Tn.data(), N_CH, N_TOK, 12, -5.0f, 5.0f, S_c.data(), S_r.data());

    std::vector<block_q2_kvarn_k> bb(N_CH);
    quantize_row_q2_kvarn_k_ref(Tn.data(), bb.data(), N_CH, N_TOK);
    for (int ch = 0; ch < N_CH; ch++) {
        const float s = ggml_fp16_to_fp32(bb[ch].s) * S_r[ch];   // fold S_r (per-channel)
        bb[ch].s = ggml_fp32_to_fp16(s);
    }
    std::vector<float> tmp(N_CH * N_TOK);
    dequantize_row_q2_kvarn_k(bb.data(), tmp.data(), N_CH, N_TOK);
    std::vector<float> rb(N_CH * N_TOK);
    for (int ch = 0; ch < N_CH; ch++)
        for (int t = 0; t < N_TOK; t++)
            rb[ch*N_TOK + t] = tmp[ch*N_TOK + t] * S_c[t];        // apply S_c (per-token)
    const double mse_varn = mse(K, rb);

    printf("  MSE no-VarN (per-channel s/z only): %.5f\n", mse_novarn);
    printf("  MSE VarN     (per-channel + fold):  %.5f\n", mse_varn);
    printf("  ratio varn/novarn: %.4f  (want < 1.0)\n", mse_varn / (mse_novarn + 1e-30));

    // Falsifiable premise: on a per-token magnitude-imbalanced tile, VarN's S_c must
    // materially cut reconstruction error vs the per-channel-only block.
    assert(mse_varn < mse_novarn);

    // Orientation guard: S_c is per-token (length N_TOK) and heavy tokens (t%16==0)
    // should get a LARGER S_c than light tokens.
    double heavy = 0.0, light = 0.0; int nh = 0, nl = 0;
    for (int t = 0; t < N_TOK; t++) {
        if (t % 16 == 0) { heavy += S_c[t]; nh++; } else { light += S_c[t]; nl++; }
    }
    printf("  mean S_c heavy tokens: %.3f  light tokens: %.3f\n", heavy/nh, light/nl);
    assert(heavy/nh > light/nl); // per-token scale tracks per-token magnitude

    printf("\nPASSED\n");
    return 0;
}
