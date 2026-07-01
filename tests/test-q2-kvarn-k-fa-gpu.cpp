// Device KQ-scores kernel test (KVARN_FAITHFUL, bespoke per-channel FA, increment 2).
// Validates that the on-device channel-major gather in kvarn_kq_scores_cuda reproduces
// the fp64 reference score[q,t] = Sc[t] * sum_c Q[c,q]*(code+z[c])*(s[c]*Sr[c]) that
// test-q2-kvarn-k-fa pinned on the host. Only built for CUDA/HIP.

#include "ggml.h"
#include "ggml-quants.h"    // block_q2_kvarn_k, QG2_KVARN

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

extern "C" void kvarn_kq_scores_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, float * scores, int D, int n_q, int G);
extern "C" void kvarn_fa_group_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, const void * V_f16, const float * mask,
        float * O, int D, int n_q, int G, float scale);

static inline float h2f(ggml_half h) { return ggml_fp16_to_fp32(h); }
static inline ggml_half f2h(float f)  { return ggml_fp32_to_fp16(f); }

int main(void) {
    printf("test-q2-kvarn-k-fa-gpu:\n");
    const int D = 128, G = QG2_KVARN, n_q = 5;
    assert(sizeof(block_q2_kvarn_k) == 36);

    srand(0xFA9E);
    std::vector<block_q2_kvarn_k> K(D);
    std::vector<float> Sr(D), Sc(G), Q((size_t) D * n_q);
    for (int c = 0; c < D; c++) {
        for (int b = 0; b < QG2_KVARN/4; b++) K[c].qs[b] = (uint8_t)(rand() & 0xFF);
        K[c].s = f2h(0.02f + (rand() % 200) * 0.005f);
        K[c].z = f2h(((rand() % 201) - 100) * 0.01f);
        Sr[c]  = h2f(f2h(0.3f + (rand() % 300) * 0.01f));
    }
    for (int t = 0; t < G; t++) Sc[t] = h2f(f2h(0.3f + (rand() % 300) * 0.01f));
    for (int i = 0; i < D * n_q; i++) Q[i] = ((rand() % 201) - 100) * 0.01f;

    std::vector<float> scores((size_t) n_q * G);
    kvarn_kq_scores_cuda(K.data(), Q.data(), Sr.data(), Sc.data(), scores.data(), D, n_q, G);

    float max_rel = 0.0f;
    for (int q = 0; q < n_q; q++) {
        const float * Qq = Q.data() + (size_t) q * D;
        for (int t = 0; t < G; t++) {
            double ref = 0.0;
            for (int c = 0; c < D; c++) {
                const double s = (double) h2f(K[c].s);
                const double z = (double) h2f(K[c].z);
                const int code = (K[c].qs[t >> 2] >> ((t & 3) * 2)) & 0x03;
                ref += (double)Qq[c] * ((double)code + z) * s * (double)Sr[c] * (double)Sc[t];
            }
            float got = scores[(size_t) q * G + t];
            float denom = fabsf((float)ref) > 1e-6f ? fabsf((float)ref) : 1e-6f;
            max_rel = fmaxf(max_rel, fabsf(got - (float)ref) / denom);
        }
    }
    printf("  scores: max rel dev over %d x %d = %.7f\n", n_q, G, max_rel);
    assert(max_rel < 1e-3f);

    // Increment 3: full attention O = softmax(scale*KQ + mask) . V, F16 V, with a
    // causal-style mask so some keys are dropped per query.
    const float scale = 1.0f / sqrtf((float) D);
    std::vector<uint16_t> Vh((size_t) D * G);
    std::vector<float>    Vf((size_t) D * G); // token-major [D,G]: Vf[c + t*D]
    for (int t = 0; t < G; t++) for (int c = 0; c < D; c++) {
        float v = ((rand() % 201) - 100) * 0.01f;
        Vf[(size_t) t * D + c] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
        Vh[(size_t) t * D + c] = ggml_fp32_to_fp16(v);
    }
    std::vector<float> mask((size_t) n_q * G);
    for (int q = 0; q < n_q; q++) for (int t = 0; t < G; t++)
        mask[(size_t) q * G + t] = (t <= (G - n_q + q)) ? 0.0f : -INFINITY; // vary kept range

    std::vector<float> O((size_t) D * n_q);
    kvarn_fa_group_cuda(K.data(), Q.data(), Sr.data(), Sc.data(), Vh.data(),
            mask.data(), O.data(), D, n_q, G, scale);

    float o_max_rel = 0.0f;
    for (int q = 0; q < n_q; q++) {
        const float * Qq = Q.data() + (size_t) q * D;
        // fp64 reference: scores, stable softmax, then V weighted sum.
        std::vector<double> sco(G);
        double mx = -1e300;
        for (int t = 0; t < G; t++) {
            double acc = 0.0;
            for (int c = 0; c < D; c++) {
                const double s = (double) h2f(K[c].s), z = (double) h2f(K[c].z);
                const int code = (K[c].qs[t >> 2] >> ((t & 3) * 2)) & 0x03;
                acc += (double)Qq[c] * ((double)code + z) * s * (double)Sr[c] * (double)Sc[t];
            }
            sco[t] = (double)scale * acc + (double) mask[(size_t) q * G + t];
            if (sco[t] > mx) mx = sco[t];
        }
        double sum = 0.0;
        for (int t = 0; t < G; t++) { sco[t] = exp(sco[t] - mx); sum += sco[t]; }
        for (int c = 0; c < D; c++) {
            double o = 0.0;
            for (int t = 0; t < G; t++) o += sco[t] * (double) Vf[(size_t) t * D + c];
            o /= sum;
            float got = O[(size_t) q * D + c];
            float denom = fabs(o) > 1e-4 ? (float) fabs(o) : 1e-4f;
            o_max_rel = fmaxf(o_max_rel, fabsf(got - (float) o) / denom);
        }
    }
    printf("  attention O: max rel dev over %d x %d = %.7f\n", n_q, D, o_max_rel);
    assert(o_max_rel < 5e-3f);

    printf("\nPASSED\n");
    return 0;
}
