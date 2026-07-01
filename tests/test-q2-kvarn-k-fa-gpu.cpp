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
extern "C" void kvarn_fa_cuda(const void * Kblocks, const float * Q,
        const float * Sr, const float * Sc, const void * V_f16, const float * mask,
        float * O, int head_dim, int n_head, int n_head_kv, int n_tok, int n_kv, float scale);

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

    // Increment 4: generalized attention -- GQA multi-head + multiple groups.
    {
        const int head_dim = 128, n_head = 4, n_head_kv = 2;
        const int n_tok = 3, n_kv = 256, ng = n_kv / G, C = head_dim * n_head_kv;
        const float gscale = 1.0f / sqrtf((float) head_dim);
        std::vector<block_q2_kvarn_k> gK((size_t) C * ng);
        for (auto & b : gK) {
            for (int j = 0; j < QG2_KVARN/4; j++) b.qs[j] = (uint8_t)(rand() & 0xFF);
            b.s = f2h(0.02f + (rand() % 200) * 0.005f);
            b.z = f2h(((rand() % 201) - 100) * 0.01f);
        }
        std::vector<float> gSr((size_t) head_dim*n_head_kv*ng), gSc((size_t) G*n_head_kv*ng);
        for (auto & x : gSr) x = h2f(f2h(0.3f + (rand() % 300) * 0.01f));
        for (auto & x : gSc) x = h2f(f2h(0.3f + (rand() % 300) * 0.01f));
        std::vector<float> gQ((size_t) head_dim*n_head*n_tok);
        for (auto & x : gQ) x = ((rand() % 201) - 100) * 0.01f;
        std::vector<uint16_t> gVh((size_t) head_dim*n_head_kv*n_kv);
        std::vector<float>    gVf((size_t) head_dim*n_head_kv*n_kv);
        for (size_t i = 0; i < gVh.size(); i++) {
            float v = ((rand() % 201) - 100) * 0.01f;
            gVf[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
            gVh[i] = ggml_fp32_to_fp16(v);
        }
        std::vector<float> gmask((size_t) n_kv*n_tok);
        for (int qt = 0; qt < n_tok; qt++) for (int T = 0; T < n_kv; T++)
            gmask[(size_t) T + n_kv*qt] = (T <= n_kv - n_tok + qt) ? 0.0f : -INFINITY;

        std::vector<float> gO((size_t) head_dim*n_head*n_tok);
        kvarn_fa_cuda(gK.data(), gQ.data(), gSr.data(), gSc.data(), gVh.data(), gmask.data(),
                gO.data(), head_dim, n_head, n_head_kv, n_tok, n_kv, gscale);

        float g_max_rel = 0.0f;
        for (int qt = 0; qt < n_tok; qt++) for (int h = 0; h < n_head; h++) {
            const int hk = h / (n_head / n_head_kv);
            const float * Qq = gQ.data() + (size_t) head_dim * (h + n_head * qt);
            std::vector<double> sco(n_kv); double mx = -1e300;
            for (int T = 0; T < n_kv; T++) {
                const int g = T / G, t = T % G;
                const block_q2_kvarn_k * Kg = gK.data() + (size_t) g * C + (size_t) hk * head_dim;
                double acc = 0.0;
                for (int d = 0; d < head_dim; d++) {
                    const double s = (double) h2f(Kg[d].s), z = (double) h2f(Kg[d].z);
                    const int code = (Kg[d].qs[t >> 2] >> ((t & 3) * 2)) & 0x03;
                    const double Sr_ = gSr[(size_t) head_dim*(hk + n_head_kv*g) + d];
                    acc += (double)Qq[d] * ((double)code + z) * s * Sr_;
                }
                const double Sc_ = gSc[(size_t) G*(hk + n_head_kv*g) + t];
                sco[T] = (double)gscale * Sc_ * acc + (double) gmask[(size_t) T + n_kv*qt];
                if (sco[T] > mx) mx = sco[T];
            }
            double sum = 0.0;
            for (int T = 0; T < n_kv; T++) { sco[T] = exp(sco[T] - mx); sum += sco[T]; }
            for (int d = 0; d < head_dim; d++) {
                double o = 0.0;
                for (int T = 0; T < n_kv; T++)
                    o += sco[T] * (double) gVf[(size_t) head_dim*(hk + n_head_kv*T) + d];
                o /= sum;
                float got = gO[(size_t) head_dim*(h + n_head*qt) + d];
                float denom = fabs(o) > 1e-4 ? (float) fabs(o) : 1e-4f;
                g_max_rel = fmaxf(g_max_rel, fabsf(got - (float) o) / denom);
            }
        }
        printf("  generalized (GQA %dx%d, %d groups): max rel dev = %.7f\n",
                n_head, n_head_kv, ng, g_max_rel);
        assert(g_max_rel < 5e-3f);
    }

    printf("\nPASSED\n");
    return 0;
}
