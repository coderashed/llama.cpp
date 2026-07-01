// Tiled fused per-channel FA kernel, increment 1
// (design: .ai/design/kvarn_tiled_fused_fa_design.md).
//
// Validates the COALESCED channel-major K tile load+dequant against an fp64 reference.
// block_q2_kvarn_k is channel-major (one block = 128 tokens' 2-bit codes for one
// channel). The loaded token-major tile element (channel d, token t) must equal
//   (code_{d,t} + z_d) * s_d * Sr_d * Sc_t
// with the VarN scales (Sr per channel, Sc per token) folded into the dequant.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "ggml.h"
#include "ggml-quants.h"   // block_q2_kvarn_k, QG2_KVARN

static float      h2f(ggml_fp16_t h) { return ggml_fp16_to_fp32(h); }
static ggml_fp16_t f2h(float f)      { return ggml_fp32_to_fp16(f); }

extern "C" void kvarn_ktile_load_cuda(const void * K, const float * Sr, const float * Sc,
        float * out, int head_dim, int G, int t0, int T_tile);

int main() {
    assert(sizeof(block_q2_kvarn_k) == 36);
    srand(20260701);

    const int G          = QG2_KVARN;      // 128 tokens per group
    const int head_dims[] = {128, 256};    // llama-class and Qwen3-35B/gemma-class
    const int t0     = 8;                  // interior tile start (spans multiple qs bytes)
    const int T_tile = 16;

    double maxdev = 0.0;
    for (int di = 0; di < 2; ++di) {
        const int D = head_dims[di];
        std::vector<block_q2_kvarn_k> K(D);
        std::vector<float> Sr(D), Sc(G), out((size_t) D * T_tile);

        for (int c = 0; c < D; ++c) {
            for (int b = 0; b < QG2_KVARN/4; ++b) K[c].qs[b] = (uint8_t)(rand() & 0xFF);
            K[c].s = f2h(0.02f + (rand() % 200) * 0.001f);   // RTN scale
            K[c].z = f2h(-1.5f + (rand() % 300) * 0.01f);    // zeropoint
            Sr[c]  = h2f(f2h(0.3f + (rand() % 300) * 0.01f)); // VarN row scale (fp16-rounded)
        }
        for (int t = 0; t < G; ++t) Sc[t] = h2f(f2h(0.3f + (rand() % 300) * 0.01f));

        kvarn_ktile_load_cuda(K.data(), Sr.data(), Sc.data(), out.data(), D, G, t0, T_tile);

        double dmax = 0.0;
        for (int c = 0; c < D; ++c) {
            const double s = (double) h2f(K[c].s);
            const double z = (double) h2f(K[c].z);
            for (int tt = 0; tt < T_tile; ++tt) {
                const int t    = t0 + tt;
                const int code = (K[c].qs[t >> 2] >> ((t & 3) * 2)) & 0x03;
                const double ref   = ((double) code + z) * s * (double) Sr[c] * (double) Sc[t];
                const double got   = (double) out[(size_t) c * T_tile + tt];
                const double denom = std::fabs(ref) > 1e-6 ? std::fabs(ref) : 1e-6;
                const double dev   = std::fabs(got - ref) / denom;
                if (dev > dmax) dmax = dev;
            }
        }
        printf("  head_dim %d: K tile load max rel dev vs fp64 = %.7f\n", D, dmax);
        if (dmax > maxdev) maxdev = dmax;
    }

    assert(maxdev < 1e-3);   // float vs double single-product accumulation
    printf("\nPASSED\n");
    return 0;
}
