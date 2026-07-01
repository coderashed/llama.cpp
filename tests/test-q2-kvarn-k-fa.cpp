// Per-channel K flash-attention dot contract (KVARN_FAITHFUL, fused FA read path).
//
// Pins the formula the fused per-channel K-dot must satisfy BEFORE any device kernel
// is written (design 5.2 test contract #1; project rule: isolate one layer with a
// unit test before touching the live path).
//
// block_q2_kvarn_k is CHANNEL-MAJOR: block c holds 128 tokens' 2-bit codes for channel
// c, plus per-channel s[c], z[c]. The full reconstruction get_k performs is
//   K[c,t] = (code_{c,t} + z[c]) * s[c] * S_r[c] * S_c[t]
// where S_r[c] (per-channel) and S_c[t] (per-token) are the VarN scales. Hence for a
// query Q and key token t the attention dot is
//   KQ[t] = sum_c Q[c] * K[c,t]
//         = S_c[t] * sum_c Q[c] * (code_{c,t} + z[c]) * (s[c] * S_r[c])
// i.e. S_c[t] is pulled once per key column (outside the channel loop) and (s[c]*S_r[c])
// is a per-channel constant. This test asserts that scalar K-dot matches an fp64
// reference Q . dequant_reconstruct(K). Host-only; no GPU required.

#include "ggml.h"           // ggml_fp32_to_fp16 / ggml_fp16_to_fp32
#include "ggml-quants.h"    // pulls ggml-common.h: block_q2_kvarn_k, QG2_KVARN

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

static inline float h2f(ggml_half h) { return ggml_fp16_to_fp32(h); }
static inline ggml_half f2h(float f)  { return ggml_fp32_to_fp16(f); }

// The scalar per-channel K-dot the device kernel must implement, for one key token t
// over D channels. K_blocks is channel-major: K_blocks[c] covers 128 tokens of chan c.
// Sr[c], Sc are the VarN scales (Sc already indexed for this token t by the caller).
static float kdot_q2_kvarn_k(const block_q2_kvarn_k * K_blocks, const float * Q,
        const float * Sr, float Sc, int D, int t) {
    float acc = 0.0f;
    for (int c = 0; c < D; c++) {
        const block_q2_kvarn_k & b = K_blocks[c];
        const float s = h2f(b.s);
        const float z = h2f(b.z);
        const int   code = (b.qs[t >> 2] >> ((t & 3) * 2)) & 0x03;
        acc += Q[c] * ((float)code + z) * (s * Sr[c]);
    }
    return Sc * acc;
}

int main(void) {
    printf("test-q2-kvarn-k-fa:\n");

    const int D = 128;   // head_dim (channels)
    const int G = QG2_KVARN; // 128 tokens per group
    assert(sizeof(block_q2_kvarn_k) == 36);

    srand(0x5EED);

    // Build a channel-major body tile: one block per channel, each with random codes,
    // and fp16-rounded scales/zeropoint (matching stored precision). Give channels a
    // spread of magnitudes so s[c]*Sr[c] genuinely varies.
    std::vector<block_q2_kvarn_k> K(D);
    std::vector<float> Sr(D), Sc(G), Q(D);
    for (int c = 0; c < D; c++) {
        for (int b = 0; b < QG2_KVARN/4; b++) K[c].qs[b] = (uint8_t)(rand() & 0xFF);
        float s = 0.02f + (rand() % 200) * 0.005f;    // per-channel scale
        float z = ((rand() % 201) - 100) * 0.01f;     // zeropoint in quantized units
        K[c].s = f2h(s);
        K[c].z = f2h(z);
        Sr[c] = h2f(f2h(0.3f + (rand() % 300) * 0.01f)); // VarN row scale, fp16-rounded
        Q[c]  = ((rand() % 201) - 100) * 0.01f;
    }
    for (int t = 0; t < G; t++) Sc[t] = h2f(f2h(0.3f + (rand() % 300) * 0.01f));

    // Check every key token in the group.
    float max_rel = 0.0f;
    for (int t = 0; t < G; t++) {
        // fp64 reference: fully reconstruct K[c,t] then dot with Q.
        double ref = 0.0;
        for (int c = 0; c < D; c++) {
            const double s = (double) h2f(K[c].s);
            const double z = (double) h2f(K[c].z);
            const int  code = (K[c].qs[t >> 2] >> ((t & 3) * 2)) & 0x03;
            const double Kct = ((double)code + z) * s * (double)Sr[c] * (double)Sc[t];
            ref += (double)Q[c] * Kct;
        }
        float got = kdot_q2_kvarn_k(K.data(), Q.data(), Sr.data(), Sc[t], D, t);
        float denom = fabsf((float)ref) > 1e-6f ? fabsf((float)ref) : 1e-6f;
        float rel = fabsf(got - (float)ref) / denom;
        if (rel > max_rel) max_rel = rel;
    }
    printf("  max rel dev over %d key tokens = %.7f\n", G, max_rel);
    assert(max_rel < 1e-4f);

    printf("\nPASSED\n");
    return 0;
}
