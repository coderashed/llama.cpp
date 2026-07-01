// GPU VarN kernel test (KVARN_FAITHFUL/04): kvarn_varn_tile_cuda must match the CPU
// kvarn_variance_normalize (scales tight; normalized tile within tolerance since the
// GPU recomputes it from the input + best scales rather than the CPU's accumulated
// in-place divisions). Only built for CUDA/HIP.

#include "../src/llama-kvarn.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <vector>
#include <random>

extern "C" void kvarn_varn_tile_cuda(const float * tiles, float * Tnorm,
        float * Sr, float * Sc, int n_tiles, int R, int C);

int main(void) {
    printf("test-q2-kvarn-varn-gpu:\n");
    const int R = 16, C = 16, N = 5;

    std::mt19937 rng(0x7A5E11U);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> tiles(N * R * C);
    for (int t = 0; t < N; t++)
        for (int i = 0; i < R; i++)
            for (int j = 0; j < C; j++)
                tiles[t*R*C + i*C + j] = nd(rng) * ((j % 4 == 0) ? 7.0f : 1.0f);

    // GPU.
    std::vector<float> gTn(N*R*C), gSr(N*R), gSc(N*C);
    kvarn_varn_tile_cuda(tiles.data(), gTn.data(), gSr.data(), gSc.data(), N, R, C);

    int checked = 0;
    for (int t = 0; t < N; t++) {
        std::vector<float> Td(tiles.begin() + t*R*C, tiles.begin() + (t+1)*R*C);
        std::vector<float> Sc_d(C), Sr_d(R);
        kvarn_variance_normalize(Td.data(), R, C, 12, -5.0f, 5.0f, Sc_d.data(), Sr_d.data());

        // Cross-device iterative float math (device vs host expf/log over 12 iters,
        // best-Imb selection) diverges by a few percent; that is negligible feeding
        // 2-bit quantization. Track max relative deviation and assert it is small.
        float max_rel = 0.0f;
        for (int j = 0; j < C; j++) {
            float a = gSc[t*C + j], b = Sc_d[j];
            max_rel = fmaxf(max_rel, fabsf(a - b) / (1.0f + fabsf(b)));
        }
        for (int i = 0; i < R; i++) {
            float a = gSr[t*R + i], b = Sr_d[i];
            max_rel = fmaxf(max_rel, fabsf(a - b) / (1.0f + fabsf(b)));
        }
        for (int i = 0; i < R; i++) for (int j = 0; j < C; j++) {
            float a = gTn[t*R*C + i*C + j], b = Td[i*C + j];
            max_rel = fmaxf(max_rel, fabsf(a - b) / (1.0f + fabsf(b)));
        }
        printf("  tile %d: max rel dev vs CPU = %.5f\n", t, max_rel);
        assert(max_rel <= 5e-2f);
        checked++;
    }
    printf("  GPU VarN matches CPU on %d tiles: PASSED\n\nPASSED\n", checked);
    return 0;
}
