// RED phase tests for Phase A of the faithful KVarN build (KVARN_FAITHFUL/03).
//
// Spec: .ai/design/kvarn_faithful_build_design.md, Section 4.
//
// Test contract (four tests):
//   1. sizeof(block_q2_kvarn_k) == 36   [format pin]
//   2. Round-trip a [128 ch x 128 tok] tile with a per-channel outlier:
//      reconstruction MSE per-channel < per-token (axis advantage). [falsifiable]
//   3. CPU vs CUDA bit-identical s, z, qs on the same tile.          [mirror guard]
//   4. Degenerate channel (all equal): s=1/code-0 convention,
//      dequant reconstructs the constant exactly within FP16 error.   [edge case]
//
// RED failure: Tests 2, 3, 4 abort with "not implemented" from the stubs in
// ggml-quants.c because quantize_row_q2_kvarn_k_ref and
// dequantize_row_q2_kvarn_k have not been implemented yet.
//
// GREEN: implement the MSE-clip per-channel quantizer (02b grid,
// F={1.0,0.9,0.8,0.7,0.6}, no s2 term) over the token axis, and the
// matching CUDA tile kernel quantize_k_q2_kvarn_perchannel_tile in
// ggml/src/ggml-cuda/cpy-utils.cuh.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#if defined(GGML_CUDA) || defined(GGML_HIP)
#include "ggml-cuda.h"
#include "ggml-backend.h"
#endif

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

// Tile dimensions (matches the design: 128 channels x 128 tokens).
static const int N_CH  = 128;  /* head-dim channels */
static const int N_TOK = 128;  /* tokens per group (QG2_KVARN) */

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Compute MSE between two float arrays of length n.
static double compute_mse(const float * a, const float * b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        acc += d * d;
    }
    return acc / (double)n;
}

// Transpose a [rows x cols] row-major matrix into [cols x rows].
static void transpose(const float * src, float * dst, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: sizeof(block_q2_kvarn_k) == 36
//
// The struct is defined in ggml-common.h with a static_assert; this runtime
// assert documents the same invariant in the test log.
// PASSES in RED once the minimal struct declaration is added (no implementation
// needed for a struct -- the layout IS the spec).
// ---------------------------------------------------------------------------
static void test_sizeof_block(void) {
    // static_assert in ggml-common.h already catches compile-time violations.
    // The runtime assert here documents the contract in ctest output.
    assert(sizeof(block_q2_kvarn_k) == 36);

    // Field layout spot-checks: qs must be the first 32 bytes.
    block_q2_kvarn_k blk;
    assert((char *)blk.qs  - (char *)&blk == 0);
    assert((char *)&blk.s  - (char *)&blk == 32);
    assert((char *)&blk.z  - (char *)&blk == 34);

    // QG2_KVARN must equal 128 (the paper's group size G).
    assert(QG2_KVARN == 128);

    // bpw: 36 bytes * 8 bits / 128 tokens = 2.25 (cheaper than per-token 2.375).
    float bpw = (float)(sizeof(block_q2_kvarn_k) * 8) / (float)QG2_KVARN;
    assert(fabsf(bpw - 2.25f) < 1e-6f);
}

// ---------------------------------------------------------------------------
// Test 2: per-channel reconstruction MSE < per-token MSE on a channel-outlier tile.
//
// Construction:
//   - Tile [N_CH x N_TOK] channel-major.  Row ch = N_TOK consecutive token values.
//   - Channel 0 has values 10x larger than channels 1..127 (persistent per-channel
//     outlier -- same magnitude spike present in ALL 128 tokens of that channel).
//   - Per-token quantizer is blind to this: the outlier in channel 0 dominates
//     the scale for every token, compressing channels 1..127 into noise.
//   - Per-channel quantizer handles channel 0 with its own scale and channels
//     1..127 each with their own (smaller) scale -> lower overall MSE.
//
// FAILS in RED: quantize_row_q2_kvarn_k_ref aborts.
// ---------------------------------------------------------------------------
static void test_round_trip_outlier(void) {
    // Build channel-major tile.
    std::mt19937 rng(0xB0AA7777U);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> tile(N_CH * N_TOK);
    for (int ch = 0; ch < N_CH; ch++) {
        float scale = (ch == 0) ? 10.0f : 1.0f;  /* channel 0 is the outlier */
        for (int tok = 0; tok < N_TOK; tok++) {
            tile[ch * N_TOK + tok] = dist(rng) * scale;
        }
    }

    // --- Per-channel MSE (new path) ---
    std::vector<block_q2_kvarn_k> kblocks(N_CH);
    // ABORTS IN RED (stub): not yet implemented.
    quantize_row_q2_kvarn_k_ref(tile.data(), kblocks.data(), N_CH, N_TOK);

    std::vector<float> dq_perchannel(N_CH * N_TOK);
    dequantize_row_q2_kvarn_k(kblocks.data(), dq_perchannel.data(), N_CH, N_TOK);

    double mse_perchannel = compute_mse(tile.data(), dq_perchannel.data(), N_CH * N_TOK);

    // --- Per-token MSE (existing path for comparison) ---
    // The per-token quantizer operates on token-major rows [N_TOK x N_CH].
    // Transpose tile to token-major for the existing API.
    std::vector<float> tile_tokenmajor(N_CH * N_TOK);
    transpose(tile.data(), tile_tokenmajor.data(), N_CH, N_TOK);

    size_t row_size = ggml_row_size(GGML_TYPE_Q2_KVARN, N_CH);
    std::vector<uint8_t> qbuf(row_size * N_TOK + 64);
    ggml_quantize_chunk(GGML_TYPE_Q2_KVARN,
        tile_tokenmajor.data(), qbuf.data(),
        0, N_TOK, N_CH, NULL);

    const ggml_type_traits * tt = ggml_get_type_traits(GGML_TYPE_Q2_KVARN);
    std::vector<float> dq_pertoken_tm(N_CH * N_TOK);
    tt->to_float(qbuf.data(), dq_pertoken_tm.data(), N_CH * N_TOK);

    // Transpose dequantized per-token result back to channel-major for fair comparison.
    std::vector<float> dq_pertoken(N_CH * N_TOK);
    transpose(dq_pertoken_tm.data(), dq_pertoken.data(), N_TOK, N_CH);

    double mse_pertoken = compute_mse(tile.data(), dq_pertoken.data(), N_CH * N_TOK);

    printf("    per-channel MSE: %f\n", (float)mse_perchannel);
    printf("    per-token   MSE: %f\n", (float)mse_pertoken);
    printf("    ratio (perchannel/pertoken): %f  (must be < 1.0)\n",
           (float)(mse_perchannel / (mse_pertoken + 1e-30)));

    // Per-channel MUST beat per-token on a channel-imbalanced tile.
    // This is the axis-advantage property the paper claims.
    assert(mse_perchannel < mse_pertoken);
}

// ---------------------------------------------------------------------------
// Test 3: CPU vs CUDA bit-identical s, z, qs on the same tile.
//
// Contract: the CUDA tile kernel quantize_k_q2_kvarn_perchannel_tile
// (ggml/src/ggml-cuda/cpy-utils.cuh) must produce bit-identical block_q2_kvarn_k
// output compared to quantize_row_q2_kvarn_k_ref on the same float tile.
// This guards the "mirror invariant" discipline established in 02b.
//
// CUDA path: in RED, the CUDA kernel does not exist yet. When CUDA is present
// the test aborts with "not implemented". When CUDA is absent, the test skips.
// GREEN: implement quantize_k_q2_kvarn_perchannel_tile and a C-linkage host
// wrapper that allocates GPU buffers, runs the kernel, and copies results back.
// Then replace the GGML_ABORT below with the real comparison.
// ---------------------------------------------------------------------------
static void test_cpu_cuda_identical(void) {
#if defined(GGML_CUDA) || defined(GGML_HIP)
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("SKIP (no CUDA/HIP device found at runtime)\n");
        return;
    }
    ggml_backend_free(cuda_backend);

    // Build a deterministic tile (same seed as Test 2).
    std::mt19937 rng(0xB0AA7777U);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> tile(N_CH * N_TOK);
    for (int ch = 0; ch < N_CH; ch++) {
        float scale = (ch == 0) ? 10.0f : 1.0f;
        for (int tok = 0; tok < N_TOK; tok++) {
            tile[ch * N_TOK + tok] = dist(rng) * scale;
        }
    }

    // CPU reference.
    std::vector<block_q2_kvarn_k> cpu_blocks(N_CH);
    quantize_row_q2_kvarn_k_ref(tile.data(), cpu_blocks.data(), N_CH, N_TOK);

    // CUDA path: GREEN must implement quantize_k_q2_kvarn_perchannel_tile in
    // ggml/src/ggml-cuda/cpy-utils.cuh and expose a host wrapper here.
    // For now this documents the RED contract.
    //
    // GREEN TODO: call the wrapper, copy results to cuda_blocks, then assert:
    //   for (int ch = 0; ch < N_CH; ch++) {
    //       assert(cpu_blocks[ch].s == cuda_blocks[ch].s);
    //       assert(cpu_blocks[ch].z == cuda_blocks[ch].z);
    //       assert(memcmp(cpu_blocks[ch].qs, cuda_blocks[ch].qs,
    //                     sizeof(cpu_blocks[ch].qs)) == 0);
    //   }
    GGML_ABORT("CUDA per-channel tile quantizer not yet implemented -- Phase A GREEN");
#else
    printf("SKIP (not a CUDA/HIP build)\n");
#endif
}

// ---------------------------------------------------------------------------
// Test 4: Degenerate channel (all tokens equal) uses s=1/code-0 convention.
//
// For a channel where all 128 token values equal constant C:
//   range = 0 -> special case: s=1.0, z=C (all codes = 0).
//   Dequant: (0 + C) * 1.0 = C.  Reconstruction exact within FP16 round-trip.
//
// Also tests that the quantizer does not NaN/Inf on degenerate input.
//
// FAILS in RED: quantize_row_q2_kvarn_k_ref aborts.
// ---------------------------------------------------------------------------
static void test_degenerate_channel(void) {
    // Tile where ALL channels are degenerate: channel ch has constant value ch+1
    // (use distinct values so we can check each channel independently).
    const int n_ch  = N_CH;
    const int n_tok = N_TOK;

    std::vector<float> tile(n_ch * n_tok);
    for (int ch = 0; ch < n_ch; ch++) {
        float cval = (float)(ch + 1) * 0.25f;  /* distinct constant per channel */
        for (int tok = 0; tok < n_tok; tok++) {
            tile[ch * n_tok + tok] = cval;
        }
    }

    std::vector<block_q2_kvarn_k> blocks(n_ch);
    // ABORTS IN RED (stub): not yet implemented.
    quantize_row_q2_kvarn_k_ref(tile.data(), blocks.data(), n_ch, n_tok);

    // Inspect each block: s should be 1.0 (within FP16), all codes should be 0.
    for (int ch = 0; ch < n_ch; ch++) {
        float s_val = ggml_fp16_to_fp32(blocks[ch].s);
        assert(isfinite(s_val));
        assert(fabsf(s_val - 1.0f) < 1e-3f);  /* s == 1.0 FP16 */
        for (int i = 0; i < (int)(QG2_KVARN / 4); i++) {
            assert(blocks[ch].qs[i] == 0x00);  /* all codes zero */
        }
    }

    // Dequantize and verify reconstruction within FP16 error.
    std::vector<float> dq(n_ch * n_tok);
    dequantize_row_q2_kvarn_k(blocks.data(), dq.data(), n_ch, n_tok);

    for (int ch = 0; ch < n_ch; ch++) {
        float cval = (float)(ch + 1) * 0.25f;
        for (int tok = 0; tok < n_tok; tok++) {
            float reconstructed = dq[ch * n_tok + tok];
            assert(isfinite(reconstructed));
            // FP16 round-trip of cval bounds the absolute error.
            float cval_fp16 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(cval));
            assert(fabsf(reconstructed - cval_fp16) < 1e-3f * (fabsf(cval) + 1.0f));
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(void) {
    ggml_cpu_init();

    printf("test-q2-kvarn-k-quant:\n");
    int passed = 0;
    int failed = 0;

    // Test 1: format pin -- PASSES in RED after minimal struct declaration.
    printf("  Test 1 (sizeof_block): ");
    fflush(stdout);
    test_sizeof_block();
    printf("PASSED\n");
    passed++;

    // Test 2: round-trip MSE quality -- FAILS in RED (quantizer stub aborts).
    printf("  Test 2 (round_trip_outlier): ");
    fflush(stdout);
    test_round_trip_outlier();
    printf("PASSED\n");
    passed++;

    // Test 3: CPU vs CUDA bit-identity -- SKIPS (no CUDA) or FAILS (CUDA present, stub aborts).
    printf("  Test 3 (cpu_cuda_identical): ");
    fflush(stdout);
    test_cpu_cuda_identical();
    printf("PASSED\n");
    passed++;

    // Test 4: degenerate channel -- FAILS in RED (quantizer stub aborts).
    printf("  Test 4 (degenerate_channel): ");
    fflush(stdout);
    test_degenerate_channel();
    printf("PASSED\n");
    passed++;

    printf("\n%d/%d tests passed\n", passed, passed + failed);
    return failed;
}
