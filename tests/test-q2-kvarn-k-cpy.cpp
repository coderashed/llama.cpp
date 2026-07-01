// Phase B (KVARN_FAITHFUL/03): the ggml cpy op F32 -> Q2_KVARN_K.
//
// The write path quantizes per-channel K by transposing each 128-token group to
// channel-major, then cpy-ing into a Q2_KVARN_K tensor. This test drives that cpy
// op through the ggml backend graph on the CPU backend and (when present) the
// CUDA/HIP backend, and asserts the stored blocks are bit-identical to the
// per-channel reference quantizer quantize_row_q2_kvarn_k_ref.
//
// Src is F32 [n_tok x n_ch] channel-major (row = channel, 128 contiguous tokens),
// i.e. the already-transposed group. One dst block per channel.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml-backend.h"

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#endif

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <cstring>
#include <vector>

static const int N_CH  = 4;    // channels (dst blocks)
static const int N_TOK = 128;  // tokens per channel (QG2_KVARN)

// Build the channel-major tile: row ch = 128 contiguous token values.
static std::vector<float> make_tile(void) {
    std::vector<float> tile(N_CH * N_TOK);
    for (int ch = 0; ch < N_CH; ch++) {
        float amp = (ch == 2) ? 9.0f : 1.0f;  // one outlier channel
        for (int tok = 0; tok < N_TOK; tok++) {
            tile[ch * N_TOK + tok] = amp * 0.03f * (float)((tok * 5 + ch * 3) % 29 - 14);
        }
    }
    return tile;
}

// Run cpy(F32 src -> Q2_KVARN_K dst) on `backend`, return the stored blocks.
static std::vector<block_q2_kvarn_k> run_cpy(ggml_backend_t backend, const std::vector<float> & tile) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    // src F32 [n_tok, n_ch]: dim0 = tokens (contiguous), dim1 = channels.
    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,        N_TOK, N_CH);
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN_K, N_TOK, N_CH);
    ggml_set_name(src, "src");
    ggml_set_name(dst, "dst");

    struct ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != NULL);

    ggml_backend_tensor_set(src, tile.data(), 0, ggml_nbytes(src));

    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    assert(st == GGML_STATUS_SUCCESS);

    // cpy's result aliases dst; read it back.
    std::vector<block_q2_kvarn_k> out(N_CH);
    ggml_backend_tensor_get(cpy, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// Run cast(Q2_KVARN_K blocks -> F32) on `backend`, return the dequantized floats
// in channel-major [n_tok, n_ch] layout (same as the input tile).
static std::vector<float> run_cast_back(ggml_backend_t backend, const std::vector<block_q2_kvarn_k> & blocks) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN_K, N_TOK, N_CH);
    struct ggml_tensor * dst = ggml_cast(ctx, src, GGML_TYPE_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != NULL);

    ggml_backend_tensor_set(src, blocks.data(), 0, ggml_nbytes(src));
    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    assert(st == GGML_STATUS_SUCCESS);

    std::vector<float> out(N_CH * N_TOK);
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static void assert_matches_ref(const std::vector<block_q2_kvarn_k> & got,
                               const std::vector<block_q2_kvarn_k> & ref) {
    for (int ch = 0; ch < N_CH; ch++) {
        assert(got[ch].s == ref[ch].s);
        assert(got[ch].z == ref[ch].z);
        assert(memcmp(got[ch].qs, ref[ch].qs, sizeof(ref[ch].qs)) == 0);
    }
}

// Reconstruction recipe test: mirror get_k. Quantize a known standard-layout K
// [C, P] into k_body [G, C, n_groups] (per group: channels' tokens -> blocks),
// then apply cast(F32) -> permute(1,0,2,3) -> cont -> reshape [C, P] and assert
// it recovers dequantize_row_q2_kvarn_k arranged in standard [channel, position].
static void test_reconstruction(ggml_backend_t backend, const char * name) {
    const int C = 8;    // channels
    const int G = 128;  // group size
    const int NG = 3;   // groups
    const int P = G*NG; // positions

    // Known standard-layout K [C, P] (channel-major: element (c,p) at c + p*C? no:
    // ggml [C, P] has C fastest -> element (c,p) at c + p*C). Build a recognizable
    // pattern where each (c,p) is distinct.
    std::vector<float> Kstd(C * P);
    for (int p = 0; p < P; p++)
        for (int c = 0; c < C; c++)
            Kstd[c + p*C] = 0.02f * (float)(((c*131 + p*17) % 61) - 30);

    // Build k_body: for group g, ref-quantize the [C, G] channel-major tile
    // (channel c's G tokens) into C blocks. Tile row c = Kstd channel c, tokens
    // [g*G, g*G+G): value at Kstd[c + (g*G+t)*C].
    std::vector<block_q2_kvarn_k> body(C * NG);
    std::vector<float> tile(C * G);
    for (int g = 0; g < NG; g++) {
        for (int c = 0; c < C; c++)
            for (int t = 0; t < G; t++)
                tile[c*G + t] = Kstd[c + (g*G + t)*C];
        quantize_row_q2_kvarn_k_ref(tile.data(), &body[g*C], C, G); // C blocks for this group
    }

    // Reference reconstruction: dequant each block into standard [C, P].
    std::vector<float> ref(C * P);
    std::vector<float> dqg(C * G);
    for (int g = 0; g < NG; g++) {
        dequantize_row_q2_kvarn_k(&body[g*C], dqg.data(), C, G); // dqg[c*G + t]
        for (int c = 0; c < C; c++)
            for (int t = 0; t < G; t++)
                ref[c + (g*G + t)*C] = dqg[c*G + t];
    }

    // ggml recipe (mirrors get_k): k_body [G, C, NG] -> cast F32 -> permute -> reshape.
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL, /*.no_alloc =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    struct ggml_tensor * kb = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_KVARN_K, G, C, NG);
    struct ggml_tensor * D  = ggml_cast(ctx, kb, GGML_TYPE_F32);              // [G, C, NG]
    struct ggml_tensor * Dp = ggml_cont(ctx, ggml_permute(ctx, D, 1, 0, 2, 3)); // [C, G, NG]
    struct ggml_tensor * K2 = ggml_reshape_2d(ctx, Dp, C, G*NG);            // [C, P]
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, K2);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != NULL);
    ggml_backend_tensor_set(kb, body.data(), 0, ggml_nbytes(kb));
    assert(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    std::vector<float> got(C * P);
    ggml_backend_tensor_get(K2, got.data(), 0, sizeof(float)*C*P);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    for (int i = 0; i < C*P; i++) {
        const float d = got[i] - ref[i];
        assert((d < 0 ? -d : d) <= 1e-4f * (1.0f + (ref[i] < 0 ? -ref[i] : ref[i])));
    }
    printf("  reconstruction on %s: PASSED\n", name);
}

int main(void) {
    ggml_cpu_init();
    printf("test-q2-kvarn-k-cpy:\n");

    const std::vector<float> tile = make_tile();

    // Reference: direct per-channel quantizer.
    std::vector<block_q2_kvarn_k> ref(N_CH);
    quantize_row_q2_kvarn_k_ref(tile.data(), ref.data(), N_CH, N_TOK);

    // CPU backend cpy must match the reference.
    printf("  cpy on CPU backend: ");
    fflush(stdout);
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        assert(cpu != NULL);
        std::vector<block_q2_kvarn_k> got = run_cpy(cpu, tile);
        assert_matches_ref(got, ref);
        ggml_backend_free(cpu);
    }
    printf("PASSED\n");

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    printf("  cpy on CUDA/HIP backend: ");
    fflush(stdout);
    {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) {
            printf("SKIP (no device)\n");
        } else {
            std::vector<block_q2_kvarn_k> got = run_cpy(gpu, tile);
            assert_matches_ref(got, ref);
            ggml_backend_free(gpu);
            printf("PASSED\n");
        }
    }
#endif

    // Reverse path: cast Q2_KVARN_K -> F32 must match the CPU ref dequant.
    std::vector<float> refdq(N_CH * N_TOK);
    dequantize_row_q2_kvarn_k(ref.data(), refdq.data(), N_CH, N_TOK);

    auto assert_dq = [&](const std::vector<float> & got) {
        for (int i = 0; i < N_CH * N_TOK; i++) {
            const float d = got[i] - refdq[i];
            assert((d < 0 ? -d : d) <= 1e-4f * (1.0f + (refdq[i] < 0 ? -refdq[i] : refdq[i])));
        }
    };

    printf("  cast-back on CPU backend: ");
    fflush(stdout);
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        assert_dq(run_cast_back(cpu, ref));
        ggml_backend_free(cpu);
    }
    printf("PASSED\n");

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    printf("  cast-back on CUDA/HIP backend: ");
    fflush(stdout);
    {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) {
            printf("SKIP (no device)\n");
        } else {
            assert_dq(run_cast_back(gpu, ref));
            ggml_backend_free(gpu);
            printf("PASSED\n");
        }
    }
#endif

    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        test_reconstruction(cpu, "CPU backend");
        ggml_backend_free(cpu);
    }
#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (gpu) { test_reconstruction(gpu, "CUDA/HIP backend"); ggml_backend_free(gpu); }
    }
#endif

    printf("\nall cpy tests passed\n");
    return 0;
}
