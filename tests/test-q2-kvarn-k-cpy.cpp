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

static void assert_matches_ref(const std::vector<block_q2_kvarn_k> & got,
                               const std::vector<block_q2_kvarn_k> & ref) {
    for (int ch = 0; ch < N_CH; ch++) {
        assert(got[ch].s == ref[ch].s);
        assert(got[ch].z == ref[ch].z);
        assert(memcmp(got[ch].qs, ref[ch].qs, sizeof(ref[ch].qs)) == 0);
    }
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

    printf("\nall cpy tests passed\n");
    return 0;
}
