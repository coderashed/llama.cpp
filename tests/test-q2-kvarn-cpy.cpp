// Item 06 (KVARN_FAITHFUL): the ggml cpy op Q2_KVARN (per-token) <-> F32.
//
// Per-token block_q2_kvarn is already read/written by ggml_quantize_chunk and the
// FA vec kernel, but ggml_cpy/ggml_cast never had a Q2_KVARN -> F32 case registered
// on the CUDA/HIP backend (supports_op claimed it worked; the dispatch switch in
// ggml_cuda_cpy fell through to the "unsupported type combination" abort). This
// test drives both directions of the cpy op through the ggml backend graph on the
// CPU backend and (when present) the CUDA/HIP backend, and asserts bit/value match
// against the reference quantizer/dequantizer.

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

static const int N_BLOCKS = 4;
static const int N_ELEM   = N_BLOCKS * QK2_KVARN;

static std::vector<float> make_tile(void) {
    std::vector<float> tile(N_ELEM);
    for (int b = 0; b < N_BLOCKS; b++) {
        float amp = (b == 2) ? 9.0f : 1.0f;  // one outlier block
        for (int j = 0; j < QK2_KVARN; j++) {
            tile[b * QK2_KVARN + j] = amp * 0.03f * (float)((j * 5 + b * 3) % 29 - 14);
        }
    }
    return tile;
}

// Run cpy(F32 src -> Q2_KVARN dst) on `backend`, return the stored blocks.
static std::vector<block_q2_kvarn> run_cpy(ggml_backend_t backend, const std::vector<float> & tile) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,       N_ELEM);
    struct ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_Q2_KVARN, N_ELEM);
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

    std::vector<block_q2_kvarn> out(N_BLOCKS);
    ggml_backend_tensor_get(cpy, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// Run cast(Q2_KVARN blocks -> F32) on `backend`, return the dequantized floats.
static std::vector<float> run_cast_back(ggml_backend_t backend, const std::vector<block_q2_kvarn> & blocks) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_Q2_KVARN, N_ELEM);
    struct ggml_tensor * dst = ggml_cast(ctx, src, GGML_TYPE_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != NULL);

    ggml_backend_tensor_set(src, blocks.data(), 0, ggml_nbytes(src));
    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    assert(st == GGML_STATUS_SUCCESS);

    std::vector<float> out(N_ELEM);
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static void assert_matches_ref(const std::vector<block_q2_kvarn> & got,
                               const std::vector<block_q2_kvarn> & ref) {
    for (int b = 0; b < N_BLOCKS; b++) {
        assert(got[b].d  == ref[b].d);
        assert(got[b].s1 == ref[b].s1);
        assert(got[b].s2 == ref[b].s2);
        assert(memcmp(got[b].qs, ref[b].qs, sizeof(ref[b].qs)) == 0);
    }
}

int main(void) {
    ggml_cpu_init();
    printf("test-q2-kvarn-cpy:\n");

    const std::vector<float> tile = make_tile();

    // Reference: direct per-token quantizer.
    std::vector<block_q2_kvarn> ref(N_BLOCKS);
    quantize_row_q2_kvarn_ref(tile.data(), ref.data(), N_ELEM);

    printf("  cpy on CPU backend: ");
    fflush(stdout);
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        assert(cpu != NULL);
        std::vector<block_q2_kvarn> got = run_cpy(cpu, tile);
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
            std::vector<block_q2_kvarn> got = run_cpy(gpu, tile);
            assert_matches_ref(got, ref);
            ggml_backend_free(gpu);
            printf("PASSED\n");
        }
    }
#endif

    // Reverse path: cast Q2_KVARN -> F32 must match the CPU ref dequant.
    std::vector<float> refdq(N_ELEM);
    dequantize_row_q2_kvarn(ref.data(), refdq.data(), N_ELEM);

    auto assert_dq = [&](const std::vector<float> & got) {
        for (int i = 0; i < N_ELEM; i++) {
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

    printf("\nall cpy tests passed\n");
    return 0;
}
