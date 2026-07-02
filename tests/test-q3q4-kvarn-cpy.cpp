// KVARN_MULTIBIT: ggml cpy for the 3/4-bit KVarN siblings (per-token
// Q3_KVARN/Q4_KVARN and per-channel Q3_KVARN_K/Q4_KVARN_K) <-> F32.
//
// Mirrors test-q2-kvarn-cpy.cpp: drives both directions of the cpy op through
// the ggml backend graph on the CPU backend and (when present) the CUDA/HIP
// backend, asserting bit/value match against the reference quantizer and
// dequantizer. Also pins the bit-width ordering: on the same data, round-trip
// MSE must strictly improve 2 -> 3 -> 4 bits.

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
static const int N_ELEM   = N_BLOCKS * 128; // all four types use 128-element blocks

static std::vector<float> make_tile(void) {
    std::vector<float> tile(N_ELEM);
    for (int b = 0; b < N_BLOCKS; b++) {
        float amp = (b == 2) ? 9.0f : 1.0f;  // one outlier block
        for (int j = 0; j < 128; j++) {
            tile[b * 128 + j] = amp * 0.03f * (float)((j * 5 + b * 3) % 29 - 14);
        }
    }
    return tile;
}

// Run cpy(F32 src -> quantized dst) on `backend`, return the raw block bytes.
static std::vector<uint8_t> run_cpy(ggml_backend_t backend, ggml_type qtype, const std::vector<float> & tile) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, N_ELEM);
    struct ggml_tensor * dst = ggml_new_tensor_1d(ctx, qtype,         N_ELEM);

    struct ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, cpy);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != NULL);

    ggml_backend_tensor_set(src, tile.data(), 0, ggml_nbytes(src));

    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    assert(st == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> out(ggml_nbytes(dst));
    ggml_backend_tensor_get(cpy, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

// Run cast(quantized blocks -> F32) on `backend`, return the dequantized floats.
static std::vector<float> run_cast_back(ggml_backend_t backend, ggml_type qtype, const std::vector<uint8_t> & blocks) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 4 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_1d(ctx, qtype, N_ELEM);
    struct ggml_tensor * dst = ggml_cast(ctx, src, GGML_TYPE_F32);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf != NULL);

    ggml_backend_tensor_set(src, blocks.data(), 0, blocks.size());
    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    assert(st == GGML_STATUS_SUCCESS);

    std::vector<float> out(N_ELEM);
    ggml_backend_tensor_get(dst, out.data(), 0, ggml_nbytes(dst));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static void assert_close(const std::vector<float> & got, const std::vector<float> & ref) {
    for (int i = 0; i < N_ELEM; i++) {
        const float d = got[i] - ref[i];
        assert((d < 0 ? -d : d) <= 1e-4f * (1.0f + (ref[i] < 0 ? -ref[i] : ref[i])));
    }
}

static double mse_vs(const std::vector<float> & got, const std::vector<float> & ref) {
    double acc = 0.0;
    for (int i = 0; i < N_ELEM; i++) {
        const double d = (double)got[i] - (double)ref[i];
        acc += d * d;
    }
    return acc / N_ELEM;
}

// One type's full check: reference quant/dequant vs backend cpy/cast on CPU and GPU.
// Returns the reference round-trip MSE vs the source tile.
static double check_type(ggml_type qtype,
        const std::vector<float> & tile,
        const std::vector<uint8_t> & ref_blocks,
        const std::vector<float> & ref_dq) {
    printf("  %s:\n", ggml_type_name(qtype));

    printf("    cpy on CPU backend: ");
    fflush(stdout);
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        assert(cpu != NULL);
        std::vector<uint8_t> got = run_cpy(cpu, qtype, tile);
        assert(got.size() == ref_blocks.size());
        assert(memcmp(got.data(), ref_blocks.data(), got.size()) == 0);
        assert_close(run_cast_back(cpu, qtype, ref_blocks), ref_dq);
        ggml_backend_free(cpu);
    }
    printf("PASSED\n");

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
    printf("    cpy on CUDA/HIP backend: ");
    fflush(stdout);
    {
        ggml_backend_t gpu = ggml_backend_cuda_init(0);
        if (!gpu) {
            printf("SKIP (no device)\n");
        } else {
            std::vector<uint8_t> got = run_cpy(gpu, qtype, tile);
            assert(got.size() == ref_blocks.size());
            assert(memcmp(got.data(), ref_blocks.data(), got.size()) == 0);
            assert_close(run_cast_back(gpu, qtype, ref_blocks), ref_dq);
            ggml_backend_free(gpu);
            printf("PASSED\n");
        }
    }
#endif

    return mse_vs(ref_dq, tile);
}

int main(void) {
    ggml_cpu_init();
    printf("test-q3q4-kvarn-cpy:\n");

    const std::vector<float> tile = make_tile();

    // Per-token references.
    std::vector<block_q2_kvarn> r2(N_BLOCKS);
    std::vector<block_q3_kvarn> r3(N_BLOCKS);
    std::vector<block_q4_kvarn> r4(N_BLOCKS);
    quantize_row_q2_kvarn_ref(tile.data(), r2.data(), N_ELEM);
    quantize_row_q3_kvarn_ref(tile.data(), r3.data(), N_ELEM);
    quantize_row_q4_kvarn_ref(tile.data(), r4.data(), N_ELEM);

    std::vector<float> dq2(N_ELEM), dq3(N_ELEM), dq4(N_ELEM);
    dequantize_row_q2_kvarn(r2.data(), dq2.data(), N_ELEM);
    dequantize_row_q3_kvarn(r3.data(), dq3.data(), N_ELEM);
    dequantize_row_q4_kvarn(r4.data(), dq4.data(), N_ELEM);

    auto bytes_of = [](const void * p, size_t n) {
        return std::vector<uint8_t>((const uint8_t *)p, (const uint8_t *)p + n);
    };

    const double mse3 = check_type(GGML_TYPE_Q3_KVARN, tile,
            bytes_of(r3.data(), N_BLOCKS * sizeof(block_q3_kvarn)), dq3);
    const double mse4 = check_type(GGML_TYPE_Q4_KVARN, tile,
            bytes_of(r4.data(), N_BLOCKS * sizeof(block_q4_kvarn)), dq4);
    const double mse2 = mse_vs(dq2, tile);

    printf("  per-token round-trip MSE: q2=%.6g q3=%.6g q4=%.6g\n", mse2, mse3, mse4);
    assert(mse3 < mse2);
    assert(mse4 < mse3);

    // Per-channel references (blocks = channels of 128 tokens each; the 1d
    // contiguous layout matches the ggml adapter contract exactly).
    std::vector<block_q3_kvarn_k> rk3(N_BLOCKS);
    std::vector<block_q4_kvarn_k> rk4(N_BLOCKS);
    quantize_row_q3_kvarn_k_ref(tile.data(), rk3.data(), N_BLOCKS, 128);
    quantize_row_q4_kvarn_k_ref(tile.data(), rk4.data(), N_BLOCKS, 128);

    std::vector<float> dqk3(N_ELEM), dqk4(N_ELEM);
    dequantize_row_q3_kvarn_k(rk3.data(), dqk3.data(), N_BLOCKS, 128);
    dequantize_row_q4_kvarn_k(rk4.data(), dqk4.data(), N_BLOCKS, 128);

    const double msek3 = check_type(GGML_TYPE_Q3_KVARN_K, tile,
            bytes_of(rk3.data(), N_BLOCKS * sizeof(block_q3_kvarn_k)), dqk3);
    const double msek4 = check_type(GGML_TYPE_Q4_KVARN_K, tile,
            bytes_of(rk4.data(), N_BLOCKS * sizeof(block_q4_kvarn_k)), dqk4);

    printf("  per-channel round-trip MSE: q3_k=%.6g q4_k=%.6g\n", msek3, msek4);
    assert(msek4 < msek3);

    printf("\nall q3/q4 kvarn cpy tests passed\n");
    return 0;
}
