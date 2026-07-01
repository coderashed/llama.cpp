// GGML_OP_KVARN_VARN graph-op test (KVARN_FAITHFUL/04, GPU graph integration).
// Validates that ggml_kvarn_varn, run through a real backend graph, produces the
// packed [T_norm ++ S_r ++ S_c] output that the CPU custom-op reference (kvarn_varn_op)
// produces:
//   - CPU backend forward must match the reference bit-close (same double math).
//   - CUDA/HIP backend forward must match within a small relative tolerance (the
//     device iterative float math diverges by a few percent on best-Imb near-ties,
//     as already characterized by test-q2-kvarn-varn-gpu).
// This is the guard that the op plumbing (dispatch, packed-offset layout, per-head
// tiling) is correct end to end. Only built for CUDA/HIP.

#include "../src/llama-kvarn.h"

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <random>

// Reference: the CPU custom-op (src/llama-kvarn.cpp), computed directly on host data.
static std::vector<float> reference_packed(const std::vector<float> & in,
        int n_tok, int head_dim, int n_head) {
    const int64_t n_ch  = (int64_t) head_dim * n_head;
    const int64_t total = n_ch*n_tok + n_ch + (int64_t) n_head*n_tok;

    struct ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (total + n_ch*n_tok) * sizeof(float) + 4*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_tok, head_dim, n_head);
    memcpy(src->data, in.data(), in.size() * sizeof(float));
    struct ggml_tensor * dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total);
    dst->src[0] = src;

    kvarn_varn_op(dst, 0, 1, nullptr);

    std::vector<float> out((size_t) total);
    memcpy(out.data(), dst->data, (size_t) total * sizeof(float));
    ggml_free(ctx);
    return out;
}

// Run ggml_kvarn_varn through a backend graph and return the packed output.
static std::vector<float> run_op_backend(ggml_backend_t backend, const std::vector<float> & in,
        int n_tok, int head_dim, int n_head) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_tok, head_dim, n_head);
    struct ggml_tensor * op  = ggml_kvarn_varn(ctx, src);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buf);

    ggml_backend_tensor_set(src, in.data(), 0, in.size() * sizeof(float));

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, op);
    ggml_status status = ggml_backend_graph_compute(backend, gf);
    assert(status == GGML_STATUS_SUCCESS);

    std::vector<float> out(ggml_nelements(op));
    ggml_backend_tensor_get(op, out.data(), 0, ggml_nbytes(op));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return out;
}

static float max_rel_dev(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        m = fmaxf(m, fabsf(a[i] - b[i]) / (1.0f + fabsf(b[i])));
    }
    return m;
}

int main(void) {
    printf("test-q2-kvarn-varn-op:\n");

    // Per-head tiles: n_head heads, each [head_dim rows x n_tok cols], row-major.
    const int n_tok = 64, head_dim = 128, n_head = 3;

    std::mt19937 rng(0xC0FFEEU);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> in((size_t) n_tok * head_dim * n_head);
    // src layout [n_tok, head_dim, n_head]: element (t,d,h) at t + n_tok*(d + head_dim*h).
    // Give a few columns (tokens) an outlier magnitude so VarN has real work to do.
    for (int h = 0; h < n_head; h++)
        for (int d = 0; d < head_dim; d++)
            for (int t = 0; t < n_tok; t++)
                in[t + n_tok*(d + head_dim*h)] = nd(rng) * ((t % 8 == 0) ? 6.0f : 1.0f);

    std::vector<float> ref = reference_packed(in, n_tok, head_dim, n_head);

    // CPU backend forward: same double math as the reference -> bit-close.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu);
    std::vector<float> cpu_out = run_op_backend(cpu, in, n_tok, head_dim, n_head);
    float cpu_dev = max_rel_dev(cpu_out, ref);
    printf("  CPU backend op vs reference: max rel dev = %.7f\n", cpu_dev);
    assert(cpu_dev <= 1e-5f);
    ggml_backend_free(cpu);

    // CUDA/HIP backend forward: device iterative float math -> small tolerance.
    ggml_backend_t cuda = ggml_backend_cuda_init(0);
    if (!cuda) {
        printf("  SKIP CUDA (no device)\n\nPASSED\n");
        return 0;
    }
    std::vector<float> cuda_out = run_op_backend(cuda, in, n_tok, head_dim, n_head);
    float cuda_dev = max_rel_dev(cuda_out, ref);
    printf("  CUDA backend op vs reference: max rel dev = %.7f\n", cuda_dev);
    assert(cuda_dev <= 5e-2f);
    ggml_backend_free(cuda);

    printf("\nPASSED\n");
    return 0;
}
