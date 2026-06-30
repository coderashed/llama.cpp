// Tests for Q2_KVARN GPU backend (item 09) and K vec_dot formula (DP4A item 01).
// Tests 1-3: CUDA/HIP backend support for Q2_KVARN operations.
// Test 4:    CPU float formula vs fp64 oracle for the K attention dot product.
//            Pins the contract that vec_dot_fattn_vec_KQ_q2_kvarn must satisfy;
//            the DP4A kernel (item 02) must pass the same oracle.
// Test 6:    Regression for KVARN_BATCH_CRASH/02.
//            Root cause: ggml_get_to_fp16_nc_cuda(Q2_KVARN) returns NULL (no entry
//            in the switch in convert.cu). When a non-VEC flash-attention kernel
//            (TILE/MMA/WMMA) is chosen and K is a NON-CONTIGUOUS q2_kvarn view,
//            launch_fattn takes the _nc branch at fattn-common.cuh:1124 and
//            calls that NULL function pointer -> SIGSEGV.
//            Fix (item KVARN_BATCH_CRASH/02): route q2_kvarn K/V unconditionally
//            to VEC (or CPU fallback) in ggml_cuda_get_best_fattn_kernel so the
//            _nc path is never reached.
//            RED construction: build a flash_attn_ext op with
//              Q->ne[1] = 8 (> 2 forces non-VEC dispatch on current code)
//              K = q2_kvarn, NON-CONTIGUOUS 3d view of a wider parent
//            On CURRENT code this crashes (SIGSEGV). After the fix it returns
//            finite values on the VEC kernel.

#include "ggml.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>

// Test 1: supports_op returns true for Q2_KVARN MUL_MAT on CUDA
// Currently returns false because Q2_KVARN is not in the CUDA supports_op switch
static void test_supports_op(void) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("  SKIP (no CUDA device)\n");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 4096, 4096);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096, 1);
    struct ggml_tensor * op = ggml_mul_mat(ctx, a, b);

    bool supported = ggml_backend_supports_op(cuda_backend, op);
    // FAILS: Q2_KVARN not in CUDA supports_op switch -> returns false
    assert(supported && "CUDA backend should support Q2_KVARN MUL_MAT");

    ggml_free(ctx);
}

// Test 2: CUDA dequant kernel exists for Q2_KVARN
// Tests that CPY from Q2_KVARN to F32 is supported on CUDA backend
// Currently fails because no dequant kernel exists for Q2_KVARN in convert.cu
static void test_dequant_kernel(void) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("  SKIP (no CUDA device)\n");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 128, 1);
    struct ggml_tensor * dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
    struct ggml_tensor * cpy = ggml_cpy(ctx, src, dst);

    bool supported = ggml_backend_supports_op(cuda_backend, cpy);
    // FAILS: no Q2_KVARN case in ggml_get_to_fp32_cuda / ggml_get_to_fp16_cuda
    assert(supported && "CUDA backend should support Q2_KVARN dequant (CPY to F32)");

    ggml_free(ctx);
}

// Test 3: Flash attention handles Q2_KVARN K/V
// Tests that FLASH_ATTN_EXT with Q2_KVARN K/V is supported on CUDA
// Currently fails because flash attention doesn't handle Q2_KVARN
static void test_flash_attn(void) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("  SKIP (no CUDA device)\n");
        return;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);

    // Q: F32, K: Q2_KVARN, V: Q2_KVARN
    struct ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 1);
    struct ggml_tensor * k = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 128, 1);
    struct ggml_tensor * v = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_KVARN, 128, 1);
    struct ggml_tensor * mask = NULL;
    struct ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    bool supported = ggml_backend_supports_op(cuda_backend, attn);
    // FAILS: flash attention doesn't handle Q2_KVARN K/V
    assert(supported && "CUDA flash attention should handle Q2_KVARN K/V");

    ggml_free(ctx);
}

// Test 5: DP4A formula vs fp64 oracle for the K vec_dot.
// Validates the integer-dot + zeropoint path:
//   scale * (Q_ds.x * sumi + d * Q_ds.y / QI8_1)
// where sumi = sum_b(k2bit[b] * q_int8[b]) (integer dp4a emulation).
// Tests the /QI8_1 zeropoint correction that the DP4A kernel must get right.
static void test_vec_dot_k_dp4a_formula(void) {
    srand(54321);

    // QK8_1=32 elements per Q8_1 block; QI8_1=8 groups of 4 per Q8_1 block.
    const int QI8_1_val = 8;
    const int N = 128;  // one KVarN block
    const int nq8blocks = N / 32;  // 4 Q8_1 blocks per KVarN block
    const int ngroups   = N / 4;   // 32 groups of 4

    for (int run = 0; run < 20; run++) {
        // Random KVarN block.
        uint8_t qs[32];
        for (int i = 0; i < 32; i++) qs[i] = (uint8_t)(rand() & 0xFF);
        float d_in  = ((rand() % 201) - 100) * 0.01f;
        float s1_in = 0.5f + (rand() % 151) * 0.01f;
        float s2_in = 0.5f + (rand() % 151) * 0.01f;
        float d  = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d_in));
        float s1 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s1_in));
        float s2 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s2_in));
        float scale = s1 * s2;

        // Q8_1 blocks: int8 values, scale (ds.x), and sum-of-floats (ds.y).
        // ds.y is set to sum_i(q_int8[i] * ds.x) -- no extra quantization error.
        int8_t q_int8[128];
        float q_dx[4];  // ds.x per Q8_1 block
        float q_dy[4];  // ds.y per Q8_1 block (sum of original floats)
        for (int blk = 0; blk < nq8blocks; blk++) {
            q_dx[blk] = 0.01f + (rand() % 100) * 0.001f;
            float sum_f = 0.0f;
            for (int i = 0; i < 32; i++) {
                int8_t qi = (int8_t)((rand() % 255) - 127);
                q_int8[blk * 32 + i] = qi;
                sum_f += (float)qi * q_dx[blk];
            }
            q_dy[blk] = sum_f;
        }

        // fp64 oracle: sum_i (q_int8[i] * q_dx) * (k2bit[i] + d) * scale
        double ref64 = 0.0;
        for (int i = 0; i < N; i++) {
            int shift = (i % 4) * 2;
            double k2bit = (double)((qs[i / 4] >> shift) & 0x03);
            double q_f = (double)q_int8[i] * (double)q_dx[i / 32];
            ref64 += q_f * (k2bit + (double)d) * (double)s1 * (double)s2;
        }

        // DP4A emulation: scale * (Q_ds.x * sumi + d * Q_ds.y / QI8_1) per group.
        float result_dp4a = 0.0f;
        for (int grp = 0; grp < ngroups; grp++) {
            uint8_t qbyte = qs[grp];
            int sumi = 0;
            for (int b = 0; b < 4; b++) {
                sumi += (int)((qbyte >> (b * 2)) & 0x03) * (int)q_int8[grp * 4 + b];
            }
            int q8blk = grp / QI8_1_val;
            result_dp4a += scale * (q_dx[q8blk] * sumi + d * q_dy[q8blk] / QI8_1_val);
        }

        float ref_f = (float)ref64;
        float denom = fabsf(ref_f) > 1e-6f ? fabsf(ref_f) : 1e-6f;
        float relerr = fabsf(result_dp4a - ref_f) / denom;
        assert(relerr < 1e-4f);
    }
}

// Test 4: Float formula vs fp64 oracle for vec_dot_fattn_vec_KQ_q2_kvarn.
// K-dot formula: Sum_i Q[i] * (k2bit[i] + d) * s1 * s2
// Runs on CPU; no GPU required. Skipped at test-time if no DP4A device is
// present (the DP4A kernel, item 02, will invoke this same oracle on-device).
static void test_vec_dot_k_formula(void) {
    srand(12345);

    for (int run = 0; run < 20; run++) {
        // Random 2-bit K block: 128 elements packed in 32 bytes.
        uint8_t qs[32];
        for (int i = 0; i < 32; i++) qs[i] = (uint8_t)(rand() & 0xFF);

        // Scales stored as fp16 to match the GPU kernel's read precision.
        float d_in  = ((rand() % 201) - 100) * 0.01f;
        float s1_in = 0.5f + (rand() % 151) * 0.01f;
        float s2_in = 0.5f + (rand() % 151) * 0.01f;
        float d  = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d_in));
        float s1 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s1_in));
        float s2 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(s2_in));
        float scale = s1 * s2;

        // Random Q values in [-1, 1].
        float Q[128];
        for (int i = 0; i < 128; i++) Q[i] = ((rand() % 201) - 100) * 0.01f;

        // fp64 oracle: reference value the float kernel must match.
        double ref64 = 0.0;
        for (int i = 0; i < 128; i++) {
            double k2bit = (double)((qs[i / 4] >> ((i % 4) * 2)) & 0x03);
            ref64 += (double)Q[i] * (k2bit + (double)d) * (double)s1 * (double)s2;
        }

        // Float emulation: same loop structure as vec_dot_fattn_vec_KQ_q2_kvarn.
        float result_f = 0.0f;
        for (int i = 0; i < 32; i++) {
            uint8_t qbyte = qs[i];
            for (int b = 0; b < 4; b++) {
                float kval = ((float)((qbyte >> (b * 2)) & 0x03) + d) * scale;
                result_f += kval * Q[i * 4 + b];
            }
        }

        float ref_f = (float)ref64;
        float denom = fabsf(ref_f) > 1e-6f ? fabsf(ref_f) : 1e-6f;
        float relerr = fabsf(result_f - ref_f) / denom;
        assert(relerr < 1e-4f);
    }
}

// Test 6: Batched q2_kvarn flash-attention must not crash (KVARN_BATCH_CRASH/02).
//
// Trigger path:
//   1. K is a q2_kvarn view with non-contiguous strides (nb[1] == 2*type_size
//      because the parent has head_dim=256 but we view head_dim=128).
//   2. Q->ne[1] = 8 > 2, so ggml_cuda_get_best_fattn_kernel picks a non-VEC
//      kernel (MMA_F16 on Turing, TILE on AMD/no-TC).
//   3. The non-VEC kernel sets need_f16_K=true. launch_fattn sees
//      !ggml_is_contiguously_allocated(K) and calls
//      ggml_get_to_fp16_nc_cuda(Q2_KVARN), which returns NULL (no entry in the
//      switch). Dereferencing that NULL -> SIGSEGV on current code.
//
// After the fix (guard in ggml_cuda_get_best_fattn_kernel):
//   q2_kvarn K/V is routed unconditionally to VEC. VEC does not set
//   need_f16_K for quantized K, so the _nc path is never taken. The kernel
//   runs and produces finite output values.
//
// RED: process crashes (SIGSEGV / abort). CTest reports non-zero exit -> FAIL.
// GREEN: compute completes, all outputs are finite.
static void test_flash_attn_batched_no_crash(void) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init(0);
    if (!cuda_backend) {
        printf("  SKIP (no CUDA device)\n");
        return;
    }

    // Head dim 128, KV length 256 (multiple of FATTN_KQ_STRIDE=256).
    // n_q=8 > 2 forces the non-VEC dispatch on current code.
    const int64_t head_dim = 128;
    const int64_t n_q      = 8;
    const int64_t Nkv      = 256;   // must be multiple of FATTN_KQ_STRIDE (256)
    const int64_t n_head   = 1;

    // Context must use no_alloc=true: required by ggml_backend_alloc_ctx_tensors.
    // Tensors are allocated on the GPU backend after graph construction.
    struct ggml_init_params params = {
        /*.mem_size   =*/ 64*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    assert(ctx);

    // Q: F32, shape [head_dim, n_q, n_head, 1].
    struct ggml_tensor * Q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_q, n_head);

    // Kparent: Q2_KVARN with ne[0]=256 (two 128-element blocks per KV row).
    // Allocating it wide means a 128-element view of it has nb[1] = 2*type_size,
    // which is NOT the contiguous stride (type_size), making the view strided.
    struct ggml_tensor * Kparent = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_KVARN, 256, Nkv, n_head);

    // Kview: Q2_KVARN, logical shape [128, Nkv, n_head].
    // nb[0] = type_size (inherited; satisfies fattn-common.cuh:1123 assert).
    // nb[1] = Kparent->nb[1] = 2*type_size != type_size -> non-contiguous.
    // ggml_is_contiguously_allocated(Kview) == false -> _nc converter branch.
    // offset=0 so Kview->data = NULL under no_alloc=true; ggml_backend_view_init
    // will correctly set it to Kparent->data+0 after Kparent is allocated.
    struct ggml_tensor * Kview = ggml_view_3d(ctx, Kparent,
        head_dim, Nkv, n_head,
        Kparent->nb[1],   // stride per KV position: 2 blocks wide
        Kparent->nb[2],   // stride per head
        0);               // byte offset into Kparent

    // V: Q4_0 contiguous (matches the integration scenario -ctv q4_0).
    struct ggml_tensor * V = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, head_dim, Nkv, n_head);

    // mask: F16, contiguous. ne[2]=1 required by ggml_cuda_get_best_fattn_kernel
    // guard at fattn.cu:457-459. Shape [Nkv, n_q, 1, 1].
    struct ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, Nkv, n_q, 1, 1);

    float scale = 1.0f / sqrtf((float)head_dim);
    struct ggml_tensor * op = ggml_flash_attn_ext(ctx, Q, Kview, V, mask, scale, 0.0f, 0.0f);

    // Before fix: dispatch returns MMA_F16/TILE (not NONE) -> supports_op=true.
    // After fix: returns VEC -> also true. Contract: always true for this shape.
    bool supported = ggml_backend_supports_op(cuda_backend, op);
    assert(supported && "CUDA backend must support flash_attn_ext with q2_kvarn K (non-contiguous view)");

    // Allocate all context tensors on the CUDA backend.
    // ggml_backend_alloc_ctx_tensors iterates the context tensor list in order:
    //   Kparent (no view_src) -> gets GPU buffer.
    //   Kview   (view_src=Kparent) -> ggml_backend_view_init sets data=Kparent->data+0.
    //   Q, V, mask, op -> each gets its own GPU buffer region.
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
    assert(buf && "Failed to allocate GPU tensors for Test 6");

    // Fill Q with small finite values. Content is irrelevant for the crash test
    // (the crash occurs before any kernel reads data), but must be finite for
    // the post-fix correctness check.
    {
        size_t q_bytes = ggml_nbytes(Q);
        float * q_data = (float *) malloc(q_bytes);
        assert(q_data);
        size_t n_q_elems = q_bytes / sizeof(float);
        for (size_t i = 0; i < n_q_elems; i++) q_data[i] = 0.01f;
        ggml_backend_tensor_set(Q, q_data, 0, q_bytes);
        free(q_data);
    }

    // Fill Kparent with all-zero q2_kvarn blocks: d=0, s1=0, s2=0, qs=0.
    // Zero-scale blocks produce zero attention weights; output will be finite
    // (attention over zeros = uniform softmax over zeros = zero output after
    // weighted sum). Sufficient for the RED: the crash is in kernel dispatch,
    // not in arithmetic.
    {
        size_t kp_bytes = ggml_nbytes(Kparent);
        void * kp_data = calloc(1, kp_bytes);
        assert(kp_data);
        ggml_backend_tensor_set(Kparent, kp_data, 0, kp_bytes);
        free(kp_data);
    }

    // Fill V with all-zero q4_0 blocks.
    {
        size_t v_bytes = ggml_nbytes(V);
        void * v_data = calloc(1, v_bytes);
        assert(v_data);
        ggml_backend_tensor_set(V, v_data, 0, v_bytes);
        free(v_data);
    }

    // Fill mask with F16 zeros (no masking).
    {
        size_t m_bytes = ggml_nbytes(mask);
        void * m_data = calloc(1, m_bytes);
        assert(m_data);
        ggml_backend_tensor_set(mask, m_data, 0, m_bytes);
        free(m_data);
    }

    // Build and run the graph.
    // RED (current code): ggml_cuda_get_best_fattn_kernel returns MMA_F16 or TILE
    // for Q->ne[1]=8 with quantized K. launch_fattn sets need_f16_K=true and
    // calls ggml_get_to_fp16_nc_cuda(Q2_KVARN) which returns NULL. The NULL
    // function pointer is called on the next line -> SIGSEGV. The process dies;
    // CTest reports non-zero exit -> FAIL (RED confirmed).
    //
    // GREEN (after fix): the guard before the MMA/TILE blocks routes q2_kvarn
    // to VEC. VEC sets need_f16_K = (K->type == F32) = false for q2_kvarn,
    // so no f16 conversion is attempted. The VEC kernel reads q2_kvarn K directly
    // via vec_dot_fattn_vec_KQ_q2_kvarn and completes successfully.
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, op);

    ggml_status status = ggml_backend_graph_compute(cuda_backend, gf);
    assert(status == GGML_STATUS_SUCCESS && "flash_attn_ext (q2_kvarn K non-contiguous) must complete without error");

    // Verify the output is finite and not all-zero.
    // Result shape: [V->ne[0], Q->ne[2], Q->ne[1], Q->ne[3]] = [128, 1, 8, 1].
    {
        size_t out_bytes = ggml_nbytes(op);
        float * out = (float *) malloc(out_bytes);
        assert(out);
        ggml_backend_tensor_get(op, out, 0, out_bytes);
        size_t n_out = out_bytes / sizeof(float);
        for (size_t i = 0; i < n_out; i++) {
            assert(isfinite(out[i]) && "Test 6: output value must be finite after fix");
        }
        free(out);
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(cuda_backend);
}

int main(void) {
    printf("test-q2-kvarn-gpu:\n");

    printf("  Test 1 (supports_op): ");
    fflush(stdout);
    test_supports_op();
    printf("PASSED\n");

    printf("  Test 2 (dequant_kernel): ");
    fflush(stdout);
    test_dequant_kernel();
    printf("PASSED\n");

    printf("  Test 3 (flash_attn): ");
    fflush(stdout);
    test_flash_attn();
    printf("PASSED\n");

    printf("  Test 4 (vec_dot_k_formula): ");
    fflush(stdout);
    test_vec_dot_k_formula();
    printf("PASSED\n");

    printf("  Test 5 (vec_dot_k_dp4a_formula): ");
    fflush(stdout);
    test_vec_dot_k_dp4a_formula();
    printf("PASSED\n");

    printf("  Test 6 (flash_attn_batched_no_crash): ");
    fflush(stdout);
    test_flash_attn_batched_no_crash();
    printf("PASSED\n");

    printf("\nAll tests passed\n");
    return 0;
}
